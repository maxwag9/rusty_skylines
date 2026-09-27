use crate::data::{Settings, ShadowType};
use crate::gpu_timestamp;
use crate::helpers::modpack::ModManager;
use crate::renderer::gizmo::gizmo::Gizmo;
use crate::renderer::gpu_profiler::GpuProfiler;
use crate::renderer::pipelines::{DEPTH_FORMAT, Pipelines};
use crate::renderer::props::{GpuPropInstance, PropVertex, Props};
use crate::renderer::ray_tracing::rt_subsystem::RTSubsystem;
use crate::renderer::render_core::{create_color_attachment_clear, create_color_attachment_load};
use crate::renderer::textures::material_keys::*;
use crate::renderer::ui::UiRenderer;
use crate::renderer::ui_pipelines::multisample_state;
use crate::ui::vertex::{TextVtxRender, ThickLineVtxRender, ThinLineVtxRender, Vertex};
use crate::world::buildings::building_mesher::BuildingVertex;
use crate::world::buildings::building_renderer::BuildingRenderer;
use crate::world::buildings::buildings::Buildings;
use crate::world::camera::Camera;
use crate::world::cars::car_mesh::CarVertex;
use crate::world::cars::car_render::CarInstance;
use crate::world::cars::car_structs::CarStorage;
use crate::world::cars::car_subsystem::CarRenderSubsystem;
use crate::world::roads::road_mesh_manager::AdvancedVertex;
use crate::world::roads::road_subsystem::RoadRenderSubsystem;
use crate::world::terrain::sky::{STAR_COUNT, STARS_VERTEX_LAYOUT};
use crate::world::terrain::terrain_subsystem::{Terrain, TerrainRenderSubsystem};
use crate::world::terrain::water::SimpleVertex;
use tracing::error;
use wgpu::CompareFunction::Always;
use wgpu::PrimitiveTopology::TriangleList;
use wgpu::*;
use wgpu_render_manager::pipelines::{FragmentOption, PipelineOptions, ShadowOptions};
use wgpu_render_manager::renderer::RenderManager;

pub struct RenderPassConfig {
    pub background_color: Color,
    pub reversed_z: bool,
}

impl RenderPassConfig {
    pub fn from_settings(settings: &Settings) -> Self {
        Self {
            background_color: Color {
                r: settings.background_color[0] as f64,
                g: settings.background_color[1] as f64,
                b: settings.background_color[2] as f64,
                a: settings.background_color[3] as f64,
            },
            reversed_z: settings.reversed_depth_z,
        }
    }
}

#[inline]
fn load_op_from_optional_clear(clear_color: Option<Color>) -> LoadOp<Color> {
    match clear_color {
        Some(c) => LoadOp::Clear(c),
        None => LoadOp::Load,
    }
}

/// Generic color attachment factory that handles MSAA <-> resolved and optionally clears.
pub fn make_color_attachment<'a>(
    msaa_view: &'a TextureView,
    resolved_view: &'a TextureView,
    msaa_samples: u32,
    clear_color: Option<Color>, // Some(color) -> Clear, None -> Load
) -> RenderPassColorAttachment<'a> {
    let load_op = load_op_from_optional_clear(clear_color);
    if msaa_samples > 1 {
        RenderPassColorAttachment {
            view: msaa_view,
            resolve_target: Some(resolved_view),
            depth_slice: None,
            ops: Operations {
                load: load_op,
                store: StoreOp::Store,
            },
        }
    } else {
        RenderPassColorAttachment {
            view: resolved_view,
            resolve_target: None,
            depth_slice: None,
            ops: Operations {
                load: load_op,
                store: StoreOp::Store,
            },
        }
    }
}

/// Special-case instance attachment factory.
/// If msaa_samples > 1 it will write into `msaa_instance_view` and resolve into `resolved_instance_view`.
/// clear_color controls whether the resolved instance texture is cleared (Some) or left intact (None).
pub fn make_motion_attachment<'a>(
    msaa_instance_view: &'a TextureView,
    resolved_instance_view: &'a TextureView,
    msaa_samples: u32,
    clear: bool,
) -> RenderPassColorAttachment<'a> {
    let clear_color = if clear { Some(Color::BLACK) } else { None };
    make_color_attachment(
        msaa_instance_view,
        resolved_instance_view,
        msaa_samples,
        clear_color,
    )
}

/// Depth attachment factory — clear or load depending on `clear`.
pub fn make_depth_attachment<'a>(
    depth_view: &'a TextureView,
    config: &RenderPassConfig,
    clear: bool,
) -> RenderPassDepthStencilAttachment<'a> {
    let clear_z = if clear {
        // Choose clear value depending on reversed z convention
        if config.reversed_z {
            LoadOp::Clear(0.0)
        } else {
            LoadOp::Clear(1.0)
        }
    } else {
        LoadOp::Load
    };

    RenderPassDepthStencilAttachment {
        view: depth_view,
        depth_ops: Some(Operations {
            load: clear_z,
            store: StoreOp::Store,
        }),
        stencil_ops: Some(Operations {
            load: if clear {
                LoadOp::Clear(0)
            } else {
                LoadOp::Load
            },
            store: StoreOp::Store,
        }),
    }
}

/// World pass: clears the main targets (color + normal + depth).
pub fn create_world_pass<'a>(
    encoder: &'a mut CommandEncoder,
    pipelines: &'a Pipelines,
    config: &'a RenderPassConfig,
    msaa_samples: u32,
    clear: bool,
) -> RenderPass<'a> {
    let color_attachment = make_color_attachment(
        &pipelines.msaa.hdr,
        &pipelines.resolved.hdr,
        msaa_samples,
        if clear {
            Some(config.background_color)
        } else {
            None
        },
    );

    let normal_attachment = make_color_attachment(
        &pipelines.msaa.normal,
        &pipelines.resolved.normal,
        msaa_samples,
        if clear { Some(Color::BLACK) } else { None },
    );

    let motion_attachment = make_motion_attachment(
        &pipelines.post_fx.dummy_motion, // dummy MSAA view (must match msaa_samples)
        &pipelines.post_fx.motion_full,  // resolved motion texture
        msaa_samples,
        clear,
    );

    encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("World Pass"),
        color_attachments: &[
            Some(color_attachment),
            Some(normal_attachment),
            Some(motion_attachment),
        ],
        depth_stencil_attachment: Some(make_depth_attachment(&pipelines.msaa.depth, config, clear)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}
pub fn create_gizmo_text_pass<'a>(
    encoder: &'a mut CommandEncoder,
    pipelines: &'a Pipelines,
    config: &'a RenderPassConfig,
    msaa_samples: u32,
) -> RenderPass<'a> {
    let color_attachment = make_color_attachment(
        &pipelines.msaa.hdr,
        &pipelines.resolved.hdr,
        msaa_samples,
        None,
    );

    encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("Gizmo Text Pass"),
        color_attachments: &[Some(color_attachment)],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}
pub fn create_id_pass<'a>(
    encoder: &'a mut CommandEncoder,
    pipelines: &'a Pipelines,
) -> RenderPass<'a> {
    let id_attachment = RenderPassColorAttachment {
        view: &pipelines.post_fx.rt_instance,
        resolve_target: None, // IMPORTANT: no resolve
        ops: Operations {
            load: LoadOp::Clear(Color::BLACK),
            store: StoreOp::Store,
        },
        depth_slice: None,
    };

    encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("Instance ID Pass"),
        color_attachments: &[Some(id_attachment)],
        depth_stencil_attachment: None,
        // Some(make_depth_attachment(
        //     &pipelines.buffers,
        //     &RenderPassConfig::from_settings(settings),
        //     false,
        // )),
        occlusion_query_set: None,
        timestamp_writes: None,
        multiview_mask: None,
    })
}
// RENDER PASSES

pub fn render_sky<'a>(
    encoder: &'a mut CommandEncoder,
    render_manager: &mut RenderManager,
    profiler: &mut GpuProfiler,
    pipelines: &Pipelines,
    settings: &Settings,
    config: &RenderPassConfig,
    msaa_samples: u32,
    mod_manager: &ModManager,
) {
    let sky_depth_stencil = Some(DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(false),
        depth_compare: if settings.reversed_depth_z {
            Some(CompareFunction::GreaterEqual)
        } else {
            Some(CompareFunction::LessEqual)
        },
        stencil: Default::default(),
        bias: Default::default(),
    });
    let targets = color_and_normals_and_motion_targets(pipelines);
    let Some(shader) = mod_manager.resource_path("shaders/stars.wgsl") else {
        error!("[Renderer] Missing shader 'shaders/stars.wgsl'");
        return;
    };
    gpu_timestamp!(encoder, profiler, "Stars", {
        let pass = &mut create_world_pass(encoder, pipelines, config, msaa_samples, false);
        //pass.draw_indexed_indirect()
        // Stars
        render_manager.render(
            &[],
            shader,
            &PipelineOptions {
                topology: PrimitiveTopology::TriangleStrip,
                depth_stencil: sky_depth_stencil.clone(),
                multisample_state: multisample_state(msaa_samples),
                vertex_layouts: Vec::from([Some(STARS_VERTEX_LAYOUT)]),
                fragment: FragmentOption::Default {
                    targets: targets.clone(),
                },
                ..Default::default()
            },
            &[&pipelines.buffers.camera, &pipelines.buffers.sky],
            pass,
        );
        pass.set_vertex_buffer(0, pipelines.resources.stars_meshes.vertex.slice(..));
        pass.draw(0..4, 0..STAR_COUNT);
    });
    let Some(shader) = mod_manager.resource_path("shaders/sky.wgsl") else {
        error!("[Renderer] Missing shader 'shaders/sky.wgsl'");
        return;
    };
    gpu_timestamp!(encoder, profiler, "Sky", {
        let pass = &mut create_world_pass(encoder, pipelines, config, msaa_samples, false);
        // Sky
        render_manager.render(
            &[],
            shader,
            &PipelineOptions {
                topology: Default::default(),
                depth_stencil: sky_depth_stencil,
                multisample_state: multisample_state(msaa_samples),
                vertex_layouts: Vec::new(),
                fragment: FragmentOption::Default { targets },
                ..Default::default()
            },
            &[&pipelines.buffers.camera, &pipelines.buffers.sky],
            pass,
        );
        pass.draw(0..3, 0..1);
    });
}
pub fn render_terrain<'a>(
    encoder: &'a mut CommandEncoder,
    render_manager: &mut RenderManager,
    terrain_renderer: &TerrainRenderSubsystem,
    terrain_subsystem: &Terrain,
    pipelines: &Pipelines,
    settings: &Settings,
    config: &RenderPassConfig,
    msaa_samples: u32,
    camera: &Camera,
    aspect: f32,
    mod_manager: &ModManager,
) {
    let pass = &mut create_world_pass(encoder, pipelines, config, msaa_samples, false);
    let keys = terrain_material_keys(mod_manager);
    let Some(shader) = mod_manager.resource_path("shaders/terrain.wgsl") else {
        error!("[Renderer] Missing shader 'shaders/terrain.wgsl'");
        return;
    };
    let shadow = make_shadow_option(settings, pipelines);

    let make_stencil = |write_mask: u32| -> DepthStencilState {
        DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: if settings.reversed_depth_z {
                Some(CompareFunction::GreaterEqual)
            } else {
                Some(CompareFunction::LessEqual)
            },
            stencil: StencilState {
                front: StencilFaceState {
                    compare: Always,
                    fail_op: StencilOperation::Keep,
                    depth_fail_op: StencilOperation::Keep,
                    pass_op: StencilOperation::Replace,
                },
                back: StencilFaceState {
                    compare: Always,
                    fail_op: StencilOperation::Keep,
                    depth_fail_op: StencilOperation::Keep,
                    pass_op: StencilOperation::Replace,
                },
                read_mask: 0xFF,
                write_mask,
            },
            bias: Default::default(),
        }
    };
    let targets = color_and_normals_and_motion_targets(pipelines);

    // Terrain Pipeline (Underwater)
    pass.set_stencil_reference(1);
    render_manager.render(
        keys.as_slice(),
        shader,
        &PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(make_stencil(0xFF)),
            multisample_state: multisample_state(msaa_samples),
            vertex_layouts: Vec::from([Some(Vertex::desc())]),
            cull_mode: Some(Face::Front),
            fragment: FragmentOption::Default {
                targets: targets.clone(),
            },
            shadow: shadow.clone(),
            ..Default::default()
        },
        &[&pipelines.buffers.camera, &pipelines.buffers.pick],
        pass,
    );
    terrain_renderer.render(pass, terrain_subsystem, camera, aspect, settings, true);

    // Terrain Pipeline (Above Water)
    pass.set_stencil_reference(0);
    render_manager.render(
        keys.as_slice(),
        shader,
        &PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(make_stencil(0)),
            multisample_state: multisample_state(msaa_samples),
            vertex_layouts: Vec::from([Some(Vertex::desc())]),
            cull_mode: Some(Face::Front),
            fragment: FragmentOption::Default { targets },
            shadow,
            ..Default::default()
        },
        &[&pipelines.buffers.camera, &pipelines.buffers.pick],
        pass,
    );
    terrain_renderer.render(pass, terrain_subsystem, camera, aspect, settings, false);
}
pub fn render_water<'a>(
    encoder: &'a mut CommandEncoder,
    render_manager: &mut RenderManager,
    pipelines: &Pipelines,
    _settings: &Settings,
    config: &RenderPassConfig,
    msaa_samples: u32,
    mod_manager: &ModManager,
) {
    let pass = &mut create_world_pass(encoder, pipelines, config, msaa_samples, false);
    let targets = color_and_normals_and_motion_targets(pipelines);
    let Some(shader) = mod_manager.resource_path("shaders/water.wgsl") else {
        error!("[Renderer] Missing shader 'shaders/water.wgsl'");
        return;
    };
    // Water
    pass.set_stencil_reference(1);
    render_manager.render(
        &[],
        shader,
        &PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(Always),
                stencil: StencilState {
                    front: StencilFaceState {
                        compare: CompareFunction::Equal,
                        fail_op: StencilOperation::Keep,
                        depth_fail_op: StencilOperation::Keep,
                        pass_op: StencilOperation::Keep,
                    },
                    back: StencilFaceState {
                        compare: CompareFunction::Equal,
                        fail_op: StencilOperation::Keep,
                        depth_fail_op: StencilOperation::Keep,
                        pass_op: StencilOperation::Keep,
                    },
                    read_mask: 0xFF,
                    write_mask: 0x00,
                },
                bias: Default::default(),
            }),
            multisample_state: multisample_state(msaa_samples),
            vertex_layouts: Vec::from([Some(SimpleVertex::layout())]),
            fragment: FragmentOption::Default { targets },
            ..Default::default()
        },
        &[
            &pipelines.buffers.camera,
            &pipelines.buffers.water,
            &pipelines.buffers.sky,
        ],
        pass,
    );

    pass.set_vertex_buffer(0, pipelines.resources.water_meshes.vertex.slice(..));
    pass.set_index_buffer(
        pipelines.resources.water_meshes.index.slice(..),
        IndexFormat::Uint32,
    );
    pass.draw_indexed(0..pipelines.resources.water_meshes.index_count, 0, 0..1);
}
pub fn render_roads<'a>(
    encoder: &'a mut CommandEncoder,
    render_manager: &mut RenderManager,
    road_renderer: &RoadRenderSubsystem,
    pipelines: &Pipelines,
    settings: &Settings,
    config: &RenderPassConfig,
    msaa_samples: u32,
    mod_manager: &ModManager,
) {
    let pass = &mut create_world_pass(encoder, pipelines, config, msaa_samples, false);
    let keys = road_material_keys(mod_manager);
    let Some(shader) = mod_manager.resource_path("shaders/road.wgsl") else {
        error!("[Renderer] Missing shader 'shaders/road.wgsl'");
        return;
    };
    let shadow = make_shadow_option(settings, pipelines);

    fn road_bias(settings: &Settings, constant: i32, slope: f32) -> DepthBiasState {
        let sign_i = if settings.reversed_depth_z { 1 } else { -1 };
        let sign_f = sign_i as f32;

        DepthBiasState {
            constant: sign_i * constant.abs(),
            slope_scale: sign_f * slope.abs(),
            clamp: 0.0,
        }
    }
    let base_bias = road_bias(settings, 3, 2.0);
    let preview_bias = road_bias(settings, 4, 2.0);
    let targets = color_and_normals_and_motion_targets(pipelines);

    // Roads
    render_manager.render(
        keys.as_slice(),
        shader,
        &PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(depth_stencil(base_bias, settings)),
            multisample_state: multisample_state(msaa_samples),
            vertex_layouts: Vec::from([Some(AdvancedVertex::layout())]),
            cull_mode: Some(Face::Back),
            fragment: FragmentOption::Default {
                targets: targets.clone(),
            },
            shadow: shadow.clone(),
            ..Default::default()
        },
        &[
            &pipelines.buffers.camera,
            &road_renderer.road_appearance.normal_buffer,
        ],
        pass,
    );

    draw_visible_roads(pass, road_renderer);

    if road_renderer.preview_gpu.is_empty() {
        return;
    }
    let (Some(vb), Some(ib)) = (&road_renderer.preview_gpu.vb, &road_renderer.preview_gpu.ib)
    else {
        return;
    };

    // Preview Roads
    render_manager.render(
        keys.as_slice(),
        shader,
        &PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(Always),
                stencil: Default::default(),
                bias: preview_bias,
            }),
            multisample_state: multisample_state(msaa_samples),
            vertex_layouts: Vec::from([Some(AdvancedVertex::layout())]),
            cull_mode: Some(Face::Back),
            fragment: FragmentOption::Default { targets },
            shadow,
            ..Default::default()
        },
        &[
            &pipelines.buffers.camera,
            &road_renderer.road_appearance.preview_buffer,
        ],
        pass,
    );

    pass.set_vertex_buffer(0, vb.slice(..));
    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
    pass.draw_indexed(0..road_renderer.preview_gpu.index_count, 0, 0..1);
}

pub fn render_buildings<'a>(
    encoder: &'a mut CommandEncoder,
    render_manager: &mut RenderManager,
    terrain: &Terrain,
    buildings: &Buildings,
    building_renderer: &mut BuildingRenderer,
    pipelines: &Pipelines,
    settings: &Settings,
    config: &RenderPassConfig,
    msaa_samples: u32,
    mod_manager: &ModManager,
) {
    let pass = &mut create_world_pass(encoder, pipelines, config, msaa_samples, false);
    let Some(shader) = mod_manager.resource_path("shaders/buildings.wgsl") else {
        error!("[Renderer] Missing shader 'shaders/buildings.wgsl'");
        return;
    };
    let shadow = make_shadow_option(settings, pipelines);

    fn building_bias(settings: &Settings, constant: i32, slope: f32) -> DepthBiasState {
        let sign_i = if settings.reversed_depth_z { 1 } else { -1 };
        let sign_f = sign_i as f32;

        DepthBiasState {
            constant: sign_i * constant.abs(),
            slope_scale: sign_f * slope.abs(),
            clamp: 0.0,
        }
    }
    let base_bias = building_bias(settings, 3, 2.0);
    let targets = color_and_normals_and_motion_targets(pipelines);
    // Buildings
    render_manager.render(
        &[],
        shader,
        &PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(depth_stencil(base_bias, settings)),
            multisample_state: multisample_state(msaa_samples),
            vertex_layouts: Vec::from([Some(BuildingVertex::layout())]),
            cull_mode: Some(Face::Back),
            fragment: FragmentOption::Default {
                targets: targets.clone(),
            },
            shadow,
            ..Default::default()
        },
        &[&pipelines.buffers.camera],
        pass,
    );

    draw_visible_buildings(pass, terrain, building_renderer);

    if terrain.cursor.preview_building.is_some() {
        let shadow = make_shadow_option(settings, pipelines);

        // Reference used both to stamp the bit (Pass 1) and to test for it (Pass 2).
        pass.set_stencil_reference(PREVIEW_STENCIL_BIT);

        // Pass 1: always beats terrain, writes depth, stamps preview bit.
        render_manager.render(
            &[],
            shader,
            &PipelineOptions {
                topology: TriangleList,
                depth_stencil: Some(preview_establish_depth_stencil()),
                multisample_state: multisample_state(msaa_samples),
                vertex_layouts: Vec::from([Some(BuildingVertex::layout())]),
                cull_mode: Some(Face::Back),
                fragment: FragmentOption::Default {
                    targets: targets.clone(),
                },
                shadow: shadow.clone(),
                ..Default::default()
            },
            &[&pipelines.buffers.camera],
            pass,
        );
        draw_preview_building(pass, terrain);

        // Pass 2: correct self-occlusion via a real depth test, restricted
        // to pixels Pass 1 just touched.
        render_manager.render(
            &[],
            shader,
            &PipelineOptions {
                topology: TriangleList,
                depth_stencil: Some(preview_resolve_depth_stencil(settings)),
                multisample_state: multisample_state(msaa_samples),
                vertex_layouts: Vec::from([Some(BuildingVertex::layout())]),
                cull_mode: Some(Face::Back),
                fragment: FragmentOption::Default { targets },
                shadow,
                ..Default::default()
            },
            &[&pipelines.buffers.camera],
            pass,
        );
        // I don't like drawing twice, ChatGPT...
        draw_preview_building(pass, terrain);
    }
}
/// Bit used by the building preview to mark "a preview fragment already
/// wrote here this frame". Must not overlap terrain's underwater bit (0x01).
const PREVIEW_STENCIL_BIT: u32 = 0x02;

/// Pass 1: preview vs. terrain. Always wins, writes depth, stamps the
/// preview stencil bit. Does not touch terrain's bit (write_mask restricts
/// the Replace op to bit 1 only).
fn preview_establish_depth_stencil() -> DepthStencilState {
    DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(true),
        depth_compare: Some(CompareFunction::Always),
        stencil: StencilState {
            front: StencilFaceState {
                compare: CompareFunction::Always,
                fail_op: StencilOperation::Keep,
                depth_fail_op: StencilOperation::Keep,
                pass_op: StencilOperation::Replace,
            },
            back: StencilFaceState {
                compare: CompareFunction::Always,
                fail_op: StencilOperation::Keep,
                depth_fail_op: StencilOperation::Keep,
                pass_op: StencilOperation::Replace,
            },
            read_mask: PREVIEW_STENCIL_BIT,
            write_mask: PREVIEW_STENCIL_BIT,
        },
        bias: DepthBiasState::default(),
    }
}

/// Pass 2: preview vs. itself. Real depth compare, only active on pixels
/// Pass 1 actually touched (Equal against the preview bit). Converges to
/// the true nearest fragment via the normal z-buffer algorithm. Never
/// writes stencil again — Pass 1 already stamped it correctly.
fn preview_resolve_depth_stencil(settings: &Settings) -> DepthStencilState {
    DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(true),
        depth_compare: depth_stencil(DepthBiasState::default(), settings).depth_compare,
        stencil: StencilState {
            front: StencilFaceState {
                compare: CompareFunction::Equal,
                fail_op: StencilOperation::Keep,
                depth_fail_op: StencilOperation::Keep,
                pass_op: StencilOperation::Keep,
            },
            back: StencilFaceState {
                compare: CompareFunction::Equal,
                fail_op: StencilOperation::Keep,
                depth_fail_op: StencilOperation::Keep,
                pass_op: StencilOperation::Keep,
            },
            read_mask: PREVIEW_STENCIL_BIT,
            write_mask: 0x00,
        },
        bias: DepthBiasState::default(),
    }
}
pub fn render_gizmo<'a>(
    encoder: &'a mut CommandEncoder,
    render_manager: &mut RenderManager,
    pipelines: &Pipelines,
    _settings: &Settings,
    config: &RenderPassConfig,
    msaa_samples: u32,
    gizmo: &mut Gizmo,
    camera: &Camera,
    device: &Device,
    queue: &Queue,
    mod_manager: &ModManager,
    ui: &mut UiRenderer,
) {
    let batches = gizmo.collect_batches(camera);

    let (thin_count, thick_count, filled_count) = gizmo.update_buffers(device, queue, &batches);

    let Some(gb) = gizmo.gizmo_buffers.as_mut() else {
        gizmo.clear();
        return;
    };

    {
        let pass = &mut create_world_pass(encoder, pipelines, config, msaa_samples, false);

        if thin_count > 0 {
            let Some(shader) = mod_manager.resource_path("shaders/lines.wgsl") else {
                error!("[Renderer] Missing shader 'shaders/lines.wgsl'");
                return;
            };

            render_manager.render(
                &[],
                shader,
                &PipelineOptions {
                    topology: PrimitiveTopology::LineList,
                    depth_stencil: Some(DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: Some(false),
                        depth_compare: Some(Always),
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample_state: multisample_state(msaa_samples),
                    vertex_layouts: Vec::from([Some(ThinLineVtxRender::layout())]),
                    fragment: FragmentOption::Default {
                        targets: color_and_normals_and_motion_targets(pipelines),
                    },
                    ..Default::default()
                },
                &[&pipelines.buffers.camera],
                pass,
            );

            pass.set_vertex_buffer(0, gb.thin_buffer.slice(..));
            pass.draw(0..thin_count, 0..1);
        }

        if thick_count > 0 {
            let Some(shader) = mod_manager.resource_path("shaders/thick_lines.wgsl") else {
                error!("[Renderer] Missing shader 'shaders/thick_lines.wgsl'");
                return;
            };

            render_manager.render(
                &[],
                shader,
                &PipelineOptions {
                    topology: PrimitiveTopology::TriangleList,
                    depth_stencil: Some(DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: Some(false),
                        depth_compare: Some(Always),
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample_state: multisample_state(msaa_samples),
                    vertex_layouts: Vec::from([Some(ThickLineVtxRender::layout())]),
                    fragment: FragmentOption::Default {
                        targets: color_and_normals_and_motion_targets(pipelines),
                    },
                    cull_mode: None,
                    ..Default::default()
                },
                &[&pipelines.buffers.camera],
                pass,
            );

            pass.set_vertex_buffer(0, gb.thick_buffer.slice(..));
            pass.draw(0..thick_count, 0..1);
        }

        if filled_count > 0 {
            let Some(shader) = mod_manager.resource_path("shaders/lines.wgsl") else {
                error!("[Renderer] Missing shader 'shaders/lines.wgsl'");
                return;
            };

            render_manager.render(
                &[],
                shader,
                &PipelineOptions {
                    topology: PrimitiveTopology::TriangleList,
                    depth_stencil: Some(DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: Some(false),
                        depth_compare: Some(Always),
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample_state: multisample_state(msaa_samples),
                    vertex_layouts: Vec::from([Some(ThinLineVtxRender::layout())]),
                    fragment: FragmentOption::Default {
                        targets: color_and_normals_and_motion_targets(pipelines),
                    },
                    ..Default::default()
                },
                &[&pipelines.buffers.camera],
                pass,
            );

            pass.set_vertex_buffer(0, gb.filled_buffer.slice(..));
            pass.draw(0..filled_count, 0..1);
        }
    }
    let text_ready = gizmo.prepare_text(
        camera,
        device,
        queue,
        encoder,
        &ui.viewport,
        &mut ui.text_atlas,
        &mut ui.font_system,
        pipelines.config.width,
        pipelines.config.height,
    );
    if text_ready {
        let Some(gb) = gizmo.gizmo_buffers.as_mut() else {
            gizmo.clear();
            error!("[Renderer] Gizmo buffers is empty");
            return;
        };

        let mut pass = create_gizmo_text_pass(encoder, pipelines, config, msaa_samples);
        if let Err(error) = gb
            .text_renderer
            .render(&ui.text_atlas, &ui.viewport, &mut pass)
        {
            error!("Failed to render gizmo text: {}", error);
        }
    }

    gizmo.clear();
}

pub fn render_cars<'a>(
    encoder: &'a mut CommandEncoder,
    render_manager: &mut RenderManager,
    rt_subsystem: &mut RTSubsystem,
    car_renderer: &mut CarRenderSubsystem,
    car_storage: &CarStorage,
    pipelines: &Pipelines,
    settings: &Settings,
    camera: &Camera,
    config: &RenderPassConfig,
    mod_manager: &ModManager,
) {
    let pass = &mut create_world_pass(encoder, pipelines, config, settings.msaa_samples, false);
    let keys = cars_material_keys(mod_manager);
    let Some(shader) = mod_manager.resource_path("shaders/car.wgsl") else {
        error!("[Renderer] Missing shader 'shaders/car.wgsl'");
        return;
    };
    let shadow = make_shadow_option(settings, pipelines);

    let targets = color_and_normals_and_motion_targets(pipelines);
    // Cars
    render_manager.render(
        keys.as_slice(),
        shader,
        &PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(depth_stencil(Default::default(), settings)),
            multisample_state: multisample_state(settings.msaa_samples),
            vertex_layouts: Vec::from([Some(CarVertex::layout()), Some(CarInstance::layout())]),
            cull_mode: Some(Face::Back),
            fragment: FragmentOption::Default { targets },
            shadow: shadow.clone(),
            ..Default::default()
        },
        &[&pipelines.buffers.camera],
        pass,
    );

    car_renderer.render(pipelines, rt_subsystem, car_storage, camera, pass);
}
pub fn render_instance_ids<'a>(
    pass: &mut RenderPass<'a>,
    render_manager: &mut RenderManager,
    pipelines: &Pipelines,
    car_renderer: &mut CarRenderSubsystem,
    settings: &Settings,
    camera: &'a Camera,
    props: &'a mut Props,
    terrain: &'a Terrain,
    mod_manager: &ModManager,
) {
    let Some(shader) = mod_manager.resource_path("shaders/car_instance_id.wgsl") else {
        error!("[Renderer] Missing shader 'shaders/car_instance_id.wgsl'");
        return;
    };

    render_manager.render(
        cars_material_keys(mod_manager).as_slice(),
        shader,
        &PipelineOptions {
            topology: TriangleList,
            depth_stencil: None,
            multisample_state: multisample_state(1), // IMPORTANT MSAA = 1!
            vertex_layouts: vec![Some(CarVertex::layout()), Some(CarInstance::layout())],
            cull_mode: Some(Face::Back),
            fragment: FragmentOption::Default {
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::R32Uint,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            },
            ..Default::default()
        },
        &[&pipelines.buffers.camera],
        pass,
    );

    car_renderer.render_last(pass);
    let Some(shader) = mod_manager.resource_path("shaders/props_instance_id.wgsl") else {
        error!("[Renderer] Missing shader 'shaders/props_instance_id.wgsl'");
        return;
    };
    let shadow = make_shadow_option(settings, pipelines);
    //let targets = color_and_normals_and_motion_targets(pipelines);

    let opts = PipelineOptions {
        topology: TriangleList,
        depth_stencil: None,
        multisample_state: multisample_state(1), // IMPORTANT MSAA = 1!
        vertex_layouts: Vec::from([Some(PropVertex::layout()), Some(GpuPropInstance::layout())]),
        cull_mode: Some(Face::Back),
        fragment: FragmentOption::Default {
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::R32Uint,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        },
        shadow: shadow.clone(),
        ..Default::default()
    };
    props.render(
        render_manager,
        pass,
        shader,
        opts,
        camera,
        terrain,
        pipelines,
        settings,
    );
}

pub fn render_props<'a>(
    encoder: &'a mut CommandEncoder,
    render_manager: &mut RenderManager,
    props: &'a mut Props,
    pipelines: &Pipelines,
    settings: &Settings,
    camera: &'a Camera,
    terrain: &'a Terrain,
    device: &Device,
    queue: &Queue,
    config: &RenderPassConfig,
    mod_manager: &ModManager,
) {
    let pass = &mut create_world_pass(encoder, pipelines, config, settings.msaa_samples, false);
    let Some(shader) = mod_manager.resource_path("shaders/props.wgsl") else {
        error!("[Renderer] Missing shader 'shaders/props.wgsl'");
        return;
    };
    let shadow = make_shadow_option(settings, pipelines);
    let targets = color_and_normals_and_motion_targets(pipelines);

    let opts = PipelineOptions {
        topology: TriangleList,
        depth_stencil: Some(depth_stencil(Default::default(), settings)),
        multisample_state: MultisampleState {
            count: settings.msaa_samples,
            mask: !0,
            alpha_to_coverage_enabled: false, //settings.msaa_samples > 2
        },
        vertex_layouts: Vec::from([Some(PropVertex::layout()), Some(GpuPropInstance::layout())]),
        cull_mode: Some(Face::Back),
        fragment: FragmentOption::Default {
            targets: targets.clone(),
        },
        shadow: shadow.clone(),
        ..Default::default()
    };

    // Draw all props
    props.render(
        render_manager,
        pass,
        shader,
        opts,
        camera,
        terrain,
        pipelines,
        settings,
    );
}

pub fn make_shadow_option(settings: &Settings, pipelines: &Pipelines) -> Option<ShadowOptions> {
    match settings.shadow_type {
        ShadowType::CSM => match settings.reversed_depth_z {
            true => Some(ShadowOptions {
                sampler: pipelines
                    .resources
                    .shadow_samplers
                    .shadow_sampler_rev_z
                    .clone(),
                view: pipelines.resources.csm_shadows.array_view.clone(),
            }),
            false => Some(ShadowOptions {
                sampler: pipelines.resources.shadow_samplers.shadow_sampler.clone(),
                view: pipelines.resources.csm_shadows.array_view.clone(),
            }),
        },
        _ => Some(ShadowOptions {
            sampler: pipelines
                .resources
                .shadow_samplers
                .shadow_sampler_off
                .clone(),
            view: pipelines.resources.csm_shadows.array_view.clone(),
        }),
    }
}

fn color_and_normals_and_motion_targets(pipelines: &Pipelines) -> Vec<Option<ColorTargetState>> {
    vec![
        Some(ColorTargetState {
            format: pipelines.msaa.hdr.texture().format(),
            blend: Some(BlendState::ALPHA_BLENDING),
            write_mask: ColorWrites::ALL,
        }),
        Some(ColorTargetState {
            format: pipelines.msaa.normal.texture().format(),
            blend: None,
            write_mask: ColorWrites::ALL,
        }),
        Some(ColorTargetState {
            format: pipelines.post_fx.motion_full.texture().format(),
            blend: None,
            write_mask: ColorWrites::ALL,
        }),
    ]
}

pub fn color_target(
    pipelines: &Pipelines,
    blend: Option<BlendState>,
) -> Vec<Option<ColorTargetState>> {
    vec![Some(ColorTargetState {
        format: pipelines.msaa.hdr.texture().format(),
        blend,
        write_mask: ColorWrites::ALL,
    })]
}
pub fn color_target_ui(
    pipelines: &Pipelines,
    blend: Option<BlendState>,
) -> Vec<Option<ColorTargetState>> {
    vec![Some(ColorTargetState {
        format: pipelines.resolved.ui.texture().format(),
        blend,
        write_mask: ColorWrites::ALL,
    })]
}
pub fn depth_stencil(bias: DepthBiasState, settings: &Settings) -> DepthStencilState {
    DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(true),
        depth_compare: if settings.reversed_depth_z {
            Some(CompareFunction::GreaterEqual)
        } else {
            Some(CompareFunction::LessEqual)
        },
        stencil: Default::default(),
        bias,
    }
}
pub fn depth_stencil_with_stencil(
    bias: DepthBiasState,
    stencil: StencilState,
    settings: &Settings,
) -> DepthStencilState {
    DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(true),
        depth_compare: if settings.reversed_depth_z {
            Some(CompareFunction::GreaterEqual)
        } else {
            Some(CompareFunction::LessEqual)
        },
        stencil,
        bias,
    }
}
pub fn draw_visible_roads(pass: &mut RenderPass, road_renderer: &RoadRenderSubsystem) {
    for chunk_coord in &road_renderer.visible_draw_list {
        if let Some(gpu) = road_renderer.chunk_gpu.get(chunk_coord) {
            pass.set_vertex_buffer(0, gpu.vertex.slice(..));
            pass.set_index_buffer(gpu.index.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..gpu.index_count, 0, 0..1);
        }
    }
}
pub fn draw_visible_buildings(
    pass: &mut RenderPass,
    terrain: &Terrain,
    building_renderer: &BuildingRenderer,
) {
    for chunk_coord in terrain.visible.iter().map(|v| v.chunk_coord) {
        if let Some(gpu) = building_renderer.chunk_gpu.get(&chunk_coord) {
            pass.set_vertex_buffer(0, gpu.vertex.slice(..));
            pass.set_index_buffer(gpu.index.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..gpu.index_count, 0, 0..1);
        }
    }
}
pub fn draw_preview_building(pass: &mut RenderPass, terrain: &Terrain) {
    if let Some(model) = terrain
        .cursor
        .preview_building
        .as_ref()
        .and_then(|pb| pb.cached_model.as_ref())
    {
        pass.set_vertex_buffer(0, model.vertex.slice(..));
        pass.set_index_buffer(model.index.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..model.mesh.indices.len() as u32, 0, 0..1);
    }
}
