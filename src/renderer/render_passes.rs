use crate::cars::car_mesh::CarVertex;
use crate::cars::car_render::CarInstance;
use crate::cars::car_structs::CarStorage;
use crate::cars::car_subsystem::CarRenderSubsystem;
use crate::data::Settings;
use crate::gpu_timestamp;
use crate::helpers::paths::shader_dir;
use crate::renderer::gizmo::Gizmo;
use crate::renderer::gpu_profiler::GpuProfiler;
use crate::renderer::pipelines::{DEPTH_FORMAT, Pipelines};
use crate::renderer::ray_tracing::rt_subsystem::RTSubsystem;
use crate::renderer::terrain_subsystem::{TerrainRenderSubsystem, TerrainSubsystem};
use crate::renderer::textures::material_keys::*;
use crate::terrain::roads::road_mesh_manager::RoadVertex;
use crate::terrain::roads::road_subsystem::RoadRenderSubsystem;
use crate::terrain::sky::{STAR_COUNT, STARS_VERTEX_LAYOUT};
use crate::terrain::water::SimpleVertex;
use crate::ui::vertex::{LineVtxRender, Vertex};
use crate::world::camera::Camera;
use wgpu::PrimitiveTopology::TriangleList;
use wgpu::*;
use wgpu_render_manager::pipelines::{PipelineOptions, ShadowOptions};
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

pub fn create_color_attachment<'a>(
    msaa_hdr_view: &'a TextureView,
    resolved_hdr_view: &'a TextureView,
    msaa_samples: u32,
    background_color: Color,
) -> RenderPassColorAttachment<'a> {
    if msaa_samples > 1 {
        RenderPassColorAttachment {
            view: msaa_hdr_view,
            resolve_target: Some(resolved_hdr_view),
            depth_slice: None,
            ops: Operations {
                load: LoadOp::Clear(background_color),
                store: StoreOp::Store,
            },
        }
    } else {
        RenderPassColorAttachment {
            view: resolved_hdr_view,
            resolve_target: None,
            depth_slice: None,
            ops: Operations {
                load: LoadOp::Clear(background_color),
                store: StoreOp::Store,
            },
        }
    }
}
pub fn create_normal_attachment<'a>(
    msaa_normal_view: &'a TextureView,
    resolved_normal_view: &'a TextureView,
    msaa_samples: u32,
) -> RenderPassColorAttachment<'a> {
    if msaa_samples > 1 {
        RenderPassColorAttachment {
            view: msaa_normal_view,
            resolve_target: Some(resolved_normal_view),
            depth_slice: None,
            ops: Operations {
                load: LoadOp::Clear(Color::BLACK),
                store: StoreOp::Store,
            },
        }
    } else {
        RenderPassColorAttachment {
            view: resolved_normal_view,
            resolve_target: None,
            depth_slice: None,
            ops: Operations {
                load: LoadOp::Clear(Color::BLACK),
                store: StoreOp::Store,
            },
        }
    }
}
pub fn create_depth_attachment<'a>(
    depth_view: &'a TextureView,
    config: &RenderPassConfig,
) -> RenderPassDepthStencilAttachment<'a> {
    let clear_z = LoadOp::Clear(if config.reversed_z { 0.0 } else { 1.0 });
    RenderPassDepthStencilAttachment {
        view: depth_view,
        depth_ops: Some(Operations {
            load: clear_z,
            store: StoreOp::Store,
        }),
        stencil_ops: Some(Operations {
            load: LoadOp::Clear(0),
            store: StoreOp::Store,
        }),
    }
}

pub fn create_world_pass<'a>(
    encoder: &'a mut CommandEncoder,
    pipelines: &'a Pipelines,
    config: &'a RenderPassConfig,
    msaa_samples: u32,
) -> RenderPass<'a> {
    let color_attachment = create_color_attachment(
        &pipelines.msaa.hdr,
        &pipelines.resolved.hdr,
        msaa_samples,
        config.background_color,
    );

    let normal_attachment = create_normal_attachment(
        &pipelines.msaa.normal,
        &pipelines.resolved.normal,
        msaa_samples,
    );

    encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("World Pass"),
        color_attachments: &[Some(color_attachment), Some(normal_attachment)],
        depth_stencil_attachment: Some(create_depth_attachment(&pipelines.msaa.depth, config)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

// RENDER PASSES

pub fn render_sky(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    profiler: &mut GpuProfiler,
    pipelines: &Pipelines,
    settings: &Settings,
    msaa_samples: u32,
) {
    let sky_depth_stencil = Some(DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: false,
        depth_compare: if settings.reversed_depth_z {
            CompareFunction::GreaterEqual
        } else {
            CompareFunction::LessEqual
        },
        stencil: Default::default(),
        bias: Default::default(),
    });
    let targets = color_and_normals_targets(pipelines);
    gpu_timestamp!(pass, profiler, "Stars", {
        // Stars
        render_manager.render(
            &[],
            shader_dir().join("stars.wgsl").as_path(),
            &PipelineOptions {
                topology: PrimitiveTopology::TriangleStrip,
                depth_stencil: sky_depth_stencil.clone(),
                msaa_samples,
                vertex_layouts: Vec::from([STARS_VERTEX_LAYOUT]),
                targets: targets.clone(),
                ..Default::default()
            },
            &[&pipelines.buffers.camera, &pipelines.buffers.sky],
            pass,
        );
        pass.set_vertex_buffer(0, pipelines.resources.stars_meshes.vertex.slice(..));
        pass.draw(0..4, 0..STAR_COUNT);
    });

    gpu_timestamp!(pass, profiler, "Sky", {
        // Sky
        render_manager.render(
            &[],
            shader_dir().join("sky.wgsl").as_path(),
            &PipelineOptions {
                topology: Default::default(),
                depth_stencil: sky_depth_stencil,
                msaa_samples,
                vertex_layouts: Vec::new(),
                targets,
                ..Default::default()
            },
            &[&pipelines.buffers.camera, &pipelines.buffers.sky],
            pass,
        );
        pass.draw(0..3, 0..1);
    });
}
pub fn render_terrain(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    terrain_renderer: &TerrainRenderSubsystem,
    terrain_subsystem: &TerrainSubsystem,
    pipelines: &Pipelines,
    settings: &Settings,
    msaa_samples: u32,
    camera: &Camera,
    aspect: f32,
) {
    let keys = terrain_material_keys();
    let shader_path = shader_dir().join("terrain.wgsl");
    let shadow = make_shadow_option(settings, pipelines);

    let make_stencil = |write_mask: u32| -> DepthStencilState {
        DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: if settings.reversed_depth_z {
                CompareFunction::GreaterEqual
            } else {
                CompareFunction::LessEqual
            },
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
                read_mask: 0xFF,
                write_mask,
            },
            bias: Default::default(),
        }
    };
    let targets = color_and_normals_targets(pipelines);

    // Terrain Pipeline (Underwater)
    pass.set_stencil_reference(1);
    render_manager.render(
        keys.as_slice(),
        shader_path.as_path(),
        &PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(make_stencil(0xFF)),
            msaa_samples,
            vertex_layouts: Vec::from([Vertex::desc()]),
            cull_mode: Some(Face::Front),
            targets: targets.clone(),
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
        shader_path.as_path(),
        &PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(make_stencil(0)),
            msaa_samples,
            vertex_layouts: Vec::from([Vertex::desc()]),
            cull_mode: Some(Face::Front),
            targets,
            shadow,
            ..Default::default()
        },
        &[&pipelines.buffers.camera, &pipelines.buffers.pick],
        pass,
    );
    terrain_renderer.render(pass, terrain_subsystem, camera, aspect, settings, false);
}
pub fn render_water(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    pipelines: &Pipelines,
    _settings: &Settings,
    msaa_samples: u32,
) {
    let targets = color_and_normals_targets(pipelines);

    // Water
    pass.set_stencil_reference(1);
    render_manager.render(
        &[],
        shader_dir().join("water.wgsl").as_path(),
        &PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Always,
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
            msaa_samples,
            vertex_layouts: Vec::from([SimpleVertex::layout()]),
            targets,
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
pub fn render_roads(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    road_renderer: &RoadRenderSubsystem,
    pipelines: &Pipelines,
    settings: &Settings,
    msaa_samples: u32,
) {
    let keys = road_material_keys();
    let shader_path = shader_dir().join("road.wgsl");
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
    let targets = color_and_normals_targets(pipelines);
    // Roads
    render_manager.render(
        keys.as_slice(),
        shader_path.as_path(),
        &PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(depth_stencil(base_bias, settings)),
            msaa_samples,
            vertex_layouts: Vec::from([RoadVertex::layout()]),
            cull_mode: Some(Face::Back),
            targets: targets.clone(),
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
    // Roads
    render_manager.render(
        keys.as_slice(),
        shader_path.as_path(),
        &PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(depth_stencil(preview_bias, settings)),
            msaa_samples,
            vertex_layouts: Vec::from([RoadVertex::layout()]),
            cull_mode: Some(Face::Back),
            targets,
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

pub fn render_gizmo(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    pipelines: &Pipelines,
    _settings: &Settings,
    msaa_samples: u32,
    gizmo: &mut Gizmo,
    camera: &Camera,
    device: &Device,
    queue: &Queue,
) {
    let targets = color_and_normals_targets(pipelines);
    // Gizmo
    render_manager.render(
        &[],
        shader_dir().join("lines.wgsl").as_path(),
        &PipelineOptions {
            topology: PrimitiveTopology::LineList,
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            msaa_samples,
            vertex_layouts: Vec::from([LineVtxRender::layout()]),
            targets,
            ..Default::default()
        },
        &[&pipelines.buffers.camera],
        pass,
    );

    let vertex_count = gizmo.update_buffer(device, queue, camera.eye_world());
    pass.set_vertex_buffer(0, gizmo.gizmo_buffer.slice(..));
    pass.draw(0..vertex_count, 0..1);
    gizmo.clear();
}

pub fn render_cars(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    rt_subsystem: &mut RTSubsystem,
    car_renderer: &mut CarRenderSubsystem,
    car_storage: &CarStorage,
    pipelines: &Pipelines,
    settings: &Settings,
    camera: &Camera,
    msaa_samples: u32,
) {
    let keys = cars_material_keys();
    let shader_path = shader_dir().join("car.wgsl");
    let shadow = make_shadow_option(settings, pipelines);

    let targets = color_and_normals_targets(pipelines);
    // Cars
    render_manager.render(
        keys.as_slice(),
        shader_path.as_path(),
        &PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(depth_stencil(Default::default(), settings)),
            msaa_samples,
            vertex_layouts: Vec::from([CarVertex::layout(), CarInstance::layout()]),
            cull_mode: Some(Face::Back),
            targets: targets.clone(),
            shadow: shadow.clone(),
            ..Default::default()
        },
        &[&pipelines.buffers.camera],
        pass,
    );

    car_renderer.render(rt_subsystem, car_storage, camera, pass);
}

fn make_shadow_option(settings: &Settings, pipelines: &Pipelines) -> Option<ShadowOptions> {
    match settings.shadows_enabled {
        true => match settings.reversed_depth_z {
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
        false => Some(ShadowOptions {
            sampler: pipelines
                .resources
                .shadow_samplers
                .shadow_sampler_off
                .clone(),
            view: pipelines.resources.csm_shadows.array_view.clone(),
        }),
    }
}

fn color_and_normals_targets(pipelines: &Pipelines) -> Vec<Option<ColorTargetState>> {
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

fn depth_stencil(bias: DepthBiasState, settings: &Settings) -> DepthStencilState {
    DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: true,
        depth_compare: if settings.reversed_depth_z {
            CompareFunction::GreaterEqual
        } else {
            CompareFunction::LessEqual
        },
        stencil: Default::default(),
        bias,
    }
}
pub fn draw_visible_roads(pass: &mut RenderPass, road_renderer: &RoadRenderSubsystem) {
    for chunk_id in &road_renderer.visible_draw_list {
        if let Some(gpu) = road_renderer.chunk_gpu.get(chunk_id) {
            pass.set_vertex_buffer(0, gpu.vertex.slice(..));
            pass.set_index_buffer(gpu.index.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..gpu.index_count, 0, 0..1);
        }
    }
}
