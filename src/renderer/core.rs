use crate::components::camera::*;
use crate::data::{DebugViewState, Settings};
use crate::gpu_timestamp;
use crate::mouse_ray::*;
use crate::paths::{shader_dir, texture_dir};
use crate::positions::WorldPos;
use crate::renderer::astronomy::*;
use crate::renderer::gizmo::Gizmo;
use crate::renderer::gpu_profiler::GpuProfiler;
use crate::renderer::pipelines::{Pipelines, SSAO_FORMAT};
use crate::renderer::procedural_bind_group_manager::FullscreenPassType;
use crate::renderer::procedural_render_manager::{
    DepthDebugParams, FullscreenDebugSwizzle, PipelineOptions, RenderManager,
    create_color_attachment_load,
};
use crate::renderer::render_passes::*;
use crate::renderer::shader_watcher::ShaderWatcher;
use crate::renderer::shadows::{
    CSM_CASCADES, ShadowMatUniform, render_roads_shadows, render_terrain_shadows,
};
use crate::renderer::ui::{ScreenUniform, UiRenderer};
use crate::renderer::uniform_updates::UniformUpdater;
use crate::renderer::world_renderer::TerrainRenderer;
use crate::resources::TimeSystem;
use crate::terrain::roads::road_subsystem::RoadRenderSubsystem;
use crate::ui::input::InputState;
use crate::ui::ui_editor::UiButtonLoader;
use crate::ui::variables::update_ui_variables;
use crate::world::CameraBundle;
use glam::Mat4;
use std::sync::Arc;
use wgpu::PrimitiveTopology::TriangleList;
use wgpu::TextureFormat::Rgba8UnormSrgb;
use wgpu::*;
use winit::dpi::PhysicalSize;
use winit::window::Window;

pub struct RenderCore {
    adapter: Adapter,
    pub surface: Surface<'static>,
    pub device: Device,
    pub queue: Queue,
    pub config: SurfaceConfiguration,
    pub msaa_samples: u32,
    pub pipelines: Pipelines,
    ui_renderer: UiRenderer,
    pub terrain_renderer: TerrainRenderer,
    pub road_renderer: RoadRenderSubsystem,
    shader_watcher: Option<ShaderWatcher>,
    render_manager: RenderManager,
    pub gizmo: Gizmo,
    astronomy: AstronomyState,
    profiler: GpuProfiler,
}

impl RenderCore {
    pub fn new(window: Arc<Window>, settings: &Settings, camera: &Camera) -> Self {
        let (surface, adapter, size) = create_surface_and_adapter(window);
        let (config, msaa_samples) = create_surface_config(&surface, &adapter, settings, size);
        let (device, queue) = create_device(&adapter);
        surface.configure(&device, &config);

        let shader_watcher = ShaderWatcher::new().ok();

        let pipelines = Pipelines::new(&device, &config, camera, settings)
            .expect("Failed to create render pipelines");
        let ui_renderer = UiRenderer::new(&device, config.format, size, msaa_samples)
            .expect("Failed to create UI pipelines");
        let terrain_renderer = TerrainRenderer::new(&device, settings);
        let road_renderer = RoadRenderSubsystem::new(&device);
        let render_manager = RenderManager::new(&device, &queue, texture_dir());
        let gizmo = Gizmo::new(&device, terrain_renderer.chunk_size);
        let profiler = GpuProfiler::new(&device, 5, 3);
        Self {
            adapter,
            surface,
            device,
            queue,
            config,
            pipelines,
            msaa_samples,
            ui_renderer,
            terrain_renderer,
            road_renderer,
            shader_watcher,
            render_manager,
            gizmo,
            astronomy: AstronomyState::default(),
            profiler,
        }
    }

    pub(crate) fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.pipelines.resize(&self.config, self.msaa_samples);
        self.render_manager.invalidate_resize_bind_groups();
    }

    pub(crate) fn render(
        &mut self,
        camera_bundle: &mut CameraBundle,
        ui_loader: &mut UiButtonLoader,
        time: &TimeSystem,
        input_state: &mut InputState,
        settings: &Settings,
    ) {
        let aspect = self.config.width as f32 / self.config.height as f32;
        self.update_render(
            camera_bundle,
            ui_loader,
            time,
            input_state,
            settings,
            aspect,
        );

        let camera = &camera_bundle.camera;
        let Some(frame) = acquire_frame(&self.surface, &self.device, &self.config) else {
            return;
        };

        let surface_view = frame.texture.create_view(&TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        self.execute_shadow_pass(&mut encoder, camera, aspect, time, settings);
        self.execute_main_pass(
            &mut encoder,
            &surface_view,
            &RenderPassConfig::from_settings(settings),
            camera,
            aspect,
            time,
            input_state,
            ui_loader,
            settings,
        );

        self.profiler.resolve(&mut encoder);

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        self.profiler
            .end_frame(&self.device, &self.queue, &mut ui_loader.variables);
    }

    pub fn update_render(
        &mut self,
        camera_bundle: &mut CameraBundle,
        ui_loader: &mut UiButtonLoader,
        time: &TimeSystem,
        input_state: &mut InputState,
        settings: &Settings,
        aspect: f32,
    ) {
        let camera = &mut camera_bundle.camera;
        self.check_shader_changes(ui_loader);

        let time_scales = TimeScales::from_game_time(time.total_game_time, settings.always_day);
        let observer = ObserverParams::new(time_scales.day_angle);
        let astronomy = compute_astronomy(&time_scales);

        update_ui_variables(
            ui_loader,
            &time_scales,
            &astronomy,
            observer.obliquity,
            settings,
        );

        let (view, proj, view_proj) = camera.matrices(aspect, settings);
        let ray = WorldRay::from_mouse(
            glam::Vec2::new(input_state.mouse.pos.x, input_state.mouse.pos.y),
            self.config.width as f32,
            self.config.height as f32,
            view,
            proj,
            camera.eye_world(),
            self.terrain_renderer.chunk_size,
        );
        self.terrain_renderer.pick_terrain_point(ray);
        self.render_manager
            .update_depth_params_buffer(DepthDebugParams {
                near: camera.near,
                far: camera.far,
                power: 20.0,
                reversed_z: settings.reversed_depth_z as u32,
                msaa_samples: self.msaa_samples,
                _pad0: 0,
                _pad1: 0,
            });
        self.update_uniforms(
            camera, view, proj, view_proj, &astronomy, time, aspect, settings,
        );

        // upload per-cascade shadow uniforms ONCE (outside encoder)
        for i in 0..CSM_CASCADES {
            let smu = ShadowMatUniform {
                light_view_proj: self.pipelines.cascaded_shadow_map.light_mats[i]
                    .to_cols_array_2d(),
            };
            self.queue.write_buffer(
                &self.pipelines.cascaded_shadow_map.shadow_mat_buffers[i],
                0,
                bytemuck::bytes_of(&smu),
            );
        }
        self.update_subsystems(
            camera,
            aspect,
            settings,
            input_state,
            time,
            ui_loader,
            &astronomy,
        );
        self.astronomy = astronomy;
    }

    fn update_uniforms(
        &mut self,
        camera: &Camera,
        view: Mat4,
        proj: Mat4,
        view_proj: Mat4,
        astronomy: &AstronomyState,
        time: &TimeSystem,
        aspect: f32,
        settings: &Settings,
    ) {
        let mut updater = UniformUpdater::new(&self.queue, &mut self.pipelines);
        updater.update_camera_uniforms(
            view,
            proj,
            view_proj,
            astronomy,
            camera,
            time.total_time,
            aspect,
            settings,
        );
        updater.update_fog_uniforms(&self.config, camera);
        updater.update_sky_uniforms(astronomy.moon_phase);
        updater.update_water_uniforms();
        updater.update_tonemapping_uniforms(&settings.tonemapping_state);
        updater.update_ssao_uniforms(settings);
    }

    fn update_subsystems(
        &mut self,
        camera: &mut Camera,
        aspect: f32,
        settings: &Settings,
        input_state: &mut InputState,
        time: &TimeSystem,
        ui_loader: &mut UiButtonLoader,
        astronomy: &AstronomyState,
    ) {
        let eye = camera.target;
        let target_pos_render = eye.to_render_pos(WorldPos::zero(), camera.chunk_size);
        ui_loader
            .variables
            .set_i32("target_pos_cx", camera.target.chunk.x);
        ui_loader
            .variables
            .set_i32("target_pos_cz", camera.target.chunk.z);
        ui_loader
            .variables
            .set_f32("target_pos_x", target_pos_render.x);
        ui_loader
            .variables
            .set_f32("target_pos_y", target_pos_render.y);
        ui_loader
            .variables
            .set_f32("target_pos_z", target_pos_render.z);
        self.terrain_renderer.update(
            &self.device,
            &self.queue,
            camera,
            aspect,
            settings,
            input_state,
            time,
        );
        self.ui_renderer.update(
            ui_loader,
            time,
            input_state,
            &self.queue,
            &PhysicalSize::new(self.config.width, self.config.height),
        );
        self.road_renderer.update(
            &self.terrain_renderer,
            &self.device,
            &self.queue,
            input_state,
            &self.terrain_renderer.last_picked,
            camera,
            &mut self.gizmo,
        );
        self.gizmo.update(
            &self.terrain_renderer,
            camera.target,
            time.total_game_time,
            &self.road_renderer.road_manager,
            settings,
        );
        self.gizmo
            .update_orbit_gizmo(camera.target, camera.orbit_radius, astronomy.sun_dir, false);
    }

    fn execute_shadow_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        camera: &Camera,
        aspect: f32,
        _time: &TimeSystem,
        settings: &Settings,
    ) {
        for cascade in 0..CSM_CASCADES {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("CSM Shadow Pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.pipelines.cascaded_shadow_map.layer_views[cascade],
                    depth_ops: Some(Operations {
                        load: if settings.reversed_depth_z {
                            LoadOp::Clear(0.0)
                        } else {
                            LoadOp::Clear(1.0)
                        },
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if !settings.shadows_enabled {
                continue;
            }
            let shadow_buf = &self.pipelines.cascaded_shadow_map.shadow_mat_buffers[cascade];

            if cascade != 0 && cascade != 1 {
                render_terrain_shadows(
                    &mut pass,
                    &mut self.render_manager,
                    &self.terrain_renderer,
                    &self.pipelines,
                    settings,
                    camera,
                    aspect,
                    shadow_buf,
                );
            }

            render_roads_shadows(
                &mut pass,
                &mut self.render_manager,
                &self.road_renderer,
                &self.pipelines,
                settings,
                shadow_buf,
            );
        }
    }

    fn execute_main_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        surface_view: &TextureView,
        config: &RenderPassConfig,
        camera: &Camera,
        aspect: f32,
        time: &TimeSystem,
        input_state: &InputState,
        ui_loader: &mut UiButtonLoader,
        settings: &Settings,
    ) {
        // -------- Pass 1: Main world pass (writes depth) --------
        self.execute_world_pass(encoder, config, camera, settings, aspect);

        self.execute_ssao_gen_pass(encoder, settings);
        self.execute_ssao_blur_pass(encoder, settings);
        self.execute_ssao_apply_pass(encoder, settings);

        // -------- Pass 2: Fog (samples depth, no depth attachment) --------
        self.execute_fog_pass(encoder, settings);

        // -------- Pass 3: UI (NOT fogged, because it draws after fog) Logic! --------
        self.execute_ui_pass(encoder, ui_loader, time, input_state);

        // -------- Pass 4: Debug preview (disabled by default) --------
        self.execute_debug_preview_pass(encoder, settings);

        // Last Pass (but no, before debug so the image isn't tampered with! Actually, no)
        self.execute_tonemap_pass(encoder, surface_view, settings);
    }

    fn execute_world_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        config: &RenderPassConfig,
        camera: &Camera,
        settings: &Settings,
        aspect: f32,
    ) {
        // Prepare Terrain Uniforms (Needs queue)
        self.terrain_renderer.make_pick_uniforms(
            &self.queue,
            &self.pipelines.pick_uniforms.buffer,
            camera,
            self.terrain_renderer.chunk_size,
        );

        let mut pass = create_world_pass(encoder, &self.pipelines, config, self.msaa_samples);

        if !settings.show_world {
            return;
        }
        // 1. Sky
        gpu_timestamp!(pass, &mut self.profiler, "Sky", {
            // All frame time names must be lowercase, I decided. (Doesn't matter anyway, cuz I .lowercase() anyway.)
            render_sky(
                &mut pass,
                &mut self.render_manager,
                &self.pipelines,
                settings,
                self.msaa_samples,
            );
        });

        // 2. Terrain
        gpu_timestamp!(pass, &mut self.profiler, "Terrain", {
            render_terrain(
                &mut pass,
                &mut self.render_manager,
                &self.terrain_renderer,
                &self.pipelines,
                settings,
                self.msaa_samples,
                camera,
                aspect,
            );
        });

        // 3. Water
        gpu_timestamp!(pass, &mut self.profiler, "Water", {
            render_water(
                &mut pass,
                &mut self.render_manager,
                &self.pipelines,
                settings,
                self.msaa_samples,
            );
        });

        // 4. Roads
        gpu_timestamp!(pass, &mut self.profiler, "Roads", {
            render_roads(
                &mut pass,
                &mut self.render_manager,
                &self.road_renderer,
                &self.pipelines,
                settings,
                self.msaa_samples,
            );
        });

        gpu_timestamp!(pass, &mut self.profiler, "Gizmo", {
            render_gizmo(
                &mut pass,
                &mut self.render_manager,
                &self.pipelines,
                settings,
                self.msaa_samples,
                &mut self.gizmo,
                camera,
                &self.device,
                &self.queue,
            );
        });
    }
    fn execute_ssao_gen_pass(&mut self, encoder: &mut CommandEncoder, settings: &Settings) {
        if !settings.show_world {
            return;
        }

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("SSAO Gen Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &self.pipelines.ssao_view,
                resolve_target: None,
                depth_slice: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::WHITE), // AO=1 default
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        let shader = if self.msaa_samples > 1 {
            "ssao_gen_msaa.wgsl"
        } else {
            "ssao_gen.wgsl"
        };

        let options = PipelineOptions {
            topology: TriangleList,
            msaa_samples: 1, // output AO is single-sample
            depth_stencil: None,
            vertex_layouts: vec![],
            cull_mode: None,
            shadow_pass: false,
            fullscreen_pass: FullscreenPassType::SsaoGen,
            targets: vec![Some(ColorTargetState {
                format: SSAO_FORMAT,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        };

        self.render_manager.render_fullscreen_pass(
            "SSAO Gen",
            shader_dir().join(shader).as_path(),
            options,
            &[
                &self.pipelines.camera_uniforms.buffer,
                &self.pipelines.ssao_uniforms.buffer,
            ],
            &mut pass,
            &self.pipelines,
            settings,
            FullscreenPassType::SsaoGen,
        );
    }
    fn execute_ssao_blur_pass(&mut self, encoder: &mut CommandEncoder, settings: &Settings) {
        if !settings.show_world {
            return;
        }

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("SSAO Blur Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &self.pipelines.ssao_blur_view,
                resolve_target: None,
                depth_slice: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::WHITE),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        let shader = if self.msaa_samples > 1 {
            "ssao_blur_msaa.wgsl"
        } else {
            "ssao_blur.wgsl"
        };

        let options = PipelineOptions {
            topology: TriangleList,
            msaa_samples: 1,
            depth_stencil: None,
            vertex_layouts: vec![],
            cull_mode: None,
            shadow_pass: false,
            fullscreen_pass: FullscreenPassType::SsaoBlur,
            targets: vec![Some(ColorTargetState {
                format: SSAO_FORMAT,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        };

        self.render_manager.render_fullscreen_pass(
            "SSAO Blur",
            shader_dir().join(shader).as_path(),
            options,
            &[
                &self.pipelines.camera_uniforms.buffer,
                &self.pipelines.ssao_uniforms.buffer,
            ],
            &mut pass,
            &self.pipelines,
            settings,
            FullscreenPassType::SsaoBlur,
        );
    }
    fn execute_ssao_apply_pass(&mut self, encoder: &mut CommandEncoder, settings: &Settings) {
        if !settings.show_world {
            return;
        }

        let color_attachment = create_color_attachment_load(
            &self.pipelines.msaa_hdr_view,
            &self.pipelines.resolved_hdr_view,
            self.msaa_samples,
        );

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("SSAO Apply Pass"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        let multiply_blend = BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::Zero,
                dst_factor: BlendFactor::Src,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::Zero,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
        };

        let options = PipelineOptions {
            topology: TriangleList,
            msaa_samples: self.msaa_samples, // output is MSAA HDR target
            depth_stencil: None,
            vertex_layouts: vec![],
            cull_mode: None,
            shadow_pass: false,
            fullscreen_pass: FullscreenPassType::SsaoApply,
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float, // HDR
                blend: Some(multiply_blend),
                write_mask: ColorWrites::ALL,
            })],
        };

        self.render_manager.render_fullscreen_pass(
            "SSAO Apply",
            shader_dir().join("ssao_apply.wgsl").as_path(),
            options,
            &[], // no uniforms needed
            &mut pass,
            &self.pipelines,
            settings,
            FullscreenPassType::SsaoApply,
        );
    }
    fn execute_fog_pass(&mut self, encoder: &mut CommandEncoder, settings: &Settings) {
        let color_attachment = create_color_attachment_load(
            &self.pipelines.msaa_hdr_view,
            &self.pipelines.resolved_hdr_view,
            self.msaa_samples,
        );

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Fog Pass"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if !settings.show_world || !settings.show_fog {
            return;
        }
        let fog_shader = if self.msaa_samples > 1 {
            "fog_msaa.wgsl"
        } else {
            "fog.wgsl"
        };

        let options = PipelineOptions {
            topology: TriangleList,
            msaa_samples: self.msaa_samples,
            depth_stencil: None,
            vertex_layouts: vec![],
            cull_mode: None,
            shadow_pass: false,
            fullscreen_pass: FullscreenPassType::Fog, // (will be overwritten anyway)
            targets: vec![Some(ColorTargetState {
                format: self.pipelines.msaa_hdr_view.texture().format(),
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            })],
        };

        self.render_manager.render_fullscreen_pass(
            "Fog",
            shader_dir().join(fog_shader).as_path(),
            options,
            &[
                &self.pipelines.camera_uniforms.buffer,
                &self.pipelines.fog_uniforms.buffer,
                &self.pipelines.pick_uniforms.buffer,
            ],
            &mut pass,
            &self.pipelines,
            settings,
            FullscreenPassType::Fog,
        );
    }

    fn execute_ui_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        ui_loader: &mut UiButtonLoader,
        time: &TimeSystem,
        input_state: &InputState,
    ) {
        let color_attachment = create_color_attachment_load(
            &self.pipelines.msaa_hdr_view,
            &self.pipelines.resolved_hdr_view,
            self.msaa_samples,
        );

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Main Pass (UI)"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        self.render_ui(&mut pass, ui_loader, time, input_state);
    }

    fn execute_debug_preview_pass(&mut self, encoder: &mut CommandEncoder, settings: &Settings) {
        match settings.debug_view_state {
            DebugViewState::None => {
                return;
            }
            DebugViewState::Normals => {
                let color_attachment = create_color_attachment_load(
                    &self.pipelines.msaa_hdr_view,
                    &self.pipelines.resolved_hdr_view,
                    self.msaa_samples,
                );

                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("Debug Normals Fullscreen Preview Pass"),
                    color_attachments: &[Some(color_attachment)],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                self.render_manager.render_fullscreen_debug_view(
                    &self.pipelines.resolved_normal_view,
                    "Debug Normals Render",
                    self.pipelines.msaa_hdr_view.texture().format(), // target format (color)
                    self.msaa_samples,
                    FullscreenDebugSwizzle::None,
                    &mut pass,
                );
            }
            DebugViewState::Depth => {
                let color_attachment = create_color_attachment_load(
                    &self.pipelines.msaa_hdr_view,
                    &self.pipelines.resolved_hdr_view,
                    self.msaa_samples,
                );

                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("Debug Depth Fullscreen Preview Pass"),
                    color_attachments: &[Some(color_attachment)],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                self.render_manager.render_fullscreen_debug_view(
                    &self.pipelines.depth_sample_view,
                    "Debug Depth Render",
                    self.pipelines.msaa_hdr_view.texture().format(), // target format (color)
                    self.msaa_samples,
                    FullscreenDebugSwizzle::None,
                    &mut pass,
                );
            }
            DebugViewState::SsaoRaw => {
                let color_attachment = create_color_attachment_load(
                    &self.pipelines.msaa_hdr_view,
                    &self.pipelines.resolved_hdr_view,
                    self.msaa_samples,
                );

                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("Debug Normals Fullscreen Preview Pass"),
                    color_attachments: &[Some(color_attachment)],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                self.render_manager.render_fullscreen_debug_view(
                    &self.pipelines.ssao_view,
                    "Debug SSAO RAW Render",
                    self.pipelines.msaa_hdr_view.texture().format(), // target format (color)
                    self.msaa_samples,
                    FullscreenDebugSwizzle::RedToRgb,
                    &mut pass,
                );
            }
            DebugViewState::SsaoBlurred => {
                let color_attachment = create_color_attachment_load(
                    &self.pipelines.msaa_hdr_view,
                    &self.pipelines.resolved_hdr_view,
                    self.msaa_samples,
                );

                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("Debug Normals Fullscreen Preview Pass"),
                    color_attachments: &[Some(color_attachment)],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                self.render_manager.render_fullscreen_debug_view(
                    &self.pipelines.ssao_blur_view,
                    "Debug SSAO Blurred Render",
                    self.pipelines.msaa_hdr_view.texture().format(), // target format (color)
                    self.msaa_samples,
                    FullscreenDebugSwizzle::RedToRgb,
                    &mut pass,
                );
            }
        }
    }

    fn execute_tonemap_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        surface_view: &TextureView,
        settings: &Settings,
    ) {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Tonemap Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: surface_view,
                resolve_target: None,
                depth_slice: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let options = PipelineOptions {
            topology: TriangleList,
            msaa_samples: 1,
            depth_stencil: None,
            vertex_layouts: vec![],
            cull_mode: None,
            shadow_pass: false,
            fullscreen_pass: FullscreenPassType::Normal,
            targets: vec![Some(ColorTargetState {
                format: surface_view.texture().format(),
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        };
        self.render_manager.render_fullscreen_pass(
            "Tonemap",
            shader_dir().join("tonemap.wgsl").as_path(),
            options,
            &[&self.pipelines.tonemapping_uniforms.buffer],
            &mut pass,
            &self.pipelines,
            settings,
            FullscreenPassType::Normal,
        );
    }

    fn render_ui(
        &mut self,
        pass: &mut RenderPass,
        ui_loader: &mut UiButtonLoader,
        time: &TimeSystem,
        input_state: &InputState,
    ) {
        let screen_uniform = ScreenUniform {
            size: [self.config.width as f32, self.config.height as f32],
            time: time.total_time as f32,
            enable_dither: 1,
            mouse: input_state.mouse.pos.to_array(),
        };

        self.queue.write_buffer(
            &self.ui_renderer.pipelines.uniform_buffer,
            0,
            bytemuck::bytes_of(&screen_uniform),
        );

        self.ui_renderer
            .render(&mut self.render_manager, pass, ui_loader, &self.pipelines);
    }

    pub(crate) fn cycle_msaa(&mut self) {
        let supported = get_supported_msaa_levels(&self.adapter, self.config.format);
        let current_idx = supported
            .iter()
            .position(|&s| s == self.msaa_samples)
            .unwrap_or(0);
        self.msaa_samples = supported[(current_idx + 1) % supported.len()];

        println!("MSAA changed to {}x", self.msaa_samples);

        self.pipelines.resize(&self.config, self.msaa_samples);
        self.ui_renderer.pipelines.msaa_samples = self.msaa_samples;
        self.ui_renderer.pipelines.rebuild_pipelines();
        self.render_manager.invalidate_resize_bind_groups();
    }

    fn reload_all_shaders(&mut self) -> anyhow::Result<()> {
        self.ui_renderer.reload_shaders()?;
        Ok(())
    }

    fn check_shader_changes(&mut self, ui_loader: &mut UiButtonLoader) {
        let Some(watcher) = &self.shader_watcher else {
            return;
        };

        let changed = watcher.take_changed_wgsl_files();
        if changed.is_empty() {
            return;
        }

        let summary = changed
            .iter()
            .filter_map(|p| p.file_name().and_then(|n| n.to_str()))
            .collect::<Vec<_>>()
            .join(", ");

        match self.reload_all_shaders() {
            Ok(()) => {
                let label = if summary.is_empty() {
                    "Shaders reloaded".to_string()
                } else {
                    format!("Shaders reloaded: {summary}")
                };
                println!("{}", label);
                ui_loader.log_console(format!("✅ {label}"));
            }
            Err(err) => ui_loader.log_console(format!("❌ Shader reload failed: {err}")),
        }
        self.render_manager.pipeline_manager.reload_shaders(changed);
    }
}

fn create_surface_and_adapter(
    window: Arc<Window>,
) -> (Surface<'static>, Adapter, PhysicalSize<u32>) {
    let instance = Instance::new(&InstanceDescriptor {
        backends: Backends::all(),
        flags: Default::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
    });

    let size = window.inner_size();
    let surface = instance
        .create_surface(window)
        .expect("Surface creation failed");

    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .expect("No suitable GPU adapters found");

    println!("Backend: {:?}", adapter.get_info().backend);

    (surface, adapter, size)
}

fn create_surface_config(
    surface: &Surface,
    adapter: &Adapter,
    settings: &Settings,
    size: PhysicalSize<u32>,
) -> (SurfaceConfiguration, u32) {
    let surface_caps = surface.get_capabilities(adapter);

    let format = surface_caps
        .formats
        .iter()
        .copied()
        .find(|f| *f == Rgba8UnormSrgb)
        .unwrap_or(surface_caps.formats[0]);
    // println!("{:?}", format);

    let alpha_mode = surface_caps
        .alpha_modes
        .iter()
        .copied()
        .find(|m| *m == CompositeAlphaMode::PostMultiplied)
        .unwrap_or(CompositeAlphaMode::Opaque);

    let present_mode = pick_present_mode(surface, adapter, settings.present_mode.clone().to_wgpu());

    let config = SurfaceConfiguration {
        usage: TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode,
        alpha_mode,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };

    let caps = adapter.get_texture_format_features(config.format);
    let msaa_samples = clamp_to_supported_msaa(caps, settings.msaa_samples);

    (config, msaa_samples)
}

fn create_device(adapter: &Adapter) -> (Device, Queue) {
    pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: Some("Device"),
        required_features: //Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
        Features::TIMESTAMP_QUERY
            | Features::TIMESTAMP_QUERY_INSIDE_PASSES
            | Features::DEPTH32FLOAT_STENCIL8
            | Features::MULTIVIEW,
        required_limits: Limits::default(),
        experimental_features: ExperimentalFeatures::disabled(),
        memory_hints: MemoryHints::default(),
        trace: Trace::Off,
    }))
    .expect("Device creation failed")
}

fn pick_present_mode(surface: &Surface, adapter: &Adapter, user_mode: PresentMode) -> PresentMode {
    let caps = surface.get_capabilities(adapter);
    let fallbacks = [
        user_mode,
        PresentMode::Mailbox,
        PresentMode::Immediate,
        PresentMode::Fifo,
    ];

    fallbacks
        .into_iter()
        .find(|mode| caps.present_modes.contains(mode))
        .unwrap_or(PresentMode::Fifo)
}

fn get_supported_msaa_levels(adapter: &Adapter, format: TextureFormat) -> Vec<u32> {
    let features = adapter.get_texture_format_features(format);

    let mut levels = Vec::new();
    for &samples in &[1, 2, 4, 8] {
        if features.flags.sample_count_supported(samples) {
            levels.push(samples);
        }
    }

    levels
}

fn clamp_to_supported_msaa(caps: TextureFormatFeatures, requested: u32) -> u32 {
    let can_render = caps
        .allowed_usages
        .contains(TextureUsages::RENDER_ATTACHMENT);

    if !can_render {
        println!("MSAA not usable: format cannot be a render attachment");
        return 1;
    }

    let supports_8 = caps
        .flags
        .contains(TextureFormatFeatureFlags::MULTISAMPLE_X8);
    let supports_4 = caps
        .flags
        .contains(TextureFormatFeatureFlags::MULTISAMPLE_X4);
    let supports_2 = caps
        .flags
        .contains(TextureFormatFeatureFlags::MULTISAMPLE_X2);

    if supports_8 {
        return if requested < 8 {
            println!("8x MSAA supported, but using {}x!", requested);
            requested
        } else {
            println!("8x MSAA supported, using {}x!", requested);
            8
        };
    }

    if supports_4 {
        if requested >= 8 {
            println!("8x requested but unsupported, falling back to 4x");
            return 4;
        }
        if requested >= 4 {
            println!("4x MSAA supported, using 4x");
            return 4;
        }
        println!("4x MSAA supported, but using {}x!", requested);
        return requested;
    }

    if supports_2 {
        if requested >= 4 {
            println!("Only 2x MSAA supported, falling back to 2x");
            return 2;
        }
        println!("2x MSAA supported, using {}x!", requested);
        return requested;
    }

    println!("MSAA not supported, falling back to 1x (WTF is this GPU huh?)");
    1
}

pub fn acquire_frame(
    surface: &Surface,
    device: &Device,
    config: &SurfaceConfiguration,
) -> Option<SurfaceTexture> {
    match surface.get_current_texture() {
        Ok(frame) => Some(frame),
        Err(SurfaceError::Outdated) => {
            // Surface is temporarily invalid
            None // skip this frame
        }
        Err(SurfaceError::Lost) => {
            surface.configure(device, config);
            None
        }
        Err(SurfaceError::OutOfMemory) => {
            panic!("OOM");
        }
        Err(e) => {
            eprintln!("Surface error: {:?}", e);
            None
        }
    }
}
