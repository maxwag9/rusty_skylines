use crate::components::camera::*;
use crate::data::{DebugViewState, Settings};
use crate::gpu_timestamp;
use crate::mouse_ray::*;
use crate::paths::{compute_shader_dir, shader_dir, texture_dir};
use crate::positions::WorldPos;
use crate::renderer::astronomy::*;
use crate::renderer::gizmo::Gizmo;
use crate::renderer::gpu_profiler::GpuProfiler;
use crate::renderer::gtao::gtao::GtaoBlurParams;
use crate::renderer::pipelines::Pipelines;
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
use wgpu::{
    Adapter, Backends, Color, ColorTargetState, ColorWrites, CommandEncoder,
    CommandEncoderDescriptor, CompositeAlphaMode, Device, DeviceDescriptor, ExperimentalFeatures,
    Features, Instance, InstanceDescriptor, Limits, LoadOp, MemoryHints, Operations,
    PowerPreference, PresentMode, Queue, RenderPass, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, RequestAdapterOptions, StoreOp,
    Surface, SurfaceConfiguration, SurfaceError, SurfaceTexture, TextureFormat,
    TextureFormatFeatureFlags, TextureFormatFeatures, TextureUsages, TextureView,
    TextureViewDescriptor, Trace,
};
// use wgpu::*;
use wgpu_render_manager::compute_system::{ComputePipelineOptions, ComputeSystem};
use wgpu_render_manager::fullscreen::{DebugVisualization, DepthDebugParams};
use wgpu_render_manager::pipelines::PipelineOptions;
use wgpu_render_manager::renderer::RenderManager;
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
    compute: ComputeSystem,
}

impl RenderCore {
    pub fn new(window: Arc<Window>, settings: &Settings, camera: &Camera) -> Self {
        let (surface, adapter, size) = create_surface_and_adapter(window);
        let (config, msaa_samples) = create_surface_config(&surface, &adapter, settings, size);
        let (device, queue) = create_device(&adapter);
        surface.configure(&device, &config);

        let shader_watcher = ShaderWatcher::new().ok();

        let pipelines = Pipelines::new(&device, &queue, &config, camera, settings)
            .expect("Failed to create render pipelines");
        let ui_renderer = UiRenderer::new(&device, config.format, size, msaa_samples)
            .expect("Failed to create UI pipelines");
        let terrain_renderer = TerrainRenderer::new(&device, settings);
        let road_renderer = RoadRenderSubsystem::new(&device);
        let render_manager = RenderManager::new(&device, &queue, texture_dir());
        let gizmo = Gizmo::new(&device, terrain_renderer.chunk_size);
        let profiler = GpuProfiler::new(&device, 5, 3);
        let compute = ComputeSystem::new(&device, &queue);
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
            compute,
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
        self.render_manager.invalidate_bind_groups();
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

        // self.execute_shadow_pass(&mut encoder, camera, aspect, time, settings);
        self.execute_main_pass(
            &surface_view,
            &RenderPassConfig::from_settings(settings),
            camera,
            aspect,
            time,
            input_state,
            ui_loader,
            settings,
        );

        //self.profiler.resolve(&mut encoder);

        frame.present();
        // self.profiler
        //     .end_frame(&self.device, &self.queue, &mut ui_loader.variables);
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
        self.terrain_renderer.make_pick_uniforms(
            &self.queue,
            &self.pipelines.buffers.pick,
            camera,
            self.terrain_renderer.chunk_size,
        );
        self.render_manager.update_depth_params(DepthDebugParams {
            near: camera.near,
            far: camera.far,
            power: 20.0,
            reversed_z: settings.reversed_depth_z as u32,
            msaa_samples: self.msaa_samples,
        });
        self.update_uniforms(
            camera, view, proj, view_proj, &astronomy, time, aspect, settings,
        );

        // upload per-cascade shadow uniforms ONCE (outside encoder)
        for i in 0..CSM_CASCADES {
            let smu = ShadowMatUniform {
                light_view_proj: self.pipelines.resources.csm_shadows.light_mats[i]
                    .to_cols_array_2d(),
            };
            self.queue.write_buffer(
                &self.pipelines.resources.csm_shadows.shadow_mat_buffers[i],
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
        updater.update_ssao_uniforms(time, settings);
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
                    view: &self.pipelines.resources.csm_shadows.layer_views[cascade],
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
            let shadow_buf = &self.pipelines.resources.csm_shadows.shadow_mat_buffers[cascade];

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
        surface_view: &TextureView,
        config: &RenderPassConfig,
        camera: &Camera,
        aspect: f32,
        time: &TimeSystem,
        input_state: &InputState,
        ui_loader: &mut UiButtonLoader,
        settings: &Settings,
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        self.execute_world_pass(&mut encoder, config, camera, settings, aspect);
        self.queue.submit(Some(encoder.finish()));

        self.execute_gtao_pass(settings);

        self.execute_fog_pass(settings);

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        self.execute_ui_pass(&mut encoder, ui_loader, time, input_state);

        self.execute_debug_preview_pass(&mut encoder, settings);

        self.execute_tonemap_pass(&mut encoder, surface_view, settings);
        self.queue.submit(Some(encoder.finish()));
    }

    fn execute_world_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        config: &RenderPassConfig,
        camera: &Camera,
        settings: &Settings,
        aspect: f32,
    ) {
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
    fn execute_gtao_pass(&mut self, settings: &Settings) {
        if !settings.show_world {
            return;
        }
        let msaa_on = settings.msaa_samples > 1;
        if msaa_on {
            self.compute.compute(
                "resolve_depth_compute_pass",
                vec![&self.pipelines.msaa.depth_sample],
                vec![&self.pipelines.resolved.depth],
                &shader_dir().join("compute/resolve_depth.wgsl"),
                ComputePipelineOptions {
                    dispatch_size: [
                        self.pipelines.resolved.depth.texture().width() / 8,
                        self.pipelines.resolved.depth.texture().height() / 8,
                        1,
                    ],
                },
                &[&self.pipelines.buffers.camera],
            );
        } else {
            self.compute.compute(
                "convert_depth_compute_pass",
                vec![&self.pipelines.msaa.depth_sample],
                vec![&self.pipelines.resolved.depth],
                &shader_dir().join("compute/resolve_depth_single_sample.wgsl"),
                ComputePipelineOptions {
                    dispatch_size: [
                        self.pipelines.msaa.depth_sample.texture().width() / 8,
                        self.pipelines.msaa.depth_sample.texture().height() / 8,
                        1,
                    ],
                },
                &[&self.pipelines.buffers.camera],
            );
        }

        let resolved_depth = &self.pipelines.resolved.depth;

        // Linearize depth pass
        self.compute.compute(
            "linearize_depth_compute_pass",
            vec![resolved_depth],
            vec![&self.pipelines.post_fx.linear_depth_full],
            &shader_dir().join("compute/linearize_depth.wgsl"),
            ComputePipelineOptions {
                dispatch_size: [
                    self.pipelines.resolved.depth.texture().width() / 8,
                    self.pipelines.resolved.depth.texture().height() / 8,
                    1,
                ],
            },
            &[&self.pipelines.buffers.camera],
        );

        // Downsample linear depth to half resolution
        self.compute.compute(
            "downsample_linear_depth_pass",
            vec![&self.pipelines.post_fx.linear_depth_full],
            vec![&self.pipelines.post_fx.linear_depth_half],
            &shader_dir().join("compute/downsample_linear_depth.wgsl"),
            ComputePipelineOptions {
                dispatch_size: [
                    (self.pipelines.post_fx.linear_depth_half.texture().width() + 7) / 8,
                    (self.pipelines.post_fx.linear_depth_half.texture().height() + 7) / 8,
                    1,
                ],
            },
            &[],
        );

        // Downsample normals to half resolution
        self.compute.compute(
            "downsample_normals_pass",
            vec![&self.pipelines.resolved.normal], // Full res normals
            vec![&self.pipelines.post_fx.normal_half], // Half res output
            &shader_dir().join("compute/downsample_normals.wgsl"),
            ComputePipelineOptions {
                dispatch_size: [
                    (self.pipelines.post_fx.normal_half.texture().width() + 7) / 8,
                    (self.pipelines.post_fx.normal_half.texture().height() + 7) / 8,
                    1,
                ],
            },
            &[],
        );

        // GTAO Generation Pass
        let half_width = self.pipelines.post_fx.linear_depth_half.texture().width();
        let half_height = self.pipelines.post_fx.linear_depth_half.texture().height();

        self.compute.compute(
            "gtao_generate_pass",
            vec![
                &self.pipelines.post_fx.linear_depth_half,
                &self.pipelines.post_fx.normal_half,
                &self.pipelines.resources.blue_noise,
            ],
            vec![&self.pipelines.post_fx.gtao_raw_half],
            &shader_dir().join("compute/gtao_generate.wgsl"),
            ComputePipelineOptions {
                dispatch_size: [(half_width + 7) / 8, (half_height + 7) / 8, 1],
            },
            &[&self.pipelines.buffers.camera, &self.pipelines.buffers.gtao],
        );

        // === Horizontal Pass ===
        let h_params = GtaoBlurParams::horizontal(half_width, half_height);
        self.queue.write_buffer(
            &self.pipelines.buffers.gtao_blur,
            0,
            bytemuck::bytes_of(&h_params),
        );

        self.compute.compute(
            "gtao_blur_horizontal",
            vec![
                &self.pipelines.post_fx.gtao_raw_half,     // AO input
                &self.pipelines.post_fx.linear_depth_half, // Depth
                &self.pipelines.post_fx.normal_half,       // Normals
            ],
            vec![&self.pipelines.post_fx.gtao_blur_horiz_half], // Temp output
            &shader_dir().join("compute/gtao_blur.wgsl"),
            ComputePipelineOptions {
                dispatch_size: [(half_width + 7) / 8, (half_height + 7) / 8, 1],
            },
            &[&self.pipelines.buffers.gtao_blur],
        );

        // === Vertical Pass ===
        let v_params = GtaoBlurParams::vertical(half_width, half_height);
        self.queue.write_buffer(
            &self.pipelines.buffers.gtao_blur,
            0,
            bytemuck::bytes_of(&v_params),
        );

        self.compute.compute(
            "gtao_blur_vertical",
            vec![
                &self.pipelines.post_fx.gtao_blur_horiz_half, // AO input (from horizontal)
                &self.pipelines.post_fx.linear_depth_half,
                &self.pipelines.post_fx.normal_half,
            ],
            vec![&self.pipelines.post_fx.gtao_blurred_half], // Final output
            &shader_dir().join("compute/gtao_blur.wgsl"),
            ComputePipelineOptions {
                dispatch_size: [(half_width + 7) / 8, (half_height + 7) / 8, 1],
            },
            &[&self.pipelines.buffers.gtao_blur],
        );

        let full_width = self.pipelines.post_fx.linear_depth_full.texture().width();
        let full_height = self.pipelines.post_fx.linear_depth_full.texture().height();

        self.compute.compute(
            "gtao_upsample_pass",
            vec![
                &self.pipelines.post_fx.gtao_blurred_half, // Half-res blurred AO
                &self.pipelines.post_fx.linear_depth_half, // Half-res depth
                &self.pipelines.post_fx.normal_half,       // Half-res normals
                &self.pipelines.post_fx.linear_depth_full, // Full-res depth
                &self.pipelines.resolved.normal,           // Full-res normals
            ],
            vec![&self.pipelines.post_fx.gtao_final_full], // Full-res output
            &compute_shader_dir().join("gtao_upsample.wgsl"),
            ComputePipelineOptions {
                dispatch_size: [(full_width + 7) / 8, (full_height + 7) / 8, 1],
            },
            &[&self.pipelines.buffers.gtao_upsample],
        );
        self.execute_gtao_apply();
    }
    fn execute_gtao_apply(&mut self) {
        let full_width = self.pipelines.resolved.hdr.texture().width();
        let full_height = self.pipelines.resolved.hdr.texture().height();

        self.compute.compute(
            "gtao_apply_pass",
            vec![
                &self.pipelines.resolved.hdr,            // HDR input
                &self.pipelines.post_fx.gtao_final_full, // GTAO texture
            ],
            vec![&self.pipelines.resolved.hdr_with_ao], // HDR output with AO
            &shader_dir().join("compute/gtao_apply.wgsl"),
            ComputePipelineOptions {
                dispatch_size: [(full_width + 7) / 8, (full_height + 7) / 8, 1],
            },
            &[&self.pipelines.buffers.gtao_apply],
        );
    }
    fn execute_fog_pass(&mut self, settings: &Settings) {
        if !settings.show_world || !settings.show_fog {
            return;
        }

        let width = self.pipelines.resolved.hdr_fogged.texture().size().width;
        let height = self.pipelines.resolved.hdr_fogged.texture().size().height;

        let workgroup_size = 8;
        let dispatch_x = (width + workgroup_size - 1) / workgroup_size;
        let dispatch_y = (height + workgroup_size - 1) / workgroup_size;

        self.compute.compute(
            "Fog Compute Pass",
            vec![
                &self.pipelines.resolved.hdr_with_ao,      // input color
                &self.pipelines.post_fx.linear_depth_full, // input depth
            ],
            vec![
                &self.pipelines.resolved.hdr_fogged, // output color
            ],
            &compute_shader_dir().join("fog.wgsl"),
            ComputePipelineOptions {
                dispatch_size: [dispatch_x, dispatch_y, 1],
            },
            &[&self.pipelines.buffers.camera, &self.pipelines.buffers.fog],
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
            &self.pipelines.msaa.hdr,
            &self.pipelines.resolved.hdr,
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

        {
            let options = PipelineOptions::default()
                .with_topology(TriangleList)
                .with_msaa(self.msaa_samples)
                .with_target(ColorTargetState {
                    format: self.pipelines.msaa.hdr.texture().format(),
                    blend: None,
                    write_mask: ColorWrites::ALL,
                });

            self.render_manager.render_with_textures(
                &[&self.pipelines.resolved.hdr_fogged],
                shader_dir().join("resolved_to_msaa_copy.wgsl").as_path(),
                &options,
                &[],
                &mut pass,
            );

            pass.draw(0..3, 0..1);
        }

        self.render_ui(&mut pass, ui_loader, time, input_state);
    }

    fn execute_debug_preview_pass(&mut self, encoder: &mut CommandEncoder, settings: &Settings) {
        match settings.debug_view_state {
            DebugViewState::None => {
                return;
            }
            DebugViewState::Normals => {
                let color_attachment = create_color_attachment_load(
                    &self.pipelines.msaa.hdr,
                    &self.pipelines.resolved.hdr,
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

                self.render_manager.render_fullscreen_debug(
                    &self.pipelines.resolved.normal,
                    DebugVisualization::Color,
                    &self.pipelines.msaa.hdr,
                    &mut pass,
                );
            }
            DebugViewState::Depth => {
                let color_attachment = create_color_attachment_load(
                    &self.pipelines.msaa.hdr,
                    &self.pipelines.resolved.hdr,
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

                self.render_manager.render_fullscreen_debug(
                    &self.pipelines.post_fx.linear_depth_full,
                    DebugVisualization::LinearDepth,
                    &self.pipelines.msaa.hdr,
                    &mut pass,
                );
            }
            DebugViewState::SsaoRaw => {
                let color_attachment = create_color_attachment_load(
                    &self.pipelines.msaa.hdr,
                    &self.pipelines.resolved.hdr,
                    self.msaa_samples,
                );

                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("Debug Raw GTAO Fullscreen Preview Pass"),
                    color_attachments: &[Some(color_attachment)],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                self.render_manager.render_fullscreen_debug(
                    &self.pipelines.post_fx.gtao_raw_half,
                    DebugVisualization::RedToGrayscale,
                    &self.pipelines.msaa.hdr,
                    &mut pass,
                );
            }
            DebugViewState::GtaoBlurred => {
                let color_attachment = create_color_attachment_load(
                    &self.pipelines.msaa.hdr,
                    &self.pipelines.resolved.hdr,
                    self.msaa_samples,
                );

                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("Debug Blurred GTAO Fullscreen Preview Pass"),
                    color_attachments: &[Some(color_attachment)],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                self.render_manager.render_fullscreen_debug(
                    &self.pipelines.resolved.hdr_with_ao,
                    DebugVisualization::Color,
                    &self.pipelines.msaa.hdr,
                    &mut pass,
                );
            }
        }
    }

    fn execute_tonemap_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        surface_view: &TextureView,
        _settings: &Settings,
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
            targets: vec![Some(ColorTargetState {
                format: surface_view.texture().format(),
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            vertex_only: false,
            shadow: None,
        };

        self.render_manager.render_with_textures(
            &[&self.pipelines.resolved.hdr],
            shader_dir().join("tonemap.wgsl").as_path(),
            &options,
            &[&self.pipelines.buffers.tonemapping],
            &mut pass,
        );

        pass.draw(0..3, 0..1);
    }

    fn render_ui<'a>(
        &'a mut self,
        pass: &mut RenderPass<'a>,
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

    pub(crate) fn cycle_msaa(&mut self, settings: &mut Settings) {
        let supported = get_supported_msaa_levels(&self.adapter, self.config.format);
        let current_idx = supported
            .iter()
            .position(|&s| s == self.msaa_samples)
            .unwrap_or(0);
        self.msaa_samples = supported[(current_idx + 1) % supported.len()];
        settings.msaa_samples = self.msaa_samples;

        println!("MSAA changed to {}x", self.msaa_samples);

        self.pipelines.resize(&self.config, self.msaa_samples);
        self.ui_renderer.pipelines.msaa_samples = self.msaa_samples;
        self.ui_renderer.pipelines.rebuild_pipelines();
        self.render_manager.invalidate_bind_groups();
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
        self.render_manager
            .reload_render_shaders(changed.as_slice());
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
        required_features: Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | Features::TIMESTAMP_QUERY
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
fn create_color_attachment_load<'a>(
    msaa_view: &'a TextureView,
    surface_view: &'a TextureView,
    msaa_samples: u32,
) -> RenderPassColorAttachment<'a> {
    if msaa_samples > 1 {
        RenderPassColorAttachment {
            view: msaa_view,
            depth_slice: None,
            resolve_target: Some(surface_view),
            ops: Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            },
        }
    } else {
        RenderPassColorAttachment {
            view: surface_view,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            },
        }
    }
}
