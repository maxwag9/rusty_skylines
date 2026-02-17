use crate::data::{DebugViewState, Settings, ShadowType};
use crate::gpu_timestamp;
use crate::helpers::paths::{compute_shader_dir, shader_dir, texture_dir};
use crate::helpers::positions::WorldPos;
use crate::renderer::gizmo::Gizmo;
use crate::renderer::gpu_profiler::GpuProfiler;
use crate::renderer::gtao::gtao::{GtaoBlurParams, GtaoUpsampleApplyParams};
use crate::renderer::pipelines::Pipelines;
use crate::renderer::ray_tracing::rt_pass::render_ray_tracing;
use crate::renderer::ray_tracing::rt_subsystem::RTSubsystem;
use crate::renderer::render_passes::*;
use crate::renderer::shader_watcher::ShaderWatcher;
use crate::renderer::shadows::{
    CSM_CASCADES, ShadowMatUniform, render_cars_shadows, render_roads_shadows,
};
use crate::renderer::ui::{ScreenUniform, UiRenderer};
use crate::renderer::uniform_updates::UniformUpdater;
use crate::resources::TimeSystem;
use crate::ui::input::InputState;
use crate::ui::ui_editor::UiButtonLoader;
use crate::world::astronomy::*;
use crate::world::camera::Camera;
use crate::world::cars::car_structs::CarStorage;
use crate::world::cars::car_subsystem::{CarRenderSubsystem, CarSubsystem};
use crate::world::roads::road_subsystem::{RoadRenderSubsystem, RoadSubsystem};
use crate::world::terrain::terrain_subsystem::{TerrainRenderSubsystem, TerrainSubsystem};
use glam::UVec2;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wgpu::PrimitiveTopology::TriangleList;
use wgpu::TextureFormat::Rgba8UnormSrgb;
use wgpu::wgt::PollType;
use wgpu::{
    Adapter, Backends, BlendComponent, BlendFactor, BlendOperation, BlendState, Color,
    ColorTargetState, ColorWrites, CommandEncoder, CommandEncoderDescriptor, CompositeAlphaMode,
    Device, DeviceDescriptor, ExperimentalFeatures, Features, Instance, InstanceDescriptor, Limits,
    LoadOp, MemoryHints, Operations, PowerPreference, PresentMode, Queue, RenderPass,
    RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor,
    RequestAdapterOptions, StoreOp, Surface, SurfaceConfiguration, SurfaceError, SurfaceTexture,
    TextureFormat, TextureFormatFeatureFlags, TextureFormatFeatures, TextureUsages, TextureView,
    TextureViewDescriptor, Trace,
};
use wgpu_render_manager::compute_system::{BufferSet, ComputePipelineOptions};
use wgpu_render_manager::fullscreen::{DebugVisualization, DepthDebugParams};
use wgpu_render_manager::pipelines::PipelineOptions;
use wgpu_render_manager::renderer::RenderManager;
use winit::dpi::PhysicalSize;
use winit::window::Window;

pub struct RenderCore {
    // gpu objects
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub config: SurfaceConfiguration,
    pub msaa_samples: u32,

    // render-only subsystems & caches
    pub render_manager: RenderManager,
    pub shader_watcher: Option<ShaderWatcher>,
    pub pipelines: Pipelines,
    pub ui_renderer: UiRenderer,
    pub profiler: GpuProfiler,
    pub rt_subsystem: RTSubsystem,
    pub terrain_renderer: TerrainRenderSubsystem,
    pub road_renderer: RoadRenderSubsystem,
    pub car_renderer: CarRenderSubsystem,
    pub gizmo: Gizmo,
}

impl RenderCore {
    pub fn new(
        device: &Device,
        queue: &Queue,
        config: &SurfaceConfiguration,
        size: PhysicalSize<u32>,
        adapter: Adapter,
        settings: &Settings,
        camera: &Camera,
    ) -> Self {
        let shader_watcher = ShaderWatcher::new().ok();

        let mut rt_subsystem = RTSubsystem::new(device);
        let ui_renderer = UiRenderer::new(device, config.format, size, settings.msaa_samples)
            .expect("Failed to create UI pipelines");
        let terrain_renderer = TerrainRenderSubsystem::new();
        let road_renderer = RoadRenderSubsystem::new(device);
        let car_renderer = CarRenderSubsystem::new(device, queue, &mut rt_subsystem);
        let mut render_manager = RenderManager::new(device, queue, texture_dir());
        let gizmo = Gizmo::new(device, settings.chunk_size);
        let profiler = GpuProfiler::new(&device, 3);
        let pipelines = Pipelines::new(
            &mut render_manager,
            device,
            queue,
            &config,
            camera,
            settings,
        )
        .expect("Failed to create render pipelines");
        Self {
            adapter,
            device: device.clone(),
            queue: queue.clone(),
            config: config.clone(),
            pipelines,
            msaa_samples: settings.msaa_samples,
            ui_renderer,
            terrain_renderer,
            road_renderer,
            car_renderer,
            shader_watcher,
            render_manager,
            gizmo,
            profiler,
            rt_subsystem,
        }
    }

    pub(crate) fn resize(&mut self, surface: &Surface, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        if new_size.width == self.config.width && new_size.height == self.config.height {
            return;
        }
        let result = self.device.poll(PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        });
        if result.is_err() {
            panic!(
                "Too long device polling time in resize()  Error: {:?}",
                result
            )
        }
        surface.configure(&self.device, &self.config);
        self.pipelines.resize(&self.config, self.msaa_samples);
        self.render_manager.invalidate_bind_groups();
    }

    pub(crate) fn render(
        &mut self,
        surface: &Surface,
        camera: &Camera,
        ui_loader: &mut UiButtonLoader,
        time: &TimeSystem,
        input_state: &mut InputState,
        settings: &Settings,
        terrain_subsystem: &mut TerrainSubsystem,
        road_subsystem: &RoadSubsystem,
        car_subsystem: &CarSubsystem,
        astronomy: &AstronomyState,
    ) {
        let total_cpu_render_time_start = Instant::now();
        let aspect = self.config.width as f32 / self.config.height as f32;
        let screen_size: UVec2 = UVec2::new(self.config.width, self.config.height);

        //let t = Instant::now();
        self.update_render(
            camera,
            terrain_subsystem,
            ui_loader,
            time,
            input_state,
            settings,
            road_subsystem,
            car_subsystem,
            astronomy,
            aspect,
            screen_size,
        );

        let Some(frame) = acquire_frame(&surface, &self.device, &self.config) else {
            return;
        };

        let surface_view = frame.texture.create_view(&TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        gpu_timestamp!(encoder, &mut self.profiler, "Total", {
            gpu_timestamp!(encoder, &mut self.profiler, "CSM", {
                self.execute_shadow_pass(
                    &mut encoder,
                    car_subsystem.car_storage(),
                    camera,
                    aspect,
                    time,
                    settings,
                );
            });
            self.execute_main_pass(
                &mut encoder,
                &surface_view,
                &RenderPassConfig::from_settings(settings),
                camera,
                aspect,
                time,
                input_state,
                ui_loader,
                terrain_subsystem,
                &car_subsystem.car_storage(),
                settings,
                astronomy,
            );
        });

        //println!("World CPU Time: {:?}", t.elapsed());
        self.profiler.resolve(&mut encoder);
        let total_cpu_render_time = total_cpu_render_time_start.elapsed().as_secs_f32() * 1000.0f32;
        ui_loader
            .variables
            .set_f32("total_cpu_render_time", total_cpu_render_time);
        self.queue.submit(Some(encoder.finish()));
        frame.present();
        self.profiler
            .end_frame(&self.device, &self.queue, &mut ui_loader.variables);
    }

    pub fn update_render(
        &mut self,
        camera: &Camera,
        terrain_subsystem: &mut TerrainSubsystem,
        ui_loader: &mut UiButtonLoader,
        time: &TimeSystem,
        input_state: &mut InputState,
        settings: &Settings,
        road_subsystem: &RoadSubsystem,
        car_subsystem: &CarSubsystem,
        astronomy: &AstronomyState,
        aspect: f32,
        screen_size: UVec2,
    ) {
        self.update_defines();

        self.check_shader_changes(ui_loader);

        let (view, proj, view_proj) = camera.matrices();
        let prev_view_proj = camera.prev_view_proj;

        terrain_subsystem.make_pick_uniforms(&self.queue, &self.pipelines.buffers.pick, camera);
        self.render_manager.update_depth_params(DepthDebugParams {
            near: camera.near,
            far: camera.far,
            power: 30.0,
            reversed_z: settings.reversed_depth_z as u32,
            msaa_samples: self.msaa_samples,
        });
        self.update_uniforms(camera, astronomy, time, aspect, terrain_subsystem, settings);

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
            terrain_subsystem,
            road_subsystem,
            car_subsystem,
            astronomy,
        );
    }

    fn update_uniforms(
        &mut self,
        camera: &Camera,
        astronomy: &AstronomyState,
        time: &TimeSystem,
        aspect: f32,
        terrain_subsystem: &TerrainSubsystem,
        settings: &Settings,
    ) {
        let mut updater = UniformUpdater::new(&self.queue, &mut self.pipelines);
        updater.update_camera_uniforms(
            terrain_subsystem,
            astronomy,
            camera,
            time,
            aspect,
            settings,
            &self.config,
        );
        updater.update_fog_uniforms(&self.config, camera);
        updater.update_sky_uniforms(astronomy.moon_phase);
        updater.update_water_uniforms();
        updater.update_tonemapping_uniforms(&settings.tonemapping_state);
        updater.update_ssao_uniforms(time, settings, camera.prev_view_proj);
    }

    fn update_subsystems(
        &mut self,
        camera: &Camera,
        aspect: f32,
        settings: &Settings,
        input_state: &mut InputState,
        time: &TimeSystem,
        ui_loader: &mut UiButtonLoader,
        terrain_subsystem: &mut TerrainSubsystem,
        road_subsystem: &RoadSubsystem,
        car_subsystem: &CarSubsystem,
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

        self.ui_renderer.update(
            ui_loader,
            time,
            input_state,
            &self.queue,
            &PhysicalSize::new(self.config.width, self.config.height),
        );
        self.road_renderer.update(
            terrain_subsystem,
            road_subsystem,
            &self.device,
            &self.queue,
            camera,
            &mut self.gizmo,
        );

        self.gizmo.update(
            terrain_subsystem,
            &self.rt_subsystem,
            time.total_game_time,
            &road_subsystem.road_manager,
            settings,
            camera,
        );
        self.gizmo
            .update_orbit_gizmo(camera.target, camera.orbit_radius, astronomy.sun_dir, false);
    }

    fn execute_shadow_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        car_storage: &CarStorage,
        camera: &Camera,
        aspect: f32,
        _time: &TimeSystem,
        settings: &Settings,
    ) {
        for cascade_idx in 0..CSM_CASCADES {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("CSM Shadow Pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.pipelines.resources.csm_shadows.layer_views[cascade_idx],
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
            if settings.shadow_type != ShadowType::CSM {
                continue;
            }
            let shadow_buf = &self.pipelines.resources.csm_shadows.shadow_mat_buffers[cascade_idx];

            //if cascade_idx != 0 && cascade_idx != 1 {
            //     render_terrain_shadows(
            //         &mut pass,
            //         &mut self.render_manager,
            //         &self.terrain_renderer,
            //         &self.pipelines,
            //         settings,
            //         camera,
            //         aspect,
            //         shadow_buf,
            //         cascade_idx
            //     );
            //}
            render_roads_shadows(
                &mut pass,
                &mut self.render_manager,
                &self.road_renderer,
                &self.pipelines,
                settings,
                shadow_buf,
                cascade_idx,
            );
            render_cars_shadows(
                &mut pass,
                &mut self.render_manager,
                &mut self.rt_subsystem,
                &mut self.car_renderer,
                car_storage,
                &self.pipelines,
                settings,
                camera,
                shadow_buf,
                cascade_idx,
            )
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
        terrain_subsystem: &TerrainSubsystem,
        car_storage: &CarStorage,
        settings: &Settings,
        astronomy: &AstronomyState,
    ) {
        self.execute_world_pass(
            encoder,
            config,
            camera,
            terrain_subsystem,
            car_storage,
            settings,
            aspect,
        );

        gpu_timestamp!(encoder, &mut self.profiler, "GTAO", {
            self.execute_gtao_pass(encoder, settings, time);
        });

        gpu_timestamp!(encoder, &mut self.profiler, "RTwd", {
            let sun_up = astronomy.sun_dir.y > 0.0;
            if sun_up && (settings.shadow_type == ShadowType::RT) {
                render_ray_tracing(
                    encoder,
                    &self.config,
                    &mut self.rt_subsystem,
                    &mut self.render_manager,
                    &mut self.pipelines,
                    &mut self.profiler,
                    self.msaa_samples,
                );
            }
        });
        self.execute_fog_pass(encoder, settings);

        gpu_timestamp!(encoder, &mut self.profiler, "UI", {
            self.execute_ui_pass(encoder, ui_loader, time, input_state);
        });

        self.execute_debug_preview_pass(encoder, settings);

        gpu_timestamp!(encoder, &mut self.profiler, "Tonemap", {
            self.execute_tonemap_pass(encoder, surface_view, settings);
        });
    }

    fn execute_world_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        config: &RenderPassConfig,
        camera: &Camera,
        terrain_subsystem: &TerrainSubsystem,
        car_storage: &CarStorage,
        settings: &Settings,
        aspect: f32,
    ) {
        let mut pass = create_world_pass(encoder, &self.pipelines, config, self.msaa_samples);

        if !settings.show_world {
            return;
        }
        // 1. Sky

        // All frame time names must be lowercase, I decided. (Doesn't matter anyway, cuz I .lowercase() anyway.)
        render_sky(
            &mut pass,
            &mut self.render_manager,
            &mut self.profiler,
            &self.pipelines,
            settings,
            self.msaa_samples,
        );

        // 2. Terrain
        gpu_timestamp!(pass, &mut self.profiler, "Terrain", {
            render_terrain(
                &mut pass,
                &mut self.render_manager,
                &self.terrain_renderer,
                terrain_subsystem,
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

        // 5
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
        pass.forget_lifetime();
        let mut pass = create_instanced_pass(encoder, &self.pipelines, config, self.msaa_samples);
        // 6
        gpu_timestamp!(pass, &mut self.profiler, "Cars", {
            render_cars(
                &mut pass,
                &mut self.render_manager,
                &mut self.rt_subsystem,
                &mut self.car_renderer,
                car_storage,
                &self.pipelines,
                settings,
                camera,
                self.msaa_samples,
            );
        });
    }
    fn execute_gtao_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        settings: &Settings,
        time: &TimeSystem,
    ) {
        if !settings.show_world {
            return;
        }
        if settings.is_gtao_prep_off() {
            return;
        }
        let msaa_on = settings.msaa_samples > 1;
        let half_w = self.pipelines.post_fx.linear_depth_half.texture().width();
        let half_h = self.pipelines.post_fx.linear_depth_half.texture().height();
        let half_disp = [(half_w + 7) / 8, (half_h + 7) / 8, 1];

        // ── Pass 1: Prep ────────────────────────────────────────────────────
        let prep_name = if msaa_on {
            "gtao_prep_msaa"
        } else {
            "gtao_prep"
        };

        gpu_timestamp!(encoder, &mut self.profiler, "GTAO_Prep", {
            self.render_manager.compute(
                Some(encoder),
                prep_name,
                vec![
                    &self.pipelines.msaa.depth_sample, // Can be non-msaa
                    &self.pipelines.resolved.normal,
                ],
                vec![
                    &self.pipelines.post_fx.linear_depth_full,
                    &self.pipelines.post_fx.linear_depth_half,
                    &self.pipelines.post_fx.normal_half,
                ],
                &compute_shader_dir().join("gtao_prep.wgsl"),
                ComputePipelineOptions {
                    dispatch_size: half_disp,
                },
                &[BufferSet::from_uniform(&self.pipelines.buffers.camera)],
            );
        });
        if !settings.gtao_enabled {
            return;
        }
        // ── Pass 2: Generate + Temporal Accumulate ──────────────────────────
        // ── History ping-pong ───────────────────────────────────────────────
        let read_idx = (time.frame_count % 2) as usize;
        let write_idx = 1 - read_idx;
        let hw = half_w as f32;
        let hh = half_h as f32;

        gpu_timestamp!(encoder, &mut self.profiler, "GTAO_Generate", {
            self.render_manager.compute(
                Some(encoder),
                "gtao_generate_temporal",
                vec![
                    &self.pipelines.post_fx.linear_depth_half,
                    &self.pipelines.post_fx.normal_half,
                    &self.pipelines.resources.blue_noise,
                    &self.pipelines.post_fx.gtao_history[read_idx],
                    &self.pipelines.post_fx.motion_full,
                ],
                vec![&self.pipelines.post_fx.gtao_history[write_idx]],
                &compute_shader_dir().join("gtao_generate.wgsl"),
                ComputePipelineOptions {
                    dispatch_size: half_disp,
                },
                &[
                    BufferSet::from_uniform(&self.pipelines.buffers.camera),
                    BufferSet::from_uniform(&self.pipelines.buffers.gtao),
                ],
            );
        });

        // ── Pass 3: 2D Bilateral Blur ───────────────────────────────────────
        let blur_params = GtaoBlurParams {
            depth_sigma: 0.02,
            normal_sigma: 0.1,
            kernel_radius: 4,
            _padding: 0,
        };
        self.queue.write_buffer(
            &self.pipelines.buffers.gtao_blur,
            0,
            bytemuck::bytes_of(&blur_params),
        );

        gpu_timestamp!(encoder, &mut self.profiler, "GTAO_Blur", {
            self.render_manager.compute(
                Some(encoder),
                "gtao_blur_2d",
                vec![
                    &self.pipelines.post_fx.gtao_history[write_idx], // just-written accumulated AO
                    &self.pipelines.post_fx.linear_depth_half,
                    &self.pipelines.post_fx.normal_half,
                ],
                vec![&self.pipelines.post_fx.gtao_blurred_half],
                &compute_shader_dir().join("gtao_blur_2d.wgsl"),
                ComputePipelineOptions {
                    dispatch_size: half_disp,
                },
                &[BufferSet::from_uniform(&self.pipelines.buffers.gtao_blur)],
            );
        });

        // ── Pass 4: Upsample + Apply (render pass — blends into msaa/resolved HDR) ──
        let fw = self.pipelines.resolved.hdr.texture().width() as f32;
        let fh = self.pipelines.resolved.hdr.texture().height() as f32;

        let upsample_apply_params = GtaoUpsampleApplyParams {
            full_size: [fw, fh],
            half_size: [hw, hh],
            inv_full_size: [1.0 / fw, 1.0 / fh],
            inv_half_size: [1.0 / hw, 1.0 / hh],
            depth_threshold: 0.05,
            normal_threshold: 0.9,
            use_normal_check: 1,
            power: 1.5,
            apply_intensity: 1.0,
            min_ao: 0.1,
            debug_mode: 0,
            _padding: 0,
        };
        self.queue.write_buffer(
            &self.pipelines.buffers.gtao_upsample_apply,
            0,
            bytemuck::bytes_of(&upsample_apply_params),
        );

        let apply_name = if msaa_on {
            "gtao_upsample_apply_msaa"
        } else {
            "gtao_upsample_apply"
        };

        gpu_timestamp!(encoder, &mut self.profiler, "GTAO_Upsample_Apply", {
            let color_attachment = create_color_attachment_load(
                &self.pipelines.msaa.hdr,
                &self.pipelines.resolved.hdr,
                self.msaa_samples,
            );

            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("GTAO Upsample Apply"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            let options = PipelineOptions::default()
                .with_topology(TriangleList)
                .with_msaa(self.msaa_samples)
                .with_target(ColorTargetState {
                    format: self.pipelines.msaa.hdr.texture().format(),
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::Zero,
                            dst_factor: BlendFactor::Src,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent::OVER,
                    }),
                    write_mask: ColorWrites::ALL,
                });
            let textures: Vec<&TextureView> = vec![
                &self.pipelines.post_fx.gtao_blurred_half,
                &self.pipelines.post_fx.linear_depth_half,
                &self.pipelines.post_fx.normal_half,
                &self.pipelines.msaa.depth_sample,
                &self.pipelines.resolved.normal,
            ];

            self.render_manager.render_with_textures(
                &textures,
                shader_dir().join("gtao_upsample_apply.wgsl").as_path(),
                &options,
                &[
                    &self.pipelines.buffers.camera,
                    &self.pipelines.buffers.gtao_upsample_apply,
                ],
                &mut pass,
            );

            pass.draw(0..3, 0..1);
        });
    }

    fn execute_fog_pass(&mut self, encoder: &mut CommandEncoder, settings: &Settings) {
        if !settings.show_world || !settings.show_fog {
            return;
        }

        let msaa_on = self.msaa_samples > 1;

        gpu_timestamp!(encoder, &mut self.profiler, "Fog", {
            let color_attachment = create_color_attachment_load(
                &self.pipelines.msaa.hdr,
                &self.pipelines.resolved.hdr,
                self.msaa_samples,
            );

            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Fog Render Pass"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // Blend: dst_color = fog_col * fog_amt + hdr_color * (1 - fog_amt)
            //   src outputs: rgb = fog_col * fog_amt, a = 1 - fog_amt
            //   color: src*One + dst*SrcAlpha  → fog_col*fog_amt + hdr*(1-fog_amt)
            //   alpha: keep destination alpha
            let options = PipelineOptions::default()
                .with_topology(TriangleList)
                .with_msaa(self.msaa_samples)
                .with_target(ColorTargetState {
                    format: self.pipelines.msaa.hdr.texture().format(),
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::SrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent::OVER,
                    }),
                    write_mask: ColorWrites::ALL,
                });

            let textures: Vec<&TextureView> = vec![
                &self.pipelines.post_fx.linear_depth_half,
                &self.pipelines.msaa.depth_sample,
            ];

            let shader_name = if msaa_on { "fog_msaa" } else { "fog" };

            self.render_manager.render_with_textures(
                &textures,
                shader_dir().join("fog.wgsl").as_path(),
                &options,
                &[&self.pipelines.buffers.camera, &self.pipelines.buffers.fog],
                &mut pass,
            );

            pass.draw(0..3, 0..1);
        });
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

        self.render_ui(&mut pass, ui_loader, time, input_state);
    }

    fn execute_debug_preview_pass(&mut self, encoder: &mut CommandEncoder, settings: &Settings) {
        match settings.debug_view_state {
            DebugViewState::Off => {
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
                    &self.pipelines.msaa.depth_sample,
                    DebugVisualization::Depth,
                    &self.pipelines.msaa.hdr,
                    &mut pass,
                );
            }
            DebugViewState::GtaoRaw => {
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
                    &self.pipelines.post_fx.gtao_history[0],
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
                    &self.pipelines.post_fx.gtao_blurred_half,
                    DebugVisualization::RedToGrayscale,
                    &self.pipelines.msaa.hdr,
                    &mut pass,
                );
            }
            DebugViewState::RTRaw => {
                let color_attachment = create_color_attachment_load(
                    &self.pipelines.msaa.hdr,
                    &self.pipelines.resolved.hdr,
                    self.msaa_samples,
                );

                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("Debug Raw Ray Tracing Fullscreen Preview Pass"),
                    color_attachments: &[Some(color_attachment)],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                self.render_manager.render_fullscreen_debug(
                    &self.pipelines.post_fx.rt_raw_half,
                    DebugVisualization::RedToGrayscale,
                    &self.pipelines.msaa.hdr,
                    &mut pass,
                );
            }
            DebugViewState::Motion => {
                let color_attachment = create_color_attachment_load(
                    &self.pipelines.msaa.hdr,
                    &self.pipelines.resolved.hdr,
                    self.msaa_samples,
                );

                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("Motion Vectors Fullscreen Preview Pass"),
                    color_attachments: &[Some(color_attachment)],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                self.render_manager.render_fullscreen_debug(
                    &self.pipelines.post_fx.motion_full,
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

        self.render_manager
            .update_define("MSAA".to_string(), self.msaa_samples > 1);
        self.pipelines.resize(&self.config, self.msaa_samples);
        self.ui_renderer.pipelines.msaa_samples = self.msaa_samples;
        self.ui_renderer.pipelines.rebuild_pipelines();
        self.render_manager.invalidate_bind_groups();
    }

    fn reload_all_shaders(&mut self, changed: &[PathBuf]) -> anyhow::Result<()> {
        self.ui_renderer.reload_shaders()?;
        self.render_manager.reload_render_shaders(changed);
        self.render_manager.compute_system().invalidate_cache();
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

        match self.reload_all_shaders(changed.as_slice()) {
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
    }

    fn update_defines(&mut self) {
        let msaa_on = self.msaa_samples > 1;
        self.render_manager
            .update_define("MSAA".to_string(), msaa_on);
    }
}

pub(crate) fn create_surface_and_adapter(
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

pub(crate) fn create_surface_config(
    surface: &Surface,
    adapter: &Adapter,
    settings: &mut Settings,
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
    settings.msaa_samples = msaa_samples;
    (config, msaa_samples)
}

pub(crate) fn create_device(adapter: &Adapter) -> (Device, Queue) {
    pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: Some("Device"),
        required_features: Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | Features::TIMESTAMP_QUERY
            | Features::TIMESTAMP_QUERY_INSIDE_PASSES
            | Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
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
pub fn create_color_attachment_load<'a>(
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
