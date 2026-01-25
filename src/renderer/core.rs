use crate::components::camera::*;
use crate::data::Settings;
use crate::mouse_ray::*;
use crate::paths::{shader_dir, texture_dir};
use crate::positions::WorldPos;
use crate::renderer::astronomy::*;
use crate::renderer::general_mesh_arena::GeneralMeshArena;
use crate::renderer::gizmo::Gizmo;
use crate::renderer::pipelines::{DEPTH_FORMAT, Pipelines};
use crate::renderer::procedural_render_manager::{
    PipelineOptions, RenderManager, create_color_attachment_load,
};
use crate::renderer::procedural_texture_manager::{MaterialKind, Params, TextureCacheKey};
use crate::renderer::render_passes::{
    RenderPassConfig, create_color_attachment, create_depth_attachment,
};
use crate::renderer::shader_watcher::ShaderWatcher;
use crate::renderer::shadows::{
    CSM_CASCADES, ShadowMatUniform, render_roads_shadows, render_terrain_shadows,
};
use crate::renderer::ui::UiRenderer;
use crate::renderer::uniform_updates::UniformUpdater;
use crate::renderer::world_renderer::TerrainRenderer;
use crate::resources::{InputState, TimeSystem, Uniforms};
use crate::terrain::roads::road_mesh_manager::RoadVertex;
use crate::terrain::roads::road_mesh_renderer::RoadRenderSubsystem;
use crate::terrain::sky::{STAR_COUNT, STARS_VERTEX_LAYOUT};
use crate::terrain::water::SimpleVertex;
use crate::ui::ui_editor::UiButtonLoader;
use crate::ui::variables::update_ui_variables;
use crate::ui::vertex::{LineVtxRender, Vertex};
use crate::world::CameraBundle;
use glam::Mat4;
use std::sync::Arc;
use wgpu::PrimitiveTopology::TriangleList;
use wgpu::TextureFormat::Rgba8UnormSrgb;
use wgpu::*;
use winit::dpi::PhysicalSize;
use winit::window::Window;

pub struct RenderCore {
    pub surface: Surface<'static>,
    pub device: Device,
    pub queue: Queue,
    pub config: SurfaceConfiguration,
    pub msaa_samples: u32,
    pub pipelines: Pipelines,
    ui_renderer: UiRenderer,
    pub terrain_renderer: TerrainRenderer,
    pub road_renderer: RoadRenderSubsystem,
    size: PhysicalSize<u32>,
    shader_watcher: Option<ShaderWatcher>,
    encoder: Option<CommandEncoder>,
    arena: GeneralMeshArena,
    render_manager: RenderManager,
    pub gizmo: Gizmo,
    astronomy: AstronomyState,
}

impl RenderCore {
    pub fn new(window: Arc<Window>, settings: &Settings, camera: &Camera) -> Self {
        let (surface, adapter, size) = create_surface_and_adapter(window);
        let (config, msaa_samples) = create_surface_config(&surface, &adapter, settings, size);
        let (device, queue) = create_device(&adapter);
        surface.configure(&device, &config);

        let shader_dir = shader_dir();
        let shader_watcher = ShaderWatcher::new(&shader_dir).ok();

        let pipelines = Pipelines::new(
            &device,
            &config,
            msaa_samples,
            &shader_dir,
            camera,
            settings.shadow_map_size,
        )
        .expect("Failed to create render pipelines");
        let ui_renderer = UiRenderer::new(&device, config.format, size, msaa_samples, &shader_dir)
            .expect("Failed to create UI pipelines");
        let terrain_renderer = TerrainRenderer::new(&device, settings);
        let road_renderer = RoadRenderSubsystem::new(&device);
        let arena = GeneralMeshArena::new(&device, 256 * 1024 * 1024, 128 * 1024 * 1024);
        let render_manager = RenderManager::new(&device, &queue, config.format, texture_dir());
        let gizmo = Gizmo::new(&device, terrain_renderer.chunk_size);

        Self {
            surface,
            device,
            queue,
            config,
            pipelines,
            msaa_samples,
            ui_renderer,
            terrain_renderer,
            road_renderer,
            size,
            shader_watcher,
            encoder: None,
            arena,
            render_manager,
            gizmo,
            astronomy: AstronomyState::default(),
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
        let frame_result = acquire_frame(&self.surface, &self.device, &mut self.config);
        if frame_result.resized {
            self.pipelines.resize(&self.config, self.msaa_samples);
        }

        let frame = frame_result.frame;
        let surface_view = frame.texture.create_view(&TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        //self.execute_shadow_pass(&mut encoder, camera, aspect, time);
        self.execute_main_pass(
            &mut encoder,
            &surface_view,
            &RenderPassConfig::from_settings(settings),
            camera,
            aspect,
            time,
            input_state,
            ui_loader,
            settings.show_world,
        );

        self.queue.submit(Some(encoder.finish()));
        frame.present();
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

        update_ui_variables(ui_loader, &time_scales, &astronomy, observer.obliquity);

        let (view, proj, view_proj) = camera.matrices(aspect);
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

        let (new_uniforms, light_mats, splits) =
            self.update_uniforms(camera, view, proj, view_proj, &astronomy, time, aspect);
        self.pipelines.uniforms_cpu = new_uniforms;
        self.pipelines.cascaded_shadow_map.light_mats = light_mats;
        self.pipelines.cascaded_shadow_map.splits = splits;

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
    ) -> (Uniforms, [Mat4; CSM_CASCADES], [f32; 4]) {
        let updater = UniformUpdater::new(&self.queue, &self.pipelines);
        let (new_uniforms, light_mats, splits) = updater.update_camera_uniforms(
            view,
            proj,
            view_proj,
            astronomy,
            camera,
            time.total_time,
            aspect,
        );
        updater.update_fog_uniforms(&self.config, camera);
        updater.update_sky_uniforms(astronomy.moon_phase);
        updater.update_water_uniforms();
        (new_uniforms, light_mats, splits)
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
        //println!("{}", target_pos_render);
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
        self.ui_renderer
            .update(ui_loader, time, input_state, &self.queue, &self.size);
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
        self.gizmo.update_gizmo_vertices(
            camera.target,
            camera.orbit_radius,
            false,
            astronomy.sun_dir,
            self.terrain_renderer.chunk_size,
        );
    }

    fn execute_shadow_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        camera: &Camera,
        aspect: f32,
        _time: &TimeSystem,
    ) {
        for cascade in 0..CSM_CASCADES {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("CSM Shadow Pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.pipelines.cascaded_shadow_map.layer_views[cascade],
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            let shadow_buf = &self.pipelines.cascaded_shadow_map.shadow_mat_buffers[cascade];

            if cascade != 0 && cascade != 1 {
                render_terrain_shadows(
                    &mut pass,
                    &mut self.render_manager,
                    &self.terrain_renderer,
                    &self.pipelines,
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
        show_world: bool,
    ) {
        // -------- Pass 1: Main world pass (writes depth) --------
        self.execute_world_pass(encoder, surface_view, config, camera, aspect, show_world);

        // -------- Pass 2: Fog (samples depth, no depth attachment) --------
        self.execute_fog_pass(encoder, surface_view);

        // -------- Pass 3: UI (NOT fogged, because it draws after fog) Logic! --------
        self.execute_ui_pass(encoder, surface_view, ui_loader, time, input_state);

        // -------- Pass 4: Debug preview (disabled by default) --------
        //self.execute_debug_preview_pass(encoder, surface_view);
    }

    fn execute_world_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        surface_view: &TextureView,
        config: &RenderPassConfig,
        camera: &Camera,
        aspect: f32,
        show_world: bool,
    ) {
        let color_attachment = create_color_attachment(
            &self.pipelines.msaa_view,
            surface_view,
            self.msaa_samples,
            config.background_color,
        );

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Main Pass (World)"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: Some(create_depth_attachment(&self.pipelines.depth_view)),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        if show_world {
            render_sky(
                &mut pass,
                &mut self.render_manager,
                &self.pipelines,
                self.msaa_samples,
            );

            self.terrain_renderer.make_pick_uniforms(
                &self.queue,
                &self.pipelines.pick_uniforms.buffer,
                camera,
                self.terrain_renderer.chunk_size,
            );
            render_terrain(
                &mut pass,
                &mut self.render_manager,
                &self.terrain_renderer,
                &self.pipelines,
                self.msaa_samples,
                camera,
                aspect,
            );

            render_water(
                &mut pass,
                &mut self.render_manager,
                &self.pipelines,
                self.msaa_samples,
            );
        }

        render_roads(
            &mut pass,
            &mut self.render_manager,
            &self.road_renderer,
            &self.pipelines,
            self.msaa_samples,
        );
        render_gizmo(
            &mut pass,
            &mut self.render_manager,
            &self.pipelines,
            self.msaa_samples,
            &mut self.gizmo,
            camera,
            &self.device,
            &self.queue,
        );
    }

    fn execute_fog_pass(&mut self, encoder: &mut CommandEncoder, surface_view: &TextureView) {
        let color_attachment = create_color_attachment_load(
            &self.pipelines.msaa_view,
            surface_view,
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

        let fog_shader = if self.msaa_samples > 1 {
            "fog_msaa.wgsl"
        } else {
            "fog.wgsl"
        };
        self.render_manager.render_fog_fullscreen(
            "Fog",
            shader_dir().join(fog_shader).as_path(),
            self.msaa_samples,
            &self.pipelines.depth_sample_view,
            &[
                &self.pipelines.uniforms.buffer,
                &self.pipelines.fog_uniforms.buffer,
                &self.pipelines.pick_uniforms.buffer,
            ],
            &mut pass,
        );
    }

    fn execute_ui_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        surface_view: &TextureView,
        ui_loader: &mut UiButtonLoader,
        time: &TimeSystem,
        input_state: &InputState,
    ) {
        let color_attachment = create_color_attachment_load(
            &self.pipelines.msaa_view,
            surface_view,
            self.msaa_samples,
        );

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Main Pass (UI)"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: Some(create_depth_attachment(&self.pipelines.depth_view)),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        self.render_ui(&mut pass, ui_loader, time, input_state);
    }

    fn execute_debug_preview_pass(
        &mut self,
        encoder: &mut CommandEncoder,
        surface_view: &TextureView,
    ) {
        let color_attachment = create_color_attachment_load(
            &self.pipelines.msaa_view,
            surface_view,
            self.msaa_samples,
        );

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Depth View Fullscreen Preview Pass"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        self.render_manager.render_fullscreen_preview(
            &self.pipelines.cascaded_shadow_map.layer_views[0],
            "Shadow Map Fullscreen Render",
            self.msaa_samples,
            &mut pass,
        );
    }

    fn render_ui(
        &self,
        pass: &mut RenderPass,
        ui_loader: &mut UiButtonLoader,
        time: &TimeSystem,
        input_state: &InputState,
    ) {
        let screen_uniform = crate::renderer::ui::ScreenUniform {
            size: [self.size.width as f32, self.size.height as f32],
            time: time.total_time as f32,
            enable_dither: 1,
            mouse: input_state.mouse.pos.to_array(),
        };

        self.queue.write_buffer(
            &self.ui_renderer.pipelines.uniform_buffer,
            0,
            bytemuck::bytes_of(&screen_uniform),
        );

        self.ui_renderer.render(pass, ui_loader);
    }

    pub(crate) fn cycle_msaa(&mut self) {
        let supported = get_supported_msaa_levels(&self.device, self.config.format);
        let current_idx = supported
            .iter()
            .position(|&s| s == self.msaa_samples)
            .unwrap_or(0);
        self.msaa_samples = supported[(current_idx + 1) % supported.len()];

        println!("MSAA changed to {}x", self.msaa_samples);

        self.pipelines.resize(&self.config, self.msaa_samples);
        self.ui_renderer.pipelines.msaa_samples = self.msaa_samples;
        self.ui_renderer.pipelines.rebuild_pipelines();
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
                ui_loader.log_console(format!("✅ {label}"));
            }
            Err(err) => ui_loader.log_console(format!("❌ Shader reload failed: {err}")),
        }
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
    println!("{:?}", format);

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
        required_features: Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
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

fn get_supported_msaa_levels(device: &Device, format: TextureFormat) -> Vec<u32> {
    let caps = device.features();
    let format_caps = device.create_texture(&TextureDescriptor {
        label: Some("MSAA probe"),
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    drop(format_caps);

    let mut levels = vec![1];
    for count in [2, 4, 8] {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            device.create_texture(&TextureDescriptor {
                label: Some("MSAA probe"),
                size: Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: count,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        }));
        if result.is_ok() {
            levels.push(count);
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

pub struct FrameResult {
    pub frame: SurfaceTexture,
    pub resized: bool,
}

pub fn acquire_frame(
    surface: &Surface,
    device: &Device,
    config: &mut SurfaceConfiguration,
) -> FrameResult {
    match surface.get_current_texture() {
        Ok(frame) => FrameResult {
            frame,
            resized: false,
        },
        Err(SurfaceError::Outdated | SurfaceError::Lost) => {
            surface.configure(device, config);
            let frame = surface.get_current_texture().unwrap();

            let size = frame.texture.size();
            config.width = size.width;
            config.height = size.height;

            FrameResult {
                frame,
                resized: true,
            }
        }
        Err(e) => panic!("{e:?}"),
    }
}

fn road_material_keys() -> Vec<TextureCacheKey> {
    vec![
        TextureCacheKey {
            kind: MaterialKind::Concrete,
            params: Params {
                seed: 1,
                scale: 2.0,
                roughness: 1.0,
                color_primary: [0.32, 0.30, 0.28, 1.0],
                color_secondary: [0.15, 0.13, 0.10, 1.0],
                moisture: 0.0,
                shadow_strength: 0.0,
                sheen_strength: 0.0,
                ..Default::default()
            },
            resolution: 512,
        },
        TextureCacheKey {
            kind: MaterialKind::Goo,
            params: Params {
                seed: 0,
                scale: 3.0,
                roughness: 0.3,
                color_primary: [0.02, 0.02, 0.03, 1.0],
                color_secondary: [0.10, 0.10, 0.12, 1.0],
                moisture: 0.0,
                shadow_strength: 0.0,
                sheen_strength: 0.0,
                ..Default::default()
            },
            resolution: 512,
        },
        TextureCacheKey {
            kind: MaterialKind::Asphalt,
            params: Params {
                seed: 0,
                scale: 16.0,
                roughness: 0.5,
                color_primary: [0.004, 0.004, 0.004, 1.0],
                color_secondary: [0.015, 0.015, 0.015, 1.0],
                moisture: 0.0,
                shadow_strength: 0.0,
                sheen_strength: 0.0,
                ..Default::default()
            },
            resolution: 512,
        },
        TextureCacheKey {
            kind: MaterialKind::Asphalt,
            params: Params {
                seed: 1,
                scale: 16.0,
                roughness: 0.3,
                color_primary: [0.006, 0.006, 0.006, 1.0],
                color_secondary: [0.020, 0.020, 0.020, 1.0],
                moisture: 0.0,
                shadow_strength: 0.0,
                sheen_strength: 0.0,
                ..Default::default()
            },
            resolution: 512,
        },
        TextureCacheKey {
            kind: MaterialKind::Asphalt,
            params: Params {
                seed: 2,
                scale: 16.0,
                roughness: 0.5,
                color_primary: [0.04, 0.04, 0.006, 1.0],
                color_secondary: [0.120, 0.120, 0.120, 1.0],
                moisture: 0.0,
                shadow_strength: 0.0,
                sheen_strength: 0.0,
                ..Default::default()
            },
            resolution: 512,
        },
        TextureCacheKey {
            kind: MaterialKind::Asphalt,
            params: Params {
                seed: 3,
                scale: 16.0,
                roughness: 0.8,
                color_primary: [0.02, 0.02, 0.02, 1.0],
                color_secondary: [0.080, 0.080, 0.080, 1.0],
                moisture: 0.0,
                shadow_strength: 0.0,
                sheen_strength: 0.0,
                ..Default::default()
            },
            resolution: 512,
        },
    ]
}

fn terrain_material_keys() -> Vec<TextureCacheKey> {
    vec![
        TextureCacheKey {
            kind: MaterialKind::Grass,
            params: Params {
                seed: 1337,
                scale: 40.0,
                roughness: 0.78, // Heavily reduced dry influence
                moisture: 0.65,  // Max lush bias
                color_primary: [0.02, 0.34, 0.01, 1.0], // Deep muted olive green—no neon
                color_secondary: [0.10, 0.60, 0.10, 1.0], // Dark neutral brown, zero yellow pop
                shadow_strength: 1.80, // Hard punchy shadows
                sheen_strength: 0.05, // Barely any highlight
                ..Default::default()
            },
            resolution: 1024,
        },
        TextureCacheKey {
            kind: MaterialKind::Grass,
            params: Params {
                seed: 42,
                scale: 80.0,
                roughness: 0.65,                       // Way down—minimal dry yellow
                moisture: 0.70,                        // Balanced but not dead
                color_primary: [0.0, 0.33, 0.00, 1.0], // Ultra-dark muted base
                color_secondary: [0.02, 0.50, 0.00, 1.0], // Super dark brown shadow tone
                shadow_strength: 1.75,                 // Even deeper volume
                sheen_strength: 0.02,                  // None basically
                ..Default::default()
            },
            resolution: 1024,
        },
    ]
}
fn road_depth_stencil(bias: DepthBiasState) -> DepthStencilState {
    DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: true,
        depth_compare: CompareFunction::LessEqual,
        stencil: Default::default(),
        bias,
    }
}

fn render_gizmo(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    pipelines: &Pipelines,
    msaa_samples: u32,
    gizmo: &mut Gizmo,
    camera: &Camera,
    device: &Device,
    queue: &Queue,
) {
    render_manager.render(
        Vec::new(),
        "Gizmo",
        shader_dir().join("lines.wgsl").as_path(),
        PipelineOptions {
            topology: PrimitiveTopology::LineList,
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            msaa_samples,
            vertex_layouts: Vec::from([LineVtxRender::layout()]),
            cull_mode: None,
            blend: Some(BlendState::REPLACE),
            shadow_pass: false,
        },
        &[&pipelines.uniforms.buffer],
        pass,
        pipelines,
    );

    let vertex_count = gizmo.update_buffer(device, queue, camera.eye_world());
    pass.set_vertex_buffer(0, gizmo.gizmo_buffer.slice(..));
    pass.draw(0..vertex_count, 0..1);
    gizmo.clear();
}

fn render_water(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    pipelines: &Pipelines,
    msaa_samples: u32,
) {
    render_manager.render(
        Vec::new(),
        "Water",
        shader_dir().join("water.wgsl").as_path(),
        PipelineOptions {
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
            cull_mode: None,
            blend: Some(BlendState::ALPHA_BLENDING),
            shadow_pass: false,
        },
        &[
            &pipelines.uniforms.buffer,
            &pipelines.water_uniforms.buffer,
            &pipelines.sky_uniforms.buffer,
        ],
        pass,
        pipelines,
    );

    pass.set_stencil_reference(1);
    pass.set_vertex_buffer(0, pipelines.water_mesh_buffers.vertex.slice(..));
    pass.set_index_buffer(
        pipelines.water_mesh_buffers.index.slice(..),
        IndexFormat::Uint32,
    );
    pass.draw_indexed(0..pipelines.water_mesh_buffers.index_count, 0, 0..1);
}

fn render_sky(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    pipelines: &Pipelines,
    msaa_samples: u32,
) {
    let sky_depth_stencil = Some(DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: false,
        depth_compare: CompareFunction::Always,
        stencil: Default::default(),
        bias: Default::default(),
    });

    render_manager.render(
        Vec::new(),
        "Stars",
        shader_dir().join("stars.wgsl").as_path(),
        PipelineOptions {
            topology: PrimitiveTopology::TriangleStrip,
            depth_stencil: sky_depth_stencil.clone(),
            msaa_samples,
            vertex_layouts: Vec::from([STARS_VERTEX_LAYOUT]),
            cull_mode: None,
            blend: Some(BlendState::ALPHA_BLENDING),
            shadow_pass: false,
        },
        &[&pipelines.uniforms.buffer, &pipelines.sky_uniforms.buffer],
        pass,
        pipelines,
    );
    pass.set_vertex_buffer(0, pipelines.stars_mesh_buffers.vertex.slice(..));
    pass.draw(0..4, 0..STAR_COUNT);

    render_manager.render(
        Vec::new(),
        "Sky",
        shader_dir().join("sky.wgsl").as_path(),
        PipelineOptions {
            topology: Default::default(),
            depth_stencil: sky_depth_stencil,
            msaa_samples,
            vertex_layouts: Vec::new(),
            cull_mode: None,
            blend: Some(BlendState::ALPHA_BLENDING),
            shadow_pass: false,
        },
        &[&pipelines.uniforms.buffer, &pipelines.sky_uniforms.buffer],
        pass,
        pipelines,
    );
    pass.draw(0..3, 0..1);
}

fn render_roads(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    road_renderer: &RoadRenderSubsystem,
    pipelines: &Pipelines,
    msaa_samples: u32,
) {
    let keys = road_material_keys();
    let shader_path = shader_dir().join("road.wgsl");

    let base_bias = DepthBiasState {
        constant: -3,
        slope_scale: -2.0,
        clamp: 0.0,
    };
    render_manager.render(
        keys.clone(),
        "Roads",
        shader_path.as_path(),
        PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(road_depth_stencil(base_bias)),
            msaa_samples,
            vertex_layouts: Vec::from([RoadVertex::layout()]),
            cull_mode: Some(Face::Back),
            blend: Some(BlendState::ALPHA_BLENDING),
            shadow_pass: false,
        },
        &[
            &pipelines.uniforms.buffer,
            &road_renderer.road_appearance.normal_buffer,
        ],
        pass,
        pipelines,
    );

    for chunk_id in &road_renderer.visible_draw_list {
        if let Some(gpu) = road_renderer.chunk_gpu.get(chunk_id) {
            pass.set_vertex_buffer(0, gpu.vertex.slice(..));
            pass.set_index_buffer(gpu.index.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..gpu.index_count, 0, 0..1);
        }
    }

    if road_renderer.preview_gpu.is_empty() {
        return;
    }
    let (Some(vb), Some(ib)) = (&road_renderer.preview_gpu.vb, &road_renderer.preview_gpu.ib)
    else {
        return;
    };

    let preview_bias = DepthBiasState {
        constant: -4,
        slope_scale: -2.0,
        clamp: 0.0,
    };
    render_manager.render(
        keys,
        "Roads",
        shader_path.as_path(),
        PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(road_depth_stencil(preview_bias)),
            msaa_samples,
            vertex_layouts: Vec::from([RoadVertex::layout()]),
            cull_mode: Some(Face::Back),
            blend: Some(BlendState::ALPHA_BLENDING),
            shadow_pass: false,
        },
        &[
            &pipelines.uniforms.buffer,
            &road_renderer.road_appearance.preview_buffer,
        ],
        pass,
        pipelines,
    );

    pass.set_vertex_buffer(0, vb.slice(..));
    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
    pass.draw_indexed(0..road_renderer.preview_gpu.index_count, 0, 0..1);
}

fn render_terrain(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    terrain_renderer: &TerrainRenderer,
    pipelines: &Pipelines,
    msaa_samples: u32,
    camera: &Camera,
    aspect: f32,
) {
    let keys = terrain_material_keys();
    let shader_path = shader_dir().join("terrain.wgsl");

    let make_stencil = |write_mask: u32| -> DepthStencilState {
        DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: CompareFunction::LessEqual,
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

    pass.set_stencil_reference(0);
    render_manager.render(
        keys.clone(),
        "Terrain Pipeline (Above Water)",
        shader_path.as_path(),
        PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(make_stencil(0)),
            msaa_samples,
            vertex_layouts: Vec::from([Vertex::desc()]),
            blend: Some(BlendState::REPLACE),
            cull_mode: Some(Face::Front),
            shadow_pass: false,
        },
        &[&pipelines.uniforms.buffer, &pipelines.pick_uniforms.buffer],
        pass,
        pipelines,
    );
    terrain_renderer.render(pass, camera, aspect, false);

    pass.set_stencil_reference(1);
    render_manager.render(
        keys,
        "Terrain Pipeline (Under Water)",
        shader_path.as_path(),
        PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(make_stencil(0xFF)),
            msaa_samples,
            vertex_layouts: Vec::from([Vertex::desc()]),
            blend: Some(BlendState::REPLACE),
            cull_mode: Some(Face::Front),
            shadow_pass: false,
        },
        &[&pipelines.uniforms.buffer, &pipelines.pick_uniforms.buffer],
        pass,
        pipelines,
    );
    terrain_renderer.render(pass, camera, aspect, true);
}
