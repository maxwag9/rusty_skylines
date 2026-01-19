use crate::components::camera::*;
use crate::data::Settings;
use crate::mouse_ray::*;
use crate::paths::{shader_dir, texture_dir};
use crate::renderer::astronomy::*;
use crate::renderer::general_mesh_arena::GeneralMeshArena;
use crate::renderer::gizmo::Gizmo;
use crate::renderer::pipelines::{DEPTH_FORMAT, Pipelines};
use crate::renderer::procedural_render_manager::{PipelineOptions, RenderManager};
use crate::renderer::procedural_texture_manager::{MaterialKind, Params, TextureCacheKey};
use crate::renderer::render_passes::{
    RenderPassConfig, create_color_attachment, create_depth_attachment,
};
use crate::renderer::shader_watcher::ShaderWatcher;
use crate::renderer::ui::UiRenderer;
use crate::renderer::uniform_updates::UniformUpdater;
use crate::renderer::world_renderer::TerrainRenderer;
use crate::resources::{InputState, TimeSystem};
use crate::terrain::roads::road_mesh_manager::RoadVertex;
use crate::terrain::roads::road_mesh_renderer::RoadRenderSubsystem;
use crate::terrain::sky::{STAR_COUNT, STARS_VERTEX_LAYOUT};
use crate::terrain::water::SimpleVertex;
use crate::ui::ui_editor::UiButtonLoader;
use crate::ui::variables::update_ui_variables;
use crate::ui::vertex::{LineVtx, Vertex};
use crate::world::CameraBundle;
use std::sync::Arc;
use wgpu::PrimitiveTopology::TriangleList;
use wgpu::TextureFormat::Rgba8UnormSrgb;
use wgpu::*;
use winit::dpi::PhysicalSize;
use winit::window::Window;
// top-level (once)

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
}

impl RenderCore {
    pub fn new(window: Arc<Window>, settings: &Settings, camera: &Camera) -> Self {
        // --- Create instance and surface ---
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
        });

        let size = window.inner_size();
        let surface = instance
            .create_surface(window.clone())
            .expect("Surface creation failed");

        // --- Pick an adapter (blocking internally) ---
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("No suitable GPU adapters found");

        println!("Backend: {:?}", adapter.get_info().backend);

        // --- Configure surface ---
        let surface_caps = surface.get_capabilities(&adapter);
        let preferred_format = Rgba8UnormSrgb;
        let format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| *f == preferred_format)
            .unwrap_or(surface_caps.formats[0]);
        println!("{:?}", format);

        let alpha_mode = surface_caps
            .alpha_modes
            .iter()
            .copied()
            .find(|m| *m == CompositeAlphaMode::PostMultiplied)
            .unwrap_or(CompositeAlphaMode::Opaque);
        let user_mode = settings.present_mode.clone().to_wgpu();
        let present_mode = pick_fail_safe_present_mode(&surface, &adapter, user_mode);
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

        let mut msaa_samples = 4;
        let caps = adapter.get_texture_format_features(config.format);
        let supported = caps.flags.intersects(
            TextureFormatFeatureFlags::MULTISAMPLE_X8
                | TextureFormatFeatureFlags::MULTISAMPLE_X4
                | TextureFormatFeatureFlags::MULTISAMPLE_X2,
        );
        let can_render = caps
            .allowed_usages
            .contains(TextureUsages::RENDER_ATTACHMENT);
        if supported && can_render {
            if msaa_samples < 8 {
                println!("8x MSAA supported, but using {}x!", msaa_samples);
            } else {
                println!("8x MSAA supported, using {}x!", msaa_samples);
            }
        } else if msaa_samples == 8 {
            println!("Falling back to 4x");
            msaa_samples = 4;
        }
        // Request device + queue
        let features = Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        let limits = Limits::default();
        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
            label: Some("Device"),
            required_features: features,
            required_limits: limits,
            experimental_features: ExperimentalFeatures::disabled(),
            memory_hints: MemoryHints::default(),
            trace: Trace::Off,
        }))
        .expect("Device creation failed");

        // Configure surface
        surface.configure(&device, &config);

        let shader_dir = shader_dir();
        let shader_watcher = ShaderWatcher::new(&shader_dir).ok();

        let _aspect = config.width as f32 / config.height as f32;
        let pipelines = Pipelines::new(&device, &config, msaa_samples, &shader_dir, camera)
            .expect("Failed to create render pipelines");
        let ui_renderer = UiRenderer::new(&device, config.format, size, msaa_samples, &shader_dir)
            .expect("Failed to create UI pipelines");
        let world = TerrainRenderer::new(&device, settings);
        let road_renderer = RoadRenderSubsystem::new(&device);
        let arena = GeneralMeshArena::new(
            &device,
            256 * 1024 * 1024, // vertex bytes per page
            128 * 1024 * 1024, // index bytes per page
        );

        let render_manager = RenderManager::new(&device, &queue, Rgba8UnormSrgb, texture_dir());
        let gizmo = Gizmo::new(&device);
        Self {
            surface,
            device,
            queue,
            config,
            pipelines,
            msaa_samples,
            ui_renderer,
            terrain_renderer: world,
            road_renderer,
            size,
            shader_watcher,
            encoder: None,
            arena,
            render_manager,
            gizmo,
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

        let cam_pos = camera.position();
        let orbit_radius = camera.orbit_radius;

        // Compute time and astronomy
        let time_scales = TimeScales::from_game_time(time.total_game_time, settings.always_day);
        let observer = ObserverParams::new(time_scales.day_angle);
        let astronomy = compute_astronomy(&time_scales);

        // Update UI variables
        update_ui_variables(ui_loader, &time_scales, &astronomy, observer.obliquity);

        // Camera matrices and ray picking
        let (view, proj, view_proj) = camera.matrices(aspect);
        let ray = ray_from_mouse_pixels(
            glam::Vec2::new(input_state.mouse.pos.x, input_state.mouse.pos.y),
            self.config.width as f32,
            self.config.height as f32,
            view,
            proj,
        );
        self.terrain_renderer.pick_terrain_point(ray);

        // Update all uniforms
        let uniform_updater = UniformUpdater::new(&self.queue, &self.pipelines);
        uniform_updater.update_camera_uniforms(
            view,
            proj,
            view_proj,
            &astronomy,
            cam_pos,
            orbit_radius,
            time.total_time as f32,
        );
        uniform_updater.update_fog_uniforms(&self.config, view, camera.position().y);
        uniform_updater.update_sky_uniforms(astronomy.moon_phase);
        uniform_updater.update_water_uniforms();

        // Update world
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
            &mut self.gizmo,
        );

        self.gizmo.update(
            &self.terrain_renderer,
            camera.target,
            time.total_game_time,
            &self.road_renderer.road_manager,
        );
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

        // Acquire frame
        let frame_result = acquire_frame(&self.surface, &self.device, &mut self.config);
        if frame_result.resized {
            self.pipelines.resize(&self.config, self.msaa_samples);
        }
        let frame = frame_result.frame;
        let surface_view = frame.texture.create_view(&TextureViewDescriptor::default());

        // Create encoder and execute render passes
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let pass_config = RenderPassConfig::from_settings(settings);
        self.execute_main_pass(
            &mut encoder,
            &surface_view,
            &pass_config,
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
        let color_attachment = create_color_attachment(
            &self.pipelines.msaa_view,
            surface_view,
            self.pipelines.msaa_samples,
            config.background_color,
        );

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Main Pass"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: Some(create_depth_attachment(&self.pipelines.depth_view)),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        if show_world {
            // Sky
            render_sky(
                &mut pass,
                &mut self.render_manager,
                &self.pipelines,
                self.msaa_samples,
            );

            // Terrain
            self.terrain_renderer
                .make_pick_uniforms(&self.queue, &self.pipelines.pick_uniforms.buffer);

            render_terrain(
                &mut pass,
                &mut self.render_manager,
                &self.terrain_renderer,
                &self.pipelines,
                self.msaa_samples,
                camera,
                aspect,
            );

            // Water
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

        // Gizmo
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

        // UI
        self.render_ui(&mut pass, ui_loader, time, input_state);

        // let view = self.render_manager.procedural_texture_manager_mut().request_texture().clone();
        // self.render_manager.render_fullscreen_preview(&view, "Road Preview", 4, &mut pass);
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
        // Pick next MSAA value
        self.msaa_samples = match self.msaa_samples {
            1 => 2,
            2 => 4,
            4 => 8,
            _ => 1,
        };

        println!("MSAA changed to {}x", self.msaa_samples);

        // Recreate pipelines with new sample count
        self.pipelines.msaa_samples = self.msaa_samples;
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
    let line_shader_path = shader_dir().join("lines.wgsl");
    render_manager.render(
        Vec::new(),
        "Gizmo",
        line_shader_path.as_path(), // file containing full vertex+fragment shader
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
            vertex_layouts: Vec::from([LineVtx::layout()]),
            cull_mode: None,
            blend: Some(BlendState::REPLACE),
        },
        &[&pipelines.uniforms.buffer],
        pass,
    );
    gizmo.update_gizmo_vertices(camera.target, camera.orbit_radius, false);
    let vertex_count = gizmo.update_buffer(device, queue);
    pass.set_vertex_buffer(0, gizmo.gizmo_buffer.slice(..));
    pass.draw(0..vertex_count, 0..1);
    gizmo.clear(); // Clears Gizmo
}
fn render_water(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    pipelines: &Pipelines,
    msaa_samples: u32,
) {
    let water_shader_path = shader_dir().join("water.wgsl");
    render_manager.render(
        Vec::new(),
        "Water",
        water_shader_path.as_path(), // file containing full vertex+fragment shader
        PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
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
        },
        &[
            &pipelines.uniforms.buffer,
            &pipelines.water_uniforms.buffer,
            &pipelines.sky_uniforms.buffer,
        ],
        pass,
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
    let stars_shader_path = shader_dir().join("stars.wgsl");
    render_manager.render(
        Vec::new(),
        "Stars",
        stars_shader_path.as_path(), // file containing full vertex+fragment shader
        PipelineOptions {
            topology: PrimitiveTopology::TriangleStrip,
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            msaa_samples,
            vertex_layouts: Vec::from([STARS_VERTEX_LAYOUT]),
            cull_mode: None,
            blend: Some(BlendState::ALPHA_BLENDING),
        },
        &[&pipelines.uniforms.buffer, &pipelines.sky_uniforms.buffer],
        pass,
    );
    pass.set_vertex_buffer(0, pipelines.stars_mesh_buffers.vertex.slice(..));
    pass.draw(0..4, 0..STAR_COUNT); // 4 verts, STAR_COUNT instances

    let sky_shader_path = shader_dir().join("sky.wgsl");
    render_manager.render(
        Vec::new(),
        "Sky",
        sky_shader_path.as_path(), // file containing full vertex+fragment shader
        PipelineOptions {
            topology: Default::default(),
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            msaa_samples,
            vertex_layouts: Vec::new(),
            cull_mode: None,
            blend: Some(BlendState::ALPHA_BLENDING),
        },
        &[&pipelines.uniforms.buffer, &pipelines.sky_uniforms.buffer],
        pass,
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
    // Ordered list of ALL possible road texture keys — the order determines layer index 0,1,2,...
    let concrete_key = TextureCacheKey {
        kind: MaterialKind::Concrete,
        params: Params {
            seed: 1,
            scale: 2.0,
            roughness: 1.0,
            _padding: 0,
            color_primary: [0.32, 0.30, 0.28, 1.0], // Light gray
            color_secondary: [0.15, 0.13, 0.10, 1.0], // Darker gray
        },
        resolution: 512,
    };
    let goo_key = TextureCacheKey {
        kind: MaterialKind::Goo,
        params: Params {
            seed: 0,
            scale: 3.0,
            roughness: 0.3,
            _padding: 0,
            color_primary: [0.02, 0.02, 0.03, 1.0], // Near black
            color_secondary: [0.10, 0.10, 0.12, 1.0], // Slight blue-ish sheen
        },
        resolution: 512,
    };
    let newest_asphalt_key = TextureCacheKey {
        kind: MaterialKind::Asphalt,
        params: Params {
            seed: 0,
            scale: 16.0,
            roughness: 0.5,
            _padding: 0,
            color_primary: [0.004, 0.004, 0.004, 1.0],
            color_secondary: [0.015, 0.015, 0.015, 1.0],
        },
        resolution: 512,
    };
    let worn_newest_asphalt_key = TextureCacheKey {
        kind: MaterialKind::Asphalt,
        params: Params {
            seed: 1,
            scale: 16.0,
            roughness: 0.3,
            _padding: 0,
            color_primary: [0.006, 0.006, 0.006, 1.0],
            color_secondary: [0.020, 0.020, 0.020, 1.0],
        },
        resolution: 512,
    };
    let orange_asphalt_key = TextureCacheKey {
        kind: MaterialKind::Asphalt,
        params: Params {
            seed: 2,
            scale: 16.0,
            roughness: 0.5,
            _padding: 0,
            color_primary: [0.04, 0.04, 0.006, 1.0],
            color_secondary: [0.120, 0.120, 0.120, 1.0],
        },
        resolution: 512,
    };
    let gray_asphalt_key = TextureCacheKey {
        kind: MaterialKind::Asphalt,
        params: Params {
            seed: 3,
            scale: 16.0,
            roughness: 0.8,
            _padding: 0,
            color_primary: [0.02, 0.02, 0.02, 1.0],
            color_secondary: [0.080, 0.080, 0.080, 1.0],
        },
        resolution: 512,
    };
    let road_material_keys = vec![
        concrete_key,            // 0
        goo_key,                 // 1
        newest_asphalt_key,      // 2
        worn_newest_asphalt_key, // 3
        orange_asphalt_key,      // 4
        gray_asphalt_key,        // 5
    ];

    // This sets the road pipeline + texture array bind group (bind group 0)
    let road_shader_path = shader_dir().join("road.wgsl");
    render_manager.render(
        road_material_keys.clone(),
        "Roads",
        road_shader_path.as_path(), // file containing full vertex+fragment shader
        PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            msaa_samples,
            vertex_layouts: Vec::from([RoadVertex::layout()]),
            cull_mode: Some(Face::Back),
            blend: Some(BlendState::ALPHA_BLENDING),
        },
        &[
            &pipelines.uniforms.buffer,
            &road_renderer.road_appearance.normal_buffer,
        ],
        pass,
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

    let nudge_bias = DepthBiasState {
        constant: -4,
        slope_scale: -2.0,
        clamp: 0.0,
    };
    render_manager.render(
        road_material_keys,
        "Roads",
        road_shader_path.as_path(), // file containing full vertex+fragment shader
        PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: nudge_bias,
            }),
            msaa_samples,
            vertex_layouts: Vec::from([RoadVertex::layout()]),
            cull_mode: Some(Face::Back),
            blend: Some(BlendState::ALPHA_BLENDING),
        },
        &[
            &pipelines.uniforms.buffer,
            &road_renderer.road_appearance.preview_buffer,
        ],
        pass,
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
    let grass_key = TextureCacheKey {
        kind: MaterialKind::Grass,
        params: Params {
            seed: 3,
            scale: 16.0,
            roughness: 0.7,
            _padding: 0,
            color_primary: [0.02, 0.02, 0.02, 1.0],
            color_secondary: [0.080, 0.080, 0.080, 1.0],
        },
        resolution: 512,
    };
    let terrain_material_keys = vec![
        grass_key, // 0
    ];
    pass.set_stencil_reference(0);
    let terrain_shader_path = shader_dir().join("terrain.wgsl");
    render_manager.render(
        terrain_material_keys.clone(),
        "Terrain Pipeline (Above Water)",
        terrain_shader_path.as_path(), // file containing full vertex+fragment shader
        PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(DepthStencilState {
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
                    write_mask: 0,
                },
                bias: Default::default(),
            }),
            msaa_samples,
            vertex_layouts: Vec::from([Vertex::desc()]),
            blend: Some(BlendState::REPLACE),
            cull_mode: Some(Face::Front),
        },
        &[
            &pipelines.uniforms.buffer,
            &pipelines.fog_uniforms.buffer,
            &pipelines.pick_uniforms.buffer,
        ],
        pass,
    );
    terrain_renderer.render(pass, camera, aspect, false);

    pass.set_stencil_reference(1);
    render_manager.render(
        terrain_material_keys,
        "Terrain Pipeline (Under Water)",
        terrain_shader_path.as_path(), // file containing full vertex+fragment shader
        PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(DepthStencilState {
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
                    write_mask: 0xFF,
                },
                bias: Default::default(),
            }),
            msaa_samples,
            vertex_layouts: Vec::from([Vertex::desc()]),
            blend: Some(BlendState::REPLACE),
            cull_mode: Some(Face::Front),
        },
        &[
            &pipelines.uniforms.buffer,
            &pipelines.fog_uniforms.buffer,
            &pipelines.pick_uniforms.buffer,
        ],
        pass,
    );
    terrain_renderer.render(pass, camera, aspect, true);
}
fn pick_fail_safe_present_mode(
    surface: &Surface,
    adapter: &Adapter,
    user_mode: PresentMode,
) -> PresentMode {
    let caps = surface.get_capabilities(adapter);

    // Fallback priority. Mailbox first, then Fifo (always supported), then Immediate.
    let fallback_chain = [
        user_mode,
        PresentMode::Mailbox,
        PresentMode::Immediate,
        PresentMode::Fifo, // required by spec...
    ];

    for mode in fallback_chain {
        if caps.present_modes.contains(&mode) {
            return mode;
        }
    }

    // Absolute guarantee
    PresentMode::Fifo
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
