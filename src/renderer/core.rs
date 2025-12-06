use crate::components::camera::Camera;
use crate::data::Settings;
use crate::paths::shader_dir;
use crate::renderer::pipelines::{FogUniforms, Pipelines, make_new_uniforms};
use crate::renderer::shader_watcher::ShaderWatcher;
use crate::renderer::ui::UiRenderer;
use crate::renderer::world_renderer::WorldRenderer;
use crate::resources::TimeSystem;
use crate::ui::input::MouseState;
use crate::ui::ui_editor::UiButtonLoader;
use crate::ui::vertex::LineVtx;
use std::sync::Arc;
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
    pub world: WorldRenderer,

    size: PhysicalSize<u32>,

    shader_watcher: Option<ShaderWatcher>,
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
        let format = surface_caps.formats[0];
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

        let aspect = config.width as f32 / config.height as f32;
        let pipelines = Pipelines::new(&device, &config, msaa_samples, &shader_dir, camera, aspect)
            .expect("Failed to create render pipelines");
        let mut ui_renderer =
            UiRenderer::new(&device, config.format, size, msaa_samples, &shader_dir)
                .expect("Failed to create UI pipelines");
        let font_ttf: &[u8] = include_bytes!("../../data/ui_data/ttf/JetBrainsMono-Regular.ttf");
        let _ = ui_renderer.build_text_atlas(&device, &queue, font_ttf, &[14, 18, 24], 1024, 1024);
        let world = WorldRenderer::new(&device);

        Self {
            surface,
            device,
            queue,
            config,
            pipelines,
            msaa_samples,
            ui_renderer,
            world,
            size,
            shader_watcher,
        }
    }

    pub(crate) fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.pipelines.resize(self.msaa_samples);
    }

    pub(crate) fn render(
        &mut self,
        camera: &Camera,
        ui_loader: &mut UiButtonLoader,
        time: &TimeSystem,
        mouse: &MouseState,
        settings: &Settings,
    ) {
        self.check_shader_changes(ui_loader);

        // camera + sun
        let aspect = self.config.width as f32 / self.config.height as f32;
        let vp = camera.view_proj(aspect);
        let cam_pos = camera.position();
        let sun = glam::Vec3::new(0.3, 1.0, 0.6).normalize();

        let new_uniforms = make_new_uniforms(vp.to_cols_array_2d(), sun, cam_pos);
        self.queue.write_buffer(
            &self.pipelines.uniform_buffer,
            0,
            bytemuck::bytes_of(&new_uniforms),
        );

        // proj params for fog depth reconstruction
        let proj_params = [
            vp.col(2).z, // proj[2][2]
            vp.col(3).z, // proj[3][2]
        ];

        let fog_uniforms = FogUniforms {
            screen_size: [self.config.width as f32, self.config.height as f32],
            proj_params,
            fog_density: 0.0002,
            fog_height: 0.0,
            cam_height: camera.position().y,
            _pad0: 0.0,
            fog_color: [0.55, 0.55, 0.6],
            _pad1: 0.0,
            fog_sky_factor: 0.4,
            fog_height_falloff: 0.12,
            fog_start: 1000.0,
            fog_end: 10000.0,
        };

        self.queue.write_buffer(
            &self.pipelines.fog_uniform_buffer,
            0,
            bytemuck::bytes_of(&fog_uniforms),
        );

        self.world.update(&self.device, camera.target);

        // get frame
        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(_) => {
                self.surface.configure(&self.device, &self.config);
                self.surface.get_current_texture().unwrap()
            }
        };
        let surface_view = frame.texture.create_view(&TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // gizmo verts (unchanged)
        let t = camera.target;
        let s = camera.radius * 0.2;
        let axes = [
            // X
            LineVtx {
                pos: [t.x, t.y, t.z],
                color: [1.0, 0.2, 0.2],
            },
            LineVtx {
                pos: [t.x + s, t.y, t.z],
                color: [1.0, 0.2, 0.2],
            },
            // Y
            LineVtx {
                pos: [t.x, t.y, t.z],
                color: [0.2, 1.0, 0.2],
            },
            LineVtx {
                pos: [t.x, t.y + s, t.z],
                color: [0.2, 1.0, 0.2],
            },
            // Z
            LineVtx {
                pos: [t.x, t.y, t.z],
                color: [0.2, 0.6, 1.0],
            },
            LineVtx {
                pos: [t.x, t.y, t.z + s],
                color: [0.2, 0.6, 1.0],
            },
        ];
        self.queue
            .write_buffer(&self.pipelines.gizmo_vbuf, 0, bytemuck::cast_slice(&axes));

        let background_color = Color {
            r: settings.background_color[0] as f64,
            g: settings.background_color[1] as f64,
            b: settings.background_color[2] as f64,
            a: settings.background_color[3] as f64,
        };

        // 1) MAIN 3D + FOG pass (MSAA -> resolve into surface_view)
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &self.pipelines.msaa_view,
                    resolve_target: Some(&surface_view), // resolve MSAA into swapchain
                    depth_slice: None,
                    ops: Operations {
                        load: LoadOp::Clear(background_color),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.pipelines.depth_view, // same sample_count as msaa_view
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // main terrain: bind camera + fog
            pass.set_pipeline(&self.pipelines.pipeline);
            pass.set_bind_group(0, &self.pipelines.uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.pipelines.fog_bind_group, &[]);

            self.world
                .render(&mut pass, &self.pipelines, camera, aspect);

            // gizmo (can use fog too, or just camera)
            pass.set_pipeline(&self.pipelines.gizmo_pipeline);
            pass.set_bind_group(0, &self.pipelines.uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.pipelines.fog_bind_group, &[]);
            pass.set_vertex_buffer(0, self.pipelines.gizmo_vbuf.slice(..));
            pass.draw(0..6, 0..1);

            // 2) UI pass on top of resolved image

            let screen_uniform = crate::renderer::ui::ScreenUniform {
                size: [self.size.width as f32, self.size.height as f32],
                time: time.total_time,
                enable_dither: 1,
                mouse: mouse.pos.to_array(),
            };

            self.queue.write_buffer(
                &self.ui_renderer.pipelines.uniform_buffer,
                0,
                bytemuck::bytes_of(&screen_uniform),
            );

            let size = (self.config.width as f32, self.config.height as f32);
            self.ui_renderer
                .render(&mut pass, ui_loader, &self.queue, time, size, mouse);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
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

        self.pipelines.resize(self.msaa_samples);

        // Recreate pipelines with new sample count
        self.pipelines.msaa_samples = self.msaa_samples;
        self.pipelines.recreate_pipelines();

        self.ui_renderer.pipelines.msaa_samples = self.msaa_samples;
        self.ui_renderer.pipelines.rebuild_pipelines();
    }

    fn reload_all_shaders(&mut self) -> anyhow::Result<()> {
        self.pipelines.reload_shaders()?;
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

fn pick_fail_safe_present_mode(
    surface: &wgpu::Surface,
    adapter: &wgpu::Adapter,
    user_mode: wgpu::PresentMode,
) -> wgpu::PresentMode {
    let caps = surface.get_capabilities(adapter);

    // Fallback priority. Mailbox first, then Fifo (always supported), then Immediate.
    let fallback_chain = [
        user_mode,
        wgpu::PresentMode::Mailbox,
        wgpu::PresentMode::Fifo, // required by spec
        wgpu::PresentMode::Immediate,
    ];

    for mode in fallback_chain {
        if caps.present_modes.contains(&mode) {
            return mode;
        }
    }

    // Absolute guarantee
    wgpu::PresentMode::Fifo
}
