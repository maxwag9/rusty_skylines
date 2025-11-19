use crate::components::camera::Camera;
pub use crate::renderer::pipelines::Pipelines;
use crate::renderer::ui::UiRenderer;
use crate::renderer::ui_editor::UiButtonLoader;
use crate::resources::{MouseState, TimeSystem, Uniforms};
use crate::vertex::{LineVtx, Vertex};
use std::sync::Arc;
use util::DeviceExt;
use wgpu::*;
use winit::dpi::PhysicalSize;
use winit::window::Window;

// top-level (once)
pub const FONT_TTF: &[u8] = include_bytes!("ui_data/ttf/JetBrainsMono-Regular.ttf");

pub struct RenderCore {
    pub surface: Surface<'static>,
    pub device: Device,
    pub queue: Queue,
    pub config: SurfaceConfiguration,

    // --- new fields ---
    pub msaa_texture: Texture,
    pub msaa_view: TextureView,
    pub msaa_samples: u32,

    pub vertex_buffer: Buffer,
    pub num_vertices: u32,
    pub pipelines: Pipelines,
    ui_renderer: UiRenderer,
    size: PhysicalSize<u32>,
}

impl RenderCore {
    pub fn new(window: Arc<Window>) -> Self {
        use wgpu::*;
        // --- Create instance and surface ---
        let instance = wgpu::Instance::new(&InstanceDescriptor {
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
            .find(|m| *m == wgpu::CompositeAlphaMode::PostMultiplied)
            .unwrap_or(wgpu::CompositeAlphaMode::Opaque);

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: PresentMode::Mailbox,
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

        let msaa_texture = device.create_texture(&TextureDescriptor {
            label: Some("MSAA Color Texture"),
            size: Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: msaa_samples,
            dimension: TextureDimension::D2,
            format: config.format,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let msaa_view = msaa_texture.create_view(&TextureViewDescriptor::default());

        // === Vertex data ===

        let vertices = [
            // bottom left triangle
            Vertex {
                position: [-1.0, 0.0, -1.0],
                color: [1.0, 1.0, 1.0],
            },
            Vertex {
                position: [1.0, 0.0, -1.0],
                color: [0.8, 0.0, 0.8],
            },
            Vertex {
                position: [-1.0, 0.0, 1.0],
                color: [0.1, 0.0, 0.2],
            },
            // top right triangle
            Vertex {
                position: [1.0, 0.0, -1.0],
                color: [0.8, 0.0, 0.8],
            },
            Vertex {
                position: [1.0, 0.0, 1.0],
                color: [0.2, 0.9, 0.2],
            },
            Vertex {
                position: [-1.0, 0.0, 1.0],
                color: [0.1, 0.0, 0.2],
            },
        ];

        let vertex_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: BufferUsages::VERTEX,
        });

        let num_vertices = vertices.len() as u32;

        let pipelines = Pipelines::new(&device, config.format, msaa_samples);
        let mut ui_renderer = UiRenderer::new(&device, config.format, size, msaa_samples);
        let _ = ui_renderer.build_text_atlas(&device, &queue, &FONT_TTF, &[14, 18, 24], 1024, 1024);

        Self {
            surface,
            device,
            queue,
            config,
            pipelines,
            msaa_texture,
            msaa_view,
            msaa_samples,
            vertex_buffer,
            num_vertices,
            ui_renderer,
            size,
        }
    }

    pub(crate) fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);

        self.msaa_texture = self.device.create_texture(&TextureDescriptor {
            label: Some("MSAA Color Texture"),
            size: Extent3d {
                width: new_size.width,
                height: new_size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: self.msaa_samples,
            dimension: TextureDimension::D2,
            format: self.config.format,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.msaa_view = self
            .msaa_texture
            .create_view(&TextureViewDescriptor::default());
    }

    pub(crate) fn render(
        &mut self,
        camera: &Camera,
        ui_loader: &mut UiButtonLoader,
        time: &TimeSystem,
        mouse: &MouseState,
    ) {
        // update camera uniforms
        let aspect = self.config.width as f32 / self.config.height as f32;
        let new_uniforms = Uniforms {
            view_proj: camera.view_proj(aspect),
        };
        self.queue.write_buffer(
            &self.pipelines.uniform_buffer,
            0,
            bytemuck::bytes_of(&new_uniforms),
        );

        // now the existing render code below...
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

        // Build gizmo vertices centered at target
        let t = camera.target;
        let s = camera.radius * 0.2; // gizmo size scales with zoom
        let axes = [
            // X axis (red)
            LineVtx {
                pos: [t.x, t.y, t.z],
                color: [1.0, 0.2, 0.2],
            },
            LineVtx {
                pos: [t.x + s, t.y, t.z],
                color: [1.0, 0.2, 0.2],
            },
            // Y axis (green)
            LineVtx {
                pos: [t.x, t.y, t.z],
                color: [0.2, 1.0, 0.2],
            },
            LineVtx {
                pos: [t.x, t.y + s, t.z],
                color: [0.2, 1.0, 0.2],
            },
            // Z axis (blue)
            LineVtx {
                pos: [t.x, t.y, t.z],
                color: [0.2, 0.6, 1.0],
            },
            LineVtx {
                pos: [t.x, t.y, t.z + s],
                color: [0.2, 0.6, 1.0],
            },
        ];
        // --- Update gizmo vertex buffer ---
        self.queue
            .write_buffer(&self.pipelines.gizmo_vbuf, 0, bytemuck::cast_slice(&axes));

        // --- Choose color attachment ---
        let color_attachment = Some(RenderPassColorAttachment {
            view: if self.msaa_samples > 1 {
                &self.msaa_view
            } else {
                &surface_view
            },
            resolve_target: if self.msaa_samples > 1 {
                Some(&surface_view)
            } else {
                None
            },
            depth_slice: None,
            ops: Operations {
                load: LoadOp::Clear(Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.3,
                    a: 1.0,
                }),
                store: StoreOp::Store,
            },
        });

        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[color_attachment],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipelines.pipeline);
            pass.set_bind_group(0, &self.pipelines.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.draw(0..self.num_vertices, 0..1);

            pass.set_pipeline(&self.pipelines.gizmo_pipeline);
            pass.set_bind_group(0, &self.pipelines.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, self.pipelines.gizmo_vbuf.slice(..));
            pass.draw(0..6, 0..1);

            // --- UI pass ---

            let elapsed = time.total_time;
            let enable_dither = 1;
            //if self.settings.dither_enabled { 1 } else { 0 };

            let screen_uniform = crate::renderer::ui::ScreenUniform {
                size: [self.size.width as f32, self.size.height as f32],
                time: elapsed,
                enable_dither,
                mouse: mouse.pos.to_array(),
            };

            self.queue.write_buffer(
                &self.ui_renderer.pipelines.uniform_buffer,
                0,
                bytemuck::bytes_of(&screen_uniform),
            );
            let size = (self.config.width as f32, self.config.height as f32);
            self.ui_renderer
                .render(&mut pass, ui_loader, &self.queue, &time, size, mouse);
        }

        // --- Submit and present ---
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

        // Recreate MSAA color texture if needed
        if self.msaa_samples > 1 {
            self.msaa_texture = self.device.create_texture(&TextureDescriptor {
                label: Some("MSAA Color Texture"),
                size: Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: self.msaa_samples,
                dimension: TextureDimension::D2,
                format: self.config.format,
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.msaa_view = self
                .msaa_texture
                .create_view(&TextureViewDescriptor::default());
        }

        // Recreate pipelines with new sample count
        self.pipelines.msaa_samples = self.msaa_samples;
        self.pipelines.recreate_pipelines();

        self.ui_renderer.pipelines.msaa_samples = self.msaa_samples;
        self.ui_renderer.pipelines.rebuild_pipelines();
    }
}
