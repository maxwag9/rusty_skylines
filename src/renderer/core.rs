use crate::camera::Camera;
pub use crate::renderer::pipelines::Pipelines;
use crate::ui::UiSystem;
use crate::vertex::{LineVtx, Vertex};
use crate::{FrameTimer, Uniforms};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::*;
use winit::window::Window;

pub struct RenderCore {
    pub surface: Surface<'static>,
    pub device: Device,
    pub queue: Queue,
    pub config: SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,

    // --- new fields ---
    pub msaa_texture: Texture,
    pub msaa_view: TextureView,
    pub msaa_samples: u32,

    pub vertex_buffer: Buffer,
    pub num_vertices: u32,
    pub pipelines: Pipelines,
    pub timer: FrameTimer,
}

impl RenderCore {
    pub async fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::default();
        let size = window.inner_size();
        let surface = instance
            .create_surface(window.clone())
            .expect("Surface creation failed");

        // Pick an adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable GPU adapters found");

        println!("Backend: {:?}", adapter.get_info().backend);

        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo, // vsync
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let mut msaa_samples = 4;
        let caps = adapter.get_texture_format_features(config.format);
        let supported = caps.flags.intersects(
            wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X8
                | wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X4
                | wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X2,
        );
        let can_render = caps
            .allowed_usages
            .contains(wgpu::TextureUsages::RENDER_ATTACHMENT);
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
        let features = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        let limits = wgpu::Limits::default();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: features,
                required_limits: limits,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Device creation failed");

        // Configure surface
        surface.configure(&device, &config);

        let msaa_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MSAA Color Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: msaa_samples,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let msaa_view = msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // === Vertex data ===

        let vertices = [
            // bottom left triangle
            Vertex {
                position: [-1.0, 0.0, -1.0],
                color: [0.2, 0.9, 0.4],
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

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: BufferUsages::VERTEX,
        });

        let num_vertices = vertices.len() as u32;

        let pipelines = Pipelines::new(&device, config.format, msaa_samples);

        Self {
            surface,
            device,
            queue,
            config,
            pipelines,
            size,
            msaa_texture,
            msaa_view,
            msaa_samples,
            vertex_buffer,
            num_vertices,
            timer: FrameTimer::new(),
        }
    }

    pub(crate) fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);

        self.msaa_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MSAA Color Texture"),
            size: wgpu::Extent3d {
                width: new_size.width,
                height: new_size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: self.msaa_samples,
            dimension: wgpu::TextureDimension::D2,
            format: self.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.msaa_view = self
            .msaa_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
    }

    pub(crate) fn render(&mut self, camera: &Camera, window: &Window, ui: &mut UiSystem) {
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
        let surface_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
        self.queue
            .write_buffer(&self.pipelines.gizmo_vbuf, 0, bytemuck::cast_slice(&axes));

        let color_attachment = if self.msaa_samples > 1 {
            // Render to MSAA texture and resolve into the swapchain
            wgpu::RenderPassColorAttachment {
                view: &self.msaa_view,
                resolve_target: Some(&surface_view),
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Discard,
                },
            }
        } else {
            // Render directly to the swapchain (no MSAA)
            RenderPassColorAttachment {
                view: &surface_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            }
        };

        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[Some(color_attachment)],
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
        }
        ui.render(
            &self.device,
            &self.queue,
            window,
            &mut encoder,
            &surface_view,
        );
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
            self.msaa_texture = self.device.create_texture(&wgpu::TextureDescriptor {
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
        self.pipelines.recreate_pipelines();
    }
}
