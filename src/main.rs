use wgpu::{VertexAttribute, VertexStepMode};
use std::collections::HashSet;
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{WindowEvent, ElementState},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
    keyboard::Key
};
use std::time::Instant;
struct FrameTimer { last: Instant }
impl FrameTimer {
    fn new() -> Self { Self { last: Instant::now() } }
    fn dt(&mut self) -> f32 {
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32();
        self.last = now;
        dt
    }
}


struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    msaa_samples: u32,
    msaa_texture: wgpu::Texture,
    msaa_view: wgpu::TextureView,
    size: winit::dpi::PhysicalSize<u32>,
    _window: Arc<Window>,

    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    //uniform_bind_group_layout: wgpu::BindGroupLayout,
    camera: Camera,
    input: InputState,
    mouse: MouseState,
    timer: FrameTimer,
    velocity: Vec3,
    gizmo_pipeline: wgpu::RenderPipeline,
    gizmo_vbuf: wgpu::Buffer,
    zoom_vel: f32,
    target_yaw: f32,
    target_pitch: f32,
    orbit_smoothness: f32,
    yaw_velocity: f32,
    pitch_velocity: f32,
    orbit_damping_release: f32,
    zoom_damping: f32,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let camera = Camera::new();

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
        let can_render = caps.allowed_usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT);
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

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: features,
                required_limits: limits,
            },
            None,
        ).await.expect("Device creation failed");

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
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Vertex {
            position: [f32; 3],
            color: [f32; 3],
        }

        impl Vertex {
            const ATTRIBS: [VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x3, // now 3 floats for position
        1 => Float32x3
    ];

            fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
                wgpu::VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &Self::ATTRIBS,
                }
            }
        }


        let vertices = [
            // bottom left triangle
            Vertex { position: [-1.0, 0.0, -1.0], color: [0.2, 0.9, 0.4] },
            Vertex { position: [ 1.0, 0.0, -1.0], color: [0.8, 0.0, 0.8] },
            Vertex { position: [-1.0, 0.0,  1.0], color: [0.1, 0.0, 0.2] },
            // top right triangle
            Vertex { position: [ 1.0, 0.0, -1.0], color: [0.8, 0.0, 0.8] },
            Vertex { position: [ 1.0, 0.0,  1.0], color: [0.2, 0.9, 0.2] },
            Vertex { position: [-1.0, 0.0,  1.0], color: [0.1, 0.0, 0.2] },
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/ground.wgsl"));

        let uniforms = Uniforms::new();

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Uniform Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let line_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/lines.wgsl"));
        let gizmo_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gizmo Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });
        let gizmo_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Gizmo Pipeline"),
            layout: Some(&gizmo_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &line_shader,
                entry_point: "vs_main",
                buffers: &[LineVtx::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &line_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None, // draw atop everything; if you have a depth buffer, you can disable write/test here instead
            multisample: wgpu::MultisampleState {
                count: msaa_samples,    // 👈 match the MSAA sample count (4)
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        let gizmo_vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gizmo VB"),
            size: (size_of::<LineVtx>() * 6) as u64, // 3 axes = 6 vertices
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });


        let num_vertices = vertices.len() as u32;

        Self {
            surface,
            device,
            queue,
            config,
            msaa_samples,
            msaa_texture,
            msaa_view,
            size,
            _window: window,
            pipeline,
            vertex_buffer,
            num_vertices,
            uniform_buffer,
            uniform_bind_group,
            //uniform_bind_group_layout,
            input: InputState::new(),
            mouse: MouseState { last_pos: None, dragging: false },
            timer: FrameTimer::new(),
            velocity: Vec3::ZERO,
            gizmo_pipeline,
            gizmo_vbuf,
            zoom_vel: 0.0,
            target_yaw: camera.yaw,
            target_pitch: camera.pitch,
            orbit_smoothness: 0.25,  // 0.0 = instant, 1.0 = very slow follow
            yaw_velocity: 0.0,
            pitch_velocity: 0.0,
            orbit_damping_release: 4.0, // how fast rotation stops after release
            zoom_damping: 12.0,
            camera,
        }
    }


    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 { return; }
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
        self.msaa_view = self.msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());
    }


    fn render(&mut self) {
        let dt = self.timer.dt();
        self.update_camera(dt);


        // update camera uniforms
        let aspect = self.config.width as f32 / self.config.height as f32;
        let new_uniforms = Uniforms {
            view_proj: self.camera.view_proj(aspect),
        };
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&new_uniforms));


        // now the existing render code below...
        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(_) => {
                self.surface.configure(&self.device, &self.config);
                self.surface.get_current_texture().unwrap()
            }
        };
        let surface_view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // Build gizmo vertices centered at target
        let t = self.camera.target;
        let s = self.camera.radius * 0.2; // gizmo size scales with zoom
        let axes = [
            // X axis (red)
            LineVtx { pos: [t.x, t.y, t.z], color: [1.0, 0.2, 0.2] },
            LineVtx { pos: [t.x + s, t.y, t.z], color: [1.0, 0.2, 0.2] },
            // Y axis (green)
            LineVtx { pos: [t.x, t.y, t.z], color: [0.2, 1.0, 0.2] },
            LineVtx { pos: [t.x, t.y + s, t.z], color: [0.2, 1.0, 0.2] },
            // Z axis (blue)
            LineVtx { pos: [t.x, t.y, t.z], color: [0.2, 0.6, 1.0] },
            LineVtx { pos: [t.x, t.y, t.z + s], color: [0.2, 0.6, 1.0] },
        ];
        self.queue.write_buffer(&self.gizmo_vbuf, 0, bytemuck::cast_slice(&axes));

        let color_attachment = if self.msaa_samples > 1 {
            // Render to MSAA texture and resolve into the swapchain
            wgpu::RenderPassColorAttachment {
                view: &self.msaa_view,
                resolve_target: Some(&surface_view),
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
            wgpu::RenderPassColorAttachment {
                view: &surface_view,
                resolve_target: None,
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
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.draw(0..self.num_vertices, 0..1);
            pass.set_pipeline(&self.gizmo_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, self.gizmo_vbuf.slice(..));
            pass.draw(0..6, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }

    fn update_camera(&mut self, dt: f32) {
        // Direction from camera to target (what you’re looking at):
        let eye = self.camera.position();
        let mut fwd3d = self.camera.target - eye;
        if fwd3d.length_squared() > 0.0 { fwd3d = fwd3d.normalize(); }

        // Flatten to XZ for planar WASD
        let mut forward = Vec3::new(fwd3d.x, 0.0, fwd3d.z);
        if forward.length_squared() > 0.0 { forward = forward.normalize(); }

        // Right = forward × up (RH system)
        let right = forward.cross(Vec3::Y).normalize();
        let up = Vec3::Y;

        // Desired move direction
        let mut wish = Vec3::ZERO;
        if self.input.pressed.contains("w") { wish += forward; }
        if self.input.pressed.contains("s") { wish -= forward; }
        if self.input.pressed.contains("a") { wish -= right;   }
        if self.input.pressed.contains("d") { wish += right;   }
        if self.input.pressed.contains("q") { wish += up;      }
        if self.input.pressed.contains("e") { wish -= up;      }

        // --- Adaptive movement speed ---
        let base_speed = 8.0;        // base units/sec
        let decay_rate = 6.0;        // damping when no input
        let dist = self.camera.radius;
        let speed_factor = (dist / 10.0).clamp(0.1, 10.0); // scales with zoom

        if wish.length_squared() > 0.0 {
            wish = wish.normalize();
            self.velocity = wish * base_speed * speed_factor;
        } else {
            // exponential decay towards zero
            let k = (1.0 - decay_rate * dt).max(0.0);
            self.velocity *= k;
            if self.velocity.length_squared() < 1e-5 {
                self.velocity = Vec3::ZERO;
            }
        }

        // Apply velocity to target (pans the scene)
        self.camera.target += self.velocity * dt;


        // smooth zoom update
        if self.zoom_vel.abs() > 0.0001 {
            self.camera.radius += self.zoom_vel * dt;
            self.zoom_vel *= (1.0 - self.zoom_damping * dt).max(0.0);
            self.camera.radius = self.camera.radius.clamp(1.0, 10_000.0);
        } else {
            self.zoom_vel = 0.0; }

        // --- Smooth orbit follow ---
        let t = 1.0 - (-self.orbit_smoothness * 60.0 * dt).exp();
        self.camera.yaw   += (self.target_yaw   - self.camera.yaw)   * t;
        self.camera.pitch += (self.target_pitch - self.camera.pitch) * t;

        // --- Gentle decel after release ---
        if !self.mouse.dragging {
            self.target_yaw   += self.yaw_velocity;
            self.target_pitch += self.pitch_velocity;
            self.yaw_velocity   *= (1.0 - self.orbit_damping_release * dt).max(0.0);
            self.pitch_velocity *= (1.0 - self.orbit_damping_release * dt).max(0.0);
        }

        // --- Clamp and pan ---
        self.camera.pitch = self.camera.pitch.clamp(10.0f32.to_radians(), 89.0f32.to_radians());
        self.camera.target += self.velocity * dt;

    }



}

#[derive(Default)]
struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create the window when resumed
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Rusty City")
                        .with_inner_size(winit::dpi::PhysicalSize::new(2560, 1400)),
                )
                .unwrap(),
        );

        let state = pollster::block_on(State::new(window.clone()));

        self.window = Some(window);
        self.state = Some(state);

    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let state = match self.state.as_mut() {
            Some(s) => s,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size),
            WindowEvent::RedrawRequested => state.render(),
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let Key::Character(ch) = &event.logical_key {
                    let key = ch.to_lowercase();
                    if ["w", "a", "s", "d", "q", "e"].contains(&key.as_str()) {
                        state.input.set_key(&key, pressed);
                    }
                }
            }

            WindowEvent::MouseInput { state: mouse_state, button, .. } => {
                if button == winit::event::MouseButton::Middle {
                    state.mouse.dragging = mouse_state == ElementState::Pressed;
                    state.mouse.last_pos = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if state.mouse.dragging {
                    if let Some((lx, ly)) = state.mouse.last_pos {
                        let dx = position.x - lx;
                        let dy = position.y - ly;
                        let pitch_sensitivity = 0.002;
                        let yaw_sensitivity = 0.0016;


                        // Apply smoothing directly to target angles
                        state.target_yaw   += dx as f32 * yaw_sensitivity;
                        state.target_pitch += dy as f32 * pitch_sensitivity;
                        state.target_pitch = state.target_pitch.clamp(10.0f32.to_radians(), 89.0f32.to_radians());

                        // record last angular velocity for soft stop
                        state.yaw_velocity = dx as f32 * yaw_sensitivity;
                        state.pitch_velocity = dy as f32 * pitch_sensitivity;
                    }
                    state.mouse.last_pos = Some((position.x, position.y));
                }



            }
            WindowEvent::MouseWheel { delta, .. } => {
                // zoom in/out
                match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => {
                        // scroll direction affects velocity, proportional to distance
                        state.zoom_vel -= y * 1.0 * state.camera.radius;
                    }
                    winit::event::MouseScrollDelta::PixelDelta(pos) => {
                        state.zoom_vel -= pos.y as f32 * 0.04 * state.camera.radius;
                    }
                }
            }

            _ => (),
        }
    }


    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Request redraw every frame
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

struct InputState {
    pressed: HashSet<String>, // store "w","a","s","d","q","e"
}

impl InputState {
    fn new() -> Self { Self { pressed: HashSet::new() } }
    fn set_key(&mut self, key: &str, down: bool) {
        if down { self.pressed.insert(key.to_string()); }
        else     { self.pressed.remove(key); }
    }
}



struct MouseState {
    last_pos: Option<(f64, f64)>,
    dragging: bool,
}


struct Camera {
    target: Vec3,
    radius: f32,
    yaw: f32,    // radians, rotates around +Y
    pitch: f32,  // radians, elevation, clamp to (-PI/2, PI/2)
}

impl Camera {
    fn new() -> Self {
        Self {
            target: Vec3::ZERO,
            radius: 5.0,
            yaw:  -45f32.to_radians(),
            pitch: 20f32.to_radians(),
        }
    }

    fn position(&self) -> Vec3 {
        // Spherical coordinates, RH system
        // x = r cos(pitch) cos(yaw)
        // y = r sin(pitch)
        // z = r cos(pitch) sin(yaw)
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        let offset = Vec3::new(self.radius * cp * cy,
                               self.radius * sp,
                               self.radius * cp * sy);
        self.target + offset
    }

    fn view_proj(&self, aspect: f32) -> [[f32; 4]; 4] {
        let eye = self.position();
        let view = Mat4::look_at_rh(eye, self.target, Vec3::Y);
        let proj = Mat4::perspective_rh_gl(45f32.to_radians(), aspect, 0.1, 1_000.0);
        (proj * view).to_cols_array_2d()
    }
}



#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
}

impl Uniforms {
    fn new() -> Self {
        // Eye position: x, y, z
        let eye = Vec3::new(5.0, 15.0, 0.0);     // elevated above origin
        let target = Vec3::new(0.0, 0.0, 0.0);  // looking at center
        let up = Vec3::Y;                       // y-axis = up

        let view = Mat4::look_at_rh(eye, target, up);
        let proj = Mat4::perspective_rh_gl(45f32.to_radians(), 16.0/9.0, 0.1, 100.0);
        Self { view_proj: (proj * view).to_cols_array_2d() }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LineVtx {
    pos: [f32; 3],
    color: [f32; 3],
}

impl LineVtx {
    fn layout<'a>() -> wgpu::VertexBufferLayout<'a> {

        const ATTRS: &[VertexAttribute] = &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
        wgpu::VertexBufferLayout {
            array_stride: size_of::<LineVtx>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: ATTRS,
        }
    }
}


fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll); // <- continuous
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("Failed to run event_loop application");
}
