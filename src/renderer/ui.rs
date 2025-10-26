use crate::data::SharedData;
use crate::renderer::ui_editor::Vertex;
use crate::vertex::UiVertex;
use util::DeviceExt;
use wgpu::*;
use winit::dpi::PhysicalSize;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScreenUniform {
    pub size: [f32; 2],
    pub time: f32,
    pub enable_dither: u32, // use 0 = off, 1 = on
}

pub struct UiRenderer {
    pub vertex_buffer: Buffer,
    pub uniform_bind_group: BindGroup,
    pub num_vertices: u32,
    circle_pipeline: RenderPipeline,
    polygon_pipeline: RenderPipeline,
    pub circle_bind_group: BindGroup,
    pub circles: Vec<CircleParams>,
    pub quad_buffer: Buffer,
    glow_pipeline: RenderPipeline,
    pub(crate) uniform_buffer: Buffer,
    pub(crate) screen_uniform: ScreenUniform,
    device: Device,
    circle_layout: BindGroupLayout,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CircleParams {
    pub center_radius_border: [f32; 4], // cx, cy, radius, border
    pub fill_color: [f32; 4],
    pub border_color: [f32; 4],
    pub glow_color: [f32; 4],
    pub glow_misc: [f32; 4], // glow_size, glow_pulse_speed, glow_pulse_intensity, 0
}

impl Default for CircleParams {
    fn default() -> Self {
        Self {
            center_radius_border: [0.0, 0.0, 0.0, 0.0],
            fill_color: [0.0; 4],
            border_color: [0.0; 4],
            glow_color: [0.0; 4],
            glow_misc: [0.0; 4],
        }
    }
}

impl UiRenderer {
    pub fn new(
        device: &Device,
        format: TextureFormat,
        size: PhysicalSize<u32>,
        data: SharedData,
    ) -> Self {
        let screen_uniform = ScreenUniform {
            size: [size.width as f32, size.height as f32],
            time: 0.0,
            enable_dither: 1,
        };

        let screen_data = bytemuck::bytes_of(&screen_uniform);

        let uniform_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("UI Uniforms"),
            contents: screen_data,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("UI Bind Layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let circle_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Circle Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT, // <-- was FRAGMENT
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // println!("CircleParams size: {}", std::mem::size_of::<CircleParams>());
        let circles = vec![CircleParams::default()];

        let quad_vertices = [
            UiVertex {
                pos: [-1.0, -1.0],
                color: [1.0; 4],
            },
            UiVertex {
                pos: [1.0, -1.0],
                color: [1.0; 4],
            },
            UiVertex {
                pos: [-1.0, 1.0],
                color: [1.0; 4],
            },
            UiVertex {
                pos: [1.0, 1.0],
                color: [1.0; 4],
            },
        ];

        let quad_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("UI Quad VB"),
            contents: bytemuck::cast_slice(&quad_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let circle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Circle Storage Buffer"),
            contents: bytemuck::cast_slice(&circles), // now 80B per element
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let circle_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Circle Bind Group"),
            layout: &circle_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: circle_buffer.as_entire_binding(),
            }],
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("UI Bind Group"),
            layout: &layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        //let ui_shader = device.create_shader_module(include_wgsl!("shaders/ui.wgsl"));
        let polygon_shader = device.create_shader_module(include_wgsl!("shaders/ui_polygon.wgsl"));
        let circle_shader = device.create_shader_module(include_wgsl!("shaders/ui_circle.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("UI Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let polygon_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("UI Polygon Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &polygon_shader,
                entry_point: Some("vs_main"),
                buffers: &[UiVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &polygon_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let circle_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Circle Pipeline Layout"),
                bind_group_layouts: &[&layout, &circle_layout],
                push_constant_ranges: &[],
            });

        let circle_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("UI Circle Pipeline"),
            layout: Some(&circle_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &circle_shader,
                entry_point: Some("vs_main"),
                buffers: &[UiVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &circle_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("UI VB"),
            size: 1024 * 1024, // 1MB buffer
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let glow_shader = device.create_shader_module(include_wgsl!("shaders/ui_circle_glow.wgsl"));

        let glow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("UI Glow Pipeline Layout"),
            bind_group_layouts: &[&layout, &circle_layout],
            push_constant_ranges: &[],
        });

        let glow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UI Glow Pipeline"),
            layout: Some(&glow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &glow_shader,
                entry_point: Some("vs_main"),
                buffers: &[UiVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &glow_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    // ✨ additive blend for halo
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            vertex_buffer,
            uniform_bind_group: bind_group,
            num_vertices: 0,
            circle_pipeline,
            polygon_pipeline,
            circle_bind_group,
            circles,
            quad_buffer,
            glow_pipeline,
            uniform_buffer,
            screen_uniform,
            device: device.clone(),
            circle_layout,
        }
    }

    pub fn _draw_rects(
        &mut self,
        queue: &Queue,
        rects: &[(f32, f32, f32, f32, [f32; 4])], // x, y, w, h, color
    ) {
        let mut vertices = Vec::new();
        for &(x, y, w, h, color) in rects {
            vertices.extend_from_slice(&[
                UiVertex { pos: [x, y], color },
                UiVertex {
                    pos: [x + w, y],
                    color,
                },
                UiVertex {
                    pos: [x, y + h],
                    color,
                },
                UiVertex {
                    pos: [x + w, y],
                    color,
                },
                UiVertex {
                    pos: [x + w, y + h],
                    color,
                },
                UiVertex {
                    pos: [x, y + h],
                    color,
                },
            ]);
        }

        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        self.num_vertices = vertices.len() as u32;
    }

    pub fn render<'a>(&'a self, pass: &mut RenderPass<'a>) {
        // Glow first (additive)
        let circle_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Circle Storage Buffer"),
                contents: bytemuck::cast_slice(&self.circles), // now 80B per element
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let circle_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Circle Bind Group"),
            layout: &self.circle_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: circle_buffer.as_entire_binding(),
            }],
        });

        pass.set_pipeline(&self.glow_pipeline);
        pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        pass.set_bind_group(1, &circle_bind_group, &[]);
        pass.set_vertex_buffer(0, self.quad_buffer.slice(..));
        pass.draw(0..4, 0..self.circles.len() as u32);

        pass.set_pipeline(&self.circle_pipeline);
        pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        pass.set_bind_group(1, &circle_bind_group, &[]);
        pass.set_vertex_buffer(0, self.quad_buffer.slice(..));
        pass.draw(0..4, 0..self.circles.len() as u32);

        pass.set_pipeline(&self.polygon_pipeline);
        pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..self.num_vertices, 0..1);
    }

    pub fn draw_custom(&mut self, queue: &Queue, vertices: &Vec<Vertex>) {
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(vertices));
        self.num_vertices = vertices.len() as u32;
    }

    pub fn generate_button_vertices(&self, btn: &UiButtonCircle) -> Vec<UiVertex> {
        let mut vertices = Vec::new();
        let cx = btn.x;
        let cy = btn.y;
        let r = btn.radius;

        // --- Background circle ---
        const SEGMENTS: usize = 32;
        for i in 0..SEGMENTS {
            let a0 = (i as f32 / SEGMENTS as f32) * std::f32::consts::TAU;
            let a1 = ((i + 1) as f32 / SEGMENTS as f32) * std::f32::consts::TAU;
            let p0 = [cx + r * a0.cos(), cy + r * a0.sin()];
            let p1 = [cx + r * a1.cos(), cy + r * a1.sin()];
            vertices.extend_from_slice(&[
                UiVertex {
                    pos: [cx, cy],
                    color: btn.color,
                },
                UiVertex {
                    pos: p0,
                    color: btn.color,
                },
                UiVertex {
                    pos: p1,
                    color: btn.color,
                },
            ]);
        }

        // --- Icon shape ---
        if btn.active {
            // Pause: two rectangles
            let bar_w = r * 0.25;
            let bar_h = r * 0.8;
            let gap = r * 0.10;

            let left_x = cx - gap - bar_w;
            let right_x = cx + gap;
            let top_y = cy - bar_h / 2.0;
            let bot_y = cy + bar_h / 2.0;

            let icon_color = [1.0, 1.0, 1.0, 0.9];

            let add_rect = |x: f32, vertices: &mut Vec<UiVertex>| {
                vertices.extend_from_slice(&[
                    UiVertex {
                        pos: [x, top_y],
                        color: icon_color,
                    },
                    UiVertex {
                        pos: [x + bar_w, top_y],
                        color: icon_color,
                    },
                    UiVertex {
                        pos: [x, bot_y],
                        color: icon_color,
                    },
                    UiVertex {
                        pos: [x + bar_w, top_y],
                        color: icon_color,
                    },
                    UiVertex {
                        pos: [x + bar_w, bot_y],
                        color: icon_color,
                    },
                    UiVertex {
                        pos: [x, bot_y],
                        color: icon_color,
                    },
                ]);
            };

            add_rect(left_x, &mut vertices);
            add_rect(right_x, &mut vertices);
        } else {
            // Start: simple triangle
            let t_w = r * 0.7;
            let t_h = r * 0.8;
            let icon_color = [1.0, 1.0, 1.0, 0.9];

            let left = cx - t_w * 0.4;
            let right = left + t_w;
            let top = cy - t_h / 2.0;
            let bottom = cy + t_h / 2.0;

            vertices.extend_from_slice(&[
                UiVertex {
                    pos: [left, top],
                    color: icon_color,
                },
                UiVertex {
                    pos: [right, cy],
                    color: icon_color,
                },
                UiVertex {
                    pos: [left, bottom],
                    color: icon_color,
                },
            ]);
        }
        vertices
    }
}

#[derive(Debug)]
pub struct UiButtonCircle {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub color: [f32; 4],
    pub active: bool, // true = running, false = paused
}
