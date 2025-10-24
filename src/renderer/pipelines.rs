use crate::Uniforms;
use crate::vertex::{LineVtx, Vertex};
use util::DeviceExt;
use wgpu::*;

pub struct Pipelines {
    pub device: Device,
    pub(crate) gizmo_vbuf: Buffer,
    pub(crate) uniform_buffer: Buffer,
    pub(crate) uniform_bind_group: BindGroup,
    pub(crate) pipeline: RenderPipeline,
    pub(crate) gizmo_pipeline: RenderPipeline,
    pub shader: ShaderModule,
    pipeline_layout: PipelineLayout,
    pub(crate) msaa_samples: u32,
    line_shader: ShaderModule,
    format: TextureFormat,
}

impl Pipelines {
    pub fn new(device: &Device, format: TextureFormat, msaa_samples: u32) -> Self {
        let shader = device.create_shader_module(include_wgsl!("shaders/ground.wgsl"));

        let uniforms = Uniforms::new();

        let uniform_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Uniform Bind Group Layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let uniform_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let gizmo_vbuf = device.create_buffer(&BufferDescriptor {
            label: Some("Gizmo VB"),
            size: (size_of::<LineVtx>() * 6) as u64, // 3 axes = 6 vertices
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let line_shader = device.create_shader_module(include_wgsl!("shaders/lines.wgsl"));

        let gizmo_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Gizmo Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &line_shader,
                entry_point: Some("vs_main"),
                buffers: &[LineVtx::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &line_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });


        Self {
            shader,
            gizmo_vbuf,
            uniform_buffer,
            uniform_bind_group,
            pipeline,
            gizmo_pipeline,
            device: device.clone(),
            pipeline_layout,
            msaa_samples,
            format,
            line_shader,
        }
    }

    pub(crate) fn recreate_pipelines(&mut self) {
        // Rebuild any pipelines that depend on sample count.
        self.pipeline = self
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("Main Pipeline"),
                layout: Some(&self.pipeline_layout),
                vertex: VertexState {
                    module: &self.shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(FragmentState {
                    module: &self.shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(ColorTargetState {
                        format: self.format,
                        blend: Some(BlendState::REPLACE),
                        write_mask:
                        ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState {
                    count: self.msaa_samples,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });

        self.gizmo_pipeline = self
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("Gizmo Pipeline"),
                layout: Some(&self.pipeline_layout),
                vertex: VertexState {
                    module: &self.line_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[LineVtx::layout()],
                    compilation_options: Default::default(),
                },
                fragment: Some(FragmentState {
                    module: &self.line_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(ColorTargetState {
                        format: self.format,
                        blend: Some(BlendState::REPLACE),
                        write_mask: ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::LineList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: MultisampleState {
                    count: self.msaa_samples,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });
    }
}
