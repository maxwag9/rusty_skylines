use crate::renderer::helper::{make_pipeline, make_uniform_layout};
use crate::renderer::ui::{ScreenUniform, TextAtlas};
use crate::vertex::{UiVertexPoly, UiVertexText};
use wgpu::util::DeviceExt;
use wgpu::*;
use winit::dpi::PhysicalSize;

pub struct UiPipelines {
    pub device: Device,
    pub(crate) uniform_buffer: Buffer,
    pub(crate) uniform_bind_group: BindGroup,

    pub(crate) msaa_samples: u32,

    format: TextureFormat,
    pub vertex_buffer: Buffer,
    pub circle_pipeline: RenderPipeline,
    pub outline_pipeline: RenderPipeline,
    pub polygon_pipeline: RenderPipeline,
    pub handle_pipeline: RenderPipeline,
    pub quad_buffer: Buffer,
    pub handle_quad_buffer: Buffer,
    pub glow_pipeline: RenderPipeline,
    pub handle_layout: BindGroupLayout,
    pub circle_layout: BindGroupLayout,
    pub text_atlas: Option<TextAtlas>,
    pub text_pipeline: RenderPipeline,
    pub text_layout: BindGroupLayout,
    pub text_bind_group: Option<BindGroup>,
    pub text_vertex_buffer: Buffer,
    pub outline_layout: BindGroupLayout,
    pub text_vertex_count: i32,
    pub num_vertices: i32,
    pub text_shader: ShaderModule,
    pub circle_shader: ShaderModule,
    pub outline_shader: ShaderModule,
    pub polygon_shader: ShaderModule,
    pub handle_shader: ShaderModule,
    pub glow_shader: ShaderModule,
    pub text_pipeline_layout: PipelineLayout,
    pub circle_pipeline_layout: PipelineLayout,
    pub glow_pipeline_layout: PipelineLayout,
    pub handle_pipeline_layout: PipelineLayout,
    pub polygon_pipeline_layout: PipelineLayout,
    pub outline_pipeline_layout: PipelineLayout,
    pub good_blend: Option<BlendState>,
    pub additive_blend: BlendState,
}

impl UiPipelines {
    pub fn new(
        device: &Device,
        format: TextureFormat,
        msaa_samples: u32,
        size: PhysicalSize<u32>,
    ) -> Self {
        let handle_quad_vertices = [
            UiVertexPoly {
                pos: [-3.0, -3.0],
                _pad: [1.0; 2],
                color: [1.0; 4],
                misc: [1.0; 4],
            },
            UiVertexPoly {
                pos: [3.0, -3.0],
                _pad: [1.0; 2],
                color: [1.0; 4],
                misc: [1.0; 4],
            },
            UiVertexPoly {
                pos: [-3.0, 3.0],
                _pad: [1.0; 2],
                color: [1.0; 4],
                misc: [1.0; 4],
            },
            UiVertexPoly {
                pos: [3.0, 3.0],
                _pad: [1.0; 2],
                color: [1.0; 4],
                misc: [1.0; 4],
            },
        ];
        let handle_quad_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("Handle Quad VB"),
            contents: bytemuck::cast_slice(&handle_quad_vertices),
            usage: BufferUsages::VERTEX,
        });

        let quad_vertices = [
            UiVertexPoly {
                pos: [-1.0, -1.0],
                _pad: [1.0; 2],
                color: [1.0; 4],
                misc: [1.0, 0.0, 0.0, 0.0],
            },
            UiVertexPoly {
                pos: [1.0, -1.0],
                _pad: [1.0; 2],
                color: [1.0; 4],
                misc: [1.0, 0.0, 0.0, 0.0],
            },
            UiVertexPoly {
                pos: [-1.0, 1.0],
                _pad: [1.0; 2],
                color: [1.0; 4],
                misc: [1.0, 0.0, 0.0, 0.0],
            },
            UiVertexPoly {
                pos: [1.0, 1.0],
                _pad: [1.0; 2],
                color: [1.0; 4],
                misc: [1.0, 0.0, 0.0, 0.0],
            },
        ];
        let quad_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("UI Quad VB"),
            contents: bytemuck::cast_slice(&quad_vertices),
            usage: BufferUsages::VERTEX,
        });
        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("UI VB"),
            size: 1024 * 1024, // 1MB buffer
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = make_uniform_layout(device, "UI Bind Layout");

        let screen_uniform = ScreenUniform {
            size: [size.width as f32, size.height as f32],
            time: 0.0,
            enable_dither: 1,
            mouse: [0.0, 0.0],
        };
        let screen_data = bytemuck::bytes_of(&screen_uniform);

        let uniform_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("UI Uniforms"),
            contents: screen_data,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let uniform_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("UI Bind Group"),
            layout: &layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let text_shader = device.create_shader_module(include_wgsl!("shaders/text.wgsl"));
        let text_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Text Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let text_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Text Pipeline Layout"),
            bind_group_layouts: &[&layout, &text_layout],
            push_constant_ranges: &[],
        });
        let text_pipeline = make_pipeline(
            device,
            "UI Text Pipeline",
            &text_pipeline_layout,
            &text_shader,
            "vs_main",
            "fs_main",
            &[UiVertexText::desc()],
            format,
            Some(BlendState::ALPHA_BLENDING),
            PrimitiveTopology::TriangleList,
            MultisampleState {
                count: msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        );

        let text_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UI Text VB"),
            size: 256 * 1024,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // println!("CircleParams size: {}", std::mem::size_of::<CircleParams>());
        let circle_shader = device.create_shader_module(include_wgsl!("shaders/ui_circle.wgsl"));
        let outline_shader =
            device.create_shader_module(include_wgsl!("shaders/ui_shape_outline.wgsl"));
        let handle_shader = device.create_shader_module(include_wgsl!("shaders/ui_handle.wgsl"));
        let circle_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Circle Layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let circle_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Circle Pipeline Layout"),
            bind_group_layouts: &[&layout, &circle_layout],
            push_constant_ranges: &[],
        });

        let circle_pipeline = make_pipeline(
            device,
            "UI Circle Pipeline",
            &circle_pipeline_layout,
            &circle_shader,
            "vs_main",
            "fs_main",
            &[UiVertexPoly::desc()],
            format,
            Some(BlendState::ALPHA_BLENDING),
            PrimitiveTopology::TriangleStrip,
            MultisampleState {
                count: msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        );

        let good_blend = Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
        });
        let outline_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Outline Layout"),
            entries: &[
                // binding 0 = ShapeParams SSBO
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1 = PolygonVertices SSBO
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let outline_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Outline Pipeline Layout"),
            bind_group_layouts: &[&layout, &outline_layout],
            push_constant_ranges: &[],
        });

        let outline_pipeline = make_pipeline(
            device,
            "UI Outline Pipeline",
            &outline_pipeline_layout,
            &outline_shader,
            "vs_main",
            "fs_main",
            &[UiVertexPoly::desc()],
            format,
            good_blend,
            PrimitiveTopology::TriangleStrip,
            MultisampleState {
                count: msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        );
        let handle_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Handle Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let handle_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Handle Pipeline Layout"),
                bind_group_layouts: &[&layout, &handle_layout],
                push_constant_ranges: &[],
            });

        let handle_pipeline = make_pipeline(
            device,
            "UI Handle Pipeline",
            &handle_pipeline_layout,
            &handle_shader,
            "vs_main",
            "fs_main",
            &[UiVertexPoly::desc()],
            format,
            good_blend,
            PrimitiveTopology::TriangleStrip,
            MultisampleState {
                count: msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        );
        let polygon_shader = device.create_shader_module(include_wgsl!("shaders/ui_polygon.wgsl"));
        let polygon_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("UI Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let polygon_pipeline = make_pipeline(
            device,
            "UI Polygon Pipeline",
            &polygon_pipeline_layout,
            &polygon_shader,
            "vs_main",
            "fs_main",
            &[UiVertexPoly::desc()],
            format,
            Some(BlendState::ALPHA_BLENDING),
            PrimitiveTopology::TriangleStrip,
            MultisampleState {
                count: msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        );

        let glow_shader = device.create_shader_module(include_wgsl!("shaders/ui_circle_glow.wgsl"));
        let glow_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("UI Glow Pipeline Layout"),
            bind_group_layouts: &[&layout, &circle_layout],
            push_constant_ranges: &[],
        });
        let additive_blend = BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
        };

        let glow_pipeline = make_pipeline(
            device,
            "UI Glow Pipeline",
            &glow_pipeline_layout,
            &glow_shader,
            "vs_main",
            "fs_main",
            &[UiVertexPoly::desc()],
            format,
            Some(additive_blend),
            PrimitiveTopology::TriangleStrip,
            MultisampleState {
                count: msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        );

        Self {
            device: device.clone(),
            uniform_buffer,
            uniform_bind_group,
            msaa_samples,
            format,

            vertex_buffer,
            quad_buffer,

            circle_shader,
            circle_layout,
            circle_pipeline_layout,
            circle_pipeline,

            outline_shader,
            outline_layout,
            outline_pipeline_layout,
            outline_pipeline,

            polygon_shader,
            polygon_pipeline_layout,
            polygon_pipeline,

            handle_quad_buffer,
            handle_shader,
            handle_layout,
            handle_pipeline_layout,
            handle_pipeline,

            glow_shader,
            glow_pipeline_layout,
            glow_pipeline,

            text_atlas: None,
            text_shader,
            text_layout,
            text_pipeline_layout,
            text_pipeline,

            text_bind_group: None,
            text_vertex_buffer,
            text_vertex_count: 0,

            additive_blend,
            good_blend,

            num_vertices: 0,
        }
    }

    pub(crate) fn rebuild_pipelines(&mut self) {
        self.text_pipeline = make_pipeline(
            &self.device,
            "UI Text Pipeline",
            &self.text_pipeline_layout,
            &self.text_shader,
            "vs_main",
            "fs_main",
            &[UiVertexText::desc()],
            self.format,
            Some(BlendState::ALPHA_BLENDING),
            PrimitiveTopology::TriangleList,
            MultisampleState {
                count: self.msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        );
        self.circle_pipeline = make_pipeline(
            &self.device,
            "UI Circle Pipeline",
            &self.circle_pipeline_layout,
            &self.circle_shader,
            "vs_main",
            "fs_main",
            &[UiVertexPoly::desc()],
            self.format,
            Some(BlendState::ALPHA_BLENDING),
            PrimitiveTopology::TriangleStrip,
            MultisampleState {
                count: self.msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        );
        self.glow_pipeline = make_pipeline(
            &self.device,
            "UI Glow Pipeline",
            &self.glow_pipeline_layout,
            &self.glow_shader,
            "vs_main",
            "fs_main",
            &[UiVertexPoly::desc()],
            self.format,
            Some(self.additive_blend),
            PrimitiveTopology::TriangleStrip,
            MultisampleState {
                count: self.msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        );
        self.handle_pipeline = make_pipeline(
            &self.device,
            "UI Handle Pipeline",
            &self.handle_pipeline_layout,
            &self.handle_shader,
            "vs_main",
            "fs_main",
            &[UiVertexPoly::desc()],
            self.format,
            self.good_blend,
            PrimitiveTopology::TriangleStrip,
            MultisampleState {
                count: self.msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        );
        self.outline_pipeline = make_pipeline(
            &self.device,
            "UI Outline Pipeline",
            &self.outline_pipeline_layout,
            &self.outline_shader,
            "vs_main",
            "fs_main",
            &[UiVertexPoly::desc()],
            self.format,
            self.good_blend,
            PrimitiveTopology::TriangleStrip,
            MultisampleState {
                count: self.msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        );
        self.polygon_pipeline = make_pipeline(
            &self.device,
            "UI Polygon Pipeline",
            &self.polygon_pipeline_layout,
            &self.polygon_shader,
            "vs_main",
            "fs_main",
            &[UiVertexPoly::desc()],
            self.format,
            Some(BlendState::ALPHA_BLENDING),
            PrimitiveTopology::TriangleStrip,
            MultisampleState {
                count: self.msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        );
    }
}
