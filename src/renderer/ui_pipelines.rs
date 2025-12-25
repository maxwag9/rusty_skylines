use crate::renderer::pipelines::load_shader;
use crate::renderer::ui::ScreenUniform;
use crate::renderer::ui_text::TextAtlas;
use crate::ui::helper::{make_pipeline, make_storage_layout, make_uniform_layout};
use crate::ui::vertex::{UiVertexPoly, UiVertexText};
use std::path::{Path, PathBuf};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    *,
};
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
    pub text_atlas: TextAtlas,
    pub text_pipeline: RenderPipeline,
    pub text_layout: BindGroupLayout,
    pub text_bind_group: BindGroup,
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
    pub polygon_layout: BindGroupLayout,
    pub polygon_pipeline_layout: PipelineLayout,
    pub outline_pipeline_layout: PipelineLayout,
    pub good_blend: Option<BlendState>,
    pub additive_blend: BlendState,

    shader_dir: PathBuf,
}

impl UiPipelines {
    pub fn new(
        device: &Device,
        format: TextureFormat,
        msaa_samples: u32,
        size: PhysicalSize<u32>,
        shader_dir: &Path,
    ) -> anyhow::Result<Self> {
        let handle_quad_vertices = [
            UiVertexPoly {
                pos: [-3.0, -3.0],
                data: [0.0, 0.0],
                color: [1.0; 4],
                misc: [1.0; 4],
            },
            UiVertexPoly {
                pos: [3.0, -3.0],
                data: [0.0, 0.0],
                color: [1.0; 4],
                misc: [1.0; 4],
            },
            UiVertexPoly {
                pos: [-3.0, 3.0],
                data: [0.0, 0.0],
                color: [1.0; 4],
                misc: [1.0; 4],
            },
            UiVertexPoly {
                pos: [3.0, 3.0],
                data: [0.0, 0.0],
                color: [1.0; 4],
                misc: [1.0; 4],
            },
        ];
        let handle_quad_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Handle Quad VB"),
            contents: bytemuck::cast_slice(&handle_quad_vertices),
            usage: BufferUsages::VERTEX,
        });

        let quad_vertices = [
            UiVertexPoly {
                pos: [-1.0, -1.0],
                data: [0.0, 0.0],
                color: [1.0; 4],
                misc: [1.0, 0.0, 0.0, 0.0],
            },
            UiVertexPoly {
                pos: [1.0, -1.0],
                data: [0.0, 0.0],
                color: [1.0; 4],
                misc: [1.0, 0.0, 0.0, 0.0],
            },
            UiVertexPoly {
                pos: [-1.0, 1.0],
                data: [0.0, 0.0],
                color: [1.0; 4],
                misc: [1.0, 0.0, 0.0, 0.0],
            },
            UiVertexPoly {
                pos: [1.0, 1.0],
                data: [0.0, 0.0],
                color: [1.0; 4],
                misc: [1.0, 0.0, 0.0, 0.0],
            },
        ];
        let quad_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("UI Quad VB"),
            contents: bytemuck::cast_slice(&quad_vertices),
            usage: BufferUsages::VERTEX,
        });
        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("UI VB"),
            size: (1024 * 1024) as u64, // 1MB buffer
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_layout = make_uniform_layout(device, "UI Bind Layout");

        let screen_uniform = ScreenUniform {
            size: [size.width as f32, size.height as f32],
            time: 0.0,
            enable_dither: 1,
            mouse: [0.0, 0.0],
        };
        let screen_data = bytemuck::bytes_of(&screen_uniform);

        let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("UI Uniforms"),
            contents: screen_data,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let uniform_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("UI Bind Group"),
            layout: &uniform_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let text_shader =
            load_shader(&device, &shader_dir.join("text.wgsl"), "UI Text Shader")?.module;
        let text_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Text Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let text_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Text Pipeline Layout"),
            bind_group_layouts: &[&uniform_layout, &text_layout],
            push_constant_ranges: &[],
        });
        let text_pipeline = build_pipeline(
            device,
            "UI Text Pipeline",
            &text_pipeline_layout,
            &text_shader,
            &[UiVertexText::desc()],
            format,
            Some(BlendState::ALPHA_BLENDING),
            PrimitiveTopology::TriangleList,
            msaa_samples,
        );

        let text_vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("UI Text VB"),
            size: 256 * 1024,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // println!("CircleParams size: {}", std::mem::size_of::<CircleParams>());
        let circle_shader = load_shader(
            &device,
            &shader_dir.join("ui_circle.wgsl"),
            "UI Circle Shader",
        )?
        .module;
        let outline_shader = load_shader(
            &device,
            &shader_dir.join("ui_shape_outline.wgsl"),
            "UI Outline Shader",
        )?
        .module;
        let handle_shader = load_shader(
            &device,
            &shader_dir.join("ui_handle.wgsl"),
            "UI Handle Shader",
        )?
        .module;
        let circle_layout = make_storage_layout(device, "Circle Layout");
        let circle_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Circle Pipeline Layout"),
            bind_group_layouts: &[&uniform_layout, &circle_layout],
            push_constant_ranges: &[],
        });

        let circle_pipeline = build_pipeline(
            device,
            "UI Circle Pipeline",
            &circle_pipeline_layout,
            &circle_shader,
            &[UiVertexPoly::desc()],
            format,
            Some(BlendState::ALPHA_BLENDING),
            PrimitiveTopology::TriangleStrip,
            msaa_samples,
        );

        let good_blend = Some(BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::SrcAlpha,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
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
            bind_group_layouts: &[&uniform_layout, &outline_layout],
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
        let handle_layout = make_storage_layout(device, "Handle Layout");
        let handle_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Handle Pipeline Layout"),
            bind_group_layouts: &[&uniform_layout, &handle_layout],
            push_constant_ranges: &[],
        });

        let handle_pipeline = build_pipeline(
            device,
            "UI Handle Pipeline",
            &handle_pipeline_layout,
            &handle_shader,
            &[UiVertexPoly::desc()],
            format,
            good_blend,
            PrimitiveTopology::TriangleStrip,
            msaa_samples,
        );
        let polygon_shader = load_shader(
            &device,
            &shader_dir.join("ui_polygon.wgsl"),
            "UI Polygon Shader",
        )?
        .module;
        let polygon_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("polygon_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let polygon_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("UI Polygon Pipeline Layout"),
            bind_group_layouts: &[
                &uniform_layout, // group(0) -> ScreenUniform
                &polygon_layout, // group(1) -> polygon_infos, polygon_edges
            ],
            push_constant_ranges: &[],
        });

        let polygon_pipeline = build_pipeline(
            device,
            "UI Polygon Pipeline",
            &polygon_pipeline_layout,
            &polygon_shader,
            &[UiVertexPoly::desc()],
            format,
            Some(BlendState::ALPHA_BLENDING),
            PrimitiveTopology::TriangleStrip,
            msaa_samples,
        );

        let glow_shader = load_shader(
            &device,
            &shader_dir.join("ui_circle_glow.wgsl"),
            "UI Glow Shader",
        )?
        .module;

        let glow_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("UI Glow Pipeline Layout"),
            bind_group_layouts: &[&uniform_layout, &circle_layout],
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

        let glow_pipeline = build_pipeline(
            device,
            "UI Glow Pipeline",
            &glow_pipeline_layout,
            &glow_shader,
            &[UiVertexPoly::desc()],
            format,
            Some(additive_blend),
            PrimitiveTopology::TriangleStrip,
            msaa_samples,
        );

        let dummy_tex = device.create_texture(&TextureDescriptor {
            label: Some("Dummy Text Atlas"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let dummy_view = dummy_tex.create_view(&TextureViewDescriptor::default());

        let dummy_sampler = device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            ..Default::default()
        });

        let text_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Text Atlas Bind Group"),
            layout: &text_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&dummy_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&dummy_sampler),
                },
            ],
        });

        Ok(Self {
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
            polygon_layout,
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

            text_atlas: TextAtlas::new(device, (1024, 1024)),
            text_shader,
            text_layout,
            text_pipeline_layout,
            text_pipeline,

            text_bind_group,
            text_vertex_buffer,
            text_vertex_count: 0,

            additive_blend,
            good_blend,

            num_vertices: 0,
            shader_dir: shader_dir.to_path_buf(),
        })
    }

    pub(crate) fn rebuild_pipelines(&mut self) {
        let samples = self.msaa_samples;
        self.text_pipeline = build_pipeline(
            &self.device,
            "UI Text Pipeline",
            &self.text_pipeline_layout,
            &self.text_shader,
            &[UiVertexText::desc()],
            self.format,
            Some(BlendState::ALPHA_BLENDING),
            PrimitiveTopology::TriangleList,
            samples,
        );
        self.circle_pipeline = build_pipeline(
            &self.device,
            "UI Circle Pipeline",
            &self.circle_pipeline_layout,
            &self.circle_shader,
            &[UiVertexPoly::desc()],
            self.format,
            Some(BlendState::ALPHA_BLENDING),
            PrimitiveTopology::TriangleStrip,
            samples,
        );
        self.glow_pipeline = build_pipeline(
            &self.device,
            "UI Glow Pipeline",
            &self.glow_pipeline_layout,
            &self.glow_shader,
            &[UiVertexPoly::desc()],
            self.format,
            Some(self.additive_blend),
            PrimitiveTopology::TriangleStrip,
            samples,
        );
        self.handle_pipeline = build_pipeline(
            &self.device,
            "UI Handle Pipeline",
            &self.handle_pipeline_layout,
            &self.handle_shader,
            &[UiVertexPoly::desc()],
            self.format,
            self.good_blend,
            PrimitiveTopology::TriangleStrip,
            samples,
        );
        self.outline_pipeline = build_pipeline(
            &self.device,
            "UI Outline Pipeline",
            &self.outline_pipeline_layout,
            &self.outline_shader,
            &[UiVertexPoly::desc()],
            self.format,
            self.good_blend,
            PrimitiveTopology::TriangleStrip,
            samples,
        );
        self.polygon_pipeline = build_pipeline(
            &self.device,
            "UI Polygon Pipeline",
            &self.polygon_pipeline_layout,
            &self.polygon_shader,
            &[UiVertexPoly::desc()],
            self.format,
            Some(BlendState::ALPHA_BLENDING),
            PrimitiveTopology::TriangleStrip,
            samples,
        );
    }

    pub fn reload_shaders(&mut self) -> anyhow::Result<()> {
        self.text_shader = load_shader(
            &self.device,
            &self.shader_dir.join("text.wgsl"),
            "UI Text Shader",
        )?
        .module;
        self.circle_shader = load_shader(
            &self.device,
            &self.shader_dir.join("ui_circle.wgsl"),
            "UI Circle Shader",
        )?
        .module;
        self.outline_shader = load_shader(
            &self.device,
            &self.shader_dir.join("ui_shape_outline.wgsl"),
            "UI Outline Shader",
        )?
        .module;
        self.handle_shader = load_shader(
            &self.device,
            &self.shader_dir.join("ui_handle.wgsl"),
            "UI Handle Shader",
        )?
        .module;
        self.polygon_shader = load_shader(
            &self.device,
            &self.shader_dir.join("ui_polygon.wgsl"),
            "UI Polygon Shader",
        )?
        .module;
        self.glow_shader = load_shader(
            &self.device,
            &self.shader_dir.join("ui_circle_glow.wgsl"),
            "UI Glow Shader",
        )?
        .module;

        self.rebuild_pipelines();
        println!("rebuilt shaders");
        Ok(())
    }
}

fn build_pipeline(
    device: &Device,
    label: &str,
    layout: &PipelineLayout,
    shader: &ShaderModule,
    buffers: &[VertexBufferLayout],
    format: TextureFormat,
    blend: Option<BlendState>,
    topology: PrimitiveTopology,
    samples: u32,
) -> RenderPipeline {
    make_pipeline(
        device,
        label,
        layout,
        shader,
        "vs_main",
        "fs_main",
        buffers,
        format,
        blend,
        topology,
        multisample_state(samples),
    )
}

fn multisample_state(samples: u32) -> MultisampleState {
    MultisampleState {
        count: samples,
        mask: !0,
        alpha_to_coverage_enabled: false,
    }
}
