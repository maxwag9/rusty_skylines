use crate::renderer::ui::ScreenUniform;
use crate::ui::helper::{make_storage_layout, make_uniform_layout};
use crate::ui::vertex::UiVertexPoly;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;
use winit::dpi::PhysicalSize;

pub struct UiPipelines {
    pub device: Device,
    pub uniform_layout: BindGroupLayout,
    pub uniform_buffer: Buffer,
    pub uniform_bind_group: BindGroup,
    pub background_buffer: Buffer,

    pub msaa_samples: u32,

    format: TextureFormat,
    pub quad_buffer: Buffer,
    pub handle_quad_buffer: Buffer,
    pub handle_layout: BindGroupLayout,
    pub circle_layout: BindGroupLayout,
    pub rect_layout: BindGroupLayout,
    pub outline_layout: BindGroupLayout,
    pub polygon_layout: BindGroupLayout,
    pub good_blend: Option<BlendState>,
    pub additive_blend: BlendState,
}
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Background {
    pub primary_color: [f32; 4],
    pub secondary_color: [f32; 4],
    pub block_size: f32,

    pub warp_strength: f32,
    pub warp_radius: f32,
    pub time_scale: f32,
    pub wave_strength: f32,
    pub _padding: [f32; 3], // forces 64 bytes
}
impl UiPipelines {
    pub fn new(
        device: &Device,
        config: &SurfaceConfiguration,
        msaa_samples: u32,
        size: PhysicalSize<u32>,
    ) -> anyhow::Result<Self> {
        let format = config.format;
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

        let uniform_layout = make_uniform_layout(device, "UI Bind Layout");

        let screen_uniform = ScreenUniform {
            size: [size.width as f32, size.height as f32],
            time: 0.0,
            enable_dither: 1,
            mouse: [0.0, 0.0],
        };
        let screen_data = bytemuck::bytes_of(&screen_uniform);
        let rect_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("rect_layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("UI Uniforms"),
            contents: screen_data,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let background_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("UI Background Uniforms"),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            size: size_of::<Background>() as u64,
            mapped_at_creation: false,
        });
        let uniform_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("UI Bind Group"),
            layout: &uniform_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

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

        let circle_layout = make_storage_layout(device, "Circle Layout");

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
        let ssbo_entries = &[
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
                    min_binding_size: Some(48.try_into()?),
                },
                count: None,
            },
        ];
        let outline_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Outline Layout"),
            entries: ssbo_entries,
        });
        let handle_layout = make_storage_layout(device, "Handle Layout");
        let polygon_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("polygon_layout"),
            entries: ssbo_entries,
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

        Ok(Self {
            device: device.clone(),
            uniform_layout,
            uniform_buffer,
            uniform_bind_group,
            background_buffer,
            msaa_samples,
            format,

            quad_buffer,

            circle_layout,

            outline_layout,

            polygon_layout,

            handle_quad_buffer,
            handle_layout,

            additive_blend,
            good_blend,
            rect_layout,
        })
    }
}

pub fn multisample_state(samples: u32) -> MultisampleState {
    MultisampleState {
        count: samples,
        mask: !0,
        alpha_to_coverage_enabled: false,
    }
}
