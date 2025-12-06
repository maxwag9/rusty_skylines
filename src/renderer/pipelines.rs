use crate::components::camera::Camera;
use crate::resources::Uniforms;
use crate::ui::vertex::{LineVtx, Vertex};
use glam::Vec3;
use std::borrow::Cow;
use std::fs;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    *,
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FogUniforms {
    pub screen_size: [f32; 2],
    pub proj_params: [f32; 2],
    pub fog_density: f32,
    pub fog_height: f32,
    pub cam_height: f32,
    pub _pad0: f32,
    pub fog_color: [f32; 3],
    pub _pad1: f32,
    pub fog_sky_factor: f32,
    pub fog_height_falloff: f32,
    pub fog_start: f32,
    pub fog_end: f32,
}

pub struct Pipelines {
    pub device: Device,

    pub(crate) gizmo_vbuf: Buffer,

    pub(crate) uniform_buffer: Buffer,
    pub(crate) uniform_bind_group: BindGroup,

    pub msaa_texture: Texture,
    pub msaa_view: TextureView,

    pub depth_texture: Texture,
    pub depth_view: TextureView,

    pub(crate) pipeline: RenderPipeline,
    pub(crate) gizmo_pipeline: RenderPipeline,

    pub fog_uniform_buffer: Buffer,
    pub fog_bind_group: BindGroup,

    pub shader: ShaderModule,
    pipeline_layout: PipelineLayout,
    pub(crate) msaa_samples: u32,
    line_shader: ShaderModule,

    config: SurfaceConfiguration,

    shader_path: PathBuf,
    line_shader_path: PathBuf,
}

impl Pipelines {
    pub fn new(
        device: &Device,
        config: &SurfaceConfiguration,
        msaa_samples: u32,
        shader_dir: &Path,
        camera: &Camera,
        aspect: f32,
    ) -> anyhow::Result<Self> {
        let shader_path = shader_dir.join("ground.wgsl");
        let line_shader_path = shader_dir.join("lines.wgsl");
        let (msaa_texture, msaa_view) = create_msaa_targets(&device, &config, msaa_samples);
        let (depth_texture, depth_view) = create_depth_texture(&device, &config, msaa_samples);
        let (scene_color, scene_color_view) = create_color_texture(&device, &config, msaa_samples);
        let (resolved_depth_texture, resolved_depth_view) =
            create_depth_texture(&device, &config, msaa_samples);
        let (resolved_scene_color, resolved_scene_color_view) =
            create_color_texture(&device, &config, msaa_samples);
        let shader = load_shader(device, &shader_path, "Ground Shader")?;

        let aspect = config.width as f32 / config.height as f32;
        let sun = glam::Vec3::new(0.3, 1.0, 0.6).normalize();
        let cam_pos = camera.position();
        let vp = camera.view_proj(aspect).to_cols_array_2d();
        let uniforms = make_new_uniforms(vp, sun, cam_pos);

        let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Uniform Bind Group Layout"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(size_of::<Uniforms>() as u64),
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

        let fog_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Fog BGL"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(size_of::<FogUniforms>() as u64),
                },
                count: None,
            }],
        });

        // Fog uniform buffer
        let fog_uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Fog Uniform Buffer"),
            size: size_of::<FogUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Fog bind group
        let fog_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Fog Bind Group"),
            layout: &fog_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: fog_uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[
                &uniform_bind_group_layout, // group(0)
                &fog_bgl,                   // group(1)
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("3D Render Pipeline"),
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
                    format: config.format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                cull_mode: Some(Face::Front),
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: MultisampleState {
                count: msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let line_shader = load_shader(device, &line_shader_path, "Line Shader")?;

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
                    format: config.format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: MultisampleState {
                count: msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Ok(Self {
            shader,
            gizmo_vbuf,
            uniform_buffer,
            uniform_bind_group,

            msaa_texture,
            msaa_view,
            depth_texture,
            depth_view,

            pipeline,
            gizmo_pipeline,
            fog_uniform_buffer,
            fog_bind_group,
            device: device.clone(),
            pipeline_layout,
            msaa_samples,
            config: config.clone(),
            line_shader,
            shader_path,
            line_shader_path,
        })
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
                        format: self.config.format,
                        blend: Some(BlendState::REPLACE),
                        write_mask: ColorWrites::ALL,
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
                        format: self.config.format,
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

    pub fn reload_shaders(&mut self) -> anyhow::Result<()> {
        self.shader = load_shader(&self.device, &self.shader_path, "Ground Shader")?;
        self.line_shader = load_shader(&self.device, &self.line_shader_path, "Line Shader")?;

        self.recreate_pipelines();
        Ok(())
    }

    pub(crate) fn resize(&mut self, msaa_samples: u32) {
        (self.msaa_texture, self.msaa_view) =
            create_msaa_targets(&self.device, &self.config, self.msaa_samples);
        (self.depth_texture, self.depth_view) =
            create_depth_texture(&self.device, &self.config, msaa_samples);
    }
}

pub fn load_shader(device: &Device, path: &Path, label: &str) -> anyhow::Result<ShaderModule> {
    let src = fs::read_to_string(path)?;
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(Cow::Owned(src)),
    });
    Ok(module)
}

pub fn create_msaa_targets(
    device: &Device,
    config: &SurfaceConfiguration,
    samples: u32,
) -> (Texture, TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("MSAA Color Texture"),
        size: Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: samples,
        dimension: TextureDimension::D2,
        format: config.format,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

fn create_depth_texture(
    device: &Device,
    config: &SurfaceConfiguration,
    msaa_samples: u32,
) -> (Texture, TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("Depth Texture"),
        size: Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: msaa_samples,

        dimension: TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

fn create_color_texture(
    device: &Device,
    config: &SurfaceConfiguration,
    msaa_samples: u32,
) -> (Texture, TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("Scene Color Texture"),
        size: Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: msaa_samples,
        dimension: TextureDimension::D2,
        format: config.format,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

fn create_resolved_depth_texture(
    device: &Device,
    config: &SurfaceConfiguration,
) -> (Texture, TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("Scene Resolved Depth Texture"),
        size: Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: config.format,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

fn create_resolved_color_texture(
    device: &Device,
    config: &SurfaceConfiguration,
) -> (Texture, TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("Scene Resolved Color Texture"),
        size: Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: config.format,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

pub fn make_new_uniforms(vp: [[f32; 4]; 4], sun: Vec3, cam_pos: Vec3) -> Uniforms {
    Uniforms {
        view_proj: vp,

        sun_direction: sun.to_array(),
        _pad0: 0.0,

        _pad1: [0.0; 4],

        camera_pos: cam_pos.to_array(),
        _pad2: 0.0,

        fog_color: [0.65, 0.78, 0.90],
        fog_start: 150.0,

        fog_end: 600.0,
        _pad3: [0.0; 3],
        _pad4: [0.0; 4],
    }
}
