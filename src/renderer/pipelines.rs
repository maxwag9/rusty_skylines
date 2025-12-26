use crate::components::camera::Camera;
use crate::mouse_ray::PickUniform;
use crate::paths::data_dir;
use crate::renderer::textures::grass::{GrassParams, generate_noise};
use crate::resources::Uniforms;
use crate::terrain::sky::SkyUniform;
use crate::terrain::water::{SimpleVertex, WaterUniform};
use crate::ui::vertex::{LineVtx, Vertex};
use glam::{Mat4, Vec3};
use std::borrow::Cow;
use std::fs;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use wgpu::TextureFormat::Rgba8Unorm;
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

pub struct RenderPipelineState {
    pub shader: ShaderAsset,
    pub pipeline: RenderPipeline,
}

pub struct ComputePipelineState {
    pub shader: ShaderAsset,
    pub pipeline: ComputePipeline,
}
pub struct GpuResourceSet {
    pub bind_group_layout: BindGroupLayout,
    pub bind_group: BindGroup,
    pub buffer: Buffer,
}
pub struct MeshBuffers {
    pub vertex: Buffer,
    pub index: Buffer,
    pub index_count: u32,
}

#[derive(Clone)]
pub struct ShaderAsset {
    pub path: PathBuf,
    pub module: ShaderModule,
}

pub struct Pipelines {
    pub device: Device,

    pub msaa_texture: Texture,
    pub msaa_view: TextureView,

    pub depth_texture: Texture,
    pub depth_view: TextureView,

    pub(crate) msaa_samples: u32,

    pub config: SurfaceConfiguration,

    pub uniforms: GpuResourceSet,
    pub sky_uniforms: GpuResourceSet,
    pub water_uniforms: GpuResourceSet,
    pub fog_uniforms: GpuResourceSet,
    pub pick_uniforms: GpuResourceSet,

    pub terrain_pipeline_above_water: RenderPipelineState,
    pub terrain_pipeline_under_water: RenderPipelineState,
    pub water_pipeline: RenderPipelineState,
    pub water_mesh_buffers: MeshBuffers,
    pub sky_pipeline: RenderPipelineState,
    pub stars_pipeline: RenderPipelineState,
    pub stars_mesh_buffers: MeshBuffers,
    pub gizmo_pipeline: RenderPipelineState,
    pub gizmo_mesh_buffers: MeshBuffers,
    pub grass_texture_pipeline: ComputePipelineState,
    pub grass_texture_resources: GpuResourceSet,
}

impl Pipelines {
    pub fn new(
        device: &Device,
        config: &SurfaceConfiguration,
        msaa_samples: u32,
        shader_dir: &Path,
        camera: &Camera,
    ) -> anyhow::Result<Self> {
        let terrain_shader_path = shader_dir.join("ground.wgsl");
        let line_shader_path = shader_dir.join("lines.wgsl");
        let (msaa_texture, msaa_view) = create_msaa_targets(&device, &config, msaa_samples);
        let (depth_texture, depth_view) = create_depth_texture(&device, &config, msaa_samples);

        let terrain_shader = load_shader(device, &terrain_shader_path, "Ground Shader")?;

        let grass_texture_shader_path = shader_dir.join("textures/grass.wgsl");
        let grass_texture_shader =
            load_shader(device, &grass_texture_shader_path, "Grass Texture Shader")?;

        let aspect = config.width as f32 / config.height as f32;
        let sun = Vec3::new(0.3, 1.0, 0.6).normalize();
        let cam_pos = camera.position();
        let (view, proj, view_proj) = camera.matrices(aspect);
        let uniforms = make_new_uniforms(
            view,
            proj,
            view_proj,
            sun,
            Vec3::new(0.0, 0.0, 0.0),
            cam_pos,
            camera.orbit_radius,
            0.0,
        );
        let pick_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Pick Uniform BGL"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pick_uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Pick Uniform Buffer"),
            contents: bytemuck::bytes_of(&PickUniform {
                pos: [0.0; 3],
                radius: 0.0,
                underwater: 0,
                _pad0: [0, 0, 0],
                color: [1.0, 0.0, 0.0],
                _pad1: 0.0,
            }),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let pick_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Pick Uniform BG"),
            layout: &pick_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: pick_uniform_buffer.as_entire_binding(),
            }],
        });
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

        let line_shader = load_shader(device, &line_shader_path, "Line Shader")?;

        let water_shader_path = shader_dir.join("water.wgsl");
        let water_shader = load_shader(device, &water_shader_path, "Water Shader")?;

        let water_vertices = [
            SimpleVertex {
                pos: [-20000.0, 0.0, -20000.0],
            },
            SimpleVertex {
                pos: [20000.0, 0.0, -20000.0],
            },
            SimpleVertex {
                pos: [20000.0, 0.0, 20000.0],
            },
            SimpleVertex {
                pos: [-20000.0, 0.0, 20000.0],
            },
        ];

        let water_indices: [u32; 6] = [0, 1, 2, 0, 2, 3];

        let water_vbuf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Water VB"),
            contents: bytemuck::cast_slice(&water_vertices),
            usage: BufferUsages::VERTEX,
        });

        let water_ibuf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Water IB"),
            contents: bytemuck::cast_slice(&water_indices),
            usage: BufferUsages::INDEX,
        });

        let water_index_count = water_indices.len() as u32;

        let wu = WaterUniform {
            sea_level: 0.0,
            _pad0: [0.0; 3],
            color: [0.05, 0.25, 0.35, 0.55],
            wave_tiling: 0.05,
            wave_strength: 0.05,
            _pad1: [0.0; 2],
        };

        let water_uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Water Uniform Buffer"),
            contents: bytemuck::bytes_of(&wu),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let water_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Water BGL"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(size_of::<WaterUniform>() as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(size_of::<SkyUniform>() as u64),
                    },
                    count: None,
                },
            ],
        });

        let sky_uniform = SkyUniform {
            exposure: 1.0,
            moon_phase: 0.0,

            sun_size: 0.05, // NDC radius for now (0.05 = big)
            sun_intensity: 510.0,

            moon_size: 0.04,
            moon_intensity: 1.0,

            _pad1: 0.0,
            _pad2: 0.0,
        };

        let sky_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Sky Uniform Buffer"),
            contents: bytemuck::bytes_of(&sky_uniform),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let water_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Water BG"),
            layout: &water_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: water_uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: sky_buffer.as_entire_binding(),
                },
            ],
        });

        let sky_shader_path = shader_dir.join("sky.wgsl");
        let sky_shader = load_shader(device, &sky_shader_path, "Sky Shader")?;

        let stars_shader_path = shader_dir.join("stars.wgsl");
        let stars_shader = load_shader(device, &stars_shader_path, "Stars Shader")?;

        let sky_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Sky Uniforms BGL"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let stars_bytes = fs::read(data_dir("stars.bin")).expect("stars.bin missing");

        let stars_vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Star Buffer"),
            contents: &stars_bytes,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });

        let sky_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Sky BG"),
            layout: &sky_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: sky_buffer.as_entire_binding(),
            }],
        });
        let grass_params = GrassParams {
            grass_color: [0.2, 0.6, 0.2, 1.0],
            blade_density: 120.0,
            blade_height: 0.8,
            wind_phase: 0.0,
            time: 0.0,
            noise_scale: 4.0,
            _pad: [0.0; 3],
        };

        let grass_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grass_params_buffer"),
            contents: bytemuck::bytes_of(&grass_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let noise_data: Vec<f32> = generate_noise(512);

        let noise_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grass_noise_buffer"),
            contents: bytemuck::cast_slice(&noise_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let grass_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("grass_texture_bgl"),
                entries: &[
                    // storage texture output
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // uniform buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // noise buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let (_, grass_texture_view) = create_grass_texture(&device, &config);
        let grass_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Grass Texture Bind Group"),
            layout: &grass_texture_bind_group_layout,
            entries: &[
                // storage texture
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&grass_texture_view),
                },
                // uniform params
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grass_params_buffer.as_entire_binding(),
                },
                // noise buffer
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: noise_buffer.as_entire_binding(),
                },
            ],
        });

        let mut this = Self {
            device: device.clone(),
            msaa_texture,
            msaa_view,
            depth_texture,
            depth_view,
            msaa_samples,
            config: config.clone(),

            uniforms: GpuResourceSet {
                bind_group_layout: uniform_bind_group_layout,
                bind_group: uniform_bind_group,
                buffer: uniform_buffer,
            },
            sky_uniforms: GpuResourceSet {
                bind_group_layout: sky_bgl,
                bind_group: sky_bind_group,
                buffer: sky_buffer,
            },
            water_uniforms: GpuResourceSet {
                bind_group_layout: water_bgl,
                bind_group: water_bind_group,
                buffer: water_uniform_buffer,
            },
            fog_uniforms: GpuResourceSet {
                bind_group_layout: fog_bgl,
                bind_group: fog_bind_group,
                buffer: fog_uniform_buffer,
            },
            pick_uniforms: GpuResourceSet {
                bind_group_layout: pick_bgl,
                bind_group: pick_bind_group,
                buffer: pick_uniform_buffer,
            },

            terrain_pipeline_above_water: make_dummy_render_pipeline_state(
                device,
                config.format,
                terrain_shader.clone(),
            ),

            terrain_pipeline_under_water: make_dummy_render_pipeline_state(
                device,
                config.format,
                terrain_shader,
            ),
            water_pipeline: make_dummy_render_pipeline_state(device, config.format, water_shader),
            water_mesh_buffers: MeshBuffers {
                vertex: water_vbuf,
                index: water_ibuf,
                index_count: water_index_count,
            },

            sky_pipeline: make_dummy_render_pipeline_state(device, config.format, sky_shader),

            gizmo_pipeline: make_dummy_render_pipeline_state(device, config.format, line_shader),
            gizmo_mesh_buffers: MeshBuffers {
                vertex: gizmo_vbuf,
                index: make_dummy_buf(&device),
                index_count: 0,
            },

            stars_pipeline: make_dummy_render_pipeline_state(device, config.format, stars_shader),
            stars_mesh_buffers: MeshBuffers {
                vertex: stars_vertex_buffer,
                index: make_dummy_buf(&device),
                index_count: 0,
            },

            grass_texture_pipeline: make_dummy_compute_pipeline_state(device, grass_texture_shader),
            grass_texture_resources: GpuResourceSet {
                bind_group_layout: grass_texture_bind_group_layout,
                bind_group: grass_texture_bind_group,
                buffer: grass_params_buffer,
            },
        };

        this.recreate_pipelines();
        Ok(this)
    }

    pub(crate) fn recreate_pipelines(&mut self) {
        (
            self.terrain_pipeline_above_water.pipeline,
            self.terrain_pipeline_under_water.pipeline,
        ) = self.build_terrain_pipelines();
        self.gizmo_pipeline.pipeline = self.build_gizmo_pipeline();
        self.water_pipeline.pipeline = self.build_water_pipeline();
        self.sky_pipeline.pipeline = self.build_sky_pipeline();
        self.stars_pipeline.pipeline = self.build_stars_pipeline();
    }

    pub fn reload_shaders(&mut self) -> anyhow::Result<()> {
        self.terrain_pipeline_above_water.shader = load_shader(
            &self.device,
            &self.terrain_pipeline_above_water.shader.path,
            "Ground Shader",
        )?;
        self.gizmo_pipeline.shader = load_shader(
            &self.device,
            &self.gizmo_pipeline.shader.path,
            "Line Shader",
        )?;
        self.water_pipeline.shader = load_shader(
            &self.device,
            &self.water_pipeline.shader.path,
            "Water Shader",
        )?;
        self.sky_pipeline.shader =
            load_shader(&self.device, &self.sky_pipeline.shader.path, "Sky Shader")?;
        self.stars_pipeline.shader = load_shader(
            &self.device,
            &self.stars_pipeline.shader.path,
            "Stars Shader",
        )?;

        self.recreate_pipelines();
        Ok(())
    }

    pub(crate) fn resize(&mut self, config: &SurfaceConfiguration, msaa_samples: u32) {
        // Keep a fresh copy of the surface configuration so our MSAA and depth textures
        // always match the swapchain size. Or ELSE, after a window resize we'd recreate
        // attachments using the old dimensions, leading to mismatched resolve targets!!!
        self.config = config.clone();
        (self.msaa_texture, self.msaa_view) =
            create_msaa_targets(&self.device, &self.config, msaa_samples);
        (self.depth_texture, self.depth_view) =
            create_depth_texture(&self.device, &self.config, msaa_samples);
    }
    fn build_terrain_pipelines(&self) -> (RenderPipeline, RenderPipeline) {
        let terrain_pipeline_layout =
            self.device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("Terrain Pipeline Layout"),
                    bind_group_layouts: &[
                        &self.uniforms.bind_group_layout,      // group(0)
                        &self.fog_uniforms.bind_group_layout,  // group(1)
                        &self.pick_uniforms.bind_group_layout, // group(2) :O
                    ],
                    immediate_size: 0,
                });

        let above_water = self
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("Terrain Pipeline"),
                layout: Some(&terrain_pipeline_layout),
                vertex: VertexState {
                    module: &self.terrain_pipeline_above_water.shader.module,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(FragmentState {
                    module: &self.terrain_pipeline_above_water.shader.module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(ColorTargetState {
                        format: self.config.format,
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
                    stencil: StencilState {
                        front: StencilFaceState {
                            compare: CompareFunction::Always,
                            fail_op: StencilOperation::Keep,
                            depth_fail_op: StencilOperation::Keep,
                            pass_op: StencilOperation::Replace,
                        },
                        back: StencilFaceState {
                            compare: CompareFunction::Always,
                            fail_op: StencilOperation::Keep,
                            depth_fail_op: StencilOperation::Keep,
                            pass_op: StencilOperation::Replace,
                        },
                        read_mask: 0xFF,
                        write_mask: 0,
                    },
                    bias: Default::default(),
                }),
                multisample: MultisampleState {
                    count: self.msaa_samples,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: None,
                multiview_mask: None,
            });
        let under_water = self
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("Terrain Pipeline"),
                layout: Some(&terrain_pipeline_layout),
                vertex: VertexState {
                    module: &self.terrain_pipeline_above_water.shader.module,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(FragmentState {
                    module: &self.terrain_pipeline_above_water.shader.module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(ColorTargetState {
                        format: self.config.format,
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
                    stencil: StencilState {
                        front: StencilFaceState {
                            compare: CompareFunction::Always,
                            fail_op: StencilOperation::Keep,
                            depth_fail_op: StencilOperation::Keep,
                            pass_op: StencilOperation::Replace,
                        },
                        back: StencilFaceState {
                            compare: CompareFunction::Always,
                            fail_op: StencilOperation::Keep,
                            depth_fail_op: StencilOperation::Keep,
                            pass_op: StencilOperation::Replace,
                        },
                        read_mask: 0xFF,
                        write_mask: 0xFF,
                    },
                    bias: Default::default(),
                }),
                multisample: MultisampleState {
                    count: self.msaa_samples,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: None,
                multiview_mask: None,
            });
        (above_water, under_water)
    }

    fn build_gizmo_pipeline(&self) -> RenderPipeline {
        let gizmo_pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Gizmo Pipeline Layout"),
                bind_group_layouts: &[
                    &self.uniforms.bind_group_layout,      // group(0)
                    &self.fog_uniforms.bind_group_layout,  // group(1)
                    &self.pick_uniforms.bind_group_layout, // group(2) :O
                ],
                immediate_size: 0,
            });
        self.device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("Gizmo Pipeline"),
                layout: Some(&gizmo_pipeline_layout),
                vertex: VertexState {
                    module: &self.gizmo_pipeline.shader.module,
                    entry_point: Some("vs_main"),
                    buffers: &[LineVtx::layout()],
                    compilation_options: Default::default(),
                },
                fragment: Some(FragmentState {
                    module: &self.gizmo_pipeline.shader.module,
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
                depth_stencil: Some(DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: CompareFunction::LessEqual,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: MultisampleState {
                    count: self.msaa_samples,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: None,
                multiview_mask: None,
            })
    }

    fn build_water_pipeline(&self) -> RenderPipeline {
        let water_pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Water Pipeline Layout"),
                bind_group_layouts: &[
                    &self.uniforms.bind_group_layout,
                    &self.water_uniforms.bind_group_layout,
                ],
                immediate_size: 0,
            });
        self.device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("Water Pipeline"),
                layout: Some(&water_pipeline_layout),
                vertex: VertexState {
                    module: &self.water_pipeline.shader.module,
                    entry_point: Some("vs_main"),
                    buffers: &[SimpleVertex::layout()],
                    compilation_options: Default::default(),
                },
                fragment: Some(FragmentState {
                    module: &self.water_pipeline.shader.module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(ColorTargetState {
                        format: self.config.format,
                        blend: Some(BlendState::ALPHA_BLENDING),
                        write_mask: ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: CompareFunction::Always,
                    stencil: StencilState {
                        front: StencilFaceState {
                            compare: CompareFunction::Equal,
                            fail_op: StencilOperation::Keep,
                            depth_fail_op: StencilOperation::Keep,
                            pass_op: StencilOperation::Keep,
                        },
                        back: StencilFaceState {
                            compare: CompareFunction::Equal,
                            fail_op: StencilOperation::Keep,
                            depth_fail_op: StencilOperation::Keep,
                            pass_op: StencilOperation::Keep,
                        },
                        read_mask: 0xFF,
                        write_mask: 0x00,
                    },
                    bias: Default::default(),
                }),
                multisample: MultisampleState {
                    count: self.msaa_samples,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: None,
                multiview_mask: None,
            })
    }

    fn build_sky_pipeline(&self) -> RenderPipeline {
        let sky_pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Sky Pipeline Layout"),
                    bind_group_layouts: &[
                        &self.uniforms.bind_group_layout,     // group 0
                        &self.sky_uniforms.bind_group_layout, // group 1
                    ],
                    immediate_size: 0,
                });
        self.device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("Sky Pipeline"),
                layout: Some(&sky_pipeline_layout),
                vertex: VertexState {
                    module: &self.sky_pipeline.shader.module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(FragmentState {
                    module: &self.sky_pipeline.shader.module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(ColorTargetState {
                        format: self.config.format,
                        blend: Some(BlendState::ALPHA_BLENDING),
                        write_mask: ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: PrimitiveState::default(),
                depth_stencil: Some(DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: CompareFunction::Always,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: MultisampleState {
                    count: self.msaa_samples,
                    ..Default::default()
                },
                cache: None,
                multiview_mask: None,
            })
    }

    fn build_stars_pipeline(&self) -> RenderPipeline {
        let stars_vertex_layout = VertexBufferLayout {
            array_stride: 16,
            step_mode: VertexStepMode::Instance, // IMPORTANT
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: 4,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        };
        let stars_pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Stars Pipeline Layout"),
                bind_group_layouts: &[
                    &self.uniforms.bind_group_layout, // group 0
                ],
                immediate_size: 0,
            });
        self.device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("Stars Pipeline"),
                layout: Some(&stars_pipeline_layout),
                vertex: VertexState {
                    module: &self.stars_pipeline.shader.module,
                    entry_point: Some("vs_main"),
                    buffers: &[stars_vertex_layout],
                    compilation_options: Default::default(),
                },
                fragment: Some(FragmentState {
                    module: &self.stars_pipeline.shader.module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(ColorTargetState {
                        format: self.config.format,
                        blend: Some(BlendState::ALPHA_BLENDING),
                        write_mask: ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: Some(DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: CompareFunction::Always,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: MultisampleState {
                    count: self.msaa_samples,
                    ..Default::default()
                },
                cache: None,
                multiview_mask: None,
            })
    }
    fn build_grass_texture_pipeline(&mut self) {
        let grass_texture_pipeline_layout = self.device.create_pipeline_layout(&Default::default());
        self.grass_texture_pipeline.pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Procedural Texture Compute Pipeline GRASS"),
                    layout: Some(&grass_texture_pipeline_layout),
                    module: &self.grass_texture_pipeline.shader.module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
    }
}

fn make_dummy_buf(device: &Device) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some("dummy ibuf"),
        size: 0,
        usage: BufferUsages::INDEX,
        mapped_at_creation: false,
    })
}

pub fn load_shader(device: &Device, path: &PathBuf, label: &str) -> anyhow::Result<ShaderAsset> {
    let src = fs::read_to_string(path)?;
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(Cow::Owned(src)),
    });
    let asset = ShaderAsset {
        path: path.clone(),
        module,
    };
    Ok(asset)
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

pub const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth24PlusStencil8;

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

fn create_grass_texture(device: &Device, config: &SurfaceConfiguration) -> (Texture, TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Grass Texture"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm, // NOT sRGB
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

pub fn make_new_uniforms(
    view: Mat4,
    proj: Mat4,
    view_proj: Mat4,
    sun: Vec3,
    moon: Vec3,
    cam_pos: Vec3,
    orbit_radius: f32,
    total_time: f32,
) -> Uniforms {
    Uniforms {
        view: view.to_cols_array_2d(),
        inv_view: view.inverse().to_cols_array_2d(),
        proj: proj.to_cols_array_2d(),
        inv_proj: proj.inverse().to_cols_array_2d(),
        view_proj: view_proj.to_cols_array_2d(),
        inv_view_proj: view_proj.inverse().to_cols_array_2d(),

        sun_direction: sun.to_array(),
        time: total_time,

        camera_pos: cam_pos.to_array(),
        orbit_radius,

        moon_direction: moon.to_array(),
        _pad0: 0.0,
    }
}

fn make_dummy_render_pipeline(device: &Device, format: TextureFormat) -> RenderPipeline {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("dummy shader"),
        source: ShaderSource::Wgsl(
            "
            @vertex
            fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
                // fullscreen triangle
                let x = f32(idx == 1u) * 4.0 - 1.0;
                let y = f32(idx == 2u) * 4.0 - 1.0;
                return vec4<f32>(x, y, 0.0, 1.0);
            }

            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 0.0, 0.0, 1.0); // red
            }
        "
            .into(),
        ),
    });

    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("dummy layout"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });

    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("dummy pipeline"),
        layout: Some(&layout),

        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },

        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(ColorTargetState {
                format,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),

        primitive: PrimitiveState::default(),
        depth_stencil: None,

        multisample: MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        cache: None,
        multiview_mask: None,
    })
}

fn make_dummy_render_pipeline_state(
    device: &Device,
    format: TextureFormat,
    shader: ShaderAsset,
) -> RenderPipelineState {
    RenderPipelineState {
        shader,
        pipeline: make_dummy_render_pipeline(device, format),
    }
}
fn make_dummy_compute_pipeline(device: &Device) -> ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("dummy compute shader"),
        source: wgpu::ShaderSource::Wgsl(
            "
            @compute @workgroup_size(1)
            fn main() {
                // do nothing
            }
            "
            .into(),
        ),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("dummy compute layout"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("dummy compute pipeline"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn make_dummy_compute_pipeline_state(device: &Device, shader: ShaderAsset) -> ComputePipelineState {
    ComputePipelineState {
        shader,
        pipeline: make_dummy_compute_pipeline(device),
    }
}
