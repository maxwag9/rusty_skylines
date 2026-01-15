use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use wgpu::util::DeviceExt;
use wgpu::{Device, Queue, StoreOp, TextureView};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaterialKind {
    Asphalt,
    Grass,
    Concrete,
    Goo,
    Brick,
    Metal,
    Wood,
    Water,
}

impl MaterialKind {
    fn shader_filename(self) -> &'static str {
        match self {
            Self::Asphalt => "asphalt.wgsl",
            Self::Grass => "grass.wgsl",
            Self::Concrete => "concrete.wgsl",
            Self::Goo => "goo.wgsl",
            Self::Brick => "brick.wgsl",
            Self::Metal => "metal.wgsl",
            Self::Wood => "wood.wgsl",
            Self::Water => "water.wgsl",
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub seed: u32,
    pub scale: f32,
    pub roughness: f32,
    pub _padding: u32,
    pub color_primary: [f32; 4],
    pub color_secondary: [f32; 4],
}

impl Default for Params {
    fn default() -> Self {
        Self {
            seed: 0,
            scale: 1.0,
            roughness: 0.5,
            _padding: 0,
            color_primary: [1.0, 1.0, 1.0, 1.0],
            color_secondary: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

impl PartialEq for Params {
    fn eq(&self, other: &Self) -> bool {
        self.seed == other.seed
            && self.scale.to_bits() == other.scale.to_bits()
            && self.roughness.to_bits() == other.roughness.to_bits()
            && self
                .color_primary
                .iter()
                .zip(&other.color_primary)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && self
                .color_secondary
                .iter()
                .zip(&other.color_secondary)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

impl Eq for Params {}

impl Hash for Params {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.seed.hash(state);
        self.scale.to_bits().hash(state);
        self.roughness.to_bits().hash(state);
        for c in &self.color_primary {
            c.to_bits().hash(state);
        }
        for c in &self.color_secondary {
            c.to_bits().hash(state);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureCacheKey {
    pub kind: MaterialKind,
    pub params: Params,
    pub resolution: u32,
}

struct CachedTexture {
    _texture: wgpu::Texture,
    view: TextureView,
}

struct PipelineEntry {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

pub struct ProceduralTextureManager {
    device: Device,
    queue: Queue,
    shader_dir: PathBuf,
    pipelines: HashMap<MaterialKind, PipelineEntry>,
    texture_cache: HashMap<TextureCacheKey, CachedTexture>,
}

impl ProceduralTextureManager {
    pub fn new(device: Device, queue: Queue, shader_dir: impl AsRef<Path>) -> Self {
        Self {
            device,
            queue,
            shader_dir: shader_dir.as_ref().to_path_buf(),
            pipelines: HashMap::new(),
            texture_cache: HashMap::new(),
        }
    }
    pub fn ensure_texture(&mut self, key: &TextureCacheKey) {
        if !self.texture_cache.contains_key(key) {
            self.ensure_pipeline(key.kind);
            self.generate_texture(*key);
        }
    }

    pub fn get_texture_view(&self, key: &TextureCacheKey) -> &TextureView {
        &self
            .texture_cache
            .get(key)
            .expect("texture must be ensured first")
            .view
    }

    pub fn reload_all_shaders(&mut self) {
        self.pipelines.clear();
        self.texture_cache.clear();
    }

    fn ensure_pipeline(&mut self, kind: MaterialKind) {
        if self.pipelines.contains_key(&kind) {
            return;
        }

        let shader_path = self.shader_dir.join(kind.shader_filename());
        let shader_source = std::fs::read_to_string(&shader_path)
            .unwrap_or_else(|e| panic!("Failed to read shader {:?}: {}", shader_path, e));

        let shader_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(kind.shader_filename()),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::Rgba8Unorm,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
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
                    ],
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        self.pipelines.insert(
            kind,
            PipelineEntry {
                pipeline,
                bind_group_layout,
            },
        );
    }

    fn generate_texture(&mut self, key: TextureCacheKey) {
        // get compute pipeline that writes the procedural texture into mip 0
        let (compute_pipeline, compute_bind_group_layout) = {
            let entry = self.pipelines.get(&key.kind).unwrap();
            (entry.pipeline.clone(), entry.bind_group_layout.clone())
        };

        // texture size and mip count
        let size = wgpu::Extent3d {
            width: key.resolution,
            height: key.resolution,
            depth_or_array_layers: 1,
        };
        let dimension = wgpu::TextureDimension::D2;
        let mip_count = size.max_mips(dimension);

        // Create texture with all mip levels and RENDER_ATTACHMENT so we can render into each mip
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: mip_count,
            sample_count: 1,
            dimension,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        // Create a view for fullscreen sampling if needed (not used for compute below)
        let _full_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create a view specifically for mip 0; we'll bind this as the storage target for the compute pass
        let view_mip0 = texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            base_mip_level: 0,
            mip_level_count: Some(1),
            ..Default::default()
        });

        // Uniform buffer used by the compute-generation pipeline
        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&key.params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Bind group for the compute pass must reference the mip0 view so compute writes to mip 0 only
        let compute_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_mip0),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // create a sampler for the blit steps (linear filtering)
        let blit_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mip_blit_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest, // we explicitly sample a single mip view
            lod_min_clamp: 0.0,
            lod_max_clamp: f32::MAX,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        // create command encoder. We will do compute then blit mips in the same encoder so GPU order is preserved.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // 1) Compute pass: generate procedural data into mip 0
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("generate_texture_compute"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &compute_bind_group, &[]);

            let workgroup_size = 8u32;
            let workgroups = key.resolution.div_ceil(workgroup_size);
            compute_pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        // 2) Create a tiny blit shader (fullscreen triangle) that samples the source mip and writes to the render target.
        // It expects binding 0 = texture_2d<f32>, binding 1 = sampler
        const BLIT_WGSL: &str = r#"
        @vertex
        fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
            // Fullscreen triangle without vertex buffer
            var pos = array<vec2<f32>, 3>(
                vec2<f32>(-1.0, -3.0),
                vec2<f32>(-1.0,  1.0),
                vec2<f32>( 3.0,  1.0)
            );
            let p = pos[vi];
            return vec4<f32>(p, 0.0, 1.0);
        }

        @group(0) @binding(0) var src_tex: texture_2d<f32>;
        @group(0) @binding(1) var src_sampler: sampler;

        struct FSOut { @location(0) color: vec4<f32> };

        @fragment
        fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> FSOut {
            // Derive uv from frag_coord and the implicit framebuffer size
            // We'll use normalized device coords to compute uv
            let ndc = frag_coord.xy / vec2<f32>(/* filled by pipeline via viewport if needed */ 1.0, 1.0);
            // But we cannot query the framebuffer size here cleanly, so instead compute uv from the vertex
            // trick: use built-in interpolated position trick avoided here by sampling using device coords in vertex.
            // For safety, sample with the normalized coordinates from gl_FragCoord style:
            // However wgpu does not give us framebuffer size, so we can rely on vertex positions mapping to full uv range if the vertex shader outputs them.
            // Instead, compute uv from the builtin position in vertex shader would be cleaner, but to keep the blit minimal, we'll use the normalized screen coords:
            let uv = (frag_coord.xy) / vec2<f32>(f32(textureDimensions(src_tex).x), f32(textureDimensions(src_tex).y));
            var out: FSOut;
            out.color = textureSampleLevel(src_tex, src_sampler, uv, 0.0);
            return out;
        }
    "#;

        // Create shader module and blit pipeline
        let blit_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mip_blit_shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(BLIT_WGSL)),
            });

        // Create pipeline layout and render pipeline for the blit passes
        let blit_bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("blit_bind_group_layout"),
                    entries: &[
                        // src texture
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        // sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

        let blit_pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("blit_pipeline_layout"),
                    bind_group_layouts: &[&blit_bind_group_layout],
                    immediate_size: 0,
                });

        let color_format = wgpu::TextureFormat::Rgba8Unorm;

        let blit_pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("mip_blit_pipeline"),
                layout: Some(&blit_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &blit_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &blit_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                cache: None,
            });

        // For each mip level: sample mip N and render into mip N+1
        for mip in 1..mip_count {
            // create a view that references mip N-1 as the source (we will bind that as sampled texture)
            let src_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("src_mip_{}", mip - 1)),
                base_mip_level: (mip - 1) as u32,
                mip_level_count: Some(1),
                ..Default::default()
            });

            // create view for destination mip N
            let dst_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("dst_mip_{}", mip)),
                base_mip_level: mip as u32,
                mip_level_count: Some(1),
                ..Default::default()
            });

            // bind group for this blit using the src_view and the blit sampler
            let blit_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("blit_bind_group_{}", mip)),
                layout: &blit_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&blit_sampler),
                    },
                ],
            });

            // Begin a render pass that writes into the destination mip
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("mip_blit_pass_{}", mip)),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // we will overwrite the whole mip, so Clear or Load both ok. We'll clear to transparent black first.
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(&blit_pipeline);
            rpass.set_bind_group(0, &blit_bind_group, &[]);
            // draw fullscreen triangle
            rpass.draw(0..3, 0..1);
            // render pass drops here
        }

        // Finish and submit
        let command_buffer = encoder.finish();
        self.queue.submit(std::iter::once(command_buffer));

        // store the texture and a default view into cache (you might prefer to store the base view)
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.texture_cache.insert(
            key,
            CachedTexture {
                _texture: texture,
                view,
            },
        );
    }

    pub fn get_views(&mut self, materials: &[TextureCacheKey]) -> Vec<&TextureView> {
        for key in materials {
            self.ensure_texture(key);
        }

        materials
            .iter()
            .map(|key| self.get_texture_view(key))
            .collect()
    }
}
