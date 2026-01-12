use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use wgpu::util::DeviceExt;
use wgpu::{Device, Queue, TextureView};

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
    view: wgpu::TextureView,
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
        let (pipeline, bind_group_layout) = {
            let entry = self.pipelines.get(&key.kind).unwrap();
            (entry.pipeline.clone(), entry.bind_group_layout.clone())
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: key.resolution,
                height: key.resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&key.params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_size = 8u32;
            let workgroups = key.resolution.div_ceil(workgroup_size);
            compute_pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

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
