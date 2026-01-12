use crate::renderer::pipelines::DEPTH_FORMAT;
use crate::renderer::procedural_bind_group_manager::MaterialBindGroupManager;
use crate::renderer::procedural_texture_manager::{
    MaterialKind, ProceduralTextureManager, TextureCacheKey,
};
use crate::terrain::roads::road_mesh_manager::RoadVertex;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use wgpu::{BindGroup, Buffer, CompareFunction, DepthStencilState, Sampler, VertexBufferLayout};

const FULLSCREEN_SHADER_SOURCE: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(idx & 1u) * 2 - 1);
    let y = f32(i32(idx >> 1u) * 2 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

@group(0) @binding(0) var t_tex: texture_2d<f32>;
@group(0) @binding(1) var s_tex: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_tex, s_tex, in.uv);
}
"#;

#[derive(Clone, Debug)]
pub struct PipelineOptions {
    pub topology: wgpu::PrimitiveTopology,
    pub msaa_samples: u32,
    pub depth_stencil: Option<wgpu::DepthStencilState>,
    pub vertex_layout: VertexBufferLayout<'static>,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            topology: wgpu::PrimitiveTopology::TriangleList,
            msaa_samples: 1,
            depth_stencil: None,
            vertex_layout: RoadVertex::layout(),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct DepthBiasStateKey {
    constant: i32,
    slope_scale_bits: u32,
    clamp_bits: u32,
}

impl From<wgpu::DepthBiasState> for DepthBiasStateKey {
    fn from(b: wgpu::DepthBiasState) -> Self {
        Self {
            constant: b.constant,
            slope_scale_bits: b.slope_scale.to_bits(),
            clamp_bits: b.clamp.to_bits(),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct StencilFaceStateKey {
    compare: wgpu::CompareFunction,
    fail_op: wgpu::StencilOperation,
    depth_fail_op: wgpu::StencilOperation,
    pass_op: wgpu::StencilOperation,
}

impl From<wgpu::StencilFaceState> for StencilFaceStateKey {
    fn from(f: wgpu::StencilFaceState) -> Self {
        Self {
            compare: f.compare,
            fail_op: f.fail_op,
            depth_fail_op: f.depth_fail_op,
            pass_op: f.pass_op,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct StencilStateKey {
    front: StencilFaceStateKey,
    back: StencilFaceStateKey,
    read_mask: u32,
    write_mask: u32,
}

impl From<wgpu::StencilState> for StencilStateKey {
    fn from(s: wgpu::StencilState) -> Self {
        Self {
            front: s.front.into(),
            back: s.back.into(),
            read_mask: s.read_mask,
            write_mask: s.write_mask,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct DepthStencilStateKey {
    format: wgpu::TextureFormat,
    depth_write_enabled: bool,
    depth_compare: wgpu::CompareFunction,
    stencil: StencilStateKey,
    bias: DepthBiasStateKey,
}

impl From<&wgpu::DepthStencilState> for DepthStencilStateKey {
    fn from(d: &wgpu::DepthStencilState) -> Self {
        Self {
            format: d.format,
            depth_write_enabled: d.depth_write_enabled,
            depth_compare: d.depth_compare,
            stencil: d.stencil.clone().into(),
            bias: d.bias.into(),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct PipelineOptionsKey {
    topology: wgpu::PrimitiveTopology,
    msaa_samples: u32,
    depth_stencil: Option<DepthStencilStateKey>,
}

impl From<&PipelineOptions> for PipelineOptionsKey {
    fn from(o: &PipelineOptions) -> Self {
        Self {
            topology: o.topology,
            msaa_samples: o.msaa_samples,
            depth_stencil: o.depth_stencil.as_ref().map(|d| d.into()),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct PipelineCacheKey {
    shader_path: PathBuf,
    material_kinds: Vec<TextureCacheKey>,
    has_uniforms: bool,
    options: PipelineOptionsKey,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct FullscreenPipelineKey {
    msaa_samples: u32,
    surface_format: wgpu::TextureFormat,
}

struct ShaderEntry {
    module: wgpu::ShaderModule,
    source_path: PathBuf,
}

pub struct PipelineManager {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_format: wgpu::TextureFormat,
    shader_cache: HashMap<PathBuf, ShaderEntry>,
    pipeline_cache: HashMap<PipelineCacheKey, wgpu::RenderPipeline>,
    fullscreen_pipeline_cache: HashMap<FullscreenPipelineKey, wgpu::RenderPipeline>,
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    fullscreen_bind_group_layout: wgpu::BindGroupLayout,
    fullscreen_shader: wgpu::ShaderModule,
}

impl PipelineManager {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Uniform BindGroup Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let fullscreen_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Fullscreen Preview BindGroup Layout"),
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

        let fullscreen_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fullscreen Preview Shader"),
            source: wgpu::ShaderSource::Wgsl(FULLSCREEN_SHADER_SOURCE.into()),
        });

        Self {
            device,
            queue,
            surface_format,
            shader_cache: HashMap::new(),
            pipeline_cache: HashMap::new(),
            fullscreen_pipeline_cache: HashMap::new(),
            uniform_bind_group_layout,
            fullscreen_bind_group_layout,
            fullscreen_shader,
        }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_format
    }

    pub fn uniform_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.uniform_bind_group_layout
    }

    pub fn fullscreen_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.fullscreen_bind_group_layout
    }

    fn load_shader(&mut self, path: &Path) {
        if !self.shader_cache.contains_key(path) {
            let source = fs::read_to_string(path).unwrap_or_else(|e| {
                panic!("Failed to load shader at {:?}: {}", path, e);
            });
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(path.to_str().unwrap_or("Shader")),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            self.shader_cache.insert(
                path.to_path_buf(),
                ShaderEntry {
                    module,
                    source_path: path.to_path_buf(),
                },
            );
        }
    }

    pub fn get_or_create_pipeline(
        &mut self,
        shader_path: &Path,
        material_kinds: &Vec<TextureCacheKey>,
        material_layout: &wgpu::BindGroupLayout,
        has_uniforms: bool,
        options: &PipelineOptions,
    ) -> &wgpu::RenderPipeline {
        let cache_key = PipelineCacheKey {
            shader_path: shader_path.to_path_buf(),
            material_kinds: material_kinds.clone(),
            has_uniforms,
            options: options.into(),
        };

        if self.pipeline_cache.contains_key(&cache_key) {
            return self.pipeline_cache.get(&cache_key).unwrap();
        }

        self.load_shader(shader_path);

        let shader = &self.shader_cache.get(shader_path).unwrap().module;
        let uniform_layout = &self.uniform_bind_group_layout;
        let device = &self.device;
        let surface_format = self.surface_format;

        let mut bind_group_layouts: Vec<&wgpu::BindGroupLayout> = vec![material_layout];
        if has_uniforms {
            bind_group_layouts.push(uniform_layout);
        }

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{} Pipeline Layout", shader_path.display())),
            bind_group_layouts: &bind_group_layouts,
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("{} Pipeline", shader_path.display())),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[options.vertex_layout.clone()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: options.topology,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: options.depth_stencil.clone(),
            multisample: wgpu::MultisampleState {
                count: options.msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            cache: None,
            multiview_mask: None,
        });

        self.pipeline_cache.insert(cache_key.clone(), pipeline);
        self.pipeline_cache.get(&cache_key).unwrap()
    }

    pub fn get_or_create_fullscreen_pipeline(
        &mut self,
        msaa_samples: u32,
    ) -> &wgpu::RenderPipeline {
        let key = FullscreenPipelineKey {
            msaa_samples,
            surface_format: self.surface_format,
        };

        if self.fullscreen_pipeline_cache.contains_key(&key) {
            return self.fullscreen_pipeline_cache.get(&key).unwrap();
        }

        let device = &self.device;
        let surface_format = self.surface_format;
        let fullscreen_shader = &self.fullscreen_shader;
        let fullscreen_bind_group_layout = &self.fullscreen_bind_group_layout;

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fullscreen Preview Pipeline Layout"),
            bind_group_layouts: &[fullscreen_bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Fullscreen Preview Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: fullscreen_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: fullscreen_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            cache: None,
            multiview_mask: None,
        });

        self.fullscreen_pipeline_cache.insert(key.clone(), pipeline);
        self.fullscreen_pipeline_cache.get(&key).unwrap()
    }

    pub fn reload_all_shaders(&mut self) {
        let paths: Vec<PathBuf> = self.shader_cache.keys().cloned().collect();

        self.shader_cache.clear();

        for path in &paths {
            self.load_shader(path);
        }

        self.pipeline_cache.clear();
    }

    pub fn create_uniform_bind_group(&self, buffer: &wgpu::Buffer, label: &str) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        })
    }
}

pub struct RenderManager {
    pipeline_manager: PipelineManager,
    material_manager: MaterialBindGroupManager,
    procedural_textures: ProceduralTextureManager,
    fullscreen_sampler: Sampler,
    uniform_bind_groups: HashMap<*const Buffer, BindGroup>,
}

impl RenderManager {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target_format: wgpu::TextureFormat, // renamed for clarity
        shader_dir: PathBuf,
    ) -> Self {
        let procedural_textures =
            ProceduralTextureManager::new(device.clone(), queue.clone(), shader_dir);
        let material_manager = MaterialBindGroupManager::new(device.clone());
        let pipeline_manager = PipelineManager::new(device.clone(), queue.clone(), target_format);

        let fullscreen_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Fullscreen Preview Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self {
            pipeline_manager,
            material_manager,
            procedural_textures,
            fullscreen_sampler,
            uniform_bind_groups: HashMap::new(),
        }
    }

    // Accessors
    pub fn device(&self) -> &wgpu::Device {
        self.pipeline_manager.device()
    }

    pub fn queue(&self) -> &wgpu::Queue {
        self.pipeline_manager.queue()
    }

    pub fn pipeline_manager(&self) -> &PipelineManager {
        &self.pipeline_manager
    }

    pub fn pipeline_manager_mut(&mut self) -> &mut PipelineManager {
        &mut self.pipeline_manager
    }
    pub fn procedural_texture_manager_mut(&mut self) -> &mut ProceduralTextureManager {
        &mut self.procedural_textures
    }
    pub fn reload_all_shaders(&mut self) {
        self.pipeline_manager.reload_all_shaders();
    }

    // Rendering (use cached bind groups)
    pub fn render(
        &mut self,
        materials: Vec<TextureCacheKey>,
        label: &str,
        shader_path: &Path,
        options: PipelineOptions,
        uniforms: Option<&Buffer>,
        pass: &mut wgpu::RenderPass,
    ) {
        let views = self.procedural_textures.get_views(&materials);
        let material_layout = &self.material_manager.get_layout(materials.len()).clone();
        let material_bind_group = self.material_manager.request_bind_group(&materials, views);

        let pipeline = self.pipeline_manager.get_or_create_pipeline(
            shader_path,
            &materials,
            material_layout,
            uniforms.is_some(),
            &options,
        );

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, material_bind_group, &[]);

        if let Some(uniform_buffer) = uniforms {
            let key = uniform_buffer as *const wgpu::Buffer;
            let bind_group = self.uniform_bind_groups.entry(key).or_insert_with(|| {
                self.pipeline_manager.create_uniform_bind_group(
                    uniform_buffer,
                    &format!("{} Uniform BindGroup", label),
                )
            });
            pass.set_bind_group(1, &*bind_group, &[]);
        }
    }

    pub fn render_fullscreen_preview(
        &mut self,
        texture: &wgpu::TextureView,
        label: &str,
        msaa_samples: u32,
        pass: &mut wgpu::RenderPass,
    ) {
        let device = self.pipeline_manager.device();
        let layout = self.pipeline_manager.fullscreen_bind_group_layout();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} BindGroup", label)),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.fullscreen_sampler),
                },
            ],
        });

        let pipeline = self
            .pipeline_manager
            .get_or_create_fullscreen_pipeline(msaa_samples);

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..4, 0..1);
    }

    pub fn create_uniform_buffer<T: bytemuck::Pod>(&self, data: &T, label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.pipeline_manager
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&[*data]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    pub fn update_uniform_buffer<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer, data: &T) {
        self.pipeline_manager
            .queue()
            .write_buffer(buffer, 0, bytemuck::cast_slice(&[*data]));
    }
}

pub struct SimpleMaterialBindGroupManager {
    device: wgpu::Device,
    layout: wgpu::BindGroupLayout,
    bind_groups: HashMap<Vec<MaterialKind>, wgpu::BindGroup>,
    textures: HashMap<MaterialKind, wgpu::TextureView>,
    sampler: wgpu::Sampler,
}

impl SimpleMaterialBindGroupManager {
    pub fn new(device: wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Material BindGroup Layout"),
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

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Material Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        Self {
            device,
            layout,
            bind_groups: HashMap::new(),
            textures: HashMap::new(),
            sampler,
        }
    }

    pub fn register_texture(&mut self, kind: MaterialKind, view: wgpu::TextureView) {
        self.textures.insert(kind, view);
        self.bind_groups.clear();
    }

    pub fn ensure_bind_group(&mut self, materials: &[MaterialKind]) {
        let key = materials.to_vec();
        if self.bind_groups.contains_key(&key) {
            return;
        }

        if materials.is_empty() {
            return;
        }

        let first_material = materials[0];
        let texture = self
            .textures
            .get(&first_material)
            .expect("Texture not registered");

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{:?} BindGroup", materials)),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        self.bind_groups.insert(key, bind_group);
    }
}

pub struct MultiTextureMaterialBindGroupManager {
    device: wgpu::Device,
    layouts: HashMap<usize, wgpu::BindGroupLayout>,
    bind_groups: HashMap<Vec<MaterialKind>, wgpu::BindGroup>,
    textures: HashMap<MaterialKind, wgpu::TextureView>,
    sampler: wgpu::Sampler,
}

impl MultiTextureMaterialBindGroupManager {
    pub fn new(device: wgpu::Device) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Multi-Material Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        Self {
            device,
            layouts: HashMap::new(),
            bind_groups: HashMap::new(),
            textures: HashMap::new(),
            sampler,
        }
    }

    pub fn register_texture(&mut self, kind: MaterialKind, view: wgpu::TextureView) {
        self.textures.insert(kind, view);
        self.bind_groups.clear();
    }

    pub fn ensure_bind_group(&mut self, materials: &[MaterialKind]) {
        let key = materials.to_vec();
        if self.bind_groups.contains_key(&key) {
            return;
        }

        let texture_count = materials.len();
        if texture_count == 0 {
            return;
        }

        if !self.layouts.contains_key(&texture_count) {
            let mut entries = Vec::with_capacity(texture_count + 1);

            for i in 0..texture_count {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                });
            }

            entries.push(wgpu::BindGroupLayoutEntry {
                binding: texture_count as u32,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });

            let layout = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("Material BindGroup Layout ({})", texture_count)),
                    entries: &entries,
                });

            self.layouts.insert(texture_count, layout);
        }

        let layout = self.layouts.get(&texture_count).unwrap();
        let device = &self.device;
        let sampler = &self.sampler;

        let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(texture_count + 1);

        for (i, material) in materials.iter().enumerate() {
            let texture = self
                .textures
                .get(material)
                .unwrap_or_else(|| panic!("Texture not registered for {:?}", material));
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::TextureView(texture),
            });
        }

        entries.push(wgpu::BindGroupEntry {
            binding: texture_count as u32,
            resource: wgpu::BindingResource::Sampler(sampler),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{:?} BindGroup", materials)),
            layout,
            entries: &entries,
        });

        self.bind_groups.insert(key, bind_group);
    }
}
