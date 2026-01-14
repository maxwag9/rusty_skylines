use crate::renderer::pipelines::DEPTH_FORMAT;
use crate::renderer::procedural_bind_group_manager::MaterialBindGroupManager;
use crate::renderer::procedural_texture_manager::{ProceduralTextureManager, TextureCacheKey};
use crate::terrain::roads::road_mesh_manager::RoadVertex;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use wgpu::{
    BindGroup, BlendState, Buffer, CompareFunction, DepthStencilState, Device, Face, Queue,
    Sampler, VertexBufferLayout,
};

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
    pub vertex_layouts: Vec<VertexBufferLayout<'static>>,
    pub blend: Option<BlendState>,
    pub cull_mode: Option<Face>,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            topology: wgpu::PrimitiveTopology::TriangleList,
            msaa_samples: 1,
            depth_stencil: None,
            vertex_layouts: Vec::from([RoadVertex::layout()]),
            blend: None,
            cull_mode: None,
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
    compare: CompareFunction,
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
    uniform_count: usize,
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
    uniform_bind_group_layouts: HashMap<usize, wgpu::BindGroupLayout>,
    fullscreen_bind_group_layout: wgpu::BindGroupLayout,
    fullscreen_shader: wgpu::ShaderModule,
}

impl PipelineManager {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
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
            uniform_bind_group_layouts: HashMap::new(),
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

    pub fn uniform_bind_group_layout(&mut self, count: usize) -> &wgpu::BindGroupLayout {
        self.ensure_uniform_bind_group_layout(count);
        self.uniform_bind_group_layouts.get(&count).unwrap()
    }

    pub fn fullscreen_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.fullscreen_bind_group_layout
    }

    fn ensure_uniform_bind_group_layout(&mut self, count: usize) {
        if count == 0 || self.uniform_bind_group_layouts.contains_key(&count) {
            return;
        }

        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..count)
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();

        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("Uniform BindGroup Layout (count: {})", count)),
                entries: &entries,
            });
        self.uniform_bind_group_layouts.insert(count, layout);
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
        uniform_count: usize,
        options: &PipelineOptions,
    ) -> &wgpu::RenderPipeline {
        let cache_key = PipelineCacheKey {
            shader_path: shader_path.to_path_buf(),
            material_kinds: material_kinds.clone(),
            uniform_count,
            options: options.into(),
        };

        if self.pipeline_cache.contains_key(&cache_key) {
            return self.pipeline_cache.get(&cache_key).unwrap();
        }

        self.load_shader(shader_path);
        self.ensure_uniform_bind_group_layout(uniform_count);

        let shader = &self.shader_cache.get(shader_path).unwrap().module;
        let device = &self.device;
        let surface_format = self.surface_format;

        let mut bind_group_layouts: Vec<&wgpu::BindGroupLayout> = vec![material_layout];
        if uniform_count > 0 {
            bind_group_layouts.push(self.uniform_bind_group_layouts.get(&uniform_count).unwrap());
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
                buffers: &options.vertex_layouts.clone(),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: options.blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: options.topology,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: options.cull_mode,
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

    pub fn create_uniform_bind_group(
        &self,
        buffers: &[&wgpu::Buffer],
        label: &str,
    ) -> wgpu::BindGroup {
        let count = buffers.len();
        let layout = self
            .uniform_bind_group_layouts
            .get(&count)
            .unwrap_or_else(|| {
                panic!(
                    "Uniform layout for {} buffers not found. Pipeline must be created first.",
                    count
                )
            });

        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();

        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &entries,
        })
    }
}

/// Cache key for uniform bind groups using buffer addresses
#[derive(Clone, Hash, PartialEq, Eq)]
struct UniformBindGroupKey(Vec<usize>);

impl UniformBindGroupKey {
    fn from_buffers(buffers: &[&Buffer]) -> Self {
        Self(
            buffers
                .iter()
                .map(|b| *b as *const Buffer as usize)
                .collect(),
        )
    }
}

pub struct RenderManager {
    pipeline_manager: PipelineManager,
    material_manager: MaterialBindGroupManager,
    procedural_textures: ProceduralTextureManager,
    fullscreen_sampler: Sampler,
    uniform_bind_groups: HashMap<UniformBindGroupKey, BindGroup>,
}

impl RenderManager {
    pub fn new(
        device: &Device,
        queue: &Queue,
        target_format: wgpu::TextureFormat,
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

    pub fn render(
        &mut self,
        materials: Vec<TextureCacheKey>,
        label: &str,
        shader_path: &Path,
        options: PipelineOptions,
        uniforms: &[&Buffer],
        pass: &mut wgpu::RenderPass,
    ) {
        let views = self.procedural_textures.get_views(&materials);
        let material_layout = self.material_manager.get_layout(materials.len()).clone();
        let material_bind_group = self.material_manager.request_bind_group(&materials, views);

        let pipeline = self.pipeline_manager.get_or_create_pipeline(
            shader_path,
            &materials,
            &material_layout,
            uniforms.len(),
            &options,
        );

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, material_bind_group, &[]);

        if !uniforms.is_empty() {
            let key = UniformBindGroupKey::from_buffers(uniforms);

            // Extract references before entry() to enable disjoint field borrowing
            let pm = &self.pipeline_manager;
            let label_owned = format!("{} Uniform BindGroup", label);

            let bind_group = self
                .uniform_bind_groups
                .entry(key)
                .or_insert_with(|| pm.create_uniform_bind_group(uniforms, &label_owned));

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
