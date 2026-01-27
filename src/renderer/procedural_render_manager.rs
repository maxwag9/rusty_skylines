use crate::data::Settings;
use crate::renderer::pipelines::Pipelines;
use crate::renderer::procedural_bind_group_manager::MaterialBindGroupManager;
use crate::renderer::procedural_texture_manager::{ProceduralTextureManager, TextureCacheKey};
use crate::terrain::roads::road_mesh_manager::RoadVertex;
use std::collections::HashMap;
use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use wgpu::*;

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
const FULLSCREEN_DEPTH_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var t_depth: texture_depth_2d;
@group(0) @binding(1) var s_depth: sampler;

// adjust these to your camera
const NEAR: f32 = 0.5;
const FAR:  f32 = 100.0;

fn linearize_depth(d: f32) -> f32 {
    let z = d * 2.0 - 1.0;
    return (2.0 * NEAR * FAR) / (FAR + NEAR - z * (FAR - NEAR));
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(idx & 1u) * 2 - 1);
    let y = f32(i32(idx >> 1u) * 2 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = textureSample(t_depth, s_depth, in.uv);

    // reversed-Z visualization
    let v = pow(d, 20.0);

    return vec4<f32>(v, v, v, 1.0);
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
    pub shadow_pass: bool,
    pub fullscreen_pass: bool,
    pub target_format: TextureFormat,
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
            shadow_pass: false,
            fullscreen_pass: false,
            target_format: TextureFormat::Rgba8UnormSrgb,
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

#[derive(Clone, Hash, PartialEq, Eq)]
struct PipelineCacheKeyRaw {
    shader_path: PathBuf,
    bind_group_layout_ptrs: Vec<usize>,
    vertex_layout_hash: u64,
    options: PipelineOptionsKey,
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
    is_depth: bool,
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct FogPipelineCacheKey {
    shader_path: PathBuf,
    msaa_samples: u32,
    depth_multisampled: bool,
    uniform_count: usize,
    target_format: TextureFormat,
}
struct ShaderEntry {
    module: wgpu::ShaderModule,
    source_path: PathBuf,
}

pub struct PipelineManager {
    device: Device,
    queue: Queue,
    surface_format: wgpu::TextureFormat,
    shader_cache: HashMap<PathBuf, ShaderEntry>,
    pipeline_cache: HashMap<PipelineCacheKey, RenderPipeline>,
    fullscreen_pipeline_cache: HashMap<FullscreenPipelineKey, RenderPipeline>,
    raw_pipeline_cache: HashMap<PipelineCacheKeyRaw, RenderPipeline>,
    uniform_bind_group_layouts: HashMap<usize, BindGroupLayout>,
    fullscreen_color_bgl: BindGroupLayout,
    fullscreen_depth_bgl: BindGroupLayout,
    fullscreen_color_shader: ShaderModule,
    fullscreen_depth_shader: ShaderModule,

    fog_pipeline_cache: HashMap<FogPipelineCacheKey, RenderPipeline>,
    fog_depth_layout: Option<BindGroupLayout>,
    fog_depth_layout_msaa: Option<BindGroupLayout>,
}

impl PipelineManager {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let fullscreen_color_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fullscreen Color Preview Shader"),
            source: wgpu::ShaderSource::Wgsl(FULLSCREEN_SHADER_SOURCE.into()),
        });
        let fullscreen_depth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fullscreen Depth Preview Shader"),
            source: wgpu::ShaderSource::Wgsl(FULLSCREEN_DEPTH_SHADER.into()),
        });
        let fullscreen_color_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Fullscreen Color BGL"),
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

        let fullscreen_depth_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Fullscreen Depth BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });

        Self {
            device,
            queue,
            surface_format,
            shader_cache: HashMap::new(),
            pipeline_cache: HashMap::new(),
            fullscreen_pipeline_cache: HashMap::new(),
            raw_pipeline_cache: HashMap::new(),
            uniform_bind_group_layouts: HashMap::new(),
            fullscreen_color_bgl,
            fullscreen_depth_bgl,
            fullscreen_color_shader,
            fullscreen_depth_shader,
            fog_pipeline_cache: Default::default(),
            fog_depth_layout: None,
            fog_depth_layout_msaa: None,
        }
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
        &self.fullscreen_depth_bgl
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

    pub fn get_or_create_pipeline(
        &mut self,
        shader_path: &Path,
        material_kinds: &Vec<TextureCacheKey>,
        material_layout: &wgpu::BindGroupLayout,
        uniform_count: usize,
        options: &PipelineOptions,
        label: &str,
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
        let vertex = VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: if options.fullscreen_pass {
                &[]
            } else {
                &options.vertex_layouts
            },
            compilation_options: PipelineCompilationOptions::default(),
        };
        let fragment = if options.shadow_pass {
            None
        } else {
            Some(FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: options.target_format,
                    blend: options.blend,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            })
        };
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(&format!("{} {} Pipeline", shader_path.display(), label)),
            layout: Some(&pipeline_layout),
            vertex,
            fragment,
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

    pub fn get_or_create_fullscreen_color_pipeline(
        &mut self,
        msaa_samples: u32,
    ) -> &wgpu::RenderPipeline {
        let key = FullscreenPipelineKey {
            msaa_samples,
            surface_format: self.surface_format,
            is_depth: false,
        };

        if self.fullscreen_pipeline_cache.get(&key).is_some() {
            return self.fullscreen_pipeline_cache.get(&key).unwrap();
        }

        let device = &self.device;

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fullscreen Color Pipeline Layout"),
            bind_group_layouts: &[&self.fullscreen_color_bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Fullscreen Color Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &self.fullscreen_color_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.fullscreen_color_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
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
            depth_stencil: None,
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
    pub fn get_or_create_fullscreen_depth_pipeline(
        &mut self,
        msaa_samples: u32,
    ) -> &wgpu::RenderPipeline {
        let key = FullscreenPipelineKey {
            msaa_samples,
            surface_format: self.surface_format,
            is_depth: true,
        };

        if self.fullscreen_pipeline_cache.get(&key).is_some() {
            return self.fullscreen_pipeline_cache.get(&key).unwrap();
        }

        let device = &self.device;

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fullscreen Depth Pipeline Layout"),
            bind_group_layouts: &[&self.fullscreen_depth_bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Fullscreen Depth Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &self.fullscreen_depth_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.fullscreen_depth_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
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
            depth_stencil: None, // IMPORTANT
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

    pub fn get_or_create_pipeline_with_layouts(
        &mut self,
        shader_path: &std::path::Path,
        bind_group_layouts: &[&wgpu::BindGroupLayout],
        options: &PipelineOptions,
        label: &str,
    ) -> &wgpu::RenderPipeline {
        let options_key: PipelineOptionsKey = options.into();

        let vertex_layouts: &[wgpu::VertexBufferLayout] = if options.fullscreen_pass {
            &[]
        } else {
            &options.vertex_layouts
        };

        let key = PipelineCacheKeyRaw {
            shader_path: shader_path.to_path_buf(),
            bind_group_layout_ptrs: bind_group_layouts
                .iter()
                .map(|l| (*l as *const wgpu::BindGroupLayout) as usize)
                .collect(),
            vertex_layout_hash: hash_vertex_layouts(vertex_layouts),
            options: options_key,
        };

        if self.raw_pipeline_cache.contains_key(&key) {
            return self.raw_pipeline_cache.get(&key).unwrap();
        }

        self.load_shader(shader_path);
        let shader = &self.shader_cache.get(shader_path).unwrap().module;

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{} RAW Pipeline Layout", shader_path.display())),
                bind_group_layouts,
                immediate_size: 0,
            });

        let vertex = wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: vertex_layouts,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        };

        let fragment = if options.shadow_pass {
            None
        } else {
            Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: options.target_format,
                    blend: options.blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            })
        };

        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{} {} RAW Pipeline", shader_path.display(), label)),
                layout: Some(&pipeline_layout),
                vertex,
                fragment,
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

        self.raw_pipeline_cache.insert(key.clone(), pipeline);
        self.raw_pipeline_cache.get(&key).unwrap()
    }

    pub fn reload_shaders(&mut self, affected_shaders: Vec<PathBuf>) {
        let paths: Vec<PathBuf> = self.shader_cache.keys().cloned().collect();

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
    fn fog_depth_bind_group_layout(&mut self, depth_multisampled: bool) -> &BindGroupLayout {
        let slot = if depth_multisampled {
            &mut self.fog_depth_layout_msaa
        } else {
            &mut self.fog_depth_layout
        };

        slot.get_or_insert_with(|| {
            self.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some(if depth_multisampled {
                        "Fog Depth BGL (MSAA)"
                    } else {
                        "Fog Depth BGL"
                    }),
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Depth,
                            view_dimension: TextureViewDimension::D2,
                            multisampled: depth_multisampled,
                        },
                        count: None,
                    }],
                })
        })
    }

    pub fn get_or_create_fog_pipeline(
        &mut self,
        shader_path: &Path,
        msaa_samples: u32,
        depth_multisampled: bool,
        uniform_count: usize,
        target_format: TextureFormat,
    ) -> &RenderPipeline {
        let key = FogPipelineCacheKey {
            shader_path: shader_path.to_path_buf(),
            msaa_samples,
            depth_multisampled,
            uniform_count,
            target_format,
        };

        if self.fog_pipeline_cache.contains_key(&key) {
            return self.fog_pipeline_cache.get(&key).unwrap();
        }

        self.load_shader(shader_path);
        self.ensure_uniform_bind_group_layout(uniform_count);

        let depth_layout = self.fog_depth_bind_group_layout(depth_multisampled).clone();
        let shader = &self.shader_cache.get(shader_path).unwrap().module;
        let uniform_layout = self.uniform_bind_group_layouts.get(&uniform_count).unwrap();

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Fog Pipeline Layout"),
                bind_group_layouts: &[&depth_layout, uniform_layout],
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("Fog Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: PipelineCompilationOptions::default(),
                },
                fragment: Some(FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(ColorTargetState {
                        format: target_format,
                        blend: Some(BlendState::ALPHA_BLENDING),
                        write_mask: ColorWrites::ALL,
                    })],
                    compilation_options: PipelineCompilationOptions::default(),
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None, // IMPORTANT: we sample depth, so do not attach it here
                multisample: MultisampleState {
                    count: msaa_samples,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                cache: None,
                multiview_mask: None,
            });

        self.fog_pipeline_cache.insert(key.clone(), pipeline);
        self.fog_pipeline_cache.get(&key).unwrap()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
fn create_fullscreen_pipeline(
    device: &Device,
    surface_format: TextureFormat,
    shader: &ShaderModule,
    bgl: &BindGroupLayout,
    msaa_samples: u32,
) -> RenderPipeline {
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Fullscreen Pipeline Layout"),
        bind_group_layouts: &[bgl],
        immediate_size: 0,
    });

    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("Fullscreen Pipeline"),
        layout: Some(&layout),
        vertex: VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleStrip,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: None,
        multisample: MultisampleState {
            count: msaa_samples,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        cache: None,
        multiview_mask: None,
    })
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
    pub pipeline_manager: PipelineManager,
    material_manager: MaterialBindGroupManager,
    procedural_textures: ProceduralTextureManager,
    fullscreen_color_sampler: Sampler,
    fullscreen_depth_sampler: Sampler,
    uniform_bind_groups: HashMap<UniformBindGroupKey, BindGroup>,
}

// impl RenderManager {
//     pub fn render_tonemap(&self, label: &str, pass: &mut RenderPass) {
//         self.pipeline_manager.
//     }
// }

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

        let fullscreen_color_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Fullscreen Preview Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let fullscreen_depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Fullscreen Depth Preview Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });
        Self {
            pipeline_manager,
            material_manager,
            procedural_textures,
            fullscreen_color_sampler,
            fullscreen_depth_sampler,
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

    pub fn render(
        &mut self,
        materials: Vec<TextureCacheKey>,
        label: &str,
        shader_path: &Path,
        options: PipelineOptions,
        uniforms: &[&Buffer],
        pass: &mut RenderPass,
        pipelines: &Pipelines,
        settings: &Settings,
    ) {
        // Fullscreen tonemap reads from resolved HDR view instead of procedural material textures
        let (views, material_count_for_layout) = if options.fullscreen_pass {
            (vec![&pipelines.resolved_hdr_view], 1usize)
        } else {
            let v = self.procedural_textures.get_views(&materials);
            (v, materials.len())
        };

        let material_layout = self
            .material_manager
            .get_layout(
                material_count_for_layout,
                options.shadow_pass,
                options.fullscreen_pass,
            )
            .clone();

        let material_bind_group = self.material_manager.request_bind_group(
            &materials,
            views,
            &pipelines.cascaded_shadow_map.array_view,
            options.shadow_pass,
            options.fullscreen_pass,
            settings,
        );

        let pipeline = self.pipeline_manager.get_or_create_pipeline(
            shader_path,
            &materials,
            &material_layout,
            uniforms.len(),
            &options,
            label,
        );

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, material_bind_group, &[]);

        if !uniforms.is_empty() {
            let key = UniformBindGroupKey::from_buffers(uniforms);
            let pm = &self.pipeline_manager;
            let label_owned = format!("{} Uniform BindGroup", label);

            let bind_group = self
                .uniform_bind_groups
                .entry(key)
                .or_insert_with(|| pm.create_uniform_bind_group(uniforms, &label_owned));

            pass.set_bind_group(1, &*bind_group, &[]);
        }
    }
    pub fn render_with_bind_groups(
        &mut self,
        label: &str,
        shader_path: &std::path::Path,
        options: PipelineOptions,
        bind_group_layouts: &[&wgpu::BindGroupLayout],
        bind_groups: &[&wgpu::BindGroup],
        pass: &mut wgpu::RenderPass,
    ) {
        let pipeline = self.pipeline_manager.get_or_create_pipeline_with_layouts(
            shader_path,
            bind_group_layouts,
            &options,
            label,
        );

        pass.set_pipeline(pipeline);

        for (i, bg) in bind_groups.iter().enumerate() {
            pass.set_bind_group(i as u32, *bg, &[]);
        }
    }
    pub fn render_fullscreen_preview(
        &mut self,
        texture: &wgpu::TextureView,
        label: &str,
        msaa_samples: u32,
        pass: &mut wgpu::RenderPass,
    ) {
        let is_depth = texture.texture().format().is_depth_stencil_format();

        let bind_group =
            self.pipeline_manager
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &if is_depth {
                        self.pipeline_manager.fullscreen_depth_bgl.clone()
                    } else {
                        self.pipeline_manager.fullscreen_color_bgl.clone()
                    },
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(texture),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(if is_depth {
                                &self.fullscreen_depth_sampler
                            } else {
                                &self.fullscreen_color_sampler
                            }),
                        },
                    ],
                });
        let pipeline = if is_depth {
            self.pipeline_manager
                .get_or_create_fullscreen_depth_pipeline(msaa_samples)
        } else {
            self.pipeline_manager
                .get_or_create_fullscreen_color_pipeline(msaa_samples)
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..4, 0..1);
    }

    pub fn render_fog_fullscreen(
        &mut self,
        label: &str,
        shader_path: &Path,
        msaa_samples: u32,
        depth_view: &TextureView,
        uniforms: &[&Buffer], // must include: uniforms, fog_uniforms, pick_uniforms
        target_format: TextureFormat,
        pass: &mut RenderPass,
    ) {
        let depth_multisampled = msaa_samples > 1;

        let depth_layout = self
            .pipeline_manager
            .fog_depth_bind_group_layout(depth_multisampled)
            .clone();
        let depth_bind_group =
            self.pipeline_manager
                .device()
                .create_bind_group(&BindGroupDescriptor {
                    label: Some(&format!("{label} Depth BindGroup")),
                    layout: &depth_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(depth_view),
                    }],
                });

        let pipeline = self.pipeline_manager.get_or_create_fog_pipeline(
            shader_path,
            msaa_samples,
            depth_multisampled,
            uniforms.len(),
            target_format,
        );

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &depth_bind_group, &[]);

        // Reuse your existing uniform bind group cache logic
        let key = UniformBindGroupKey::from_buffers(uniforms);
        let label_owned = format!("{label} Uniform BindGroup");

        let bind_group = self.uniform_bind_groups.entry(key).or_insert_with(|| {
            self.pipeline_manager
                .create_uniform_bind_group(uniforms, &label_owned)
        });

        pass.set_bind_group(1, &*bind_group, &[]);

        pass.draw(0..3, 0..1);
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

pub fn create_color_attachment_load<'a>(
    msaa_view: &'a TextureView,
    surface_view: &'a TextureView,
    msaa_samples: u32,
) -> RenderPassColorAttachment<'a> {
    if msaa_samples > 1 {
        RenderPassColorAttachment {
            view: msaa_view,
            depth_slice: None,
            resolve_target: Some(surface_view),
            ops: Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            },
        }
    } else {
        RenderPassColorAttachment {
            view: surface_view,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            },
        }
    }
}
fn hash_vertex_layouts(layouts: &[VertexBufferLayout]) -> u64 {
    let mut h = DefaultHasher::new();
    layouts.len().hash(&mut h);

    for l in layouts {
        // stride + step mode
        (l.array_stride as u64).hash(&mut h);
        (l.step_mode as u32).hash(&mut h);

        // attributes
        l.attributes.len().hash(&mut h);
        for a in l.attributes {
            (a.shader_location as u32).hash(&mut h);
            (a.offset as u64).hash(&mut h);
            (a.format as u32).hash(&mut h);
        }
    }

    h.finish()
}
