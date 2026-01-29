use crate::data::Settings;
use crate::renderer::pipelines::Pipelines;
use crate::renderer::procedural_bind_group_manager::{
    FullscreenPassType, MaterialBindGroupManager,
};
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

struct DepthDebugParams {
    near: f32,
    far: f32,
    power: f32,
    reversed_z: u32,
    msaa_samples: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var t_depth: texture_depth_2d;
@group(0) @binding(1) var s_depth: sampler;
@group(1) @binding(0) var<uniform> u: DepthDebugParams;

// wgpu/WebGPU depth is 0..1. This is the D3D/Vulkan linearization.
fn linearize_depth_01(d01: f32, near: f32, far: f32) -> f32 {
    return (near * far) / (far - d01 * (far - near));
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
    var d = textureSample(t_depth, s_depth, in.uv);

    // If reversed-z (near=1, far=0), flip for linearization:
    if (u.reversed_z != 0u) {
        d = 1.0 - d;
    }

    let z = linearize_depth_01(d, u.near, u.far);
    // Normalize and make “near bright” like your pow(d,20) did:
    let v01 = 1.0 - clamp(z / u.far, 0.0, 1.0);
    let v = pow(v01, u.power);

    return vec4<f32>(v, v, v, 1.0);
}
"#;
const FULLSCREEN_SHADER_SOURCE_UNFILTERABLE: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );

    var uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
    );

    var out: VsOut;
    out.pos = vec4<f32>(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Unfilterable float textures must be sampled with explicit LOD
    return textureSampleLevel(tex, samp, in.uv, 0.0);
}
"#;
pub const FULLSCREEN_DEPTH_MSAA_SHADER: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct DepthDebugParams {
    near: f32,
    far: f32,
    power: f32,
    reversed_z: u32,
    msaa_samples: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var depth_tex: texture_depth_multisampled_2d;
@group(1) @binding(0) var<uniform> u: DepthDebugParams;

fn linearize_depth_01(d01: f32, near: f32, far: f32) -> f32 {
    return (near * far) / (far - d01 * (far - near));
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );

    var uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
    );

    var out: VsOut;
    out.pos = vec4<f32>(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let dims_u: vec2<u32> = textureDimensions(depth_tex);
    let dims: vec2<f32> = vec2<f32>(f32(dims_u.x), f32(dims_u.y));

    var p = vec2<i32>(
        i32(in.uv.x * dims.x),
        i32(in.uv.y * dims.y),
    );
    p.x = clamp(p.x, 0, i32(dims_u.x) - 1);
    p.y = clamp(p.y, 0, i32(dims_u.y) - 1);

    var z_min: f32 = 1e30;

    // NOTE: u.msaa_samples must match the actual texture sample count.
    for (var i: u32 = 0u; i < u.msaa_samples; i = i + 1u) {
        var d = textureLoad(depth_tex, p, i);
        if (u.reversed_z != 0u) {
            d = 1.0 - d;
        }
        let z = linearize_depth_01(d, u.near, u.far);
        z_min = min(z_min, z);
    }

    let v01 = 1.0 - clamp(z_min / u.far, 0.0, 1.0);
    let v = pow(v01, u.power);
    return vec4<f32>(v, v, v, 1.0);
}
"#;
pub const FULLSCREEN_COLOR_MSAA_SHADER: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );

    var uvs = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
    );

    var out: VsOut;
    out.pos = vec4<f32>(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

@group(0) @binding(0) var tex_msaa: texture_multisampled_2d<f32>;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let dims_u: vec2<u32> = textureDimensions(tex_msaa);
    let dims: vec2<f32> = vec2<f32>(f32(dims_u.x), f32(dims_u.y));

    var p = vec2<i32>(
        i32(in.uv.x * dims.x),
        i32(in.uv.y * dims.y),
    );
    p.x = clamp(p.x, 0, i32(dims_u.x) - 1);
    p.y = clamp(p.y, 0, i32(dims_u.y) - 1);

    // show sample 0
    return textureLoad(tex_msaa, p, 0);
}
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DepthDebugParams {
    pub near: f32,
    pub far: f32,
    pub power: f32,        // e.g. 20.0
    pub reversed_z: u32,   // 0 or 1
    pub msaa_samples: u32, // 1,2,4,8...
    pub _pad0: u32,
    pub _pad1: u32,
}
#[derive(Clone, Debug)]
pub struct PipelineOptions {
    pub topology: wgpu::PrimitiveTopology,
    pub msaa_samples: u32,
    pub depth_stencil: Option<wgpu::DepthStencilState>,
    pub vertex_layouts: Vec<VertexBufferLayout<'static>>,
    pub cull_mode: Option<Face>,
    pub shadow_pass: bool,
    pub fullscreen_pass: FullscreenPassType,
    pub targets: Vec<Option<ColorTargetState>>,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            topology: wgpu::PrimitiveTopology::TriangleList,
            msaa_samples: 1,
            depth_stencil: None,
            vertex_layouts: Vec::from([RoadVertex::layout()]),
            cull_mode: None,
            shadow_pass: false,
            fullscreen_pass: FullscreenPassType::None,
            targets: vec![],
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct FullscreenPipelineKey {
    msaa_samples: u32, // output render target MSAA
    target_format: TextureFormat,
    kind: FullscreenDebugKind, // based on source format class
    src_multisampled: bool,    // NEW: source texture view samples > 1
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
    fullscreen_color_msaa_bgl: wgpu::BindGroupLayout,
    fullscreen_depth_bgl: BindGroupLayout,
    fullscreen_depth_msaa_bgl: BindGroupLayout,

    fullscreen_color_shader: ShaderModule,
    fullscreen_color_msaa_shader: ShaderModule,
    fullscreen_depth_shader: ShaderModule,
    fullscreen_depth_msaa_shader: ShaderModule,

    fog_pipeline_cache: HashMap<FogPipelineCacheKey, RenderPipeline>,
    fog_depth_layout: Option<BindGroupLayout>,
    fog_depth_layout_msaa: Option<BindGroupLayout>,
    fullscreen_color_unfilterable_bgl: wgpu::BindGroupLayout,
    fullscreen_color_unfilterable_shader: wgpu::ShaderModule,
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
        let fullscreen_color_unfilterable_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Fullscreen Color Preview Shader (Unfilterable Float)"),
                source: wgpu::ShaderSource::Wgsl(FULLSCREEN_SHADER_SOURCE_UNFILTERABLE.into()),
            });

        let fullscreen_color_unfilterable_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Fullscreen Color BGL (Unfilterable Float)"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
        let fullscreen_depth_msaa_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Fullscreen Depth MSAA Preview Shader"),
                source: wgpu::ShaderSource::Wgsl(FULLSCREEN_DEPTH_MSAA_SHADER.into()),
            });

        let fullscreen_depth_msaa_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Fullscreen Depth MSAA BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: true, // IMPORTANT
                    },
                    count: None,
                }],
            });
        let fullscreen_color_msaa_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Fullscreen Color MSAA Preview Shader"),
                source: wgpu::ShaderSource::Wgsl(FULLSCREEN_COLOR_MSAA_SHADER.into()),
            });

        let fullscreen_color_msaa_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Fullscreen Color MSAA BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: true,
                    },
                    count: None,
                }],
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
            fullscreen_color_msaa_bgl,
            fullscreen_depth_bgl,
            fullscreen_color_shader,
            fullscreen_color_msaa_shader,
            fullscreen_depth_shader,
            fog_pipeline_cache: Default::default(),
            fog_depth_layout: None,
            fog_depth_layout_msaa: None,
            fullscreen_color_unfilterable_bgl,
            fullscreen_color_unfilterable_shader,
            fullscreen_depth_msaa_bgl,
            fullscreen_depth_msaa_shader,
        }
    }
    pub fn fullscreen_debug_bgl(
        &self,
        kind: FullscreenDebugKind,
        src_multisampled: bool,
    ) -> &BindGroupLayout {
        match (kind, src_multisampled) {
            (FullscreenDebugKind::Depth, false) => &self.fullscreen_depth_bgl,
            (FullscreenDebugKind::Depth, true) => &self.fullscreen_depth_msaa_bgl,

            // For MSAA color, filterable/unfilterable doesn't matter (textureLoad).
            (FullscreenDebugKind::FloatFilterable, false) => &self.fullscreen_color_bgl,
            (FullscreenDebugKind::FloatUnfilterable, false) => {
                &self.fullscreen_color_unfilterable_bgl
            }
            (FullscreenDebugKind::FloatFilterable, true) => &self.fullscreen_color_msaa_bgl,
            (FullscreenDebugKind::FloatUnfilterable, true) => &self.fullscreen_color_msaa_bgl,
        }
    }

    pub fn get_or_create_fullscreen_debug_pipeline(
        &mut self,
        kind: FullscreenDebugKind,
        src_multisampled: bool,
        msaa_samples: u32, // OUTPUT target MSAA
        target_format: wgpu::TextureFormat,
    ) -> &wgpu::RenderPipeline {
        let key = FullscreenPipelineKey {
            msaa_samples,
            target_format,
            kind,
            src_multisampled,
        };

        // 1) Fast path: already cached (only immutably borrows the cache for this statement)
        if self.fullscreen_pipeline_cache.contains_key(&key) {
            return self.fullscreen_pipeline_cache.get(&key).unwrap();
        }

        // 2) Ensure the depth params uniform layout exists (DO NOT store the returned &BGL)
        if kind == FullscreenDebugKind::Depth {
            let _ = self.uniform_bind_group_layout(1);
        }

        // 3) Create pipeline using only short-lived immutable borrows (in this block)
        let pipeline = {
            // Pick shader + BGL0
            let (shader, bgl0, label): (&wgpu::ShaderModule, &wgpu::BindGroupLayout, &str) =
                match (kind, src_multisampled) {
                    (FullscreenDebugKind::Depth, false) => (
                        &self.fullscreen_depth_shader,
                        &self.fullscreen_depth_bgl,
                        "Fullscreen Depth Debug Pipeline",
                    ),
                    (FullscreenDebugKind::Depth, true) => (
                        &self.fullscreen_depth_msaa_shader,
                        &self.fullscreen_depth_msaa_bgl,
                        "Fullscreen Depth MSAA Debug Pipeline",
                    ),

                    (FullscreenDebugKind::FloatFilterable, false) => (
                        &self.fullscreen_color_shader,
                        &self.fullscreen_color_bgl,
                        "Fullscreen Color Debug Pipeline (Filterable Float)",
                    ),
                    (FullscreenDebugKind::FloatUnfilterable, false) => (
                        &self.fullscreen_color_unfilterable_shader,
                        &self.fullscreen_color_unfilterable_bgl,
                        "Fullscreen Color Debug Pipeline (Unfilterable Float)",
                    ),

                    // MSAA color uses textureLoad; one MSAA color shader/BGL is enough.
                    (FullscreenDebugKind::FloatFilterable, true)
                    | (FullscreenDebugKind::FloatUnfilterable, true) => (
                        &self.fullscreen_color_msaa_shader,
                        &self.fullscreen_color_msaa_bgl,
                        "Fullscreen Color MSAA Debug Pipeline",
                    ),
                };

            // Build pipeline layout (depth gets group(1) params)
            let pipeline_layout = if kind == FullscreenDebugKind::Depth {
                let ubgl = self
                    .uniform_bind_group_layouts
                    .get(&1)
                    .expect("uniform_bind_group_layout(1) must have been ensured above");

                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(&format!("{label} Layout")),
                        bind_group_layouts: &[bgl0, ubgl],
                        immediate_size: 0,
                    })
            } else {
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(&format!("{label} Layout")),
                        bind_group_layouts: &[bgl0],
                        immediate_size: 0,
                    })
            };

            self.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(&format!("{label} -> {target_format:?}")),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: target_format,
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
                })
        };

        // 4) Now we can mutably borrow the cache and insert (no outstanding borrows now)
        self.fullscreen_pipeline_cache.insert(key, pipeline);

        self.fullscreen_pipeline_cache.get(&key).unwrap()
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
        let vertex_layouts: &[VertexBufferLayout] =
            if options.fullscreen_pass == FullscreenPassType::None {
                &options.vertex_layouts
            } else {
                &[]
            };
        let vertex = VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: vertex_layouts,

            compilation_options: PipelineCompilationOptions::default(),
        };
        let fragment = if options.shadow_pass {
            None
        } else {
            Some(FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: options.targets.as_slice(),
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
                polygon_mode: PolygonMode::Fill,
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

    pub fn get_or_create_pipeline_with_layouts(
        &mut self,
        shader_path: &std::path::Path,
        bind_group_layouts: &[&wgpu::BindGroupLayout],
        options: &PipelineOptions,
        label: &str,
    ) -> &wgpu::RenderPipeline {
        let options_key: PipelineOptionsKey = options.into();

        let vertex_layouts: &[VertexBufferLayout] =
            if options.fullscreen_pass == FullscreenPassType::None {
                &options.vertex_layouts
            } else {
                &[]
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
                targets: options.targets.as_slice(),
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FullscreenDebugKind {
    Depth,
    FloatFilterable,
    FloatUnfilterable,
}

impl FullscreenDebugKind {
    fn from_format(format: wgpu::TextureFormat) -> Self {
        if format.is_depth_stencil_format() {
            return Self::Depth;
        }

        // 32-bit float formats are unfilterable in WebGPU/wgpu.
        // (This is the main practical reason to branch by format here.)
        match format {
            wgpu::TextureFormat::R32Float
            | wgpu::TextureFormat::Rg32Float
            | wgpu::TextureFormat::Rgba32Float => Self::FloatUnfilterable,
            _ => Self::FloatFilterable,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct FullscreenDebugBindGroupKey {
    view_ptr: usize,
    format: TextureFormat,
    src_multisampled: bool,
}
pub struct RenderManager {
    pub pipeline_manager: PipelineManager,
    fullscreen_debug_bind_groups: HashMap<FullscreenDebugBindGroupKey, BindGroup>,
    material_manager: MaterialBindGroupManager,
    procedural_textures: ProceduralTextureManager,
    fullscreen_color_sampler: Sampler,
    fullscreen_depth_sampler: Sampler,
    uniform_bind_groups: HashMap<UniformBindGroupKey, BindGroup>,
    depth_debug_params_buffer: Option<Buffer>,
    depth_debug_params_bind_group: Option<BindGroup>,
}

impl RenderManager {
    pub fn new(
        device: &Device,
        queue: &Queue,
        target_format: TextureFormat,
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
            fullscreen_debug_bind_groups: Default::default(),
            material_manager,
            procedural_textures,
            fullscreen_color_sampler,
            fullscreen_depth_sampler,
            uniform_bind_groups: HashMap::new(),
            depth_debug_params_buffer: None,
            depth_debug_params_bind_group: None,
        }
    }
    pub fn invalidate_resize_bind_groups(&mut self) {
        self.fullscreen_debug_bind_groups.clear();
        self.material_manager.clear_bind_groups()
    }
    pub fn device(&self) -> &Device {
        self.pipeline_manager.device()
    }

    pub fn queue(&self) -> &Queue {
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
        let (views, material_count_for_layout) = match options.fullscreen_pass {
            FullscreenPassType::None => {
                let v = self.procedural_textures.get_views(&materials);
                (v, materials.len())
            }
            FullscreenPassType::Normal => (vec![&pipelines.resolved_hdr_view], 1usize),
            FullscreenPassType::Fog => (vec![&pipelines.depth_sample_view], 1usize),
        };

        let material_layout = self
            .material_manager
            .get_layout(
                material_count_for_layout,
                options.shadow_pass,
                options.fullscreen_pass,
                options.msaa_samples,
            )
            .clone();

        let material_bind_group = self.material_manager.request_bind_group(
            &materials,
            views,
            &pipelines.cascaded_shadow_map.array_view,
            options.shadow_pass,
            options.fullscreen_pass,
            options.msaa_samples,
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

        if let Some(bg) = self.get_or_create_uniform_bind_group(label, uniforms) {
            pass.set_bind_group(1, bg, &[]);
        }
    }
    pub fn render_with_bind_groups(
        &mut self,
        label: &str,
        shader_path: &std::path::Path,
        options: PipelineOptions,
        bind_group_layouts: &[&wgpu::BindGroupLayout],
        bind_groups: &[&wgpu::BindGroup],
        pass: &mut RenderPass,
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

    pub fn render_fullscreen_debug_view(
        &mut self,
        texture: &TextureView,
        label: &str,
        target_format: TextureFormat,
        target_msaa_samples: u32,
        pass: &mut RenderPass,
    ) {
        let src_format = texture.texture().format();
        let kind = FullscreenDebugKind::from_format(src_format);
        let src_multisampled = texture.texture().sample_count() > 1;

        // --- BG0 cache key ---
        let key = FullscreenDebugBindGroupKey {
            view_ptr: (texture as *const wgpu::TextureView) as usize,
            format: src_format,
            src_multisampled,
        };

        // Choose the correct BGL for group(0) based on kind + src_multisampled.
        let bgl0 = self
            .pipeline_manager
            .fullscreen_debug_bgl(kind, src_multisampled);

        // Create/cached bind group(0). MSAA paths have no sampler.
        let bg0 = self
            .fullscreen_debug_bind_groups
            .entry(key)
            .or_insert_with(|| {
                let entries: Vec<wgpu::BindGroupEntry> = match (kind, src_multisampled) {
                    // MSAA: textureLoad path, no sampler
                    (FullscreenDebugKind::Depth, true) => vec![wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(texture),
                    }],
                    (FullscreenDebugKind::FloatFilterable, true)
                    | (FullscreenDebugKind::FloatUnfilterable, true) => {
                        vec![wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(texture),
                        }]
                    }

                    // Non-MSAA depth: depth texture + non-filtering sampler
                    (FullscreenDebugKind::Depth, false) => vec![
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(texture),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                &self.fullscreen_depth_sampler,
                            ),
                        },
                    ],

                    // Non-MSAA color filterable: texture + filtering sampler
                    (FullscreenDebugKind::FloatFilterable, false) => vec![
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(texture),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                &self.fullscreen_color_sampler,
                            ),
                        },
                    ],

                    // Non-MSAA float unfilterable: texture + non-filtering sampler (and shader uses SampleLevel)
                    (FullscreenDebugKind::FloatUnfilterable, false) => vec![
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(texture),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(
                                &self.fullscreen_depth_sampler,
                            ),
                        },
                    ],
                };

                self.pipeline_manager
                    .device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(label),
                        layout: bgl0,
                        entries: &entries,
                    })
            });

        // Create/select pipeline (depends on source MSAA + kind, and output target settings)
        let pipeline = self
            .pipeline_manager
            .get_or_create_fullscreen_debug_pipeline(
                kind,
                src_multisampled,
                target_msaa_samples,
                target_format,
            );

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &*bg0, &[]);

        // Depth debug uses group(1) params
        if kind == FullscreenDebugKind::Depth {
            let bg1 = self.depth_debug_params_bind_group.as_ref().unwrap_or_else(|| {
                panic!(
                    "Depth debug requires depth params. Call render_manager.update_depth_params_buffer(...) before render_fullscreen_debug_view()."
                )
            });
            pass.set_bind_group(1, bg1, &[]);
        }

        pass.draw(0..4, 0..1);
    }

    pub fn render_fog_fullscreen(
        &mut self,
        label: &str,
        shader_path: &Path,
        msaa_samples: u32,
        depth_view: &TextureView,
        uniforms: &[&Buffer],
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

        if let Some(bg) = self.get_or_create_uniform_bind_group(label, uniforms) {
            pass.set_bind_group(1, bg, &[]);
        }
        pass.draw(0..3, 0..1);
    }
    pub fn render_fullscreen_pass(
        &mut self,
        label: &str,
        shader_path: &Path,
        mut options: PipelineOptions,
        uniforms: &[&wgpu::Buffer],
        pass: &mut wgpu::RenderPass,
        pipelines: &Pipelines,
        settings: &Settings,
        fullscreen_pass_type: FullscreenPassType,
    ) {
        options.fullscreen_pass = fullscreen_pass_type;

        // Materials ignored for fullscreen passes!!
        self.render(
            Vec::new(),
            label,
            shader_path,
            options,
            uniforms,
            pass,
            pipelines,
            settings,
        );

        // fullscreen triangulation (strangulation)
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
    fn get_or_create_uniform_bind_group<'a>(
        &'a mut self,
        label: &str,
        uniforms: &[&wgpu::Buffer],
    ) -> Option<&'a wgpu::BindGroup> {
        if uniforms.is_empty() {
            return None;
        }

        let key = UniformBindGroupKey::from_buffers(uniforms);
        let label_owned = format!("{label} Uniform BindGroup");
        let pm = &self.pipeline_manager;

        Some(
            self.uniform_bind_groups
                .entry(key)
                .or_insert_with(|| pm.create_uniform_bind_group(uniforms, &label_owned)),
        )
    }
    pub fn update_depth_params_buffer(&mut self, params: DepthDebugParams) {
        // Ensure the uniform BGL for 1 buffer exists (PipelineManager will create/cache it).
        // This avoids create_uniform_bind_group() panicking.
        self.pipeline_manager.uniform_bind_group_layout(1);

        // Create buffer once, then just update it.
        if self.depth_debug_params_buffer.is_none() {
            use wgpu::util::DeviceExt;

            let buf = self.pipeline_manager.device().create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Depth Debug Params Buffer"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                },
            );
            self.depth_debug_params_buffer = Some(buf);

            // Create bind group once (points at the buffer).
            let buffer_ref = self.depth_debug_params_buffer.as_ref().unwrap();
            let bg = self
                .pipeline_manager
                .create_uniform_bind_group(&[buffer_ref], "Depth Debug Params BindGroup");
            self.depth_debug_params_bind_group = Some(bg);
        } else {
            // Update existing buffer
            let buf = self.depth_debug_params_buffer.as_ref().unwrap();
            self.pipeline_manager
                .queue()
                .write_buffer(buf, 0, bytemuck::bytes_of(&params));

            // Bind group still valid; no need to recreate.
        }
    }
}

pub fn create_color_attachment_load<'a>(
    msaa_view: &'a TextureView,
    surface_view: &'a TextureView,
    msaa_samples: u32,
) -> RenderPassColorAttachment<'a> {
    println!(
        "Creating color attachment load {:?} {:?}",
        msaa_view.texture().size(),
        surface_view.texture().size()
    );
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
