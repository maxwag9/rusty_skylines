use crate::components::camera::Camera;
use crate::data::Settings;
use crate::renderer::pipelines_outsource::*;
use crate::renderer::shadows::{CSM_CASCADES, CascadedShadowMap, create_csm_shadow_texture};
use crate::resources::Uniforms;
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::fs;
use std::path::PathBuf;
use wgpu::TextureFormat::{Depth32Float, Rgba8Unorm, Rgba16Float};
use wgpu::*;

#[macro_export]
macro_rules! time_call {
    ($label:expr, $expr:expr) => {{
        let start = Instant::now();
        let result = $expr;
        let elapsed = start.elapsed();
        println!("{:<40} {:>8.3} ms", $label, elapsed.as_secs_f64() * 1000.0);
        result
    }};
}

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
impl Default for FogUniforms {
    fn default() -> Self {
        Self {
            screen_size: [1920.0, 1080.0],
            proj_params: [0.1, 1000.0],
            fog_density: 1.0,
            fog_height: 10.0, // Fog is thickest below y=10
            cam_height: 50.0,
            _pad0: 0.0,
            fog_color: [0.7, 0.75, 0.8], // Light gray-blue
            _pad1: 0.0,
            fog_sky_factor: 0.3,
            fog_height_falloff: 0.05, // How quickly fog thins above fog_height
            fog_start: 1000.0,
            fog_end: 10000.0,
        }
    }
}
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub enum ToneMappingState {
    #[default]
    Cinematic,
    Off,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ToneMappingUniforms {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
}
impl Default for ToneMappingUniforms {
    fn default() -> Self {
        Self {
            a: 2.55,
            b: 0.02,
            c: 2.43,
            d: 0.59,
            e: 0.14,
        }
    }
}
impl ToneMappingUniforms {
    pub fn from_state(tonemapping_state: &ToneMappingState) -> Self {
        match tonemapping_state {
            ToneMappingState::Cinematic => Self::cinematic(),
            ToneMappingState::Off => Self::off(),
        }
    }
    fn cinematic() -> Self {
        Self {
            a: 2.55,
            b: 0.02,
            c: 2.43,
            d: 0.59,
            e: 0.14,
        }
    }
    fn off() -> Self {
        Self {
            a: 0.0,
            b: 1.0,
            c: 0.0,
            d: 0.0,
            e: 1.0,
        }
    }
}
pub struct GpuResourceSet {
    pub _bind_group_layout: BindGroupLayout,
    pub _bind_group: BindGroup,
    pub buffer: Buffer,
}
pub struct MeshBuffers {
    pub vertex: Buffer,
    pub index: Buffer,
    pub index_count: u32,
}

#[derive(Clone)]
pub struct ShaderAsset {
    pub _path: PathBuf,
    pub module: ShaderModule,
}
pub struct ShadowSamplers {
    pub shadow_sampler: Sampler,
    pub shadow_sampler_rev_z: Sampler,
    pub shadow_sampler_off: Sampler,
}
pub struct Pipelines {
    pub device: Device,

    pub msaa_hdr_view: TextureView,
    pub resolved_hdr_view: TextureView,
    pub msaa_normal_view: TextureView,
    pub resolved_normals_full: TextureView,
    pub resolved_normals_half: TextureView,
    pub ssao_raw_view_half: TextureView,
    pub ssao_blur_view_half: TextureView,
    pub ssao_history_half: TextureView,
    pub depth_view: TextureView,
    pub depth_sample_view: TextureView, // sampling (DepthOnly)
    pub resolved_depth_view_full: TextureView,
    pub linear_depth_view_full: TextureView,
    pub linear_depth_view_half: TextureView,

    pub cascaded_shadow_map: CascadedShadowMap,
    pub shadow_samplers: ShadowSamplers,

    pub config: SurfaceConfiguration,

    pub camera_uniforms: GpuResourceSet,
    pub sky_uniforms: GpuResourceSet,
    pub water_uniforms: GpuResourceSet,
    pub fog_uniforms: GpuResourceSet,
    pub tonemapping_uniforms: GpuResourceSet,
    pub pick_uniforms: GpuResourceSet,
    pub ssao_uniforms: GpuResourceSet,

    pub water_mesh_buffers: MeshBuffers,
    pub stars_mesh_buffers: MeshBuffers,
}

impl Pipelines {
    pub fn new(
        device: &Device,
        config: &SurfaceConfiguration,
        camera: &Camera,
        settings: &Settings,
    ) -> anyhow::Result<Self> {
        let msaa_samples = settings.msaa_samples;
        // Create render targets
        let (msaa_hdr_view, msaa_normal_view) = create_msaa_targets(&device, &config, msaa_samples);
        let (resolved_hdr_view, resolved_normal_view, resolved_depth_view) =
            create_resolved_targets(&device, &config);
        let (depth_view, depth_sample_view) = create_depth_texture(device, config, msaa_samples);
        let linear_depth_view_full = create_linear_depth_texture(device, config, 1.0);
        let linear_depth_view_half = create_linear_depth_texture(device, config, 0.5);
        let resolved_normals_half = create_normals_texture(device, config, 0.5);
        let (ssao_raw_view_half, ssao_blur_view_half) = create_ssao_textures(device, config, 0.5);
        let ssao_history_half = create_ao_history_texture(device, config, 0.5);
        let cascaded_shadow_map =
            create_csm_shadow_texture(device, settings.shadow_map_size, "Sun CSM");
        let shadow_samplers = create_shadow_samplers(&device);
        let camera_uniforms = create_camera_uniforms(device, camera, config, settings);
        let sky_uniforms = create_sky_uniforms(device);
        let fog_uniforms = create_fog_uniforms(device);
        let tonemapping_uniforms = create_tonemapping_uniforms(device);
        let pick_uniforms = create_pick_uniforms(device);
        let ssao_uniforms = create_ssao_uniforms(device, settings);

        let water_uniforms = create_water_uniforms(device, &sky_uniforms.buffer);
        let water_mesh = create_water_mesh(device);
        let stars_mesh = create_stars_mesh(device);

        let this = Self {
            device: device.clone(),
            config: config.clone(),
            msaa_hdr_view,
            resolved_hdr_view,
            msaa_normal_view,
            resolved_normals_full: resolved_normal_view,
            resolved_normals_half,
            ssao_raw_view_half,
            ssao_blur_view_half,
            ssao_history_half,
            depth_view,
            depth_sample_view,
            resolved_depth_view_full: resolved_depth_view,
            linear_depth_view_full,
            linear_depth_view_half,

            cascaded_shadow_map,

            camera_uniforms,
            sky_uniforms,
            water_uniforms,
            fog_uniforms,
            tonemapping_uniforms,
            pick_uniforms,

            ssao_uniforms,
            water_mesh_buffers: water_mesh,

            stars_mesh_buffers: stars_mesh,
            shadow_samplers,
        };

        Ok(this)
    }

    pub(crate) fn resize(&mut self, config: &SurfaceConfiguration, msaa_samples: u32) {
        // Keep a fresh copy of the surface configuration so our MSAA and depth textures
        // always match the swapchain size. Or ELSE, after a window resize we'd recreate
        // attachments using the old dimensions, leading to mismatched resolve targets!!!
        self.config = config.clone();
        (self.msaa_hdr_view, self.msaa_normal_view) =
            create_msaa_targets(&self.device, &self.config, msaa_samples);
        (
            self.resolved_hdr_view,
            self.resolved_normals_full,
            self.resolved_depth_view_full,
        ) = create_resolved_targets(&self.device, &self.config);
        (self.depth_view, self.depth_sample_view) =
            create_depth_texture(&self.device, &self.config, msaa_samples);
    }
}

fn create_shadow_samplers(device: &Device) -> ShadowSamplers {
    let shadow_sampler = device.create_sampler(&SamplerDescriptor {
        label: Some("Shadow Sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: MipmapFilterMode::Nearest,
        compare: Some(CompareFunction::LessEqual), // ||| Great!
        ..Default::default()
    });
    let shadow_sampler_rev_z = device.create_sampler(&SamplerDescriptor {
        label: Some("Shadow Sampler (Reversed-Z)"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: MipmapFilterMode::Nearest,
        compare: Some(CompareFunction::GreaterEqual), // <-- important
        ..Default::default()
    });
    let shadow_sampler_off = device.create_sampler(&SamplerDescriptor {
        label: Some("Shadows Off Sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: MipmapFilterMode::Nearest,
        compare: Some(CompareFunction::Always), // <--- Mission CRITICAL
        ..Default::default()
    });
    ShadowSamplers {
        shadow_sampler,
        shadow_sampler_rev_z,
        shadow_sampler_off,
    }
}

pub fn make_dummy_buf(device: &Device) -> Buffer {
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
        _path: path.clone(),
        module,
    };
    Ok(asset)
}

pub fn create_msaa_targets(
    device: &Device,
    config: &SurfaceConfiguration,
    samples: u32,
) -> (TextureView, TextureView) {
    let color_texture = device.create_texture(&TextureDescriptor {
        label: Some("MSAA Color Texture"),
        size: Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: samples,
        dimension: TextureDimension::D2,
        format: Rgba16Float,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let color_view = color_texture.create_view(&TextureViewDescriptor::default());
    let normal_texture = device.create_texture(&TextureDescriptor {
        label: Some("MSAA Normals Texture"),
        size: Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: samples,
        dimension: TextureDimension::D2,
        format: NORMAL_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let normal_view = normal_texture.create_view(&TextureViewDescriptor::default());
    (color_view, normal_view)
}
const NORMAL_FORMAT: TextureFormat = Rgba8Unorm;
pub fn create_resolved_targets(
    device: &Device,
    config: &SurfaceConfiguration,
) -> (TextureView, TextureView, TextureView) {
    let color_texture = device.create_texture(&TextureDescriptor {
        label: Some("Resolved, non-MSAA Color Texture"),
        size: Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: Rgba16Float,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let color_view = color_texture.create_view(&TextureViewDescriptor::default());
    let normal_texture = device.create_texture(&TextureDescriptor {
        label: Some("Resolved, non-MSAA Normals Texture"),
        size: Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: NORMAL_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let normal_view = normal_texture.create_view(&TextureViewDescriptor::default());
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("Depth Texture"),
        size: Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: Depth32Float,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    // For sampling in fog shader: MUST be DepthOnly for Depth24PlusStencil8
    let depth_only_view = texture.create_view(&TextureViewDescriptor {
        label: Some("Depth Texture View (DepthOnly)"),
        aspect: TextureAspect::DepthOnly,
        ..Default::default()
    });
    (color_view, normal_view, depth_only_view)
}
pub fn create_normals_texture(
    device: &Device,
    config: &SurfaceConfiguration,
    resolution_factor: f32,
) -> TextureView {
    let tex = device.create_texture(&TextureDescriptor {
        label: Some("Normal Half"),
        size: Extent3d {
            width: (config.width as f32 * resolution_factor) as u32,
            height: (config.height as f32 * resolution_factor) as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: NORMAL_FORMAT,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    tex.create_view(&Default::default())
}

pub const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32FloatStencil8;

fn create_depth_texture(
    device: &Device,
    config: &SurfaceConfiguration,
    msaa_samples: u32,
) -> (TextureView, TextureView) {
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
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    // For using as depth-stencil attachment (can include stencil aspect)
    let attachment_view = texture.create_view(&TextureViewDescriptor::default());

    // For sampling in fog shader: MUST be DepthOnly for Depth24PlusStencil8
    let depth_only_view = texture.create_view(&TextureViewDescriptor {
        label: Some("Depth Texture View (DepthOnly)"),
        aspect: TextureAspect::DepthOnly,
        ..Default::default()
    });

    (attachment_view, depth_only_view)
}
fn create_linear_depth_texture(
    device: &Device,
    config: &SurfaceConfiguration,
    resolution_factor: f32,
) -> TextureView {
    // linear depth (filterable!)
    let linear_depth_tex = device.create_texture(&TextureDescriptor {
        label: Some("Linear Depth"),
        size: Extent3d {
            width: (config.width as f32 * resolution_factor) as u32,
            height: (config.height as f32 * resolution_factor) as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1, // single-sample
        dimension: TextureDimension::D2,
        format: TextureFormat::R16Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let linear_depth_view = linear_depth_tex.create_view(&Default::default());

    linear_depth_view
}

pub const SSAO_FORMAT: TextureFormat = TextureFormat::R8Unorm;

fn create_ssao_textures(
    device: &Device,
    config: &SurfaceConfiguration,
    resolution_factor: f32,
) -> (TextureView, TextureView) {
    let make = |label: &str| {
        let tex = device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: (config.width as f32 * resolution_factor) as u32,
                height: (config.height as f32 * resolution_factor) as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: SSAO_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&TextureViewDescriptor::default());
        (tex, view)
    };

    let (_, ssao_view) = make("SSAO Texture");
    let (_, ssao_blur_view) = make("SSAO Blurred Texture");
    (ssao_view, ssao_blur_view)
}
pub fn create_ao_history_texture(
    device: &Device,
    config: &SurfaceConfiguration,
    resolution_factor: f32,
) -> TextureView {
    let tex = device.create_texture(&TextureDescriptor {
        label: Some("AO History"),
        size: Extent3d {
            width: (config.width as f32 * resolution_factor) as u32,
            height: (config.height as f32 * resolution_factor) as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: SSAO_FORMAT,
        usage: TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    tex.create_view(&Default::default())
}

pub fn make_new_uniforms_csm(
    view: Mat4,
    proj: Mat4,
    view_proj: Mat4,
    sun: Vec3,
    moon: Vec3,
    total_time: f64,
    light_view_proj: [Mat4; CSM_CASCADES],
    cascade_splits: [f32; 4],
    camera: &Camera,
    settings: &Settings,
) -> Uniforms {
    let eye = camera.eye_world();
    Uniforms {
        view: view.to_cols_array_2d(),
        inv_view: view.inverse().to_cols_array_2d(),
        proj: proj.to_cols_array_2d(),
        inv_proj: proj.inverse().to_cols_array_2d(),
        view_proj: view_proj.to_cols_array_2d(),
        inv_view_proj: view_proj.inverse().to_cols_array_2d(),
        lighting_view_proj: light_view_proj.map(|m| m.to_cols_array_2d()),
        cascade_splits,
        sun_direction: sun.to_array(),
        time: total_time as f32,

        camera_local: [eye.local.x, eye.local.y, eye.local.z],
        chunk_size: camera.chunk_size as f32,
        camera_chunk: [eye.chunk.x, eye.chunk.z],

        _pad_cam: [1, 1],
        moon_direction: moon.to_array(),
        orbit_radius: camera.orbit_radius,
        reversed_depth_z: settings.reversed_depth_z as u32,
        shadows_enabled: settings.shadows_enabled as u32,

        near_far_depth: [camera.near, camera.far],
    }
}
