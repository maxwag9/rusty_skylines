use crate::data::Settings;
use crate::renderer::pipelines_outsource::*;
use crate::renderer::shadows::{CSM_CASCADES, CascadedShadowMap, create_csm_shadow_texture};
use crate::renderer::textures::noise::create_blue_noise_texture_gpu;
use crate::resources::Uniforms;
use crate::world::camera::Camera;
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::fs;
use std::path::PathBuf;
use wgpu::TextureFormat::{R32Float, Rgba8Unorm, Rgba16Float};
use wgpu::*;
use wgpu_render_manager::renderer::RenderManager;

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
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FogUniforms {
    pub fog_density: f32,
    pub fog_height: f32,
    pub fog_height_falloff: f32,
    pub fog_start: f32,

    pub fog_end: f32,
    pub fog_sky_factor: f32,
    pub _pad0: f32,
    pub _pad1: f32,

    pub fog_color: [f32; 3],
    pub _pad2: f32,
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
pub struct MsaaTextures {
    pub hdr: TextureView,
    pub normal: TextureView,
    pub depth: TextureView,
    pub depth_sample: TextureView,
}

pub struct ResolvedTextures {
    pub hdr: TextureView,
    pub normal: TextureView,
}

pub struct PostFxTextures {
    pub linear_depth_half: TextureView,
    pub normal_half: TextureView,
    pub gtao_blurred_half: TextureView,
    pub gtao_history: [TextureView; 2],
    pub rt_raw_half: TextureView,
    pub rt_denoised_half: TextureView,
    pub rt_instance: TextureView,
    pub dummy_msaa_rt_instance: TextureView,
}

pub struct UniformBuffers {
    pub camera: Buffer,
    pub sky: Buffer,
    pub water: Buffer,
    pub fog: Buffer,
    pub tonemapping: Buffer,
    pub pick: Buffer,
    pub gtao: Buffer,
    pub gtao_blur: Buffer,
    pub gtao_upsample_apply: Buffer,
}

pub struct SceneResources {
    pub blue_noise: TextureView,
    pub csm_shadows: CascadedShadowMap,
    pub shadow_samplers: ShadowSamplers,
    pub water_meshes: MeshBuffers,
    pub stars_meshes: MeshBuffers,
}

pub struct Pipelines {
    pub device: Device,
    pub config: SurfaceConfiguration,

    pub msaa: MsaaTextures,
    pub resolved: ResolvedTextures,
    pub post_fx: PostFxTextures,
    pub buffers: UniformBuffers,
    pub resources: SceneResources,
}

impl Pipelines {
    pub fn new(
        render_manager: &mut RenderManager,
        device: &Device,
        queue: &Queue,
        config: &SurfaceConfiguration,
        camera: &Camera,
        settings: &Settings,
    ) -> anyhow::Result<Self> {
        let msaa = Self::create_msaa_textures(device, config, settings.msaa_samples);
        let resolved = Self::create_resolved_textures(device, config);
        let post_fx = Self::create_post_fx_textures(device, config, settings.msaa_samples);
        let blue_noise = create_blue_noise_texture_gpu(render_manager, device, queue, 32, 69);
        // ^ Only GTAO needs it, it doesn't give a shit. This is expensive O(size⁴) computation. 32 is enough, 64 is bigger and still doesn't hog the game, but it's no use.

        let csm_shadow_map = create_csm_shadow_texture(device, settings.shadow_map_size, "Sun CSM");
        let shadow_samplers = create_shadow_samplers(device);
        let water_meshes = create_water_mesh(device);
        let stars_meshes = create_stars_mesh(device);

        let resources = SceneResources {
            blue_noise,
            csm_shadows: csm_shadow_map,
            shadow_samplers,
            water_meshes,
            stars_meshes,
        };

        let uniforms = UniformBuffers {
            camera: create_camera_buffer(device, camera, config, settings),
            sky: create_sky_buffer(device),
            water: create_water_buffer(device),
            fog: create_fog_buffer(device),
            tonemapping: create_tonemapping_buffer(device),
            pick: create_pick_buffer(device),
            gtao: create_gtao_buffer(device, settings),
            gtao_blur: create_gtao_blur_buffer(device, settings),
            gtao_upsample_apply: create_gtao_upsample_apply_buffer(device, settings),
        };

        Ok(Self {
            device: device.clone(),
            config: config.clone(),
            msaa,
            resolved,
            post_fx,
            buffers: uniforms,
            resources,
        })
    }

    pub fn resize(&mut self, config: &SurfaceConfiguration, msaa_samples: u32) {
        self.config = config.clone();
        self.msaa = Self::create_msaa_textures(&self.device, &self.config, msaa_samples);
        self.resolved = Self::create_resolved_textures(&self.device, &self.config);
        self.post_fx = Self::create_post_fx_textures(&self.device, &self.config, msaa_samples);
    }

    fn create_msaa_textures(
        device: &Device,
        config: &SurfaceConfiguration,
        samples: u32,
    ) -> MsaaTextures {
        let (hdr, normal) = create_msaa_targets(device, config, samples);
        let (depth, depth_sample) = create_depth_texture(device, config, samples);

        MsaaTextures {
            hdr,
            normal,
            depth,
            depth_sample,
        }
    }

    fn create_resolved_textures(
        device: &Device,
        config: &SurfaceConfiguration,
    ) -> ResolvedTextures {
        let (hdr, normal) = create_resolved_targets(device, config);

        ResolvedTextures { hdr, normal }
    }

    fn create_post_fx_textures(
        device: &Device,
        config: &SurfaceConfiguration,
        msaa_samples: u32,
    ) -> PostFxTextures {
        let linear_depth_half = create_linear_depth_texture(device, config, 0.5);
        let normal_half = create_normals_texture(device, config, 0.5);

        let (_, gtao_blurred_half) = create_gtao_textures(device, config, 0.5);
        let (rt_raw_half, rt_denoised_half) = create_rt_textures(device, config, 0.5);

        let dummy_msaa_rt_instance = create_instance_texture(device, config, 1.0, msaa_samples);
        let rt_instance = create_instance_texture(device, config, 1.0, 1);

        let gtao_history = [
            create_gtao_texture(device, config, 0.5),
            create_gtao_texture(device, config, 0.5),
        ];

        PostFxTextures {
            linear_depth_half,
            normal_half,
            gtao_blurred_half,
            gtao_history,
            dummy_msaa_rt_instance,
            rt_instance,
            rt_raw_half,
            rt_denoised_half,
        }
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
    let mut view_descriptor = TextureViewDescriptor::default();
    view_descriptor.label = Some("MSAA Color View");
    let color_view = color_texture.create_view(&view_descriptor);
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
) -> (TextureView, TextureView) {
    let create_hdr = |label: &str| {
        let tex = device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING, // Added for compute
            view_formats: &[],
        });
        tex.create_view(&TextureViewDescriptor::default())
    };

    let resolved_hdr = create_hdr("Resolved HDR Texture");
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
    (resolved_hdr, normal_view)
}
pub fn create_resolved_hdr(
    device: &Device,
    config: &SurfaceConfiguration,
    label: &str,
) -> TextureView {
    let create_hdr = |label: &str| {
        let tex = device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING, // Added for compute
            view_formats: &[],
        });
        tex.create_view(&TextureViewDescriptor::default())
    };

    let resolved_hdr = create_hdr(label);

    resolved_hdr
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

    let attachment_view = texture.create_view(&TextureViewDescriptor::default());

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
        format: R32Float,
        usage: TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });

    let linear_depth_view = linear_depth_tex.create_view(&Default::default());

    linear_depth_view
}

pub const GTAO_FORMAT: TextureFormat = R32Float;

fn create_gtao_textures(
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
            format: GTAO_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&TextureViewDescriptor::default());
        view
    };

    let gtao_view = make("GTAO Raw");
    let gtao_blurred_view = make("GTAO Blurred");
    (gtao_view, gtao_blurred_view)
}
const RT_FORMAT: TextureFormat = TextureFormat::R8Unorm;
fn create_rt_textures(
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
            format: RT_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&TextureViewDescriptor::default());
        view
    };

    let rt_view = make("RT Raw");
    let rt_blurred_view = make("RT Blurred");
    (rt_view, rt_blurred_view)
}
const RT_INSTANCE_FORMAT: TextureFormat = TextureFormat::R32Uint;
fn create_instance_texture(
    device: &Device,
    config: &SurfaceConfiguration,
    resolution_factor: f32,
    msaa_samples: u32,
) -> TextureView {
    let make = |label: &str| {
        let usage = if msaa_samples > 1 {
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING
        } else {
            TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING
        };
        let tex = device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: (config.width as f32 * resolution_factor) as u32,
                height: (config.height as f32 * resolution_factor) as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: msaa_samples,
            dimension: TextureDimension::D2,
            format: RT_INSTANCE_FORMAT,
            usage,
            view_formats: &[],
        });
        let view = tex.create_view(&TextureViewDescriptor::default());
        view
    };

    let rt_view = make("RT Instances");
    rt_view
}
pub fn create_gtao_texture(
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
        format: GTAO_FORMAT,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });

    tex.create_view(&Default::default())
}

pub fn make_new_camera_uniforms(
    sun: Vec3,
    moon: Vec3,
    total_time: f64,
    light_view_proj: [Mat4; CSM_CASCADES],
    cascade_splits: [f32; 4],
    camera: &Camera,
    settings: &Settings,
) -> Uniforms {
    let eye = camera.eye_world();
    let (view, proj, view_proj) = camera.matrices();
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
