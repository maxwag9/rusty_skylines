use crate::components::camera::Camera;
use crate::renderer::pipelines_outsource::*;
use crate::renderer::shadows::{CSM_CASCADES, CascadedShadowMap, create_csm_shadow_texture};
use crate::resources::Uniforms;
use glam::{Mat4, Vec3};
use std::borrow::Cow;
use std::fs;
use std::path::PathBuf;
use wgpu::TextureFormat::Rgba16Float;
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

pub struct RenderPipelineState {
    pub shader: ShaderAsset,
    pub pipeline: RenderPipeline,
}

pub struct ComputePipelineState {
    pub shader: ShaderAsset,
    pub pipeline: ComputePipeline,
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

pub struct Pipelines {
    pub device: Device,

    pub msaa_hdr_texture: Texture,
    pub msaa_hdr_view: TextureView,
    pub resolved_hdr_texture: Texture,
    pub resolved_hdr_view: TextureView,
    pub depth_texture: Texture,
    pub depth_view: TextureView,
    pub depth_sample_view: TextureView, // sampling (DepthOnly)
    pub cascaded_shadow_map: CascadedShadowMap,

    pub config: SurfaceConfiguration,

    pub uniforms: GpuResourceSet,
    pub sky_uniforms: GpuResourceSet,
    pub water_uniforms: GpuResourceSet,
    pub fog_uniforms: GpuResourceSet,
    pub pick_uniforms: GpuResourceSet,

    pub water_mesh_buffers: MeshBuffers,
    pub stars_mesh_buffers: MeshBuffers,
    pub gizmo_mesh_buffers: MeshBuffers,
    pub grass_texture_resources: GpuResourceSet,
    pub uniforms_cpu: Uniforms,
}

impl Pipelines {
    pub fn new(
        device: &Device,
        config: &SurfaceConfiguration,
        msaa_samples: u32,
        camera: &Camera,
        shadow_map_size: u32,
    ) -> anyhow::Result<Self> {
        // Create render targets
        let (msaa_hdr_texture, msaa_hdr_view) = create_msaa_targets(&device, &config, msaa_samples);
        let (resolved_hdr_texture, resolved_hdr_view) =
            create_resolved_targets(&device, &config, msaa_samples);
        let (depth_texture, depth_view, depth_sample_view) =
            create_depth_texture(device, config, msaa_samples);
        let csm = create_csm_shadow_texture(device, shadow_map_size, "Sun CSM"); // 2048 or 4096
        // Load all shaders
        let shaders = load_all_shaders(device)?;

        let (uniforms_set, uniforms_cpu) = create_camera_uniforms(device, camera, config);
        let sky_uniforms = create_sky_uniforms(device);
        let fog_uniforms = create_fog_uniforms(device);
        let pick_uniforms = create_pick_uniforms(device);
        //let road_uniforms = create_road_uniforms(device);
        let water_uniforms = create_water_uniforms(device, &sky_uniforms.buffer);
        let water_mesh = create_water_mesh(device);
        let gizmo_mesh = create_gizmo_mesh(device);
        let stars_mesh = create_stars_mesh(device);
        let grass_texture_resources = create_grass_texture_resources(device, config);

        let this = Self {
            device: device.clone(),
            msaa_hdr_texture,
            msaa_hdr_view,
            resolved_hdr_texture,
            resolved_hdr_view,
            depth_texture,
            depth_view,
            depth_sample_view,
            cascaded_shadow_map: csm,
            config: config.clone(),

            uniforms: uniforms_set,
            sky_uniforms,
            water_uniforms,
            fog_uniforms,
            pick_uniforms,

            water_mesh_buffers: water_mesh,
            gizmo_mesh_buffers: gizmo_mesh,

            stars_mesh_buffers: stars_mesh,

            grass_texture_resources,
            uniforms_cpu,
        };

        Ok(this)
    }

    pub(crate) fn resize(&mut self, config: &SurfaceConfiguration, msaa_samples: u32) {
        // Keep a fresh copy of the surface configuration so our MSAA and depth textures
        // always match the swapchain size. Or ELSE, after a window resize we'd recreate
        // attachments using the old dimensions, leading to mismatched resolve targets!!!
        self.config = config.clone();
        (self.msaa_hdr_texture, self.msaa_hdr_view) =
            create_msaa_targets(&self.device, &self.config, msaa_samples);
        (self.depth_texture, self.depth_view, self.depth_sample_view) =
            create_depth_texture(&self.device, &self.config, msaa_samples);
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
        format: Rgba16Float,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

pub fn create_resolved_targets(
    device: &Device,
    config: &SurfaceConfiguration,
    samples: u32,
) -> (Texture, TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
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

    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
}

pub const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth24PlusStencil8;

fn create_depth_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    msaa_samples: u32,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: msaa_samples,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    // For using as depth-stencil attachment (can include stencil aspect)
    let attachment_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    // For sampling in fog shader: MUST be DepthOnly for Depth24PlusStencil8
    let depth_only_view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("Depth Texture View (DepthOnly)"),
        aspect: wgpu::TextureAspect::DepthOnly,
        ..Default::default()
    });

    (texture, attachment_view, depth_only_view)
}

pub fn create_grass_texture(
    device: &Device,
    config: &SurfaceConfiguration,
) -> (Texture, TextureView) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("Grass Texture"),
        size: Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm, // NOT sRGB
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let view = texture.create_view(&TextureViewDescriptor::default());
    (texture, view)
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
    }
}
