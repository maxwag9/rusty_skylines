use crate::components::camera::Camera;
use crate::renderer::pipelines_outsource::*;
use crate::resources::Uniforms;
use glam::{Mat4, Vec3};
use std::borrow::Cow;
use std::fs;
use std::path::{Path, PathBuf};
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
    pub depth_sample_view: TextureView, // sampling (DepthOnly)

    pub msaa_samples: u32,

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
}

impl Pipelines {
    pub fn new(
        device: &Device,
        config: &SurfaceConfiguration,
        msaa_samples: u32,
        shader_dir: &Path,
        camera: &Camera,
    ) -> anyhow::Result<Self> {
        // Create render targets
        let (msaa_texture, msaa_view) = create_msaa_targets(&device, &config, msaa_samples);
        let (depth_texture, depth_view, depth_sample_view) =
            create_depth_texture(device, config, msaa_samples);
        // Load all shaders
        let shaders = load_all_shaders(device, shader_dir)?;

        let uniforms = create_camera_uniforms(device, camera, config);
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
            msaa_texture,
            msaa_view,
            depth_texture,
            depth_view,
            depth_sample_view,
            msaa_samples,
            config: config.clone(),

            uniforms,
            sky_uniforms,
            water_uniforms,
            fog_uniforms,
            pick_uniforms,

            water_mesh_buffers: water_mesh,
            gizmo_mesh_buffers: gizmo_mesh,

            stars_mesh_buffers: stars_mesh,

            grass_texture_resources,
        };

        Ok(this)
    }

    pub(crate) fn resize(&mut self, config: &SurfaceConfiguration, msaa_samples: u32) {
        // Keep a fresh copy of the surface configuration so our MSAA and depth textures
        // always match the swapchain size. Or ELSE, after a window resize we'd recreate
        // attachments using the old dimensions, leading to mismatched resolve targets!!!
        self.config = config.clone();
        (self.msaa_texture, self.msaa_view) =
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

pub fn make_dummy_render_pipeline(device: &Device, format: TextureFormat) -> RenderPipeline {
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
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("dummy compute shader"),
        source: ShaderSource::Wgsl(
            "
            @compute @workgroup_size(1)
            fn main() {
                // do nothing
            }
            "
            .into(),
        ),
    });

    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("dummy compute layout"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });

    device.create_compute_pipeline(&ComputePipelineDescriptor {
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
