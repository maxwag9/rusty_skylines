use crate::components::camera::Camera;
use crate::data::Settings;
use crate::hsv::lerp;
use crate::mouse_ray::PickUniform;
use crate::paths::data_dir;
use crate::renderer::pipelines::{
    FogUniforms, GpuResourceSet, MeshBuffers, ToneMappingUniforms, make_dummy_buf,
    make_new_uniforms_csm,
};
use crate::renderer::procedural_render_manager::DepthDebugParams;
use crate::renderer::shadows::compute_csm_matrices;
use crate::resources::Uniforms;
use crate::terrain::sky::SkyUniform;
use crate::terrain::water::{SimpleVertex, WaterUniform};
use crate::ui::vertex::LineVtxRender;
use glam::Vec3;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};
use std::fs;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;

pub fn create_camera_uniforms(
    device: &Device,
    camera: &Camera,
    config: &SurfaceConfiguration,
    settings: &Settings,
) -> (GpuResourceSet, Uniforms) {
    let aspect = config.width as f32 / config.height as f32;
    let sun = Vec3::new(0.3, 1.0, 0.6).normalize();
    let (view, proj, view_proj) = camera.matrices(aspect, settings);
    // Build 4 cascade matrices + splits (defaults baked in: shadow distance, lambda, padding).
    let (light_mats, splits) = compute_csm_matrices(
        view,
        camera.fov.to_radians(),
        aspect,
        camera.near,
        camera.far,
        Vec3::ONE,
        /*shadow_map_size:*/ 2048, // or the actual CSM texture size
        /*stabilize:*/ true,
        settings.reversed_depth_z,
    );

    // This is the uniforms used for *normal* rendering (shadow_cascade_index unused there).
    let uniforms = make_new_uniforms_csm(
        view,
        proj,
        view_proj,
        Vec3::ONE,
        Vec3::ONE,
        0.0,
        light_mats,
        splits,
        camera,
        settings,
    );

    let buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Uniform Bind Group Layout"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: BufferSize::new(size_of::<Uniforms>() as u64),
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Uniform Bind Group"),
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    (
        GpuResourceSet {
            _bind_group_layout: bind_group_layout,
            _bind_group: bind_group,
            buffer,
        },
        uniforms,
    )
}

pub fn create_sky_uniforms(device: &Device) -> GpuResourceSet {
    let sky_uniform = SkyUniform {
        exposure: 1.0,
        moon_phase: 0.0,
        sun_size: 0.05,
        sun_intensity: 510.0,
        moon_size: 0.04,
        moon_intensity: 1.0,
        _pad1: 0.0,
        _pad2: 0.0,
    };

    let buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Sky Uniform Buffer"),
        contents: bytemuck::bytes_of(&sky_uniform),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Sky Uniforms BGL"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Sky BG"),
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    GpuResourceSet {
        _bind_group_layout: bind_group_layout,
        _bind_group: bind_group,
        buffer,
    }
}
// pub fn create_road_uniforms(device: &Device) -> GpuResourceSet {
//     let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
//         label: Some("Road BGL"),
//         entries: &[BindGroupLayoutEntry {
//             binding: 0,
//             visibility: ShaderStages::FRAGMENT,
//             ty: BindingType::Texture {
//                 sample_type: TextureSampleType::Float { filterable: true },
//                 view_dimension: TextureViewDimension::D2,
//                 multisampled: false,
//             },
//             count: None,
//         }],
//     });
//     GpuResourceSet {
//         bind_group_layout,
//         bind_group,
//         buffer,
//     }
// }
pub fn create_fog_uniforms(device: &Device) -> GpuResourceSet {
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Fog BGL"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: BufferSize::new(size_of::<FogUniforms>() as u64),
            },
            count: None,
        }],
    });

    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Fog Uniform Buffer"),
        size: size_of::<FogUniforms>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Fog Bind Group"),
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    GpuResourceSet {
        _bind_group_layout: bind_group_layout,
        _bind_group: bind_group,
        buffer,
    }
}
pub fn create_tonemapping_uniforms(device: &Device) -> GpuResourceSet {
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Tonemapping BGL"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: BufferSize::new(size_of::<ToneMappingUniforms>() as u64),
            },
            count: None,
        }],
    });

    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Tonemapping Uniform Buffer"),
        size: size_of::<ToneMappingUniforms>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Tonemapping Bind Group"),
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    GpuResourceSet {
        _bind_group_layout: bind_group_layout,
        _bind_group: bind_group,
        buffer,
    }
}

pub fn create_pick_uniforms(device: &Device) -> GpuResourceSet {
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Pick Uniform BGL"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Pick Uniform Buffer"),
        contents: bytemuck::bytes_of(&PickUniform {
            pos: [0.0; 3],
            radius: 0.0,
            underwater: 0,
            _pad0: [0, 0, 0],
            color: [1.0, 0.0, 0.0],
            _pad1: 0.0,
        }),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Pick Uniform BG"),
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    GpuResourceSet {
        _bind_group_layout: bind_group_layout,
        _bind_group: bind_group,
        buffer,
    }
}
pub fn create_depth_debug_uniforms(
    device: &Device,
    camera: &Camera,
    msaa_samples: u32,
) -> GpuResourceSet {
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Depth Debug Uniform BGL"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let params = DepthDebugParams {
        near: camera.near,
        far: camera.far,
        power: 20.0,
        reversed_z: 0, // if you use reversed-z, else 0
        msaa_samples,
        _pad0: 0,
        _pad1: 0,
    };

    let buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Depth Debug Uniform Buffer"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Depth Debug Uniform BG"),
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    GpuResourceSet {
        _bind_group_layout: bind_group_layout,
        _bind_group: bind_group,
        buffer,
    }
}
pub const SSAO_KERNEL_AMOUNT: usize = 32;
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SsaoUniforms {
    pub kernel: [[f32; 4]; SSAO_KERNEL_AMOUNT],
    pub params0: [f32; 4], // radius, bias, intensity, power
    pub params1: [u32; 4], // reversed_z, noise_tile_px, 0, 0
}
pub fn create_ssao_uniforms(
    device: &Device,
    camera: &Camera,
    settings: &Settings,
) -> GpuResourceSet {
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("SSAO Uniform BGL"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let radius = 3.0f32;
    let bias = 0.10f32;
    let intensity = 1.0f32;
    let power = 1.22f32;
    let reversed_z = settings.reversed_depth_z as u32;
    let noise_tile_px = 8u32;
    let params = SsaoUniforms {
        kernel: make_ssao_kernel(69420, 2.0),
        params0: [radius, bias, intensity, power],
        params1: [reversed_z, noise_tile_px, 0, 0],
    };

    let buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("SSAO Uniform Buffer"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("SSAO Uniform BG"),
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    GpuResourceSet {
        _bind_group_layout: bind_group_layout,
        _bind_group: bind_group,
        buffer,
    }
}
pub fn create_water_uniforms(device: &Device, sky_buffer: &Buffer) -> GpuResourceSet {
    let wu = WaterUniform {
        sea_level: 0.0,
        _pad0: [0.0; 3],
        color: [0.05, 0.25, 0.35, 0.55],
        wave_tiling: 0.05,
        wave_strength: 0.05,
        _pad1: [0.0; 2],
    };

    let buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Water Uniform Buffer"),
        contents: bytemuck::bytes_of(&wu),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Water BGL"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(size_of::<WaterUniform>() as u64),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(size_of::<SkyUniform>() as u64),
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Water BG"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: sky_buffer.as_entire_binding(),
            },
        ],
    });

    GpuResourceSet {
        _bind_group_layout: bind_group_layout,
        _bind_group: bind_group,
        buffer,
    }
}

pub fn create_water_mesh(device: &Device) -> MeshBuffers {
    let water_vertices = [
        SimpleVertex {
            pos: [-20000.0, 0.0, -20000.0],
        },
        SimpleVertex {
            pos: [20000.0, 0.0, -20000.0],
        },
        SimpleVertex {
            pos: [20000.0, 0.0, 20000.0],
        },
        SimpleVertex {
            pos: [-20000.0, 0.0, 20000.0],
        },
    ];

    let water_indices: [u32; 6] = [0, 1, 2, 0, 2, 3];

    let vertex = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Water VB"),
        contents: bytemuck::cast_slice(&water_vertices),
        usage: BufferUsages::VERTEX,
    });

    let index = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Water IB"),
        contents: bytemuck::cast_slice(&water_indices),
        usage: BufferUsages::INDEX,
    });

    MeshBuffers {
        vertex,
        index,
        index_count: water_indices.len() as u32,
    }
}

pub fn create_gizmo_mesh(device: &Device) -> MeshBuffers {
    let vertex = device.create_buffer(&BufferDescriptor {
        label: Some("Gizmo VB"),
        size: (size_of::<LineVtxRender>() * 6) as u64,
        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    MeshBuffers {
        vertex,
        index: make_dummy_buf(&device),
        index_count: 0,
    }
}

pub fn create_stars_mesh(device: &Device) -> MeshBuffers {
    let stars_bytes = fs::read(data_dir("stars.bin")).expect("stars.bin missing");

    let vertex = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Star Buffer"),
        contents: &stars_bytes,
        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
    });

    MeshBuffers {
        vertex,
        index: make_dummy_buf(&device),
        index_count: 0,
    }
}

pub fn make_ssao_kernel(seed: u64, horizon_bias: f32) -> [[f32; 4]; SSAO_KERNEL_AMOUNT] {
    // horizon_bias:
    // 1.0 = uniform hemisphere
    // 2.0..4.0 = more rays near tangent plane (often helps corners)
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut kernel = [[0.0f32; 4]; SSAO_KERNEL_AMOUNT];

    for i in 0..SSAO_KERNEL_AMOUNT {
        let u1: f32 = rng.random();
        let u2: f32 = rng.random();

        let phi = std::f32::consts::TAU * u1;

        // cos(theta) in [0..1], biased toward 0 (horizon) if horizon_bias > 1
        let cos_theta = u2.powf(horizon_bias);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let x = sin_theta * phi.cos();
        let y = sin_theta * phi.sin();
        let z = cos_theta; // >= 0 hemisphere  // hemisphere (IMPORTANT) and cool, crysis had a full circle so it had some unique look

        // deterministic per-sample radius scale (less grain than random scaling)
        let fi = i as f32 / SSAO_KERNEL_AMOUNT as f32;
        let scale = lerp(0.05, 1.0, fi * fi);

        kernel[i] = [x * scale, y * scale, z * scale, 0.0];
    }

    kernel
}
