use crate::components::camera::Camera;
use crate::mouse_ray::PickUniform;
use crate::paths::{data_dir, shader_dir};
use crate::renderer::pipelines::{
    FogUniforms, GpuResourceSet, MeshBuffers, ShaderAsset, ToneMappingUniforms,
    create_grass_texture, load_shader, make_dummy_buf, make_new_uniforms_csm,
};
use crate::renderer::shadows::compute_csm_matrices;
use crate::renderer::textures::grass::{GrassParams, generate_noise};
use crate::resources::Uniforms;
use crate::terrain::sky::SkyUniform;
use crate::terrain::water::{SimpleVertex, WaterUniform};
use crate::ui::vertex::LineVtxRender;
use glam::Vec3;
use std::fs;
use wgpu::TextureFormat::Rgba8Unorm;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;

// Helper struct to hold all shaders
pub struct Shaders {
    pub(crate) line: ShaderAsset,
    pub(crate) water: ShaderAsset,
    pub(crate) sky: ShaderAsset,
    pub(crate) stars: ShaderAsset,
    pub(crate) grass_texture: ShaderAsset,
    pub(crate) road: ShaderAsset,
}

pub fn load_all_shaders(device: &Device) -> anyhow::Result<Shaders> {
    Ok(Shaders {
        line: load_shader(device, &shader_dir().join("lines.wgsl"), "Line Shader")?,
        water: load_shader(device, &shader_dir().join("water.wgsl"), "Water Shader")?,
        sky: load_shader(device, &shader_dir().join("sky.wgsl"), "Sky Shader")?,
        stars: load_shader(device, &shader_dir().join("stars.wgsl"), "Stars Shader")?,
        grass_texture: load_shader(
            device,
            &shader_dir().join("textures/grass.wgsl"),
            "Grass Texture Shader",
        )?,
        road: load_shader(device, &shader_dir().join("road.wgsl"), "Road Shader")?,
    })
}

pub fn create_camera_uniforms(
    device: &Device,
    camera: &Camera,
    config: &SurfaceConfiguration,
) -> (GpuResourceSet, Uniforms) {
    let aspect = config.width as f32 / config.height as f32;
    let sun = Vec3::new(0.3, 1.0, 0.6).normalize();
    let (view, proj, view_proj) = camera.matrices(aspect);
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

pub fn create_grass_texture_resources(
    device: &Device,
    config: &SurfaceConfiguration,
) -> GpuResourceSet {
    let grass_params = GrassParams {
        grass_color: [0.2, 0.6, 0.2, 1.0],
        blade_density: 120.0,
        blade_height: 0.8,
        wind_phase: 0.0,
        time: 0.0,
        noise_scale: 4.0,
        _pad: [0.0; 3],
    };

    let buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("grass_params_buffer"),
        contents: bytemuck::bytes_of(&grass_params),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    let noise_data: Vec<f32> = generate_noise(512);
    let noise_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("grass_noise_buffer"),
        contents: bytemuck::cast_slice(&noise_data),
        usage: BufferUsages::STORAGE,
    });

    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("grass_texture_bgl"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: Rgba8Unorm,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let (_, grass_texture_view) = create_grass_texture(&device, &config);

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("Grass Texture Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&grass_texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: noise_buffer.as_entire_binding(),
            },
        ],
    });

    GpuResourceSet {
        _bind_group_layout: bind_group_layout,
        _bind_group: bind_group,
        buffer,
    }
}
