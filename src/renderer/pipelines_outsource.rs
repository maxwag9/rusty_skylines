use crate::data::Settings;
use crate::helpers::mouse_ray::PickUniform;
use crate::helpers::paths::data_dir;
use crate::renderer::gtao::gtao::{GtaoBlurParams, GtaoParams, GtaoUpsampleApplyParams};
use crate::renderer::pipelines::{FogUniforms, MeshBuffers, ToneMappingUniforms, make_dummy_buf};
use crate::resources::Uniforms;
use crate::world::camera::Camera;
use crate::world::terrain::sky::SkyUniform;
use crate::world::terrain::water::{SimpleVertex, WaterUniform};
use std::fs;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;

pub fn create_camera_buffer(
    device: &Device,
    camera: &Camera,
    config: &SurfaceConfiguration,
    settings: &Settings,
) -> Buffer {
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Uniform Buffer"),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        size: size_of::<Uniforms>() as u64,
        mapped_at_creation: false,
    });

    buffer
}

pub fn create_sky_buffer(device: &Device) -> Buffer {
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Sky Uniform Buffer"),
        size: size_of::<SkyUniform>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    buffer
}

pub fn create_fog_buffer(device: &Device) -> Buffer {
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Fog Uniform Buffer"),
        size: size_of::<FogUniforms>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    buffer
}
pub fn create_tonemapping_buffer(device: &Device) -> Buffer {
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Tonemapping Uniform Buffer"),
        size: size_of::<ToneMappingUniforms>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    buffer
}

pub fn create_pick_buffer(device: &Device) -> Buffer {
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

    buffer
}

pub fn create_gtao_buffer(device: &Device, settings: &Settings) -> Buffer {
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("GTAO Uniform Buffer"),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        size: size_of::<GtaoParams>() as u64,
        mapped_at_creation: false,
    });

    buffer
}
pub fn create_gtao_blur_buffer(device: &Device, settings: &Settings) -> Buffer {
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Blur Uniform Buffer"),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        size: size_of::<GtaoBlurParams>() as u64,
        mapped_at_creation: false,
    });

    buffer
}
pub fn create_gtao_upsample_apply_buffer(device: &Device, settings: &Settings) -> Buffer {
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("GTAO Upsample Uniform Buffer"),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        size: size_of::<GtaoUpsampleApplyParams>() as u64,
        mapped_at_creation: false,
    });

    buffer
}

pub fn create_water_buffer(device: &Device) -> Buffer {
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

    buffer
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
