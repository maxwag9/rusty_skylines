use wgpu::{VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode};

pub const STAR_COUNT: u32 = 116812;
pub const STARS_VERTEX_LAYOUT: VertexBufferLayout = VertexBufferLayout {
    array_stride: 16,
    step_mode: VertexStepMode::Instance, // IMPORTANT
    attributes: &[
        VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: VertexFormat::Float32,
        },
        VertexAttribute {
            offset: 4,
            shader_location: 1,
            format: VertexFormat::Float32,
        },
        VertexAttribute {
            offset: 8,
            shader_location: 2,
            format: VertexFormat::Float32,
        },
        VertexAttribute {
            offset: 12,
            shader_location: 3,
            format: VertexFormat::Float32,
        },
    ],
};
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyUniform {
    pub exposure: f32,
    pub moon_phase: f32,

    pub sun_size: f32,
    pub sun_intensity: f32,

    pub moon_size: f32,
    pub moon_intensity: f32,

    pub _pad1: f32,
    pub _pad2: f32,
}
