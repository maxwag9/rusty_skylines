// cars_render.rs
use bytemuck::{Pod, Zeroable};
use wgpu::{VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CarInstance {
    pub model: [[f32; 4]; 4],      // transform
    pub prev_model: [[f32; 4]; 4], // transform (previous ofc)
    pub color: [f32; 3],
    pub _pad: f32,
}
impl CarInstance {
    pub fn layout<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            array_stride: size_of::<CarInstance>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: &[
                // mat4 = 4 vec4s
                VertexAttribute {
                    shader_location: 4,
                    format: VertexFormat::Float32x4,
                    offset: 0,
                },
                VertexAttribute {
                    shader_location: 5,
                    format: VertexFormat::Float32x4,
                    offset: 16,
                },
                VertexAttribute {
                    shader_location: 6,
                    format: VertexFormat::Float32x4,
                    offset: 32,
                },
                VertexAttribute {
                    shader_location: 7,
                    format: VertexFormat::Float32x4,
                    offset: 48,
                },
                // prev mat4 = 4 vec4s
                VertexAttribute {
                    shader_location: 8,
                    format: VertexFormat::Float32x4,
                    offset: 64,
                },
                VertexAttribute {
                    shader_location: 9,
                    format: VertexFormat::Float32x4,
                    offset: 80,
                },
                VertexAttribute {
                    shader_location: 10,
                    format: VertexFormat::Float32x4,
                    offset: 96,
                },
                VertexAttribute {
                    shader_location: 11,
                    format: VertexFormat::Float32x4,
                    offset: 112,
                },
                // color
                VertexAttribute {
                    shader_location: 12,
                    format: VertexFormat::Float32x3,
                    offset: 128,
                },
            ],
        }
    }
}
