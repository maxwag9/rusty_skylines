use bytemuck::{Pod, Zeroable};
use wgpu::{VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct WaterUniform {
    pub sea_level: f32,
    pub _pad0: [f32; 3],
    pub color: [f32; 4],
    pub wave_tiling: f32,
    pub wave_strength: f32,
    pub _pad1: [f32; 2],
}

const WATER_VERTS: &[SimpleVertex] = &[
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

const WATER_INDICES: &[u32] = &[0, 1, 2, 0, 2, 3];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SimpleVertex {
    pub pos: [f32; 3],
}

impl SimpleVertex {
    pub(crate) fn layout() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: size_of::<SimpleVertex>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: &[VertexAttribute {
                format: VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            }],
        }
    }
}
