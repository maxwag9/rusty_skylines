use crate::renderer::ui_editor::UiLayer;
use serde::Deserialize;
use wgpu::{
    Buffer, BufferAddress, VertexAttribute, VertexBufferLayout, VertexStepMode, vertex_attr_array,
};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LineVtx {
    pub(crate) pos: [f32; 3],
    pub(crate) color: [f32; 3],
}

impl LineVtx {
    pub(crate) fn layout<'a>() -> VertexBufferLayout<'a> {
        const ATTRS: &[VertexAttribute] = &vertex_attr_array![0 => Float32x3, 1 => Float32x3];
        VertexBufferLayout {
            array_stride: size_of::<LineVtx>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: ATTRS,
        }
    }
}

pub struct LayerGpu {
    pub circle_ssbo: Option<Buffer>,
    pub circle_count: u32,

    pub poly_vbo: Option<Buffer>, // rectangles + triangles + polygons in one stream
    pub poly_count: u32,          // vertex count

    pub text_vbo: Option<Buffer>, // UiVertexText stream
    pub text_count: u32,
}

impl Default for LayerGpu {
    fn default() -> Self {
        Self {
            circle_ssbo: None,
            circle_count: 0,
            poly_vbo: None,
            poly_count: 0,
            text_vbo: None,
            text_count: 0,
        }
    }
}
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Vertex {
    pub(crate) position: [f32; 3],
    pub(crate) color: [f32; 3],
}

impl Vertex {
    const ATTRIBS: [VertexAttribute; 2] = vertex_attr_array![
        0 => Float32x3, // now 3 floats for position
        1 => Float32x3
    ];

    pub(crate) fn desc<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            array_stride: size_of::<Vertex>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct UiVertexPoly {
    pub pos: [f32; 2],
    pub _pad: [f32; 2], // pad to 16 bytes
    pub color: [f32; 4],
    pub misc: [f32; 4],
}
impl UiVertexPoly {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: 48,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

// For text — pos + uv + color
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiVertexText {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}
impl UiVertexText {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem::size_of;
        wgpu::VertexBufferLayout {
            array_stride: size_of::<UiVertexText>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: size_of::<[f32; 2]>() as _,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: (size_of::<[f32; 2]>() * 2) as _,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

#[derive(Deserialize, Debug, Clone, Copy)]
pub struct UiVertex {
    pub pos: [f32; 2],
    pub color: [f32; 4],
    pub roundness: f32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct GlowMisc {
    pub glow_size: f32,
    pub glow_speed: f32,
    pub glow_intensity: f32,
}
#[derive(Deserialize, Debug, Clone)]
pub struct MiscButtonSettings {
    pub active: bool,
    pub touched_time: f32,
    pub is_touched: bool,
}
#[derive(Deserialize, Debug)]
pub struct GuiLayout {
    pub layers: Vec<UiLayer>,
}

// --- all possible button shapes ---
#[derive(Deserialize, Debug, Clone)]
pub struct UiButtonText {
    pub id: Option<String>,
    pub x: f32,
    pub y: f32,
    pub stretch_x: f32,
    pub stretch_y: f32,
    pub top_left_vertex: UiVertex,
    pub bottom_left_vertex: UiVertex,
    pub top_right_vertex: UiVertex,
    pub bottom_right_vertex: UiVertex,
    pub px: u16,
    pub color: [f32; 4],
    pub text: String,
    pub misc: MiscButtonSettings,
}

#[derive(Deserialize, Debug, Clone)]
pub struct UiButtonCircle {
    pub id: Option<String>,
    pub x: f32,
    pub y: f32,
    pub stretch_x: f32,
    pub stretch_y: f32,
    pub radius: f32,
    pub fill_color: [f32; 4],
    pub border_color: [f32; 4],
    pub glow_color: [f32; 4],
    pub glow_misc: GlowMisc,
    pub misc: MiscButtonSettings,
}

#[derive(Deserialize, Debug, Clone)]
pub struct UiButtonRectangle {
    pub id: Option<String>,
    pub top_left_vertex: UiVertex,
    pub bottom_left_vertex: UiVertex,
    pub top_right_vertex: UiVertex,
    pub bottom_right_vertex: UiVertex,
    pub misc: MiscButtonSettings,
}

#[derive(Deserialize, Debug, Clone)]
pub struct UiButtonPolygon {
    pub id: Option<String>,
    pub(crate) vertices: Vec<UiVertex>,
    pub misc: MiscButtonSettings,
}

#[derive(Deserialize, Debug, Clone)]
pub struct UiButtonTriangle {
    pub id: Option<String>,
    pub top_vertex: UiVertex,
    pub left_vertex: UiVertex,
    pub right_vertex: UiVertex,
    pub misc: MiscButtonSettings,
}
