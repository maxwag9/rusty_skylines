use crate::renderer::helper::ensure_ccw;
use crate::renderer::ui::{CircleParams, HandleParams, OutlineParams, TextParams};
use serde::{Deserialize, Serialize};
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

#[derive(Debug)]
pub struct LayerGpu {
    pub circle_ssbo: Option<Buffer>,
    pub circle_count: u32,

    pub outline_poly_vertices_ssbo: Option<Buffer>,
    pub outline_shapes_ssbo: Option<Buffer>,
    pub outline_count: u32,

    pub handle_ssbo: Option<Buffer>,
    pub handle_count: u32,

    pub poly_vbo: Option<Buffer>, // polygons, I know, right??
    pub poly_count: u32,          // vertex count

    pub text_vbo: Option<Buffer>, // UiVertexText stream
    pub text_count: u32,
}

impl Default for LayerGpu {
    fn default() -> Self {
        Self {
            circle_ssbo: None,
            circle_count: 0,
            outline_poly_vertices_ssbo: None,
            outline_shapes_ssbo: None,
            outline_count: 0,
            handle_ssbo: None,
            handle_count: 0,
            poly_vbo: None,
            poly_count: 0,
            text_vbo: None,
            text_count: 0,
        }
    }
}

pub enum TouchState {
    Pressed,
    Held,
    Released,
    Idle,
}

#[derive(Debug)]
pub struct LayerCache {
    pub texts: Vec<TextParams>,
    pub circle_params: Vec<CircleParams>,
    pub outline_params: Vec<OutlineParams>,
    pub handle_params: Vec<HandleParams>,
    pub polygon_vertices: Vec<UiVertexPoly>,
    pub outline_poly_vertices: Vec<[f32; 2]>,
}

impl Default for LayerCache {
    fn default() -> Self {
        Self {
            texts: vec![],
            circle_params: vec![],
            outline_params: vec![],
            handle_params: vec![],
            polygon_vertices: vec![],
            outline_poly_vertices: vec![],
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SelectedUiElement {
    pub layer_name: String,
    pub element_id: String,
    pub active: bool,
}

#[derive(Clone, Debug)]
pub enum UiElement {
    Circle(UiButtonCircle),
    Handle(UiButtonHandle),
    Polygon(UiButtonPolygon),
    Text(UiButtonText),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ButtonRuntime {
    pub touched_time: f32,
    pub is_down: bool,
    pub just_pressed: bool,
    pub just_released: bool,
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

#[derive(Debug)]
pub struct RuntimeLayer {
    pub name: String,
    pub order: u32,
    pub texts: Vec<UiButtonText>,
    pub circles: Vec<UiButtonCircle>,
    pub outlines: Vec<UiButtonOutline>,
    pub handles: Vec<UiButtonHandle>,
    pub polygons: Vec<UiButtonPolygon>,
    pub active: bool,
    // NEW: cached GPU data!!!
    pub cache: LayerCache,

    pub dirty: bool, // set true when anything changes or the screen will be dirty asf!
    pub gpu: LayerGpu,
    pub opaque: bool,
}

#[derive(Debug, serde::Deserialize)]
pub struct UiLayer {
    pub name: String,
    pub order: u32,
    pub texts: Option<Vec<UiButtonText>>,
    pub circles: Option<Vec<UiButtonCircle>>,
    pub outlines: Option<Vec<UiButtonOutline>>,
    pub handles: Option<Vec<UiButtonHandle>>,
    pub polygons: Option<Vec<UiButtonPolygon>>,
    pub active: Option<bool>,
    pub opaque: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct UiLayerJson {
    pub name: String,
    pub order: u32,
    pub texts: Option<Vec<UiButtonTextJson>>,
    pub circles: Option<Vec<UiButtonCircleJson>>,
    pub outlines: Option<Vec<UiButtonOutlineJson>>,
    pub handles: Option<Vec<UiButtonHandleJson>>,
    pub polygons: Option<Vec<UiButtonPolygonJson>>,
    pub active: Option<bool>,
    pub opaque: Option<bool>,
}

#[derive(Deserialize, Debug, Clone, Copy)]
pub struct UiVertex {
    pub pos: [f32; 2],
    pub color: [f32; 4],
    pub roundness: f32,
    pub selected: bool,
    pub id: usize,
}

impl UiVertex {
    fn from_json(v: UiVertexJson, id: usize) -> Self {
        UiVertex {
            pos: v.pos,
            color: v.color,
            roundness: v.roundness,
            selected: false,
            id,
        }
    }

    pub fn to_json(&self) -> UiVertexJson {
        UiVertexJson {
            pos: self.pos,
            color: self.color,
            roundness: self.roundness,
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy)]
pub struct UiVertexJson {
    pub pos: [f32; 2],
    pub color: [f32; 4],
    pub roundness: f32,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct GlowMisc {
    pub glow_size: f32,
    pub glow_speed: f32,
    pub glow_intensity: f32,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct DashMisc {
    pub dash_len: f32,
    pub dash_spacing: f32,
    pub dash_roundness: f32,
    pub dash_speed: f32,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ShapeData {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub border_thickness: f32,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct HandleMisc {
    pub handle_len: f32,
    pub handle_width: f32,
    pub handle_roundness: f32,
    pub handle_speed: f32,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MiscButtonSettings {
    pub active: bool,
    pub touched_time: f32,
    pub is_touched: bool,
    pub pressable: bool,
    pub editable: bool,
}

impl MiscButtonSettings {
    pub fn to_json(&self) -> MiscButtonSettingsJson {
        MiscButtonSettingsJson {
            active: self.active,
            pressable: self.pressable,
            editable: self.editable,
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MiscButtonSettingsJson {
    pub active: bool,
    pub pressable: bool,
    pub editable: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GuiLayout {
    pub layers: Vec<UiLayerJson>,
}

// --- all possible button shapes ---
#[derive(Deserialize, Debug, Clone)]
pub struct UiButtonText {
    pub id: Option<String>,
    pub z_index: i32,
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
pub struct UiButtonPolygon {
    pub id: Option<String>,
    pub z_index: i32,
    pub vertices: Vec<UiVertex>,
    pub misc: MiscButtonSettings,
    pub tri_count: u32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct UiButtonCircle {
    pub id: Option<String>,
    pub z_index: i32,

    pub x: f32,
    pub y: f32,
    pub stretch_x: f32,
    pub stretch_y: f32,
    pub radius: f32,
    pub border_thickness: f32,
    pub fade: [f32; 4],
    pub fill_color: [f32; 4],
    pub border_color: [f32; 4],
    pub glow_color: [f32; 4],
    pub glow_misc: GlowMisc,
    pub misc: MiscButtonSettings,
}

#[derive(Deserialize, Debug, Clone)]
pub struct UiButtonOutline {
    pub id: Option<String>,
    pub z_index: i32,
    pub parent_id: Option<String>,

    pub mode: f32, // 0 = circle, 1 = polygon

    pub vertex_offset: u32,    // index into global vertex buffer
    pub vertex_count: u32,     // how many vertices
    pub shape_data: ShapeData, // cx, cy, radius, thickness OR unused for poly

    pub dash_color: [f32; 4],
    pub dash_misc: DashMisc,
    pub sub_dash_color: [f32; 4],
    pub sub_dash_misc: DashMisc,

    pub misc: MiscButtonSettings,
}

#[derive(Deserialize, Debug, Clone)]
pub struct UiButtonHandle {
    pub id: Option<String>,
    pub z_index: i32,
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub handle_thickness: f32,
    pub handle_color: [f32; 4],
    pub handle_misc: HandleMisc,
    pub sub_handle_color: [f32; 4],
    pub sub_handle_misc: HandleMisc,
    pub misc: MiscButtonSettings,
    pub parent_id: Option<String>,
}

impl UiButtonText {
    pub(crate) fn from_json(t: UiButtonTextJson) -> Self {
        UiButtonText {
            id: t.id,
            z_index: t.z_index,

            x: t.x,
            y: t.y,
            stretch_x: t.stretch_x,
            stretch_y: t.stretch_y,
            top_left_vertex: UiVertex {
                pos: t.top_left_vertex.pos,
                color: t.top_left_vertex.color,
                roundness: t.top_left_vertex.roundness,
                selected: false,
                id: 0,
            },
            bottom_left_vertex: UiVertex {
                pos: t.bottom_left_vertex.pos,
                color: t.bottom_left_vertex.color,
                roundness: t.bottom_left_vertex.roundness,
                selected: false,
                id: 0,
            },
            top_right_vertex: UiVertex {
                pos: t.top_right_vertex.pos,
                color: t.top_right_vertex.color,
                roundness: t.top_right_vertex.roundness,
                selected: false,
                id: 0,
            },
            bottom_right_vertex: UiVertex {
                pos: t.bottom_right_vertex.pos,
                color: t.bottom_right_vertex.color,
                roundness: t.bottom_right_vertex.roundness,
                selected: false,
                id: 0,
            },
            px: t.px,
            color: t.color,
            text: t.text,
            misc: MiscButtonSettings {
                active: t.misc.active,
                touched_time: 0.0,
                is_touched: false,
                pressable: t.misc.pressable,
                editable: t.misc.editable,
            },
        }
    }

    pub fn to_json(&self) -> UiButtonTextJson {
        UiButtonTextJson {
            id: self.id.clone(),
            z_index: self.z_index,

            x: self.x,
            y: self.y,
            stretch_x: self.stretch_x,
            stretch_y: self.stretch_y,

            top_left_vertex: self.top_left_vertex.to_json(),
            bottom_left_vertex: self.bottom_left_vertex.to_json(),
            top_right_vertex: self.top_right_vertex.to_json(),
            bottom_right_vertex: self.bottom_right_vertex.to_json(),

            px: self.px,
            color: self.color,
            text: self.text.clone(),
            misc: self.misc.to_json(),
        }
    }
}

impl UiButtonCircle {
    pub(crate) fn from_json(c: UiButtonCircleJson) -> Self {
        UiButtonCircle {
            id: c.id,

            z_index: c.z_index,
            x: c.x,
            y: c.y,
            stretch_x: c.stretch_x,
            stretch_y: c.stretch_y,
            radius: c.radius,
            border_thickness: c.border_thickness,
            fade: [0.0; 4],
            fill_color: c.fill_color,
            border_color: c.border_color,
            glow_color: c.glow_color,
            glow_misc: c.glow_misc,
            misc: MiscButtonSettings {
                active: c.misc.active,
                touched_time: 0.0,
                is_touched: false,
                pressable: c.misc.pressable,
                editable: c.misc.editable,
            },
        }
    }

    pub fn to_json(&self) -> UiButtonCircleJson {
        UiButtonCircleJson {
            id: self.id.clone(),
            z_index: self.z_index,

            x: self.x,
            y: self.y,
            stretch_x: self.stretch_x,
            stretch_y: self.stretch_y,

            radius: self.radius,
            border_thickness: self.border_thickness,

            fill_color: self.fill_color,
            border_color: self.border_color,
            glow_color: self.glow_color,
            glow_misc: self.glow_misc.clone(),

            misc: self.misc.to_json(),
        }
    }
}

impl UiButtonHandle {
    pub(crate) fn from_json(h: UiButtonHandleJson) -> Self {
        UiButtonHandle {
            id: h.id,
            z_index: h.z_index,
            x: h.x,
            y: h.y,
            radius: h.radius,
            handle_thickness: h.handle_thickness,
            handle_color: h.handle_color,
            handle_misc: h.handle_misc,
            sub_handle_color: h.sub_handle_color,
            sub_handle_misc: h.sub_handle_misc,
            parent_id: h.parent_id,
            misc: MiscButtonSettings {
                active: h.misc.active,
                touched_time: 0.0,
                is_touched: false,
                pressable: h.misc.pressable,
                editable: h.misc.editable,
            },
        }
    }

    pub fn to_json(&self) -> UiButtonHandleJson {
        UiButtonHandleJson {
            id: self.id.clone(),
            z_index: self.z_index,
            x: self.x,
            y: self.y,
            radius: self.radius,

            handle_thickness: self.handle_thickness,
            handle_color: self.handle_color,
            handle_misc: self.handle_misc.clone(),

            sub_handle_color: self.sub_handle_color,
            sub_handle_misc: self.sub_handle_misc.clone(),

            parent_id: self.parent_id.clone(),
            misc: self.misc.to_json(),
        }
    }
}

impl UiButtonOutline {
    pub(crate) fn from_json(o: UiButtonOutlineJson) -> Self {
        UiButtonOutline {
            id: o.id,

            z_index: o.z_index,
            parent_id: o.parent_id,
            mode: o.mode,
            vertex_offset: 0,
            vertex_count: 0,
            shape_data: o.shape_data,
            dash_color: o.dash_color,
            dash_misc: o.dash_misc,
            sub_dash_color: o.sub_dash_color,
            sub_dash_misc: o.sub_dash_misc,
            misc: MiscButtonSettings {
                active: o.misc.active,
                touched_time: 0.0,
                is_touched: false,
                pressable: o.misc.pressable,
                editable: o.misc.editable,
            },
        }
    }

    pub fn to_json(&self) -> UiButtonOutlineJson {
        UiButtonOutlineJson {
            id: self.id.clone(),
            z_index: self.z_index,
            parent_id: self.parent_id.clone(),

            mode: self.mode,
            shape_data: self.shape_data.clone(),

            dash_color: self.dash_color,
            dash_misc: self.dash_misc.clone(),
            sub_dash_color: self.sub_dash_color,
            sub_dash_misc: self.sub_dash_misc.clone(),

            misc: self.misc.to_json(),
        }
    }
}

impl UiButtonPolygon {
    pub(crate) fn from_json(p: UiButtonPolygonJson, id_gen: &mut usize) -> Self {
        let mut verts: Vec<UiVertex> = p
            .vertices
            .into_iter()
            .map(|vj| {
                let id = *id_gen;
                *id_gen += 1;
                UiVertex::from_json(vj, id)
            })
            .collect();

        ensure_ccw(&mut verts);

        UiButtonPolygon {
            id: p.id,
            z_index: p.z_index,
            vertices: verts,
            misc: MiscButtonSettings {
                active: p.misc.active,
                touched_time: 0.0,
                is_touched: false,
                pressable: p.misc.pressable,
                editable: p.misc.editable,
            },
            tri_count: 0,
        }
    }

    pub fn to_json(&self) -> UiButtonPolygonJson {
        UiButtonPolygonJson {
            id: self.id.clone(),
            z_index: self.z_index,

            vertices: self.vertices.iter().map(|v| v.to_json()).collect(),

            misc: self.misc.to_json(),
        }
    }
}

impl Default for UiButtonText {
    fn default() -> Self {
        Self {
            id: None,
            z_index: 0,
            x: 0.0,
            y: 0.0,
            stretch_x: 1.0,
            stretch_y: 1.0,
            top_left_vertex: UiVertex::default(),
            bottom_left_vertex: UiVertex::default(),
            top_right_vertex: UiVertex::default(),
            bottom_right_vertex: UiVertex::default(),
            px: 14,
            color: [1.0, 1.0, 1.0, 1.0],
            text: "".into(),
            misc: MiscButtonSettings::default(),
        }
    }
}

impl Default for UiButtonPolygon {
    fn default() -> Self {
        let verts = vec![
            UiVertex {
                pos: [-30.0, 30.0],
                color: [1.0, 1.0, 1.0, 1.0],
                roundness: 0.0,
                selected: false,
                id: 0,
            },
            UiVertex {
                pos: [0.0, -30.0],
                color: [1.0, 1.0, 1.0, 1.0],
                roundness: 0.0,
                selected: false,
                id: 1,
            },
            UiVertex {
                pos: [30.0, 30.0],
                color: [1.0, 1.0, 1.0, 1.0],
                roundness: 0.0,
                selected: false,
                id: 2,
            },
            UiVertex {
                pos: [30.0, 50.0],
                color: [1.0, 1.0, 0.0, 1.0],
                roundness: 0.0,
                selected: false,
                id: 3,
            },
            UiVertex {
                pos: [50.0, 30.0],
                color: [1.0, 0.0, 1.0, 1.0],
                roundness: 0.0,
                selected: false,
                id: 5,
            },
        ];

        Self {
            id: None,
            z_index: 0,
            vertices: verts,
            misc: MiscButtonSettings::default(),
            tri_count: 0,
        }
    }
}

impl Default for UiButtonCircle {
    fn default() -> Self {
        Self {
            id: None,
            z_index: 0,
            x: 0.0,
            y: 0.0,
            stretch_x: 1.0,
            stretch_y: 1.0,
            radius: 10.0,
            border_thickness: 1.0,
            fade: [0.0; 4],
            fill_color: [1.0, 1.0, 1.0, 1.0],
            border_color: [0.0, 0.0, 0.0, 1.0],
            glow_color: [1.0, 1.0, 1.0, 0.0],
            glow_misc: GlowMisc::default(),
            misc: MiscButtonSettings::default(),
        }
    }
}

impl Default for UiButtonOutline {
    fn default() -> Self {
        Self {
            id: None,
            z_index: 0,
            parent_id: None,
            mode: 1.0,
            vertex_offset: 0,
            vertex_count: 0,
            shape_data: ShapeData::default(),
            dash_color: [1.0, 1.0, 1.0, 1.0],
            dash_misc: DashMisc::default(),
            sub_dash_color: [1.0, 1.0, 1.0, 1.0],
            sub_dash_misc: DashMisc::default(),
            misc: MiscButtonSettings::default(),
        }
    }
}

impl Default for UiButtonHandle {
    fn default() -> Self {
        Self {
            id: None,
            z_index: 0,
            x: 0.0,
            y: 0.0,
            radius: 6.0,
            handle_thickness: 2.0,
            handle_color: [1.0, 1.0, 1.0, 1.0],
            handle_misc: HandleMisc::default(),
            sub_handle_color: [1.0, 1.0, 1.0, 1.0],
            sub_handle_misc: HandleMisc::default(),
            misc: MiscButtonSettings::default(),
            parent_id: None,
        }
    }
}

impl Default for GlowMisc {
    fn default() -> Self {
        Self {
            glow_size: 0.0,
            glow_speed: 0.0,
            glow_intensity: 0.0,
        }
    }
}

impl Default for DashMisc {
    fn default() -> Self {
        Self {
            dash_len: 20.0,
            dash_spacing: 10.0,
            dash_roundness: 0.0,
            dash_speed: 0.0,
        }
    }
}

impl Default for ShapeData {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            radius: 10.0,
            border_thickness: 1.0,
        }
    }
}

impl Default for HandleMisc {
    fn default() -> Self {
        Self {
            handle_len: 10.0,
            handle_width: 2.0,
            handle_roundness: 0.0,
            handle_speed: 0.0,
        }
    }
}

impl Default for MiscButtonSettings {
    fn default() -> Self {
        Self {
            active: true,
            touched_time: 0.0,
            is_touched: false,
            pressable: true,
            editable: false,
        }
    }
}

impl Default for UiVertex {
    fn default() -> Self {
        Self {
            pos: [0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            roundness: 0.0,
            selected: false,
            id: 0,
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct UiButtonTextJson {
    pub id: Option<String>,
    pub z_index: i32,

    pub x: f32,
    pub y: f32,
    pub stretch_x: f32,
    pub stretch_y: f32,
    pub top_left_vertex: UiVertexJson,
    pub bottom_left_vertex: UiVertexJson,
    pub top_right_vertex: UiVertexJson,
    pub bottom_right_vertex: UiVertexJson,
    pub px: u16,
    pub color: [f32; 4],
    pub text: String,
    pub misc: MiscButtonSettingsJson,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct UiButtonCircleJson {
    pub id: Option<String>,
    pub z_index: i32,
    pub x: f32,
    pub y: f32,
    pub stretch_x: f32,
    pub stretch_y: f32,
    pub radius: f32,
    pub border_thickness: f32,
    pub fill_color: [f32; 4],
    pub border_color: [f32; 4],
    pub glow_color: [f32; 4],
    pub glow_misc: GlowMisc,
    pub misc: MiscButtonSettingsJson,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct UiButtonHandleJson {
    pub id: Option<String>,
    pub z_index: i32,
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub handle_thickness: f32,
    pub handle_color: [f32; 4],
    pub handle_misc: HandleMisc,
    pub sub_handle_color: [f32; 4],
    pub sub_handle_misc: HandleMisc,
    pub misc: MiscButtonSettingsJson,
    pub parent_id: Option<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct UiButtonOutlineJson {
    pub id: Option<String>,
    pub z_index: i32,
    pub parent_id: Option<String>,
    pub mode: f32,             // 0 = circle, 1 = polygon
    pub shape_data: ShapeData, // cx, cy, radius, thickness OR unused for poly

    pub dash_color: [f32; 4],
    pub dash_misc: DashMisc,
    pub sub_dash_color: [f32; 4],
    pub sub_dash_misc: DashMisc,

    pub misc: MiscButtonSettingsJson,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct UiButtonPolygonJson {
    pub id: Option<String>,
    pub z_index: i32,
    pub vertices: Vec<UiVertexJson>,
    pub misc: MiscButtonSettingsJson,
}

pub fn ensure_id<T: UiElementId>(mut item: T) -> T {
    if item.get_id().is_none() {
        item.set_id(Some(gen_id()));
    }
    item
}

pub fn gen_id() -> String {
    nanoid::nanoid!(8) // 8 chars is enough, change if you want longer
}

pub trait UiElementId {
    fn get_id(&self) -> &Option<String>;
    fn set_id(&mut self, id: Option<String>);
}

impl UiElementId for UiButtonCircle {
    fn get_id(&self) -> &Option<String> {
        &self.id
    }
    fn set_id(&mut self, id: Option<String>) {
        self.id = id;
    }
}

impl UiElementId for UiButtonPolygon {
    fn get_id(&self) -> &Option<String> {
        &self.id
    }
    fn set_id(&mut self, id: Option<String>) {
        self.id = id;
    }
}

impl UiElementId for UiButtonText {
    fn get_id(&self) -> &Option<String> {
        &self.id
    }
    fn set_id(&mut self, id: Option<String>) {
        self.id = id;
    }
}

impl UiElementId for UiButtonHandle {
    fn get_id(&self) -> &Option<String> {
        &self.id
    }
    fn set_id(&mut self, id: Option<String>) {
        self.id = id;
    }
}

impl UiElementId for UiButtonOutline {
    fn get_id(&self) -> &Option<String> {
        &self.id
    }
    fn set_id(&mut self, id: Option<String>) {
        self.id = id;
    }
}
