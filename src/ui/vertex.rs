use crate::renderer::ui::{CircleParams, HandleParams, OutlineParams, TextParams};
use crate::renderer::ui_text::Anchor;
use crate::ui::helper::ensure_ccw;
use crate::ui::ui_editor::UiVariableRegistry;
use serde::{Deserialize, Serialize};
use std::mem::size_of;
use wgpu::{vertex_attr_array, *};
use winit::dpi::PhysicalSize;

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
    pub poly_info_ssbo: Option<Buffer>,
    pub poly_edge_ssbo: Option<Buffer>,

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
            poly_info_ssbo: None,
            poly_edge_ssbo: None,
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

#[derive(Debug, Clone, Copy)]
pub struct LayerDirty {
    pub texts: bool,
    pub circles: bool,
    pub outlines: bool,
    pub handles: bool,
    pub polygons: bool,
}

impl LayerDirty {
    pub fn all() -> Self {
        Self {
            texts: true,
            circles: true,
            outlines: true,
            handles: true,
            polygons: true,
        }
    }

    pub fn none() -> Self {
        Self {
            texts: false,
            circles: false,
            outlines: false,
            handles: false,
            polygons: false,
        }
    }

    pub fn any(self) -> bool {
        self.texts || self.circles || self.outlines || self.handles || self.polygons
    }

    pub fn mark_texts(&mut self) {
        self.texts = true;
    }

    pub fn mark_circles(&mut self) {
        self.circles = true;
    }

    pub fn mark_outlines(&mut self) {
        self.outlines = true;
    }

    pub fn mark_handles(&mut self) {
        self.handles = true;
    }

    pub fn mark_polygons(&mut self) {
        self.polygons = true;
    }

    pub fn mark_all(&mut self) {
        *self = Self::all();
    }

    pub fn clear(&mut self, done: LayerDirty) {
        if done.texts {
            self.texts = false;
        }
        if done.circles {
            self.circles = false;
        }
        if done.outlines {
            self.outlines = false;
        }
        if done.handles {
            self.handles = false;
        }
        if done.polygons {
            self.polygons = false;
        }
    }
}

impl Default for LayerDirty {
    fn default() -> Self {
        Self::all()
    }
}

#[derive(Debug, Clone)]
pub enum UiElement {
    Circle(UiButtonCircle),
    Handle(UiButtonHandle),
    Polygon(UiButtonPolygon),
    Text(UiButtonText),
    Outline(UiButtonOutline),
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
    pub normal: [f32; 3],
    pub(crate) color: [f32; 3],
}

impl Vertex {
    pub fn desc() -> VertexBufferLayout<'static> {
        use std::mem::size_of;
        VertexBufferLayout {
            array_stride: size_of::<Vertex>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &[
                // @location(0) position
                VertexAttribute {
                    shader_location: 0,
                    offset: 0,
                    format: VertexFormat::Float32x3,
                },
                // @location(1) normal
                VertexAttribute {
                    shader_location: 1,
                    offset: 12,
                    format: VertexFormat::Float32x3,
                },
                // @location(2) color
                VertexAttribute {
                    shader_location: 2,
                    offset: 24,
                    format: VertexFormat::Float32x3,
                },
            ],
        }
    }
}
pub trait VertexWithPosition {
    fn position(&self) -> [f32; 3];
    fn lerp(a: &Self, b: &Self, t: f32) -> Self;
}

impl VertexWithPosition for Vertex {
    fn position(&self) -> [f32; 3] {
        self.position
    }

    // produce a vertex that is a linear interpolation between a and b with factor t in [0,1]
    // must interpolate all vertex attributes consistently (position, normal, color).
    fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        // Linear interpolation helper
        fn mix(x: [f32; 3], y: [f32; 3], t: f32) -> [f32; 3] {
            [
                x[0] + (y[0] - x[0]) * t,
                x[1] + (y[1] - x[1]) * t,
                x[2] + (y[2] - x[2]) * t,
            ]
        }

        let position = mix(a.position, b.position, t);
        let mut normal = mix(a.normal, b.normal, t);
        let color = mix(a.color, b.color, t);

        // Normalize normal
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if len > 0.0 {
            normal[0] /= len;
            normal[1] /= len;
            normal[2] /= len;
        }

        Vertex {
            position,
            normal,
            color,
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct UiVertexPoly {
    pub pos: [f32; 2],
    pub data: [f32; 2], // [roundness_px, polygon_index]
    pub color: [f32; 4],
    pub misc: [f32; 4], // active, touched_time, is_touched, hash
}

impl UiVertexPoly {
    pub fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: size_of::<UiVertexPoly>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: &[
                VertexAttribute {
                    shader_location: 0,
                    format: VertexFormat::Float32x2,
                    offset: 0,
                },
                VertexAttribute {
                    shader_location: 1,
                    format: VertexFormat::Float32x2,
                    offset: 8,
                },
                VertexAttribute {
                    shader_location: 2,
                    format: VertexFormat::Float32x4,
                    offset: 16,
                },
                VertexAttribute {
                    shader_location: 3,
                    format: VertexFormat::Float32x4,
                    offset: 32,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PolygonInfoGpu {
    pub edge_offset: u32,
    pub edge_count: u32,
    pub _pad0: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PolygonEdgeGpu {
    pub p0: [f32; 2],
    pub p1: [f32; 2],
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
    pub fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: size_of::<UiVertexText>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &[
                VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: VertexFormat::Float32x2,
                },
                VertexAttribute {
                    offset: size_of::<[f32; 2]>() as _,
                    shader_location: 1,
                    format: VertexFormat::Float32x2,
                },
                VertexAttribute {
                    offset: (size_of::<[f32; 2]>() * 2) as _,
                    shader_location: 2,
                    format: VertexFormat::Float32x4,
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

    pub dirty: LayerDirty, // set true when anything changes or the screen will be dirty asf!
    pub gpu: LayerGpu,
    pub opaque: bool,
    pub saveable: bool,
}

pub enum UiElementRef<'a> {
    Text(&'a UiButtonText),
    Circle(&'a UiButtonCircle),
    Outline(&'a UiButtonOutline),
    Handle(&'a UiButtonHandle),
    Polygon(&'a UiButtonPolygon),
}

impl<'a> UiElementRef<'a> {
    pub fn id(&self) -> &str {
        match self {
            UiElementRef::Text(t) => t.id.as_deref().unwrap_or(""),
            UiElementRef::Circle(c) => c.id.as_deref().unwrap_or(""),
            UiElementRef::Outline(o) => o.id.as_deref().unwrap_or(""),
            UiElementRef::Handle(h) => h.id.as_deref().unwrap_or(""),
            UiElementRef::Polygon(p) => p.id.as_deref().unwrap_or(""),
        }
    }

    pub fn center(&self) -> (f32, f32) {
        match self {
            UiElementRef::Text(t) => (t.x, t.y),
            UiElementRef::Circle(c) => (c.x, c.y),
            UiElementRef::Handle(h) => (h.x, h.y),
            UiElementRef::Outline(o) => (o.shape_data.x, o.shape_data.y),
            UiElementRef::Polygon(p) => {
                let count = p.vertices.len().max(1);
                let sum = p
                    .vertices
                    .iter()
                    .fold((0.0, 0.0), |acc, v| (acc.0 + v.pos[0], acc.1 + v.pos[1]));
                (sum.0 / count as f32, sum.1 / count as f32)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementKind {
    Text,
    Circle,
    Outline,
    Handle,
    Polygon,
    None,
}

impl std::fmt::Display for ElementKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
pub enum UiElementMut<'a> {
    Text(&'a mut UiButtonText),
    Circle(&'a mut UiButtonCircle),
    Outline(&'a mut UiButtonOutline),
    Handle(&'a mut UiButtonHandle),
    Polygon(&'a mut UiButtonPolygon),
}

impl<'a> UiElementMut<'a> {
    pub fn id(&self) -> &str {
        match self {
            UiElementMut::Text(t) => t.id.as_deref().unwrap_or(""),
            UiElementMut::Circle(c) => c.id.as_deref().unwrap_or(""),
            UiElementMut::Outline(o) => o.id.as_deref().unwrap_or(""),
            UiElementMut::Handle(h) => h.id.as_deref().unwrap_or(""),
            UiElementMut::Polygon(p) => p.id.as_deref().unwrap_or(""),
        }
    }

    pub fn center(&self) -> (f32, f32) {
        match self {
            UiElementMut::Text(t) => (t.x, t.y),
            UiElementMut::Circle(c) => (c.x, c.y),
            UiElementMut::Handle(h) => (h.x, h.y),
            UiElementMut::Outline(o) => (o.shape_data.x, o.shape_data.y),
            UiElementMut::Polygon(p) => {
                let count = p.vertices.len().max(1);
                let sum = p
                    .vertices
                    .iter()
                    .fold((0.0, 0.0), |acc, v| (acc.0 + v.pos[0], acc.1 + v.pos[1]));
                (sum.0 / count as f32, sum.1 / count as f32)
            }
        }
    }

    pub fn bump_z(&mut self, delta: i32) {
        match self {
            UiElementMut::Text(t) => t.z_index += delta,
            UiElementMut::Circle(c) => c.z_index += delta,
            UiElementMut::Outline(o) => o.z_index += delta,
            UiElementMut::Handle(h) => h.z_index += delta,
            UiElementMut::Polygon(p) => p.z_index += delta,
        }
    }
    pub fn z_index(&self) -> i32 {
        match self {
            UiElementMut::Text(t) => t.z_index,
            UiElementMut::Circle(c) => c.z_index,
            UiElementMut::Outline(o) => o.z_index,
            UiElementMut::Handle(h) => h.z_index,
            UiElementMut::Polygon(p) => p.z_index,
        }
    }
    pub fn update_z(&mut self, delta: i32, variables: &mut UiVariableRegistry) {
        self.bump_z(delta);
        variables.set_i32("selected_element.z_index", self.z_index());
    }
    pub fn translate(&mut self, dx: f32, dy: f32) {
        match self {
            UiElementMut::Text(t) => {
                t.x += dx;
                t.y += dy;
            }
            UiElementMut::Circle(c) => {
                c.x += dx;
                c.y += dy;
            }
            UiElementMut::Handle(h) => {
                h.x += dx;
                h.y += dy;
            }
            UiElementMut::Outline(o) => {
                o.shape_data.x += dx;
                o.shape_data.y += dy;
            }
            UiElementMut::Polygon(p) => {
                for v in &mut p.vertices {
                    v.pos[0] += dx;
                    v.pos[1] += dy;
                }
            }
        }
    }

    pub fn resize(&mut self, scale: f32) {
        match self {
            UiElementMut::Text(t) => {
                // preserve previous intent: px scaled as float then cast back to u16
                t.px = ((t.px as f32) * scale).round() as u16;

                println!("Text size: {}", t.px)
            }
            UiElementMut::Circle(c) => {
                c.radius *= scale;
            }
            UiElementMut::Outline(_) | UiElementMut::Handle(_) => {
                // outlines and handles have no generic resize behaviour here.
                // If you want, implement per-field scaling.
            }
            UiElementMut::Polygon(p) => {
                // compute centroid
                let mut cx = 0.0f32;
                let mut cy = 0.0f32;
                let count = p.vertices.len() as f32;
                if count > 0.0 {
                    for v in &p.vertices {
                        cx += v.pos[0];
                        cy += v.pos[1];
                    }
                    cx /= count;
                    cy /= count;

                    for v in &mut p.vertices {
                        v.pos[0] = cx + (v.pos[0] - cx) * scale;
                        v.pos[1] = cy + (v.pos[1] - cy) * scale;
                    }
                }
            }
        }
    }
}
impl RuntimeLayer {
    pub fn iter_all_elements(&self) -> impl Iterator<Item = (UiElementRef<'_>, ElementKind)> {
        self.texts
            .iter()
            .map(|t| (UiElementRef::Text(t), ElementKind::Text))
            .chain(
                self.circles
                    .iter()
                    .map(|c| (UiElementRef::Circle(c), ElementKind::Circle)),
            )
            .chain(
                self.outlines
                    .iter()
                    .map(|o| (UiElementRef::Outline(o), ElementKind::Outline)),
            )
            .chain(
                self.handles
                    .iter()
                    .map(|h| (UiElementRef::Handle(h), ElementKind::Handle)),
            )
            .chain(
                self.polygons
                    .iter()
                    .map(|p| (UiElementRef::Polygon(p), ElementKind::Polygon)),
            )
    }

    pub fn iter_all_elements_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(UiElementMut<'_>, ElementKind),
    {
        for t in &mut self.texts {
            f(UiElementMut::Text(t), ElementKind::Text);
        }
        for c in &mut self.circles {
            f(UiElementMut::Circle(c), ElementKind::Circle);
        }
        for o in &mut self.outlines {
            f(UiElementMut::Outline(o), ElementKind::Outline);
        }
        for h in &mut self.handles {
            f(UiElementMut::Handle(h), ElementKind::Handle);
        }
        for p in &mut self.polygons {
            f(UiElementMut::Polygon(p), ElementKind::Polygon);
        }
    }
    pub fn sort_by_z(&mut self) {
        self.texts.sort_by_key(|e| e.z_index);
        self.circles.sort_by_key(|e| e.z_index);
        self.outlines.sort_by_key(|e| e.z_index);
        self.handles.sort_by_key(|e| e.z_index);
        self.polygons.sort_by_key(|e| e.z_index);
    }

    pub fn bump_element_z(&mut self, id: &str, delta: i32, variables: &mut UiVariableRegistry) {
        self.iter_all_elements_mut(|mut elem, _| {
            if elem.id() == id {
                elem.update_z(delta, variables);
            }
        });
    }

    // simplified bump_element_xy using the helper
    pub fn bump_element_xy(&mut self, id: &str, dx: f32, dy: f32) {
        if let Some(mut el) = self.find_element_mut(id) {
            el.translate(dx, dy);
        }
    }

    // simplified resize_element using the helper
    pub fn resize_element(&mut self, id: &str, scale: f32) {
        if let Some(mut el) = self.find_element_mut(id) {
            el.resize(scale);
        }
    }
    pub fn find_element_mut(&mut self, id: &str) -> Option<UiElementMut<'_>> {
        if let Some(e) = self.texts.iter_mut().find(|e| e.id.as_deref() == Some(id)) {
            return Some(UiElementMut::Text(e));
        }
        if let Some(e) = self
            .circles
            .iter_mut()
            .find(|e| e.id.as_deref() == Some(id))
        {
            return Some(UiElementMut::Circle(e));
        }
        if let Some(e) = self
            .outlines
            .iter_mut()
            .find(|e| e.id.as_deref() == Some(id))
        {
            return Some(UiElementMut::Outline(e));
        }
        if let Some(e) = self
            .handles
            .iter_mut()
            .find(|e| e.id.as_deref() == Some(id))
        {
            return Some(UiElementMut::Handle(e));
        }
        if let Some(e) = self
            .polygons
            .iter_mut()
            .find(|e| e.id.as_deref() == Some(id))
        {
            return Some(UiElementMut::Polygon(e));
        }
        None
    }
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
    fn from_json(v: UiVertexJson, id: usize, window_size: PhysicalSize<u32>) -> Self {
        let mut pos = v.pos;
        pos[0] *= window_size.width as f32;
        pos[1] *= window_size.height as f32;
        UiVertex {
            pos,
            color: v.color,
            roundness: v.roundness,
            selected: false,
            id,
        }
    }

    pub fn to_json(&self, window_size: PhysicalSize<u32>) -> UiVertexJson {
        let mut pos = self.pos;
        pos[0] /= window_size.width as f32;
        pos[1] /= window_size.height as f32;
        UiVertexJson {
            pos,
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

impl ShapeData {
    pub(crate) fn scale_from_normalized(
        &self,
        window_size: PhysicalSize<u32>,
        scale: f32,
    ) -> ShapeData {
        ShapeData {
            x: self.x * window_size.width as f32,
            y: self.y * window_size.height as f32,
            radius: self.radius * scale,
            border_thickness: self.border_thickness * scale,
        }
    }

    pub(crate) fn scale_to_normalized(
        &self,
        window_size: PhysicalSize<u32>,
        scale: f32,
    ) -> ShapeData {
        ShapeData {
            x: self.x / window_size.width as f32,
            y: self.y / window_size.height as f32,
            radius: self.radius / scale,
            border_thickness: self.border_thickness / scale,
        }
    }
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
    pub menus: Vec<MenuJson>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MenuJson {
    pub name: String,
    pub layers: Vec<UiLayerJson>,
}

// --- all possible button shapes ---
#[derive(Deserialize, Debug, Clone)]
pub struct UiButtonText {
    pub id: Option<String>,
    pub action: String,
    pub style: String,
    pub z_index: i32,
    pub x: f32,
    pub y: f32,
    pub top_left_offset: [f32; 2],
    pub bottom_left_offset: [f32; 2],
    pub top_right_offset: [f32; 2],
    pub bottom_right_offset: [f32; 2],
    pub px: u16,
    pub color: [f32; 4],
    pub text: String,
    pub template: String,
    pub misc: MiscButtonSettings,

    pub natural_width: f32,
    pub natural_height: f32,
    pub ascent: f32,
    pub being_edited: bool,
    pub caret: usize,
    pub being_hovered: bool,
    pub just_unhovered: bool,

    pub sel_start: usize, // selection start index
    pub sel_end: usize,   // selection end index
    pub has_selection: bool,
    pub glyph_bounds: Vec<(f32, f32)>,

    pub input_box: bool,
    pub anchor: Option<Anchor>,
}

impl UiButtonText {
    pub fn clear_selection(&mut self) {
        self.sel_start = self.caret;
        self.sel_end = self.caret;
        self.has_selection = false;
    }

    pub fn selection_range(&self) -> (usize, usize) {
        if !self.has_selection {
            return (self.caret, self.caret);
        }
        if self.sel_start <= self.sel_end {
            (self.sel_start, self.sel_end)
        } else {
            (self.sel_end, self.sel_start)
        }
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct UiButtonPolygon {
    pub id: Option<String>,
    pub action: String,
    pub style: String,
    pub z_index: i32,
    pub vertices: Vec<UiVertex>,
    pub misc: MiscButtonSettings,
    pub tri_count: u32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct UiButtonCircle {
    pub id: Option<String>,
    pub action: String,
    pub style: String,
    pub z_index: i32,

    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub inside_border_thickness: f32,
    pub border_thickness: f32,
    pub fade: f32,
    pub fill_color: [f32; 4],
    pub inside_border_color: [f32; 4],
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
    pub(crate) fn from_json(t: UiButtonTextJson, window_size: PhysicalSize<u32>) -> Self {
        let scale = (window_size.width as f32 * window_size.height as f32).sqrt();
        let length = t.text.len();
        UiButtonText {
            id: t.id,
            action: t.action.clone(),
            style: t.style.clone(),
            z_index: t.z_index,

            x: window_size.width as f32 * t.x,
            y: window_size.height as f32 * t.y,
            top_left_offset: [scale * t.top_left_offset[0], scale * t.top_left_offset[1]],
            bottom_left_offset: [
                scale * t.bottom_left_offset[0],
                scale * t.bottom_left_offset[1],
            ],
            top_right_offset: [scale * t.top_right_offset[0], scale * t.top_right_offset[1]],
            bottom_right_offset: [
                scale * t.bottom_right_offset[0],
                scale * t.bottom_right_offset[1],
            ],
            px: (scale * t.px) as u16,
            color: t.color,
            text: t.text.clone(),
            template: t.text,
            misc: MiscButtonSettings {
                active: t.misc.active,
                touched_time: 0.0,
                is_touched: false,
                pressable: t.misc.pressable,
                editable: t.misc.editable,
            },
            natural_width: 50.0,
            natural_height: 20.0,
            ascent: 10.0,
            being_edited: false,
            caret: length,
            being_hovered: false,
            just_unhovered: false,
            sel_start: 0,
            sel_end: 0,
            has_selection: false,
            glyph_bounds: vec![],
            input_box: t.input_box,
            anchor: t.anchor,
        }
    }

    pub fn to_json(&self, window_size: PhysicalSize<u32>) -> UiButtonTextJson {
        let scale = (window_size.width as f32 * window_size.height as f32).sqrt();
        UiButtonTextJson {
            id: self.id.clone(),
            action: self.action.clone(),
            style: self.style.clone(),
            z_index: self.z_index,

            x: self.x / window_size.width as f32,
            y: self.y / window_size.height as f32,

            top_left_offset: [
                self.top_left_offset[0] / scale,
                self.top_left_offset[1] / scale,
            ],
            bottom_left_offset: [
                self.bottom_left_offset[0] / scale,
                self.bottom_left_offset[1] / scale,
            ],
            top_right_offset: [
                self.top_right_offset[0] / scale,
                self.top_right_offset[1] / scale,
            ],
            bottom_right_offset: [
                self.bottom_right_offset[0] / scale,
                self.bottom_right_offset[1] / scale,
            ],

            px: self.px as f32 / scale,
            color: self.color,
            text: self.template.clone(),
            misc: self.misc.to_json(),
            input_box: self.input_box,
            anchor: self.anchor,
        }
    }
}

impl UiButtonCircle {
    pub(crate) fn from_json(c: UiButtonCircleJson, window_size: PhysicalSize<u32>) -> Self {
        let scale = (window_size.width as f32 * window_size.height as f32).sqrt();
        UiButtonCircle {
            id: c.id,
            action: c.action,
            style: c.style,
            z_index: c.z_index,
            x: window_size.width as f32 * c.x,
            y: window_size.height as f32 * c.y,
            radius: scale * c.radius,
            inside_border_thickness: scale * c.inside_border_thickness,
            border_thickness: scale * c.border_thickness,
            fade: c.fade,
            fill_color: c.fill_color,
            inside_border_color: c.inside_border_color,
            border_color: c.border_color,
            glow_color: c.glow_color,
            glow_misc: GlowMisc {
                glow_size: scale * c.glow_misc.glow_size,
                glow_speed: c.glow_misc.glow_speed,
                glow_intensity: c.glow_misc.glow_intensity,
            },
            misc: MiscButtonSettings {
                active: c.misc.active,
                touched_time: 0.0,
                is_touched: false,
                pressable: c.misc.pressable,
                editable: c.misc.editable,
            },
        }
    }

    pub fn to_json(&self, window_size: PhysicalSize<u32>) -> UiButtonCircleJson {
        let scale = (window_size.width as f32 * window_size.height as f32).sqrt();
        let mut glow_misc = self.glow_misc.clone();
        glow_misc.glow_size /= scale;
        UiButtonCircleJson {
            id: self.id.clone(),
            action: self.action.clone(),
            style: self.style.clone(),
            z_index: self.z_index,

            x: self.x / window_size.width as f32,
            y: self.y / window_size.height as f32,

            radius: self.radius / scale,
            inside_border_thickness: self.inside_border_thickness / scale,
            border_thickness: self.border_thickness / scale,

            fade: self.fade,
            fill_color: self.fill_color,
            inside_border_color: self.inside_border_color,
            border_color: self.border_color,
            glow_color: self.glow_color,
            glow_misc,

            misc: self.misc.to_json(),
        }
    }
}

impl UiButtonHandle {
    pub(crate) fn from_json(h: UiButtonHandleJson, window_size: PhysicalSize<u32>) -> Self {
        let scale = (window_size.width as f32 * window_size.height as f32).sqrt();
        UiButtonHandle {
            id: h.id,
            z_index: h.z_index,
            x: window_size.width as f32 * h.x,
            y: window_size.height as f32 * h.y,
            radius: scale * h.radius,
            handle_thickness: scale * h.handle_thickness,
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

    pub fn to_json(&self, window_size: PhysicalSize<u32>) -> UiButtonHandleJson {
        let scale = (window_size.width as f32 * window_size.height as f32).sqrt();
        UiButtonHandleJson {
            id: self.id.clone(),
            z_index: self.z_index,
            x: self.x / window_size.width as f32,
            y: self.y / window_size.height as f32,
            radius: self.radius / scale,

            handle_thickness: self.handle_thickness / scale,
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
    pub(crate) fn from_json(o: UiButtonOutlineJson, window_size: PhysicalSize<u32>) -> Self {
        let scale = (window_size.width as f32 * window_size.height as f32).sqrt();
        UiButtonOutline {
            id: o.id,
            z_index: o.z_index,
            parent_id: o.parent_id,
            mode: o.mode,
            vertex_offset: 0,
            vertex_count: 0,
            shape_data: o.shape_data.scale_from_normalized(window_size, scale),
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

    pub fn to_json(&self, window_size: PhysicalSize<u32>) -> UiButtonOutlineJson {
        let scale = (window_size.width as f32 * window_size.height as f32).sqrt();
        UiButtonOutlineJson {
            id: self.id.clone(),
            z_index: self.z_index,
            parent_id: self.parent_id.clone(),

            mode: self.mode,
            shape_data: self.shape_data.scale_to_normalized(window_size, scale),

            dash_color: self.dash_color,
            dash_misc: self.dash_misc.clone(),
            sub_dash_color: self.sub_dash_color,
            sub_dash_misc: self.sub_dash_misc.clone(),

            misc: self.misc.to_json(),
        }
    }
}

impl UiButtonPolygon {
    pub(crate) fn from_json(
        p: UiButtonPolygonJson,
        id_gen: &mut usize,
        window_size: PhysicalSize<u32>,
    ) -> Self {
        let mut verts: Vec<UiVertex> = p
            .vertices
            .into_iter()
            .map(|vj| {
                let id = *id_gen;
                *id_gen += 1;
                UiVertex::from_json(vj, id, window_size)
            })
            .collect();

        ensure_ccw(&mut verts);

        UiButtonPolygon {
            id: p.id,
            action: p.action,
            style: p.style,
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

    pub fn to_json(&self, window_size: PhysicalSize<u32>) -> UiButtonPolygonJson {
        UiButtonPolygonJson {
            id: self.id.clone(),
            action: self.action.clone(),
            style: self.style.clone(),
            z_index: self.z_index,

            vertices: self
                .vertices
                .iter()
                .map(|v| v.to_json(window_size))
                .collect(),

            misc: self.misc.to_json(),
        }
    }
}

impl Default for UiButtonText {
    fn default() -> Self {
        Self {
            id: None,
            action: "None".to_string(),
            style: "None".to_string(),
            z_index: 0,
            x: 0.0,
            y: 0.0,
            top_left_offset: [0.0; 2],
            bottom_left_offset: [0.0; 2],
            top_right_offset: [0.0; 2],
            bottom_right_offset: [0.0; 2],
            px: 14,
            color: [1.0, 1.0, 1.0, 1.0],
            text: "".into(),
            template: "".to_string(),
            misc: MiscButtonSettings::default(),
            natural_width: 50.0,
            natural_height: 20.0,
            ascent: 10.0,
            being_edited: false,
            caret: 0,
            being_hovered: false,
            just_unhovered: false,
            sel_start: 0,
            sel_end: 0,
            has_selection: false,
            glyph_bounds: vec![],
            input_box: false,
            anchor: None,
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
            action: "None".to_string(),
            style: "None".to_string(),
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
            action: "None".to_string(),
            style: "None".to_string(),
            z_index: 0,
            x: 0.0,
            y: 0.0,
            radius: 10.0,
            inside_border_thickness: 0.0,
            border_thickness: 1.0,
            fade: 0.0,
            fill_color: [1.0, 1.0, 1.0, 1.0],
            inside_border_color: [0.0, 0.0, 0.0, 1.0],
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
            editable: true,
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
    pub action: String,
    pub style: String,
    pub z_index: i32,

    pub x: f32,
    pub y: f32,
    pub top_left_offset: [f32; 2],
    pub bottom_left_offset: [f32; 2],
    pub top_right_offset: [f32; 2],
    pub bottom_right_offset: [f32; 2],
    pub px: f32,
    pub color: [f32; 4],
    pub text: String,
    pub misc: MiscButtonSettingsJson,
    pub input_box: bool,
    pub anchor: Option<Anchor>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct UiButtonCircleJson {
    pub id: Option<String>,
    pub action: String,
    pub style: String,
    pub z_index: i32,
    pub x: f32,                       // normalized 0.0-1.0 of screen width!
    pub y: f32,                       // normalized 0.0-1.0 of screen height!
    pub radius: f32,                  // normalized 0.0-1.0
    pub inside_border_thickness: f32, // normalized 0.0-1.0
    pub border_thickness: f32,        // normalized 0.0-1.0
    pub fade: f32,                    // normalized 0.0-1.0
    pub fill_color: [f32; 4],
    pub inside_border_color: [f32; 4],
    pub border_color: [f32; 4],
    pub glow_color: [f32; 4],
    pub glow_misc: GlowMisc, // normalized 0.0-1.0
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
    pub action: String,
    pub style: String,
    pub z_index: i32,
    pub vertices: Vec<UiVertexJson>,
    pub misc: MiscButtonSettingsJson,
}
