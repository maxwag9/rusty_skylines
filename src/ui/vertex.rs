use crate::renderer::ui::{CircleParams, HandleParams, OutlineParams, TextParams};
use crate::renderer::ui_text_rendering::Anchor;
use crate::ui::helper::ensure_ccw;
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

#[derive(Debug, Clone)]
pub enum UiElementCache {
    Circle(CircleParams),
    Handle(HandleParams),
    Polygon(Vec<UiVertexPoly>),
    Text(TextParams),
    Outline(OutlineParams),
}

#[derive(Debug)]
pub struct LayerCache {
    pub elements: Vec<UiElementCache>,
    pub outline_poly_vertices: Vec<[f32; 2]>,
}

impl UiElementCache {
    // non-mutable
    pub fn as_circle(&self) -> Option<&CircleParams> {
        match self {
            UiElementCache::Circle(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_handle(&self) -> Option<&HandleParams> {
        match self {
            UiElementCache::Handle(h) => Some(h),
            _ => None,
        }
    }

    pub fn as_polygon(&self) -> Option<&Vec<UiVertexPoly>> {
        match self {
            UiElementCache::Polygon(p) => Some(p),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&TextParams> {
        match self {
            UiElementCache::Text(t) => Some(t),
            _ => None,
        }
    }

    pub fn as_outline(&self) -> Option<&OutlineParams> {
        match self {
            UiElementCache::Outline(o) => Some(o),
            _ => None,
        }
    }

    // mutable
    pub fn as_circle_mut(&mut self) -> Option<&mut CircleParams> {
        match self {
            UiElementCache::Circle(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_handle_mut(&mut self) -> Option<&mut HandleParams> {
        match self {
            UiElementCache::Handle(h) => Some(h),
            _ => None,
        }
    }

    pub fn as_polygon_mut(&mut self) -> Option<&mut Vec<UiVertexPoly>> {
        match self {
            UiElementCache::Polygon(p) => Some(p),
            _ => None,
        }
    }

    pub fn as_text_mut(&mut self) -> Option<&mut TextParams> {
        match self {
            UiElementCache::Text(t) => Some(t),
            _ => None,
        }
    }

    pub fn as_outline_mut(&mut self) -> Option<&mut OutlineParams> {
        match self {
            UiElementCache::Outline(o) => Some(o),
            _ => None,
        }
    }
}

impl LayerCache {
    // non-mutable iterators
    pub fn iter_circles(&self) -> impl Iterator<Item = &CircleParams> {
        self.elements.iter().filter_map(UiElementCache::as_circle)
    }

    pub fn iter_handles(&self) -> impl Iterator<Item = &HandleParams> {
        self.elements.iter().filter_map(UiElementCache::as_handle)
    }

    pub fn iter_polygons(&self) -> impl Iterator<Item = &Vec<UiVertexPoly>> {
        self.elements.iter().filter_map(UiElementCache::as_polygon)
    }

    pub fn iter_texts(&self) -> impl Iterator<Item = &TextParams> {
        self.elements.iter().filter_map(UiElementCache::as_text)
    }

    pub fn iter_outlines(&self) -> impl Iterator<Item = &OutlineParams> {
        self.elements.iter().filter_map(UiElementCache::as_outline)
    }
}

impl Default for LayerCache {
    fn default() -> Self {
        Self {
            elements: vec![],
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
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum UiElementYaml {
    Circle(UiButtonCircleYaml),
    Handle(UiButtonHandleYaml),
    Polygon(UiButtonPolygonYaml),
    Text(UiButtonTextYaml),
    Outline(UiButtonOutlineYaml),
}
#[derive(Debug, Clone)]
pub enum UiElement {
    Circle(UiButtonCircle),
    Handle(UiButtonHandle),
    Polygon(UiButtonPolygon),
    Text(UiButtonText),
    Outline(UiButtonOutline),
}

impl UiElement {
    pub(crate) fn from_yaml(element: UiElementYaml, window_size: PhysicalSize<u32>) -> UiElement {
        match element {
            UiElementYaml::Circle(e) => {
                UiElement::Circle(UiButtonCircle::from_yaml(e, window_size))
            }
            UiElementYaml::Handle(e) => {
                UiElement::Handle(UiButtonHandle::from_yaml(e, window_size))
            }
            UiElementYaml::Polygon(e) => {
                UiElement::Polygon(UiButtonPolygon::from_yaml(e, window_size))
            }
            UiElementYaml::Text(e) => UiElement::Text(UiButtonText::from_yaml(e, window_size)),
            UiElementYaml::Outline(e) => {
                UiElement::Outline(UiButtonOutline::from_yaml(e, window_size))
            }
        }
    }
    pub(crate) fn to_yaml(&self, window_size: PhysicalSize<u32>) -> UiElementYaml {
        match self {
            UiElement::Circle(e) => UiElementYaml::Circle(UiButtonCircle::to_yaml(e, window_size)),
            UiElement::Handle(e) => UiElementYaml::Handle(UiButtonHandle::to_yaml(e, window_size)),
            UiElement::Polygon(e) => {
                UiElementYaml::Polygon(UiButtonPolygon::to_yaml(e, window_size))
            }
            UiElement::Text(e) => UiElementYaml::Text(UiButtonText::to_yaml(e, window_size)),
            UiElement::Outline(e) => {
                UiElementYaml::Outline(UiButtonOutline::to_yaml(e, window_size))
            }
        }
    }
    pub fn as_text_mut(&mut self) -> Option<&mut UiButtonText> {
        match self {
            UiElement::Text(t) => Some(t),
            _ => None,
        }
    }
    pub fn as_circle_mut(&mut self) -> Option<&mut UiButtonCircle> {
        match self {
            UiElement::Circle(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_polygon_mut(&mut self) -> Option<&mut UiButtonPolygon> {
        match self {
            UiElement::Polygon(p) => Some(p),
            _ => None,
        }
    }

    pub fn as_handle_mut(&mut self) -> Option<&mut UiButtonHandle> {
        match self {
            UiElement::Handle(h) => Some(h),
            _ => None,
        }
    }

    pub fn as_outline_mut(&mut self) -> Option<&mut UiButtonOutline> {
        match self {
            UiElement::Outline(o) => Some(o),
            _ => None,
        }
    }
    pub fn as_circle(&self) -> Option<&UiButtonCircle> {
        match self {
            UiElement::Circle(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_handle(&self) -> Option<&UiButtonHandle> {
        match self {
            UiElement::Handle(h) => Some(h),
            _ => None,
        }
    }

    pub fn as_polygon(&self) -> Option<&UiButtonPolygon> {
        match self {
            UiElement::Polygon(p) => Some(p),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&UiButtonText> {
        match self {
            UiElement::Text(t) => Some(t),
            _ => None,
        }
    }

    pub fn as_outline(&self) -> Option<&UiButtonOutline> {
        match self {
            UiElement::Outline(o) => Some(o),
            _ => None,
        }
    }

    /// Get element kind name for descriptions
    pub fn kind_name(&self) -> &'static str {
        match self {
            UiElement::Circle(_) => "Circle",
            UiElement::Text(_) => "Text",
            UiElement::Polygon(_) => "Polygon",
            UiElement::Handle(_) => "Handle",
            UiElement::Outline(_) => "Outline",
        }
    }
    pub fn id(&self) -> &str {
        match self {
            UiElement::Text(t) => t.id.as_deref().unwrap_or(""),
            UiElement::Circle(c) => c.id.as_deref().unwrap_or(""),
            UiElement::Outline(o) => o.id.as_deref().unwrap_or(""),
            UiElement::Handle(h) => h.id.as_deref().unwrap_or(""),
            UiElement::Polygon(p) => p.id.as_deref().unwrap_or(""),
        }
    }

    pub fn center(&self) -> (f32, f32) {
        match self {
            UiElement::Text(t) => (t.x, t.y),
            UiElement::Circle(c) => (c.x, c.y),
            UiElement::Handle(h) => (h.x, h.y),
            UiElement::Outline(o) => (o.shape_data.x, o.shape_data.y),
            UiElement::Polygon(p) => {
                let count = p.vertices.len().max(1);
                let sum = p
                    .vertices
                    .iter()
                    .fold((0.0, 0.0), |acc, v| (acc.0 + v.pos[0], acc.1 + v.pos[1]));
                (sum.0 / count as f32, sum.1 / count as f32)
            }
        }
    }
    pub fn kind(&self) -> ElementKind {
        match self {
            UiElement::Circle(_) => ElementKind::Circle,
            UiElement::Text(_) => ElementKind::Text,
            UiElement::Polygon(_) => ElementKind::Polygon,
            UiElement::Outline(_) => ElementKind::Outline,
            UiElement::Handle(_) => ElementKind::Handle,
        }
    }
    pub fn resize(&mut self, scale: f32) {
        match self {
            UiElement::Text(t) => {
                t.px = ((t.px as f32) * scale).round() as u16;

                println!("Text size: {}", t.px)
            }
            UiElement::Circle(c) => {
                c.radius *= scale;
            }
            UiElement::Outline(_) | UiElement::Handle(_) => {}
            UiElement::Polygon(p) => {
                // compute centroid
                let mut cx = 0.0f32;
                let mut cy = 0.0f32;
                let count = p.vertices.len() as f32;
                if count <= 0.0 {
                    return;
                }
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
    pub fn translate(&mut self, dx: f32, dy: f32) {
        match self {
            UiElement::Text(t) => {
                t.x += dx;
                t.y += dy;
            }
            UiElement::Circle(c) => {
                c.x += dx;
                c.y += dy;
            }
            UiElement::Handle(h) => {
                h.x += dx;
                h.y += dy;
            }
            UiElement::Outline(o) => {
                o.shape_data.x += dx;
                o.shape_data.y += dy;
            }
            UiElement::Polygon(p) => {
                for v in &mut p.vertices {
                    v.pos[0] += dx;
                    v.pos[1] += dy;
                }
            }
        }
    }
    /// Replaces self if same variant and matching id. Returns true if replaced.
    pub fn replace_if_matches(&mut self, new_state: &UiElement) -> bool {
        match (self, new_state) {
            (UiElement::Polygon(p), UiElement::Polygon(new_p)) if p.id == new_p.id => {
                *p = new_p.clone();
                true
            }
            (UiElement::Circle(c), UiElement::Circle(new_c)) if c.id == new_c.id => {
                *c = new_c.clone();
                true
            }
            (UiElement::Text(t), UiElement::Text(new_t)) if t.id == new_t.id => {
                *t = new_t.clone();
                true
            }
            _ => false,
        }
    }

    pub fn mark_dirty(&self, dirty: &mut LayerDirty) {
        match self {
            UiElement::Polygon(_) => dirty.mark_polygons(),
            UiElement::Circle(_) => dirty.mark_circles(),
            UiElement::Text(_) => dirty.mark_texts(),
            UiElement::Handle(_) => dirty.mark_handles(),
            UiElement::Outline(_) => dirty.mark_outlines(),
        }
    }
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
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
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
    pub elements: Vec<UiElement>,
    pub active: bool,
    // NEW: cached GPU data!!!
    pub cache: LayerCache,

    pub dirty: LayerDirty, // set true when anything changes or the screen will be dirty asf!
    pub gpu: LayerGpu,
    pub opaque: bool,
    pub saveable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementKind {
    Text,
    Circle,
    Outline,
    Handle,
    Polygon,
    None,
}
impl From<&UiElement> for ElementKind {
    fn from(element: &UiElement) -> Self {
        match element {
            UiElement::Circle(_) => ElementKind::Circle,
            UiElement::Handle(_) => ElementKind::Handle,
            UiElement::Polygon(_) => ElementKind::Polygon,
            UiElement::Text(_) => ElementKind::Text,
            UiElement::Outline(_) => ElementKind::Outline,
        }
    }
}
impl std::fmt::Display for ElementKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl RuntimeLayer {
    pub fn bump_element_z(&mut self, id: &str, delta: i32) {
        let len = self.elements.len();
        let Some(idx) = self.elements.iter().position(|e| e.id() == id) else {
            return;
        };

        let new_idx = (idx as i32 + delta).clamp(0, (len - 1) as i32) as usize;

        if idx == new_idx {
            return;
        }

        let element = self.elements.remove(idx);
        self.elements.insert(new_idx, element);
    }

    // simplified bump_element_xy using the helper
    pub fn bump_element_xy(&mut self, id: &str, dx: f32, dy: f32) {
        if let Some(el) = self.find_element_mut(id) {
            el.translate(dx, dy);
        }
    }

    // simplified resize_element using the helper
    pub fn resize_element(&mut self, id: &str, scale: f32) {
        if let Some(el) = self.find_element_mut(id) {
            el.resize(scale);
        }
    }
    pub fn find_element_mut(&mut self, id: &str) -> Option<&mut UiElement> {
        self.elements.iter_mut().find(|e| e.id() == id)
    }

    pub fn iter_circles(&self) -> impl Iterator<Item = &UiButtonCircle> {
        self.elements.iter().filter_map(UiElement::as_circle)
    }

    pub fn iter_handles(&self) -> impl Iterator<Item = &UiButtonHandle> {
        self.elements.iter().filter_map(UiElement::as_handle)
    }

    pub fn iter_polygons(&self) -> impl Iterator<Item = &UiButtonPolygon> {
        self.elements.iter().filter_map(UiElement::as_polygon)
    }

    pub fn iter_texts(&self) -> impl Iterator<Item = &UiButtonText> {
        self.elements.iter().filter_map(UiElement::as_text)
    }

    pub fn iter_outlines(&self) -> impl Iterator<Item = &UiButtonOutline> {
        self.elements.iter().filter_map(UiElement::as_outline)
    }

    pub fn iter_all(&self) -> impl Iterator<Item = &UiElement> {
        self.elements.iter()
    }

    pub fn iter_all_mut(&mut self) -> impl Iterator<Item = &mut UiElement> {
        self.elements.iter_mut()
    }
    /// Clear all Circle elements
    pub fn clear_circles(&mut self) {
        self.elements.retain(|e| !matches!(e, UiElement::Circle(_)));
    }

    /// Clear all Handle elements
    pub fn clear_handles(&mut self) {
        self.elements.retain(|e| !matches!(e, UiElement::Handle(_)));
    }

    /// Clear all Polygon elements
    pub fn clear_polygons(&mut self) {
        self.elements
            .retain(|e| !matches!(e, UiElement::Polygon(_)));
    }

    /// Clear all Text elements
    pub fn clear_texts(&mut self) {
        self.elements.retain(|e| !matches!(e, UiElement::Text(_)));
    }

    /// Clear all Outline elements
    pub fn clear_outlines(&mut self) {
        self.elements
            .retain(|e| !matches!(e, UiElement::Outline(_)));
    }
    pub fn replace_element(&mut self, new_state: &UiElement) -> bool {
        for elem in &mut self.elements {
            if elem.replace_if_matches(new_state) {
                new_state.mark_dirty(&mut self.dirty);
                return true;
            }
        }
        false
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(default)] // This tells serde to fill missing fields with defaults when loading
pub struct UiLayerYaml {
    pub name: String,

    #[serde(default, skip_serializing_if = "is_default")] // Skips if 0
    pub order: u32,

    // Skips if None or Empty Vector
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elements: Option<Vec<UiElementYaml>>,

    // Default is true. Skip if true.
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub active: bool,

    // Default is false. Skip if false.
    #[serde(default, skip_serializing_if = "is_default")]
    pub opaque: bool,
}

// Manual Default impl required because of the custom default values (like active=true)
impl Default for UiLayerYaml {
    fn default() -> Self {
        Self {
            name: "Layer".to_string(),
            order: 0,
            elements: None,
            active: true,
            opaque: false,
        }
    }
}

#[derive(Deserialize, Debug, Clone, Copy)]
pub struct UiVertex {
    pub pos: [f32; 2],
    pub color: [f32; 4],
    pub roundness: f32,
    pub _selected: bool,
    pub id: usize,
}

impl UiVertex {
    fn from_yaml(v: UiVertexYaml, id: usize, window_size: PhysicalSize<u32>) -> Self {
        let mut pos = v.pos;
        pos[0] *= window_size.width as f32;
        pos[1] *= window_size.height as f32;
        UiVertex {
            pos,
            color: v.color,
            roundness: v.roundness,
            _selected: false,
            id,
        }
    }

    pub fn to_yaml(&self, window_size: PhysicalSize<u32>) -> UiVertexYaml {
        let mut pos = self.pos;
        pos[0] /= window_size.width as f32;
        pos[1] /= window_size.height as f32;
        UiVertexYaml {
            pos,
            color: self.color,
            roundness: self.roundness,
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq)]
#[serde(default)]
pub struct UiVertexYaml {
    // Position usually shouldn't be skipped as it defines the shape
    pub pos: [f32; 2],
    pub color: [f32; 4],

    #[serde(skip_serializing_if = "is_default")]
    pub roundness: f32,
}

impl Default for UiVertexYaml {
    fn default() -> Self {
        Self {
            pos: [0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            roundness: 0.0,
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct GlowMisc {
    #[serde(skip_serializing_if = "is_default")]
    pub glow_size: f32,
    #[serde(skip_serializing_if = "is_default")]
    pub glow_speed: f32,
    #[serde(skip_serializing_if = "is_default")]
    pub glow_intensity: f32,
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

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct DashMisc {
    #[serde(skip_serializing_if = "is_default")]
    pub dash_len: f32,
    #[serde(skip_serializing_if = "is_default")]
    pub dash_spacing: f32,
    #[serde(skip_serializing_if = "is_default")]
    pub dash_roundness: f32,
    #[serde(skip_serializing_if = "is_default")]
    pub dash_speed: f32,
}
impl Default for DashMisc {
    fn default() -> Self {
        Self {
            dash_len: 1.0,
            dash_spacing: 10.0,
            dash_roundness: 0.0,
            dash_speed: 1.0,
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
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

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
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
    pub fn to_yaml(&self) -> MiscButtonSettingsYaml {
        MiscButtonSettingsYaml {
            active: self.active,
            pressable: self.pressable,
            editable: self.editable,
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
pub struct MiscButtonSettingsYaml {
    pub active: bool,
    pub pressable: bool,
    pub editable: bool,
}

impl Default for MiscButtonSettingsYaml {
    fn default() -> Self {
        Self {
            active: true,
            pressable: true,
            editable: true,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GuiLayout {
    pub menus: Vec<MenuYaml>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MenuYaml {
    pub name: String,
    pub layers: Vec<UiLayerYaml>,
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
// --- all possible button shapes ---
#[derive(Deserialize, Debug, Clone)]
pub struct UiButtonText {
    pub id: Option<String>,
    pub action: String,
    pub style: String,
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

#[derive(Deserialize, Debug, Clone)]
pub struct UiButtonPolygon {
    pub id: Option<String>,
    pub action: String,
    pub style: String,
    pub vertices: Vec<UiVertex>,
    pub misc: MiscButtonSettings,
    pub tri_count: u32,
}

#[derive(Deserialize, Debug, Clone)]
pub struct UiButtonCircle {
    pub id: Option<String>,
    pub action: String,
    pub style: String,
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub inside_border_thickness_percentage: f32,
    pub border_thickness_percentage: f32,
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
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub handle_color: [f32; 4],
    pub handle_misc: HandleMisc,
    pub sub_handle_color: [f32; 4],
    pub sub_handle_misc: HandleMisc,
    pub misc: MiscButtonSettings,
    pub parent_id: Option<String>,
}

impl UiButtonText {
    pub(crate) fn from_yaml(t: UiButtonTextYaml, window_size: PhysicalSize<u32>) -> Self {
        let scale = (window_size.width as f32 * window_size.height as f32).sqrt();
        let length = t.text.len();
        UiButtonText {
            id: t.id,
            action: t.action.clone(),
            style: t.style.clone(),
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

    pub fn to_yaml(&self, window_size: PhysicalSize<u32>) -> UiButtonTextYaml {
        let scale = (window_size.width as f32 * window_size.height as f32).sqrt();
        UiButtonTextYaml {
            id: self.id.clone(),
            action: self.action.clone(),
            style: self.style.clone(),

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
            misc: self.misc.to_yaml(),
            input_box: self.input_box,
            anchor: self.anchor,
        }
    }
}

impl UiButtonCircle {
    pub(crate) fn from_yaml(c: UiButtonCircleYaml, window_size: PhysicalSize<u32>) -> Self {
        let scale = (window_size.width as f32 * window_size.height as f32).sqrt();
        let radius = scale * c.radius;
        UiButtonCircle {
            id: c.id,
            action: c.action,
            style: c.style,
            x: window_size.width as f32 * c.x,
            y: window_size.height as f32 * c.y,
            radius,
            inside_border_thickness_percentage: c.inside_border_thickness_percentage,
            border_thickness_percentage: c.border_thickness_percentage,
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

    pub fn to_yaml(&self, window_size: PhysicalSize<u32>) -> UiButtonCircleYaml {
        let scale = (window_size.width as f32 * window_size.height as f32).sqrt();
        let mut glow_misc = self.glow_misc.clone();
        glow_misc.glow_size /= scale;
        UiButtonCircleYaml {
            id: self.id.clone(),
            action: self.action.clone(),
            style: self.style.clone(),
            x: self.x / window_size.width as f32,
            y: self.y / window_size.height as f32,

            radius: self.radius / scale,
            inside_border_thickness_percentage: self.inside_border_thickness_percentage,
            border_thickness_percentage: self.border_thickness_percentage,

            fade: self.fade,
            fill_color: self.fill_color,
            inside_border_color: self.inside_border_color,
            border_color: self.border_color,
            glow_color: self.glow_color,
            glow_misc,

            misc: self.misc.to_yaml(),
        }
    }
}

impl UiButtonHandle {
    pub(crate) fn from_yaml(h: UiButtonHandleYaml, window_size: PhysicalSize<u32>) -> Self {
        let scale = (window_size.width as f32 * window_size.height as f32).sqrt();
        UiButtonHandle {
            id: h.id,
            x: window_size.width as f32 * h.x,
            y: window_size.height as f32 * h.y,
            radius: scale * h.radius,
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

    pub fn to_yaml(&self, window_size: PhysicalSize<u32>) -> UiButtonHandleYaml {
        let scale = (window_size.width as f32 * window_size.height as f32).sqrt();
        UiButtonHandleYaml {
            id: self.id.clone(),
            x: self.x / window_size.width as f32,
            y: self.y / window_size.height as f32,
            radius: self.radius / scale,

            handle_color: self.handle_color,
            handle_misc: self.handle_misc.clone(),

            sub_handle_color: self.sub_handle_color,
            sub_handle_misc: self.sub_handle_misc.clone(),

            parent_id: self.parent_id.clone(),
            misc: self.misc.to_yaml(),
        }
    }
}

impl UiButtonOutline {
    pub(crate) fn from_yaml(o: UiButtonOutlineYaml, window_size: PhysicalSize<u32>) -> Self {
        let scale = (window_size.width as f32 * window_size.height as f32).sqrt();
        UiButtonOutline {
            id: o.id,
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

    pub fn to_yaml(&self, window_size: PhysicalSize<u32>) -> UiButtonOutlineYaml {
        let scale = (window_size.width as f32 * window_size.height as f32).sqrt();
        UiButtonOutlineYaml {
            id: self.id.clone(),
            parent_id: self.parent_id.clone(),

            mode: self.mode,
            shape_data: self.shape_data.scale_to_normalized(window_size, scale),

            dash_color: self.dash_color,
            dash_misc: self.dash_misc.clone(),
            sub_dash_color: self.sub_dash_color,
            sub_dash_misc: self.sub_dash_misc.clone(),

            misc: self.misc.to_yaml(),
        }
    }
}

impl UiButtonPolygon {
    pub(crate) fn from_yaml(p: UiButtonPolygonYaml, window_size: PhysicalSize<u32>) -> Self {
        let mut id_gen = 1;
        let mut verts: Vec<UiVertex> = p
            .vertices
            .into_iter()
            .map(|vj| {
                id_gen += 1;
                UiVertex::from_yaml(vj, id_gen, window_size)
            })
            .collect();

        ensure_ccw(&mut verts);

        UiButtonPolygon {
            id: p.id,
            action: p.action,
            style: p.style,
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

    pub fn to_yaml(&self, window_size: PhysicalSize<u32>) -> UiButtonPolygonYaml {
        UiButtonPolygonYaml {
            id: self.id.clone(),
            action: self.action.clone(),
            style: self.style.clone(),

            vertices: self
                .vertices
                .iter()
                .map(|v| v.to_yaml(window_size))
                .collect(),

            misc: self.misc.to_yaml(),
        }
    }

    pub fn center(&self) -> (f32, f32) {
        let count = self.vertices.len().max(1);
        let sum = self
            .vertices
            .iter()
            .fold((0.0, 0.0), |acc, v| (acc.0 + v.pos[0], acc.1 + v.pos[1]));
        (sum.0 / count as f32, sum.1 / count as f32)
    }
}

impl Default for UiButtonText {
    fn default() -> Self {
        Self {
            id: None,
            action: "None".to_string(),
            style: "None".to_string(),
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
                _selected: false,
                id: 0,
            },
            UiVertex {
                pos: [0.0, -30.0],
                color: [1.0, 1.0, 1.0, 1.0],
                roundness: 0.0,
                _selected: false,
                id: 1,
            },
            UiVertex {
                pos: [30.0, 30.0],
                color: [1.0, 1.0, 1.0, 1.0],
                roundness: 0.0,
                _selected: false,
                id: 2,
            },
            UiVertex {
                pos: [30.0, 50.0],
                color: [1.0, 1.0, 0.0, 1.0],
                roundness: 0.0,
                _selected: false,
                id: 3,
            },
            UiVertex {
                pos: [50.0, 30.0],
                color: [1.0, 0.0, 1.0, 1.0],
                roundness: 0.0,
                _selected: false,
                id: 5,
            },
        ];

        Self {
            id: None,
            action: "None".to_string(),
            style: "None".to_string(),
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
            x: 0.0,
            y: 0.0,
            radius: 10.0,
            inside_border_thickness_percentage: 0.0,
            border_thickness_percentage: 1.0,
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
            x: 0.0,
            y: 0.0,
            radius: 6.0,
            handle_color: [1.0, 1.0, 1.0, 1.0],
            handle_misc: HandleMisc::default(),
            sub_handle_color: [1.0, 1.0, 1.0, 1.0],
            sub_handle_misc: HandleMisc::default(),
            misc: MiscButtonSettings::default(),
            parent_id: None,
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
            handle_len: 0.15,
            handle_width: 0.1,
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
            _selected: false,
            id: 0,
        }
    }
}

// --- TEXT ---
#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(default)]
pub struct UiButtonTextYaml {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    #[serde(
        default = "default_none_string",
        skip_serializing_if = "is_none_string"
    )]
    pub action: String,

    #[serde(
        default = "default_none_string",
        skip_serializing_if = "is_none_string"
    )]
    pub style: String,

    pub x: f32,
    pub y: f32,

    #[serde(skip_serializing_if = "is_zero_vec2")]
    pub top_left_offset: [f32; 2],
    #[serde(skip_serializing_if = "is_zero_vec2")]
    pub bottom_left_offset: [f32; 2],
    #[serde(skip_serializing_if = "is_zero_vec2")]
    pub top_right_offset: [f32; 2],
    #[serde(skip_serializing_if = "is_zero_vec2")]
    pub bottom_right_offset: [f32; 2],

    #[serde(skip_serializing_if = "is_default")]
    pub px: f32,

    pub color: [f32; 4],
    pub text: String,

    // If 'misc' matches defaults (active:true, pressable:false, editable:false),
    // this entire block is removed from YAML.
    #[serde(skip_serializing_if = "is_default")]
    pub misc: MiscButtonSettingsYaml,

    #[serde(skip_serializing_if = "is_default")]
    pub input_box: bool,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub anchor: Option<Anchor>,
}

impl Default for UiButtonTextYaml {
    fn default() -> Self {
        Self {
            id: None,
            action: "None".to_string(),
            style: "None".to_string(),
            x: 0.0,
            y: 0.0,
            top_left_offset: [0.0; 2],
            bottom_left_offset: [0.0; 2],
            top_right_offset: [0.0; 2],
            bottom_right_offset: [0.0; 2],
            px: 0.0,
            color: [1.0, 1.0, 1.0, 1.0],
            text: String::new(),
            misc: MiscButtonSettingsYaml::default(),
            input_box: false,
            anchor: None,
        }
    }
}

// --- CIRCLE ---
#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(default)]
pub struct UiButtonCircleYaml {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    #[serde(
        default = "default_none_string",
        skip_serializing_if = "is_none_string"
    )]
    pub action: String,

    #[serde(
        default = "default_none_string",
        skip_serializing_if = "is_none_string"
    )]
    pub style: String,

    pub x: f32,
    pub y: f32,
    pub radius: f32,

    #[serde(skip_serializing_if = "is_default")]
    pub inside_border_thickness_percentage: f32,

    #[serde(skip_serializing_if = "is_default")]
    pub border_thickness_percentage: f32,

    #[serde(skip_serializing_if = "is_default")]
    pub fade: f32,
    #[serde(skip_serializing_if = "is_default")]
    pub fill_color: [f32; 4],
    #[serde(skip_serializing_if = "is_default")]
    pub inside_border_color: [f32; 4],
    #[serde(skip_serializing_if = "is_default")]
    pub border_color: [f32; 4],

    // Default implies [0,0,0,0] (transparent). Skip if so.
    #[serde(skip_serializing_if = "is_default")]
    pub glow_color: [f32; 4],

    // If glow settings are all 0.0, remove block
    #[serde(skip_serializing_if = "is_default")]
    pub glow_misc: GlowMisc,

    #[serde(skip_serializing_if = "is_default")]
    pub misc: MiscButtonSettingsYaml,
}

impl Default for UiButtonCircleYaml {
    fn default() -> Self {
        Self {
            id: None,
            action: "None".to_string(),
            style: "None".to_string(),
            x: 0.0,
            y: 0.0,
            radius: 0.0,
            inside_border_thickness_percentage: 0.0,
            border_thickness_percentage: 0.0,
            fade: 0.0,
            fill_color: [1.0, 1.0, 1.0, 1.0],
            inside_border_color: [0.0, 0.0, 0.0, 0.0],
            border_color: [0.0, 0.0, 0.0, 0.0],
            glow_color: [0.0, 0.0, 0.0, 0.0],
            glow_misc: GlowMisc::default(),
            misc: MiscButtonSettingsYaml::default(),
        }
    }
}

// --- HANDLE ---
#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(default)]
pub struct UiButtonHandleYaml {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    pub x: f32,
    pub y: f32,
    pub radius: f32,

    pub handle_color: [f32; 4],

    #[serde(skip_serializing_if = "is_default")]
    pub handle_misc: HandleMisc, // Assuming HandleMisc exists and derives Default+PartialEq

    pub sub_handle_color: [f32; 4],

    #[serde(skip_serializing_if = "is_default")]
    pub sub_handle_misc: HandleMisc,

    #[serde(skip_serializing_if = "is_default")]
    pub misc: MiscButtonSettingsYaml,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
}

impl Default for UiButtonHandleYaml {
    fn default() -> Self {
        Self {
            id: None,
            x: 0.0,
            y: 0.0,
            radius: 0.0,
            handle_color: [1.0, 1.0, 1.0, 1.0],
            handle_misc: HandleMisc::default(),
            sub_handle_color: [1.0, 1.0, 1.0, 1.0],
            sub_handle_misc: HandleMisc::default(),
            misc: MiscButtonSettingsYaml::default(),
            parent_id: None,
        }
    }
}

// --- OUTLINE ---
#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(default)]
pub struct UiButtonOutlineYaml {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,

    #[serde(skip_serializing_if = "is_default")]
    pub mode: f32,

    // Assuming ShapeData derives Default/PartialEq
    #[serde(skip_serializing_if = "is_default")]
    pub shape_data: ShapeData,

    pub dash_color: [f32; 4],

    #[serde(skip_serializing_if = "is_default")]
    pub dash_misc: DashMisc,

    pub sub_dash_color: [f32; 4],

    #[serde(skip_serializing_if = "is_default")]
    pub sub_dash_misc: DashMisc,

    #[serde(skip_serializing_if = "is_default")]
    pub misc: MiscButtonSettingsYaml,
}

impl Default for UiButtonOutlineYaml {
    fn default() -> Self {
        Self {
            id: None,
            parent_id: None,
            mode: 0.0,
            shape_data: ShapeData::default(),
            dash_color: [1.0, 1.0, 1.0, 1.0],
            dash_misc: DashMisc::default(),
            sub_dash_color: [1.0, 1.0, 1.0, 1.0],
            sub_dash_misc: DashMisc::default(),
            misc: MiscButtonSettingsYaml::default(),
        }
    }
}

// --- POLYGON ---
#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(default)]
pub struct UiButtonPolygonYaml {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    #[serde(
        default = "default_none_string",
        skip_serializing_if = "is_none_string"
    )]
    pub action: String,

    #[serde(
        default = "default_none_string",
        skip_serializing_if = "is_none_string"
    )]
    pub style: String,

    // If vertices are empty, we might as well skip, but usually polygon has data
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub vertices: Vec<UiVertexYaml>,

    #[serde(skip_serializing_if = "is_default")]
    pub misc: MiscButtonSettingsYaml,
}

impl Default for UiButtonPolygonYaml {
    fn default() -> Self {
        Self {
            id: None,
            action: "None".to_string(),
            style: "None".to_string(),
            vertices: Vec::new(),
            misc: MiscButtonSettingsYaml::default(),
        }
    }
}
// Checks if a standard type matches its default (e.g., false for bool, 0 for u32)
fn is_default<T: Default + PartialEq>(t: &T) -> bool {
    t == &T::default()
}

// Checks if a boolean is true (useful for things like 'active' where default is true)
fn is_true(b: &bool) -> bool {
    *b
}

// Checks if the [f32; 2] offset is [0.0, 0.0]
fn is_zero_offset(v: &[f32; 2]) -> bool {
    v[0] == 0.0 && v[1] == 0.0
}

// Checks if the string is "None" or empty
fn is_none_string(s: &String) -> bool {
    s == "None" || s.is_empty()
}

// Returns "None" for the default value of strings
fn default_none_string() -> String {
    "None".to_string()
}

// Returns true for default active states
fn default_true() -> bool {
    true
}

// Check: is [f32; 2] == [0.0, 0.0]?
fn is_zero_vec2(v: &[f32; 2]) -> bool {
    v[0] == 0.0 && v[1] == 0.0
}
