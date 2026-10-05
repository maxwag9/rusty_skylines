use crate::data::Settings;
use crate::helpers::modpack::ModManager;
use crate::helpers::positions::WorldPos;
use crate::helpers::{rgba_from_stupid_color, stupid_color_from_rgba};
use crate::renderer::ui::{CircleParams, HandleParams, OutlineParams, TextParams};
use crate::renderer::ui_text_rendering::Anchor;
use crate::ui::action_parser::CompiledAction;
use crate::ui::helper::ensure_ccw;
use crate::ui::parser::Value;
use crate::ui::ui_edit_manager::ColorComponent;
use crate::ui::ui_edits::SizeProperty;
use crate::ui::ui_touch_manager::{ElementRef, Touchable};
use crate::ui::variables::Variables;
use serde::de::Visitor;
use serde::{Deserialize, Deserializer, Serialize, de};
use sluggrs_skylines::cosmic_text::Metrics;
use sluggrs_skylines::{DecorationMode, TextDecoration};
use std::collections::HashMap;
use std::fmt;
use std::mem::size_of;
use tracing::error;
use wgpu::{vertex_attr_array, *};
use winit::dpi::PhysicalSize;

const SCALING_EPSILON: f32 = 0.1; // pixels, 0.1 is good because You probably won't move an element by just 0.1 pixels willingly, impossible.
fn snap(value: f32, original: f32) -> f32 {
    if (value - original).abs() < SCALING_EPSILON {
        // pixels
        original
    } else {
        value
    }
}
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ThinLineVtxRender {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}
impl ThinLineVtxRender {
    pub fn layout<'a>() -> VertexBufferLayout<'a> {
        const ATTRS: &[VertexAttribute] = &vertex_attr_array![0 => Float32x3, 1 => Float32x4];
        VertexBufferLayout {
            array_stride: size_of::<ThinLineVtxRender>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: ATTRS,
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ThickLineVtxRender {
    pub start: [f32; 3],
    pub end: [f32; 3],
    pub end_sign: f32,  // -1.0 for start side, +1.0 for end side
    pub side_sign: f32, // -1.0 or +1.0
    pub width: f32,
    pub color: [f32; 4],
}
impl ThickLineVtxRender {
    pub fn layout<'a>() -> VertexBufferLayout<'a> {
        const ATTRS: &[VertexAttribute] = &vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32, 3 => Float32, 4 => Float32, 5 => Float32x4];
        VertexBufferLayout {
            array_stride: size_of::<ThickLineVtxRender>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: ATTRS,
        }
    }
}
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TextVtxRender {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

impl TextVtxRender {
    pub fn layout<'a>() -> VertexBufferLayout<'a> {
        const ATTRS: &[VertexAttribute] =
            &vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x4];
        VertexBufferLayout {
            array_stride: size_of::<TextVtxRender>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: ATTRS,
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FlowingThickLineVtxRender {
    pub start: [f32; 3],
    pub end: [f32; 3],
    pub thickness: f32,
    pub start_dist: f32,
    pub end_dist: f32,
    pub color_a: [f32; 4],
    pub color_b: [f32; 4],
    pub pattern_len: f32,
    pub band_len: f32,
}
impl FlowingThickLineVtxRender {
    pub const ATTRIBUTES: [wgpu::VertexAttribute; 9] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32,
        3 => Float32,
        4 => Float32,
        5 => Float32x4,
        6 => Float32x4,
        7 => Float32,
        8 => Float32,
    ];

    pub fn layout<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[derive(Clone)]
pub struct LineVtxWorld {
    pub pos: WorldPos,
    pub color: [f32; 4],
}
#[derive(Debug, Clone)]
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

    pub rect_ssbo: Option<Buffer>,
    pub rect_count: u32,

    pub text_misc_vbo: Option<Buffer>,
    pub text_misc_vertex_count: u32,
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
            rect_ssbo: None,
            rect_count: 0,

            text_misc_vbo: None,
            text_misc_vertex_count: 0,
        }
    }
}

pub enum TouchState {
    Pressed,
    Held,
    Released,
    Idle,
}

#[derive(Debug, Clone, Copy)]
pub struct LayerDirty {
    pub texts: bool,
    pub circles: bool,
    pub outlines: bool,
    pub handles: bool,
    pub polygons: bool,
    pub rects: bool,
    pub aps: bool,
}

impl LayerDirty {
    pub fn all() -> Self {
        Self {
            texts: true,
            circles: true,
            outlines: true,
            handles: true,
            polygons: true,
            rects: true,
            aps: true,
        }
    }

    pub fn none() -> Self {
        Self {
            texts: false,
            circles: false,
            outlines: false,
            handles: false,
            polygons: false,
            rects: false,
            aps: false,
        }
    }

    pub fn any(self) -> bool {
        self.texts || self.circles || self.outlines || self.handles || self.polygons || self.aps
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
    pub fn mark_rects(&mut self) {
        self.rects = true;
    }
    pub fn mark_handles(&mut self) {
        self.handles = true;
    }

    pub fn mark_polygons(&mut self) {
        self.polygons = true;
    }
    pub fn mark_advanced_primitives(&mut self) {
        self.aps = true;
    }
    pub fn mark_all(&mut self) {
        *self = Self::all();
    }

    pub fn clear(&mut self, d: LayerDirty) {
        self.texts &= !d.texts;
        self.circles &= !d.circles;
        self.outlines &= !d.outlines;
        self.handles &= !d.handles;
        self.polygons &= !d.polygons;
        self.rects &= !d.rects;
        self.aps &= !d.aps
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
    Advanced(AdvancedPrimitiveYaml),
    Rect(UiButtonRectYaml),
}
impl UiElementYaml {
    pub fn kind(&self) -> ElementKind {
        match self {
            UiElementYaml::Rect(_) => ElementKind::Rect,
            UiElementYaml::Advanced(_) => ElementKind::Advanced,
            UiElementYaml::Circle(_) => ElementKind::Circle,
            UiElementYaml::Text(_) => ElementKind::Text,
            UiElementYaml::Polygon(_) => ElementKind::Polygon,
            UiElementYaml::Outline(_) => ElementKind::Outline,
            UiElementYaml::Handle(_) => ElementKind::Handle,
        }
    }
    pub fn advanced_primitive(&self) -> Option<AdvancedPrimitive> {
        match self {
            UiElementYaml::Advanced(ap) => Some(AdvancedPrimitive::from_yaml(&ap)),
            _ => None,
        }
    }
}
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AdvancedPrimitiveYaml {
    pub name: String,
    #[serde(default)]
    pub ap_name: String,

    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        alias = "ap_var",
        deserialize_with = "deserialize_string_or_vec"
    )]
    pub ap_vars: Vec<String>,
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "deserialize_string_or_vec"
    )]
    pub actions: Vec<String>,
    #[serde(deserialize_with = "deserialize_i16_from_number")]
    pub x: i16,
    #[serde(deserialize_with = "deserialize_i16_from_number")]
    pub y: i16,

    #[serde(default)]
    pub scale: f32,
    #[serde(default, skip_serializing_if = "is_default")]
    pub misc: MiscButtonSettingsYaml,
    #[serde(default, skip_serializing_if = "is_false")]
    pub editing_tool: bool,
}
#[derive(Debug, Clone)]
pub struct AdvancedPrimitive {
    pub id: String,
    pub ap_name: String,
    pub ap_vars: Vec<String>,
    pub actions: Vec<String>,
    pub x: f32,
    pub y: f32,
    pub scale: f32,
    pub misc: MiscButtonSettings,
    pub editing_tool: bool,
    pub is_temporary: bool,
    pub scale_my_coords: bool,
}

impl AdvancedPrimitive {
    pub fn from_yaml(yaml: &AdvancedPrimitiveYaml) -> Self {
        Self {
            id: yaml.name.clone(),
            ap_name: yaml.ap_name.clone(),
            ap_vars: yaml.ap_vars.clone(),
            actions: yaml.actions.clone(),
            x: yaml.x as f32,
            y: yaml.y as f32,
            scale: yaml.scale,
            misc: MiscButtonSettings {
                active: yaml.misc.active,
                touched_time: 0.0,
                is_touched: false,
                touchable: yaml.misc.touchable,
                editable: Editability::from_bool(yaml.misc.editable),
            },
            editing_tool: yaml.editing_tool,
            is_temporary: false,
            scale_my_coords: true,
        }
    }
    pub fn to_yaml(&self) -> AdvancedPrimitiveYaml {
        AdvancedPrimitiveYaml {
            name: self.id.clone(),
            ap_name: self.ap_name.clone(),
            ap_vars: self.ap_vars.clone(),
            actions: self.actions.clone(),
            x: self.x as i16,
            y: self.y as i16,
            scale: self.scale,
            misc: MiscButtonSettingsYaml {
                active: self.misc.active,
                touchable: self.misc.touchable,
                editable: self.misc.editable.to_bool(),
            },
            editing_tool: self.editing_tool,
        }
    }
    pub fn to_layer(
        self,
        settings: &Settings,
        variables: &Variables,
        advanced_primitives: &HashMap<String, UiLayerYaml>,
        order: u32,
        window_size: PhysicalSize<f32>,
        device: &Device,
        queue: &Queue,
        mod_manager: &ModManager,
    ) -> RuntimeLayer {
        let x_scale = window_size.width / 1920.0;
        let y_scale = window_size.height / 1080.0;
        let x = if self.scale_my_coords {
            self.x * x_scale
        } else {
            self.x
        };
        let y = if self.scale_my_coords {
            self.y * y_scale
        } else {
            self.y
        };

        let elements = if let Some(ap_template) = advanced_primitives.get(&self.ap_name) {
            ap_template
                .elements
                .clone()
                .unwrap_or_default()
                .into_iter()
                .filter_map(|e| UiElement::from_yaml(e, window_size, device, queue, mod_manager))
                .map(|mut el| {
                    //el.scale_by(x_scale, y_scale, );// WTF??!?!?!
                    el.translate(x, y);
                    el.set_editable(&self.misc.editable);
                    el.set_aps(settings, variables, self.ap_vars.as_slice());
                    el
                })
                .collect()
        } else {
            Vec::new()
        };
        //println!("AP {}: {}", self.id, elements.len());
        RuntimeLayer {
            name: self.id,
            ap_name: Some(self.ap_name),
            order,
            string_actions: self.actions.clone(),
            compiled_actions: vec![],
            elements,
            active: self.misc.active,
            ap_vars: self.ap_vars,
            dirty: LayerDirty::all(),
            gpu: Default::default(),
            opaque: false,
            saveable: false,
            editing_tool: self.editing_tool,
            outline_poly_vertices: vec![],
        }
    }

    pub fn set_pos(&mut self, position: [f32; 2]) {
        self.x = position[0];
        self.y = position[1];
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResizeAxis {
    #[serde(alias = "x")]
    X,
    #[serde(alias = "y")]
    Y,
    #[serde(alias = "uniform")]
    #[default]
    Uniform,
    #[serde(alias = "none")]
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResizeBehaviour {
    #[serde(default, skip_serializing_if = "is_default")]
    pub width: ResizeAxis,
    #[serde(default, skip_serializing_if = "is_default")]
    pub height: ResizeAxis,
}
impl Default for ResizeBehaviour {
    fn default() -> Self {
        Self {
            width: ResizeAxis::X,
            height: ResizeAxis::Y,
        }
    }
}
impl ResizeBehaviour {
    #[inline]
    pub fn scale_for(axis: ResizeAxis, x_scale: f32, y_scale: f32, uniform_scale: f32) -> f32 {
        match axis {
            ResizeAxis::X => x_scale,
            ResizeAxis::Y => y_scale,
            ResizeAxis::Uniform => uniform_scale,
            ResizeAxis::None => 1.0,
        }
    }

    #[inline]
    pub fn width_scale(self, x_scale: f32, y_scale: f32, uniform_scale: f32) -> f32 {
        Self::scale_for(self.width, x_scale, y_scale, uniform_scale)
    }

    #[inline]
    pub fn height_scale(self, x_scale: f32, y_scale: f32, uniform_scale: f32) -> f32 {
        Self::scale_for(self.height, x_scale, y_scale, uniform_scale)
    }

    #[inline]
    pub fn uniform_scale(self, x_scale: f32, y_scale: f32, uniform_scale: f32) -> f32 {
        match self.width {
            ResizeAxis::X => x_scale,
            ResizeAxis::Y => y_scale,
            ResizeAxis::Uniform => uniform_scale,
            ResizeAxis::None => 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub enum RectTextureType {
    Shader(String),
    Image(TextureView),
}

impl RectTextureType {
    pub fn from_str(
        kind: &str,
        path: &str,
        device: &Device,
        queue: &Queue,
        mod_manager: &ModManager,
    ) -> Option<Self> {
        match kind.to_lowercase().as_str() {
            "shader" => {
                let relative_path = path;

                Some(Self::Shader(relative_path.to_owned()))
            }

            "image" => {
                let Some(path) = mod_manager.resource_path(path) else {
                    error!("[Renderer] Missing shader '{path}' for UI Rect Image");
                    return None;
                };

                //println!("Image path: {:?}", path);
                if !path.is_file() {
                    error!("Image path isn't file: {}", path.display());
                    return None;
                }

                let image = match image::open(&path) {
                    Ok(image) => image.to_rgba8(),
                    Err(err) => {
                        error!("Could not load image: {}", err.to_string());
                        return None;
                    }
                };
                let dimensions = image.dimensions();

                let texture = device.create_texture(&TextureDescriptor {
                    label: Some("Rect Image Texture"),
                    size: Extent3d {
                        width: dimensions.0,
                        height: dimensions.1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba8UnormSrgb,
                    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                    view_formats: &[],
                });

                queue.write_texture(
                    TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: Origin3d::ZERO,
                        aspect: TextureAspect::All,
                    },
                    &image,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * dimensions.0),
                        rows_per_image: Some(dimensions.1),
                    },
                    Extent3d {
                        width: dimensions.0,
                        height: dimensions.1,
                        depth_or_array_layers: 1,
                    },
                );

                let view = texture.create_view(&TextureViewDescriptor::default());

                Some(Self::Image(view))
            }

            _ => None,
        }
    }
}
fn get_rect_texture(
    texture_string: Option<String>,
    device: &Device,
    queue: &Queue,
    mod_manager: &ModManager,
) -> Option<RectTextureType> {
    let Some(texture_string) = texture_string.as_ref() else {
        return None;
    };
    match texture_string.split_once(':') {
        None => None,
        Some((texture_type, path)) => {
            //println!("Trying texture: '{texture_type}', '{path}'");
            RectTextureType::from_str(texture_type, path, device, queue, mod_manager)
        }
    }
    // 1. Path to shader relative to data folder  2. Path to an actual texture image relative to the data folder    // -3. Procedural texture (NO NOT YET)-
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct UiButtonRectYaml {
    #[serde(
        default = "default_none_string",
        skip_serializing_if = "is_none_string"
    )]
    pub id: String,

    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "deserialize_string_or_vec"
    )]
    pub actions: Vec<String>,

    #[serde(
        default = "default_none_string",
        skip_serializing_if = "is_none_string"
    )]
    pub style: String,
    #[serde(deserialize_with = "deserialize_i16_from_number")]
    pub x: i16,
    #[serde(deserialize_with = "deserialize_i16_from_number")]
    pub y: i16,
    #[serde(deserialize_with = "deserialize_u16_from_number")]
    pub w: u16,
    #[serde(deserialize_with = "deserialize_u16_from_number")]
    pub h: u16,
    #[serde(default, skip_serializing_if = "is_default")]
    pub resize_behaviour: ResizeBehaviour,
    #[serde(default)]
    pub rotation: f32, // DEGREES!!! 360 ftw!
    #[serde(default)]
    pub color: [f32; 4], // tint for texture
    #[serde(default)]
    pub border_color: [f32; 4],
    #[serde(default)]
    pub texture: Option<String>,
    #[serde(default)]
    pub roundness: f32, // corner radius
    #[serde(default)]
    pub border_thickness: f32,
    #[serde(default)]
    pub fade: f32,
    #[serde(default)]
    pub blur: f32,
    #[serde(default)]
    pub glow_color: [f32; 4],

    // If glow settings are all 0.0, remove bloc
    #[serde(default)]
    pub glow_misc: GlowMisc,
    #[serde(default, skip_serializing_if = "is_default")]
    pub misc: MiscButtonSettingsYaml,
}
#[derive(Debug, Clone)]
pub struct UiButtonRect {
    pub id: String,
    pub string_actions: Vec<String>,
    pub compiled_actions: Vec<CompiledAction>,
    pub style: String,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub resize_behaviour: ResizeBehaviour,
    pub rotation: f32,   // DEGREES!!! 360 ftw!
    pub color: [f32; 4], // tint for texture
    pub border_color: [f32; 4],
    pub texture: Option<String>,
    pub cached_texture: Option<RectTextureType>,
    pub roundness: f32, // corner radius
    pub border_thickness: f32,
    pub fade: f32,
    pub blur: f32,
    pub glow_color: [f32; 4],
    pub glow_misc: GlowMisc,
    pub misc: MiscButtonSettings,
    pub yaml_element: Option<UiButtonRectYaml>,
    pub cache: Option<RectParams>,
}

impl UiButtonRect {
    pub fn from_yaml(
        e: UiButtonRectYaml,
        device: &Device,
        queue: &Queue,
        mod_manager: &ModManager,
    ) -> Self {
        let yaml_element = Some(e.clone());

        UiButtonRect {
            id: e.id,
            string_actions: e.actions,
            compiled_actions: vec![],
            style: e.style,

            x: e.x as f32,
            y: e.y as f32,
            w: e.w as f32,
            h: e.h as f32,

            resize_behaviour: e.resize_behaviour,

            rotation: e.rotation,
            color: e.color,
            border_color: e.border_color,
            texture: e.texture.clone(),
            cached_texture: get_rect_texture(e.texture, device, queue, mod_manager),
            roundness: e.roundness,
            border_thickness: e.border_thickness,
            fade: e.fade,
            blur: e.blur,
            glow_color: e.glow_color,
            glow_misc: e.glow_misc,

            misc: MiscButtonSettings {
                active: e.misc.active,
                touched_time: 0.0,
                is_touched: false,
                touchable: e.misc.touchable,
                editable: Editability::from_bool(e.misc.editable),
            },

            yaml_element,
            cache: None,
        }
    }

    pub fn to_yaml(&self) -> UiButtonRectYaml {
        let (x, y, w, h) = if let Some(yaml_element) = self.yaml_element.as_ref() {
            let x = snap(self.x, yaml_element.x as f32);
            let y = snap(self.y, yaml_element.y as f32);
            let w = snap(self.w, yaml_element.w as f32);
            let h = snap(self.h, yaml_element.h as f32);
            (x, y, w, h)
        } else {
            (self.x, self.y, self.w, self.h)
        };

        UiButtonRectYaml {
            id: self.id.clone(),
            actions: self.string_actions.clone(),
            style: self.style.clone(),

            x: x as i16,
            y: y as i16,
            w: w as u16,
            h: h as u16,

            resize_behaviour: self.resize_behaviour,

            rotation: self.rotation,
            color: self.color,
            border_color: self.border_color,
            texture: self.texture.clone(),
            roundness: self.roundness,
            border_thickness: self.border_thickness,
            fade: self.fade,
            blur: self.blur,
            glow_color: self.glow_color,
            glow_misc: self.glow_misc.clone(),

            misc: self.misc.to_yaml(),
        }
    }

    #[inline]
    pub fn mean_size(&self) -> f32 {
        (self.w + self.h) * 0.5
    }
    #[inline]
    pub fn size(&self) -> [f32; 2] {
        [self.w, self.h]
    }
    #[inline]
    pub fn set_pos(&mut self, position: [f32; 2]) {
        self.x = position[0];
        self.y = position[1];
    }
}
#[derive(Debug, Clone)]
pub enum UiElement {
    Circle(UiButtonCircle),
    Handle(UiButtonHandle),
    Polygon(UiButtonPolygon),
    Text(UiButtonText),
    Outline(UiButtonOutline),
    Rect(UiButtonRect),
    Advanced(AdvancedPrimitive),
}

impl UiElement {
    pub fn from_yaml(
        element: UiElementYaml,
        window_size: PhysicalSize<f32>,
        device: &Device,
        queue: &Queue,
        mod_manager: &ModManager,
    ) -> Option<UiElement> {
        let mut element = match element {
            UiElementYaml::Circle(e) => UiElement::Circle(UiButtonCircle::from_yaml(e)),
            UiElementYaml::Handle(e) => UiElement::Handle(UiButtonHandle::from_yaml(e)),
            UiElementYaml::Polygon(e) => {
                UiElement::Polygon(UiButtonPolygon::from_yaml(e, window_size))
            }
            UiElementYaml::Text(e) => UiElement::Text(UiButtonText::from_yaml(e)),
            UiElementYaml::Outline(e) => UiElement::Outline(UiButtonOutline::from_yaml(e)),
            UiElementYaml::Rect(e) => {
                UiElement::Rect(UiButtonRect::from_yaml(e, device, queue, mod_manager))
            }
            UiElementYaml::Advanced(ap) => UiElement::Advanced(AdvancedPrimitive::from_yaml(&ap)),
        };

        element.rescale_to_window(PhysicalSize::new(1920.0, 1080.0), window_size);

        Some(element)
    }

    pub fn to_yaml(&self, window_size: PhysicalSize<f32>) -> UiElementYaml {
        let mut element = self.clone();

        element.rescale_to_window(window_size, PhysicalSize::new(1920.0, 1080.0));

        match &element {
            UiElement::Circle(e) => UiElementYaml::Circle(e.to_yaml()),
            UiElement::Handle(e) => UiElementYaml::Handle(e.to_yaml()),
            UiElement::Polygon(e) => UiElementYaml::Polygon(e.to_yaml(window_size)),
            UiElement::Text(e) => UiElementYaml::Text(e.to_yaml()),
            UiElement::Outline(e) => UiElementYaml::Outline(e.to_yaml()),
            UiElement::Rect(e) => UiElementYaml::Rect(e.to_yaml()),
            UiElement::Advanced(ap) => UiElementYaml::Advanced(ap.to_yaml()),
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
    pub fn as_rect_mut(&mut self) -> Option<&mut UiButtonRect> {
        match self {
            UiElement::Rect(r) => Some(r),
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

    pub fn as_rect(&self) -> Option<&UiButtonRect> {
        match self {
            UiElement::Rect(r) => Some(r),
            _ => None,
        }
    }

    pub fn as_ap(&self) -> Option<&AdvancedPrimitive> {
        match self {
            UiElement::Advanced(ap) => Some(ap),
            _ => None,
        }
    }
    pub fn as_ap_mut(&mut self) -> Option<&mut AdvancedPrimitive> {
        match self {
            UiElement::Advanced(ap) => Some(ap),
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
            UiElement::Rect(_) => "Rect",
            UiElement::Advanced(_) => "Advanced",
        }
    }

    pub fn id(&self) -> &str {
        match self {
            UiElement::Text(t) => &t.id,
            UiElement::Circle(c) => &c.id,
            UiElement::Outline(o) => &o.id,
            UiElement::Handle(h) => &h.id,
            UiElement::Polygon(p) => &p.id,
            UiElement::Rect(r) => &r.id,
            UiElement::Advanced(ap) => &ap.id,
        }
    }

    pub fn set_id(&mut self, new_id: &String) {
        let new_id = new_id.clone();
        match self {
            UiElement::Text(t) => t.id = new_id,
            UiElement::Circle(c) => c.id = new_id,
            UiElement::Outline(o) => o.id = new_id,
            UiElement::Handle(h) => h.id = new_id,
            UiElement::Polygon(p) => p.id = new_id,
            UiElement::Rect(r) => r.id = new_id,
            UiElement::Advanced(ap) => ap.id = new_id,
        }
    }

    pub fn is_editable(&self, override_mode: bool) -> bool {
        match self {
            UiElement::Text(t) => t.misc.editable.editable(override_mode),
            UiElement::Circle(c) => c.misc.editable.editable(override_mode),
            UiElement::Outline(o) => o.misc.editable.editable(override_mode),
            UiElement::Handle(h) => h.misc.editable.editable(override_mode),
            UiElement::Polygon(p) => p.misc.editable.editable(override_mode),
            UiElement::Rect(r) => r.misc.editable.editable(override_mode),
            UiElement::Advanced(ap) => ap.misc.editable.editable(override_mode),
        }
    }

    #[inline]
    pub fn is_active(&self) -> bool {
        match self {
            UiElement::Text(t) => t.misc.active,
            UiElement::Circle(c) => c.misc.active,
            UiElement::Outline(o) => o.misc.active,
            UiElement::Handle(h) => h.misc.active,
            UiElement::Polygon(p) => p.misc.active,
            UiElement::Rect(r) => r.misc.active,
            UiElement::Advanced(ap) => ap.misc.active,
        }
    }

    pub fn is_touchable(&self) -> bool {
        match self {
            UiElement::Text(t) => t.misc.touchable,
            UiElement::Circle(c) => c.misc.touchable,
            UiElement::Outline(o) => o.misc.touchable,
            UiElement::Handle(h) => h.misc.touchable,
            UiElement::Polygon(p) => p.misc.touchable,
            UiElement::Rect(r) => r.misc.touchable,
            UiElement::Advanced(ap) => ap.misc.touchable,
        }
    }

    pub fn string_actions(&self) -> Vec<String> {
        match self {
            UiElement::Text(t) => t.string_actions.clone(),
            UiElement::Circle(c) => c.string_actions.clone(),
            UiElement::Outline(_) => vec![],
            UiElement::Handle(_) => vec![],
            UiElement::Polygon(p) => p.string_actions.clone(),
            UiElement::Rect(r) => r.string_actions.clone(),
            UiElement::Advanced(_) => vec![],
        }
    }

    pub fn compiled_actions(&self) -> Vec<CompiledAction> {
        match self {
            UiElement::Text(t) => t.compiled_actions.clone(),
            UiElement::Circle(c) => c.compiled_actions.clone(),
            UiElement::Outline(_) => vec![],
            UiElement::Handle(_) => vec![],
            UiElement::Polygon(p) => p.compiled_actions.clone(),
            UiElement::Rect(r) => r.compiled_actions.clone(),
            UiElement::Advanced(_) => vec![],
        }
    }

    pub fn set_actions(&mut self, actions: Vec<String>) {
        match self {
            UiElement::Text(e) => {
                e.string_actions = actions;
            }
            UiElement::Circle(e) => {
                e.string_actions = actions;
            }
            UiElement::Handle(e) => {}
            UiElement::Outline(e) => {}
            UiElement::Polygon(e) => {
                e.string_actions = actions;
            }
            UiElement::Rect(e) => {
                e.string_actions = actions;
            }
            UiElement::Advanced(e) => {}
        }
    }

    pub fn set_compiled_actions(&mut self, actions: Vec<CompiledAction>) {
        match self {
            UiElement::Text(e) => {
                e.compiled_actions = actions;
            }
            UiElement::Circle(e) => {
                e.compiled_actions = actions;
            }
            UiElement::Handle(e) => {}
            UiElement::Outline(e) => {}
            UiElement::Polygon(e) => {
                e.compiled_actions = actions;
            }
            UiElement::Rect(e) => {
                e.compiled_actions = actions;
            }
            UiElement::Advanced(e) => {}
        }
    }

    pub fn set_text(&mut self, text: String) {
        match self {
            UiElement::Text(e) => {
                e.text = text;
            }
            _ => {}
        }
    }
    pub fn set_template(&mut self, template: String) {
        match self {
            UiElement::Text(e) => {
                e.template = template;
            }
            _ => {}
        }
    }

    pub fn set_active(&mut self, active: bool) {
        match self {
            UiElement::Text(e) => {
                e.misc.active = active;
            }
            UiElement::Circle(e) => {
                e.misc.active = active;
            }
            UiElement::Handle(e) => {
                e.misc.active = active;
            }
            UiElement::Outline(e) => {
                e.misc.active = active;
            }
            UiElement::Polygon(e) => {
                e.misc.active = active;
            }
            UiElement::Rect(e) => {
                e.misc.active = active;
            }
            UiElement::Advanced(e) => {
                e.misc.active = active;
            }
        }
    }

    pub fn center(&self) -> [f32; 2] {
        match self {
            UiElement::Text(t) => [t.x, t.y],
            UiElement::Circle(c) => [c.x, c.y],
            UiElement::Handle(h) => [h.x, h.y],
            UiElement::Outline(o) => [o.shape_data.x, o.shape_data.y],
            UiElement::Polygon(p) => p.center(),
            UiElement::Rect(r) => [r.x, r.y],
            UiElement::Advanced(ap) => [ap.x, ap.y],
        }
    }
    pub fn size2(&self) -> Option<[f32; 2]> {
        match self {
            UiElement::Text(e) => Some([e.width, e.height]),
            UiElement::Rect(e) => Some(e.size()),
            _ => None,
        }
    }
    pub fn radius(&self) -> Option<f32> {
        match self {
            UiElement::Circle(e) => Some(e.radius),
            UiElement::Handle(e) => Some(e.radius),
            UiElement::Outline(e) => Some(e.shape_data.radius),
            _ => None,
        }
    }
    pub fn pt(&self) -> Option<f32> {
        if let Some(text) = self.as_text() {
            Some(text.pt)
        } else {
            None
        }
    }
    pub fn text(&self) -> Option<String> {
        match self {
            UiElement::Text(t) => Some(t.text.clone()),
            _ => None,
        }
    }
    pub fn template(&self) -> Option<String> {
        match self {
            UiElement::Text(t) => Some(t.template.clone()),
            _ => None,
        }
    }
    pub fn sizes(&self) -> Vec<SizeProperty> {
        match self {
            UiElement::Text(e) => e.sizes(),
            UiElement::Circle(e) => e.sizes(),
            UiElement::Handle(e) => e.sizes(),
            UiElement::Outline(e) => vec![
                SizeProperty::Radius(e.shape_data.radius),
                SizeProperty::Border(e.shape_data.border_thickness),
            ],
            UiElement::Polygon(e) => e.sizes(),
            UiElement::Rect(e) => e.sizes(),
            UiElement::Advanced(e) => vec![SizeProperty::AdvancedPrimitiveScale(e.scale)],
        }
    }
    pub fn main_size(&self) -> SizeProperty {
        match self {
            UiElement::Text(e) => e.main_size(),
            UiElement::Circle(e) => e.main_size(),
            UiElement::Handle(e) => e.main_size(),
            UiElement::Outline(e) => SizeProperty::Radius(e.shape_data.radius),
            UiElement::Polygon(e) => e.main_size(),
            UiElement::Rect(e) => e.main_size(),
            UiElement::Advanced(e) => SizeProperty::AdvancedPrimitiveScale(e.scale),
        }
    }
    pub fn kind(&self) -> ElementKind {
        match self {
            UiElement::Circle(_) => ElementKind::Circle,
            UiElement::Text(_) => ElementKind::Text,
            UiElement::Polygon(_) => ElementKind::Polygon,
            UiElement::Outline(_) => ElementKind::Outline,
            UiElement::Handle(_) => ElementKind::Handle,
            UiElement::Rect(_) => ElementKind::Rect,
            UiElement::Advanced(_) => ElementKind::Advanced,
        }
    }

    pub fn color_components(&self) -> Vec<ColorComponent> {
        match self {
            UiElement::Circle(_) => vec![
                ColorComponent::Fill,
                ColorComponent::Border,
                ColorComponent::InsideBorder,
            ],
            UiElement::Text(_) => vec![ColorComponent::Fill],
            UiElement::Polygon(_) => vec![ColorComponent::Fill, ColorComponent::VertexIndex(1)],
            UiElement::Outline(_) => vec![ColorComponent::DashColor, ColorComponent::SubDashColor],
            UiElement::Handle(_) => vec![ColorComponent::DashColor, ColorComponent::SubDashColor],
            UiElement::Rect(_) => vec![ColorComponent::Fill, ColorComponent::Border],
            UiElement::Advanced(_) => vec![],
        }
    }
    pub fn color(&self, component: ColorComponent) -> Option<[f32; 4]> {
        match self {
            UiElement::Circle(c) => match component {
                ColorComponent::Fill => Some(c.fill_color),
                ColorComponent::Border => Some(c.border_color),
                ColorComponent::InsideBorder => Some(c.inside_border_color),
                ColorComponent::Glow => Some(c.glow_color),
                _ => None,
            },

            UiElement::Text(t) => match component {
                ColorComponent::Fill => Some(t.color),
                _ => None,
            },

            UiElement::Polygon(p) => match component {
                ColorComponent::Fill => p.unscaled_vertices.first().map(|v| v.color),
                ColorComponent::VertexIndex(i) => {
                    p.unscaled_vertices.get(i as usize).map(|v| v.color)
                }
                _ => None,
            },

            UiElement::Outline(o) => match component {
                ColorComponent::DashColor => Some(o.dash_color),
                ColorComponent::SubDashColor => Some(o.sub_dash_color),
                _ => None,
            },

            UiElement::Handle(h) => match component {
                ColorComponent::DashColor => Some(h.handle_color),
                ColorComponent::SubDashColor => Some(h.sub_handle_color),
                _ => None,
            },

            UiElement::Rect(r) => match component {
                ColorComponent::Fill => Some(r.color),
                ColorComponent::Border => Some(r.border_color),
                _ => None,
            },

            UiElement::Advanced(_) => None,
        }
    }
    pub fn resize_behaviour(&self) -> ResizeBehaviour {
        match self {
            UiElement::Text(t) => t.resize_behaviour,
            UiElement::Circle(c) => c.resize_behaviour,
            UiElement::Outline(o) => ResizeBehaviour::default(),
            UiElement::Handle(h) => h.resize_behaviour,
            UiElement::Polygon(p) => p.resize_behaviour,
            UiElement::Rect(r) => r.resize_behaviour,
            UiElement::Advanced(ap) => ResizeBehaviour::default(),
        }
    }
    pub fn scale_by(&mut self, x_scale: f32, y_scale: f32, uniform_scale: f32) {
        let resize_behaviour = self.resize_behaviour();

        match self {
            UiElement::Text(t) => {
                let scale = resize_behaviour.uniform_scale(x_scale, y_scale, uniform_scale);

                t.pt *= scale;
            }

            UiElement::Circle(c) => {
                let scale = resize_behaviour.uniform_scale(x_scale, y_scale, uniform_scale);
                //r.border_thickness *= ;
                c.radius *= scale; // scale;
                c.border_thickness *= x_scale * y_scale;
            }

            UiElement::Outline(o) => {
                let scale = resize_behaviour.uniform_scale(x_scale, y_scale, uniform_scale);

                o.shape_data.radius *= scale;
                o.shape_data.border_thickness *= x_scale * y_scale;
            }

            UiElement::Handle(h) => {
                let scale = resize_behaviour.uniform_scale(x_scale, y_scale, uniform_scale);

                h.radius *= scale;
            }

            UiElement::Polygon(p) => {
                let scale = resize_behaviour.uniform_scale(x_scale, y_scale, uniform_scale);

                p.scale_by(scale);
            }

            UiElement::Rect(r) => {
                let width_scale = resize_behaviour.width_scale(x_scale, y_scale, uniform_scale);

                let height_scale = resize_behaviour.height_scale(x_scale, y_scale, uniform_scale);

                r.w *= width_scale;
                r.h *= height_scale;
                r.border_thickness *= x_scale * y_scale;
            }

            UiElement::Advanced(_) => {
                //Doesn't make sense
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
                p.x += dx;
                p.y += dy;
            }
            UiElement::Rect(r) => {
                r.x += dx;
                r.y += dy;
            }
            UiElement::Advanced(ap) => {
                ap.x += dx;
                ap.y += dy;
            }
        }
    }

    pub fn set_pos(&mut self, x: f32, y: f32) {
        match self {
            UiElement::Text(t) => {
                t.x = x;
                t.y = y;
            }
            UiElement::Circle(c) => {
                c.x = x;
                c.y = y;
            }
            UiElement::Handle(h) => {
                h.x = x;
                h.y = y;
            }
            UiElement::Outline(o) => {
                o.shape_data.x = x;
                o.shape_data.y = y;
            }
            UiElement::Polygon(p) => {
                p.x = x;
                p.y = y;
            }
            UiElement::Rect(r) => {
                r.x = x;
                r.y = y;
            }
            UiElement::Advanced(ap) => {
                ap.x = x;
                ap.y = y;
            }
        }
    }

    fn misc_mut(&mut self) -> &mut MiscButtonSettings {
        match self {
            UiElement::Text(t) => &mut t.misc,
            UiElement::Circle(c) => &mut c.misc,
            UiElement::Handle(h) => &mut h.misc,
            UiElement::Outline(o) => &mut o.misc,
            UiElement::Polygon(p) => &mut p.misc,
            UiElement::Rect(r) => &mut r.misc,
            UiElement::Advanced(ap) => &mut ap.misc,
        }
    }

    pub fn set_aps(&mut self, settings: &Settings, variables: &Variables, ap_vars: &[String]) {
        let ap_vars = ap_vars
            .iter()
            .map(|var| Value::from_str_pure(var).into_string())
            .collect::<Vec<String>>();
        let ap_vars = ap_vars.as_slice();
        match self {
            UiElement::Circle(e) => Self::replace_actions(&mut e.string_actions, ap_vars),
            UiElement::Handle(e) => {}
            UiElement::Polygon(e) => Self::replace_actions(&mut e.string_actions, ap_vars),
            UiElement::Text(e) => {
                Self::replace_actions(&mut e.string_actions, ap_vars);

                let mut text = e.template.clone();

                Self::replace_aps(&mut text, ap_vars);

                e.text = text.clone();
                e.template = text;
            }
            UiElement::Outline(e) => {}
            UiElement::Rect(e) => Self::replace_actions(&mut e.string_actions, ap_vars),
            UiElement::Advanced(e) => {}
        }
    }
    fn replace_aps(text: &mut String, ap_vars: &[String]) {
        while let Some(start) = text.find("{ap") {
            let rest = &text[start..];

            let (end, replacement) = if rest.starts_with("{ap}") {
                (start + 4, ap_vars.first().cloned().unwrap_or("None".into()))
            } else if rest.starts_with("{ap.all}") {
                let end = start + 8;
                (end, format!("[{}]", ap_vars.join(", ")))
            } else if rest.starts_with("{ap.") {
                if let Some(close_rel) = rest.find('}') {
                    let end = start + close_rel + 1;
                    let component = &text[start + 4..end - 1];
                    let idx = Variables::component_index(component).unwrap_or(0);

                    (end, ap_vars.get(idx).cloned().unwrap_or("None".into()))
                } else {
                    break;
                }
            } else {
                break;
            };

            text.replace_range(start..end, &replacement);
        }
    }
    fn replace_actions(actions: &mut Vec<String>, ap_vars: &[String]) {
        for action in actions {
            Self::replace_aps(action, ap_vars);
        }
    }
    pub fn set_editable(&mut self, editable: &Editability) {
        self.misc_mut().editable = editable.clone();
    }
    pub fn rescale_to_window(
        &mut self,
        old_window_size: PhysicalSize<f32>,
        new_window_size: PhysicalSize<f32>,
    ) {
        let old_x_scale = old_window_size.width / 1920.0;
        let old_y_scale = old_window_size.height / 1080.0;

        let new_x_scale = new_window_size.width / 1920.0;
        let new_y_scale = new_window_size.height / 1080.0;

        let x_scale = new_x_scale / old_x_scale;
        let y_scale = new_y_scale / old_y_scale;

        let old_uniform_scale = old_x_scale.min(old_y_scale);
        let new_uniform_scale = new_x_scale.min(new_y_scale);

        let uniform_scale = new_uniform_scale / old_uniform_scale;

        let current_pos = self.center();

        self.set_pos(current_pos[0] * x_scale, current_pos[1] * y_scale);

        self.scale_by(x_scale, y_scale, uniform_scale);
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
            UiElement::Rect(_) => dirty.mark_rects(),
            UiElement::Advanced(_) => dirty.mark_advanced_primitives(),
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
pub struct Vertex {
    /// @location(0) chunk-local position
    pub chunk_xz: [i32; 2],
    pub local_position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    pub quad_uv: [f32; 2],
}

impl Vertex {
    pub fn desc() -> VertexBufferLayout<'static> {
        use std::mem::size_of;
        VertexBufferLayout {
            array_stride: size_of::<Vertex>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &[
                // loc0 chunk_xz
                VertexAttribute {
                    shader_location: 0,
                    offset: 0,
                    format: VertexFormat::Sint32x2,
                },
                // @location(1) chunk-local position
                VertexAttribute {
                    shader_location: 1,
                    offset: 8,
                    format: VertexFormat::Float32x3,
                },
                // @location(2) normal
                VertexAttribute {
                    shader_location: 2,
                    offset: 20,
                    format: VertexFormat::Float32x3,
                },
                // @location(3) color
                VertexAttribute {
                    shader_location: 3,
                    offset: 32,
                    format: VertexFormat::Float32x3,
                },
                // @location(4) quad_uv
                VertexAttribute {
                    shader_location: 4,
                    offset: 44,
                    format: VertexFormat::Float32x2,
                }, // // @location(5) texture_id
                   // VertexAttribute {
                   //     shader_location: 5,
                   //     offset: 52,
                   //     format: VertexFormat::Uint32,
                   // },
            ],
        }
    }
}
pub trait VertexWithPosition {
    fn local_position(&self) -> [f32; 3];
    fn lerp(a: &Self, b: &Self, t: f32) -> Self;
}

impl VertexWithPosition for Vertex {
    fn local_position(&self) -> [f32; 3] {
        self.local_position
    }

    // produce a vertex that is a linear interpolation between a and b with factor t in [0,1]
    // must interpolate all vertex attributes consistently (position, normal, color).
    fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        // Linear interpolation helper for [f32; 3]
        fn mix3(x: [f32; 3], y: [f32; 3], t: f32) -> [f32; 3] {
            [
                x[0] + (y[0] - x[0]) * t,
                x[1] + (y[1] - x[1]) * t,
                x[2] + (y[2] - x[2]) * t,
            ]
        }

        // Linear interpolation helper for [f32; 2]
        fn mix2(x: [f32; 2], y: [f32; 2], t: f32) -> [f32; 2] {
            [x[0] + (y[0] - x[0]) * t, x[1] + (y[1] - x[1]) * t]
        }

        let position = mix3(a.local_position, b.local_position, t);
        let mut normal = mix3(a.normal, b.normal, t);
        let color = mix3(a.color, b.color, t);
        let quad_uv = mix2(a.quad_uv, b.quad_uv, t);

        // Normalize normal
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if len > 0.0 {
            normal[0] /= len;
            normal[1] /= len;
            normal[2] /= len;
        }

        // For chunk_xz, pick based on which vertex we're closer to
        let chunk_xz = if t <= 0.5 { a.chunk_xz } else { b.chunk_xz };
        //let texture_id = if t <= 0.5 { a.texture_id } else { b.texture_id };
        Vertex {
            local_position: position,
            normal,
            color,
            chunk_xz,
            quad_uv,
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
    pub depth: f32,
    pub _pad0: [f32; 3],
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
                VertexAttribute {
                    shader_location: 4,
                    format: VertexFormat::Float32,
                    offset: 48,
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
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RectParams {
    pub center: [f32; 2],    // center position
    pub half_size: [f32; 2], // width, height
    pub color: [f32; 4],     // RGBA
    pub border_color: [f32; 4],
    pub roundness: f32,        // corner radius
    pub border_thickness: f32, // computed from percentage
    pub rotation: f32,         // RAD to the GPU! DEG User-Facing in YAML and Runtime.
    pub fade: f32,
    pub glow_color: [f32; 4],
    pub glow_misc: [f32; 4], // glow_size, glow_speed, glow_intensity
    pub misc: [f32; 4],      // active, touched_time, is_down, hash
    pub blur: f32,
    pub depth: f32,
    pub _pad0: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Debug)]
pub struct UiVertexText {
    pub pos: [f32; 2],
    pub color: [f32; 4],
    pub depth: f32,
    pub _pad0: [f32; 3],
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
                    format: VertexFormat::Float32x4,
                },
                VertexAttribute {
                    offset: (size_of::<[f32; 2]>() + size_of::<[f32; 4]>()) as _,
                    shader_location: 2,
                    format: VertexFormat::Float32,
                },
            ],
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ElementKind {
    #[default]
    None,
    Text,
    Circle,
    Outline,
    Handle,
    Polygon,
    Advanced,
    Rect,
}

impl From<&UiElement> for ElementKind {
    fn from(element: &UiElement) -> Self {
        match element {
            UiElement::Circle(_) => ElementKind::Circle,
            UiElement::Handle(_) => ElementKind::Handle,
            UiElement::Polygon(_) => ElementKind::Polygon,
            UiElement::Text(_) => ElementKind::Text,
            UiElement::Outline(_) => ElementKind::Outline,
            UiElement::Rect(_) => ElementKind::Rect,
            UiElement::Advanced(_) => ElementKind::Advanced,
        }
    }
}
impl fmt::Display for ElementKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl ElementKind {
    pub fn from_string(kind: &str) -> Self {
        match kind.to_lowercase().as_str() {
            "rect" => ElementKind::Rect,
            "text" => ElementKind::Text,
            "circle" => ElementKind::Circle,
            "polygon" => ElementKind::Polygon,
            "advanced" => ElementKind::Advanced,
            "handle" => ElementKind::Handle,
            "outline" => ElementKind::Outline,
            _ => ElementKind::None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeLayer {
    pub name: String,
    pub ap_name: Option<String>,
    pub order: u32,
    pub string_actions: Vec<String>,
    pub compiled_actions: Vec<CompiledAction>,
    pub elements: Vec<UiElement>,
    pub active: bool,
    pub ap_vars: Vec<String>,

    pub dirty: LayerDirty, // set true when anything changes or the screen will be dirty asf!
    pub gpu: LayerGpu,
    pub opaque: bool,
    pub saveable: bool,
    pub editing_tool: bool,
    pub outline_poly_vertices: Vec<[f32; 2]>,
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

    pub fn find_element_mut(&mut self, id: &str) -> Option<&mut UiElement> {
        self.elements.iter_mut().find(|e| e.id() == id)
    }
    pub fn find_element(&self, id: &str) -> Option<&UiElement> {
        self.elements.iter().find(|e| e.id() == id)
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
    pub fn iter_texts_mut(&mut self) -> impl Iterator<Item = &mut UiButtonText> {
        self.elements.iter_mut().filter_map(UiElement::as_text_mut)
    }
    pub fn iter_rects(&self) -> impl Iterator<Item = &UiButtonRect> {
        self.elements.iter().filter_map(UiElement::as_rect)
    }
    pub fn iter_aps(&self) -> impl Iterator<Item = &AdvancedPrimitive> {
        self.elements.iter().filter_map(UiElement::as_ap)
    }
    pub fn iter_aps_mut(&mut self) -> impl Iterator<Item = &mut AdvancedPrimitive> {
        self.elements.iter_mut().filter_map(UiElement::as_ap_mut)
    }
    pub fn iter_outlines(&self) -> impl Iterator<Item = &UiButtonOutline> {
        self.elements.iter().filter_map(UiElement::as_outline)
    }

    #[inline]
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

    /// Clear all Polygon elements
    pub fn clear_rects(&mut self) {
        self.elements.retain(|e| !matches!(e, UiElement::Rect(_)));
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

    pub fn activate_all_elements(&mut self) {
        self.elements.iter_mut().for_each(|e| e.set_active(true))
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(default)] // This tells serde to fill missing fields with defaults when loading
pub struct UiLayerYaml {
    pub name: String,

    #[serde(default, skip_serializing_if = "is_default")] // Skips if 0
    pub order: u32,

    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "deserialize_string_or_vec"
    )]
    pub actions: Vec<String>,

    // Skips if None or Empty Vector
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elements: Option<Vec<UiElementYaml>>,

    // Default is true. Skip if true.
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub active: bool,

    // Default is false. Skip if false.
    #[serde(default, skip_serializing_if = "is_default")]
    pub opaque: bool,

    #[serde(default, skip_serializing_if = "is_false")]
    pub editing_tool: bool,
}

// Manual Default impl required because of the custom default values (like active=true)
impl Default for UiLayerYaml {
    fn default() -> Self {
        Self {
            name: "Layer".to_string(),
            order: 0,
            actions: vec![],
            elements: None,
            active: true,
            opaque: false,
            editing_tool: false,
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
    fn from_yaml(v: UiVertexYaml, id: usize, window_size: PhysicalSize<f32>) -> Self {
        let pos = v.pos;
        // let x_scale_reverse = window_size.width as f32 / 1920.0;
        // let y_scale_reverse = window_size.height as f32 / 1080.0;
        // pos[0] *= x_scale_reverse;
        // pos[1] *= y_scale_reverse;
        UiVertex {
            pos,
            color: v.color,
            roundness: v.roundness,
            _selected: false,
            id,
        }
    }

    pub fn to_yaml(&self, window_size: PhysicalSize<f32>) -> UiVertexYaml {
        let pos = self.pos;
        // let x_scale_reverse = 1920.0 / window_size.width as f32;
        // let y_scale_reverse = 1080.0 / window_size.height as f32;
        // pos[0] *= x_scale_reverse;
        // pos[1] *= y_scale_reverse;
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
    // relative to position of Polygon. Positions are in 1920x1080 space.
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
    pub fn scale_from_normalized(&self, window_size: PhysicalSize<f32>, scale: f32) -> ShapeData {
        let x = if self.x < 2.0 {
            window_size.width * self.x
        } else {
            self.x
        };
        let y = if self.y < 2.0 {
            window_size.height * self.y
        } else {
            self.y
        };
        let radius = if self.radius < 2.0 {
            scale * self.radius
        } else {
            self.radius
        };
        ShapeData {
            x,
            y,
            radius,
            border_thickness: self.border_thickness,
        }
    }

    pub fn scale_to_normalized(&self, window_size: PhysicalSize<f32>, scale: f32) -> ShapeData {
        ShapeData {
            x: self.x / window_size.width,
            y: self.y / window_size.height,
            radius: self.radius / scale,
            border_thickness: self.border_thickness,
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
pub enum Editability {
    Editable,
    NotEditable,
    HARDNOTEDITABLE,
}
impl Editability {
    pub fn from_bool(editable: bool) -> Self {
        if editable {
            Editability::Editable
        } else {
            Editability::NotEditable
        }
    }
    pub fn to_bool(&self) -> bool {
        match self {
            Editability::Editable => true,
            Editability::NotEditable => false,
            Editability::HARDNOTEDITABLE => false,
        }
    }
    pub fn editable(&self, override_mode: bool) -> bool {
        match self {
            Editability::Editable => true,
            Editability::NotEditable => override_mode,
            Editability::HARDNOTEDITABLE => false,
        }
    }
}
#[derive(Debug, Clone)]
pub struct MiscButtonSettings {
    pub active: bool,
    pub touched_time: f32,
    pub is_touched: bool,
    pub touchable: bool,
    pub editable: Editability,
}

impl MiscButtonSettings {
    pub fn to_yaml(&self) -> MiscButtonSettingsYaml {
        MiscButtonSettingsYaml {
            active: self.active,
            touchable: self.touchable,
            editable: self.editable.to_bool(),
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
#[serde(default)]
pub struct MiscButtonSettingsYaml {
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub active: bool,

    #[serde(
        default = "default_true",
        alias = "pressable",
        skip_serializing_if = "is_true"
    )]
    pub touchable: bool,

    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub editable: bool,
}

impl Default for MiscButtonSettingsYaml {
    fn default() -> Self {
        Self {
            active: true,
            touchable: true,
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
#[derive(Debug, Clone)]
pub struct UiButtonText {
    pub id: String,
    pub string_actions: Vec<String>,
    pub compiled_actions: Vec<CompiledAction>,
    pub style: String,
    pub x: f32,
    pub y: f32,
    pub pt: f32,
    pub resize_behaviour: ResizeBehaviour,
    pub original_pt: f32,
    pub color: [f32; 4],
    pub text: String,
    pub template: String,
    pub misc: MiscButtonSettings,

    pub width: f32,
    pub height: f32,
    pub being_edited: bool,
    pub caret: usize,
    pub being_hovered: bool,
    pub just_unhovered: bool,

    pub sel_start: usize, // selection start index
    pub sel_end: usize,   // selection end index
    pub has_selection: bool,

    pub buffer: sluggrs_skylines::Buffer,
    pub input_box: bool,
    pub anchor: Anchor,
    pub yaml_element: Option<UiButtonTextYaml>,

    pub cache: Option<TextParams>,
    pub decorations: Vec<TextDecoration>,
}

#[derive(Debug, Clone)]
pub struct UiButtonPolygon {
    pub id: String,
    pub x: f32,
    pub y: f32,
    pub resize_behaviour: ResizeBehaviour,
    pub scale: f32,
    cache_valid: bool,
    pub string_actions: Vec<String>,
    pub compiled_actions: Vec<CompiledAction>,
    pub style: String,
    cached_scaled_vertices: Vec<UiVertex>,
    pub unscaled_vertices: Vec<UiVertex>,
    pub misc: MiscButtonSettings,
    pub tri_count: u32,
    pub yaml_element: Option<UiButtonPolygonYaml>,

    pub cache: Option<Vec<UiVertexPoly>>,
}

#[derive(Debug, Clone)]
pub struct UiButtonCircle {
    pub id: String,
    pub string_actions: Vec<String>,
    pub compiled_actions: Vec<CompiledAction>,
    pub style: String,
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub resize_behaviour: ResizeBehaviour,
    pub original_radius: f32,
    pub inside_border_thickness: f32,
    pub border_thickness: f32,
    pub fade: f32,
    pub fill_color: [f32; 4],
    pub inside_border_color: [f32; 4],
    pub border_color: [f32; 4],
    pub glow_color: [f32; 4],
    pub glow_misc: GlowMisc,
    pub misc: MiscButtonSettings,
    pub yaml_element: Option<UiButtonCircleYaml>,
    pub cache: Option<CircleParams>,
}

#[derive(Debug, Clone)]
pub struct UiButtonOutline {
    pub id: String,
    pub parent: Option<ElementRef>,

    pub mode: f32, // 0 = circle, 1 = polygon

    pub vertex_offset: u32,    // index into global vertex buffer
    pub vertex_count: u32,     // how many vertices
    pub shape_data: ShapeData, // cx, cy, radius, thickness OR unused for poly

    pub dash_color: [f32; 4],
    pub dash_misc: DashMisc,
    pub sub_dash_color: [f32; 4],
    pub sub_dash_misc: DashMisc,

    pub misc: MiscButtonSettings,
    pub yaml_element: Option<UiButtonOutlineYaml>,
    pub cache: Option<OutlineParams>,
}

#[derive(Debug, Clone)]
pub struct UiButtonHandle {
    pub id: String,
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub resize_behaviour: ResizeBehaviour,
    pub handle_color: [f32; 4],
    pub handle_misc: HandleMisc,
    pub sub_handle_color: [f32; 4],
    pub sub_handle_misc: HandleMisc,
    pub misc: MiscButtonSettings,
    pub parent: Option<ElementRef>,
    pub yaml_element: Option<UiButtonHandleYaml>,

    pub cache: Option<HandleParams>,
}

impl UiButtonText {
    pub fn from_yaml(e: UiButtonTextYaml) -> Self {
        let length = e.text.len();
        let yaml_element = Some(e.clone());
        let (decorations, errors) = e.decorations_to_runtime();
        for err in errors {
            error!(
                "[UI] Error while converting YAML text to Runtime Text: '{:?}' in element: {}",
                err, e.id
            );
        }
        UiButtonText {
            id: e.id,
            string_actions: e.actions.clone(),
            compiled_actions: vec![],
            style: e.style.clone(),

            x: e.x as f32,
            y: e.y as f32,
            pt: e.pt,
            original_pt: e.pt,

            resize_behaviour: e.resize_behaviour,

            color: e.color,

            text: e.text.clone(),
            template: e.text,

            misc: MiscButtonSettings {
                active: e.misc.active,
                touched_time: 0.0,
                is_touched: false,
                touchable: e.misc.touchable,
                editable: Editability::from_bool(e.misc.editable),
            },

            width: 50.0,
            height: 20.0,

            being_edited: false,
            caret: length,
            being_hovered: false,
            just_unhovered: false,

            sel_start: 0,
            sel_end: 0,
            has_selection: false,

            input_box: e.input_box,
            anchor: e.anchor,

            yaml_element,
            cache: None,

            buffer: sluggrs_skylines::Buffer::new_empty(Metrics::new(e.pt, 20.0)),
            decorations,
        }
    }

    pub fn to_yaml(&self) -> UiButtonTextYaml {
        let (x, y, pt) = if let Some(yaml_element) = self.yaml_element.as_ref() {
            let x = snap(self.x, yaml_element.x as f32);
            let y = snap(self.y, yaml_element.y as f32);
            let pt = snap(self.pt, self.original_pt);
            (x, y, pt)
        } else {
            (self.x, self.y, self.pt)
        };

        UiButtonTextYaml {
            id: self.id.clone(),
            actions: self.string_actions.clone(),
            style: self.style.clone(),

            x: x as i16,
            y: y as i16,
            pt,

            resize_behaviour: self.resize_behaviour,

            color: self.color,
            text: self.template.clone(),

            misc: self.misc.to_yaml(),

            input_box: self.input_box,
            anchor: self.anchor,
            decorations: Self::decorations_to_yaml(self.decorations.clone()),
        }
    }

    fn decorations_to_yaml(decs: Vec<TextDecoration>) -> Vec<TextDecorationYaml> {
        decs.into_iter()
            .map(|d| TextDecorationYaml::from_runtime(d))
            .collect()
    }

    pub fn set_pos(&mut self, position: [f32; 2]) {
        self.x = position[0];
        self.y = position[1];
    }
}

impl UiButtonCircle {
    pub fn from_yaml(e: UiButtonCircleYaml) -> Self {
        let yaml_element = Some(e.clone());

        UiButtonCircle {
            id: e.id,
            string_actions: e.actions,
            compiled_actions: vec![],
            style: e.style,

            x: e.x as f32,
            y: e.y as f32,
            radius: e.radius,
            original_radius: e.radius,

            resize_behaviour: e.resize_behaviour,

            inside_border_thickness: e.inside_border_thickness,
            border_thickness: e.border_thickness,

            fade: e.fade,
            fill_color: e.fill_color,
            inside_border_color: e.inside_border_color,
            border_color: e.border_color,
            glow_color: e.glow_color,

            glow_misc: GlowMisc {
                glow_size: e.glow_misc.glow_size,
                glow_speed: e.glow_misc.glow_speed,
                glow_intensity: e.glow_misc.glow_intensity,
            },

            misc: MiscButtonSettings {
                active: e.misc.active,
                touched_time: 0.0,
                is_touched: false,
                touchable: e.misc.touchable,
                editable: Editability::from_bool(e.misc.editable),
            },

            yaml_element,
            cache: None,
        }
    }

    pub fn to_yaml(&self) -> UiButtonCircleYaml {
        let (x, y, radius) = if let Some(yaml_element) = self.yaml_element.as_ref() {
            let x = snap(self.x, yaml_element.x as f32);
            let y = snap(self.y, yaml_element.y as f32);
            let radius = snap(self.radius, yaml_element.radius);
            (x, y, radius)
        } else {
            (self.x, self.y, self.radius)
        };

        UiButtonCircleYaml {
            id: self.id.clone(),
            actions: self.string_actions.clone(),
            style: self.style.clone(),

            x: x as i16,
            y: y as i16,
            radius,

            resize_behaviour: self.resize_behaviour,

            inside_border_thickness: self.inside_border_thickness,
            border_thickness: self.border_thickness,

            fade: self.fade,
            fill_color: self.fill_color,
            inside_border_color: self.inside_border_color,
            border_color: self.border_color,
            glow_color: self.glow_color,
            glow_misc: self.glow_misc.clone(),

            misc: self.misc.to_yaml(),
        }
    }

    pub fn set_pos(&mut self, position: [f32; 2]) {
        self.x = position[0];
        self.y = position[1];
    }
}

impl UiButtonHandle {
    pub fn from_yaml(e: UiButtonHandleYaml) -> Self {
        let yaml_element = Some(e.clone());

        UiButtonHandle {
            id: e.id,

            x: e.x as f32,
            y: e.y as f32,
            radius: e.radius,

            resize_behaviour: e.resize_behaviour,

            handle_color: e.handle_color,
            handle_misc: e.handle_misc,

            sub_handle_color: e.sub_handle_color,
            sub_handle_misc: e.sub_handle_misc,

            parent: e.parent,

            misc: MiscButtonSettings {
                active: e.misc.active,
                touched_time: 0.0,
                is_touched: false,
                touchable: e.misc.touchable,
                editable: Editability::from_bool(e.misc.editable),
            },

            yaml_element,
            cache: None,
        }
    }

    pub fn to_yaml(&self) -> UiButtonHandleYaml {
        let (x, y, radius) = if let Some(yaml_element) = self.yaml_element.as_ref() {
            let x = snap(self.x, yaml_element.x as f32);
            let y = snap(self.y, yaml_element.y as f32);
            let radius = snap(self.radius, yaml_element.radius);
            (x, y, radius)
        } else {
            (self.x, self.y, self.radius)
        };

        UiButtonHandleYaml {
            id: self.id.clone(),

            x: x as i16,
            y: y as i16,
            radius,

            resize_behaviour: self.resize_behaviour,

            handle_color: self.handle_color,
            handle_misc: self.handle_misc.clone(),

            sub_handle_color: self.sub_handle_color,
            sub_handle_misc: self.sub_handle_misc.clone(),

            parent: self.parent.clone(),

            misc: self.misc.to_yaml(),
        }
    }

    pub fn set_pos(&mut self, position: [f32; 2]) {
        self.x = position[0];
        self.y = position[1];
    }
}

impl UiButtonOutline {
    pub fn from_yaml(e: UiButtonOutlineYaml) -> Self {
        let yaml_element = Some(e.clone());

        UiButtonOutline {
            id: e.id,
            parent: e.parent,
            mode: e.mode,

            vertex_offset: 0,
            vertex_count: 0,

            shape_data: ShapeData {
                x: e.shape_data.x,
                y: e.shape_data.y,
                radius: e.shape_data.radius,
                border_thickness: e.shape_data.border_thickness,
            },

            dash_color: e.dash_color,
            dash_misc: e.dash_misc,

            sub_dash_color: e.sub_dash_color,
            sub_dash_misc: e.sub_dash_misc,

            misc: MiscButtonSettings {
                active: e.misc.active,
                touched_time: 0.0,
                is_touched: false,
                touchable: e.misc.touchable,
                editable: Editability::from_bool(e.misc.editable),
            },

            yaml_element,
            cache: None,
        }
    }

    pub fn to_yaml(&self) -> UiButtonOutlineYaml {
        let (x, y, radius) = if let Some(yaml_element) = self.yaml_element.as_ref() {
            let x = snap(self.shape_data.x, yaml_element.shape_data.x);
            let y = snap(self.shape_data.y, yaml_element.shape_data.y);
            let radius = snap(self.shape_data.radius, yaml_element.shape_data.radius);
            (x, y, radius)
        } else {
            (self.shape_data.x, self.shape_data.y, self.shape_data.radius)
        };

        UiButtonOutlineYaml {
            id: self.id.clone(),
            parent: self.parent.clone(),

            mode: self.mode,

            shape_data: ShapeData {
                x,
                y,
                radius,

                border_thickness: self.shape_data.border_thickness,
            },

            dash_color: self.dash_color,
            dash_misc: self.dash_misc.clone(),

            sub_dash_color: self.sub_dash_color,
            sub_dash_misc: self.sub_dash_misc.clone(),

            misc: self.misc.to_yaml(),
        }
    }

    pub fn set_pos(&mut self, position: [f32; 2]) {
        self.shape_data.x = position[0];
        self.shape_data.y = position[1];
    }
}

impl UiButtonPolygon {
    pub fn from_yaml(e: UiButtonPolygonYaml, window_size: PhysicalSize<f32>) -> Self {
        let mut id_gen = 1;
        let yaml_element = Some(e.clone());

        let mut verts: Vec<UiVertex> = e
            .vertices
            .into_iter()
            .map(|vj| {
                id_gen += 1;
                UiVertex::from_yaml(vj, id_gen, window_size)
            })
            .collect();

        ensure_ccw(&mut verts);

        let mut polygon = UiButtonPolygon {
            id: e.id,

            x: e.x as f32,
            y: e.y as f32,
            scale: e.scale,

            resize_behaviour: e.resize_behaviour,

            cache_valid: false,

            string_actions: e.actions,
            compiled_actions: vec![],

            style: e.style,

            cached_scaled_vertices: vec![],
            unscaled_vertices: verts,

            misc: MiscButtonSettings {
                active: e.misc.active,
                touched_time: 0.0,
                is_touched: false,
                touchable: e.misc.touchable,
                editable: Editability::from_bool(e.misc.editable),
            },

            tri_count: 0,

            yaml_element,
            cache: None,
        };

        polygon.update_scaled_vertices();
        polygon
    }

    pub fn to_yaml(&self, window_size: PhysicalSize<f32>) -> UiButtonPolygonYaml {
        let (x, y, scale) = if let Some(yaml_element) = self.yaml_element.as_ref() {
            let x = snap(self.x, yaml_element.x as f32);
            let y = snap(self.y, yaml_element.y as f32);
            let scale = snap(self.scale, yaml_element.scale);
            (x, y, scale)
        } else {
            (self.x, self.y, self.scale)
        };

        UiButtonPolygonYaml {
            id: self.id.clone(),

            actions: self.string_actions.clone(),
            style: self.style.clone(),

            x: x as i16,
            y: y as i16,
            scale,

            resize_behaviour: self.resize_behaviour,

            vertices: self
                .unscaled_vertices
                .iter()
                .map(|v| v.to_yaml(window_size))
                .collect(),

            misc: self.misc.to_yaml(),
        }
    }

    pub fn center(&self) -> [f32; 2] {
        [self.x, self.y]
    }

    /// Resizes the polygon uniformly from its center.
    /// `scale` represents the factor to resize by.
    pub fn scale_by(&mut self, scale: f32) {
        self.scale *= scale;
        self.update_scaled_vertices();
    }
    pub fn scale_to(&mut self, scale: f32) {
        self.scale = scale;
        self.update_scaled_vertices();
    }
    /// Returns the current size of the polygon.
    /// Size is defined as the maximum distance from center to any vertex.
    pub fn max_size(&self) -> f32 {
        let center = self.center();

        self.scaled_vertices()
            .iter()
            .map(|v| {
                let dx = v.pos[0] - center[0];
                let dy = v.pos[1] - center[1];
                (dx * dx + dy * dy).sqrt()
            })
            .fold(0.0_f32, |a, b| a.max(b))
    }
    pub fn scaled_vertices_are_valid(&self) -> bool {
        self.cache_valid
    }
    pub fn invalidate_scaled_vertices_cache(&mut self) {
        self.cache_valid = false;
    }
    fn validate_scaled_vertices_cache(&mut self) {
        self.cache_valid = true;
    }
    pub fn scaled_vertices(&self) -> Vec<UiVertex> {
        if self.scaled_vertices_are_valid() {
            return self.cached_scaled_vertices.clone();
        };
        let (x, y) = (self.x, self.y);

        self.unscaled_vertices
            .iter()
            .map(|v| UiVertex {
                pos: [
                    x + (v.pos[0] - x) * self.scale,
                    y + (v.pos[1] - y) * self.scale,
                ],
                ..*v
            })
            .collect()
    }
    pub fn update_scaled_vertices(&mut self) {
        self.cached_scaled_vertices = self.scaled_vertices();
        self.validate_scaled_vertices_cache();
    }

    pub fn set_pos(&mut self, position: [f32; 2]) {
        self.x = position[0];
        self.y = position[1];
    }
}

impl Default for UiButtonText {
    fn default() -> Self {
        Self {
            id: "None".to_string(),
            string_actions: vec![],
            compiled_actions: vec![],
            style: "None".to_string(),
            x: 0.0,
            y: 0.0,
            pt: 14.0,
            resize_behaviour: Default::default(),
            original_pt: 14.0,
            color: [1.0, 1.0, 1.0, 1.0],
            text: "default".into(),
            template: "default".to_string(),
            misc: MiscButtonSettings::default(),
            width: 50.0,
            height: 20.0,
            being_edited: false,
            caret: 0,
            being_hovered: false,
            just_unhovered: false,
            sel_start: 0,
            sel_end: 0,
            has_selection: false,
            input_box: false,
            anchor: Anchor::default(),
            yaml_element: None,
            cache: None,
            buffer: sluggrs_skylines::Buffer::new_empty(Metrics::new(14.0, 20.0)),
            decorations: vec![],
        }
    }
}

impl Default for UiButtonPolygon {
    fn default() -> Self {
        let verts = vec![
            UiVertex {
                pos: [-60.0, 60.0],
                color: [1.0, 1.0, 1.0, 1.0],
                roundness: 0.0,
                _selected: false,
                id: 0,
            },
            UiVertex {
                pos: [0.0, -60.0],
                color: [1.0, 1.0, 1.0, 1.0],
                roundness: 0.0,
                _selected: false,
                id: 1,
            },
            UiVertex {
                pos: [60.0, 60.0],
                color: [1.0, 1.0, 1.0, 1.0],
                roundness: 0.0,
                _selected: false,
                id: 2,
            },
            UiVertex {
                pos: [60.0, 100.0],
                color: [1.0, 1.0, 0.0, 1.0],
                roundness: 0.0,
                _selected: false,
                id: 3,
            },
        ];

        Self {
            id: "None".to_string(),
            x: 0.0,
            y: 0.0,
            resize_behaviour: Default::default(),
            scale: 1.0,
            cache_valid: false,
            string_actions: vec![],
            compiled_actions: vec![],
            style: "None".to_string(),
            unscaled_vertices: verts.clone(),
            cached_scaled_vertices: vec![],
            misc: MiscButtonSettings::default(),
            tri_count: 0,
            yaml_element: None,
            cache: None,
        }
    }
}

impl Default for UiButtonCircle {
    fn default() -> Self {
        Self {
            id: "None".to_string(),
            string_actions: vec![],
            compiled_actions: vec![],
            style: "None".to_string(),
            x: 0.0,
            y: 0.0,
            radius: 50.0,
            resize_behaviour: Default::default(),
            original_radius: 50.0,
            inside_border_thickness: 0.0,
            border_thickness: 0.05,
            fade: 0.0,
            fill_color: [1.0, 1.0, 1.0, 1.0],
            inside_border_color: [0.0, 0.0, 0.0, 1.0],
            border_color: [0.0, 0.0, 0.0, 1.0],
            glow_color: [1.0, 1.0, 1.0, 0.0],
            glow_misc: GlowMisc::default(),
            misc: MiscButtonSettings::default(),
            yaml_element: None,
            cache: None,
        }
    }
}

impl Default for UiButtonOutline {
    fn default() -> Self {
        Self {
            id: "None".to_string(),
            parent: None,
            mode: 1.0,
            vertex_offset: 0,
            vertex_count: 0,
            shape_data: ShapeData::default(),
            dash_color: [1.0, 1.0, 1.0, 1.0],
            dash_misc: DashMisc::default(),
            sub_dash_color: [1.0, 1.0, 1.0, 1.0],
            sub_dash_misc: DashMisc::default(),
            misc: MiscButtonSettings::default(),
            yaml_element: None,
            cache: None,
        }
    }
}

impl Default for UiButtonHandle {
    fn default() -> Self {
        Self {
            id: "None".to_string(),
            x: 0.0,
            y: 0.0,
            radius: 6.0,
            resize_behaviour: Default::default(),
            handle_color: [1.0, 1.0, 1.0, 1.0],
            handle_misc: HandleMisc::default(),
            sub_handle_color: [1.0, 1.0, 1.0, 1.0],
            sub_handle_misc: HandleMisc::default(),
            misc: MiscButtonSettings::default(),
            parent: None,
            yaml_element: None,
            cache: None,
        }
    }
}
impl Default for UiButtonRect {
    fn default() -> Self {
        Self {
            id: "None".to_string(),
            string_actions: vec![],
            compiled_actions: vec![],
            style: "".to_string(),
            x: 0.0,
            y: 0.0,
            w: 10.0,
            h: 10.0,
            resize_behaviour: ResizeBehaviour::default(),
            rotation: 30.0,
            color: [1.0, 1.0, 1.0, 1.0],
            border_color: [1.0, 1.0, 1.0, 1.0],
            texture: None,
            cached_texture: None,
            roundness: 0.0,
            border_thickness: 0.0,
            fade: 0.0,
            blur: 0.2,
            glow_color: [0.2, 0.0, 0.8, 0.98],
            glow_misc: GlowMisc::default(),
            misc: MiscButtonSettings::default(),
            yaml_element: None,
            cache: None,
        }
    }
}
impl Default for AdvancedPrimitive {
    fn default() -> Self {
        Self {
            id: "default".to_string(),
            ap_name: "".to_string(),
            ap_vars: vec![],
            actions: vec![],
            x: 0.0,
            y: 0.0,
            scale: 1.0,
            misc: MiscButtonSettings::default(),
            editing_tool: false,
            is_temporary: false,
            scale_my_coords: true,
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
            touchable: true,
            editable: Editability::Editable,
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
    #[serde(
        default = "default_none_string",
        skip_serializing_if = "is_none_string"
    )]
    pub id: String,

    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "deserialize_string_or_vec"
    )]
    pub actions: Vec<String>,

    #[serde(
        default = "default_none_string",
        skip_serializing_if = "is_none_string"
    )]
    pub style: String,

    #[serde(deserialize_with = "deserialize_i16_from_number")]
    pub x: i16,
    #[serde(deserialize_with = "deserialize_i16_from_number")]
    pub y: i16,

    #[serde(skip_serializing_if = "is_default")]
    pub pt: f32,
    #[serde(default, skip_serializing_if = "is_default")]
    pub resize_behaviour: ResizeBehaviour,
    #[serde(default, skip_serializing_if = "is_default")]
    pub color: [f32; 4],
    pub text: String,

    // If 'misc' matches defaults (active:true, pressable:false, editable:false),
    // this entire block is removed from YAML.
    #[serde(default, skip_serializing_if = "is_default")]
    pub misc: MiscButtonSettingsYaml,

    #[serde(skip_serializing_if = "is_default")]
    pub input_box: bool,

    #[serde(default, skip_serializing_if = "is_default")]
    pub anchor: Anchor,

    #[serde(default, skip_serializing_if = "is_default")]
    pub decorations: Vec<TextDecorationYaml>,
}

impl UiButtonTextYaml {
    pub fn decorations_to_runtime(&self) -> (Vec<TextDecoration>, Vec<(usize, String)>) {
        let mut result = Vec::with_capacity(self.decorations.len());
        let mut ring_count = 0;
        let mut errors = Vec::new();
        for (idx, decoration) in self.decorations.iter().enumerate() {
            let decoration = match decoration.to_runtime() {
                Ok(decoration) => decoration,
                Err(e) => {
                    errors.push((idx, e));
                    continue;
                }
            };

            if decoration.mode == DecorationMode::Ring {
                ring_count += 1;
                if ring_count > 1 {
                    errors.push((
                        idx,
                        "At most one Ring decoration per text area!! 🖕".to_string(),
                    ));
                    continue;
                }
            }

            result.push(decoration);
        }

        (result, errors)
    }
}
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
pub struct TextDecorationYaml {
    #[serde(default = "default_border_color", skip_serializing_if = "is_default")]
    pub color: [f32; 4],
    #[serde(default, skip_serializing_if = "is_default")]
    pub spread: f32,
    #[serde(default, skip_serializing_if = "is_default")]
    pub offset: [f32; 2],
    #[serde(default, skip_serializing_if = "is_default")]
    pub blur: f32,
    #[serde(default, skip_serializing_if = "is_default")]
    pub mode: Option<DecorationMode>,
}

impl TextDecorationYaml {
    fn to_runtime(&self) -> Result<TextDecoration, String> {
        if self.blur > 0.0 && self.spread > 0.0 {
            return Err("A blurred text decoration cannot also carry a spread! 😂".to_string());
        }

        let mode = self.mode.unwrap_or_else(|| {
            if self.spread > 0.0 {
                DecorationMode::Ring
            } else {
                DecorationMode::Solid
            }
        });

        Ok(TextDecoration {
            color: stupid_color_from_rgba(self.color),
            spread: self.spread,
            offset: self.offset,
            blur: self.blur,
            mode,
        })
    }
    fn from_runtime(dec: TextDecoration) -> TextDecorationYaml {
        TextDecorationYaml {
            color: rgba_from_stupid_color(dec.color),
            spread: dec.spread,
            offset: dec.offset,
            blur: dec.blur,
            mode: Some(dec.mode),
        }
    }
}
impl Default for UiButtonTextYaml {
    fn default() -> Self {
        Self {
            id: "None".to_string(),
            actions: Vec::new(),
            style: "None".to_string(),
            x: 0,
            y: 0,
            pt: 14.0,
            resize_behaviour: Default::default(),
            color: [1.0, 1.0, 1.0, 1.0],
            text: String::new(),
            misc: MiscButtonSettingsYaml::default(),
            input_box: false,
            anchor: Anchor::default(),
            decorations: vec![],
        }
    }
}

fn default_border_color() -> [f32; 4] {
    [0.0, 0.0, 0.0, 1.0]
}
// --- CIRCLE ---
#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(default)]
pub struct UiButtonCircleYaml {
    #[serde(
        default = "default_none_string",
        skip_serializing_if = "is_none_string"
    )]
    pub id: String,

    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "deserialize_string_or_vec"
    )]
    pub actions: Vec<String>,

    #[serde(
        default = "default_none_string",
        skip_serializing_if = "is_none_string"
    )]
    pub style: String,

    #[serde(deserialize_with = "deserialize_i16_from_number")]
    pub x: i16,
    #[serde(deserialize_with = "deserialize_i16_from_number")]
    pub y: i16,
    pub radius: f32,
    #[serde(default, skip_serializing_if = "is_default")]
    pub resize_behaviour: ResizeBehaviour,

    #[serde(skip_serializing_if = "is_default")]
    pub inside_border_thickness: f32,

    #[serde(skip_serializing_if = "is_default")]
    pub border_thickness: f32,

    #[serde(skip_serializing_if = "is_default")]
    pub fade: f32,
    #[serde(skip_serializing_if = "is_default")]
    pub fill_color: [f32; 4],
    #[serde(skip_serializing_if = "is_default")]
    pub inside_border_color: [f32; 4],
    #[serde(skip_serializing_if = "is_default")]
    pub border_color: [f32; 4],

    #[serde(default)]
    pub glow_color: [f32; 4],
    #[serde(default)]
    pub glow_misc: GlowMisc,

    #[serde(default, skip_serializing_if = "is_default")]
    pub misc: MiscButtonSettingsYaml,
}

impl Default for UiButtonCircleYaml {
    fn default() -> Self {
        Self {
            id: "None".to_string(),
            actions: Vec::new(),
            style: "None".to_string(),
            x: 0,
            y: 0,
            radius: 0.0,
            resize_behaviour: Default::default(),
            inside_border_thickness: 0.0,
            border_thickness: 0.0,
            fade: 0.0,
            fill_color: [0.0, 0.0, 0.0, 0.0],
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
    #[serde(
        default = "default_none_string",
        skip_serializing_if = "is_none_string"
    )]
    pub id: String,

    #[serde(deserialize_with = "deserialize_i16_from_number")]
    pub x: i16,
    #[serde(deserialize_with = "deserialize_i16_from_number")]
    pub y: i16,
    pub radius: f32,
    #[serde(default, skip_serializing_if = "is_default")]
    pub resize_behaviour: ResizeBehaviour,

    pub handle_color: [f32; 4],

    #[serde(skip_serializing_if = "is_default")]
    pub handle_misc: HandleMisc, // Assuming HandleMisc exists and derives Default+PartialEq

    pub sub_handle_color: [f32; 4],

    #[serde(skip_serializing_if = "is_default")]
    pub sub_handle_misc: HandleMisc,

    #[serde(default, skip_serializing_if = "is_default")]
    pub misc: MiscButtonSettingsYaml,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<ElementRef>,
}

impl Default for UiButtonHandleYaml {
    fn default() -> Self {
        Self {
            id: "None".to_string(),
            x: 0,
            y: 0,
            radius: 0.0,
            resize_behaviour: Default::default(),
            handle_color: [1.0, 1.0, 1.0, 1.0],
            handle_misc: HandleMisc::default(),
            sub_handle_color: [1.0, 1.0, 1.0, 1.0],
            sub_handle_misc: HandleMisc::default(),
            misc: MiscButtonSettingsYaml::default(),
            parent: None,
        }
    }
}

// --- OUTLINE ---
#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(default)]
pub struct UiButtonOutlineYaml {
    #[serde(
        default = "default_none_string",
        skip_serializing_if = "is_none_string"
    )]
    pub id: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<ElementRef>,

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

    #[serde(default, skip_serializing_if = "is_default")]
    pub misc: MiscButtonSettingsYaml,
}

impl Default for UiButtonOutlineYaml {
    fn default() -> Self {
        Self {
            id: "None".to_string(),
            parent: None,
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
    #[serde(
        default = "default_none_string",
        skip_serializing_if = "is_none_string"
    )]
    pub id: String,

    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "deserialize_string_or_vec"
    )]
    pub actions: Vec<String>,

    #[serde(
        default = "default_none_string",
        skip_serializing_if = "is_none_string"
    )]
    pub style: String,
    #[serde(deserialize_with = "deserialize_i16_from_number")]
    pub x: i16,
    #[serde(deserialize_with = "deserialize_i16_from_number")]
    pub y: i16,
    #[serde(skip_serializing_if = "is_one")]
    pub scale: f32,
    #[serde(default, skip_serializing_if = "is_default")]
    pub resize_behaviour: ResizeBehaviour,
    // If vertices are empty, we might as well skip, but usually polygon has data
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub vertices: Vec<UiVertexYaml>,

    #[serde(default, skip_serializing_if = "is_default")]
    pub misc: MiscButtonSettingsYaml,
}

impl Default for UiButtonPolygonYaml {
    fn default() -> Self {
        Self {
            id: "None".to_string(),
            actions: vec![],
            style: "None".to_string(),
            x: 0,
            y: 0,
            scale: 1.0,
            resize_behaviour: Default::default(),
            vertices: Vec::new(),
            misc: MiscButtonSettingsYaml::default(),
        }
    }
}
// Checks if a standard type matches its default (e.g., false for bool, 0 for u32)
fn is_default<T: Default + PartialEq>(t: &T) -> bool {
    t == &T::default()
}

fn is_false(value: &bool) -> bool {
    !*value
}

// Checks if a boolean is true (useful for things like 'active' where default is true)
fn is_true(b: &bool) -> bool {
    *b
}
fn is_one(v: &f32) -> bool {
    (*v - 1.0).abs() < 1e-6
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
pub fn default_true() -> bool {
    true
}

// Check: is [f32; 2] == [0.0, 0.0]?
fn is_zero_vec2(v: &[f32; 2]) -> bool {
    v[0] == 0.0 && v[1] == 0.0
}
fn string_or_vec<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct StringOrVec;

    impl<'de> Visitor<'de> for StringOrVec {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a string or a list of strings")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(vec![value.to_string()])
        }

        fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(vec![value])
        }

        fn visit_seq<A>(self, seq: A) -> Result<Self::Value, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }
    }

    deserializer.deserialize_any(StringOrVec)
}

fn deserialize_i16_from_number<'de, D>(deserializer: D) -> Result<i16, D::Error>
where
    D: Deserializer<'de>,
{
    let value = serde_yaml::Value::deserialize(deserializer)?;

    match value {
        serde_yaml::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i as i16)
            } else if let Some(f) = n.as_f64() {
                Ok(f.round() as i16)
            } else {
                Err(de::Error::custom("invalid number"))
            }
        }
        _ => Err(de::Error::custom("expected number")),
    }
}
fn deserialize_u16_from_number<'de, D>(deserializer: D) -> Result<u16, D::Error>
where
    D: Deserializer<'de>,
{
    let value = serde_yaml::Value::deserialize(deserializer)?;

    match value {
        serde_yaml::Value::Number(n) => {
            if let Some(i) = n.as_u64() {
                Ok(i as u16)
            } else if let Some(f) = n.as_f64() {
                Ok(f.round() as u16)
            } else {
                Err(de::Error::custom("invalid number"))
            }
        }
        _ => Err(de::Error::custom("expected number")),
    }
}

fn deserialize_string_or_vec<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct StringOrVecVisitor;

    impl<'de> Visitor<'de> for StringOrVecVisitor {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a string or a sequence of strings")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(vec![value.to_string()])
        }

        fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(vec![value])
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            let mut values = Vec::new();

            while let Some(value) = seq.next_element::<String>()? {
                values.push(value);
            }

            Ok(values)
        }
    }

    deserializer.deserialize_any(StringOrVecVisitor)
}
