use crate::renderer::helper;
use crate::renderer::ui::{CircleOutlineParams, CircleParams, TextParams};
use crate::resources::MouseState;
use crate::vertex::{
    DashMisc, GuiLayout, LayerGpu, UiButtonCircle, UiButtonCircleOutline, UiButtonPolygon,
    UiButtonRectangle, UiButtonText, UiButtonTriangle, UiVertexPoly,
};
use std::collections::HashMap;
use std::fs;
use std::io::Result;
use std::path::PathBuf;

#[derive(Debug, serde::Deserialize)]
pub struct UiLayer {
    pub name: String,
    pub order: u32,
    pub texts: Option<Vec<UiButtonText>>,
    pub circles: Option<Vec<UiButtonCircle>>,
    pub circle_outlines: Option<Vec<UiButtonCircleOutline>>,
    pub rectangles: Option<Vec<UiButtonRectangle>>,
    pub triangles: Option<Vec<UiButtonTriangle>>,
    pub polygons: Option<Vec<UiButtonPolygon>>,
    pub active: Option<bool>,
    pub opaque: Option<bool>,
}
pub struct LayerCache {
    pub texts: Vec<TextParams>,
    pub circle_params: Vec<CircleParams>,
    pub circle_outline_params: Vec<CircleOutlineParams>,
    pub rect_vertices: Vec<UiVertexPoly>,
    pub triangle_vertices: Vec<UiVertexPoly>,
    pub polygon_vertices: Vec<UiVertexPoly>,
}
#[derive(Debug, Clone, Default)]
pub struct SelectedUiElement {
    pub layer_name: String,
    pub element_id: String,
    pub active: bool,
}

impl Default for LayerCache {
    fn default() -> Self {
        Self {
            texts: vec![],
            circle_params: vec![],
            circle_outline_params: vec![],
            rect_vertices: vec![],
            triangle_vertices: vec![],
            polygon_vertices: vec![],
        }
    }
}

#[derive(Clone, Debug)]
pub enum UiElement {
    Circle(UiButtonCircle),
    Rectangle(UiButtonRectangle),
    Triangle(UiButtonTriangle),
    Polygon(UiButtonPolygon),
    Text(UiButtonText),
}

pub struct RuntimeLayer {
    pub name: String,
    pub order: u32,
    pub texts: Vec<UiButtonText>,
    pub circles: Vec<UiButtonCircle>,
    pub circle_outlines: Vec<UiButtonCircleOutline>,
    pub rectangles: Vec<UiButtonRectangle>,
    pub triangles: Vec<UiButtonTriangle>,
    pub polygons: Vec<UiButtonPolygon>,
    pub active: bool,
    // NEW: cached GPU data!!!
    pub cache: LayerCache,

    pub dirty: bool, // set true when anything changes or the screen will be dirty asf!
    pub gpu: LayerGpu,
    pub opaque: bool,
}

pub struct UiButtonLoader {
    pub layers: Vec<RuntimeLayer>,
    pub ui_runtime: UiRuntime,
    pub id_lookup: HashMap<u32, String>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ButtonRuntime {
    pub touched_time: f32,
    pub is_down: bool,
    pub just_pressed: bool,
    pub just_released: bool,
}

pub struct UiRuntime {
    pub elements: HashMap<String, ButtonRuntime>,
    pub selected_ui_element: SelectedUiElement,
}

impl UiRuntime {
    pub fn new() -> Self {
        Self {
            elements: HashMap::new(),
            selected_ui_element: SelectedUiElement::default(),
        }
    }

    pub fn update_touch(
        &mut self,
        id: &str,
        touched_now: bool,
        dt: f32,
        layer_name: &String,
    ) -> bool {
        let entry = self
            .elements
            .entry(id.to_string())
            .or_insert_with(ButtonRuntime::default);

        entry.just_pressed = false;
        entry.just_released = false;

        match (entry.is_down, touched_now) {
            // pressed this frame
            (false, true) => {
                entry.is_down = true;
                entry.just_pressed = true;
                entry.touched_time = 0.0;
                println!("{dt}Pressed {}", id);
                false
            }

            // still holding
            (true, true) => {
                entry.touched_time += dt;
                false
            }

            // released this frame
            (true, false) => {
                entry.is_down = false;
                entry.just_released = true;
                println!("Released {}", id);
                self.selected_ui_element = SelectedUiElement {
                    layer_name: layer_name.clone(),
                    element_id: id.parse().unwrap(),
                    active: true,
                };
                true
            }

            // idle
            (false, false) => {
                entry.touched_time = 0.0;
                false
            }
        }
    }

    pub fn get(&self, id: &str) -> ButtonRuntime {
        *self.elements.get(id).unwrap_or(&ButtonRuntime::default())
    }
}

impl UiButtonLoader {
    pub fn new() -> Self {
        let layout = Self::load_gui_from_file("ui_data/gui_layout.json").unwrap_or_else(|e| {
            eprintln!("❌ Failed to load GUI layout: {e}");
            GuiLayout { layers: vec![] }
        });

        let mut loader = Self {
            layers: Vec::new(),
            ui_runtime: UiRuntime::new(),
            id_lookup: HashMap::new(),
        };

        // JSON layers to runtime layers
        for l in layout.layers {
            loader.layers.push(RuntimeLayer {
                name: l.name,
                order: l.order,
                active: l.active.unwrap_or(true),
                cache: Default::default(),
                texts: l.texts.unwrap_or_default(),
                circles: l.circles.unwrap_or_default(),
                circle_outlines: l.circle_outlines.unwrap_or_default(),
                rectangles: l.rectangles.unwrap_or_default(),
                triangles: l.triangles.unwrap_or_default(),
                polygons: l.polygons.unwrap_or_default(),
                dirty: true,
                gpu: LayerGpu::default(),
                opaque: l.opaque.unwrap_or(false),
            });
        }

        loader.add_editor_layers();
        loader.layers.sort_by_key(|l| l.order); // SORT!

        loader
    }

    pub fn load_gui_from_file(path: &str) -> Result<GuiLayout> {
        let mut full_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        full_path.push("src/renderer");
        full_path.push(path);

        let data = fs::read_to_string(&full_path)?;
        let parsed: GuiLayout = serde_json::from_str(&data)?;
        Ok(parsed)
    }

    fn add_editor_layers(&mut self) {
        self.layers.push(RuntimeLayer {
            name: "editor_selection".into(),
            order: 900,
            active: false,
            cache: LayerCache::default(),
            texts: vec![],
            circles: vec![],
            circle_outlines: vec![],
            rectangles: vec![],
            triangles: vec![],
            polygons: vec![],
            dirty: true,
            gpu: LayerGpu::default(),
            opaque: true,
        });

        self.layers.push(RuntimeLayer {
            name: "editor_handles".into(),
            order: 950,
            active: false,
            cache: LayerCache::default(),
            texts: vec![],
            circles: vec![],
            circle_outlines: vec![],
            rectangles: vec![],
            triangles: vec![],
            polygons: vec![],
            dirty: true,
            gpu: LayerGpu::default(),
            opaque: true,
        });
    }

    pub fn handle_touches(&mut self, mouse: &MouseState, dt: f32) {
        let mut trigger_selection = false;

        for layer in self.layers.iter().filter(|l| l.active) {
            // === Circles ===
            for c in &layer.circles {
                if !c.misc.active {
                    continue;
                }
                let dx = mouse.pos.x - c.x;
                let dy = mouse.pos.y - c.y;
                let dist = (dx * dx + dy * dy).sqrt();
                let touched = dist <= c.radius && mouse.left_pressed;

                if let Some(id) = &c.id {
                    if self.ui_runtime.update_touch(id, touched, dt, &layer.name) {
                        trigger_selection = true;
                    }
                }
            }

            // === Rectangles ===
            for r in &layer.rectangles {
                if !r.misc.active {
                    continue;
                }

                let verts = [
                    r.top_left_vertex.pos,
                    r.bottom_left_vertex.pos,
                    r.bottom_right_vertex.pos,
                    r.top_right_vertex.pos,
                ];
                let vertices_struct = [
                    r.top_left_vertex,
                    r.bottom_left_vertex,
                    r.bottom_right_vertex,
                    r.top_right_vertex,
                ];

                let mouse_pos = [mouse.pos.x, mouse.pos.y];
                let (min_x, min_y, max_x, max_y) = helper::get_aabb(&verts);
                if mouse_pos[0] < min_x
                    || mouse_pos[0] > max_x
                    || mouse_pos[1] < min_y
                    || mouse_pos[1] > max_y
                {
                    continue;
                }

                let mut inside = false;
                let is_rounded = helper::has_roundness(&vertices_struct);

                if !is_rounded && helper::is_axis_aligned_rect(&verts) {
                    inside = mouse_pos[0] >= min_x
                        && mouse_pos[0] <= max_x
                        && mouse_pos[1] >= min_y
                        && mouse_pos[1] <= max_y;
                } else if is_rounded {
                    let radius = vertices_struct
                        .iter()
                        .map(|v| v.roundness)
                        .fold(0.0, f32::max);
                    inside = helper::is_point_in_rounded_rect(mouse_pos, &verts, radius);
                } else {
                    inside = helper::is_point_in_quad(mouse_pos, &verts);
                }

                if let Some(id) = &r.id {
                    if self.ui_runtime.update_touch(
                        id,
                        inside && mouse.left_pressed,
                        dt,
                        &layer.name,
                    ) {
                        trigger_selection = true;
                    }
                }
            }
        }

        if trigger_selection {
            self.update_selection();
        }
    }

    pub(crate) fn hash_id(id: &str) -> f32 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        id.hash(&mut hasher);
        let hash_u64 = hasher.finish(); // 64-bit
        // map to [0, 1]
        (hash_u64 as f64 / u64::MAX as f64) as f32
    }

    pub fn update_selection(&mut self) {
        if !self.ui_runtime.selected_ui_element.active {
            return;
        }

        let sel = self.ui_runtime.selected_ui_element.clone();

        if let Some(element) = self.find_element(&sel.layer_name, &sel.element_id) {
            if let Some(editor_layer) = self
                .layers
                .iter_mut()
                .find(|l| l.name == "editor_selection")
            {
                editor_layer.active = true;
                editor_layer.dirty = true;
                editor_layer.circles.clear();
                editor_layer.rectangles.clear();
                editor_layer.triangles.clear();
                editor_layer.polygons.clear();
                println!("here");
                match element {
                    UiElement::Circle(c) => {
                        let circle_outline = UiButtonCircleOutline {
                            id: Some("Circle Outline".to_string()),
                            x: c.x,
                            y: c.y,
                            stretch_x: c.stretch_x,
                            stretch_y: c.stretch_y,
                            radius: c.radius * 1.1,
                            dash_thickness: 8.0,
                            dash_color: [0.0, 0.0, 0.6, 0.8],
                            dash_misc: DashMisc {
                                dash_len: 3.0,
                                dash_spacing: 1.5,
                                dash_roundness: 1.0,
                                dash_speed: 2.0,
                            },
                            sub_dash_color: [0.8, 0.8, 0.8, 0.9],
                            sub_dash_misc: DashMisc {
                                dash_len: 3.0,
                                dash_spacing: 1.5,
                                dash_roundness: 1.0,
                                dash_speed: 2.0,
                            },
                            misc: c.misc,
                        };
                        editor_layer.circle_outlines.push(circle_outline);
                    }
                    UiElement::Rectangle(r) => {
                        //r. = [1.0, 0.8, 0.2, 0.7];
                        //editor_layer.rectangles.push(r);
                    }
                    UiElement::Triangle(t) => {
                        //t.misc.color = [1.0, 1.0, 0.2, 0.7];
                        //editor_layer.triangles.push(t);
                    }
                    UiElement::Polygon(p) => {
                        //p.misc.color = [1.0, 0.7, 0.1, 0.7];
                        //editor_layer.polygons.push(p);
                    }
                    UiElement::Text(mut tx) => {
                        //tx.color = [1.0, 1.0, 0.0, 1.0];
                        //tx.misc.active = true;
                        //editor_layer.texts.push(tx);
                    }
                }
            }
        }

        self.ui_runtime.selected_ui_element.active = false;
    }

    pub fn find_element(&self, layer_name: &str, element_id: &str) -> Option<UiElement> {
        let layer = self.layers.iter().find(|l| l.name == layer_name)?;

        // Circles
        for c in &layer.circles {
            if let Some(id) = &c.id {
                if id == element_id {
                    return Some(UiElement::Circle(c.clone()));
                }
            }
        }

        // Rectangles
        for r in &layer.rectangles {
            if let Some(id) = &r.id {
                if id == element_id {
                    return Some(UiElement::Rectangle(r.clone()));
                }
            }
        }

        // Triangles
        for t in &layer.triangles {
            if let Some(id) = &t.id {
                if id == element_id {
                    return Some(UiElement::Triangle(t.clone()));
                }
            }
        }

        // Polygons
        for p in &layer.polygons {
            if let Some(id) = &p.id {
                if id == element_id {
                    return Some(UiElement::Polygon(p.clone()));
                }
            }
        }

        // Texts
        for tx in &layer.texts {
            if let Some(id) = &tx.id {
                if id == element_id {
                    return Some(UiElement::Text(tx.clone()));
                }
            }
        }

        None
    }
}
