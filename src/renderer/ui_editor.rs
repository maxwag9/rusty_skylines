use crate::renderer::helper;
use crate::renderer::ui::{CircleOutlineParams, CircleParams, HandleParams, TextParams};
use crate::resources::MouseState;
use crate::vertex::{
    DashMisc, GuiLayout, HandleMisc, LayerGpu, MiscButtonSettings, UiButtonCircle,
    UiButtonCircleOutline, UiButtonHandle, UiButtonPolygon, UiButtonRectangle, UiButtonText,
    UiButtonTriangle, UiVertexPoly,
};
use std::collections::HashMap;
use std::fs;
use std::io::Result;
use std::path::PathBuf;

pub enum TouchState {
    Pressed,
    Held,
    Released,
    Idle,
}

#[derive(Debug, serde::Deserialize)]
pub struct UiLayer {
    pub name: String,
    pub order: u32,
    pub texts: Option<Vec<UiButtonText>>,
    pub circles: Option<Vec<UiButtonCircle>>,
    pub circle_outlines: Option<Vec<UiButtonCircleOutline>>,
    pub handles: Option<Vec<UiButtonHandle>>,
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
    pub handle_params: Vec<HandleParams>,
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
            handle_params: vec![],
            rect_vertices: vec![],
            triangle_vertices: vec![],
            polygon_vertices: vec![],
        }
    }
}

#[derive(Clone, Debug)]
pub enum UiElement {
    Circle(UiButtonCircle),
    Handle(UiButtonHandle),
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
    pub handles: Vec<UiButtonHandle>,
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
    ) -> TouchState {
        let entry = self
            .elements
            .entry(id.to_string())
            .or_insert_with(ButtonRuntime::default);

        entry.just_pressed = false;
        entry.just_released = false;

        match (entry.is_down, touched_now) {
            (false, true) => {
                entry.is_down = true;
                entry.just_pressed = true;
                entry.touched_time = 0.0;
                TouchState::Pressed
            }
            (true, true) => {
                entry.touched_time += dt;
                TouchState::Held
            }
            (true, false) => {
                entry.is_down = false;
                entry.just_released = true;
                TouchState::Released
            }
            (false, false) => TouchState::Idle,
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
                handles: l.handles.unwrap_or_default(),
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
            handles: vec![],
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
            handles: vec![],
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
        let mut pending_circle_updates: Vec<(String, f32, f32)> = Vec::new();

        // Cache mouse state
        let mx = mouse.pos.x;
        let my = mouse.pos.y;
        let pressed = mouse.left_pressed;
        let just_pressed = mouse.left_just_pressed;
        let just_released = mouse.left_just_released;

        // ============================================================
        // 1) PRE-PASS: did THIS click start on any UI element?
        // ============================================================
        let mut press_began_on_ui = false;
        if just_pressed {
            'outer_scan: for layer in self.layers.iter().filter(|l| l.active) {
                // circles
                for c in &layer.circles {
                    if !c.misc.active {
                        continue;
                    }
                    let dx = mx - c.x;
                    let dy = my - c.y;
                    if dx * dx + dy * dy <= c.radius * c.radius {
                        press_began_on_ui = true;
                        break 'outer_scan;
                    }
                }
                // rectangles
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
                    let mouse_pos = [mx, my];
                    let (min_x, min_y, max_x, max_y) = helper::get_aabb(&verts);
                    if mouse_pos[0] < min_x
                        || mouse_pos[0] > max_x
                        || mouse_pos[1] < min_y
                        || mouse_pos[1] > max_y
                    {
                        // not in AABB, skip
                    } else {
                        let is_rounded = helper::has_roundness(&vertices_struct);
                        let inside = if !is_rounded && helper::is_axis_aligned_rect(&verts) {
                            mouse_pos[0] >= min_x
                                && mouse_pos[0] <= max_x
                                && mouse_pos[1] >= min_y
                                && mouse_pos[1] <= max_y
                        } else if is_rounded {
                            let radius = vertices_struct
                                .iter()
                                .map(|v| v.roundness)
                                .fold(0.0, f32::max);
                            helper::is_point_in_rounded_rect(mouse_pos, &verts, radius)
                        } else {
                            helper::is_point_in_quad(mouse_pos, &verts)
                        };
                        if inside {
                            press_began_on_ui = true;
                            break 'outer_scan;
                        }
                    }
                }
                // handles: start only if inside the visible band
                for h in &layer.handles {
                    if !h.misc.active {
                        continue;
                    }
                    let dx = mx - h.x;
                    let dy = my - h.y;
                    let dist2 = dx * dx + dy * dy;

                    let width_ratio = h.handle_misc.handle_width; // fraction of radius
                    let half_thick = 0.5 * h.radius * width_ratio;
                    let inner = h.radius - half_thick;
                    let outer = h.radius + half_thick;
                    let inner2 = inner * inner;
                    let outer2 = outer * outer;

                    let inside_band = dist2 >= inner2 && dist2 <= outer2;
                    if inside_band {
                        press_began_on_ui = true;
                        break 'outer_scan;
                    }
                }
            }
        }
        // println!("2: {:?} {:?}", just_pressed, press_began_on_ui);
        if just_pressed && !press_began_on_ui {
            // Released after a click that began on empty space -> deselect
            self.ui_runtime.selected_ui_element.active = false;
            println!("2: {:?}", self.ui_runtime.selected_ui_element);
            self.update_selection();
        }
        // ============================================================
        // 2) MAIN PASS: update touches, selection and handle drags
        // ============================================================
        for layer_index in 0..self.layers.len() {
            let (_before, layer_rest) = self.layers.split_at_mut(layer_index);
            let (layer, _after) = layer_rest.split_first_mut().unwrap();

            if !layer.active {
                continue;
            }

            // --- circles ---
            for c in &layer.circles {
                if !c.misc.active {
                    continue;
                }
                let dx = mx - c.x;
                let dy = my - c.y;
                let touched_now = dx * dx + dy * dy <= c.radius * c.radius && just_pressed;

                if let Some(id) = &c.id {
                    match self
                        .ui_runtime
                        .update_touch(id, touched_now, dt, &layer.name)
                    {
                        TouchState::Released => {
                            trigger_selection = true;
                            self.ui_runtime.selected_ui_element = SelectedUiElement {
                                layer_name: layer.name.clone(),
                                element_id: id.clone(),
                                active: true,
                            };
                        }
                        _ => {}
                    }
                }
            }

            // --- rectangles ---
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

                let mouse_pos = [mx, my];
                let (min_x, min_y, max_x, max_y) = helper::get_aabb(&verts);
                let mut inside = false;
                if !(mouse_pos[0] < min_x
                    || mouse_pos[0] > max_x
                    || mouse_pos[1] < min_y
                    || mouse_pos[1] > max_y)
                {
                    let is_rounded = helper::has_roundness(&vertices_struct);
                    inside = if !is_rounded && helper::is_axis_aligned_rect(&verts) {
                        mouse_pos[0] >= min_x
                            && mouse_pos[0] <= max_x
                            && mouse_pos[1] >= min_y
                            && mouse_pos[1] <= max_y
                    } else if is_rounded {
                        let radius = vertices_struct
                            .iter()
                            .map(|v| v.roundness)
                            .fold(0.0, f32::max);
                        helper::is_point_in_rounded_rect(mouse_pos, &verts, radius)
                    } else {
                        helper::is_point_in_quad(mouse_pos, &verts)
                    };
                }

                if let Some(id) = &r.id {
                    // Make rectangles behave like handles: start only on just_pressed,
                    // then stay active while held if this rectangle already had the grab.
                    let runtime = self.ui_runtime.get(id);
                    let touched_now = if !runtime.is_down {
                        inside && just_pressed
                    } else {
                        // continue only while the mouse is still down
                        pressed
                    };

                    if let TouchState::Released =
                        self.ui_runtime
                            .update_touch(id, touched_now, dt, &layer.name)
                    {
                        trigger_selection = true;
                        self.ui_runtime.selected_ui_element = SelectedUiElement {
                            layer_name: layer.name.clone(),
                            element_id: id.clone(),
                            active: true,
                        };
                    }
                }
            }

            // --- handles (sticky once grabbed; must start on band) ---
            for h in &mut layer.handles {
                if !h.misc.active {
                    continue;
                }

                let dx = mx - h.x;
                let dy = my - h.y;
                let dist2 = dx * dx + dy * dy;

                let width_ratio = h.handle_misc.handle_width;
                let half_thick = 0.5 * h.radius * width_ratio;
                let inner = h.radius - half_thick;
                let outer = h.radius + half_thick;
                let inner2 = inner * inner;
                let outer2 = outer * outer;

                let inside_band = dist2 >= inner2 && dist2 <= outer2;

                let runtime = self.ui_runtime.get(h.id.as_ref().unwrap());
                let touched_now = if !runtime.is_down {
                    // only start grab when the mouse was just pressed over the handle
                    inside_band && just_pressed
                } else {
                    // once grabbed, stay active as long as the button is held
                    pressed
                };

                if let Some(id) = &h.id {
                    let state = self
                        .ui_runtime
                        .update_touch(id, touched_now, dt, &layer.name);

                    if let Some(parent_id) = &h.parent_id {
                        match state {
                            TouchState::Held => {
                                pending_circle_updates.push((parent_id.clone(), mx, my));
                            }
                            TouchState::Released => {}
                            TouchState::Pressed => {}
                            TouchState::Idle => {}
                        }
                    }
                }
            }
        }

        // ============================================================
        // 3) Apply deferred handle-driven radius updates (smoothed)
        // ============================================================
        for (parent_id, mx, my) in pending_circle_updates {
            let mut target_radius = 0.0f32;
            let mut current_radius = 0.0f32;
            let mut layer_name: Option<String> = None;
            let mut cx = 0.0f32;
            let mut cy = 0.0f32;

            for layer in &mut self.layers {
                for c in &mut layer.circles {
                    if let Some(id) = &c.id {
                        if id == &parent_id {
                            current_radius = c.radius;
                            cx = c.x;
                            cy = c.y;
                            // absolute distance ensures we always get positive radius
                            target_radius = ((mx - cx).powi(2) + (my - cy).powi(2)).sqrt();
                            layer_name = Some(layer.name.clone());
                            break;
                        }
                    }
                }
                if layer_name.is_some() {
                    break;
                }
            }

            if let Some(layer_name) = layer_name {
                // --- smoothing ---
                let smoothing_speed = 10.0;
                let dt_effective = dt.clamp(1.0 / 240.0, 0.1);
                let k = 1.0 - (-smoothing_speed * dt_effective).exp();

                // Smooth radius, absolute, and clamp to a reasonable minimum
                let new_radius = (current_radius + (target_radius - current_radius) * k)
                    .abs()
                    .max(2.0); // avoid collapse at center

                // Update the parent circle
                for layer in &mut self.layers {
                    for c in &mut layer.circles {
                        if let Some(id) = &c.id {
                            if id == &parent_id {
                                c.radius = new_radius;
                            }
                        }
                    }
                }

                // Update handles + outlines
                for layer in &mut self.layers {
                    for h in &mut layer.handles {
                        if let Some(pid) = &h.parent_id {
                            if pid == &parent_id {
                                h.radius = new_radius;
                                layer.dirty = true;
                            }
                        }
                    }
                    for o in &mut layer.circle_outlines {
                        if let Some(pid) = &o.parent_id {
                            if pid == &parent_id {
                                o.radius = new_radius;
                                layer.dirty = true;
                            }
                        }
                    }
                }

                self.mark_layer_dirty(&layer_name);
            }
        }

        // ============================================================
        // 4) Selection / deselection
        // ============================================================
        if trigger_selection {
            self.update_selection();
        }
    }

    fn mark_layer_dirty(&mut self, layer_name: &str) {
        if let Some(layer) = self.layers.iter_mut().find(|l| l.name == layer_name) {
            layer.dirty = true;
        }
    }

    fn update_circle_radius_and_handle(
        &mut self,
        parent_id: &str,
        mx: f32,
        my: f32,
    ) -> Option<(String, f32, f32, f32)> {
        for layer in &mut self.layers {
            for c in &mut layer.circles {
                if let Some(id) = &c.id {
                    if id == parent_id {
                        let dx = mx - c.x;
                        let dy = my - c.y;
                        let new_radius = (dx * dx + dy * dy).sqrt().max(1.0);
                        c.radius = new_radius;
                        return Some((layer.name.clone(), new_radius, c.x, c.y));
                    }
                }
            }
        }
        None
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
            if let Some(editor_layer) = self
                .layers
                .iter_mut()
                .find(|l| l.name == "editor_selection")
            {
                editor_layer.active = false;
                editor_layer.circles.clear();
                editor_layer.circle_outlines.clear();
                editor_layer.handles.clear();
                editor_layer.rectangles.clear();
                editor_layer.triangles.clear();
                editor_layer.polygons.clear();
            }
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
                editor_layer.circle_outlines.clear();
                editor_layer.handles.clear();
                editor_layer.rectangles.clear();
                editor_layer.triangles.clear();
                editor_layer.polygons.clear();
                match element {
                    UiElement::Circle(c) => {
                        let circle_outline = UiButtonCircleOutline {
                            id: Some("Circle Outline".to_string()),
                            parent_id: c.id.clone(),
                            x: c.x,
                            y: c.y,
                            stretch_x: c.stretch_x,
                            stretch_y: c.stretch_y,
                            radius: c.radius,
                            dash_thickness: 6.0,
                            dash_color: [0.2, 0.0, 0.4, 0.8],
                            dash_misc: DashMisc {
                                dash_len: 4.0,
                                dash_spacing: 1.5,
                                dash_roundness: 1.0,
                                dash_speed: 4.0,
                            },
                            sub_dash_color: [0.8, 1.0, 1.0, 0.5],
                            sub_dash_misc: DashMisc {
                                dash_len: 1.0,
                                dash_spacing: 1.0,
                                dash_roundness: 0.0,
                                dash_speed: -2.0,
                            },
                            misc: c.misc,
                        };
                        editor_layer.circle_outlines.push(circle_outline);

                        let handle = UiButtonHandle {
                            id: Some("Circle Handle".to_string()),
                            parent_id: c.id,
                            x: c.x,
                            y: c.y,
                            radius: c.radius,
                            handle_thickness: 10.0,
                            handle_color: [0.65, 0.22, 0.05, 1.0],
                            handle_misc: HandleMisc {
                                handle_len: 0.09,
                                handle_width: 0.2,
                                handle_roundness: 0.3,
                                handle_speed: 0.0,
                            },
                            sub_handle_color: [0.0, 0.0, 0.0, 0.7],
                            sub_handle_misc: HandleMisc {
                                handle_len: 0.08,
                                handle_width: 0.05,
                                handle_roundness: 0.5,
                                handle_speed: 0.0,
                            },
                            misc: MiscButtonSettings {
                                active: true,
                                touched_time: 0.0,
                                is_touched: false,
                            },
                        };
                        editor_layer.handles.push(handle);
                    }
                    UiElement::Handle(h) => {}
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

        self.ui_runtime.selected_ui_element.active = true;
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

    pub fn find_element_index_by_hash(&self, id_hash: f32) -> Option<(String, usize)> {
        for layer in &self.layers {
            // Circles
            for (i, c) in layer.circles.iter().enumerate() {
                if let Some(id) = &c.id {
                    if (UiButtonLoader::hash_id(id) - id_hash).abs() < f32::EPSILON {
                        return Some((layer.name.clone(), i));
                    }
                }
            }

            // Rectangles
            for (i, r) in layer.rectangles.iter().enumerate() {
                if let Some(id) = &r.id {
                    if (UiButtonLoader::hash_id(id) - id_hash).abs() < f32::EPSILON {
                        return Some((layer.name.clone(), i));
                    }
                }
            }

            // Polygons
            for (i, p) in layer.polygons.iter().enumerate() {
                if let Some(id) = &p.id {
                    if (UiButtonLoader::hash_id(id) - id_hash).abs() < f32::EPSILON {
                        return Some((layer.name.clone(), i));
                    }
                }
            }

            // Texts
            for (i, t) in layer.texts.iter().enumerate() {
                if let Some(id) = &t.id {
                    if (UiButtonLoader::hash_id(id) - id_hash).abs() < f32::EPSILON {
                        return Some((layer.name.clone(), i));
                    }
                }
            }
        }

        None
    }
}
