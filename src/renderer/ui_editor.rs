use crate::renderer::helper::{dist, order_vertices_ccw, polygon_sdf, triangulate_polygon};
use crate::renderer::ui::{CircleParams, HandleParams, OutlineParams, TextParams};
use crate::renderer::ui_editor::UiElement::Polygon;
use crate::resources::MouseState;
use crate::vertex::{
    DashMisc, GuiLayout, HandleMisc, LayerGpu, MiscButtonSettings, ShapeData, UiButtonCircle,
    UiButtonHandle, UiButtonOutline, UiButtonPolygon, UiButtonText, UiButtonVertexSelection,
    UiVertex, UiVertexPoly,
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
    pub outlines: Option<Vec<UiButtonOutline>>,
    pub handles: Option<Vec<UiButtonHandle>>,
    pub vertex_selections: Option<Vec<UiButtonVertexSelection>>,
    pub polygons: Option<Vec<UiButtonPolygon>>,
    pub active: Option<bool>,
    pub opaque: Option<bool>,
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
            outline_params: vec![],
            handle_params: vec![],
            polygon_vertices: vec![],
            outline_poly_vertices: vec![],
        }
    }
}

#[derive(Clone, Debug)]
pub enum UiElement {
    Circle(UiButtonCircle),
    Handle(UiButtonHandle),
    VertexSelection(UiButtonVertexSelection),
    Polygon(UiButtonPolygon),
    Text(UiButtonText),
}

#[derive(Debug)]
pub struct RuntimeLayer {
    pub name: String,
    pub order: u32,
    pub texts: Vec<UiButtonText>,
    pub circles: Vec<UiButtonCircle>,
    pub outlines: Vec<UiButtonOutline>,
    pub handles: Vec<UiButtonHandle>,
    pub vertex_selections: Vec<UiButtonVertexSelection>,
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
    pub active_vertex: Option<usize>,
    pub drag_offset: Option<(f32, f32)>,
    pub editor_mode: bool,
}

impl UiRuntime {
    pub fn new(editor_mode: bool) -> Self {
        Self {
            elements: HashMap::new(),
            selected_ui_element: SelectedUiElement::default(),
            active_vertex: None,
            drag_offset: None,
            editor_mode,
        }
    }

    pub fn update_touch(
        &mut self,
        id: &str,
        touched_now: bool,
        dt: f32,
        _layer_name: &String,
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

    pub fn update_editor_mode(&mut self, editor_mode: bool) {
        self.editor_mode = editor_mode;
    }

    pub fn get(&self, id: &str) -> ButtonRuntime {
        *self.elements.get(id).unwrap_or(&ButtonRuntime::default())
    }
}

impl UiButtonLoader {
    pub fn new(editor_mode: bool) -> Self {
        let layout = Self::load_gui_from_file("ui_data/gui_layout.json").unwrap_or_else(|e| {
            eprintln!("❌ Failed to load GUI layout: {e}");
            GuiLayout { layers: vec![] }
        });

        let mut loader = Self {
            layers: Vec::new(),
            ui_runtime: UiRuntime::new(editor_mode),
            id_lookup: HashMap::new(),
        };

        // JSON layers to runtime layers
        for l in layout.layers {
            let mut polys = l.polygons.unwrap_or_default();

            // ORDER THE REAL POLYGONS
            for p in &mut polys {
                order_vertices_ccw(&mut p.vertices);
            }
            loader.layers.push(RuntimeLayer {
                name: l.name,
                order: l.order,
                active: l.active.unwrap_or(true),
                cache: Default::default(),
                texts: l.texts.unwrap_or_default(),
                circles: l.circles.unwrap_or_default(),
                outlines: l.outlines.unwrap_or_default(),
                handles: l.handles.unwrap_or_default(),
                vertex_selections: l.vertex_selections.unwrap_or_default(),
                polygons: polys,
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
            outlines: vec![],
            handles: vec![],
            vertex_selections: vec![],
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
            outlines: vec![],
            handles: vec![],
            vertex_selections: vec![],
            polygons: vec![],
            dirty: true,
            gpu: LayerGpu::default(),
            opaque: true,
        });
    }

    pub fn handle_touches(&mut self, mouse: &MouseState, dt: f32) {
        let mut trigger_selection = false;
        let mut pending_circle_updates: Vec<(String, f32, f32)> = Vec::new();
        let mut moved_any_selected_object = false;

        // Cache mouse state
        let mx = mouse.pos.x;
        let my = mouse.pos.y;
        let pressed = mouse.left_pressed;
        let just_pressed = mouse.left_just_pressed;
        //let just_released = mouse.left_just_released;
        let scroll = mouse.scroll_delta.y;

        // Current mode
        let editor_mode = self.ui_runtime.editor_mode;

        // ============================================================
        // 1) PRE-PASS: detect clicks (shared between modes)
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

                // polygons
                for poly in &layer.polygons {
                    if !poly.misc.active {
                        continue;
                    }

                    let verts = &poly.vertices;

                    let sdf = polygon_sdf(mx, my, verts);

                    let inside = sdf < 0.0;
                    let near_edge = sdf.abs() < 8.0;
                    let hit = inside || near_edge;

                    if hit {
                        press_began_on_ui = true;
                        break 'outer_scan;
                    }
                }

                // handles       -
                for h in &layer.handles {
                    if !h.misc.active || !editor_mode {
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
                    if inside_band {
                        press_began_on_ui = true;
                        break 'outer_scan;
                    }
                }
            }
        }

        if just_pressed && !press_began_on_ui && editor_mode {
            // Prevent accidental deselection if near a handle
            let mut near_handle = false;
            for layer in self.layers.iter().filter(|l| l.active) {
                for h in &layer.handles {
                    if !h.misc.active {
                        continue;
                    }
                    let dx = mx - h.x;
                    let dy = my - h.y;
                    let dist = (dx * dx + dy * dy).sqrt();
                    let margin = (h.radius * 0.2).max(12.0);
                    if (dist - h.radius).abs() < margin {
                        near_handle = true;
                        break;
                    }
                }
                if near_handle {
                    break;
                }
            }

            if !near_handle {
                self.ui_runtime.selected_ui_element.active = false;
                self.update_selection();
            }
        }

        // ============================================================
        // 2) EDITOR-ONLY interactions (drag, resize, etc.)
        // ============================================================
        if editor_mode {
            for layer_index in 0..self.layers.len() {
                let (_before, layer_rest) = self.layers.split_at_mut(layer_index);
                let (layer, _after) = layer_rest.split_first_mut().unwrap();

                if !layer.active {
                    continue;
                }

                // --- circles (draggable, identical behavior restored) ---
                for c in &mut layer.circles {
                    if !c.misc.active {
                        continue;
                    }

                    let dx = mx - c.x;
                    let dy = my - c.y;

                    let drag_radius = (c.radius * 0.8).max(8.0);
                    let inside = dx * dx + dy * dy <= drag_radius * drag_radius;

                    if let Some(id) = &c.id {
                        let runtime = self.ui_runtime.get(id);
                        let touched_now = if !runtime.is_down {
                            inside && just_pressed
                        } else {
                            pressed
                        };

                        let state = self
                            .ui_runtime
                            .update_touch(id, touched_now, dt, &layer.name);

                        match state {
                            TouchState::Pressed => {
                                // Store grab offset so it doesn’t snap to mouse center
                                self.ui_runtime.drag_offset = Some((mx - c.x, my - c.y));
                            }
                            TouchState::Held => {
                                let (ox, oy) = self.ui_runtime.drag_offset.unwrap_or((0.0, 0.0));
                                let new_x = mx - ox;
                                let new_y = my - oy;

                                if (new_x - c.x).abs() > 0.001 || (new_y - c.y).abs() > 0.001 {
                                    c.x = new_x;
                                    c.y = new_y;
                                    layer.dirty = true;
                                    moved_any_selected_object = true;
                                }
                            }
                            TouchState::Released => {
                                self.ui_runtime.drag_offset = None;
                                trigger_selection = true;
                                self.ui_runtime.selected_ui_element = SelectedUiElement {
                                    layer_name: layer.name.clone(),
                                    element_id: id.clone(),
                                    active: true,
                                };
                            }
                            TouchState::Idle => {}
                        }
                    }
                }

                // --- handles (resize, original sticky logic restored) ---
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
                    // Expand the grab zone by a constant margin (in pixels)
                    let margin = (h.radius * 0.15).max(10.0); // 15% of radius, at least 10 px
                    let inner_grab = (inner - margin).max(0.0);
                    let outer_grab = outer + margin;
                    let inside_band =
                        dist2 >= inner_grab * inner_grab && dist2 <= outer_grab * outer_grab;

                    let runtime = self.ui_runtime.get(h.id.as_ref().unwrap());
                    let touched_now = if !runtime.is_down {
                        inside_band && just_pressed
                    } else {
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

                // POLYGON selection --------------------------------
                for poly in &mut layer.polygons {
                    if !poly.misc.active {
                        continue;
                    }

                    let verts = &mut poly.vertices;
                    if verts.is_empty() {
                        continue;
                    }

                    // --- stable centroid for anchoring ---
                    let mut cx = 0.0f32;
                    let mut cy = 0.0f32;
                    for v in verts.iter() {
                        cx += v.pos[0];
                        cy += v.pos[1];
                    }
                    let inv_n = 1.0 / verts.len() as f32;
                    cx *= inv_n;
                    cy *= inv_n;

                    // polygon hit
                    let sdf = polygon_sdf(mx, my, verts);
                    let inside = sdf < 0.0;
                    let near_edge = sdf.abs() < 8.0;
                    let poly_hit = inside || near_edge;

                    // vertex hit
                    const VERTEX_RADIUS: f32 = 8.0;
                    let vertex_hit = verts
                        .iter()
                        .enumerate()
                        .find(|(_, v)| dist(mx, my, v.pos[0], v.pos[1]) < VERTEX_RADIUS)
                        .map(|(i, _)| i);

                    // final hit priority:
                    // 1. vertex hit > 2. polygon hit
                    let hit = match vertex_hit {
                        Some(_) => true,
                        None => poly_hit,
                    };

                    if let Some(id) = &poly.id {
                        let runtime = self.ui_runtime.get(id);

                        let touched_now = if !runtime.is_down {
                            hit && just_pressed
                        } else {
                            pressed // continue drag
                        };

                        match self
                            .ui_runtime
                            .update_touch(id, touched_now, dt, &layer.name)
                        {
                            TouchState::Pressed => {
                                if let Some(vidx) = vertex_hit {
                                    // start a vertex drag
                                    self.ui_runtime.drag_offset =
                                        Some((mx - verts[vidx].pos[0], my - verts[vidx].pos[1]));
                                    self.ui_runtime.active_vertex = Some(vidx); // <-- NEW
                                } else {
                                    // start a centroid drag
                                    self.ui_runtime.drag_offset = Some((mx - cx, my - cy));
                                    self.ui_runtime.active_vertex = None;
                                }
                            }

                            TouchState::Held => {
                                if let Some(vidx) = self.ui_runtime.active_vertex {
                                    // === drag vertex only ===
                                    let (ox, oy) = self.ui_runtime.drag_offset.unwrap();
                                    let new_x = mx - ox;
                                    let new_y = my - oy;
                                    verts[vidx].pos = [new_x, new_y];
                                    layer.dirty = true;
                                    moved_any_selected_object = true;
                                } else {
                                    // === centroid drag ===
                                    let (ox, oy) = self.ui_runtime.drag_offset.unwrap();

                                    // recompute centroid
                                    let mut ccx = 0.0;
                                    let mut ccy = 0.0;
                                    for v in verts.iter() {
                                        ccx += v.pos[0];
                                        ccy += v.pos[1];
                                    }
                                    let inv_n = 1.0 / verts.len() as f32;
                                    ccx *= inv_n;
                                    ccy *= inv_n;

                                    let new_cx = mx - ox;
                                    let new_cy = my - oy;

                                    let dx = new_cx - ccx;
                                    let dy = new_cy - ccy;

                                    if dx.abs() > 0.001 || dy.abs() > 0.001 {
                                        for v in verts.iter_mut() {
                                            v.pos[0] += dx;
                                            v.pos[1] += dy;
                                        }
                                        layer.dirty = true;
                                        moved_any_selected_object = true;
                                    }
                                }
                            }

                            TouchState::Released => {
                                self.ui_runtime.drag_offset = None;
                                self.ui_runtime.active_vertex = None;

                                trigger_selection = true;
                                self.ui_runtime.selected_ui_element = SelectedUiElement {
                                    layer_name: layer.name.clone(),
                                    element_id: id.clone(),
                                    active: true,
                                };
                            }

                            TouchState::Idle => {}
                        }
                    }
                }
            }

            // ============================================================
            // Apply deferred handle-driven radius updates
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
                    let smoothing_speed = 10.0;
                    let dt_effective = dt.clamp(1.0 / 240.0, 0.1);
                    let k = 1.0 - (-smoothing_speed * dt_effective).exp();
                    let new_radius = (current_radius + (target_radius - current_radius) * k)
                        .abs()
                        .max(2.0);

                    for layer in &mut self.layers {
                        for c in &mut layer.circles {
                            if let Some(id) = &c.id {
                                if id == &parent_id {
                                    c.radius = new_radius;
                                }
                            }
                        }
                    }

                    for layer in &mut self.layers {
                        for h in &mut layer.handles {
                            if let Some(pid) = &h.parent_id {
                                if pid == &parent_id {
                                    h.radius = new_radius;
                                    layer.dirty = true;
                                }
                            }
                        }
                        for o in &mut layer.outlines {
                            if let Some(pid) = &o.parent_id {
                                if pid == &parent_id {
                                    o.shape_data.radius = new_radius;
                                    layer.dirty = true;
                                }
                            }
                        }
                    }

                    self.mark_layer_dirty(&layer_name);
                }
            }

            if moved_any_selected_object {
                for layer in &mut self.layers {
                    if layer.name == "editor_selection" || layer.name == "editor_handles" {
                        layer.dirty = true;
                    }
                }
                trigger_selection = true;
            }

            // Scroll-wheel resize (editor only)
            if self.ui_runtime.selected_ui_element.active && scroll.abs() > 0.0 {
                for layer in &mut self.layers {
                    for c in &mut layer.circles {
                        if let Some(id) = &c.id {
                            if *id == self.ui_runtime.selected_ui_element.element_id {
                                c.radius = (c.radius + scroll * 3.0).max(2.0);
                                layer.dirty = true;
                                trigger_selection = true;
                            }
                        }
                    }
                }
            }
        } else if self.ui_runtime.selected_ui_element.active {
            self.ui_runtime.selected_ui_element.active = false;
            self.update_selection();
        }
        // ============================================================
        // 3) Selection / deselection
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
                editor_layer.outlines.clear();
                editor_layer.handles.clear();
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
                editor_layer.outlines.clear();
                editor_layer.handles.clear();
                editor_layer.polygons.clear();
                match element {
                    UiElement::Circle(c) => {
                        let circle_outline = UiButtonOutline {
                            id: Some("Circle Outline".to_string()),
                            parent_id: c.id.clone(),
                            mode: 0.0,
                            vertex_offset: 0,
                            vertex_count: 0,
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
                            shape_data: ShapeData {
                                x: c.x,
                                y: c.y,
                                radius: c.radius,
                                border_thickness: c.border_thickness,
                            },
                        };
                        println!("{:?}", circle_outline);
                        editor_layer.outlines.push(circle_outline);

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
                                pressable: true,
                                editable: false,
                            },
                        };
                        editor_layer.handles.push(handle);
                    }
                    UiElement::Handle(_h) => {}
                    UiElement::VertexSelection(_) => {}
                    Polygon(p) => {
                        let mut cx = 0.0;
                        let mut cy = 0.0;
                        for v in &p.vertices {
                            cx += v.pos[0];
                            cy += v.pos[1];
                        }
                        cx /= p.vertices.len() as f32;
                        cy /= p.vertices.len() as f32;

                        let mut radius: f32 = 0.0;
                        for v in &p.vertices {
                            let dx = v.pos[0] - cx;
                            let dy = v.pos[1] - cy;
                            radius = radius.max((dx * dx + dy * dy).sqrt());
                        }
                        let polygon_outline = UiButtonOutline {
                            id: Some("Polygon Outline".to_string()),
                            parent_id: p.id.clone(),
                            mode: 1.0,
                            vertex_offset: 0,
                            vertex_count: p.vertices.len() as u32,
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
                            misc: p.misc,
                            shape_data: ShapeData {
                                x: cx,
                                y: cy,
                                radius,
                                border_thickness: 3.0,
                            },
                        };
                        editor_layer.outlines.push(polygon_outline);
                    }
                    UiElement::Text(_tx) => {
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

    pub fn rebuild_layer_cache_index(&mut self, layer_index: usize) {
        let runtime = &self.ui_runtime;
        // Split so we can have &mut to this layer and & to all others.
        let (before, rest) = self.layers.split_at_mut(layer_index);
        let (layer, after) = rest.split_first_mut().unwrap();
        let l = layer;

        l.cache = LayerCache::default();

        // ------- TEXTS -------
        for t in &l.texts {
            let id_str = t.id.as_deref().unwrap_or("");
            let rt = runtime.get(id_str);

            let hash = if id_str.is_empty() {
                f32::MAX
            } else {
                UiButtonLoader::hash_id(id_str)
            };

            l.cache.texts.push(TextParams {
                pos: [t.x, t.y],
                px: t.px,
                color: t.color,
                id_hash: hash,
                misc: [
                    f32::from(t.misc.active),
                    rt.touched_time,
                    f32::from(rt.is_down),
                    hash,
                ],
                text: t.text.clone(),
            });
        }

        // ------- CIRCLES -------
        for c in &l.circles {
            let id_str = c.id.as_deref().unwrap_or("");
            let rt = runtime.get(id_str);

            let hash = if id_str.is_empty() {
                f32::MAX
            } else {
                UiButtonLoader::hash_id(id_str)
            };

            l.cache.circle_params.push(CircleParams {
                center_radius_border: [c.x, c.y, c.radius, c.border_thickness],
                fill_color: c.fill_color,
                border_color: c.border_color,
                glow_color: c.glow_color,
                glow_misc: [
                    c.glow_misc.glow_size,
                    c.glow_misc.glow_speed,
                    c.glow_misc.glow_intensity,
                    1.0,
                ],
                misc: [
                    f32::from(c.misc.active),
                    rt.touched_time,
                    f32::from(rt.is_down),
                    hash,
                ],
            });
        }

        // Local helper that searches ALL OTHER layers (before + after) for a polygon id
        fn find_polygon_by_id<'a>(
            id: &Option<String>,
            before: &'a [RuntimeLayer],
            after: &'a [RuntimeLayer],
        ) -> Option<&'a UiButtonPolygon> {
            let target = match id {
                Some(s) => s,
                None => return None,
            };

            for layer in before.iter().chain(after.iter()) {
                for p in &layer.polygons {
                    if let Some(pid) = &p.id {
                        if pid == target {
                            return Some(p);
                        }
                    }
                }
            }
            None
        }

        // ------- OUTLINES -------
        for o in &mut l.outlines {
            if o.mode == 1.0 {
                if let Some(poly) = find_polygon_by_id(&o.parent_id, before, after) {
                    o.vertex_offset = l.cache.outline_poly_vertices.len() as u32;

                    for v in &poly.vertices {
                        l.cache.outline_poly_vertices.push([v.pos[0], v.pos[1]]);
                    }

                    o.vertex_count = poly.vertices.len() as u32;
                }
            }

            let id_str = o.id.as_deref().unwrap_or("");
            let rt = runtime.get(id_str);

            let hash = if id_str.is_empty() {
                f32::MAX
            } else {
                UiButtonLoader::hash_id(id_str)
            };

            l.cache.outline_params.push(OutlineParams {
                mode: o.mode,
                vertex_offset: o.vertex_offset,
                vertex_count: o.vertex_count,
                _pad0: 0, // u32 padding, must be integer
                shape_data: [
                    o.shape_data.x,
                    o.shape_data.y,
                    o.shape_data.radius,
                    o.shape_data.border_thickness,
                ],
                dash_color: o.dash_color,
                dash_misc: [
                    o.dash_misc.dash_len,
                    o.dash_misc.dash_spacing,
                    o.dash_misc.dash_roundness,
                    o.dash_misc.dash_speed,
                ],
                sub_dash_color: o.sub_dash_color,
                sub_dash_misc: [
                    o.sub_dash_misc.dash_len,
                    o.sub_dash_misc.dash_spacing,
                    o.sub_dash_misc.dash_roundness,
                    o.sub_dash_misc.dash_speed,
                ],
                misc: [
                    f32::from(o.misc.active),
                    rt.touched_time,
                    f32::from(rt.is_down),
                    hash,
                ],
            });
        }

        // ------- HANDLES -------
        for h in &l.handles {
            let id_str = h.id.as_deref().unwrap_or("");
            let rt = runtime.get(id_str);

            let hash = if id_str.is_empty() {
                f32::MAX
            } else {
                UiButtonLoader::hash_id(id_str)
            };

            l.cache.handle_params.push(HandleParams {
                center_radius_mode: [h.x, h.y, h.radius, 1.0],
                handle_color: h.handle_color,
                handle_misc: [
                    h.handle_misc.handle_len,
                    h.handle_misc.handle_width,
                    h.handle_misc.handle_roundness,
                    h.handle_misc.handle_speed,
                ],
                sub_handle_color: h.sub_handle_color,
                sub_handle_misc: [
                    h.sub_handle_misc.handle_len,
                    h.sub_handle_misc.handle_width,
                    h.sub_handle_misc.handle_roundness,
                    h.sub_handle_misc.handle_speed,
                ],
                misc: [
                    f32::from(h.misc.active),
                    rt.touched_time,
                    f32::from(rt.is_down),
                    hash,
                ],
            });
        }

        // Common builder for vertex-emitting shapes
        let push_with_misc = |v: &UiVertex, misc: [f32; 4], out: &mut Vec<UiVertexPoly>| {
            out.push(UiVertexPoly {
                pos: v.pos,
                _pad: [1.0; 2],
                color: v.color,
                misc,
            });
        };

        // ------- POLYGONS (N verts) -------
        for poly in &l.polygons {
            let id_str = poly.id.as_deref().unwrap_or("");
            let rt = runtime.get(id_str);

            let hash = if id_str.is_empty() {
                f32::MAX
            } else {
                UiButtonLoader::hash_id(id_str)
            };

            let misc = [
                f32::from(poly.misc.active),
                rt.touched_time,
                f32::from(rt.is_down),
                hash,
            ];

            let tris = triangulate_polygon(&poly.vertices);

            for [i0, i1, i2] in tris {
                let v0 = &poly.vertices[i0];
                let v1 = &poly.vertices[i1];
                let v2 = &poly.vertices[i2];

                push_with_misc(v0, misc, &mut l.cache.polygon_vertices);
                push_with_misc(v1, misc, &mut l.cache.polygon_vertices);
                push_with_misc(v2, misc, &mut l.cache.polygon_vertices);
            }
        }

        l.dirty = false;
    }
}
