use crate::renderer::helper::triangulate_polygon;
use crate::renderer::touches::{
    EditorInteractionResult, MouseSnapshot, apply_pending_circle_updates, find_top_hit,
    handle_editor_mode_interactions, handle_scroll_resize, mark_editor_layers_dirty, near_handle,
    press_began_on_ui,
};
use crate::renderer::ui::{CircleParams, HandleParams, OutlineParams, TextParams};
use crate::resources::{InputState, MouseState};
use crate::vertex::UiElement::*;
pub(crate) use crate::vertex::*;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use winit::keyboard::{KeyCode, PhysicalKey};

pub struct UiButtonLoader {
    pub layers: Vec<RuntimeLayer>,
    pub ui_runtime: UiRuntime,
    pub id_lookup: HashMap<u32, String>,
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

        // JSON layers to runtime layers...…
        let mut id_gen: usize = 1;

        for l in layout.layers {
            let texts = l
                .texts
                .unwrap_or_default()
                .into_iter()
                .map(UiButtonText::from_json)
                .collect();

            let circles = l
                .circles
                .unwrap_or_default()
                .into_iter()
                .map(UiButtonCircle::from_json)
                .collect();

            let outlines = l
                .outlines
                .unwrap_or_default()
                .into_iter()
                .map(UiButtonOutline::from_json)
                .collect();

            let handles = l
                .handles
                .unwrap_or_default()
                .into_iter()
                .map(UiButtonHandle::from_json)
                .collect();

            let polygons = l
                .polygons
                .unwrap_or_default()
                .into_iter()
                .map(|p| UiButtonPolygon::from_json(p, &mut id_gen))
                .collect();

            loader.layers.push(RuntimeLayer {
                name: l.name,
                order: l.order,
                active: l.active.unwrap_or(true),
                opaque: l.opaque.unwrap_or(false),
                texts,
                circles,
                outlines,
                handles,
                polygons,
                cache: LayerCache::default(),
                gpu: LayerGpu::default(),
                dirty: true,
            });
        }

        loader.add_editor_layers();
        loader.layers.sort_by_key(|l| l.order); // SORT!

        loader
    }

    pub fn load_gui_from_file(path: &str) -> Result<GuiLayout, Box<dyn std::error::Error>> {
        let mut full_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        full_path.push("src/renderer");
        full_path.push(path);

        let data = fs::read_to_string(&full_path)?;
        let parsed: GuiLayout = serde_json::from_str(&data)?;
        Ok(parsed)
    }

    pub fn save_gui_to_file(&self, path: &str) -> anyhow::Result<()> {
        use std::fs;
        use std::path::PathBuf;

        let mut full_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        full_path.push("src/renderer");
        full_path.push(path);

        // Convert runtime → serializable JSON
        let layout = self.to_gui_layout();

        let json = serde_json::to_string_pretty(&layout)?;
        fs::write(&full_path, json)?;

        println!("GUI saved to {}", full_path.display());
        Ok(())
    }

    fn to_gui_layout(&self) -> GuiLayout {
        let mut layers = Vec::new();

        for l in &self.layers {
            // Skip editor-only layers to avoid saving internal junk
            if l.name == "editor" || l.name == "editor_selection" || l.name == "editor_handles" {
                continue;
            }

            layers.push(UiLayerJson {
                name: l.name.clone(),
                order: l.order,
                active: Some(l.active),
                opaque: Some(l.opaque),

                texts: Some(l.texts.iter().map(|t| t.to_json()).collect()),
                circles: Some(l.circles.iter().map(|c| c.to_json()).collect()),
                outlines: Some(l.outlines.iter().map(|o| o.to_json()).collect()),
                handles: Some(l.handles.iter().map(|h| h.to_json()).collect()),
                polygons: Some(l.polygons.iter().map(|p| p.to_json()).collect()),
            });
        }

        GuiLayout { layers }
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
            polygons: vec![],
            dirty: true,
            gpu: LayerGpu::default(),
            opaque: true,
        });
    }

    pub fn handle_touches(&mut self, mouse: &MouseState, dt: f32, input_state: &InputState) {
        let mut trigger_selection = false;
        let mouse_snapshot = MouseSnapshot::from_mouse(mouse);
        let editor_mode = self.ui_runtime.editor_mode;

        let press_started_on_ui = if mouse_snapshot.just_pressed {
            press_began_on_ui(&self.layers, &mouse_snapshot, editor_mode)
        } else {
            false
        };

        if mouse_snapshot.just_pressed && !press_started_on_ui && editor_mode {
            if !near_handle(&self.layers, &mouse_snapshot) {
                self.ui_runtime.selected_ui_element.active = false;
                self.update_selection(mouse);
            }
        }

        if editor_mode {
            let top_hit = find_top_hit(&self.layers, &mouse_snapshot, editor_mode);
            let EditorInteractionResult {
                trigger_selection: mut selection,
                pending_circle_updates,
                moved_any_selected_object,
            } = handle_editor_mode_interactions(self, dt, &mouse_snapshot, top_hit);

            apply_pending_circle_updates(self, dt, pending_circle_updates);

            if moved_any_selected_object {
                mark_editor_layers_dirty(&mut self.layers);
                selection = true;
            }

            if handle_scroll_resize(self, mouse_snapshot.scroll) {
                selection = true;
            }

            trigger_selection = selection;
        } else if self.ui_runtime.selected_ui_element.active {
            self.ui_runtime.selected_ui_element.active = false;
            self.update_selection(mouse);
        }

        if trigger_selection {
            self.update_selection(mouse);
        }

        if input_state.pressed_physical(&PhysicalKey::Code(KeyCode::KeyX))
            && self.ui_runtime.selected_ui_element.active
        {
            println!("deleting");
            let element_id = self.ui_runtime.selected_ui_element.element_id.clone();
            let layer_name = self.ui_runtime.selected_ui_element.layer_name.clone();

            self.ui_runtime.selected_ui_element.active = false;

            let _ = self.delete_element(&layer_name, &element_id);
        }
    }

    pub fn add_element(
        &mut self,
        layer_name: &str,
        mut element: UiElement,
        mouse: &MouseState,
    ) -> Result<String, String> {
        let layer = self
            .layers
            .iter_mut()
            .find(|l| l.name == layer_name)
            .ok_or_else(|| format!("Layer '{layer_name}' not found"))?;

        // === apply mouse positioning ===
        match &mut element {
            UiElement::Circle(c) => {
                c.x = mouse.pos.x;
                c.y = mouse.pos.y;
            }
            UiElement::Handle(h) => {
                h.x = mouse.pos.x;
                h.y = mouse.pos.y;
            }
            UiElement::Text(t) => {
                t.x = mouse.pos.x;
                t.y = mouse.pos.y;
            }
            UiElement::Polygon(p) => {
                if !p.vertices.is_empty() {
                    // compute centroid!
                    let mut cx = 0.0;
                    let mut cy = 0.0;
                    for v in &p.vertices {
                        cx += v.pos[0];
                        cy += v.pos[1];
                    }
                    cx /= p.vertices.len() as f32;
                    cy /= p.vertices.len() as f32;

                    let dx = mouse.pos.x - cx;
                    let dy = mouse.pos.y - cy;

                    // translate all vertices
                    for v in &mut p.vertices {
                        v.pos[0] += dx;
                        v.pos[1] += dy;
                    }
                }
            }
        };

        // === insert into layer ===
        let id = match &mut element {
            UiElement::Circle(c) => {
                if c.id.is_none() {
                    c.id = Some(gen_id());
                }
                let id = c.id.clone().unwrap();
                layer.circles.push(c.clone());
                id
            }
            UiElement::Handle(h) => {
                if h.id.is_none() {
                    h.id = Some(gen_id());
                }
                let id = h.id.clone().unwrap();
                layer.handles.push(h.clone());
                id
            }
            UiElement::Polygon(p) => {
                if p.id.is_none() {
                    p.id = Some(gen_id());
                }
                let id = p.id.clone().unwrap();
                layer.polygons.push(p.clone());
                id
            }
            UiElement::Text(t) => {
                if t.id.is_none() {
                    t.id = Some(gen_id());
                }
                let id = t.id.clone().unwrap();
                layer.texts.push(t.clone());
                id
            }
        };

        layer.dirty = true;
        Ok(id)
    }

    pub(crate) fn mark_layer_dirty(&mut self, layer_name: &str) {
        if let Some(layer) = self.layers.iter_mut().find(|l| l.name == layer_name) {
            layer.dirty = true;
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

    pub fn update_selection(&mut self, _mouse: &MouseState) {
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
                    Circle(c) => {
                        let circle_outline = UiButtonOutline {
                            id: Some("Circle Outline".to_string()),
                            z_index: c.z_index,
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
                            z_index: c.z_index,
                        };
                        editor_layer.handles.push(handle);
                    }
                    Handle(_h) => {}
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
                        for (i, v) in p.vertices.iter().enumerate() {
                            let dx = v.pos[0] - cx;
                            let dy = v.pos[1] - cy;
                            let distance = (dx * dx + dy * dy).sqrt();

                            radius = radius.max(distance);
                            let vertex_outline = UiButtonCircle {
                                id: Some(format!("vertex_outline_{}", i)),
                                z_index: i as i32,
                                x: v.pos[0],
                                y: v.pos[1],
                                stretch_x: 0.0,
                                stretch_y: 0.0,
                                radius: 15.0,
                                border_thickness: 0.0,
                                fade: [1.0, 0.0, 0.0, 0.0],
                                fill_color: [0.0, 1.0, 0.0, 1.0],
                                border_color: [0.5, 0.0, 0.0, 1.0],
                                glow_color: [0.0, 0.0, 0.5, 0.0],
                                glow_misc: GlowMisc {
                                    glow_size: 0.0,
                                    glow_speed: 0.0,
                                    glow_intensity: 0.0,
                                },
                                misc: MiscButtonSettings {
                                    active: false,
                                    touched_time: 0.0,
                                    is_touched: false,
                                    pressable: false,
                                    editable: false,
                                },
                            };
                            editor_layer.circles.push(vertex_outline);
                        }
                        let polygon_outline = UiButtonOutline {
                            id: Some("Polygon Outline".to_string()),
                            z_index: p.z_index,
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
                                border_thickness: 2.0,
                            },
                        };
                        editor_layer.outlines.push(polygon_outline);
                    }
                    Text(_tx) => {
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

    pub fn delete_element(&mut self, layer_name: &str, element_id: &str) -> Option<UiElement> {
        // get mutable layer
        let layer = self.layers.iter_mut().find(|l| l.name == layer_name)?;

        // Circles
        if let Some(pos) = layer
            .circles
            .iter()
            .position(|c| c.id.as_deref() == Some(element_id))
        {
            let removed = layer.circles.remove(pos);
            return Some(UiElement::Circle(removed));
        }

        // Polygons
        if let Some(pos) = layer
            .polygons
            .iter()
            .position(|p| p.id.as_deref() == Some(element_id))
        {
            let removed = layer.polygons.remove(pos);
            return Some(UiElement::Polygon(removed));
        }

        // Texts
        if let Some(pos) = layer
            .texts
            .iter()
            .position(|t| t.id.as_deref() == Some(element_id))
        {
            let removed = layer.texts.remove(pos);
            return Some(UiElement::Text(removed));
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
                fade: c.fade,
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
        for poly in &mut l.polygons {
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

            let tris = triangulate_polygon(&mut poly.vertices);
            poly.tri_count = tris.len() as u32 / 3;

            for vertex in &tris {
                push_with_misc(vertex, misc, &mut l.cache.polygon_vertices);
            }
        }

        l.dirty = false;
    }
}
