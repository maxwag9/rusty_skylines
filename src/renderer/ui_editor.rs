use crate::renderer::helper::triangulate_polygon;
use crate::renderer::parser::resolve_template;
use crate::renderer::touches::{
    EditorInteractionResult, MouseSnapshot, apply_pending_circle_updates, find_top_hit,
    handle_editor_mode_interactions, handle_scroll_resize, handle_text_editing,
    mark_editor_layers_dirty, near_handle, press_began_on_ui,
};
use crate::renderer::ui::{CircleParams, HandleParams, OutlineParams, TextParams};
use crate::resources::{InputState, MouseState, TimeSystem};
use crate::vertex::UiElement::*;
pub(crate) use crate::vertex::*;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;

pub struct UiVariableRegistry {
    pub(crate) vars: HashMap<String, String>,
}

impl UiVariableRegistry {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }

    pub fn set(&mut self, name: &str, value: impl Into<String>) {
        self.vars.insert(name.to_string(), value.into());
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.vars.get(name).map(|s| s.as_str())
    }
}

#[derive(Debug)]
pub struct Menu {
    pub layers: Vec<RuntimeLayer>,
    pub active: bool,
}

impl Menu {
    pub fn rebuild_layer_cache_index(&mut self, layer_index: usize, runtime: &UiRuntime) {
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

                natural_width: t.natural_width,
                natural_height: t.natural_height,
                id: t.id.clone(),
                caret: t.text.len(),
                glyph_bounds: vec![],
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

        // Common builder for vertex-emitting shapes
        let push_with_misc = |v: &UiVertex, misc: [f32; 4], out: &mut Vec<UiVertexPoly>| {
            out.push(UiVertexPoly {
                pos: v.pos,
                _pad: [1.0; 2],
                color: v.color,
                misc,
            });
        };

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

#[derive(Debug)]
pub struct UiRuntime {
    pub elements: HashMap<String, ButtonRuntime>,
    pub selected_ui_element: SelectedUiElement,
    pub active_vertex: Option<usize>,
    pub drag_offset: Option<(f32, f32)>,
    pub editor_mode: bool,
    pub editing_text: bool,
    pub clipboard: String,
    pub dragging_text: bool,
}

impl UiRuntime {
    pub fn new(editor_mode: bool) -> Self {
        Self {
            elements: HashMap::new(),
            selected_ui_element: SelectedUiElement::default(),
            active_vertex: None,
            drag_offset: None,
            editor_mode,
            editing_text: false,
            clipboard: "".to_string(),
            dragging_text: false,
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

pub struct UiButtonLoader {
    pub menus: HashMap<String, Menu>,

    pub console_lines: VecDeque<String>,
    pub ui_runtime: UiRuntime,
    pub variables: UiVariableRegistry,
}

impl UiButtonLoader {
    pub fn new(editor_mode: bool) -> Self {
        let layout = Self::load_gui_from_file("ui_data/gui_layout.json").unwrap_or_else(|e| {
            eprintln!("❌ Failed to load GUI layout: {e}");
            GuiLayout { menus: vec![] }
        });

        let mut loader = Self {
            menus: Default::default(),
            ui_runtime: UiRuntime::new(editor_mode),
            console_lines: VecDeque::new(),
            variables: UiVariableRegistry::new(),
        };

        // JSON layers to runtime layers...…
        let mut id_gen: usize = 1;
        for menu_json in layout.menus {
            let mut layers = Vec::new();

            for l in menu_json.layers {
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

                layers.push(RuntimeLayer {
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
                    saveable: true,
                });
            }

            layers.sort_by_key(|l| l.order); // SORT!
            // Insert this menu into the loader
            loader.menus.insert(
                menu_json.name.clone(),
                Menu {
                    layers,
                    active: true,
                },
            );
        }

        loader.add_editor_layers();
        loader.ensure_console_layer();

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
        let layout = self.to_json_gui_layout();

        let json = serde_json::to_string_pretty(&layout)?;
        fs::write(&full_path, json)?;

        println!("GUI saved to {}", full_path.display());
        Ok(())
    }

    fn to_json_gui_layout(&self) -> GuiLayout {
        let mut menus = Vec::new();
        for (menu_name, menu) in &self.menus {
            let mut layers = Vec::new();

            for l in &menu.layers {
                // Skip editor-only layers to avoid saving internal crap
                if !l.saveable {
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

            menus.push(MenuJson {
                name: menu_name.to_string(),
                layers,
            })
        }

        GuiLayout { menus }
    }

    fn add_editor_layers(&mut self) {
        let menu = self.menus.entry("Editor_Menu".into()).or_insert(Menu {
            layers: Vec::new(),
            active: true,
        });

        // Insert layers
        menu.layers.push(RuntimeLayer {
            name: "editor_selection".into(),
            order: 900,
            active: true,
            cache: LayerCache::default(),
            texts: vec![],
            circles: vec![],
            outlines: vec![],
            handles: vec![],
            polygons: vec![],
            dirty: true,
            gpu: LayerGpu::default(),
            opaque: true,
            saveable: false,
        });

        menu.layers.push(RuntimeLayer {
            name: "editor_handles".into(),
            order: 950,
            active: true,
            cache: LayerCache::default(),
            texts: vec![],
            circles: vec![],
            outlines: vec![],
            handles: vec![],
            polygons: vec![],
            dirty: true,
            gpu: LayerGpu::default(),
            opaque: true,
            saveable: false,
        });

        // ensure correct draw order
        menu.layers.sort_by_key(|l| l.order);
    }

    pub fn ensure_console_layer(&mut self) -> &mut RuntimeLayer {
        // 1. Ensure Debug_Menu exists
        let menu = self.menus.entry("Debug_Menu".into()).or_insert(Menu {
            layers: Vec::new(),
            active: true,
        });

        // 2. If shader_console layer exists, return it
        if let Some(idx) = menu.layers.iter().position(|l| l.name == "shader_console") {
            return &mut menu.layers[idx];
        }

        // 3. Create shader_console layer
        menu.layers.push(RuntimeLayer {
            name: "shader_console".into(),
            order: 980,
            active: true,
            cache: LayerCache::default(),
            texts: vec![],
            circles: vec![],
            outlines: vec![],
            handles: vec![],
            polygons: vec![],
            dirty: true,
            gpu: LayerGpu::default(),
            opaque: false,
            saveable: false,
        });

        // 4. Sort by order
        menu.layers.sort_by_key(|l| l.order);

        // 5. Return newly created layer
        let idx = menu
            .layers
            .iter()
            .position(|l| l.name == "shader_console")
            .expect("shader_console layer missing");

        &mut menu.layers[idx]
    }

    pub fn sync_console_ui(&mut self) {
        let lines: Vec<String> = self.console_lines.iter().cloned().collect();

        let layer = self.ensure_console_layer();

        layer.texts.clear();
        for (i, line) in lines.into_iter().enumerate() {
            layer.texts.push(UiButtonText {
                id: Some(format!("console_line_{}", i)),
                z_index: 980 + i as i32,
                x: 20.0,
                y: 20.0 + i as f32 * 22.0,
                stretch_x: 0.0,
                stretch_y: 0.0,
                top_left_offset: [0.0, 0.0],
                bottom_left_offset: [0.0, 0.0],
                top_right_offset: [0.0, 0.0],
                bottom_right_offset: [0.0, 0.0],
                px: 18,
                color: [0.95, 0.9, 0.8, 0.95],
                text: line.to_string(),
                template: line.to_string(),
                misc: MiscButtonSettings {
                    active: true,
                    touched_time: 0.0,
                    is_touched: false,
                    pressable: false,
                    editable: false,
                },
                natural_width: 50.0,
                natural_height: 20.0,
                being_edited: false,
                caret: line.len(),
                being_hovered: false,
                just_unhovered: false,
                sel_start: 0,
                sel_end: 0,
                has_selection: false,
                glyph_bounds: vec![],
            });
        }

        layer.dirty = true;
    }

    pub fn log_console(&mut self, message: impl Into<String>) {
        self.console_lines.push_back(message.into());

        while self.console_lines.len() > 6 {
            self.console_lines.pop_front();
        }
    }

    pub fn update_dynamic_texts(&mut self) {
        for (_, menu) in &mut self.menus {
            for layer in &mut menu.layers {
                layer.dirty = true;
                let mut any_changed = false;

                for t in &mut layer.texts {
                    // Skip if no template braces exist!
                    if !t.template.contains('{') || !t.template.contains('}') || t.being_edited {
                        continue;
                    }

                    // Resolve template
                    let new_text = resolve_template(&t.template, &self.variables);
                    if new_text != t.text {
                        t.text = new_text;
                        any_changed = true;
                    }
                    if t.being_hovered || t.just_unhovered {
                        any_changed = true;
                    }
                }

                if any_changed {
                    layer.dirty = true;
                }
            }
        }
    }

    pub fn handle_touches(
        &mut self,
        mouse: &MouseState,
        dt: f32,
        input_state: &mut InputState,
        time_system: &TimeSystem,
    ) {
        self.ui_runtime.selected_ui_element.just_deselected = false;
        let mut trigger_selection = false;
        let mouse_snapshot = MouseSnapshot::from_mouse(mouse);
        let editor_mode = self.ui_runtime.editor_mode;

        let press_started_on_ui = if mouse_snapshot.just_pressed {
            press_began_on_ui(&self.menus, &mouse_snapshot, editor_mode)
        } else {
            false
        };

        if mouse_snapshot.just_pressed && !press_started_on_ui && editor_mode {
            if !near_handle(&self.menus, &mouse_snapshot) {
                self.ui_runtime.selected_ui_element.active = false;
                self.ui_runtime.editing_text = false;
                println!("deselection");
                self.ui_runtime.selected_ui_element.just_deselected = true;
                self.update_selection();
            }
        }

        if editor_mode {
            let top_hit = find_top_hit(&mut self.menus, &mouse_snapshot, editor_mode);
            let EditorInteractionResult {
                trigger_selection: mut selection,
                pending_circle_updates,
                moved_any_selected_object,
            } = handle_editor_mode_interactions(
                self,
                time_system,
                &mouse_snapshot,
                top_hit,
                input_state,
            );
            self.variables
                .set("editing_text", format!("{}", self.ui_runtime.editing_text));
            handle_text_editing(
                &mut self.ui_runtime,
                &mut self.menus,
                input_state,
                mouse_snapshot,
                time_system,
            );

            apply_pending_circle_updates(self, dt, pending_circle_updates);

            if moved_any_selected_object {
                mark_editor_layers_dirty(self.menus.get_mut("Editor_Menu"));
                selection = true;
            }

            if handle_scroll_resize(self, mouse_snapshot.scroll) {
                selection = true;
            }

            trigger_selection = selection;
        } else if self.ui_runtime.selected_ui_element.active {
            self.ui_runtime.selected_ui_element.active = false;
            self.update_selection();
        }

        if trigger_selection {
            self.update_selection();
        }
        // ----------------------------------------------------------
        // MOVE ELEMENT UP/DOWN & MOVE LAYER UP/DOWN
        // ----------------------------------------------------------
        if self.ui_runtime.selected_ui_element.active {
            let menu_name = self.ui_runtime.selected_ui_element.menu_name.clone();
            let layer_name = self.ui_runtime.selected_ui_element.layer_name.clone();
            let element_id = self.ui_runtime.selected_ui_element.element_id.clone();

            if let Some(menu) = self.menus.get_mut(&menu_name) {
                if let Some(layer_index) = menu.layers.iter().position(|l| l.name == layer_name) {
                    let layer = &mut menu.layers[layer_index];

                    // ------------------------------------------------------
                    // ELEMENT MOVEMENT
                    // ------------------------------------------------------
                    if let Some(kind) = layer.find_element(&element_id) {
                        let idx = kind.index();

                        let len = match kind {
                            ElementRefMut::Text(_) => layer.texts.len(),
                            ElementRefMut::Polygon(_) => layer.polygons.len(),
                            ElementRefMut::Circle(_) => layer.circles.len(),
                            ElementRefMut::Outline(_) => layer.outlines.len(),
                            ElementRefMut::Handle(_) => layer.handles.len(),
                        };

                        if input_state.action_pressed_once("Move Element Up") && idx > 0 {
                            layer.swap_elements(&kind, idx, idx - 1);
                            println!(
                                "Moved element '{}' up in layer '{}'",
                                element_id, layer_name
                            );
                        }

                        if input_state.action_pressed_once("Move Element Down") && idx + 1 < len {
                            layer.swap_elements(&kind, idx, idx + 1);
                            println!(
                                "Moved element '{}' down in layer '{}'",
                                element_id, layer_name
                            );
                        }
                    }

                    // ------------------------------------------------------
                    // LAYER MOVEMENT
                    // ------------------------------------------------------
                    if input_state.action_pressed_once("Move Layer Up") && layer_index > 0 {
                        menu.layers.swap(layer_index, layer_index - 1);
                        println!("Moved layer '{}' up", layer_name);
                    }

                    if input_state.action_pressed_once("Move Layer Down")
                        && layer_index + 1 < menu.layers.len()
                    {
                        menu.layers.swap(layer_index, layer_index + 1);
                        println!("Moved layer '{}' down", layer_name);
                    }
                }
            }
        }

        if input_state.action_pressed_once("Delete selected GUI Element")
            && self.ui_runtime.selected_ui_element.active
            && !self.ui_runtime.editing_text
        {
            println!("deleting");
            let element_id = self.ui_runtime.selected_ui_element.element_id.clone();
            let layer_name = self.ui_runtime.selected_ui_element.layer_name.clone();
            let menu_name = self.ui_runtime.selected_ui_element.menu_name.clone();
            self.ui_runtime.selected_ui_element.active = false;

            let _ = self.delete_element(&menu_name, &layer_name, &element_id);
        }
    }

    pub fn add_element(
        &mut self,
        layer_name: &str,
        mut element: UiElement,
        mouse: &MouseState,
    ) -> Result<(), String> {
        // 1. Get selected menu
        let menu = self
            .menus
            .get_mut(&self.ui_runtime.selected_ui_element.menu_name)
            .ok_or_else(|| {
                format!(
                    "Menu *{}* doesn't exist",
                    self.ui_runtime.selected_ui_element.menu_name
                )
            })?;

        // 2. Get selected layer
        let layer = menu
            .layers
            .iter_mut()
            .find(|l| l.name == layer_name)
            .ok_or_else(|| {
                format!(
                    "Layer *{}* not found in *{}* menu",
                    layer_name, self.ui_runtime.selected_ui_element.menu_name
                )
            })?;
        let id = mouse.pos.x as u32 - mouse.pos.y as u32;

        // 3. Apply mouse positioning (editor placement)
        match &mut element {
            UiElement::Text(t) => {
                t.x = mouse.pos.x;
                t.y = mouse.pos.y;
                t.id = Option::from(id.to_string());
            }
            UiElement::Circle(c) => {
                c.x = mouse.pos.x;
                c.y = mouse.pos.y;
                c.id = Option::from(id.to_string());
            }
            UiElement::Outline(o) => {
                o.id = Option::from(id.to_string());
            }
            UiElement::Handle(h) => {
                h.x = mouse.pos.x;
                h.y = mouse.pos.y;
                h.id = Option::from(id.to_string());
            }
            UiElement::Polygon(p) => {
                p.id = Option::from(id.to_string());
            }
        }

        // 4. Insert element (NO ids needed)
        match element {
            UiElement::Text(t) => layer.texts.push(t),
            UiElement::Circle(c) => layer.circles.push(c),
            UiElement::Outline(o) => layer.outlines.push(o),
            UiElement::Handle(h) => layer.handles.push(h),
            UiElement::Polygon(p) => layer.polygons.push(p),
        }

        // 5. Mark layer dirty
        layer.dirty = true;

        Ok(())
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
        self.ui_runtime.editing_text = false;
        if !self.ui_runtime.selected_ui_element.active {
            if let Some(editor_menu) = self.menus.get_mut("Editor_Menu") {
                if let Some(editor_layer) = editor_menu
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
            }
            return;
        }
        let sel = self.ui_runtime.selected_ui_element.clone();

        if let Some(menu) = self.menus.get_mut(&sel.menu_name) {
            if let Some(layer) = menu
                .layers
                .iter()
                .find(|l| l.name == sel.layer_name.to_string())
            {
                self.variables
                    .set("selected_layer.order", layer.order.to_string());
            }
        }
        if let Some(element) = self.find_element(&sel.menu_name, &sel.layer_name, &sel.element_id) {
            if let Some(editor_menu) = self.menus.get_mut("Editor_Menu") {
                if let Some(editor_layer) = editor_menu
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
                            self.variables
                                .set("selected_element.z_index", c.z_index.to_string());
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
                            self.variables
                                .set("selected_element.z_index", p.z_index.to_string());
                        }
                        Text(tx) => {
                            //tx.color = [1.0, 1.0, 0.0, 1.0];
                            //tx.misc.active = true;
                            //editor_layer.texts.push(tx);
                            self.variables
                                .set("selected_element.z_index", tx.z_index.to_string());
                        }
                        Outline(_o) => {}
                    }
                }
            }
        }

        self.ui_runtime.selected_ui_element.active = true;
    }

    pub fn find_element(
        &mut self,
        menu_name: &String,
        layer_name: &String,
        element_id: &String,
    ) -> Option<UiElement> {
        if let Some(menu) = self.menus.get_mut(menu_name) {
            let layer = menu
                .layers
                .iter()
                .find(|l| l.name == layer_name.to_string())?;

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
        }
        None
    }

    pub fn delete_element(
        &mut self,
        menu_name: &str,
        layer_name: &str,
        element_id: &str,
    ) -> Option<UiElement> {
        // get mutable layer
        if let Some(menu) = self.menus.get_mut(menu_name) {
            let layer = menu.layers.iter_mut().find(|l| l.name == layer_name)?;

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
        }
        None
    }
}
