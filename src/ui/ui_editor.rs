use crate::data::BendMode;
use crate::paths::data_dir;
use crate::resources::{InputState, TimeSystem};
pub(crate) use crate::ui::actions::{activate_action, execute_action};
use crate::ui::helper::calc_move_speed;
use crate::ui::input::MouseState;
pub(crate) use crate::ui::menu::Menu;
use crate::ui::menu::get_selected_element_color;
use crate::ui::parser::{resolve_template, set_input_box};
use crate::ui::selections::{SelectedUiElement, deselect_everything};
use crate::ui::special_actions::rgb_to_hsv;
use crate::ui::touches::{
    EditorInteractionResult, MouseSnapshot, apply_pending_circle_updates, find_top_hit,
    handle_editor_mode_interactions, handle_scroll_resize, handle_text_editing, near_handle,
    press_began_on_ui,
};
use crate::ui::ui_loader::load_gui_from_file;
pub(crate) use crate::ui::ui_runtime::UiRuntime;
pub(crate) use crate::ui::variables::UiVariableRegistry;
use crate::ui::vertex::UiElement::*;
pub(crate) use crate::ui::vertex::*;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;

pub struct UiButtonLoader {
    pub menus: HashMap<String, Menu>,

    pub console_lines: VecDeque<String>,
    pub ui_runtime: UiRuntime,
    pub variables: UiVariableRegistry,
}

impl UiButtonLoader {
    pub fn new(
        editor_mode: bool,
        override_mode: bool,
        show_gui: bool,
        bend_mode: BendMode,
    ) -> Self {
        let layout_path = data_dir("ui_data/gui_layout.json");
        let layout = load_gui_from_file(layout_path, bend_mode).unwrap_or_else(|e| {
            eprintln!("❌ Failed to load GUI layout: {e}");
            GuiLayout { menus: vec![] }
        });
        println!("{:?}", layout);
        let mut loader = Self {
            menus: Default::default(),
            ui_runtime: UiRuntime::new(editor_mode, override_mode, show_gui),
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
                    dirty: LayerDirty::all(),
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

    pub fn save_gui_to_file(&self, path: PathBuf) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Convert runtime → serializable JSON
        let layout = self.to_json_gui_layout();

        let json = serde_json::to_string_pretty(&layout)?;
        fs::write(&path, json)?;

        println!("GUI saved to {}", path.display());
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
            dirty: LayerDirty::all(),
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
            dirty: LayerDirty::all(),
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
            dirty: LayerDirty::all(),
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
                action: "None".to_string(),
                style: "None".to_string(),
                z_index: 980 + i as i32,
                x: 20.0,
                y: 20.0 + i as f32 * 22.0,
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
                ascent: 10.0,
                being_edited: false,
                caret: line.len(),
                being_hovered: false,
                just_unhovered: false,
                sel_start: 0,
                sel_end: 0,
                has_selection: false,
                glyph_bounds: vec![],
                input_box: false,
                anchor: None,
            });
        }

        layer.dirty.mark_texts();
    }

    pub(crate) fn mark_editor_layers_dirty(&mut self) {
        const TARGETS: [&str; 3] = ["editor_selection", "editor_handles", "shader_console"];

        let menu = self.menus.get_mut("Debug_Menu");
        if let Some(menu) = menu {
            for layer in &mut menu.layers {
                if TARGETS.contains(&layer.name.as_str()) {
                    layer.dirty.mark_all();
                }
            }
        }
    }
    pub fn log_console(&mut self, message: impl Into<String>) {
        self.console_lines.push_back(message.into());

        while self.console_lines.len() > 6 {
            self.console_lines.pop_front();
        }
    }

    pub fn update_dynamic_texts(&mut self) {
        let mut being_hovered = false;
        let mut selected_being_hovered = false;
        for (menu_name, menu) in &mut self.menus {
            for layer in &mut menu.layers {
                let mut any_changed = false;

                for t in &mut layer.texts {
                    if self.ui_runtime.selected_ui_element_primary.just_deselected
                        || self.ui_runtime.selected_ui_element_primary.just_selected
                    {
                        t.clear_selection();
                        layer.dirty.mark_texts();
                    }
                    if t.being_edited || t.being_hovered || t.just_unhovered {
                        any_changed = true;
                    }

                    if !being_hovered && t.being_hovered {
                        being_hovered = true;
                        if t.id
                            == Option::from(
                                self.ui_runtime
                                    .selected_ui_element_primary
                                    .element_id
                                    .clone(),
                            )
                        {
                            selected_being_hovered = true;
                        }
                    }

                    // Skip if no template braces exist!
                    if !t.template.contains('{')
                        || !t.template.contains('}')
                        || (t.being_edited && !t.input_box)
                    {
                        continue;
                    }

                    // Resolve template
                    if !t.input_box || self.ui_runtime.override_mode {
                        // Not an input box, always update!
                        let new_text = resolve_template(&t.template, &self.variables);
                        if new_text != t.text {
                            t.text = new_text;
                            any_changed = true;
                        }
                    } else {
                        // input box stuff
                        if this_text(
                            &self.ui_runtime.selected_ui_element_primary,
                            menu_name,
                            layer.name.as_str(),
                            t.id.clone(),
                        ) {
                            // Editing input box (selected) update the associated variable.
                            let new_text = set_input_box(&t.template, &t.text, &mut self.variables);
                            if new_text != t.text {
                                t.text = new_text;
                                any_changed = true;
                            }
                        } else {
                            // Not editing an input box, just update
                            let new_text = resolve_template(&t.template, &self.variables);
                            if new_text != t.text {
                                t.text = new_text;
                                any_changed = true;
                            }
                        }
                    }
                    if t.input_box && self.ui_runtime.selected_ui_element_primary.just_deselected {
                        if this_text(
                            &self.ui_runtime.selected_ui_element_primary,
                            menu_name,
                            layer.name.as_str(),
                            t.id.clone(),
                        ) {
                            // Editing input box (selected) update the associated variable.
                            let new_text = set_input_box(&t.template, &t.text, &mut self.variables);
                            if new_text != t.text {
                                t.text = new_text;
                                any_changed = true;
                            }
                        }
                    }
                }

                if any_changed {
                    layer.dirty.mark_texts();
                }
            }
        }
        self.variables
            .set_bool("any_text.being_hovered", being_hovered);
        self.variables
            .set_bool("selected_text.being_hovered", selected_being_hovered);
    }

    pub fn handle_touches(
        &mut self,
        dt: f32,
        input_state: &mut InputState,
        time_system: &TimeSystem,
    ) {
        if !self.ui_runtime.show_gui {
            return;
        }
        self.ui_runtime.selected_ui_element_primary.just_deselected = false;
        self.ui_runtime.selected_ui_element_primary.just_selected = false;

        if let Some(color) = get_selected_element_color(self) {
            let r = color[0];
            let g = color[1];
            let b = color[2];

            let (h, s, v) = rgb_to_hsv(r, g, b);

            self.variables.set_f32("color_picker.r", r);
            self.variables.set_f32("color_picker.g", g);
            self.variables.set_f32("color_picker.b", b);
            self.variables.set_f32("color_picker.h", h);
            self.variables.set_f32("color_picker.s", s);
            self.variables.set_f32("color_picker.v", v);
        }

        let mouse_snapshot = MouseSnapshot::from_mouse(&input_state.mouse);
        let editor_mode = self.ui_runtime.editor_mode;

        let press_started_on_ui = if mouse_snapshot.just_pressed {
            press_began_on_ui(&self.menus, &mouse_snapshot, editor_mode)
        } else {
            (false, "None".to_string())
        };

        if mouse_snapshot.just_pressed && !press_started_on_ui.0 {
            if !near_handle(&self.menus, &mouse_snapshot) {
                deselect_everything(self);
            }
        }

        let top_hit = find_top_hit(&mut self.menus, &mouse_snapshot, editor_mode);

        let EditorInteractionResult {
            trigger_selection: mut selection,
            pending_circle_updates,
            moved_any_selected_object,
        } = handle_editor_mode_interactions(
            self,
            time_system,
            &mouse_snapshot,
            top_hit.clone(),
            input_state,
        );
        self.variables
            .set_bool("editing_text", self.ui_runtime.editing_text);
        if editor_mode {
            handle_text_editing(
                &mut self.ui_runtime,
                &mut self.menus,
                input_state,
                mouse_snapshot,
            );

            apply_pending_circle_updates(self, dt, pending_circle_updates);
        }

        if moved_any_selected_object {
            self.mark_editor_layers_dirty();
            selection = true;
        }

        if editor_mode {
            if handle_scroll_resize(self, mouse_snapshot.scroll) {
                selection = true;
            }
        }

        let trigger_selection = selection;

        // if self.ui_runtime.selected_ui_element.active {
        //     self.ui_runtime.selected_ui_element.active = false;
        //     self.update_selection();
        //
        // }

        if trigger_selection {
            self.update_selection();
        }

        if self.ui_runtime.selected_ui_element_primary.just_selected
            || self.ui_runtime.selected_ui_element_primary.just_deselected
        {
            if self.ui_runtime.selected_ui_element_primary.action_name != "Drag Hue Point" {
                if let Some(menu) = self.menus.get_mut("Editor_Menu") {
                    if let Some(_layer) = menu.layers.iter_mut().find(|l| l.name == "Color Picker")
                    {
                        //layer.active = false;
                    }
                }
            }
        }
        if input_state.mouse.right_just_pressed
            && self.ui_runtime.selected_ui_element_primary.element_type != ElementKind::None
        {
            if let Some(menu) = self.menus.get_mut("Editor_Menu") {
                if let Some(layer) = menu.layers.iter_mut().find(|l| l.name == "Color Picker") {
                    layer.active = true;
                }
            }
        }

        activate_action(self, &top_hit, input_state);

        execute_action(self, &top_hit, &input_state.mouse, time_system);

        if editor_mode {
            self.apply_ui_edit_movement(input_state);

            if input_state.action_pressed_once("Delete selected GUI Element")
                && self.ui_runtime.selected_ui_element_primary.active
                && !self.ui_runtime.editing_text
            {
                println!("deleting");
                let element_id = self
                    .ui_runtime
                    .selected_ui_element_primary
                    .element_id
                    .clone();
                let layer_name = self
                    .ui_runtime
                    .selected_ui_element_primary
                    .layer_name
                    .clone();
                let menu_name = self
                    .ui_runtime
                    .selected_ui_element_primary
                    .menu_name
                    .clone();
                deselect_everything(self);

                let _ = self.delete_element(&menu_name, &layer_name, &element_id);
            }
        }
    }

    pub fn add_element(
        &mut self,
        menu_name: &str,
        layer_name: &str,
        mut element: UiElement,
        mouse: &MouseState,
        on_mouse_pos: bool,
    ) -> Result<(), String> {
        // 1. Get selected menu
        let menu = self
            .menus
            .get_mut(menu_name)
            .ok_or_else(|| format!("Menu *{}* doesn't exist", menu_name))?;

        // 2. Get selected layer
        let layer = menu
            .layers
            .iter_mut()
            .find(|l| l.name == layer_name)
            .ok_or_else(|| format!("Layer *{}* not found in *{}* menu", layer_name, menu_name))?;
        let id = mouse.pos.x as u32 - mouse.pos.y as u32;

        // 3. Apply mouse positioning (editor placement)
        if on_mouse_pos {
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
        }
        // 4. Insert element (NO ids needed)
        match element {
            UiElement::Text(t) => {
                layer.texts.push(t);
                layer.dirty.mark_texts();
            }
            UiElement::Circle(c) => {
                layer.circles.push(c);
                layer.dirty.mark_circles();
            }
            UiElement::Outline(o) => {
                layer.outlines.push(o);
                layer.dirty.mark_outlines();
            }
            UiElement::Handle(h) => {
                layer.handles.push(h);
                layer.dirty.mark_handles();
            }
            UiElement::Polygon(p) => {
                layer.polygons.push(p);
                layer.dirty.mark_polygons();
            }
        }

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
        if self.ui_runtime.selected_ui_element_primary.just_deselected {
            self.ui_runtime.editing_text = false;
        }

        if !self.ui_runtime.selected_ui_element_primary.active {
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
                    editor_layer.dirty.mark_all()
                }
            }
            return;
        }
        let sel = self.ui_runtime.selected_ui_element_primary.clone();

        if let Some(menu) = self.menus.get_mut(&sel.menu_name) {
            if let Some(layer) = menu
                .layers
                .iter()
                .find(|l| l.name == sel.layer_name.to_string())
            {
                self.variables
                    .set_i32("selected_layer.order", layer.order as i32);
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
                    editor_layer.dirty.mark_all();
                    editor_layer.circles.clear();
                    editor_layer.outlines.clear();
                    editor_layer.handles.clear();
                    editor_layer.polygons.clear();
                    match element {
                        Circle(c) => {
                            if self.ui_runtime.editor_mode && c.misc.editable {
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
                                        border_thickness: 0.1 * c.radius,
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
                            self.variables
                                .set_i32("selected_element.z_index", c.z_index);
                        }
                        Handle(_h) => {}
                        Polygon(p) => {
                            if self.ui_runtime.editor_mode && p.misc.editable {
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
                                        action: "None".to_string(),
                                        style: "None".to_string(),
                                        z_index: i as i32,
                                        x: v.pos[0],
                                        y: v.pos[1],
                                        radius: 10.0,
                                        inside_border_thickness: 2.0,
                                        border_thickness: 0.0,
                                        fade: 1.0,
                                        fill_color: [0.0, 0.8, 0.0, 0.6],
                                        inside_border_color: [0.0; 4],
                                        border_color: [0.0, 0.0, 0.0, 0.0],
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
                            self.variables
                                .set_i32("selected_element.z_index", p.z_index);
                        }
                        Text(tx) => {
                            // if self.ui_runtime.editor_mode {
                            //
                            // }

                            self.variables
                                .set_i32("selected_element.z_index", tx.z_index);
                        }
                        Outline(_o) => {}
                    }
                }
            }
        }

        self.ui_runtime.selected_ui_element_primary.active = true;
        if !self.ui_runtime.selected_ui_element_primary.just_deselected {
            self.ui_runtime.selected_ui_element_primary.just_selected = true;
        }
    }

    pub fn find_element(
        &mut self,
        menu_name: &str,
        layer_name: &str,
        element_id: &str,
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
                        return Some(Circle(c.clone()));
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

    pub fn edit_circle(
        &mut self,
        menu_name: &str,
        layer_name: &str,
        element_id: &str,
        x: Option<f32>,
        y: Option<f32>,
        radius: Option<f32>,
        fill_color: Option<[f32; 4]>,
        border_color: Option<[f32; 4]>,
    ) -> bool {
        if let Some(menu) = self.menus.get_mut(menu_name) {
            let layer = menu.layers.iter_mut().find(|l| {
                //println!("COMPARING [{}] with [{}]", l.name, layer_name);
                l.name == layer_name
            });

            // Circles
            if let Some(layer) = layer {
                for c in &mut layer.circles {
                    if let Some(id) = &c.id {
                        if id.as_str() == element_id {
                            c.x = x.unwrap_or(c.x);
                            c.y = y.unwrap_or(c.y);
                            c.radius = radius.unwrap_or(c.radius);
                            c.fill_color = fill_color.unwrap_or(c.fill_color);
                            c.border_color = border_color.unwrap_or(c.border_color);
                            layer.dirty.mark_circles();
                            return true;
                        }
                    }
                }
            }
        }
        false
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
    pub fn apply_ui_edit_movement(&mut self, input_state: &mut InputState) {
        let sel = &self.ui_runtime.selected_ui_element_primary;
        if !sel.active {
            return;
        }

        let mut changed = false;

        // ============================================================
        // BLOCK 1: Element XY movement, resizing, z-index movement
        // ============================================================
        {
            let menu = match self.menus.get_mut(&sel.menu_name) {
                Some(m) => m,
                None => return,
            };

            let layer = match menu.layers.iter_mut().find(|l| l.name == sel.layer_name) {
                Some(l) => l,
                None => return,
            };

            // Movement (WASD)
            let speed = calc_move_speed(input_state);
            let mut dx = 0.0;
            let mut dy = 0.0;

            if input_state.action_repeat("Move Element Left") {
                dx -= speed;
            }
            if input_state.action_repeat("Move Element Right") {
                dx += speed;
            }
            if input_state.action_repeat("Move Element Up") {
                dy -= speed;
            }
            if input_state.action_repeat("Move Element Down") {
                dy += speed;
            }

            if dx != 0.0 || dy != 0.0 {
                layer.bump_element_xy(&sel.element_id, dx, dy);
                layer.dirty.mark_all();
                changed = true;
            }

            // -----------------------------
            // Resizing (+, -, scroll)
            // -----------------------------
            let mut scale = 1.0;

            if input_state.action_repeat("Resize Element Bigger") {
                scale = 1.05;
            }
            if input_state.action_repeat("Resize Element Smaller") {
                scale = 0.95;
            }
            if input_state.action_repeat("Resize Element Bigger Scroll") {
                scale = 1.05;
            }
            if input_state.action_repeat("Resize Element Smaller Scroll") {
                scale = 0.95;
            }

            if scale != 1.0 {
                layer.resize_element(&sel.element_id, scale);
                layer.dirty.mark_all();
                changed = true;
            }

            // -----------------------------
            // Element Z movement
            // -----------------------------
            if !input_state.shift {
                if input_state.action_repeat("Move Element Z Up") {
                    layer.bump_element_z(&sel.element_id, 1, &mut self.variables);
                    layer.sort_by_z();
                    layer.dirty.mark_all();
                    changed = true;
                }

                if input_state.action_repeat("Move Element Z Down") {
                    layer.bump_element_z(&sel.element_id, -1, &mut self.variables);
                    layer.sort_by_z();
                    layer.dirty.mark_all();
                    changed = true;
                }
            }
        }
        // ====== end block 1 (layer + menu borrow released) ======

        // ============================================================
        // BLOCK 2: LAYER ORDERING
        // ============================================================
        {
            let menu = match self.menus.get_mut(&sel.menu_name) {
                Some(m) => m,
                None => return,
            };

            if !input_state.shift {
                if input_state.action_repeat("Move Layer Up") {
                    menu.bump_layer_order(&sel.layer_name, 1, &mut self.variables);
                    menu.sort_layers();
                    changed = true;
                }

                if input_state.action_repeat("Move Layer Down") {
                    menu.bump_layer_order(&sel.layer_name, -1, &mut self.variables);
                    menu.sort_layers();
                    changed = true;
                }
            }
        }
        // ====== end block 2 ======

        // ============================================================
        // FINAL: only dirty if something changed
        // ============================================================
        if changed {
            self.update_selection();
            self.mark_editor_layers_dirty();
        }
    }
}

fn this_text(
    selected: &SelectedUiElement,
    menu: &str,
    layer: &str,
    element_id: Option<String>,
) -> bool {
    if selected.menu_name != menu {
        return false;
    }
    if selected.layer_name != layer {
        return false;
    }

    if let Some(id) = element_id {
        return selected.element_id == id;
    }

    false
}
