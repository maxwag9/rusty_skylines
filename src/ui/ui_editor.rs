//! UI Button Loader with integrated undo/redo support
//!
//! Uses Command pattern for all undoable operations.

use crate::data::BendMode;
use crate::hsv::{HSV, rgb_to_hsv};
use crate::paths::data_dir;
use crate::renderer::world_renderer::WorldRenderer;
use crate::resources::{InputState, TimeSystem};
use crate::ui::actions::{ActionSystem, activate_action, execute_action};
use crate::ui::helper::calc_move_speed;
use crate::ui::input::MouseState;
use crate::ui::menu::{Menu, get_selected_element_color};
use crate::ui::parser::{resolve_template, set_input_box};
use crate::ui::selections::{SelectedUiElement, deselect_everything};
use crate::ui::touches::{
    MouseSnapshot, apply_pending_circle_updates, find_top_hit, handle_editor_mode_interactions,
    handle_scroll_resize, handle_text_editing, near_handle, press_began_on_ui,
};
use crate::ui::ui_edit_manager::{
    ChangeZIndexCommand, Command, ModifyPolygonCommand, MoveElementCommand, ResizeElementCommand,
    UiEditManager,
};
use crate::ui::ui_edits::*;
use crate::ui::ui_loader::load_gui_from_file;
use crate::ui::ui_runtime::UiRuntime;
use crate::ui::variables::UiVariableRegistry;
use crate::ui::vertex::*;
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::PathBuf;
use winit::dpi::PhysicalSize;

// ============================================================================
// DRAG STATE FOR UNDO TRACKING
// ============================================================================

/// Captured state when a drag operation begins
#[derive(Clone, Debug)]
pub struct DragStartState {
    pub menu: String,
    pub layer: String,
    pub element_id: String,
    pub element_kind: ElementKind,
    pub start_pos: (f32, f32),
    pub start_size: Option<f32>,
    pub start_vertices: Option<Vec<[f32; 2]>>,
}

/// Snap-to-grid settings
#[derive(Clone, Debug)]
pub struct SnapSettings {
    pub enabled: bool,
    pub grid_size: f32,
}

impl Default for SnapSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            grid_size: 10.0,
        }
    }
}

// ============================================================================
// UI BUTTON LOADER
// ============================================================================

/// Main UI loader with editor and undo support
pub struct UiButtonLoader {
    /// All menus keyed by name
    pub menus: HashMap<String, Menu>,
    /// Runtime state for UI interactions
    pub ui_runtime: UiRuntime,
    /// Console output lines
    pub console_lines: VecDeque<String>,
    /// Variable registry for dynamic text
    pub variables: UiVariableRegistry,
    /// UiEdit/Undo/Redo manager
    pub ui_edit_manager: UiEditManager,
    /// Current drag operation state (for undo tracking)
    pub drag_start_state: Option<DragStartState>,
    /// Snap-to-grid settings
    pub snap_settings: SnapSettings,
    /// Multi-selection support (element IDs)
    pub multi_selection: Vec<SelectedUiElement>,
    /// Clipboard for copy/paste elements
    pub element_clipboard: Option<UiElement>,
}

impl UiButtonLoader {
    pub fn new(
        editor_mode: bool,
        override_mode: bool,
        show_gui: bool,
        bend_mode: BendMode,
        window_size: PhysicalSize<u32>,
    ) -> Self {
        let layout_path = data_dir("ui_data/gui_layout.yaml");
        let layout = load_gui_from_file(layout_path, bend_mode).unwrap_or_else(|e| {
            eprintln!("❌ Failed to load GUI layout: {e}");
            GuiLayout { menus: vec![] }
        });

        let mut loader = Self {
            menus: Default::default(),
            ui_runtime: UiRuntime::new(editor_mode, override_mode, show_gui),
            console_lines: VecDeque::new(),
            variables: UiVariableRegistry::new(),
            ui_edit_manager: UiEditManager::new(),
            drag_start_state: None,
            snap_settings: SnapSettings::default(),
            multi_selection: Vec::new(),
            element_clipboard: None,
        };

        // Load menus from layout
        let mut id_gen: usize = 1;
        for menu_json in layout.menus {
            let mut layers = Vec::new();

            for l in menu_json.layers {
                let texts = l
                    .texts
                    .unwrap_or_default()
                    .into_iter()
                    .map(|t| UiButtonText::from_json(t, window_size))
                    .collect();

                let circles = l
                    .circles
                    .unwrap_or_default()
                    .into_iter()
                    .map(|c| UiButtonCircle::from_json(c, window_size))
                    .collect();

                let outlines = l
                    .outlines
                    .unwrap_or_default()
                    .into_iter()
                    .map(|o| UiButtonOutline::from_json(o, window_size))
                    .collect();

                let handles = l
                    .handles
                    .unwrap_or_default()
                    .into_iter()
                    .map(|h| UiButtonHandle::from_json(h, window_size))
                    .collect();

                let polygons = l
                    .polygons
                    .unwrap_or_default()
                    .into_iter()
                    .map(|p| UiButtonPolygon::from_json(p, &mut id_gen, window_size))
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

            layers.sort_by_key(|l| l.order);
            loader.menus.insert(
                menu_json.name.clone(),
                Menu {
                    layers,
                    active: true,
                },
            );
        }

        loader.add_editor_layers();
        loader
    }

    pub fn save_gui_to_file(
        &mut self,
        path: PathBuf,
        window_size: PhysicalSize<u32>,
    ) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let layout = self.to_json_gui_layout(window_size);
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("yaml");

        let content = match extension {
            "json" => serde_json::to_string_pretty(&layout)?,
            "yaml" | "yml" | _ => serde_yaml::to_string(&layout)?,
        };

        fs::write(&path, &content)?;
        self.ui_edit_manager.mark_saved();

        println!("GUI saved to {} ({} bytes)", path.display(), content.len());
        Ok(())
    }

    fn to_json_gui_layout(&self, window_size: PhysicalSize<u32>) -> GuiLayout {
        let mut menus = Vec::new();
        for (menu_name, menu) in &self.menus {
            let mut layers = Vec::new();

            for l in &menu.layers {
                if !l.saveable {
                    continue;
                }

                layers.push(UiLayerJson {
                    name: l.name.clone(),
                    order: l.order,
                    active: Some(l.active),
                    opaque: Some(l.opaque),
                    texts: Some(
                        l.texts
                            .iter()
                            .map(|t: &UiButtonText| t.to_json(window_size))
                            .collect(),
                    ),
                    circles: Some(l.circles.iter().map(|c| c.to_json(window_size)).collect()),
                    outlines: Some(l.outlines.iter().map(|o| o.to_json(window_size)).collect()),
                    handles: Some(l.handles.iter().map(|h| h.to_json(window_size)).collect()),
                    polygons: Some(l.polygons.iter().map(|p| p.to_json(window_size)).collect()),
                });
            }

            menus.push(MenuJson {
                name: menu_name.to_string(),
                layers,
            })
        }

        GuiLayout { menus }
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

                    if !t.template.contains('{')
                        || !t.template.contains('}')
                        || (t.being_edited && !t.input_box)
                    {
                        continue;
                    }

                    if !t.input_box || self.ui_runtime.override_mode {
                        let new_text = resolve_template(&t.template, &self.variables);
                        if new_text != t.text {
                            t.text = new_text;
                            any_changed = true;
                        }
                    } else {
                        if this_text(
                            &self.ui_runtime.selected_ui_element_primary,
                            menu_name,
                            layer.name.as_str(),
                            t.id.clone(),
                        ) {
                            let new_text = set_input_box(&t.template, &t.text, &mut self.variables);
                            if new_text != t.text {
                                t.text = new_text;
                                any_changed = true;
                            }
                        } else {
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

        if let Some(element) = get_element(
            &self.menus,
            &sel.menu_name,
            &sel.layer_name,
            &sel.element_id,
        ) {
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
                        UiElement::Circle(c) => {
                            if self.ui_runtime.editor_mode && c.misc.editable {
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
                                };
                                editor_layer.handles.push(handle);
                            }
                        }
                        UiElement::Handle(_h) => {}
                        UiElement::Polygon(p) => {
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
                        }
                        UiElement::Text(_tx) => {}
                        UiElement::Outline(_o) => {}
                    }
                }
            }
        }

        self.ui_runtime.selected_ui_element_primary.active = true;
        if !self.ui_runtime.selected_ui_element_primary.just_deselected {
            self.ui_runtime.selected_ui_element_primary.just_selected = true;
        }
    }

    pub(crate) fn mark_editor_layers_dirty(&mut self) {
        const TARGETS: [&str; 3] = ["editor_selection", "editor_handles", "shader_console"];

        if let Some(menu) = self.menus.get_mut("Debug_Menu") {
            for layer in &mut menu.layers {
                if TARGETS.contains(&layer.name.as_str()) {
                    layer.dirty.mark_all();
                }
            }
        }
    }

    // ========================================================================
    // UNDO/REDO OPERATIONS
    // ========================================================================

    /// Handle Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y input
    pub fn handle_undo_redo_input(&mut self, input_state: &mut InputState, dt: f32) {
        self.ui_edit_manager.update(dt);

        if !self.ui_runtime.editor_mode {
            return;
        }

        // Ctrl+Shift+Z
        if input_state.action_pressed_once("Redo") {
            self.perform_redo(&input_state.mouse);
            return;
        }
        // Ctrl+Z = Undo
        if input_state.action_pressed_once("Undo") {
            println!("hi");
            self.perform_undo(&input_state.mouse);
            return;
        }
    }

    /// Perform undo operation
    pub fn perform_undo(&mut self, mouse: &MouseState) {
        println!("performing UNDERWOOD");
        if let Some(desc) = self.ui_edit_manager.undo(
            &mut self.ui_runtime,
            &mut self.menus,
            &mut self.variables,
            mouse,
        ) {
            self.log_console(format!("↩ Undo: {}", desc));
            self.mark_editor_layers_dirty();
            self.update_selection();
        } else {
            self.log_console("Nothing to undo");
        }
    }

    /// Perform redo operation
    pub fn perform_redo(&mut self, mouse: &MouseState) {
        if let Some(desc) = self.ui_edit_manager.redo(
            &mut self.ui_runtime,
            &mut self.menus,
            &mut self.variables,
            mouse,
        ) {
            self.log_console(format!("↪ Redo: {}", desc));
            self.mark_editor_layers_dirty();
            self.update_selection();
        } else {
            self.log_console("Nothing to redo");
        }
    }

    /// Delete element by ID (internal, for undo system)
    pub fn delete_element_by_id(&mut self, menu_name: &str, layer_name: &str, element_id: &str) {
        if let Some(element) = get_element(&self.menus, menu_name, layer_name, element_id) {
            delete_element(&mut self.menus, menu_name, layer_name, &element);
        }
    }

    /// Called when drag starts - captures initial state
    pub fn begin_drag(
        &mut self,
        menu: &str,
        layer: &str,
        element_id: &str,
        element_kind: ElementKind,
    ) {
        let start_pos = get_element_position(&self.menus, menu, layer, element_id);
        let start_size = get_element_size(&self.menus, menu, layer, element_id, element_kind);
        let start_vertices = if element_kind == ElementKind::Polygon {
            get_polygon_vertices(&self.menus, menu, layer, element_id)
        } else {
            None
        };

        self.drag_start_state = Some(DragStartState {
            menu: menu.to_string(),
            layer: layer.to_string(),
            element_id: element_id.to_string(),
            element_kind,
            start_pos,
            start_size,
            start_vertices,
        });
    }

    /// Called when drag ends - records undo action if state changed
    pub fn end_drag(&mut self) {
        let Some(start) = self.drag_start_state.take() else {
            return;
        };

        // Check for position change
        let new_pos =
            get_element_position(&self.menus, &start.menu, &start.layer, &start.element_id);
        let dx = (new_pos.0 - start.start_pos.0).abs();
        let dy = (new_pos.1 - start.start_pos.1).abs();

        if dx > 0.5 || dy > 0.5 {
            self.ui_edit_manager.push_command(MoveElementCommand {
                menu: start.menu.clone(),
                layer: start.layer.clone(),
                element_id: start.element_id.clone(),
                element_kind: start.element_kind,
                before: start.start_pos,
                after: new_pos,
            });
        }

        // Check for size change
        if let (Some(old_size), Some(new_size)) = (
            start.start_size,
            get_element_size(
                &self.menus,
                &start.menu,
                &start.layer,
                &start.element_id,
                start.element_kind,
            ),
        ) {
            if (new_size - old_size).abs() > 0.5 {
                self.ui_edit_manager.push_command(ResizeElementCommand {
                    menu: start.menu.clone(),
                    layer: start.layer.clone(),
                    element_id: start.element_id.clone(),
                    element_kind: start.element_kind,
                    before: old_size,
                    after: new_size,
                });
            }
        }

        // Check for vertex changes (polygons)
        if start.element_kind == ElementKind::Polygon {
            if let (Some(old_verts), Some(new_verts)) = (
                &start.start_vertices,
                get_polygon_vertices(&self.menus, &start.menu, &start.layer, &start.element_id),
            ) {
                let changed = old_verts.iter().zip(new_verts.iter()).any(|(old, new)| {
                    (old[0] - new[0]).abs() > 0.5 || (old[1] - new[1]).abs() > 0.5
                });

                if changed {
                    if let (Some(old_state), Some(new_state)) = (
                        self.create_polygon_snapshot_with_vertices(
                            &start.menu,
                            &start.layer,
                            &start.element_id,
                            old_verts,
                        ),
                        self.get_polygon(&start.menu, &start.layer, &start.element_id),
                    ) {
                        self.ui_edit_manager.push_command(ModifyPolygonCommand {
                            menu: start.menu,
                            layer: start.layer,
                            before: old_state,
                            after: new_state,
                        });
                    }
                }
            }
        }
    }

    /// Finalize any in-progress drag operation (alias for end_drag)
    pub fn finalize_drag(&mut self) {
        self.end_drag();
    }

    // ========================================================================
    // ELEMENT POSITION/SIZE/COLOR SETTERS (for undo system)
    // ========================================================================

    pub fn set_element_position(
        &mut self,
        menu: &str,
        layer: &str,
        id: &str,
        kind: ElementKind,
        pos: (f32, f32),
    ) {
        let Some(menu) = self.menus.get_mut(menu) else {
            return;
        };
        let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer) else {
            return;
        };

        match kind {
            ElementKind::Circle => {
                if let Some(c) = layer
                    .circles
                    .iter_mut()
                    .find(|c| c.id.as_deref() == Some(id))
                {
                    c.x = pos.0;
                    c.y = pos.1;
                    layer.dirty.mark_circles();
                }
            }
            ElementKind::Text => {
                if let Some(t) = layer.texts.iter_mut().find(|t| t.id.as_deref() == Some(id)) {
                    t.x = pos.0;
                    t.y = pos.1;
                    layer.dirty.mark_texts();
                }
            }
            ElementKind::Polygon => {
                if let Some(p) = layer
                    .polygons
                    .iter_mut()
                    .find(|p| p.id.as_deref() == Some(id))
                {
                    let (cx, cy) = p.center();
                    let dx = pos.0 - cx;
                    let dy = pos.1 - cy;
                    for v in &mut p.vertices {
                        v.pos[0] += dx;
                        v.pos[1] += dy;
                    }
                    layer.dirty.mark_polygons();
                }
            }
            ElementKind::Handle => {
                if let Some(h) = layer
                    .handles
                    .iter_mut()
                    .find(|h| h.id.as_deref() == Some(id))
                {
                    h.x = pos.0;
                    h.y = pos.1;
                    layer.dirty.mark_handles();
                }
            }
            _ => {}
        }
    }

    pub fn set_element_size(
        &mut self,
        menu: &str,
        layer: &str,
        id: &str,
        kind: ElementKind,
        size: f32,
    ) {
        let Some(menu) = self.menus.get_mut(menu) else {
            return;
        };
        let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer) else {
            return;
        };

        match kind {
            ElementKind::Circle => {
                if let Some(c) = layer
                    .circles
                    .iter_mut()
                    .find(|c| c.id.as_deref() == Some(id))
                {
                    c.radius = size.max(2.0);
                    layer.dirty.mark_circles();
                }
            }
            ElementKind::Text => {
                if let Some(t) = layer.texts.iter_mut().find(|t| t.id.as_deref() == Some(id)) {
                    t.px = size.max(4.0) as u16;
                    layer.dirty.mark_texts();
                }
            }
            _ => {}
        }
    }

    /// Change z-index (internal, for undo system - does NOT record undo)
    pub fn change_z_index(&mut self, menu: &str, layer: &str, id: &str, delta: i32) {
        let Some(menu) = self.menus.get_mut(menu) else {
            return;
        };
        let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer) else {
            return;
        };

        layer.bump_element_z(id, delta);
        layer.dirty.mark_all();
    }

    /// Change z-index with undo support
    pub fn change_z_index_undoable(&mut self, menu: &str, layer: &str, id: &str, delta: i32) {
        self.ui_edit_manager.push_command(ChangeZIndexCommand {
            menu: menu.to_string(),
            layer: layer.to_string(),
            element_id: id.to_string(),
            delta,
        });

        self.change_z_index(menu, layer, id, delta);
    }

    pub fn replace_circle(&mut self, menu: &str, layer: &str, new_state: &UiButtonCircle) {
        let Some(menu) = self.menus.get_mut(menu) else {
            return;
        };
        let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer) else {
            return;
        };

        if let Some(c) = layer.circles.iter_mut().find(|c| c.id == new_state.id) {
            *c = new_state.clone();
            layer.dirty.mark_circles();
        }
    }

    pub fn replace_text(&mut self, menu: &str, layer: &str, new_state: &UiButtonText) {
        let Some(menu) = self.menus.get_mut(menu) else {
            return;
        };
        let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer) else {
            return;
        };

        if let Some(t) = layer.texts.iter_mut().find(|t| t.id == new_state.id) {
            *t = new_state.clone();
            layer.dirty.mark_texts();
        }
    }

    pub fn replace_polygon(&mut self, menu: &str, layer: &str, new_state: &UiButtonPolygon) {
        let Some(menu) = self.menus.get_mut(menu) else {
            return;
        };
        let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer) else {
            return;
        };

        if let Some(p) = layer.polygons.iter_mut().find(|p| p.id == new_state.id) {
            *p = new_state.clone();
            layer.dirty.mark_polygons();
        }
    }

    fn get_polygon(&self, menu: &str, layer: &str, id: &str) -> Option<UiButtonPolygon> {
        let menu = self.menus.get(menu)?;
        let layer = menu.layers.iter().find(|l| l.name == layer)?;
        layer
            .polygons
            .iter()
            .find(|p| p.id.as_deref() == Some(id))
            .cloned()
    }

    fn create_polygon_snapshot_with_vertices(
        &self,
        menu: &str,
        layer: &str,
        id: &str,
        vertices: &[[f32; 2]],
    ) -> Option<UiButtonPolygon> {
        let mut poly = self.get_polygon(menu, layer, id)?;
        for (i, pos) in vertices.iter().enumerate() {
            if let Some(v) = poly.vertices.get_mut(i) {
                v.pos = *pos;
            }
        }
        Some(poly)
    }

    // ========================================================================
    // UTILITY FUNCTIONS
    // ========================================================================

    fn generate_unique_id(&self) -> u32 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u32)
            .unwrap_or(0)
    }

    fn apply_snap(&self, pos: (f32, f32)) -> (f32, f32) {
        if !self.snap_settings.enabled {
            return pos;
        }
        let grid = self.snap_settings.grid_size;
        ((pos.0 / grid).round() * grid, (pos.1 / grid).round() * grid)
    }

    pub fn log_console(&mut self, message: impl Into<String>) {
        self.console_lines.push_back(message.into());
        while self.console_lines.len() > 6 {
            self.console_lines.pop_front();
        }
    }

    pub fn has_unsaved_changes(&self) -> bool {
        self.ui_edit_manager.is_dirty()
    }

    pub fn handle_touches(
        &mut self,
        action_system: &mut ActionSystem,
        dt: f32,
        input_state: &mut InputState,
        time_system: &TimeSystem,
        world_renderer: &mut WorldRenderer,
        window_size: PhysicalSize<u32>,
    ) {
        if !self.ui_runtime.show_gui {
            return;
        }

        self.ui_edit_manager.update(dt);
        self.handle_undo_redo_input(input_state, dt);
        self.reset_selection_flags();
        self.sync_selected_element_color();

        let mouse = MouseSnapshot::from_mouse(&input_state.mouse);
        let editor_mode = self.ui_runtime.editor_mode;

        if mouse.just_pressed && !self.press_started_on_ui(&mouse) && !self.near_any_handle(&mouse)
        {
            deselect_everything(self);
        }

        let top_hit = find_top_hit(&mut self.menus, &mouse, editor_mode);

        let interaction = handle_editor_mode_interactions(
            self,
            time_system,
            &mouse,
            top_hit.clone(),
            input_state,
        );

        self.variables
            .set_bool("editing_text", self.ui_runtime.editing_text);

        let mut trigger_selection =
            interaction.trigger_selection || interaction.moved_any_selected_object;

        if editor_mode {
            handle_text_editing(
                &mut self.ui_runtime,
                &mut self.menus,
                &mut self.ui_edit_manager,
                input_state,
                mouse,
            );
            apply_pending_circle_updates(self, dt, interaction.pending_circle_updates);

            if handle_scroll_resize(self, mouse.scroll) {
                trigger_selection = true;
            }
        }

        // Check if drag ended (mouse released while we had a drag in progress)
        if !mouse.pressed && self.drag_start_state.is_some() {
            self.finalize_drag();
        }

        if interaction.moved_any_selected_object {
            self.mark_editor_layers_dirty();
        }

        if trigger_selection {
            self.update_selection();
        }

        activate_action(
            action_system,
            self,
            &top_hit,
            &input_state.mouse,
            input_state,
            time_system,
            world_renderer,
            window_size,
        );

        execute_action(
            action_system,
            self,
            &top_hit,
            &input_state.mouse,
            input_state,
            time_system,
            world_renderer,
            window_size,
        );

        if editor_mode {
            self.apply_ui_edit_movement(input_state);
        }
    }

    pub fn toggle_snap(&mut self) {
        self.snap_settings.enabled = !self.snap_settings.enabled;
        self.log_console(format!(
            "Snap to grid: {}",
            if self.snap_settings.enabled {
                "ON"
            } else {
                "OFF"
            }
        ));
    }

    pub fn set_grid_size(&mut self, size: f32) {
        self.snap_settings.grid_size = size.max(1.0);
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

            // Resizing (+, -, scroll)
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

            // Element Z movement
            if !input_state.shift {
                if input_state.action_repeat("Move Element Z Up") {
                    layer.bump_element_z(&sel.element_id, 1);
                    layer.dirty.mark_all();
                    changed = true;
                }

                if input_state.action_repeat("Move Element Z Down") {
                    layer.bump_element_z(&sel.element_id, -1);
                    layer.dirty.mark_all();
                    changed = true;
                }
            }
        }

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

        if changed {
            self.update_selection();
            self.mark_editor_layers_dirty();
        }
    }

    fn reset_selection_flags(&mut self) {
        self.ui_runtime.selected_ui_element_primary.just_deselected = false;
        self.ui_runtime.selected_ui_element_primary.just_selected = false;
    }

    fn sync_selected_element_color(&mut self) {
        let Some(color) = get_selected_element_color(self) else {
            return;
        };

        let HSV { h, s, v } = rgb_to_hsv(color);

        self.variables.set_f32("color_picker.r", color[0]);
        self.variables.set_f32("color_picker.g", color[1]);
        self.variables.set_f32("color_picker.b", color[2]);
        self.variables.set_f32("color_picker.h", h);
        self.variables.set_f32("color_picker.s", s);
        self.variables.set_f32("color_picker.v", v);
    }

    fn press_started_on_ui(&self, mouse: &MouseSnapshot) -> bool {
        if !mouse.just_pressed {
            return false;
        }
        press_began_on_ui(&self.menus, mouse, self.ui_runtime.editor_mode).0
    }

    fn near_any_handle(&self, mouse: &MouseSnapshot) -> bool {
        near_handle(&self.menus, mouse)
    }

    fn add_editor_layers(&mut self) {
        let menu = self.menus.entry("Editor_Menu".into()).or_insert(Menu {
            layers: Vec::new(),
            active: true,
        });

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

        menu.layers.sort_by_key(|l| l.order);
    }
    pub(crate) fn hash_id(id: &str) -> f32 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        id.hash(&mut hasher);
        let hash_u64 = hasher.finish(); // 64-bit
        // map to [0, 1]
        (hash_u64 as f64 / u64::MAX as f64) as f32
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn colors_equal(a: &[f32; 4], b: &[f32; 4]) -> bool {
    const EPSILON: f32 = 0.001;
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < EPSILON)
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

pub fn get_element(
    menus: &HashMap<String, Menu>,
    menu_name: &str,
    layer_name: &str,
    element_id: &str,
) -> Option<UiElement> {
    let menu = menus.get(menu_name)?;
    let layer = menu.layers.iter().find(|l| l.name == layer_name)?;

    if let Some(c) = layer
        .circles
        .iter()
        .find(|c| c.id.as_deref() == Some(element_id))
    {
        return Some(UiElement::Circle(c.clone()));
    }
    if let Some(t) = layer
        .texts
        .iter()
        .find(|t| t.id.as_deref() == Some(element_id))
    {
        return Some(UiElement::Text(t.clone()));
    }
    if let Some(p) = layer
        .polygons
        .iter()
        .find(|p| p.id.as_deref() == Some(element_id))
    {
        return Some(UiElement::Polygon(p.clone()));
    }
    if let Some(h) = layer
        .handles
        .iter()
        .find(|h| h.id.as_deref() == Some(element_id))
    {
        return Some(UiElement::Handle(h.clone()));
    }
    if let Some(o) = layer
        .outlines
        .iter()
        .find(|o| o.id.as_deref() == Some(element_id))
    {
        return Some(UiElement::Outline(o.clone()));
    }

    None
}

pub fn get_element_position(
    menus: &HashMap<String, Menu>,
    menu_name: &str,
    layer_name: &str,
    element_id: &str,
) -> (f32, f32) {
    if let Some(element) = get_element(menus, menu_name, layer_name, element_id) {
        element.center()
    } else {
        (0.0, 0.0)
    }
}
pub fn get_element_size(
    menus: &HashMap<String, Menu>,
    menu_name: &str,
    layer_name: &str,
    id: &str,
    kind: ElementKind,
) -> Option<f32> {
    let menu = menus.get(menu_name)?;
    let layer = menu.layers.iter().find(|l| l.name == layer_name)?;

    match kind {
        ElementKind::Circle => layer
            .circles
            .iter()
            .find(|c| c.id.as_deref() == Some(id))
            .map(|c| c.radius),
        ElementKind::Text => layer
            .texts
            .iter()
            .find(|t| t.id.as_deref() == Some(id))
            .map(|t| t.px as f32),
        ElementKind::Handle => layer
            .handles
            .iter()
            .find(|h| h.id.as_deref() == Some(id))
            .map(|h| h.radius),
        _ => None,
    }
}

pub fn get_polygon_vertices(
    menus: &HashMap<String, Menu>,
    menu_name: &str,
    layer_name: &str,
    id: &str,
) -> Option<Vec<[f32; 2]>> {
    let menu = menus.get(menu_name)?;
    let layer = menu.layers.iter().find(|l| l.name == layer_name)?;

    layer
        .polygons
        .iter()
        .find(|p| p.id.as_deref() == Some(id))
        .map(|p| p.vertices.iter().map(|v| v.pos).collect())
}
