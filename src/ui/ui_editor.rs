//! UI Button Loader with integrated undo/redo support
//!
//! Uses Command pattern for all undoable operations.

use crate::data::BendMode;
use crate::hsv::{HSV, rgb_to_hsv};
use crate::paths::data_dir;
use crate::renderer::world_renderer::TerrainRenderer;
use crate::resources::{InputState, TimeSystem};
use crate::ui::actions::{ActionSystem, activate_action, execute_action};
use crate::ui::helper::calc_move_speed;
use crate::ui::input::MouseState;
use crate::ui::menu::{Menu, get_selected_element_color};
use crate::ui::parser::{resolve_template, set_input_box};
use crate::ui::selections::SelectedUiElement;
use crate::ui::ui_edit_manager::{
    Command, DeselectAllCommand, MoveElementCommand, MoveVertexCommand, ResizeElementCommand,
    UiEditManager,
};
use crate::ui::ui_edits::*;
use crate::ui::ui_loader::{
    load_legacy_gui_layout_legacy, load_menus_from_directory, sanitize_filename,
};
use crate::ui::ui_text_editing::{MouseSnapshot, handle_text_editing};
use crate::ui::ui_touch_manager::{
    DragCoordinator, ElementRef, HitDetector, InputSnapshot, NavigationDirection, TouchEvent,
    UiTouchManager,
};
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
    pub affected_element: ElementRef,
    pub start_pos: (f32, f32),
    pub start_size: Option<f32>,
    pub start_vertices: Option<Vec<[f32; 2]>>,
}

/// Results from processing touch events
#[derive(Default)]
struct EventProcessingResult {
    /// Commands to push to undo system
    commands: Vec<Box<dyn Command>>,
    /// Whether to update selection visuals
    update_selection: bool,
    /// Whether to mark layers dirty
    mark_dirty: bool,
    /// Action to trigger
    trigger_action: Option<String>,
    /// Whether a drag operation ended
    drag_ended: bool,
}

pub struct GuiOptions {
    pub(crate) show_gui: bool,
    pub(crate) override_mode: bool,
}

// ============================================================================
// UI BUTTON LOADER
// ============================================================================

pub struct UiButtonLoader {
    // Core data
    pub menus: HashMap<String, Menu>,
    pub variables: UiVariableRegistry,
    pub console_lines: VecDeque<String>,

    // NEW: Touch manager (primary touch handling)
    pub touch_manager: UiTouchManager,

    // Undo/Redo
    pub ui_edit_manager: UiEditManager,

    // Drag tracking for undo
    pub drag_start_state: Option<DragStartState>,

    // Multi-selection (now delegated to touch_manager.selection)
    pub multi_selection: Vec<SelectedUiElement>,

    // Element clipboard
    pub element_clipboard: Option<UiElement>,
}

impl UiButtonLoader {
    pub fn new(
        editor_mode: bool,
        override_mode: bool,
        show_gui: bool,
        bend_mode: &BendMode,
        window_size: PhysicalSize<u32>,
    ) -> Self {
        let menus_dir = data_dir("ui_data/menus");
        let legacy_path = data_dir("ui_data/gui_layout.yaml");

        let menu_files = load_menus_from_directory(&menus_dir, bend_mode)
            .ok()
            .filter(|menus| !menus.is_empty())
            .unwrap_or_else(|| {
                println!("No menus in directory, trying legacy file...");
                load_legacy_gui_layout_legacy(&legacy_path, bend_mode)
            });

        let mut loader = Self {
            menus: HashMap::new(),
            variables: UiVariableRegistry::new(),
            console_lines: VecDeque::new(),
            touch_manager: UiTouchManager::new(editor_mode, override_mode, show_gui),
            ui_edit_manager: UiEditManager::new(),
            drag_start_state: None,
            multi_selection: Vec::new(),
            element_clipboard: None,
        };

        // Load menus
        for menu_yaml in menu_files {
            let mut layers = Vec::new();

            for l in menu_yaml.layers {
                let elements = l
                    .elements
                    .unwrap_or_default()
                    .into_iter()
                    .map(|t| UiElement::from_yaml(t, window_size))
                    .collect();

                layers.push(RuntimeLayer {
                    name: l.name,
                    order: l.order,
                    active: l.active,
                    opaque: l.opaque,
                    elements,
                    cache: LayerCache::default(),
                    gpu: LayerGpu::default(),
                    dirty: LayerDirty::all(),
                    saveable: true,
                });
            }

            layers.sort_by_key(|l| l.order);
            loader.menus.insert(
                menu_yaml.name.clone(),
                Menu {
                    layers,
                    active: true,
                },
            );
        }

        loader.add_editor_layers();
        loader
    }

    // ========================================================================
    // MAIN UPDATE LOOP - NEW IMPLEMENTATION
    // ========================================================================

    pub fn handle_touches(
        &mut self,
        action_system: &mut ActionSystem,
        dt: f32,
        input_state: &mut InputState,
        time_system: &TimeSystem,
        world_renderer: &mut TerrainRenderer,
        window_size: PhysicalSize<u32>,
    ) {
        if !self.touch_manager.options.show_gui {
            return;
        }

        // Update undo manager timing
        self.ui_edit_manager.update(dt);
        self.handle_undo_redo_input(input_state, dt);

        // Reset frame flags
        self.reset_selection_flags();
        self.sync_selected_element_color();

        // Create input snapshot
        let input_snapshot = self.create_input_snapshot(&input_state.mouse, input_state);

        // Collect elements - borrow only self.menus
        let elements = Self::collect_touchable_elements_from(&self.menus);

        // Now we can mutably borrow touch_manager separately
        self.touch_manager
            .update(dt, input_snapshot, elements.into_iter());
        // Process all emitted events
        let result = self.process_touch_events(&input_state.mouse);
        if result.drag_ended || result.update_selection {
            println!(
                "Result drag_ended: {}, Result Selection updated: {}, Text editing: {}",
                result.drag_ended,
                result.update_selection,
                self.touch_manager.editor.editing_text.is_some()
            );
        }

        // Apply results
        self.apply_event_results(result, &input_state.mouse);

        // Handle text editing
        if self.touch_manager.editor.enabled {
            self.handle_text_editing(input_state, input_snapshot);
        }

        // Handle keyboard navigation
        self.handle_keyboard_navigation(input_state);

        // Handle keyboard movement of elements
        if self.touch_manager.editor.enabled {
            self.apply_ui_edit_movement(input_state);
        }

        // Execute actions
        let top_hit = self.get_current_hit_for_actions();
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
    }

    /// Create input snapshot from mouse state
    fn create_input_snapshot(&self, mouse: &MouseState, input: &InputState) -> InputSnapshot {
        InputSnapshot {
            position: (mouse.pos.x, mouse.pos.y),
            pressed: mouse.left_pressed,
            just_pressed: mouse.left_just_pressed,
            just_released: !mouse.left_pressed && self.touch_manager.last_input.pressed,
            scroll_delta: mouse.scroll_delta.y,
            ctrl_held: input.ctrl,
            shift_held: input.shift,
            alt_held: input.alt,
        }
    }

    /// Collect all touchable elements from menus - return references, not owned Strings
    fn collect_touchable_elements_from(
        menus: &HashMap<String, Menu>, // adjust type as needed
    ) -> Vec<(&str, &str, u32, usize, &UiElement)> {
        let mut elements = Vec::new();

        for (menu_name, menu) in menus {
            if !menu.active {
                continue;
            }

            for layer in &menu.layers {
                if !layer.active {
                    continue;
                }

                for (idx, element) in layer.elements.iter().enumerate() {
                    elements.push((
                        menu_name.as_str(),  // &'a str - borrows from HashMap key
                        layer.name.as_str(), // &'a str - borrows from Layer in Menu
                        layer.order,
                        idx,
                        element,
                    ));
                }
            }
        }

        elements
    }

    /// Process all touch events and return combined result
    fn process_touch_events(&mut self, mouse: &MouseState) -> EventProcessingResult {
        let mut result = EventProcessingResult::default();

        // Drain events from touch manager
        let events: Vec<_> = self.touch_manager.events.drain().collect();

        for event in events {
            self.handle_touch_event(&event, &mut result, mouse);
        }

        result
    }

    /// Handle a single touch event
    fn handle_touch_event(
        &mut self,
        event: &TouchEvent,
        result: &mut EventProcessingResult,
        mouse: &MouseState,
    ) {
        println!("handle_touch_event: {:?}", event);

        match event {
            // ----------------------------------------------------------------
            // HOVER EVENTS
            // ----------------------------------------------------------------
            TouchEvent::HoverEnter { element } => {
                self.handle_hover_enter(element, result);
            }
            TouchEvent::HoverExit { element } => {
                self.handle_hover_exit(element, result);
            }

            // ----------------------------------------------------------------
            // PRESS/RELEASE EVENTS
            // ----------------------------------------------------------------
            TouchEvent::Press {
                element,
                position,
                vertex_index,
            } => {
                self.handle_press(element, *position, *vertex_index, result);
            }
            TouchEvent::Release {
                element,
                position,
                was_drag,
                action,
            } => {
                self.handle_release(element, *position, *was_drag, action.clone(), result);
            }
            TouchEvent::Click {
                element,
                position,
                action,
            } => {
                self.handle_click(element, *position, action.clone(), result);
            }
            TouchEvent::DoubleClick { element, position } => {
                self.handle_double_click(element, *position, result);
            }

            // ----------------------------------------------------------------
            // DRAG EVENTS
            // ----------------------------------------------------------------
            TouchEvent::DragStart {
                element,
                start_position,
                vertex_index,
            } => {
                self.handle_drag_start(element, *start_position, *vertex_index, result);
            }
            TouchEvent::DragMove {
                element,
                current_position,
                delta,
                total_delta,
            } => {
                self.handle_drag_move(
                    element,
                    *current_position,
                    *delta,
                    *total_delta,
                    result,
                    mouse,
                );
            }
            TouchEvent::DragEnd {
                element,
                start_position,
                end_position,
                vertex_index,
            } => {
                self.handle_drag_end(
                    element,
                    *start_position,
                    *end_position,
                    *vertex_index,
                    result,
                );
            }

            // ----------------------------------------------------------------
            // SELECTION EVENTS
            // ----------------------------------------------------------------
            TouchEvent::SelectionRequested {
                element,
                additive,
                multi,
            } => {
                self.handle_selection_requested(element, *additive, *multi, result);
            }
            TouchEvent::DeselectAllRequested => {
                self.handle_deselect_all(result, mouse);
            }
            TouchEvent::BoxSelectStart { start } => {
                self.handle_box_select_start(*start, result);
            }
            TouchEvent::BoxSelectMove { current } => {
                self.handle_box_select_move(*current, result);
            }
            TouchEvent::BoxSelectEnd { start, end } => {
                self.handle_box_select_end(*start, *end, result);
            }

            // ----------------------------------------------------------------
            // SCROLL/RESIZE EVENTS
            // ----------------------------------------------------------------
            TouchEvent::ScrollOnElement { element, delta } => {
                self.handle_scroll_on_element(element, *delta, result);
            }

            // ----------------------------------------------------------------
            // TEXT EVENTS
            // ----------------------------------------------------------------
            TouchEvent::TextEditRequested { element } => {
                self.handle_text_edit_requested(element, result);
            }
            TouchEvent::TextEditEnded { element } => {
                self.handle_text_edit_ended(element, result);
            }

            // ----------------------------------------------------------------
            // NAVIGATION EVENTS
            // ----------------------------------------------------------------
            TouchEvent::NavigateDirection { direction } => {
                self.handle_navigate_direction(*direction, result);
            }
        }
    }

    // ========================================================================
    // EVENT HANDLERS
    // ========================================================================

    fn handle_hover_enter(&mut self, element: &ElementRef, result: &mut EventProcessingResult) {
        // Update text hover state
        if element.kind == ElementKind::Text {
            if let Some(text) = self.get_text_mut(&element.menu, &element.layer, &element.id) {
                text.being_hovered = true;
                text.just_unhovered = false;
                result.mark_dirty = true;
            }
        }

        self.variables.set_bool("any_text.being_hovered", true);
    }

    fn handle_hover_exit(&mut self, element: &ElementRef, result: &mut EventProcessingResult) {
        if element.kind == ElementKind::Text {
            if let Some(text) = self.get_text_mut(&element.menu, &element.layer, &element.id) {
                text.being_hovered = false;
                text.just_unhovered = true;
                result.mark_dirty = true;
            }
        }
    }

    fn handle_press(
        &mut self,
        element: &ElementRef,
        position: (f32, f32),
        vertex_index: Option<usize>,
        result: &mut EventProcessingResult,
    ) {
        // Store drag offset for editor mode
        if self.touch_manager.editor.enabled {
            if let Some(elem) = self.get_element(&element.menu, &element.layer, &element.id) {
                if vertex_index.is_none() {
                    let center = elem.center();
                    let offset = (position.0 - center.0, position.1 - center.1);
                    if let Some(active_drag) = &mut self.touch_manager.drag.active_drag {
                        active_drag.offset = offset;
                    }
                }
                self.touch_manager.editor.active_vertex = vertex_index;
            }
        }

        result.mark_dirty = true;
    }

    fn handle_release(
        &mut self,
        _element: &ElementRef,
        _position: (f32, f32),
        was_drag: bool,
        action: Option<String>,
        result: &mut EventProcessingResult,
    ) {
        self.touch_manager.drag.active_drag = None;
        self.touch_manager.editor.active_vertex = None;

        if !was_drag {
            if let Some(action_name) = action {
                result.trigger_action = Some(action_name);
            }
        }

        result.update_selection = true;
        result.mark_dirty = true;
    }

    fn handle_click(
        &mut self,
        _element: &ElementRef,
        _position: (f32, f32),
        action: Option<String>,
        result: &mut EventProcessingResult,
    ) {
        if let Some(action_name) = action {
            result.trigger_action = Some(action_name);
        }
    }

    fn handle_double_click(
        &mut self,
        element: &ElementRef,
        _position: (f32, f32),
        result: &mut EventProcessingResult,
    ) {
        if element.kind != ElementKind::Text || !self.touch_manager.editor.enabled {
            return;
        }

        // Check if editable first (immutable borrow)
        let is_editable = self
            .get_element(&element.menu, &element.layer, &element.id)
            .and_then(|e| e.as_text())
            .map(|t| t.misc.editable)
            .unwrap_or(false);

        if !is_editable {
            return;
        }

        // Now do mutations (borrow is dropped)
        self.touch_manager.editor.editing_text = Some(element.clone());

        // Get mutable reference and update
        if let Some(text) = self.get_text_mut(&element.menu, &element.layer, &element.id) {
            text.being_edited = true;
            text.text = text.template.clone();
        }

        result.mark_dirty = true;
    }

    fn handle_drag_start(
        &mut self,
        element: &ElementRef,
        start_position: (f32, f32),
        vertex_index: Option<usize>,
        _result: &mut EventProcessingResult,
    ) {
        if !self.touch_manager.editor.enabled {
            return;
        }

        // Capture start state for undo
        let start_size = self.get_element_size_by_ref(element);
        let start_vertices = if element.kind == ElementKind::Polygon {
            self.get_polygon_vertices(&element.menu, &element.layer, &element.id)
        } else {
            None
        };

        self.drag_start_state = Some(DragStartState {
            affected_element: element.clone(),
            start_pos: start_position,
            start_size,
            start_vertices,
        });

        self.touch_manager.editor.active_vertex = vertex_index;
    }

    fn handle_drag_move(
        &mut self,
        element: &ElementRef,
        current_position: (f32, f32),
        delta: (f32, f32),
        _total_delta: (f32, f32),
        result: &mut EventProcessingResult,
        mouse: &MouseState,
    ) {
        if !self.touch_manager.editor.enabled {
            return;
        }
        let Some(active_drag) = &mut self.touch_manager.drag.active_drag else {
            return;
        };

        let offset = active_drag.offset;
        let new_pos = (current_position.0 - offset.0, current_position.1 - offset.1);

        // Apply snap if enabled
        let snapped_pos = if self.touch_manager.config.snap_enabled {
            DragCoordinator::apply_snap(new_pos, &self.touch_manager.config)
        } else {
            new_pos
        };

        match element.kind {
            ElementKind::Circle => {
                self.move_circle(&element.menu, &element.layer, &element.id, snapped_pos);
            }
            ElementKind::Text => {
                self.move_text(&element.menu, &element.layer, &element.id, snapped_pos);
            }
            ElementKind::Polygon => {
                if let Some(vertex_idx) = self.touch_manager.editor.active_vertex {
                    self.move_polygon_vertex(
                        &element.menu,
                        &element.layer,
                        &element.id,
                        vertex_idx,
                        snapped_pos,
                    );
                } else {
                    self.move_polygon(&element.menu, &element.layer, &element.id, delta);
                }
            }
            ElementKind::Handle => {
                self.handle_handle_drag(
                    &element.menu,
                    &element.layer,
                    &element.id,
                    current_position,
                    mouse,
                );
            }
            _ => {}
        }

        result.mark_dirty = true;
        result.update_selection = true;
    }

    fn handle_drag_end(
        &mut self,
        element: &ElementRef,
        _start_position: (f32, f32),
        _end_position: (f32, f32),
        vertex_index: Option<usize>,
        result: &mut EventProcessingResult,
    ) {
        if let Some(start_state) = self.drag_start_state.take() {
            // Create undo command for the completed drag
            let new_pos = self.get_element_position(&element.menu, &element.layer, &element.id);

            let dx = (new_pos.0 - start_state.start_pos.0).abs();
            let dy = (new_pos.1 - start_state.start_pos.1).abs();

            if dx > 0.5 || dy > 0.5 {
                self.ui_edit_manager.push_command(MoveElementCommand {
                    affected_element: element.clone(),
                    before: start_state.start_pos,
                    after: new_pos,
                });
            }

            // Handle vertex moves for polygons
            if element.kind == ElementKind::Polygon {
                if let Some(idx) = vertex_index {
                    if let (Some(old_verts), Some(new_verts)) = (
                        &start_state.start_vertices,
                        self.get_polygon_vertices(&element.menu, &element.layer, &element.id),
                    ) {
                        if let (Some(old_pos), Some(new_pos)) =
                            (old_verts.get(idx), new_verts.get(idx))
                        {
                            self.ui_edit_manager.push_command(MoveVertexCommand {
                                affected_element: element.clone(),
                                vertex_index: idx,
                                before: *old_pos,
                                after: *new_pos,
                            });
                        }
                    }
                }
            }
        }

        self.touch_manager.drag.active_drag = None;
        self.touch_manager.editor.active_vertex = None;
        result.drag_ended = true;
        result.update_selection = true;
        result.mark_dirty = true;
    }

    fn handle_selection_requested(
        &mut self,
        element: &ElementRef,
        additive: bool,
        multi: bool,
        result: &mut EventProcessingResult,
    ) {
        // Get action for element
        let action = self.get_element_action(&element.menu, &element.layer, &element.id);
        let is_input_box = self.is_input_box(&element.menu, &element.layer, &element.id);

        if multi {
            self.touch_manager
                .selection
                .move_primary_to_multi(element.clone(), action);
        } else if additive {
            self.touch_manager
                .selection
                .toggle_selection(element.clone());
        } else {
            self.touch_manager
                .selection
                .select(element.clone(), action, is_input_box);
        }

        result.update_selection = true;
    }

    fn handle_deselect_all(&mut self, result: &mut EventProcessingResult, mouse: &MouseState) {
        self.ui_edit_manager.execute_command(
            DeselectAllCommand {
                primary: self.touch_manager.selection.primary.clone(),
                secondary: self.touch_manager.selection.secondary.clone(),
            },
            &mut self.touch_manager,
            &mut self.menus,
            &mut self.variables,
            &mouse,
        );
    }

    fn handle_box_select_start(&mut self, start: (f32, f32), _result: &mut EventProcessingResult) {
        self.touch_manager.selection.begin_box_select(start);
    }

    fn handle_box_select_move(
        &mut self,
        _current: (f32, f32),
        _result: &mut EventProcessingResult,
    ) {
        // Could draw selection rectangle here
    }

    fn handle_box_select_end(
        &mut self,
        start: (f32, f32),
        end: (f32, f32),
        result: &mut EventProcessingResult,
    ) {
        // Find all elements in box
        let elements: Vec<_> = Self::collect_touchable_elements_from(&self.menus)
            .into_iter()
            .map(|(menu, layer, _, _, elem)| (menu, layer, elem))
            .collect();

        let selected =
            HitDetector::find_in_box(start, end, elements.iter().map(|(m, l, e)| (*m, *l, *e)));

        self.touch_manager
            .selection
            .set_from_box(selected, &mut self.menus);
        result.update_selection = true;
    }

    fn handle_scroll_on_element(
        &mut self,
        element: &ElementRef,
        delta: f32,
        result: &mut EventProcessingResult,
    ) {
        if element.kind != ElementKind::Circle {
            return;
        }

        if let Some(old_size) = self.get_element_size_by_ref(element) {
            let new_size = (old_size + delta * 3.0).max(2.0);

            // Push resize command
            self.ui_edit_manager.push_command(ResizeElementCommand {
                affected_element: element.clone(),
                before: old_size,
                after: new_size,
            });

            // Apply resize
            self.set_element_size(
                &element.menu,
                &element.layer,
                &element.id,
                element.kind,
                new_size,
            );
            result.mark_dirty = true;
        }
    }

    fn handle_text_edit_requested(
        &mut self,
        element: &ElementRef,
        result: &mut EventProcessingResult,
    ) {
        // Check existence first
        let exists = self
            .get_element(&element.menu, &element.layer, &element.id)
            .and_then(|e| e.as_text())
            .is_some();

        if !exists {
            return;
        }

        // Update non-menu fields first
        self.touch_manager.editor.editing_text = Some(element.clone());
        self.variables.set_bool("selected_text.being_edited", true);

        // Then update the text element
        if let Some(text) = self.get_text_mut(&element.menu, &element.layer, &element.id) {
            text.being_edited = true;
            text.text = text.template.clone();
        }

        result.mark_dirty = true;
    }

    fn handle_text_edit_ended(&mut self, element: &ElementRef, result: &mut EventProcessingResult) {
        if let Some(text) = self.get_text_mut(&element.menu, &element.layer, &element.id) {
            text.template = text.text.clone();
            text.being_edited = false;
            self.touch_manager.editor.editing_text = None;
            self.variables.set_bool("selected_text.being_edited", false);
            result.mark_dirty = true;
        }
    }

    fn handle_navigate_direction(
        &mut self,
        direction: NavigationDirection,
        result: &mut EventProcessingResult,
    ) {
        let current = match &self.touch_manager.selection.primary {
            Some(elem) => elem.clone(),
            None => return,
        };

        let current_pos = self.get_element_position(&current.menu, &current.layer, &current.id);

        // Find the best element in direction
        if let Some(next) = self.find_element_in_direction(&current, current_pos, direction) {
            self.touch_manager.selection.select(next, None, false);
            result.update_selection = true;
        }
    }

    // ========================================================================
    // APPLY RESULTS
    // ========================================================================

    fn apply_event_results(&mut self, result: EventProcessingResult, mouse: &MouseState) {
        // Push undo commands
        for cmd in result.commands {
            self.ui_edit_manager.execute(
                cmd,
                &mut self.touch_manager,
                &mut self.menus,
                &mut self.variables,
                mouse,
            );
        }

        // Mark layers dirty
        if result.mark_dirty {
            self.mark_all_layers_dirty();
        }

        // Update selection visuals
        if result.update_selection {
            self.update_selection();
        }
    }

    // ========================================================================
    // ELEMENT MANIPULATION HELPERS
    // ========================================================================

    fn move_circle(&mut self, menu: &str, layer: &str, id: &str, pos: (f32, f32)) {
        if let Some(menu_data) = self.menus.get_mut(menu) {
            if let Some(layer_data) = menu_data.layers.iter_mut().find(|l| l.name == layer) {
                if let Some(circle) = layer_data
                    .elements
                    .iter_mut()
                    .filter_map(UiElement::as_circle_mut)
                    .find(|c| c.id == id)
                {
                    circle.x = pos.0;
                    circle.y = pos.1;
                    layer_data.dirty.mark_circles();
                }
            }
        }
    }

    fn move_text(&mut self, menu: &str, layer: &str, id: &str, pos: (f32, f32)) {
        if let Some(menu_data) = self.menus.get_mut(menu) {
            if let Some(layer_data) = menu_data.layers.iter_mut().find(|l| l.name == layer) {
                if let Some(text) = layer_data
                    .elements
                    .iter_mut()
                    .filter_map(UiElement::as_text_mut)
                    .find(|t| t.id == id)
                {
                    if text.being_edited {
                        return;
                    }
                    text.x = pos.0;
                    text.y = pos.1;
                    layer_data.dirty.mark_texts();
                }
            }
        }
    }

    fn move_polygon(&mut self, menu: &str, layer: &str, id: &str, delta: (f32, f32)) {
        if let Some(menu_data) = self.menus.get_mut(menu) {
            if let Some(layer_data) = menu_data.layers.iter_mut().find(|l| l.name == layer) {
                if let Some(poly) = layer_data
                    .elements
                    .iter_mut()
                    .filter_map(UiElement::as_polygon_mut)
                    .find(|p| p.id == id)
                {
                    for v in &mut poly.vertices {
                        v.pos[0] += delta.0;
                        v.pos[1] += delta.1;
                    }
                    layer_data.dirty.mark_polygons();
                }
            }
        }
    }

    fn move_polygon_vertex(
        &mut self,
        menu: &str,
        layer: &str,
        id: &str,
        vertex_idx: usize,
        pos: (f32, f32),
    ) {
        if let Some(menu_data) = self.menus.get_mut(menu) {
            if let Some(layer_data) = menu_data.layers.iter_mut().find(|l| l.name == layer) {
                if let Some(poly) = layer_data
                    .elements
                    .iter_mut()
                    .filter_map(UiElement::as_polygon_mut)
                    .find(|p| p.id == id)
                {
                    if let Some(v) = poly.vertices.get_mut(vertex_idx) {
                        v.pos = [pos.0, pos.1];
                        layer_data.dirty.mark_polygons();
                    }
                }
            }
        }
    }

    fn handle_handle_drag(
        &mut self,
        menu_name: &str,
        layer_name: &str,
        id: &str,
        position: (f32, f32),
        mouse: &MouseState,
    ) {
        // Extract parent_id first (clone to own the data)
        let affected_element = {
            let handle = match self.get_handle(menu_name, layer_name, id) {
                Some(h) => h,
                None => return,
            };
            match &handle.parent {
                Some(affected_element) => affected_element.clone(),
                None => return,
            }
        };
        // Find parent circle and update its radius
        if let Some(menu) = self.menus.get_mut(&affected_element.menu) {
            if let Some(layer) = menu
                .layers
                .iter_mut()
                .find(|l| l.name == affected_element.layer)
            {
                if let Some(circle) = layer
                    .elements
                    .iter_mut()
                    .filter_map(UiElement::as_circle_mut)
                    .find(|c| c.id == affected_element.id.as_str())
                {
                    let dx = position.0 - circle.x;
                    let dy = position.1 - circle.y;
                    let new_radius = (dx * dx + dy * dy).sqrt().max(2.0);
                    self.ui_edit_manager.execute_command(
                        ResizeElementCommand {
                            affected_element,
                            before: circle.radius,
                            after: new_radius,
                        },
                        &mut self.touch_manager,
                        &mut self.menus,
                        &mut self.variables,
                        mouse,
                    );
                    return;
                }
            }
        }
    }

    // ========================================================================
    // ELEMENT GETTERS
    // ========================================================================

    fn get_element(&self, menu: &str, layer: &str, id: &str) -> Option<&UiElement> {
        let menu_data = self.menus.get(menu)?;
        let layer_data = menu_data.layers.iter().find(|l| l.name == layer)?;
        layer_data.elements.iter().find(|e| e.id() == id)
    }

    fn get_text_mut(&mut self, menu: &str, layer: &str, id: &str) -> Option<&mut UiButtonText> {
        let menu_data = self.menus.get_mut(menu)?;
        let layer_data = menu_data.layers.iter_mut().find(|l| l.name == layer)?;
        layer_data
            .elements
            .iter_mut()
            .filter_map(UiElement::as_text_mut)
            .find(|t| t.id == id)
    }

    fn get_handle(&self, menu: &str, layer: &str, id: &str) -> Option<&UiButtonHandle> {
        let menu_data = self.menus.get(menu)?;
        let layer_data = menu_data.layers.iter().find(|l| l.name == layer)?;
        layer_data.iter_handles().find(|h| h.id == id)
    }

    fn get_element_position(&self, menu: &str, layer: &str, id: &str) -> (f32, f32) {
        self.get_element(menu, layer, id)
            .map(|e| e.center())
            .unwrap_or((0.0, 0.0))
    }

    fn get_element_size_by_ref(&self, element: &ElementRef) -> Option<f32> {
        get_element_size(
            &self.menus,
            &element.menu,
            &element.layer,
            &element.id,
            element.kind,
        )
    }

    fn get_polygon_vertices(&self, menu: &str, layer: &str, id: &str) -> Option<Vec<[f32; 2]>> {
        let menu_data = self.menus.get(menu)?;
        let layer_data = menu_data.layers.iter().find(|l| l.name == layer)?;
        layer_data
            .iter_polygons()
            .find(|p| p.id == id)
            .map(|p| p.vertices.iter().map(|v| v.pos).collect())
    }

    fn get_element_action(&self, menu: &str, layer: &str, id: &str) -> Option<String> {
        self.get_element(menu, layer, id).and_then(|e| match e {
            UiElement::Circle(c) => Some(c.action.clone()),
            UiElement::Polygon(p) => Some(p.action.clone()),
            UiElement::Text(t) => Some(t.action.clone()),
            _ => None,
        })
    }

    fn is_input_box(&self, menu: &str, layer: &str, id: &str) -> bool {
        self.get_element(menu, layer, id)
            .and_then(|e| e.as_text())
            .map(|t| t.input_box)
            .unwrap_or(false)
    }

    fn set_element_size(
        &mut self,
        menu: &str,
        layer: &str,
        id: &str,
        kind: ElementKind,
        size: f32,
    ) {
        set_element_size(&mut self.menus, menu, layer, id, kind, size);
    }

    // ========================================================================
    // KEYBOARD NAVIGATION
    // ========================================================================

    fn handle_keyboard_navigation(&mut self, input: &mut InputState) {
        if input.ctrl || !self.touch_manager.editor.enabled {
            return;
        }

        let direction = if input.action_repeat("Navigate UI Left") {
            Some(NavigationDirection::Left)
        } else if input.action_repeat("Navigate UI Right") {
            Some(NavigationDirection::Right)
        } else if input.action_repeat("Navigate UI Up") {
            Some(NavigationDirection::Up)
        } else if input.action_repeat("Navigate UI Down") {
            Some(NavigationDirection::Down)
        } else {
            None
        };

        if let Some(dir) = direction {
            self.touch_manager.process_navigation(dir);
        }
    }

    fn find_element_in_direction(
        &self,
        from: &ElementRef,
        from_pos: (f32, f32),
        direction: NavigationDirection,
    ) -> Option<ElementRef> {
        let dir_vec = match direction {
            NavigationDirection::Up => (0.0, -1.0),
            NavigationDirection::Down => (0.0, 1.0),
            NavigationDirection::Left => (-1.0, 0.0),
            NavigationDirection::Right => (1.0, 0.0),
        };

        let menu = self.menus.get(&from.menu)?;
        let mut best: Option<(ElementRef, f32)> = None;
        let max_angle = 45.0_f32.to_radians();
        let cos_max = max_angle.cos();

        for layer in &menu.layers {
            if !layer.active || !layer.saveable {
                continue;
            }

            for elem in &layer.elements {
                let id = match elem.id() {
                    id if id != from.id => id,
                    _ => continue,
                };

                let pos = elem.center();
                let dx = pos.0 - from_pos.0;
                let dy = pos.1 - from_pos.1;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist < 1.0 {
                    continue;
                }

                // Check direction
                let cos_theta = (dx * dir_vec.0 + dy * dir_vec.1) / dist;
                if cos_theta < cos_max {
                    continue;
                }

                // Prefer closer elements
                match &best {
                    Some((_, best_dist)) if dist >= *best_dist => continue,
                    _ => {}
                }

                best = Some((
                    ElementRef::new(&from.menu, &layer.name, id, elem.kind()),
                    dist,
                ));
            }
        }

        best.map(|(elem, _)| elem)
    }

    // ========================================================================
    // LEGACY SYNC
    // ========================================================================

    fn reset_selection_flags(&mut self) {
        self.touch_manager.selection.reset_frame_flags();
    }

    // ========================================================================
    // UNDO/REDO
    // ========================================================================

    pub fn handle_undo_redo_input(&mut self, input: &mut InputState, dt: f32) {
        self.ui_edit_manager.update(dt);

        if !self.touch_manager.editor.enabled {
            return;
        }

        if input.action_pressed_once("Redo") {
            self.perform_redo(&input.mouse);
            return;
        }

        if input.action_pressed_once("Undo") {
            self.perform_undo(&input.mouse);
            return;
        }
    }

    pub fn perform_undo(&mut self, mouse: &MouseState) {
        if let Some(desc) = self.ui_edit_manager.undo(
            &mut self.touch_manager,
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

    pub fn perform_redo(&mut self, mouse: &MouseState) {
        if let Some(desc) = self.ui_edit_manager.redo(
            &mut self.touch_manager,
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

    // ========================================================================
    // UTILITY METHODS (kept from original)
    // ========================================================================

    pub fn log_console(&mut self, message: impl Into<String>) {
        self.console_lines.push_back(message.into());
        while self.console_lines.len() > 6 {
            self.console_lines.pop_front();
        }
    }

    pub fn has_unsaved_changes(&self) -> bool {
        self.ui_edit_manager.is_dirty()
    }

    pub fn toggle_snap(&mut self) {
        self.touch_manager.toggle_snap();
        self.log_console(format!(
            "Snap to grid: {}",
            if self.touch_manager.config.snap_enabled {
                "ON"
            } else {
                "OFF"
            }
        ));
    }

    pub fn set_grid_size(&mut self, size: f32) {
        self.touch_manager.set_snap_grid(size);
    }

    fn mark_all_layers_dirty(&mut self) {
        for menu in self.menus.values_mut() {
            for layer in &mut menu.layers {
                layer.dirty.mark_all();
            }
        }
    }

    pub fn mark_editor_layers_dirty(&mut self) {
        const TARGETS: [&str; 3] = ["editor_selection", "editor_handles", "shader_console"];

        if let Some(menu) = self.menus.get_mut("Editor_Menu") {
            for layer in &mut menu.layers {
                if TARGETS.contains(&layer.name.as_str()) {
                    layer.dirty.mark_all();
                }
            }
        }
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

    fn get_current_hit_for_actions(&self) -> Option<crate::ui::ui_text_editing::HitResult> {
        // Convert touch manager's current hover to legacy HitResult for action system
        self.touch_manager
            .hovered()
            .map(|elem| crate::ui::ui_text_editing::HitResult {
                menu_name: elem.menu.clone(),
                layer_name: elem.layer.clone(),
                element: match elem.kind {
                    ElementKind::Circle => crate::ui::ui_text_editing::HitElement::Circle(0),
                    ElementKind::Polygon => crate::ui::ui_text_editing::HitElement::Polygon(0),
                    ElementKind::Text => crate::ui::ui_text_editing::HitElement::Text(0),
                    ElementKind::Handle => crate::ui::ui_text_editing::HitElement::Handle(0),
                    _ => crate::ui::ui_text_editing::HitElement::Circle(0),
                },
                layer_order: 0,
                element_order: 0,
                action: self.get_element_action(&elem.menu, &elem.layer, &elem.id),
            })
    }

    fn handle_text_editing(&mut self, input: &mut InputState, snapshot: InputSnapshot) {
        if self.touch_manager.editor.editing_text.is_some() {
            let mouse_snapshot = MouseSnapshot {
                mx: snapshot.position.0,
                my: snapshot.position.1,
                pressed: snapshot.pressed,
                just_pressed: snapshot.just_pressed,
                scroll: snapshot.scroll_delta,
            };

            handle_text_editing(
                &mut self.touch_manager.selection,
                &mut self.touch_manager.editor,
                &mut self.menus,
                &mut self.ui_edit_manager,
                input,
                mouse_snapshot,
            );
        }
    }

    pub fn save_gui_to_file(
        &mut self,
        menus_dir: PathBuf,
        window_size: PhysicalSize<u32>,
    ) -> anyhow::Result<()> {
        fs::create_dir_all(&menus_dir)?;

        let mut total_bytes = 0usize;
        let mut saved_count = 0usize;

        for (menu_name, menu) in &self.menus {
            let menu_yaml = self.menu_to_yaml(menu_name, menu, window_size);

            // Skip menus with no saveable layers
            if menu_yaml.layers.is_empty() {
                continue;
            }
            let safe_name = sanitize_filename(menu_name);
            let file_path = menus_dir.join(format!("{}.{}", safe_name, "yaml"));

            let content = serde_yaml::to_string(&menu_yaml)?;

            fs::write(&file_path, &content)?;
            total_bytes += content.len();
            saved_count += 1;

            println!("  📁 {} ({} bytes)", file_path.display(), content.len());
        }

        self.ui_edit_manager.mark_saved();
        println!(
            "✅ GUI saved: {} menus, {} total bytes in {}",
            saved_count,
            total_bytes,
            menus_dir.display()
        );

        Ok(())
    }

    fn menu_to_yaml(
        &self,
        menu_name: &str,
        menu: &Menu,
        window_size: PhysicalSize<u32>,
    ) -> MenuYaml {
        let layers = menu
            .layers
            .iter()
            .filter(|l| l.saveable)
            .map(|l| UiLayerYaml {
                name: l.name.clone(),
                order: l.order,
                active: l.active,
                opaque: l.opaque,
                elements: Some(
                    l.elements
                        .iter()
                        .map(|e| UiElement::to_yaml(e, window_size))
                        .collect(),
                ),
            })
            .collect();

        MenuYaml {
            name: menu_name.to_string(),
            layers,
        }
    }

    pub fn update_dynamic_texts(&mut self) {
        let mut being_hovered = false;
        let mut selected_being_hovered = false;

        for (menu_name, menu) in &mut self.menus {
            for layer in &mut menu.layers {
                let mut any_changed = false;

                for t in layer.elements.iter_mut().filter_map(UiElement::as_text_mut) {
                    if self.touch_manager.selection.just_deselected
                        || self.touch_manager.selection.just_selected
                    {
                        t.clear_selection();
                        layer.dirty.mark_texts();
                    }
                    if t.being_edited || t.being_hovered || t.just_unhovered {
                        any_changed = true;
                    }

                    if !being_hovered && t.being_hovered {
                        being_hovered = true;
                        if let Some(sel) = &self.touch_manager.selection.primary {
                            if t.id == sel.id {
                                selected_being_hovered = true;
                            }
                        }
                    }

                    if !t.template.contains('{')
                        || !t.template.contains('}')
                        || (t.being_edited && !t.input_box)
                    {
                        continue;
                    }

                    if !t.input_box || self.touch_manager.options.override_mode {
                        let new_text = resolve_template(&t.template, &self.variables);
                        if new_text != t.text {
                            t.text = new_text;
                            any_changed = true;
                        }
                    } else {
                        if this_text(
                            &self.touch_manager.selection.primary,
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

                    if t.input_box && self.touch_manager.selection.just_deselected {
                        if this_text(
                            &self.touch_manager.selection.primary,
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

    /// Updates existing handles and outlines based on their parent elements' positions and sizes
    fn update_selection_visuals(menus: &HashMap<String, Menu>, editor_layer: &mut RuntimeLayer) {
        for element in &mut editor_layer.elements {
            match element {
                UiElement::Handle(h) => {
                    if let Some(ref parent) = h.parent {
                        if let Some(menu) = menus.get(&parent.menu) {
                            if let Some(layer) = menu.layers.iter().find(|l| l.name == parent.layer)
                            {
                                if let Some(c) = layer.iter_circles().find(|c| c.id == parent.id) {
                                    h.x = c.x;
                                    h.y = c.y;
                                    h.radius = c.radius;
                                }
                            }
                        }
                    }
                }
                UiElement::Outline(o) => {
                    if let Some(ref parent) = o.parent {
                        if let Some(menu) = menus.get(&parent.menu) {
                            if let Some(layer) = menu.layers.iter().find(|l| l.name == parent.layer)
                            {
                                match parent.kind {
                                    ElementKind::Circle => {
                                        if let Some(c) =
                                            layer.iter_circles().find(|c| c.id == parent.id)
                                        {
                                            o.shape_data.x = c.x;
                                            o.shape_data.y = c.y;
                                            o.shape_data.radius = c.radius;
                                            o.misc = c.misc.clone();
                                        }
                                    }
                                    ElementKind::Polygon => {
                                        if let Some(p) =
                                            layer.iter_polygons().find(|p| p.id == parent.id)
                                        {
                                            let mut cx = 0.0;
                                            let mut cy = 0.0;
                                            for v in &p.vertices {
                                                cx += v.pos[0];
                                                cy += v.pos[1];
                                            }
                                            if !p.vertices.is_empty() {
                                                cx /= p.vertices.len() as f32;
                                                cy /= p.vertices.len() as f32;
                                            }

                                            let mut radius: f32 = 0.0;
                                            for v in &p.vertices {
                                                let dx = v.pos[0] - cx;
                                                let dy = v.pos[1] - cy;
                                                radius = radius.max((dx * dx + dy * dy).sqrt());
                                            }

                                            o.shape_data.x = cx;
                                            o.shape_data.y = cy;
                                            o.shape_data.radius = radius;
                                            o.vertex_count = p.vertices.len() as u32;
                                            o.misc = p.misc.clone();
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Creates selection visuals (outlines and handles) for the selected element
    fn create_selection_visuals(
        editor_layer: &mut RuntimeLayer,
        element: &UiElement,
        sel: &ElementRef,
        editor_mode: bool,
    ) {
        match element {
            UiElement::Circle(c) => {
                if editor_mode && c.misc.editable {
                    let circle_outline = UiButtonOutline {
                        id: "Circle Outline".to_string(),
                        parent: Some(ElementRef {
                            menu: sel.menu.clone(),
                            layer: sel.layer.clone(),
                            id: c.id.clone(),
                            kind: ElementKind::Circle,
                        }),
                        mode: 0.0,
                        vertex_offset: 0,
                        vertex_count: 0,
                        dash_color: [0.2, 0.0, 0.4, 0.8],
                        dash_misc: DashMisc {
                            dash_len: 0.12,
                            dash_spacing: 0.04,
                            dash_roundness: 1.0,
                            dash_speed: 0.15,
                        },
                        sub_dash_color: [0.8, 1.0, 1.0, 0.5],
                        sub_dash_misc: DashMisc {
                            dash_len: 0.12,
                            dash_spacing: 0.08,
                            dash_roundness: 1.0,
                            dash_speed: -0.25,
                        },
                        misc: c.misc.clone(),
                        shape_data: ShapeData {
                            x: c.x,
                            y: c.y,
                            radius: c.radius,
                            border_thickness: 0.08,
                        },
                    };
                    editor_layer
                        .elements
                        .push(UiElement::Outline(circle_outline));

                    let handle = UiButtonHandle {
                        id: "Circle Handle".to_string(),
                        parent: Some(ElementRef {
                            menu: sel.menu.clone(),
                            layer: sel.layer.clone(),
                            id: c.id.clone(),
                            kind: ElementKind::Circle,
                        }),
                        x: c.x,
                        y: c.y,
                        radius: c.radius,
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
                    editor_layer.elements.push(UiElement::Handle(handle));
                }
            }
            UiElement::Polygon(p) => {
                if editor_mode && p.misc.editable {
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
                        radius = radius.max((dx * dx + dy * dy).sqrt());

                        let vertex_outline = UiButtonCircle {
                            id: format!("vertex_outline_{}", i),
                            action: "None".to_string(),
                            style: "None".to_string(),
                            x: v.pos[0],
                            y: v.pos[1],
                            radius: 10.0,
                            inside_border_thickness_percentage: 2.0,
                            border_thickness_percentage: 0.0,
                            fade: 0.0,
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
                                active: true,
                                touched_time: 0.0,
                                is_touched: false,
                                pressable: false,
                                editable: false,
                            },
                        };
                        editor_layer
                            .elements
                            .push(UiElement::Circle(vertex_outline));
                    }

                    let polygon_outline = UiButtonOutline {
                        id: "Polygon Outline".to_string(),
                        parent: Some(ElementRef {
                            menu: sel.menu.clone(),
                            layer: sel.layer.clone(),
                            id: p.id.clone(),
                            kind: ElementKind::Polygon,
                        }),
                        mode: 1.0,
                        vertex_offset: 0,
                        vertex_count: p.vertices.len() as u32,
                        dash_color: [0.2, 0.0, 0.4, 0.8],
                        dash_misc: DashMisc {
                            dash_len: 0.1,
                            dash_spacing: 0.05,
                            dash_roundness: 1.0,
                            dash_speed: 0.15,
                        },
                        sub_dash_color: [0.8, 1.0, 1.0, 0.5],
                        sub_dash_misc: DashMisc {
                            dash_len: 0.1,
                            dash_spacing: 0.1,
                            dash_roundness: 0.0,
                            dash_speed: -0.25,
                        },
                        misc: p.misc.clone(),
                        shape_data: ShapeData {
                            x: cx,
                            y: cy,
                            radius,
                            border_thickness: 0.9,
                        },
                    };
                    println!("YEES");
                    editor_layer
                        .elements
                        .push(UiElement::Outline(polygon_outline));
                }
            }
            _ => {}
        }
    }

    pub fn update_selection(&mut self) {
        if self.touch_manager.selection.just_deselected
            && !self.touch_manager.selection.just_selected
        {
            self.touch_manager.editor.editing_text = None;
        }
        let Some(sel) = &self.touch_manager.selection.primary else {
            if let Some(editor_menu) = self.menus.get_mut("Editor_Menu") {
                if let Some(editor_layer) = editor_menu
                    .layers
                    .iter_mut()
                    .find(|l| l.name == "editor_selection")
                {
                    editor_layer.active = false;
                    editor_layer.clear_circles();
                    editor_layer.clear_handles();
                    editor_layer.clear_outlines();
                    editor_layer.clear_polygons();
                    editor_layer.dirty.mark_all()
                }
            }
            return;
        };

        if let Some(menu) = self.menus.get(&sel.menu) {
            if let Some(layer) = menu.layers.iter().find(|l| l.name == sel.layer.to_string()) {
                self.variables
                    .set_i32("selected_layer.order", layer.order as i32);
            }
        }

        let element = get_element(&self.menus, &sel.menu, &sel.layer, &sel.id);

        if let Some(element) = element {
            let is_handle = element.kind() == ElementKind::Handle;
            let editor_mode = self.touch_manager.editor.enabled;

            // Temporarily remove Editor_Menu to avoid borrow conflicts
            if let Some(mut editor_menu) = self.menus.remove("Editor_Menu") {
                if let Some(editor_layer) = editor_menu
                    .layers
                    .iter_mut()
                    .find(|l| l.name == "editor_selection")
                {
                    editor_layer.active = true;
                    editor_layer.dirty.mark_all();
                    editor_layer.clear_circles();
                    editor_layer.clear_polygons();

                    if is_handle {
                        // Update existing handles/outlines from parent
                        Self::update_selection_visuals(&self.menus, editor_layer);
                    } else {
                        // Create new selection visuals
                        editor_layer.clear_handles();
                        editor_layer.clear_outlines();
                        Self::create_selection_visuals(editor_layer, &element, sel, editor_mode);
                    }
                }
                self.menus.insert("Editor_Menu".to_string(), editor_menu);
            }
        }
        // Maybe just selected if just not deselected idk hopefully not
    }
    // ========================================================================
    // UNDO/REDO OPERATIONS
    // ========================================================================

    /// Delete element by ID (internal, for undo system)
    pub fn delete_element_by_id(&mut self, menu_name: &str, layer_name: &str, element_id: &str) {
        if let Some(element) = get_element(&self.menus, menu_name, layer_name, element_id) {
            delete_element(&mut self.menus, menu_name, layer_name, &element);
        }
    }

    fn get_polygon(&self, menu: &str, layer: &str, id: &str) -> Option<UiButtonPolygon> {
        let menu = self.menus.get(menu)?;
        let layer = menu.layers.iter().find(|l| l.name == layer)?;
        layer.iter_polygons().find(|p| p.id == id).cloned()
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

    pub fn apply_ui_edit_movement(&mut self, input_state: &mut InputState) {
        let Some(sel) = &self.touch_manager.selection.primary else {
            return;
        };
        let mut changed = false;

        // ============================================================
        // BLOCK 1: Element XY movement, resizing, z-index movement
        // ============================================================
        {
            let menu = match self.menus.get_mut(&sel.menu) {
                Some(m) => m,
                None => return,
            };

            let layer = match menu.layers.iter_mut().find(|l| l.name == sel.layer) {
                Some(l) => l,
                None => return,
            };

            // Movement (WASD)
            let speed = calc_move_speed(input_state);
            let mut dx = 0.0;
            let mut dy = 0.0;

            if input_state.gameplay_down("Move Element Left") {
                dx -= speed;
            }
            if input_state.gameplay_down("Move Element Right") {
                dx += speed;
            }
            if input_state.gameplay_down("Move Element Up") {
                dy -= speed;
            }
            if input_state.gameplay_down("Move Element Down") {
                dy += speed;
            }

            if dx != 0.0 || dy != 0.0 {
                layer.bump_element_xy(&sel.id, dx, dy);
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
                layer.resize_element(&sel.id, scale);
                layer.dirty.mark_all();
                changed = true;
            }

            // Element Z movement
            if !input_state.shift {
                if input_state.action_repeat("Move Element Z Up") {
                    layer.bump_element_z(&sel.id, 1);
                    layer.dirty.mark_all();
                    changed = true;
                }

                if input_state.action_repeat("Move Element Z Down") {
                    layer.bump_element_z(&sel.id, -1);
                    layer.dirty.mark_all();
                    changed = true;
                }
            }
        }

        // ============================================================
        // BLOCK 2: LAYER ORDERING
        // ============================================================
        {
            let menu = match self.menus.get_mut(&sel.menu) {
                Some(m) => m,
                None => return,
            };

            if !input_state.shift {
                if input_state.action_repeat("Move Layer Up") {
                    menu.bump_layer_order(&sel.layer, 1, &mut self.variables);
                    menu.sort_layers();
                    changed = true;
                }

                if input_state.action_repeat("Move Layer Down") {
                    menu.bump_layer_order(&sel.layer, -1, &mut self.variables);
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
            elements: vec![],
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
            elements: vec![],
            dirty: LayerDirty::all(),
            gpu: LayerGpu::default(),
            opaque: true,
            saveable: false,
        });

        menu.sort_layers();
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
fn this_text(selected: &Option<ElementRef>, menu: &str, layer: &str, element_id: String) -> bool {
    let Some(sel) = selected else { return false };
    if sel.menu != menu {
        return false;
    }
    if sel.layer != layer {
        return false;
    }

    sel.id == element_id
}

pub fn get_element(
    menus: &HashMap<String, Menu>,
    menu_name: &str,
    layer_name: &str,
    element_id: &str,
) -> Option<UiElement> {
    let menu = menus.get(menu_name)?;
    let layer = menu.layers.iter().find(|l| l.name == layer_name)?;

    if let Some(c) = layer.iter_circles().find(|c| c.id == element_id) {
        return Some(UiElement::Circle(c.clone()));
    }
    if let Some(t) = layer.iter_texts().find(|t| t.id == element_id) {
        return Some(UiElement::Text(t.clone()));
    }
    if let Some(p) = layer.iter_polygons().find(|p| p.id == element_id) {
        return Some(UiElement::Polygon(p.clone()));
    }
    if let Some(h) = layer.iter_handles().find(|h| h.id == element_id) {
        return Some(UiElement::Handle(h.clone()));
    }
    if let Some(o) = layer.iter_outlines().find(|o| o.id == element_id) {
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
        ElementKind::Circle => layer.iter_circles().find(|c| c.id == id).map(|c| c.radius),
        ElementKind::Text => layer.iter_texts().find(|t| t.id == id).map(|t| t.px as f32),
        ElementKind::Handle => layer.iter_handles().find(|h| h.id == id).map(|h| h.radius),
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
        .iter_polygons() // mutable iterator
        .find(|p| p.id == id)
        .map(|p| p.vertices.iter().map(|v| v.pos).collect())
}
