use crate::resources::{InputState, TimeSystem};
use crate::ui::actions::selection_needed;
use crate::ui::helper::{dist, polygon_sdf};
use crate::ui::input::MouseState;
use crate::ui::menu::Menu;
use crate::ui::selections::{
    SelectedUiElement, select_move_primary_to_multi, select_to_multi, select_ui_element,
};
use crate::ui::ui_edit_manager::{
    MoveElementCommand, MoveVertexCommand, ResizeElementCommand, TextEditCommand, UiEditManager,
};
use crate::ui::ui_editor::{DragStartState, UiButtonLoader, get_element, get_element_size};
use crate::ui::ui_runtime::UiRuntime;
use crate::ui::variables::UiVariableRegistry;
use crate::ui::vertex::{
    ElementKind, LayerDirty, RuntimeLayer, TouchState, UiButtonCircle, UiButtonHandle,
    UiButtonPolygon, UiButtonText,
};
use std::collections::{HashMap, HashSet};
// ============================================================================
// TYPES
// ============================================================================

#[derive(Clone, Copy)]
pub(crate) struct MouseSnapshot {
    pub mx: f32,
    pub my: f32,
    pub pressed: bool,
    pub just_pressed: bool,
    pub scroll: f32,
}

impl MouseSnapshot {
    pub fn from_mouse(mouse: &MouseState) -> Self {
        Self {
            mx: mouse.pos.x,
            my: mouse.pos.y,
            pressed: mouse.left_pressed,
            just_pressed: mouse.left_just_pressed,
            scroll: mouse.scroll_delta.y,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum HitElement {
    Circle(usize),
    Handle(usize),
    Polygon(usize),
    Text(usize),
}

#[derive(Clone, Debug)]
pub(crate) struct HitResult {
    pub menu_name: String,
    pub layer_name: String,
    pub element: HitElement,
    pub layer_order: u32,
    pub action: Option<String>,
    element_order: usize,
}

impl HitResult {
    pub fn matches(&self, menu_name: &str, layer_name: &str, element: HitElement) -> bool {
        self.menu_name == menu_name && self.layer_name == layer_name && self.element == element
    }
}

#[derive(Clone, Debug)]
pub struct PendingDragStart {
    pub menu: String,
    pub layer: String,
    pub element_id: String,
    pub element_kind: ElementKind,
    pub start_pos: (f32, f32),
    pub start_size: Option<f32>,
}

#[derive(Default)]
pub(crate) struct EditorInteractionResult {
    pub trigger_selection: bool,
    pub pending_circle_updates: Vec<(String, f32, f32)>,
    pub moved_any_selected_object: bool,
    pub pending_drag_start: Option<PendingDragStart>,
    pub pending_move_commands: Vec<MoveElementCommand>,
    pub pending_vertex_commands: Vec<MoveVertexCommand>,
    pub pending_resize_commands: Vec<ResizeElementCommand>,
}

#[derive(Copy, Clone, PartialEq)]
enum Direction {
    Left,
    Right,
    Up,
    Down,
}

// ============================================================================
// COMMON HELPERS
// ============================================================================

fn active_menus(menus: &mut HashMap<String, Menu>) -> impl Iterator<Item = (&String, &mut Menu)> {
    menus.iter_mut().filter(|(_, menu)| menu.active)
}

fn active_layers(layers: &mut Vec<RuntimeLayer>) -> impl Iterator<Item = &mut RuntimeLayer> {
    layers.iter_mut().filter(|layer| layer.active)
}

fn active_saveable_layers(
    layers: &mut Vec<RuntimeLayer>,
) -> impl Iterator<Item = &mut RuntimeLayer> {
    layers
        .iter_mut()
        .filter(|layer| layer.active && layer.saveable)
}

fn compute_element_hit(
    top_hit: Option<&HitResult>,
    menu_name: &str,
    layer_name: &str,
    element: HitElement,
) -> bool {
    top_hit
        .map(|hit| hit.matches(menu_name, layer_name, element))
        .unwrap_or(false)
}

fn compute_element_selected(
    ui_runtime: &UiRuntime,
    menu_name: &str,
    layer_name: &str,
    id: &str,
) -> bool {
    let sel = &ui_runtime.selected_ui_element_primary;
    sel.active && sel.menu_name == menu_name && sel.layer_name == layer_name && sel.element_id == id
}

fn compute_touched_now(is_down: bool, is_hit: bool, mouse: &MouseSnapshot) -> bool {
    if !is_down {
        is_hit && mouse.just_pressed
    } else {
        mouse.pressed
    }
}

fn should_skip_processing(is_down: bool, is_hit: bool, is_selected: bool) -> bool {
    (!is_down && !is_hit) || (is_down && !is_selected)
}

fn create_selection(
    menu_name: &str,
    layer_name: &str,
    element_id: &str,
    element_type: ElementKind,
    action_name: String,
    input_box: bool,
) -> SelectedUiElement {
    SelectedUiElement {
        active: true,
        menu_name: menu_name.to_string(),
        layer_name: layer_name.to_string(),
        element_id: element_id.to_string(),
        just_deselected: false,
        dragging: false,
        element_type,
        just_selected: true,
        action_name,
        input_box,
    }
}

pub fn select_correctly(
    pending_selection: Option<SelectedUiElement>,
    loader: &mut UiButtonLoader,
    input_state: &InputState,
) {
    if let Some(p) = pending_selection {
        if selection_needed(loader, p.action_name.as_str()) {
            select_move_primary_to_multi(loader, p)
        } else if input_state.ctrl {
            select_to_multi(loader, p);
        } else {
            select_ui_element(loader, p);
        }
    }
}

// ============================================================================
// HIT DETECTION - BASIC
// ============================================================================

fn hit_circle(mx: f32, my: f32, circle: &UiButtonCircle, radius: f32) -> bool {
    let dx = mx - circle.x;
    let dy = my - circle.y;
    dx * dx + dy * dy <= radius * radius
}

fn hit_handle(mx: f32, my: f32, handle: &UiButtonHandle) -> bool {
    let dx = mx - handle.x;
    let dy = my - handle.y;
    let dist2 = dx * dx + dy * dy;

    let width_ratio = handle.handle_misc.handle_width;
    let half_thick = 0.5 * handle.radius * width_ratio;
    let inner = handle.radius - half_thick;
    let outer = handle.radius + half_thick;
    let margin = (handle.radius * 0.15).max(10.0);
    let inner_grab = (inner - margin).max(0.0);
    let outer_grab = outer + margin;

    dist2 >= inner_grab * inner_grab && dist2 <= outer_grab * outer_grab
}

fn hit_polygon(mx: f32, my: f32, poly: &UiButtonPolygon) -> bool {
    let verts = &poly.vertices;
    if verts.is_empty() {
        return false;
    }

    const VERTEX_RADIUS: f32 = 20.0;
    let vertex_hit = verts
        .iter()
        .any(|v| dist(mx, my, v.pos[0], v.pos[1]) < VERTEX_RADIUS);

    if vertex_hit {
        return true;
    }

    let sdf = polygon_sdf(mx, my, verts);
    let inside = sdf < 0.0;
    let near_edge = sdf.abs() < 8.0;
    inside || near_edge
}

fn hit_text(text: &UiButtonText, mx: f32, my: f32) -> bool {
    let x0 = text.x + text.top_left_offset[0];
    let y0 = text.y + text.top_left_offset[1];
    let x1 = x0 + text.natural_width + text.top_right_offset[0];
    let y1 = y0 + text.natural_height + text.bottom_left_offset[1];

    mx >= x0 && mx <= x1 && my >= y0 && my <= y1
}

// ============================================================================
// HIT DETECTION - LAYER LEVEL
// ============================================================================

fn circle_hit(layer: &RuntimeLayer, mx: f32, my: f32) -> (bool, String) {
    for circle in &layer.circles {
        if !circle.misc.active {
            continue;
        }
        if hit_circle(mx, my, circle, circle.radius) {
            return (true, circle.action.clone());
        }
    }
    (false, "None".to_string())
}

fn polygon_hit(layer: &RuntimeLayer, mx: f32, my: f32) -> (bool, String) {
    for poly in &layer.polygons {
        if !poly.misc.active {
            continue;
        }
        if hit_polygon(mx, my, poly) {
            return (true, poly.action.clone());
        }
    }
    (false, "None".to_string())
}

fn handle_hit(layer: &RuntimeLayer, mx: f32, my: f32) -> (bool, String) {
    for handle in &layer.handles {
        if !handle.misc.active {
            continue;
        }
        if hit_handle(mx, my, handle) {
            return (true, "None".to_string());
        }
    }
    (false, "None".to_string())
}

// ============================================================================
// PUBLIC HIT DETECTION
// ============================================================================

pub(crate) fn press_began_on_ui(
    menus: &HashMap<String, Menu>,
    mouse: &MouseSnapshot,
    editor_mode: bool,
) -> (bool, String) {
    for (_, menu) in menus.iter().filter(|(_, menu)| menu.active) {
        for layer in menu.layers.iter().filter(|l| l.active) {
            if let result @ (true, _) = circle_hit(layer, mouse.mx, mouse.my) {
                return result;
            }
            if let result @ (true, _) = polygon_hit(layer, mouse.mx, mouse.my) {
                return result;
            }
            if editor_mode {
                if let result @ (true, _) = handle_hit(layer, mouse.mx, mouse.my) {
                    return result;
                }
            }
            for text in layer.texts.iter().filter(|t| t.misc.active) {
                if hit_text(text, mouse.mx, mouse.my) {
                    return (true, text.action.clone());
                }
            }
        }
    }
    (false, "None".to_string())
}

pub(crate) fn near_handle(menus: &HashMap<String, Menu>, mouse: &MouseSnapshot) -> bool {
    menus
        .iter()
        .filter(|(_, menu)| menu.active)
        .flat_map(|(_, menu)| menu.layers.iter().filter(|l| l.active))
        .any(|layer| {
            layer.handles.iter().any(|h| {
                if !h.misc.active {
                    return false;
                }
                let dx = mouse.mx - h.x;
                let dy = mouse.my - h.y;
                let dist = (dx * dx + dy * dy).sqrt();
                let margin = (h.radius * 0.2).max(12.0);
                (dist - h.radius).abs() < margin
            })
        })
}

// ============================================================================
// FIND TOP HIT
// ============================================================================

pub(crate) fn find_top_hit(
    menus: &mut HashMap<String, Menu>,
    mouse: &MouseSnapshot,
    editor_mode: bool,
) -> Option<HitResult> {
    let mut best: Option<HitResult> = None;

    for (menu_name, menu) in menus.iter_mut().filter(|(_, menu)| menu.active) {
        for layer in menu.layers.iter_mut().filter(|l| l.active) {
            find_top_hit_circles(&mut best, menu_name, layer, mouse, editor_mode);

            if editor_mode {
                find_top_hit_handles(&mut best, menu_name, layer, mouse);
            }

            find_top_hit_polygons(&mut best, menu_name, layer, mouse);
            find_top_hit_texts(&mut best, menu_name, layer, mouse);
        }
    }

    best
}

fn find_top_hit_circles(
    best: &mut Option<HitResult>,
    menu_name: &str,
    layer: &RuntimeLayer,
    mouse: &MouseSnapshot,
    editor_mode: bool,
) {
    for (circle_index, circle) in layer.circles.iter().enumerate() {
        if !circle.misc.active || (!circle.misc.pressable && !circle.misc.editable) {
            continue;
        }

        let drag_radius = if editor_mode {
            (circle.radius * 0.9).max(8.0)
        } else {
            circle.radius
        };

        if hit_circle(mouse.mx, mouse.my, circle, drag_radius) {
            consider_candidate(
                best,
                HitResult {
                    menu_name: menu_name.to_string(),
                    layer_name: layer.name.clone(),
                    element: HitElement::Circle(circle_index),
                    layer_order: layer.order,
                    element_order: circle_index,
                    action: Some(circle.action.clone()),
                },
            );
        }
    }
}

fn find_top_hit_handles(
    best: &mut Option<HitResult>,
    menu_name: &str,
    layer: &RuntimeLayer,
    mouse: &MouseSnapshot,
) {
    for (handle_index, handle) in layer.handles.iter().enumerate() {
        if !handle.misc.active || (!handle.misc.pressable && !handle.misc.editable) {
            continue;
        }

        if hit_handle(mouse.mx, mouse.my, handle) {
            consider_candidate(
                best,
                HitResult {
                    menu_name: menu_name.to_string(),
                    layer_name: layer.name.clone(),
                    element: HitElement::Handle(handle_index),
                    element_order: handle_index,
                    layer_order: layer.order,
                    action: None,
                },
            );
        }
    }
}

fn find_top_hit_polygons(
    best: &mut Option<HitResult>,
    menu_name: &str,
    layer: &RuntimeLayer,
    mouse: &MouseSnapshot,
) {
    for (poly_index, poly) in layer.polygons.iter().enumerate() {
        if !poly.misc.active || (!poly.misc.pressable && !poly.misc.editable) {
            continue;
        }

        if hit_polygon(mouse.mx, mouse.my, poly) {
            consider_candidate(
                best,
                HitResult {
                    menu_name: menu_name.to_string(),
                    layer_name: layer.name.clone(),
                    element: HitElement::Polygon(poly_index),
                    layer_order: layer.order,
                    element_order: poly_index,
                    action: Some(poly.action.clone()),
                },
            );
        }
    }
}

fn find_top_hit_texts(
    best: &mut Option<HitResult>,
    menu_name: &str,
    layer: &mut RuntimeLayer,
    mouse: &MouseSnapshot,
) {
    for (text_index, text) in layer.texts.iter_mut().enumerate() {
        if !text.misc.active || (!text.misc.pressable && !text.misc.editable) {
            continue;
        }

        text.being_hovered = false;

        if hit_text(text, mouse.mx, mouse.my) {
            text.being_hovered = true;
            text.just_unhovered = false;

            consider_candidate(
                best,
                HitResult {
                    menu_name: menu_name.to_string(),
                    layer_name: layer.name.clone(),
                    element: HitElement::Text(text_index),
                    layer_order: layer.order,
                    element_order: text_index,
                    action: Some(text.action.clone()),
                },
            );
        }
    }
}

fn consider_candidate(best: &mut Option<HitResult>, candidate: HitResult) {
    let candidate_key = (candidate.layer_order, candidate.element_order);

    match best {
        Some(current) => {
            let current_key = (current.layer_order, current.element_order);
            if candidate_key > current_key {
                *current = candidate;
            }
        }
        None => {
            *best = Some(candidate);
        }
    }
}

// ============================================================================
// EDITOR INTERACTIONS - MAIN
// ============================================================================

pub(crate) fn handle_editor_mode_interactions(
    loader: &mut UiButtonLoader,
    time_system: &TimeSystem,
    mouse: &MouseSnapshot,
    top_hit: Option<HitResult>,
    input_state: &mut InputState,
) -> EditorInteractionResult {
    let mut result = EditorInteractionResult::default();
    let top_hit_ref = top_hit.as_ref();

    process_circles(
        loader,
        time_system.sim_dt,
        mouse,
        top_hit_ref,
        &mut result,
        input_state,
    );
    process_handles(
        loader,
        time_system.sim_dt,
        mouse,
        top_hit_ref,
        &mut result,
        input_state,
    );
    process_polygons(
        loader,
        time_system.sim_dt,
        mouse,
        top_hit_ref,
        &mut result,
        input_state,
    );
    process_text(
        loader,
        time_system,
        mouse,
        top_hit_ref,
        &mut result,
        input_state,
    );

    if loader.ui_runtime.editor_mode {
        process_keyboard_ui_navigation(loader, input_state);
    }

    result
}

// ============================================================================
// PROCESS CIRCLES
// ============================================================================

fn process_circles(
    loader: &mut UiButtonLoader,
    dt: f32,
    mouse: &MouseSnapshot,
    top_hit: Option<&HitResult>,
    result: &mut EditorInteractionResult,
    input_state: &InputState,
) {
    let mut pending_selection: Option<SelectedUiElement> = None;
    let mut pending_drag: Option<PendingDragStart> = None;

    for (menu_name, menu) in active_menus(&mut loader.menus) {
        for layer in active_saveable_layers(&mut menu.layers) {
            process_circles_layer(
                &mut loader.ui_runtime,
                dt,
                mouse,
                top_hit,
                result,
                menu_name,
                layer,
                &mut pending_selection,
                &mut pending_drag,
            );
        }
    }

    // Push pending move commands after iteration (coalescing handles rapid updates)
    for cmd in result.pending_move_commands.drain(..) {
        loader.ui_edit_manager.push_command(cmd);
    }

    apply_pending_drag(loader, pending_drag);
    select_correctly(pending_selection, loader, input_state);
}
fn process_circles_layer(
    ui_runtime: &mut UiRuntime,
    dt: f32,
    mouse: &MouseSnapshot,
    top_hit: Option<&HitResult>,
    result: &mut EditorInteractionResult,
    menu_name: &str,
    layer: &mut RuntimeLayer,
    pending_selection: &mut Option<SelectedUiElement>,
    pending_drag: &mut Option<PendingDragStart>,
) {
    for (circle_index, circle) in layer.circles.iter_mut().enumerate() {
        process_single_circle(
            ui_runtime,
            dt,
            mouse,
            top_hit,
            result,
            menu_name,
            &mut layer.dirty,
            &layer.name,
            circle_index,
            circle,
            pending_selection,
            pending_drag,
        );
    }
}

fn process_single_circle(
    ui_runtime: &mut UiRuntime,
    dt: f32,
    mouse: &MouseSnapshot,
    top_hit: Option<&HitResult>,
    result: &mut EditorInteractionResult,
    menu_name: &str,
    dirty: &mut LayerDirty,
    layer_name: &str,
    circle_index: usize,
    circle: &mut UiButtonCircle,
    pending_selection: &mut Option<SelectedUiElement>,
    pending_drag: &mut Option<PendingDragStart>,
) {
    let id = match get_circle_id(circle) {
        Some(id) => id,
        None => return,
    };

    let is_down = ui_runtime.get(&id).is_down;
    let is_hit = compute_element_hit(
        top_hit,
        menu_name,
        layer_name,
        HitElement::Circle(circle_index),
    );
    let is_selected = compute_element_selected(ui_runtime, menu_name, layer_name, &id);

    if should_skip_processing(is_down, is_hit, is_selected) {
        return;
    }

    let touched_now = compute_touched_now(is_down, is_hit, mouse);
    let state = ui_runtime.update_touch(&id, touched_now, dt, &layer_name.to_string());

    handle_circle_touch_state(
        ui_runtime,
        mouse,
        result,
        menu_name,
        layer_name,
        dirty,
        circle,
        &id,
        state,
        pending_selection,
        pending_drag,
    );
}

fn get_circle_id(circle: &UiButtonCircle) -> Option<String> {
    if !circle.misc.active {
        return None;
    }
    circle.id.clone()
}

fn handle_circle_touch_state(
    ui_runtime: &mut UiRuntime,
    mouse: &MouseSnapshot,
    result: &mut EditorInteractionResult,
    menu_name: &str,
    layer_name: &str,
    dirty: &mut LayerDirty,
    circle: &mut UiButtonCircle,
    id: &str,
    state: TouchState,
    pending_selection: &mut Option<SelectedUiElement>,
    pending_drag: &mut Option<PendingDragStart>,
) {
    match state {
        TouchState::Pressed => {
            handle_circle_pressed(
                ui_runtime,
                mouse,
                menu_name,
                layer_name,
                dirty,
                circle,
                id,
                pending_selection,
                pending_drag,
            );
        }
        TouchState::Held => {
            handle_circle_held(
                ui_runtime, mouse, result, dirty, circle, menu_name, layer_name,
            );
        }
        TouchState::Released => {
            handle_circle_released(ui_runtime, dirty, result);
        }
        TouchState::Idle => {}
    }
}

fn handle_circle_pressed(
    ui_runtime: &mut UiRuntime,
    mouse: &MouseSnapshot,
    menu_name: &str,
    layer_name: &str,
    dirty: &mut LayerDirty,
    circle: &UiButtonCircle,
    id: &str,
    pending_selection: &mut Option<SelectedUiElement>,
    pending_drag: &mut Option<PendingDragStart>,
) {
    if ui_runtime.editor_mode {
        ui_runtime.drag_offset = Some((mouse.mx - circle.x, mouse.my - circle.y));

        // Collect drag info instead of calling loader.begin_drag
        *pending_drag = Some(PendingDragStart {
            menu: menu_name.to_string(),
            layer: layer_name.to_string(),
            element_id: id.to_string(),
            element_kind: ElementKind::Circle,
            start_pos: (circle.x, circle.y),
            start_size: Some(circle.radius),
        });
    }

    *pending_selection = Some(create_selection(
        menu_name,
        layer_name,
        id,
        ElementKind::Circle,
        circle.action.clone(),
        false,
    ));

    dirty.mark_circles();
}

fn handle_circle_held(
    ui_runtime: &mut UiRuntime,
    mouse: &MouseSnapshot,
    result: &mut EditorInteractionResult,
    dirty: &mut LayerDirty,
    circle: &mut UiButtonCircle,
    menu_name: &str,
    layer_name: &str,
) {
    ui_runtime.selected_ui_element_primary.dragging = true;

    if !ui_runtime.editor_mode || !circle.misc.editable {
        return;
    }

    if let Some((ox, oy)) = ui_runtime.drag_offset {
        let new_x = mouse.mx - ox;
        let new_y = mouse.my - oy;

        if (new_x - circle.x).abs() > 0.001 || (new_y - circle.y).abs() > 0.001 {
            let id = circle.id.clone().unwrap_or_default();

            // Push command for undo (coalescing will merge rapid updates)
            result.pending_move_commands.push(MoveElementCommand {
                menu: menu_name.to_string(),
                layer: layer_name.to_string(),
                element_id: id,
                element_kind: ElementKind::Circle,
                before: (circle.x, circle.y),
                after: (new_x, new_y),
            });

            // Visual update
            circle.x = new_x;
            circle.y = new_y;
            dirty.mark_circles();
            result.moved_any_selected_object = true;
        }
    }
}

fn handle_circle_released(
    ui_runtime: &mut UiRuntime,
    dirty: &mut LayerDirty,
    result: &mut EditorInteractionResult,
) {
    ui_runtime.selected_ui_element_primary.dragging = false;
    ui_runtime.drag_offset = None;
    dirty.mark_circles();

    // Signal that drag ended - will be handled in handle_touches
    result.trigger_selection = true;
}

// ============================================================================
// PROCESS HANDLES - COMPLETE REWRITE
// ============================================================================

fn process_handles(
    loader: &mut UiButtonLoader,
    dt: f32,
    mouse: &MouseSnapshot,
    top_hit: Option<&HitResult>,
    result: &mut EditorInteractionResult,
    input_state: &InputState,
) {
    let mut pending_selection: Option<SelectedUiElement> = None;
    let mut pending_drag: Option<PendingDragStart> = None;

    for (menu_name, menu) in active_menus(&mut loader.menus) {
        for layer in active_layers(&mut menu.layers) {
            process_handles_layer(
                &mut loader.ui_runtime,
                dt,
                mouse,
                top_hit,
                result,
                menu_name,
                layer,
                &mut pending_selection,
                &mut pending_drag,
            );
        }
    }

    // Apply pending drag after iteration
    apply_pending_drag(loader, pending_drag);

    select_correctly(pending_selection, loader, input_state);
}

fn apply_pending_drag(loader: &mut UiButtonLoader, pending_drag: Option<PendingDragStart>) {
    if let Some(drag) = pending_drag {
        if loader.drag_start_state.is_none() {
            loader.drag_start_state = Some(DragStartState {
                menu: drag.menu,
                layer: drag.layer,
                element_id: drag.element_id,
                element_kind: drag.element_kind,
                start_pos: drag.start_pos,
                start_size: drag.start_size,
                start_vertices: None,
            });
        }
    }
}

fn process_handles_layer(
    ui_runtime: &mut UiRuntime,
    dt: f32,
    mouse: &MouseSnapshot,
    top_hit: Option<&HitResult>,
    result: &mut EditorInteractionResult,
    menu_name: &str,
    layer: &mut RuntimeLayer,
    pending_selection: &mut Option<SelectedUiElement>,
    pending_drag: &mut Option<PendingDragStart>,
) {
    for (handle_index, handle) in layer.handles.iter_mut().enumerate() {
        process_single_handle(
            ui_runtime,
            dt,
            mouse,
            top_hit,
            result,
            menu_name,
            &mut layer.dirty,
            &layer.name,
            handle_index,
            handle,
            pending_selection,
            pending_drag,
        );
    }
}

fn process_single_handle(
    ui_runtime: &mut UiRuntime,
    dt: f32,
    mouse: &MouseSnapshot,
    top_hit: Option<&HitResult>,
    result: &mut EditorInteractionResult,
    menu_name: &str,
    dirty: &mut LayerDirty,
    layer_name: &str,
    handle_index: usize,
    handle: &mut UiButtonHandle,
    pending_selection: &mut Option<SelectedUiElement>,
    pending_drag: &mut Option<PendingDragStart>,
) {
    let id = match get_handle_id(handle) {
        Some(id) => id,
        None => return,
    };

    let is_down = ui_runtime.get(&id).is_down;
    let is_hit = compute_element_hit(
        top_hit,
        menu_name,
        layer_name,
        HitElement::Handle(handle_index),
    );
    let is_selected = compute_element_selected(ui_runtime, menu_name, layer_name, &id);

    if should_skip_processing(is_down, is_hit, is_selected) {
        return;
    }

    let touched_now = compute_touched_now(is_down, is_hit, mouse);
    let state = ui_runtime.update_touch(&id, touched_now, dt, &layer_name.to_string());

    handle_handle_touch_state(
        ui_runtime,
        mouse,
        result,
        menu_name,
        layer_name,
        dirty,
        handle,
        &id,
        state,
        pending_selection,
        pending_drag,
    );
}

fn get_handle_id(handle: &UiButtonHandle) -> Option<String> {
    if !(handle.misc.active || handle.misc.pressable) {
        return None;
    }
    handle.id.clone()
}

fn handle_handle_touch_state(
    ui_runtime: &mut UiRuntime,
    mouse: &MouseSnapshot,
    result: &mut EditorInteractionResult,
    menu_name: &str,
    layer_name: &str,
    dirty: &mut LayerDirty,
    handle: &UiButtonHandle,
    id: &str,
    state: TouchState,
    pending_selection: &mut Option<SelectedUiElement>,
    pending_drag: &mut Option<PendingDragStart>,
) {
    match state {
        TouchState::Pressed => {
            handle_handle_pressed(
                ui_runtime,
                mouse,
                menu_name,
                layer_name,
                dirty,
                handle,
                id,
                pending_selection,
                pending_drag,
            );
        }
        TouchState::Held => {
            handle_handle_held(ui_runtime, mouse, result, handle);
        }
        TouchState::Released => {
            handle_handle_released(ui_runtime, dirty, result);
        }
        TouchState::Idle => {}
    }
}

fn handle_handle_pressed(
    ui_runtime: &mut UiRuntime,
    mouse: &MouseSnapshot,
    menu_name: &str,
    layer_name: &str,
    dirty: &mut LayerDirty,
    handle: &UiButtonHandle,
    id: &str,
    pending_selection: &mut Option<SelectedUiElement>,
    pending_drag: &mut Option<PendingDragStart>,
) {
    if ui_runtime.editor_mode {
        ui_runtime.drag_offset = Some((mouse.mx - handle.x, mouse.my - handle.y));

        *pending_drag = Some(PendingDragStart {
            menu: menu_name.to_string(),
            layer: layer_name.to_string(),
            element_id: id.to_string(),
            element_kind: ElementKind::Handle,
            start_pos: (handle.x, handle.y),
            start_size: Some(handle.radius),
        });
    }

    *pending_selection = Some(create_selection(
        menu_name,
        layer_name,
        id,
        ElementKind::Handle,
        "None".to_string(),
        false,
    ));

    dirty.mark_handles();
}

fn handle_handle_held(
    ui_runtime: &mut UiRuntime,
    mouse: &MouseSnapshot,
    result: &mut EditorInteractionResult,
    handle: &UiButtonHandle,
) {
    ui_runtime.selected_ui_element_primary.dragging = true;

    if let Some(parent_id) = &handle.parent_id {
        if ui_runtime.editor_mode {
            result
                .pending_circle_updates
                .push((parent_id.clone(), mouse.mx, mouse.my));
        }
    }
}

fn handle_handle_released(
    ui_runtime: &mut UiRuntime,
    dirty: &mut LayerDirty,
    result: &mut EditorInteractionResult,
) {
    ui_runtime.selected_ui_element_primary.dragging = false;
    ui_runtime.drag_offset = None;
    dirty.mark_handles();
    result.trigger_selection = true;
}

// ============================================================================
// PROCESS POLYGONS - COMPLETE REWRITE
// ============================================================================

fn process_polygons(
    loader: &mut UiButtonLoader,
    dt: f32,
    mouse: &MouseSnapshot,
    top_hit: Option<&HitResult>,
    result: &mut EditorInteractionResult,
    input_state: &InputState,
) {
    let mut pending_selection: Option<SelectedUiElement> = None;
    let mut pending_drag: Option<PendingDragStart> = None;
    let mut pending_vertices: Option<Vec<[f32; 2]>> = None;

    for (menu_name, menu) in active_menus(&mut loader.menus) {
        for layer in active_saveable_layers(&mut menu.layers) {
            process_polygons_layer(
                &mut loader.ui_runtime,
                dt,
                mouse,
                top_hit,
                result,
                menu_name,
                layer,
                &mut pending_selection,
                &mut pending_drag,
                &mut pending_vertices,
            );
        }
    }

    // Push pending commands after iteration
    for cmd in result.pending_move_commands.drain(..) {
        loader.ui_edit_manager.push_command(cmd);
    }
    for cmd in result.pending_vertex_commands.drain(..) {
        loader.ui_edit_manager.push_command(cmd);
    }

    apply_pending_drag(loader, pending_drag);
    select_correctly(pending_selection, loader, input_state);
}

fn process_polygons_layer(
    ui_runtime: &mut UiRuntime,
    dt: f32,
    mouse: &MouseSnapshot,
    top_hit: Option<&HitResult>,
    result: &mut EditorInteractionResult,
    menu_name: &str,
    layer: &mut RuntimeLayer,
    pending_selection: &mut Option<SelectedUiElement>,
    pending_drag: &mut Option<PendingDragStart>,
    pending_vertices: &mut Option<Vec<[f32; 2]>>,
) {
    for (poly_index, poly) in layer.polygons.iter_mut().enumerate() {
        process_single_polygon(
            ui_runtime,
            dt,
            mouse,
            top_hit,
            result,
            menu_name,
            &mut layer.dirty,
            &layer.name,
            poly_index,
            poly,
            pending_selection,
            pending_drag,
            pending_vertices,
        );
    }
}

fn process_single_polygon(
    ui_runtime: &mut UiRuntime,
    dt: f32,
    mouse: &MouseSnapshot,
    top_hit: Option<&HitResult>,
    result: &mut EditorInteractionResult,
    menu_name: &str,
    dirty: &mut LayerDirty,
    layer_name: &str,
    poly_index: usize,
    poly: &mut UiButtonPolygon,
    pending_selection: &mut Option<SelectedUiElement>,
    pending_drag: &mut Option<PendingDragStart>,
    pending_vertices: &mut Option<Vec<[f32; 2]>>,
) {
    let id = match get_polygon_id(poly) {
        Some(id) => id,
        None => return,
    };

    let is_down = ui_runtime.get(&id).is_down;
    let is_hit_top = compute_element_hit(
        top_hit,
        menu_name,
        layer_name,
        HitElement::Polygon(poly_index),
    );
    let is_selected = compute_element_selected(ui_runtime, menu_name, layer_name, &id);

    let (vertex_hit, poly_hit) = compute_polygon_hits(poly, mouse);
    let hit = vertex_hit.is_some() || poly_hit;

    if should_skip_polygon_processing(is_down, hit, is_hit_top, is_selected) {
        return;
    }

    let touched_now = compute_polygon_touched_now(is_down, mouse);
    let state = ui_runtime.update_touch(&id, touched_now, dt, &layer_name.to_string());

    handle_polygon_touch_state(
        ui_runtime,
        mouse,
        result,
        menu_name,
        layer_name,
        dirty,
        poly,
        &id,
        state,
        vertex_hit,
        pending_selection,
        pending_drag,
        pending_vertices,
    );
}

fn get_polygon_id(poly: &UiButtonPolygon) -> Option<String> {
    if !(poly.misc.active && !poly.vertices.is_empty()) {
        return None;
    }
    poly.id.clone()
}
const VERTEX_RADIUS: f32 = 20.0;
fn compute_polygon_hits(poly: &UiButtonPolygon, mouse: &MouseSnapshot) -> (Option<usize>, bool) {
    let verts = &poly.vertices;

    let vertex_hit = verts
        .iter()
        .enumerate()
        .find(|(_, v)| dist(mouse.mx, mouse.my, v.pos[0], v.pos[1]) < VERTEX_RADIUS)
        .map(|(i, _)| i);

    let sdf = polygon_sdf(mouse.mx, mouse.my, verts);
    let inside = sdf < 0.0;
    let near_edge = sdf.abs() < 8.0;
    let poly_hit = inside || near_edge;

    (vertex_hit, poly_hit)
}

fn should_skip_polygon_processing(
    is_down: bool,
    hit: bool,
    is_hit_top: bool,
    is_selected: bool,
) -> bool {
    (!is_down && !(hit && is_hit_top)) || (is_down && !is_selected)
}

fn compute_polygon_touched_now(is_down: bool, mouse: &MouseSnapshot) -> bool {
    if !is_down {
        mouse.just_pressed
    } else {
        mouse.pressed
    }
}

fn compute_polygon_centroid(poly: &UiButtonPolygon) -> (f32, f32) {
    let verts = &poly.vertices;
    let inv_n = 1.0 / verts.len() as f32;

    let mut cx = 0.0;
    let mut cy = 0.0;
    for v in verts.iter() {
        cx += v.pos[0];
        cy += v.pos[1];
    }

    (cx * inv_n, cy * inv_n)
}

fn handle_polygon_touch_state(
    ui_runtime: &mut UiRuntime,
    mouse: &MouseSnapshot,
    result: &mut EditorInteractionResult,
    menu_name: &str,
    layer_name: &str,
    dirty: &mut LayerDirty,
    poly: &mut UiButtonPolygon,
    id: &str,
    state: TouchState,
    vertex_hit: Option<usize>,
    pending_selection: &mut Option<SelectedUiElement>,
    pending_drag: &mut Option<PendingDragStart>,
    pending_vertices: &mut Option<Vec<[f32; 2]>>,
) {
    match state {
        TouchState::Pressed => {
            handle_polygon_pressed(
                ui_runtime,
                mouse,
                menu_name,
                layer_name,
                dirty,
                poly,
                id,
                vertex_hit,
                pending_selection,
                pending_drag,
                pending_vertices,
            );
        }
        TouchState::Held => {
            handle_polygon_held(
                ui_runtime, mouse, result, dirty, poly, menu_name, layer_name,
            );
        }
        TouchState::Released => {
            handle_polygon_released(ui_runtime, dirty, result);
        }
        TouchState::Idle => {}
    }
}

fn handle_polygon_pressed(
    ui_runtime: &mut UiRuntime,
    mouse: &MouseSnapshot,
    menu_name: &str,
    layer_name: &str,
    dirty: &mut LayerDirty,
    poly: &mut UiButtonPolygon,
    id: &str,
    vertex_hit: Option<usize>,
    pending_selection: &mut Option<SelectedUiElement>,
    pending_drag: &mut Option<PendingDragStart>,
    pending_vertices: &mut Option<Vec<[f32; 2]>>,
) {
    if ui_runtime.editor_mode && poly.misc.editable {
        setup_polygon_drag(ui_runtime, mouse, poly, vertex_hit);

        let (cx, cy) = compute_polygon_centroid(poly);

        // Collect all vertex positions for undo
        let vertices: Vec<[f32; 2]> = poly.vertices.iter().map(|v| v.pos).collect();
        *pending_vertices = Some(vertices);

        *pending_drag = Some(PendingDragStart {
            menu: menu_name.to_string(),
            layer: layer_name.to_string(),
            element_id: id.to_string(),
            element_kind: ElementKind::Polygon,
            start_pos: (cx, cy),
            start_size: None,
        });
    }

    *pending_selection = Some(create_selection(
        menu_name,
        layer_name,
        id,
        ElementKind::Polygon,
        poly.action.clone(),
        false,
    ));

    dirty.mark_polygons();
}

fn setup_polygon_drag(
    ui_runtime: &mut UiRuntime,
    mouse: &MouseSnapshot,
    poly: &UiButtonPolygon,
    vertex_hit: Option<usize>,
) {
    if let Some(vidx) = vertex_hit {
        let vx = poly.vertices[vidx].pos[0];
        let vy = poly.vertices[vidx].pos[1];
        ui_runtime.drag_offset = Some((mouse.mx - vx, mouse.my - vy));
        ui_runtime.active_vertex = Some(poly.vertices[vidx].id);
    } else {
        let (cx, cy) = compute_polygon_centroid(poly);
        ui_runtime.drag_offset = Some((mouse.mx - cx, mouse.my - cy));
        ui_runtime.active_vertex = None;
    }
}

fn handle_polygon_held(
    ui_runtime: &mut UiRuntime,
    mouse: &MouseSnapshot,
    result: &mut EditorInteractionResult,
    dirty: &mut LayerDirty,
    poly: &mut UiButtonPolygon,
    menu_name: &str,
    layer_name: &str,
) {
    ui_runtime.selected_ui_element_primary.dragging = true;

    if !ui_runtime.editor_mode || !poly.misc.editable {
        return;
    }

    if let Some(active_id) = ui_runtime.active_vertex {
        apply_vertex_drag(
            ui_runtime, mouse, result, dirty, poly, active_id, menu_name, layer_name,
        );
    } else {
        apply_polygon_drag(
            ui_runtime, mouse, result, dirty, poly, menu_name, layer_name,
        );
    }
}

fn apply_vertex_drag(
    ui_runtime: &UiRuntime,
    mouse: &MouseSnapshot,
    result: &mut EditorInteractionResult,
    dirty: &mut LayerDirty,
    poly: &mut UiButtonPolygon,
    active_id: usize,
    menu_name: &str,
    layer_name: &str,
) {
    let (ox, oy) = ui_runtime.drag_offset.unwrap_or((0.0, 0.0));
    let new_x = mouse.mx - ox;
    let new_y = mouse.my - oy;

    if let Some((idx, v)) = poly
        .vertices
        .iter_mut()
        .enumerate()
        .find(|(_, v)| v.id == active_id)
    {
        let old_pos = v.pos;

        // Push command for undo
        if let Some(poly_id) = &poly.id {
            result.pending_vertex_commands.push(MoveVertexCommand {
                menu: menu_name.to_string(),
                layer: layer_name.to_string(),
                element_id: poly_id.clone(),
                vertex_index: idx,
                before: old_pos,
                after: [new_x, new_y],
            });
        }

        v.pos = [new_x, new_y];
        dirty.mark_polygons();
        dirty.mark_outlines();
        result.moved_any_selected_object = true;
    }
}

fn apply_polygon_drag(
    ui_runtime: &UiRuntime,
    mouse: &MouseSnapshot,
    result: &mut EditorInteractionResult,
    dirty: &mut LayerDirty,
    poly: &mut UiButtonPolygon,
    menu_name: &str,
    layer_name: &str,
) {
    let Some((ox, oy)) = ui_runtime.drag_offset else {
        return;
    };

    let (ccx, ccy) = compute_polygon_centroid(poly);
    let new_cx = mouse.mx - ox;
    let new_cy = mouse.my - oy;
    let dx = new_cx - ccx;
    let dy = new_cy - ccy;

    if dx.abs() > 0.001 || dy.abs() > 0.001 {
        if let Some(poly_id) = &poly.id {
            // Push move command for the polygon centroid
            result.pending_move_commands.push(MoveElementCommand {
                menu: menu_name.to_string(),
                layer: layer_name.to_string(),
                element_id: poly_id.clone(),
                element_kind: ElementKind::Polygon,
                before: (ccx, ccy),
                after: (new_cx, new_cy),
            });
        }

        for v in poly.vertices.iter_mut() {
            v.pos[0] += dx;
            v.pos[1] += dy;
        }
        dirty.mark_polygons();
        dirty.mark_outlines();
        result.moved_any_selected_object = true;
    }
}

fn handle_polygon_released(
    ui_runtime: &mut UiRuntime,
    dirty: &mut LayerDirty,
    result: &mut EditorInteractionResult,
) {
    ui_runtime.selected_ui_element_primary.dragging = false;
    ui_runtime.drag_offset = None;
    ui_runtime.active_vertex = None;
    dirty.mark_polygons();
    result.trigger_selection = true;
}

// ============================================================================
// PROCESS TEXT - COMPLETE REWRITE
// ============================================================================

fn process_text(
    loader: &mut UiButtonLoader,
    time_system: &TimeSystem,
    mouse: &MouseSnapshot,
    top_hit: Option<&HitResult>,
    result: &mut EditorInteractionResult,
    input_state: &InputState,
) {
    let mut pending_selection: Option<SelectedUiElement> = None;
    let mut pending_drag: Option<PendingDragStart> = None;

    for (menu_name, menu) in active_menus(&mut loader.menus) {
        for layer in active_saveable_layers(&mut menu.layers) {
            process_text_layer(
                &mut loader.ui_runtime,
                &mut loader.variables,
                time_system,
                mouse,
                top_hit,
                result,
                menu_name,
                layer,
                &mut pending_selection,
                &mut pending_drag,
            );
        }
    }

    // Apply pending drag after iteration
    apply_pending_drag(loader, pending_drag);

    select_correctly(pending_selection, loader, input_state);
}

fn process_text_layer(
    ui_runtime: &mut UiRuntime,
    variables: &mut UiVariableRegistry,
    time_system: &TimeSystem,
    mouse: &MouseSnapshot,
    top_hit: Option<&HitResult>,
    result: &mut EditorInteractionResult,
    menu_name: &str,
    layer: &mut RuntimeLayer,
    pending_selection: &mut Option<SelectedUiElement>,
    pending_drag: &mut Option<PendingDragStart>,
) {
    for (text_index, text) in layer.texts.iter_mut().enumerate() {
        process_single_text(
            ui_runtime,
            variables,
            time_system,
            mouse,
            top_hit,
            result,
            menu_name,
            &mut layer.dirty,
            &layer.name,
            text_index,
            text,
            pending_selection,
            pending_drag,
        );
    }
}

fn process_single_text(
    ui_runtime: &mut UiRuntime,
    variables: &mut UiVariableRegistry,
    time_system: &TimeSystem,
    mouse: &MouseSnapshot,
    top_hit: Option<&HitResult>,
    result: &mut EditorInteractionResult,
    menu_name: &str,
    dirty: &mut LayerDirty,
    layer_name: &str,
    text_index: usize,
    text: &mut UiButtonText,
    pending_selection: &mut Option<SelectedUiElement>,
    pending_drag: &mut Option<PendingDragStart>,
) {
    let id = match get_text_id(text) {
        Some(id) => id,
        None => return,
    };

    let is_down = ui_runtime.get(&id).is_down;
    let is_hit = compute_element_hit(top_hit, menu_name, layer_name, HitElement::Text(text_index));
    let is_selected = compute_element_selected(ui_runtime, menu_name, layer_name, &id);

    if ui_runtime.editor_mode && text.misc.editable {
        let should_return = handle_text_editor_mode(
            ui_runtime,
            variables,
            mouse,
            dirty,
            text,
            is_hit,
            is_selected,
            is_down,
        );
        if should_return {
            return;
        }
    }

    let touched_now = compute_touched_now(is_down, is_hit, mouse);
    let state = ui_runtime.update_touch(
        &id,
        touched_now,
        time_system.sim_dt,
        &layer_name.to_string(),
    );

    if should_skip_text_touch_handling(ui_runtime, is_down, is_selected) {
        return;
    }

    handle_text_touch_state(
        ui_runtime,
        mouse,
        result,
        menu_name,
        layer_name,
        dirty,
        text,
        &id,
        state,
        pending_selection,
        pending_drag,
    );
}

fn get_text_id(text: &UiButtonText) -> Option<String> {
    if !text.misc.active {
        return None;
    }
    text.id.clone()
}

fn should_skip_text_touch_handling(
    ui_runtime: &UiRuntime,
    is_down: bool,
    is_selected: bool,
) -> bool {
    (is_down && !is_selected) || ui_runtime.selected_ui_element_primary.just_deselected
}

fn handle_text_editor_mode(
    ui_runtime: &mut UiRuntime,
    variables: &mut UiVariableRegistry,
    mouse: &MouseSnapshot,
    dirty: &mut LayerDirty,
    text: &mut UiButtonText,
    is_hit: bool,
    is_selected: bool,
    is_down: bool,
) -> bool {
    sync_editing_state(ui_runtime, variables, text);

    if text.being_edited {
        dirty.mark_texts();
    }

    if is_selected && ui_runtime.editing_text {
        return true;
    }

    let want_enter_edit = check_want_enter_edit(ui_runtime, mouse, text, is_hit, is_selected);

    if want_enter_edit {
        enter_text_editing_mode(ui_runtime, variables, text);
        dirty.mark_texts();
    }

    if should_exit_edit_mode(ui_runtime, mouse, text, is_hit, want_enter_edit) {
        exit_text_editing_mode(ui_runtime, dirty, text, is_selected);
        return true;
    }

    if !is_down && !is_hit {
        return true;
    }

    false
}

fn sync_editing_state(
    ui_runtime: &UiRuntime,
    variables: &mut UiVariableRegistry,
    text: &mut UiButtonText,
) {
    if !ui_runtime.editing_text {
        text.being_edited = false;
        variables.set_bool("selected_text.being_edited", text.being_edited);
    }
}

fn check_want_enter_edit(
    ui_runtime: &UiRuntime,
    mouse: &MouseSnapshot,
    text: &UiButtonText,
    is_hit: bool,
    is_selected: bool,
) -> bool {
    let normal_text_enter = mouse.just_pressed && is_hit && !ui_runtime.editing_text && is_selected;
    let input_box_enter =
        mouse.just_pressed && is_hit && !ui_runtime.editing_text && !is_selected && text.input_box;

    normal_text_enter || input_box_enter
}

fn should_exit_edit_mode(
    ui_runtime: &UiRuntime,
    mouse: &MouseSnapshot,
    text: &UiButtonText,
    is_hit: bool,
    want_enter_edit: bool,
) -> bool {
    let clicked_outside = mouse.just_pressed
        && !want_enter_edit
        && ui_runtime.editing_text
        && text.being_edited
        && !is_hit;

    clicked_outside || ui_runtime.selected_ui_element_primary.just_deselected
}

pub fn enter_text_editing_mode(
    ui_runtime: &mut UiRuntime,
    variables: &mut UiVariableRegistry,
    text: &mut UiButtonText,
) {
    ui_runtime.editing_text = true;
    text.being_edited = true;
    variables.set_bool("selected_text.being_edited", text.being_edited);

    if !text.input_box || ui_runtime.override_mode {
        text.text = text.template.clone();
    }
}

fn exit_text_editing_mode(
    ui_runtime: &mut UiRuntime,
    dirty: &mut LayerDirty,
    text: &mut UiButtonText,
    is_selected: bool,
) {
    if is_selected && (!text.input_box || ui_runtime.override_mode) {
        text.template = text.text.clone();
        dirty.mark_texts();
    }

    ui_runtime.editing_text = false;
    text.being_edited = false;
    ui_runtime.selected_ui_element_primary.just_deselected = false;
}

fn handle_text_touch_state(
    ui_runtime: &mut UiRuntime,
    mouse: &MouseSnapshot,
    result: &mut EditorInteractionResult,
    menu_name: &str,
    layer_name: &str,
    dirty: &mut LayerDirty,
    text: &mut UiButtonText,
    id: &str,
    state: TouchState,
    pending_selection: &mut Option<SelectedUiElement>,
    pending_drag: &mut Option<PendingDragStart>,
) {
    match state {
        TouchState::Pressed => {
            handle_text_pressed(
                ui_runtime,
                mouse,
                menu_name,
                layer_name,
                dirty,
                text,
                id,
                pending_selection,
                pending_drag,
            );
        }
        TouchState::Held => {
            handle_text_held(
                ui_runtime, mouse, result, dirty, text, menu_name, layer_name,
            );
        }
        TouchState::Released => {
            handle_text_released(ui_runtime, dirty, result);
        }
        TouchState::Idle => {}
    }
}

fn handle_text_pressed(
    ui_runtime: &mut UiRuntime,
    mouse: &MouseSnapshot,
    menu_name: &str,
    layer_name: &str,
    dirty: &mut LayerDirty,
    text: &UiButtonText,
    id: &str,
    pending_selection: &mut Option<SelectedUiElement>,
    pending_drag: &mut Option<PendingDragStart>,
) {
    if ui_runtime.editor_mode {
        ui_runtime.drag_offset = Some((mouse.mx - text.x, mouse.my - text.y));

        *pending_drag = Some(PendingDragStart {
            menu: menu_name.to_string(),
            layer: layer_name.to_string(),
            element_id: id.to_string(),
            element_kind: ElementKind::Text,
            start_pos: (text.x, text.y),
            start_size: Some(text.px as f32),
        });
    }

    dirty.mark_texts();

    *pending_selection = Some(create_selection(
        menu_name,
        layer_name,
        id,
        ElementKind::Text,
        text.action.clone(),
        text.input_box,
    ));
}

fn handle_text_held(
    ui_runtime: &mut UiRuntime,
    mouse: &MouseSnapshot,
    result: &mut EditorInteractionResult,
    dirty: &mut LayerDirty,
    text: &mut UiButtonText,
    menu_name: &str,
    layer_name: &str,
) {
    if text.being_edited {
        return;
    }

    ui_runtime.selected_ui_element_primary.dragging = true;

    if ui_runtime.editor_mode && text.misc.editable {
        apply_text_drag(
            ui_runtime, mouse, result, dirty, text, menu_name, layer_name,
        );
    }
}

fn apply_text_drag(
    ui_runtime: &UiRuntime,
    mouse: &MouseSnapshot,
    result: &mut EditorInteractionResult,
    dirty: &mut LayerDirty,
    text: &mut UiButtonText,
    menu_name: &str,
    layer_name: &str,
) {
    let Some((ox, oy)) = ui_runtime.drag_offset else {
        return;
    };

    let new_x = mouse.mx - ox;
    let new_y = mouse.my - oy;

    if (new_x - text.x).abs() > 0.001 || (new_y - text.y).abs() > 0.001 {
        if let Some(id) = &text.id {
            result.pending_move_commands.push(MoveElementCommand {
                menu: menu_name.to_string(),
                layer: layer_name.to_string(),
                element_id: id.clone(),
                element_kind: ElementKind::Text,
                before: (text.x, text.y),
                after: (new_x, new_y),
            });
        }

        text.x = new_x;
        text.y = new_y;
        dirty.mark_texts();
        result.moved_any_selected_object = true;
    }
}

fn handle_text_released(
    ui_runtime: &mut UiRuntime,
    dirty: &mut LayerDirty,
    result: &mut EditorInteractionResult,
) {
    ui_runtime.selected_ui_element_primary.dragging = false;
    ui_runtime.drag_offset = None;
    dirty.mark_texts();
    result.trigger_selection = true;
}

// ============================================================================
// KEYBOARD UI NAVIGATION
// ============================================================================

fn process_keyboard_ui_navigation(loader: &mut UiButtonLoader, input: &mut InputState) {
    if input.ctrl {
        return;
    }

    let dir = determine_navigation_direction(input);
    let Some(dir) = dir else { return };

    let sel = loader.ui_runtime.selected_ui_element_primary.clone();

    let sel_pos = match find_selected_center(&loader.menus, &sel) {
        Some(p) => p,
        None => return,
    };

    let next = find_best_element_in_direction(loader, &sel, sel_pos, dir);

    let Some((next_layer, next_id, element_type)) = next else {
        return;
    };

    select_ui_element(
        loader,
        SelectedUiElement {
            menu_name: sel.menu_name.clone(),
            layer_name: next_layer,
            element_id: next_id,
            active: false,
            just_deselected: false,
            dragging: false,
            element_type,
            just_selected: false,
            action_name: "None".to_string(),
            input_box: false,
        },
    );
}

fn determine_navigation_direction(input: &mut InputState) -> Option<Direction> {
    if input.action_repeat("Navigate UI Left") {
        Some(Direction::Left)
    } else if input.action_repeat("Navigate UI Right") {
        Some(Direction::Right)
    } else if input.action_repeat("Navigate UI Up") {
        Some(Direction::Up)
    } else if input.action_repeat("Navigate UI Down") {
        Some(Direction::Down)
    } else {
        None
    }
}

fn find_best_element_in_direction(
    loader: &UiButtonLoader,
    selected: &SelectedUiElement,
    sel_pos: (f32, f32),
    dir: Direction,
) -> Option<(String, String, ElementKind)> {
    if let Some(result) = find_best_element_in_direction_inner(loader, selected, sel_pos, dir, 40.0)
    {
        return Some(result);
    }

    let opposite = match dir {
        Direction::Up => Direction::Down,
        Direction::Down => Direction::Up,
        Direction::Left => Direction::Right,
        Direction::Right => Direction::Left,
    };

    find_best_element_in_direction_inner(loader, selected, sel_pos, opposite, 65.0)
}

fn find_best_element_in_direction_inner(
    loader: &UiButtonLoader,
    selected: &SelectedUiElement,
    sel_pos: (f32, f32),
    dir: Direction,
    max_angle_deg: f32,
) -> Option<(String, String, ElementKind)> {
    let max_angle_rad = max_angle_deg.to_radians();
    let cos_max = max_angle_rad.cos();

    let dir_vec = direction_to_vector(dir);

    let menu = loader.menus.get(&selected.menu_name)?;
    let items = collect_navigation_candidates(menu);

    find_best_candidate(selected, sel_pos, dir, dir_vec, cos_max, items)
}

fn direction_to_vector(dir: Direction) -> (f32, f32) {
    match dir {
        Direction::Up => (0.0, -1.0),
        Direction::Down => (0.0, 1.0),
        Direction::Left => (-1.0, 0.0),
        Direction::Right => (1.0, 0.0),
    }
}

fn collect_navigation_candidates(menu: &Menu) -> Vec<(String, String, (f32, f32), ElementKind)> {
    let mut items = Vec::with_capacity(64);

    for layer in &menu.layers {
        if !layer.active || !layer.saveable {
            continue;
        }
        for (elem, kind) in layer.iter_all_elements() {
            items.push((
                layer.name.clone(),
                elem.id().to_string(),
                elem.center(),
                kind,
            ));
        }
    }

    items
}

fn find_best_candidate(
    selected: &SelectedUiElement,
    sel_pos: (f32, f32),
    dir: Direction,
    dir_vec: (f32, f32),
    cos_max: f32,
    items: Vec<(String, String, (f32, f32), ElementKind)>,
) -> Option<(String, String, ElementKind)> {
    let mut best: Option<(String, String, ElementKind)> = None;
    let mut best_forward = f32::INFINITY;
    let mut best_lateral = f32::INFINITY;
    let mut best_cos = -1.0_f32;
    let dist_eps = 0.5_f32;

    for (layer_name, elem_id, pos, element_type) in items {
        if elem_id == selected.element_id && layer_name == selected.layer_name {
            continue;
        }

        let dx = pos.0 - sel_pos.0;
        let dy = pos.1 - sel_pos.1;

        let (forward, lateral) = match compute_forward_lateral(dir, dx, dy) {
            Some(fl) => fl,
            None => continue,
        };

        if forward < 0.0001 {
            continue;
        }

        let mag2 = dx * dx + dy * dy;
        if mag2 < 1e-4 {
            continue;
        }
        let mag = mag2.sqrt();

        let cos_theta = ((dx * dir_vec.0 + dy * dir_vec.1) / mag).clamp(-1.0, 1.0);

        if cos_theta < cos_max {
            continue;
        }

        if is_better_candidate(
            forward,
            lateral,
            cos_theta,
            best_forward,
            best_lateral,
            best_cos,
            dist_eps,
        ) {
            best_forward = forward;
            best_lateral = lateral;
            best_cos = cos_theta;
            best = Some((layer_name, elem_id, element_type));
        }
    }

    best
}

fn compute_forward_lateral(dir: Direction, dx: f32, dy: f32) -> Option<(f32, f32)> {
    match dir {
        Direction::Up => {
            if dy >= 0.0 {
                None
            } else {
                Some((-dy, dx.abs()))
            }
        }
        Direction::Down => {
            if dy <= 0.0 {
                None
            } else {
                Some((dy, dx.abs()))
            }
        }
        Direction::Left => {
            if dx >= 0.0 {
                None
            } else {
                Some((-dx, dy.abs()))
            }
        }
        Direction::Right => {
            if dx <= 0.0 {
                None
            } else {
                Some((dx, dy.abs()))
            }
        }
    }
}

fn is_better_candidate(
    forward: f32,
    lateral: f32,
    cos_theta: f32,
    best_forward: f32,
    best_lateral: f32,
    best_cos: f32,
    dist_eps: f32,
) -> bool {
    if forward + dist_eps < best_forward {
        true
    } else if (forward - best_forward).abs() <= dist_eps && lateral + dist_eps < best_lateral {
        true
    } else if (forward - best_forward).abs() <= dist_eps
        && (lateral - best_lateral).abs() <= dist_eps
        && cos_theta > best_cos
    {
        true
    } else {
        false
    }
}

fn find_selected_center(
    menus: &HashMap<String, Menu>,
    sel: &SelectedUiElement,
) -> Option<(f32, f32)> {
    let elem = get_element(menus, &sel.menu_name, &sel.layer_name, &sel.element_id)?;

    Some(elem.center())
}

// ============================================================================
// PENDING CIRCLE UPDATES
// ============================================================================

pub(crate) fn apply_pending_circle_updates(
    loader: &mut UiButtonLoader,
    dt: f32,
    pending_circle_updates: Vec<(String, f32, f32)>,
) {
    for (parent_id, mx, my) in pending_circle_updates {
        let update_info = find_circle_update_info(loader, &parent_id, mx, my);

        let Some((current_radius, target_radius)) = update_info else {
            continue;
        };

        let new_radius = calculate_smoothed_radius(current_radius, target_radius, dt);
        apply_radius_to_related_elements(loader, &parent_id, new_radius);
    }
}

fn find_circle_update_info(
    loader: &mut UiButtonLoader,
    parent_id: &str,
    mx: f32,
    my: f32,
) -> Option<(f32, f32)> {
    for (_, menu) in loader.menus.iter_mut().filter(|(_, m)| m.active) {
        for layer in &mut menu.layers {
            for circle in &mut layer.circles {
                if circle.id.as_ref() == Some(&parent_id.to_string()) {
                    let current_radius = circle.radius;
                    let target_radius = ((mx - circle.x).powi(2) + (my - circle.y).powi(2)).sqrt();
                    return Some((current_radius, target_radius));
                }
            }
        }
    }
    None
}

fn calculate_smoothed_radius(current_radius: f32, target_radius: f32, dt: f32) -> f32 {
    let smoothing_speed = 10.0;
    let dt_effective = dt.clamp(1.0 / 240.0, 0.1);
    let k = 1.0 - (-smoothing_speed * dt_effective).exp();

    (current_radius + (target_radius - current_radius) * k)
        .abs()
        .max(2.0)
}

fn apply_radius_to_related_elements(loader: &mut UiButtonLoader, parent_id: &str, new_radius: f32) {
    let mut resize_info: Option<(String, String, f32, f32)> = None;

    for (menu_name, menu) in loader.menus.iter_mut().filter(|(_, m)| m.active) {
        for layer in &mut menu.layers {
            for circle in &mut layer.circles {
                if circle.id.as_ref() == Some(&parent_id.to_string()) {
                    let old_radius = circle.radius;
                    circle.radius = new_radius;
                    layer.dirty.mark_circles();
                    resize_info = Some((
                        menu_name.clone(),
                        layer.name.clone(),
                        old_radius,
                        new_radius,
                    ));
                }
            }

            for handle in &mut layer.handles {
                if handle.parent_id.as_ref() == Some(&parent_id.to_string()) {
                    handle.radius = new_radius;
                    layer.dirty.mark_handles();
                }
            }

            for outline in &mut layer.outlines {
                if outline.parent_id.as_ref() == Some(&parent_id.to_string()) {
                    outline.shape_data.radius = new_radius;
                    layer.dirty.mark_outlines();
                }
            }
        }
    }

    if let Some((menu, layer, old_radius, new_radius)) = resize_info {
        loader.ui_edit_manager.push_command(ResizeElementCommand {
            menu,
            layer,
            element_id: parent_id.to_string(),
            element_kind: ElementKind::Circle,
            before: old_radius,
            after: new_radius,
        });
    }
}

/// Handle scroll-wheel resizing with undo support
pub fn handle_scroll_resize(loader: &mut UiButtonLoader, scroll: f32) -> bool {
    if !loader.ui_runtime.selected_ui_element_primary.active || scroll == 0.0 {
        return false;
    }

    let selected = loader.ui_runtime.selected_ui_element_primary.clone();

    if selected.element_type != ElementKind::Circle {
        return false;
    }

    let Some(old_size) = get_element_size(
        &loader.menus,
        &selected.menu_name,
        &selected.layer_name,
        &selected.element_id,
        selected.element_type,
    ) else {
        return false;
    };

    let new_size = (old_size + scroll * 3.0).max(2.0);

    loader.ui_edit_manager.push_command(ResizeElementCommand {
        menu: selected.menu_name.clone(),
        layer: selected.layer_name.clone(),
        element_id: selected.element_id.clone(),
        element_kind: selected.element_type,
        before: old_size,
        after: new_size,
    });

    loader.set_element_size(
        &selected.menu_name,
        &selected.layer_name,
        &selected.element_id,
        selected.element_type,
        new_size,
    );

    true
}

// ============================================================================
// TEXT EDITING
// ============================================================================

/// Handle text editing with undo support
pub fn handle_text_editing(
    ui_runtime: &mut UiRuntime,
    menus: &mut HashMap<String, Menu>,
    undo_manager: &mut UiEditManager,
    input: &mut InputState,
    mouse_snapshot: MouseSnapshot,
) {
    if !ui_runtime.editing_text {
        return;
    }

    let sel = &ui_runtime.selected_ui_element_primary;
    if !sel.active {
        return;
    }

    let sel_menu = sel.menu_name.clone();
    let sel_layer = sel.layer_name.clone();
    let sel_element_id = sel.element_id.clone();

    for (menu_name, menu) in menus
        .iter_mut()
        .filter(|(n, m)| **n == sel_menu && m.active)
    {
        for layer in &mut menu.layers {
            if layer.name != sel_layer {
                continue;
            }

            for text in &mut layer.texts {
                if text.id.as_ref() != Some(&sel_element_id) {
                    continue;
                }

                let before_text = text.text.clone();
                let before_template = text.template.clone();
                let before_caret = text.caret;

                process_text_editing_input(
                    ui_runtime,
                    input,
                    mouse_snapshot,
                    text,
                    &mut layer.dirty,
                );

                if text.text != before_text || text.template != before_template {
                    undo_manager.push_command(TextEditCommand {
                        menu: menu_name.clone(),
                        layer: layer.name.clone(),
                        element_id: sel_element_id.clone(),
                        before_text,
                        after_text: text.text.clone(),
                        before_template,
                        after_template: text.template.clone(),
                        before_caret,
                        after_caret: text.caret,
                    });
                }

                return;
            }
        }
    }
}

pub(crate) fn process_text_editing_input(
    ui_runtime: &mut UiRuntime,
    input: &mut InputState,
    mouse_snapshot: MouseSnapshot,
    text: &mut UiButtonText,
    dirty: &mut LayerDirty,
) {
    if handle_mouse_caret_selection(ui_runtime, mouse_snapshot, text) {
        return;
    }

    if handle_ctrl_commands(ui_runtime, input, text, dirty) {
        return;
    }

    if handle_backspace(ui_runtime, input, text, dirty) {
        return;
    }

    if handle_character_input(ui_runtime, input, text, dirty) {
        return;
    }

    handle_arrow_navigation(input, text, dirty);
}

fn handle_mouse_caret_selection(
    ui_runtime: &mut UiRuntime,
    mouse_snapshot: MouseSnapshot,
    text: &mut UiButtonText,
) -> bool {
    let mx = mouse_snapshot.mx;
    let my = mouse_snapshot.my;

    let x0 = text.x;
    let y0 = text.y;
    let x1 = x0 + text.natural_width;
    let y1 = y0 + text.natural_height;

    if mouse_snapshot.just_pressed && mx >= x0 && mx <= x1 && my >= y0 && my <= y1 {
        let new_caret = pick_caret(text, mx);
        text.caret = new_caret;
        text.sel_start = new_caret;
        text.sel_end = new_caret;
        text.has_selection = false;
        ui_runtime.dragging_text = true;
        return true;
    }

    if ui_runtime.dragging_text && mouse_snapshot.pressed {
        let new_pos = pick_caret(text, mx);
        text.sel_end = new_pos;
        text.has_selection = text.sel_end != text.sel_start;
        text.caret = new_pos;
        return true;
    }

    if ui_runtime.dragging_text && !mouse_snapshot.pressed {
        ui_runtime.dragging_text = false;
    }

    false
}

fn handle_ctrl_commands(
    ui_runtime: &mut UiRuntime,
    input: &mut InputState,
    text: &mut UiButtonText,
    dirty: &mut LayerDirty,
) -> bool {
    if !input.ctrl {
        return false;
    }

    if handle_paste(ui_runtime, input, text, dirty) {
        return true;
    }

    if handle_copy(ui_runtime, input, text) {
        return true;
    }

    if handle_cut(ui_runtime, input, text, dirty) {
        return true;
    }

    true
}

fn handle_paste(
    ui_runtime: &mut UiRuntime,
    input: &mut InputState,
    text: &mut UiButtonText,
    dirty: &mut LayerDirty,
) -> bool {
    if !input.action_repeat("Paste text") {
        return false;
    }

    let is_template_mode = !text.input_box || ui_runtime.override_mode;

    if text.has_selection {
        let (l, r) = text.selection_range();
        if is_template_mode {
            text.template.replace_range(l..r, &ui_runtime.clipboard);
        } else {
            text.text.replace_range(l..r, &ui_runtime.clipboard);
        }
        text.caret = l + ui_runtime.clipboard.len();
        text.clear_selection();
    } else {
        for c in ui_runtime.clipboard.clone().chars() {
            if is_template_mode {
                text.template.insert(text.caret, c);
            } else {
                text.text.insert(text.caret, c);
            }
            text.caret += 1;
        }
    }

    text.text = text.template.clone();
    dirty.mark_texts();

    true
}

fn handle_copy(ui_runtime: &mut UiRuntime, input: &mut InputState, text: &UiButtonText) -> bool {
    if !input.action_pressed_once("Copy text") {
        return false;
    }

    let (l, r) = text.selection_range();
    let is_template_mode = !text.input_box || ui_runtime.override_mode;

    ui_runtime.clipboard = if is_template_mode {
        text.template.get(l..r).unwrap_or("").to_string()
    } else {
        text.text.get(l..r).unwrap_or("").to_string()
    };

    true
}

fn handle_cut(
    ui_runtime: &mut UiRuntime,
    input: &mut InputState,
    text: &mut UiButtonText,
    dirty: &mut LayerDirty,
) -> bool {
    if !input.action_pressed_once("Cut text") {
        return false;
    }

    let (l, r) = text.selection_range();
    let is_template_mode = !text.input_box || ui_runtime.override_mode;

    if is_template_mode {
        ui_runtime.clipboard = text.template.get(l..r).unwrap_or("").to_string();
        text.template.replace_range(l..r, "");
        text.text = text.template.clone();
    } else {
        ui_runtime.clipboard = text.text.get(l..r).unwrap_or("").to_string();
        text.text.replace_range(l..r, "");
    }

    text.caret = l;
    text.clear_selection();
    dirty.mark_texts();

    true
}

fn handle_backspace(
    ui_runtime: &UiRuntime,
    input: &mut InputState,
    text: &mut UiButtonText,
    dirty: &mut LayerDirty,
) -> bool {
    if !input.action_repeat("Backspace") {
        return false;
    }

    let is_template_mode = !text.input_box || ui_runtime.override_mode;

    if text.has_selection {
        let (l, r) = text.selection_range();
        if is_template_mode {
            text.template.replace_range(l..r, "");
            text.text = text.template.clone();
        } else {
            text.text.replace_range(l..r, "");
        }
        text.caret = l;
        text.clear_selection();
        dirty.mark_texts();
        return true;
    }

    if text.caret > 0 {
        if is_template_mode {
            text.template.remove(text.caret - 1);
            text.text = text.template.clone();
        } else {
            text.text.remove(text.caret - 1);
        }
        text.caret -= 1;
        dirty.mark_texts();
    }

    true
}

fn handle_character_input(
    ui_runtime: &UiRuntime,
    input: &mut InputState,
    text: &mut UiButtonText,
    dirty: &mut LayerDirty,
) -> bool {
    if !input.repeat("char_repeat", !input.text_chars.is_empty()) {
        return false;
    }

    let is_template_mode = !text.input_box || ui_runtime.override_mode;

    if text.has_selection {
        delete_selection(text, is_template_mode);
    }

    insert_characters(text, &input.text_chars, is_template_mode);
    dirty.mark_texts();

    true
}

fn delete_selection(text: &mut UiButtonText, is_template_mode: bool) {
    let (l, r) = text.selection_range();

    if is_template_mode {
        let bl = caret_to_byte(&text.template, l);
        let br = caret_to_byte(&text.template, r);
        text.template.replace_range(bl..br, "");
        text.text = text.template.clone();
    } else {
        let bl = caret_to_byte(&text.text, l);
        let br = caret_to_byte(&text.text, r);
        text.text.replace_range(bl..br, "");
    }

    text.caret = l;
    text.clear_selection();
}

fn insert_characters(text: &mut UiButtonText, chars: &HashSet<String>, is_template_mode: bool) {
    for s in chars {
        if s.is_empty() {
            continue;
        }

        if is_template_mode {
            let bi = caret_to_byte(&text.template, text.caret);
            text.template.insert_str(bi, s);
            text.text = text.template.clone();
        } else {
            let bi = caret_to_byte(&text.text, text.caret);
            text.text.insert_str(bi, s);
        }

        text.caret += s.chars().count();
    }
}

fn handle_arrow_navigation(
    input: &mut InputState,
    text: &mut UiButtonText,
    dirty: &mut LayerDirty,
) {
    if text.has_selection {
        handle_selection_collapse(input, text, dirty);
        return;
    }

    if input.action_repeat("Move Cursor Left") && text.caret > 0 {
        text.caret -= 1;
        dirty.mark_texts();
    }

    if input.action_repeat("Move Cursor Right") && text.caret < text.template.len() {
        text.caret += 1;
        dirty.mark_texts();
    }
}

fn handle_selection_collapse(
    input: &mut InputState,
    text: &mut UiButtonText,
    dirty: &mut LayerDirty,
) {
    let (l, r) = text.selection_range();

    if input.action_pressed_once("Move Cursor Left") {
        text.caret = l;
        text.clear_selection();
        dirty.mark_texts();
    }

    if input.action_pressed_once("Move Cursor Right") {
        text.caret = r;
        text.clear_selection();
        dirty.mark_texts();
    }
}

fn pick_caret(text: &UiButtonText, mx: f32) -> usize {
    for (i, (_, gx1)) in text.glyph_bounds.iter().enumerate() {
        if mx < *gx1 {
            return i;
        }
    }
    text.glyph_bounds.len()
}

fn caret_to_byte(text: &str, caret: usize) -> usize {
    text.char_indices()
        .nth(caret)
        .map(|(i, _)| i)
        .unwrap_or_else(|| text.len())
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

pub fn colors_equal(a: &[f32; 4], b: &[f32; 4]) -> bool {
    const EPSILON: f32 = 0.001;
    (a[0] - b[0]).abs() < EPSILON
        && (a[1] - b[1]).abs() < EPSILON
        && (a[2] - b[2]).abs() < EPSILON
        && (a[3] - b[3]).abs() < EPSILON
}
