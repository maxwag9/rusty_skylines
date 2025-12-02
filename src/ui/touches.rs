use crate::resources::{InputState, TimeSystem};
use crate::ui::actions::selected_needed;
use crate::ui::helper::{dist, polygon_sdf};
use crate::ui::input::MouseState;
use crate::ui::selections::{select_move_primary_to_multi, select_to_multi, select_ui_element};
use crate::ui::ui_editor::{Menu, UiButtonLoader, UiRuntime};
use crate::ui::vertex::{
    ElementKind, RuntimeLayer, SelectedUiElement, TouchState, UiButtonCircle, UiButtonHandle,
    UiButtonPolygon, UiButtonText, UiElement, UiElementRef,
};
use std::collections::HashMap;

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
    pub z_index: i32,
    pub layer_order: u32,
    pub action: Option<String>,
}

impl HitResult {
    pub fn matches(&self, menu_name: &str, layer_name: &str, element: HitElement) -> bool {
        self.menu_name == menu_name && self.layer_name == layer_name && self.element == element
    }
}

#[derive(Default)]
pub(crate) struct EditorInteractionResult {
    pub trigger_selection: bool,
    pub pending_circle_updates: Vec<(String, f32, f32)>,
    pub moved_any_selected_object: bool,
}

pub(crate) fn press_began_on_ui(
    menus: &HashMap<String, Menu>,
    mouse: &MouseSnapshot,
    editor_mode: bool,
) -> (bool, String) {
    for (_, menu) in menus.iter().filter(|(_, menu)| menu.active) {
        for layer in menu.layers.iter().filter(|l| l.active) {
            let (hit, action) = circle_hit(layer, mouse.mx, mouse.my);
            if hit {
                return (hit, action);
            }
            let (hit, action) = polygon_hit(layer, mouse.mx, mouse.my);
            if hit {
                return (hit, action);
            }

            if editor_mode {
                let (hit, action) = handle_hit(layer, mouse.mx, mouse.my);
                if hit {
                    return (hit, action);
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

pub(crate) fn find_top_hit(
    menus: &mut HashMap<String, Menu>,
    mouse: &MouseSnapshot,
    editor_mode: bool,
) -> Option<HitResult> {
    let mut best: Option<HitResult> = None;
    for (menu_name, menu) in menus.iter_mut().filter(|(_, menu)| menu.active) {
        for layer in menu.layers.iter_mut().filter(|l| l.active) {
            // Circles
            for (circle_index, circle) in layer.circles.iter().enumerate() {
                if !circle.misc.active {
                    continue;
                }
                if !circle.misc.pressable && !circle.misc.editable {
                    continue;
                }
                let mut drag_radius: f32 = circle.radius;
                if editor_mode {
                    drag_radius = (circle.radius * 0.9).max(8.0);
                }

                if hit_circle(mouse.mx, mouse.my, circle, drag_radius) {
                    consider_candidate(
                        &mut best,
                        HitResult {
                            menu_name: menu_name.clone(),
                            layer_name: layer.name.clone(),
                            element: HitElement::Circle(circle_index),
                            z_index: circle.z_index,
                            layer_order: layer.order,
                            action: Some(circle.action.clone()),
                        },
                    );
                }
            }

            // Handles (editor only)
            if editor_mode {
                for (handle_index, handle) in layer.handles.iter().enumerate() {
                    if !handle.misc.active {
                        continue;
                    }
                    if !handle.misc.pressable && !handle.misc.editable {
                        continue;
                    }

                    if hit_handle(mouse.mx, mouse.my, handle) {
                        consider_candidate(
                            &mut best,
                            HitResult {
                                menu_name: menu_name.clone(),
                                layer_name: layer.name.clone(),
                                element: HitElement::Handle(handle_index),
                                z_index: handle.z_index,
                                layer_order: layer.order,
                                action: None,
                            },
                        );
                    }
                }
            }

            // Polygons
            for (poly_index, poly) in layer.polygons.iter().enumerate() {
                if !poly.misc.active {
                    continue;
                }
                if !poly.misc.pressable && !poly.misc.editable {
                    continue;
                }

                if hit_polygon(mouse.mx, mouse.my, poly) {
                    consider_candidate(
                        &mut best,
                        HitResult {
                            menu_name: menu_name.clone(),
                            layer_name: layer.name.clone(),
                            element: HitElement::Polygon(poly_index),
                            z_index: poly.z_index,
                            layer_order: layer.order,
                            action: Some(poly.action.clone()),
                        },
                    );
                }
            }

            for (text_index, text) in layer.texts.iter_mut().enumerate() {
                if !text.misc.active {
                    continue;
                }
                if !text.misc.pressable && !text.misc.editable {
                    continue;
                }
                text.being_hovered = false;

                if hit_text(text, mouse.mx, mouse.my) {
                    text.being_hovered = true;
                    text.just_unhovered = false;
                    consider_candidate(
                        &mut best,
                        HitResult {
                            menu_name: menu_name.clone(),
                            layer_name: layer.name.clone(),
                            element: HitElement::Text(text_index),
                            z_index: text.z_index,
                            layer_order: layer.order,
                            action: Some(text.action.clone()),
                        },
                    );
                }
            }
        }
    }
    best
}

#[derive(Copy, Clone, PartialEq)]
enum Direction {
    Left,
    Right,
    Up,
    Down,
}

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

fn process_keyboard_ui_navigation(loader: &mut UiButtonLoader, input: &mut InputState) {
    // Determine which arrow is held
    if input.ctrl {
        return;
    }

    let dir = if input.action_repeat("Navigate UI Left") {
        Some(Direction::Left)
    } else if input.action_repeat("Navigate UI Right") {
        Some(Direction::Right)
    } else if input.action_repeat("Navigate UI Up") {
        Some(Direction::Up)
    } else if input.action_repeat("Navigate UI Down") {
        Some(Direction::Down)
    } else {
        None
    };

    let Some(dir) = dir else { return };

    // Extract selection WITHOUT borrowing loader.menus
    let sel = {
        let s = &loader.ui_runtime.selected_ui_element_primary;
        s.clone()
    };

    // Resolve selected element center — borrow ends here
    let sel_pos = match find_selected_center(loader, &sel) {
        Some(p) => p,
        None => return,
    };

    // NEXT target element — another dead-end borrow
    let next = find_best_element_in_direction(loader, &sel, sel_pos, dir);

    let Some((next_layer, next_id, element_type)) = next else {
        return;
    };

    select_ui_element(
        loader,
        sel.menu_name.clone(),
        next_layer,
        next_id,
        false,
        element_type,
        "None".to_string(),
    );
}

fn find_best_element_in_direction(
    loader: &UiButtonLoader,
    selected: &SelectedUiElement,
    sel_pos: (f32, f32),
    dir: Direction,
) -> Option<(String, String, ElementKind)> {
    // First: normal directional navigation with a fairly tight cone
    if let Some(result) = find_best_element_in_direction_inner(loader, selected, sel_pos, dir, 40.0)
    {
        return Some(result);
    }

    // Nothing found, wrap in the opposite direction with a wider cone
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

    let dir_vec = match dir {
        Direction::Up => (0.0_f32, -1.0_f32),
        Direction::Down => (0.0_f32, 1.0_f32),
        Direction::Left => (-1.0_f32, 0.0_f32),
        Direction::Right => (1.0_f32, 0.0_f32),
    };

    let menu = loader.menus.get(&selected.menu_name)?;

    // Collect candidates: (layer_name, elem_id, center_pos)
    let mut items = Vec::with_capacity(64);
    for layer in &menu.layers {
        if !layer.active || !layer.saveable {
            continue;
        }
        for elem in layer.iter_all_elements() {
            items.push((
                layer.name.clone(),
                elem.0.id().to_string(),
                element_center(elem.0),
                elem.1,
            ));
        }
    }

    // Best candidate tracked lexicographically by:
    // 1) forward distance (primary)
    // 2) lateral distance (tiebreaker)
    // 3) angle (tie breaker, via cos(angle), higher is better)
    let mut best: Option<(String, String, ElementKind)> = None;
    let mut best_forward = f32::INFINITY;
    let mut best_lateral = f32::INFINITY;
    let mut best_cos = -1.0_f32;

    // Small epsilon so tiny differences do not cause jitter
    let dist_eps = 0.5_f32;

    for (layer_name, elem_id, pos, element_type) in items {
        // Skip currently selected element
        if elem_id == selected.element_id && layer_name == selected.layer_name {
            continue;
        }

        let dx = pos.0 - sel_pos.0;
        let dy = pos.1 - sel_pos.1;

        // Compute forward and lateral distance based on direction
        let (forward, lateral) = match dir {
            Direction::Up => {
                if dy >= 0.0 {
                    continue;
                }
                (-dy, dx.abs())
            }
            Direction::Down => {
                if dy <= 0.0 {
                    continue;
                }
                (dy, dx.abs())
            }
            Direction::Left => {
                if dx >= 0.0 {
                    continue;
                }
                (-dx, dy.abs())
            }
            Direction::Right => {
                if dx <= 0.0 {
                    continue;
                }
                (dx, dy.abs())
            }
        };

        // Ignore almost zero movement, avoids degenerate cases
        if forward < 0.0001 {
            continue;
        }

        let mag2 = dx * dx + dy * dy;
        if mag2 < 1e-4 {
            continue;
        }
        let mag = mag2.sqrt();

        // Angle check via cos(theta)
        let cos_theta_raw = (dx * dir_vec.0 + dy * dir_vec.1) / mag;
        let cos_theta = cos_theta_raw.clamp(-1.0, 1.0);

        if cos_theta < cos_max {
            // Outside allowed cone
            continue;
        }

        // Lexicographic better check
        let better = if forward + dist_eps < best_forward {
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
        };

        if better {
            best_forward = forward;
            best_lateral = lateral;
            best_cos = cos_theta;
            best = Some((layer_name, elem_id, element_type));
        }
    }

    best
}

fn element_center(e: UiElementRef) -> (f32, f32) {
    match e {
        UiElementRef::Text(t) => (t.x, t.y),
        UiElementRef::Circle(c) => (c.x, c.y),
        UiElementRef::Handle(h) => (h.x, h.y),
        UiElementRef::Outline(o) => (o.shape_data.x, o.shape_data.y),
        UiElementRef::Polygon(p) => {
            let count = p.vertices.len().max(1);
            let sum = p
                .vertices
                .iter()
                .fold((0.0, 0.0), |acc, v| (acc.0 + v.pos[0], acc.1 + v.pos[1]));
            (sum.0 / count as f32, sum.1 / count as f32)
        }
    }
}

fn find_selected_center(
    loader: &mut UiButtonLoader,
    sel: &SelectedUiElement,
) -> Option<(f32, f32)> {
    let elem = loader.find_element(&sel.menu_name, &sel.layer_name, &sel.element_id)?;

    Some(match elem {
        UiElement::Text(t) => (t.x, t.y),
        UiElement::Circle(c) => (c.x, c.y),
        UiElement::Handle(h) => (h.x, h.y),

        UiElement::Outline(o) => (o.shape_data.x, o.shape_data.y),

        UiElement::Polygon(p) => {
            let count = p.vertices.len().max(1);
            let sum = p
                .vertices
                .iter()
                .fold((0.0, 0.0), |acc, v| (acc.0 + v.pos[0], acc.1 + v.pos[1]));
            (sum.0 / count as f32, sum.1 / count as f32)
        }
    })
}

pub(crate) fn apply_pending_circle_updates(
    loader: &mut UiButtonLoader,
    dt: f32,
    pending_circle_updates: Vec<(String, f32, f32)>,
) {
    for (parent_id, mx, my) in pending_circle_updates {
        let mut current_radius = 0.0f32;
        let mut target_radius = 0.0f32;
        let mut found_layer_name: Option<String> = None;

        // 1. Find the element across active menus
        for (_, menu) in loader.menus.iter_mut().filter(|(_, m)| m.active) {
            for layer in &mut menu.layers {
                for circle in &mut layer.circles {
                    if circle.id.as_ref() == Some(&parent_id) {
                        current_radius = circle.radius;
                        target_radius = ((mx - circle.x).powi(2) + (my - circle.y).powi(2)).sqrt();
                        found_layer_name = Some(layer.name.clone());
                    }
                }
            }
        }

        // 2. If not found, skip update
        let Some(_layer_name) = found_layer_name else {
            continue;
        };

        // 3. Smooth transitioning
        let smoothing_speed = 10.0;
        let dt_effective = dt.clamp(1.0 / 240.0, 0.1);
        let k = 1.0 - (-smoothing_speed * dt_effective).exp();
        let new_radius = (current_radius + (target_radius - current_radius) * k)
            .abs()
            .max(2.0);
        // 4. Apply updated radius to ALL relevant objects (same parent_id)
        for (_, menu) in loader.menus.iter_mut().filter(|(_, m)| m.active) {
            for layer in &mut menu.layers {
                for circle in &mut layer.circles {
                    if circle.id.as_ref() == Some(&parent_id) {
                        circle.radius = new_radius;
                        layer.dirty.mark_circles();
                    }
                }

                for handle in &mut layer.handles {
                    if handle.parent_id.as_ref() == Some(&parent_id) {
                        handle.radius = new_radius;
                        layer.dirty.mark_handles();
                    }
                }

                for outline in &mut layer.outlines {
                    if outline.parent_id.as_ref() == Some(&parent_id) {
                        outline.shape_data.radius = new_radius;
                        layer.dirty.mark_outlines();
                    }
                }
            }
        }
    }
}

pub(crate) fn handle_scroll_resize(loader: &mut UiButtonLoader, scroll: f32) -> bool {
    if !loader.ui_runtime.selected_ui_element_primary.active || scroll == 0.0 {
        return false;
    }

    let selected = loader.ui_runtime.selected_ui_element_primary.clone();
    let mut selection_changed = false;

    for (menu_name, menu) in loader.menus.iter_mut().filter(|(_, menu)| menu.active) {
        for layer in &mut menu.layers {
            // Only affect the currently selected layer + menu
            if selected.menu_name != *menu_name || selected.layer_name != layer.name {
                continue;
            }

            for circle in &mut layer.circles {
                if let Some(id) = &circle.id {
                    if *id == selected.element_id {
                        circle.radius = (circle.radius + scroll * 3.0).max(2.0);
                        layer.dirty.mark_circles();
                        selection_changed = true;
                    }
                }
            }
        }
    }
    selection_changed
}

fn hit_text(text: &UiButtonText, mx: f32, my: f32) -> bool {
    // 1. compute the natural text bounds (you already have this somewhere in the text layout)
    let w = text.natural_width;
    let h = text.natural_height;

    let x = text.x;
    let y = text.y;

    // 2. build warped quad from natural box + offsets
    let quad = [
        [x + text.top_left_offset[0], y + text.top_left_offset[1]],
        [
            x + w + text.top_right_offset[0],
            y + text.top_right_offset[1],
        ],
        [
            x + w + text.bottom_right_offset[0],
            y + h + text.bottom_right_offset[1],
        ],
        [
            x + text.bottom_left_offset[0],
            y + h + text.bottom_left_offset[1],
        ],
    ];

    point_in_quad(mx, my, &quad)
}

fn point_in_quad(px: f32, py: f32, quad: &[[f32; 2]; 4]) -> bool {
    fn edge(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
        (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    }

    let p = [px, py];

    let mut sign = 0.0;
    for i in 0..4 {
        let a = quad[i];
        let b = quad[(i + 1) % 4];
        let e = edge(a, b, p);

        if e.abs() < 1e-6 {
            continue; // on edge
        }
        let s = e.signum();
        if sign == 0.0 {
            sign = s;
        } else if s != sign {
            return false;
        }
    }
    true
}

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

fn consider_candidate(best: &mut Option<HitResult>, candidate: HitResult) {
    if let Some(current) = best {
        // First compare layer_order (layer z), then element z_index inside the layer
        let current_key = (current.layer_order, current.z_index);
        let candidate_key = (candidate.layer_order, candidate.z_index);
        if candidate_key > current_key {
            *current = candidate;
        }
    } else {
        *best = Some(candidate);
    }
}

fn process_circles(
    loader: &mut UiButtonLoader,
    dt: f32,
    mouse: &MouseSnapshot,
    top_hit: Option<&HitResult>,
    result: &mut EditorInteractionResult,
    input_state: &InputState,
) {
    // Start with no selection pending
    let mut pending_selection: Option<SelectedUiElement> = None;

    for (menu_name, menu) in loader.menus.iter_mut().filter(|(_, m)| m.active) {
        for layer in menu.layers.iter_mut().filter(|l| l.active && l.saveable) {
            for (circle_index, circle) in layer.circles.iter_mut().enumerate() {
                if !circle.misc.active {
                    continue;
                }

                let Some(id) = &circle.id else { continue };
                let runtime = loader.ui_runtime.get(id);

                let is_hit = top_hit
                    .map(|hit| {
                        hit.matches(menu_name, &layer.name, HitElement::Circle(circle_index))
                    })
                    .unwrap_or(false);

                if !runtime.is_down && !is_hit {
                    continue;
                }

                let is_selected = loader.ui_runtime.selected_ui_element_primary.active
                    && loader.ui_runtime.selected_ui_element_primary.menu_name == *menu_name
                    && loader.ui_runtime.selected_ui_element_primary.layer_name == layer.name
                    && loader.ui_runtime.selected_ui_element_primary.element_id == *id;

                if runtime.is_down && !is_selected {
                    continue;
                }

                let touched_now = if !runtime.is_down {
                    is_hit && mouse.just_pressed
                } else {
                    mouse.pressed
                };

                let state = loader
                    .ui_runtime
                    .update_touch(id, touched_now, dt, &layer.name);

                match state {
                    TouchState::Pressed => {
                        if loader.ui_runtime.editor_mode {
                            loader.ui_runtime.drag_offset =
                                Some((mouse.mx - circle.x, mouse.my - circle.y));
                        }
                        // Store the element that should become selected
                        pending_selection = Some(SelectedUiElement {
                            active: true,
                            menu_name: menu_name.to_string(),
                            layer_name: layer.name.clone(),
                            element_id: id.clone(),
                            just_deselected: false,
                            dragging: false,
                            element_type: ElementKind::Circle,
                            just_selected: true,
                            action_name: circle.action.clone(),
                        });

                        result.trigger_selection = true;
                    }

                    TouchState::Held => {
                        loader.ui_runtime.selected_ui_element_primary.dragging = true;
                        if loader.ui_runtime.editor_mode && circle.misc.editable {
                            if let Some((ox, oy)) = loader.ui_runtime.drag_offset {
                                let new_x = mouse.mx - ox;
                                let new_y = mouse.my - oy;
                                if (new_x - circle.x).abs() > 0.001
                                    || (new_y - circle.y).abs() > 0.001
                                {
                                    circle.x = new_x;
                                    circle.y = new_y;
                                    layer.dirty.mark_circles();
                                    result.moved_any_selected_object = true;
                                }
                            }
                        }
                    }

                    TouchState::Released => {
                        loader.ui_runtime.selected_ui_element_primary.dragging = false;
                        loader.ui_runtime.drag_offset = None;
                    }

                    TouchState::Idle => {}
                }
            }
        }
    }

    // Apply the selection after processing all circles
    if let Some(p) = pending_selection {
        if selected_needed(loader, p.action_name.as_str()) {
            select_move_primary_to_multi(
                loader,
                p.menu_name,
                p.layer_name,
                p.element_id,
                p.dragging,
                p.element_type,
                p.action_name,
            )
        } else if input_state.ctrl {
            select_to_multi(
                loader,
                p.menu_name,
                p.layer_name,
                p.element_id,
                p.dragging,
                p.element_type,
                p.action_name,
            );
        } else {
            select_ui_element(
                loader,
                p.menu_name,
                p.layer_name,
                p.element_id,
                p.dragging,
                p.element_type,
                p.action_name,
            );
        }
    }
}

fn process_handles(
    loader: &mut UiButtonLoader,
    dt: f32,
    mouse: &MouseSnapshot,
    top_hit: Option<&HitResult>,
    result: &mut EditorInteractionResult,
    input_state: &InputState,
) {
    let mut pending_selection: Option<SelectedUiElement> = None;

    for (menu_name, menu) in loader.menus.iter_mut().filter(|(_, m)| m.active) {
        for layer in menu.layers.iter_mut().filter(|l| l.active) {
            for (handle_index, handle) in layer.handles.iter_mut().enumerate() {
                if !(handle.misc.active || handle.misc.pressable) {
                    continue;
                }

                let Some(id) = &handle.id else { continue };
                let runtime = loader.ui_runtime.get(id);

                let is_hit = top_hit
                    .map(|hit| {
                        hit.matches(menu_name, &layer.name, HitElement::Handle(handle_index))
                    })
                    .unwrap_or(false);

                let is_selected = loader.ui_runtime.selected_ui_element_primary.active
                    && loader.ui_runtime.selected_ui_element_primary.menu_name == *menu_name
                    && loader.ui_runtime.selected_ui_element_primary.layer_name == layer.name
                    && loader.ui_runtime.selected_ui_element_primary.element_id == *id;

                if !runtime.is_down && !is_hit {
                    continue;
                }

                if runtime.is_down && !is_selected {
                    continue;
                }

                let touched_now = if !runtime.is_down {
                    is_hit && mouse.just_pressed
                } else {
                    mouse.pressed
                };

                let state = loader
                    .ui_runtime
                    .update_touch(id, touched_now, dt, &layer.name);

                match state {
                    TouchState::Pressed => {
                        if loader.ui_runtime.editor_mode {
                            loader.ui_runtime.drag_offset =
                                Some((mouse.mx - handle.x, mouse.my - handle.y));
                        }

                        pending_selection = Some(SelectedUiElement {
                            active: true,
                            menu_name: menu_name.to_string(),
                            layer_name: layer.name.clone(),
                            element_id: id.clone(),
                            just_deselected: false,
                            dragging: false,
                            element_type: ElementKind::Handle,
                            just_selected: true,
                            action_name: "None".to_string(),
                        });

                        result.trigger_selection = true;
                    }

                    TouchState::Held => {
                        loader.ui_runtime.selected_ui_element_primary.dragging = true;
                        // live circle radius update
                        if let Some(parent_id) = &handle.parent_id {
                            if loader.ui_runtime.editor_mode {
                                result.pending_circle_updates.push((
                                    parent_id.clone(),
                                    mouse.mx,
                                    mouse.my,
                                ));
                            }
                        }
                    }

                    TouchState::Released => {
                        loader.ui_runtime.selected_ui_element_primary.dragging = false;
                        loader.ui_runtime.drag_offset = None;
                    }

                    TouchState::Idle => {}
                }
            }
        }
    }

    if let Some(p) = pending_selection {
        if selected_needed(loader, p.action_name.as_str()) {
            select_move_primary_to_multi(
                loader,
                p.menu_name,
                p.layer_name,
                p.element_id,
                p.dragging,
                p.element_type,
                p.action_name,
            )
        } else if input_state.ctrl {
            select_to_multi(
                loader,
                p.menu_name,
                p.layer_name,
                p.element_id,
                p.dragging,
                p.element_type,
                p.action_name,
            );
        } else {
            select_ui_element(
                loader,
                p.menu_name,
                p.layer_name,
                p.element_id,
                p.dragging,
                p.element_type,
                p.action_name,
            );
        }
    }
}

fn process_polygons(
    loader: &mut UiButtonLoader,
    dt: f32,
    mouse: &MouseSnapshot,
    top_hit: Option<&HitResult>,
    result: &mut EditorInteractionResult,
    input_state: &InputState,
) {
    const VERTEX_RADIUS: f32 = 20.0;

    let mut pending_selection: Option<SelectedUiElement> = None;

    for (menu_name, menu) in loader.menus.iter_mut().filter(|(_, m)| m.active) {
        for layer in menu.layers.iter_mut().filter(|l| l.active && l.saveable) {
            for (poly_index, poly) in layer.polygons.iter_mut().enumerate() {
                if !(poly.misc.active & !poly.vertices.is_empty()) {
                    continue;
                }

                let Some(id) = &poly.id else { continue };
                let runtime = loader.ui_runtime.get(id);

                let is_hit_top = top_hit
                    .map(|hit| hit.matches(menu_name, &layer.name, HitElement::Polygon(poly_index)))
                    .unwrap_or(false);

                // Compute centroid
                let verts = &mut poly.vertices;
                let inv_n = 1.0 / verts.len() as f32;

                let mut cx = 0.0;
                let mut cy = 0.0;
                for v in verts.iter() {
                    cx += v.pos[0];
                    cy += v.pos[1];
                }
                cx *= inv_n;
                cy *= inv_n;

                // Per vertex hit
                let vertex_hit = verts
                    .iter()
                    .enumerate()
                    .find(|(_, v)| dist(mouse.mx, mouse.my, v.pos[0], v.pos[1]) < VERTEX_RADIUS)
                    .map(|(i, _)| i);

                // Polygon SDF
                let sdf = polygon_sdf(mouse.mx, mouse.my, verts);
                let inside = sdf < 0.0;
                let near_edge = sdf.abs() < 8.0;
                let poly_hit = inside || near_edge;
                let hit = vertex_hit.is_some() || poly_hit;

                let is_selected = loader.ui_runtime.selected_ui_element_primary.active
                    && loader.ui_runtime.selected_ui_element_primary.menu_name == *menu_name
                    && loader.ui_runtime.selected_ui_element_primary.layer_name == layer.name
                    && loader.ui_runtime.selected_ui_element_primary.element_id == *id;

                if !runtime.is_down && !(hit && is_hit_top) {
                    continue;
                }

                if runtime.is_down && !is_selected {
                    continue;
                }

                let touched_now = if !runtime.is_down {
                    mouse.just_pressed
                } else {
                    mouse.pressed
                };

                let state = loader
                    .ui_runtime
                    .update_touch(id, touched_now, dt, &layer.name);

                match state {
                    TouchState::Pressed => {
                        if loader.ui_runtime.editor_mode && poly.misc.editable {
                            if let Some(vidx) = vertex_hit {
                                let vx = verts[vidx].pos[0];
                                let vy = verts[vidx].pos[1];
                                loader.ui_runtime.drag_offset =
                                    Some((mouse.mx - vx, mouse.my - vy));
                                loader.ui_runtime.active_vertex = Some(verts[vidx].id);
                            } else {
                                loader.ui_runtime.drag_offset =
                                    Some((mouse.mx - cx, mouse.my - cy));
                                loader.ui_runtime.active_vertex = None;
                            }
                        }

                        pending_selection = Some(SelectedUiElement {
                            active: true,
                            menu_name: menu_name.to_string(),
                            layer_name: layer.name.clone(),
                            element_id: id.clone(),
                            just_deselected: false,
                            dragging: false,
                            element_type: ElementKind::Polygon,
                            just_selected: true,
                            action_name: poly.action.clone(),
                        });

                        result.trigger_selection = true;
                    }

                    TouchState::Held => {
                        loader.ui_runtime.selected_ui_element_primary.dragging = true;
                        if loader.ui_runtime.editor_mode && poly.misc.editable {
                            if let Some(active_id) = loader.ui_runtime.active_vertex {
                                let (ox, oy) = loader.ui_runtime.drag_offset.unwrap_or((0.0, 0.0));
                                let new_x = mouse.mx - ox;
                                let new_y = mouse.my - oy;

                                if let Some(v) = verts.iter_mut().find(|v| v.id == active_id) {
                                    v.pos = [new_x, new_y];
                                    layer.dirty.mark_polygons();
                                    layer.dirty.mark_outlines();
                                    result.moved_any_selected_object = true;
                                }
                            } else if let Some((ox, oy)) = loader.ui_runtime.drag_offset {
                                let mut ccx = 0.0;
                                let mut ccy = 0.0;
                                for v in verts.iter() {
                                    ccx += v.pos[0];
                                    ccy += v.pos[1];
                                }
                                ccx *= inv_n;
                                ccy *= inv_n;

                                let new_cx = mouse.mx - ox;
                                let new_cy = mouse.my - oy;

                                let dx = new_cx - ccx;
                                let dy = new_cy - ccy;

                                if dx.abs() > 0.001 || dy.abs() > 0.001 {
                                    for v in verts.iter_mut() {
                                        v.pos[0] += dx;
                                        v.pos[1] += dy;
                                    }
                                    layer.dirty.mark_polygons();
                                    layer.dirty.mark_outlines();
                                    result.moved_any_selected_object = true;
                                }
                            }
                        }
                    }

                    TouchState::Released => {
                        loader.ui_runtime.selected_ui_element_primary.dragging = false;
                        loader.ui_runtime.drag_offset = None;
                        loader.ui_runtime.active_vertex = None;
                    }

                    TouchState::Idle => {}
                }
            }
        }
    }

    if let Some(p) = pending_selection {
        if selected_needed(loader, p.action_name.as_str()) {
            select_move_primary_to_multi(
                loader,
                p.menu_name,
                p.layer_name,
                p.element_id,
                p.dragging,
                p.element_type,
                p.action_name,
            )
        } else if input_state.ctrl {
            select_to_multi(
                loader,
                p.menu_name,
                p.layer_name,
                p.element_id,
                p.dragging,
                p.element_type,
                p.action_name,
            );
        } else {
            select_ui_element(
                loader,
                p.menu_name,
                p.layer_name,
                p.element_id,
                p.dragging,
                p.element_type,
                p.action_name,
            );
        }
    }
}

fn process_text(
    loader: &mut UiButtonLoader,
    time_system: &TimeSystem,
    mouse: &MouseSnapshot,
    top_hit: Option<&HitResult>,
    result: &mut EditorInteractionResult,
    input_state: &InputState,
) {
    let mut pending_selection: Option<SelectedUiElement> = None;
    for (menu_name, menu) in loader.menus.iter_mut().filter(|(_, m)| m.active) {
        for layer in menu.layers.iter_mut().filter(|l| l.active && l.saveable) {
            for (text_index, text) in layer.texts.iter_mut().enumerate() {
                if !text.misc.active {
                    continue;
                }

                let Some(id) = &text.id else { continue };
                let runtime = loader.ui_runtime.get(id);

                let is_hit = top_hit
                    .map(|hit| hit.matches(menu_name, &layer.name, HitElement::Text(text_index)))
                    .unwrap_or(false);

                let is_selected = loader.ui_runtime.selected_ui_element_primary.active
                    && loader.ui_runtime.selected_ui_element_primary.menu_name == *menu_name
                    && loader.ui_runtime.selected_ui_element_primary.layer_name == layer.name
                    && loader.ui_runtime.selected_ui_element_primary.element_id == *id;

                if loader.ui_runtime.editor_mode && text.misc.editable {
                    // if not in global editing mode, no text should be flagged as being_edited
                    if !loader.ui_runtime.editing_text {
                        loader
                            .variables
                            .set("selected_text.being_edited", text.being_edited.to_string());
                        text.being_edited = false;
                    }

                    // if this text is being edited, make sure the layer is redrawn
                    if text.being_edited {
                        layer.dirty.mark_texts();
                    }

                    // enter edit mode: second click on already selected text
                    if mouse.just_pressed
                        && is_hit
                        && is_selected
                        && !loader.ui_runtime.editing_text
                    {
                        loader.ui_runtime.editing_text = true;
                        text.being_edited = true;
                        loader
                            .variables
                            .set("selected_text.being_edited", text.being_edited.to_string());
                        text.text = text.template.clone();
                        layer.dirty.mark_texts();
                        continue;
                    }

                    // exit edit mode when clicking outside after deselection
                    if mouse.just_pressed
                        && loader.ui_runtime.editing_text
                        && text.being_edited
                        && !is_hit
                        && loader
                            .ui_runtime
                            .selected_ui_element_primary
                            .just_deselected
                    {
                        println!("Inside editing text mode EXIT");
                        if is_selected {
                            text.template = text.text.clone();
                            layer.dirty.mark_texts();
                        }

                        loader.ui_runtime.editing_text = false;
                        text.being_edited = false;
                        loader
                            .ui_runtime
                            .selected_ui_element_primary
                            .just_deselected = false;
                        continue;
                    }

                    // when editing this text, do not drag it
                    if is_selected && loader.ui_runtime.editing_text {
                        continue;
                    }

                    // drag / selection logic (your new version)
                    if !runtime.is_down && !is_hit {
                        continue;
                    }
                }
                if runtime.is_down && !is_selected {
                    continue;
                }

                let touched_now = if !runtime.is_down {
                    is_hit && mouse.just_pressed
                } else {
                    mouse.pressed
                };

                let state = loader.ui_runtime.update_touch(
                    id,
                    touched_now,
                    time_system.sim_dt,
                    &layer.name,
                );

                match state {
                    TouchState::Pressed => {
                        if loader.ui_runtime.editor_mode {
                            loader.ui_runtime.drag_offset =
                                Some((mouse.mx - text.x, mouse.my - text.y));
                        }

                        pending_selection = Some(SelectedUiElement {
                            active: true,
                            menu_name: menu_name.to_string(),
                            layer_name: layer.name.clone(),
                            element_id: id.clone(),
                            just_deselected: false,
                            dragging: false,
                            element_type: ElementKind::Text,
                            just_selected: true,
                            action_name: text.action.clone(),
                        });

                        result.trigger_selection = true;
                    }

                    TouchState::Held => {
                        loader.ui_runtime.selected_ui_element_primary.dragging = true;
                        if loader.ui_runtime.editor_mode && text.misc.editable {
                            if let Some((ox, oy)) = loader.ui_runtime.drag_offset {
                                let new_x = mouse.mx - ox;
                                let new_y = mouse.my - oy;

                                if (new_x - text.x).abs() > 0.001 || (new_y - text.y).abs() > 0.001
                                {
                                    text.x = new_x;
                                    text.y = new_y;
                                    layer.dirty.mark_texts();
                                    result.moved_any_selected_object = true;
                                }
                            }
                        }
                    }

                    TouchState::Released => {
                        loader.ui_runtime.selected_ui_element_primary.dragging = false;
                        loader.ui_runtime.drag_offset = None;
                    }

                    TouchState::Idle => {}
                }
            }
        }
    }

    if let Some(p) = pending_selection {
        if selected_needed(loader, p.action_name.as_str()) {
            select_move_primary_to_multi(
                loader,
                p.menu_name,
                p.layer_name,
                p.element_id,
                p.dragging,
                p.element_type,
                p.action_name,
            )
        } else if input_state.ctrl {
            select_to_multi(
                loader,
                p.menu_name,
                p.layer_name,
                p.element_id,
                p.dragging,
                p.element_type,
                p.action_name,
            );
        } else {
            select_ui_element(
                loader,
                p.menu_name,
                p.layer_name,
                p.element_id,
                p.dragging,
                p.element_type,
                p.action_name,
            );
        }
    }
}

pub fn handle_text_editing(
    ui_runtime: &mut UiRuntime,
    menus: &mut HashMap<String, Menu>,
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

    for (_, menu) in menus.iter_mut().filter(|(_, m)| m.active) {
        for layer in &mut menu.layers {
            if layer.name != sel.layer_name {
                continue;
            }

            for text in &mut layer.texts {
                if text.id.as_ref() != Some(&sel.element_id) {
                    continue;
                }

                let mx = mouse_snapshot.mx;
                let my = mouse_snapshot.my;

                let x0 = text.x;
                let y0 = text.y;
                let x1 = x0 + text.natural_width;
                let y1 = y0 + text.natural_height;

                // -------------------------------------
                // MOUSE → CARET / SELECTION
                // -------------------------------------

                if mouse_snapshot.just_pressed && mx >= x0 && mx <= x1 && my >= y0 && my <= y1 {
                    let new_caret = pick_caret(text, mx);
                    text.caret = new_caret;
                    text.sel_start = new_caret;
                    text.sel_end = new_caret;
                    text.has_selection = false;

                    ui_runtime.dragging_text = true;
                    return;
                }

                if ui_runtime.dragging_text && mouse_snapshot.pressed {
                    let new_pos = pick_caret(text, mx);
                    text.sel_end = new_pos;
                    text.has_selection = text.sel_end != text.sel_start;
                    text.caret = new_pos;
                    return;
                }

                if ui_runtime.dragging_text && !mouse_snapshot.pressed {
                    ui_runtime.dragging_text = false;
                }

                // -------------------------------------
                // CTRL + X / C / V
                // -------------------------------------
                if input.ctrl {
                    // -------------------------------
                    // PASTE (repeating if held)
                    // -------------------------------
                    if input.action_repeat("Paste text") {
                        if text.has_selection {
                            let (l, r) = text.selection_range();
                            text.template.replace_range(l..r, &ui_runtime.clipboard);
                            text.caret = l + ui_runtime.clipboard.len();
                            text.clear_selection();
                        } else {
                            for c in ui_runtime.clipboard.clone().chars() {
                                text.template.insert(text.caret, c);
                                text.caret += 1;
                            }
                        }

                        text.text = text.template.clone();
                        layer.dirty.mark_texts();
                        return;
                    }

                    // -------------------------------
                    // COPY (single press)
                    // -------------------------------
                    if input.action_pressed_once("Copy text") {
                        let (l, r) = text.selection_range();
                        ui_runtime.clipboard = text.template.get(l..r).unwrap_or("").to_string();
                        return;
                    }

                    // -------------------------------
                    // CUT (single press)
                    // -------------------------------
                    if input.action_pressed_once("Cut text") {
                        let (l, r) = text.selection_range();
                        ui_runtime.clipboard = text.template.get(l..r).unwrap_or("").to_string();

                        text.template.replace_range(l..r, "");
                        text.caret = l;
                        text.clear_selection();

                        text.text = text.template.clone();
                        layer.dirty.mark_texts();
                        return;
                    }

                    return;
                }

                // ============================================================
                // BACKSPACE (using new repeat)
                // ============================================================
                if input.action_repeat("Backspace") {
                    if text.has_selection {
                        let (l, r) = text.selection_range();
                        text.template.replace_range(l..r, "");
                        text.caret = l;
                        text.clear_selection();

                        text.text = text.template.clone();
                        layer.dirty.mark_texts();
                        return;
                    }

                    if text.caret > 0 {
                        text.template.remove(text.caret - 1);
                        text.caret -= 1;

                        text.text = text.template.clone();
                        layer.dirty.mark_texts();
                    }

                    return;
                }

                // ============================================================
                // CHARACTER INPUT (repeat for held printable chars)
                // ============================================================
                if input.repeat("char_repeat", !input.text_chars.is_empty()) {
                    if text.has_selection {
                        let (l, r) = text.selection_range();
                        let bl = caret_to_byte(&text.template, l);
                        let br = caret_to_byte(&text.template, r);
                        text.template.replace_range(bl..br, "");
                        text.caret = l;
                        text.clear_selection();
                    }

                    for s in input.text_chars.clone() {
                        if !s.is_empty() {
                            let bi = caret_to_byte(&text.template, text.caret);
                            text.template.insert_str(bi, &s);
                            text.caret += s.chars().count();
                        }
                    }

                    text.text = text.template.clone();
                    layer.dirty.mark_texts();
                    return;
                }

                // ============================================================
                // ARROWS (repeat)
                // ============================================================

                // selection collapse
                if text.has_selection {
                    let (l, r) = text.selection_range();

                    if input.action_pressed_once("Move Cursor Left") {
                        text.caret = l;
                        text.clear_selection();
                        layer.dirty.mark_texts();
                    }
                    if input.action_pressed_once("Move Cursor Right") {
                        text.caret = r;
                        text.clear_selection();
                        layer.dirty.mark_texts();
                    }
                    return;
                }

                // move caret
                if input.action_repeat("Move Cursor Left") && text.caret > 0 {
                    println!("Left caret: {}", text.caret);
                    text.caret -= 1;
                    layer.dirty.mark_texts();
                }

                if input.action_repeat("Move Cursor Right") && text.caret < text.template.len() {
                    text.caret += 1;
                    layer.dirty.mark_texts();
                }

                return;
            }
        }
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
