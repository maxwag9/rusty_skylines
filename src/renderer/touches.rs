use crate::renderer::helper::{dist, polygon_sdf};
use crate::renderer::ui_editor::{Menu, UiButtonLoader, UiRuntime};
use crate::resources::MouseState;
use crate::vertex::{
    RuntimeLayer, SelectedUiElement, TouchState, UiButtonCircle, UiButtonHandle, UiButtonPolygon,
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum HitElement {
    Circle(usize),
    Handle(usize),
    Polygon(usize),
}

#[derive(Clone, Copy)]
pub(crate) struct HitResult {
    pub layer_index: usize,
    pub element: HitElement,
    pub z_index: i32,
    pub layer_order: u32,
}

impl HitResult {
    pub fn matches(&self, layer_index: usize, element: HitElement) -> bool {
        self.layer_index == layer_index && self.element == element
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
) -> bool {
    for (_, menu) in menus.iter().filter(|(_, menu)| menu.active) {
        for layer in menu.layers.iter().filter(|l| l.active) {
            if circle_hit(layer, mouse.mx, mouse.my) {
                return true;
            }

            if polygon_hit(layer, mouse.mx, mouse.my) {
                return true;
            }

            if editor_mode && handle_hit(layer, mouse.mx, mouse.my) {
                return true;
            }
        }
    }
    false
}

pub(crate) fn near_handle(menus: &HashMap<String, Menu>, mouse: &MouseSnapshot) -> bool {
    for (_, menu) in menus.iter().filter(|(_, menu)| menu.active) {
        for layer in menu.layers.iter().filter(|l| l.active) {
            for h in &layer.handles {
                if !h.misc.active {
                    continue;
                }
                let dx = mouse.mx - h.x;
                let dy = mouse.my - h.y;
                let dist = (dx * dx + dy * dy).sqrt();
                let margin = (h.radius * 0.2).max(12.0);
                if (dist - h.radius).abs() < margin {
                    return true;
                }
            }
        }
    }
    false
}

pub(crate) fn find_top_hit(
    menus: &HashMap<String, Menu>,
    mouse: &MouseSnapshot,
    editor_mode: bool,
) -> Option<HitResult> {
    let mut best: Option<HitResult> = None;
    for (_, menu) in menus.iter().filter(|(_, menu)| menu.active) {
        for (layer_index, layer) in menu.layers.iter().enumerate() {
            if !layer.active {
                continue;
            }

            for (circle_index, circle) in layer.circles.iter().enumerate() {
                if !circle.misc.active {
                    continue;
                }

                let drag_radius = (circle.radius * 0.8).max(8.0);
                if hit_circle(mouse.mx, mouse.my, circle, drag_radius) {
                    consider_candidate(
                        &mut best,
                        HitResult {
                            layer_index,
                            element: HitElement::Circle(circle_index),
                            z_index: circle.z_index,
                            layer_order: layer.order,
                        },
                    );
                }
            }

            if editor_mode {
                for (handle_index, handle) in layer.handles.iter().enumerate() {
                    if !handle.misc.active {
                        continue;
                    }

                    if hit_handle(mouse.mx, mouse.my, handle) {
                        consider_candidate(
                            &mut best,
                            HitResult {
                                layer_index,
                                element: HitElement::Handle(handle_index),
                                z_index: handle.z_index,
                                layer_order: layer.order,
                            },
                        );
                    }
                }
            }

            for (poly_index, poly) in layer.polygons.iter().enumerate() {
                if !poly.misc.active {
                    continue;
                }

                if hit_polygon(mouse.mx, mouse.my, poly) {
                    consider_candidate(
                        &mut best,
                        HitResult {
                            layer_index,
                            element: HitElement::Polygon(poly_index),
                            z_index: poly.z_index,
                            layer_order: layer.order,
                        },
                    );
                }
            }
        }
    }
    best
}

pub(crate) fn handle_editor_mode_interactions(
    loader: &mut UiButtonLoader,
    dt: f32,
    mouse: &MouseSnapshot,
    top_hit: Option<HitResult>,
) -> EditorInteractionResult {
    let mut result = EditorInteractionResult::default();

    let ui_runtime = &mut loader.ui_runtime;
    for (menu_name, mut menu) in loader.menus.iter_mut().filter(|(_, menu)| menu.active) {
        for (layer_index, layer) in menu.layers.iter_mut().enumerate() {
            if !layer.active {
                continue;
            }

            process_circles(
                ui_runtime,
                menu_name,
                layer,
                layer_index,
                dt,
                mouse,
                top_hit,
                &mut result,
            );
            process_handles(
                ui_runtime,
                layer,
                layer_index,
                dt,
                mouse,
                top_hit,
                &mut result,
            );
            process_polygons(
                ui_runtime,
                menu_name,
                layer,
                layer_index,
                dt,
                mouse,
                top_hit,
                &mut result,
            );
        }
    }
    result
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
        let Some(layer_name) = found_layer_name else {
            continue;
        };

        // 3. Smooth transitioning
        let smoothing_speed = 10.0;
        let dt_effective = dt.clamp(1.0 / 240.0, 0.1);
        let k = 1.0 - (-smoothing_speed * dt_effective).exp();
        let new_radius = (current_radius + (target_radius - current_radius) * k)
            .abs()
            .max(2.0);

        // 4. Apply updated radius to ALL relevant objects
        for (_, menu) in loader.menus.iter_mut().filter(|(_, m)| m.active) {
            for layer in &mut menu.layers {
                for circle in &mut layer.circles {
                    if circle.id.as_ref() == Some(&parent_id) {
                        circle.radius = new_radius;
                        layer.dirty = true;
                    }
                }

                for handle in &mut layer.handles {
                    if handle.parent_id.as_ref() == Some(&parent_id) {
                        handle.radius = new_radius;
                        layer.dirty = true;
                    }
                }

                for outline in &mut layer.outlines {
                    if outline.parent_id.as_ref() == Some(&parent_id) {
                        outline.shape_data.radius = new_radius;
                        layer.dirty = true;
                    }
                }
            }
        }
    }
}

pub(crate) fn mark_editor_layers_dirty(menu: Option<&mut Menu>) {
    const TARGETS: [&str; 2] = ["editor_selection", "editor_handles"];

    if let Some(menu) = menu {
        for layer in &mut menu.layers {
            if TARGETS.contains(&layer.name.as_str()) {
                layer.dirty = true;
            }
        }
    }
}

pub(crate) fn handle_scroll_resize(loader: &mut UiButtonLoader, scroll: f32) -> bool {
    if !loader.ui_runtime.selected_ui_element.active || scroll.abs() <= 0.0 {
        return false;
    }

    let selected_id = loader.ui_runtime.selected_ui_element.element_id.clone();
    let mut selection_changed = false;
    for (_, menu) in loader.menus.iter_mut().filter(|(_, menu)| menu.active) {
        for layer in &mut menu.layers {
            for circle in &mut layer.circles {
                if let Some(id) = &circle.id {
                    if *id == selected_id {
                        circle.radius = (circle.radius + scroll * 3.0).max(2.0);
                        layer.dirty = true;
                        selection_changed = true;
                    }
                }
            }
        }
    }
    selection_changed
}

fn circle_hit(layer: &RuntimeLayer, mx: f32, my: f32) -> bool {
    for circle in &layer.circles {
        if !circle.misc.active {
            continue;
        }
        if hit_circle(mx, my, circle, circle.radius) {
            return true;
        }
    }
    false
}

fn polygon_hit(layer: &RuntimeLayer, mx: f32, my: f32) -> bool {
    for poly in &layer.polygons {
        if !poly.misc.active {
            continue;
        }
        if hit_polygon(mx, my, poly) {
            return true;
        }
    }
    false
}

fn handle_hit(layer: &RuntimeLayer, mx: f32, my: f32) -> bool {
    for handle in &layer.handles {
        if !handle.misc.active {
            continue;
        }
        if hit_handle(mx, my, handle) {
            return true;
        }
    }
    false
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
        let current_key = (current.z_index, current.layer_order);
        let candidate_key = (candidate.z_index, candidate.layer_order);
        if candidate_key > current_key {
            *current = candidate;
        }
    } else {
        *best = Some(candidate);
    }
}

fn process_circles(
    ui_runtime: &mut UiRuntime,
    menu_name: &str,
    layer: &mut RuntimeLayer,
    layer_index: usize,
    dt: f32,
    mouse: &MouseSnapshot,
    top_hit: Option<HitResult>,
    result: &mut EditorInteractionResult,
) {
    for (circle_index, circle) in layer.circles.iter_mut().enumerate() {
        if !circle.misc.active {
            continue;
        }

        let Some(id) = &circle.id else {
            continue;
        };

        let runtime = ui_runtime.get(id);
        let is_hit = top_hit
            .map(|hit| hit.matches(layer_index, HitElement::Circle(circle_index)))
            .unwrap_or(false);

        if !runtime.is_down && !is_hit {
            continue;
        }

        let touched_now = if !runtime.is_down {
            is_hit && mouse.just_pressed
        } else {
            mouse.pressed
        };

        let state = ui_runtime.update_touch(id, touched_now, dt, &layer.name);

        match state {
            TouchState::Pressed => {
                ui_runtime.drag_offset = Some((mouse.mx - circle.x, mouse.my - circle.y));
            }
            TouchState::Held => {
                if let Some((ox, oy)) = ui_runtime.drag_offset {
                    let new_x = mouse.mx - ox;
                    let new_y = mouse.my - oy;
                    if (new_x - circle.x).abs() > 0.001 || (new_y - circle.y).abs() > 0.001 {
                        circle.x = new_x;
                        circle.y = new_y;
                        layer.dirty = true;
                        result.moved_any_selected_object = true;
                    }
                }
            }
            TouchState::Released => {
                ui_runtime.drag_offset = None;
                ui_runtime.selected_ui_element = SelectedUiElement {
                    menu_name: menu_name.to_string(),
                    layer_name: layer.name.clone(),
                    element_id: id.clone(),
                    active: true,
                };
                result.trigger_selection = true;
            }
            TouchState::Idle => {}
        }
    }
}

fn process_handles(
    ui_runtime: &mut UiRuntime,
    layer: &mut RuntimeLayer,
    layer_index: usize,
    dt: f32,
    mouse: &MouseSnapshot,
    top_hit: Option<HitResult>,
    result: &mut EditorInteractionResult,
) {
    for (handle_index, handle) in layer.handles.iter_mut().enumerate() {
        if !handle.misc.active {
            continue;
        }

        let Some(id) = &handle.id else {
            continue;
        };

        let runtime = ui_runtime.get(id);
        let is_hit = top_hit
            .map(|hit| hit.matches(layer_index, HitElement::Handle(handle_index)))
            .unwrap_or(false);

        if !runtime.is_down && !is_hit {
            continue;
        }

        let touched_now = if !runtime.is_down {
            is_hit && mouse.just_pressed
        } else {
            mouse.pressed
        };

        let state = ui_runtime.update_touch(id, touched_now, dt, &layer.name);

        if let Some(parent_id) = &handle.parent_id {
            if matches!(state, TouchState::Held) {
                result
                    .pending_circle_updates
                    .push((parent_id.clone(), mouse.mx, mouse.my));
            }
        }
    }
}

fn process_polygons(
    ui_runtime: &mut UiRuntime,
    menu_name: &str,
    layer: &mut RuntimeLayer,
    layer_index: usize,
    dt: f32,
    mouse: &MouseSnapshot,
    top_hit: Option<HitResult>,
    result: &mut EditorInteractionResult,
) {
    const VERTEX_RADIUS: f32 = 20.0;

    for (poly_index, poly) in layer.polygons.iter_mut().enumerate() {
        if !poly.misc.active {
            continue;
        }

        if poly.vertices.is_empty() {
            continue;
        }

        let Some(id) = &poly.id else {
            continue;
        };

        let runtime = ui_runtime.get(id);
        let is_hit = top_hit
            .map(|hit| hit.matches(layer_index, HitElement::Polygon(poly_index)))
            .unwrap_or(false);

        if !runtime.is_down && !is_hit {
            continue;
        }

        let verts = &mut poly.vertices;

        let mut cx = 0.0f32;
        let mut cy = 0.0f32;
        for v in verts.iter() {
            cx += v.pos[0];
            cy += v.pos[1];
        }
        let inv_n = 1.0 / verts.len() as f32;
        cx *= inv_n;
        cy *= inv_n;

        let vertex_hit = verts
            .iter()
            .enumerate()
            .find(|(_, v)| dist(mouse.mx, mouse.my, v.pos[0], v.pos[1]) < VERTEX_RADIUS)
            .map(|(i, _)| i);

        let sdf = polygon_sdf(mouse.mx, mouse.my, verts);
        let inside = sdf < 0.0;
        let near_edge = sdf.abs() < 8.0;
        let poly_hit = inside || near_edge;

        let hit = vertex_hit.is_some() || poly_hit;

        if !runtime.is_down && !(hit && is_hit) {
            continue;
        }

        let touched_now = if !runtime.is_down {
            mouse.just_pressed
        } else {
            mouse.pressed
        };

        let state = ui_runtime.update_touch(id, touched_now, dt, &layer.name);

        match state {
            TouchState::Pressed => {
                if let Some(vidx) = vertex_hit {
                    let vx = verts[vidx].pos[0];
                    let vy = verts[vidx].pos[1];
                    ui_runtime.drag_offset = Some((mouse.mx - vx, mouse.my - vy));
                    ui_runtime.active_vertex = Some(verts[vidx].id);
                } else {
                    ui_runtime.drag_offset = Some((mouse.mx - cx, mouse.my - cy));
                    ui_runtime.active_vertex = None;
                }
            }
            TouchState::Held => {
                if let Some(active_id) = ui_runtime.active_vertex {
                    let (ox, oy) = ui_runtime.drag_offset.unwrap_or((0.0, 0.0));
                    let new_x = mouse.mx - ox;
                    let new_y = mouse.my - oy;
                    if let Some(v) = verts.iter_mut().find(|v| v.id == active_id) {
                        v.pos = [new_x, new_y];
                        layer.dirty = true;
                        result.moved_any_selected_object = true;
                    }
                } else if let Some((ox, oy)) = ui_runtime.drag_offset {
                    let mut ccx = 0.0;
                    let mut ccy = 0.0;
                    for v in verts.iter() {
                        ccx += v.pos[0];
                        ccy += v.pos[1];
                    }
                    let inv_n = 1.0 / verts.len() as f32;
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
                        layer.dirty = true;
                        result.moved_any_selected_object = true;
                    }
                }
            }
            TouchState::Released => {
                ui_runtime.drag_offset = None;
                ui_runtime.active_vertex = None;
                ui_runtime.selected_ui_element = SelectedUiElement {
                    menu_name: menu_name.to_string(),
                    layer_name: layer.name.clone(),
                    element_id: id.clone(),
                    active: true,
                };

                result.trigger_selection = true;
            }
            TouchState::Idle => {}
        }
    }
}
