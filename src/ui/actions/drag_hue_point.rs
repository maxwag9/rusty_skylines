use crate::resources::TimeSystem;
use crate::ui::actions::{ActionArg, ActionState, deactivate_action};
use crate::ui::input::MouseState;
use crate::ui::touches::HitResult;
use crate::ui::ui_editor::{MiscButtonSettings, UiButtonCircle, UiButtonLoader, UiElement::Circle};
use glam::Vec2;
use std::f32::consts::{FRAC_PI_2, TAU};

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s;
    let h_prime = h * 6.0;
    let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match h_prime as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    [r + m, g + m, b + m]
}

pub fn drag_hue_point(
    loader: &mut UiButtonLoader,
    mouse_state: &MouseState,
    top_hit: &Option<HitResult>,
    time: &TimeSystem,
) {
    let selection = &loader.ui_runtime.selected_ui_element_primary;
    let dragging = selection.dragging;
    let just_selected = selection.just_selected;
    let just_deselected = selection.just_deselected;

    let Some((hue, handle)) = get_picker_circles(loader) else {
        return;
    };

    ensure_action_state(loader, hue.radius, handle.radius);

    let pointer = smooth_pointer(loader, mouse_state, dragging);
    let state = loader.ui_runtime.action_states.get("Drag Hue Point");
    let og_handle_r = state
        .and_then(|s| s.get_data("handle_radius"))
        .and_then(|a| a.as_float())
        .unwrap_or(handle.radius);
    let og_hue_r = state
        .and_then(|s| s.get_data("hue_radius"))
        .and_then(|a| a.as_float())
        .unwrap_or(hue.radius);

    let (new_pos, color, handle_radius, hue_radius) = if dragging {
        compute_drag_state(&hue, pointer, og_handle_r, og_hue_r, loader)
    } else {
        compute_idle_state(
            &hue,
            &handle,
            pointer,
            og_handle_r,
            og_hue_r,
            loader,
            time.sim_target_step,
        )
    };

    let handle_color = get_handle_color(loader);

    let created = loader.edit_circle(
        "Editor_Menu",
        "Color Picker",
        "color picker handle circle",
        new_pos.map(|p| p.x),
        new_pos.map(|p| p.y),
        Some(handle_radius),
        color.or(handle_color),
        None,
    );

    loader.edit_circle(
        "Editor_Menu",
        "Color Picker",
        "Color Picker Color Circle",
        None,
        None,
        Some(hue_radius),
        None,
        None,
    );

    if just_deselected {
        deactivate_action(loader, "Drag Hue Point");
        let _ = loader.delete_element("Editor_Menu", "Color Picker", "color picker handle circle");
    }

    if !created && just_selected && is_action_active(loader) {
        spawn_handle_circle(loader, mouse_state);
    }

    if let Some(color) = color {
        apply_to_multi_selection(loader, color);
    }
}

fn get_picker_circles(loader: &UiButtonLoader) -> Option<(CircleSnapshot, CircleSnapshot)> {
    let layer = loader
        .menus
        .get("Editor_Menu")?
        .layers
        .iter()
        .find(|l| l.name == "Color Picker")?;

    let hue = layer.circles.iter().find(|c| c.style == "Hue Circle")?;
    let handle = layer
        .circles
        .iter()
        .find(|c| c.id.as_deref() == Some("color picker handle circle"))?;

    Some((
        CircleSnapshot {
            x: hue.x,
            y: hue.y,
            radius: hue.radius,
        },
        CircleSnapshot {
            x: handle.x,
            y: handle.y,
            radius: handle.radius,
        },
    ))
}

#[derive(Clone, Copy)]
struct CircleSnapshot {
    x: f32,
    y: f32,
    radius: f32,
}

fn ensure_action_state(loader: &mut UiButtonLoader, hue_radius: f32, handle_radius: f32) {
    let state = loader
        .ui_runtime
        .action_states
        .entry("Drag Hue Point".to_string())
        .or_insert_with(|| ActionState::new("Drag Hue Point"));

    if state.get_data("handle_radius").is_none() {
        state.set_data("handle_radius", ActionArg::Float(handle_radius));
    }
    if state.get_data("hue_radius").is_none() {
        state.set_data("hue_radius", ActionArg::Float(hue_radius));
    }
}

fn smooth_pointer(loader: &mut UiButtonLoader, mouse: &MouseState, dragging: bool) -> Vec2 {
    if !dragging {
        return mouse.pos;
    }

    let state = match loader.ui_runtime.action_states.get_mut("Drag Hue Point") {
        Some(s) => s,
        None => return mouse.pos,
    };

    let last = state.last_pos.unwrap_or(mouse.last_pos);
    let base = if last.x == 0.0 { mouse.last_pos } else { last };

    let smoothed = Vec2::new(
        lerp(base.x, mouse.pos.x, 0.5),
        lerp(base.y, mouse.pos.y, 0.4),
    );

    state.last_pos = Some(smoothed);
    smoothed
}

fn compute_drag_state(
    hue: &CircleSnapshot,
    pointer: Vec2,
    og_handle_r: f32,
    og_hue_r: f32,
    loader: &mut UiButtonLoader,
) -> (Option<Vec2>, Option<[f32; 4]>, f32, f32) {
    let offset = pointer - Vec2::new(hue.x, hue.y);
    let angle = offset.y.atan2(offset.x);
    let clamped_dist = offset.length().min(hue.radius);

    let new_pos = Vec2::new(
        hue.x + angle.cos() * clamped_dist,
        hue.y + angle.sin() * clamped_dist,
    );

    let ang_shifted = angle + FRAC_PI_2;
    let ang_wrapped = ang_shifted.sin().atan2(ang_shifted.cos());
    let h = ang_wrapped / TAU + 0.5;
    let s = (clamped_dist / hue.radius).clamp(0.0, 1.0).powf(0.47);
    let v = 1.0;

    let rgb = hsv_to_rgb(h, s, v);

    loader.variables.set_f32("color_picker.r", rgb[0]);
    loader.variables.set_f32("color_picker.g", rgb[1]);
    loader.variables.set_f32("color_picker.b", rgb[2]);
    loader.variables.set_f32("color_picker.h", h);
    loader.variables.set_f32("color_picker.s", s);
    loader.variables.set_f32("color_picker.v", v);

    let handle_radius = loader.ui_runtime.original_radius + 5.0;
    let hue_radius = og_hue_r * 1.1;

    (
        Some(new_pos),
        Some([rgb[0], rgb[1], rgb[2], 1.0]),
        handle_radius,
        hue_radius,
    )
}

fn compute_idle_state(
    hue: &CircleSnapshot,
    handle: &CircleSnapshot,
    pointer: Vec2,
    og_handle_r: f32,
    og_hue_r: f32,
    loader: &UiButtonLoader,
    dt: f32,
) -> (Option<Vec2>, Option<[f32; 4]>, f32, f32) {
    let h = loader.variables.get_f32("color_picker.h").unwrap_or(0.0);
    let s = loader.variables.get_f32("color_picker.s").unwrap_or(1.0);

    let mut angle = h * TAU - FRAC_PI_2;
    angle = angle.rem_euclid(TAU);

    let r = s.powf(1.0 / 0.47).clamp(0.0, 1.0) * og_hue_r;
    let target = Vec2::new(hue.x + angle.cos() * r, hue.y + angle.sin() * r);

    let alpha = (dt / 0.2).clamp(0.0, 1.0);
    let smoothed = Vec2::new(
        lerp(handle.x, target.x, alpha),
        lerp(handle.y, target.y, alpha),
    );

    let dist_to_mouse = (Vec2::new(handle.x, handle.y) - pointer).length();
    let handle_radius = if dist_to_mouse < og_handle_r + 4.0 {
        og_handle_r + 4.0
    } else {
        og_handle_r
    };

    (Some(smoothed), None, handle_radius, og_hue_r)
}

fn get_handle_color(loader: &UiButtonLoader) -> Option<[f32; 4]> {
    let r = loader.variables.get_f32("color_picker.r").unwrap_or(1.0);
    let g = loader.variables.get_f32("color_picker.g").unwrap_or(1.0);
    let b = loader.variables.get_f32("color_picker.b").unwrap_or(1.0);
    Some([r, g, b, 1.0])
}

fn is_action_active(loader: &UiButtonLoader) -> bool {
    loader
        .ui_runtime
        .action_states
        .get("Drag Hue Point")
        .map(|s| s.active)
        .unwrap_or(false)
}

fn spawn_handle_circle(loader: &mut UiButtonLoader, mouse: &MouseState) {
    let handle = UiButtonCircle {
        id: Some("color picker handle circle".to_string()),
        action: "None".to_string(),
        style: "None".to_string(),
        z_index: 990,
        x: mouse.pos.x,
        y: mouse.pos.y,
        radius: 6.0,
        inside_border_thickness: 0.002,
        border_thickness: 1.0,
        fade: 0.0,
        fill_color: [0.2, 0.2, 0.2, 0.0],
        inside_border_color: [0.4; 4],
        border_color: [0.1, 0.1, 0.1, 0.8],
        glow_color: [0.0; 4],
        glow_misc: Default::default(),
        misc: MiscButtonSettings {
            active: true,
            touched_time: 0.0,
            is_touched: false,
            pressable: false,
            editable: false,
        },
    };

    let _ = loader.add_element("Editor_Menu", "Color Picker", Circle(handle), mouse, false);
}

fn apply_to_multi_selection(loader: &mut UiButtonLoader, color: [f32; 4]) {
    let elements: Vec<_> = loader
        .ui_runtime
        .selected_ui_element_multi
        .iter()
        .cloned()
        .collect();

    for elem in elements {
        if let Some(menu) = loader.menus.get_mut(&elem.menu_name) {
            menu.change_element_color(&elem.layer_name, &elem.element_id, elem.element_type, color);
        }
    }
}
