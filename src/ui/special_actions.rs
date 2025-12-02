use crate::ui::actions::deactivate_action;
use crate::ui::input::MouseState;
use crate::ui::touches::HitResult;
use crate::ui::ui_editor::UiElement::Circle;
use crate::ui::ui_editor::{MiscButtonSettings, UiButtonCircle, UiButtonLoader};
use glam::Vec2;
use std::f32::consts::{FRAC_PI_2, PI};

pub fn drag_hue_point(
    loader: &mut UiButtonLoader,
    mouse_state: &MouseState,
    top_hit: &Option<HitResult>,
) {
    let mut new_x = None;
    let mut new_y = None;
    let mut color = None;
    let border_color = None;
    let mut radius = None;
    let mut hue_circle_radius = None;

    if loader.ui_runtime.selected_ui_element_primary.dragging {
        if let Some(action_state) = loader.ui_runtime.action_states.get_mut("Drag Hue Point") {
            if let Some(last_pos) = action_state.last_pos {
                if last_pos.x == 0.0 {
                    new_x = Some(lerp(mouse_state.last_pos.x, mouse_state.pos.x, 0.5));
                    new_y = Some(lerp(mouse_state.last_pos.y, mouse_state.pos.y, 0.4));
                } else {
                    new_x = Some(lerp(last_pos.x, mouse_state.pos.x, 0.5));
                    new_y = Some(lerp(last_pos.y, mouse_state.pos.y, 0.4));
                }
            }

            action_state.last_pos = Some(Vec2::from((
                new_x.unwrap_or(mouse_state.last_pos.x),
                new_y.unwrap_or(mouse_state.last_pos.y),
            )));
        }
    }
    if let Some(circle_layer) = loader
        .menus
        .get("Editor_Menu")
        .and_then(|m| m.layers.iter().find(|l| l.name == "Color Picker"))
    {
        let hue_circle = circle_layer
            .circles
            .iter()
            .find(|c| c.style == "Hue Circle");

        let handle_circle = circle_layer
            .circles
            .iter()
            .find(|c| c.id.as_deref() == Some("color picker handle circle"));

        if let (Some(hue), Some(handle)) = (hue_circle, handle_circle) {
            // Use smoothed position if available, otherwise raw mouse
            let pointer_x = new_x.unwrap_or(mouse_state.pos.x);
            let pointer_y = new_y.unwrap_or(mouse_state.pos.y);

            // Hover radius stuff still uses pointer, not raw mouse
            let dmx = handle.x - pointer_x;
            let dmy = handle.y - pointer_y;
            let dist_to_mouse = (dmx * dmx + dmy * dmy).sqrt();
            if let Some(action_state) = loader.ui_runtime.action_states.get_mut("Drag Hue Point") {
                if let Some(og_radius_1) = action_state.original_radius_1 {
                    if dist_to_mouse < og_radius_1 + 4.0 {
                        radius = Some(og_radius_1 + 4.0);
                    } else {
                        radius = Some(og_radius_1);
                    }
                } else {
                    action_state.original_radius_1 = Some(handle.radius);
                }
                if let Some(og_radius_2) = action_state.original_radius_2 {
                    if loader.ui_runtime.selected_ui_element_primary.dragging {
                        hue_circle_radius = Some(og_radius_2 * 1.1);
                    } else {
                        hue_circle_radius = Some(og_radius_2);
                    }
                } else {
                    action_state.original_radius_2 = Some(hue.radius);
                }
            }
            if loader.ui_runtime.selected_ui_element_primary.dragging {
                radius = Some(loader.ui_runtime.original_radius + 5.0);

                // 1. Vector from center to (smoothed) pointer
                let mx = pointer_x - hue.x;
                let my = pointer_y - hue.y;

                let angle = my.atan2(mx);
                let dist_mouse = (mx * mx + my * my).sqrt();

                // Clamp to hue radius
                let clamped_r = dist_mouse.min(hue.radius);

                // New, clamped handle position
                let hx = hue.x + angle.cos() * clamped_r;
                let hy = hue.y + angle.sin() * clamped_r;

                new_x = Some(hx);
                new_y = Some(hy);

                // 2. Now compute HSV from the *new* handle position
                let dx = hx - hue.x;
                let dy = hy - hue.y;

                let angle = dy.atan2(dx);
                let angle_shifted = angle + FRAC_PI_2;
                let angle_wrapped = angle_shifted.sin().atan2(angle_shifted.cos());

                let h = angle_wrapped / (PI * 2.0) + 0.5;

                let dist = (dx * dx + dy * dy).sqrt();
                let s_linear = (dist / hue.radius).clamp(0.0, 1.0);
                let s = s_linear.powf(0.47);

                let v = 1.0;
                let rgb = hsv_to_rgb(h, s, v);

                color = Some([rgb[0], rgb[1], rgb[2], 1.0]);

                loader.variables.set("color_picker.r", rgb[0].to_string());
                loader.variables.set("color_picker.g", rgb[1].to_string());
                loader.variables.set("color_picker.b", rgb[2].to_string());

                loader.variables.set("color_picker.h", h.to_string());
                loader.variables.set("color_picker.s", s.to_string());
                loader.variables.set("color_picker.v", v.to_string());
            } else {
                // === NEW: Snap handle back to hue circle on release (even for one frame) ===
                if let Some(action_state) = loader.ui_runtime.action_states.get("Drag Hue Point") {
                    if action_state.active {
                        // Use last known handle position (or current if somehow missing)
                        let current_x = handle.x;
                        let current_y = handle.y;

                        let dx = current_x - hue.x;
                        let dy = current_y - hue.y;
                        let dist = (dx * dx + dy * dy).sqrt();

                        if dist > 0.0 {
                            let angle = dy.atan2(dx);
                            let clamped_r = hue.radius; // Force exactly on ring

                            let snapped_x = hue.x + angle.cos() * clamped_r;
                            let snapped_y = hue.y + angle.sin() * clamped_r;

                            new_x = Some(snapped_x);
                            new_y = Some(snapped_y);
                        }
                    }
                }
            }
        }

        loader.ui_runtime.action_states.get("Drag Hue Point");
    }

    let result = loader.edit_circle(
        "Editor_Menu",
        "Color Picker",
        "color picker handle circle",
        new_x,
        new_y,
        radius,
        color,
        border_color,
    );

    let _ = loader.edit_circle(
        "Editor_Menu",
        "Color Picker",
        "Color Picker Color Circle",
        None,
        None,
        hue_circle_radius,
        None,
        None,
    );

    if let Some(hit) = top_hit {
        if hit.action != Some("Drag Hue Point".to_string()) {
            if loader.ui_runtime.selected_ui_element_primary.just_selected {
                deactivate_action(loader, "Drag Hue Point");
                let _ = loader.delete_element(
                    "Editor_Menu",
                    "Color Picker",
                    "color picker handle circle",
                );
            }
        }
    }

    // Handle delete on deselection
    if loader
        .ui_runtime
        .selected_ui_element_primary
        .just_deselected
    {
        deactivate_action(loader, "Drag Hue Point");
        let _ = loader.delete_element("Editor_Menu", "Color Picker", "color picker handle circle");
    }

    if let Some(action) = loader.ui_runtime.action_states.get("Drag Hue Point") {
        let active = action.active.clone();
        if !result {
            if loader.ui_runtime.selected_ui_element_primary.just_selected && active {
                let handle_circle = UiButtonCircle {
                    id: Some("color picker handle circle".to_string()),
                    action: "None".to_string(),
                    style: "None".to_string(),
                    z_index: 990,
                    x: mouse_state.pos.x,
                    y: mouse_state.pos.y,
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
                loader.ui_runtime.last_pos = (mouse_state.pos.x, mouse_state.pos.y);
                loader.ui_runtime.original_radius = handle_circle.radius;
                let _ = loader.add_element(
                    "Editor_Menu",
                    "Color Picker",
                    Circle(handle_circle),
                    mouse_state,
                    false,
                );
            }
        }
    }

    if !loader.ui_runtime.selected_ui_element_multi.is_empty() {
        for color_element in loader.ui_runtime.selected_ui_element_multi.iter() {
            let menu = loader.menus.get_mut(&color_element.menu_name);
            if let Some(menu) = menu {
                if let Some(color) = color {
                    menu.change_element_color(
                        color_element.layer_name.as_str(),
                        color_element.element_id.as_str(),
                        color_element.element_type,
                        color,
                    );
                }
            }
        }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h6 = h * 6.0;
    let i = h6.floor();
    let f = h6 - i;

    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    if i == 0.0 {
        [v, t, p]
    } else if i == 1.0 {
        [q, v, p]
    } else if i == 2.0 {
        [p, v, t]
    } else if i == 3.0 {
        [p, q, v]
    } else if i == 4.0 {
        [t, p, v]
    } else {
        [v, p, q]
    }
}
