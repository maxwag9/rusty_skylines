use crate::resources::TimeSystem;
use crate::ui::actions::deactivate_action;
use crate::ui::input::MouseState;
use crate::ui::touches::HitResult;
use crate::ui::ui_editor::UiElement::Circle;
use crate::ui::ui_editor::{MiscButtonSettings, UiButtonCircle, UiButtonLoader};
use glam::Vec2;
use std::f32::consts::{FRAC_PI_2, PI, TAU};

pub fn drag_hue_point(
    loader: &mut UiButtonLoader,
    mouse_state: &MouseState,
    top_hit: &Option<HitResult>,
    time: &TimeSystem,
) {
    let mut new_x = None;
    let mut new_y = None;
    let r_handle = loader.variables.get_f32("color_picker.r").unwrap_or(1.0);
    let g_handle = loader.variables.get_f32("color_picker.g").unwrap_or(1.0);
    let b_handle = loader.variables.get_f32("color_picker.b").unwrap_or(1.0);
    let mut color = None;
    let mut handle_color = Some([r_handle, g_handle, b_handle, 1.0]);
    let border_color = None;
    let mut radius = None;
    let mut hue_circle_radius = None;
    let dt = time.sim_target_step;

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

                let mx = pointer_x - hue.x;
                let my = pointer_y - hue.y;
                let angle = my.atan2(mx);
                let dist_mouse = (mx * mx + my * my).sqrt();
                let clamped_r = dist_mouse.min(hue.radius);

                let hx = hue.x + angle.cos() * clamped_r;
                let hy = hue.y + angle.sin() * clamped_r;

                new_x = Some(hx);
                new_y = Some(hy);

                // update HSV in variables
                let dx = hx - hue.x;
                let dy = hy - hue.y;

                let ang = dy.atan2(dx);
                let ang_shifted = ang + FRAC_PI_2;
                let ang_wrapped = ang_shifted.sin().atan2(ang_shifted.cos());

                let h = ang_wrapped / (PI * 2.0) + 0.5;

                let dist = (dx * dx + dy * dy).sqrt();
                let s = (dist / hue.radius).clamp(0.0, 1.0).powf(0.47);
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
                // ---------------------------------------------
                // NOT DRAGGING: HANDLE ANIMATES TO STORED COLOR
                // ---------------------------------------------
                if let Some(action_state) =
                    loader.ui_runtime.action_states.get_mut("Drag Hue Point")
                {
                    let h = loader.variables.get_f32("color_picker.h").unwrap_or(0.0);
                    let s = loader.variables.get_f32("color_picker.s").unwrap_or(1.0);

                    // target angle and radius
                    let target_angle = (h * TAU) - FRAC_PI_2;
                    let target_r = s.powf(1.0 / 0.47).clamp(0.0, 1.0)
                        * -action_state.original_radius_2.unwrap_or(hue.radius);

                    let target_x = hue.x + target_angle.cos() * target_r;
                    let target_y = hue.y + target_angle.sin() * target_r;

                    // smooth 100 ms = 0.1 s
                    let alpha = (dt / 0.2).clamp(0.0, 1.0);

                    let smoothed_x = lerp(handle.x, target_x, alpha);
                    let smoothed_y = lerp(handle.y, target_y, alpha);

                    new_x = Some(smoothed_x);
                    new_y = Some(smoothed_y);
                }
            }
        }

        loader.ui_runtime.action_states.get("Drag Hue Point");
    }
    let mut result = None;
    if color.is_some() {
        result = Some(loader.edit_circle(
            "Editor_Menu",
            "Color Picker",
            "color picker handle circle",
            new_x,
            new_y,
            radius,
            color,
            border_color,
        ));
    } else {
        result = Some(loader.edit_circle(
            "Editor_Menu",
            "Color Picker",
            "color picker handle circle",
            new_x,
            new_y,
            radius,
            handle_color,
            border_color,
        ));
    }

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
                // deactivate_action(loader, "Drag Hue Point");
                // let _ = loader.delete_element(
                //     "Editor_Menu",
                //     "Color Picker",
                //     "color picker handle circle",
                // );
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
        println!("deleting handle circle picker");
        let _ = loader.delete_element("Editor_Menu", "Color Picker", "color picker handle circle");
    }

    if let Some(action) = loader.ui_runtime.action_states.get("Drag Hue Point") {
        let active = action.active.clone();
        if let Some(result) = result {
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

pub fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    // Hue
    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        ((g - b) / delta) % 6.0
    } else if max == g {
        ((b - r) / delta) + 2.0
    } else {
        ((r - g) / delta) + 4.0
    };

    let h_norm = (h / 6.0).rem_euclid(1.0);

    // Saturation
    let s = if max == 0.0 { 0.0 } else { delta / max };

    // Value
    let v = max;

    (h_norm, s, v)
}
