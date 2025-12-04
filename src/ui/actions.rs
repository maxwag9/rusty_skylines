use crate::resources::{InputState, TimeSystem};
use crate::ui::input::MouseState;
use crate::ui::special_actions::drag_hue_point;
use crate::ui::touches::HitResult;
use crate::ui::ui_editor::UiButtonLoader;
use glam::Vec2;

pub fn style_to_u32(style: &str) -> u32 {
    match style {
        "Hue Circle" => 1,
        "None" => 0,
        &_ => 0,
    }
}

pub fn execute_action(
    loader: &mut UiButtonLoader,
    top_hit: &Option<HitResult>,
    mouse_state: &MouseState,
    time: &TimeSystem,
) {
    let actions: Vec<String> = loader.ui_runtime.action_states.keys().cloned().collect();

    for action in actions {
        match action.as_str() {
            "Drag Hue Point" => {
                drag_hue_point(loader, mouse_state, top_hit, time);
            }
            "None" => {}
            _ => {}
        }
    }
}

pub fn activate_action(
    loader: &mut UiButtonLoader,
    top_hit: &Option<HitResult>,
    input_state: &InputState,
) {
    if let Some(hit) = top_hit {
        let action = hit.action.clone().unwrap_or("None".to_string());

        match action.as_str() {
            "Drag Hue Point" => {
                if let Some(action_state) =
                    loader.ui_runtime.action_states.get_mut("Drag Hue Point")
                {
                    action_state.active = true;
                } else {
                    let action_state = ActionState {
                        action_name: "Drag Hue Point".to_string(),
                        position: None,
                        last_pos: None,
                        radius: None,
                        original_radius_1: None,
                        original_radius_2: None,
                        active: true,
                    };
                    loader
                        .ui_runtime
                        .action_states
                        .insert("Drag Hue Point".to_string(), action_state);
                }
            }
            // "Toggle Color Picker" => {
            //     if !input_state.mouse.right_just_pressed {
            //         return;
            //     }
            //     if let Some(menu) = loader.menus.get_mut("Editor_Menu"){
            //         if let Some(layer) = menu.layers.iter_mut().find(|l| l.name == "Color Picker") {
            //             layer.active = !layer.active;
            //         }
            //     }
            // }
            "None" => {}
            &_ => {}
        }
    }
}

pub fn deactivate_action(loader: &mut UiButtonLoader, action_name: &str) {
    if let Some(action_state) = loader.ui_runtime.action_states.get_mut(action_name) {
        action_state.active = false;
    }
}

pub fn selected_needed(loader: &UiButtonLoader, action_name: &str) -> bool {
    match action_name {
        "Drag Hue Point" => true,
        "None" => false,
        _ => false,
    }
}

#[derive(Debug)]
pub struct ActionState {
    pub(crate) action_name: String,
    pub(crate) position: Option<Vec2>,
    pub(crate) last_pos: Option<Vec2>,
    pub(crate) radius: Option<f32>,
    pub(crate) original_radius_1: Option<f32>,
    pub(crate) original_radius_2: Option<f32>,

    pub active: bool,
}
