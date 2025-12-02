use crate::ui::actions::ActionState;
use crate::ui::ui_editor::{ButtonRuntime, SelectedUiElement, TouchState};
use std::collections::HashMap;

#[derive(Debug)]
pub struct UiRuntime {
    pub elements: HashMap<String, ButtonRuntime>,
    pub selected_ui_element_primary: SelectedUiElement,
    pub selected_ui_element_multi: Vec<SelectedUiElement>,
    pub active_vertex: Option<usize>,
    pub drag_offset: Option<(f32, f32)>,
    pub editor_mode: bool,
    pub editing_text: bool,
    pub clipboard: String,
    pub dragging_text: bool,
    pub last_pos: (f32, f32),
    pub original_radius: f32,
    pub action_states: HashMap<String, ActionState>,
}

impl UiRuntime {
    pub fn new(editor_mode: bool) -> Self {
        Self {
            elements: HashMap::new(),
            selected_ui_element_primary: SelectedUiElement::default(),
            selected_ui_element_multi: vec![],
            active_vertex: None,
            drag_offset: None,
            editor_mode,
            editing_text: false,
            clipboard: "".to_string(),
            dragging_text: false,
            last_pos: (0.0, 0.0),
            original_radius: 0.0,
            action_states: HashMap::new(),
        }
    }

    pub fn update_touch(
        &mut self,
        id: &str,
        touched_now: bool,
        dt: f32,
        _layer_name: &String,
    ) -> TouchState {
        let entry = self
            .elements
            .entry(id.to_string())
            .or_insert_with(ButtonRuntime::default);

        entry.just_pressed = false;
        entry.just_released = false;

        match (entry.is_down, touched_now) {
            (false, true) => {
                entry.is_down = true;
                entry.just_pressed = true;
                entry.touched_time = 0.0;
                TouchState::Pressed
            }
            (true, true) => {
                entry.touched_time += dt;
                TouchState::Held
            }
            (true, false) => {
                entry.is_down = false;
                entry.just_released = true;
                TouchState::Released
            }
            (false, false) => TouchState::Idle,
        }
    }

    pub fn update_editor_mode(&mut self, editor_mode: bool) {
        self.editor_mode = editor_mode;
    }

    pub fn get(&self, id: &str) -> ButtonRuntime {
        *self.elements.get(id).unwrap_or(&ButtonRuntime::default())
    }
}
