use crate::ui::actions::ActionState;
use crate::ui::vertex::*;
use std::collections::HashMap;

#[derive(Debug)]
pub struct UiRuntimes {
    pub elements: HashMap<String, ButtonRuntime>,
    pub action_states: HashMap<String, ActionState>,
}

impl UiRuntimes {
    pub fn new() -> Self {
        Self {
            elements: HashMap::new(),
            action_states: HashMap::new(),
        }
    }

    pub fn update_touch(&mut self, id: &str, touched_now: bool, dt: f32) -> TouchState {
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

    pub fn get(&self, id: &str) -> ButtonRuntime {
        *self.elements.get(id).unwrap_or(&ButtonRuntime::default())
    }
}
