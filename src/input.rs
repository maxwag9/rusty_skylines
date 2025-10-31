use std::collections::HashSet;

pub(crate) struct InputState {
    pub(crate) pressed: HashSet<String>, // store "w","a","s","d","q","e"
    pub shift_pressed: bool,
    pub ctrl_pressed: bool,
}

impl InputState {
    pub(crate) fn new() -> Self {
        Self {
            pressed: HashSet::new(),
            shift_pressed: false,
            ctrl_pressed: false,
        }
    }
    pub(crate) fn set_key(&mut self, key: &str, down: bool) {
        if down {
            self.pressed.insert(key.to_string());
        } else {
            self.pressed.remove(key);
        }
    }
}

pub(crate) struct MouseState {
    pub(crate) last_pos: Option<(f32, f32)>,
    pub(crate) pos_x: f32,
    pub(crate) pos_y: f32,
    pub(crate) middle_pressed: bool,
    pub(crate) left_pressed: bool,
    pub(crate) right_pressed: bool,
    pub(crate) back_pressed: bool,
    pub(crate) forward_pressed: bool,
}
