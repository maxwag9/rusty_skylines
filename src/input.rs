use std::collections::HashSet;

pub(crate) struct InputState {
    pub(crate) pressed: HashSet<String>, // store "w","a","s","d","q","e"
}

impl InputState {
    pub(crate) fn new() -> Self {
        Self {
            pressed: HashSet::new(),
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
    pub(crate) last_pos: Option<(f64, f64)>,
    pub(crate) dragging: bool,
}
