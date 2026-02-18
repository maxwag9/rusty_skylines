pub struct GameState {
    current_save: Option<SaveState>,
}

impl GameState {
    pub fn new() -> Self {
        Self { current_save: None }
    }
}

impl Default for GameState {
    fn default() -> Self {
        let mut save = Self::new();
        save.current_save = Some(SaveState::new(128));
        save
    }
}

pub struct SaveState {
    pub chunk_size: usize,
}

impl SaveState {
    pub fn new(chunk_size: usize) -> Self {
        Self { chunk_size }
    }
}
