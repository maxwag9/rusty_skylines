use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Settings {
    pub target_fps: f32,
    pub present_mode: String, // "Fifo", "Mailbox", or "Immediate"
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            target_fps: 60.0,
            present_mode: "Mailbox".to_string(),
        }
    }
}

impl Settings {
    pub fn load<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref();
        if let Ok(content) = fs::read_to_string(path) {
            match toml::from_str::<Self>(&content) {
                Ok(cfg) => cfg,
                Err(e) => {
                    eprintln!("Error parsing {:?}: {e}", path);
                    Self::default()
                }
            }
        } else {
            eprintln!("No settings file found, creating default {:?}", path);
            let default = Self::default();
            let toml_str = toml::to_string_pretty(&default).unwrap();
            let _ = fs::write(path, toml_str);
            default
        }
    }
}
