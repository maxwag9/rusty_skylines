use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")] // "fifo", "mailbox", ...
pub enum PresentModeSetting {
    Immediate,
    Mailbox,
    Fifo,
}

impl PresentModeSetting {
    pub fn to_wgpu(self) -> wgpu::PresentMode {
        match self {
            PresentModeSetting::Immediate => wgpu::PresentMode::Immediate,
            PresentModeSetting::Mailbox => wgpu::PresentMode::Mailbox,
            PresentModeSetting::Fifo => wgpu::PresentMode::Fifo,
        }
    }
}

impl Default for PresentModeSetting {
    fn default() -> Self {
        PresentModeSetting::Mailbox
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Settings {
    pub target_fps: f32,
    pub target_tps: f32,

    pub present_mode: PresentModeSetting,
    pub editor_mode: bool,
    pub background_color: [f32; 4],
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            target_fps: 60.0,
            target_tps: 60.0,
            present_mode: PresentModeSetting::Mailbox,
            editor_mode: false,
            background_color: [0.0, 0.0, 0.0, 1.0],
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
