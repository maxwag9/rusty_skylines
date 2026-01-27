use crate::positions::ChunkSize;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};
use wgpu::*;

/// Mode switch: Strict attempts to deserialize the file as normal JSON into GuiLayout.
/// Bent ignores JSON structure and deterministically synthesizes a GuiLayout from the file bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BendMode {
    Strict,
    Bent,
    #[serde(other)]
    Unknown,
}
impl Default for BendMode {
    fn default() -> Self {
        BendMode::Strict
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")] // "fifo", "mailbox", ...
pub enum PresentModeSetting {
    Immediate,
    Mailbox,
    Fifo,
    #[serde(other)]
    Unknown,
}

impl PresentModeSetting {
    pub fn to_wgpu(self) -> PresentMode {
        match self {
            PresentModeSetting::Immediate => PresentMode::Immediate,
            PresentModeSetting::Mailbox => PresentMode::Mailbox,
            PresentModeSetting::Fifo => PresentMode::Fifo,
            _ => PresentMode::Mailbox,
        }
    }
}

impl Default for PresentModeSetting {
    fn default() -> Self {
        PresentModeSetting::Mailbox
    }
}
fn default_chunk_size() -> ChunkSize {
    128
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(default)]
pub struct Settings {
    #[serde(default)]
    pub target_fps: f32,
    #[serde(default)]
    pub target_tps: f32,
    #[serde(default)]
    pub present_mode: PresentModeSetting,
    #[serde(default)]
    pub editor_mode: bool,
    #[serde(default)]
    pub override_mode: bool,
    #[serde(default)]
    pub show_gui: bool,
    #[serde(default)]
    pub background_color: [f32; 4],
    #[serde(default)]
    pub total_game_time: f64,
    #[serde(default)]
    pub world_generation_benchmark_mode: bool,
    #[serde(default)]
    pub bend_mode: BendMode,
    #[serde(default)]
    pub show_world: bool,
    #[serde(default)]
    pub always_day: bool,
    #[serde(default)]
    pub msaa_samples: u32,
    #[serde(default)]
    pub shadow_map_size: u32,
    #[serde(default)]
    pub shadows_enabled: bool,
    #[serde(default)]
    pub zoom_speed: f32,
    #[serde(default)]
    pub render_lanes_gizmo: bool,
    #[serde(default)]
    pub render_chunk_bounds: bool,
    #[serde(default = "default_chunk_size")]
    pub chunk_size: ChunkSize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            target_fps: 60.0,
            target_tps: 60.0,
            present_mode: PresentModeSetting::Mailbox,
            editor_mode: false,
            override_mode: false,
            show_gui: true,
            background_color: [0.0, 0.0, 0.0, 1.0],
            total_game_time: 0.0,
            world_generation_benchmark_mode: false,
            bend_mode: BendMode::Strict,
            show_world: true,
            always_day: false,
            msaa_samples: 4,
            shadow_map_size: 4096,
            shadows_enabled: true,
            zoom_speed: 10.0,
            render_lanes_gizmo: false,
            render_chunk_bounds: false,
            chunk_size: default_chunk_size(),
        }
    }
}

impl Settings {
    pub fn load<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref();

        match fs::read_to_string(path) {
            Ok(content) => match toml::from_str::<Settings>(&content) {
                Ok(settings) => settings,
                Err(err) => {
                    eprintln!("Error parsing {:?}: {err}", path);
                    Self::default()
                }
            },
            Err(_) => {
                eprintln!("No settings file found, creating default {:?}", path);
                let default = Self::default();
                if let Ok(toml_str) = toml::to_string_pretty(&default) {
                    let _ = fs::write(path, toml_str);
                }
                default
            }
        }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let path = path.as_ref();

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let toml_str = toml::to_string_pretty(self)?;
        fs::write(path, toml_str)?;

        Ok(())
    }
}
