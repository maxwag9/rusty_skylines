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
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")] // "fifo", "mailbox", ...
pub enum PresentModeSetting {
    Immediate,
    Mailbox,
    Fifo,
}

impl PresentModeSetting {
    pub fn to_wgpu(self) -> PresentMode {
        match self {
            PresentModeSetting::Immediate => PresentMode::Immediate,
            PresentModeSetting::Mailbox => PresentMode::Mailbox,
            PresentModeSetting::Fifo => PresentMode::Fifo,
        }
    }
}

impl Default for PresentModeSetting {
    fn default() -> Self {
        PresentModeSetting::Mailbox
    }
}

#[derive(Debug, Deserialize)]
struct PartialSettings {
    target_fps: Option<f32>,
    target_tps: Option<f32>,
    present_mode: Option<PresentModeSetting>,
    editor_mode: Option<bool>,
    override_mode: Option<bool>,
    show_gui: Option<bool>,
    background_color: Option<[f32; 4]>,
    total_game_time: Option<f64>,
    world_generation_benchmark_mode: Option<bool>,
    bend_mode: Option<BendMode>,
    show_world: Option<bool>,
    always_day: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(default)]
pub struct Settings {
    pub target_fps: f32,
    pub target_tps: f32,

    pub present_mode: PresentModeSetting,
    pub editor_mode: bool,
    pub override_mode: bool,
    pub show_gui: bool,
    pub background_color: [f32; 4],
    pub total_game_time: f64,
    pub world_generation_benchmark_mode: bool,
    pub bend_mode: BendMode,
    pub show_world: bool,
    pub always_day: bool,
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
        }
    }
}

impl Settings {
    pub fn load<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref();

        match fs::read_to_string(path) {
            Ok(content) => match toml::from_str::<PartialSettings>(&content) {
                Ok(partial) => Settings::from_partial(partial),
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
    fn from_partial(p: PartialSettings) -> Self {
        let mut s = Self::default();

        if let Some(v) = p.target_fps {
            s.target_fps = v;
        }
        if let Some(v) = p.target_tps {
            s.target_tps = v;
        }
        if let Some(v) = p.present_mode {
            s.present_mode = v;
        }
        if let Some(v) = p.editor_mode {
            s.editor_mode = v;
        }
        if let Some(v) = p.override_mode {
            s.override_mode = v;
        }
        if let Some(v) = p.show_gui {
            s.show_gui = v;
        }
        if let Some(v) = p.background_color {
            s.background_color = v;
        }
        if let Some(v) = p.total_game_time {
            s.total_game_time = v;
        }
        if let Some(v) = p.world_generation_benchmark_mode {
            s.world_generation_benchmark_mode = v;
        }
        if let Some(v) = p.bend_mode {
            s.bend_mode = v;
        }
        if let Some(v) = p.show_world {
            s.show_world = v;
        }
        if let Some(v) = p.always_day {
            s.always_day = v;
        }
        s
    }
}
