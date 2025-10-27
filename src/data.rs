use crate::renderer::Renderer;
use crate::renderer::ui_editor::UiButtonLoader;
use crate::simulation::Simulation;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::{fs, path::Path};

#[derive(Clone)]
pub struct Data {
    pub renderer: Option<Arc<Mutex<Renderer>>>,
    pub simulation: Option<Arc<Mutex<Simulation>>>,
    pub ui_loader: Option<Arc<Mutex<UiButtonLoader>>>,
}

impl Data {
    pub fn empty() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            renderer: None,
            simulation: None,
            ui_loader: None,
        }))
    }

    pub fn set_cores(
        &mut self,
        renderer: Arc<Mutex<Renderer>>,
        simulation: Arc<Mutex<Simulation>>,
        ui_loader: Arc<Mutex<UiButtonLoader>>,
    ) {
        self.renderer = Some(renderer);
        self.simulation = Some(simulation);
        self.ui_loader = Some(ui_loader);
    }
}

pub type SharedData = Arc<Mutex<Data>>;

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
