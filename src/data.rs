use crate::renderer::Renderer;
use crate::renderer::ui_editor::UiButtonLoader;
use crate::simulation::Simulation;
use std::sync::{Arc, Mutex};

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
