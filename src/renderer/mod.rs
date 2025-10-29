pub mod core;
pub mod pipelines;
pub mod render_passes;
mod ui;
pub mod ui_editor;

use crate::app::TimingData;
use crate::camera::Camera;
use crate::renderer::ui_editor::UiButtonLoader;
use core::RenderCore;
use std::sync::{Arc, Mutex};

pub struct Renderer {
    pub core: RenderCore,
}

impl Renderer {
    pub fn new(window: Arc<winit::window::Window>) -> Self {
        let core = RenderCore::new(window);
        Self { core }
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.core.resize(size);
    }

    pub fn render(
        &mut self,
        camera: &Camera,
        ui_loader: &UiButtonLoader,
        timing_data: Arc<Mutex<TimingData>>,
    ) {
        self.core.render(camera, ui_loader, timing_data);
    }
}
