pub mod core;
pub mod pipelines;
pub mod render_passes;
mod ui;
pub mod ui_editor;

use crate::camera::Camera;
use crate::data::SharedData;
use core::RenderCore;
use std::sync::Arc;

pub struct Renderer {
    pub core: RenderCore,
}

impl Renderer {
    pub fn new(window: Arc<winit::window::Window>, data: SharedData) -> Self {
        let core = RenderCore::new(window, data.clone());
        Self { core }
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.core.resize(size);
    }

    pub fn render(&mut self, camera: &Camera) {
        self.core.render(camera);
    }
}
