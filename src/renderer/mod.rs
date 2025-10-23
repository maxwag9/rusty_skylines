pub mod core;
pub mod pipelines;
pub mod render_passes;

use crate::camera::Camera;
use crate::ui::UiSystem;
use core::RenderCore;
use std::sync::Arc;
use winit::window::Window;

pub struct Renderer {
    pub core: RenderCore,
}

impl Renderer {
    pub async fn new(window: Arc<winit::window::Window>) -> Self {
        let core = RenderCore::new(window).await;
        Self { core }
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.core.resize(size);
    }

    pub fn render(&mut self, camera: &Camera, window: &Window, ui: &mut UiSystem) {
        self.core.render(camera, window, ui);
    }
}
