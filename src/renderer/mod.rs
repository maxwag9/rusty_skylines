pub mod core;
pub mod helper;
pub mod pipelines;
pub mod render_passes;
mod ui;
pub mod ui_editor;

use crate::components::camera::Camera;
use crate::renderer::ui_editor::UiButtonLoader;
use crate::resources::{MouseState, TimeSystem};
use core::RenderCore;
use std::sync::Arc;

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
        ui_loader: &mut UiButtonLoader,
        time: &TimeSystem,
        mouse: &MouseState,
    ) {
        self.core.render(camera, ui_loader, time, mouse);
    }
}
