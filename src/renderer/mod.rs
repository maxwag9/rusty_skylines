pub mod core;
pub mod pipelines;
pub mod render_passes;
pub(crate) mod shader_watcher;
pub(crate) mod ui;
pub(crate) mod ui_pipelines;
mod world_renderer;

use crate::components::camera::Camera;
use crate::data::Settings;
use crate::resources::TimeSystem;
use crate::ui::input::MouseState;
use crate::ui::ui_editor::UiButtonLoader;
use core::RenderCore;
use std::sync::Arc;

pub struct Renderer {
    pub core: RenderCore,
}

impl Renderer {
    pub fn new(window: Arc<winit::window::Window>, settings: &Settings) -> Self {
        let core = RenderCore::new(window, settings);
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
        settings: &Settings,
    ) {
        self.core.render(camera, ui_loader, time, mouse, settings);
    }
}
