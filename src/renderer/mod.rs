pub mod core;
pub(crate) mod mesh_arena;
pub mod pipelines;
pub mod render_passes;
pub(crate) mod shader_watcher;
pub(crate) mod textures;
pub(crate) mod ui;
pub(crate) mod ui_pipelines;
pub(crate) mod world_renderer;

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
    pub fn new(window: Arc<winit::window::Window>, settings: &Settings, camera: &Camera) -> Self {
        let core = RenderCore::new(window, settings, camera);
        Self { core }
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.core.resize(size);
    }

    pub fn render(
        &mut self,
        camera: &mut Camera,
        ui_loader: &mut UiButtonLoader,
        time: &TimeSystem,
        mouse: &MouseState,
        settings: &Settings,
    ) {
        self.core.render(camera, ui_loader, time, mouse, settings);
    }
}
