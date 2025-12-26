pub(crate) mod astronomy;
mod benchmark;
pub(crate) mod core;
pub(crate) mod mesh_arena;
pub(crate) mod pipelines;
pub(crate) mod pipelines_outsource;
pub(crate) mod render_passes;
pub(crate) mod shader_watcher;
pub(crate) mod textures;
pub(crate) mod ui;
pub(crate) mod ui_pipelines;
pub(crate) mod ui_text;
pub(crate) mod ui_upload;
pub(crate) mod uniform_updates;
pub(crate) mod world_renderer;

use crate::components::camera::Camera;
use crate::data::Settings;
use crate::resources::{InputState, TimeSystem};
use crate::ui::ui_editor::UiButtonLoader;
use crate::world::CameraBundle;
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
        camera_bundle: &mut CameraBundle,
        ui_loader: &mut UiButtonLoader,
        time: &TimeSystem,
        input_state: &mut InputState,
        settings: &Settings,
    ) {
        self.core
            .render(camera_bundle, ui_loader, time, input_state, settings);
    }
}
