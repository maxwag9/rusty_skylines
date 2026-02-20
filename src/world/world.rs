use crate::data::Settings;
use crate::resources::Time;
use crate::ui::ui_editor::UiButtonLoader;
use crate::ui::variables::update_ui_variables;
use crate::world::astronomy::{AstronomyState, ObserverParams, TimeScales, compute_astronomy};
use crate::world::camera::{Camera, CameraController};
use glam::Mat4;

pub struct WorldState {
    pub camera: Camera,
    pub cam_controller: CameraController,
    pub astronomy: AstronomyState,
}

impl WorldState {
    pub fn new() -> Self {
        let camera: Camera = Camera::new();
        let cam_controller: CameraController = CameraController::new(&camera);
        let world = Self {
            camera,
            cam_controller,
            astronomy: AstronomyState::default(),
        };
        world
    }

    pub fn update(
        &mut self,
        ui_loader: &mut UiButtonLoader,
        time: &Time,
        settings: &Settings,
        proj: Mat4,
    ) {
        let time_scales = TimeScales::from_game_time(time.total_game_time, settings.always_day);
        let observer = ObserverParams::new(time_scales.day_angle);
        let astronomy = compute_astronomy(&time_scales, proj);

        update_ui_variables(
            ui_loader,
            &time_scales,
            &astronomy,
            observer.obliquity,
            settings,
        );
        self.astronomy = astronomy;
    }
}
