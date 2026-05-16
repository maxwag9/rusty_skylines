use crate::data::Settings;
use crate::resources::Time;
use crate::ui::ui_editor::Ui;
use crate::ui::variables::update_ui_variables;
use crate::world::astronomy::{ObserverParams, TimeScales, compute_astronomy};
use crate::world::camera::{Camera, CameraController};
use glam::Mat4;

pub struct WorldState {
    pub camera: Camera,
    pub cam_controller: CameraController,
}

impl WorldState {
    pub fn new() -> Self {
        let camera: Camera = Camera::new();
        let cam_controller: CameraController = CameraController::new(&camera);
        let world = Self {
            camera,
            cam_controller,
        };
        world
    }

    pub fn update(&mut self, ui_loader: &mut Ui, time: &mut Time, settings: &Settings, proj: Mat4) {
        let time_scales =
            TimeScales::from_game_time(time.total_game_time, time.day_length, settings.always_day);
        let observer = ObserverParams::from_jd(time_scales.jd);
        time.astronomy = compute_astronomy(&time_scales, proj);

        update_ui_variables(
            ui_loader,
            &time_scales,
            &time.astronomy,
            observer.obliquity,
            settings,
        );
    }
}
