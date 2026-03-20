use crate::data::Settings;
use crate::resources::Time;
use crate::ui::ui_editor::Ui;
use crate::ui::variables::update_ui_variables;
use crate::world::astronomy::{Astronomy, ObserverParams, TimeScales, compute_astronomy};
use crate::world::camera::{Camera, CameraController};
use crate::world::game_state::GameState;
use glam::Mat4;

pub struct WorldState {
    pub camera: Camera,
    pub cam_controller: CameraController,
    pub astronomy: Astronomy,
    pub game_state: GameState,
}

impl WorldState {
    pub fn new() -> Self {
        let camera: Camera = Camera::new();
        let cam_controller: CameraController = CameraController::new(&camera);
        let world = Self {
            camera,
            cam_controller,
            astronomy: Astronomy::default(),
            game_state: GameState::default(),
        };
        world
    }

    pub fn update(&mut self, ui_loader: &mut Ui, time: &Time, settings: &Settings, proj: Mat4) {
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
