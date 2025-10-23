use egui::{Button, Response, RichText, Ui, Vec2};

use crate::simulation_controls::SimulationSpeed;

pub fn play_pause_button(ui: &mut Ui, running: bool) -> Response {
    let label = if running { "Pause" } else { "Start" };
    let text = RichText::new(label).strong();
    ui.add(Button::new(text).min_size(Vec2::new(68.0, 28.0)))
}

pub fn speed_button(ui: &mut Ui, speed: SimulationSpeed, active: bool) -> Response {
    let mut text = RichText::new(speed.to_string());
    if active {
        text = text.strong();
    }

    ui.add(Button::new(text).min_size(Vec2::new(44.0, 28.0)))
}
