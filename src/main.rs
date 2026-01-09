#![allow(dead_code, unused_variables)]

mod app;
mod components;
mod data;
mod events;
mod hsv;
mod mouse_ray;
mod paths;
mod renderer;
mod resources;
mod simulation;
mod systems;
mod ui;
mod world;

mod terrain;

use app::App;
use winit::event_loop::{ControlFlow, EventLoop};

fn main() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop
        .run_app(&mut app)
        .expect("Failed to run event loop application");
}
