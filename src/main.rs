mod app;
mod chunk_builder;
mod components;
mod data;
mod events;
mod paths;
mod renderer;
mod resources;
mod simulation;
mod systems;
mod terrain;
mod threads;
mod ui;
mod world;

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
