#![allow(dead_code, unused_variables)]
extern crate core;

mod app;
mod cars;
mod data;
mod events;
mod helpers;
mod renderer;
mod resources;
mod simulation;
mod systems;
mod terrain;
mod ui;
pub mod world;

use app::App;
use winit::event_loop::{ControlFlow, EventLoop};

fn main() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(
        #[cfg(target_arch = "wasm32")]
        &event_loop,
    );
    event_loop
        .run_app(&mut app)
        .expect("Failed to run event loop application");
}
