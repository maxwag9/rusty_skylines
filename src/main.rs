#![allow(dead_code, unused_variables, private_interfaces)]

mod app;
mod commands;
mod data;
mod helpers;
mod renderer;
mod resources;
mod simulation;
mod systems;
mod ui;
pub mod world;

use crate::helpers::print_start_message;
use app::App;
use tracing_subscriber::FmtSubscriber;
use winit::event_loop::{ControlFlow, EventLoop};

fn main() {
    print_start_message();
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let subscriber = FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).unwrap();
    let app = App::new(
        #[cfg(target_arch = "wasm32")]
        &event_loop,
    );
    event_loop
        .run_app(app)
        .expect("Failed to run event loop application");
}
