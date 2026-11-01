#![allow(dead_code, unused_variables, private_interfaces, unused_imports)]

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

use crate::app::GAME_VERSION;
use crate::helpers::{get_start_message, print_exit_message};
use app::App;
use tracing_subscriber::FmtSubscriber;
use winit::event_loop::{ControlFlow, EventLoop};

fn main() {
    let start_message = get_start_message();
    println!("{}", start_message);
    println!("Rusty Skylines Alpha v{}", GAME_VERSION);
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
    print_exit_message()
}
