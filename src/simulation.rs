use crate::renderer::ui_editor::UiButtonLoader;
use crate::state::State;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct Simulation {
    pub running: bool,
    pub tick: u64,
    last_update: Instant,
}

impl Simulation {
    pub fn new() -> Self {
        Self {
            running: true,
            tick: 0,
            last_update: Instant::now(),
        }
    }

    pub fn toggle(&mut self) {
        if self.running {
            self.stop()
        } else {
            self.start()
        }
        println!(
            "Simulation {}",
            if self.running { "started" } else { "paused" }
        );
    }

    pub fn start(&mut self) {
        self.running = true;
        self.last_update = Instant::now();
    }

    pub fn stop(&mut self) {
        self.running = false;
    }

    pub fn update(
        &mut self,
        dt: f32,
        ui_loader: &Arc<Mutex<UiButtonLoader>>,
        state: &Arc<Mutex<State>>,
    ) {
        {
            let mut ui_loader_lock = ui_loader.lock().unwrap();
            let mouse = &state.lock().unwrap().mouse;
            ui_loader_lock.handle_touches(&mouse, dt);
        }
        // up here update regardless of is the simulation running or not ^
        if !self.running {
            return;
        }
        // Down here update only if the simulation is running \/
        self.tick += 1;

        // TODO: Add city logic here
        //println!("Sim tick {} (dt = {:.3} s)", self.tick, dt);
    }
}
