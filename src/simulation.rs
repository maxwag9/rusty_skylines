use std::time::{Duration, Instant};

pub struct Simulation {
    pub running: bool,
    pub tick: u64,
    last_update: Instant,
}

impl Simulation {
    pub fn new() -> Self {
        Self {
            running: false,
            tick: 0,
            last_update: Instant::now(),
        }
    }

    pub fn _toggle(&mut self) {
        self.running = !self.running;
        println!("Simulation {}", if self.running { "started" } else { "paused" });
    }

    pub fn start(&mut self) {
        self.running = true;
        self.last_update = Instant::now();
    }

    pub fn stop(&mut self) {
        self.running = false;
    }

    pub fn update(&mut self, dt: f32) {
        if !self.running {
            return;
        }
        self.tick += 1;
        // TODO: Add city logic here
        println!("Sim tick {} (dt = {:.3} s)", self.tick, dt);
    }
}
