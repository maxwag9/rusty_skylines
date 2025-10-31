use crate::events::{Event, Events};
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
            self.stop();
        } else {
            self.start();
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

    pub fn process_events(&mut self, events: &mut Events) {
        for event in events.drain() {
            match event {
                Event::ToggleSimulation => self.toggle(),
            }
        }
    }

    pub fn update(&mut self, _dt: f32) {
        if !self.running {
            return;
        }

        self.tick += 1;
        self.last_update = Instant::now();
        // TODO: Add simulation logic here
    }
}
