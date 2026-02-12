use crate::cars::car_player::drive_car;
use crate::cars::car_subsystem::CarSubsystem;
use crate::data::Settings;
use crate::events::{Event, Events};
use crate::renderer::world_renderer::TerrainRenderer;
use crate::resources::TimeSystem;
use crate::terrain::roads::road_subsystem::RoadRenderSubsystem;
use crate::ui::input::InputState;
use crate::world::CameraBundle;
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
                Event::SetCursorMode(mode) => {}
            }
        }
    }
    pub fn process_event(&mut self, event: &Event) {
        match event {
            Event::ToggleSimulation => self.toggle(),
            _ => {}
        }
    }
    pub fn update(
        &mut self,
        terrain_renderer: &TerrainRenderer,
        road_renderer: &RoadRenderSubsystem,
        car_subsystem: &mut CarSubsystem,
        settings: &Settings,
        time: &TimeSystem,
        input: &mut InputState,
        camera_bundle: &mut CameraBundle,
    ) {
        if !self.running {
            return;
        }

        self.tick += 1;
        self.last_update = Instant::now();
        let camera = &mut camera_bundle.camera;
        let cam_ctrl = &mut camera_bundle.controller;
        car_subsystem.update(
            &road_renderer.road_manager,
            &terrain_renderer,
            input,
            time,
            camera.target,
        );
        drive_car(
            car_subsystem,
            terrain_renderer,
            settings,
            input,
            cam_ctrl,
            camera,
            time.target_sim_dt,
        );

        // TODO: Add simulation logic here
    }
}
