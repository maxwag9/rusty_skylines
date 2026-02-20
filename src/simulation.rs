use crate::commands::Command;
use crate::data::Settings;
use crate::helpers::mouse_ray::WorldRay;
use crate::renderer::gizmo::gizmo::Gizmo;
use crate::resources::Time;
use crate::ui::input::Input;
use crate::ui::variables::UiVariableRegistry;
use crate::world::camera::{Camera, CameraController};
use crate::world::cars::car_player::drive_car;
use crate::world::cars::car_subsystem::CarSubsystem;
use crate::world::roads::road_subsystem::RoadSubsystem;
use crate::world::terrain::terrain_subsystem::TerrainSubsystem;
use glam::Vec2;
use std::time::Instant;
use wgpu::{Device, Queue, SurfaceConfiguration};

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

    pub fn process_simulation_state_commands(&mut self, command: &Command) {
        match command {
            Command::ToggleSimulation => self.toggle(),
            _ => {}
        }
    }
    pub fn update(
        &mut self,
        terrain: &mut TerrainSubsystem,
        road_subsystem: &mut RoadSubsystem,
        car_subsystem: &mut CarSubsystem,
        settings: &Settings,
        time: &Time,
        input: &mut Input,
        variables: &mut UiVariableRegistry,
        camera: &mut Camera,
        cam_controller: &mut CameraController,
        device: &Device,
        queue: &Queue,
        config: &SurfaceConfiguration,
        gizmo: &mut Gizmo,
    ) {
        if !self.running {
            return;
        }

        self.tick += 1;
        self.last_update = Instant::now();
        update_picked_pos(terrain, camera, settings, config, input);
        let aspect = config.width as f32 / config.height as f32;

        car_subsystem.update(
            gizmo,
            &road_subsystem.road_manager,
            &terrain,
            input,
            time,
            variables,
            camera.target,
        );
        drive_car(
            car_subsystem,
            terrain,
            settings,
            input,
            cam_controller,
            camera,
            time.target_sim_dt,
        );
        //println!("Simulation finished: {}", time.frame_count);
    }
}

fn update_picked_pos(
    terrain_subsystem: &mut TerrainSubsystem,
    camera: &Camera,
    settings: &Settings,
    config: &SurfaceConfiguration,
    input_state: &Input,
) {
    let (view, proj, view_proj) = camera.matrices();
    let ray = WorldRay::from_mouse(
        Vec2::new(input_state.mouse.pos.x, input_state.mouse.pos.y),
        config.width as f32,
        config.height as f32,
        view,
        proj,
        camera.eye_world(),
        settings.chunk_size,
    );
    terrain_subsystem.pick_terrain_point(ray);
}

pub struct Ticker {
    interval: f32, // seconds per tick
    accumulator: f32,
}

impl Ticker {
    pub fn new(hz: f32) -> Self {
        Self {
            interval: 1.0 / hz,
            accumulator: 0.0,
        }
    }

    pub fn tick(&mut self, dt: f32) -> bool {
        self.accumulator += dt;
        if self.accumulator >= self.interval {
            self.accumulator -= self.interval;
            true
        } else {
            false
        }
    }
}
