use crate::cars::car_player::drive_car;
use crate::cars::car_subsystem::CarSubsystem;
use crate::data::Settings;
use crate::events::{Event, Events};
use crate::helpers::mouse_ray::WorldRay;
use crate::renderer::gizmo::Gizmo;
use crate::renderer::terrain_subsystem::TerrainSubsystem;
use crate::resources::TimeSystem;
use crate::terrain::roads::road_subsystem::RoadSubsystem;
use crate::ui::input::InputState;
use crate::world::camera::Camera;
use crate::world::world::CameraBundle;
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

    pub fn process_events(&mut self, events: &mut Events) {
        for event in events.drain() {
            match event {
                Event::ToggleSimulation => self.toggle(),
                Event::SetCursorMode(mode) => {}
                _ => {}
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
        terrain_subsystem: &mut TerrainSubsystem,
        road_subsystem: &mut RoadSubsystem,
        car_subsystem: &mut CarSubsystem,
        settings: &Settings,
        time: &TimeSystem,
        input: &mut InputState,
        camera_bundle: &mut CameraBundle,
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
        let camera = &mut camera_bundle.camera;
        update_picked_pos(terrain_subsystem, camera, settings, config, input);
        let aspect = config.width as f32 / config.height as f32;
        terrain_subsystem.update(device, queue, camera, aspect, settings, input, time);
        let cam_ctrl = &mut camera_bundle.controller;
        road_subsystem.update(terrain_subsystem, car_subsystem, input, gizmo);
        car_subsystem.update(
            &road_subsystem.road_manager,
            &terrain_subsystem,
            input,
            time,
            camera.target,
        );
        drive_car(
            car_subsystem,
            terrain_subsystem,
            settings,
            input,
            cam_ctrl,
            camera,
            time.target_sim_dt,
        );
    }
}

fn update_picked_pos(
    terrain_subsystem: &mut TerrainSubsystem,
    camera: &Camera,
    settings: &Settings,
    config: &SurfaceConfiguration,
    input_state: &InputState,
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
