use crate::commands::Command;
use crate::data::Settings;
use crate::helpers::mouse_ray::WorldRay;
use crate::renderer::render_core::Renderer;
use crate::resources::Time;
use crate::ui::input::Input;
use crate::ui::ui_editor::Ui;
use crate::world::buildings::buildings::Buildings;
use crate::world::camera::Camera;
use crate::world::cars::car_player::drive_car;
use crate::world::cars::car_subsystem::Cars;
use crate::world::roads::road_subsystem::Roads;
use crate::world::terrain::terrain_subsystem::Terrain;
use crate::world::world_state::WorldState;
use glam::Vec2;
use std::time::Instant;
use wgpu::SurfaceConfiguration;

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
        world_state: &mut WorldState,
        cars: &mut Cars,
        roads: &Roads,
        terrain: &Terrain,
        input: &mut Input,
        time: &Time,
        buildings: &mut Buildings,
        renderer: &mut Renderer,
        ui: &mut Ui,
        settings: &Settings,
    ) {
        if !self.running {
            return;
        }

        self.tick += 1;
        self.last_update = Instant::now();

        let camera = &mut world_state.camera;
        let cam_controller = &mut world_state.cam_controller;

        cars.update(
            &mut renderer.gizmo,
            &roads.road_manager,
            terrain,
            input,
            time,
            &mut ui.variables,
            camera.target,
        );

        drive_car(
            cars,
            terrain,
            settings,
            input,
            cam_controller,
            camera,
            time.target_sim_dt,
        );

        buildings
            .zoning
            .update_districts(time, &renderer.road_renderer.mesh_manager.road_edge_storage);
    }
}

pub fn update_picked_pos(
    terrain_subsystem: &mut Terrain,
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
    );
    terrain_subsystem.pick_terrain_point(ray);
}

#[derive(Clone)]
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
impl Default for Ticker {
    fn default() -> Self {
        Self::new(0.1)
    }
}
