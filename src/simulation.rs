use crate::commands::Command;
use crate::data::Settings;
use crate::helpers::mouse_ray::WorldRay;
use crate::renderer::render_core::Renderer;
use crate::ui::input::Input;
use crate::ui::ui_editor::Ui;
use crate::world::buildings::zoning::Zoning;
use crate::world::camera::Camera;
use crate::world::cars::car_player::drive_car;
use crate::world::terrain::terrain_subsystem::Terrain;
use crate::world::world::World;
use glam::Vec2;
use std::time::Instant;
use wgpu::SurfaceConfiguration;

pub struct Simulation {
    pub tick: u64,
    last_update: Instant,
    speed: f32,
    pub old_speed: f32,
    temporary_speed: Option<f32>, // While a key is held
}

impl Simulation {
    pub fn new() -> Self {
        Self {
            tick: 0,
            last_update: Instant::now(),
            speed: 1.0,
            old_speed: 1.0,
            temporary_speed: None,
        }
    }

    pub fn toggle(&mut self) {
        if self.running() {
            self.stop();
        } else {
            self.start();
        }
    }
    fn print_toggle(&self) {
        println!(
            "Simulation {}",
            if self.running() { "started" } else { "paused" }
        );
    }
    pub fn start(&mut self) {
        if self.running() {
            return;
        }

        self.last_update = Instant::now();
        self.speed = self.old_speed.max(1.0);
        self.print_toggle();
    }

    pub fn stop(&mut self) {
        if !self.running() {
            return;
        }

        self.old_speed = self.speed;
        self.speed = 0.0;
        self.print_toggle();
    }
    pub fn set_running(&mut self, new_running: bool) {
        if new_running {
            self.start()
        } else {
            self.stop()
        }
    }
    pub fn set_speed_permanent(&mut self, speed: f32) {
        self.speed = speed;
    }
    pub fn set_speed_temporary(&mut self, speed: Option<f32>) {
        self.temporary_speed = speed;
    }
    pub fn running(&self) -> bool {
        self.speed > 0.0
    }
    pub fn speed(&self) -> f32 {
        self.temporary_speed.unwrap_or(self.speed)
    }
    pub fn process_simulation_state_commands(&mut self, command: &Command) {
        match command {
            Command::ToggleSimulation => self.toggle(),
            _ => {}
        }
    }
    pub fn update(
        &mut self,
        world: &mut World,
        renderer: &mut Renderer,
        ui: &mut Ui,
        settings: &Settings,
    ) {
        if !self.running() {
            return;
        }

        self.tick += 1;
        self.last_update = Instant::now();

        let camera = &mut world.world_state.camera;
        let cam_controller = &mut world.world_state.cam_controller;

        world.cars.update(
            &world.buildings.partitions,
            &world.zoning.zoning_storage,
            &mut renderer.gizmo,
            &world.roads.road_manager,
            &world.terrain,
            &mut world.input,
            &world.time,
            &mut ui.variables,
            camera.target,
        );

        drive_car(
            &mut world.cars,
            &world.terrain,
            settings,
            &mut world.input,
            cam_controller,
            camera,
            world.time.target_sim_dt,
        );
        world.buildings.storage.update(camera.target.chunk);
        Zoning::update_districts(
            &mut renderer.gizmo,
            &world.terrain,
            &mut world.zoning,
            &mut world.buildings,
            world.cars.car_storage_mut(),
            &mut world.city_state,
            &world.time,
            &world.roads.road_manager.roads,
            &renderer.road_renderer.mesh_manager.road_edge_storage,
            camera.target,
        );
        world
            .city_state
            .update(&world.time, &mut world.zoning, &world.buildings);
        world.buildings.utilities.check_network(&mut world.roads);
    }
}

pub fn update_picked_pos(
    terrain: &mut Terrain,
    camera: &Camera,
    settings: &Settings,
    config: &SurfaceConfiguration,
    input: &Input,
    ui: &Ui,
) {
    if !settings.show_world || ui.touch_manager.hovered().is_some() {
        terrain.last_picked = None;
        return;
    }

    let (view, proj, view_proj) = camera.matrices();
    let ray = WorldRay::from_mouse(
        Vec2::new(input.mouse.pos.x, input.mouse.pos.y),
        config.width as f32,
        config.height as f32,
        view,
        proj,
        camera.eye_world(),
    );
    terrain.pick_terrain_point(ray);
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
