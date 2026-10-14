use crate::commands::CommandBuffer;
use crate::data::Settings;
use crate::helpers::modpack::ModManager;
use crate::helpers::positions::chunk_size;
use crate::renderer::props::Props;
use crate::resources::{Time, TimeChangeOperation};
use crate::ui::input::Input;
use crate::world::buildings::buildings::Buildings;
use crate::world::buildings::zoning::Zoning;
use crate::world::camera::CameraMode;
use crate::world::cars::car_subsystem::Cars;
use crate::world::game_state::GameState;
use crate::world::roads::road_subsystem::Roads;
use crate::world::sound::sound::Sounds;
use crate::world::statisticals::CityState;
use crate::world::terrain::terrain_subsystem::Terrain;
use crate::world::world_state::WorldState;
use glam::Vec3;
use rand::RngExt;
use rand::rngs::ThreadRng;
use wgpu::{Device, Queue};

pub struct World {
    pub world_state: WorldState,
    pub time: Time,
    pub input: Input,
    pub events: CommandBuffer, // main-thread swap/flips, core consumes on sim tick
    pub terrain: Terrain,
    pub roads: Roads,
    pub cars: Cars,
    pub buildings: Buildings,
    pub zoning: Zoning,
    pub city_state: CityState, // ... other sim-only subsystems (economy, citizens, etc.)
    pub sounds: Sounds,
    pub panorama: bool,
}

impl World {
    pub fn new(
        device: &Device,
        queue: &Queue,
        settings: &Settings,
        props: &Props,
        mod_manager: &ModManager,
    ) -> Self {
        let world_state = WorldState::new();
        let terrain = Terrain::new(device, queue, settings, props);
        Self {
            world_state,
            time: Time::new(),
            input: Input::new(),
            terrain,
            roads: Roads::new(),
            cars: Cars::new(),
            buildings: Buildings::new(mod_manager),
            zoning: Zoning::new(),
            events: CommandBuffer::new(),
            city_state: CityState::new(),
            sounds: Sounds::new(mod_manager),
            panorama: false,
        }
    }
    pub fn recreate(&mut self, settings: &Settings, props: &Props, mod_manager: &ModManager) {
        *self = World::new(
            &self.terrain.device.clone(),
            &self.terrain.queue.clone(),
            settings,
            props,
            mod_manager,
        )
    }

    pub fn start_panorama(&mut self, game_state: &GameState) {
        println!("Starting panorama");
        if self.panorama {
            return;
        }
        if game_state.current_save_info.is_some() {
            return;
        };
        //self.world_state.camera.set_mode(CameraMode::FirstPerson);
        self.panorama = true;
        self.world_state.cam_controller.target_yaw = 90.0_f32.to_radians();
        self.world_state.cam_controller.target_pitch = 15.0_f32.to_radians();
        self.world_state.camera.yaw = 90.0_f32.to_radians();
        self.world_state.camera.pitch = 15.0_f32.to_radians();
        let mut rng = ThreadRng::default();
        self.time.change_game_time(TimeChangeOperation::Set(
            self.time.day_length * rng.random_range(0.1..0.8),
        ));
        let hc = chunk_size() as f32 * 0.5;
        self.world_state.camera.target = self.world_state.camera.target.add_vec3(Vec3::new(
            rng.random_range(-hc..hc),
            rng.random_range(0.0..100.0),
            rng.random_range(-hc..hc),
        ));
    }
    pub fn end_panorama(&mut self) {
        println!("Ending panorama");
        if !self.panorama {
            return;
        }
        self.world_state.camera.set_mode(CameraMode::Orbit);
        self.panorama = false
    }

    pub fn update_panorama(&mut self, game_state: &GameState) {
        if !self.panorama {
            return;
        };
        if game_state.current_save_info.is_some() {
            return;
        };

        //self.world_state.camera.yaw += 0.1*self.time.render_dt;
        self.world_state.cam_controller.target_yaw += 1.0_f32.to_radians() * self.time.render_dt;
        //self.world_state.camera.target.local.y = self.
        self.terrain.last_picked = None;
    }
}
