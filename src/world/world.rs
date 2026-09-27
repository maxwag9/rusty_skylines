use crate::commands::CommandBuffer;
use crate::data::Settings;
use crate::helpers::modpack::ModManager;
use crate::renderer::props::Props;
use crate::resources::Time;
use crate::ui::input::Input;
use crate::world::buildings::buildings::Buildings;
use crate::world::buildings::zoning::Zoning;
use crate::world::cars::car_subsystem::Cars;
use crate::world::roads::road_subsystem::Roads;
use crate::world::sound::sound::Sounds;
use crate::world::statisticals::CityState;
use crate::world::terrain::terrain_subsystem::Terrain;
use crate::world::world_state::WorldState;
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
}
