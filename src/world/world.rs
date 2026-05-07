use crate::commands::CommandBuffer;
use crate::data::Settings;
use crate::resources::Time;
use crate::ui::input::Input;
use crate::world::buildings::buildings::Buildings;
use crate::world::buildings::zoning::Zoning;
use crate::world::cars::car_subsystem::Cars;
use crate::world::game_state::GameState;
use crate::world::roads::road_subsystem::Roads;
use crate::world::terrain::terrain_subsystem::Terrain;
use crate::world::world_state::WorldState;
use wgpu::Device;

pub struct World {
    pub world_state: WorldState,
    pub time: Time,
    pub input: Input,          // main-thread owned
    pub events: CommandBuffer, // main-thread swap/flips, core consumes on sim tick
    pub terrain: Terrain,
    pub roads: Roads,
    pub cars: Cars,
    pub buildings: Buildings,
    pub zoning: Zoning,
    //pub job_pool: JobPool,         // persistent worker threads + channels
    // ... other sim-only subsystems (economy, citizens, etc)
}

impl World {
    pub fn new(device: &Device, settings: &Settings, game_state: &mut GameState) -> Self {
        let world_state = WorldState::new();
        let terrain = Terrain::new(device, settings, &mut game_state.current_save);
        Self {
            world_state,
            time: Time::new(),
            input: Input::new(),
            terrain,
            roads: Roads::new(),
            cars: Cars::new(),
            buildings: Buildings::new(),
            zoning: Zoning::new(),
            events: CommandBuffer::new(),
        }
    }
}
