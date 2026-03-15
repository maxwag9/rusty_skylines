use crate::commands::CommandBuffer;
use crate::data::Settings;
use crate::resources::Time;
use crate::simulation::Simulation;
use crate::ui::input::Input;
use crate::world::cars::car_subsystem::Cars;
use crate::world::roads::road_subsystem::Roads;
use crate::world::terrain::terrain_subsystem::Terrain;
use crate::world::world::WorldState;
use wgpu::Device;

pub struct WorldCore {
    pub world_state: WorldState,
    pub time: Time,
    pub input: Input,          // main-thread owned
    pub events: CommandBuffer, // main-thread swap/flips, core consumes on sim tick
    pub simulation: Simulation,
    pub terrain: Terrain,
    pub road_subsystem: Roads,
    pub cars: Cars,
    //pub job_pool: JobPool,         // persistent worker threads + channels
    // ... other sim-only subsystems (economy, citizens, etc)
}

impl WorldCore {
    pub(crate) fn new(device: &Device, settings: &Settings) -> Self {
        Self {
            world_state: WorldState::new(),
            time: Time::new(),
            input: Input::new(),
            simulation: Simulation::new(),
            terrain: Terrain::new(device, settings),
            road_subsystem: Roads::new(),
            cars: Cars::new(),
            events: CommandBuffer::new(),
        }
    }
}
