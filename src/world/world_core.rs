use crate::cars::car_subsystem::CarSubsystem;
use crate::data::Settings;
use crate::events::Events;
use crate::renderer::terrain_subsystem::TerrainSubsystem;
use crate::resources::TimeSystem;
use crate::simulation::Simulation;
use crate::terrain::roads::road_subsystem::RoadSubsystem;
use crate::ui::input::InputState;
use crate::world::world::WorldState;
use wgpu::Device;

pub struct WorldCore {
    pub world_state: WorldState,
    pub time: TimeSystem,
    pub input: InputState, // main-thread owned
    pub events: Events,    // main-thread swap/flips, core consumes on sim tick
    pub simulation: Simulation,
    pub terrain_subsystem: TerrainSubsystem,
    pub road_subsystem: RoadSubsystem,
    pub car_subsystem: CarSubsystem,
    //pub job_pool: JobPool,         // persistent worker threads + channels
    // ... other sim-only subsystems (economy, citizens, etc)
}

impl WorldCore {
    pub(crate) fn new(device: &Device, settings: &Settings) -> Self {
        Self {
            world_state: WorldState::new(),
            time: TimeSystem::new(),
            input: InputState::new(),
            simulation: Simulation::new(),
            terrain_subsystem: TerrainSubsystem::new(device, settings),
            road_subsystem: RoadSubsystem::new(),
            car_subsystem: CarSubsystem::new(),
            events: Events::new(),
        }
    }
}
