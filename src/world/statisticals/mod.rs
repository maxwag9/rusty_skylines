use crate::resources::Time;
use crate::world::buildings::buildings::Buildings;
use crate::world::buildings::zoning::Zoning;
use crate::world::statisticals::money::Economy;
use crate::world::statisticals::schedule::Schedule;
use serde::{Deserialize, Serialize};

pub mod demands;
pub mod money;
pub mod schedule;
pub mod transports;

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct CityState {
    pub world_population: u64,
    pub schedule: Schedule,
    pub economy: Economy,
}
impl CityState {
    pub fn new() -> Self {
        CityState {
            world_population: 0,
            schedule: Schedule::default(),
            economy: Economy::default(), // Get it? Like Defaulting on your debt
        }
    }

    // Start of intercontinental district manager

    pub fn update(&mut self, time: &Time, zoning: &mut Zoning, buildings: &Buildings) {
        self.world_population = zoning
            .zoning_storage
            .iter_districts()
            .map(|d| d.zoning_demand.demography.population as u64)
            .sum();
        self.schedule.update(time);
    }
}
