use crate::world::statisticals::money::Economy;
use crate::world::statisticals::schedule::Schedule;
use serde::{Deserialize, Serialize};

pub mod demands;
pub mod money;
pub mod schedule;

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct CityState {
    pub schedule: Schedule,
    pub economy: Economy,
}
impl CityState {
    pub fn new() -> Self {
        CityState {
            schedule: Schedule::default(),
            economy: Economy::default(), // Get it? Like Defaulting on your debt
        }
    }
}
