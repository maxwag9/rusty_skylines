use crate::resources::Time;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct ZoningDemand {
    pub residential: f32,
    pub commercial: f32,
    pub industrial: f32,
    pub office: f32,
    pub last_updated: f64,
}
impl ZoningDemand {
    pub fn new() -> Self {
        ZoningDemand {
            residential: 1.0,
            commercial: -1.0,
            industrial: -1.0,
            office: -1.0,
            last_updated: 0.0,
        }
    }
    pub fn update_demands(&mut self, time: &Time) {
        self.last_updated = time.total_game_time;
    }
}

pub struct Demands {}
impl Demands {
    pub fn new() -> Demands {
        Demands {}
    }
    pub fn update_demands(&mut self) {}
}
