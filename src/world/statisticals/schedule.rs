use crate::resources::Time;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulePhase {
    Night,
    CommuteToWork,
    Work,
    Lunch,
    CommuteHome,
    Evening,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Schedule {
    pub phase: SchedulePhase,

    pub residential_trip_rate: f32,
    pub commercial_trip_rate: f32,
    pub industrial_trip_rate: f32,

    pub traffic_multiplier: f32,
    pub pedestrian_multiplier: f32,

    pub parking_demand: f32,
    pub leisure_demand: f32,
}

impl Default for Schedule {
    fn default() -> Self {
        Self {
            phase: SchedulePhase::Night,

            residential_trip_rate: 0.0,
            commercial_trip_rate: 0.0,
            industrial_trip_rate: 0.0,

            traffic_multiplier: 0.0,
            pedestrian_multiplier: 0.0,

            parking_demand: 0.0,
            leisure_demand: 0.0,
        }
    }
}
impl Schedule {
    pub fn update(&mut self, time: &Time) {
        match time.hour as i32 {
            0..6 => self.night_phase(),
            6..8 => self.commute_to_work_phase(),
            8..12 => self.work_phase(),
            12..13 => self.lunch_phase(),
            13..17 => self.work_phase(),
            17..19 => self.commute_home_phase(),
            19..22 => self.evening_phase(),
            22..24 => self.night_phase(),
            _ => {}
        }
    }

    fn set_all_zero(&mut self) {
        self.residential_trip_rate = 0.0;
        self.commercial_trip_rate = 0.0;
        self.industrial_trip_rate = 0.0;

        self.traffic_multiplier = 0.0;
        self.pedestrian_multiplier = 0.0;

        self.parking_demand = 0.0;
        self.leisure_demand = 0.0;
    }

    fn night_phase(&mut self) {
        self.set_all_zero();

        self.phase = SchedulePhase::Night;

        self.traffic_multiplier = 0.15;
        self.pedestrian_multiplier = 0.05;

        self.parking_demand = 0.95;
    }

    fn commute_to_work_phase(&mut self) {
        self.set_all_zero();

        self.phase = SchedulePhase::CommuteToWork;

        self.residential_trip_rate = 1.0;
        self.industrial_trip_rate = 0.8;
        self.commercial_trip_rate = 0.4;

        self.traffic_multiplier = 1.0;
        self.pedestrian_multiplier = 0.4;

        self.parking_demand = 0.7;
    }

    fn work_phase(&mut self) {
        self.set_all_zero();

        self.phase = SchedulePhase::Work;

        self.commercial_trip_rate = 0.5;
        self.industrial_trip_rate = 0.7;

        self.traffic_multiplier = 0.45;
        self.pedestrian_multiplier = 0.25;

        self.parking_demand = 0.85;
    }

    fn lunch_phase(&mut self) {
        self.set_all_zero();

        self.phase = SchedulePhase::Lunch;

        self.commercial_trip_rate = 1.0;

        self.traffic_multiplier = 0.6;
        self.pedestrian_multiplier = 0.9;

        self.leisure_demand = 0.5;
    }

    fn commute_home_phase(&mut self) {
        self.set_all_zero();

        self.phase = SchedulePhase::CommuteHome;

        self.residential_trip_rate = 1.0;
        self.commercial_trip_rate = 0.6;

        self.traffic_multiplier = 1.2;
        self.pedestrian_multiplier = 0.5;

        self.parking_demand = 1.0;
    }

    fn evening_phase(&mut self) {
        self.set_all_zero();

        self.phase = SchedulePhase::Evening;

        self.commercial_trip_rate = 0.8;

        self.traffic_multiplier = 0.5;
        self.pedestrian_multiplier = 0.7;

        self.leisure_demand = 1.0;
    }
}
