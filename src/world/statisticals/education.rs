use crate::resources::Time;
use rand::RngExt;
use rand::prelude::ThreadRng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy)]
pub enum EducationLevel {
    None,
    Low,
    Medium,
    High,
}

impl EducationLevel {
    /// prestige:
    /// 0.0 = terrible city, mostly uneducated immigrants
    /// 1.0 = average city
    /// 2.0 = highly prestigious city
    ///
    /// You can feed this from attractiveness, land value,
    /// education availability, jobs, etc.
    pub fn random_weighted(prestige: f32) -> Self {
        let mut rng = rand::rng();

        let prestige = prestige.max(0.0);

        // Simple smooth curve control.
        // Higher prestige shifts weight upward.
        let none_weight = (1.8 - prestige * 0.7).max(0.05);
        let low_weight = (1.5 - prestige * 0.3).max(0.05);
        let med_weight = (0.7 + prestige * 0.5).max(0.05);
        let high_weight = (0.2 + prestige * prestige * 0.8).max(0.01);

        let total = none_weight + low_weight + med_weight + high_weight;

        let roll = rng.random::<f32>() * total;

        if roll < none_weight {
            Self::None
        } else if roll < none_weight + low_weight {
            Self::Low
        } else if roll < none_weight + low_weight + med_weight {
            Self::Medium
        } else {
            Self::High
        }
    }
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Education {
    non_educated_population_factor: f64,
    low_educated_population_factor: f64,
    medium_educated_population_factor: f64,
    highly_educated_population_factor: f64,
    school_year: u32,
}

impl Education {
    pub fn new() -> Self {
        Self {
            non_educated_population_factor: 1.0,
            low_educated_population_factor: 0.0,
            medium_educated_population_factor: 0.0,
            highly_educated_population_factor: 0.0,
            school_year: 0,
        }
    }
    #[inline]
    pub fn non_educated_population(&self, district_population: u32) -> u32 {
        (district_population as f64 * self.non_educated_population_factor) as u32
    }
    #[inline]
    pub fn low_educated_population(&self, district_population: u32) -> u32 {
        (district_population as f64 * self.low_educated_population_factor) as u32
    }
    #[inline]
    pub fn medium_educated_population(&self, district_population: u32) -> u32 {
        (district_population as f64 * self.medium_educated_population_factor) as u32
    }
    #[inline]
    pub fn highly_educated_population(&self, district_population: u32) -> u32 {
        (district_population as f64 * self.highly_educated_population_factor) as u32
    }

    pub fn get_citizen_education(&self) -> EducationLevel {
        let mut rng = ThreadRng::default();

        let none = self.non_educated_population_factor.max(0.0);
        let low = self.low_educated_population_factor.max(0.0);
        let med = self.medium_educated_population_factor.max(0.0);
        let high = self.highly_educated_population_factor.max(0.0);

        let sum = none + low + med + high;

        // fallback if everything is zero
        if sum <= 0.0 {
            return EducationLevel::None;
        }

        let roll: f64 = rng.random_range(0.0..sum);

        if roll < none {
            EducationLevel::None
        } else if roll < none + low {
            EducationLevel::Low
        } else if roll < none + low + med {
            EducationLevel::Medium
        } else {
            EducationLevel::High
        }
    }

    pub fn update(&mut self, time: &Time, population: u32) {
        let current_school_year = time.school_year();
        if self.school_year != current_school_year {
            // New school year! People graduated!
            self.school_year = current_school_year;
            //println!("{}", current_school_year);
        }
    }
    pub fn add_person(&mut self, prev_population: u32, ed_level: EducationLevel) {
        let mut none = self.non_educated_population(prev_population);
        let mut low = self.low_educated_population(prev_population);
        let mut med = self.medium_educated_population(prev_population);
        let mut high = self.highly_educated_population(prev_population);

        match ed_level {
            EducationLevel::None => none += 1,
            EducationLevel::Low => low += 1,
            EducationLevel::Medium => med += 1,
            EducationLevel::High => high += 1,
        }

        let new_population = prev_population + 1;

        self.non_educated_population_factor = none as f64 / new_population as f64;

        self.low_educated_population_factor = low as f64 / new_population as f64;

        self.medium_educated_population_factor = med as f64 / new_population as f64;

        self.highly_educated_population_factor = high as f64 / new_population as f64;
    }
}
