use crate::resources::Time;
use crate::world::statisticals::demography::{Groups, LifeStage};
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
    pub const LEVELS: usize = 4;
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

    pub fn get_citizen_education(&self, age: LifeStage) -> EducationLevel {
        let mut rng = ThreadRng::default();

        let mut none = self.non_educated_population_factor.max(0.0);
        let mut low = self.low_educated_population_factor.max(0.0);
        let mut med = self.medium_educated_population_factor.max(0.0);
        let mut high = self.highly_educated_population_factor.max(0.0);

        // age-based skewing
        match age {
            LifeStage::Infant => {
                none *= 3.0;
                low *= 0.5;
                med *= 0.1;
                high *= 0.01;
            }

            LifeStage::Child => {
                none *= 2.0;
                low *= 1.2;
                med *= 0.3;
                high *= 0.05;
            }

            LifeStage::YoungAdult => {
                none *= 1.2;
                low *= 1.3;
                med *= 1.0;
                high *= 0.5;
            }

            LifeStage::Adult => {
                none *= 0.8;
                low *= 1.0;
                med *= 1.4;
                high *= 1.2;
            }

            LifeStage::Elder => {
                none *= 0.7;
                low *= 0.9;
                med *= 1.1;
                high *= 1.3;
            }
        }

        let sum = none + low + med + high;

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

    pub fn update(&mut self, time: &Time, groups: &Groups) {
        let current_school_year = time.school_year();
        if self.school_year != current_school_year {
            self.school_year = current_school_year;
        }

        self.update_from_groups(groups);
    }

    pub fn update_from_groups(&mut self, groups: &Groups) {
        let population = groups.whole_population();
        if population == 0 {
            self.non_educated_population_factor = 1.0;
            self.low_educated_population_factor = 0.0;
            self.medium_educated_population_factor = 0.0;
            self.highly_educated_population_factor = 0.0;
            return;
        }

        let population = population as f64;

        self.non_educated_population_factor =
            groups.education_total(EducationLevel::None) as f64 / population;
        self.low_educated_population_factor =
            groups.education_total(EducationLevel::Low) as f64 / population;
        self.medium_educated_population_factor =
            groups.education_total(EducationLevel::Medium) as f64 / population;
        self.highly_educated_population_factor =
            groups.education_total(EducationLevel::High) as f64 / population;
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
    pub fn remove_person(&mut self, prev_population: u32, ed_level: EducationLevel) {
        if prev_population == 0 {
            return;
        }

        // Compute absolute counts BEFORE the removal (same pattern as add_person)
        let mut none = self.non_educated_population(prev_population);
        let mut low = self.low_educated_population(prev_population);
        let mut med = self.medium_educated_population(prev_population);
        let mut high = self.highly_educated_population(prev_population);

        match ed_level {
            EducationLevel::None => none = none.saturating_sub(1),
            EducationLevel::Low => low = low.saturating_sub(1),
            EducationLevel::Medium => med = med.saturating_sub(1),
            EducationLevel::High => high = high.saturating_sub(1),
        }

        let new_pop = prev_population - 1;
        if new_pop > 0 {
            self.non_educated_population_factor = none as f64 / new_pop as f64;
            self.low_educated_population_factor = low as f64 / new_pop as f64;
            self.medium_educated_population_factor = med as f64 / new_pop as f64;
            self.highly_educated_population_factor = high as f64 / new_pop as f64;
        } else {
            // city emptied — reset to defaults
            self.non_educated_population_factor = 1.0;
            self.low_educated_population_factor = 0.0;
            self.medium_educated_population_factor = 0.0;
            self.highly_educated_population_factor = 0.0;
        }
    }
}
