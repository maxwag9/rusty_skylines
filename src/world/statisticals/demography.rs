use crate::resources::{DAYS_PER_YEAR, HOURS_PER_YEAR, Time};
use crate::world::statisticals::education::{Education, EducationLevel};
use crate::world::statisticals::money::{IncomeDistribution, NUM_INCOME_CLASSES, TaxConfig};
use rand::{Rng, RngExt};
use rand_distr::Distribution;
use rand_distr::Poisson;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::ops::RangeInclusive;
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

pub const MAX_AGE: usize = 23;
pub const INFANT_AGE_RANGE: RangeInclusive<usize> = 0..=1;
pub const CHILD_AGE_RANGE: RangeInclusive<usize> = 2..=7;
pub const YOUNG_ADULT_AGE_RANGE: RangeInclusive<usize> = 8..=12;
pub const ADULT_AGE_RANGE: RangeInclusive<usize> = 13..=19;
pub const ELDER_AGE_RANGE: RangeInclusive<usize> = 20..=22;

// Workhorse population (YoungAdult and Adult combined)
pub const WORKHORSE_AGE_RANGE: RangeInclusive<usize> =
    *YOUNG_ADULT_AGE_RANGE.start()..=*ADULT_AGE_RANGE.end();

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, EnumIter)]
pub enum LifeStage {
    Infant,     // 0–1
    Child,      // 2–7
    YoungAdult, // 8–12
    Adult,      // 13–19
    Elder,      // 20–22
}

impl LifeStage {
    pub fn from_int<T>(year: T) -> Self
    where
        T: Into<usize>,
    {
        match year.into() {
            y if INFANT_AGE_RANGE.contains(&y) => Self::Infant,
            y if CHILD_AGE_RANGE.contains(&y) => Self::Child,
            y if YOUNG_ADULT_AGE_RANGE.contains(&y) => Self::YoungAdult,
            y if ADULT_AGE_RANGE.contains(&y) => Self::Adult,
            _ => Self::Elder,
        }
    }

    pub fn random_life_stage() -> LifeStage {
        let roll = rand::random::<f32>();

        match roll {
            x if x < 0.04 => LifeStage::Infant,
            x if x < 0.22 => LifeStage::Child,
            x if x < 0.72 => LifeStage::YoungAdult,
            x if x < 0.94 => LifeStage::Adult,
            _ => LifeStage::Elder,
        }
    }
    pub fn to_u8(&self) -> u8 {
        match self {
            LifeStage::Infant => *INFANT_AGE_RANGE.start() as u8,
            LifeStage::Child => *CHILD_AGE_RANGE.start() as u8,
            LifeStage::YoungAdult => *YOUNG_ADULT_AGE_RANGE.start() as u8,
            LifeStage::Adult => *ADULT_AGE_RANGE.start() as u8,
            LifeStage::Elder => *ELDER_AGE_RANGE.start() as u8,
        }
    }

    pub fn is_workhorse(&self) -> bool {
        matches!(self, LifeStage::YoungAdult | LifeStage::Adult)
    }
}

impl From<LifeStage> for usize {
    fn from(value: LifeStage) -> Self {
        match value {
            LifeStage::Infant => 0,
            LifeStage::Child => 2,
            LifeStage::YoungAdult => 9,
            LifeStage::Adult => 14,
            LifeStage::Elder => 20,
        }
    }
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LifeStageConfig {
    /// Multiplied against `base_mortality_rate` for citizens in this stage.
    pub mortality_multiplier: f64,
    /// 0.0 = infertile, 1.0 = full birth contribution.
    pub fertility_multiplier: f64,
    pub work_productivity: f64,
}
impl LifeStageConfig {
    pub fn defaults() -> HashMap<LifeStage, Self> {
        LifeStage::iter()
            .map(|stage| {
                let cfg = match stage {
                    LifeStage::Infant => Self {
                        mortality_multiplier: 2.0,
                        fertility_multiplier: 0.0,
                        work_productivity: 0.0,
                    },
                    LifeStage::Child => Self {
                        mortality_multiplier: 0.4,
                        fertility_multiplier: 0.0,
                        work_productivity: 0.0,
                    },
                    LifeStage::YoungAdult => Self {
                        mortality_multiplier: 0.6,
                        fertility_multiplier: 0.8,
                        work_productivity: 0.9,
                    },
                    LifeStage::Adult => Self {
                        mortality_multiplier: 1.0,
                        fertility_multiplier: 0.4,
                        work_productivity: 1.0,
                    },
                    LifeStage::Elder => Self {
                        mortality_multiplier: 10.0,
                        fertility_multiplier: 0.0,
                        work_productivity: 0.3,
                    },
                };
                (stage, cfg)
            })
            .collect()
    }
}

/// Demographic state at a single point in time
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DemographySnapshot {
    pub total_game_time: f64,
    pub population: u32,
    pub ages: Groups,
    pub births: u32,
    pub deaths: u32,
}

/// Rolling daily snapshots, capped at `capacity` (oldest dropped automatically).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DemographyHistory {
    /// One entry per in-game day, newest at the back.
    pub daily: VecDeque<DemographySnapshot>,

    // Accumulators — reset on each flush
    pending_births: u32,
    pending_deaths: u32,
}

impl DemographyHistory {
    pub fn new() -> Self {
        Self {
            daily: VecDeque::with_capacity((10.0 * DAYS_PER_YEAR) as usize),
            pending_births: 0,
            pending_deaths: 0,
        }
    }

    pub fn latest(&self) -> Option<&DemographySnapshot> {
        self.daily.back()
    }

    fn accumulate(&mut self, births: u32, deaths: u32) {
        self.pending_births += births;
        self.pending_deaths += deaths;
    }

    fn new_day(&mut self, total_game_time: f64, population: u32, ages: Groups) {
        self.daily.push_back(DemographySnapshot {
            total_game_time,
            population,
            ages,
            births: self.pending_births,
            deaths: self.pending_deaths,
        });
        self.pending_births = 0;
        self.pending_deaths = 0;
    }
}

/// Factors that influence demographic change each hour.
pub struct DemographyTick {
    /// 0.0 = starving, 1.0 = well-fed. Affects both birth rate and mortality.
    pub food_availability: f64,
    /// 0.0 = miserable, 1.0 = thriving. Affects birth rate.
    pub happiness: f64,
    pub prestige: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, Hash)]
pub struct Groups {
    age_groups: [u32; MAX_AGE],
    education_groups: [[u32; EducationLevel::LEVELS]; MAX_AGE],
}

impl Groups {
    pub fn new() -> Self {
        Self {
            age_groups: [0; MAX_AGE],
            education_groups: [[0; EducationLevel::LEVELS]; MAX_AGE],
        }
    }
    #[inline]
    pub fn get_age<T: Into<usize>>(&self, age: T) -> u32 {
        let age = age.into();
        self.age_groups.get(age).copied().unwrap_or(0)
    }

    #[inline]
    pub fn set_age<T: Into<usize>>(&mut self, age: T, value: u32) {
        let age = age.into();
        if let Some(slot) = self.age_groups.get_mut(age) {
            *slot = value;
        }
    }

    #[inline]
    pub fn add_age<T: Into<usize>>(&mut self, age: T, rhs: u32) {
        let age = age.into();
        if let Some(slot) = self.age_groups.get_mut(age) {
            *slot = slot.saturating_add(rhs);
        }
    }

    #[inline]
    pub fn remove_age<T: Into<usize>>(&mut self, age: T, rhs: u32) {
        let age = age.into();
        if let Some(slot) = self.age_groups.get_mut(age) {
            *slot = slot.saturating_sub(rhs);
        }
    }
    #[inline]
    pub fn get_education<T: Into<usize>>(&self, age: T, level: EducationLevel) -> u32 {
        let age = age.into();
        let lvl = level as usize;

        self.education_groups
            .get(age)
            .and_then(|row| row.get(lvl))
            .copied()
            .unwrap_or(0)
    }

    #[inline]
    pub fn set_education<T: Into<usize>>(&mut self, age: T, level: EducationLevel, value: u32) {
        let age = age.into();
        let lvl = level as usize;

        if let Some(row) = self.education_groups.get_mut(age) {
            if let Some(slot) = row.get_mut(lvl) {
                *slot = value;
            }
        }
    }

    #[inline]
    pub fn add_education<T: Into<usize>>(&mut self, age: T, level: EducationLevel, rhs: u32) {
        let age = age.into();
        let lvl = level as usize;

        if let Some(row) = self.education_groups.get_mut(age) {
            if let Some(slot) = row.get_mut(lvl) {
                *slot = slot.saturating_add(rhs);
            }
        }
    }

    #[inline]
    pub fn remove_education<T: Into<usize>>(&mut self, age: T, level: EducationLevel, rhs: u32) {
        let age = age.into();
        let lvl = level as usize;

        if let Some(row) = self.education_groups.get_mut(age) {
            if let Some(slot) = row.get_mut(lvl) {
                *slot = slot.saturating_sub(rhs);
            }
        }
    }
    #[inline]
    pub fn whole_population(&self) -> u32 {
        self.age_groups.iter().sum()
    }
    #[inline]
    pub fn workhorse_population(&self) -> u32 {
        self.age_groups[WORKHORSE_AGE_RANGE].iter().sum()
    }
    #[inline]
    pub fn clear(&mut self) {
        self.age_groups = [0; MAX_AGE];
        self.education_groups = [[0; EducationLevel::LEVELS]; MAX_AGE];
    }
    #[inline]
    pub fn add_groups(&mut self, other: &Groups) {
        for i in 0..MAX_AGE {
            self.age_groups[i] = self.age_groups[i].saturating_add(other.age_groups[i]);

            for j in 0..EducationLevel::LEVELS {
                self.education_groups[i][j] =
                    self.education_groups[i][j].saturating_add(other.education_groups[i][j]);
            }
        }
    }
    #[inline]
    pub fn education_total(&self, level: EducationLevel) -> u32 {
        let lvl = level as usize;
        self.education_groups
            .iter()
            .map(|row| row.get(lvl).copied().unwrap_or(0))
            .sum()
    }
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Demography {
    pub population: u32,

    pub age_groups: Groups,
    pub lifestage_config: HashMap<LifeStage, LifeStageConfig>,
    pub income: IncomeDistribution,
    pub tax_config: TaxConfig,
    pub education: Education,

    pub history: DemographyHistory,

    base_birth_rate: f64,
    base_mortality_rate: f64,
    /// Multiplier on base_mortality_rate from food scarcity (at food=0: applies at full strength).
    pub starvation_mortality_scale: f64,
    day_counter: u32,
}

impl Demography {
    pub fn new() -> Self {
        let avg_lifespan_hours = 20.0 * HOURS_PER_YEAR;
        let base_mortality_rate = 1.0 / avg_lifespan_hours;

        Self {
            population: 0,
            age_groups: Groups::new(),
            lifestage_config: LifeStageConfig::defaults(),
            income: IncomeDistribution::new(),
            tax_config: TaxConfig::default(),
            education: Education::new(),
            history: DemographyHistory::new(),
            base_birth_rate: base_mortality_rate,
            base_mortality_rate,
            starvation_mortality_scale: 2.0,
            day_counter: 0,
        }
    }
    #[inline]
    pub fn working_age_population(&self) -> u32 {
        let start = 9usize;
        let end = 19usize;

        (start..=end).map(|age| self.age_groups.get_age(age)).sum()
    }

    /// Advance one in-game hour. Returns births
    pub fn age_building_groups(
        &self,
        groups: &mut Groups,
        rng: &mut impl Rng,
        tick: DemographyTick,
    ) -> (u32, u32) {
        let food = tick.food_availability.clamp(0.0, 1.0);
        let happiness = tick.happiness.clamp(0.0, 1.0);

        let food_mort_mod = 1.0 + self.starvation_mortality_scale * (1.0 - food);

        let graduation_rate = 1.0 / (HOURS_PER_YEAR);

        let mut graduating = [0u32; MAX_AGE];
        let mut total_deaths = 0u32;
        for year in 0..MAX_AGE {
            let pop = groups.get_age(year);
            let life_stage = LifeStage::from_int(year);
            let cfg = &self.lifestage_config[&life_stage];

            let mort = self.base_mortality_rate * cfg.mortality_multiplier * food_mort_mod;
            let deaths = poisson(rng, pop as f64 * mort).min(pop);
            let survivors = pop - deaths;
            total_deaths += deaths;

            graduating[year] = if year < MAX_AGE - 1 {
                poisson(rng, survivors as f64 * graduation_rate).min(survivors)
            } else {
                0
            };

            groups.set_age(year, survivors - graduating[year]);
        }

        for year in 1..MAX_AGE {
            let new_population = groups.get_age(year).saturating_add(graduating[year - 1]);
            groups.set_age(year, new_population);
        }

        let fertile_pop: f64 = (0..MAX_AGE)
            .map(|y| {
                groups.get_age(y) as f64
                    * self.lifestage_config[&LifeStage::from_int(y)].fertility_multiplier
            })
            .sum();

        let births = poisson(rng, fertile_pop * self.base_birth_rate * food * happiness);

        (births, total_deaths)
    }
    /// Tax revenue collected this in-game day, in €.
    ///
    /// Iterates over every age cohort, weights each income class by its share
    /// and the cohort's LifeStage productivity, then divides annual income down
    /// to a per-hour figure. Infants and children short-circuit at productivity = 0.0.
    pub fn collect_taxes(&self, day_length: f64) -> i64 {
        let mut total = 0.0f64;

        for age in 0..MAX_AGE {
            let pop = self.age_groups.get_age(age) as f64;
            if pop == 0.0 {
                continue;
            }

            let productivity = self.lifestage_config[&LifeStage::from_int(age)].work_productivity;
            if productivity == 0.0 {
                continue;
            }

            let fractions = self.income.fractions_for_age(age);

            for ic in 0..NUM_INCOME_CLASSES {
                total += pop
                    * fractions[ic]
                    * productivity
                    * self.tax_config.secondly_tax_per_worker(ic, day_length);
            }
        }

        total.round() as i64
    }
    pub fn update(&mut self, time: &Time, births: u32, deaths: u32) {
        self.education.update(time, &self.age_groups);

        self.income.on_births(births, &self.age_groups);

        self.history.accumulate(births, deaths);

        if time.is_new_day() {
            self.day_counter += 1;

            self.history.new_day(
                time.total_game_time,
                self.population,
                self.age_groups.clone(),
            );
            if self.day_counter % DAYS_PER_YEAR as u32 == 0 && self.day_counter > 0 {
                self.income.apply_annual_transitions();
            }
        }
    }

    pub fn get_random_age(&self, rng: &mut impl Rng) -> usize {
        let total = self.age_groups.whole_population();

        if total == 0 {
            return 0;
        }

        let mut pick = rng.random_range(0..total);

        for age in 0..MAX_AGE {
            let count = self.age_groups.get_age(age);

            if pick < count {
                return age;
            }

            pick -= count;
        }

        0 // fallback (should never hit if totals are consistent)
    }
    pub fn get_random_person(&self, rng: &mut impl Rng) -> Option<Person> {
        if self.population == 0 {
            return None;
        }
        let age = self.get_random_age(rng);
        let ed = self
            .education
            .get_citizen_education(LifeStage::from_int(age));
        Some(Person {
            education_level: ed,
            age: age as u8,
        })
    }
}
#[derive(Debug, Clone, Copy)]
pub struct Person {
    pub education_level: EducationLevel,
    pub age: u8,
}
/// Poisson sample — correct variance unlike floor+Bernoulli.
/// Falls back gracefully on degenerate λ.
fn poisson(rng: &mut impl Rng, lambda: f64) -> u32 {
    if lambda <= 0.0 {
        return 0;
    }
    Poisson::new(lambda)
        .map(|d| d.sample(rng) as u32)
        .unwrap_or(0)
}
