use crate::resources::{DAYS_PER_YEAR, HOURS_PER_YEAR, Time};
use crate::world::statisticals::education::{Education, EducationLevel};
use crate::world::statisticals::money::{IncomeDistribution, NUM_INCOME_CLASSES, TaxConfig};
use rand::{Rng, RngExt};
use rand_distr::Distribution;
use rand_distr::Poisson;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

pub const MAX_AGE: usize = 23;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash, EnumIter)]
pub enum LifeStage {
    Infant,     // 0–1
    Child,      // 2–8
    YoungAdult, // 9–13
    Adult,      // 14–19
    Elder,      // 20–22
}

impl LifeStage {
    pub fn from_usize(year: usize) -> Self {
        match year {
            0..=1 => Self::Infant,
            2..=8 => Self::Child,
            9..=13 => Self::YoungAdult,
            14..=19 => Self::Adult,
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
    pub ages: AgeGroups,
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

    fn new_day(&mut self, total_game_time: f64, population: u32, ages: AgeGroups) {
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
    /// 0.0 = starving, 1.0 = well fed. Affects both birth rate and mortality.
    pub food_availability: f64,
    /// 0.0 = miserable, 1.0 = thriving. Affects birth rate.
    pub happiness: f64,
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AgeGroups {
    groups: [u32; MAX_AGE],
}
impl AgeGroups {
    pub fn new() -> Self {
        Self {
            groups: [0; MAX_AGE],
        }
    }
    pub fn get<T: Into<usize>>(&self, age: T) -> u32 {
        let age: usize = age.into();

        self.groups.get(age).copied().unwrap_or(0) // age as usize is a bug, that would be the ENUM index, not the age. Use age.to_usize(), just for my future self. I changed it anyway.
    }

    pub fn set<T: Into<usize>>(&mut self, age: T, new_population: u32) {
        let age: usize = age.into();

        self.groups.get_mut(age).map(|p| *p = new_population);
    }
    pub fn add<T: Into<usize>>(&mut self, age: T, rhs: u32) {
        let age: usize = age.into();

        self.groups.get_mut(age).map(|p| *p += rhs);
    }
    pub fn whole_population(&self) -> u32 {
        self.groups.iter().sum()
    }
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Demography {
    pub population: u32,

    /// Age buckets: [children 0–4, youth 5–9, young_adult 10–14, adult 15–19, elderly 20+]1
    pub age_groups: AgeGroups,
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
            age_groups: AgeGroups::new(),
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

        (start..=end).map(|age| self.age_groups.get(age)).sum()
    }
    #[inline]
    pub fn add_person(&mut self, life_stage: LifeStage, ed_level: EducationLevel) {
        self.education.add_person(self.population, ed_level); // MUST be before population increments!!
        self.population += 1;
        self.age_groups.add(life_stage, 1);
    }

    /// Advance one in-game hour. Returns births
    pub fn age(&mut self, rng: &mut impl Rng, time: &Time, tick: &DemographyTick) {
        let food = tick.food_availability.clamp(0.0, 1.0);
        let happiness = tick.happiness.clamp(0.0, 1.0);

        // At food=0: mortality_modifier = 1 + starvation_scale
        // At food=1: mortality_modifier = 1.0 (no penalty)
        let food_mort_mod = 1.0 + self.starvation_mortality_scale * (1.0 - food);

        // Probability of aging one year per hour = 1 / hours_per_year
        let graduation_rate = 1.0 / (HOURS_PER_YEAR);

        let mut graduating = [0u32; MAX_AGE];
        let mut total_deaths = 0u32;

        for year in 0..MAX_AGE {
            let pop = self.age_groups.get(year);
            let cfg = &self.lifestage_config[&LifeStage::from_usize(year)];

            // Deaths this hour for this cohort
            let mort = self.base_mortality_rate * cfg.mortality_multiplier * food_mort_mod;
            let deaths = poisson(rng, pop as f64 * mort).min(pop);
            let survivors = pop - deaths;
            total_deaths += deaths;

            // Graduate survivors into the next year slot (no graduation at max age)
            graduating[year] = if year < MAX_AGE - 1 {
                poisson(rng, survivors as f64 * graduation_rate).min(survivors)
            } else {
                0
            };

            self.age_groups.set(year, survivors - graduating[year]);
        }

        // Pour graduating cohorts into the next year slot
        for year in 1..MAX_AGE {
            let new_population = self
                .age_groups
                .get(year)
                .saturating_add(graduating[year - 1]);
            self.age_groups.set(year, new_population);
        }

        // Births — summed fertility contribution weighted by age group
        let fertile_pop: f64 = (0..MAX_AGE)
            .map(|y| {
                self.age_groups.get(y) as f64
                    * self.lifestage_config[&LifeStage::from_usize(y)].fertility_multiplier
            })
            .sum();
        let births = poisson(rng, fertile_pop * self.base_birth_rate * food * happiness);
        self.age_groups.add(0usize, births);

        self.income.on_births(births, &self.age_groups);

        self.population = self.age_groups.whole_population();
        self.history.accumulate(births, total_deaths);

        // Snapshot once per in-game day
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
    /// Tax revenue collected this in-game day, in €.
    ///
    /// Iterates over every age cohort, weights each income class by its share
    /// and the cohort's LifeStage productivity, then divides annual income down
    /// to a per-hour figure. Infants and children short-circuit at productivity = 0.0.
    pub fn collect_taxes(&self, day_length: f64) -> u64 {
        let mut total = 0.0f64;

        for age in 0..MAX_AGE {
            let pop = self.age_groups.get(age) as f64;
            if pop == 0.0 {
                continue;
            }

            let productivity = self.lifestage_config[&LifeStage::from_usize(age)].work_productivity;
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

        total.round() as u64
    }
    pub fn update(&mut self, rng: &mut impl Rng, time: &Time) {
        self.education.update(time, self.population);
        let demography_tick = DemographyTick {
            food_availability: 1.0,
            happiness: 1.0,
        };
        self.age(rng, time, &demography_tick);
    }

    pub fn get_random_age(&self, rng: &mut impl Rng) -> usize {
        let total = self.age_groups.whole_population();

        if total == 0 {
            return 0;
        }

        let mut pick = rng.random_range(0..total);

        for age in 0..MAX_AGE {
            let count = self.age_groups.get(age);

            if pick < count {
                return age;
            }

            pick -= count;
        }

        0 // fallback (should never hit if totals are consistent)
    }
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
