use crate::resources::{DAYS_PER_YEAR, Time};
use crate::world::buildings::buildings::{BuildingId, Buildings};
use crate::world::buildings::zoning::{Zoning, ZoningType};
use crate::world::statisticals::transports::TransportStats;
use rand::rngs::ThreadRng;
use rand::{Rng, RngExt};
use rand_distr::{Distribution, Poisson};
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

pub const HOURS_PER_DAY: f64 = 24.0;
// assumes DAYS_PER_YEAR is defined in your crate

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
        let avg_lifespan_hours = 20.0 * DAYS_PER_YEAR * HOURS_PER_DAY;
        let base_mortality_rate = 1.0 / avg_lifespan_hours;

        Self {
            population: 0,
            age_groups: AgeGroups::new(),
            lifestage_config: Default::default(),
            education: Education::new(),
            history: DemographyHistory::new(),
            base_birth_rate: base_mortality_rate,
            base_mortality_rate,
            starvation_mortality_scale: 2.0,
            day_counter: 0,
        }
    }
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
    pub fn age(&mut self, rng: &mut impl Rng, time: &Time, tick: &DemographyTick) -> u32 {
        let food = tick.food_availability.clamp(0.0, 1.0);
        let happiness = tick.happiness.clamp(0.0, 1.0);

        // At food=0: mortality_modifier = 1 + starvation_scale
        // At food=1: mortality_modifier = 1.0 (no penalty)
        let food_mort_mod = 1.0 + self.starvation_mortality_scale * (1.0 - food);

        // Probability of aging one year per hour = 1 / hours_per_year
        let graduation_rate = 1.0 / (DAYS_PER_YEAR * HOURS_PER_DAY);

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
        }

        births
    }

    fn update(&mut self, time: &Time) {
        self.education.update(time, self.population);
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

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ZoningDemand {
    pub demography: Demography, // BTW! This is ONE District!!
    pub transport_stats: TransportStats,
    pub housing_capacity: u32,
    pub commercial_capacity: u32,
    pub industrial_capacity: u32,
    pub office_capacity: u32,

    // Internal migration attractiveness
    // -1.0 = people want to move out
    //  0.0 = balanced
    //  1.0 = highly attractive
    pub residential_attractiveness: f32,
    pub commercial_attractiveness: f32,
    pub industrial_attractiveness: f32,
    pub office_attractiveness: f32,

    // What the player sees
    // 0.0 = no demand
    // 1.0 = build more
    pub residential: f32,
    pub commercial: f32,
    pub industrial: f32,
    pub office: f32,

    pub last_updated: f64,

    pub prestige: f32,
}

impl ZoningDemand {
    pub fn new() -> Self {
        ZoningDemand {
            demography: Demography::new(),
            transport_stats: TransportStats::new(),

            housing_capacity: 0,
            commercial_capacity: 0,
            industrial_capacity: 0,
            office_capacity: 0,

            residential_attractiveness: 0.0,
            commercial_attractiveness: 0.0,
            industrial_attractiveness: 0.0,
            office_attractiveness: 0.0,

            residential: 1.0,
            commercial: 0.0,
            industrial: 0.0,
            office: 0.0,

            last_updated: 0.0,
            prestige: 0.0,
        }
    }
    #[inline]
    fn smooth(current: f32, target: f32, rate: f32) -> f32 {
        current + (target - current) * rate
    }

    #[inline]
    fn deficit_demand(need: f32, capacity: u32, scale: f32) -> f32 {
        if need <= 0.0 || scale <= 0.0 {
            return 0.0;
        }

        // How much of the need is unmet? (0 = fully served, 1 = nothing is built)
        let unmet = ((need - capacity as f32) / need).clamp(0.0, 1.0);

        // Is the need significant enough to matter?
        // Full weight once need reaches ~5% of population
        // Below that it ramps up linearly, so truly negligible need stays quiet
        const RELEVANCE_THRESHOLD: f32 = 0.05;
        let relevance = ((need / scale) / RELEVANCE_THRESHOLD).clamp(0.0, 1.0);

        unmet * relevance
    }
    pub fn update_demands(&mut self, time: &Time, average_land_value: f32) {
        self.last_updated = time.total_game_time;
        self.demography.update(time);

        let pop = self.demography.population.max(1);
        let pop_f = pop as f32;

        let working_age = self.demography.working_age_population() as f32;
        let none = self.demography.education.non_educated_population(pop) as f32;
        let low = self.demography.education.low_educated_population(pop) as f32;
        let med = self.demography.education.medium_educated_population(pop) as f32;
        let high = self.demography.education.highly_educated_population(pop) as f32;

        self.prestige =
            (self.residential_attractiveness * 0.05 + average_land_value * 0.5).clamp(0.0, 1.0);

        let housing_deficit = ((pop_f - self.housing_capacity as f32) / pop_f).clamp(0.0, 1.0);
        let job_capacity =
            self.commercial_capacity + self.industrial_capacity + self.office_capacity;
        let job_pull = (job_capacity as f32 / pop_f).clamp(0.0, 1.0);

        let target_residential = (0.85 * housing_deficit + 0.35 * job_pull).clamp(0.0, 1.0);

        let commercial_need = med * 1.15 + working_age * 0.10;
        let industrial_need = (none + low) * 1.10 + working_age * 0.12;

        let office_need = high * 1.20 + working_age * 0.005 * (1.0 + self.prestige);

        // Pass pop_f as the scale so tiny needs don't generate full demand
        let target_commercial =
            Self::deficit_demand(commercial_need, self.commercial_capacity, pop_f);
        let target_industrial =
            Self::deficit_demand(industrial_need, self.industrial_capacity, pop_f);
        // Office is a prestige-gated zone: scale the *target* by prestige, not the need!!
        let target_office =
            Self::deficit_demand(office_need, self.office_capacity, pop_f) * self.prestige.max(0.1); // clamp floor so a brand-new city can still start building

        // Smooth bars so they don't jump
        self.residential = Self::smooth(self.residential, target_residential, 0.35);
        self.commercial = Self::smooth(self.commercial, target_commercial, 0.35);
        self.industrial = Self::smooth(self.industrial, target_industrial, 0.35);
        self.office = Self::smooth(self.office, target_office, 0.35);

        // Attractiveness (supply/demand ratio per education tier)
        self.residential_attractiveness = (job_pull - housing_deficit + 0.2).clamp(-1.0, 1.0);
        self.commercial_attractiveness =
            ((med / pop_f) - (self.commercial_capacity as f32 / pop_f)).clamp(-1.0, 1.0);
        self.industrial_attractiveness =
            (((none + low) / pop_f) - (self.industrial_capacity as f32 / pop_f)).clamp(-1.0, 1.0);
        self.office_attractiveness =
            ((high / pop_f) - (self.office_capacity as f32 / pop_f)).clamp(-1.0, 1.0);

        // Immigration
        if self.residential_attractiveness > 0.0 {
            let free_capacity = self
                .housing_capacity
                .saturating_sub(self.demography.population);
            let expected = self.residential_attractiveness as f64 * 0.02 * free_capacity as f64;
            let guaranteed = expected.floor() as u32;
            let fractional = expected.fract();
            let prestige = self.residential_attractiveness * 0.1;

            for _ in 0..guaranteed {
                self.demography.add_person(
                    LifeStage::random_life_stage(),
                    EducationLevel::random_weighted(prestige),
                );
            }
            if rand::random::<f64>() < fractional {
                self.demography.add_person(
                    LifeStage::random_life_stage(),
                    EducationLevel::random_weighted(prestige),
                );
            }
        }
    }

    pub fn spawn_building(buildings: &mut Buildings, zoning: &mut Zoning, building_id: BuildingId) {
        let Some(building) = buildings.storage.get(building_id) else {
            return;
        };
        let Some(lot) = zoning.zoning_storage.get_lot(building.lot_id) else {
            return;
        };
        let tiles = lot.get_tiles();
        let story_area: f64 = lot.get_floor_area();
        let max_people = building.current_level_params().max_people(story_area);
        let zoning_type = lot.zoning_type;
        let Some(district) = zoning.zoning_storage.get_mut_district(lot.district_id) else {
            return;
        };
        //println!("{}", max_people);
        match zoning_type {
            ZoningType::None => {}
            ZoningType::Residential => district.zoning_demand.housing_capacity += max_people,
            ZoningType::Commercial => district.zoning_demand.commercial_capacity += max_people,
            ZoningType::Industrial => district.zoning_demand.industrial_capacity += max_people,
            ZoningType::Office => district.zoning_demand.office_capacity += max_people,
        }
    }
    pub fn despawn_building(
        buildings: &mut Buildings,
        zoning: &mut Zoning,
        building_id: BuildingId,
    ) {
        let Some(building) = buildings.storage.get(building_id) else {
            return;
        };
        let Some(lot) = zoning.zoning_storage.get_lot(building.lot_id) else {
            return;
        };
        let tiles = lot.get_tiles();
        let story_area: f64 = lot.get_floor_area();
        let max_people = building.current_level_params().max_people(story_area);
        let zoning_type = lot.zoning_type;
        let Some(district) = zoning.zoning_storage.get_mut_district(lot.district_id) else {
            return;
        };
        //println!("{}", max_people);
        match zoning_type {
            ZoningType::None => {}
            ZoningType::Residential => district.zoning_demand.housing_capacity -= max_people,
            ZoningType::Commercial => district.zoning_demand.commercial_capacity -= max_people,
            ZoningType::Industrial => district.zoning_demand.industrial_capacity -= max_people,
            ZoningType::Office => district.zoning_demand.office_capacity -= max_people,
        }
    }

    pub fn demand_from_zoning_type(&self, zoning_type: ZoningType) -> f32 {
        match zoning_type {
            ZoningType::None => 0.0,
            ZoningType::Residential => self.residential,
            ZoningType::Commercial => self.commercial,
            ZoningType::Industrial => self.industrial,
            ZoningType::Office => self.office,
        }
    }
}

pub struct Demands {}
impl Demands {
    pub fn new() -> Demands {
        Demands {}
    }
    pub fn update_demands(&mut self) {}
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
