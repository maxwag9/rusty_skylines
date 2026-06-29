use crate::resources::Time;
use crate::world::buildings::buildings::{BuildingLevel, BuildingOccupancy};
use crate::world::buildings::zoning::ZoningType;
use crate::world::statisticals::demography::{Demography, LifeStage, Person};
use crate::world::statisticals::education::EducationLevel;
use crate::world::statisticals::money::CorporateTaxConfig;
use crate::world::statisticals::transports::TransportStats;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub const HOURS_PER_DAY: f64 = 24.0;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ZoningDemand {
    pub demography: Demography, // BTW! This is ONE District!!
    pub corporate_tax_config: CorporateTaxConfig,
    pub transport_stats: TransportStats,
    pub residential_capacity: u32,
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
    pub total_taxes_collected: i64,
}

impl ZoningDemand {
    pub fn new() -> Self {
        ZoningDemand {
            demography: Demography::new(),
            corporate_tax_config: CorporateTaxConfig::default(),
            transport_stats: TransportStats::new(),

            residential_capacity: 0,
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
            total_taxes_collected: 0,
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

    pub fn distribute_occupancy_to_building(
        &self,
        zoning_type: ZoningType,
        building_capacity: u32,
        job_occupancy: &JobOccupancy,
        mut building_occupancy: BuildingOccupancy,
    ) -> BuildingOccupancy {
        if building_capacity == 0 {
            return building_occupancy;
        }

        match zoning_type {
            ZoningType::Residential => {
                building_occupancy.jobs_capacity = 0;
                building_occupancy.workers = 0;
                building_occupancy.residential_capacity = building_capacity;
                let unemployed = building_occupancy.unemployed();
                for _ in 0..=unemployed {

                    // let Some(workplace_id) = zoning_storage.get_work_place(residence.pos, building_storage, person, rng) else { continue };
                    // let Some(workplace) = building_storage.get_mut(workplace_id) else { continue };
                    // workplace.occupancy.workers +=1;
                }
            }
            ZoningType::Commercial => {
                building_occupancy.residential_capacity = 0;
                building_occupancy.jobs_capacity = building_capacity;
                building_occupancy.groups.clear(); // NO ONE lives here!! For now.
            }
            ZoningType::Industrial => {
                building_occupancy.residential_capacity = 0;
                building_occupancy.jobs_capacity = building_capacity;
                let total_cap = self.industrial_capacity.max(1);
                building_occupancy.groups.clear();
            }
            ZoningType::Office => {
                building_occupancy.residential_capacity = 0;
                building_occupancy.jobs_capacity = building_capacity;
                building_occupancy.groups.clear();
            }
            ZoningType::None => {}
        }
        building_occupancy
    }

    // Target land value for one lot's EMA.
    // Outputs ~0.04–0.35 so that average_land_value (true average across lots)
    // feeds cleanly into the prestige formula below (* 3.0 → 0.0–1.0).
    pub fn target_land_value(
        &self,
        zoning_type: ZoningType,
        building_level: BuildingLevel,
        occ: &BuildingOccupancy,
    ) -> f32 {
        let level_mult = 1.0 + building_level.to_u8() as f32 * 0.3;
        let fill_mult = 0.1 + occ.fill_rate(zoning_type) * 1.2;
        let prestige_mult = 1.0 + self.prestige * 1.5;
        let attractiveness_bonus = match zoning_type {
            ZoningType::Residential => self.residential_attractiveness.max(0.0),
            ZoningType::Commercial => self.commercial_attractiveness.max(0.0),
            ZoningType::Industrial => self.industrial_attractiveness.max(0.0),
            ZoningType::Office => self.office_attractiveness.max(0.0),
            ZoningType::None => 0.0,
        };
        (0.04 * level_mult * fill_mult * prestige_mult * (1.0 + attractiveness_bonus * 0.4))
            .max(0.0)
    }
    pub fn update_demands(
        &mut self,
        rng: &mut impl Rng,
        time: &Time,
        average_land_value: f32,
    ) -> (i64, Vec<Person>, Vec<Person>) {
        self.last_updated = time.total_game_time;

        let pop = self.demography.population.max(1);
        let pop_f = pop as f32;

        let job_occ = self.infer_job_occupancy();
        let employed = (job_occ.commercial_workers
            + job_occ.industrial_workers
            + job_occ.office_workers) as f32;
        let job_capacity =
            (self.commercial_capacity + self.industrial_capacity + self.office_capacity) as f32;

        let working_age = self.demography.working_age_population() as f32;
        let none = self.demography.education.non_educated_population(pop) as f32;
        let low = self.demography.education.low_educated_population(pop) as f32;
        let med = self.demography.education.medium_educated_population(pop) as f32;
        let high = self.demography.education.highly_educated_population(pop) as f32;

        self.prestige = (average_land_value * 3.0).clamp(0.0, 1.0);

        let housing_balance = if self.residential_capacity == 0 {
            -1.0
        } else {
            1.0 - (self.demography.population as f32 / self.residential_capacity as f32)
        };
        let housing_balance = housing_balance.clamp(-1.0, 1.0);

        let job_vacancy = if job_capacity <= 0.0 {
            0.0
        } else {
            ((job_capacity - employed) / job_capacity).clamp(0.0, 1.0)
        };

        let unemployment_pressure = if working_age > 0.0 {
            ((working_age - employed) / working_age).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let housing_deficit = if self.demography.population == 0 {
            0.0
        } else {
            ((pop_f - self.residential_capacity as f32) / pop_f).clamp(0.0, 1.0)
        };

        let target_residential =
            (0.9 * housing_balance.max(0.0) + 0.5 * job_vacancy + 0.2 * self.prestige)
                .clamp(0.0, 1.0);

        self.residential = Self::smooth(self.residential, target_residential, 0.35);
        self.commercial = Self::smooth(
            self.commercial,
            Self::deficit_demand(
                med * 1.15 + working_age * 0.10,
                self.commercial_capacity,
                pop_f,
            ),
            0.35,
        );
        self.industrial = Self::smooth(
            self.industrial,
            Self::deficit_demand(
                (none + low) * 1.10 + working_age * 0.12,
                self.industrial_capacity,
                pop_f,
            ),
            0.35,
        );
        self.office = Self::smooth(
            self.office,
            Self::deficit_demand(
                high * 1.20 + working_age * 0.005 * (1.0 + self.prestige),
                self.office_capacity,
                pop_f,
            ) * self.prestige.max(0.1),
            0.35,
        );

        self.residential_attractiveness =
            (housing_balance * 0.75 + job_vacancy * 0.45 + self.prestige * 0.20
                - unemployment_pressure * 0.25)
                .clamp(-1.0, 1.0);

        self.commercial_attractiveness =
            ((med / pop_f) - (self.commercial_capacity as f32 / pop_f)).clamp(-1.0, 1.0);
        self.industrial_attractiveness =
            (((none + low) / pop_f) - (self.industrial_capacity as f32 / pop_f)).clamp(-1.0, 1.0);
        self.office_attractiveness =
            ((high / pop_f) - (self.office_capacity as f32 / pop_f)).clamp(-1.0, 1.0);

        // Immigration
        let mut immigrants = vec![];
        let mut emigrants = vec![];
        if self.residential_attractiveness > 0.0 {
            let free_capacity = self
                .residential_capacity
                .saturating_sub(self.demography.population);
            let expected = self.residential_attractiveness as f64 * 0.02 * free_capacity as f64;
            let guaranteed = expected.floor() as u32;
            let fractional = expected.fract();
            let prestige = self.residential_attractiveness * 0.1;
            //println!("Guaranteed Immigrants: {}, fractional: {}, {} {} {} {} {}", guaranteed, fractional, self.residential_capacity, self.demography.population, free_capacity, self.residential_attractiveness, expected);
            for _ in 0..guaranteed {
                let person = Person {
                    education_level: EducationLevel::random_weighted(prestige),
                    age: LifeStage::random_life_stage().to_u8(),
                };
                immigrants.push(person);
            }
            if rand::random::<f64>() < fractional {
                let person = Person {
                    education_level: EducationLevel::random_weighted(prestige),
                    age: LifeStage::random_life_stage().to_u8(),
                };
                immigrants.push(person);
            }
        } else if self.residential_attractiveness < 0.0 {
            let leave_rate = (-self.residential_attractiveness) as f64 * 0.02;
            let to_remove = (leave_rate * self.demography.population as f64).floor() as u32;
            for _ in 0..to_remove.min(self.demography.population) {
                let Some(person) = self.demography.get_random_person(rng) else {
                    continue;
                };
                emigrants.push(person);
            }
        }

        // Residential Taxes
        let taxes = self.demography.collect_taxes(time.day_length);

        self.total_taxes_collected += taxes;

        (taxes, immigrants, emigrants)
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

    pub fn infer_job_occupancy(&self) -> JobOccupancy {
        let pop = self.demography.population;

        let working_age = self.demography.working_age_population();

        if working_age == 0 {
            return JobOccupancy {
                commercial_workers: 0,
                industrial_workers: 0,
                office_workers: 0,
                commercial_fill: 0.0,
                industrial_fill: 0.0,
                office_fill: 0.0,
                unemployed: 0,
            };
        }

        let none = self.demography.education.non_educated_population(pop);
        let low = self.demography.education.low_educated_population(pop);
        let med = self.demography.education.medium_educated_population(pop);
        let high = self.demography.education.highly_educated_population(pop);

        let industrial_desire = none as f32 * 1.2 + low as f32 * 1.0 + med as f32 * 0.25;

        let commercial_desire = low as f32 * 0.3 + med as f32 * 1.2 + high as f32 * 0.5;

        let office_desire = med as f32 * 0.4 + high as f32 * 2.0;

        let total_desire = industrial_desire + commercial_desire + office_desire;

        if total_desire <= 0.0 {
            return JobOccupancy {
                commercial_workers: 0,
                industrial_workers: 0,
                office_workers: 0,
                commercial_fill: 0.0,
                industrial_fill: 0.0,
                office_fill: 0.0,
                unemployed: working_age,
            };
        }

        let mut industrial_workers =
            ((industrial_desire / total_desire) * working_age as f32) as u32;

        let mut commercial_workers =
            ((commercial_desire / total_desire) * working_age as f32) as u32;

        let mut office_workers = ((office_desire / total_desire) * working_age as f32) as u32;

        industrial_workers = industrial_workers.min(self.industrial_capacity);

        commercial_workers = commercial_workers.min(self.commercial_capacity);

        office_workers = office_workers.min(self.office_capacity);

        let employed = industrial_workers + commercial_workers + office_workers;

        let unemployed = working_age.saturating_sub(employed);

        JobOccupancy {
            commercial_workers,
            industrial_workers,
            office_workers,

            commercial_fill: commercial_workers as f32 / self.commercial_capacity.max(1) as f32,

            industrial_fill: industrial_workers as f32 / self.industrial_capacity.max(1) as f32,

            office_fill: office_workers as f32 / self.office_capacity.max(1) as f32,

            unemployed,
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

#[derive(Debug, Clone, Copy)]
pub struct JobOccupancy {
    pub commercial_workers: u32,
    pub industrial_workers: u32,
    pub office_workers: u32,

    pub commercial_fill: f32,
    pub industrial_fill: f32,
    pub office_fill: f32,

    pub unemployed: u32,
}
