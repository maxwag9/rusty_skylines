use crate::resources::Time;
use crate::world::buildings::buildings::{BuildingId, Buildings};
use crate::world::buildings::zoning::{Zoning, ZoningType};
use crate::world::statisticals::demography::{Demography, LifeStage};
use crate::world::statisticals::education::EducationLevel;
use crate::world::statisticals::transports::TransportStats;
use rand::Rng;
use serde::{Deserialize, Serialize};

pub const HOURS_PER_DAY: f64 = 24.0;

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
    pub total_taxes_collected: u64,
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
    pub fn update_demands(
        &mut self,
        rng: &mut impl Rng,
        time: &Time,
        average_land_value: f32,
    ) -> i64 {
        self.last_updated = time.total_game_time;
        self.demography.update(rng, time);

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

        // Taxes
        let taxes = self.demography.collect_taxes(time.day_length);

        self.total_taxes_collected += taxes;

        taxes as i64
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
