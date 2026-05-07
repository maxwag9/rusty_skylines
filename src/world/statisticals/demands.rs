use crate::helpers::positions::WorldPos;
use crate::resources::Time;
use crate::world::buildings::buildings::{BuildingId, Buildings};
use crate::world::buildings::zoning::{Tile, TileType, Zoning};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ZoningDemand {
    pub population: u32,
    pub housing_capacity: u32,
    pub residential: f32,
    pub commercial: f32,
    pub industrial: f32,
    pub office: f32,
    pub last_updated: f64,
}
impl ZoningDemand {
    pub fn new() -> Self {
        ZoningDemand {
            population: 0,
            housing_capacity: 0,
            residential: 1.0,
            commercial: -1.0,
            industrial: -1.0,
            office: -1.0,
            last_updated: 0.0,
        }
    }
    pub fn update_demands(&mut self, time: &Time) {
        self.last_updated = time.total_game_time;

        let shortage = self.population as i32 - self.housing_capacity as i32;

        // if shortage > 0 → not enough housing → demand is higher
        // if shortage < 0 → surplus → demand is low
        self.residential = shortage as f32 * 0.001;

        self.residential = self.residential.clamp(-1.0, 1.0);
    }
    pub fn spawn_building(buildings: &mut Buildings, zoning: &mut Zoning, building_id: BuildingId) {
        let Some(building) = buildings.storage.get(building_id) else {
            return;
        };
        let Some(lot) = zoning.zoning_storage.get_lot(building.lot_id) else {
            return;
        };
        let tiles = lot.get_tiles();
        let story_area: f64 = tiles
            .values()
            .filter_map(|tile| match tile {
                Tile::Square(TileType::House) => Some(1.0),
                Tile::Polygon(TileType::House, points) => Some(WorldPos::area(points)),
                _ => None,
            })
            .sum();
        let max_people = building.current_level_params().max_people(story_area);
        let Some(district) = zoning.zoning_storage.get_mut_district(building.lot_id) else {
            return;
        };
        district.zoning_demand.housing_capacity += max_people;
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
        let story_area: f64 = tiles
            .values()
            .filter_map(|tile| match tile {
                Tile::Square(TileType::House) => Some(1.0),
                Tile::Polygon(TileType::House, points) => Some(WorldPos::area(points)),
                _ => None,
            })
            .sum();
        let max_people = building.current_level_params().max_people(story_area);
        let Some(district) = zoning.zoning_storage.get_mut_district(building.lot_id) else {
            return;
        };
        district.zoning_demand.housing_capacity -= max_people;
    }
}

pub struct Demands {}
impl Demands {
    pub fn new() -> Demands {
        Demands {}
    }
    pub fn update_demands(&mut self) {}
}
