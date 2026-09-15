use crate::helpers::positions::WorldPos;
use crate::world::buildings::zoning::{
    Lot, LotEntrance, ParkingSpot, ParkingSpotLotInfo, Tile, TilePos, TileType, point_in_polygon_xz,
};
use crate::world::cars::parking::{PARK_L, PARK_W, ParkingSpotId, ParkingStorage};
use glam::{Vec2, Vec3};
use rand::RngExt;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

#[derive(Clone, Copy)]
struct Rect {
    min_x: i16,
    max_x: i16,
    min_z: i16,
    max_z: i16,
}

impl Rect {
    fn contains(&self, x: i16, z: i16) -> bool {
        x >= self.min_x && x <= self.max_x && z >= self.min_z && z <= self.max_z
    }

    fn width(&self) -> i16 {
        self.max_x - self.min_x + 1
    }

    fn depth(&self) -> i16 {
        self.max_z - self.min_z + 1
    }

    fn center_x(&self) -> i16 {
        (self.min_x + self.max_x) / 2
    }

    fn center_z(&self) -> i16 {
        (self.min_z + self.max_z) / 2
    }
}

pub struct LotFrame {
    origin: WorldPos,
    right: Vec2,
    forward: Vec2,
    min_x: i16,
    max_x: i16,
    min_z: i16,
    max_z: i16,
}

impl LotFrame {
    fn width(&self) -> i16 {
        self.max_x - self.min_x + 1
    }

    fn depth(&self) -> i16 {
        self.max_z - self.min_z + 1
    }

    fn world_pos(&self, x: i16, z: i16) -> WorldPos {
        self.origin
            .add_vec2(self.right * (x as f32 + 0.5) + self.forward * (z as f32 + 0.5))
    }

    fn world_direction(&self) -> Vec3 {
        Vec3::new(self.forward.x, 0.0, self.forward.y)
    }

    pub fn from_lot(lot: &Lot) -> Self {
        let origin = lot.entrance.pos;
        let direction = lot.entrance.dir;

        let forward = Vec2::new(direction.x, direction.z).normalize_or_zero();
        let right = Vec2::new(forward.y, -forward.x);

        let mut min_x = i16::MAX;
        let mut max_x = i16::MIN;
        let mut min_z = i16::MAX;
        let mut max_z = i16::MIN;

        for p in &lot.bounds {
            let local = origin.delta_xz(*p, right, forward);
            let x = local.x.floor() as i16;
            let z = local.y.floor() as i16;

            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_z = min_z.min(z);
            max_z = max_z.max(z);
        }

        Self {
            origin,
            right,
            forward,
            min_x,
            max_x,
            min_z,
            max_z,
        }
    }
}
pub struct LotPlan {
    house: Rect,
    garage: Rect,
    driveway: Rect,
    balcony: Option<Rect>,
    entrance: TilePos,
}
impl LotPlan {
    pub fn generate(frame: &LotFrame, rng: &mut ChaCha8Rng) -> Self {
        let width = frame.width();
        let depth = frame.depth();

        let house_w = ((width as f32) * rng.random_range(0.38..0.72)).round() as i16;
        let house_d = ((depth as f32) * rng.random_range(0.32..0.75)).round() as i16;

        let house_w = house_w.clamp(4, (width - 2).max(4));
        let house_d = house_d.clamp(4, (depth - 3).max(4));

        let garage_w = rng.random_range(2..=3);
        let garage_d = rng.random_range(2..=4);
        let driveway_w = rng.random_range(2..=3);

        let center_x = (frame.min_x + frame.max_x) / 2;
        let house_offset = rng.random_range(-1..=1);

        let house_x_min = frame.min_x + 1;
        let house_x_max = (frame.max_x - house_w).max(house_x_min);

        let house_x0 = (center_x - house_w / 2 + house_offset).clamp(house_x_min, house_x_max);

        let house_x1 = house_x0 + house_w - 1;
        let house_z1 = frame.max_z - 1;

        let front_setback = rng.random_range(0..=2);

        let house_z_min = frame.min_z + 2;
        let house_z_max = (frame.max_z - house_d).max(house_z_min);

        let house_z0 = (house_z1 - house_d + 1 - front_setback).clamp(house_z_min, house_z_max);

        let garage_on_right = rng.random_bool(0.7) && house_x1 + 1 + garage_w <= frame.max_x;

        let garage_x0 = if garage_on_right {
            house_x1 + 1
        } else {
            (house_x0 - 1 - garage_w).max(frame.min_x)
        };

        let garage_z0 = house_z0 + rng.random_range(0..=1);
        let garage_z1 = (garage_z0 + garage_d - 1).min(house_z1);

        let garage = Rect {
            min_x: garage_x0,
            max_x: garage_x0 + garage_w - 1,
            min_z: garage_z0,
            max_z: garage_z1,
        };

        let driveway_center_x = garage.center_x();
        let driveway_x0 = driveway_center_x - driveway_w / 2;

        let driveway = Rect {
            min_x: driveway_x0,
            max_x: driveway_x0 + driveway_w - 1,
            min_z: frame.min_z + 1,
            max_z: house_z0,
        };

        let balcony = if rng.random_bool(0.45) && width >= 10 {
            Some(Rect {
                min_x: (house_x0 + rng.random_range(0..=1)).clamp(house_x0, house_x1),
                max_x: (house_x1 - rng.random_range(0..=1)).clamp(house_x0, house_x1),
                min_z: house_z1,
                max_z: house_z1,
            })
        } else {
            None
        };

        Self {
            house: Rect {
                min_x: house_x0,
                max_x: house_x1,
                min_z: house_z0,
                max_z: house_z1,
            },
            garage,
            driveway,
            balcony,
            entrance: TilePos::new(driveway_center_x, frame.min_z + 1),
        }
    }
    fn tile_at(&self, x: i16, z: i16, lot: &Lot) -> Tile {
        if self.driveway.contains(x, z) {
            if z <= self.driveway.min_z {
                return Tile::Square(TileType::LotEntrance);
            }

            return Tile::Square(TileType::Driveway);
        }

        if self.garage.contains(x, z) {
            return Tile::Square(TileType::Garage);
        }

        if self.house.contains(x, z) {
            if self.balcony.is_some_and(|balcony| balcony.contains(x, z)) {
                return Tile::Square(TileType::HouseBalcony);
            }

            if z == self.house.min_z && x >= self.driveway.min_x && x <= self.driveway.max_x {
                return Tile::Square(TileType::HouseEntrance);
            }

            return Tile::Square(TileType::House);
        }

        let hash = (lot.id as u64)
            ^ ((x as i64 as u64).wrapping_mul(0x9E3779B97F4A7C15))
            ^ ((z as i64 as u64).wrapping_mul(0xC2B2AE3D27D4EB4F));

        if hash % 100 == 0 {
            Tile::Square(TileType::Tree)
        } else {
            Tile::Square(TileType::Garden)
        }
    }
    pub fn rasterize(&self, lot: &Lot) -> HashMap<TilePos, Tile> {
        let frame = LotFrame::from_lot(lot);
        let mut tiles = HashMap::new();

        for x in frame.min_x..=frame.max_x {
            for z in frame.min_z..=frame.max_z {
                let world_pos = frame.world_pos(x, z);

                if point_in_polygon_xz(world_pos, &lot.bounds) {
                    tiles.insert(TilePos::new(x, z), self.tile_at(x, z, lot));
                }
            }
        }

        tiles
    }
    pub fn generate_parking(
        &self,
        lot: &Lot,
        frame: &LotFrame,
        tiles: &HashMap<TilePos, Tile>,
        parking_storage: &mut ParkingStorage,
    ) -> Vec<ParkingSpotId> {
        let mut parking_spots = Vec::new();

        let mut z = frame.min_z + 1;

        while z <= self.house.min_z - PARK_L + 1 {
            let mut x = self.driveway.min_x;

            while x + PARK_W - 1 <= self.driveway.max_x {
                let parking_tiles = [
                    TilePos::new(x, z),
                    TilePos::new(x + 1, z),
                    TilePos::new(x, z + 1),
                    TilePos::new(x + 1, z + 1),
                    TilePos::new(x, z + 2),
                    TilePos::new(x + 1, z + 2),
                    TilePos::new(x, z + 3),
                    TilePos::new(x + 1, z + 3),
                ];

                let valid = parking_tiles.iter().all(|tile_pos| {
                    matches!(
                        tiles.get(tile_pos),
                        Some(Tile::Square(TileType::Driveway))
                            | Some(Tile::Square(TileType::LotEntrance))
                    )
                });

                if valid {
                    let center = frame.origin.add_vec2(
                        frame.right * (x as f32 + PARK_W as f32 * 0.5)
                            + frame.forward * (z as f32 + PARK_L as f32 * 0.5),
                    );

                    let id = parking_storage.spawn(ParkingSpot::new(
                        center,
                        frame.world_direction(),
                        Some(ParkingSpotLotInfo {
                            lot_id: lot.id,
                            tiles: parking_tiles,
                        }),
                    ));

                    parking_spots.push(id);
                    x += PARK_W;
                } else {
                    x += 1;
                }
            }

            z += PARK_L;
        }

        parking_spots
    }
    pub fn driveway_entrances(&self, frame: &LotFrame) -> Vec<LotEntrance> {
        let entrance_pos = frame.world_pos(self.entrance.x, self.entrance.z);

        vec![LotEntrance::new(entrance_pos, frame.world_direction())]
    }
}
