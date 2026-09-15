use crate::helpers::implementations::SerializableVec3;
use crate::helpers::positions::{ChunkCoord, WorldPos};
use crate::renderer::gizmo::gizmo::Gizmo;
use crate::renderer::props::Props;
use crate::resources::Time;
use crate::simulation::Ticker;
use crate::ui::input::Input;
use crate::ui::parser::Value;
use crate::ui::variables::Variables;
use crate::world::buildings::building_mesher::{
    Color, DrivewayMaterial, RoofMaterial, WallMaterial,
};
use crate::world::buildings::buildings::{
    Building, BuildingComplaint, BuildingDesignSource, BuildingId, BuildingOccupancy,
    BuildingParams, BuildingStorage, BuildingUsage, Buildings, GarageParams, MiscBuildingParams,
    RevisionedSmallVec, RoofType,
};
use crate::world::buildings::lot_fitting::{
    LOT_FRONT_OFFSET, LOT_MIN_DEPTH, LOT_PREFERRED_FRONTAGE, LotFit, LotPoint, collect_road_points,
    fit_lot_to_neighbors, gather_closest_lot_point,
};
use crate::world::buildings::lot_layout::{LotFrame, LotPlan};
use crate::world::camera::Camera;
use crate::world::cars::car_structs::{Car, CarId, CarMode, CarStorage, SimTime};
use crate::world::cars::car_subsystem::make_random_car;
use crate::world::cars::parking::{ParkingSpotId, ParkingStorage};
use crate::world::cars::partitions::Destination;
use crate::world::roads::road_mesh_manager::{Edges, RoadEdgeStorage, RoadEdges, RoadMeshManager};
use crate::world::roads::road_structs::{LaneId, SegmentId};
use crate::world::roads::road_subsystem::Roads;
use crate::world::roads::roads::RoadStorage;
use crate::world::statisticals::CityState;
use crate::world::statisticals::demands::ZoningDemand;
use crate::world::statisticals::demography::{DemographyTick, Groups, LifeStage, Person};
use crate::world::statisticals::education::EducationLevel;
use crate::world::statisticals::schedule::{Schedule, SchedulePhase};
use crate::world::statisticals::transports::CarTripType;
use crate::world::terrain::terrain_subsystem::{CursorMode, PickedPoint, Terrain};
use glam::{Quat, Vec2, Vec3};
use rand::rngs::ThreadRng;
use rand::{Rng, RngExt};
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use rayon::iter::ParallelIterator;
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator};
use revision::revisioned;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::slice::IterMut;
use tracing::error;

const SNAP_RADIUS: f64 = 20.0;
const EPS: f64 = 0.0001;

#[derive(Serialize, Deserialize, Clone, Debug)]
#[revisioned(revision = 1)]
pub enum DistrictType {
    AutomaticallyMade,
    PlayerMade,
}
#[derive(Serialize, Deserialize, Clone, Debug)]
#[revisioned(revision = 1)]
pub struct District {
    pub id: DistrictId,
    pub name: String,
    pub district_type: DistrictType,
    pub center: WorldPos,
    // Raw points from all lots in the district
    raw_points: Vec<WorldPos>,

    // Cached convex hull of raw_points
    pub points: Vec<WorldPos>,
    pub lot_ids: Vec<LotId>,
    pub zoning_demand: ZoningDemand,
}

impl District {
    pub fn new(name: String, points: Vec<WorldPos>, district_type: DistrictType) -> District {
        let center = WorldPos::centroid(&points);
        Self {
            id: 3346243577,
            name,
            district_type,
            center,
            raw_points: points.clone(),
            points,
            lot_ids: vec![],
            zoning_demand: ZoningDemand::new(),
        }
    }
    #[inline]
    pub fn add_point(&mut self, point: WorldPos) {
        self.points.push(point);
        self.center = WorldPos::centroid(&self.points);
    }
    #[inline]
    pub fn add_points(&mut self, mut points: Vec<WorldPos>) {
        self.points.append(&mut points);
        self.points = WorldPos::convex_hull(&self.points);
        self.center = WorldPos::centroid(&self.points);
    }
    #[inline]
    pub fn replace_points(&mut self, points: Vec<WorldPos>) {
        self.points = WorldPos::convex_hull(&points);
        self.center = WorldPos::centroid(&self.points);
    }
    #[inline]
    pub fn remove_points(&mut self, points: &Vec<WorldPos>) {
        self.points.retain(|p| !points.contains(p)); // TODO, RISKY!!
        self.points = WorldPos::convex_hull(&self.points);
        self.center = WorldPos::centroid(&self.points);
    }
    pub fn update(
        terrain: &Terrain,
        time: &Time,
        buildings: &Buildings,
        zoning: &Zoning,
        schedule: &Schedule,
        district_id: DistrictId,
        road_edge_storage: &RoadEdgeStorage,
        target_chunk: ChunkCoord,
    ) -> DistrictUpdateCallback {
        let mut callback = DistrictUpdateCallback::new(district_id);
        let Some(district) = zoning
            .zoning_storage
            .districts
            .get(district_id as usize)
            .and_then(|d| d.as_ref())
        else {
            return callback;
        };

        let job_occupancy = district.zoning_demand.infer_job_occupancy();
        let rng = &mut ThreadRng::default();
        for lot_id in district.lot_ids.clone() {
            let (chunk, zoning_type, maybe_building) = {
                if let Some((lot_entrance, car_trip_type)) =
                    Lot::get_car_spawn(zoning, buildings, schedule, lot_id, rng)
                {
                    let dir = -lot_entrance.dir.as_vec3(); // Flip in_dir to become out_dir
                    let mut car = make_random_car(lot_entrance.pos, rng);
                    let forward = dir.normalize();
                    let up = Vec3::Y;

                    car.quat = Quat::from_rotation_arc(forward, up);
                    if let Some(destination) = zoning.zoning_storage.get_work_place_destination(
                        &buildings.storage,
                        car.pos,
                        car_trip_type,
                        rng,
                    ) {
                        car.mode = CarMode::Driving(destination);
                        callback.new_cars.push((lot_id, Some(car)));
                    }
                };

                let Some(lot) = zoning
                    .zoning_storage
                    .lots
                    .get(lot_id as usize)
                    .and_then(|lot| lot.as_ref())
                else {
                    continue;
                };

                if let Some(building) = buildings.storage.get(lot.building_id) {
                    let Some(zoning_type) = lot.zoning_type else {
                        continue;
                    };
                    if building.pos.chunk.dist2(target_chunk)
                        < ((terrain.view_radius_render as f32 * 0.8)
                            * (terrain.view_radius_render as f32 * 0.8))
                            as u64
                    {
                        let complaint = building.occupancy.complaint(
                            zoning_type,
                            &district.zoning_demand,
                            &job_occupancy,
                        );
                        callback
                            .complaining_buildings
                            .push((building.id, building.pos, complaint));
                        // match complaint {
                        //     None => {}
                        //     Some(c) => {
                        //         match c {
                        //             BuildingComplaint::NotEnoughWorkers => {}
                        //             BuildingComplaint::NotEnoughCustomers => {}
                        //             BuildingComplaint::NotEnoughJobs => {
                        //
                        //             }
                        //         }
                        //     }
                        // }
                    };
                    let floor_area = lot.floor_area_or_zero();
                    let Some(current_level) = building.current_level_params(&buildings.catalog)
                    else {
                        error!("Building had no Building Params");
                        continue;
                    };
                    let building_capacity = current_level.max_people(floor_area);
                    let full_area = floor_area * current_level.num_stories as f64;

                    if zoning_type.is_workplace() {
                        let tax = district.zoning_demand.corporate_tax_config.compute_lot_tax(
                            zoning_type,
                            full_area,
                            building.level,
                            &job_occupancy,
                            time.day_length,
                        );

                        callback.corporate_taxes += tax as i64;
                    }

                    let building_occ = district.zoning_demand.distribute_occupancy_to_building(
                        zoning_type,
                        building_capacity,
                        &job_occupancy,
                        building.occupancy.clone(),
                    );
                    let target_lv = district.zoning_demand.target_land_value(
                        zoning_type,
                        building.level,
                        &building_occ,
                    );
                    // EMA: 2% of the way toward target each tick, slow enough to be stable
                    // even though prestige feeds back into target_lv.
                    let new_land_value = lot.land_value + (target_lv - lot.land_value) * 0.02;

                    match zoning_type {
                        ZoningType::Residential => {
                            callback.residential_capacity += building_capacity;
                            let mut aged_groups = building.occupancy.groups.clone();
                            let demography_tick = DemographyTick {
                                food_availability: 1.0,
                                happiness: 1.0,
                                prestige: district.zoning_demand.prestige,
                            };

                            let (births, deaths) = district
                                .zoning_demand
                                .demography
                                .age_building_groups(&mut aged_groups, rng, demography_tick);
                            callback.total_births += births;
                            callback.total_deaths += deaths;
                            callback.new_age_groups.add_groups(&aged_groups);
                            callback
                                .building_demography_updates
                                .push((building.id, aged_groups));

                            for _ in 0..building.occupancy.unemployed() {
                                let age = building.occupancy.random_employable_age();
                                let person = Person {
                                    education_level: district
                                        .zoning_demand
                                        .demography
                                        .education
                                        .get_citizen_education(LifeStage::from_int(age)),
                                    age: age as u8,
                                };
                                let Some(workplace_id) = zoning.zoning_storage.get_work_place(
                                    building.pos,
                                    &buildings.storage,
                                    person,
                                    rng,
                                ) else {
                                    break;
                                };
                                callback.new_workers.push((building.id, workplace_id));
                            }
                        }
                        ZoningType::Commercial => callback.commercial_capacity += building_capacity,
                        ZoningType::Industrial => callback.industrial_capacity += building_capacity,
                        ZoningType::Office => callback.office_capacity += building_capacity,
                    }

                    callback.occupancy_updates.push((building.id, building_occ));
                    callback.land_value_updates.push((lot_id, new_land_value));
                    continue;
                } else {
                    if let Some(zoning_type) = lot.zoning_type {
                        let Some(district) = zoning
                            .zoning_storage
                            .districts
                            .get(district_id as usize)
                            .and_then(|d| d.as_ref())
                        else {
                            return callback;
                        };
                        if !rng.random_bool(
                            district
                                .zoning_demand
                                .demand_from_zoning_type(zoning_type)
                                .clamp(0.0, 1.0) as f64,
                        ) {
                            continue;
                        }

                        let building = generate_building(terrain, lot);
                        (lot.center.chunk, lot.zoning_type, building)
                    } else {
                        (lot.center.chunk, lot.zoning_type, None)
                    }
                }
            };

            callback.new_buildings.push((lot_id, maybe_building));
        }

        let average_land_value = zoning.average_land_value(district_id);
        callback.average_land_value = Some(average_land_value);
        let Some(district) = zoning.zoning_storage.get_district(district_id) else {
            return callback;
        };

        if !matches!(district.district_type, DistrictType::PlayerMade) && district.should_split() {
            let split_result = Self::split(&zoning.zoning_storage, district_id, road_edge_storage);
            //println!("{:?}", split_result);
            callback.district_split = Some(split_result);
        }
        callback
    }
    fn should_split(&self) -> bool {
        if self.points.len() < 3 {
            return false;
        }
        if self.lot_ids.len() > 500 {
            return true;
        }
        let area = WorldPos::area(self.points.as_slice());
        //println!("Area: {}", area);
        if area > 50_000_000.0 {
            return true;
        }
        false
    }

    /// Split the district along the road geometry of one of its boundary segments.
    ///
    /// Strategy:
    ///   1. Among all road segments referenced by our lots, find the one whose
    ///      right-sidewalk edge actually bisects the district and whose midpoint
    ///      is closest to our centroid.
    ///   2. Find the first and last boundary crossings of that edge polyline.
    ///   3. Cut the district polygon at those two crossing points, keeping the
    ///      road-edge vertices in between as the seam.
    ///   4. Assign each lot to whichever half its centroid falls in.
    ///   5. Mutate `self` to be the first half; return the second half.
    ///
    /// Fallback:
    ///   If no road-edge split works, split the polygon in two using a simple
    ///   axis-aligned line through the centroid.
    ///
    /// Everything stays in WorldPos — dx/dz for geometry, delta_to for offsets.
    pub fn split(
        zoning_storage: &ZoningStorage,
        district_id: DistrictId,
        road_edge_storage: &RoadEdgeStorage,
    ) -> DistrictSplitResult {
        let Some(district) = zoning_storage.get_district(district_id) else {
            return DistrictSplitResult::DistrictDoesntExist;
        };

        if district.points.len() < 6 {
            return DistrictSplitResult::NotEnoughPoints;
        }

        #[derive(Clone, Copy)]
        struct Crossing {
            boundary_edge: usize,
            t_boundary: f32,
            split_pos: f32,
            point: WorldPos,
        }

        const EPS: f64 = 1e-9;
        const POINT_EPS: f64 = 0.001;

        fn push_unique(poly: &mut Vec<WorldPos>, p: WorldPos) {
            if poly
                .last()
                .map_or(true, |last| last.distance_to(p) > POINT_EPS)
            {
                poly.push(p);
            }
        }

        fn side_of(p: WorldPos, centroid: WorldPos, split_on_x: bool) -> f64 {
            if split_on_x {
                p.dx(centroid)
            } else {
                p.dz(centroid)
            }
        }

        fn clip_halfspace(
            poly: &[WorldPos],
            centroid: WorldPos,
            split_on_x: bool,
            keep_positive: bool,
        ) -> Vec<WorldPos> {
            let mut out = Vec::<WorldPos>::new();
            if poly.len() < 3 {
                return out;
            }

            let inside = |s: f64| {
                if keep_positive { s >= -EPS } else { s <= EPS }
            };

            let mut prev = poly[poly.len() - 1];
            let mut prev_s = side_of(prev, centroid, split_on_x);
            let mut prev_in = inside(prev_s);

            for &curr in poly {
                let curr_s = side_of(curr, centroid, split_on_x);
                let curr_in = inside(curr_s);

                if prev_in != curr_in {
                    let denom = prev_s - curr_s;
                    if denom.abs() > EPS {
                        let t = (prev_s / (prev_s - curr_s)) as f32;
                        let p = lerp_on_segment(prev, curr, t);
                        push_unique(&mut out, p);
                    }
                }

                if curr_in {
                    push_unique(&mut out, curr);
                }

                prev = curr;
                prev_s = curr_s;
                prev_in = curr_in;
            }

            if out.len() >= 2 && out[0].distance_to(*out.last().unwrap()) <= POINT_EPS {
                out.pop();
            }

            out
        }

        fn collect_crossings(split_line: &[WorldPos], boundary: &[WorldPos]) -> Vec<Crossing> {
            let mut crossings = Vec::<Crossing>::new();
            let n = boundary.len();

            if split_line.len() < 2 || n < 3 {
                return crossings;
            }

            for si in 0..(split_line.len() - 1) {
                let s0 = split_line[si];
                let s1 = split_line[si + 1];

                for bi in 0..n {
                    let b0 = boundary[bi];
                    let b1 = boundary[(bi + 1) % n];

                    if let Some((t_b, t_s)) = segment_xz_intersect(b0, b1, s0, s1, 0.0) {
                        crossings.push(Crossing {
                            boundary_edge: bi,
                            t_boundary: t_b as f32,
                            split_pos: si as f32 + t_s as f32,
                            point: lerp_on_segment(b0, b1, t_b as f32),
                        });
                    }
                }
            }

            crossings.sort_by(|a, b| {
                a.split_pos
                    .partial_cmp(&b.split_pos)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut deduped = Vec::<Crossing>::with_capacity(crossings.len());
            for c in crossings {
                let keep = deduped
                    .last()
                    .map_or(true, |prev| prev.point.distance_to(c.point) > POINT_EPS);
                if keep {
                    deduped.push(c);
                }
            }

            deduped
        }

        fn build_half(
            boundary: &[WorldPos],
            start_bi: usize,
            end_bi: usize,
            start_pt: WorldPos,
            end_pt: WorldPos,
        ) -> Vec<WorldPos> {
            let n = boundary.len();
            let mut poly = Vec::<WorldPos>::new();

            poly.push(start_pt);

            let mut i = (start_bi + 1) % n;
            let stop = (end_bi + 1) % n;
            let mut steps = 0usize;

            while i != stop && steps < n {
                poly.push(boundary[i]);
                i = (i + 1) % n;
                steps += 1;
            }

            poly.push(end_pt);
            poly
        }

        let centroid = district.center;
        let boundary = &district.points;

        // ── 1. Find the best bisecting road edge ────────────────────────────────
        let mut best_line: Option<Vec<WorldPos>> = None;
        let mut best_crossing_count: usize = usize::MAX;
        let mut best_dist = f64::MAX;
        let mut seen_segments = std::collections::HashSet::new();

        for &lot_id in &district.lot_ids {
            let Some(lot) = zoning_storage.get_lot(lot_id) else {
                continue;
            };

            if !seen_segments.insert(lot.segment_id) {
                continue;
            }
            let Some(segment_id) = lot.segment_id else {
                continue;
            };
            let Some(road_edges) = road_edge_storage.get(&segment_id) else {
                continue;
            };

            let candidate = &road_edges.right_sidewalk_edge.right_points;
            if candidate.len() < 2 {
                continue;
            }

            let crossings = collect_crossings(candidate, boundary);
            if crossings.len() < 2 {
                continue;
            }

            let mid = candidate[candidate.len() / 2];
            let dist = centroid.distance_to(mid);

            let crossing_count = crossings.len();
            let better = match best_line {
                None => true,
                Some(_) => {
                    crossing_count < best_crossing_count
                        || (crossing_count == best_crossing_count && dist < best_dist)
                }
            };

            if better {
                best_crossing_count = crossing_count;
                best_dist = dist;
                best_line = Some(candidate.clone());
            }
        }

        // ── 2. Try road-edge split first ────────────────────────────────────────
        let mut poly_a: Option<Vec<WorldPos>> = None;
        let mut poly_b: Option<Vec<WorldPos>> = None;

        if let Some(split_line) = best_line {
            let crossings = collect_crossings(&split_line, boundary);

            if crossings.len() >= 2 {
                let entry = crossings.first().copied().unwrap();
                let exit = crossings.last().copied().unwrap();

                let mut cut: Vec<WorldPos> = Vec::new();
                cut.push(entry.point);

                let first_vi = entry.split_pos.ceil() as usize;
                let last_vi = exit.split_pos.floor() as usize;
                for vi in first_vi..=last_vi {
                    if vi < split_line.len() {
                        cut.push(split_line[vi]);
                    }
                }

                cut.push(exit.point);

                let mut a = build_half(
                    boundary,
                    entry.boundary_edge,
                    exit.boundary_edge,
                    entry.point,
                    exit.point,
                );

                if cut.len() > 2 {
                    for p in cut[1..cut.len() - 1].iter().rev() {
                        a.push(*p);
                    }
                }

                let mut b = build_half(
                    boundary,
                    exit.boundary_edge,
                    entry.boundary_edge,
                    exit.point,
                    entry.point,
                );

                if cut.len() > 2 {
                    for p in cut[1..cut.len() - 1].iter() {
                        b.push(*p);
                    }
                }

                if a.len() >= 3 && b.len() >= 3 {
                    poly_a = Some(a);
                    poly_b = Some(b);
                }
            }
        }

        // ── 3. Fallback: simple centroid split in two ──────────────────────────
        if poly_a.is_none() || poly_b.is_none() {
            let mut min_dx = 0.0f64;
            let mut max_dx = 0.0f64;
            let mut min_dz = 0.0f64;
            let mut max_dz = 0.0f64;

            for &p in boundary.iter().skip(1) {
                let dx = boundary[0].dx(p);
                let dz = boundary[0].dz(p);

                min_dx = min_dx.min(dx);
                max_dx = max_dx.max(dx);
                min_dz = min_dz.min(dz);
                max_dz = max_dz.max(dz);
            }

            let x_extent = max_dx - min_dx;
            let z_extent = max_dz - min_dz;

            let split_on_x = x_extent >= z_extent;

            let mut a = clip_halfspace(boundary, centroid, split_on_x, true);
            let mut b = clip_halfspace(boundary, centroid, split_on_x, false);

            if a.len() < 3 || b.len() < 3 {
                let a2 = clip_halfspace(boundary, centroid, !split_on_x, true);
                let b2 = clip_halfspace(boundary, centroid, !split_on_x, false);

                if a2.len() >= 3 && b2.len() >= 3 {
                    a = a2;
                    b = b2;
                }
            }

            if a.len() < 3 || b.len() < 3 {
                return DistrictSplitResult::DegeneratePolygon;
            }

            poly_a = Some(a);
            poly_b = Some(b);
        }

        let poly_a = poly_a.unwrap();
        let poly_b = poly_b.unwrap();

        if poly_a.len() < 3 || poly_b.len() < 3 {
            return DistrictSplitResult::DegeneratePolygon;
        }

        // ── 4. Assign lots: centroid-in-polygon test ────────────────────────────
        let mut lots_a: Vec<LotId> = Vec::new();
        let mut lots_b: Vec<LotId> = Vec::new();

        for &lot_id in &district.lot_ids {
            let Some(lot) = zoning_storage.get_lot(lot_id) else {
                lots_b.push(lot_id);
                continue;
            };

            let lot_centroid = WorldPos::centroid(&lot.bounds);
            if point_in_polygon_xz(lot_centroid, &poly_a) {
                lots_a.push(lot_id);
            } else {
                lots_b.push(lot_id);
            }
        }

        let mut rng = ThreadRng::default();
        let name = generate_district_name(&mut rng);

        let mut new_district = District::new(name, poly_b, DistrictType::AutomaticallyMade); // TODO: poly_b MUST be all points, and NOT a convex hull!
        new_district.lot_ids = lots_b;
        new_district.zoning_demand = ZoningDemand::new();

        DistrictSplitResult::Alright(new_district, (district_id, poly_a, lots_a))
    }

    pub fn add_immigrants(
        zoning_storage: &mut ZoningStorage,
        building_storage: &mut BuildingStorage,
        district_id: DistrictId,
        immigrants: Vec<Person>,
        rng: &mut impl Rng,
    ) {
        //println!("IMMIGRANTS: {}", immigrants.len());
        for person in immigrants {
            let Some(residence_id) =
                zoning_storage.get_residence(district_id, building_storage, person, rng)
            else {
                continue;
            };
            let Some(residence) = building_storage.get_mut(residence_id) else {
                continue;
            };
            //println!("IMMIGRANT moving into building id: {}", building_id);
            residence.occupancy.groups.add_age(person.age, 1);
            residence
                .occupancy
                .groups
                .add_education(person.age, person.education_level, 1);
            let Some(workplace_id) =
                zoning_storage.get_work_place(residence.pos, building_storage, person, rng)
            else {
                continue;
            };
            let Some(workplace) = building_storage.get_mut(workplace_id) else {
                continue;
            };
            workplace.occupancy.workers += 1;
            let Some(residence) = building_storage.get_mut(residence_id) else {
                continue;
            };
            residence.occupancy.employed_tenants += 1;
            //println!("Added immigrant, total tenants in building: {}, Building object address: {:p}, Building ID: {}", building.occupancy.tenants, building, building.id);
        }
    }
    pub fn remove_emigrants(
        zoning_storage: &mut ZoningStorage,
        building_storage: &mut BuildingStorage,
        district_id: DistrictId,
        emigrants: Vec<Person>,
        rng: &mut impl Rng,
    ) {
        //println!("EMIGRANTS: {}", emigrants.len());
        for person in emigrants {
            let Some(residence_id) =
                zoning_storage.get_residence(district_id, building_storage, person, rng)
            else {
                continue;
            };
            let Some(residence) = building_storage.get_mut(residence_id) else {
                continue;
            };
            residence.occupancy.groups.remove_age(person.age, 1);
            residence
                .occupancy
                .groups
                .remove_education(person.age, person.education_level, 1);
            let Some(workplace_id) =
                zoning_storage.get_work_place(residence.pos, building_storage, person, rng)
            else {
                continue;
            };
            let Some(workplace) = building_storage.get_mut(workplace_id) else {
                continue;
            };
            workplace.occupancy.workers -= 1;
            let Some(residence) = building_storage.get_mut(residence_id) else {
                continue;
            };
            residence.occupancy.employed_tenants -= 1;
            //println!("Removed emigrant, total tenants in building: {}, Building object address: {:p}, Building ID: {}", building.occupancy.tenants, building, building.id);
        }
    }
    pub fn add_births(
        zoning_storage: &mut ZoningStorage,
        building_storage: &mut BuildingStorage,
        district_id: DistrictId,
        births: u32,
        rng: &mut impl Rng,
    ) {
        let baby = Person {
            education_level: EducationLevel::None,
            age: 0,
        };
        let bb = 0..births;
        for _ in bb {
            // Add baby 🥰😳🖕
            let Some(residence_id) =
                zoning_storage.get_residence(district_id, building_storage, baby, rng)
            else {
                continue;
            };
            let Some(residence) = building_storage.get_mut(residence_id) else {
                continue;
            };
            residence.occupancy.groups.add_age(baby.age, 1);
            residence
                .occupancy
                .groups
                .add_education(baby.age, EducationLevel::None, 1);
        }
    }
}
#[derive(Default)]
pub struct DistrictUpdateCallback {
    pub district_id: DistrictId,
    pub average_land_value: Option<f32>,
    pub district_split: Option<DistrictSplitResult>,
    pub new_cars: Vec<(LotId, Option<Car>)>,
    pub new_buildings: Vec<(LotId, Option<Building>)>,
    pub corporate_taxes: i64,
    pub occupancy_updates: Vec<(BuildingId, BuildingOccupancy)>,
    pub land_value_updates: Vec<(LotId, f32)>,
    pub complaining_buildings: Vec<(BuildingId, WorldPos, Option<BuildingComplaint>)>,
    pub residential_capacity: u32,
    pub commercial_capacity: u32,
    pub industrial_capacity: u32,
    pub office_capacity: u32,
    pub new_age_groups: Groups,
    pub building_demography_updates: Vec<(BuildingId, Groups)>,
    pub total_births: u32,
    pub total_deaths: u32,
    pub new_workers: Vec<(BuildingId, BuildingId)>,
}
impl DistrictUpdateCallback {
    pub fn new(district_id: DistrictId) -> Self {
        Self {
            district_id,
            ..Default::default()
        }
    }
}
fn generate_building(terrain: &Terrain, lot: &Lot) -> Option<Building> {
    let Some(zoning_type) = lot.zoning_type else {
        return None;
    };

    let roof = match zoning_type {
        ZoningType::Residential => RoofType::Triangle(30.0),
        ZoningType::Commercial => RoofType::Flat,
        ZoningType::Industrial => RoofType::Flat,
        ZoningType::Office => RoofType::Flat,
    };
    let roof_material = match zoning_type {
        ZoningType::Residential => RoofMaterial::Shingles,
        ZoningType::Commercial => RoofMaterial::Metal,
        ZoningType::Industrial => RoofMaterial::Metal,
        ZoningType::Office => RoofMaterial::Shingles,
    };
    let wall_material = match zoning_type {
        ZoningType::Residential => WallMaterial::Paint(Color([0.9f32, 0.9, 0.9, 1.0])),
        ZoningType::Commercial => WallMaterial::Paint(Color([0.9f32, 0.9, 0.9, 1.0])),
        ZoningType::Industrial => WallMaterial::Paint(Color([0.4f32, 0.4, 0.4, 1.0])),
        ZoningType::Office => WallMaterial::Paint(Color([0.9f32, 0.9, 0.9, 1.0])),
    };
    let story_height = match zoning_type {
        ZoningType::Residential => 2.7,
        ZoningType::Commercial => 3.0,
        ZoningType::Industrial => 4.0,
        ZoningType::Office => 3.0,
    };
    let num_stories = match zoning_type {
        ZoningType::Residential => 2,
        ZoningType::Commercial => 1,
        ZoningType::Industrial => 2,
        ZoningType::Office => 3,
    };
    let garage = match zoning_type {
        ZoningType::Residential => Some(GarageParams {
            story_height,
            num_stories: 1,
        }),
        ZoningType::Commercial => None,
        ZoningType::Industrial => None,
        ZoningType::Office => None,
    };
    let miscellaneous = MiscBuildingParams {
        window_material_accent: Default::default(),
        solar_modules: false,
        antenna: false,
        usage: BuildingUsage::from_zoning_type(zoning_type),
    };
    let level0 = BuildingParams {
        roof,
        roof_material,
        wall_material,
        driveway_material: DrivewayMaterial::Bricks,
        story_height,
        num_stories,
        basement: Default::default(),
        garden: Default::default(),
        garage,
        miscellaneous: Default::default(),
    };
    let levels = RevisionedSmallVec(SmallVec::from_vec(vec![
        level0,
        BuildingParams::default(),
        BuildingParams::default(),
        BuildingParams::default(),
        BuildingParams::default(),
        BuildingParams::default(),
    ]));

    Some(Building {
        id: 631864891,
        pos: lot.center,
        segment_id: lot.segment_id,
        lot_id: lot.id,
        level: Default::default(),
        design_source: BuildingDesignSource::BuildingParams { levels },
        edit_id: None,
        prop_instance_ids: vec![],
        occupancy: Default::default(),
        misc: Default::default(),
    })
}

#[derive(Clone, Default)]
struct ZoningState {
    pub district_id: DistrictId,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[revisioned(revision = 1)]
pub enum ZoningType {
    Residential,
    Commercial,
    Industrial,
    Office,
}
impl ZoningType {
    pub fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::String(s) => match s.to_lowercase().as_str() {
                "residential" => Some(ZoningType::Residential),
                "commercial" => Some(ZoningType::Commercial),
                "industrial" => Some(ZoningType::Industrial),
                "office" => Some(ZoningType::Office),
                _ => None,
            },
            _ => None,
        }
    }
    pub fn is_workplace(&self) -> bool {
        // TODO: Too black and white for later... Later, I want buildings with multiple zoning types in percentages stored in the building (Or rather, lot layout?). So a Residential building with small shops on the ground floor can work, like in Baltimor, California. (New Hampshire)
        match self {
            ZoningType::Residential => false,
            ZoningType::Commercial => true,
            ZoningType::Industrial => true,
            ZoningType::Office => true,
        }
    }
}
impl Display for ZoningType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ZoningType::Residential => write!(f, "residential"),
            ZoningType::Commercial => write!(f, "commercial"),
            ZoningType::Industrial => write!(f, "industrial"),
            ZoningType::Office => write!(f, "office"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum PlacePos {
    Free(WorldPos, f64),
    RoadSnap(WorldPos, f64),
    CurrentDistrictFirstPoint(WorldPos, f64),
    CurrentDistrictPoint(WorldPos, f64),
    CurrentDistrictLastPoint(WorldPos, f64),
    OtherDistrictPoint(WorldPos, f64),
}

impl PlacePos {
    fn pos(self) -> WorldPos {
        match self {
            PlacePos::RoadSnap(pos, _)
            | PlacePos::Free(pos, _)
            | PlacePos::CurrentDistrictFirstPoint(pos, _)
            | PlacePos::CurrentDistrictPoint(pos, _)
            | PlacePos::CurrentDistrictLastPoint(pos, _)
            | PlacePos::OtherDistrictPoint(pos, _) => pos,
        }
    }

    fn dist(self) -> f64 {
        match self {
            PlacePos::Free(_, dist) => dist,
            PlacePos::RoadSnap(_, dist)
            | PlacePos::CurrentDistrictFirstPoint(_, dist)
            | PlacePos::CurrentDistrictPoint(_, dist)
            | PlacePos::CurrentDistrictLastPoint(_, dist)
            | PlacePos::OtherDistrictPoint(_, dist) => dist,
        }
    }
}

#[derive(Clone, Default)]
pub struct Zoning {
    pub ticker: Ticker,
    zoning_state: Option<ZoningState>,
    pub zoning_storage: ZoningStorage,
}

impl Zoning {
    pub fn new() -> Self {
        Self {
            ticker: Ticker::new(1.0),
            zoning_state: None,
            zoning_storage: ZoningStorage::new(),
        }
    }
    pub fn update(
        &mut self,
        camera: &Camera,
        terrain: &mut Terrain,
        buildings: &mut Buildings,
        roads: &Roads,
        road_mesh_manager: &RoadMeshManager,
        input: &mut Input,
        variables: &Variables,
        props: &mut Props,
        gizmo: &mut Gizmo,
    ) {
        self.zoning_storage.update_target(camera.target.chunk);
        if terrain.cursor.mode != CursorMode::Area && terrain.cursor.mode != CursorMode::Zoning {
            return;
        };
        let Some(new_zone_type) = terrain.cursor.zoning_type else {
            return;
        };
        let active_district_id = self.zoning_state.as_ref().map(|state| state.district_id);
        for district in self.zoning_storage.iter_districts() {
            if Some(district.id) != active_district_id {
                draw_area(
                    district.points.as_slice(),
                    None,
                    variables,
                    gizmo,
                    Some([1.0, 1.0, 1.0, 0.2]),
                    None,
                );
            }
            //println!("{}", district.points.as_slice().len());
            gizmo.polyline(
                district.points.as_slice(),
                [0.14, 0.21, 0.5, 0.5],
                0.0,
                true,
                0.1,
                0.0,
            );

            gizmo.text(
                district.name.clone(),
                district.center,
                1.0,
                [0.05, 0.05, 0.05, 0.4],
                None,
                true,
                0.0,
                0.0,
            )

            // This was to test if the lateral is ACTUALLY right! It was perfect!
            // for idx in 0..district.points.len() {
            //     let (_, right) = tangent_and_lateral_right(&*district.points, idx, camera.chunk_size);
            //     gizmo.direction(district.points[idx], right, [1.0, 0.0, 0.0, 1.0], 0.0, 0.0);
            // }
        }
        // After this, if the mouse is over the sky or UI, stuff doesn't get rendered cuz there is no picked point ofc!
        let Some(picked) = terrain.last_picked.clone() else {
            return;
        };

        let mut closest_point = gather_closest_lot_point(roads, road_mesh_manager, picked.pos);

        for segment_id in roads
            .road_manager
            .roads
            .segment_ids_touching_chunks(&picked.chunk.chunk_coord.get_chunks_3x3())
        {
            let Some(road_edges) = road_mesh_manager.road_edge_storage.get(&segment_id) else {
                continue;
            };

            for point in road_edges
                .right_sidewalk_edge
                .right_points
                .iter()
                .chain(road_edges.left_sidewalk_edge.right_points.iter())
            {
                if point.distance_squared(picked.pos) < SNAP_RADIUS * SNAP_RADIUS {
                    gizmo.circle(*point, 0.4, [0.8, 0.3, 1.0, 1.0], 0.1, 0.0);
                }
            }

            if road_edges.right_sidewalk_edge.is_empty() || road_edges.left_sidewalk_edge.is_empty()
            {
                for edges in road_edges.lane_edges.values() {
                    for point in &edges.right_points {
                        if point.distance_squared(picked.pos) < SNAP_RADIUS * SNAP_RADIUS {
                            gizmo.circle(*point, 0.4, [0.8, 0.3, 1.0, 1.0], 0.1, 0.0);
                        }
                    }
                }
            }
        }

        let lot_snap_point = closest_point.take().and_then(|snap_point| {
            if snap_point.dist < SNAP_RADIUS {
                gizmo.circle(snap_point.pos, 1.0, [0.8, 0.3, 1.0, 1.0], 0.15, 0.0);
                Some(snap_point)
            } else {
                None
            }
        });

        match terrain.cursor.mode.clone() {
            CursorMode::Zoning => {
                self.run_lot_zoning(
                    terrain,
                    buildings,
                    roads,
                    road_mesh_manager,
                    input,
                    variables,
                    lot_snap_point,
                    &picked,
                    new_zone_type,
                    props,
                    gizmo,
                    None,
                );
            }
            CursorMode::Area => {
                self.run_area_zoning(
                    terrain,
                    roads,
                    input,
                    variables,
                    &picked,
                    active_district_id,
                    new_zone_type,
                    gizmo,
                );
            }
            _ => {}
        }
    }
    pub fn average_land_value(&self, district_id: DistrictId) -> f32 {
        let Some(district) = self.zoning_storage.get_district(district_id) else {
            return 0.0;
        };
        let (sum, count) = district
            .lot_ids
            .iter()
            .flat_map(|lot_id| self.zoning_storage.get_lot(*lot_id))
            .fold((0.0f32, 0usize), |(s, c), lot| (s + lot.land_value, c + 1));
        if count == 0 { 0.0 } else { sum / count as f32 }
    }
    pub fn update_districts(
        gizmo: &mut Gizmo,
        terrain: &Terrain,
        zoning: &mut Zoning,
        buildings: &mut Buildings,
        car_storage: &mut CarStorage,
        city_state: &mut CityState,
        time: &Time,
        road_storage: &RoadStorage,
        road_edge_storage: &RoadEdgeStorage,
        target_pos: WorldPos,
    ) {
        let callbacks: Vec<DistrictUpdateCallback> = zoning
            .zoning_storage
            .get_district_ids_to_tick(time.sim_time())
            .par_iter()
            .map(|&district_id| {
                District::update(
                    terrain,
                    time,
                    buildings,
                    zoning,
                    &city_state.schedule,
                    district_id,
                    road_edge_storage,
                    target_pos.chunk,
                )
            })
            .collect();

        let rng = &mut ThreadRng::default();

        for callback in callbacks {
            let mut immigration_changes: Option<(Vec<Person>, Vec<Person>)> = None;

            for (building_id, occ) in callback.occupancy_updates {
                // MUST BE FIRST
                if let Some(building) = buildings.storage.get_mut(building_id) {
                    building.occupancy = occ;
                }
            }

            for (building_id, groups) in callback.building_demography_updates {
                if let Some(building) = buildings.storage.get_mut(building_id) {
                    building.occupancy.groups = groups;
                }
            }

            if let Some(district) = zoning.zoning_storage.get_mut_district(callback.district_id) {
                district.zoning_demand.demography.age_groups = callback.new_age_groups;
                district.zoning_demand.demography.population = district
                    .zoning_demand
                    .demography
                    .age_groups
                    .whole_population();

                if let Some(average_land_value) = callback.average_land_value {
                    let (taxes, i, e) =
                        district
                            .zoning_demand
                            .update_demands(rng, time, average_land_value);
                    immigration_changes = Some((i, e));

                    city_state.economy.add_money(taxes);
                }

                // Corporate taxes computed per-lot above, fully separate
                if callback.corporate_taxes > 0 {
                    city_state.economy.add_money(callback.corporate_taxes);
                    district.zoning_demand.total_taxes_collected += callback.corporate_taxes;
                }

                district.zoning_demand.residential_capacity = callback.residential_capacity;
                district.zoning_demand.commercial_capacity = callback.commercial_capacity;
                district.zoning_demand.industrial_capacity = callback.industrial_capacity;
                district.zoning_demand.office_capacity = callback.office_capacity;
                for &(residence_id, workplace_id) in callback.new_workers.iter() {
                    if let Some(workplace) = buildings.storage.get_mut(workplace_id) {
                        workplace.occupancy.workers += 1;
                        let Some(residence) = buildings.storage.get_mut(residence_id) else {
                            continue;
                        };
                        residence.occupancy.employed_tenants += 1;
                    }
                }

                //println!("Callback Population: {}", callback.population);

                district.zoning_demand.demography.update(
                    time,
                    callback.total_births,
                    callback.total_deaths,
                );
            }

            District::add_births(
                &mut zoning.zoning_storage,
                &mut buildings.storage,
                callback.district_id,
                callback.total_births,
                rng,
            );

            if let Some((immigrants, emigrants)) = immigration_changes {
                // Add immigrants
                District::add_immigrants(
                    &mut zoning.zoning_storage,
                    &mut buildings.storage,
                    callback.district_id,
                    immigrants,
                    rng,
                );
                // Remove emigrants and dead people
                District::remove_emigrants(
                    &mut zoning.zoning_storage,
                    &mut buildings.storage,
                    callback.district_id,
                    emigrants,
                    rng,
                );
            }

            if let Some(split_result) = callback.district_split {
                match split_result {
                    DistrictSplitResult::Alright(
                        new_other_district,
                        (district_id, poly_a, lots_a),
                    ) => {
                        if let Some(district) = zoning.zoning_storage.get_mut_district(district_id)
                        {
                            district.replace_points(poly_a);
                            district.lot_ids = lots_a;
                            district.district_type = DistrictType::AutomaticallyMade;
                        }

                        zoning.zoning_storage.spawn_district(new_other_district);
                    }
                    _ => {}
                };
            }

            for (lot_id, maybe_car) in callback.new_cars {
                if let Some(car) = maybe_car {
                    car_storage.spawn(car, road_storage);
                }
            }

            for (lot_id, maybe_building) in callback.new_buildings {
                if let Some(building) = maybe_building {
                    let building_id = BuildingStorage::spawn(buildings, zoning, building);

                    if let Some(lot) = zoning
                        .zoning_storage
                        .lots
                        .get_mut(lot_id as usize)
                        .and_then(|lot| lot.as_mut())
                    {
                        lot.building_id = Some(building_id);
                    }
                }
            }

            for (lot_id, land_value) in callback.land_value_updates {
                if let Some(lot) = zoning
                    .zoning_storage
                    .lots
                    .get_mut(lot_id as usize)
                    .and_then(|lot| lot.as_mut())
                {
                    lot.land_value = land_value;
                }
            }
            for complainant in callback.complaining_buildings.into_iter() {
                let Some(complaint) = complainant.2 else {
                    continue;
                };
                let pos = complainant.1;
                //println!("{:?}", complaint);
                match complaint {
                    BuildingComplaint::NotEnoughWorkers => {
                        gizmo.text(
                            "Not enough Workers!",
                            pos,
                            3.0,
                            [1.0, 0.0, 0.0, 1.0],
                            None,
                            false,
                            0.0,
                            1.0,
                        );
                    }
                    BuildingComplaint::NotEnoughCustomers => {
                        gizmo.text(
                            "Not enough Customers!",
                            pos,
                            3.0,
                            [1.0, 0.0, 0.0, 1.0],
                            None,
                            false,
                            0.0,
                            1.0,
                        );
                    }
                    BuildingComplaint::NotEnoughJobs => {
                        gizmo.text(
                            "Not enough Jobs!",
                            pos,
                            3.0,
                            [1.0, 0.0, 0.0, 1.0],
                            None,
                            false,
                            0.0,
                            1.0,
                        );
                    }
                }
            }
        }

        // for chunk in terrain.visible.iter().map(|v|v.coords.chunk_coord) {
        //     let Some(lots) =
        // }
    }
    fn can_place_zoning_point(
        &self,
        gizmo: &mut Gizmo,
        place_pos: PlacePos,
        active_district_id: Option<DistrictId>,
    ) -> bool {
        let Some(district_id) = active_district_id else {
            return true;
        };
        let Some(district) = self.zoning_storage.get_district(district_id) else {
            return false;
        };

        let Some(&last) = district.points.last() else {
            return false;
        };
        gizmo.line(last, place_pos.pos(), [0.0, 1.0, 0.0, 1.0], 0.5, 0.0);
        if self.segment_intersects_any_district(place_pos.pos(), last, active_district_id) {
            return false;
        };
        //println!("{:?}", place_pos);
        match place_pos {
            PlacePos::Free(_, _) => true,
            PlacePos::RoadSnap(_, _) => true,
            PlacePos::CurrentDistrictFirstPoint(_, _) => {
                let point_amount_requirement = district.points.len() >= 3;

                let enclosing_another_area = self
                    .zoning_storage
                    .iter_districts()
                    .filter(|other_district| other_district.id != district.id)
                    .any(|other_district| {
                        other_district
                            .points
                            .iter()
                            .any(|point| point_in_polygon_xz(*point, &district.points))
                    });

                point_amount_requirement && !enclosing_another_area
            }
            PlacePos::CurrentDistrictPoint(_, _) => false,
            PlacePos::CurrentDistrictLastPoint(_, _) => false,
            PlacePos::OtherDistrictPoint(_, _) => true,
        }
    }

    fn segment_intersects_any_district(
        &self,
        a: WorldPos,
        b: WorldPos,
        active_district_id: Option<DistrictId>,
    ) -> bool {
        const CLEARANCE: f64 = 0.0; // 1 meter clearance

        for district in self.zoning_storage.iter_districts() {
            let points = &district.points;
            if points.len() < 2 {
                continue;
            }

            for edge in points.windows(2) {
                let c = edge[0];
                let d = edge[1];

                if segment_xz_intersect(a, b, c, d, CLEARANCE).is_some() {
                    //println!("first one, {}", a.local.x);
                    return true;
                }
            }

            if Some(district.id) != active_district_id && points.len() >= 3 {
                let c = *points.last().unwrap();
                let d = points[0];

                if segment_xz_intersect(a, b, c, d, CLEARANCE).is_some() {
                    //println!("second one");
                    return true;
                }
            }
        }

        false
    }
    fn lot_intersects_any_road(
        &self,
        gizmo: &mut Gizmo,
        roads: &Roads,
        road_mesh_manager: &RoadMeshManager,
        lot_points: &[WorldPos],
    ) -> bool {
        if lot_points.len() < 4 {
            return false;
        }

        let segment_ids = lot_points
            .iter()
            .map(|point| {
                roads
                    .road_manager
                    .roads
                    .segment_ids_touching_chunk(point.chunk)
            })
            .flatten()
            .collect::<Vec<SegmentId>>();

        for segment_id in segment_ids.iter() {
            let Some(edges) = road_mesh_manager.road_edge_storage.get(segment_id) else {
                continue;
            };

            let road_points = collect_road_points(edges);

            for point in road_points {
                if point_in_polygon_xz(*point, lot_points) {
                    return true;
                }
            }
        }

        false
    }

    fn lot_intersects_any_lot(&self, points: &[WorldPos], ignore_lot_id: Option<LotId>) -> bool {
        if points.len() < 4 {
            return false;
        }

        for existing_lot in self.zoning_storage.iter_lots() {
            if Some(existing_lot.id) == ignore_lot_id {
                continue;
            }

            if existing_lot.bounds.len() < 4 {
                continue;
            }

            match self.lot_intersects_lot(points, existing_lot.bounds.as_slice()) {
                None => continue,
                Some(_) => return true,
            }
        }

        false
    }

    fn lot_intersects_lot(
        &self,
        points_a: &[WorldPos],
        points_b: &[WorldPos],
    ) -> Option<SegmentIntersectionType> {
        if points_a.len() < 4 || points_b.len() < 4 {
            return None;
        }

        const CLEARANCE: f64 = 0.05;

        for edge_a in points_a.windows(2) {
            let a = edge_a[0];
            let b = edge_a[1];

            for edge_b in points_b.windows(2) {
                let c = edge_b[0];
                let d = edge_b[1];

                if segment_xz_intersect(a, b, c, d, CLEARANCE).is_some() {
                    return Some(SegmentIntersectionType::OtherEdges);
                }
            }
        }

        if points_a[0].in_polygon(points_b) || points_b[0].in_polygon(points_a) {
            return Some(SegmentIntersectionType::PointInsidePolygon);
        }

        None
    }

    pub fn run_lot_zoning(
        &mut self,
        terrain: &mut Terrain,
        buildings: &mut Buildings,
        roads: &Roads,
        road_mesh_manager: &RoadMeshManager,
        input: &mut Input,
        variables: &Variables,
        lot_snap_point: Option<LotPoint>,
        picked: &PickedPoint,
        new_zoning_type: ZoningType,
        props: &mut Props,
        gizmo: &mut Gizmo,
        preview_lot_id: Option<LotId>,
    ) -> bool {
        let mut inside_lot_id = None;

        for lot in self.zoning_storage.iter_lots() {
            if point_in_polygon_xz(picked.pos, lot.bounds.as_slice()) {
                inside_lot_id = Some(lot.id);
                break;
            }
        }
        if let Some(lot_id) = inside_lot_id {
            if inside_lot_id == preview_lot_id {
                let fit = self.zoning_storage.get_lot(lot_id).and_then(|lot| {
                    let target = lot.entrance.pos;

                    let snap = lot_snap_point.clone().or_else(|| {
                        let snapped = gather_closest_lot_point(roads, road_mesh_manager, target)?;

                        (snapped.dist <= SNAP_RADIUS).then_some(snapped)
                    })?;

                    fit_lot_to_neighbors(
                        &self.zoning_storage,
                        road_mesh_manager,
                        &snap,
                        Some(lot_id),
                    )
                });

                let valid = if let Some(fit) = &fit {
                    let mut preview = fit.bounds.clone();

                    for point in &mut preview {
                        point.local.y = terrain.get_height_at(*point, true);
                    }

                    let intersects_lot =
                        self.lot_intersects_any_lot(preview.as_slice(), Some(lot_id));

                    let intersects_roads = self.lot_intersects_any_road(
                        gizmo,
                        roads,
                        road_mesh_manager,
                        preview.as_slice(),
                    );

                    let invalid = intersects_lot || intersects_roads;

                    gizmo.polyline(
                        preview.as_slice(),
                        if invalid {
                            [0.8, 0.1, 0.1, 1.0]
                        } else {
                            [0.1, 0.8, 0.1, 1.0]
                        },
                        8.0,
                        false,
                        if invalid { 0.25 } else { 0.2 },
                        0.0,
                    );

                    !invalid
                } else {
                    if let Some(lot) = self.zoning_storage.get_lot(lot_id) {
                        gizmo.polyline(
                            lot.bounds.as_slice(),
                            [0.8, 0.1, 0.1, 1.0],
                            8.0,
                            false,
                            0.25,
                            0.0,
                        );
                    }

                    false
                };

                for lot in self.zoning_storage.iter_lots() {
                    if lot.id == lot_id {
                        draw_area(
                            lot.bounds.as_slice(),
                            lot.zoning_type,
                            variables,
                            gizmo,
                            Some([1.2, 1.2, 1.2, 1.0]),
                            Some(new_zoning_type),
                        );
                    } else {
                        draw_area(
                            lot.bounds.as_slice(),
                            lot.zoning_type,
                            variables,
                            gizmo,
                            Some([1.0, 1.0, 1.0, 0.3]),
                            None,
                        );
                    }

                    gizmo.polyline(
                        lot.bounds.as_slice(),
                        [0.1, 0.3, 0.7, 0.8],
                        10.0,
                        false,
                        0.10,
                        0.0,
                    );
                }

                return valid;
            }
        }

        let removing_lot = input.action_down("Remove Lot");
        let finished_removing_lot = input.action_released("Remove Lot");

        if let Some(lot_id) = inside_lot_id {
            let mut fit: Option<LotFit> = None;

            if let Some(lot) = self.zoning_storage.get_lot(lot_id) {
                let target = lot.entrance.pos;

                let snap = lot_snap_point.clone().or_else(|| {
                    let snapped = gather_closest_lot_point(roads, road_mesh_manager, target)?;

                    if snapped.dist > SNAP_RADIUS {
                        return None;
                    }

                    Some(snapped)
                });

                if let Some(snap) = snap {
                    fit = fit_lot_to_neighbors(
                        &self.zoning_storage,
                        road_mesh_manager,
                        &snap,
                        Some(lot_id),
                    );
                }
            }

            for lot in self.zoning_storage.iter_mut_lots() {
                if inside_lot_id == Some(lot.id) {
                    if removing_lot {
                        draw_area(
                            lot.bounds.as_slice(),
                            lot.zoning_type,
                            variables,
                            gizmo,
                            Some([3.0, 0.2, 0.2, 1.0]),
                            Some(new_zoning_type),
                        );
                    } else {
                        if let Some(fit) = &fit {
                            gizmo.polyline(
                                fit.bounds.as_slice(),
                                [0.1, 0.8, 0.1, 1.0],
                                8.0,
                                false,
                                0.2,
                                0.0,
                            );
                        } else {
                            gizmo.polyline(
                                lot.bounds.as_slice(),
                                [0.8, 0.1, 0.1, 1.0],
                                8.0,
                                false,
                                0.25,
                                0.0,
                            );
                        }

                        draw_area(
                            lot.bounds.as_slice(),
                            lot.zoning_type,
                            variables,
                            gizmo,
                            Some([1.2, 1.2, 1.2, 1.0]),
                            Some(new_zoning_type),
                        );

                        if input.action_pressed_once("Place Zoning Point") {
                            if let Some(fit) = fit.clone() {
                                lot.bounds = fit.bounds;
                                lot.center =
                                    WorldPos::centroid(&lot.bounds[..lot.bounds.len() - 1]);
                                lot.entrance =
                                    LotEntrance::new(fit.snap_point.pos, fit.snap_point.lateral);
                                lot.segment_id = Some(fit.snap_point.segment_id);
                                lot.zoning_type = Some(new_zoning_type);
                            }
                        }
                    }
                } else {
                    draw_area(
                        lot.bounds.as_slice(),
                        lot.zoning_type,
                        variables,
                        gizmo,
                        Some([1.0, 1.0, 1.0, 0.3]),
                        None,
                    );
                }

                gizmo.polyline(
                    lot.bounds.as_slice(),
                    [0.1, 0.3, 0.7, 0.8],
                    10.0,
                    false,
                    0.10,
                    0.0,
                );
            }

            if removing_lot {
                if let Some(building_id) = self
                    .zoning_storage
                    .get_lot(lot_id)
                    .and_then(|lot| lot.building_id)
                {
                    BuildingStorage::despawn(
                        buildings,
                        self,
                        &mut terrain.terrain_editor,
                        props,
                        building_id,
                    );
                }

                self.zoning_storage.despawn_lot(lot_id);
            }

            return true;
        }

        let Some(snap_point) = lot_snap_point else {
            for lot in self.zoning_storage.iter_lots() {
                draw_area(
                    lot.bounds.as_slice(),
                    lot.zoning_type,
                    variables,
                    gizmo,
                    Some([1.0, 1.0, 1.0, 0.3]),
                    None,
                );

                gizmo.polyline(
                    lot.bounds.as_slice(),
                    [0.1, 0.3, 0.7, 0.8],
                    10.0,
                    false,
                    0.10,
                    0.0,
                );
            }

            return false;
        };

        gizmo.circle(snap_point.pos, 0.3, [0.1, 0.3, 0.8, 1.0], 0.1, 0.0);

        let fit = fit_lot_to_neighbors(&self.zoning_storage, road_mesh_manager, &snap_point, None);

        if let Some(fit) = &fit {
            let mut preview = fit.bounds.clone();

            for point in &mut preview {
                point.local.y = terrain.get_height_at(*point, true);
            }

            let intersects_lot = self.lot_intersects_any_lot(preview.as_slice(), None);

            let intersects_roads =
                self.lot_intersects_any_road(gizmo, roads, road_mesh_manager, preview.as_slice());

            let invalid = intersects_lot || intersects_roads;

            if invalid {
                gizmo.polyline(
                    preview.as_slice(),
                    [0.8, 0.1, 0.1, 1.0],
                    8.0,
                    false,
                    0.25,
                    0.0,
                );
            } else {
                gizmo.polyline(
                    preview.as_slice(),
                    [0.1, 0.8, 0.1, 1.0],
                    8.0,
                    false,
                    0.2,
                    0.0,
                );

                if input.action_repeat("Place Zoning Point") {
                    let lot_center = WorldPos::centroid(&preview[..preview.len() - 1]);

                    let lot = Lot {
                        id: 6945220,
                        bounds: preview,
                        bounds_version: 0,
                        center: lot_center,
                        entrance: LotEntrance::new(fit.snap_point.pos, fit.snap_point.lateral),
                        layout: None,
                        zoning_type: Some(new_zoning_type),
                        segment_id: Some(fit.snap_point.segment_id),
                        district_id: 6378186,
                        building_id: None,
                        land_value: self.zoning_storage.sample_land_value(lot_center.chunk),
                    };

                    self.zoning_storage.spawn_lot(lot);
                }
            }

            for lot in self.zoning_storage.iter_lots() {
                draw_area(
                    lot.bounds.as_slice(),
                    lot.zoning_type,
                    variables,
                    gizmo,
                    Some([1.0, 1.0, 1.0, 0.3]),
                    None,
                );

                gizmo.polyline(
                    lot.bounds.as_slice(),
                    [0.1, 0.3, 0.7, 0.8],
                    10.0,
                    false,
                    0.10,
                    0.0,
                );
            }

            return !invalid;
        }

        let width = LOT_PREFERRED_FRONTAGE as f32;
        let origin = snap_point
            .pos
            .add_vec3(snap_point.lateral * LOT_FRONT_OFFSET as f32);

        let half_width = width * 0.5;

        let front_left = origin.sub_vec3(snap_point.tangent * half_width);

        let front_right = origin.add_vec3(snap_point.tangent * half_width);

        let back_left = front_left.add_vec3(snap_point.lateral * LOT_MIN_DEPTH as f32);

        let back_right = front_right.add_vec3(snap_point.lateral * LOT_MIN_DEPTH as f32);

        gizmo.polyline(
            &[front_left, front_right, back_right, back_left],
            [0.8, 0.1, 0.1, 1.0],
            8.0,
            true,
            0.25,
            0.0,
        );

        for lot in self.zoning_storage.iter_lots() {
            draw_area(
                lot.bounds.as_slice(),
                lot.zoning_type,
                variables,
                gizmo,
                Some([1.0, 1.0, 1.0, 0.3]),
                None,
            );

            gizmo.polyline(
                lot.bounds.as_slice(),
                [0.1, 0.3, 0.7, 0.8],
                10.0,
                false,
                0.10,
                0.0,
            );
        }

        false
    }
    fn run_area_zoning(
        &mut self,
        terrain: &Terrain,
        roads: &Roads,
        input: &mut Input,
        variables: &Variables,
        picked: &PickedPoint,
        active_district_id: Option<DistrictId>,
        new_zone_type: ZoningType,
        gizmo: &mut Gizmo,
    ) {
        let mut best_place: PlacePos = PlacePos::Free(picked.pos, 0.0);

        let mut consider = |candidate: PlacePos| {
            use PlacePos::*;

            let priority = |p: &PlacePos| match p {
                CurrentDistrictFirstPoint(_, _) => 5,
                CurrentDistrictLastPoint(_, _) => 4,
                CurrentDistrictPoint(_, _) => 3,
                OtherDistrictPoint(_, _) => 2,
                RoadSnap(_, _) => 1,
                Free(_, _) => 0,
            };

            let best_pri = priority(&best_place);
            let cand_pri = priority(&candidate);
            let best_dist = best_place.dist();
            let cand_dist = candidate.dist();

            let should_replace = match (
                best_pri.cmp(&cand_pri),
                best_dist >= SNAP_RADIUS,
                cand_dist < SNAP_RADIUS,
            ) {
                (std::cmp::Ordering::Less, _, true) => true, // Promote if candidate in radius
                (std::cmp::Ordering::Greater, true, _) => true, // Demote if best outside radius
                (std::cmp::Ordering::Equal, _, _) => cand_dist < best_dist, // Same priority: closer wins
                _ => false,
            };

            if should_replace {
                best_place = candidate;
            }
        };

        if let Some(projection) = roads
            .road_manager
            .roads
            .closest_lane_point_to(picked.pos)
            .filter(|projection| projection.distance <= SNAP_RADIUS)
        {
            consider(PlacePos::RoadSnap(projection.position, projection.distance));
        }

        if let Some(district_id) = active_district_id {
            if let Some(district) = self.zoning_storage.get_district(district_id) {
                let len = district.points.len();

                for (i, point) in district.points.iter().copied().enumerate() {
                    let dist = point.distance_to(picked.pos);

                    if dist > SNAP_RADIUS {
                        continue;
                    }

                    let candidate = if i == 0 {
                        PlacePos::CurrentDistrictFirstPoint(point, dist)
                    } else if i + 1 == len {
                        PlacePos::CurrentDistrictLastPoint(point, dist)
                    } else {
                        PlacePos::CurrentDistrictPoint(point, dist)
                    };
                    //println!("Considering: {:?}", candidate);
                    consider(candidate);
                }
            }
        }

        for district in self.zoning_storage.iter_districts() {
            if Some(district.id) == active_district_id {
                continue;
            }

            for point in district.points.iter().copied() {
                let dist = point.distance_to(picked.pos);

                if dist <= SNAP_RADIUS {
                    consider(PlacePos::OtherDistrictPoint(point, dist));
                }
            }
        }

        let can_place = self.can_place_zoning_point(gizmo, best_place, active_district_id);

        let preview_color = if can_place {
            [0.1, 0.3, 0.7, 1.0]
        } else {
            [0.7, 0.1, 0.1, 1.0]
        };
        let mut inside_district_id: Option<DistrictId> = None;
        let inside_lot_id: Option<LotId> = None;
        if can_place && matches!(best_place, PlacePos::Free(_, _)) {
            for district in self.zoning_storage.iter_districts() {
                if Some(district.id) == active_district_id {
                    continue;
                }
                if point_in_polygon_xz(best_place.pos(), &district.points) {
                    // best_pos is inside this district
                    inside_district_id.replace(district.id);
                    // for lot in district
                    //     .lots
                    //     .iter()
                    //     .map(|lot_id| self.zoning_storage.get_lot(*lot_id))
                    // {
                    //     let Some(lot) = lot else { continue };
                    //     if point_inside_polygon(
                    //         best_place.pos(),
                    //         &lot.bounds,
                    //         gizmo.chunk_size,
                    //         false,
                    //     ) {
                    //         inside_lot_id.replace(lot.id);
                    //     }
                    // }
                }
            }
        }

        let canceling = input.action_pressed_once("Cancel");
        if canceling {
            if let Some(zoning_state) = &self.zoning_state {
                let district_id = zoning_state.district_id;
                if let Some(district) = self.zoning_storage.get_mut_district(district_id) {
                    district.points.pop();
                    if district.points.is_empty() {
                        self.zoning_storage.despawn_district(district_id);
                        self.zoning_state = None;
                    }
                }
            }
        }
        if let Some(id) = inside_district_id {
            if input.action_pressed_once("Place Zoning Point") {
                if let Some(district) = self.zoning_storage.get_mut_district(id) {
                    //district.district_type = *new_district_type; // Update zoning type of district, but district don't actually have a zoning type so whatever...
                }
            }
            if let Some(district) = self.zoning_storage.get_district(id) {
                draw_area(
                    district.points.as_slice(),
                    None,
                    variables,
                    gizmo,
                    Some([1.1, 1.1, 1.1, 1.0]),
                    Some(new_zone_type),
                );
                // for lot_id in &district.lots {
                //     let Some(lot) = self.zoning_storage.get_lot(*lot_id) else {
                //         continue;
                //     };
                //
                //     if Some(*lot_id) == inside_lot_id {
                //         // mouse inside lot, highlighted
                //         draw_district_area(
                //             lot.bounds.as_slice(),
                //             &lot.district_type,
                //             variables,
                //             gizmo,
                //             Some([1.1, 1.1, 1.1, 1.0]),
                //             Some(new_district_type),
                //         );
                //         gizmo.polyline(
                //             lot.bounds.as_slice(),
                //             [0.07, 0.48, 0.5, 1.0],
                //             0.0,
                //             0.15,
                //             0.0,
                //         );
                //     } else {
                //         // Normal lot drawing, not highlighted
                //         draw_district_area(
                //             lot.bounds.as_slice(),
                //             &lot.district_type,
                //             variables,
                //             gizmo,
                //             Some([1.0, 1.0, 1.0, 1.0]),
                //             Some(new_district_type),
                //         );
                //         gizmo.polyline(
                //             lot.bounds.as_slice(),
                //             [0.07, 0.48, 0.5, 1.0],
                //             0.0,
                //             0.1,
                //             0.0,
                //         );
                //     }
                // }
            }
        } else {
            gizmo.circle(best_place.pos(), 0.5, preview_color, 0.0, 0.0);

            if let Some(last_pos) = active_district_id
                .and_then(|district_id| self.zoning_storage.get_district(district_id))
                .and_then(|district| district.points.last().copied())
            {
                gizmo.line(best_place.pos(), last_pos, preview_color, 0.0, 0.0);
            }

            if input.action_pressed_once("Place Zoning Point") && can_place {
                if let Some(zoning_state) = &self.zoning_state {
                    let district_id = zoning_state.district_id;
                    match best_place {
                        PlacePos::Free(pos, _) => {
                            if let Some(district) =
                                self.zoning_storage.get_mut_district(district_id)
                            {
                                district.add_point(pos);
                            }
                        }
                        PlacePos::CurrentDistrictFirstPoint(pos, _) => {
                            if let Some(district) =
                                self.zoning_storage.get_mut_district(district_id)
                            {
                                if district.points.len() >= 3 {
                                    district.add_point(pos);
                                    //println!("Closed zoning loop");
                                    self.zoning_state = None;
                                }
                            }
                            if self.zoning_state.is_none() {
                                //self.zoning_storage.calculate_lots_for_district(district_id);
                                // ↑ Doesn't exist, is bullshit anyway now, because Districts just have LotIds which reference Lots which are the actual lots, pretty independent and that's good.
                            }
                        }

                        PlacePos::RoadSnap(pos, _) => {
                            if let Some(district) =
                                self.zoning_storage.get_mut_district(district_id)
                            {
                                district.add_point(pos);
                            }
                        }

                        PlacePos::CurrentDistrictPoint(_, _) => {}
                        PlacePos::CurrentDistrictLastPoint(_, _) => {}
                        PlacePos::OtherDistrictPoint(pos, _) => {
                            if let Some(district) =
                                self.zoning_storage.get_mut_district(district_id)
                            {
                                district.add_point(pos);
                            }
                        }
                    }
                } else {
                    let mut rng = ThreadRng::default();
                    let name = generate_district_name(&mut rng);
                    let district_id = self.zoning_storage.spawn_district(District::new(
                        name,
                        vec![best_place.pos()],
                        DistrictType::PlayerMade,
                    ));
                    self.zoning_state = Some(ZoningState { district_id });
                }
            }
        }
    }
}

pub type DistrictId = u32;
pub type LotId = u32;
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[revisioned(revision = 1)]
pub enum TileType {
    Grass,
    Tree,
    Garden,
    House,
    HouseBalcony,
    HouseEntrance,
    LotEntrance,
    Garage,
    Driveway,
}
impl TileType {
    #[inline]
    pub fn is_drivable(&self) -> bool {
        match self {
            TileType::Grass => false,
            TileType::Tree => false,
            TileType::Garden => false,
            TileType::House => false,
            TileType::HouseBalcony => false,
            TileType::HouseEntrance => false,
            TileType::LotEntrance => true,
            TileType::Garage => true, // TODO: lol drive through he garage
            TileType::Driveway => true,
        }
    }
}
#[derive(Debug, Serialize, Deserialize, Clone)]
#[revisioned(revision = 1)]
pub enum Tile {
    Square(TileType),
    Polygon(TileType, Vec<WorldPos>),
}
impl Tile {
    #[inline]
    pub fn get_tile_type(&self) -> TileType {
        match self {
            Tile::Square(tile_type) => *tile_type,
            Tile::Polygon(tile_type, _) => *tile_type,
        }
    }
}
#[revisioned(revision = 1)]
pub struct Lot {
    pub id: LotId,
    pub bounds: Vec<WorldPos>,
    pub bounds_version: u32,
    pub center: WorldPos,
    pub entrance: LotEntrance, // From this point into the lot
    pub layout: Option<LotLayout>,
    pub zoning_type: Option<ZoningType>,

    pub segment_id: Option<SegmentId>,

    pub district_id: DistrictId,
    pub building_id: Option<BuildingId>,

    pub land_value: f32, // Money € per square meter m²
}
impl Clone for Lot {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            bounds: self.bounds.clone(),
            bounds_version: self.bounds_version,
            center: self.center,
            entrance: self.entrance.clone(),
            layout: None, // Expensive and useless to clone
            zoning_type: self.zoning_type,
            segment_id: self.segment_id,
            district_id: self.district_id,
            building_id: self.building_id,
            land_value: self.land_value,
        }
    }
}
impl Lot {
    #[inline]
    pub fn chunk_coord(&self) -> ChunkCoord {
        self.center.chunk
    }

    pub fn generate_layout(&self, parking_storage: &mut ParkingStorage) -> LotLayout {
        let mut rng = ChaCha8Rng::seed_from_u64(self.id as u64);
        let frame = LotFrame::from_lot(self);
        let plan = LotPlan::generate(&frame, &mut rng);

        let mut tiles = plan.rasterize(self);
        let driveway_entrances = plan.driveway_entrances(&frame);
        let parking_spots = plan.generate_parking(self, &frame, &tiles, parking_storage);

        let mut layout = LotLayout {
            tiles,
            area: 0.0,
            floor_area: 0.0,
            driveway_entrances,
            unoccupied_parking_spots: parking_spots,
            occupied_parking_spots: Vec::new(),
        };
        layout.compute_areas();
        layout
    }

    pub fn remove_layout(&mut self, parking_storage: &mut ParkingStorage) {
        if let Some(layout) = self.layout.take() {
            for parking_spot_id in layout
                .occupied_parking_spots
                .into_iter()
                .chain(layout.unoccupied_parking_spots.into_iter())
            {
                parking_storage.despawn(parking_spot_id)
            }
        }
    }

    /// Designed for once per second.
    pub fn get_car_spawn(
        zoning: &Zoning,
        buildings: &Buildings,
        schedule: &Schedule,
        lot_id: LotId,
        rng: &mut impl Rng,
    ) -> Option<(LotEntrance, CarTripType)> {
        //println!("Trying car spawn...");
        let lot = zoning.zoning_storage.get_lot(lot_id)?;
        let building = buildings.storage.get(lot.building_id)?;
        let district = zoning.zoning_storage.get_district(lot.district_id)?;

        let tenants = building
            .current_level_params(&buildings.catalog)?
            .max_people(lot.floor_area_or_zero()); // TODO: Beware the building params Optionality!
        if tenants == 0 {
            return None;
        }

        let age = district.zoning_demand.demography.get_random_age(rng);

        let Some(zoning_type) = lot.zoning_type else {
            return None;
        };

        let phase_factor = match schedule.phase {
            SchedulePhase::Night => match zoning_type {
                ZoningType::Residential => 0.002,
                ZoningType::Commercial => 0.0005,
                ZoningType::Industrial => 0.0002,
                ZoningType::Office => 0.0004,
            },

            SchedulePhase::CommuteToWork => match zoning_type {
                ZoningType::Residential => 0.020,
                ZoningType::Commercial => 0.004,
                ZoningType::Industrial => 0.006,
                ZoningType::Office => 0.006,
            },

            SchedulePhase::Work => match zoning_type {
                ZoningType::Residential => 0.005,
                ZoningType::Commercial => 0.012,
                ZoningType::Industrial => 0.018,
                ZoningType::Office => 0.013,
            },

            SchedulePhase::Lunch => match zoning_type {
                ZoningType::Residential => 0.006,
                ZoningType::Commercial => 0.016,
                ZoningType::Industrial => 0.035,
                ZoningType::Office => 0.024,
            },

            SchedulePhase::CommuteHome => match zoning_type {
                ZoningType::Residential => 0.002,
                ZoningType::Commercial => 0.035,
                ZoningType::Industrial => 0.042,
                ZoningType::Office => 0.036,
            },

            SchedulePhase::Evening => match zoning_type {
                ZoningType::Residential => 0.010,
                ZoningType::Commercial => 0.012,
                ZoningType::Industrial => 0.003,
                ZoningType::Office => 0.003,
            },
        };

        let age_factor = match schedule.phase {
            SchedulePhase::CommuteToWork | SchedulePhase::Work => match age {
                0..=14 => 0.05,
                15..=24 => 0.70,
                25..=64 => 1.00,
                _ => 0.35,
            },

            SchedulePhase::CommuteHome => match age {
                0..=14 => 0.10,
                15..=24 => 0.75,
                25..=64 => 1.00,
                _ => 0.40,
            },

            SchedulePhase::Lunch | SchedulePhase::Evening => match age {
                0..=14 => 0.50,
                15..=24 => 0.90,
                25..=64 => 1.00,
                _ => 0.80,
            },

            SchedulePhase::Night => match age {
                0..=14 => 0.02,
                15..=24 => 0.08,
                25..=64 => 0.04,
                _ => 0.06,
            },
        };

        let size_factor = ((tenants as f32).sqrt() / 4.0).clamp(0.25, 2.0);

        let probability = phase_factor * age_factor * size_factor * schedule.traffic_multiplier;
        //println!("Car spawn probability: {}", probability);
        if !rng.random_bool(probability.clamp(0.0, 1.0) as f64) {
            return None;
        }

        let car_trip_type = CarTripType::pick_car_trip_type(Some(zoning_type), schedule.phase, rng); // TODO: I will change the zoning type naturally later.

        Some((lot.entrance.clone(), car_trip_type)) // TODO: huh? entrance
    }

    pub fn floor_area_or_zero(&self) -> f64 {
        self.layout.as_ref().map(|l| l.floor_area).unwrap_or(0.0)
    }
}
#[derive(Serialize, Deserialize, Clone, Copy, Eq, PartialEq, Hash, Debug)]
#[revisioned(revision = 1)]
pub struct TilePos {
    pub x: i16,
    pub z: i16,
}
impl TilePos {
    pub fn new(x: i16, z: i16) -> Self {
        Self { x, z }
    }
    #[inline]
    pub fn offset(self, dx: i16, dz: i16) -> Self {
        Self {
            x: self.x + dx,
            z: self.z + dz,
        }
    }
    #[inline]
    pub fn get_tiles_plus(&self) -> [TilePos; 5] {
        [
            *self,
            self.offset(0, 1),
            self.offset(1, 0),
            self.offset(0, -1),
            self.offset(-1, 0),
        ]
    }
    #[inline]
    pub fn get_neighbors_plus(&self) -> [TilePos; 4] {
        [
            self.offset(0, 1),
            self.offset(1, 0),
            self.offset(0, -1),
            self.offset(-1, 0),
        ]
    }
}
#[derive(Clone)]
#[revisioned(revision = 1)]
pub struct LotLayout {
    pub tiles: HashMap<TilePos, Tile>,
    pub area: f64,
    pub floor_area: f64,
    pub driveway_entrances: Vec<LotEntrance>,
    pub unoccupied_parking_spots: Vec<ParkingSpotId>,
    pub occupied_parking_spots: Vec<ParkingSpotId>,
}

impl LotLayout {
    #[inline]
    pub fn get_tilepos_for_pos(&self, pos: WorldPos, lot_entrance: LotEntrance) -> TilePos {
        let origin = lot_entrance.pos;
        let direction = lot_entrance.dir;
        let forward = Vec2::new(direction.x, direction.z).normalize_or_zero();
        let right = Vec2::new(forward.y, -forward.x);

        let local = origin.delta_xz(pos, right, forward);

        TilePos::new(local.x.floor() as i16, local.y.floor() as i16)
    }

    #[inline]
    pub fn get_tile_for_pos(&self, pos: WorldPos, lot_entrance: LotEntrance) -> Option<&Tile> {
        self.tiles.get(&self.get_tilepos_for_pos(pos, lot_entrance))
    }

    #[inline]
    pub fn get_pos_for_tilepos(&self, tile_pos: TilePos, lot_entrance: LotEntrance) -> WorldPos {
        let origin = lot_entrance.pos;
        let direction = lot_entrance.dir;

        let forward = Vec2::new(direction.x, direction.z).normalize_or_zero();
        let right = Vec2::new(forward.y, -forward.x);

        origin.add_vec2(right * (tile_pos.x as f32 + 0.5) + forward * (tile_pos.z as f32 + 0.5))
    }

    pub fn compute_areas(&mut self) {
        self.area = self.tiles.len() as f64 * self.tiles.len() as f64;
        self.floor_area = self
            .tiles
            .values()
            .filter_map(|tile| match tile {
                Tile::Square(TileType::House) => Some(1.0),
                Tile::Polygon(TileType::House, points) => Some(WorldPos::area(points.as_slice())),
                _ => None,
            })
            .sum()
    }
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
#[revisioned(revision = 1)]
pub struct LotEntrance {
    pub pos: WorldPos,
    pub dir: SerializableVec3,
}
impl LotEntrance {
    pub fn new(pos: WorldPos, dir: Vec3) -> Self {
        LotEntrance {
            pos,
            dir: SerializableVec3::from(dir),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ParkingSpot {
    pub id: ParkingSpotId,
    pub pos: WorldPos,
    pub dir: Vec3,
    pub occupied: Option<CarId>,
    pub lot_info: Option<ParkingSpotLotInfo>,
}
impl ParkingSpot {
    pub fn new(pos: WorldPos, dir: Vec3, lot_info: Option<ParkingSpotLotInfo>) -> Self {
        ParkingSpot {
            id: 0,
            pos,
            dir,
            occupied: None,
            lot_info,
        }
    }
}
#[derive(Serialize, Deserialize, Clone)]
pub struct ParkingSpotLotInfo {
    pub lot_id: LotId,
    pub tiles: [TilePos; 8],
}
#[derive(Default, Clone)]
#[revisioned(revision = 1)]
pub struct ZoningStorage {
    districts: Vec<Option<District>>,
    district_next_ticks: HashMap<DistrictId, SimTime>,
    lots: Vec<Option<Lot>>,
    district_free_list: Vec<DistrictId>,
    lot_free_list: Vec<LotId>,
    lot_chunk_storage: HashMap<ChunkCoord, Vec<LotId>>,
    center_chunk: ChunkCoord,
}

impl ZoningStorage {
    pub fn get_work_place_destination(
        &self,
        buildings: &BuildingStorage,
        pos: WorldPos,
        car_trip_type: CarTripType,
        rng: &mut impl Rng,
    ) -> Option<Destination> {
        const MAX_COMMUTE_DISTANCE: f64 = 1500.0;
        const MAX_DIST2: f64 = MAX_COMMUTE_DISTANCE * MAX_COMMUTE_DISTANCE;
        const EPS: f64 = 1.0;

        let chunk_coords = pos.chunk.get_chunks_in_distance(MAX_COMMUTE_DISTANCE);
        let lot_ids_to_consider = self.lots_in_chunks(chunk_coords);

        let mut candidates: Vec<(LotId, f64)> = Vec::new();

        for lot_id in lot_ids_to_consider {
            let lot = self.get_lot(lot_id)?;
            if lot.layout.is_none() {
                continue;
            };
            if lot.zoning_type.is_none_or(|z| !z.is_workplace()) {
                // Right?
                continue;
            }

            let dist2 = lot.entrance.pos.distance_squared(pos);

            if dist2 > MAX_DIST2 {
                continue;
            }

            let weight = 1.0 / (dist2 + EPS);
            candidates.push((lot_id, weight));
        }

        if candidates.is_empty() {
            return None;
        }

        // weighted random pick
        let total: f64 = candidates.iter().map(|(_, w)| w).sum();
        let mut pick = rng.random::<f64>() * total;

        let mut chosen = None;
        for (lot_id, weight) in candidates {
            pick -= weight;
            if pick <= 0.0 {
                chosen = Some(lot_id);
                break;
            }
        }

        let lot = self.get_lot(chosen?)?;
        let building_id = lot.building_id?;
        let building = buildings.get(building_id)?;
        let partition_id = buildings.get_partition_of_building(building_id)?;

        Some(Destination::Building(
            lot.district_id,
            partition_id,
            lot.segment_id,
            building_id,
        ))
    }
    pub fn get_work_place(
        &self,
        pos: WorldPos,
        building_storage: &BuildingStorage,
        person: Person,
        rng: &mut impl Rng,
    ) -> Option<BuildingId> {
        const MAX_COMMUTE_DISTANCE: f32 = 1500.0;
        const MAX_DIST2: f32 = MAX_COMMUTE_DISTANCE * MAX_COMMUTE_DISTANCE;
        const EPS: f32 = 1.0;

        // Precomputed bias tables
        const EDUCATION_BIASES: [[f32; 4]; 5] = [
            // None, Low, Medium, High
            [0.0, 0.0, 0.0, 0.0], // None
            [0.0, 0.0, 0.4, 1.0], // Office
            [0.0, 0.5, 0.9, 0.7], // Commercial
            [0.5, 1.0, 0.9, 0.3], // Industrial
            [0.0, 0.0, 0.0, 0.0], // Residential
        ];

        const AGE_BIASES: [[f32; 5]; 5] = [
            // Infant, Child, YoungAdult, Adult, Elder
            [0.0, 0.0, 0.0, 0.0, 0.0], // None
            [0.0, 0.0, 0.1, 1.0, 0.0], // Office
            [0.0, 0.0, 0.7, 1.0, 0.0], // Commercial
            [0.0, 0.0, 0.7, 1.0, 0.0], // Industrial
            [0.0, 0.0, 0.0, 0.0, 0.0], // Residential
        ];

        let lifestage = LifeStage::from_int(person.age);
        if !lifestage.is_workhorse() {
            return None;
        }

        let education_idx = match person.education_level {
            EducationLevel::None => 0,
            EducationLevel::Low => 1,
            EducationLevel::Medium => 2,
            EducationLevel::High => 3,
        };

        let age_idx = match lifestage {
            LifeStage::Infant => 0,
            LifeStage::Child => 1,
            LifeStage::YoungAdult => 2,
            LifeStage::Adult => 3,
            LifeStage::Elder => 4,
        };
        let chunk_coords = pos
            .chunk
            .get_chunks_in_distance(MAX_COMMUTE_DISTANCE as f64);
        let lot_ids = self.lots_in_chunks(chunk_coords);

        let mut total_weight: f32 = 0.0;
        let mut chosen: Option<BuildingId> = None;

        for lot_id in lot_ids {
            let Some(lot) = self.get_lot(lot_id) else {
                continue;
            };

            if lot.zoning_type.is_none_or(|z| !z.is_workplace()) {
                // Right?
                continue;
            }

            let dist2 = lot.entrance.pos.distance_squared(pos) as f32;
            if dist2 > MAX_DIST2 {
                continue;
            }

            let Some(job_building_id) = lot.building_id else {
                continue;
            };
            let Some(building) = building_storage.get(job_building_id) else {
                continue;
            };

            let fill_rate = building.occupancy.fill_rate_workplace();
            if fill_rate >= 1.0 {
                continue;
            }

            let free_rate = 1.0 - fill_rate;

            let zoning_idx = match lot.zoning_type {
                None => 0,
                Some(ZoningType::Residential) => 4,
                Some(ZoningType::Commercial) => 2,
                Some(ZoningType::Industrial) => 3,
                Some(ZoningType::Office) => 1,
            };
            let education_bias = EDUCATION_BIASES[zoning_idx][education_idx];
            let age_bias = AGE_BIASES[zoning_idx][age_idx];

            if education_bias <= 0.0 || age_bias <= 0.0 {
                continue;
            }

            let weight = (free_rate * free_rate) * education_bias * age_bias / (dist2 + EPS);

            if weight <= 0.0 {
                continue;
            }

            total_weight += weight;

            // single-pass weighted selection (no Vec, no second loop)
            if rng.random::<f32>() * total_weight < weight {
                chosen = Some(job_building_id);
            }
        }

        chosen
    }
    pub fn get_residence(
        &self,
        district_id: DistrictId,
        building_storage: &BuildingStorage,
        person: Person,
        rng: &mut impl Rng,
    ) -> Option<BuildingId> {
        let lot_ids_to_consider = self.get_district(district_id)?.lot_ids.clone();

        let base_desired: f32 = match person.education_level {
            EducationLevel::None => 0.18,
            EducationLevel::Low => 0.32,
            EducationLevel::Medium => 0.55,
            EducationLevel::High => 0.78,
        };
        let lifestage = LifeStage::from_int(person.age);
        let age_shift: f32 = match lifestage {
            LifeStage::Infant => -0.06,
            LifeStage::Child => -0.04,
            LifeStage::YoungAdult => 0.05,
            LifeStage::Adult => 0.02,
            LifeStage::Elder => -0.02,
        };

        let desired_land_value: f32 = (base_desired + age_shift).clamp(0.0, 1.0);

        let sigma = match person.education_level {
            EducationLevel::None => 0.34,
            EducationLevel::Low => 0.30,
            EducationLevel::Medium => 0.26,
            EducationLevel::High => 0.22,
        };

        let mut candidates: Vec<(BuildingId, f32)> = Vec::new();

        for lot_id in lot_ids_to_consider {
            let lot = self.get_lot(lot_id)?;
            if lot.zoning_type.is_some_and(|z| z.is_workplace()) {
                // Right?
                continue;
            }

            let building_id = lot.building_id?;
            let building = building_storage.get(building_id)?;

            let fill_rate: f32 = building.occupancy.fill_rate_residential();

            let land_value = lot.land_value;
            let gap = land_value - desired_land_value;

            let desirability_weight = (-gap * gap / (2.0 * sigma * sigma)).exp();
            let fill_weight = 0.08 + (1.0 - fill_rate).powf(2.0) * 0.92;
            let age_weight: f32 = match lifestage {
                LifeStage::Infant => 0.95,
                LifeStage::Child => 1.00,
                LifeStage::YoungAdult => 1.10,
                LifeStage::Adult => 1.00,
                LifeStage::Elder => 0.90,
            };

            let weight = desirability_weight * fill_weight * age_weight;

            if weight > 0.0 {
                candidates.push((building_id, weight));
            }
        }

        if candidates.is_empty() {
            return None;
        }

        let total: f32 = candidates.iter().map(|(_, w)| w).sum();
        if total <= 0.0 {
            return None;
        }

        let mut pick = rng.random::<f32>() * total;

        for (building_id, weight) in candidates {
            pick -= weight;
            if pick <= 0.0 {
                return Some(building_id);
            }
        }

        None
    }
    pub fn update_lot_geometry(
        &mut self,
        id: LotId,
        entrance: LotEntrance,
        bounds: Vec<WorldPos>,
        center: WorldPos,
    ) {
        let old_chunk = match self.get_lot(id) {
            Some(lot) => lot.chunk_coord(),
            None => return,
        };

        let new_chunk = center.chunk;

        if let Some(lot) = self.get_mut_lot(id) {
            lot.entrance = entrance;
            // if lot.bounds != bounds {
            //     lot.bounds_version = lot.bounds_version.saturating_add(1)
            // };
            lot.bounds = bounds;
            lot.center = center;
        } else {
            return;
        }

        if old_chunk == new_chunk {
            return;
        }

        if let Some(chunk_lots) = self.lot_chunk_storage.get_mut(&old_chunk) {
            chunk_lots.retain(|&lot_id| lot_id != id);

            if chunk_lots.is_empty() {
                self.lot_chunk_storage.remove(&old_chunk);
            }
        }

        self.lot_chunk_storage
            .entry(new_chunk)
            .or_default()
            .push(id);
    }
    pub fn sample_land_value(&self, center: ChunkCoord) -> f32 {
        let lot_ids = self.lots_in_chunk_plus(center);
        if lot_ids.is_empty() {
            return 0.0;
        }

        let sum: f32 = lot_ids
            .iter()
            .flat_map(|lot_id| self.get_lot(*lot_id))
            .map(|lot| lot.land_value)
            .sum();
        let average = sum / lot_ids.len() as f32;

        average
    }
    pub fn update_target(&mut self, target_chunk: ChunkCoord) {
        self.center_chunk = target_chunk;
    }
    pub fn district_ids(&self) -> Vec<DistrictId> {
        self.districts
            .iter()
            .flatten()
            .map(|d| d.id)
            .collect::<Vec<DistrictId>>()
    }
    pub fn get_district_ids_to_tick(&mut self, current_time: SimTime) -> Vec<DistrictId> {
        const DISTRICT_AVERAGE_TICK_TIME: f64 = 1.0; // s

        let mut out = Vec::new();

        for district_id in self.district_ids() {
            let offset = self.stable_offset_seconds(district_id, DISTRICT_AVERAGE_TICK_TIME as f32);
            let next_tick = self
                .district_next_ticks
                .entry(district_id)
                .or_insert_with(|| {
                    // Stable offset so not all districts tick on the same frame.
                    current_time + offset as f64
                });

            if current_time >= *next_tick {
                out.push(district_id);

                // Keep it roughly once per second, even if we lag behind.
                let mut due = *next_tick;
                while due <= current_time {
                    due += DISTRICT_AVERAGE_TICK_TIME;
                }
                *next_tick = due;
            }
        }
        //println!("Ticking {} Districts", out.len());
        out
    }

    fn stable_offset_seconds(&self, district_id: DistrictId, max: f32) -> f32 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        district_id.hash(&mut hasher);
        let h = hasher.finish();

        let frac = (h as f64 / u64::MAX as f64) as f32;
        frac * max
    }
    pub fn lots_in_chunk(&self, chunk_coord: ChunkCoord) -> Vec<LotId> {
        self.lot_chunk_storage
            .get(&chunk_coord)
            .cloned()
            .unwrap_or_default()
    }
    pub fn lots_in_chunk_plus(&self, chunk_coord: ChunkCoord) -> Vec<LotId> {
        let chunk_coords = chunk_coord.get_chunks_plus();

        self.lots_in_chunks(chunk_coords)
    }
    pub fn lots_in_chunks(&self, chunk_coords: Vec<ChunkCoord>) -> Vec<LotId> {
        chunk_coords
            .into_iter()
            .flat_map(|chunk_coord| {
                // CHUNK weird word. CH is weird.
                self.lot_chunk_storage
                    .get(&chunk_coord)
                    .into_iter()
                    .flatten()
                    .copied()
            })
            .collect()
    }
    pub fn get_closest_district(&self, position: WorldPos) -> Option<&District> {
        self.iter_districts().min_by(|a, b| {
            let da = position.polygon_distance_squared(&a.points);
            let db = position.polygon_distance_squared(&b.points);

            da.partial_cmp(&db).unwrap()
        })
    }

    // pub fn get_closest_lots(&self, position: WorldPos) -> Option<&Lot> {
    //     self.iter_lots().min_by(|a, b| {
    //         let da = a.center.distance_squared(position);
    //         let db = b.center.distance_squared(position);
    //
    //         da.partial_cmp(&db).unwrap()
    //     })
    // }
    pub fn iter_districts(&self) -> impl Iterator<Item = &District> {
        self.districts.iter().filter_map(|z| z.as_ref())
    }
    pub fn iter_lots(&self) -> impl Iterator<Item = &Lot> {
        self.lots.iter().filter_map(|l| l.as_ref())
    }
    pub fn iter_mut_lots(&mut self) -> impl Iterator<Item = &mut Lot> {
        self.lots.iter_mut().filter_map(|l| l.as_mut())
    }
    pub fn iter_mut_districts(&mut self) -> IterMut<'_, Option<District>> {
        self.districts.iter_mut()
    }
    /// Returns a parallel mutable iterator over building slots.
    /// Each slot is independent, so this is safe for rayon.
    pub fn par_iter_mut_districts(&mut self) -> rayon::slice::IterMut<'_, Option<District>> {
        self.districts.par_iter_mut()
    }
    pub fn new() -> Self {
        Self {
            districts: Vec::new(),
            district_next_ticks: HashMap::new(),
            lots: Vec::new(),
            district_free_list: Vec::new(),
            lot_free_list: Vec::new(),
            lot_chunk_storage: Default::default(),
            center_chunk: ChunkCoord::zero(),
        }
    }

    pub fn spawn_district(&mut self, mut district: District) -> DistrictId {
        let district_id = if let Some(reused_id) = self.district_free_list.pop() {
            // Reuse slot - III know it's None because it's in free_list
            district.id = reused_id;
            self.districts[reused_id as usize] = Some(district);
            reused_id
        } else {
            let new_id = self.districts.len() as u32;
            district.id = new_id;
            self.districts.push(Some(district));
            new_id
        };
        self.ensure_lots_reference_district(district_id);
        district_id
    }
    pub fn ensure_lots_reference_district(&mut self, district_id: DistrictId) {
        let Some(district) = self
            .districts
            .get_mut(district_id as usize)
            .and_then(|d| d.as_mut())
        else {
            return;
        };
        for &lot in district.lot_ids.iter() {
            self.lots[lot as usize]
                .as_mut()
                .map(|l| l.district_id = district_id);
        }
    }
    pub fn despawn_district(&mut self, id: DistrictId) {
        let mut lots_to_despawn = Vec::new();
        if self
            .districts
            .get(id as usize)
            .and_then(|opt| opt.as_ref())
            .is_some()
        {
            if let Some(district) = self.districts.get(id as usize) {
                if let Some(district) = district {
                    lots_to_despawn = district.lot_ids.clone();
                }
            }
            // Actually free the slot
            self.districts[id as usize] = None;
            self.district_free_list.push(id);
        }
        for lot_id in lots_to_despawn {
            self.despawn_lot(lot_id);
        }
    }

    pub fn district_count(&self) -> usize {
        self.districts.len() - self.district_free_list.len()
    }

    #[inline]
    pub fn get_district(&self, id: DistrictId) -> Option<&District> {
        self.districts.get(id as usize)?.as_ref()
    }

    #[inline]
    pub fn get_mut_district(&mut self, id: DistrictId) -> Option<&mut District> {
        self.districts.get_mut(id as usize)?.as_mut()
    }

    pub fn spawn_lot(&mut self, mut lot: Lot) -> LotId {
        lot.center = WorldPos::centroid(&lot.bounds);
        let lot_id = if let Some(reused_id) = self.lot_free_list.pop() {
            lot.id = reused_id;
            self.lot_chunk_storage
                .entry(lot.chunk_coord())
                .or_default()
                .push(lot.id);
            self.lots[reused_id as usize] = Some(lot);
            reused_id
        } else {
            let new_id = self.lots.len() as u32;
            lot.id = new_id;
            self.lot_chunk_storage
                .entry(lot.chunk_coord())
                .or_default()
                .push(lot.id);
            self.lots.push(Some(lot));
            new_id
        };

        // Clone bounds out — breaks the borrow conflict with iter_districts()
        let bounds: Vec<WorldPos> = self.lots[lot_id as usize].as_ref().unwrap().bounds.clone();

        let mut best_with_points: Option<(DistrictId, u32)> = None;
        let mut best_no_points: Option<(DistrictId, f64)> = None;

        for district in self.iter_districts() {
            let count = bounds
                .iter()
                .filter(|&&pt| point_in_polygon_xz(pt, &district.points))
                .count() as u32;

            let mut dist: f64 = f64::INFINITY;

            for p1 in &district.points {
                for p2 in &bounds {
                    dist = dist.min(p1.distance_to(*p2));
                }
            }
            if dist > 100.0 {
                // At most 100.0 meters away
                continue; // skip this district entirely
            }
            if count > 0 {
                match best_with_points {
                    Some((_, best_count)) if count <= best_count => {}
                    _ => best_with_points = Some((district.id, count)),
                }
            } else {
                match best_no_points {
                    Some((_, best_dist)) if dist >= best_dist => {}
                    _ => best_no_points = Some((district.id, dist)),
                }
            }
        }
        let best = if let Some((id, count)) = best_with_points {
            Some((id, count))
        } else {
            // None of the lot bound points are inside any district, so take the closest district at least!
            best_no_points.map(|(id, _)| (id, 0))
        };

        if let Some((district_id, _)) = best {
            if let Some(lot) = self.lots[lot_id as usize].as_mut() {
                lot.district_id = district_id;
            }
            if let Some(district) = self.get_mut_district(district_id) {
                district.lot_ids.push(lot_id);
                district.add_points(bounds);
            }
        } else {
            let mut rng = ThreadRng::default();
            let name = generate_district_name(&mut rng);
            let mut district = District::new(name, bounds, DistrictType::AutomaticallyMade);
            district.lot_ids.push(lot_id);
            let district_id = self.spawn_district(district);
            if let Some(lot) = self.lots[lot_id as usize].as_mut() {
                lot.district_id = district_id;
            }
        }

        lot_id
    }

    pub fn despawn_lot(&mut self, id: LotId) {
        let mut despawn_district = None;
        if let Some(Some(lot)) = self.lots.get(id as usize) {
            // Remove from district
            if let Some(Some(district)) = self.districts.get_mut(lot.district_id as usize) {
                district
                    .lot_ids
                    .retain(|district_lot_id| district_lot_id != &lot.id);
                district.remove_points(&lot.bounds);
                if district.lot_ids.is_empty() {
                    despawn_district = Some(district.id);
                }
            }

            // Remove from chunk storage
            if let Some(chunk_lots) = self.lot_chunk_storage.get_mut(&lot.chunk_coord()) {
                chunk_lots.retain(|lot_id| lot_id != &id);

                if chunk_lots.is_empty() {
                    self.lot_chunk_storage.remove(&lot.chunk_coord());
                }
            }

            // Free slot
            self.lots[id as usize] = None;
            self.lot_free_list.push(id);
        }
        if let Some(district_id) = despawn_district {
            self.despawn_district(district_id)
        }
    }

    pub fn lot_count(&self) -> usize {
        self.lots.len() - self.lot_free_list.len()
    }

    #[inline]
    pub fn get_lot(&self, id: LotId) -> Option<&Lot> {
        self.lots.get(id as usize)?.as_ref()
    }

    #[inline]
    pub fn get_mut_lot(&mut self, id: LotId) -> Option<&mut Lot> {
        self.lots.get_mut(id as usize)?.as_mut()
    }
}

/// Calculates the minimum distance from a point `p` to the line segment defined by `a` and `b`.
/// Returns the distance in world units (squared distance for performance).
pub fn point_to_segment_distance_sq(p: WorldPos, a: WorldPos, b: WorldPos) -> f32 {
    // 1. Convert relevant points to render coordinates (f32)
    let p_render = p.to_relative_pos(p);
    let a_render = a.to_relative_pos(a);
    let b_render = b.to_relative_pos(b);

    // 2. Vector from A to B
    let ab_x = b_render.x - a_render.x;
    let ab_z = b_render.z - a_render.z;

    // 3. Vector from A to P
    let ap_x = p_render.x - a_render.x;
    let ap_z = p_render.z - a_render.z;

    // 4. Calculate the squared length of the segment AB
    let len_sq = ab_x * ab_x + ab_z * ab_z;

    // Handle degenerate case: segment is a point
    if len_sq < 1e-10 {
        return ap_x * ap_x + ap_z * ap_z;
    }

    // 5. Project AP onto AB to find the closest point on the infinite line
    // t represents how far along the segment (0.0 to 1.0) the closest point is
    let t = (ap_x * ab_x + ap_z * ab_z) / len_sq;

    // 6. Clamp t to the segment bounds [0.0, 1.0]
    // If t < 0, closest point is A. If t > 1, closest point is B.
    let t_clamped = t.max(0.0).min(1.0);

    // 7. Calculate the coordinates of the closest point on the segment
    let closest_x = a_render.x + t_clamped * ab_x;
    let closest_z = a_render.z + t_clamped * ab_z;

    // 8. Calculate squared distance from P to the closest point
    let dx = p_render.x - closest_x;
    let dz = p_render.z - closest_z;

    dx * dx + dz * dz
}

/// Wrapper to return the actual distance (not squared).
pub fn point_to_segment_distance(p: WorldPos, a: WorldPos, b: WorldPos) -> f32 {
    point_to_segment_distance_sq(p, a, b).sqrt()
}

pub fn draw_area(
    points: &[WorldPos],
    zone_type: Option<ZoningType>,
    variables: &Variables,
    gizmo: &mut Gizmo,
    color_multiplier: Option<[f32; 4]>,
    predicted_zone_type: Option<ZoningType>,
) {
    let zone_type = predicted_zone_type.or(zone_type);

    let key = match zone_type {
        None => "none_zone_color",
        Some(ZoningType::Residential) => "residential_zone_color",
        Some(ZoningType::Commercial) => "commercial_zone_color",
        Some(ZoningType::Industrial) => "industrial_zone_color",
        Some(ZoningType::Office) => "office_zone_color",
    };
    //println!("{:?}", points.len());
    if let Some(mut c) = variables
        .get(key)
        .as_deref()
        .unwrap_or(&Value::None)
        .as_color4()
    {
        if let Some(m) = color_multiplier {
            for i in 0..4 {
                c[i] *= m[i];
            }
        }
        gizmo.area(points, c, 0.0);
    }
}

#[derive(Debug)]
enum SegmentIntersectionType {
    OtherEdges,
    ClosingEdgeOfB,
    ClosingEdgeOfA,
    PointInsidePolygon,
}

/// Parametric XZ intersection of segment a→b with segment c→d.
/// Returns (t along a→b, u along c→d), both ∈ [0, 1].
/// Uses only WorldPos::dx / dz — no raw world coordinates.
/// `tolerance`: minimum distance from endpoints in world units.
fn shrink_segment(a: WorldPos, b: WorldPos, clearance: f64) -> Option<(WorldPos, WorldPos)> {
    if clearance <= 0.0 {
        return Some((a, b));
    }

    let len = a.distance_to(b);

    if len <= clearance * 2.0 + 1e-6 {
        return None;
    }

    let t = clearance / len;

    Some((a.lerp(b, t), a.lerp(b, 1.0 - t)))
}

fn segment_xz_intersect(
    a: WorldPos,
    b: WorldPos,
    c: WorldPos,
    d: WorldPos,
    clearance: f64,
) -> Option<(f64, f64)> {
    let (a, b) = shrink_segment(a, b, clearance)?;
    let (c, d) = shrink_segment(c, d, clearance)?;

    let r_x = a.dx(b);
    let r_z = a.dz(b);
    let s_x = c.dx(d);
    let s_z = c.dz(d);

    let denom = r_x * s_z - r_z * s_x;

    let ac_x = a.dx(c);
    let ac_z = a.dz(c);

    if denom.abs() < 1e-9 {
        let r_len2 = r_x * r_x + r_z * r_z;

        if r_len2 < 1e-12 {
            return None;
        }

        let cross = ac_x * r_z - ac_z * r_x;

        if cross.abs() > 1e-6 * r_len2.sqrt() {
            return None;
        }

        let t0 = (ac_x * r_x + ac_z * r_z) / r_len2;
        let ad_x = a.dx(d);
        let ad_z = a.dz(d);
        let t1 = (ad_x * r_x + ad_z * r_z) / r_len2;

        let (lo, hi) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };

        if hi < 0.0 || lo > 1.0 {
            return None;
        }

        return Some((lo.clamp(0.0, 1.0), hi.clamp(0.0, 1.0)));
    }

    let t = (ac_x * s_z - ac_z * s_x) / denom;
    let u = (ac_x * r_z - ac_z * r_x) / denom;

    if t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 {
        Some((t, u))
    } else {
        None
    }
}

// /// Find intersection point of two segments (XZ plane).
// /// Returns the intersection point in WorldPos, or None if parallel/non-intersecting.
// ///
// /// `tolerance`: If provided, segments must intersect with at least this much
// /// clearance from their endpoints (in world units). Use this to allow segments
// /// to get close without triggering an intersection.
// pub fn segment_intersection_xz(
//     a1: WorldPos,
//     a2: WorldPos,
//     b1: WorldPos,
//     b2: WorldPos,
//     tolerance: f64
// ) -> Option<WorldPos> {
//     let d1 = a2.sub_world_pos(a1);
//     let d2 = b2.sub_world_pos(b1);
//     let d12 = b1.sub_world_pos(a1);
//
//     // ✅ Convert to full world coordinates
//     let d1_x = d1.chunk.x as f64 * chunk_size as f64 + d1.local.x as f64;
//     let d1_z = d1.chunk.z as f64 * chunk_size as f64 + d1.local.z as f64;
//     let d2_x = d2.chunk.x as f64 * chunk_size as f64 + d2.local.x as f64;
//     let d2_z = d2.chunk.z as f64 * chunk_size as f64 + d2.local.z as f64;
//     let d12_x = d12.chunk.x as f64 * chunk_size as f64 + d12.local.x as f64;
//     let d12_z = d12.chunk.z as f64 * chunk_size as f64 + d12.local.z as f64;
//
//     let cross = d1_x * d2_z - d1_z * d2_x;
//
//     if cross.abs() < 1e-10 {
//         return None;
//     }
//
//     let t = (d12_x * d2_z - d12_z * d2_x) / cross;
//     let u = (d12_x * d1_z - d12_z * d1_x) / cross;
//
//     // Convert tolerance directly to parametric epsilon
//     let eps = if tolerance == 0.0 {
//         0.0 // Actually respect zero tolerance!
//     } else {
//         let len_a = (d1_x * d1_x + d1_z * d1_z).sqrt();
//         let len_b = (d2_x * d2_x + d2_z * d2_z).sqrt();
//         // Use MINIMUM or average, not MAXIMUM!
//         let eps_a = tolerance / len_a.max(1e-6);
//         let eps_b = tolerance / len_b.max(1e-6);
//         eps_a.min(eps_b) // Most conservative
//     };
//
//     if t > eps && t < 1.0 - eps && u > eps && u < 1.0 - eps {
//         let intersection = WorldPos {
//             chunk: ChunkCoord::new(a1.chunk.x + d1.chunk.x, a1.chunk.z + d1.chunk.z),
//             local: LocalPos::new(
//                 a1.local.x + t as f32 * d1.local.x,
//                 a1.local.y + t as f32 * d1.local.y,
//                 a1.local.z + t as f32 * d1.local.z,
//             ),
//         }
//         .normalize();
//
//         Some(intersection)
//     } else {
//         None
//     }
// }

/// Point at parameter `t` ∈ [0, 1] along segment a→b.
#[inline]
fn lerp_on_segment(a: WorldPos, b: WorldPos, t: f32) -> WorldPos {
    a.add_vec3(a.delta_to(b) * t)
}

/// Ray-casting point-in-polygon test, XZ plane only.
/// Ray direction is +X from `point`; uses only dx/dz offsets.
/// WorldPos Native!!!
pub fn point_in_polygon_xz(point: WorldPos, polygon: &[WorldPos]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;

    for i in 0..n {
        // Vertex positions relative to `point` — dx/dz keep everything WorldPos-native
        let xi = point.dx(polygon[i]);
        let zi = point.dz(polygon[i]);
        let xj = point.dx(polygon[j]);
        let zj = point.dz(polygon[j]);

        // Does edge j→i cross z = 0 to the right of the origin?
        if (zi > 0.0) != (zj > 0.0) {
            let x_cross = xj + (xi - xj) * (-zj) / (zi - zj);
            if x_cross > 0.0 {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// How many times does a polyline cross a closed polygon boundary?
/// Used to score candidate split lines before committing.
fn boundary_crossing_count(polyline: &[WorldPos], boundary: &[WorldPos]) -> usize {
    let n = boundary.len();
    let mut count = 0;
    for si in 0..polyline.len().saturating_sub(1) {
        for bi in 0..n {
            if segment_xz_intersect(
                boundary[bi],
                boundary[(bi + 1) % n],
                polyline[si],
                polyline[si + 1],
                0.0,
            )
            .is_some()
            {
                count += 1;
            }
        }
    }
    count
}

fn generate_district_name(rng: &mut impl Rng) -> String {
    let prefixes = [
        "North", "South", "East", "West", "New", "Old", "Upper", "Lower",
    ];
    let cores = [
        "Oak", "River", "Stone", "Linden", "Brook", "Hill", "Maple", "Iron",
    ];
    let suffixes = ["District", "Heights", "Quarter", "Park", "Gardens", "Zone"];

    let use_prefix = rng.random_bool(0.4);
    let prefix = if use_prefix {
        Some(prefixes[rng.random_range(0..prefixes.len())])
    } else {
        None
    };

    let core = cores[rng.random_range(0..cores.len())];
    let suffix = suffixes[rng.random_range(0..suffixes.len())];

    match prefix {
        Some(p) => format!("{} {} {}", p, core, suffix),
        None => format!("{} {}", core, suffix),
    }
}

#[derive(Debug)]
enum DistrictSplitResult {
    Alright(District, (DistrictId, Vec<WorldPos>, Vec<LotId>)),
    NotEnoughPoints,

    NoBestLine,
    NotEnoughCrossings,
    DegeneratePolygon,
    DistrictDoesntExist,
}
