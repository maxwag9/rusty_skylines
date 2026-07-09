//! cars_render.rs

use crate::helpers::positions::{LocalPos, WorldPos};
use crate::resources::Time;
use crate::world::cars::car_player::conform_car_to_terrain;
use crate::world::cars::car_structs::{CarId, CarMode, CarStorage};
use crate::world::terrain::terrain_subsystem::Terrain;
use crate::world::world::World;
use rayon::iter::ParallelIterator;

use crate::data::Settings;
use crate::ui::variables::Variables;
use crate::world::buildings::buildings::{BuildingId, BuildingStorage, Buildings};
use crate::world::buildings::zoning::{DistrictId, LotLayout, ParkingSpot, ZoningStorage};
use crate::world::cars::car_simulation::CarTrajectory;
use crate::world::cars::parking::ParkingStorage;
use crate::world::cars::partitions::PartitionId;
use crate::world::cars::signfinding::*;
use crate::world::roads::road_structs::SegmentId;
use crate::world::roads::roads::{LaneRef, RoadStorage, Segment};
use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};
use num_cpus::get;
use rand::RngExt;
use rand::rngs::ThreadRng;
use rayon::iter::IntoParallelRefIterator;
use tracing::error;
use wgpu::{VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CarInstance {
    pub model: [[f32; 4]; 4],      // transform
    pub prev_model: [[f32; 4]; 4], // transform (previous ofc)
    pub color: [f32; 3],
    pub _pad: f32,
}
impl CarInstance {
    pub fn layout<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            array_stride: size_of::<CarInstance>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: &[
                // mat4 = 4 vec4s
                VertexAttribute {
                    shader_location: 4,
                    format: VertexFormat::Float32x4,
                    offset: 0,
                },
                VertexAttribute {
                    shader_location: 5,
                    format: VertexFormat::Float32x4,
                    offset: 16,
                },
                VertexAttribute {
                    shader_location: 6,
                    format: VertexFormat::Float32x4,
                    offset: 32,
                },
                VertexAttribute {
                    shader_location: 7,
                    format: VertexFormat::Float32x4,
                    offset: 48,
                },
                // prev mat4 = 4 vec4s
                VertexAttribute {
                    shader_location: 8,
                    format: VertexFormat::Float32x4,
                    offset: 64,
                },
                VertexAttribute {
                    shader_location: 9,
                    format: VertexFormat::Float32x4,
                    offset: 80,
                },
                VertexAttribute {
                    shader_location: 10,
                    format: VertexFormat::Float32x4,
                    offset: 96,
                },
                VertexAttribute {
                    shader_location: 11,
                    format: VertexFormat::Float32x4,
                    offset: 112,
                },
                // color
                VertexAttribute {
                    shader_location: 12,
                    format: VertexFormat::Float32x3,
                    offset: 128,
                },
            ],
        }
    }
}

#[derive(Clone)]
pub enum CarChange {
    Position(WorldPos),
    Quat(Quat),
    Velocity(Vec3),
    Snap(WorldPos, Quat, Vec3),
    PhysicalTrajectory(Option<CarTrajectory>),
    SignfindingTrajectory(Option<CarSignfindingTrajectory>),
    CurrentLane(Option<LaneRef>),
    LastTurn(Option<SFTurnIdentification>),
    EngineState {
        rpm: f32,
        throttle: f32,
        brake: f32,
        gear: i8,
        accel: f32,
        decel: f32,
    },
    ReachedDestination {
        district_id: DistrictId,
        partition_id: PartitionId,
        segment_id: SegmentId,
    },
    CarMode(CarMode),
}

fn interpolate_car(
    time: &Time,
    road_storage: &RoadStorage,
    storage: &CarStorage,
    car_id: CarId,
    buildings: &Buildings,
    zoning: &ZoningStorage,
    parking_storage: &ParkingStorage,
    sf_options: &SignFindingOptions,
    target_pos: WorldPos,
) -> Vec<CarChange> {
    const INTERP_BACKTIME: f64 = 0.0;
    const MAX_EXTRAP: f64 = 0.60;

    let mut changes = Vec::new();
    let Some(car) = storage.get(car_id) else {
        return changes;
    };

    let owned_trajectory: Option<CarTrajectory>;

    let traj = match &car.physical_trajectory {
        Some(t) => Some(t),
        None => {
            let (signfinding, physical, cb) = match make_new_trajectory(
                time,
                car,
                storage,
                road_storage,
                buildings,
                zoning,
                sf_options,
            ) {
                Ok(ok) => ok,
                Err(e) => match e {
                    MakeNewTrajectoryError::Parked => {
                        let callback = RoadPathCallback {
                            is_last_turn: false,
                            car_changes: vec![
                                CarChange::CarMode(CarMode::Parked),
                                CarChange::LastTurn(None),
                            ],
                        };
                        (None, None, Some(callback))
                    }
                    MakeNewTrajectoryError::ParkingDone => {
                        let callback = RoadPathCallback {
                            is_last_turn: false,
                            car_changes: vec![
                                CarChange::CarMode(CarMode::Parked),
                                CarChange::LastTurn(None),
                            ],
                        };
                        (None, None, Some(callback))
                    }
                    _ => {
                        error!("Make new trajectory failed for car_id {}: {:?}", car_id, e);
                        (None, None, None)
                    }
                },
            };
            if let Some(mut cb) = cb {
                changes.append(&mut cb.car_changes);
            }

            if let Some(s) = signfinding {
                changes.push(CarChange::SignfindingTrajectory(Some(s)));
            }

            owned_trajectory = physical;
            changes.push(CarChange::PhysicalTrajectory(owned_trajectory.clone()));
            owned_trajectory.as_ref()
        }
    };
    let Some(traj) = traj else {
        return changes;
    }; // Don't interpolate parked cars
    if traj.points.len() < 2 {
        return changes;
    }

    let mut now = time.sim_time() - INTERP_BACKTIME;
    if !now.is_finite() {
        now = time.sim_time();
    }

    let first = &traj.points[0];
    let last = traj.points.last().unwrap();

    if now <= first.time {
        let world_pos = traj.origin.add_vec3(first.pos);
        let rot = first.quat;
        let vel = first.velocity;

        let delta = car.pos.delta_to(world_pos);
        if delta.length_squared() > 100.0 * 100.0 {
            changes.push(CarChange::Snap(world_pos, rot, vel));
            return changes;
        }

        changes.push(CarChange::Position(world_pos));
        changes.push(CarChange::Quat(rot));
        changes.push(CarChange::Velocity(vel));
        changes.push(CarChange::CurrentLane(first.lane_ref));
        return changes;
    }

    if now >= last.time {
        let dt_ex = (now - last.time).clamp(0.0, MAX_EXTRAP) as f32;

        let last_world = traj.origin.add_vec3(last.pos);
        let world_pos = last_world.add_vec3(last.velocity * dt_ex);
        let rot = last.quat;
        let vel = last.velocity;

        let delta = car.pos.delta_to(world_pos);
        if delta.length_squared() > 100.0 * 100.0 {
            changes.push(CarChange::Snap(world_pos, rot, vel));
            return changes;
        }

        changes.push(CarChange::Position(world_pos));
        changes.push(CarChange::Quat(rot));

        changes.push(CarChange::Velocity(vel));
        changes.push(CarChange::PhysicalTrajectory(None));
        changes.push(CarChange::CurrentLane(last.lane_ref));

        if traj.is_last_turn_of_sf_traj {
            changes.push(CarChange::SignfindingTrajectory(None));

            //changes.push(CarChange::LastTurn(None));
        }

        const PARKING_DISTANCE_TO_BUILDING: f64 = 50.0 * 50.0;

        if let Some(building) = buildings.storage.get(
            car.destination_addr
                .as_ref()
                .and_then(|a| a.destination.as_building_id()),
        ) {
            if let Some(lot) = zoning.get_lot(building.lot_id) {
                if lot.entrance.pos.distance_squared(world_pos) < PARKING_DISTANCE_TO_BUILDING {
                    // if let Some(car_segment_id) = last.lane_ref.and_then(|l|l.as_lane()).and_then(|(lane_id, _)| road_storage.segment_of_lane(lane_id)) {
                    //     if lot.segment_id == car_segment_id {
                    //     }
                    // }
                    let mut path = None;
                    let car_segment = last
                        .lane_ref
                        .and_then(|l| l.as_lane())
                        .and_then(|(lane_id, _)| road_storage.segment_of_lane(lane_id))
                        .and_then(|seg_id| road_storage.segment_safe(seg_id));
                    let parking_spot =
                        get_parking_spots(parking_storage, lot.layout.as_ref(), car_segment);
                    if let Some(parking_spot) = parking_spot {
                        let p = make_path_to_parking_spot(road_storage, zoning, &parking_spot, car);
                        match p {
                            Ok(p) => {
                                path = Some(p);
                            }
                            Err(e) => {
                                error!("{:?}", e)
                            }
                        };
                    };

                    if let Some(path) = path {
                        changes.push(CarChange::CarMode(CarMode::Parking { path }))
                    } else {
                        changes.push(CarChange::CarMode(CarMode::Driving));
                    }
                }

                // TODO: Not yet
                // if let Some(partition_id) = buildings.storage.get_partition_of_building(building.id) {
                //     changes.push(CarChange::ReachedDestination {district_id: lot.district_id, partition_id, segment_id: building.segment_id})
                // }
            }
        }

        return changes;
    }

    let idx = traj
        .points
        .binary_search_by(|p| {
            p.time
                .partial_cmp(&now)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or_else(|i| i);

    let i1 = idx.clamp(1, traj.points.len() - 1);
    let i0 = i1 - 1;

    let p0 = &traj.points[i0];
    let p1 = &traj.points[i1];

    let dt = (p1.time - p0.time) as f32;
    let t = if dt > 1e-9 {
        ((now - p0.time) as f32 / dt).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let pos_rel = Vec3::lerp(p0.pos, p1.pos, t);
    let world_pos = traj.origin.add_vec3(pos_rel);

    let rot = Quat::slerp(p0.quat, p1.quat, t);
    let vel = Vec3::lerp(p0.velocity, p1.velocity, t);

    let delta = car.pos.distance_squared(world_pos);

    if delta > 100.0 * 100.0 {
        changes.push(CarChange::Snap(world_pos, rot, vel));
        return changes;
    }

    changes.push(CarChange::Position(world_pos));
    changes.push(CarChange::Quat(rot));
    changes.push(CarChange::Velocity(vel));
    changes.push(CarChange::CurrentLane(p0.lane_ref));

    const MAX_DISTANCE: f64 = 300.0 * 300.0;

    let d2 = car.pos.distance_squared(target_pos);

    if d2 > MAX_DISTANCE {
        return changes;
    }

    let speed = vel.length();

    let acc: f32 = if i0 >= 2 && i0 + 1 < traj.points.len() {
        // Central difference over multiple points for stability
        let prev_p = &traj.points[i0 - 2];
        let next_p = &traj.points[i0 + 1];
        let dt_wide = (next_p.time - prev_p.time) as f32;
        if dt_wide > 1e-5 {
            (next_p.velocity.length() - prev_p.velocity.length()) / dt_wide
        } else {
            // fallback
            let dt = (p1.time - p0.time) as f32;
            if dt > 1e-5 {
                (speed - traj.points[i0 - 1].velocity.length()) / dt
            } else {
                0.0
            }
        }
    } else if i0 > 0 {
        let dt = (p1.time - p0.time) as f32;
        if dt > 1e-5 {
            (speed - traj.points[i0 - 1].velocity.length()) / dt
        } else {
            0.0
        }
    } else {
        0.0
    };

    let acc_clamped = acc.clamp(-9.0, 7.0); // m/s²

    let throttle = if acc_clamped > 0.0 {
        (acc_clamped / 6.5).clamp(0.0, 1.0)
    } else {
        0.08
    };

    let brake = if acc_clamped < -0.5 {
        ((-acc_clamped) / 8.0).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let wheel_rpm = (speed * 60.0) / (2.0 * std::f32::consts::PI * car.wheel_radius.max(0.1));

    let gear: i8 = match speed {
        s if s < 4.5 => 1,
        s if s < 10.0 => 2,
        s if s < 18.0 => 3,
        s if s < 29.0 => 4,
        s if s < 45.0 => 5,
        _ => 6,
    };

    let ratios = [0.0f32, 4.1, 2.6, 1.75, 1.3, 1.0, 0.78];
    let gear_ratio = ratios[gear as usize];

    let base_rpm = (wheel_rpm * gear_ratio).clamp(750.0, 7200.0);
    let rpm = (base_rpm + throttle * 950.0).clamp(750.0, 7200.0);

    let accel = acc_clamped.max(0.0);
    let decel = (-acc_clamped).max(0.0);

    // Push improved engine state
    changes.push(CarChange::EngineState {
        rpm,
        throttle,
        brake,
        gear,
        accel, // m/s²
        decel, // m/s²
    });
    changes
}

fn get_parking_spots(
    parking_storage: &ParkingStorage,
    lot_layout: Option<&LotLayout>,
    segment: Option<&Segment>,
) -> Option<ParkingSpot> {
    // 2 Parking spot possibilities:
    // Inside a lot
    // Beside the curb
    //
    // I guess a building needs to define parking spaces?
    // Yes all Parking Spots will be tracked in ParkingStorage

    let mut spot: Option<ParkingSpot>;
    let rng = &mut ThreadRng::default();
    let parking_spot_id = lot_layout.map(|layout| {
        layout.unoccupied_parking_spots[rng.random_range(0..layout.unoccupied_parking_spots.len())]
    });
    spot = parking_storage.get(parking_spot_id).cloned();
    if spot.is_some() {
        return spot;
    };
    let parking_spot_id = segment.map(|s| s.parking_spots.iter());

    spot
    // match car.current_lane.and_then(|l|l.as_lane()) {
    //     Some((lane_id, poly_idx)) => {
    //         let Some(segment) = road_storage.lane_safe(lane_id).and_then(|l|road_storage.segment_safe(l.segment())) else { return Err(ParkingPathError::DespawnTheCar) };
    //
    //
    //         Ok(Path)
    //     }
    //     None => {return Err(ParkingPathError::DespawnTheCar)}
    // }
}

fn apply_car_changes(
    terrain: &Terrain,
    time: &Time,
    road_storage: &mut RoadStorage,
    car_storage: &mut CarStorage,
    car_id: CarId,
    delta: Vec<CarChange>,
) {
    let mut lane_change = None;
    {
        let Some(car) = car_storage.get_mut(car_id) else {
            return;
        };
        for delta in delta {
            match delta {
                CarChange::Position(pos) => car.pos = pos,
                CarChange::Quat(quat) => car.quat = quat,
                CarChange::Velocity(v) => car.current_velocity = v,
                CarChange::Snap(pos, quat, v) => {
                    car.pos = pos;
                    car.quat = quat;
                    car.current_velocity = v;
                    car.physical_trajectory = None;
                }
                CarChange::PhysicalTrajectory(traj) => car.physical_trajectory = traj,
                CarChange::SignfindingTrajectory(traj) => {
                    car.signfinding_trajectory = traj;
                    if car.trip.is_none() {
                        car.trip = Some(SignFindingTrip {
                            start_time: time.sim_time(),
                            sections: vec![],
                        });
                    };
                }
                CarChange::CurrentLane(lane_ref) => lane_change = Some(lane_ref),
                CarChange::LastTurn(last_turn) => {
                    car.last_turn = last_turn;
                    if let Some((node_id, segment_id)) =
                        car.last_turn.as_ref().and_then(|t| match t.turn_type {
                            SFTurnType::SegmentLanes { .. } => None,
                            SFTurnType::IntersectionLanes {
                                node_id,
                                to_segment_id,
                                ..
                            } => Some((node_id, to_segment_id)),
                        })
                    {
                        if let Some(trip) = car.trip.as_mut() {
                            //let last_time = trip.sections.last().map(|l|l.end_time).unwrap_or(trip.start_time);
                            let same_as_last = trip
                                .sections
                                .last()
                                .map(|l| l.segment_id == segment_id && l.node_id == node_id)
                                .unwrap_or(false);
                            if !same_as_last {
                                trip.sections.push(SFSection {
                                    node_id,
                                    segment_id,
                                    end_time: time.sim_time(),
                                })
                            }
                        }
                    }
                }
                CarChange::EngineState {
                    rpm,
                    throttle,
                    brake,
                    gear,
                    accel,
                    decel,
                } => {
                    car.engine_rpm = rpm;

                    car.throttle = throttle;
                    car.brake = brake;
                    car.gear = gear;
                    car.accel = accel;
                    car.decel = decel;
                }
                CarChange::ReachedDestination {
                    district_id,
                    partition_id,
                    segment_id,
                } => {
                    if let Some(trip) = car.trip.as_mut() {
                        let last_time = trip
                            .sections
                            .last()
                            .map(|l| l.end_time)
                            .unwrap_or(trip.start_time);
                        for section in trip.sections.iter() {
                            let Some(arm) = road_storage
                                .node_mut_safe(section.node_id)
                                .and_then(|node| node.arm_for_segment_mut(section.segment_id))
                            else {
                                continue;
                            };
                            let duration = last_time - section.end_time;
                            arm.update_travel_time(
                                district_id,
                                partition_id,
                                segment_id,
                                duration as f32,
                            )
                        }
                    }
                    car.trip = None;
                }
                CarChange::CarMode(car_mode) => {
                    car.mode = car_mode;
                }
            }
        }
        conform_car_to_terrain(car, terrain, time.render_dt);
    }
    if let Some(lane_ref) = lane_change {
        car_storage.set_car_lane(car_id, lane_ref, road_storage);
    }
}

pub fn interpolate_cars(world: &mut World, variables: &Variables, settings: &Settings) {
    let car_subsystem = &mut world.cars;

    // Get all CLOSE cars (medium and far away cars are ghosts btw)
    let close_ids: Vec<CarId> = {
        let storage = car_subsystem.car_storage();
        storage.car_chunk_storage.close_car_ids().collect()
    };
    let sf_options = SignFindingOptions::new(variables, settings);
    // Calculate interpolation
    let changes: Vec<(CarId, Vec<CarChange>)> = {
        let storage = car_subsystem.car_storage();
        close_ids
            .par_iter()
            .map(|&car_id| {
                (
                    car_id,
                    if car_id == 0 && settings.drive_car {
                        Vec::new()
                    } else {
                        interpolate_car(
                            &world.time,
                            &world.roads.road_manager.roads,
                            storage,
                            car_id,
                            &world.buildings,
                            &world.zoning.zoning_storage,
                            &world.roads.parking,
                            &sf_options,
                            world.world_state.camera.eye_world(),
                        )
                    },
                )
            })
            .collect()
    };

    // Apply
    {
        let car_storage = car_subsystem.car_storage_mut();
        for (car_id, changes) in changes {
            apply_car_changes(
                &world.terrain,
                &world.time,
                &mut world.roads.road_manager.roads,
                car_storage,
                car_id,
                changes,
            );
        }
    }
}
