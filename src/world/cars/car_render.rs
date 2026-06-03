//! cars_render.rs

use crate::helpers::positions::WorldPos;
use crate::resources::Time;
use crate::world::cars::car_player::conform_car_to_terrain;
use crate::world::cars::car_structs::{CarId, CarStorage};
use crate::world::terrain::terrain_subsystem::Terrain;
use crate::world::world::World;
use rayon::iter::ParallelIterator;

use crate::data::Settings;
use crate::ui::variables::Variables;
use crate::world::buildings::buildings::Buildings;
use crate::world::buildings::zoning::ZoningStorage;
use crate::world::cars::car_simulation::CarTrajectory;
use crate::world::cars::signfinding::*;
use crate::world::roads::roads::{LaneRef, RoadStorage};
use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};
use rayon::iter::IntoParallelRefIterator;
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
enum CarChange {
    Position(WorldPos),
    Quat(Quat),
    Velocity(Vec3),
    Snap(WorldPos, Quat, Vec3),
    PhysicalTrajectory(Option<CarTrajectory>),
    SignfindingTrajectory(Option<CarSignfindingTrajectory>),
    LaneRef(Option<LaneRef>),
}

fn interpolate_car(
    time: &Time,
    road_storage: &RoadStorage,
    storage: &CarStorage,
    car_id: CarId,
    buildings: &Buildings,
    zoning: &ZoningStorage,
    sf_options: &SignFindingOptions,
) -> Vec<CarChange> {
    const INTERP_BACKTIME: f64 = 0.0;
    const MAX_EXTRAP: f64 = 0.60;

    let mut changes = Vec::new();
    let Some(car) = storage.get(car_id) else {
        return changes;
    };

    let owned_trajectory: Option<CarTrajectory>;

    let traj = match &car.physical_trajectory {
        Some(t) => t,
        None => {
            let (signfinding, physical) = make_new_trajectory(
                time,
                car,
                storage,
                road_storage,
                buildings,
                zoning,
                sf_options,
            );
            if let Some(s) = signfinding {
                changes.push(CarChange::SignfindingTrajectory(Some(s)));
            }

            owned_trajectory = Some(physical);
            changes.push(CarChange::PhysicalTrajectory(owned_trajectory.clone()));
            owned_trajectory.as_ref().unwrap()
        }
    };
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
    if delta > 10.0 * 10.0 {
        changes.push(CarChange::Snap(world_pos, rot, vel));
        return changes;
    }

    changes.push(CarChange::Position(world_pos));
    changes.push(CarChange::Quat(rot));
    changes.push(CarChange::Velocity(vel));
    changes.push(CarChange::LaneRef(p0.lane_ref));

    changes
}

fn apply_car_changes(
    terrain: &Terrain,
    time: &Time,
    road_storage: &RoadStorage,
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
                CarChange::SignfindingTrajectory(traj) => car.signfinding_trajectory = traj,
                CarChange::LaneRef(lane_ref) => lane_change = Some(lane_ref),
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
    let sf_options = SignFindingOptions::new(variables);
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
                            &sf_options,
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
                &world.roads.road_manager.roads,
                car_storage,
                car_id,
                changes,
            );
        }
    }
}
