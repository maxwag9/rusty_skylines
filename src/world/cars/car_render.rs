//! cars_render.rs
use crate::helpers::positions::{ChunkSize, WorldPos};
use crate::resources::Time;
use crate::world::cars::car_player::conform_car_to_terrain;
use crate::world::cars::car_structs::{Car, CarId};
use crate::world::terrain::terrain_subsystem::Terrain;
use crate::world::world_core::WorldCore;
use rayon::iter::ParallelIterator;

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
}

fn interpolate_car(time: &Time, car: &Car, chunk_size: ChunkSize) -> Vec<CarChange> {
    const INTERP_BACKTIME: f64 = 0.12;
    const MAX_EXTRAP: f64 = 0.60;

    let mut changes = Vec::new();

    let Some(traj) = &car.trajectory else {
        return changes;
    };
    if traj.points.len() < 2 {
        return changes;
    }

    let mut now = time.total_game_time - INTERP_BACKTIME;
    if !now.is_finite() {
        now = time.total_game_time;
    }

    let first = &traj.points[0];
    let last = traj.points.last().unwrap();

    if now <= first.time {
        let world_pos = traj.origin.add_vec3(first.pos, chunk_size);
        let rot = first.quat;
        let vel = first.velocity;

        let delta = car.pos.delta_to(world_pos, chunk_size);
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

        let last_world = traj.origin.add_vec3(last.pos, chunk_size);
        let world_pos = last_world.add_vec3(last.velocity * dt_ex, chunk_size);
        let rot = last.quat;
        let vel = last.velocity;

        let delta = car.pos.delta_to(world_pos, chunk_size);
        if delta.length_squared() > 100.0 * 100.0 {
            changes.push(CarChange::Snap(world_pos, rot, vel));
            return changes;
        }

        changes.push(CarChange::Position(world_pos));
        changes.push(CarChange::Quat(rot));
        changes.push(CarChange::Velocity(vel));
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
    let world_pos = traj.origin.add_vec3(pos_rel, chunk_size);

    let rot = Quat::slerp(p0.quat, p1.quat, t);
    let vel = Vec3::lerp(p0.velocity, p1.velocity, t);

    let delta = car.pos.delta_to(world_pos, chunk_size);
    if delta.length_squared() > 100.0 * 100.0 {
        changes.push(CarChange::Snap(world_pos, rot, vel));
        return changes;
    }

    changes.push(CarChange::Position(world_pos));
    changes.push(CarChange::Quat(rot));
    changes.push(CarChange::Velocity(vel));

    changes
}

fn apply_car_changes(terrain: &Terrain, time: &Time, car: &mut Car, delta: Vec<CarChange>) {
    for delta in delta {
        match delta {
            CarChange::Position(pos) => car.pos = pos,
            CarChange::Quat(quat) => car.quat = quat,
            CarChange::Velocity(v) => car.current_velocity = v,
            CarChange::Snap(pos, quat, v) => {
                car.pos = pos;
                car.quat = quat;
                car.current_velocity = v;
                car.trajectory = None;
            }
        }
    }
    conform_car_to_terrain(car, terrain, time.render_dt);
}

pub fn interpolate_cars(world_core: &mut WorldCore) {
    let car_subsystem = &mut world_core.car_subsystem;

    // Get all CLOSE cars (far away cars are ghosts)
    let close_ids: Vec<CarId> = {
        let storage = car_subsystem.car_storage();
        storage.car_chunk_storage.close_car_ids().collect()
    };

    // Calculate interpolation
    let changes: Vec<(CarId, Vec<CarChange>)> = {
        let storage = car_subsystem.car_storage();
        close_ids
            .par_iter()
            .filter_map(|&id| {
                storage.get(id).map(|car| {
                    (
                        id,
                        interpolate_car(
                            &world_core.time,
                            car,
                            world_core.terrain_subsystem.chunk_size,
                        ),
                    )
                })
            })
            .collect()
    };

    // Apply
    {
        let storage_mut = car_subsystem.car_storage_mut();
        for (id, changes) in changes {
            let Some(car) = storage_mut.get_mut(id) else {
                continue;
            };
            apply_car_changes(
                &world_core.terrain_subsystem,
                &world_core.time,
                car,
                changes,
            );
        }
    }
}
