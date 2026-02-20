//! cars_render.rs
use crate::helpers::positions::{ChunkSize, WorldPos};
use crate::resources::Time;
use crate::world::cars::car_player::conform_car_to_terrain;
use crate::world::cars::car_structs::{Car, CarId};
use crate::world::terrain::terrain_subsystem::TerrainSubsystem;
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
}

fn interpolate_car(time: &Time, car: &Car, chunk_size: ChunkSize) -> Vec<CarChange> {
    let mut car_changes = Vec::new();
    let next_splines = &car.next_splines;

    if next_splines.is_empty() {
        return car_changes;
    }

    let current_time = time.total_game_time;

    let active_segment = next_splines.iter().find(|seg| {
        let duration = 1.0 / seg.inv_duration as f64;
        current_time >= seg.t0 && current_time < seg.t0 + duration
    });

    let Some(segment) = active_segment else {
        eprintln!(
            "No segment! time={}, segments: {:?}",
            current_time,
            next_splines
                .iter()
                .map(|s| (s.t0, s.t0 + 1.0 / s.inv_duration as f64))
                .collect::<Vec<_>>()
        );
        return car_changes;
    };

    let local_t = ((current_time - segment.t0) as f32 * segment.inv_duration).clamp(0.0, 1.0);

    let t = local_t;
    let t2 = t * t;
    let t3 = t2 * t;

    // Position basis functions
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;

    // Derivative basis functions (d/dt of above)
    let dh00 = 6.0 * t2 - 6.0 * t;
    let dh10 = 3.0 * t2 - 4.0 * t + 1.0;
    let dh01 = -6.0 * t2 + 6.0 * t;
    let dh11 = 3.0 * t2 - 2.0 * t;

    let dt = 1.0 / segment.inv_duration;

    // Position
    let interpolated =
        segment.p0 * h00 + (segment.v0 * dt) * h10 + segment.p1 * h01 + (segment.v1 * dt) * h11;

    // Velocity/tangent (multiply by inv_duration to get world-space velocity)
    let tangent = (segment.p0 * dh00
        + (segment.v0 * dt) * dh10
        + segment.p1 * dh01
        + (segment.v1 * dt) * dh11)
        * segment.inv_duration;

    let next_pos = segment.origin.add_vec3(interpolated, chunk_size);
    car_changes.push(CarChange::Position(next_pos));

    // Compute rotation from tangent direction
    let direction = Vec3::new(tangent.x, 0.0, tangent.z); // Project to XZ plane for ground vehicles
    if direction.length_squared() > 1e-6 {
        let direction = direction.normalize();
        // Assuming car's forward is +Z, rotate from +Z to direction
        let forward = Vec3::Z;
        let rotation = Quat::from_rotation_arc(forward, direction);
        car_changes.push(CarChange::Quat(rotation));
    }

    car_changes
}

fn apply_car_changes(
    terrain: &TerrainSubsystem,
    time: &Time,
    car: &mut Car,
    delta: Vec<CarChange>,
) {
    for delta in delta {
        match delta {
            CarChange::Position(pos) => {
                car.pos = pos;
            }
            CarChange::Quat(quat) => {
                car.quat = quat;
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
