use crate::helpers::positions::{ChunkCoord, ChunkSize, LocalPos, WorldPos};
use crate::world::cars::car_player::sanitize_quat;
use crate::world::cars::car_structs::{CarChunkStorage, CarId, CarStorage, SimTime};
use crate::world::terrain::terrain_subsystem::Terrain;
use glam::{Quat, Vec3};
use rand::{Rng, RngExt, seq::SliceRandom};
use rayon::prelude::*;
use std::collections::HashMap;
// ============================================================================
// Constants
// ============================================================================

pub const CHUNK_TICK_INTERVAL: SimTime = 2.0; // 2 second ticks
pub const TICK_JITTER_MAX: SimTime = 0.5; // Spread updates over 0.5s window
pub const MAX_CHUNKS_PER_UPDATE: usize = 128; // Prevent frame spikes
pub const JOB_BATCH_SIZE: usize = 16; // Chunks per parallel batch

// ============================================================================
// Job Structs
// ============================================================================

#[derive(Clone, Debug)]
pub struct CarChunkJob {
    pub chunk_coord: ChunkCoord,
    pub car_ids: Vec<CarId>,
    pub delta_time: SimTime,
}
#[derive(Debug, Clone)]
pub struct CarTrajectoryPoint {
    pub time: f64,
    pub pos: Vec3, // relative to origin
    pub quat: Quat,
    pub velocity: Vec3, // world space, for conforming to terrain etc
}

#[derive(Debug, Clone)]
pub struct CarTrajectory {
    pub car_id: u32,
    pub origin: WorldPos,
    pub points: Vec<CarTrajectoryPoint>,
    pub end_quat: Quat,
    pub end_yaw_rate: f32,
    pub end_steering_angle: f32,
    pub end_steering_vel: f32,
}
#[derive(Debug)]
pub struct CarChunkJobResult {
    pub chunk_coord: ChunkCoord,
    pub completed_time: SimTime,
    /// Cars that need to move to different chunks: (car_id, new_chunk)
    pub chunk_transfers: Vec<(CarId, ChunkCoord)>,
    /// Cars to despawn
    pub despawns: Vec<CarId>,
    /// Next positions as splines for cars for the great interpolator! "He's a nice guy, you know, I've known him for many years, he's a great guy, brilliant I must say.
    /// Joe Biden could never have been like him, not even before he got dementia, that's the truth!"
    pub trajectories: Vec<CarTrajectory>,
}

impl CarChunkJobResult {
    pub fn empty(chunk_coord: ChunkCoord, completed_time: SimTime) -> Self {
        Self {
            chunk_coord,
            completed_time,
            chunk_transfers: Vec::new(),
            despawns: Vec::new(),
            trajectories: Vec::new(),
        }
    }
}

// Job System

pub struct CarSimSystem {
    /// Per-chunk jitter to prevent thundering herd
    chunk_jitter: HashMap<ChunkCoord, SimTime>,
    batch_size: usize,
    max_per_update: usize,
}

impl Default for CarSimSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl CarSimSystem {
    pub fn new() -> Self {
        Self {
            chunk_jitter: HashMap::new(),
            batch_size: JOB_BATCH_SIZE,
            max_per_update: MAX_CHUNKS_PER_UPDATE,
        }
    }

    pub fn with_limits(mut self, batch_size: usize, max_per_update: usize) -> Self {
        self.batch_size = batch_size;
        self.max_per_update = max_per_update;
        self
    }

    /// Main entry point: collect due chunks, dispatch in parallel batches, apply results
    /// Returns the list of chunk coords that were updated this frame
    pub fn update_chunks(
        &mut self,
        car_storage: &mut CarStorage,
        terrain: &Terrain,
        current_time: SimTime,
        rng: &mut impl Rng,
        chunk_size: ChunkSize,
    ) -> Vec<ChunkCoord> {
        let jobs = self.collect_due_chunks(&car_storage.car_chunk_storage, current_time, rng);

        if jobs.is_empty() {
            return Vec::new();
        }

        // Collect which chunks are being updated
        let updated_chunks: Vec<ChunkCoord> = jobs.iter().map(|j| j.chunk_coord).collect();

        let results = self.dispatch_batched(&jobs, car_storage, terrain, current_time, chunk_size);
        self.apply_results(results, car_storage, rng, current_time);

        updated_chunks
    }

    /// Simulate specific chunks (for external batch calls)
    pub fn simulate_chunks(
        &mut self,
        chunk_coords: Vec<ChunkCoord>,
        car_storage: &mut CarStorage,
        terrain: &Terrain,
        current_time: SimTime,
        rng: &mut impl Rng,
        chunk_size: ChunkSize,
    ) {
        let jobs: Vec<CarChunkJob> = chunk_coords
            .into_iter()
            .filter_map(|coord| {
                let chunk = car_storage.car_chunk_storage.close().get(&coord)?;
                if chunk.car_ids.is_empty() {
                    return None;
                }
                Some(CarChunkJob {
                    chunk_coord: coord,
                    car_ids: chunk.car_ids.clone(),
                    delta_time: current_time - chunk.last_update_time,
                })
            })
            .collect();

        if jobs.is_empty() {
            return;
        }

        let results = self.dispatch_batched(&jobs, car_storage, terrain, current_time, chunk_size);
        self.apply_results(results, car_storage, rng, current_time);
    }

    fn collect_due_chunks(
        &mut self,
        chunk_storage: &CarChunkStorage,
        current_time: SimTime,
        rng: &mut impl Rng,
    ) -> Vec<CarChunkJob> {
        let mut due_jobs = Vec::new();

        for (coord, chunk) in chunk_storage.close().iter() {
            if chunk.car_ids.is_empty() {
                continue;
            }

            // Assign or retrieve per-chunk jitter
            let jitter = *self
                .chunk_jitter
                .entry(*coord)
                .or_insert_with(|| rng.random_range(0.0..TICK_JITTER_MAX));

            let required_interval = CHUNK_TICK_INTERVAL + jitter;
            let elapsed = current_time - chunk.last_update_time;

            if elapsed >= required_interval {
                due_jobs.push(CarChunkJob {
                    chunk_coord: *coord,
                    car_ids: chunk.car_ids.clone(),
                    delta_time: elapsed,
                });
            }
        }

        // Shuffle for spatial distribution (random order = naturally spread out)
        due_jobs.shuffle(rng);

        // Cap to prevent massive frame spikes
        if due_jobs.len() > self.max_per_update {
            due_jobs.truncate(self.max_per_update);
        }

        due_jobs
    }

    fn dispatch_batched(
        &self,
        jobs: &[CarChunkJob],
        car_storage: &CarStorage,
        terrain: &Terrain,
        current_time: SimTime,
        chunk_size: ChunkSize,
    ) -> Vec<CarChunkJobResult> {
        // Parallel processing with rayon
        jobs.par_chunks(self.batch_size)
            .flat_map_iter(|batch| {
                batch
                    .iter()
                    .map(|job| self.process_chunk_job(job, car_storage, current_time, chunk_size))
            })
            .collect()
    }

    fn process_chunk_job(
        &self,
        job: &CarChunkJob,
        car_storage: &CarStorage,
        current_time: SimTime,
        chunk_size: ChunkSize,
    ) -> CarChunkJobResult {
        let mut result = CarChunkJobResult::empty(job.chunk_coord, current_time);

        const TOTAL_DURATION: f32 = 4.0;
        const PHYSICS_STEP: f32 = 1.0 / 120.0;
        const SAMPLE_EVERY: u32 = 2;

        const G: f32 = 9.81;
        const MASS: f32 = 1500.0;
        const IZ: f32 = 2500.0;
        const MU: f32 = 1.05;
        const C_AF: f32 = 90_000.0;
        const C_AR: f32 = 100_000.0;
        const ROLL_DRAG: f32 = 0.02;
        const AERO_DRAG: f32 = 0.0007;
        const STEER_RATE_MAX: f32 = 6.0;
        const STEER_KP: f32 = 70.0;
        const STEER_KD: f32 = 14.0;
        const STEER_LOCK_LOW: f32 = 0.62;
        const STEER_LOCK_HIGH: f32 = 0.14;
        const STEER_LOCK_FADE_MPS: f32 = 28.0;
        const MIN_U: f32 = 0.5;
        const MAX_SPEED: f32 = 50.0;
        const MAX_YAW_RATE: f32 = 3.5;

        let cs = chunk_size as f64;
        let total_steps = (TOTAL_DURATION / PHYSICS_STEP) as u32;
        let num_samples = total_steps / SAMPLE_EVERY + 2;

        for &car_id in &job.car_ids {
            let Some(car) = car_storage.get(car_id) else {
                continue;
            };

            let (
                mut sim_pos,
                mut sim_quat,
                mut sim_vel,
                mut sim_yaw_rate,
                mut sim_steering_angle,
                mut sim_steering_vel,
                start_time,
            ) = if let Some(traj) = &car.trajectory {
                if let Some(last) = traj.points.last() {
                    let end_pos = traj.origin.add_vec3(last.pos, chunk_size);
                    (
                        end_pos,
                        traj.end_quat,
                        last.velocity,
                        traj.end_yaw_rate,
                        traj.end_steering_angle,
                        traj.end_steering_vel,
                        last.time,
                    )
                } else {
                    (
                        car.pos,
                        car.quat,
                        car.current_velocity,
                        car.yaw_rate,
                        car.steering_angle,
                        car.steering_vel,
                        current_time,
                    )
                }
            } else {
                (
                    car.pos,
                    car.quat,
                    car.current_velocity,
                    car.yaw_rate,
                    car.steering_angle,
                    car.steering_vel,
                    current_time,
                )
            };

            sim_quat = sanitize_quat(sim_quat);

            let ref_origin = car
                .trajectory
                .as_ref()
                .map(|t| t.origin)
                .unwrap_or_else(|| WorldPos::new(sim_pos.chunk, LocalPos::zero()));

            let wheelbase = car.length.max(0.1);
            let a_wb = 0.5 * wheelbase;
            let b_wb = wheelbase - a_wb;
            let fz_f = MASS * G * (b_wb / wheelbase);
            let fz_r = MASS * G * (a_wb / wheelbase);

            let inv_q = sim_quat.conjugate();
            let local_v = inv_q * sim_vel;
            let mut u = local_v.z;
            let mut v_lat = -local_v.x;
            let mut r = sim_yaw_rate;

            let mut points = Vec::with_capacity(num_samples as usize);

            points.push(CarTrajectoryPoint {
                time: start_time,
                pos: ref_origin.delta_to(sim_pos, chunk_size),
                quat: sim_quat,
                velocity: sim_vel,
            });

            let mut sim_time = start_time;
            let mut step_counter: u32 = 0;

            for _ in 0..total_steps {
                let h = PHYSICS_STEP;
                sim_time += h as f64;
                step_counter += 1;

                let global_x = sim_pos.chunk.x as f64 * cs + sim_pos.local.x as f64;
                let global_z = sim_pos.chunk.z as f64 * cs + sim_pos.local.z as f64;
                let wander_phase = global_x * 0.012 + global_z * 0.008 + (car_id as f64 * 7.3);
                let wander = wander_phase.sin() as f32;

                let steer_input = wander * 0.55;
                let throttle_input = 0.85 + wander * 0.15;
                let brake_input = if wander.abs() > 0.8 { 0.3 } else { 0.0 };

                let speed = u.abs();
                let t_lock = (speed / STEER_LOCK_FADE_MPS).clamp(0.0, 1.0);
                let steer_lock =
                    STEER_LOCK_HIGH + (STEER_LOCK_LOW - STEER_LOCK_HIGH) * (1.0 - t_lock).powf(1.7);
                let delta_cmd = (steer_input * steer_lock).clamp(-steer_lock, steer_lock);
                let ax_cmd = throttle_input * car.accel - brake_input * car.accel * 3.0;

                let delta = sim_steering_angle;
                let mut delta_dot = sim_steering_vel;
                delta_dot += (STEER_KP * (delta_cmd - delta) - STEER_KD * delta_dot) * h;
                delta_dot = delta_dot.clamp(-STEER_RATE_MAX, STEER_RATE_MAX);
                let mut delta_new = delta + delta_dot * h;
                delta_new = delta_new.clamp(-steer_lock, steer_lock);
                sim_steering_angle = delta_new;
                sim_steering_vel = delta_dot;

                let sign_u = if u >= 0.0 { 1.0 } else { -1.0 };
                let u_safe = if u.abs() < MIN_U { MIN_U * sign_u } else { u };

                let alpha_f = (v_lat + a_wb * r).atan2(u_safe) - sim_steering_angle;
                let alpha_r = (v_lat - b_wb * r).atan2(u_safe);

                let fy_f = (-C_AF * alpha_f).clamp(-MU * fz_f, MU * fz_f);
                let fy_r = (-C_AR * alpha_r).clamp(-MU * fz_r, MU * fz_r);

                let ax_drag = -ROLL_DRAG * u - AERO_DRAG * u * u.abs();

                let u_dot = ax_cmd + ax_drag + r * v_lat;
                let v_dot = (fy_f + fy_r) / MASS - r * u;
                let r_dot = (a_wb * fy_f - b_wb * fy_r) / IZ;

                u += u_dot * h;
                v_lat += v_dot * h;
                r += r_dot * h;

                u = u.clamp(-MAX_SPEED, MAX_SPEED);
                v_lat = v_lat.clamp(-MAX_SPEED * 0.4, MAX_SPEED * 0.4);
                r = r.clamp(-MAX_YAW_RATE, MAX_YAW_RATE);

                let yaw_delta = -r * h;
                if yaw_delta.abs() > 1e-9 {
                    let yaw_q = Quat::from_axis_angle(Vec3::Y, yaw_delta);
                    sim_quat = sanitize_quat(yaw_q * sim_quat);
                }

                let local_velocity = Vec3::new(-v_lat, 0.0, u);
                let world_vel = sim_quat * local_velocity;
                sim_vel = world_vel;
                sim_pos = sim_pos.add_vec3(world_vel * h, chunk_size);

                if step_counter % SAMPLE_EVERY == 0 {
                    points.push(CarTrajectoryPoint {
                        time: sim_time,
                        pos: ref_origin.delta_to(sim_pos, chunk_size),
                        quat: sim_quat,
                        velocity: world_vel,
                    });
                }
            }

            sim_yaw_rate = r;

            if points.last().map(|p| p.time).unwrap_or(0.0) < sim_time {
                points.push(CarTrajectoryPoint {
                    time: sim_time,
                    pos: ref_origin.delta_to(sim_pos, chunk_size),
                    quat: sim_quat,
                    velocity: sim_vel,
                });
            }

            result.trajectories.push(CarTrajectory {
                car_id,
                origin: ref_origin,
                points,
                end_quat: sim_quat,
                end_yaw_rate: sim_yaw_rate,
                end_steering_angle: sim_steering_angle,
                end_steering_vel: sim_steering_vel,
            });

            if sim_pos.chunk != job.chunk_coord {
                result.chunk_transfers.push((car_id, sim_pos.chunk));
            }
        }

        result
    }

    fn apply_results(
        &mut self,
        results: Vec<CarChunkJobResult>,
        car_storage: &mut CarStorage,
        rng: &mut impl Rng,
        current_time: SimTime,
    ) {
        const HISTORY_KEEP: f64 = 1.25;
        const EPS: f64 = 1e-6;

        fn trim_history(points: &mut Vec<CarTrajectoryPoint>, cutoff: f64) {
            if points.len() < 3 {
                return;
            }
            let k = points.partition_point(|p| p.time < cutoff);
            if k > 1 {
                points.drain(0..(k - 1));
            }
        }

        for result in results {
            if let Some(chunk) = car_storage
                .car_chunk_storage
                .close_mut()
                .get_mut(&result.chunk_coord)
            {
                chunk.last_update_time = result.completed_time;
            }

            for (car_id, new_chunk) in result.chunk_transfers {
                car_storage.move_car_between_chunks(result.chunk_coord, new_chunk, car_id);
            }

            for car_id in result.despawns {
                car_storage.despawn(car_id);
            }

            for mut trajectory in result.trajectories {
                let Some(car) = car_storage.get_mut(trajectory.car_id) else {
                    continue;
                };
                if trajectory.points.len() < 2 {
                    continue;
                }

                if let Some(existing) = &mut car.trajectory {
                    if existing.origin != trajectory.origin {
                        car.trajectory = Some(trajectory);
                    } else {
                        let start_new = trajectory.points[0].time;
                        existing.points.retain(|p| p.time < start_new - EPS);
                        existing.points.append(&mut trajectory.points);
                        existing.end_quat = trajectory.end_quat;
                        existing.end_yaw_rate = trajectory.end_yaw_rate;
                        existing.end_steering_angle = trajectory.end_steering_angle;
                        existing.end_steering_vel = trajectory.end_steering_vel;
                        trim_history(&mut existing.points, current_time - HISTORY_KEEP);
                    }
                } else {
                    trim_history(&mut trajectory.points, current_time - HISTORY_KEEP);
                    car.trajectory = Some(trajectory);
                }
            }

            self.chunk_jitter
                .insert(result.chunk_coord, rng.random_range(0.0..TICK_JITTER_MAX));
        }
    }

    /// Cleanup jitter map for removed chunks (call periodically)
    pub fn cleanup_stale_jitter(&mut self, chunk_storage: &CarChunkStorage) {
        self.chunk_jitter
            .retain(|coord, _| chunk_storage.contains(coord));
    }
}
