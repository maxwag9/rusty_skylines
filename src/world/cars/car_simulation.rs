use crate::helpers::positions::{ChunkCoord, ChunkSize, LocalPos, WorldPos};
use crate::world::cars::car_structs::{CarChunkStorage, CarId, CarStorage, SimTime};
use glam::Vec3;
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
#[derive(Debug)]
pub struct CarSplineSegment {
    pub t0: SimTime,
    pub inv_duration: f32, // 1.0 / (t1 - t0)
    pub p0: Vec3,
    pub v0: Vec3,
    pub p1: Vec3,
    pub v1: Vec3,
    pub origin: WorldPos,
}
#[derive(Debug)]
pub struct CarSpline {
    pub car_id: CarId,
    pub next_splines: Vec<CarSplineSegment>,
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
    pub next_positions: Vec<CarSpline>,
}

impl CarChunkJobResult {
    pub fn empty(chunk_coord: ChunkCoord, completed_time: SimTime) -> Self {
        Self {
            chunk_coord,
            completed_time,
            chunk_transfers: Vec::new(),
            despawns: Vec::new(),
            next_positions: Vec::new(),
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

        let results = self.dispatch_batched(&jobs, car_storage, current_time, chunk_size);
        self.apply_results(results, car_storage, rng);

        updated_chunks
    }

    /// Simulate specific chunks (for external batch calls)
    pub fn simulate_chunks(
        &mut self,
        chunk_coords: Vec<ChunkCoord>,
        car_storage: &mut CarStorage,
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

        let results = self.dispatch_batched(&jobs, car_storage, current_time, chunk_size);
        self.apply_results(results, car_storage, rng);
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

        const NUM_SEGMENTS: usize = 5;
        const TOTAL_DURATION: f32 = 2.0; // seconds
        const SEGMENT_DURATION: f32 = TOTAL_DURATION / NUM_SEGMENTS as f32; // 0.4s each

        for &car_id in &job.car_ids {
            let Some(car) = car_storage.get(car_id) else {
                continue;
            };

            // Use car's starting chunk as reference for all spline positions
            let ref_origin = WorldPos::new(car.pos.chunk, LocalPos::zero());

            // Deterministic random generator seeded by car ID + time
            let mut rng_state =
                (car_id as u64).wrapping_mul(0x517cc1b727220a95) ^ (current_time.to_bits());

            let mut rand = || -> f32 {
                rng_state ^= rng_state >> 12;
                rng_state ^= rng_state << 25;
                rng_state ^= rng_state >> 27;
                (rng_state.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / 16777216.0
            };

            // Each car has consistent driving characteristics
            let base_speed = 6.0 + (car_id % 12) as f32; // 6-18 m/s
            let mut heading = (car_id as f32 * 2.39996323) % std::f32::consts::TAU;

            let mut pos = car.pos;
            let mut vel = Vec3::new(heading.cos(), 0.0, heading.sin()) * base_speed;

            let mut segments = Vec::with_capacity(NUM_SEGMENTS);

            for i in 0..NUM_SEGMENTS {
                let t0 = (current_time as f32 + i as f32 * SEGMENT_DURATION) as SimTime;

                // Position relative to reference chunk origin (handles cross-chunk movement)
                let p0 = ref_origin.delta_to(pos, chunk_size);
                let v0 = vel;

                // Beautiful smooth curves: gentle steering + subtle speed wobble
                let turn = (rand() - 0.5) * 0.4; // ±0.2 rad per segment (~11°)
                let speed_wobble = 0.88 + rand() * 0.24; // 88-112% speed variation

                heading += turn;
                let speed = base_speed * speed_wobble;
                vel = Vec3::new(heading.cos(), 0.0, heading.sin()) * speed;

                // Advance position using average velocity (matches Hermite tangents nicely)
                let avg_vel = (v0 + vel) * 0.5;
                pos = pos.add_vec3(avg_vel * SEGMENT_DURATION, chunk_size);

                let p1 = ref_origin.delta_to(pos, chunk_size);
                let v1 = vel;

                segments.push(CarSplineSegment {
                    origin: ref_origin,
                    t0,
                    inv_duration: 1.0 / SEGMENT_DURATION,
                    p0,
                    v0,
                    p1,
                    v1,
                });
            }

            result.next_positions.push(CarSpline {
                car_id,
                next_splines: segments,
            });

            // Final position determines chunk transfer
            if pos.chunk != job.chunk_coord {
                result.chunk_transfers.push((car_id, pos.chunk));
            }
        }

        result
    }

    fn apply_results(
        &mut self,
        results: Vec<CarChunkJobResult>,
        car_storage: &mut CarStorage,
        rng: &mut impl Rng,
    ) {
        for result in results {
            // Update the chunk's last simulation time
            if let Some(chunk) = car_storage
                .car_chunk_storage
                .close_mut()
                .get_mut(&result.chunk_coord)
            {
                chunk.last_update_time = result.completed_time;
            }

            // Process chunk transfers
            for (car_id, new_chunk) in result.chunk_transfers {
                car_storage.move_car_between_chunks(result.chunk_coord, new_chunk, car_id);
            }

            // Process despawns
            for car_id in result.despawns {
                car_storage.despawn(car_id);
            }

            // Process positions
            for next_positions in result.next_positions {
                let Some(car) = car_storage.get_mut(next_positions.car_id) else {
                    continue;
                };
                car.next_splines = next_positions.next_splines;
            }

            // Regenerate jitter for next cycle (prevents patterns)
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
