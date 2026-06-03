use crate::helpers::positions::{ChunkCoord, WorldPos};
use crate::world::cars::car_structs::{CarId, SimTime};
use crate::world::roads::roads::LaneRef;
use glam::{Quat, Vec3};
use std::collections::HashMap;

const FINAL_DRIVE: f32 = 3.9;
const GEAR_RATIOS: [f32; 6] = [3.6, 2.19, 1.41, 1.12, 0.95, 0.80];
const REVERSE_RATIO: f32 = -3.2;
const IDLE_RPM: f32 = 900.0;
const REDLINE_RPM: f32 = 6500.0;
const MAX_ENGINE_TORQUE: f32 = 260.0; // Nm
const DRIVETRAIN_EFF: f32 = 0.85;

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
    pub time: SimTime,
    pub pos: Vec3, // relative to origin
    pub quat: Quat,
    pub velocity: Vec3, // world space, for conforming to terrain etc
    pub lane_ref: Option<LaneRef>,
}

#[derive(Debug, Clone)]
pub struct CarTrajectory {
    pub car_id: CarId,
    pub origin: WorldPos,
    pub points: Vec<CarTrajectoryPoint>,
    pub end_quat: Quat,
    pub end_yaw_rate: f32,
    pub end_steering_angle: f32,
    pub end_steering_vel: f32,
}
#[derive(Debug)]
pub struct CarEngineResult {
    pub car_id: CarId,
    pub rpm: f32,
    pub gear: i8,
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
    pub engine_results: Vec<CarEngineResult>,
}

impl CarChunkJobResult {
    pub fn empty(chunk_coord: ChunkCoord, completed_time: SimTime) -> Self {
        Self {
            chunk_coord,
            completed_time,
            chunk_transfers: Vec::new(),
            despawns: Vec::new(),
            trajectories: Vec::new(),
            engine_results: Vec::new(),
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
}

/// Discard trajectory points older than `cutoff`, keeping at least 2 points.
fn trim_old_trajectory_points(traj: &mut CarTrajectory, cutoff: f64) {
    let points = &mut traj.points;
    if points.len() < 3 {
        return;
    }
    // Find first point at-or-after cutoff, keep one point before it for interpolation.
    let keep_from = points
        .partition_point(|p| p.time < cutoff)
        .saturating_sub(1);
    if keep_from > 0 {
        points.drain(0..keep_from);
    }
}

fn engine_torque_from_rpm(rpm: f32) -> f32 {
    let x = (rpm / 4000.0).clamp(0.0, 2.0);

    // Peaky NA petrol curve
    let torque = if x < 1.0 {
        x * 1.2
    } else {
        1.2 - (x - 1.0) * 0.8
    };

    torque.clamp(0.0, 1.0) * MAX_ENGINE_TORQUE
}
