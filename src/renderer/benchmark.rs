use crate::components::camera::Camera;
use rand::RngExt;
use std::time::Instant;

const MAX_CLOSE_JOBS: usize = 8;
const MAX_FAR_JOBS: usize = 8;
const MAX_BATCH: usize = 128;

#[derive(Clone, Copy)]
struct BenchRange {
    min: usize,
}

struct BenchmarkState {
    active: bool,
    start: Instant,
    best_score: f32,
    step: u32,
    close_jobs: BenchRange,
    far_jobs: BenchRange,
    close_batch: BenchRange,
    far_batch: BenchRange,
}

pub struct ChunkJobConfig {
    pub max_close_jobs_per_frame: usize,
    pub max_far_jobs_per_frame: usize,
    pub max_close_chunks_per_batch: usize,
    pub max_far_chunks_per_batch: usize,
}

impl Default for ChunkJobConfig {
    fn default() -> Self {
        Self {
            max_close_jobs_per_frame: 1,
            max_far_jobs_per_frame: 1,
            max_close_chunks_per_batch: 2,
            max_far_chunks_per_batch: 100,
        }
    }
}

pub struct Benchmark {
    state: Option<BenchmarkState>,
}

impl Default for Benchmark {
    fn default() -> Self {
        Self { state: None }
    }
}

impl Benchmark {
    pub fn run(
        &mut self,
        _camera: &mut Camera,
        chunk_count: usize,
        config: &mut ChunkJobConfig,
        clear_state: impl FnOnce(),
    ) {
        //camera.target.x += 220.0;

        if self.state.is_none() {
            self.state = Some(BenchmarkState {
                active: true,
                start: Instant::now(),
                best_score: 0.0,
                step: 0,
                close_jobs: BenchRange { min: 4 },
                far_jobs: BenchRange { min: 4 },
                close_batch: BenchRange { min: 16 },
                far_batch: BenchRange { min: 64 },
            });
        }

        let bench = self.state.as_mut().unwrap();

        if bench.active {
            self.start_run(config, clear_state);
            return;
        }

        self.evaluate_run(chunk_count, config);
    }

    fn start_run(&mut self, config: &mut ChunkJobConfig, clear_state: impl FnOnce()) {
        let bench = self.state.as_mut().unwrap();
        let mut rng = rand::rng();
        let explore = rng.random_bool(0.2);

        if explore {
            config.max_close_jobs_per_frame = rng.random_range(1..=MAX_CLOSE_JOBS);
            config.max_far_jobs_per_frame = rng.random_range(1..=MAX_FAR_JOBS);
            config.max_close_chunks_per_batch = rng.random_range(4..=MAX_BATCH);
            config.max_far_chunks_per_batch = rng.random_range(8..=MAX_BATCH);
        } else {
            config.max_close_jobs_per_frame = bench.close_jobs.min;
            config.max_far_jobs_per_frame = bench.far_jobs.min;
            config.max_close_chunks_per_batch = bench.close_batch.min;
            config.max_far_chunks_per_batch = bench.far_batch.min;
        }

        clear_state();
        bench.start = Instant::now();
        bench.active = false;

        println!(
            "Benchmark step {} | close jobs {} batch {} | far jobs {} batch {}{}",
            bench.step,
            config.max_close_jobs_per_frame,
            config.max_close_chunks_per_batch,
            config.max_far_jobs_per_frame,
            config.max_far_chunks_per_batch,
            if explore { " (explore)" } else { "" }
        );
    }

    fn evaluate_run(&mut self, chunk_count: usize, config: &mut ChunkJobConfig) {
        let bench = self.state.as_mut().unwrap();
        let seconds = bench.start.elapsed().as_secs_f32();
        let score = chunk_count as f32 / seconds;

        if bench.best_score > 0.0 && score < bench.best_score && seconds > 0.2 {
            println!("Current run slower than best, skipping");
            bench.step += 1;
            bench.active = true;
            return;
        }

        if seconds < 2.0 {
            return;
        }

        println!(
            "Generated {} chunks in {:.2}s ({:.1} chunks/s)",
            chunk_count, seconds, score
        );

        if score > bench.best_score {
            bench.best_score = score;
            bench.close_jobs.min = config.max_close_jobs_per_frame;
            bench.far_jobs.min = config.max_far_jobs_per_frame;
            bench.close_batch.min = config.max_close_chunks_per_batch;
            bench.far_batch.min = config.max_far_chunks_per_batch;
            println!("New best configuration");
        }

        bench.step += 1;
        bench.active = true;
    }
}
