use crate::chunk_builder::{ChunkBuilder, CpuChunkMesh};
use crate::terrain::TerrainGenerator;
use crossbeam_channel::{Receiver, Sender, unbounded};

// threads.rs

#[derive(Clone, Debug)]
pub struct ChunkJob {
    /// (cx, cz, self_step, neg_x_step, pos_x_step, neg_z_step, pos_z_step)
    pub chunks: Vec<(i32, i32, usize, usize, usize, usize, usize)>,
}

pub struct ChunkWorkerPool {
    pub job_tx: Sender<ChunkJob>,
    pub result_rx: Receiver<CpuChunkMesh>,
}

impl ChunkWorkerPool {
    pub fn new(worker_count: usize, terrain_gen: TerrainGenerator, chunk_size: u32) -> Self {
        use crossbeam_channel::unbounded;

        let (job_tx, job_rx) = unbounded::<ChunkJob>();
        let (result_tx, result_rx) = unbounded::<CpuChunkMesh>();

        for worker_id in 0..worker_count {
            let job_rx = job_rx.clone();
            let result_tx = result_tx.clone();
            let terrain = terrain_gen.clone();

            std::thread::spawn(move || {
                loop {
                    match job_rx.recv() {
                        Ok(job) => {
                            for (cx, cz, self_step, nx_neg, nx_pos, nz_neg, nz_pos) in
                                job.chunks.iter().copied()
                            {
                                let cpu = ChunkBuilder::build_chunk_cpu(
                                    cx, cz, chunk_size, self_step, nx_neg, nx_pos, nz_neg, nz_pos,
                                    &terrain,
                                );

                                if result_tx.send(cpu).is_err() {
                                    break;
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
            });
        }

        Self { job_tx, result_rx }
    }
}
