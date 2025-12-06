use crate::chunk_builder::{ChunkBuilder, CpuChunkMesh};
use crate::terrain::TerrainGenerator;
use crossbeam_channel::{Receiver, Sender};

#[derive(Clone)]
pub struct ChunkJob {
    pub cx: i32,
    pub cz: i32,
    pub step: usize,
}

pub struct ChunkWorker {
    pub job_tx: Sender<ChunkJob>,
    pub result_rx: Receiver<CpuChunkMesh>,
}

impl ChunkWorker {
    pub fn new(terrain_gen: TerrainGenerator, chunk_size: u32) -> Self {
        let (job_tx, job_rx) = crossbeam_channel::unbounded::<ChunkJob>();
        let (result_tx, result_rx) = crossbeam_channel::unbounded::<CpuChunkMesh>();

        std::thread::spawn(move || {
            let terrain = terrain_gen;
            loop {
                match job_rx.recv() {
                    Ok(job) => {
                        let cpu = ChunkBuilder::build_chunk_cpu(
                            job.cx, job.cz, chunk_size, job.step, &terrain,
                        );
                        if result_tx.send(cpu).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        Self { job_tx, result_rx }
    }
}

pub struct ChunkWorkerPool {
    pub job_tx: Sender<ChunkJob>,
    pub result_rx: Receiver<CpuChunkMesh>,
}

impl ChunkWorkerPool {
    pub fn new(worker_count: usize, terrain_gen: TerrainGenerator, chunk_size: u32) -> Self {
        let (job_tx, job_rx) = crossbeam_channel::unbounded::<ChunkJob>();
        let (result_tx, result_rx) = crossbeam_channel::unbounded::<CpuChunkMesh>();

        for worker_id in 0..worker_count {
            let job_rx = job_rx.clone();
            let result_tx = result_tx.clone();
            let terrain = terrain_gen.clone();

            std::thread::spawn(move || {
                loop {
                    match job_rx.recv() {
                        Ok(job) => {
                            let cpu = ChunkBuilder::build_chunk_cpu(
                                job.cx, job.cz, chunk_size, job.step, &terrain,
                            );
                            let _ = result_tx.send(cpu);
                        }
                        Err(_) => break,
                    }
                }
            });

            println!("Spawned chunk worker {}", worker_id);
        }

        Self { job_tx, result_rx }
    }
}
