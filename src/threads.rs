use crate::chunk_builder::{ChunkBuilder, CpuChunkMesh};
use crate::terrain::TerrainGenerator;
use crossbeam_channel::{Receiver, Sender, unbounded};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

// threads.rs
#[derive(Clone, Debug)]
pub struct ChunkJob {
    /// (cx, cz, self_step, neg_x_step, pos_x_step, neg_z_step, pos_z_step, version)
    pub chunks: Vec<(i32, i32, usize, usize, usize, usize, usize, u64)>,
}

pub struct ChunkWorkerPool {
    pub job_tx: Sender<ChunkJob>,
    pub result_rx: Receiver<CpuChunkMesh>,

    versions: Arc<RwLock<HashMap<(i32, i32), u64>>>,
    next_version: AtomicU64,
}

impl ChunkWorkerPool {
    pub fn new(worker_count: usize, terrain_gen: TerrainGenerator, chunk_size: u32) -> Self {
        let (job_tx, job_rx) = unbounded::<ChunkJob>();
        let (result_tx, result_rx) = unbounded::<CpuChunkMesh>();

        let versions = Arc::new(RwLock::new(HashMap::new()));
        let next_version = AtomicU64::new(0);

        for _worker_id in 0..worker_count {
            let job_rx = job_rx.clone();
            let result_tx = result_tx.clone();
            let terrain = terrain_gen.clone();
            let versions = versions.clone();

            std::thread::spawn(move || {
                loop {
                    match job_rx.recv() {
                        Ok(job) => {
                            for (cx, cz, self_step, nx_neg, nx_pos, nz_neg, nz_pos, version) in
                                job.chunks.iter().copied()
                            {
                                let keep = {
                                    let guard = versions.read().unwrap();
                                    match guard.get(&(cx, cz)) {
                                        Some(&v) => v == version,
                                        None => false,
                                    }
                                };

                                if !keep {
                                    continue;
                                }

                                let cpu = ChunkBuilder::build_chunk_cpu(
                                    cx, cz, chunk_size, self_step, nx_neg, nx_pos, nz_neg, nz_pos,
                                    version, &terrain,
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

        Self {
            job_tx,
            result_rx,
            versions,
            next_version,
        }
    }

    pub fn new_version_for(&self, coord: (i32, i32)) -> u64 {
        let v = self.next_version.fetch_add(1, Ordering::Relaxed) + 1;
        let mut guard = self.versions.write().unwrap();
        guard.insert(coord, v);
        v
    }

    pub fn is_current_version(&self, coord: (i32, i32), version: u64) -> bool {
        let guard = self.versions.read().unwrap();
        matches!(guard.get(&coord), Some(&v) if v == version)
    }

    pub fn forget_chunk(&self, coord: (i32, i32)) {
        let mut guard = self.versions.write().unwrap();
        guard.remove(&coord);
    }
}
