use crate::helpers::positions::{ChunkCoord, ChunkSize, LodStep};
use crate::terrain::chunk_builder::{ChunkBuilder, CpuChunkMesh};
use crate::terrain::terrain::TerrainGenerator;
use crossbeam_channel::{Receiver, Sender, unbounded};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

// threads.rs
#[derive(Clone, Debug)]
pub struct ChunkJob {
    /// (ChunkCoord, LodStep, neg_x_step, pos_x_step, neg_z_step, pos_z_step, version, version again?)
    pub chunks: Vec<(
        ChunkCoord,
        LodStep,
        LodStep,
        LodStep,
        LodStep,
        LodStep,
        u64,
        Arc<AtomicU64>,
        bool,
    )>,
}

pub struct ChunkWorkerPool {
    pub job_tx: Sender<ChunkJob>,
    pub result_rx: Receiver<CpuChunkMesh>,

    versions: Arc<RwLock<HashMap<ChunkCoord, Arc<AtomicU64>>>>,
}

impl ChunkWorkerPool {
    pub fn new(worker_count: usize, terrain_gen: TerrainGenerator, chunk_size: ChunkSize) -> Self {
        let (job_tx, job_rx) = unbounded::<ChunkJob>();
        let (result_tx, result_rx) = unbounded::<CpuChunkMesh>();

        let versions = Arc::new(RwLock::new(HashMap::new()));

        for _worker_id in 0..worker_count {
            let job_rx = job_rx.clone();
            let result_tx = result_tx.clone();
            let terrain = terrain_gen.clone();

            std::thread::spawn(move || {
                loop {
                    match job_rx.recv() {
                        Ok(job) => {
                            for (
                                chunk_coord,
                                self_step,
                                nx_neg,
                                nx_pos,
                                nz_neg,
                                nz_pos,
                                version,
                                version_atomic,
                                has_edits,
                            ) in job.chunks.iter().cloned()
                            {
                                // fast pre-check before doing any work
                                if version_atomic.load(Ordering::Relaxed) != version {
                                    continue;
                                }

                                if let Some(cpu) = ChunkBuilder::build_chunk_cpu(
                                    chunk_coord,
                                    chunk_size,
                                    self_step,
                                    nx_neg,
                                    nx_pos,
                                    nz_neg,
                                    nz_pos,
                                    version,
                                    &version_atomic,
                                    &terrain,
                                    has_edits,
                                ) {
                                    if result_tx.send(cpu).is_err() {
                                        break;
                                    }
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
        }
    }

    pub fn is_current_version(&self, coord: ChunkCoord, version: u64) -> bool {
        let guard = self.versions.read().unwrap();
        match guard.get(&coord) {
            Some(v_atomic) => v_atomic.load(Ordering::Relaxed) == version,
            None => false,
        }
    }

    #[inline(always)]
    pub fn still_current(v: &AtomicU64, expected: u64) -> bool {
        v.load(Ordering::Relaxed) == expected
    }

    pub fn new_version_for(&self, coord: ChunkCoord) -> (u64, Arc<AtomicU64>) {
        let atomic = {
            let mut g = self.versions.write().unwrap();
            g.entry(coord)
                .or_insert_with(|| Arc::new(AtomicU64::new(0)))
                .clone()
        };

        let v = atomic.fetch_add(1, Ordering::Relaxed) + 1;
        (v, atomic)
    }

    pub fn forget_chunk(&self, coord: ChunkCoord) {
        let mut guard = self.versions.write().unwrap();
        guard.remove(&coord);
    }
}
