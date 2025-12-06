use crate::chunk_builder::{ChunkBuilder, ChunkMeshLod, GpuChunkMesh, lod_step_for_distance};
use crate::renderer::pipelines::Pipelines;
use crate::terrain::TerrainGenerator;
use crate::threads::{ChunkJob, ChunkWorkerPool};
use glam::Vec3;
use std::collections::{HashMap, HashSet};
use wgpu::{Device, IndexFormat, RenderPass};

pub struct WorldRenderer {
    pub chunks: HashMap<(i32, i32), ChunkMeshLod>, // GPU-ready
    pub pending: HashSet<(i32, i32)>,              // requested but not uploaded yet
    pub terrain_gen: TerrainGenerator,
    pub chunk_size: i32,
    pub view_radius_generate: i32,
    pub view_radius_render: i32,
    pub workers: ChunkWorkerPool,
    pub max_jobs_per_frame: usize,
}

impl WorldRenderer {
    pub fn new(device: &Device) -> Self {
        let terrain_gen = TerrainGenerator::new(0);
        let chunk_size = 256;
        let view_radius_render = 16;
        let view_radius_generate = 16;

        let threads = num_cpus::get_physical().saturating_sub(1).max(1);
        println!("Using {} chunk workers", threads);

        let workers = ChunkWorkerPool::new(threads, terrain_gen.clone(), chunk_size as u32);

        let mut chunks = HashMap::new();
        let pending = HashSet::new();

        // Build origin chunk immediately with the highest LOD
        let cpu_origin = ChunkBuilder::build_chunk_cpu(
            0,
            0,
            chunk_size as u32,
            1, // highest detail
            &terrain_gen,
        );
        let gpu_origin = GpuChunkMesh::from_cpu(device, &cpu_origin);

        chunks.insert(
            (0, 0),
            ChunkMeshLod {
                step: 1,
                mesh: gpu_origin,
            },
        );

        Self {
            chunks,
            pending,
            terrain_gen,
            chunk_size,
            view_radius_generate,
            view_radius_render,
            workers,
            max_jobs_per_frame: 8,
        }
    }

    pub fn update(&mut self, device: &Device, camera_pos: Vec3) {
        let cs = self.chunk_size as f32;
        let cam_cx = (camera_pos.x / cs).floor() as i32;
        let cam_cz = (camera_pos.z / cs).floor() as i32;

        // 1. Drain finished CPU meshes  (Step 6)
        while let Ok(cpu) = self.workers.result_rx.try_recv() {
            let coord = (cpu.cx, cpu.cz);
            self.pending.remove(&coord);

            let gpu = GpuChunkMesh::from_cpu(device, &cpu);

            self.chunks.insert(
                coord,
                ChunkMeshLod {
                    step: cpu.step,
                    mesh: gpu,
                },
            );
        }

        // 2. Decide which chunks to generate
        let mut jobs_sent = 0usize;

        for dx in -self.view_radius_generate..=self.view_radius_generate {
            for dz in -self.view_radius_generate..=self.view_radius_generate {
                let cx = cam_cx + dx;
                let cz = cam_cz + dz;
                let dist2 = dx * dx + dz * dz;

                let desired_step = lod_step_for_distance(dist2);
                let coord = (cx, cz);

                // If chunk exists AND LOD is equal or better: skip
                if let Some(existing) = self.chunks.get(&coord) {
                    if existing.step <= desired_step {
                        continue;
                    }
                }

                // If chunk is already being generated: skip
                if self.pending.contains(&coord) {
                    continue;
                }

                // Send job to worker
                if jobs_sent < self.max_jobs_per_frame {
                    self.pending.insert(coord);
                    self.workers
                        .job_tx
                        .send(ChunkJob {
                            cx,
                            cz,
                            step: desired_step,
                        })
                        .ok();
                    jobs_sent += 1;
                }
            }
        }
    }

    pub fn render<'a>(
        &'a self,
        pass: &mut RenderPass<'a>,
        pipelines: &'a Pipelines,
        camera_pos: Vec3,
    ) {
        pass.set_pipeline(&pipelines.pipeline);
        pass.set_bind_group(0, &pipelines.uniform_bind_group, &[]);

        let cs = self.chunk_size as f32;
        let cam_cx = (camera_pos.x / cs).floor() as i32;
        let cam_cz = (camera_pos.z / cs).floor() as i32;
        let r2 = (self.view_radius_render * self.view_radius_render) as i32;

        for (&(cx, cz), chunk) in self.chunks.iter() {
            let dx = cx - cam_cx;
            let dz = cz - cam_cz;
            if dx * dx + dz * dz > r2 {
                continue;
            }

            pass.set_vertex_buffer(0, chunk.mesh.vertex_buf.slice(..));
            pass.set_index_buffer(chunk.mesh.index_buf.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..chunk.mesh.index_count, 0, 0..1);
        }
    }
}
