use crate::chunk_builder::{
    ChunkMeshLod, GpuChunkMesh, generate_spiral_offsets, lod_step_for_distance,
};
use crate::components::camera::Camera;
use crate::renderer::pipelines::Pipelines;
use crate::terrain::TerrainGenerator;
use crate::threads::{ChunkJob, ChunkWorkerPool};
use glam::Vec3;
use std::collections::HashMap;
use wgpu::{Device, IndexFormat, RenderPass};

pub struct WorldRenderer {
    pub chunks: HashMap<(i32, i32), ChunkMeshLod>,
    pub pending: HashMap<(i32, i32), (u64, usize)>, // coord -> (version, desired_step)
    pub terrain_gen: TerrainGenerator,
    pub chunk_size: u32,
    pub view_radius_generate: u32,
    pub view_radius_render: u32,
    pub workers: ChunkWorkerPool,
    pub max_jobs_per_frame: usize,
    pub max_chunks_per_batch: usize,
    pub max_far_batches_per_frame: usize,

    pub spiral: Vec<(i32, i32)>,
    pub lod_map: HashMap<(i32, i32), usize>,
}

impl WorldRenderer {
    pub fn new() -> Self {
        let terrain_gen = TerrainGenerator::new(201035458);
        let chunk_size = 512;
        let view_radius_render = 16;
        let view_radius_generate = 8;

        let threads = num_cpus::get_physical().saturating_sub(1).max(1);
        println!("Using {} chunk workers", threads);

        let workers = ChunkWorkerPool::new(threads, terrain_gen.clone(), chunk_size as u32);

        let mut chunks = HashMap::new();
        let pending = HashMap::new();

        // Build origin chunk immediately with the highest LOD
        // let cpu_origin =
        //     ChunkBuilder::build_chunk_cpu(0, 0, chunk_size, 1, 1, 1, 1, 1, &terrain_gen);
        //
        // let gpu_origin = GpuChunkMesh::from_cpu(device, &cpu_origin);

        // chunks.insert(
        //     (0, 0),
        //     ChunkMeshLod {
        //         step: 1,
        //         mesh: gpu_origin,
        //     },
        // );

        Self {
            chunks,
            pending,
            terrain_gen,
            chunk_size,
            view_radius_generate,
            view_radius_render,
            workers,
            max_jobs_per_frame: 2,
            max_chunks_per_batch: 8,
            max_far_batches_per_frame: 2,

            spiral: generate_spiral_offsets(view_radius_generate as i32),
            lod_map: HashMap::new(),
        }
    }

    pub fn update(&mut self, device: &Device, camera_pos: Vec3) {
        let cs = self.chunk_size as f32;
        let cam_cx = (camera_pos.x / cs).floor() as i32;
        let cam_cz = (camera_pos.z / cs).floor() as i32;

        // 1. Drain finished CPU meshes
        while let Ok(cpu) = self.workers.result_rx.try_recv() {
            let coord = (cpu.cx, cpu.cz);

            if !self.workers.is_current_version(coord, cpu.version) {
                // obsolete result from an old job, ignore
                continue;
            }

            // this job is now done
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

        // 2. Compute desired LOD for all candidate chunks
        self.lod_map.clear();
        let r_gen = self.view_radius_generate as i32;
        let r_gen2 = r_gen * r_gen;

        for (dx, dz) in &self.spiral {
            let dx = *dx;
            let dz = *dz;
            let dist2 = dx * dx + dz * dz;
            if dist2 > r_gen2 {
                continue;
            }

            let cx = cam_cx + dx;
            let cz = cam_cz + dz;
            let step = lod_step_for_distance(dist2);

            self.lod_map.insert((cx, cz), step);
        }

        // 2b. Smooth LODs
        for _iter in 0..2 {
            let current = self.lod_map.clone();
            for (&(cx, cz), step) in current.iter() {
                let s = *step;

                let n0 = current.get(&(cx - 1, cz)).copied().unwrap_or(s);
                let n1 = current.get(&(cx + 1, cz)).copied().unwrap_or(s);
                let n2 = current.get(&(cx, cz - 1)).copied().unwrap_or(s);
                let n3 = current.get(&(cx, cz + 1)).copied().unwrap_or(s);

                let min_step = s.min(n0).min(n1).min(n2).min(n3);
                self.lod_map.insert((cx, cz), min_step);
            }
        }

        // 3. Job creation: unified rebuild logic (upgrade + downgrade)
        let mut batch: Vec<(i32, i32, usize, usize, usize, usize, usize, u64)> = Vec::new();
        let mut near_jobs_sent = 0usize;
        let mut far_batches_sent = 0usize;

        for (dx, dz) in &self.spiral {
            if near_jobs_sent >= self.max_jobs_per_frame
                && far_batches_sent >= self.max_far_batches_per_frame
            {
                break;
            }

            let dx = *dx;
            let dz = *dz;
            let cx = cam_cx + dx;
            let cz = cam_cz + dz;
            let coord = (cx, cz);

            let r_render = self.view_radius_render as i32;
            let r_render2 = r_render * r_render;
            let dist2 = dx * dx + dz * dz;

            // outside render radius: ignore completely
            if dist2 > r_render2 {
                continue;
            }

            // inside outer ring: force coarse LOD
            let step = if dist2 > r_gen2 {
                let coarse = lod_step_for_distance(r_gen2 + 1);
                self.lod_map.insert(coord, coarse);
                coarse
            } else {
                let s = lod_step_for_distance(dist2);
                self.lod_map.insert(coord, s);
                s
            };

            let desired_step = *self.lod_map.get(&coord).unwrap_or(&1);
            let existing_step = self.chunks.get(&coord).map(|c| c.step);
            let pending_entry = self.pending.get(&coord).copied();

            // decide whether to schedule a new job
            let need_job = match (existing_step, pending_entry) {
                // already have a chunk at desired LOD
                (Some(cur), _) if cur == desired_step => false,
                // already have a job in flight for this coord and LOD
                (_, Some((_version, pending_step))) if pending_step == desired_step => false,
                // otherwise we need a job
                _ => true,
            };

            if !need_job {
                continue;
            }

            let step = desired_step;

            // neighbor LODs
            let n_x_neg = *self.lod_map.get(&(cx - 1, cz)).unwrap_or(&step);
            let n_x_pos = *self.lod_map.get(&(cx + 1, cz)).unwrap_or(&step);
            let n_z_neg = *self.lod_map.get(&(cx, cz - 1)).unwrap_or(&step);
            let n_z_pos = *self.lod_map.get(&(cx, cz + 1)).unwrap_or(&step);

            if step <= 2 {
                // near chunks: high priority, one per job
                if near_jobs_sent < self.max_jobs_per_frame {
                    let version = self.workers.new_version_for(coord);
                    self.pending.insert(coord, (version, step));

                    let job = ChunkJob {
                        chunks: vec![(cx, cz, step, n_x_neg, n_x_pos, n_z_neg, n_z_pos, version)],
                    };
                    let _ = self.workers.job_tx.send(job);
                    near_jobs_sent += 1;
                }
                continue;
            }

            // far chunks: batched
            if far_batches_sent < self.max_far_batches_per_frame {
                let version = self.workers.new_version_for(coord);
                self.pending.insert(coord, (version, step));

                batch.push((cx, cz, step, n_x_neg, n_x_pos, n_z_neg, n_z_pos, version));

                if batch.len() >= self.max_chunks_per_batch {
                    let job = ChunkJob {
                        chunks: batch.clone(),
                    };
                    let _ = self.workers.job_tx.send(job);
                    batch.clear();
                    far_batches_sent += 1;
                }
            }
        }

        // flush leftover far batch
        if !batch.is_empty() && far_batches_sent < self.max_far_batches_per_frame {
            let job = ChunkJob { chunks: batch };
            let _ = self.workers.job_tx.send(job);
        }
    }

    pub fn render<'a>(
        &'a self,
        pass: &mut RenderPass<'a>,
        pipelines: &'a Pipelines,
        camera: &'a Camera,
        aspect: f32,
    ) {
        pass.set_pipeline(&pipelines.pipeline);
        pass.set_bind_group(0, &pipelines.uniform_bind_group, &[]);
        pass.set_bind_group(1, &pipelines.fog_bind_group, &[]);
        // compute planes once per frame
        let planes = extract_frustum_planes(camera.view_proj(aspect));

        let cs = self.chunk_size as f32;
        let cam_pos = camera.position();
        let cam_cx = (cam_pos.x / cs).floor() as i32;
        let cam_cz = (cam_pos.z / cs).floor() as i32;

        let r2 = (self.view_radius_render * self.view_radius_render) as i32;

        // fixed terrain height bounds (can be dynamic later)
        let min_y = -50.0;
        let max_y = 200.0;

        for (&(cx, cz), chunk) in self.chunks.iter() {
            // distance radius culling (your old check)
            let dx = cx - cam_cx;
            let dz = cz - cam_cz;
            if dx * dx + dz * dz > r2 {
                continue;
            }

            // world-space bounds
            let world_x = cx as f32 * cs;
            let world_z = cz as f32 * cs;

            let min = glam::Vec3::new(world_x, min_y, world_z);
            let max = glam::Vec3::new(world_x + cs, max_y, world_z + cs);

            // frustum culling
            if !aabb_in_frustum(&planes, min, max) {
                continue;
            }

            // draw chunk
            let mesh = &chunk.mesh;
            pass.set_vertex_buffer(0, mesh.vertex_buf.slice(..));
            pass.set_index_buffer(mesh.index_buf.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
    }
}
#[derive(Clone, Copy)]
pub struct Plane {
    pub normal: glam::Vec3,
    pub d: f32,
}

impl Plane {
    pub fn distance(&self, p: glam::Vec3) -> f32 {
        self.normal.dot(p) + self.d
    }
}

pub fn extract_frustum_planes(view_proj: glam::Mat4) -> [Plane; 6] {
    let m = view_proj.to_cols_array_2d();

    let planes = [
        (
            m[0][3] + m[0][0],
            m[1][3] + m[1][0],
            m[2][3] + m[2][0],
            m[3][3] + m[3][0],
        ), // left
        (
            m[0][3] - m[0][0],
            m[1][3] - m[1][0],
            m[2][3] - m[2][0],
            m[3][3] - m[3][0],
        ), // right
        (
            m[0][3] + m[0][1],
            m[1][3] + m[1][1],
            m[2][3] + m[2][1],
            m[3][3] + m[3][1],
        ), // bottom
        (
            m[0][3] - m[0][1],
            m[1][3] - m[1][1],
            m[2][3] - m[2][1],
            m[3][3] - m[3][1],
        ), // top
        (
            m[0][3] + m[0][2],
            m[1][3] + m[1][2],
            m[2][3] + m[2][2],
            m[3][3] + m[3][2],
        ), // near
        (
            m[0][3] - m[0][2],
            m[1][3] - m[1][2],
            m[2][3] - m[2][2],
            m[3][3] - m[3][2],
        ), // far
    ];

    planes.map(|(a, b, c, d)| {
        let n = glam::Vec3::new(a, b, c);
        let inv_len = 1.0 / n.length();
        Plane {
            normal: n * inv_len,
            d: d * inv_len,
        }
    })
}

pub fn aabb_in_frustum(planes: &[Plane; 6], min: glam::Vec3, max: glam::Vec3) -> bool {
    for p in planes {
        let vx = if p.normal.x >= 0.0 { max.x } else { min.x };
        let vy = if p.normal.y >= 0.0 { max.y } else { min.y };
        let vz = if p.normal.z >= 0.0 { max.z } else { min.z };

        if p.distance(glam::Vec3::new(vx, vy, vz)) < 0.0 {
            return false;
        }
    }
    true
}
