use crate::components::camera::Camera;
use crate::mouse_ray::*;
use crate::renderer::mesh_arena::{GeometryScratch, MeshArena};
use crate::renderer::pipelines::Pipelines;
use crate::terrain::chunk_builder::{
    ChunkHeightGrid, ChunkMeshLod, EditedChunk, GpuChunkHandle, apply_sparse_deltas_to_height_grid,
    generate_spiral_offsets, lod_step_for_distance, regenerate_vertices_from_height_grid,
};
use crate::terrain::terrain::{TerrainGenerator, TerrainParams};
use crate::terrain::threads::{ChunkJob, ChunkWorkerPool};
use crate::ui::vertex::Vertex;
use glam::Vec3;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use wgpu::{Buffer, Device, IndexFormat, Queue, RenderPass};

use crate::data::Settings;
use crate::resources::{InputState, TimeSystem};
use crate::terrain::terrain_editing::*;
use rayon::prelude::*;
use std::time::Instant;

#[derive(Clone, Copy)]
enum BenchParam {
    CloseJobs,
    FarJobs,
    CloseBatch,
    FarBatch,
}

struct HillState {
    param: BenchParam,
    direction: i32,
    step: usize,
}

#[derive(Clone, Copy)]
struct BenchRange {
    min: usize,
    max: usize,
}

struct BenchmarkState {
    active: bool,
    start: Instant,
    best_score: f32,

    close_jobs: BenchRange,
    far_jobs: BenchRange,
    close_batch: BenchRange,
    far_batch: BenchRange,

    step: usize,
    best_score_seconds: f32,
}

pub struct PickedPoint {
    pub pos: glam::Vec3,
    pub radius: f32,
}

pub struct WorldRenderer {
    pub arena: MeshArena,

    pub chunks: HashMap<(i32, i32), ChunkMeshLod>,
    pub pending: HashMap<(i32, i32), (u64, usize)>, // coord -> (version, desired_step)
    pub edited_chunks: HashMap<(i32, i32), EditedChunk>,

    pub terrain_gen: TerrainGenerator,
    pub chunk_size: u32,
    pub view_radius_generate: u32,
    pub view_radius_render: u32,

    pub workers: ChunkWorkerPool,
    pub max_close_jobs_per_frame: usize,
    pub max_close_chunks_per_batch: usize,
    pub max_far_chunks_per_batch: usize,
    pub max_far_jobs_per_frame: usize,

    pub spiral: Vec<(i32, i32)>,
    pub lod_map: HashMap<(i32, i32), usize>,
    pub pick_radius_m: f32, // controlled by UI slider
    last_picked: Option<PickedPoint>,

    benchmark: Option<BenchmarkState>,
    pub frame_timings: FrameTimings,
}

#[derive(Clone, Copy)]
struct FrameState {
    cs: f32,
    cam_cx: i32,
    cam_cz: i32,
    planes: [Plane; 6],
    r2_render: i32,
    r2_gen: i32,
}

impl WorldRenderer {
    pub fn new(device: &Device) -> Self {
        let mut terrain_params = TerrainParams::default();
        terrain_params.seed = 201035458;
        let terrain_gen = TerrainGenerator::new(terrain_params);

        let chunk_size = 128;
        let view_radius_render = 128;
        let view_radius_generate = 64;

        // Paged arena: interpret these as "page sizes" if your MeshArena is paged.
        let arena = MeshArena::new(
            device,
            256 * 1024 * 1024, // vertex bytes per page
            128 * 1024 * 1024, // index bytes per page
        );

        let threads = num_cpus::get_physical().saturating_sub(1).max(1);
        println!("Using {} chunk workers", threads);

        let workers = ChunkWorkerPool::new(threads, terrain_gen.clone(), chunk_size as u32);

        Self {
            arena,
            chunks: HashMap::new(),
            pending: HashMap::new(),

            edited_chunks: HashMap::new(),
            terrain_gen,
            chunk_size,
            view_radius_generate,
            view_radius_render,

            workers,
            max_close_jobs_per_frame: 1,
            max_close_chunks_per_batch: 2,
            max_far_chunks_per_batch: 100,
            max_far_jobs_per_frame: 1,

            spiral: generate_spiral_offsets(view_radius_generate as i32),
            lod_map: HashMap::new(),
            pick_radius_m: 10.0,
            last_picked: None,

            benchmark: None,
            frame_timings: FrameTimings::default(),
        }
    }

    pub fn run_benchmark(&mut self, camera: &mut Camera) {
        use rand::Rng;
        camera.target.x += 220.0;
        const MAX_CLOSE_JOBS: usize = 8;
        const MAX_FAR_JOBS: usize = 8;
        const MAX_BATCH: usize = 128;

        if self.benchmark.is_none() {
            self.benchmark = Some(BenchmarkState {
                active: true,
                start: Instant::now(),
                best_score: 0.0,

                close_jobs: BenchRange { min: 4, max: 4 },
                far_jobs: BenchRange { min: 4, max: 16 },
                close_batch: BenchRange { min: 16, max: 16 },
                far_batch: BenchRange { min: 64, max: 64 },

                step: 0,
                best_score_seconds: 0.0,
            });
        }

        let bench = self.benchmark.as_mut().unwrap();

        // Start run
        if bench.active {
            let mut rng = rand::thread_rng();

            let explore = rng.gen_bool(0.2);

            if explore {
                self.max_close_jobs_per_frame = rng.gen_range(1..=MAX_CLOSE_JOBS);
                self.max_far_jobs_per_frame = rng.gen_range(1..=MAX_FAR_JOBS);
                self.max_close_chunks_per_batch = rng.gen_range(4..=MAX_BATCH);
                self.max_far_chunks_per_batch = rng.gen_range(8..=MAX_BATCH);
            } else {
                self.max_close_jobs_per_frame = bench.close_jobs.min;
                self.max_far_jobs_per_frame = bench.far_jobs.min;
                self.max_close_chunks_per_batch = bench.close_batch.min;
                self.max_far_chunks_per_batch = bench.far_batch.min;
            }

            self.chunks.clear();
            self.pending.clear();
            self.lod_map.clear();

            bench.start = Instant::now();
            bench.active = false;

            println!(
                "Benchmark step {} | close jobs per frame {}, close batch chunk amount {} | far jobs per frame {} far batch chunk amount {}{}",
                bench.step,
                self.max_close_jobs_per_frame,
                self.max_close_chunks_per_batch,
                self.max_far_jobs_per_frame,
                self.max_far_chunks_per_batch,
                if explore { " (explore)" } else { "" }
            );

            return;
        }

        // Wait / skip if current run is already slower than best
        let seconds = bench.start.elapsed().as_secs_f32();
        let score = self.chunks.len() as f32 / seconds;
        if bench.best_score > 0.0 && score < bench.best_score && seconds > 0.2 {
            println!("Current run slower than best, skipping to next iteration");
            bench.step += 1;
            bench.active = true;
            return;
        }

        if seconds < 2.0 {
            return;
        }

        // Score
        bench.best_score_seconds = seconds;
        println!(
            "Generated {} chunks in {:.2}s ({:.1} chunks/s)",
            self.chunks.len(),
            seconds,
            score
        );

        if score > bench.best_score {
            bench.best_score = score;

            bench.close_jobs.min = self.max_close_jobs_per_frame;
            bench.far_jobs.min = self.max_far_jobs_per_frame;
            bench.close_batch.min = self.max_close_chunks_per_batch;
            bench.far_batch.min = self.max_far_chunks_per_batch;

            println!("New best configuration accepted");
        }

        bench.step += 1;
        bench.active = true;
    }

    pub fn update(
        &mut self,
        device: &Device,
        queue: &Queue,
        camera: &mut Camera,
        aspect: f32,
        settings: &Settings,
        input_state: &mut InputState,
        time_system: &TimeSystem,
    ) {
        let t_frame = std::time::Instant::now();

        if settings.world_generation_benchmark_mode {
            self.run_benchmark(camera);
        }

        let frame = self.frame_state(camera, aspect);

        // 1) Drain finished meshes (CPU → GPU uploads, allocations)
        let t0 = std::time::Instant::now();
        self.drain_finished_meshes(device, queue);
        self.frame_timings.drain_ms = t0.elapsed().as_secs_f32() * 1000.0;

        // 2) Visible set build (culling cost)
        let t0 = std::time::Instant::now();
        let mut visible = self.collect_visible(&frame);
        self.frame_timings.collect_visible_ms = t0.elapsed().as_secs_f32() * 1000.0;

        // 2.5) Sorting visible (often non-trivial!)
        let t0 = std::time::Instant::now();
        visible.sort_unstable_by_key(|&(_cx, _cz, dist2)| dist2);
        self.frame_timings.sort_visible_ms = t0.elapsed().as_secs_f32() * 1000.0;

        // 3) LOD decisions (math + data access)
        let t0 = std::time::Instant::now();
        self.compute_lod_for_visible(&visible, frame.r2_gen);
        self.frame_timings.lod_ms = t0.elapsed().as_secs_f32() * 1000.0;

        // 4) Job dispatch (allocs, queues, atomics)
        let t0 = std::time::Instant::now();
        self.dispatch_jobs_for_visible(&visible);
        self.frame_timings.dispatch_ms = t0.elapsed().as_secs_f32() * 1000.0;

        // 5) Unload chunks (hashmaps + GPU frees)
        let t0 = std::time::Instant::now();
        self.unload_out_of_range(&frame, &visible);
        self.frame_timings.unload_ms = t0.elapsed().as_secs_f32() * 1000.0;

        // 6) Terrain editing (this is where spikes happen)
        let t0 = std::time::Instant::now();
        if input_state.action_down("Edit Terrain +") {
            if let Some(last_picked) = &self.last_picked {
                self.edit_terrain_with_brush::<SmoothFalloff, Raise>(
                    device,
                    queue,
                    last_picked.pos,
                    self.pick_radius_m,
                    1.0,
                    &mut GeometryScratch::default(),
                    time_system,
                    false,
                );
            }
        } else if input_state.action_down("Edit Terrain -") {
            if let Some(last_picked) = &self.last_picked {
                self.edit_terrain_with_brush::<SmoothFalloff, Raise>(
                    device,
                    queue,
                    last_picked.pos,
                    self.pick_radius_m,
                    -1.0,
                    &mut GeometryScratch::default(),
                    time_system,
                    false,
                );
            }
        }
        if input_state.action_released("Edit Terrain -")
            || input_state.action_released("Edit Terrain +")
        {
            if let Some(last_picked) = &self.last_picked {
                println!("RUN");
                self.edit_terrain_with_brush::<SmoothFalloff, Raise>(
                    device,
                    queue,
                    last_picked.pos,
                    self.pick_radius_m,
                    -1.0,
                    &mut GeometryScratch::default(),
                    time_system,
                    true,
                );
            }
        }
        self.frame_timings.edit_ms = t0.elapsed().as_secs_f32() * 1000.0;

        self.frame_timings.total_ms = t_frame.elapsed().as_secs_f32() * 1000.0;

        if (time_system.total_time / time_system.target_frametime as f64) as u32 % 12 == 0 {
            println!(
                "frame {:.2} | drain {:.2} | vis {:.2}+{:.2} | lod {:.2} | disp {:.2} | unload {:.2} | edit {:.2}",
                self.frame_timings.total_ms,
                self.frame_timings.drain_ms,
                self.frame_timings.collect_visible_ms,
                self.frame_timings.sort_visible_ms,
                self.frame_timings.lod_ms,
                self.frame_timings.dispatch_ms,
                self.frame_timings.unload_ms,
                self.frame_timings.edit_ms,
            );
        }
    }

    fn frame_state(&self, camera: &Camera, aspect: f32) -> FrameState {
        let cs = self.chunk_size as f32;

        // IMPORTANT: use the same reference point everywhere.
        // position() is what you use in render; use it here too.
        let cam_pos = camera.target;
        let cam_cx = (cam_pos.x / cs).floor() as i32;
        let cam_cz = (cam_pos.z / cs).floor() as i32;

        let (_, _, view_proj) = camera.matrices(aspect);
        let planes = extract_frustum_planes(view_proj);

        let r_render = self.view_radius_render as i32;
        let r_gen = self.view_radius_generate as i32;

        FrameState {
            cs,
            cam_cx,
            cam_cz,
            planes,
            r2_render: r_render * r_render,
            r2_gen: r_gen * r_gen,
        }
    }

    pub fn drain_finished_meshes(&mut self, device: &Device, queue: &Queue) {
        while let Ok(cpu) = self.workers.result_rx.try_recv() {
            let coord = (cpu.cx, cpu.cz);
            // Drop obsolete results
            if !self.workers.is_current_version(coord, cpu.version) {
                continue;
            }

            // Replace old chunk
            self.remove_chunk(coord);

            // Move out of cpu to avoid clone
            let mut vertices = cpu.vertices;
            let indices = cpu.indices;
            // start with worker's height grid (base, generated from terrain_gen)
            let mut height_grid = (*cpu.height_grid).clone();

            // If we have edits for this chunk, apply them to a new height grid (copy-on-write).
            if let Some(edit) = self.edited_chunks.get(&coord) {
                if !edit.deltas.is_empty() {
                    // apply sparse deltas to get a new grid
                    let new_grid = apply_sparse_deltas_to_height_grid(&height_grid, &edit.deltas);
                    height_grid = new_grid;

                    // regenerate vertex heights & normals from the new height grid
                    regenerate_vertices_from_height_grid(&mut vertices, &height_grid);
                }
            }

            // Upload new chunk to GPU
            let handle = self.arena.alloc_and_upload(
                device,
                queue,
                &vertices,
                &indices,
                &mut GeometryScratch::default(),
            );

            // Insert chunk with moved buffers and updated height grid (Arc)
            self.chunks.insert(
                coord,
                ChunkMeshLod {
                    step: cpu.step,
                    handle,
                    cpu_vertices: vertices,
                    cpu_indices: indices,
                    height_grid: Arc::new(height_grid),
                },
            );
        }
    }

    fn remove_chunk(&mut self, coord: (i32, i32)) {
        // remove pending job
        self.pending.remove(&coord);

        // remove GPU chunk
        if let Some(old) = self.chunks.remove(&coord) {
            self.arena.free::<Vertex>(old.handle);
        }

        // tell workers to forget it
        self.workers.forget_chunk(coord);
    }

    fn collect_visible(&self, frame: &FrameState) -> Vec<(i32, i32, i32)> {
        let mut visible = Vec::new();
        for &(dx, dz) in &self.spiral {
            let dist2 = dx * dx + dz * dz;
            if dist2 > frame.r2_render {
                continue;
            }

            let cx = frame.cam_cx + dx;
            let cz = frame.cam_cz + dz;

            let (min, max) = chunk_aabb_world(cx, cz, frame.cs);
            if aabb_in_frustum(&frame.planes, min, max) {
                visible.push((cx, cz, dist2));
            }
        }
        visible
    }

    fn compute_lod_for_visible(&mut self, visible: &[(i32, i32, i32)], r2_gen: i32) {
        self.lod_map.clear();

        for &(cx, cz, dist2) in visible {
            let step = if dist2 > r2_gen {
                lod_step_for_distance(r2_gen + 1)
            } else {
                lod_step_for_distance(dist2)
            };
            self.lod_map.insert((cx, cz), step);
        }

        // Smooth within visible set.
        for _ in 0..2 {
            let current = self.lod_map.clone();
            for &(cx, cz, _dist2) in visible {
                let s = *current.get(&(cx, cz)).unwrap_or(&1);

                // Neighbors default to s if not visible, to keep edges stable.
                let n0 = current.get(&(cx - 1, cz)).copied().unwrap_or(s);
                let n1 = current.get(&(cx + 1, cz)).copied().unwrap_or(s);
                let n2 = current.get(&(cx, cz - 1)).copied().unwrap_or(s);
                let n3 = current.get(&(cx, cz + 1)).copied().unwrap_or(s);

                self.lod_map
                    .insert((cx, cz), s.min(n0).min(n1).min(n2).min(n3));
            }
        }
    }

    fn dispatch_jobs_for_visible(&mut self, visible_sorted_near_to_far: &[(i32, i32, i32)]) {
        let mut close_batch = Vec::new();
        let mut far_batch = Vec::new();

        let mut close_jobs_sent = 0usize;
        let mut far_batches_sent = 0usize;

        for &(cx, cz, _dist2) in visible_sorted_near_to_far {
            if close_jobs_sent >= self.max_close_jobs_per_frame
                && far_batches_sent >= self.max_far_jobs_per_frame
            {
                break;
            }

            let coord = (cx, cz);
            let desired_step = *self.lod_map.get(&coord).unwrap_or(&1);

            // Already good?
            if let Some(ch) = self.chunks.get(&coord) {
                if ch.step == desired_step {
                    continue;
                }
            }
            // Already pending same step?
            if let Some((_ver, pending_step)) = self.pending.get(&coord).copied() {
                if pending_step == desired_step {
                    continue;
                }
            }

            // Neighbor LODs (fallback to step if neighbor not visible)
            let step = desired_step;
            let n_x_neg = *self.lod_map.get(&(cx - 1, cz)).unwrap_or(&step);
            let n_x_pos = *self.lod_map.get(&(cx + 1, cz)).unwrap_or(&step);
            let n_z_neg = *self.lod_map.get(&(cx, cz - 1)).unwrap_or(&step);
            let n_z_pos = *self.lod_map.get(&(cx, cz + 1)).unwrap_or(&step);

            if step <= 2 {
                if close_jobs_sent < self.max_close_jobs_per_frame {
                    let (version, version_atomic) = self.workers.new_version_for(coord);
                    self.pending.insert(coord, (version, step));

                    close_batch.push((
                        cx,
                        cz,
                        step,
                        n_x_neg,
                        n_x_pos,
                        n_z_neg,
                        n_z_pos,
                        version,
                        version_atomic,
                    ));

                    if close_batch.len() >= self.max_close_chunks_per_batch {
                        let job = ChunkJob {
                            chunks: std::mem::take(&mut close_batch),
                        };
                        let _ = self.workers.job_tx.send(job);
                        close_jobs_sent += 1;
                    }
                }
            } else {
                if far_batches_sent < self.max_far_jobs_per_frame {
                    let (version, version_atomic) = self.workers.new_version_for(coord);
                    self.pending.insert(coord, (version, step));

                    far_batch.push((
                        cx,
                        cz,
                        step,
                        n_x_neg,
                        n_x_pos,
                        n_z_neg,
                        n_z_pos,
                        version,
                        version_atomic,
                    ));

                    if far_batch.len() >= self.max_far_chunks_per_batch {
                        let job = ChunkJob {
                            chunks: std::mem::take(&mut far_batch),
                        };
                        let _ = self.workers.job_tx.send(job);
                        far_batches_sent += 1;
                    }
                }
            }
        }
        if !close_batch.is_empty() && close_jobs_sent < self.max_close_jobs_per_frame {
            let job = ChunkJob {
                chunks: close_batch,
            };
            let _ = self.workers.job_tx.send(job);
        }

        if !far_batch.is_empty() && far_batches_sent < self.max_far_jobs_per_frame {
            let job = ChunkJob { chunks: far_batch };
            let _ = self.workers.job_tx.send(job);
        }
    }

    fn lod_step_for_distance_with_hysteresis(&self, dist2: i32, current: usize) -> usize {
        let desired = lod_step_for_distance(dist2);

        if desired > current {
            desired
        } else if desired < current.saturating_sub(1) {
            desired
        } else {
            current
        }
    }

    fn unload_out_of_range(&mut self, frame: &FrameState, visible: &[(i32, i32, i32)]) {
        // Avoid thrash: unload outside render radius + generous margin.
        let margin = 12;
        let r = self.view_radius_render as i32 + margin;
        let r2 = r * r;

        // Build a set of currently-visible coords so we never unload them.
        let mut visible_set: HashSet<(i32, i32)> = HashSet::with_capacity(visible.len());
        for &(cx, cz, _) in visible {
            visible_set.insert((cx, cz));
        }

        let mut to_remove = Vec::new();
        for (&coord @ (cx, cz), _) in self.chunks.iter() {
            if visible_set.contains(&coord) {
                continue;
            }
            let dx = cx - frame.cam_cx;
            let dz = cz - frame.cam_cz;
            if dx * dx + dz * dz > r2 {
                to_remove.push(coord);
            }
        }

        for coord in to_remove {
            self.remove_chunk(coord);
        }
    }

    pub fn render<'a>(
        &'a self,
        pass: &mut RenderPass<'a>,
        pipelines: &'a Pipelines,
        camera: &'a Camera,
        aspect: f32,
        underwater: bool,
    ) {
        if !underwater {
            pass.set_pipeline(&pipelines.terrain_pipeline_above_water.pipeline);
        } else {
            pass.set_pipeline(&pipelines.terrain_pipeline_under_water.pipeline);
        }

        pass.set_bind_group(0, &pipelines.uniforms.bind_group, &[]);
        pass.set_bind_group(1, &pipelines.fog_uniforms.bind_group, &[]);
        pass.set_bind_group(2, &pipelines.pick_uniforms.bind_group, &[]);

        let (_, _, view_proj) = camera.matrices(aspect);

        let planes = extract_frustum_planes(view_proj);

        let cs = self.chunk_size as f32;
        let cam_pos = camera.target;
        let cam_cx = (cam_pos.x / cs).floor() as i32;
        let cam_cz = (cam_pos.z / cs).floor() as i32;

        let r = self.view_radius_render as i32;
        let r2 = r * r;

        // Bucket visible chunks by page index.
        let mut per_page: Vec<Vec<GpuChunkHandle>> = vec![Vec::new(); self.arena.pages.len()];

        for (&(cx, cz), chunk) in self.chunks.iter() {
            let dx = cx - cam_cx;
            let dz = cz - cam_cz;
            if dx * dx + dz * dz > r2 {
                continue;
            }

            let (min, max) = chunk_aabb_world(cx, cz, cs);
            if !aabb_in_frustum(&planes, min, max) {
                continue;
            }

            let p = chunk.handle.page as usize;
            if p < per_page.len() {
                per_page[p].push(chunk.handle);
            }
        }

        // Draw page by page (bind once per page).
        for (pi, handles) in per_page.iter().enumerate() {
            if handles.is_empty() {
                continue;
            }

            let page = &self.arena.pages[pi];
            pass.set_vertex_buffer(0, page.vertex_buf.slice(..));
            pass.set_index_buffer(page.index_buf.slice(..), IndexFormat::Uint32);

            for h in handles {
                let (start, count) = if underwater {
                    (h.first_index_under, h.index_count_under)
                } else {
                    (h.first_index_above, h.index_count_above)
                };
                pass.draw_indexed(start..start + count, h.base_vertex, 0..1);
            }
        }
    }

    pub fn pick_terrain_point(&mut self, ray: Ray) -> Option<(i32, i32, Vec3)> {
        let cs = self.chunk_size as f32;
        let eps = 1e-6 * cs;

        let mut cx = (ray.origin.x / cs).floor() as i32;
        let mut cz = (ray.origin.z / cs).floor() as i32;

        let step_x = ray.dir.x.signum() as i32;
        let step_z = ray.dir.z.signum() as i32;

        let next_x = (cx + if step_x > 0 { 1 } else { 0 }) as f32 * cs;
        let next_z = (cz + if step_z > 0 { 1 } else { 0 }) as f32 * cs;

        let mut t_max_x = if ray.dir.x.abs() < 1e-8 {
            f32::INFINITY
        } else {
            (next_x - ray.origin.x) / ray.dir.x
        };

        let mut t_max_z = if ray.dir.z.abs() < 1e-8 {
            f32::INFINITY
        } else {
            (next_z - ray.origin.z) / ray.dir.z
        };

        let t_delta_x = if ray.dir.x.abs() < 1e-8 {
            f32::INFINITY
        } else {
            cs / ray.dir.x.abs()
        };

        let t_delta_z = if ray.dir.z.abs() < 1e-8 {
            f32::INFINITY
        } else {
            cs / ray.dir.z.abs()
        };

        let mut t = 0.0;
        let max_t = 10_000.0;

        while t < max_t {
            let next_t = t_max_x.min(t_max_z);

            if let Some(chunk) = self.chunks.get(&(cx, cz)) {
                if let Some((_, pos)) =
                    raycast_chunk_heightgrid(ray, &chunk.height_grid, t, next_t + eps)
                {
                    self.last_picked = Some(PickedPoint {
                        pos,
                        radius: self.pick_radius_m,
                    });
                    return Some((cx, cz, pos));
                }
            }

            let tie = (t_max_x - t_max_z).abs() < 1e-7;

            if tie {
                cx += step_x;
                cz += step_z;
                t = t_max_x;
                t_max_x += t_delta_x;
                t_max_z += t_delta_z;
            } else if t_max_x < t_max_z {
                cx += step_x;
                t = t_max_x;
                t_max_x += t_delta_x;
            } else {
                cz += step_z;
                t = t_max_z;
                t_max_z += t_delta_z;
            }
        }

        self.last_picked = None;
        None
    }

    pub fn make_pick_uniforms(&self, queue: &Queue, pick_uniform_buffer: &Buffer) {
        let u = if let Some(p) = &self.last_picked {
            PickUniform {
                pos: p.pos.into(),
                radius: self.pick_radius_m,
                underwater: 1,
                _pad0: [0, 0, 0],
                color: [1.0, 0.0, 0.0],
                _pad1: 0.0,
            }
        } else {
            PickUniform {
                pos: [0.0; 3],
                radius: 0.0,
                underwater: 0,
                _pad0: [0, 0, 0],
                color: [1.0, 0.0, 0.0],
                _pad1: 0.0,
            }
        };

        queue.write_buffer(&pick_uniform_buffer, 0, bytemuck::bytes_of(&u));
    }
    /// Apply brush edits. Writes sparse deltas into EditedChunk.deltas (gx, gz, delta).
    /// No cloning of vertex arrays. Only operates on chunks that exist and step == 1.
    pub fn edit_terrain_with_brush<F, B>(
        &mut self,
        _device: &Device,
        _queue: &Queue,
        center: Vec3,
        radius: f32,
        strength: f32,
        _scratch: &mut GeometryScratch<Vertex>,
        time_system: &TimeSystem,
        end: bool,
    ) where
        F: Falloff + Sync + Send,
        B: BrushOp + Sync + Send,
    {
        let cs = self.chunk_size as f32;
        let (min_cx, max_cx, min_cz, max_cz) = affected_chunks(center, radius, cs);
        let r2 = radius * radius;

        let t0 = Instant::now();
        // 1) Determine which chunks in range exist and are editable (step == 1).
        let mut target_coords = Vec::new();
        for cx in min_cx..=max_cx {
            for cz in min_cz..=max_cz {
                target_coords.push((cx, cz));
            }
        }
        let affected_ms = t0.elapsed();
        let t0 = Instant::now();
        // 2) Ensure EditedChunk exists for each target chunk. Do not clone vertices.
        for coord in &target_coords {
            self.edited_chunks
                .entry(*coord)
                .or_insert_with(|| EditedChunk {
                    deltas: Vec::new(),
                    dirty: false,
                });
        }
        let ensure_existence_ms = t0.elapsed();
        let t0 = Instant::now();
        // 3) For each target chunk, compute sparse deltas by iterating grid cells.
        for coord in target_coords {
            // safe unwrap because we ensured existence
            let edited = self.edited_chunks.get_mut(&coord).unwrap();
            //edited.deltas.clear(); // replace previous in-flight deltas for this brush op
            edited.dirty = false;

            // get chunk height grid to sample current heights
            let chunk = match self.chunks.get(&coord) {
                Some(c) => c,
                None => continue,
            };
            let hg = &*chunk.height_grid;
            let nx = hg.nx;
            let nz = hg.nz;
            let stepf = hg.cell;
            let base_x = hg.base_x;
            let base_z = hg.base_z;

            // iterate over grid cells and compute weight & delta
            // compute bounding grid range to reduce iterations: convert circle in world coords to grid index range
            // local grid index bounds
            // find min/max gx/gz inside the radius
            // world -> local grid index
            let min_fx = (((center.x - radius) - base_x) / stepf).floor() as isize;
            let max_fx = (((center.x + radius) - base_x) / stepf).ceil() as isize;
            let min_fz = (((center.z - radius) - base_z) / stepf).floor() as isize;
            let max_fz = (((center.z + radius) - base_z) / stepf).ceil() as isize;

            let mut chunk_changed = false;

            for fx in min_fx..=max_fx {
                if fx < 0 || (fx as usize) >= nx {
                    continue;
                }
                let gx = fx as usize;
                for fz in min_fz..=max_fz {
                    if fz < 0 || (fz as usize) >= nz {
                        continue;
                    }
                    let gz = fz as usize;

                    // world position of this grid cell
                    let wx = base_x + (gx as f32) * stepf;
                    let wz = base_z + (gz as f32) * stepf;
                    let dx = wx - center.x;
                    let dz = wz - center.z;
                    let d2 = dx * dx + dz * dz;
                    if d2 >= r2 {
                        continue;
                    }

                    let w = F::weight(d2, r2);
                    if w <= 0.0001 {
                        continue;
                    }

                    // compute delta by invoking brush op on a copy of current height
                    let idx = gx * nz + gz;
                    let current_h = hg.heights[idx];
                    let mut new_h = current_h;
                    B::apply(&mut new_h, strength, w);
                    let delta = new_h - current_h;
                    if delta.abs() < f32::EPSILON {
                        continue;
                    }

                    edited.deltas.push((gx, gz, delta));
                    chunk_changed = true;
                }
            }

            if chunk_changed {
                edited.dirty = true;
            } else {
                edited.dirty = false;
            }
        }
        let compute_sparse_deltas_ms = t0.elapsed();
        let t0 = Instant::now();
        // 4) After brush complete, upload modified chunks (applies deltas copy-on-write).
        self.upload_edited_chunks(_device, _queue, _scratch, end);
        let upload_edited_ms = t0.elapsed();
        if (time_system.total_time / time_system.target_frametime as f64) as u32 % 12 == 0 {
            println!(
                "affected {:?} | ensure_existence {:?} | compute_sparse_deltas {:?} | upload_edited {:?}",
                affected_ms, ensure_existence_ms, compute_sparse_deltas_ms, upload_edited_ms,
            );
        }
    }

    /// Upload all edited chunks. This applies deltas copy-on-write to the height grid,
    /// regenerates vertex heights, normals, colors, uploads, and replaces chunk.height_grid Arc.
    fn upload_edited_chunks(
        &mut self,
        device: &Device,
        queue: &Queue,
        scratch: &mut GeometryScratch<Vertex>,
        end: bool,
    ) {
        let dirty_coords: Vec<(i32, i32)> = self
            .edited_chunks
            .iter()
            .filter(|(_, e)| e.dirty && !e.deltas.is_empty())
            .map(|(c, _)| *c)
            .collect();

        for coord in dirty_coords {
            let t_total = Instant::now();

            // ---- stage edited deltas ----
            let t0 = Instant::now();
            let deltas = {
                let edited = match self.edited_chunks.get_mut(&coord) {
                    Some(e) => e,
                    None => continue,
                };
                edited.dirty = false;
                if end {
                    edited.deltas.clone()
                } else {
                    //edited.deltas.clone()
                    std::mem::take(&mut edited.deltas)
                }
            };
            let stage_deltas_ms = t0.elapsed();

            // ---- stage base chunk data (immutable borrow only) ----
            let t0 = Instant::now();
            let (step, indices, mut grid, mut vertices) = match self.chunks.get(&coord) {
                Some(c) => (
                    c.step,
                    c.cpu_indices.clone(),
                    (*c.height_grid).clone(),
                    c.cpu_vertices.clone(),
                ),
                None => continue,
            };
            let stage_stage_ms = t0.elapsed();

            // ---- apply sparse deltas to height grid ----
            let t0 = Instant::now();
            for (gx, gz, delta) in deltas.iter() {
                if *gx < grid.nx && *gz < grid.nz {
                    let idx = gx * grid.nz + gz;
                    grid.heights[idx] += *delta;
                }
            }
            let stage_apply_ms = t0.elapsed();

            // ---- recompute patch min/max ----
            let t0 = Instant::now();
            let patch_cells = 8usize;
            let nx = grid.nx;
            let nz = grid.nz;
            let px = (nx - 1) / patch_cells;
            let pz = (nz - 1) / patch_cells;

            grid.patch_minmax.clear();
            grid.patch_minmax.reserve(px * pz);

            for px_i in 0..px {
                for pz_i in 0..pz {
                    let mut min_y = f32::INFINITY;
                    let mut max_y = -f32::INFINITY;

                    for lx in 0..=patch_cells {
                        for lz in 0..=patch_cells {
                            let gx = px_i * patch_cells + lx;
                            let gz = pz_i * patch_cells + lz;
                            if gx < nx && gz < nz {
                                let h = grid.heights[gx * nz + gz];
                                min_y = min_y.min(h);
                                max_y = max_y.max(h);
                            }
                        }
                    }

                    grid.patch_minmax.push((min_y, max_y));
                }
            }
            let stage_patch_ms = t0.elapsed();

            // ---- regenerate vertices (positions + normals + colors) ----
            let t0 = Instant::now();
            regenerate_vertices_from_height_grid_and_color(&mut vertices, &grid, &self.terrain_gen);
            let stage_vertices_ms = t0.elapsed();

            // ---- upload ----
            let t0 = Instant::now();
            let handle = self
                .arena
                .alloc_and_upload(device, queue, &vertices, &indices, scratch);
            let stage_upload_ms = t0.elapsed();

            // ---- replace chunk ----
            let t0 = Instant::now();
            self.remove_chunk(coord);
            self.chunks.insert(
                coord,
                ChunkMeshLod {
                    step,
                    handle,
                    cpu_vertices: vertices,
                    cpu_indices: indices,
                    height_grid: Arc::new(grid),
                },
            );
            let stage_replace_ms = t0.elapsed();

            // ---- print timing ----
            println!(
                "chunk {:?} | deltas {:?} | stage {:?} | apply {:?} | patch {:?} | vertices {:?} | upload {:?} | replace {:?} | total {:?}",
                coord,
                stage_deltas_ms,
                stage_stage_ms,
                stage_apply_ms,
                stage_patch_ms,
                stage_vertices_ms,
                stage_upload_ms,
                stage_replace_ms,
                t_total.elapsed(),
            );
        }
    }

    fn apply_persistent_edits_to_vertices(
        &self,
        height_grid: &ChunkHeightGrid,
        vertices: &mut [Vertex],
        edited: &EditedChunk,
    ) {
        if edited.deltas.is_empty() {
            return;
        }

        let verts_x = height_grid.nx;
        let verts_z = height_grid.nz;
        let stepf = height_grid.cell;

        // Track which vertex indices we changed so we only recompute normals locally.
        let mut changed_indices: HashSet<usize> = HashSet::new();

        for &(gx, gz, delta) in edited.deltas.iter() {
            if gx >= verts_x || gz >= verts_z {
                continue;
            }
            let idx = gx * verts_z + gz;
            vertices[idx].position[1] += delta;
            changed_indices.insert(idx);
        }

        if changed_indices.is_empty() {
            return;
        }

        // Recompute normals around changed vertices (3x3 neighborhood)
        let inv = 1.0 / stepf;
        let mut to_recompute: HashSet<usize> = HashSet::new();
        for &idx in &changed_indices {
            let gx = idx / verts_z;
            let gz = idx % verts_z;

            // collect neighborhood indices
            for nx in gx.saturating_sub(1)..=(gx + 1).min(verts_x - 1) {
                for nz in gz.saturating_sub(1)..=(gz + 1).min(verts_z - 1) {
                    to_recompute.insert(nx * verts_z + nz);
                }
            }
        }

        for &idx in &to_recompute {
            let gx = idx / verts_z;
            let gz = idx % verts_z;

            // neighbor heights from vertices (which we've already updated for changed cells)
            let h_l = if gx > 0 {
                vertices[(gx - 1) * verts_z + gz].position[1]
            } else {
                vertices[gx * verts_z + gz].position[1]
            };

            let h_r = if gx + 1 < verts_x {
                vertices[(gx + 1) * verts_z + gz].position[1]
            } else {
                vertices[gx * verts_z + gz].position[1]
            };

            let h_d = if gz > 0 {
                vertices[gx * verts_z + (gz - 1)].position[1]
            } else {
                vertices[gx * verts_z + gz].position[1]
            };

            let h_u = if gz + 1 < verts_z {
                vertices[gx * verts_z + (gz + 1)].position[1]
            } else {
                vertices[gx * verts_z + gz].position[1]
            };

            let dhdx = (h_r - h_l) * 0.5 * inv;
            let dhdz = (h_u - h_d) * 0.5 * inv;
            let n = Vec3::new(-dhdx, 1.0, -dhdz).normalize();

            // update normal and color
            let v_pos = vertices[idx].position;
            vertices[idx].normal = [n.x, n.y, n.z];
            vertices[idx].color = self.terrain_gen.color(
                v_pos[0],
                v_pos[2],
                v_pos[1],
                self.terrain_gen.moisture(v_pos[0], v_pos[2], v_pos[1]),
            );
        }
    }
}
/// Regenerate vertex heights, normals and colors from the provided height grid.
/// This helper uses TerrainGenerator to recompute color + moisture.
fn regenerate_vertices_from_height_grid_and_color(
    vertices: &mut [Vertex],
    height_grid: &ChunkHeightGrid,
    terrain_gen: &TerrainGenerator,
) {
    let verts_x = height_grid.nx;
    let verts_z = height_grid.nz;
    let stepf = height_grid.cell;

    if vertices.len() != verts_x * verts_z {
        return;
    }

    // update positions' y from grid
    for gx in 0..verts_x {
        for gz in 0..verts_z {
            let idx = gx * verts_z + gz;
            let h = height_grid.heights[idx];
            vertices[idx].position[1] = h;
        }
    }

    // recompute normals and colors
    let inv = 1.0 / stepf;
    for gx in 0..verts_x {
        for gz in 0..verts_z {
            let idx = gx * verts_z + gz;

            let h_l = if gx > 0 {
                vertices[(gx - 1) * verts_z + gz].position[1]
            } else {
                vertices[idx].position[1]
            };

            let h_r = if gx + 1 < verts_x {
                vertices[(gx + 1) * verts_z + gz].position[1]
            } else {
                vertices[idx].position[1]
            };

            let h_d = if gz > 0 {
                vertices[gx * verts_z + (gz - 1)].position[1]
            } else {
                vertices[idx].position[1]
            };

            let h_u = if gz + 1 < verts_z {
                vertices[gx * verts_z + (gz + 1)].position[1]
            } else {
                vertices[idx].position[1]
            };

            let dhdx = (h_r - h_l) * 0.5 * inv;
            let dhdz = (h_u - h_d) * 0.5 * inv;
            let n = Vec3::new(-dhdx, 1.0, -dhdz).normalize();
            vertices[idx].normal = [n.x, n.y, n.z];

            let v_pos = vertices[idx].position;
            //vertices[idx].color = terrain_gen.color(v_pos[0], v_pos[2], v_pos[1], terrain_gen.moisture(v_pos[0], v_pos[2], v_pos[1]));
        }
    }
}
fn chunk_intersects_circle(coord: (i32, i32), chunk_size: f32, center: Vec3, radius: f32) -> bool {
    let (cx, cz) = coord;
    let half = chunk_size * 0.5;
    let chunk_center_x = (cx as f32) * chunk_size + half;
    let chunk_center_z = (cz as f32) * chunk_size + half;

    // AABB min/max
    let minx = chunk_center_x - half;
    let maxx = chunk_center_x + half;
    let minz = chunk_center_z - half;
    let maxz = chunk_center_z + half;

    // find closest point on AABB to circle center
    let closest_x = clamp(center.x, minx, maxx);
    let closest_z = clamp(center.z, minz, maxz);

    let dx = center.x - closest_x;
    let dz = center.z - closest_z;
    dx * dx + dz * dz <= radius * radius
}

#[inline]
fn clamp(x: f32, a: f32, b: f32) -> f32 {
    if x < a {
        a
    } else if x > b {
        b
    } else {
        x
    }
}
fn recompute_normals_partial(
    vertices: &mut [Vertex],
    min_i: usize,
    max_i: usize,
    step: usize,
    terrain_gen: &TerrainGenerator,
) {
    let start = min_i.saturating_sub(1);
    let end = std::cmp::min(max_i + 1, vertices.len().saturating_sub(1));

    for i in start..=end {
        recompute_normal_for_vertex(vertices, i, step, terrain_gen);
    }
}

fn recompute_normal_for_vertex(
    vertices: &mut [Vertex],
    i: usize,
    step: usize,
    terrain_gen: &TerrainGenerator,
) {
    vertices[i].normal = [0.0, 1.0, 0.0];
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
    // glam is column-major; transpose to treat axes as rows.
    let mt = view_proj.transpose();
    let r0 = mt.x_axis; // row 0
    let r1 = mt.y_axis; // row 1
    let r2 = mt.z_axis; // row 2
    let r3 = mt.w_axis; // row 3

    // left, right, bottom, top, near, far
    let eqs = [
        r3 + r0,
        r3 - r0,
        r3 + r1,
        r3 - r1,
        r2,      // near: z >= 0   (wgpu/D3D/Vulkan)
        r3 - r2, // far:  z <= w
    ];

    eqs.map(|v| {
        let n = glam::Vec3::new(v.x, v.y, v.z);
        let inv_len = 1.0 / n.length();
        Plane {
            normal: n * inv_len,
            d: v.w * inv_len,
        }
    })
}

pub fn aabb_in_frustum(planes: &[Plane; 6], min: Vec3, max: Vec3) -> bool {
    // 0.5..2.0 meters is usually enough to remove micro-popping
    let margin = 1.0_f32;

    for p in planes {
        let vx = if p.normal.x >= 0.0 { max.x } else { min.x };
        let vy = if p.normal.y >= 0.0 { max.y } else { min.y };
        let vz = if p.normal.z >= 0.0 { max.z } else { min.z };

        if p.distance(glam::Vec3::new(vx, vy, vz)) < -margin {
            return false;
        }
    }
    true
}

const TERRAIN_MIN_Y: f32 = -4096.0;
const TERRAIN_MAX_Y: f32 = 4096.0;

#[inline]
fn chunk_aabb_world(cx: i32, cz: i32, chunk_size: f32) -> (Vec3, Vec3) {
    let x0 = cx as f32 * chunk_size;
    let z0 = cz as f32 * chunk_size;
    let min = glam::Vec3::new(x0, TERRAIN_MIN_Y, z0);
    let max = glam::Vec3::new(x0 + chunk_size, TERRAIN_MAX_Y, z0 + chunk_size);
    (min, max)
}
#[derive(Default)]
pub struct FrameTimings {
    pub drain_ms: f32,
    pub collect_visible_ms: f32,
    pub sort_visible_ms: f32,
    pub lod_ms: f32,
    pub dispatch_ms: f32,
    pub unload_ms: f32,
    pub edit_ms: f32,
    pub total_ms: f32,
}
