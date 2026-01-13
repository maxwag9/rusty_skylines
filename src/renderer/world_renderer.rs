use crate::components::camera::Camera;
use crate::data::Settings;
use crate::mouse_ray::*;
use crate::paths::data_dir;
use crate::renderer::benchmark::{Benchmark, ChunkJobConfig};
use crate::renderer::mesh_arena::{GeometryScratch, TerrainMeshArena};
use crate::resources::{InputState, TimeSystem};
use crate::terrain::chunk_builder::*;
use crate::terrain::roads::road_mesh_manager::{ChunkId, chunk_coord_to_id};
use crate::terrain::terrain::{TerrainGenerator, TerrainParams};
use crate::terrain::terrain_editing::*;
use crate::terrain::threads::{ChunkJob, ChunkWorkerPool};
use crate::ui::vertex::Vertex;
use glam::Vec3;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use wgpu::{Buffer, Device, IndexFormat, Queue, RenderPass};

pub struct PickedPoint {
    pub pos: Vec3,
    pub chunk: VisibleChunk,
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
pub struct ChunkCoords {
    x: i32,
    z: i32, // Y IS UP/DOWN LIKE IN MINECRAFT NOT CRINGE Z LIKE BLENDER ETC.
    dist2: i32,
}
pub struct VisibleChunk {
    pub coords: ChunkCoords,
    pub id: ChunkId,
}
pub struct TerrainRenderer {
    pub arena: TerrainMeshArena,

    pub chunks: HashMap<(i32, i32), ChunkMeshLod>,
    pub pending: HashMap<(i32, i32), (u64, usize)>,
    pub terrain_editor: TerrainEditor,

    pub terrain_gen: TerrainGenerator,
    pub chunk_size: usize,
    pub view_radius_generate: usize,
    pub view_radius_render: usize,

    pub workers: ChunkWorkerPool,
    pub max_close_jobs_per_frame: usize,
    pub max_close_chunks_per_batch: usize,
    pub max_far_chunks_per_batch: usize,
    pub max_far_jobs_per_frame: usize,

    pub spiral: Vec<(i32, i32)>,
    pub lod_map: HashMap<(i32, i32), usize>,

    pub pick_radius_m: f32,
    pub last_picked: Option<PickedPoint>,

    pub benchmark: Benchmark,
    pub frame_timings: FrameTimings,
    pub job_config: ChunkJobConfig,
    pub visible: Vec<VisibleChunk>,
    pub terrain_editing_enabled: bool,
}

impl TerrainRenderer {
    pub fn new(device: &Device, settings: &Settings) -> Self {
        let mut terrain_params = TerrainParams::default();
        terrain_params.seed = 144;
        let terrain_gen = TerrainGenerator::new(terrain_params);

        let chunk_size = 64;
        let view_radius_render = 128;
        let view_radius_generate = 64;

        let arena = TerrainMeshArena::new(
            device,
            256 * 1024 * 1024, // vertex bytes per page
            128 * 1024 * 1024, // index bytes per page
        );

        let threads = num_cpus::get_physical().saturating_sub(1).max(1);
        println!("Using {} chunk workers", threads);

        let workers = ChunkWorkerPool::new(threads, terrain_gen.clone(), chunk_size as u32);
        let terrain_editor: TerrainEditor;
        if settings.show_world {
            terrain_editor = match TerrainEditor::load_edits(data_dir("edited_chunks")) {
                Ok(te) => {
                    println!("World loaded");
                    te
                }
                Err(e) => {
                    eprintln!("Failed to load World: {e}");
                    TerrainEditor::default()
                }
            };
        } else {
            terrain_editor = TerrainEditor::default();
        }

        Self {
            arena,
            chunks: HashMap::new(),
            pending: HashMap::new(),
            terrain_editor,

            terrain_gen,
            chunk_size,
            view_radius_generate,
            view_radius_render,

            workers,
            job_config: ChunkJobConfig::default(),
            max_close_jobs_per_frame: 1,
            max_close_chunks_per_batch: 2,
            max_far_chunks_per_batch: 100,
            max_far_jobs_per_frame: 1,

            spiral: generate_spiral_offsets(view_radius_generate as i32),
            lod_map: HashMap::new(),
            pick_radius_m: 1.0,
            last_picked: None,

            benchmark: Benchmark::default(),
            frame_timings: FrameTimings::default(),
            visible: vec![],
            terrain_editing_enabled: false,
        }
    }

    pub fn update(
        &mut self,
        device: &Device,
        queue: &Queue,
        camera: &mut Camera,
        aspect: f32,
        settings: &Settings,
        input_state: &mut InputState,
        _time_system: &TimeSystem,
    ) {
        let t_frame = Instant::now();

        if settings.world_generation_benchmark_mode {
            self.benchmark
                .run(camera, self.chunks.len(), &mut self.job_config, || {
                    self.chunks.clear();
                    self.pending.clear();
                    self.lod_map.clear();
                });
        }

        let frame = self.frame_state(camera, aspect);

        let t0 = Instant::now();
        self.drain_finished_meshes(device, queue);
        self.frame_timings.drain_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let t0 = Instant::now();
        self.visible = self.collect_visible(&frame);
        self.frame_timings.collect_visible_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let t0 = Instant::now();
        self.visible.sort_unstable_by_key(|v| v.coords.dist2);
        self.frame_timings.sort_visible_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let t0 = Instant::now();
        self.compute_lod_for_visible(frame.r2_gen);
        self.frame_timings.lod_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let t0 = Instant::now();
        self.dispatch_jobs_for_visible();
        self.frame_timings.dispatch_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let t0 = Instant::now();
        self.unload_out_of_range(&frame);
        self.frame_timings.unload_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let t0 = Instant::now();
        if settings.show_world && self.terrain_editing_enabled {
            self.handle_terrain_editing(device, queue, input_state);
        }

        self.frame_timings.edit_ms = t0.elapsed().as_secs_f32() * 1000.0;

        self.frame_timings.total_ms = t_frame.elapsed().as_secs_f32() * 1000.0;
    }

    fn handle_terrain_editing(
        &mut self,
        device: &Device,
        queue: &Queue,
        input_state: &mut InputState,
    ) {
        let editing =
            input_state.action_down("Edit Terrain +") || input_state.action_down("Edit Terrain -");
        let released = input_state.action_released("Edit Terrain +")
            || input_state.action_released("Edit Terrain -");

        if !editing && !released {
            return;
        }

        let Some(picked) = &self.last_picked else {
            return;
        };
        let pos = picked.pos;

        let strength = if input_state.action_down("Edit Terrain +") {
            1.0
        } else {
            -1.0
        };

        self.terrain_editor.apply_brush::<SmoothFalloff, Raise>(
            pos,
            self.pick_radius_m,
            strength,
            self.chunk_size as f32,
            &self.chunks,
        );

        let mut scratch = GeometryScratch::default();
        let freed_handles = self.terrain_editor.upload_dirty_chunks(
            device,
            queue,
            &mut self.arena,
            &mut self.chunks,
            &self.terrain_gen,
            &mut scratch,
            released,
        );

        for handle in freed_handles {
            self.arena.free::<Vertex>(handle);
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

            if !self.workers.is_current_version(coord, cpu.version) {
                continue;
            }

            self.remove_chunk(coord);

            let mut vertices = cpu.vertices;
            let indices = cpu.indices;
            let mut height_grid = (*cpu.height_grid).clone();
            let step = cpu.step;

            // Check if this chunk has accumulated edits
            let has_edits = self
                .terrain_editor
                .edited_chunks
                .get(&coord)
                .map_or(false, |e| !e.accumulated_deltas.is_empty());

            // Check if any neighbor is STRICTLY coarser (higher step = needs stitching on that edge)
            let has_coarser_neighbor = false;

            // Only apply delta/stitching logic if there are edits OR coarser neighbors exist
            if has_edits || has_coarser_neighbor {
                height_grid = apply_accumulated_deltas_with_stitching(
                    &height_grid,
                    coord,
                    &self.terrain_editor.edited_chunks,
                    &self.chunks,
                    step,
                );

                recompute_patch_minmax(&mut height_grid);

                // Gather neighbor edge heights for proper normal calculation at boundaries
                let neighbor_edges = gather_neighbor_edge_heights(
                    coord,
                    &height_grid,
                    &self.chunks,
                    &self.terrain_editor.edited_chunks,
                );
                regenerate_vertices_from_height_grid(
                    &mut vertices,
                    &height_grid,
                    &self.terrain_gen,
                    Some(&neighbor_edges),
                    false,
                );
            }

            let handle = self.arena.alloc_and_upload(
                device,
                queue,
                &vertices,
                &indices,
                &mut GeometryScratch::default(),
            );

            self.chunks.insert(
                coord,
                ChunkMeshLod {
                    step,
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

    fn collect_visible(&self, frame: &FrameState) -> Vec<VisibleChunk> {
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
                visible.push(VisibleChunk {
                    coords: ChunkCoords {
                        x: cx,
                        z: cz,
                        dist2,
                    },
                    id: chunk_coord_to_id(cx, cz),
                });
            }
        }
        visible
    }

    fn compute_lod_for_visible(&mut self, r2_gen: i32) {
        self.lod_map.clear();

        for v in self.visible.iter() {
            let step = if v.coords.dist2 > r2_gen {
                lod_step_for_distance(r2_gen + 1)
            } else {
                lod_step_for_distance(v.coords.dist2)
            };
            self.lod_map.insert((v.coords.x, v.coords.z), step);
        }

        // Smooth within visible set.
        for _ in 0..2 {
            let current = self.lod_map.clone();
            for v in self.visible.iter() {
                let s = *current.get(&(v.coords.x, v.coords.z)).unwrap_or(&1);

                // Neighbors default to s if not visible, to keep edges stable.
                let n0 = current
                    .get(&(v.coords.x - 1, v.coords.z))
                    .copied()
                    .unwrap_or(s);
                let n1 = current
                    .get(&(v.coords.x + 1, v.coords.z))
                    .copied()
                    .unwrap_or(s);
                let n2 = current
                    .get(&(v.coords.x, v.coords.z - 1))
                    .copied()
                    .unwrap_or(s);
                let n3 = current
                    .get(&(v.coords.x, v.coords.z + 1))
                    .copied()
                    .unwrap_or(s);

                self.lod_map
                    .insert((v.coords.x, v.coords.z), s.min(n0).min(n1).min(n2).min(n3));
            }
        }
    }

    fn dispatch_jobs_for_visible(&mut self) {
        let mut close_batch = Vec::new();
        let mut far_batch = Vec::new();

        let mut close_jobs_sent = 0usize;
        let mut far_batches_sent = 0usize;

        for v in self.visible.iter() {
            let cx = v.coords.x;
            let cz = v.coords.z;
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

    fn unload_out_of_range(&mut self, frame: &FrameState) {
        // Avoid thrash: unload outside render radius + generous margin.
        let margin = 12;
        let r = self.view_radius_render as i32 + margin;
        let r2 = r * r;

        // Build a set of currently-visible coords so we never unload them.
        let mut visible_set: HashSet<(i32, i32)> = HashSet::with_capacity(self.visible.len());
        for v in self.visible.iter() {
            visible_set.insert((v.coords.x, v.coords.z));
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

    pub fn render(&self, pass: &mut RenderPass, camera: &Camera, aspect: f32, underwater: bool) {
        let (_, _, view_proj) = camera.matrices(aspect);
        let planes = extract_frustum_planes(view_proj);

        let cs = self.chunk_size as f32;
        let cam_pos = camera.target;
        let cam_cx = (cam_pos.x / cs).floor() as i32;
        let cam_cz = (cam_pos.z / cs).floor() as i32;

        let r = self.view_radius_render as i32;
        let r2 = r * r;

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
                    let chunk = VisibleChunk {
                        coords: ChunkCoords {
                            x: cx,
                            z: cz,
                            dist2: 0,
                        },
                        id: chunk_coord_to_id(cx, cz),
                    };
                    self.last_picked = Some(PickedPoint { pos, chunk });
                    return Some((cx, cz, pos));
                }
            }

            dda_advance(
                &mut cx,
                &mut cz,
                step_x,
                step_z,
                &mut t,
                &mut t_max_x,
                &mut t_max_z,
                t_delta_x,
                t_delta_z,
            );
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

        queue.write_buffer(pick_uniform_buffer, 0, bytemuck::bytes_of(&u));
    }
    pub fn get_height_at(&self, pos: [f32; 2]) -> f32 {
        let chunk_size = self.chunk_size as f32;

        let [cx, cz] = pos_to_chunk_coord(pos, chunk_size);

        let chunk = match self.chunks.get(&(cx, cz)) {
            Some(c) => c,
            None => return 0.0,
        };

        let [lx, lz] = pos_to_chunk_local(pos, chunk_size);

        height_bilinear(&chunk.height_grid, lx, lz)
    }
}

fn pos_to_chunk_coord(pos: [f32; 2], chunk_size: f32) -> [i32; 2] {
    [
        (pos[0] / chunk_size).floor() as i32,
        (pos[1] / chunk_size).floor() as i32,
    ]
}
fn pos_to_chunk_local(pos: [f32; 2], chunk_size: f32) -> [f32; 2] {
    [
        pos[0] - (pos[0] / chunk_size).floor() * chunk_size,
        pos[1] - (pos[1] / chunk_size).floor() * chunk_size,
    ]
}

#[derive(Clone, Copy)]
pub struct Plane {
    pub normal: Vec3,
    pub d: f32,
}

impl Plane {
    pub fn distance(&self, p: Vec3) -> f32 {
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
        let n = Vec3::new(v.x, v.y, v.z);
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

        if p.distance(Vec3::new(vx, vy, vz)) < -margin {
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
    let min = Vec3::new(x0, TERRAIN_MIN_Y, z0);
    let max = Vec3::new(x0 + chunk_size, TERRAIN_MAX_Y, z0 + chunk_size);
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
