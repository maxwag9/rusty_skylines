use crate::components::camera::Camera;
use crate::data::{LodCenterType, Settings};
use crate::mouse_ray::*;
use crate::paths::data_dir;
use crate::positions::*;
use crate::renderer::benchmark::{Benchmark, ChunkJobConfig};
use crate::renderer::mesh_arena::{GeometryScratch, TerrainMeshArena};
use crate::resources::TimeSystem;
use crate::terrain::chunk_builder::*;
use crate::terrain::roads::road_mesh_manager::{ChunkId, chunk_coord_to_id};
use crate::terrain::roads::road_structs::RoadType;
use crate::terrain::terrain::{TerrainGenerator, TerrainParams};
use crate::terrain::terrain_editing::*;
use crate::terrain::threads::{ChunkJob, ChunkWorkerPool};
use crate::ui::input::InputState;
use crate::ui::vertex::Vertex;
use glam::{Mat4, Vec3};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use wgpu::{Buffer, Device, IndexFormat, Queue, RenderPass};

pub struct ChunkCoords {
    chunk_coord: ChunkCoord, // Y IS UP/DOWN LIKE IN MINECRAFT NOT CRINGE Z LIKE BLENDER ETC.
    dist3: i32,
}
pub struct VisibleChunk {
    pub coords: ChunkCoords,
    pub id: ChunkId,
}
pub struct PickedPoint {
    pub pos: WorldPos,
    pub chunk: VisibleChunk,
}

#[derive(Clone, Copy)]
struct FrameState {
    cs: ChunkSize,
    cam_pos: WorldPos,
    planes: [Plane; 6],
    r2_render: i32,
    r2_gen: i32,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CursorMode {
    None,
    Roads(RoadType),
    TerrainEditing,
}
#[derive(Debug)]
pub struct Cursor {
    pub mode: CursorMode,
}

impl Cursor {
    pub fn new() -> Self {
        Self {
            mode: CursorMode::Roads(RoadType::default()),
        }
    }
}

pub struct TerrainRenderer {
    pub cursor: Cursor,
    pub arena: TerrainMeshArena,

    pub chunks: HashMap<ChunkCoord, ChunkMeshLod>,
    pub pending: HashMap<ChunkCoord, (u64, LodStep)>,
    pub terrain_editor: TerrainEditor,

    pub terrain_gen: TerrainGenerator,
    pub chunk_size: ChunkSize,
    pub view_radius_generate: usize,
    pub view_radius_render: usize,

    pub workers: ChunkWorkerPool,
    pub max_close_jobs_per_frame: usize,
    pub max_close_chunks_per_batch: usize,
    pub max_far_chunks_per_batch: usize,
    pub max_far_jobs_per_frame: usize,

    pub spiral: Vec<ChunkCoord>,
    pub lod_map: HashMap<ChunkCoord, LodStep>,

    pub pick_radius_m: f32,
    pub last_picked: Option<PickedPoint>,

    pub benchmark: Benchmark,
    pub frame_timings: FrameTimings,
    pub job_config: ChunkJobConfig,
    pub visible: Vec<VisibleChunk>,
}
const VERTEX_SIZE_BYTES: usize = size_of::<Vertex>();
impl TerrainRenderer {
    pub fn new(device: &Device, settings: &Settings) -> Self {
        let mut terrain_params = TerrainParams::default();
        terrain_params.seed = 144;
        let terrain_gen = TerrainGenerator::new(terrain_params);

        let chunk_size: ChunkSize = settings.chunk_size;
        let view_radius_render = 64;
        let view_radius_generate = 32;

        let arena = TerrainMeshArena::new(
            device,
            256 * 1024 * 1024, // vertex bytes per page
            128 * 1024 * 1024, // index bytes per page
        );

        let threads = num_cpus::get_physical().saturating_sub(1).max(1);
        println!("Using {} chunk workers", threads);

        let workers = ChunkWorkerPool::new(threads, terrain_gen.clone(), chunk_size);
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
            cursor: Cursor::new(),
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

        let frame = self.frame_state(settings, camera, aspect);

        let t0 = Instant::now();
        self.drain_finished_meshes(device, queue);
        self.frame_timings.drain_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let t0 = Instant::now();
        self.visible = self.collect_visible(camera, &frame);
        self.frame_timings.collect_visible_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let t0 = Instant::now();
        self.visible.sort_unstable_by_key(|v| v.coords.dist3);
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
        if settings.show_world {
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
        match self.cursor.mode {
            CursorMode::TerrainEditing => {} // just continue lol
            _ => {
                return;
            } //Fuck off, this is not your mode!!
        };
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
            self.chunk_size,
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

    fn frame_state(&self, settings: &Settings, camera: &Camera, aspect: f32) -> FrameState {
        let cs = self.chunk_size;

        let cam_pos = match settings.lod_center {
            LodCenterType::Eye => camera.eye_world(),
            LodCenterType::Target => camera.target,
        };

        let (_, _, view_proj) = camera.matrices(aspect, settings);
        let planes = extract_frustum_planes(view_proj);

        let r_render = self.view_radius_render as i32;
        let r_gen = self.view_radius_generate as i32;

        FrameState {
            cs,
            cam_pos,
            planes,
            r2_render: r_render * r_render,
            r2_gen: r_gen * r_gen,
        }
    }

    pub fn drain_finished_meshes(&mut self, device: &Device, queue: &Queue) {
        while let Ok(cpu) = self.workers.result_rx.try_recv() {
            let coord = cpu.chunk_coord;

            if !self.workers.is_current_version(coord, cpu.version) {
                continue;
            }

            self.remove_chunk(coord);

            let mut vertices = cpu.vertices;
            let mut indices = cpu.indices;
            let mut height_grid = (*cpu.height_grid).clone();
            let step = cpu.step;

            let has_edits = self
                .terrain_editor
                .edited_chunks
                .get(&coord)
                .map_or(false, |e| !e.accumulated_deltas.is_empty());

            let has_coarser_neighbor = false;

            if has_edits || has_coarser_neighbor {
                height_grid = apply_accumulated_deltas_with_stitching(
                    &height_grid,
                    coord,
                    &self.terrain_editor.edited_chunks,
                    &self.chunks,
                    step,
                );

                recompute_patch_minmax(&mut height_grid);

                let expected_verts = height_grid.nx * height_grid.nz;

                if vertices.len() == expected_verts {
                    // Fast path: simple grid layout, update in-place
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
                } else {
                    // Greedy-meshed chunk: rebuild with simple grid layout
                    (vertices, indices) =
                        build_simple_grid_mesh(&height_grid, coord, &self.terrain_gen);
                }
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
    fn remove_chunk(&mut self, coord: ChunkCoord) {
        // remove pending job
        self.pending.remove(&coord);
        // remove GPU chunk
        if let Some(old) = self.chunks.remove(&coord) {
            self.arena.free::<Vertex>(old.handle);
        }
        // tell workers to forget it
        self.workers.forget_chunk(coord);
    }

    fn collect_visible(&self, camera: &Camera, frame: &FrameState) -> Vec<VisibleChunk> {
        let mut visible = Vec::new();
        for &chunk_coord in &self.spiral {
            let (dx, dz) = (chunk_coord.x, chunk_coord.z);
            //let dy =
            let dist3 = dx * dx + dz * dz;
            if dist3 > frame.r2_render {
                continue;
            }
            let cx = frame.cam_pos.chunk.x + dx;
            let cz = frame.cam_pos.chunk.z + dz;

            let (min, max) = chunk_aabb_render(cx, cz, frame.cs, camera.eye_world());
            if aabb_in_frustum(&frame.planes, min, max) {
                visible.push(VisibleChunk {
                    coords: ChunkCoords {
                        chunk_coord: ChunkCoord::new(cx, cz),
                        dist3,
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
            let step = if v.coords.dist3 > r2_gen {
                lod_step_for_distance(r2_gen + 1)
            } else {
                lod_step_for_distance(v.coords.dist3)
            };
            self.lod_map.insert(v.coords.chunk_coord, step);
        }

        // Smooth to prevent T-junctions: no chunk should be >2x finer than neighbors
        for _ in 0..2 {
            let current = self.lod_map.clone();
            for v in self.visible.iter() {
                let s = *current.get(&v.coords.chunk_coord).unwrap_or(&32);

                let n0 = current
                    .get(&ChunkCoord::new(
                        v.coords.chunk_coord.x - 1,
                        v.coords.chunk_coord.z,
                    ))
                    .copied()
                    .unwrap_or(s);
                let n1 = current
                    .get(&ChunkCoord::new(
                        v.coords.chunk_coord.x + 1,
                        v.coords.chunk_coord.z,
                    ))
                    .copied()
                    .unwrap_or(s);
                let n2 = current
                    .get(&ChunkCoord::new(
                        v.coords.chunk_coord.x,
                        v.coords.chunk_coord.z - 1,
                    ))
                    .copied()
                    .unwrap_or(s);
                let n3 = current
                    .get(&ChunkCoord::new(
                        v.coords.chunk_coord.x,
                        v.coords.chunk_coord.z + 1,
                    ))
                    .copied()
                    .unwrap_or(s);

                // FIX: Prevent being more than 2x finer than the coarsest neighbor
                let max_neighbor = n0.max(n1).max(n2).max(n3);
                let min_allowed_step = (max_neighbor / 2).max(1);
                self.lod_map
                    .insert(v.coords.chunk_coord, s.max(min_allowed_step));
            }
        }
    }

    fn dispatch_jobs_for_visible(&mut self) {
        let mut close_batch = Vec::new();
        let mut far_batch = Vec::new();

        let mut close_jobs_sent = 0usize;
        let mut far_batches_sent = 0usize;

        for v in self.visible.iter() {
            let cx = v.coords.chunk_coord.x;
            let cz = v.coords.chunk_coord.z;
            if close_jobs_sent >= self.max_close_jobs_per_frame
                && far_batches_sent >= self.max_far_jobs_per_frame
            {
                break;
            }

            let coord = v.coords.chunk_coord;
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
            let has_edits = self
                .terrain_editor
                .edited_chunks
                .get(&coord)
                .map_or(false, |e| !e.accumulated_deltas.is_empty());
            // Neighbor LODs (fallback to step if neighbor not visible)
            let step = desired_step;
            let n_x_neg = *self
                .lod_map
                .get(&ChunkCoord::new(cx - 1, cz))
                .unwrap_or(&step);
            let n_x_pos = *self
                .lod_map
                .get(&ChunkCoord::new(cx + 1, cz))
                .unwrap_or(&step);
            let n_z_neg = *self
                .lod_map
                .get(&ChunkCoord::new(cx, cz - 1))
                .unwrap_or(&step);
            let n_z_pos = *self
                .lod_map
                .get(&ChunkCoord::new(cx, cz + 1))
                .unwrap_or(&step);

            if step <= 2 {
                if close_jobs_sent < self.max_close_jobs_per_frame {
                    let (version, version_atomic) = self.workers.new_version_for(coord);
                    self.pending.insert(coord, (version, step));

                    close_batch.push((
                        coord,
                        step,
                        n_x_neg,
                        n_x_pos,
                        n_z_neg,
                        n_z_pos,
                        version,
                        version_atomic,
                        has_edits,
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
                        coord,
                        step,
                        n_x_neg,
                        n_x_pos,
                        n_z_neg,
                        n_z_pos,
                        version,
                        version_atomic,
                        has_edits,
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
        let margin = 4;
        let r = self.view_radius_render as i32 + margin;
        let r2 = r * r;

        // Build a set of currently-visible coords so we never unload them.
        let mut visible_set: HashSet<ChunkCoord> = HashSet::with_capacity(self.visible.len());
        for v in self.visible.iter() {
            visible_set.insert(v.coords.chunk_coord);
        }

        let mut to_remove = Vec::new();
        for (&coord @ chunk_coord, _) in self.chunks.iter() {
            if visible_set.contains(&coord) {
                continue;
            }
            let dx = chunk_coord.x - frame.cam_pos.chunk.x;
            let dz = chunk_coord.z - frame.cam_pos.chunk.z;
            if dx * dx + dz * dz > r2 {
                to_remove.push(coord);
            }
        }

        for coord in to_remove {
            self.remove_chunk(coord);
        }
    }

    pub fn render(
        &self,
        pass: &mut RenderPass,
        camera: &Camera,
        aspect: f32,
        settings: &Settings,
        underwater: bool,
    ) {
        let (_, _, view_proj) = camera.matrices(aspect, settings);
        let planes = extract_frustum_planes(view_proj);

        let cs = self.chunk_size;
        let target_pos = camera.target;
        let target_cx = target_pos.chunk.x;
        let target_cz = target_pos.chunk.z;

        let r = self.view_radius_render as i32;
        let r2 = r * r;

        let mut per_page: Vec<Vec<GpuChunkHandle>> = vec![Vec::new(); self.arena.pages.len()];

        for (&chunk_coord, chunk) in self.chunks.iter() {
            let dx = chunk_coord.x - target_cx;
            let dz = chunk_coord.z - target_cz;
            if dx * dx + dz * dz > r2 {
                continue;
            }

            let (min, max) =
                chunk_aabb_render(chunk_coord.x, chunk_coord.z, cs, camera.eye_world());
            if !aabb_in_frustum(&planes, min, max) {
                continue;
            }

            let p = chunk.handle.page as usize;
            if p < per_page.len() {
                per_page[p].push(chunk.handle);
            }
        }
        let mut total_indices: usize = 0;
        let mut total_vertices: usize = 0;

        for handles in &per_page {
            for h in handles {
                let count = if underwater {
                    h.index_count_under as usize
                } else {
                    h.index_count_above as usize
                };

                total_indices += count;

                // Worst-case but correct for GPU memory usage
                // base_vertex + max index touched
                total_vertices += count; // triangles already expanded
            }
        }

        let index_bytes = total_indices * 4;
        let vertex_bytes = total_vertices * VERTEX_SIZE_BYTES;
        let _total_bytes = index_bytes + vertex_bytes;

        // println!(
        //     "render: {} vertices, {} indices, {:.2} MB total",
        //     total_vertices,
        //     total_indices,
        //     _total_bytes as f32 / 1024.0 / 1024.0
        // );

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

    /// Pick a terrain point by casting a ray through loaded chunks.
    /// Uses WorldRay for maximum precision at any distance.
    pub fn pick_terrain_point(&mut self, ray: WorldRay) -> Option<WorldPos> {
        let cs = self.chunk_size;
        let cs_f32 = cs as f32;
        let eps = 1e-6 * cs_f32;

        // Start DDA from ray origin's chunk (exact, no precision loss)
        let mut cx = ray.origin.chunk.x;
        let mut cz = ray.origin.chunk.z;

        let step_x = if ray.dir.x >= 0.0 { 1i32 } else { -1i32 };
        let step_z = if ray.dir.z >= 0.0 { 1i32 } else { -1i32 };

        // Compute t to first chunk boundaries using integer chunk math
        let first_boundary_x = if step_x > 0 { cx + 1 } else { cx };
        let first_boundary_z = if step_z > 0 { cz + 1 } else { cz };

        let mut t_max_x = ray.t_to_chunk_x_boundary(first_boundary_x, cs);
        let mut t_max_z = ray.t_to_chunk_z_boundary(first_boundary_z, cs);

        // Handle ray starting exactly on boundary moving negative
        if t_max_x <= 0.0 && step_x < 0 {
            cx -= 1;
            t_max_x = ray.t_to_chunk_x_boundary(cx, cs);
        }
        if t_max_z <= 0.0 && step_z < 0 {
            cz -= 1;
            t_max_z = ray.t_to_chunk_z_boundary(cz, cs);
        }

        let t_delta_x = if ray.dir.x.abs() < 1e-12 {
            f32::INFINITY
        } else {
            cs_f32 / ray.dir.x.abs()
        };

        let t_delta_z = if ray.dir.z.abs() < 1e-12 {
            f32::INFINITY
        } else {
            cs_f32 / ray.dir.z.abs()
        };

        let mut t = 0.0f32;
        let max_t = 100_000.0f32;

        while t < max_t {
            let next_t = t_max_x.min(t_max_z).min(max_t);
            let chunk_coord = ChunkCoord::new(cx, cz);

            if let Some(chunk) = self.chunks.get(&chunk_coord) {
                if let Some((_, hit_pos)) =
                    raycast_chunk_heightgrid(&ray, &chunk.height_grid, t, next_t + eps)
                {
                    // Update last picked info
                    let visible_chunk = VisibleChunk {
                        coords: ChunkCoords {
                            chunk_coord,
                            dist3: 0,
                        },
                        id: chunk_coord_to_id(cx, cz),
                    };

                    self.last_picked = Some(PickedPoint {
                        pos: hit_pos,
                        chunk: visible_chunk,
                    });

                    return Some(hit_pos);
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

    pub fn make_pick_uniforms(
        &self,
        queue: &Queue,
        pick_uniform_buffer: &Buffer,
        camera: &Camera,
        chunk_size: ChunkSize,
    ) {
        let u = if let Some(p) = &self.last_picked {
            PickUniform {
                pos: p
                    .pos
                    .to_render_pos(camera.eye_world(), chunk_size)
                    .to_array(),
                radius: self.pick_radius_m,
                underwater: 1,
                _pad0: [0, 0, 0],
                color: [0.01, 0.01, 0.01],
                _pad1: 0.0,
            }
        } else {
            PickUniform {
                pos: [0.0; 3],
                radius: 0.0,
                underwater: 0,
                _pad0: [0, 0, 0],
                color: [0.01, 0.01, 0.01],
                _pad1: 0.0,
            }
        };

        queue.write_buffer(pick_uniform_buffer, 0, bytemuck::bytes_of(&u));
    }
    pub fn get_height_at(&self, pos: WorldPos) -> f32 {
        let chunk = match self.chunks.get(&pos.chunk) {
            Some(c) => c,
            None => return self.terrain_gen.height(&pos, self.chunk_size),
        };

        height_bilinear_world(&chunk.height_grid, pos)
    }
}

#[derive(Clone, Copy)]
pub struct Plane {
    pub normal: Vec3,
    pub d: f32,
}

pub fn extract_frustum_planes(view_proj: Mat4) -> [Plane; 6] {
    let m = view_proj.transpose();

    let r0 = m.x_axis;
    let r1 = m.y_axis;
    let r2 = m.z_axis;
    let r3 = m.w_axis;

    let raw = [
        r3 + r0, // left
        r3 - r0, // right
        r3 + r1, // bottom
        r3 - r1, // top
        r2,      // near (wgpu z >= 0)
        r3 - r2, // far  (z <= w)
    ];

    raw.map(|p| {
        let n = Vec3::new(p.x, p.y, p.z);
        let inv_len = 1.0 / n.length();

        Plane {
            normal: n * inv_len,
            d: p.w * inv_len,
        }
    })
}

pub fn aabb_in_frustum(planes: &[Plane; 6], min: Vec3, max: Vec3) -> bool {
    let center = (min + max) * 0.5;
    let extents = (max - min) * 0.5;

    const EPS: f32 = 0.5;

    for p in planes {
        let r = extents.dot(p.normal.abs());
        let s = p.normal.dot(center) + p.d;

        if s + r < -EPS {
            return false;
        }
    }
    true
}

#[inline]
fn chunk_aabb_render(cx: i32, cz: i32, cs: ChunkSize, eye: WorldPos) -> (Vec3, Vec3) {
    let origin =
        WorldPos::new(ChunkCoord::new(cx, cz), LocalPos::new(0.0, 0.0, 0.0)).to_render_pos(eye, cs);

    let min = origin + Vec3::new(0.0, CHUNK_MIN_Y, 0.0);
    let max = origin + Vec3::new(cs as f32, CHUNK_MAX_Y, cs as f32);

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
