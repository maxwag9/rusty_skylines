use crate::commands::Command;
use crate::data::{LodCenterType, Settings};
use crate::helpers::mouse_ray::*;
use crate::helpers::positions::*;
use crate::renderer::benchmark::{Benchmark, ChunkJobConfig};
use crate::renderer::gizmo::gizmo::Gizmo;
use crate::renderer::mesh_arena::{GeometryScratch, TerrainMeshArena};
use crate::resources::Time;
use crate::simulation::Ticker;
use crate::ui::input::Input;
use crate::ui::vertex::Vertex;
use crate::world::buildings::zoning::ZoningType;
use crate::world::camera::Camera;
use crate::world::game_state::SaveState;
use crate::world::roads::road_mesh_manager::{ChunkId, chunk_coord_to_id};
use crate::world::roads::road_structs::RoadType;
use crate::world::roads::road_subsystem::Roads;
use crate::world::terrain::chunk_builder::*;
use crate::world::terrain::terrain_editing::*;
use crate::world::terrain::terrain_gen::{TerrainGenerator, TerrainParams};
use crate::world::terrain::terrain_threads::{
    ChunkWorkerPool, LoadedChunkSnapshot, LoadedChunksSnapshot, PendingChunkRequest,
    TerrainEditsSnapshot,
};
use glam::{Mat4, Vec3};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;
use wgpu::{Buffer, Device, IndexFormat, Queue, RenderPass};

#[derive(Clone)]
pub struct ChunkCoords {
    pub chunk_coord: ChunkCoord, // Y IS UP/DOWN LIKE IN MINECRAFT NOT CRINGE Z LIKE BLENDER ETC. (Blender is awesome)
    pub dist2: i32,
}
#[derive(Clone)]
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
    cam_pos: WorldPos,
    planes: [Plane; 6],
    r2_render: i32,
    r2_gen: i32,
}
#[derive(Debug, Clone, PartialEq)]
pub enum CursorMode {
    None,
    Roads,
    Zoning,
    Props,
    TerrainEditing,
    Cars,
    Area,
    Destruction,
}
impl From<String> for CursorMode {
    fn from(string: String) -> Self {
        match string.to_lowercase().as_str() {
            "none" => CursorMode::None,
            "roads" => CursorMode::Roads,
            "zoning" => CursorMode::Zoning,
            "props" => CursorMode::Props,
            "terrain_editing" => CursorMode::TerrainEditing,
            "cars" => CursorMode::Cars,
            "area" => CursorMode::Area,
            "destruction" => CursorMode::Destruction,
            _ => CursorMode::None,
        }
    }
}
impl CursorMode {
    fn order() -> [CursorMode; 8] {
        [
            CursorMode::Roads,
            CursorMode::Zoning,
            CursorMode::Area,
            CursorMode::Props,
            CursorMode::TerrainEditing,
            CursorMode::Cars,
            CursorMode::Destruction,
            CursorMode::None,
        ]
    }

    fn kind_eq(a: &CursorMode, b: &CursorMode) -> bool {
        matches!(
            (a, b),
            (CursorMode::None, CursorMode::None)
                | (CursorMode::Props, CursorMode::Props)
                | (CursorMode::Cars, CursorMode::Cars)
                | (CursorMode::TerrainEditing, CursorMode::TerrainEditing)
                | (CursorMode::Roads, CursorMode::Roads)
                | (CursorMode::Zoning, CursorMode::Zoning)
                | (CursorMode::Area, CursorMode::Area)
                | (CursorMode::Destruction, CursorMode::Destruction)
        )
    }

    pub fn next(&self) -> CursorMode {
        let order = Self::order();
        let i = order.iter().position(|m| Self::kind_eq(m, self)).unwrap();
        order[(i + 1) % order.len()].to_owned()
    }

    pub fn next_command(&self) -> Command {
        Command::SetCursorMode(self.next())
    }
}
#[derive(Debug)]
pub struct Cursor {
    pub mode: CursorMode,
    pub road_type: Option<RoadType>,
    pub prop_name: Option<String>,
    pub zoning_type: ZoningType,
}

impl Cursor {
    pub fn new() -> Self {
        Self {
            mode: CursorMode::Roads,
            road_type: Some(RoadType::default()),
            prop_name: Some("oak".to_string()),
            zoning_type: ZoningType::None,
        }
    }
}

pub struct TerrainJobs {
    pub workers: ChunkWorkerPool,
    pub job_config: ChunkJobConfig,
    pub max_close_jobs_per_frame: usize,
    pub max_far_jobs_per_frame: usize,
}

impl TerrainJobs {
    fn new(workers: ChunkWorkerPool) -> Self {
        Self {
            workers,
            job_config: ChunkJobConfig::default(),
            max_close_jobs_per_frame: 4, // was 1
            max_far_jobs_per_frame: 4,   // was 1
        }
    }
    fn dispatch_jobs_for_visible(
        &mut self,
        terrain_editor: &TerrainEditor,
        chunks: &HashMap<ChunkCoord, ChunkMeshLod>,
        pending: &VecDeque<CpuChunkMesh>,
        visible: &Vec<VisibleChunk>,
        lod_map: &HashMap<ChunkCoord, LodStep>,
    ) {
        let pending_coords: HashSet<ChunkCoord> =
            pending.iter().map(|cpu| cpu.chunk_coord).collect();
        let mut close_jobs_sent = 0usize;
        let mut far_jobs_sent = 0usize;

        for v in visible.iter() {
            if close_jobs_sent >= self.max_close_jobs_per_frame
                || far_jobs_sent >= self.max_far_jobs_per_frame
            {
                break;
            }

            let coord = v.coords.chunk_coord;
            let coord_x_neg = ChunkCoord::new(coord.x - 1, coord.z);
            let coord_x_pos = ChunkCoord::new(coord.x + 1, coord.z);
            let coord_z_neg = ChunkCoord::new(coord.x, coord.z - 1);
            let coord_z_pos = ChunkCoord::new(coord.x, coord.z + 1);
            let step = *lod_map.get(&coord).unwrap_or(&1);
            let nx_neg = *lod_map.get(&coord_x_neg).unwrap_or(&step);
            let nx_pos = *lod_map.get(&coord_x_pos).unwrap_or(&step);
            let nz_neg = *lod_map.get(&coord_z_neg).unwrap_or(&step);
            let nz_pos = *lod_map.get(&coord_z_pos).unwrap_or(&step);
            let desired = ChunkState {
                step,
                nx_neg,
                nx_pos,
                nz_neg,
                nz_pos,
            };

            if let Some(existing) = chunks.get(&coord) {
                if existing.state.same_as(&desired) {
                    continue;
                }
            }
            // Check if already completed and waiting in pending_results
            if pending_coords.contains(&coord) {
                continue; // Don't re-submit!
            }
            if self.workers.has_request(coord, &desired) || self.workers.is_building(coord) {
                continue;
            }

            let has_edits = terrain_editor.has_edits_on_chunk(coord);

            let (version, version_atomic) = self.workers.new_version_for(coord);
            let priority = (u64::MAX - v.coords.dist2 as u64) / 16;

            let terrain_edits_snapshot = TerrainEditsSnapshot {
                edits: Arc::new(terrain_editor.edits.clone()),
                affected_chunks: Arc::new(terrain_editor.affected_chunks.clone()),
            };

            let mut loaded_snapshot: LoadedChunksSnapshot = HashMap::new();

            for (&coord, chunk) in chunks.iter() {
                loaded_snapshot.insert(
                    coord,
                    LoadedChunkSnapshot {
                        step: chunk.state.step,
                        height_grid: chunk.height_grid.clone(),
                    },
                );
            }
            self.workers.submit_request(PendingChunkRequest {
                coord,
                state: desired,

                version,
                version_atomic,

                has_edits,

                priority,
                in_progress: Arc::new(AtomicBool::new(false)),
                terrain_edits_snapshot,
                loaded_snapshot: Arc::new(loaded_snapshot),
            });

            if close_jobs_sent < self.max_close_jobs_per_frame {
                close_jobs_sent += 1;
            } else if far_jobs_sent < self.max_far_jobs_per_frame {
                far_jobs_sent += 1;
            }
        }
    }
}

pub struct TerrainRenderSubsystem {}
impl TerrainRenderSubsystem {
    pub fn new() -> Self {
        Self { /* fields */ }
    }
    pub fn render(
        &self,
        pass: &mut RenderPass,
        terrain: &Terrain,
        camera: &Camera,
        _aspect: f32,
        _settings: &Settings,
        underwater: bool,
    ) {
        let t_frame = Instant::now();
        let view_proj = camera.view_proj();
        let planes = extract_frustum_planes(view_proj);

        let target_pos = camera.target;
        let target_cx = target_pos.chunk.x;
        let target_cz = target_pos.chunk.z;

        let r = terrain.view_radius_render as i32;
        let r2 = r * r;

        let mut per_page: Vec<Vec<GpuChunkHandle>> = vec![Vec::new(); terrain.arena.pages.len()];

        for (&chunk_coord, chunk) in terrain.chunks.iter() {
            let dx = (chunk_coord.x - target_cx).abs();
            if dx > r {
                continue;
            }

            let dz = (chunk_coord.z - target_cz).abs();
            if dz > r {
                continue;
            }

            if (dx as i64 * dx as i64 + dz as i64 * dz as i64) > r2 as i64 {
                continue;
            }

            let (min, max) = chunk_aabb_render(chunk_coord.x, chunk_coord.z, camera.eye_world());
            if !aabb_in_frustum(&planes, min, max) {
                continue;
            }

            let p = chunk.handle.page;
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

        let total_vertex_bytes = terrain.arena.pages.len() as u64 * terrain.arena.page_v_bytes;

        let total_index_bytes = terrain.arena.pages.len() as u64 * terrain.arena.page_i_bytes;

        let total_bytes = total_vertex_bytes + total_index_bytes;

        // println!(
        //     "Terrain: {} pages, {:.2} MB VRAM ({:.2} MB vertex, {:.2} MB index) \n\
        //                                         {} Vertices, {} Indices",
        //     terrain.arena.pages.len(),
        //     total_bytes as f64 / 1024.0 / 1024.0,
        //     total_vertex_bytes as f64 / 1024.0 / 1024.0,
        //     total_index_bytes as f64 / 1024.0 / 1024.0,
        //     total_vertices,
        //     total_indices
        // );

        for (pi, handles) in per_page.iter().enumerate() {
            if handles.is_empty() {
                continue;
            }

            let page = &terrain.arena.pages[pi];
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
        //let total_ms = t_frame.elapsed().as_secs_f32() * 1000.0;
        // println!("{}", total_ms);
    }
}

pub struct Terrain {
    pub tick: Ticker,
    pub cursor: Cursor,
    pub arena: TerrainMeshArena,

    pub chunks: HashMap<ChunkCoord, ChunkMeshLod>,
    pub terrain_editor: TerrainEditor,

    pub view_radius_generate: usize,
    pub view_radius_render: usize,

    pub terrain_jobs: TerrainJobs,

    pub spiral: Vec<ChunkCoord>,
    pub lod_map: HashMap<ChunkCoord, LodStep>,

    pub pick_radius_m: f32,
    pub last_picked: Option<PickedPoint>,

    pub benchmark: Benchmark,
    pub frame_timings: FrameTimings,
    pub visible: Vec<VisibleChunk>,
    pub terrain_gen: TerrainGenerator,

    // Cache for LOD computation to avoid rebuilding every frame
    lod_coord_to_index: HashMap<ChunkCoord, usize>,
    lod_steps_buffer: Vec<LodStep>,

    last_camera_chunk: ChunkCoord,
    last_visible: Vec<VisibleChunk>,

    pub pending_results: VecDeque<CpuChunkMesh>,
    device: Device,
    queue: Queue,
}
const VERTEX_SIZE_BYTES: usize = size_of::<Vertex>();
impl Terrain {
    pub fn new(
        device: &Device,
        queue: &Queue,
        settings: &Settings,
        save_state: &mut SaveState,
    ) -> Self {
        let cs = chunk_size() as f32;
        let view_radius_render = (64f32 * (64f32 / cs)) as usize;
        let view_radius_generate = (32f32 * (64f32 / cs)) as usize;

        let arena = TerrainMeshArena::new(
            device,
            64 * 1024 * 1024, // vertex bytes per page
            32 * 1024 * 1024, // index bytes per page
        );

        let terrain_editor = TerrainEditor::default();

        let mut terrain_params = TerrainParams::default();
        terrain_params.seed = 144;
        let terrain_gen = TerrainGenerator::new(terrain_params);

        let threads = num_cpus::get_physical().saturating_sub(1).max(1);
        println!("Using {} chunk workers", threads);

        let workers = ChunkWorkerPool::new(threads, terrain_gen.clone());
        Self {
            tick: Ticker::new(60.0),
            cursor: Cursor::new(),
            arena,
            chunks: HashMap::new(),
            terrain_editor,

            view_radius_generate,
            view_radius_render,

            terrain_jobs: TerrainJobs::new(workers),

            terrain_gen,

            spiral: generate_spiral_offsets(view_radius_generate as i32),
            lod_map: HashMap::new(),
            pick_radius_m: 5.0,
            last_picked: None,

            benchmark: Benchmark::default(),
            frame_timings: FrameTimings::default(),
            visible: vec![],

            lod_coord_to_index: HashMap::new(),
            lod_steps_buffer: vec![],
            last_camera_chunk: ChunkCoord::new(9999999, 99984496),
            last_visible: vec![],

            pending_results: VecDeque::with_capacity(64),
            device: device.clone(),
            queue: queue.clone(),
        }
    }

    pub fn update(
        &mut self,
        gizmo: &mut Gizmo,
        camera: &Camera,
        aspect: f32,
        settings: &Settings,
        input_state: &mut Input,
        _time: &Time,
        roads: &mut Roads,
    ) {
        let t_frame = Instant::now();

        if settings.world_generation_benchmark_mode {
            self.benchmark
                .run(self.chunks.len(), &mut self.terrain_jobs.job_config, || {
                    self.chunks.clear();
                    self.terrain_jobs.workers.clear();
                    self.lod_map.clear();
                });
        }

        let frame = self.frame_state(settings, camera, aspect);

        let t0 = Instant::now();
        self.drain_finished_meshes();
        self.frame_timings.drain_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let t0 = Instant::now();
        let current_chunk = camera.target.chunk;
        // if current_chunk != self.last_camera_chunk {
        //     self.last_camera_chunk = current_chunk;
        self.visible = self.collect_visible(camera, &frame);
        // } else {
        //     // Just re-sort by distance
        //     self.visible.sort_unstable_by_key(|v| {
        //         let dx = v.coords.chunk_coord.x - current_chunk.x;
        //         let dz = v.coords.chunk_coord.z - current_chunk.z;
        //         dx*dx + dz*dz
        //     });
        // }
        self.frame_timings.collect_visible_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let t0 = Instant::now();
        self.visible.sort_unstable_by_key(|v| v.coords.dist2);
        self.frame_timings.sort_visible_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let t0 = Instant::now();
        self.compute_lod_for_visible(frame.r2_gen);
        self.frame_timings.lod_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let t0 = Instant::now();
        self.terrain_jobs.dispatch_jobs_for_visible(
            &self.terrain_editor,
            &self.chunks,
            &self.pending_results,
            &self.visible,
            &self.lod_map,
        );
        self.frame_timings.dispatch_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let t0 = Instant::now();
        self.unload_out_of_range(&frame);
        self.frame_timings.unload_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let t0 = Instant::now();
        if settings.show_world {
            self.handle_terrain_editing(input_state, roads);
        }

        self.frame_timings.edit_ms = t0.elapsed().as_secs_f32() * 1000.0;

        self.frame_timings.total_ms = t_frame.elapsed().as_secs_f32() * 1000.0;
        //println!("{:#?}", self.frame_timings);
        // println!("Queue: {} pending, {} results backlog, {} chunks loaded",
        //          self.terrain_jobs.workers.pending_count(),
        //          self.pending_results.len(),
        //          self.chunks.len()
        // );
    }
    pub fn flush_dirty_chunks(&mut self) {
        let freed = self.terrain_editor.flush_dirty_chunks(
            &self.device,
            &self.queue,
            &mut self.arena,
            &mut self.chunks,
            &self.terrain_gen,
        );
        for h in freed {
            self.arena.free::<Vertex>(h);
        }
    }
    fn handle_terrain_editing(&mut self, input_state: &mut Input, roads: &mut Roads) {
        self.flush_dirty_chunks();
        match self.cursor.mode {
            CursorMode::TerrainEditing => {} // just continue lol
            _ => {
                return;
            } // Fuck off, this is not your mode!!
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
            TerrainEditSource::Player,
        );
    }

    fn frame_state(&self, settings: &Settings, camera: &Camera, _aspect: f32) -> FrameState {
        let cam_pos = match settings.lod_center {
            LodCenterType::Eye => camera.eye_world(),
            LodCenterType::Target => camera.target,
        };

        let view_proj = camera.view_proj();
        let planes = extract_frustum_planes(view_proj);

        let r_render = self.view_radius_render as i32;
        let r_gen = self.view_radius_generate as i32;

        FrameState {
            cam_pos,
            planes,
            r2_render: r_render * r_render,
            r2_gen: r_gen * r_gen,
        }
    }

    pub fn drain_finished_meshes(&mut self) {
        let mut recv_ms = 0.0;
        let mut rebuild_ms = 0.0;
        let mut gpu_ms = 0.0;
        let mut insert_ms = 0.0;
        let detail = DrainDetailTimings::default();

        let t0 = Instant::now();
        while let Ok(cpu) = self.terrain_jobs.workers.result_rx.try_recv() {
            self.pending_results.push_back(cpu);
        }
        recv_ms += t0.elapsed().as_secs_f32() * 1000.0;

        let mut processed = 0;
        let max_per_frame = 8;

        while processed < max_per_frame {
            let Some(cpu) = self.pending_results.pop_front() else {
                break;
            };
            processed += 1;

            let coord = cpu.chunk_coord;

            if !self
                .terrain_jobs
                .workers
                .is_current_version(coord, cpu.version)
            {
                continue;
            }

            let t0 = Instant::now();

            let (vertices, indices, height_grid) = (cpu.vertices, cpu.indices, cpu.height_grid);

            rebuild_ms += t0.elapsed().as_secs_f32() * 1000.0;

            let t0 = Instant::now();

            self.free_chunk_gpu(coord);

            let handle = self.arena.alloc_and_upload(
                &self.device,
                &self.queue,
                &vertices,
                &indices,
                &mut GeometryScratch::default(),
            );

            gpu_ms += t0.elapsed().as_secs_f32() * 1000.0;

            let t0 = Instant::now();

            let coord_x_neg = ChunkCoord::new(coord.x - 1, coord.z);
            let coord_x_pos = ChunkCoord::new(coord.x + 1, coord.z);
            let coord_z_neg = ChunkCoord::new(coord.x, coord.z - 1);
            let coord_z_pos = ChunkCoord::new(coord.x, coord.z + 1);

            let step = *self.lod_map.get(&coord).unwrap_or(&1);

            let nx_neg = *self.lod_map.get(&coord_x_neg).unwrap_or(&step);
            let nx_pos = *self.lod_map.get(&coord_x_pos).unwrap_or(&step);
            let nz_neg = *self.lod_map.get(&coord_z_neg).unwrap_or(&step);
            let nz_pos = *self.lod_map.get(&coord_z_pos).unwrap_or(&step);

            self.chunks.insert(
                coord,
                ChunkMeshLod {
                    state: ChunkState {
                        step,
                        nx_neg,
                        nx_pos,
                        nz_neg,
                        nz_pos,
                    },
                    handle,
                    cpu_vertices: vertices,
                    cpu_indices: indices,
                    height_grid,
                },
            );

            insert_ms += t0.elapsed().as_secs_f32() * 1000.0;
        }

        self.frame_timings.drain_recv_ms = recv_ms;
        self.frame_timings.drain_cpu_rebuild_ms = rebuild_ms;
        self.frame_timings.drain_detail = detail;
        self.frame_timings.drain_gpu_upload_ms = gpu_ms;
        self.frame_timings.drain_chunk_insert_ms = insert_ms;
    }
    fn free_chunk_gpu(&mut self, coord: ChunkCoord) {
        if let Some(old) = self.chunks.remove(&coord) {
            self.arena.free::<Vertex>(old.handle);
        }
    }

    fn unload_chunk(&mut self, coord: ChunkCoord) {
        self.free_chunk_gpu(coord);
        self.terrain_jobs.workers.forget_chunk(coord);
    }

    fn collect_visible(&self, camera: &Camera, frame: &FrameState) -> Vec<VisibleChunk> {
        let mut visible = Vec::with_capacity(400);
        let cam_chunk = frame.cam_pos.chunk;
        let r2_render = frame.r2_render;

        // Iterate using pre-computed spiral for perfect distance ordering
        for offset in &self.spiral {
            let dist2 = offset.x * offset.x + offset.z * offset.z;

            if dist2 > r2_render {
                continue;
            }
            let coord = cam_chunk.offset(offset.x, offset.z);

            let (min, max) = chunk_aabb_render(coord.x, coord.z, camera.eye_world());

            if aabb_in_frustum(&frame.planes, min, max) {
                visible.push(VisibleChunk {
                    coords: ChunkCoords {
                        chunk_coord: coord,
                        dist2,
                    },
                    id: chunk_coord_to_id(coord.x, coord.z),
                });
            }
        }

        visible
    }

    fn compute_lod_for_visible(&mut self, r2_gen: i32) {
        let n = self.visible.len();
        if n == 0 {
            self.lod_map.clear();
            return;
        }

        self.lod_coord_to_index.clear();
        self.lod_steps_buffer.clear();
        self.lod_steps_buffer.reserve(n);

        let mut base_steps = Vec::with_capacity(n);

        for (i, v) in self.visible.iter().enumerate() {
            let dist2 = v.coords.dist2;

            let step = if dist2 > r2_gen {
                lod_step_for_distance(r2_gen + 1)
            } else {
                lod_step_for_distance(dist2)
            };

            base_steps.push(step);
            self.lod_coord_to_index.insert(v.coords.chunk_coord, i);
        }

        self.lod_steps_buffer = base_steps.clone();

        // single stabilization pass only
        for (i, v) in self.visible.iter().enumerate() {
            let coord = v.coords.chunk_coord;
            let s = base_steps[i];

            let mut max_n = s;

            if let Some(&j) = self
                .lod_coord_to_index
                .get(&ChunkCoord::new(coord.x - 1, coord.z))
            {
                max_n = max_n.max(base_steps[j]);
            }
            if let Some(&j) = self
                .lod_coord_to_index
                .get(&ChunkCoord::new(coord.x + 1, coord.z))
            {
                max_n = max_n.max(base_steps[j]);
            }
            if let Some(&j) = self
                .lod_coord_to_index
                .get(&ChunkCoord::new(coord.x, coord.z - 1))
            {
                max_n = max_n.max(base_steps[j]);
            }
            if let Some(&j) = self
                .lod_coord_to_index
                .get(&ChunkCoord::new(coord.x, coord.z + 1))
            {
                max_n = max_n.max(base_steps[j]);
            }

            let min_allowed = (max_n / 2).max(1);
            self.lod_steps_buffer[i] = s.max(min_allowed);
        }

        self.lod_map.clear();
        self.lod_map.reserve(n);

        for (i, v) in self.visible.iter().enumerate() {
            self.lod_map
                .insert(v.coords.chunk_coord, self.lod_steps_buffer[i]);
        }
    }

    fn unload_out_of_range(&mut self, frame: &FrameState) {
        // Avoid thrash: unload outside render radius + generous margin.
        let margin = (self.view_radius_render / 2) as u64;
        let r = self.view_radius_render as u64 + margin;
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
            let dist2 = chunk_coord.dist2(&frame.cam_pos.chunk);

            if dist2 > r2 {
                to_remove.push(coord);
            }
        }

        for coord in to_remove {
            self.unload_chunk(coord);
            self.terrain_editor.pristine_grids.remove(&coord);
        }
    }

    /// Pick a terrain point by casting a ray through loaded chunks.
    /// Uses WorldRay for maximum precision at any distance.
    pub fn pick_terrain_point(&mut self, ray: WorldRay) -> Option<WorldPos> {
        let cs = chunk_size() as f32;
        let eps = 1e-6 * cs;

        // Start DDA from ray origin's chunk (exact, no precision loss)
        let mut cx = ray.origin.chunk.x;
        let mut cz = ray.origin.chunk.z;

        let step_x = if ray.dir.x >= 0.0 { 1i32 } else { -1i32 };
        let step_z = if ray.dir.z >= 0.0 { 1i32 } else { -1i32 };

        // Compute t to first chunk boundaries using integer chunk math
        let first_boundary_x = if step_x > 0 { cx + 1 } else { cx };
        let first_boundary_z = if step_z > 0 { cz + 1 } else { cz };

        let mut t_max_x = ray.t_to_chunk_x_boundary(first_boundary_x);
        let mut t_max_z = ray.t_to_chunk_z_boundary(first_boundary_z);

        // Handle ray starting exactly on boundary moving negative
        if t_max_x <= 0.0 && step_x < 0 {
            cx -= 1;
            t_max_x = ray.t_to_chunk_x_boundary(cx);
        }
        if t_max_z <= 0.0 && step_z < 0 {
            cz -= 1;
            t_max_z = ray.t_to_chunk_z_boundary(cz);
        }
        let t_delta_x = if ray.dir.x.abs() < 1e-12 {
            f32::INFINITY
        } else {
            cs / ray.dir.x.abs()
        };

        let t_delta_z = if ray.dir.z.abs() < 1e-12 {
            f32::INFINITY
        } else {
            cs / ray.dir.z.abs()
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
                            dist2: 0,
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

    pub fn make_pick_uniforms(&self, queue: &Queue, pick_uniform_buffer: &Buffer, camera: &Camera) {
        let u = if let Some(p) = &self.last_picked {
            PickUniform {
                pos: p.pos.to_relative_pos(camera.eye_world()).to_array(),
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

    pub fn get_height_at(&self, pos: WorldPos, high_res: bool) -> f32 {
        let chunk = match self.chunks.get(&pos.chunk) {
            Some(c) => c,
            None => return self.terrain_gen.height(&pos),
        };
        if high_res && chunk.state.step != 1 {
            return self.terrain_gen.height(&pos);
        }
        height_bilinear_world(&chunk.height_grid, &pos)
    }
    pub fn get_height_at_explicit(
        chunks: &HashMap<ChunkCoord, ChunkMeshLod>,
        terrain_gen: &TerrainGenerator,
        pos: WorldPos,
        high_res: bool,
    ) -> f32 {
        let chunk = match chunks.get(&pos.chunk) {
            Some(c) => c,
            None => return terrain_gen.height(&pos),
        };
        if high_res && chunk.state.step != 1 {
            return terrain_gen.height(&pos);
        }
        height_bilinear_world(&chunk.height_grid, &pos)
    }
    fn rebuild_colors(&self, grid: &ChunkHeightGrid) -> Vec<[f32; 3]> {
        let mut colors = Vec::with_capacity(grid.nx * grid.nz);
        let cell = grid.step_f32();

        for gx in 0..grid.nx {
            for gz in 0..grid.nz {
                let idx = gx * grid.nz + gz;
                let h = grid.heights[idx];
                let pos = WorldPos::new(
                    grid.chunk_coord,
                    LocalPos::new(gx as f32 * cell, h, gz as f32 * cell),
                );
                let m = self.terrain_gen.moisture(&pos, h);
                colors.push(self.terrain_gen.color(&pos, h, m));
            }
        }
        colors
    }

    fn rebuild_normals(&self, grid: &ChunkHeightGrid) -> Vec<[f32; 3]> {
        let nx = grid.nx;
        let nz = grid.nz;
        let cell = grid.step_f32();
        let inv = 1.0 / cell;
        let mut normals = vec![[0.0f32, 1.0, 0.0]; nx * nz];

        for gx in 0..nx {
            for gz in 0..nz {
                let idx = gx * nz + gz;

                let h_l = if gx > 0 {
                    grid.heights[(gx - 1) * nz + gz]
                } else {
                    grid.heights[idx]
                };
                let h_r = if gx + 1 < nx {
                    grid.heights[(gx + 1) * nz + gz]
                } else {
                    grid.heights[idx]
                };
                let h_d = if gz > 0 {
                    grid.heights[gx * nz + gz - 1]
                } else {
                    grid.heights[idx]
                };
                let h_u = if gz + 1 < nz {
                    grid.heights[gx * nz + gz + 1]
                } else {
                    grid.heights[idx]
                };

                let dhdx = (h_r - h_l) * 0.5 * inv;
                let dhdz = (h_u - h_d) * 0.5 * inv;
                let n = Vec3::new(-dhdx, 1.0, -dhdz).normalize();
                normals[idx] = [n.x, n.y, n.z];
            }
        }
        normals
    }

    fn chunk_needs_stitching(&self, coord: ChunkCoord, own_step: LodStep) -> bool {
        for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let nb = ChunkCoord::new(coord.x + dx, coord.z + dz);
            if self.terrain_editor.has_edits_on_chunk(nb) {
                continue;
            }
            if let Some(c) = self.chunks.get(&nb) {
                if c.state.step > own_step {
                    return true;
                }
            }
        }
        false
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
fn chunk_aabb_render(cx: i32, cz: i32, eye: WorldPos) -> (Vec3, Vec3) {
    let cs = chunk_size() as f32;
    let origin =
        WorldPos::new(ChunkCoord::new(cx, cz), LocalPos::new(0.0, 0.0, 0.0)).to_relative_pos(eye);

    let min = origin + Vec3::new(0.0, CHUNK_MIN_Y, 0.0);
    let max = origin + Vec3::new(cs, CHUNK_MAX_Y, cs);

    (min, max)
}
#[derive(Default, Debug)]
pub struct DrainDetailTimings {
    pub edits_ms: f32,
    pub patch_minmax_ms: f32,
    pub neighbor_extract_ms: f32,
    pub build_mesh_ms: f32,
    pub regenerate_ms: f32,
    pub skirts_ms: f32,
}
#[derive(Default, Debug)]
pub struct FrameTimings {
    pub drain_ms: f32,

    pub drain_recv_ms: f32,
    pub drain_cpu_rebuild_ms: f32,
    pub drain_detail: DrainDetailTimings,
    pub drain_gpu_upload_ms: f32,
    pub drain_chunk_insert_ms: f32,

    pub collect_visible_ms: f32,
    pub sort_visible_ms: f32,
    pub lod_ms: f32,
    pub dispatch_ms: f32,
    pub unload_ms: f32,
    pub edit_ms: f32,
    pub total_ms: f32,
}
/// Skirt depth below terrain surface
const SKIRT_DEPTH: f32 = 8.0;

fn make_vertex(
    terrain_generator: &TerrainGenerator,
    chunk_coord: ChunkCoord,
    pos: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    chunk_xz: [i32; 2],
) -> Vertex {
    let world_pos = WorldPos::new(chunk_coord, LocalPos::new(pos[0], pos[1], pos[2]));
    let h = pos[1];
    let moisture = terrain_generator.moisture(&world_pos, h);
    let color = terrain_generator.color(&world_pos, h, moisture);
    Vertex {
        local_position: pos,
        normal,
        color,
        chunk_xz,
        quad_uv: uv,
    }
}

/// Append skirt geometry for +X and +Z edges of the chunk.
/// Skirts are vertical strips that extend downward, TRYING to hide LOD seam gaps.
pub fn append_edge_skirts(
    terrain_generator: &TerrainGenerator,
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    height_grid: &ChunkHeightGrid,
    chunk_coord: ChunkCoord,
) {
    let nx = height_grid.nx;
    let nz = height_grid.nz;
    let cell = height_grid.step_f32();
    let chunk_xz = [chunk_coord.x, chunk_coord.z];

    // +X edge skirt
    let gx = nx - 1;
    let x = gx as f32 * cell;
    for gz in 0..(nz - 1) {
        let z0 = gz as f32 * cell;
        let z1 = (gz + 1) as f32 * cell;
        let h0 = height_grid.heights[gx * nz + gz];
        let h1 = height_grid.heights[gx * nz + gz + 1];
        let base = vertices.len() as u32;

        vertices.push(make_vertex(
            terrain_generator,
            chunk_coord,
            [x, h0, z0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0],
            chunk_xz,
        ));
        vertices.push(make_vertex(
            terrain_generator,
            chunk_coord,
            [x, h1, z1],
            [0.0, 1.0, 0.0],
            [1.0, 0.0],
            chunk_xz,
        ));
        vertices.push(make_vertex(
            terrain_generator,
            chunk_coord,
            [x, h0 - SKIRT_DEPTH, z0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0],
            chunk_xz,
        ));
        vertices.push(make_vertex(
            terrain_generator,
            chunk_coord,
            [x, h1 - SKIRT_DEPTH, z1],
            [0.0, 1.0, 0.0],
            [1.0, 1.0],
            chunk_xz,
        ));

        indices.extend_from_slice(&[base, base + 2, base + 1, base + 1, base + 2, base + 3]);
    }

    // +Z edge skirt
    let gz = nz - 1;
    let z = gz as f32 * cell;
    for gx in 0..(nx - 1) {
        let x0 = gx as f32 * cell;
        let x1 = (gx + 1) as f32 * cell;
        let h0 = height_grid.heights[gx * nz + gz];
        let h1 = height_grid.heights[(gx + 1) * nz + gz];
        let base = vertices.len() as u32;

        vertices.push(make_vertex(
            terrain_generator,
            chunk_coord,
            [x0, h0, z],
            [0.0, 1.0, 0.0],
            [0.0, 0.0],
            chunk_xz,
        ));
        vertices.push(make_vertex(
            terrain_generator,
            chunk_coord,
            [x1, h1, z],
            [0.0, 1.0, 0.0],
            [1.0, 0.0],
            chunk_xz,
        ));
        vertices.push(make_vertex(
            terrain_generator,
            chunk_coord,
            [x0, h0 - SKIRT_DEPTH, z],
            [0.0, 1.0, 0.0],
            [0.0, 1.0],
            chunk_xz,
        ));
        vertices.push(make_vertex(
            terrain_generator,
            chunk_coord,
            [x1, h1 - SKIRT_DEPTH, z],
            [0.0, 1.0, 0.0],
            [1.0, 1.0],
            chunk_xz,
        ));

        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 1, base + 3]);
    }
}
