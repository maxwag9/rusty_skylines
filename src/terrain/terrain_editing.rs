use crate::helpers::positions::{ChunkCoord, ChunkSize, LocalPos, LodStep, WorldPos};
use crate::renderer::mesh_arena::{GeometryScratch, TerrainMeshArena};
use crate::terrain::chunk_builder::{
    ChunkHeightGrid, ChunkMeshLod, GpuChunkHandle, gather_neighbor_edge_heights,
    regenerate_vertices_from_height_grid,
};
use crate::terrain::terrain::TerrainGenerator;
use crate::ui::vertex::Vertex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use wgpu::{Device, Queue};

#[derive(Serialize, Deserialize)]
struct PersistedChunk {
    accumulated_deltas: Vec<((u16, u16), f32)>,
    dirty: bool,
}

#[derive(Serialize, Deserialize)]
struct EditFile {
    version: u32,
    timestamp_unix: u64,
    edits: HashMap<ChunkCoord, PersistedChunk>,
}

pub trait Falloff {
    fn weight(d2: f32, r2: f32) -> f32;
}

pub struct SmoothFalloff;
impl Falloff for SmoothFalloff {
    fn weight(d2: f32, r2: f32) -> f32 {
        let t = 1.0 - (d2 / r2);
        if t <= 0.0 {
            return 0.0;
        }
        t * t * (3.0 - 2.0 * t)
    }
}

pub trait BrushOp {
    fn apply(y: &mut f32, strength: f32, weight: f32);
}

pub struct Raise;
impl BrushOp for Raise {
    fn apply(y: &mut f32, strength: f32, weight: f32) {
        *y += strength * weight;
    }
}

/// Calculate which chunks are affected by a brush centered at `center` with given `radius`.
/// Returns (min_chunk_x, max_chunk_x, min_chunk_z, max_chunk_z).
pub fn affected_chunks(
    center: WorldPos,
    radius: f32,
    chunk_size: ChunkSize,
) -> (i32, i32, i32, i32) {
    let chunk_size_f64 = chunk_size as f64;
    let radius_f64 = radius as f64;

    // Compute world coordinates in f64 for precision
    let wx = center.chunk.x as f64 * chunk_size_f64 + center.local.x as f64;
    let wz = center.chunk.z as f64 * chunk_size_f64 + center.local.z as f64;

    let min_x = ((wx - radius_f64) / chunk_size_f64).floor() as i32;
    let max_x = ((wx + radius_f64) / chunk_size_f64).floor() as i32;
    let min_z = ((wz - radius_f64) / chunk_size_f64).floor() as i32;
    let max_z = ((wz + radius_f64) / chunk_size_f64).floor() as i32;

    (min_x, max_x, min_z, max_z)
}

#[derive(Clone, Copy, PartialEq)]
enum Edge {
    XNeg,
    XPos,
    ZNeg,
    ZPos,
}

pub struct EditedChunk {
    pub pending_deltas: Vec<(u16, u16, f32)>,
    /// Accumulated deltas stored at LOD 1 (step=1) coordinates
    pub accumulated_deltas: HashMap<(u16, u16), f32>,
    pub dirty: bool,
}

impl Default for EditedChunk {
    fn default() -> Self {
        Self {
            pending_deltas: Vec::new(),
            accumulated_deltas: HashMap::new(),
            dirty: false,
        }
    }
}

pub struct TerrainEditor {
    pub(crate) edited_chunks: HashMap<ChunkCoord, EditedChunk>,
}

impl Default for TerrainEditor {
    fn default() -> Self {
        Self {
            edited_chunks: HashMap::new(),
        }
    }
}

impl TerrainEditor {
    pub fn save_edits<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("tmp");

        let mut edits: HashMap<ChunkCoord, PersistedChunk> = HashMap::new();
        for (chunk_coord, chunk) in &self.edited_chunks {
            if chunk.accumulated_deltas.is_empty() && !chunk.dirty {
                continue;
            }
            let vec = chunk
                .accumulated_deltas
                .iter()
                .filter(|&(_, &h)| h.abs() > 0.001)
                .map(|(&(x, z), &h)| ((x, z), h))
                .collect();

            edits.insert(
                *chunk_coord,
                PersistedChunk {
                    accumulated_deltas: vec,
                    dirty: chunk.dirty,
                },
            );
        }

        let meta = EditFile {
            version: 1,
            timestamp_unix: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            edits,
        };

        let bytes = postcard::to_allocvec(&meta)?;
        let compressed = zstd::encode_all(&bytes[..], 3)?;
        // for (coord, chunk) in &self.edited_chunks {
        //     println!(
        //         "Chunk {:?}: {} deltas, dirty={}",
        //         coord,
        //         chunk.accumulated_deltas.len(),
        //         chunk.dirty
        //     );
        // }
        let total: usize = self
            .edited_chunks
            .values()
            .map(|c| c.accumulated_deltas.len())
            .sum();
        println!("Total deltas across all chunks: {}", total);
        println!("Uncompressed size: {} bytes", bytes.len());
        println!("Compressed size: {} bytes", compressed.len());

        fs::write(&tmp, compressed)?;
        fs::rename(&tmp, path)?;

        Ok(())
    }

    pub fn load_edits<P: AsRef<Path>>(
        path: P,
    ) -> Result<TerrainEditor, Box<dyn std::error::Error>> {
        let data = fs::read(path)?;
        let decompressed = zstd::decode_all(&data[..])?;
        let meta: EditFile = postcard::from_bytes(&decompressed)?;
        let total_entries: usize = meta
            .edits
            .values()
            .map(|c| c.accumulated_deltas.len())
            .sum();
        println!("Total entries: {}", total_entries);
        println!("Avg per chunk: {}", total_entries / meta.edits.len().max(1));
        // Add this before serialization:
        println!("Chunk count: {}", meta.edits.len());
        let mut editor = TerrainEditor {
            edited_chunks: HashMap::new(),
        };
        for (chunk_coord, persisted_chunk) in meta.edits {
            let mut acc = HashMap::with_capacity(persisted_chunk.accumulated_deltas.len());
            for ((x16, z16), h) in persisted_chunk.accumulated_deltas {
                acc.insert((x16, z16), h);
            }
            editor.edited_chunks.insert(
                chunk_coord,
                EditedChunk {
                    pending_deltas: Vec::new(),
                    accumulated_deltas: acc,
                    dirty: persisted_chunk.dirty,
                },
            );
        }

        Ok(editor)
    }

    pub fn apply_brush<F: Falloff, B: BrushOp>(
        &mut self,
        center: WorldPos,
        radius: f32,
        strength: f32,
        chunk_size: ChunkSize,
        chunks: &HashMap<ChunkCoord, ChunkMeshLod>,
    ) {
        let (min_cx, max_cx, min_cz, max_cz) = affected_chunks(center, radius, chunk_size);
        let r2 = radius * radius;

        let target_coords: Vec<ChunkCoord> = (min_cx..=max_cx)
            .flat_map(|cx| (min_cz..=max_cz).map(move |cz| ChunkCoord::new(cx, cz)))
            .collect();

        for &coord in &target_coords {
            self.edited_chunks.entry(coord).or_default();
        }

        for coord in target_coords {
            self.compute_deltas_for_chunk::<F, B>(coord, center, radius, r2, strength, chunks);
        }
    }

    /// Compute deltas for a chunk based on brush application.
    /// `center` is the brush center as a WorldPos.
    fn compute_deltas_for_chunk<F: Falloff, B: BrushOp>(
        &mut self,
        coord: ChunkCoord,
        center: WorldPos,
        radius: f32,
        r2: f32,
        strength: f32,
        chunks: &HashMap<ChunkCoord, ChunkMeshLod>,
    ) {
        let chunk = match chunks.get(&coord) {
            Some(c) => c,
            None => return,
        };

        let hg = &*chunk.height_grid;
        let step = chunk.step as usize;
        let chunk_size = hg.chunk_size as f64;

        // Calculate base (step=1) resolution parameters
        let base_cell = hg.cell_f32() / step as f32;
        let base_cell_f64 = base_cell as f64;
        let base_nx = (hg.nx - 1) * step + 1;
        let base_nz = (hg.nz - 1) * step + 1;

        // Compute chunk offset from center's chunk in f64 for precision
        let chunk_offset_x = (coord.x - center.chunk.x) as f64 * chunk_size;
        let chunk_offset_z = (coord.z - center.chunk.z) as f64 * chunk_size;
        let center_local_x = center.local.x as f64;
        let center_local_z = center.local.z as f64;

        // Grid bounds at base resolution (in local coordinates of this chunk)
        let radius_f64 = radius as f64;

        // Transform center to this chunk's local space for bounds calculation
        let center_in_local_x = -chunk_offset_x + center_local_x;
        let center_in_local_z = -chunk_offset_z + center_local_z;

        let min_base_gx = ((center_in_local_x - radius_f64) / base_cell_f64)
            .floor()
            .max(0.0) as usize;
        let max_base_gx = ((center_in_local_x + radius_f64) / base_cell_f64)
            .ceil()
            .min((base_nx - 1) as f64) as usize;
        let min_base_gz = ((center_in_local_z - radius_f64) / base_cell_f64)
            .floor()
            .max(0.0) as usize;
        let max_base_gz = ((center_in_local_z + radius_f64) / base_cell_f64)
            .ceil()
            .min((base_nz - 1) as f64) as usize;

        let edited = self.edited_chunks.get_mut(&coord).unwrap();
        let mut changed = false;

        for base_gx in min_base_gx..=max_base_gx {
            for base_gz in min_base_gz..=max_base_gz {
                // Local position within this chunk
                let local_x = (base_gx as f64) * base_cell_f64;
                let local_z = (base_gz as f64) * base_cell_f64;

                // Distance from brush center (computed in f64 then converted)
                let dx = (chunk_offset_x + local_x - center_local_x) as f32;
                let dz = (chunk_offset_z + local_z - center_local_z) as f32;
                let d2 = dx * dx + dz * dz;

                if d2 >= r2 {
                    continue;
                }

                let w = F::weight(d2, r2);
                if w <= 0.0001 {
                    continue;
                }

                // Sample current height using bilinear interpolation from actual grid
                let current_h = sample_height_at_local(hg, local_x as f32, local_z as f32);

                let mut new_h = current_h;
                B::apply(&mut new_h, strength, w);
                let delta = new_h - current_h;

                if delta.abs() < f32::EPSILON {
                    continue;
                }

                // Always store at base resolution
                let base_gx_u16 = base_gx as u16;
                let base_gz_u16 = base_gz as u16;
                *edited
                    .accumulated_deltas
                    .entry((base_gx_u16, base_gz_u16))
                    .or_insert(0.0) += delta;

                // Only add to pending_deltas if this aligns with current LOD grid
                if base_gx % step == 0 && base_gz % step == 0 {
                    let gx = (base_gx / step) as u16;
                    let gz = (base_gz / step) as u16;
                    edited.pending_deltas.push((gx, gz, delta));
                }

                changed = true;
            }
        }

        if changed {
            edited.dirty = true;
        }
    }
    pub(crate) fn upload_dirty_chunks(
        &mut self,
        device: &Device,
        queue: &Queue,
        arena: &mut TerrainMeshArena,
        chunks: &mut HashMap<ChunkCoord, ChunkMeshLod>,
        terrain_gen: &TerrainGenerator,
        scratch: &mut GeometryScratch<Vertex>,
        _finalize: bool,
    ) -> Vec<GpuChunkHandle> {
        let dirty_coords: Vec<_> = self
            .edited_chunks
            .iter()
            .filter(|(_, e)| e.dirty && !e.pending_deltas.is_empty())
            .map(|(c, _)| *c)
            .collect();

        let mut freed_handles = Vec::new();

        for coord in dirty_coords {
            if let Some(handle) =
                self.upload_single_chunk(coord, device, queue, arena, chunks, terrain_gen, scratch)
            {
                freed_handles.push(handle);
            }
        }

        freed_handles
    }

    fn upload_single_chunk(
        &mut self,
        coord: ChunkCoord,
        device: &Device,
        queue: &Queue,
        arena: &mut TerrainMeshArena,
        chunks: &mut HashMap<ChunkCoord, ChunkMeshLod>,
        terrain_gen: &TerrainGenerator,
        scratch: &mut GeometryScratch<Vertex>,
    ) -> Option<GpuChunkHandle> {
        let edited = self.edited_chunks.get_mut(&coord)?;
        edited.dirty = false;

        let pending = std::mem::take(&mut edited.pending_deltas);

        if pending.is_empty() {
            return None;
        }

        let (step, mut indices, mut grid, mut vertices) = match chunks.get(&coord) {
            Some(c) => (
                c.step,
                c.cpu_indices.clone(),
                (*c.height_grid).clone(),
                c.cpu_vertices.clone(),
            ),
            None => return None,
        };

        // Apply pending deltas
        for (gx, gz, delta) in &pending {
            let gx_usize = *gx as usize;
            let gz_usize = *gz as usize;
            if gx_usize < grid.nx && gz_usize < grid.nz {
                grid.heights[gx_usize * grid.nz + gz_usize] += *delta;
            }
        }

        recompute_patch_minmax(&mut grid);

        // FIX: Check if vertex layout is compatible with in-place update
        let expected_verts = grid.nx * grid.nz;

        if vertices.len() == expected_verts {
            // Fast path: simple grid layout, update in-place
            let neighbor_edges =
                gather_neighbor_edge_heights(coord, &grid, chunks, &self.edited_chunks);
            regenerate_vertices_from_height_grid(
                &mut vertices,
                &grid,
                terrain_gen,
                Some(&neighbor_edges),
                false,
            );
        } else {
            // Greedy-meshed chunk: rebuild with simple grid layout (once per chunk)
            (vertices, indices) = build_simple_grid_mesh(&grid, coord, terrain_gen);
        }

        let new_handle = arena.alloc_and_upload(device, queue, &vertices, &indices, scratch);
        let old_handle = chunks.remove(&coord).map(|c| c.handle);

        chunks.insert(
            coord,
            ChunkMeshLod {
                step,
                handle: new_handle,
                cpu_vertices: vertices,
                cpu_indices: indices,
                height_grid: Arc::new(grid),
            },
        );

        old_handle
    }
}
/// Build mesh with simple shared-vertex grid layout for edited chunks.
pub fn build_simple_grid_mesh(
    grid: &ChunkHeightGrid,
    coord: ChunkCoord,
    terrain_gen: &TerrainGenerator,
) -> (Vec<Vertex>, Vec<u32>) {
    let nx = grid.nx;
    let nz = grid.nz;
    let cell = grid.cell_f32();
    let step = grid.cell_size as usize;
    let chunk_size = grid.chunk_size;
    let inv = 1.0 / cell;

    let mut vertices = Vec::with_capacity(nx * nz);
    let mut indices = Vec::with_capacity((nx - 1) * (nz - 1) * 6);

    for gx in 0..nx {
        let local_x = (gx * step) as f32;
        for gz in 0..nz {
            let local_z = (gz * step) as f32;
            let h = grid.heights[gx * nz + gz];

            // Normal via central difference
            let h_l = if gx > 0 {
                grid.heights[(gx - 1) * nz + gz]
            } else {
                h
            };
            let h_r = if gx + 1 < nx {
                grid.heights[(gx + 1) * nz + gz]
            } else {
                h
            };
            let h_d = if gz > 0 {
                grid.heights[gx * nz + (gz - 1)]
            } else {
                h
            };
            let h_u = if gz + 1 < nz {
                grid.heights[gx * nz + (gz + 1)]
            } else {
                h
            };

            let dhdx = (h_r - h_l) * 0.5 * inv;
            let dhdz = (h_u - h_d) * 0.5 * inv;
            let n = glam::Vec3::new(-dhdx, 1.0, -dhdz).normalize();

            let pos = WorldPos::new(coord, LocalPos::new(local_x, h, local_z));
            let m = terrain_gen.moisture(&pos, h, chunk_size);
            let c = terrain_gen.color(&pos, h, m, chunk_size);

            vertices.push(Vertex {
                local_position: [local_x, h, local_z],
                normal: [n.x, n.y, n.z],
                color: c,
                chunk_xz: [coord.x, coord.z],
                quad_uv: [0.0, 0.0],
            });
        }
    }

    for gx in 0..(nx - 1) {
        for gz in 0..(nz - 1) {
            let i00 = (gx * nz + gz) as u32;
            let i10 = ((gx + 1) * nz + gz) as u32;
            let i01 = (gx * nz + (gz + 1)) as u32;
            let i11 = ((gx + 1) * nz + (gz + 1)) as u32;
            indices.extend_from_slice(&[i00, i10, i01, i01, i10, i11]);
        }
    }

    (vertices, indices)
}
/// Sample height at local position using bilinear interpolation.
/// `local_x` and `local_z` should be in range [0, chunk_size].
#[inline]
fn sample_height_at_local(hg: &ChunkHeightGrid, local_x: f32, local_z: f32) -> f32 {
    let cell = hg.cell_f32();

    let gx_f = local_x / cell;
    let gz_f = local_z / cell;

    let gx0 = (gx_f.floor() as usize).min(hg.nx.saturating_sub(1));
    let gz0 = (gz_f.floor() as usize).min(hg.nz.saturating_sub(1));
    let gx1 = (gx0 + 1).min(hg.nx - 1);
    let gz1 = (gz0 + 1).min(hg.nz - 1);

    let tx = (gx_f - gx0 as f32).clamp(0.0, 1.0);
    let tz = (gz_f - gz0 as f32).clamp(0.0, 1.0);

    let h00 = hg.heights[gx0 * hg.nz + gz0];
    let h10 = hg.heights[gx1 * hg.nz + gz0];
    let h01 = hg.heights[gx0 * hg.nz + gz1];
    let h11 = hg.heights[gx1 * hg.nz + gz1];

    let h0 = h00 + tx * (h10 - h00);
    let h1 = h01 + tx * (h11 - h01);

    h0 + tz * (h1 - h0)
}

pub fn recompute_patch_minmax(grid: &mut ChunkHeightGrid) {
    const PATCH_CELLS: usize = 8;
    let nx = grid.nx;
    let nz = grid.nz;
    let px = (nx - 1) / PATCH_CELLS;
    let pz = (nz - 1) / PATCH_CELLS;

    grid.patch_minmax.clear();
    grid.patch_minmax.reserve(px * pz);

    for px_i in 0..px {
        for pz_i in 0..pz {
            let mut min_y = f32::INFINITY;
            let mut max_y = f32::NEG_INFINITY;

            for lx in 0..=PATCH_CELLS {
                for lz in 0..=PATCH_CELLS {
                    let gx = px_i * PATCH_CELLS + lx;
                    let gz = pz_i * PATCH_CELLS + lz;
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
}

/// Apply accumulated deltas and handle LOD stitching correctly.
pub fn apply_accumulated_deltas_with_stitching(
    height_grid: &ChunkHeightGrid,
    coord: ChunkCoord,
    edited_chunks: &HashMap<ChunkCoord, EditedChunk>,
    loaded_chunks: &HashMap<ChunkCoord, ChunkMeshLod>,
    own_step: LodStep,
) -> ChunkHeightGrid {
    let mut grid = height_grid.clone();

    let has_own_edits = edited_chunks
        .get(&coord)
        .map_or(false, |e| !e.accumulated_deltas.is_empty());

    // Apply THIS chunk's accumulated deltas
    if let Some(edit) = edited_chunks.get(&coord) {
        for (&(base_x, base_z), &delta) in &edit.accumulated_deltas {
            // Check if this base coordinate aligns with current LOD grid
            if base_x % own_step != 0 || base_z % own_step != 0 {
                continue;
            }

            let gx = (base_x / own_step) as usize;
            let gz = (base_z / own_step) as usize;

            if gx < grid.nx && gz < grid.nz {
                grid.heights[gx * grid.nz + gz] += delta;
            }
        }
    }

    // If this chunk has edits, don't do any stitching - edits are authoritative
    if has_own_edits {
        return grid;
    }

    // Only stitch specific edges where neighbor is STRICTLY coarser and has no edits
    let neighbors = [
        (coord.offset(-1, 0), Edge::XNeg),
        (coord.offset(1, 0), Edge::XPos),
        (coord.offset(0, -1), Edge::ZNeg),
        (coord.offset(0, 1), Edge::ZPos),
    ];

    for (neighbor_coord, edge) in neighbors {
        // Skip if neighbor has edits
        let neighbor_has_edits = edited_chunks
            .get(&neighbor_coord)
            .map_or(false, |e| !e.accumulated_deltas.is_empty());

        if neighbor_has_edits {
            continue;
        }

        if let Some(neighbor_chunk) = loaded_chunks.get(&neighbor_coord) {
            // Only stitch THIS specific edge if neighbor is strictly coarser
            if neighbor_chunk.step > own_step {
                stitch_edge_to_neighbor(&mut grid, &neighbor_chunk.height_grid, edge);
            }
        }
    }

    grid
}

/// Stitch our X edge to match a neighbor's X edge.
/// Both grids use local coordinates [0, chunk_size].
fn stitch_x_edge(
    grid: &mut ChunkHeightGrid,
    neighbor_grid: &ChunkHeightGrid,
    our_gx: usize,
    neighbor_gx: usize,
) {
    let our_cell = grid.cell_f32();
    let neighbor_cell = neighbor_grid.cell_f32();

    for our_gz in 0..grid.nz {
        // Our local Z at this grid position
        let local_z = our_gz as f32 * our_cell;

        // Convert to neighbor's grid coordinates
        let neighbor_gz_f = local_z / neighbor_cell;
        let neighbor_gz_lo = (neighbor_gz_f.floor() as usize).min(neighbor_grid.nz - 1);
        let neighbor_gz_hi = (neighbor_gz_lo + 1).min(neighbor_grid.nz - 1);
        let t = (neighbor_gz_f - neighbor_gz_lo as f32).clamp(0.0, 1.0);

        let h_lo = neighbor_grid.heights[neighbor_gx * neighbor_grid.nz + neighbor_gz_lo];
        let h_hi = neighbor_grid.heights[neighbor_gx * neighbor_grid.nz + neighbor_gz_hi];

        grid.heights[our_gx * grid.nz + our_gz] = h_lo + t * (h_hi - h_lo);
    }
}

/// Stitch our Z edge to match a neighbor's Z edge.
fn stitch_z_edge(
    grid: &mut ChunkHeightGrid,
    neighbor_grid: &ChunkHeightGrid,
    our_gz: usize,
    neighbor_gz: usize,
) {
    let our_cell = grid.cell_f32();
    let neighbor_cell = neighbor_grid.cell_f32();

    for our_gx in 0..grid.nx {
        // Our local X at this grid position
        let local_x = our_gx as f32 * our_cell;

        // Convert to neighbor's grid coordinates
        let neighbor_gx_f = local_x / neighbor_cell;
        let neighbor_gx_lo = (neighbor_gx_f.floor() as usize).min(neighbor_grid.nx - 1);
        let neighbor_gx_hi = (neighbor_gx_lo + 1).min(neighbor_grid.nx - 1);
        let t = (neighbor_gx_f - neighbor_gx_lo as f32).clamp(0.0, 1.0);

        let h_lo = neighbor_grid.heights[neighbor_gx_lo * neighbor_grid.nz + neighbor_gz];
        let h_hi = neighbor_grid.heights[neighbor_gx_hi * neighbor_grid.nz + neighbor_gz];

        grid.heights[our_gx * grid.nz + our_gz] = h_lo + t * (h_hi - h_lo);
    }
}

/// Stitch a specific edge to match the neighbor's corresponding edge.
fn stitch_edge_to_neighbor(
    grid: &mut ChunkHeightGrid,
    neighbor_grid: &ChunkHeightGrid,
    edge: Edge,
) {
    match edge {
        Edge::XNeg => {
            // Our -X edge (gx=0) should match neighbor's +X edge (gx=nx-1)
            stitch_x_edge(grid, neighbor_grid, 0, neighbor_grid.nx - 1);
        }
        Edge::XPos => {
            // Our +X edge (gx=nx-1) should match neighbor's -X edge (gx=0)
            stitch_x_edge(grid, neighbor_grid, grid.nx - 1, 0);
        }
        Edge::ZNeg => {
            // Our -Z edge (gz=0) should match neighbor's +Z edge (gz=nz-1)
            stitch_z_edge(grid, neighbor_grid, 0, neighbor_grid.nz - 1);
        }
        Edge::ZPos => {
            // Our +Z edge (gz=nz-1) should match neighbor's -Z edge (gz=0)
            stitch_z_edge(grid, neighbor_grid, grid.nz - 1, 0);
        }
    }
}
