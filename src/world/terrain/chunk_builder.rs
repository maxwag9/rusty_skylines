use crate::helpers::positions::{ChunkCoord, LocalPos, LodStep, WorldPos, chunk_size};
use crate::ui::vertex::Vertex;
use crate::world::terrain::terrain_editing::{apply_edits_with_stitching, recompute_patch_minmax};
use crate::world::terrain::terrain_gen::TerrainGenerator;
use crate::world::terrain::terrain_subsystem::append_edge_skirts;
use crate::world::terrain::terrain_threads::{
    ChunkWorkerPool, LoadedChunksSnapshot, TerrainEditsSnapshot,
};
use glam::Vec3;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

#[derive(Clone)]
pub struct ChunkHeightGrid {
    pub chunk_coord: ChunkCoord,
    pub step: LodStep,
    pub nx: usize,
    pub nz: usize,
    pub heights: Vec<f32>,             // indexed as x * nz + z
    pub patch_minmax: Vec<(f32, f32)>, // 8x8 patches for culling
}

impl ChunkHeightGrid {
    #[inline]
    pub fn step_f32(&self) -> f32 {
        self.step as f32
    }

    /// Grid extent in local X direction
    #[inline]
    pub fn extent_x(&self) -> f32 {
        (self.nx - 1) as f32 * self.step_f32()
    }

    /// Grid extent in local Z direction
    #[inline]
    pub fn extent_z(&self) -> f32 {
        (self.nz - 1) as f32 * self.step_f32()
    }
    /// Convert a WorldPos to local coordinates relative to this grid's chunk.
    /// Returns (local_x, local_z) which may be outside [0, chunk_size] if pos is in a different chunk.
    #[inline]
    pub fn _world_to_local(&self, pos: &WorldPos) -> (f32, f32) {
        let cs = chunk_size() as f32;
        let chunk_offset_x = (pos.chunk.x - self.chunk_coord.x) as f32 * cs;
        let chunk_offset_z = (pos.chunk.z - self.chunk_coord.z) as f32 * cs;
        (chunk_offset_x + pos.local.x, chunk_offset_z + pos.local.z)
    }

    /// Convert local coordinates to a WorldPos (normalizing if needed).
    #[inline]
    pub fn _local_to_world(&self, local_x: f32, local_y: f32, local_z: f32) -> WorldPos {
        WorldPos::new(self.chunk_coord, LocalPos::new(local_x, local_y, local_z)).normalize()
    }

    /// Check if a WorldPos falls within this chunk's boundaries.
    #[inline]
    pub fn _contains(&self, pos: &WorldPos) -> bool {
        pos.chunk.x == self.chunk_coord.x && pos.chunk.z == self.chunk_coord.z
    }
}
#[derive(Clone)]
pub struct ChunkState {
    pub step: LodStep,

    pub nx_neg: LodStep,
    pub nx_pos: LodStep,
    pub nz_neg: LodStep,
    pub nz_pos: LodStep,
}
impl ChunkState {
    #[inline]
    pub fn same_as(&self, other: &ChunkState) -> bool {
        self.step == other.step
            && self.nx_neg == other.nx_neg
            && self.nx_pos == other.nx_pos
            && self.nz_neg == other.nz_neg
            && self.nz_pos == other.nz_pos
    }
}

pub struct ChunkMeshLod {
    pub state: ChunkState,
    pub handle: GpuChunkHandle,
    pub cpu_vertices: Vec<Vertex>,
    pub cpu_indices: Vec<u32>,
    pub height_grid: Arc<ChunkHeightGrid>,
}

/// Holds edge heights from neighboring chunks for proper normal calculation at boundaries
pub struct NeighborEdgeHeights {
    /// Heights along the -X edge of the +X neighbor (indexed by gz)
    pub pos_x: Option<Vec<f32>>,
    /// Heights along the +X edge of the -X neighbor (indexed by gz)
    pub neg_x: Option<Vec<f32>>,
    /// Heights along the -Z edge of the +Z neighbor (indexed by gx)
    pub pos_z: Option<Vec<f32>>,
    /// Heights along the +Z edge of the -Z neighbor (indexed by gx)
    pub neg_z: Option<Vec<f32>>,
}

impl NeighborEdgeHeights {
    pub fn empty() -> Self {
        Self {
            pos_x: None,
            neg_x: None,
            pos_z: None,
            neg_z: None,
        }
    }
}

/// Regenerate vertex positions (y), normals, and optionally colors from the height grid.
/// Vertices are expected to have LOCAL positions (relative to chunk origin).
pub fn regenerate_vertices_from_height_grid(
    vertices: &mut [Vertex],
    height_grid: &ChunkHeightGrid,
    terrain_gen: &TerrainGenerator,
    neighbor_edges: Option<&NeighborEdgeHeights>,
    update_colors: bool,
) {
    let verts_x = height_grid.nx;
    let verts_z = height_grid.nz;
    let cell = height_grid.step_f32();
    let chunk = height_grid.chunk_coord;

    // sanity check
    if vertices.len() != verts_x * verts_z {
        return;
    }

    // Update positions' y from grid
    for gx in 0..verts_x {
        for gz in 0..verts_z {
            let idx = gx * verts_z + gz;
            let h = height_grid.heights[idx];
            vertices[idx].local_position[1] = h;
        }
    }

    // Recompute normals using central differences
    let inv = 1.0 / cell;
    for gx in 0..verts_x {
        for gz in 0..verts_z {
            let idx = gx * verts_z + gz;

            let local_x = gx as f32 * cell;
            let local_z = gz as f32 * cell;

            // --- X Axis Gradient ---
            let h_l = if gx > 0 {
                height_grid.heights[(gx - 1) * verts_z + gz]
            } else {
                neighbor_edges
                    .and_then(|n| n.neg_x.as_ref())
                    .and_then(|edge| edge.get(gz).copied())
                    .unwrap_or_else(|| {
                        let pos = WorldPos::new(chunk, LocalPos::new(local_x - cell, 0.0, local_z))
                            .normalize();
                        terrain_gen.height(&pos)
                    })
            };

            let h_r = if gx + 1 < verts_x {
                height_grid.heights[(gx + 1) * verts_z + gz]
            } else {
                neighbor_edges
                    .and_then(|n| n.pos_x.as_ref())
                    .and_then(|edge| edge.get(gz).copied())
                    .unwrap_or_else(|| {
                        let pos = WorldPos::new(chunk, LocalPos::new(local_x + cell, 0.0, local_z))
                            .normalize();
                        terrain_gen.height(&pos)
                    })
            };

            // --- Z Axis Gradient ---
            let h_d = if gz > 0 {
                height_grid.heights[gx * verts_z + (gz - 1)]
            } else {
                neighbor_edges
                    .and_then(|n| n.neg_z.as_ref())
                    .and_then(|edge| edge.get(gx).copied())
                    .unwrap_or_else(|| {
                        let pos = WorldPos::new(chunk, LocalPos::new(local_x, 0.0, local_z - cell))
                            .normalize();
                        terrain_gen.height(&pos)
                    })
            };

            let h_u = if gz + 1 < verts_z {
                height_grid.heights[gx * verts_z + (gz + 1)]
            } else {
                neighbor_edges
                    .and_then(|n| n.pos_z.as_ref())
                    .and_then(|edge| edge.get(gx).copied())
                    .unwrap_or_else(|| {
                        let pos = WorldPos::new(chunk, LocalPos::new(local_x, 0.0, local_z + cell))
                            .normalize();
                        terrain_gen.height(&pos)
                    })
            };

            // Central Difference
            let dhdx = (h_r - h_l) * 0.5 * inv;
            let dhdz = (h_u - h_d) * 0.5 * inv;

            let n = Vec3::new(-dhdx, 1.0, -dhdz).normalize();
            vertices[idx].normal = [n.x, n.y, n.z];

            // Optionally update colors
            if update_colors {
                let v_pos = vertices[idx].local_position;
                let pos = WorldPos::new(chunk, LocalPos::new(v_pos[0], v_pos[1], v_pos[2]));
                let h = v_pos[1];
                let m = terrain_gen.moisture(&pos, h);
                vertices[idx].color = terrain_gen.color(&pos, h, m);
            }
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct GpuChunkHandle {
    pub base_vertex: i32,
    pub(crate) first_index_above: u32,
    pub(crate) index_count_above: u32,
    pub(crate) first_index_under: u32,
    pub(crate) index_count_under: u32,
    pub(crate) page: u8,
    pub(crate) vertex_count: u32,
}

pub struct ChunkBuilder;

pub struct CpuChunkMesh {
    pub chunk_coord: ChunkCoord,
    pub step: LodStep,
    pub version: u64,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub height_grid: Arc<ChunkHeightGrid>,
}

/// Pre-computed cell data for greedy meshing decisions
#[derive(Clone, Copy)]
struct CellData {
    _heights: [f32; 4], // Corner heights: [0,0], [1,0], [0,1], [1,1]
    color: [f32; 3],    // Average color for merging comparison
    flat_height: f32,   // Average height if flat
    is_flat: bool,      // Can this cell participate in greedy merge?
}

/// Represents a merged rectangular region
struct MergedQuad {
    x: usize,
    z: usize,
    width: usize,
    depth: usize,
    height: f32,
    color: [f32; 3],
}

impl ChunkBuilder {
    // Tunable thresholds for greedy merging
    const HEIGHT_TOLERANCE: f32 = 0.003; // Max height variance to consider "flat"
    const COLOR_TOLERANCE: f32 = 0.02; // Max color difference per channel

    pub fn build_chunk_cpu(
        chunk_coord: ChunkCoord,
        state: ChunkState,
        version: u64,
        version_atomic: &AtomicU64,
        terrain_gen: &TerrainGenerator,
        terrain_edits_snapshot: &TerrainEditsSnapshot,
        loaded_chunks_snapshot: &LoadedChunksSnapshot,
    ) -> Option<CpuChunkMesh> {
        let step = state.step;
        let stepf = step as f32;
        let step_usize = step as usize;
        let inv_step = 1.0 / stepf;
        let cs = chunk_size();

        let verts_x = (cs / step + 1) as usize;
        let verts_z = (cs / step + 1) as usize;
        let cells_x = verts_x - 1;
        let cells_z = verts_z - 1;
        let total_cells = cells_x * cells_z;

        let (mut heights, colors) = Self::sample_terrain_batch(chunk_coord, step, terrain_gen);

        if !ChunkWorkerPool::still_current(version_atomic, version) {
            return None;
        }

        // Build initial height grid from sampled heights
        let mut height_grid = build_height_grid_from_heights(chunk_coord, step, heights);

        // Apply edits + stitching ON WORKER
        height_grid = apply_edits_with_stitching(
            &height_grid,
            chunk_coord,
            terrain_edits_snapshot,
            loaded_chunks_snapshot,
            step,
        );

        recompute_patch_minmax(&mut height_grid);

        heights = height_grid.heights.clone();

        let normals = Self::compute_normals_batch(
            chunk_coord,
            step,
            stepf,
            inv_step,
            verts_x,
            verts_z,
            &heights,
            terrain_gen,
        );

        if !ChunkWorkerPool::still_current(version_atomic, version) {
            return None;
        }

        let has_edits = terrain_edits_snapshot.has_edits_on_chunk(chunk_coord);

        let (mut vertices, mut indices) = if has_edits {
            Self::build_simple_grid(
                chunk_coord,
                step_usize,
                verts_x,
                verts_z,
                &heights,
                &colors,
                &normals,
            )
        } else {
            Self::build_greedy_mesh(
                chunk_coord,
                step_usize,
                verts_x,
                verts_z,
                cells_x,
                cells_z,
                total_cells,
                &heights,
                &colors,
                &normals,
                version,
                version_atomic,
            )?
        };

        append_edge_skirts(
            terrain_gen,
            &mut vertices,
            &mut indices,
            &height_grid,
            chunk_coord,
        );

        Some(CpuChunkMesh {
            chunk_coord,
            step,
            version,
            vertices,
            indices,
            height_grid: Arc::new(height_grid),
        })
    }

    /// Build a simple grid mesh where vertices[gx * nz + gz] = vertex at grid position (gx, gz).
    /// This layout is required for in-place height updates on edited chunks.
    pub fn build_simple_grid(
        chunk_coord: ChunkCoord,
        step: usize,
        verts_x: usize,
        verts_z: usize,
        heights: &[f32],
        colors: &[[f32; 3]],
        normals: &[[f32; 3]],
    ) -> (Vec<Vertex>, Vec<u32>) {
        let total_verts = verts_x * verts_z;
        let cells_x = verts_x - 1;
        let cells_z = verts_z - 1;

        let mut vertices = Vec::with_capacity(total_verts);
        let mut indices = Vec::with_capacity(cells_x * cells_z * 6);

        let step_f = step as f32;

        // Emit vertices in row-major order: gx * verts_z + gz
        for gx in 0..verts_x {
            for gz in 0..verts_z {
                let idx = gx * verts_z + gz;

                let local_x = gx as f32 * step_f;
                let local_z = gz as f32 * step_f;

                vertices.push(Vertex {
                    local_position: [local_x, heights[idx], local_z],
                    normal: normals[idx],
                    color: colors[idx],
                    chunk_xz: [chunk_coord.x, chunk_coord.z],
                    quad_uv: [1.0, 0.0],
                });
            }
        }

        // Emit indices for each cell
        for cx in 0..cells_x {
            for cz in 0..cells_z {
                let v00 = (cx * verts_z + cz) as u32;
                let v10 = ((cx + 1) * verts_z + cz) as u32;
                let v01 = (cx * verts_z + (cz + 1)) as u32;
                let v11 = ((cx + 1) * verts_z + (cz + 1)) as u32;

                // Two triangles per cell
                indices.extend_from_slice(&[v00, v10, v11, v00, v11, v01]);
            }
        }

        (vertices, indices)
    }

    /// Build a greedy-meshed geometry for non-edited chunks.
    fn build_greedy_mesh(
        chunk_coord: ChunkCoord,
        step_usize: usize,
        _verts_x: usize,
        verts_z: usize,
        cells_x: usize,
        cells_z: usize,
        total_cells: usize,
        heights: &[f32],
        colors: &[[f32; 3]],
        normals: &[[f32; 3]],
        version: u64,
        version_atomic: &AtomicU64,
    ) -> Option<(Vec<Vertex>, Vec<u32>)> {
        // Build cell classification for greedy meshing
        let cells = Self::build_cell_data(cells_x, cells_z, verts_z, heights, colors);

        // Greedy meshing - merge flat regions
        let mut merged = vec![false; total_cells];
        let mut merged_quads: Vec<MergedQuad> = Vec::new();
        let mut non_flat_cells: Vec<(usize, usize)> = Vec::new();

        let cell_idx = |x: usize, z: usize| x * cells_z + z;

        for cx in 0..cells_x {
            if cx & 0xF == 0 && !ChunkWorkerPool::still_current(version_atomic, version) {
                return None;
            }

            for cz in 0..cells_z {
                let idx = cell_idx(cx, cz);
                if merged[idx] {
                    continue;
                }

                let cell = &cells[idx];

                if cell.is_flat {
                    // Greedy expansion
                    let (width, depth) = Self::find_max_rect(
                        &cells,
                        &merged,
                        cx,
                        cz,
                        cells_x,
                        cells_z,
                        cell.flat_height,
                        &cell.color,
                    );

                    // Mark all cells in the rectangle as merged
                    for dx in 0..width {
                        for dz in 0..depth {
                            merged[cell_idx(cx + dx, cz + dz)] = true;
                        }
                    }

                    merged_quads.push(MergedQuad {
                        x: cx,
                        z: cz,
                        width,
                        depth,
                        height: cell.flat_height,
                        color: cell.color,
                    });
                } else {
                    merged[idx] = true;
                    non_flat_cells.push((cx, cz));
                }
            }
        }

        // Generate optimized mesh
        let estimated_verts = merged_quads.len() * 4 + non_flat_cells.len() * 4;
        let estimated_indices = merged_quads.len() * 6 + non_flat_cells.len() * 6;

        let mut vertices = Vec::with_capacity(estimated_verts);
        let mut indices = Vec::with_capacity(estimated_indices);

        // Emit merged flat quads
        for quad in &merged_quads {
            Self::emit_merged_quad(&mut vertices, &mut indices, chunk_coord, step_usize, quad);
        }

        // Emit non-flat cells with full vertex data
        for &(cx, cz) in &non_flat_cells {
            Self::emit_detailed_cell(
                &mut vertices,
                &mut indices,
                chunk_coord,
                step_usize,
                cx,
                cz,
                verts_z,
                heights,
                colors,
                normals,
            );
        }

        vertices.shrink_to_fit();
        indices.shrink_to_fit();

        Some((vertices, indices))
    }

    #[inline]
    fn sample_terrain_batch(
        chunk_coord: ChunkCoord,
        step: LodStep,
        terrain_gen: &TerrainGenerator,
    ) -> (Vec<f32>, Vec<[f32; 3]>) {
        let cs = chunk_size();
        let step_usize = step as usize;
        let verts_x = (cs / step + 1) as usize;
        let verts_z = (cs / step + 1) as usize;
        let total = verts_x * verts_z;

        let mut heights = Vec::with_capacity(total);
        let mut colors = Vec::with_capacity(total);

        for gx in 0..verts_x {
            let local_x = (gx * step_usize) as f32;

            for gz in 0..verts_z {
                let local_z = (gz * step_usize) as f32;
                let world_pos = WorldPos::new(chunk_coord, LocalPos::new(local_x, 0.0, local_z));

                let h = terrain_gen.height(&world_pos);
                let m = terrain_gen.moisture(&world_pos, h);
                let c = terrain_gen.color(&world_pos, h, m);

                heights.push(h);
                colors.push(c);
            }
        }

        (heights, colors)
    }

    // ═══════════════════════════════════════════════════════════════════
    // NORMAL COMPUTATION
    // ═══════════════════════════════════════════════════════════════════

    fn compute_normals_batch(
        chunk_coord: ChunkCoord,
        step: LodStep,
        stepf: f32,
        inv_step: f32,
        verts_x: usize,
        verts_z: usize,
        heights: &[f32],
        terrain_gen: &TerrainGenerator,
    ) -> Vec<[f32; 3]> {
        let total = verts_x * verts_z;
        let mut normals = vec![[0.0f32, 1.0, 0.0]; total];
        let step_usize = step as usize;

        for gx in 0..verts_x {
            let local_x = (gx * step_usize) as f32;

            for gz in 0..verts_z {
                let local_z = (gz * step_usize) as f32;
                let idx = gx * verts_z + gz;

                // Sample neighbors (with chunk boundary lookups)
                let h_left = if gx > 0 {
                    heights[(gx - 1) * verts_z + gz]
                } else {
                    Self::sample_neighbor_height(chunk_coord, terrain_gen, local_x - stepf, local_z)
                };

                let h_right = if gx < verts_x - 1 {
                    heights[(gx + 1) * verts_z + gz]
                } else {
                    Self::sample_neighbor_height(chunk_coord, terrain_gen, local_x + stepf, local_z)
                };

                let h_back = if gz > 0 {
                    heights[gx * verts_z + (gz - 1)]
                } else {
                    Self::sample_neighbor_height(chunk_coord, terrain_gen, local_x, local_z - stepf)
                };

                let h_front = if gz < verts_z - 1 {
                    heights[gx * verts_z + (gz + 1)]
                } else {
                    Self::sample_neighbor_height(chunk_coord, terrain_gen, local_x, local_z + stepf)
                };

                // Central difference
                let dhdx = (h_right - h_left) * 0.5 * inv_step;
                let dhdz = (h_front - h_back) * 0.5 * inv_step;

                let n = Vec3::new(-dhdx, 1.0, -dhdz).normalize();
                normals[idx] = [n.x, n.y, n.z];
            }
        }

        normals
    }

    #[inline]
    fn sample_neighbor_height(
        chunk_coord: ChunkCoord,
        terrain_gen: &TerrainGenerator,
        local_x: f32,
        local_z: f32,
    ) -> f32 {
        let pos = WorldPos::new(chunk_coord, LocalPos::new(local_x, 0.0, local_z)).normalize();
        terrain_gen.height(&pos)
    }

    fn build_cell_data(
        cells_x: usize,
        cells_z: usize,
        verts_z: usize,
        heights: &[f32],
        colors: &[[f32; 3]],
    ) -> Vec<CellData> {
        let total_cells = cells_x * cells_z;
        let mut cells = Vec::with_capacity(total_cells);

        for cx in 0..cells_x {
            for cz in 0..cells_z {
                let i00 = cx * verts_z + cz;
                let i10 = (cx + 1) * verts_z + cz;
                let i01 = cx * verts_z + (cz + 1);
                let i11 = (cx + 1) * verts_z + (cz + 1);

                let h = [heights[i00], heights[i10], heights[i01], heights[i11]];

                // Fast min/max without branching
                let min_h = h[0].min(h[1]).min(h[2]).min(h[3]);
                let max_h = h[0].max(h[1]).max(h[2]).max(h[3]);
                let height_range = max_h - min_h;
                let is_flat = height_range <= Self::HEIGHT_TOLERANCE;

                // Compute average color
                let c00 = colors[i00];
                let c10 = colors[i10];
                let c01 = colors[i01];
                let c11 = colors[i11];

                let avg_color = [
                    (c00[0] + c10[0] + c01[0] + c11[0]) * 0.25,
                    (c00[1] + c10[1] + c01[1] + c11[1]) * 0.25,
                    (c00[2] + c10[2] + c01[2] + c11[2]) * 0.25,
                    //(c00[3] + c10[3] + c01[3] + c11[3]) * 0.25,
                ];

                cells.push(CellData {
                    _heights: h,
                    color: avg_color,
                    flat_height: (min_h + max_h) * 0.5,
                    is_flat,
                });
            }
        }

        cells
    }

    fn find_max_rect(
        cells: &[CellData],
        merged: &[bool],
        start_x: usize,
        start_z: usize,
        cells_x: usize,
        cells_z: usize,
        ref_height: f32,
        ref_color: &[f32; 3],
    ) -> (usize, usize) {
        let cell_idx = |x: usize, z: usize| x * cells_z + z;

        // Step 1: Expand in Z direction (find max depth)
        let mut depth = 1usize;
        while start_z + depth < cells_z {
            let idx = cell_idx(start_x, start_z + depth);
            if !Self::can_merge_cell(merged, cells, idx, ref_height, ref_color) {
                break;
            }
            depth += 1;
        }

        // Step 2: Expand in X direction (must validate entire column each time)
        let mut width = 1usize;
        'expand_x: while start_x + width < cells_x {
            // Check all cells in this column within our depth
            for dz in 0..depth {
                let idx = cell_idx(start_x + width, start_z + dz);
                if !Self::can_merge_cell(merged, cells, idx, ref_height, ref_color) {
                    break 'expand_x;
                }
            }
            width += 1;
        }

        (width, depth)
    }

    #[inline(always)]
    fn can_merge_cell(
        merged: &[bool],
        cells: &[CellData],
        idx: usize,
        ref_height: f32,
        ref_color: &[f32; 3],
    ) -> bool {
        if merged[idx] {
            return false;
        }

        let cell = &cells[idx];

        cell.is_flat
            && (cell.flat_height - ref_height).abs() <= Self::HEIGHT_TOLERANCE
            && Self::colors_match(ref_color, &cell.color)
    }

    #[inline(always)]
    fn colors_match(a: &[f32; 3], b: &[f32; 3]) -> bool {
        let dr = (a[0] - b[0]).abs();
        let dg = (a[1] - b[1]).abs();
        let db = (a[2] - b[2]).abs();

        dr <= Self::COLOR_TOLERANCE && dg <= Self::COLOR_TOLERANCE && db <= Self::COLOR_TOLERANCE
    }

    #[inline]
    fn emit_merged_quad(
        vertices: &mut Vec<Vertex>,
        indices: &mut Vec<u32>,
        chunk_coord: ChunkCoord,
        step: usize,
        quad: &MergedQuad,
    ) {
        let base = vertices.len() as u32;

        let x0 = (quad.x * step) as f32;
        let z0 = (quad.z * step) as f32;
        let x1 = ((quad.x + quad.width) * step) as f32;
        let z1 = ((quad.z + quad.depth) * step) as f32;

        let normal = [0.0, 1.0, 0.0];
        let chunk_xz = [chunk_coord.x, chunk_coord.z];
        let h = quad.height;
        let c = quad.color;

        // 4 vertices with quad-local UVs for edge detection
        vertices.extend_from_slice(&[
            Vertex {
                local_position: [x0, h, z0],
                normal,
                color: c,
                chunk_xz,
                quad_uv: [0.0, 0.0],
            },
            Vertex {
                local_position: [x1, h, z0],
                normal,
                color: c,
                chunk_xz,
                quad_uv: [1.0, 0.0],
            },
            Vertex {
                local_position: [x0, h, z1],
                normal,
                color: c,
                chunk_xz,
                quad_uv: [0.0, 1.0],
            },
            Vertex {
                local_position: [x1, h, z1],
                normal,
                color: c,
                chunk_xz,
                quad_uv: [1.0, 1.0],
            },
        ]);

        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 1, base + 3]);
    }

    #[inline]
    fn emit_detailed_cell(
        vertices: &mut Vec<Vertex>,
        indices: &mut Vec<u32>,
        chunk_coord: ChunkCoord,
        step: usize,
        cx: usize,
        cz: usize,
        verts_z: usize,
        heights: &[f32],
        colors: &[[f32; 3]],
        normals: &[[f32; 3]],
    ) {
        let base = vertices.len() as u32;

        let x0 = (cx * step) as f32;
        let z0 = (cz * step) as f32;
        let x1 = ((cx + 1) * step) as f32;
        let z1 = ((cz + 1) * step) as f32;

        let i00 = cx * verts_z + cz;
        let i10 = (cx + 1) * verts_z + cz;
        let i01 = cx * verts_z + (cz + 1);
        let i11 = (cx + 1) * verts_z + (cz + 1);

        let chunk_xz = [chunk_coord.x, chunk_coord.z];

        vertices.extend_from_slice(&[
            Vertex {
                local_position: [x0, heights[i00], z0],
                normal: normals[i00],
                color: colors[i00],
                chunk_xz,
                quad_uv: [0.0, 0.0],
            },
            Vertex {
                local_position: [x1, heights[i10], z0],
                normal: normals[i10],
                color: colors[i10],
                chunk_xz,
                quad_uv: [1.0, 0.0],
            },
            Vertex {
                local_position: [x0, heights[i01], z1],
                normal: normals[i01],
                color: colors[i01],
                chunk_xz,
                quad_uv: [0.0, 1.0],
            },
            Vertex {
                local_position: [x1, heights[i11], z1],
                normal: normals[i11],
                color: colors[i11],
                chunk_xz,
                quad_uv: [1.0, 1.0],
            },
        ]);

        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 1, base + 3]);
    }
}

pub fn lod_step_for_distance(dist2_chunks: i32) -> LodStep {
    if dist2_chunks <= 0 {
        return 1;
    }

    // Scale thresholds to maintain consistent world-space LOD boundaries.
    // Reference size is 128; smaller chunks get proportionally larger thresholds.
    // For chunk_size > 128, minimum thresholds ensure at least 9 chunks at LOD 1.
    let cs = chunk_size() as i32;
    let scale = (128 / cs).max(1);
    let scale2 = scale * scale;

    // Each LOD level covers ~2x the world-space distance of the previous.
    // Thresholds quadruple per level (distance² relationship).
    // Base thresholds calibrated for chunk_size=128:
    //   LOD 1: dist² ≤ 2  → center + 8 neighbors (~181 world units)
    //   LOD 2: dist² ≤ 8  → ~362 world units (2x)
    //   LOD 4: dist² ≤ 32 → ~724 world units (2x)
    //   etc.
    if dist2_chunks <= 2 * scale2 {
        1
    } else if dist2_chunks <= 8 * scale2 {
        2
    } else if dist2_chunks <= 32 * scale2 {
        4
    } else if dist2_chunks <= 128 * scale2 {
        8
    } else if dist2_chunks <= 512 * scale2 {
        16
    } else if dist2_chunks <= 2048 * scale2 {
        32
    } else if dist2_chunks <= 8192 * scale2 {
        64
    } else {
        128
    }
}

fn _density_from_chunk_dist2(dist2_chunks: i32) -> f32 {
    let d = (dist2_chunks as f32).sqrt(); // distance in chunks
    let t = (d / 8.0).clamp(0.0, 1.0); // 8 chunks = full fade
    1.0 - t * t * (3.0 - 2.0 * t)
}

pub fn generate_spiral_offsets(radius: i32) -> Vec<ChunkCoord> {
    let mut v: Vec<ChunkCoord> = Vec::new();
    for dx in -radius..=radius {
        for dz in -radius..=radius {
            v.push(ChunkCoord::new(dx, dz));
        }
    }

    // sort by distance from center
    v.sort_by_key(|chunk_coord| chunk_coord.x * chunk_coord.x + chunk_coord.z * chunk_coord.z);
    v
}
pub fn generate_height_grid(
    chunk_coord: ChunkCoord,
    step: LodStep,
    terrain_gen: &TerrainGenerator,
) -> ChunkHeightGrid {
    let cs = chunk_size();
    let step_usize = step as usize;
    let verts_x = (cs / step + 1) as usize;
    let verts_z = (cs / step + 1) as usize;
    let total = verts_x * verts_z;

    let mut heights = Vec::with_capacity(total);

    for gx in 0..verts_x {
        let local_x = (gx * step_usize) as f32;

        for gz in 0..verts_z {
            let local_z = (gz * step_usize) as f32;
            let world_pos = WorldPos::new(chunk_coord, LocalPos::new(local_x, 0.0, local_z));

            let h = terrain_gen.height(&world_pos);

            heights.push(h);
        }
    }

    build_height_grid_from_heights(chunk_coord, step, heights)
}

fn build_height_grid_from_heights(
    chunk_coord: ChunkCoord,
    step: LodStep,
    heights: Vec<f32>,
) -> ChunkHeightGrid {
    let cs = chunk_size();
    let nx = (cs / step + 1) as usize;
    let nz = (cs / step + 1) as usize;
    const PATCH_CELLS: usize = 8;

    let px = (nx - 1) / PATCH_CELLS;
    let pz = (nz - 1) / PATCH_CELLS;

    let mut patch_minmax = Vec::with_capacity(px * pz);

    for px_i in 0..px {
        for pz_i in 0..pz {
            let mut min_y = f32::INFINITY;
            let mut max_y = f32::NEG_INFINITY;

            for lx in 0..=PATCH_CELLS {
                for lz in 0..=PATCH_CELLS {
                    let gx = px_i * PATCH_CELLS + lx;
                    let gz = pz_i * PATCH_CELLS + lz;

                    if gx < nx && gz < nz {
                        let h = heights[gx * nz + gz];
                        min_y = min_y.min(h);
                        max_y = max_y.max(h);
                    }
                }
            }

            patch_minmax.push((min_y, max_y));
        }
    }

    ChunkHeightGrid {
        chunk_coord,
        step,
        nx,
        nz,
        heights,
        patch_minmax,
    }
}
