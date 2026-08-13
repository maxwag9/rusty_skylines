use std::collections::HashSet;
use crate::helpers::positions::{chunk_size, ChunkCoord, LocalPos, LodStep, WorldPos};
use crate::renderer::gizmo::gizmo::{push_gizmo_renders, Gizmo};
use crate::renderer::props::{ArchetypeId, PropInstance, Props};
use crate::ui::vertex::Vertex;
use crate::world::terrain::terrain_editing::{apply_edits_with_stitching, recompute_patch_minmax};
use crate::world::terrain::terrain_gen::TerrainGenerator;
use crate::world::terrain::terrain_subsystem::append_edge_skirts;
use crate::world::terrain::terrain_threads::{
    ChunkWorkerPool, LoadedChunksSnapshot, TerrainEditsSnapshot,
};
use fastnoise_lite::{FastNoiseLite, NoiseType};
use glam::Vec3;
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};
use std::mem::take;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter};

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
                        let pos = WorldPos::new(chunk, LocalPos::new(local_x, 0.0, local_z + cell)).normalize();
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
pub struct GpuChunkHandle {
    pub base_vertex: i32,
    pub first_index_above: u32,
    pub index_count_above: u32,
    pub first_index_under: u32,
    pub index_count_under: u32,
    pub page: usize,
    pub vertex_count: u32
}

pub struct ChunkBuilder;

pub struct CpuChunkMesh {
    pub chunk_coord: ChunkCoord,
    pub step: LodStep,
    pub version: u64,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub height_grid: Arc<ChunkHeightGrid>,
    pub tree_placements: Vec<PropInstance>,
    pub archetypes: Vec<String>
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
        tree_spawning_params: TreeSpawningParams
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

        let (heights, moistures, colors) = Self::sample_terrain_batch(chunk_coord, step, terrain_gen);

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

        let heights = height_grid.heights.as_slice();

        let normals = Self::compute_normals_batch(
            chunk_coord,
            step,
            stepf,
            inv_step,
            verts_x,
            verts_z,
            heights,
            terrain_gen,
        );

        if !ChunkWorkerPool::still_current(version_atomic, version) {
            return None;
        }

        let has_edits = terrain_edits_snapshot.has_edits_on_chunk(chunk_coord);
        let (tree_placements, archetypes) = if has_edits {
            // conservative: skip auto-trees on hand-edited terrain to avoid
            // floating/clipping trees where the player flattened ground
            (Vec::new(), Vec::new())
        } else {
            let veg_samples = TreeSpawner::gather_vegetation_samples(chunk_coord, terrain_gen);
            terrain_gen.tree_spawner.spawn_trees_for_chunk(chunk_coord, &veg_samples, terrain_gen, tree_spawning_params)
        };
        //println!("Trees: {}", tree_placements.len());
        let (mut vertices, mut indices) = if has_edits {
            Self::build_simple_grid(
                chunk_coord,
                step_usize,
                verts_x,
                verts_z,
                heights,
                &colors,
                &normals
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
                heights,
                &colors,
                &normals,
                version,
                version_atomic
            )?
        };

        append_edge_skirts(
            terrain_gen,
            &mut vertices,
            &mut indices,
            &height_grid,
            chunk_coord
        );

        Some(CpuChunkMesh {
            chunk_coord,
            step,
            version,
            vertices,
            indices,
            height_grid: Arc::new(height_grid),
            tree_placements,
            archetypes
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

    // Expensive as fuck!!
    #[inline]
    fn sample_terrain_batch(
        chunk_coord: ChunkCoord,
        step: LodStep,
        terrain_gen: &TerrainGenerator,
    ) -> (Vec<f32>, Vec<f32>, Vec<[f32; 3]>) {
        let cs = chunk_size();
        let step_usize = step as usize;
        let verts_x = (cs / step + 1) as usize;
        let verts_z = (cs / step + 1) as usize;
        let total = verts_x * verts_z;

        let mut heights = Vec::with_capacity(total);
        let mut moistures = Vec::with_capacity(total);
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
                moistures.push(m);
                colors.push(c);
            }
        }

        (heights, moistures, colors)
    }

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
    let dist2_chunks = dist2_chunks as f32;
    // Scale thresholds to maintain consistent world-space LOD boundaries.
    // Reference size is 128; smaller chunks get proportionally larger thresholds.
    // For chunk_size > 128, minimum thresholds ensure at least 9 chunks at LOD 1.
    let cs = chunk_size() as f32;
    let scale = (128.0 / cs).max(0.01);
    let scale2 = scale * scale;

    // Each LOD level covers ~2x the world-space distance of the previous.
    // Thresholds quadruple per level (distance² relationship).
    // Base thresholds calibrated for chunk_size=128:
    //   LOD 1: dist² ≤ 2  → center + 8 neighbors (~181 world units)
    //   LOD 2: dist² ≤ 8  → ~362 world units (2x)
    //   LOD 4: dist² ≤ 32 → ~724 world units (2x)
    //   etc.
    if dist2_chunks <= 2.0 {
        1
    } else if dist2_chunks <= 8.0 * scale2 {
        2
    } else if dist2_chunks <= 32.0 * scale2 {
        4
    } else if dist2_chunks <= 128.0 * scale2 {
        8
    } else if dist2_chunks <= 512.0 * scale2 {
        16
    } else if dist2_chunks <= 2048.0 * scale2 {
        32
    } else if dist2_chunks <= 8192.0 * scale2 {
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

#[derive(Clone, Copy, Debug, Display, EnumIter, Eq, Hash, PartialEq)]
pub enum TreeKind {
    Oak,
    Pine,
    Birch,
    DeadTree,
}

pub const VEG_GRID_SIZE: usize = 32;

#[derive(Clone, Copy)]
pub struct VegetationSample {
    pub position: LocalPos, // .y = terrain height at this sample
    pub moisture: f32,
    pub normal: [f32; 3],
}
#[derive(Clone, Copy)]
pub struct TreeSpawningParams {
    pub forest_cluster_strength: f32,
    pub dense_forests: bool
}
#[derive(Clone)]
pub struct TreeSpawner {
    pub kind_to_archetype_id: Vec<ArchetypeId>,

    /// Broad deterministic forest map.
    forest_noise: Arc<FastNoiseLite>,
    /// Smaller deterministic carving noise to break up blobs.
    forest_carve_noise: Arc<FastNoiseLite>,
}

impl TreeSpawner {
    pub fn new(props: &Props) -> TreeSpawner {
        let mut kind_to_archetype_id: Vec<ArchetypeId> = Vec::new();
        for kind in TreeKind::iter() {
            let Some(archetype_id) = props.get_archetype_id_for_name(kind.to_string().as_str()) else { continue };
            kind_to_archetype_id.push(archetype_id);
        }

        let mut forest_noise = FastNoiseLite::new();
        forest_noise.set_noise_type(Some(NoiseType::OpenSimplex2));

        let mut forest_carve_noise = FastNoiseLite::new();
        forest_carve_noise.set_noise_type(Some(NoiseType::OpenSimplex2));

        TreeSpawner {
            kind_to_archetype_id,
            forest_noise: Arc::new(forest_noise),
            forest_carve_noise: Arc::new(forest_carve_noise),
        }
    }

    pub fn gather_vegetation_samples(
        chunk_coord: ChunkCoord,
        terrain_gen: &TerrainGenerator,
    ) -> Vec<VegetationSample> {
        //let gizmo = &mut Gizmo::new_empty();
        let cs = chunk_size() as f32;
        let cell_size = cs / VEG_GRID_SIZE as f32;
        let mut samples = Vec::with_capacity(VEG_GRID_SIZE * VEG_GRID_SIZE);

        for gx in 0..VEG_GRID_SIZE {
            for gz in 0..VEG_GRID_SIZE {
                let x = (gx as f32 + 0.5) * cell_size;
                let z = (gz as f32 + 0.5) * cell_size;

                let wp = WorldPos::new(chunk_coord, LocalPos::new(x, 0.0, z));
                let h = terrain_gen.height(&wp);
                let m = terrain_gen.moisture(&wp, h);
                let n = Self::normal_at(terrain_gen, chunk_coord, x, z, cell_size);
                let local_pos = LocalPos::new(x, h, z);
                //gizmo.cross(WorldPos::new(chunk_coord, local_pos), 20.0, [1.0, 0.0, 0.0, 1.0], 0.0, 10.0);

                samples.push(VegetationSample {
                    position: local_pos,
                    moisture: m,
                    normal: n,
                });
            }
        }

        //println!("gether vegetation samples gizmo pending render length: {}", gizmo.pending_renders.len());
        //push_gizmo_renders(take(&mut gizmo.pending_renders));

        samples
    }

    fn normal_at(
        terrain_gen: &TerrainGenerator,
        chunk_coord: ChunkCoord,
        x: f32,
        z: f32,
        cell_size: f32,
    ) -> [f32; 3] {
        let eps = (cell_size * 0.25).max(0.1);
        let h = |x: f32, z: f32| {
            terrain_gen.height(&WorldPos::new(chunk_coord, LocalPos::new(x, 0.0, z)))
        };

        let dx = (h(x + eps, z) - h(x - eps, z)) / (2.0 * eps);
        let dz = (h(x, z + eps) - h(x, z - eps)) / (2.0 * eps);

        let n = [-dx, 1.0, -dz];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-6);
        [n[0] / len, n[1] / len, n[2] / len]
    }

    pub fn spawn_trees_for_chunk(
        &self,
        chunk_coord: ChunkCoord,
        veg_samples: &[VegetationSample],
        terrain_gen: &TerrainGenerator,
        tree_spawning_params: TreeSpawningParams,
    ) -> (Vec<PropInstance>, Vec<String>) {
        debug_assert_eq!(veg_samples.len(), VEG_GRID_SIZE * VEG_GRID_SIZE);
        let gizmo = &mut Gizmo::new_empty();
        let cs = chunk_size() as f32;
        let cell_size = cs / VEG_GRID_SIZE as f32;
        let mut rng = Self::rng_for_chunk(chunk_coord);
        let mut placements = Vec::new();
        let mut archetypes: HashSet<String> = HashSet::new();
        let strength = tree_spawning_params.forest_cluster_strength.clamp(0.0, 1.0);

        for sample in veg_samples {
            if sample.normal[1] < 0.75 {
                continue;
            }

            let h = sample.position.y;
            let m = sample.moisture;

            let base_density = Self::tree_density(h, m);
            if base_density <= 0.0 {
                continue;
            }

            let world_x = chunk_coord.x as f32 * cs + sample.position.x;
            let world_z = chunk_coord.z as f32 * cs + sample.position.z;

            let forest_blob: f32 = self.forest_blob_factor(world_x, world_z, h, m);
            // min = min.min(forest_blob);
            // max = max.max(forest_blob);
            // sum += forest_blob;
            //
            // let color = [
            //     forest_blob,
            //     0.0,
            //     1.0 - forest_blob,
            //     1.0
            // ];
            //
            // gizmo.cross(
            //     WorldPos::new(chunk_coord, sample.position),
            //     2.0,
            //     color,
            //     0.0,
            //     20.0,
            // );
            // Hard gate when strength is high.
            // This is what stops the "tree blanket".
            let mut density = base_density;
            density *= 1.0 - strength + forest_blob * strength;

            // Strong clustering should make plains mostly empty.
            // if forest_blob < 0.35 {
            //     continue;
            // }

            // If the blob is strong, let it spawn multiple trees.
            let mut tree_count = 1usize;
            if tree_spawning_params.dense_forests && strength > 0.0 {
                let extra = (forest_blob * forest_blob * 6.0 * strength).floor() as usize;
                tree_count += extra;
            }

            for _ in 0..tree_count {
                if rng.random::<f32>() > density {
                    continue;
                }

                let jitter_x = rng.random_range(-cell_size * 0.5..cell_size * 0.5);
                let jitter_z = rng.random_range(-cell_size * 0.5..cell_size * 0.5);

                let local_x = (sample.position.x + jitter_x).clamp(0.0, cs - 0.01);
                let local_z = (sample.position.z + jitter_z).clamp(0.0, cs - 0.01);

                let h_exact = terrain_gen.height(&WorldPos::new(
                    chunk_coord,
                    LocalPos::new(local_x, 0.0, local_z),
                ));

                let scale = rng.random_range(0.85..1.2);
                let rotation_y_rad = rng.random_range(0.0..std::f32::consts::TAU);
                let world_pos = WorldPos::new(chunk_coord, LocalPos::new(local_x, h_exact, local_z));
                //gizmo.cross(world_pos, 2.0, [1.0, 0.0, 0.0, 1.0], 0.0, 10.0);
                let tree_kind = self.pick_kind(h, m, &mut rng);
                archetypes.insert(tree_kind.to_string());
                placements.push(PropInstance {
                    id: None,
                    archetype_id: Some(self.archetype_id_of_kind(tree_kind)),
                    pos: world_pos,
                    color: [1.0, 1.0, 1.0, 1.0],
                    scale,
                    rotation_y_rad,
                    seed: rng.random(),
                    variant: 0,
                    wind_strength: 1.0,
                    generated: true,
                });
            }
        }
        push_gizmo_renders(take(&mut gizmo.pending_renders));


        // println!(
        //     "blob min {:.2} max {:.2} avg {:.2}",
        //     min,
        //     max,
        //     sum / veg_samples.len() as f32
        // );

        (placements, archetypes.into_iter().collect())
    }

    /// This is the actual forest map.
    /// It is broad, deterministic, and has hard-ish blob boundaries.
    /// Terrain only nudges it, it does not decide it alone.
    fn forest_blob_factor(&self, world_x: f32, world_z: f32, height: f32, moisture: f32) -> f32 {
        let cs = chunk_size() as f32;

        // Very broad scale for large forest regions.
        let macro_scale = 1.0 / (cs * 0.5);
        let carve_scale = 1.0 / (cs * 4.5);

        let macro_n = self.forest_noise.get_noise_2d(world_x * macro_scale, world_z * macro_scale);
        let carve_n = self.forest_carve_noise.get_noise_2d(world_x * carve_scale, world_z * carve_scale);

        let macro01 = ((macro_n + 1.0) * 0.5).clamp(0.0, 1.0);
        let carve01 = ((carve_n + 1.0) * 0.5).clamp(0.0, 1.0);

        // Broad blob mask, not a tiny speckle mask.
        let blob = Self::smoothstep(0.56, 0.80, macro01);

        // Carve holes inside blobs so it does not become a solid carpet.
        let holes = 1.0 - Self::smoothstep(0.38, 0.72, carve01);

        // Terrain influences it, but weakly.
        // Low wet areas are more likely forest, high dry areas less likely.
        let plains_factor = if height <= 18.0 {
            1.0
        } else if height >= 90.0 {
            0.0
        } else {
            1.0 - ((height - 18.0) / (90.0 - 18.0)).clamp(0.0, 1.0)
        };

        let wet_factor = moisture.clamp(0.0, 1.0);

        let terrain = (0.35 + 0.45 * wet_factor + 0.20 * (1.0 - plains_factor)).clamp(0.0, 1.0);
        // println!(
        //     "macro={} macro01={} blob={} carve01={} holes={}",
        //     macro_n,
        //     macro01,
        //     blob,
        //     carve01,
        //     holes,
        // );
        (blob * holes * terrain).clamp(0.0, 1.0)
    }

    fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
        let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    fn tree_density(height: f32, moisture: f32) -> f32 {
        if height < 2.0 || height > 1800.0 {
            return 0.0;
        }

        let altitude_factor = (1.0 - ((height - 60.0) / 1200.0).max(0.0)).clamp(0.0, 1.0);
        let moisture_factor = moisture.clamp(0.0, 1.0);

        (altitude_factor * moisture_factor).powf(1.5) * 0.35
    }

    fn pick_kind(&self, height: f32, moisture: f32, rng: &mut impl Rng) -> TreeKind {
        if moisture > 0.7 {
            TreeKind::Oak //TreeKind::Pine
        } else if height > 120.0 {
            TreeKind::Oak //TreeKind::DeadTree
        } else if rng.random_bool(0.9) {
            TreeKind::Oak
        } else {
            TreeKind::Birch
        }
    }

    fn archetype_id_of_kind(&self, kind: TreeKind) -> ArchetypeId {
        let idx = TreeKind::iter().position(|k| k == kind).unwrap_or(0);
        self.kind_to_archetype_id.get(idx).copied().unwrap_or(0)
    }

    fn rng_for_chunk(chunk_coord: ChunkCoord) -> SmallRng {
        const TREE_SALT: u64 = 0x7A_5E_ED_5E_ED_00_01;
        let x = chunk_coord.x as i64 as u64;
        let z = chunk_coord.z as i64 as u64;
        let mut seed = x
            .wrapping_mul(0x9E3779B97F4A7C15)
            ^ z.wrapping_mul(0xC2B2AE3D27D4EB4F)
            ^ TREE_SALT;

        seed = (seed ^ (seed >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        seed = (seed ^ (seed >> 27)).wrapping_mul(0x94D049BB133111EB);
        seed ^= seed >> 31;

        SmallRng::seed_from_u64(seed)
    }
}
