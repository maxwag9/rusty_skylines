use crate::helpers::positions::{ChunkCoord, ChunkSize, LocalPos, LodStep, WorldPos};
use crate::ui::vertex::Vertex;
use crate::world::terrain::terrain::TerrainGenerator;
use crate::world::terrain::terrain_editing::EditedChunk;
use crate::world::terrain::threads::ChunkWorkerPool;
use glam::Vec3;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

#[derive(Clone)]
pub struct ChunkHeightGrid {
    pub chunk_coord: ChunkCoord,
    pub chunk_size: ChunkSize,
    pub cell_size: LodStep,
    pub nx: usize,
    pub nz: usize,
    pub heights: Vec<f32>,             // indexed as x * nz + z
    pub patch_minmax: Vec<(f32, f32)>, // 8x8 patches for culling
}

impl ChunkHeightGrid {
    #[inline]
    pub fn cell_f32(&self) -> f32 {
        self.cell_size as f32
    }

    #[inline]
    pub fn chunk_size_f32(&self) -> f32 {
        self.chunk_size as f32
    }

    /// Grid extent in local X direction
    #[inline]
    pub fn extent_x(&self) -> f32 {
        (self.nx - 1) as f32 * self.cell_f32()
    }

    /// Grid extent in local Z direction
    #[inline]
    pub fn extent_z(&self) -> f32 {
        (self.nz - 1) as f32 * self.cell_f32()
    }
    /// Convert a WorldPos to local coordinates relative to this grid's chunk.
    /// Returns (local_x, local_z) which may be outside [0, chunk_size] if pos is in a different chunk.
    #[inline]
    pub fn _world_to_local(&self, pos: &WorldPos) -> (f32, f32) {
        let chunk_size = self.chunk_size_f32();
        let chunk_offset_x = (pos.chunk.x - self.chunk_coord.x) as f32 * chunk_size;
        let chunk_offset_z = (pos.chunk.z - self.chunk_coord.z) as f32 * chunk_size;
        (chunk_offset_x + pos.local.x, chunk_offset_z + pos.local.z)
    }

    /// Convert local coordinates to a WorldPos (normalizing if needed).
    #[inline]
    pub fn _local_to_world(&self, local_x: f32, local_y: f32, local_z: f32) -> WorldPos {
        WorldPos::new(self.chunk_coord, LocalPos::new(local_x, local_y, local_z))
            .normalize(self.chunk_size)
    }

    /// Check if a WorldPos falls within this chunk's boundaries.
    #[inline]
    pub fn _contains(&self, pos: &WorldPos) -> bool {
        pos.chunk.x == self.chunk_coord.x && pos.chunk.z == self.chunk_coord.z
    }
}

pub struct ChunkMeshLod {
    pub step: LodStep,
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

/// Gather edge heights from neighboring chunks for proper normal calculation.
/// Samples at positions one step past our edges (for central difference normals).
/// Handles different LOD resolutions via bilinear interpolation.
pub fn gather_neighbor_edge_heights(
    coord: ChunkCoord,
    own_grid: &ChunkHeightGrid,
    chunks: &HashMap<ChunkCoord, ChunkMeshLod>,
    edited_chunks: &HashMap<ChunkCoord, EditedChunk>,
) -> NeighborEdgeHeights {
    let mut result = NeighborEdgeHeights::empty();

    let own_nx = own_grid.nx;
    let own_nz = own_grid.nz;
    let own_cell = own_grid.cell_f32();
    let chunk_size = own_grid.chunk_size_f32();

    /// Sample height from neighbor grid at a local position within that neighbor's chunk,
    /// including pending deltas (accumulated deltas are already baked into heights).
    fn sample_neighbor_with_pending(
        neighbor_coord: ChunkCoord,
        chunk: &ChunkMeshLod,
        neighbor_local_x: f32,
        neighbor_local_z: f32,
        edited_chunks: &HashMap<ChunkCoord, EditedChunk>,
    ) -> f32 {
        let grid = chunk.height_grid.as_ref();
        let cell = grid.cell_f32();

        let gx_f = neighbor_local_x / cell;
        let gz_f = neighbor_local_z / cell;

        let gx0 = (gx_f.floor() as usize).min(grid.nx.saturating_sub(1));
        let gz0 = (gz_f.floor() as usize).min(grid.nz.saturating_sub(1));
        let gx1 = (gx0 + 1).min(grid.nx - 1);
        let gz1 = (gz0 + 1).min(grid.nz - 1);

        let tx = (gx_f - gx0 as f32).clamp(0.0, 1.0);
        let tz = (gz_f - gz0 as f32).clamp(0.0, 1.0);

        // Get pending delta for a grid position (only pending, not accumulated)
        let get_pending = |gx: usize, gz: usize| -> f32 {
            edited_chunks
                .get(&neighbor_coord)
                .map(|e| {
                    e.pending_deltas
                        .iter()
                        .filter(|&&(px, pz, _)| px as usize == gx && pz as usize == gz)
                        .map(|&(_, _, d)| d)
                        .sum::<f32>()
                })
                .unwrap_or(0.0)
        };

        let h00 = grid.heights[gx0 * grid.nz + gz0] + get_pending(gx0, gz0);
        let h10 = grid.heights[gx1 * grid.nz + gz0] + get_pending(gx1, gz0);
        let h01 = grid.heights[gx0 * grid.nz + gz1] + get_pending(gx0, gz1);
        let h11 = grid.heights[gx1 * grid.nz + gz1] + get_pending(gx1, gz1);

        // Bilinear interpolation
        let h0 = h00 + tx * (h10 - h00);
        let h1 = h01 + tx * (h11 - h01);
        h0 + tz * (h1 - h0)
    }

    // +X neighbor: sample one step past our +X edge
    // Our +X edge is at local_x = chunk_size (last vertex)
    // One step past: local_x = chunk_size + own_cell
    // In neighbor's local space: (chunk_size + own_cell) - chunk_size = own_cell
    let pos_x_coord = coord.offset(1, 0);
    if let Some(chunk) = chunks.get(&pos_x_coord) {
        let neighbor_local_x = own_cell;
        let mut edge = Vec::with_capacity(own_nz);
        for gz in 0..own_nz {
            let our_local_z = gz as f32 * own_cell;
            // Z coordinate is same in neighbor's space (no chunk boundary crossed in Z)
            edge.push(sample_neighbor_with_pending(
                pos_x_coord,
                chunk,
                neighbor_local_x,
                our_local_z,
                edited_chunks,
            ));
        }
        result.pos_x = Some(edge);
    }

    // -X neighbor: sample one step before our -X edge
    // Our -X edge is at local_x = 0
    // One step before: local_x = -own_cell
    // In -X neighbor's local space: -own_cell + chunk_size = chunk_size - own_cell
    let neg_x_coord = coord.offset(-1, 0);
    if let Some(chunk) = chunks.get(&neg_x_coord) {
        let neighbor_local_x = chunk_size - own_cell;
        let mut edge = Vec::with_capacity(own_nz);
        for gz in 0..own_nz {
            let our_local_z = gz as f32 * own_cell;
            edge.push(sample_neighbor_with_pending(
                neg_x_coord,
                chunk,
                neighbor_local_x,
                our_local_z,
                edited_chunks,
            ));
        }
        result.neg_x = Some(edge);
    }

    // +Z neighbor: sample one step past our +Z edge
    let pos_z_coord = coord.offset(0, 1);
    if let Some(chunk) = chunks.get(&pos_z_coord) {
        let neighbor_local_z = own_cell;
        let mut edge = Vec::with_capacity(own_nx);
        for gx in 0..own_nx {
            let our_local_x = gx as f32 * own_cell;
            edge.push(sample_neighbor_with_pending(
                pos_z_coord,
                chunk,
                our_local_x,
                neighbor_local_z,
                edited_chunks,
            ));
        }
        result.pos_z = Some(edge);
    }

    // -Z neighbor: sample one step before our -Z edge
    let neg_z_coord = coord.offset(0, -1);
    if let Some(chunk) = chunks.get(&neg_z_coord) {
        let neighbor_local_z = chunk_size - own_cell;
        let mut edge = Vec::with_capacity(own_nx);
        for gx in 0..own_nx {
            let our_local_x = gx as f32 * own_cell;
            edge.push(sample_neighbor_with_pending(
                neg_z_coord,
                chunk,
                our_local_x,
                neighbor_local_z,
                edited_chunks,
            ));
        }
        result.neg_z = Some(edge);
    }

    result
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
    let cell = height_grid.cell_f32();
    let chunk = height_grid.chunk_coord;
    let chunk_size = height_grid.chunk_size;

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
                            .normalize(chunk_size);
                        terrain_gen.height(&pos, chunk_size)
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
                            .normalize(chunk_size);
                        terrain_gen.height(&pos, chunk_size)
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
                            .normalize(chunk_size);
                        terrain_gen.height(&pos, chunk_size)
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
                            .normalize(chunk_size);
                        terrain_gen.height(&pos, chunk_size)
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
                let m = terrain_gen.moisture(&pos, h, chunk_size);
                vertices[idx].color = terrain_gen.color(&pos, h, m, chunk_size);
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
    const HEIGHT_TOLERANCE: f32 = 0.05; // Max height variance to consider "flat"
    const COLOR_TOLERANCE: f32 = 0.02; // Max color difference per channel

    pub fn build_chunk_cpu(
        chunk_coord: ChunkCoord,
        chunk_size: ChunkSize,
        step: LodStep,
        _ns_x_neg: LodStep,
        _ns_x_pos: LodStep,
        _ns_z_neg: LodStep,
        _ns_z_pos: LodStep,
        version: u64,
        version_atomic: &AtomicU64,
        terrain_gen: &TerrainGenerator,
        has_edits: bool,
    ) -> Option<CpuChunkMesh> {
        let stepf = step as f32;
        let step_usize = step as usize;
        let inv_step = 1.0 / stepf;

        let verts_x = (chunk_size / step + 1) as usize;
        let verts_z = (chunk_size / step + 1) as usize;
        let cells_x = verts_x - 1;
        let cells_z = verts_z - 1;
        let _total_verts = verts_x * verts_z;
        let total_cells = cells_x * cells_z;

        let (heights, colors) = Self::sample_terrain_batch(
            chunk_coord,
            chunk_size,
            step,
            verts_x,
            verts_z,
            terrain_gen,
        );

        if !ChunkWorkerPool::still_current(version_atomic, version) {
            return None;
        }

        let normals = Self::compute_normals_batch(
            chunk_coord,
            chunk_size,
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

        let (vertices, indices) = if has_edits {
            // Simple grid layout: vertices[gx * verts_z + gz] = vertex at (gx, gz)
            // This allows in-place updates via regenerate_vertices_from_height_grid
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
            // Greedy meshing path for non-edited chunks
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

        Some(CpuChunkMesh {
            chunk_coord,
            step,
            version,
            vertices,
            indices,
            height_grid: Arc::new(Self::build_height_grid(
                chunk_coord,
                chunk_size,
                step,
                terrain_gen,
            )),
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
        chunk_size: ChunkSize,
        step: LodStep,
        verts_x: usize,
        verts_z: usize,
        terrain_gen: &TerrainGenerator,
    ) -> (Vec<f32>, Vec<[f32; 3]>) {
        let step_usize = step as usize;
        let total = verts_x * verts_z;

        let mut heights = Vec::with_capacity(total);
        let mut colors = Vec::with_capacity(total);

        for gx in 0..verts_x {
            let local_x = (gx * step_usize) as f32;

            for gz in 0..verts_z {
                let local_z = (gz * step_usize) as f32;
                let world_pos = WorldPos::new(chunk_coord, LocalPos::new(local_x, 0.0, local_z));

                let h = terrain_gen.height(&world_pos, chunk_size);
                let m = terrain_gen.moisture(&world_pos, h, chunk_size);
                let c = terrain_gen.color(&world_pos, h, m, chunk_size);

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
        chunk_size: ChunkSize,
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
                    Self::sample_neighbor_height(
                        chunk_coord,
                        chunk_size,
                        terrain_gen,
                        local_x - stepf,
                        local_z,
                    )
                };

                let h_right = if gx < verts_x - 1 {
                    heights[(gx + 1) * verts_z + gz]
                } else {
                    Self::sample_neighbor_height(
                        chunk_coord,
                        chunk_size,
                        terrain_gen,
                        local_x + stepf,
                        local_z,
                    )
                };

                let h_back = if gz > 0 {
                    heights[gx * verts_z + (gz - 1)]
                } else {
                    Self::sample_neighbor_height(
                        chunk_coord,
                        chunk_size,
                        terrain_gen,
                        local_x,
                        local_z - stepf,
                    )
                };

                let h_front = if gz < verts_z - 1 {
                    heights[gx * verts_z + (gz + 1)]
                } else {
                    Self::sample_neighbor_height(
                        chunk_coord,
                        chunk_size,
                        terrain_gen,
                        local_x,
                        local_z + stepf,
                    )
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
        chunk_size: ChunkSize,
        terrain_gen: &TerrainGenerator,
        local_x: f32,
        local_z: f32,
    ) -> f32 {
        let pos =
            WorldPos::new(chunk_coord, LocalPos::new(local_x, 0.0, local_z)).normalize(chunk_size);
        terrain_gen.height(&pos, chunk_size)
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

    pub fn build_height_grid(
        chunk_coord: ChunkCoord,
        chunk_size: ChunkSize,
        cell_size: LodStep,
        terrain_gen: &TerrainGenerator,
    ) -> ChunkHeightGrid {
        let cell_size_f = cell_size as f32;
        let nx = (chunk_size / cell_size + 1) as usize;
        let nz = (chunk_size / cell_size + 1) as usize;

        let mut heights = vec![0.0f32; nx * nz];

        for x in 0..nx {
            let local_x = x as f32 * cell_size_f;
            for z in 0..nz {
                let local_z = z as f32 * cell_size_f;
                let pos = WorldPos::new(chunk_coord, LocalPos::new(local_x, 0.0, local_z));
                heights[x * nz + z] = terrain_gen.height(&pos, chunk_size);
            }
        }

        // Build 8x8 patch min/max for fast culling
        let patch_cells = 8usize;
        let px = (nx - 1) / patch_cells;
        let pz = (nz - 1) / patch_cells;

        let mut patch_minmax = Vec::with_capacity(px * pz);

        for px_i in 0..px {
            for pz_i in 0..pz {
                let mut min_y = f32::INFINITY;
                let mut max_y = f32::NEG_INFINITY;

                for lx in 0..=patch_cells {
                    for lz in 0..=patch_cells {
                        let gx = px_i * patch_cells + lx;
                        let gz = pz_i * patch_cells + lz;
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
            chunk_size,
            cell_size,
            nx,
            nz,
            heights,
            patch_minmax,
        }
    }
}

pub fn lod_step_for_distance(dist2_chunks: i32, chunk_size: ChunkSize) -> LodStep {
    if dist2_chunks <= 0 {
        return 1; // center chunk always max detail
    }

    let dist_world2 = (dist2_chunks as f32) * (chunk_size as f32).powi(2);

    if dist_world2 < 1024.0 {
        1
    }
    // < ~32 world units
    else if dist_world2 < 4096.0 {
        2
    }
    // < ~64
    else if dist_world2 < 16384.0 {
        4
    }
    // < ~128
    else if dist_world2 < 65536.0 {
        8
    }
    // < ~256
    else if dist_world2 < 262144.0 {
        16
    }
    // < ~512
    else if dist_world2 < 1048576.0 {
        32
    }
    // < ~1024
    else if dist_world2 < 4194304.0 {
        64
    }
    // < ~2048
    else {
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
