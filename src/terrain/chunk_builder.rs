use crate::positions::{ChunkCoord, ChunkSize, LocalPos, LodStep, WorldPos};
use crate::terrain::terrain::TerrainGenerator;
use crate::terrain::terrain_editing::EditedChunk;
use crate::terrain::threads::ChunkWorkerPool;
use crate::ui::vertex::Vertex;
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
    pub fn world_to_local(&self, pos: &WorldPos) -> (f32, f32) {
        let chunk_size = self.chunk_size_f32();
        let chunk_offset_x = (pos.chunk.x - self.chunk_coord.x) as f32 * chunk_size;
        let chunk_offset_z = (pos.chunk.z - self.chunk_coord.z) as f32 * chunk_size;
        (chunk_offset_x + pos.local.x, chunk_offset_z + pos.local.z)
    }

    /// Convert local coordinates to a WorldPos (normalizing if needed).
    #[inline]
    pub fn local_to_world(&self, local_x: f32, local_y: f32, local_z: f32) -> WorldPos {
        WorldPos::new(self.chunk_coord, LocalPos::new(local_x, local_y, local_z))
            .normalize(self.chunk_size)
    }

    /// Check if a WorldPos falls within this chunk's boundaries.
    #[inline]
    pub fn contains(&self, pos: &WorldPos) -> bool {
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

impl ChunkBuilder {
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
    ) -> Option<CpuChunkMesh> {
        let stepf = step as f32;

        let verts_x = (chunk_size / step + 1) as usize;
        let verts_z = (chunk_size / step + 1) as usize;

        let mut vertices = Vec::with_capacity(verts_x * verts_z);
        let mut heights = vec![0.0f32; verts_x * verts_z];

        // 1. Populate heights (Inner Chunk Data)
        for gx in 0..verts_x {
            for gz in 0..verts_z {
                // Local position within the chunk
                let local_x = (gx * step as usize) as f32;
                let local_z = (gz * step as usize) as f32;

                let world_pos = WorldPos::new(chunk_coord, LocalPos::new(local_x, 0.0, local_z));

                let h = terrain_gen.height(&world_pos, chunk_size);
                heights[gx * verts_z + gz] = h;

                let m = terrain_gen.moisture(&world_pos, h, chunk_size);
                let color = terrain_gen.color(&world_pos, h, m, chunk_size);

                // Store LOCAL position - rendering will offset by chunk position relative to camera
                vertices.push(Vertex {
                    local_position: [local_x, h, local_z],
                    normal: [0.0, 1.0, 0.0], // Dummy normal
                    color,
                    chunk_xz: [chunk_coord.x, chunk_coord.z],
                });
            }
        }

        // 2. Compute Normals (Using Neighbor Lookups for Edges)
        let inv = 1.0 / stepf;
        for gx in 0..verts_x {
            for gz in 0..verts_z {
                let local_x = (gx * step as usize) as f32;
                let local_z = (gz * step as usize) as f32;

                // --- X Axis Gradient ---
                let h_l = if gx > 0 {
                    heights[(gx - 1) * verts_z + gz]
                } else {
                    // Query generator for x - step (normalizes to previous chunk if needed)
                    let pos =
                        WorldPos::new(chunk_coord, LocalPos::new(local_x - stepf, 0.0, local_z))
                            .normalize(chunk_size);
                    terrain_gen.height(&pos, chunk_size)
                };

                let h_r = if gx < verts_x - 1 {
                    heights[(gx + 1) * verts_z + gz]
                } else {
                    let pos =
                        WorldPos::new(chunk_coord, LocalPos::new(local_x + stepf, 0.0, local_z))
                            .normalize(chunk_size);
                    terrain_gen.height(&pos, chunk_size)
                };

                // --- Z Axis Gradient ---
                let h_d = if gz > 0 {
                    heights[gx * verts_z + (gz - 1)]
                } else {
                    let pos =
                        WorldPos::new(chunk_coord, LocalPos::new(local_x, 0.0, local_z - stepf))
                            .normalize(chunk_size);
                    terrain_gen.height(&pos, chunk_size)
                };

                let h_u = if gz < verts_z - 1 {
                    heights[gx * verts_z + (gz + 1)]
                } else {
                    let pos =
                        WorldPos::new(chunk_coord, LocalPos::new(local_x, 0.0, local_z + stepf))
                            .normalize(chunk_size);
                    terrain_gen.height(&pos, chunk_size)
                };

                // Central Difference
                let dhdx = (h_r - h_l) * 0.5 * inv;
                let dhdz = (h_u - h_d) * 0.5 * inv;

                let n = Vec3::new(-dhdx, 1.0, -dhdz).normalize();
                vertices[gx * verts_z + gz].normal = [n.x, n.y, n.z];
            }
        }

        // -------- indices --------
        let mut indices = Vec::with_capacity((verts_x - 1) * (verts_z - 1) * 6);
        for gx in 0..verts_x - 1 {
            if gx & 0b1111 == 0 && !ChunkWorkerPool::still_current(version_atomic, version) {
                return None;
            }

            for gz in 0..verts_z - 1 {
                let i0 = (gx * verts_z + gz) as u32;
                let i1 = ((gx + 1) * verts_z + gz) as u32;
                let i2 = (gx * verts_z + gz + 1) as u32;
                let i3 = ((gx + 1) * verts_z + gz + 1) as u32;
                indices.extend_from_slice(&[i0, i1, i2, i2, i1, i3]);
            }
        }

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
            for z in 0..nz {
                let local_x = x as f32 * cell_size_f;
                let local_z = z as f32 * cell_size_f;

                let pos = WorldPos::new(chunk_coord, LocalPos::new(local_x, 0.0, local_z));
                heights[x * nz + z] = terrain_gen.height(&pos, chunk_size);
            }
        }

        // 8x8 logical cell patches
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
                        let h = heights[gx * nz + gz];
                        min_y = min_y.min(h);
                        max_y = max_y.max(h);
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

pub fn lod_step_for_distance(dist2_chunks: i32) -> LodStep {
    // distance in chunks
    // sqrt(dist2) gives number of chunks away

    if dist2_chunks < 1 {
        1 // same chunk
    } else if dist2_chunks < 4 {
        2 // < 2 chunks
    } else if dist2_chunks < 12 {
        4 // < 4 chunks
    } else if dist2_chunks < 48 {
        8 // < 8 chunks
    } else if dist2_chunks < 128 {
        16 // < 16 chunks
    } else {
        32
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
