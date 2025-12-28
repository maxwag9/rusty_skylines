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
    pub base_x: f32,
    pub base_z: f32,
    pub cell: f32,
    pub nx: usize,
    pub nz: usize,
    pub heights: Vec<f32>,             // x * nz + z
    pub patch_minmax: Vec<(f32, f32)>, // 8x8 patches
}

pub struct ChunkMeshLod {
    pub step: usize,
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
    coord: (i32, i32),
    own_grid: &ChunkHeightGrid,
    chunks: &HashMap<(i32, i32), ChunkMeshLod>,
    edited_chunks: &HashMap<(i32, i32), EditedChunk>,
) -> NeighborEdgeHeights {
    let mut result = NeighborEdgeHeights::empty();

    let own_nx = own_grid.nx;
    let own_nz = own_grid.nz;
    let own_cell = own_grid.cell;
    let own_base_x = own_grid.base_x;
    let own_base_z = own_grid.base_z;

    // Sample height from neighbor grid at world position, including pending deltas only
    // (accumulated deltas are already baked into height_grid.heights)
    let sample_neighbor =
        |neighbor_coord: (i32, i32), chunk: &ChunkMeshLod, wx: f32, wz: f32| -> f32 {
            let grid = chunk.height_grid.as_ref();

            let gx_f = (wx - grid.base_x) / grid.cell;
            let gz_f = (wz - grid.base_z) / grid.cell;

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
        };

    // +X neighbor: sample one step past our +X edge
    let pos_x_coord = (coord.0 + 1, coord.1);
    if let Some(chunk) = chunks.get(&pos_x_coord) {
        let sample_x = own_base_x + (own_nx as f32) * own_cell;
        let mut edge = Vec::with_capacity(own_nz);
        for gz in 0..own_nz {
            let wz = own_base_z + (gz as f32) * own_cell;
            edge.push(sample_neighbor(pos_x_coord, chunk, sample_x, wz));
        }
        result.pos_x = Some(edge);
    }

    // -X neighbor: sample one step before our -X edge
    let neg_x_coord = (coord.0 - 1, coord.1);
    if let Some(chunk) = chunks.get(&neg_x_coord) {
        let sample_x = own_base_x - own_cell;
        let mut edge = Vec::with_capacity(own_nz);
        for gz in 0..own_nz {
            let wz = own_base_z + (gz as f32) * own_cell;
            edge.push(sample_neighbor(neg_x_coord, chunk, sample_x, wz));
        }
        result.neg_x = Some(edge);
    }

    // +Z neighbor: sample one step past our +Z edge
    let pos_z_coord = (coord.0, coord.1 + 1);
    if let Some(chunk) = chunks.get(&pos_z_coord) {
        let sample_z = own_base_z + (own_nz as f32) * own_cell;
        let mut edge = Vec::with_capacity(own_nx);
        for gx in 0..own_nx {
            let wx = own_base_x + (gx as f32) * own_cell;
            edge.push(sample_neighbor(pos_z_coord, chunk, wx, sample_z));
        }
        result.pos_z = Some(edge);
    }

    // -Z neighbor: sample one step before our -Z edge
    let neg_z_coord = (coord.0, coord.1 - 1);
    if let Some(chunk) = chunks.get(&neg_z_coord) {
        let sample_z = own_base_z - own_cell;
        let mut edge = Vec::with_capacity(own_nx);
        for gx in 0..own_nx {
            let wx = own_base_x + (gx as f32) * own_cell;
            edge.push(sample_neighbor(neg_z_coord, chunk, wx, sample_z));
        }
        result.neg_z = Some(edge);
    }

    result
}

/// Regenerate vertex positions (y), normals, and optionally colors from the provided height grid.
/// We update `vertices` in-place.
/// Expects vertices.len() == nx * nz.
///
/// `neighbor_edges` provides heights from adjacent chunks for correct normal calculation at edges.
/// If a neighbor edge is not available, falls back to terrain_gen.height().
pub fn regenerate_vertices_from_height_grid(
    vertices: &mut [Vertex],
    height_grid: &ChunkHeightGrid,
    terrain_gen: &TerrainGenerator,
    neighbor_edges: Option<&NeighborEdgeHeights>,
    update_colors: bool,
) {
    let verts_x = height_grid.nx;
    let verts_z = height_grid.nz;
    let stepf = height_grid.cell;
    let base_x = height_grid.base_x;
    let base_z = height_grid.base_z;

    // sanity check
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

    // recompute normals using central differences
    let inv = 1.0 / stepf;
    for gx in 0..verts_x {
        for gz in 0..verts_z {
            let idx = gx * verts_z + gz;

            // Calculate world position for fallback neighbor queries
            let wx = base_x + (gx as f32 * stepf);
            let wz = base_z + (gz as f32 * stepf);

            // --- X Axis Gradient ---
            let h_l = if gx > 0 {
                height_grid.heights[(gx - 1) * verts_z + gz]
            } else {
                // Need height from -X neighbor's +X edge
                neighbor_edges
                    .and_then(|n| n.neg_x.as_ref())
                    .and_then(|edge| edge.get(gz).copied())
                    .unwrap_or_else(|| terrain_gen.height(wx - stepf, wz))
            };

            let h_r = if gx + 1 < verts_x {
                height_grid.heights[(gx + 1) * verts_z + gz]
            } else {
                // Need height from +X neighbor's -X edge
                neighbor_edges
                    .and_then(|n| n.pos_x.as_ref())
                    .and_then(|edge| edge.get(gz).copied())
                    .unwrap_or_else(|| terrain_gen.height(wx + stepf, wz))
            };

            // --- Z Axis Gradient ---
            let h_d = if gz > 0 {
                height_grid.heights[gx * verts_z + (gz - 1)]
            } else {
                // Need height from -Z neighbor's +Z edge
                neighbor_edges
                    .and_then(|n| n.neg_z.as_ref())
                    .and_then(|edge| edge.get(gx).copied())
                    .unwrap_or_else(|| terrain_gen.height(wx, wz - stepf))
            };

            let h_u = if gz + 1 < verts_z {
                height_grid.heights[gx * verts_z + (gz + 1)]
            } else {
                // Need height from +Z neighbor's -Z edge
                neighbor_edges
                    .and_then(|n| n.pos_z.as_ref())
                    .and_then(|edge| edge.get(gx).copied())
                    .unwrap_or_else(|| terrain_gen.height(wx, wz + stepf))
            };

            // Central Difference
            let dhdx = (h_r - h_l) * 0.5 * inv;
            let dhdz = (h_u - h_d) * 0.5 * inv;

            let n = Vec3::new(-dhdx, 1.0, -dhdz).normalize();
            vertices[idx].normal = [n.x, n.y, n.z];

            // Optionally update colors
            if update_colors {
                let v_pos = vertices[idx].position;
                vertices[idx].color = terrain_gen.color(
                    v_pos[0],
                    v_pos[2],
                    v_pos[1],
                    terrain_gen.moisture(v_pos[0], v_pos[2], v_pos[1]),
                );
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
    pub cx: i32,
    pub cz: i32,
    pub step: usize,
    pub version: u64,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub height_grid: Arc<ChunkHeightGrid>,
}

impl ChunkBuilder {
    pub fn build_chunk_cpu(
        chunk_x: i32,
        chunk_z: i32,
        size: u32,
        step: usize,
        _ns_x_neg: usize,
        _ns_x_pos: usize,
        _ns_z_neg: usize,
        _ns_z_pos: usize,
        version: u64,
        version_atomic: &AtomicU64,
        terrain_gen: &TerrainGenerator,
    ) -> Option<CpuChunkMesh> {
        let size = size as usize;
        let base_x = chunk_x as f32 * size as f32;
        let base_z = chunk_z as f32 * size as f32;
        let stepf = step as f32;

        let verts_x = size / step + 1;
        let verts_z = size / step + 1;

        // Compute normals after storing all heights
        let mut vertices = Vec::with_capacity(verts_x * verts_z);
        let mut heights = vec![0.0; verts_x * verts_z];

        // 1. Populate heights (Inner Chunk Data)
        for gx in 0..verts_x {
            for gz in 0..verts_z {
                let wx = base_x + (gx * step) as f32;
                let wz = base_z + (gz * step) as f32;
                let h = terrain_gen.height(wx, wz);
                heights[gx * verts_z + gz] = h;

                vertices.push(Vertex {
                    position: [wx, h, wz],
                    normal: [0.0, 1.0, 0.0], // Dummy normal
                    color: terrain_gen.color(wx, wz, h, terrain_gen.moisture(wx, wz, h)),
                });
            }
        }

        // 2. Compute Normals (Using Neighbor Lookups for Edges for Normals!!)
        let inv = 1.0 / stepf;
        for gx in 0..verts_x {
            for gz in 0..verts_z {
                // Recalculate World position for neighbor
                let wx = base_x + (gx * step) as f32;
                let wz = base_z + (gz * step) as f32;

                // --- X Axis Gradient ---
                let h_l = if gx > 0 {
                    heights[(gx - 1) * verts_z + gz]
                } else {
                    // FIXED: Query generator for x - 1
                    terrain_gen.height(wx - stepf, wz)
                };

                let h_r = if gx < verts_x - 1 {
                    heights[(gx + 1) * verts_z + gz]
                } else {
                    terrain_gen.height(wx + stepf, wz)
                };

                // --- Z Axis Gradient ---
                let h_d = if gz > 0 {
                    heights[gx * verts_z + (gz - 1)]
                } else {
                    terrain_gen.height(wx, wz - stepf)
                };

                let h_u = if gz < verts_z - 1 {
                    heights[gx * verts_z + (gz + 1)]
                } else {
                    terrain_gen.height(wx, wz + stepf)
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
            cx: chunk_x,
            cz: chunk_z,
            step,
            version,
            vertices,
            indices,
            height_grid: Arc::new(Self::build_height_grid(
                chunk_x,
                chunk_z,
                size,
                step,
                terrain_gen,
            )),
        })
    }

    pub fn build_height_grid(
        chunk_x: i32,
        chunk_z: i32,
        size: usize,
        cell_size: usize,
        terrain_gen: &TerrainGenerator,
    ) -> ChunkHeightGrid {
        let size_f = size as f32;
        let cell_size_f = cell_size as f32;
        let nx = (size_f / cell_size_f) as usize + 1;
        let nz = (size_f / cell_size_f) as usize + 1;

        let base_x = chunk_x as f32 * size_f;
        let base_z = chunk_z as f32 * size_f;

        let mut heights = vec![0.0f32; nx * nz];

        for x in 0..nx {
            for z in 0..nz {
                let wx = base_x + x as f32 * cell_size_f;
                let wz = base_z + z as f32 * cell_size_f;
                heights[x * nz + z] = terrain_gen.height(wx, wz);
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
                let mut max_y = -f32::INFINITY;

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
            base_x,
            base_z,
            cell: cell_size as f32,
            nx,
            nz,
            heights,
            patch_minmax,
        }
    }
}

pub fn lod_step_for_distance(dist2_chunks: i32) -> usize {
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

pub fn generate_spiral_offsets(radius: i32) -> Vec<(i32, i32)> {
    let mut v = Vec::new();
    for dx in -radius..=radius {
        for dz in -radius..=radius {
            v.push((dx, dz));
        }
    }

    // sort by distance from center
    v.sort_by_key(|(dx, dz)| dx * dx + dz * dz);
    v
}
