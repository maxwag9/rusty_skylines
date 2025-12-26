use crate::renderer::mesh_arena::{GeometryScratch, MeshArena};
use crate::renderer::world_renderer::regenerate_vertices_from_height_grid_and_color;
use crate::terrain::chunk_builder::{ChunkHeightGrid, ChunkMeshLod, GpuChunkHandle};
use crate::terrain::terrain::TerrainGenerator;
use crate::ui::vertex::Vertex;
use glam::Vec3;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{Device, Queue};

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
pub fn affected_chunks(center: Vec3, radius: f32, chunk_size: f32) -> (i32, i32, i32, i32) {
    let min_x = ((center.x - radius) / chunk_size).floor() as i32;
    let max_x = ((center.x + radius) / chunk_size).floor() as i32;
    let min_z = ((center.z - radius) / chunk_size).floor() as i32;
    let max_z = ((center.z + radius) / chunk_size).floor() as i32;
    (min_x, max_x, min_z, max_z)
}

pub struct EditedChunk {
    pub deltas: Vec<(usize, usize, f32)>,
    pub dirty: bool,
}

pub struct TerrainEditor {
    edited_chunks: HashMap<(i32, i32), EditedChunk>,
}

impl Default for TerrainEditor {
    fn default() -> Self {
        Self {
            edited_chunks: HashMap::new(),
        }
    }
}

impl TerrainEditor {
    pub fn get_edits(&self, coord: &(i32, i32)) -> Option<&EditedChunk> {
        self.edited_chunks.get(coord)
    }

    pub fn apply_brush<F: Falloff, B: BrushOp>(
        &mut self,
        center: Vec3,
        radius: f32,
        strength: f32,
        chunk_size: f32,
        chunks: &HashMap<(i32, i32), ChunkMeshLod>,
    ) {
        let (min_cx, max_cx, min_cz, max_cz) = affected_chunks(center, radius, chunk_size);
        let r2 = radius * radius;

        let target_coords: Vec<_> = (min_cx..=max_cx)
            .flat_map(|cx| (min_cz..=max_cz).map(move |cz| (cx, cz)))
            .collect();

        for &coord in &target_coords {
            self.edited_chunks
                .entry(coord)
                .or_insert_with(|| EditedChunk {
                    deltas: Vec::new(),
                    dirty: false,
                });
        }

        for coord in target_coords {
            self.compute_deltas_for_chunk::<F, B>(coord, center, radius, r2, strength, chunks);
        }
    }

    fn compute_deltas_for_chunk<F: Falloff, B: BrushOp>(
        &mut self,
        coord: (i32, i32),
        center: Vec3,
        radius: f32,
        r2: f32,
        strength: f32,
        chunks: &HashMap<(i32, i32), ChunkMeshLod>,
    ) {
        let edited = self.edited_chunks.get_mut(&coord).unwrap();
        edited.dirty = false;

        let chunk = match chunks.get(&coord) {
            Some(c) => c,
            None => return,
        };

        let hg = &*chunk.height_grid;
        let (min_fx, max_fx, min_fz, max_fz) = grid_bounds(hg, center, radius);

        let mut changed = false;

        for fx in min_fx..=max_fx {
            if fx < 0 || (fx as usize) >= hg.nx {
                continue;
            }
            let gx = fx as usize;

            for fz in min_fz..=max_fz {
                if fz < 0 || (fz as usize) >= hg.nz {
                    continue;
                }
                let gz = fz as usize;

                if let Some(delta) = compute_delta::<F, B>(hg, gx, gz, center, r2, strength) {
                    edited.deltas.push((gx, gz, delta));
                    changed = true;
                }
            }
        }

        edited.dirty = changed;
    }

    pub fn upload_dirty_chunks(
        &mut self,
        device: &Device,
        queue: &Queue,
        arena: &mut MeshArena,
        chunks: &mut HashMap<(i32, i32), ChunkMeshLod>,
        terrain_gen: &TerrainGenerator,
        scratch: &mut GeometryScratch<Vertex>,
        finalize: bool,
    ) -> Vec<GpuChunkHandle> {
        let dirty_coords: Vec<_> = self
            .edited_chunks
            .iter()
            .filter(|(_, e)| e.dirty && !e.deltas.is_empty())
            .map(|(c, _)| *c)
            .collect();

        let mut freed_handles = Vec::new();

        for coord in dirty_coords {
            if let Some(handle) = self.upload_single_chunk(
                coord,
                device,
                queue,
                arena,
                chunks,
                terrain_gen,
                scratch,
                finalize,
            ) {
                freed_handles.push(handle);
            }
        }

        freed_handles
    }

    fn upload_single_chunk(
        &mut self,
        coord: (i32, i32),
        device: &Device,
        queue: &Queue,
        arena: &mut MeshArena,
        chunks: &mut HashMap<(i32, i32), ChunkMeshLod>,
        terrain_gen: &TerrainGenerator,
        scratch: &mut GeometryScratch<Vertex>,
        finalize: bool,
    ) -> Option<GpuChunkHandle> {
        let deltas = {
            let edited = self.edited_chunks.get_mut(&coord)?;
            edited.dirty = false;
            if finalize {
                edited.deltas.clone()
            } else {
                std::mem::take(&mut edited.deltas)
            }
        };

        let (step, indices, mut grid, mut vertices) = match chunks.get(&coord) {
            Some(c) => (
                c.step,
                c.cpu_indices.clone(),
                (*c.height_grid).clone(),
                c.cpu_vertices.clone(),
            ),
            None => return None,
        };

        for (gx, gz, delta) in &deltas {
            if *gx < grid.nx && *gz < grid.nz {
                grid.heights[gx * grid.nz + gz] += *delta;
            }
        }

        recompute_patch_minmax(&mut grid);
        regenerate_vertices_from_height_grid_and_color(&mut vertices, &grid, terrain_gen);

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

fn grid_bounds(hg: &ChunkHeightGrid, center: Vec3, radius: f32) -> (isize, isize, isize, isize) {
    let min_fx = (((center.x - radius) - hg.base_x) / hg.cell).floor() as isize;
    let max_fx = (((center.x + radius) - hg.base_x) / hg.cell).ceil() as isize;
    let min_fz = (((center.z - radius) - hg.base_z) / hg.cell).floor() as isize;
    let max_fz = (((center.z + radius) - hg.base_z) / hg.cell).ceil() as isize;
    (min_fx, max_fx, min_fz, max_fz)
}

fn compute_delta<F: Falloff, B: BrushOp>(
    hg: &ChunkHeightGrid,
    gx: usize,
    gz: usize,
    center: Vec3,
    r2: f32,
    strength: f32,
) -> Option<f32> {
    let wx = hg.base_x + (gx as f32) * hg.cell;
    let wz = hg.base_z + (gz as f32) * hg.cell;
    let dx = wx - center.x;
    let dz = wz - center.z;
    let d2 = dx * dx + dz * dz;

    if d2 >= r2 {
        return None;
    }

    let w = F::weight(d2, r2);
    if w <= 0.0001 {
        return None;
    }

    let idx = gx * hg.nz + gz;
    let current_h = hg.heights[idx];
    let mut new_h = current_h;
    B::apply(&mut new_h, strength, w);
    let delta = new_h - current_h;

    if delta.abs() < f32::EPSILON {
        None
    } else {
        Some(delta)
    }
}

fn recompute_patch_minmax(grid: &mut ChunkHeightGrid) {
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

pub fn apply_sparse_deltas(
    height_grid: &ChunkHeightGrid,
    deltas: &[(usize, usize, f32)],
) -> ChunkHeightGrid {
    let mut grid = height_grid.clone();
    for &(gx, gz, delta) in deltas {
        if gx < grid.nx && gz < grid.nz {
            grid.heights[gx * grid.nz + gz] += delta;
        }
    }
    grid
}
