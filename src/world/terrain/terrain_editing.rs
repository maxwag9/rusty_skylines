use crate::helpers::positions::{ChunkCoord, ChunkSize, LocalPos, LodStep, WorldPos};
use crate::renderer::mesh_arena::{GeometryScratch, TerrainMeshArena};
use crate::ui::vertex::Vertex;
use crate::world::roads::road_structs::{NodeId, SegmentId};
use crate::world::terrain::chunk_builder::{
    ChunkHeightGrid, ChunkMeshLod, GpuChunkHandle, gather_neighbor_edge_heights,
    regenerate_vertices_from_height_grid,
};
use crate::world::terrain::terrain::TerrainGenerator;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use wgpu::{Device, Queue};

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RoadEditElement {
    Segment(SegmentId),
    Intersection(NodeId),
}

#[derive(Serialize, Deserialize)]
struct PersistedChunk {
    accumulated_deltas: Vec<((u16, u16), f32)>,
    road_deltas: Vec<(RoadEditElement, Vec<((u16, u16), f32)>)>,
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

pub fn affected_chunks(
    center: WorldPos,
    radius: f32,
    chunk_size: ChunkSize,
) -> (i32, i32, i32, i32) {
    let chunk_size_f64 = chunk_size as f64;
    let radius_f64 = radius as f64;

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
    pub accumulated_deltas: HashMap<(u16, u16), f32>,
    pub road_deltas: HashMap<RoadEditElement, HashMap<(u16, u16), f32>>,
    pub dirty: bool,
}

impl Default for EditedChunk {
    fn default() -> Self {
        Self {
            pending_deltas: Vec::new(),
            accumulated_deltas: HashMap::new(),
            road_deltas: HashMap::new(),
            dirty: false,
        }
    }
}

impl EditedChunk {
    pub fn total_delta_at(&self, gx: u16, gz: u16) -> f32 {
        let key = (gx, gz);
        let manual = self.accumulated_deltas.get(&key).copied().unwrap_or(0.0);
        let road: f32 = self
            .road_deltas
            .values()
            .filter_map(|deltas| deltas.get(&key))
            .sum();
        manual + road
    }

    pub fn has_any_deltas(&self) -> bool {
        !self.accumulated_deltas.is_empty() || !self.road_deltas.is_empty()
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
            if !chunk.has_any_deltas() && !chunk.dirty {
                continue;
            }

            let accumulated_vec: Vec<_> = chunk
                .accumulated_deltas
                .iter()
                .filter(|&(_, &h)| h.abs() > 0.001)
                .map(|(&(x, z), &h)| ((x, z), h))
                .collect();

            let road_vec: Vec<_> = chunk
                .road_deltas
                .iter()
                .map(|(&elem, deltas)| {
                    let delta_vec: Vec<_> = deltas
                        .iter()
                        .filter(|&(_, &h)| h.abs() > 0.001)
                        .map(|(&(x, z), &h)| ((x, z), h))
                        .collect();
                    (elem, delta_vec)
                })
                .filter(|(_, v)| !v.is_empty())
                .collect();

            if accumulated_vec.is_empty() && road_vec.is_empty() && !chunk.dirty {
                continue;
            }

            edits.insert(
                *chunk_coord,
                PersistedChunk {
                    accumulated_deltas: accumulated_vec,
                    road_deltas: road_vec,
                    dirty: chunk.dirty,
                },
            );
        }

        let meta = EditFile {
            version: 3,
            timestamp_unix: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            edits,
        };

        let bytes = postcard::to_allocvec(&meta)?;
        let compressed = zstd::encode_all(&bytes[..], 3)?;

        let manual_count: usize = self
            .edited_chunks
            .values()
            .map(|c| c.accumulated_deltas.len())
            .sum();
        let road_count: usize = self
            .edited_chunks
            .values()
            .map(|c| c.road_deltas.values().map(|r| r.len()).sum::<usize>())
            .sum();

        println!(
            "Deltas: {} manual, {} road/intersection, {} total",
            manual_count,
            road_count,
            manual_count + road_count
        );
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

        let mut editor = TerrainEditor {
            edited_chunks: HashMap::new(),
        };

        for (chunk_coord, persisted_chunk) in meta.edits {
            let mut acc = HashMap::with_capacity(persisted_chunk.accumulated_deltas.len());
            for ((x16, z16), h) in persisted_chunk.accumulated_deltas {
                acc.insert((x16, z16), h);
            }

            let mut road = HashMap::new();
            for (elem, deltas) in persisted_chunk.road_deltas {
                let mut elem_deltas = HashMap::with_capacity(deltas.len());
                for ((x16, z16), h) in deltas {
                    elem_deltas.insert((x16, z16), h);
                }
                road.insert(elem, elem_deltas);
            }

            editor.edited_chunks.insert(
                chunk_coord,
                EditedChunk {
                    pending_deltas: Vec::new(),
                    accumulated_deltas: acc,
                    road_deltas: road,
                    dirty: persisted_chunk.dirty,
                },
            );
        }

        Ok(editor)
    }

    pub fn apply_segment_flattening(
        &mut self,
        segment_id: SegmentId,
        centerline: &[WorldPos],
        road_heights: &[f32],
        half_width: f32,
        terrain_offset: f32,
        falloff_distance: f32,
        chunk_size: ChunkSize,
        chunks: &HashMap<ChunkCoord, ChunkMeshLod>,
    ) -> HashSet<ChunkCoord> {
        self.apply_polyline_flattening(
            RoadEditElement::Segment(segment_id),
            centerline,
            road_heights,
            half_width,
            terrain_offset,
            falloff_distance,
            chunk_size,
            chunks,
        )
    }

    pub fn apply_intersection_flattening(
        &mut self,
        node_id: NodeId,
        center: WorldPos,
        center_height: f32,
        radius: f32,
        terrain_offset: f32,
        falloff_distance: f32,
        chunk_size: ChunkSize,
        chunks: &HashMap<ChunkCoord, ChunkMeshLod>,
    ) -> HashSet<ChunkCoord> {
        self.apply_circular_flattening(
            RoadEditElement::Intersection(node_id),
            center,
            center_height,
            radius,
            terrain_offset,
            falloff_distance,
            chunk_size,
            chunks,
        )
    }

    pub fn apply_intersection_polygon_flattening(
        &mut self,
        node_id: NodeId,
        polygon: &[WorldPos],
        center_height: f32,
        terrain_offset: f32,
        falloff_distance: f32,
        chunk_size: ChunkSize,
        chunks: &HashMap<ChunkCoord, ChunkMeshLod>,
    ) -> HashSet<ChunkCoord> {
        self.apply_polygon_flattening(
            RoadEditElement::Intersection(node_id),
            polygon,
            center_height,
            terrain_offset,
            falloff_distance,
            chunk_size,
            chunks,
        )
    }

    pub fn remove_segment_flattening(&mut self, segment_id: SegmentId) -> HashSet<ChunkCoord> {
        self.remove_road_element_flattening(RoadEditElement::Segment(segment_id))
    }

    pub fn remove_intersection_flattening(&mut self, node_id: NodeId) -> HashSet<ChunkCoord> {
        self.remove_road_element_flattening(RoadEditElement::Intersection(node_id))
    }

    fn remove_road_element_flattening(&mut self, element: RoadEditElement) -> HashSet<ChunkCoord> {
        let mut affected = HashSet::new();

        for (&coord, edited) in self.edited_chunks.iter_mut() {
            if edited.road_deltas.remove(&element).is_some() {
                edited.dirty = true;
                affected.insert(coord);
            }
        }

        affected
    }

    fn apply_polyline_flattening(
        &mut self,
        element: RoadEditElement,
        centerline: &[WorldPos],
        road_heights: &[f32],
        half_width: f32,
        terrain_offset: f32,
        falloff_distance: f32,
        chunk_size: ChunkSize,
        chunks: &HashMap<ChunkCoord, ChunkMeshLod>,
    ) -> HashSet<ChunkCoord> {
        let mut affected = HashSet::new();

        if centerline.is_empty() || centerline.len() != road_heights.len() {
            return affected;
        }

        let total_width = half_width + falloff_distance;

        let mut affected_chunk_coords: HashSet<ChunkCoord> = HashSet::new();
        for pos in centerline {
            let (min_cx, max_cx, min_cz, max_cz) = affected_chunks(*pos, total_width, chunk_size);
            for cx in min_cx..=max_cx {
                for cz in min_cz..=max_cz {
                    affected_chunk_coords.insert(ChunkCoord::new(cx, cz));
                }
            }
        }

        for chunk_coord in affected_chunk_coords {
            let chunk = match chunks.get(&chunk_coord) {
                Some(c) => c,
                None => continue,
            };

            let grid = &chunk.height_grid;
            let step = chunk.step as usize;
            let base_cell = grid.cell_f32() / step as f32;
            let base_nx = (grid.nx - 1) * step + 1;
            let base_nz = (grid.nz - 1) * step + 1;

            let mut element_deltas: HashMap<(u16, u16), f32> = HashMap::new();

            for base_gx in 0..base_nx {
                for base_gz in 0..base_nz {
                    let local_x = base_gx as f32 * base_cell;
                    let local_z = base_gz as f32 * base_cell;

                    let grid_pos = WorldPos::new(chunk_coord, LocalPos::new(local_x, 0.0, local_z));

                    let (dist, road_height) = closest_point_on_polyline_xz(
                        &grid_pos,
                        centerline,
                        road_heights,
                        chunk_size,
                    );

                    if dist > total_width {
                        continue;
                    }

                    let weight = if dist <= half_width {
                        1.0
                    } else {
                        let t = (dist - half_width) / falloff_distance;
                        let s = 1.0 - t;
                        s * s * (3.0 - 2.0 * s)
                    };

                    if weight < 0.01 {
                        continue;
                    }

                    let current_height = sample_height_at_local(grid, local_x, local_z);
                    let target_height = road_height + terrain_offset;
                    let delta = (target_height - current_height) * weight;

                    if delta.abs() > 0.001 {
                        element_deltas.insert((base_gx as u16, base_gz as u16), delta);
                    }
                }
            }

            if !element_deltas.is_empty() {
                let edited = self.edited_chunks.entry(chunk_coord).or_default();
                edited.road_deltas.insert(element, element_deltas);
                edited.dirty = true;
                affected.insert(chunk_coord);
            }
        }

        affected
    }

    fn apply_circular_flattening(
        &mut self,
        element: RoadEditElement,
        center: WorldPos,
        center_height: f32,
        radius: f32,
        terrain_offset: f32,
        falloff_distance: f32,
        chunk_size: ChunkSize,
        chunks: &HashMap<ChunkCoord, ChunkMeshLod>,
    ) -> HashSet<ChunkCoord> {
        let mut affected = HashSet::new();

        let total_radius = radius + falloff_distance;
        let (min_cx, max_cx, min_cz, max_cz) = affected_chunks(center, total_radius, chunk_size);

        let center_wx = center.chunk.x as f64 * chunk_size as f64 + center.local.x as f64;
        let center_wz = center.chunk.z as f64 * chunk_size as f64 + center.local.z as f64;

        for cx in min_cx..=max_cx {
            for cz in min_cz..=max_cz {
                let chunk_coord = ChunkCoord::new(cx, cz);
                let chunk = match chunks.get(&chunk_coord) {
                    Some(c) => c,
                    None => continue,
                };

                let grid = &chunk.height_grid;
                let step = chunk.step as usize;
                let base_cell = grid.cell_f32() / step as f32;
                let base_nx = (grid.nx - 1) * step + 1;
                let base_nz = (grid.nz - 1) * step + 1;

                let chunk_wx_base = cx as f64 * chunk_size as f64;
                let chunk_wz_base = cz as f64 * chunk_size as f64;

                let mut element_deltas: HashMap<(u16, u16), f32> = HashMap::new();

                for base_gx in 0..base_nx {
                    for base_gz in 0..base_nz {
                        let local_x = base_gx as f32 * base_cell;
                        let local_z = base_gz as f32 * base_cell;

                        let wx = chunk_wx_base + local_x as f64;
                        let wz = chunk_wz_base + local_z as f64;

                        let dx = wx - center_wx;
                        let dz = wz - center_wz;
                        let dist = ((dx * dx + dz * dz) as f32).sqrt();

                        if dist > total_radius {
                            continue;
                        }

                        let weight = if dist <= radius {
                            1.0
                        } else {
                            let t = (dist - radius) / falloff_distance;
                            let s = 1.0 - t;
                            s * s * (3.0 - 2.0 * s)
                        };

                        if weight < 0.01 {
                            continue;
                        }

                        let current_height = sample_height_at_local(grid, local_x, local_z);
                        let target_height = center_height + terrain_offset;
                        let delta = (target_height - current_height) * weight;

                        if delta.abs() > 0.001 {
                            element_deltas.insert((base_gx as u16, base_gz as u16), delta);
                        }
                    }
                }

                if !element_deltas.is_empty() {
                    let edited = self.edited_chunks.entry(chunk_coord).or_default();
                    edited.road_deltas.insert(element, element_deltas);
                    edited.dirty = true;
                    affected.insert(chunk_coord);
                }
            }
        }

        affected
    }

    fn apply_polygon_flattening(
        &mut self,
        element: RoadEditElement,
        polygon: &[WorldPos],
        center_height: f32,
        terrain_offset: f32,
        falloff_distance: f32,
        chunk_size: ChunkSize,
        chunks: &HashMap<ChunkCoord, ChunkMeshLod>,
    ) -> HashSet<ChunkCoord> {
        let mut affected = HashSet::new();

        if polygon.len() < 3 {
            return affected;
        }

        let mut min_wx = f64::MAX;
        let mut max_wx = f64::MIN;
        let mut min_wz = f64::MAX;
        let mut max_wz = f64::MIN;

        let poly_world: Vec<(f64, f64)> = polygon
            .iter()
            .map(|p| {
                let wx = p.chunk.x as f64 * chunk_size as f64 + p.local.x as f64;
                let wz = p.chunk.z as f64 * chunk_size as f64 + p.local.z as f64;
                min_wx = min_wx.min(wx);
                max_wx = max_wx.max(wx);
                min_wz = min_wz.min(wz);
                max_wz = max_wz.max(wz);
                (wx, wz)
            })
            .collect();

        let falloff_f64 = falloff_distance as f64;
        min_wx -= falloff_f64;
        max_wx += falloff_f64;
        min_wz -= falloff_f64;
        max_wz += falloff_f64;

        let min_cx = (min_wx / chunk_size as f64).floor() as i32;
        let max_cx = (max_wx / chunk_size as f64).floor() as i32;
        let min_cz = (min_wz / chunk_size as f64).floor() as i32;
        let max_cz = (max_wz / chunk_size as f64).floor() as i32;

        for cx in min_cx..=max_cx {
            for cz in min_cz..=max_cz {
                let chunk_coord = ChunkCoord::new(cx, cz);
                let chunk = match chunks.get(&chunk_coord) {
                    Some(c) => c,
                    None => continue,
                };

                let grid = &chunk.height_grid;
                let step = chunk.step as usize;
                let base_cell = grid.cell_f32() / step as f32;
                let base_nx = (grid.nx - 1) * step + 1;
                let base_nz = (grid.nz - 1) * step + 1;

                let chunk_wx_base = cx as f64 * chunk_size as f64;
                let chunk_wz_base = cz as f64 * chunk_size as f64;

                let mut element_deltas: HashMap<(u16, u16), f32> = HashMap::new();

                for base_gx in 0..base_nx {
                    for base_gz in 0..base_nz {
                        let local_x = base_gx as f32 * base_cell;
                        let local_z = base_gz as f32 * base_cell;

                        let wx = chunk_wx_base + local_x as f64;
                        let wz = chunk_wz_base + local_z as f64;

                        let (inside, dist_to_edge) = point_polygon_distance(&poly_world, wx, wz);

                        let dist = dist_to_edge as f32;

                        let weight = if inside {
                            1.0
                        } else if dist <= falloff_distance {
                            let t = dist / falloff_distance;
                            let s = 1.0 - t;
                            s * s * (3.0 - 2.0 * s)
                        } else {
                            continue;
                        };

                        if weight < 0.01 {
                            continue;
                        }

                        let current_height = sample_height_at_local(grid, local_x, local_z);
                        let target_height = center_height + terrain_offset;
                        let delta = (target_height - current_height) * weight;

                        if delta.abs() > 0.001 {
                            element_deltas.insert((base_gx as u16, base_gz as u16), delta);
                        }
                    }
                }

                if !element_deltas.is_empty() {
                    let edited = self.edited_chunks.entry(chunk_coord).or_default();
                    edited.road_deltas.insert(element, element_deltas);
                    edited.dirty = true;
                    affected.insert(chunk_coord);
                }
            }
        }

        affected
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

        let base_cell = hg.cell_f32() / step as f32;
        let base_cell_f64 = base_cell as f64;
        let base_nx = (hg.nx - 1) * step + 1;
        let base_nz = (hg.nz - 1) * step + 1;

        let chunk_offset_x = (coord.x - center.chunk.x) as f64 * chunk_size;
        let chunk_offset_z = (coord.z - center.chunk.z) as f64 * chunk_size;
        let center_local_x = center.local.x as f64;
        let center_local_z = center.local.z as f64;

        let radius_f64 = radius as f64;

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
                let local_x = (base_gx as f64) * base_cell_f64;
                let local_z = (base_gz as f64) * base_cell_f64;

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

                let current_h = sample_height_at_local(hg, local_x as f32, local_z as f32);

                let mut new_h = current_h;
                B::apply(&mut new_h, strength, w);
                let delta = new_h - current_h;

                if delta.abs() < f32::EPSILON {
                    continue;
                }

                let base_gx_u16 = base_gx as u16;
                let base_gz_u16 = base_gz as u16;
                *edited
                    .accumulated_deltas
                    .entry((base_gx_u16, base_gz_u16))
                    .or_insert(0.0) += delta;

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
            .filter(|(_, e)| e.dirty)
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

        let (step, mut indices, mut grid, mut vertices) = match chunks.get(&coord) {
            Some(c) => (
                c.step,
                c.cpu_indices.clone(),
                (*c.height_grid).clone(),
                c.cpu_vertices.clone(),
            ),
            None => return None,
        };

        for (gx, gz, delta) in &pending {
            let gx_usize = *gx as usize;
            let gz_usize = *gz as usize;
            if gx_usize < grid.nx && gz_usize < grid.nz {
                grid.heights[gx_usize * grid.nz + gz_usize] += *delta;
            }
        }

        for seg_deltas in edited.road_deltas.values() {
            for (&(base_gx, base_gz), &delta) in seg_deltas {
                if base_gx as usize % step as usize != 0 || base_gz as usize % step as usize != 0 {
                    continue;
                }
                let gx = base_gx as usize / step as usize;
                let gz = base_gz as usize / step as usize;
                if gx < grid.nx && gz < grid.nz {
                    grid.heights[gx * grid.nz + gz] += delta;
                }
            }
        }

        recompute_patch_minmax(&mut grid);

        let expected_verts = grid.nx * grid.nz;

        if vertices.len() == expected_verts {
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

fn point_polygon_distance(polygon: &[(f64, f64)], px: f64, pz: f64) -> (bool, f64) {
    let n = polygon.len();
    if n < 3 {
        return (false, f64::MAX);
    }

    let mut inside = false;
    let mut min_dist_sq = f64::MAX;

    for i in 0..n {
        let j = (i + 1) % n;
        let (ax, az) = polygon[i];
        let (bx, bz) = polygon[j];

        if ((az > pz) != (bz > pz)) && (px < (bx - ax) * (pz - az) / (bz - az) + ax) {
            inside = !inside;
        }

        let dx = bx - ax;
        let dz = bz - az;
        let len_sq = dx * dx + dz * dz;

        let t = if len_sq < 1e-10 {
            0.0
        } else {
            (((px - ax) * dx + (pz - az) * dz) / len_sq).clamp(0.0, 1.0)
        };

        let closest_x = ax + t * dx;
        let closest_z = az + t * dz;
        let dist_sq = (px - closest_x).powi(2) + (pz - closest_z).powi(2);

        min_dist_sq = min_dist_sq.min(dist_sq);
    }

    (inside, min_dist_sq.sqrt())
}

fn closest_point_on_polyline_xz(
    point: &WorldPos,
    polyline: &[WorldPos],
    heights: &[f32],
    chunk_size: ChunkSize,
) -> (f32, f32) {
    let mut best_dist = f32::MAX;
    let mut best_height = 0.0;

    let px = point.chunk.x as f64 * chunk_size as f64 + point.local.x as f64;
    let pz = point.chunk.z as f64 * chunk_size as f64 + point.local.z as f64;

    for i in 0..polyline.len().saturating_sub(1) {
        let a = &polyline[i];
        let b = &polyline[i + 1];

        let ax = a.chunk.x as f64 * chunk_size as f64 + a.local.x as f64;
        let az = a.chunk.z as f64 * chunk_size as f64 + a.local.z as f64;
        let bx = b.chunk.x as f64 * chunk_size as f64 + b.local.x as f64;
        let bz = b.chunk.z as f64 * chunk_size as f64 + b.local.z as f64;

        let dx = bx - ax;
        let dz = bz - az;
        let len_sq = dx * dx + dz * dz;

        let t = if len_sq < 1e-10 {
            0.0
        } else {
            (((px - ax) * dx + (pz - az) * dz) / len_sq).clamp(0.0, 1.0)
        };

        let closest_x = ax + t * dx;
        let closest_z = az + t * dz;

        let dist = (((px - closest_x).powi(2) + (pz - closest_z).powi(2)) as f32).sqrt();

        if dist < best_dist {
            best_dist = dist;
            best_height = heights[i] + (heights[i + 1] - heights[i]) * t as f32;
        }
    }

    if polyline.len() == 1 {
        let a = &polyline[0];
        let ax = a.chunk.x as f64 * chunk_size as f64 + a.local.x as f64;
        let az = a.chunk.z as f64 * chunk_size as f64 + a.local.z as f64;
        best_dist = (((px - ax).powi(2) + (pz - az).powi(2)) as f32).sqrt();
        best_height = heights[0];
    }

    (best_dist, best_height)
}

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
        .map_or(false, |e| e.has_any_deltas());

    if let Some(edit) = edited_chunks.get(&coord) {
        for (&(base_x, base_z), &delta) in &edit.accumulated_deltas {
            if base_x % own_step != 0 || base_z % own_step != 0 {
                continue;
            }

            let gx = (base_x / own_step) as usize;
            let gz = (base_z / own_step) as usize;

            if gx < grid.nx && gz < grid.nz {
                grid.heights[gx * grid.nz + gz] += delta;
            }
        }

        for seg_deltas in edit.road_deltas.values() {
            for (&(base_x, base_z), &delta) in seg_deltas {
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
    }

    if has_own_edits {
        return grid;
    }

    let neighbors = [
        (coord.offset(-1, 0), Edge::XNeg),
        (coord.offset(1, 0), Edge::XPos),
        (coord.offset(0, -1), Edge::ZNeg),
        (coord.offset(0, 1), Edge::ZPos),
    ];

    for (neighbor_coord, edge) in neighbors {
        let neighbor_has_edits = edited_chunks
            .get(&neighbor_coord)
            .map_or(false, |e| e.has_any_deltas());

        if neighbor_has_edits {
            continue;
        }

        if let Some(neighbor_chunk) = loaded_chunks.get(&neighbor_coord) {
            if neighbor_chunk.step > own_step {
                stitch_edge_to_neighbor(&mut grid, &neighbor_chunk.height_grid, edge);
            }
        }
    }

    grid
}

fn stitch_x_edge(
    grid: &mut ChunkHeightGrid,
    neighbor_grid: &ChunkHeightGrid,
    our_gx: usize,
    neighbor_gx: usize,
) {
    let our_cell = grid.cell_f32();
    let neighbor_cell = neighbor_grid.cell_f32();

    for our_gz in 0..grid.nz {
        let local_z = our_gz as f32 * our_cell;

        let neighbor_gz_f = local_z / neighbor_cell;
        let neighbor_gz_lo = (neighbor_gz_f.floor() as usize).min(neighbor_grid.nz - 1);
        let neighbor_gz_hi = (neighbor_gz_lo + 1).min(neighbor_grid.nz - 1);
        let t = (neighbor_gz_f - neighbor_gz_lo as f32).clamp(0.0, 1.0);

        let h_lo = neighbor_grid.heights[neighbor_gx * neighbor_grid.nz + neighbor_gz_lo];
        let h_hi = neighbor_grid.heights[neighbor_gx * neighbor_grid.nz + neighbor_gz_hi];

        grid.heights[our_gx * grid.nz + our_gz] = h_lo + t * (h_hi - h_lo);
    }
}

fn stitch_z_edge(
    grid: &mut ChunkHeightGrid,
    neighbor_grid: &ChunkHeightGrid,
    our_gz: usize,
    neighbor_gz: usize,
) {
    let our_cell = grid.cell_f32();
    let neighbor_cell = neighbor_grid.cell_f32();

    for our_gx in 0..grid.nx {
        let local_x = our_gx as f32 * our_cell;

        let neighbor_gx_f = local_x / neighbor_cell;
        let neighbor_gx_lo = (neighbor_gx_f.floor() as usize).min(neighbor_grid.nx - 1);
        let neighbor_gx_hi = (neighbor_gx_lo + 1).min(neighbor_grid.nx - 1);
        let t = (neighbor_gx_f - neighbor_gx_lo as f32).clamp(0.0, 1.0);

        let h_lo = neighbor_grid.heights[neighbor_gx_lo * neighbor_grid.nz + neighbor_gz];
        let h_hi = neighbor_grid.heights[neighbor_gx_hi * neighbor_grid.nz + neighbor_gz];

        grid.heights[our_gx * grid.nz + our_gz] = h_lo + t * (h_hi - h_lo);
    }
}

fn stitch_edge_to_neighbor(
    grid: &mut ChunkHeightGrid,
    neighbor_grid: &ChunkHeightGrid,
    edge: Edge,
) {
    match edge {
        Edge::XNeg => {
            stitch_x_edge(grid, neighbor_grid, 0, neighbor_grid.nx - 1);
        }
        Edge::XPos => {
            stitch_x_edge(grid, neighbor_grid, grid.nx - 1, 0);
        }
        Edge::ZNeg => {
            stitch_z_edge(grid, neighbor_grid, 0, neighbor_grid.nz - 1);
        }
        Edge::ZPos => {
            stitch_z_edge(grid, neighbor_grid, grid.nz - 1, 0);
        }
    }
}

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
