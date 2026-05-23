// terrain_editing.rs
use crate::helpers::positions::{ChunkCoord, LocalPos, LodStep, WorldPos, chunk_size};
use crate::renderer::mesh_arena::{GeometryScratch, TerrainMeshArena};
use crate::ui::vertex::Vertex;
use crate::world::buildings::buildings::BuildingId;
use crate::world::roads::road_structs::{NodeId, SegmentId};
use crate::world::terrain::chunk_builder::{
    ChunkHeightGrid, ChunkMeshLod, GpuChunkHandle, NeighborEdgeHeights, generate_height_grid,
    regenerate_vertices_from_height_grid,
};
use crate::world::terrain::terrain_gen::TerrainGenerator;
use crate::world::terrain::terrain_subsystem::Terrain;
use crate::world::terrain::terrain_threads::{LoadedChunksSnapshot, TerrainEditsSnapshot};
use glam::Vec2;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use wgpu::{Device, Queue};

pub type EditId = u64;

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TerrainEditSource {
    Player,
    Building(BuildingId),
    Segment(SegmentId),
    Intersection(NodeId),
}

/// A single terrain editing operation. This is the canonical, persistent
/// representation of all terrain changes. On load the list is replayed from
/// scratch against the base (generator) heights to reproduce the exact terrain.
///
/// Operations are applied in order; later flat-type ops take priority through
/// weighted blending. Raise is always additive regardless of order.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum TerrainEdit {
    /// Additive raise / lower brush with smooth radial falloff.
    /// Use `strength < 0` to lower.
    Raise {
        id: EditId,
        center: WorldPos,
        radius: f32,
        strength: f32,
        edit_source: TerrainEditSource,
    },

    /// Flatten a convex-or-concave polygon to exactly `height`.
    /// Inside the polygon weight = 1.0 (completely flat).
    /// Outside, weight smoothly falls to 0 over `falloff_distance`.
    ///
    /// Ideal for: building lots, plazas, parks, road intersections.
    FlatPolygon {
        id: EditId,
        /// Polygon vertices in world space. Must have ≥ 3 points.
        polygon: Vec<WorldPos>,
        /// Offset from the polygon heights
        height_offset: f32,
        /// Distance outside the polygon edge over which the terrain
        /// transitions back to its original height.
        falloff_distance: f32,

        edit_source: TerrainEditSource,
    },

    /// Flatten terrain along a polyline (road segment).
    FlatPolyline {
        id: EditId,
        centerline: Vec<WorldPos>,
        half_width: f32,
        terrain_offset: f32,
        falloff_distance: f32,
        edit_source: TerrainEditSource,
    },

    /// Flatten terrain in a circle (road intersection or roundabout).
    FlatCircle {
        id: EditId,
        center: WorldPos,
        radius: f32,
        terrain_offset: f32,
        falloff_distance: f32,
        edit_source: TerrainEditSource,
    },
}

impl TerrainEdit {
    pub fn id(&self) -> EditId {
        match self {
            Self::Raise { id, .. }
            | Self::FlatPolygon { id, .. }
            | Self::FlatPolyline { id, .. }
            | Self::FlatCircle { id, .. } => *id,
        }
    }

    pub fn source(&self) -> &TerrainEditSource {
        match self {
            Self::Raise { edit_source, .. } => edit_source,
            Self::FlatPolygon { edit_source, .. } => edit_source,
            Self::FlatPolyline { edit_source, .. } => edit_source,
            Self::FlatCircle { edit_source, .. } => edit_source,
        }
    }

    /// `(min_cx, max_cx, min_cz, max_cz)` chunk-coordinate bounding box.
    pub fn chunk_bounds(&self) -> (i32, i32, i32, i32) {
        let cs = chunk_size() as f64;
        match self {
            Self::Raise { center, radius, .. } => affected_chunks(*center, *radius),
            Self::FlatCircle {
                center,
                radius,
                falloff_distance,
                ..
            } => affected_chunks(*center, radius + falloff_distance),
            Self::FlatPolyline {
                centerline,
                half_width,
                falloff_distance,
                ..
            } => {
                let total = half_width + falloff_distance;
                let mut min_cx = i32::MAX;
                let mut max_cx = i32::MIN;
                let mut min_cz = i32::MAX;
                let mut max_cz = i32::MIN;
                for pos in centerline {
                    let (a, b, c, d) = affected_chunks(*pos, total);
                    min_cx = min_cx.min(a);
                    max_cx = max_cx.max(b);
                    min_cz = min_cz.min(c);
                    max_cz = max_cz.max(d);
                }
                if min_cx == i32::MAX {
                    (0, 0, 0, 0)
                } else {
                    (min_cx, max_cx, min_cz, max_cz)
                }
            }
            Self::FlatPolygon {
                polygon,
                falloff_distance,
                ..
            } => polygon_chunk_bounds(polygon, *falloff_distance as f64, cs),
        }
    }

    /// Set of every chunk coord this edit touches.
    pub fn affected_chunks(&self) -> HashSet<ChunkCoord> {
        let (min_cx, max_cx, min_cz, max_cz) = self.chunk_bounds();
        let mut set = HashSet::new();
        for cx in min_cx..=max_cx {
            for cz in min_cz..=max_cz {
                set.insert(ChunkCoord::new(cx, cz));
            }
        }
        set
    }
}

pub trait Falloff {
    fn weight(d2: f32, r2: f32) -> f32;
}

pub struct SmoothFalloff;
impl Falloff for SmoothFalloff {
    fn weight(d2: f32, r2: f32) -> f32 {
        smooth_weight(d2, r2)
    }
}

pub trait BrushOp {
    /// Returns the signed delta to apply (positive = raise, negative = lower).
    fn delta(strength: f32, weight: f32) -> f32;
}

pub struct Raise;
impl BrushOp for Raise {
    fn delta(strength: f32, weight: f32) -> f32 {
        strength * weight
    }
}

pub struct Lower;
impl BrushOp for Lower {
    fn delta(strength: f32, weight: f32) -> f32 {
        -strength * weight
    }
}

#[derive(Serialize, Deserialize, Default)]
struct SaveFile {
    version: String,
    timestamp_unix: u64,
    /// The complete ordered list of terrain edit operations.
    edits: Vec<TerrainEdit>,
}

/// Manages terrain modifications as an ordered list of operations.
///
/// **Design contract**
/// - `edits` is the single source of truth and the only thing serialised.
/// - `base_grids` stores the pristine generator heights for each chunk.
///   Register them via [`TerrainEditor::register_base_grid`] when chunks are
///   first created (before any edits are applied).
/// - When the GPU mesh for a dirty chunk is rebuilt, all edits are replayed
///   against the base grid in order.
pub struct TerrainEditor {
    /// Ordered list of all edit operations. Serialised to disk on save.
    pub edits: Vec<TerrainEdit>,
    next_id: EditId,

    /// Pristine (pre-edit) height grids. Required for correct replay.
    /// Must be populated via [`register_base_grid`] before chunks can be rebuilt.
    pub pristine_grids: HashMap<ChunkCoord, ChunkHeightGrid>,

    /// This replaces the O(edits × chunks) scan every frame
    pub affected_chunks: HashSet<ChunkCoord>,

    /// Chunks that need GPU upload (after successful worker job or direct edit)
    dirty_chunks: HashSet<ChunkCoord>,
}

impl Default for TerrainEditor {
    fn default() -> Self {
        Self {
            edits: Vec::new(),
            next_id: 1,
            pristine_grids: HashMap::new(),
            affected_chunks: HashSet::new(),
            dirty_chunks: HashSet::new(),
        }
    }
}

impl TerrainEditor {
    // Base grid Gen!!
    pub fn get_pristine_grid(
        &mut self,
        terrain_generator: &TerrainGenerator,
        coord: ChunkCoord,
    ) -> &ChunkHeightGrid {
        self.pristine_grids
            .entry(coord)
            .or_insert_with(|| generate_height_grid(coord, 1, terrain_generator))
    }

    // ── ID allocation ─────────────────────────────────────────────────────────
    pub fn alloc_id(&mut self) -> EditId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    // ── Adding / removing edits ───────────────────────────────────────────────

    #[inline(always)]
    pub fn has_edits_on_chunk(&self, coord: ChunkCoord) -> bool {
        self.affected_chunks.contains(&coord)
    }

    // ── Add / remove edits – maintain affected_chunks incrementally ────────
    pub fn add_edit(&mut self, edit: TerrainEdit) -> EditId {
        let id = edit.id();
        let affected = edit.affected_chunks();
        self.affected_chunks.extend(affected.iter().copied());
        self.dirty_chunks.extend(affected);
        self.edits.push(edit);
        id
    }

    pub fn remove_edit(&mut self, edit_id: EditId) -> bool {
        if let Some(pos) = self.edits.iter().position(|e| e.id() == edit_id) {
            let removed = self.edits.remove(pos);
            // Rebuild entire affected set conservatively – still way faster than per-frame scan
            self.rebuild_affected_set();
            self.dirty_chunks.extend(removed.affected_chunks());
            return true;
        }
        false
    }

    /// Remove all edits whose `source` matches `element`.
    /// Returns the set of chunk coords that must be rebuilt.
    pub fn remove_edits_for_source(&mut self, source: TerrainEditSource) -> HashSet<ChunkCoord> {
        let mut affected = HashSet::new();
        let mut i = 0;
        while i < self.edits.len() {
            if *self.edits[i].source() == source {
                let removed = self.edits.remove(i);
                affected.extend(removed.affected_chunks());
            } else {
                i += 1;
            }
        }
        if !affected.is_empty() {
            self.rebuild_affected_set();
            self.dirty_chunks.extend(affected.iter().copied());
        }
        affected
    }

    /// Called only when edits are removed (rare) – O(edits) but acceptable
    fn rebuild_affected_set(&mut self) {
        self.affected_chunks.clear();
        for edit in &self.edits {
            self.affected_chunks.extend(edit.affected_chunks());
        }
    }

    // ── Convenience constructors ──────────────────────────────────────────────

    /// Add a raise / lower brush stroke.
    /// `strength > 0` raises, `strength < 0` lowers.
    pub fn push_raise(
        &mut self,
        center: WorldPos,
        radius: f32,
        strength: f32,
        edit_source: TerrainEditSource,
    ) -> EditId {
        let id = self.alloc_id();
        self.add_edit(TerrainEdit::Raise {
            id,
            center,
            radius,
            strength,
            edit_source,
        })
    }

    /// Flatten a polygon region to exactly `height` with a smooth outer transition.
    ///
    /// * Inside `polygon` the terrain is set to `height` with weight 1.
    /// * Outside `polygon`, the weight smoothly falls from 1 to 0 over
    ///   `falloff_distance` (smoothstep curve), blending back to original terrain.
    ///
    /// Use this for building lots, roads, parks, airfields, etc.
    pub fn push_flat_polygon(
        &mut self,
        polygon: Vec<WorldPos>,
        height_offset: f32,
        falloff_distance: f32,
        edit_source: TerrainEditSource,
    ) -> EditId {
        let id = self.alloc_id();
        self.add_edit(TerrainEdit::FlatPolygon {
            id,
            polygon,
            height_offset,
            falloff_distance,
            edit_source,
        })
    }

    // ── Generic brush (backward-compat entry point) ───────────────────────────

    /// Apply a single brush stroke. Use [`push_raise`] for a simpler API.
    pub fn apply_brush<F: Falloff, B: BrushOp>(
        &mut self,
        center: WorldPos,
        radius: f32,
        strength: f32,
        edit_source: TerrainEditSource,
    ) {
        // Encode the brush as a signed-strength Raise op.
        // B::delta(1.0, 1.0) gives +1 for Raise and -1 for Lower, so we
        // multiply strength by that sign to get the right polarity.
        let sign = B::delta(1.0, 1.0).signum();
        let id = self.alloc_id();
        self.add_edit(TerrainEdit::Raise {
            id,
            center,
            radius,
            strength: strength * sign,
            edit_source,
        });
    }

    // ── Road flattening (public API unchanged from old implementation) ─────────

    pub fn apply_segment_flattening(
        &mut self,
        segment_id: SegmentId,
        centerline: &[WorldPos],
        half_width: f32,
        terrain_offset: f32,
        falloff_distance: f32,
    ) -> HashSet<ChunkCoord> {
        let edit_source = TerrainEditSource::Segment(segment_id);
        // Replace any previous edit for this segment.
        self.remove_edits_for_source(edit_source);
        let id = self.alloc_id();
        let edit = TerrainEdit::FlatPolyline {
            id,
            centerline: centerline.to_vec(),
            half_width,
            terrain_offset,
            falloff_distance,
            edit_source,
        };
        let affected = edit.affected_chunks();
        self.add_edit(edit);
        affected
    }

    pub fn apply_intersection_flattening(
        &mut self,
        node_id: NodeId,
        center: WorldPos,
        radius: f32,
        terrain_offset: f32,
        falloff_distance: f32,
    ) -> HashSet<ChunkCoord> {
        let edit_source = TerrainEditSource::Intersection(node_id);
        self.remove_edits_for_source(edit_source);
        let id = self.alloc_id();
        let edit = TerrainEdit::FlatCircle {
            id,
            center,
            radius,
            terrain_offset,
            falloff_distance,
            edit_source,
        };
        let affected = edit.affected_chunks();
        self.add_edit(edit);
        affected
    }

    pub fn apply_intersection_polygon_flattening(
        &mut self,
        node_id: NodeId,
        polygon: &[WorldPos],
        height_offset: f32,
        falloff_distance: f32,
    ) -> HashSet<ChunkCoord> {
        //let direction = WorldPos::polygon_direction(polygon);

        let edit_source = TerrainEditSource::Intersection(node_id);
        self.remove_edits_for_source(edit_source);
        let id = self.alloc_id();
        let edit = TerrainEdit::FlatPolygon {
            id,
            polygon: polygon.to_vec(),
            height_offset,
            falloff_distance,
            edit_source,
        };
        let affected = edit.affected_chunks();
        self.add_edit(edit);
        affected
    }
    pub fn remove_segment_flattening(&mut self, segment_id: SegmentId) -> HashSet<ChunkCoord> {
        self.remove_edits_for_source(TerrainEditSource::Segment(segment_id))
    }

    pub fn remove_intersection_flattening(&mut self, node_id: NodeId) -> HashSet<ChunkCoord> {
        self.remove_edits_for_source(TerrainEditSource::Intersection(node_id))
    }

    // ── Save / load ───────────────────────────────────────────────────────────

    /// Returns the edit list to be serialized (e.g. stored inside a larger save struct).
    pub fn get_edits_for_save(&self) -> &[TerrainEdit] {
        &self.edits
    }

    /// Restore the edit list from a previously-saved slice (e.g. embedded in a
    /// larger save format). All affected chunks are marked dirty so they get
    /// rebuilt as soon as their base grids are registered.
    pub fn load_edits_from_vec(&mut self, edits: Vec<TerrainEdit>) {
        let max_id = edits.iter().map(|e| e.id()).max().unwrap_or(0);
        self.next_id = max_id + 1;
        self.edits = edits;

        self.affected_chunks.clear();
        self.dirty_chunks.clear();
        for edit in &self.edits {
            let set = edit.affected_chunks();
            self.affected_chunks.extend(set.iter().copied());
            self.dirty_chunks.extend(set.iter().copied());
        }
    }

    // ── Core: apply all edits to a height grid ────────────────────────────────

    /// Apply every relevant edit to `grid` in-place for chunk `coord`.
    ///
    /// The grid should contain the **base** (generator) heights before calling
    /// this. Each edit is applied in registration order; the result is
    /// deterministic and matches what was authored at edit time.
    pub fn apply_edits_to_grid(&self, grid: &mut ChunkHeightGrid, coord: ChunkCoord) {
        let cs = chunk_size() as f64;
        let cell = grid.step_f32() as f64;
        let chunk_wx = coord.x as f64 * cs;
        let chunk_wz = coord.z as f64 * cs;

        for edit in &self.edits {
            let (min_cx, max_cx, min_cz, max_cz) = edit.chunk_bounds();
            // Cheap AABB cull before doing any per-vertex work.
            if coord.x < min_cx || coord.x > max_cx || coord.z < min_cz || coord.z > max_cz {
                continue;
            }
            apply_edit_to_grid(edit, grid, chunk_wx, chunk_wz, cell, cs);
        }
    }

    /// Force immediate rebuild of all dirty chunks.
    /// Call this after batch-adding edits (e.g., placing multiple lots).
    pub fn flush_dirty_chunks(
        &mut self,
        device: &Device,
        queue: &Queue,
        arena: &mut TerrainMeshArena,
        chunks: &mut HashMap<ChunkCoord, ChunkMeshLod>,
        terrain_gen: &TerrainGenerator,
    ) -> Vec<GpuChunkHandle> {
        if self.dirty_chunks.is_empty() {
            return Vec::new();
        }

        let mut scratch = GeometryScratch::default();
        self.upload_dirty_chunks(device, queue, arena, chunks, terrain_gen, &mut scratch)
    }

    // ── GPU upload ────────────────────────────────────────────────────────────

    pub fn upload_dirty_chunks(
        &mut self,
        device: &Device,
        queue: &Queue,
        arena: &mut TerrainMeshArena,
        chunks: &mut HashMap<ChunkCoord, ChunkMeshLod>,
        terrain_gen: &TerrainGenerator,
        scratch: &mut GeometryScratch<Vertex>,
    ) -> Vec<GpuChunkHandle> {
        // Drain here so we don't hold a borrow on self during the loop.
        let dirty: Vec<ChunkCoord> = self.dirty_chunks.drain().collect();
        let mut freed = Vec::new();

        for coord in dirty.iter() {
            if let Some(old) =
                self.upload_single_chunk(*coord, device, queue, arena, chunks, terrain_gen, scratch)
            {
                freed.push(old);
            }
        }
        // for coord in dirty {
        //     roads.road_manager.roads.update_heights_in_chunk(chunks, terrain_gen, coord);
        //     roads.road_editor.pending_chunk_rebuilds.push(coord.chunk_id());
        //     for node_id in roads.road_manager.roads.nodes_in_chunk(coord.chunk_id()) {
        //         // roads.road_editor.pending_outside_commands.push(RoadEditorCommand::Road(MakeIntersection {
        //         //     node_id,
        //         //     intersection_params: IntersectionBuildParams::default(),
        //         //     chunk_id: coord.chunk_id(),
        //         //     recalc_clearance: true
        //         // }))
        //     }
        // };
        freed
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
        // We need the base grid to replay edits from scratch.
        let mut grid = self.get_pristine_grid(terrain_gen, coord).clone();
        let state = chunks.get(&coord).map(|c| c.state.clone())?;

        self.apply_edits_to_grid(&mut grid, coord);
        recompute_patch_minmax(&mut grid);

        // Reuse existing vertex buffer if dimensions match, otherwise rebuild.
        let (vertices, indices) = if chunks
            .get(&coord)
            .map_or(false, |c| c.cpu_vertices.len() == grid.nx * grid.nz)
        {
            let existing = chunks.get(&coord).unwrap();
            let mut verts = existing.cpu_vertices.clone();
            regenerate_vertices_from_height_grid(
                &mut verts,
                &grid,
                terrain_gen,
                Some(&extract_neighbor_edges(chunks, coord)),
                false,
            );
            (verts, existing.cpu_indices.clone())
        } else {
            build_simple_grid_mesh(&grid, coord, terrain_gen)
        };

        let new_handle = arena.alloc_and_upload(device, queue, &vertices, &indices, scratch);
        let old_handle = chunks.remove(&coord).map(|c| c.handle);

        chunks.insert(
            coord,
            ChunkMeshLod {
                state,
                handle: new_handle,
                cpu_vertices: vertices,
                cpu_indices: indices,
                height_grid: Arc::new(grid),
            },
        );

        old_handle
    }
}

// ── apply_edit_to_grid ────────────────────────────────────────────────────────
//
// Applies one TerrainEdit to every grid point in the chunk.
// `chunk_wx/wz` are the world-space origins of the chunk corner.
// `cell` is the world-space distance between adjacent grid points.
// `cs`   is the world-space chunk size.

fn apply_edit_to_grid(
    edit: &TerrainEdit,
    grid: &mut ChunkHeightGrid,
    chunk_wx: f64,
    chunk_wz: f64,
    cell: f64,
    cs: f64,
) {
    match edit {
        // ── Raise ─────────────────────────────────────────────────────────────
        TerrainEdit::Raise {
            center,
            radius,
            strength,
            ..
        } => {
            let cx = center.chunk.x as f64 * cs + center.local.x as f64;
            let cz = center.chunk.z as f64 * cs + center.local.z as f64;
            let r2 = (radius * radius) as f64;

            for gx in 0..grid.nx {
                let wx = chunk_wx + gx as f64 * cell;
                let dx = (wx - cx) as f32;
                if (dx * dx) as f64 > r2 {
                    continue; // row-level early-out
                }
                for gz in 0..grid.nz {
                    let wz = chunk_wz + gz as f64 * cell;
                    let dz = (wz - cz) as f32;
                    let d2 = dx * dx + dz * dz;
                    if d2 as f64 >= r2 {
                        continue;
                    }
                    let w = smooth_weight(d2, *radius * *radius);
                    grid.heights[gx * grid.nz + gz] += strength * w;
                }
            }
        }

        // ── FlatCircle ────────────────────────────────────────────────────────
        TerrainEdit::FlatCircle {
            center,
            radius,
            terrain_offset,
            falloff_distance,
            ..
        } => {
            let cx = center.chunk.x as f64 * cs + center.local.x as f64;
            let cz = center.chunk.z as f64 * cs + center.local.z as f64;
            let total = radius + falloff_distance;
            let target = center.local.y + terrain_offset;

            for gx in 0..grid.nx {
                for gz in 0..grid.nz {
                    let wx = chunk_wx + gx as f64 * cell;
                    let wz = chunk_wz + gz as f64 * cell;
                    let dx = wx - cx;
                    let dz = wz - cz;
                    let dist = ((dx * dx + dz * dz) as f32).sqrt();
                    if dist > total {
                        continue;
                    }
                    let w = flat_weight(dist, *radius, *falloff_distance);
                    let h = &mut grid.heights[gx * grid.nz + gz];
                    *h += (target - *h) * w;
                }
            }
        }

        // ── FlatPolyline ──────────────────────────────────────────────────────
        TerrainEdit::FlatPolyline {
            centerline,
            half_width,
            terrain_offset,
            falloff_distance,
            ..
        } => {
            if centerline.is_empty() {
                return;
            }
            let total = half_width + falloff_distance;

            for gx in 0..grid.nx {
                for gz in 0..grid.nz {
                    let wx = chunk_wx + gx as f64 * cell;
                    let wz = chunk_wz + gz as f64 * cell;
                    let (dist, road_h) = closest_point_on_polyline_world(wx, wz, centerline, cs);
                    if dist > total {
                        continue;
                    }
                    let w = flat_weight(dist, *half_width, *falloff_distance);
                    let target = road_h + terrain_offset;
                    let h = &mut grid.heights[gx * grid.nz + gz];
                    *h += (target - *h) * w;
                }
            }
        }

        // ── FlatPolygon ───────────────────────────────────────────────────────
        TerrainEdit::FlatPolygon {
            polygon,
            height_offset,
            falloff_distance,
            ..
        } => {
            if polygon.len() < 3 {
                return;
            }

            let poly: Vec<(f64, f64)> = polygon
                .iter()
                .map(|p| {
                    (
                        p.chunk.x as f64 * cs + p.local.x as f64,
                        p.chunk.z as f64 * cs + p.local.z as f64,
                    )
                })
                .collect();

            for gx in 0..grid.nx {
                for gz in 0..grid.nz {
                    let wx = chunk_wx + gx as f64 * cell;
                    let wz = chunk_wz + gz as f64 * cell;

                    let (inside, dist_to_edge) = point_polygon_signed_dist(&poly, wx, wz);

                    let dist = dist_to_edge as f32;

                    let w = if inside {
                        1.0
                    } else if dist <= *falloff_distance {
                        let t = dist / falloff_distance;
                        let s = 1.0 - t;
                        s * s * (3.0 - 2.0 * s)
                    } else {
                        continue;
                    };

                    let interpolated_height =
                        WorldPos::polygon_height_at_imprecise(polygon, wx, wz);

                    let target_height = interpolated_height + *height_offset;

                    let h = &mut grid.heights[gx * grid.nz + gz];

                    *h += (target_height - *h) * w;
                }
            }
        }
    }
}

// ── Public helper: apply_edits_with_stitching ─────────────────────────────────
//
// Replaces the old `apply_accumulated_deltas_with_stitching`.
// Call from chunk_builder when building a LOD grid for a chunk that may need
// both edit replay and seam stitching to lower-detail neighbours.

#[derive(Clone, Copy, PartialEq)]
enum Edge {
    XNeg,
    XPos,
    ZNeg,
    ZPos,
}

pub fn apply_edits_with_stitching(
    base_grid: &ChunkHeightGrid,
    coord: ChunkCoord,
    terrain_edits_snapshot: &TerrainEditsSnapshot,
    loaded_chunks_snapshot: &LoadedChunksSnapshot,
    own_step: LodStep,
) -> ChunkHeightGrid {
    let has_own_edits = terrain_edits_snapshot.has_edits_on_chunk(coord);

    // FAST PATH 1: If this chunk has edits, apply them and skip stitching
    if has_own_edits {
        let mut grid = base_grid.clone();
        apply_edits_to_grid(&terrain_edits_snapshot.edits, &mut grid, coord);
        return grid;
    }

    // FAST PATH 2: Pre-check if ANY neighbor needs stitching before doing lookups
    let mut needs_stitching = false;

    // Only check if neighbor exists AND has higher step
    // Avoid the edited neighbor check in the first pass
    for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        let nb_coord = ChunkCoord::new(coord.x + dx, coord.z + dz);

        // Skip if neighbor has edits
        if terrain_edits_snapshot.has_edits_on_chunk(nb_coord) {
            continue;
        }

        // Check if loaded and needs stitching
        if let Some(nb_chunk) = loaded_chunks_snapshot.get(&nb_coord) {
            if nb_chunk.step > own_step {
                needs_stitching = true;
                break;
            }
        }
    }

    // FAST PATH 3: No edits and no stitching - return base grid directly (NO CLONE!)
    if !needs_stitching {
        return base_grid.clone(); // Still need to clone because we return owned
    }

    // Only clone when we actually need to modify
    let mut grid = base_grid.clone();

    // Define edges once
    const EDGES: [(i32, i32, Edge); 4] = [
        (-1, 0, Edge::XNeg),
        (1, 0, Edge::XPos),
        (0, -1, Edge::ZNeg),
        (0, 1, Edge::ZPos),
    ];

    // Stitch edges
    for (dx, dz, edge) in EDGES {
        let nb_coord = ChunkCoord::new(coord.x + dx, coord.z + dz);

        if terrain_edits_snapshot.has_edits_on_chunk(nb_coord) {
            continue;
        }

        if let Some(nb_chunk) = loaded_chunks_snapshot.get(&nb_coord) {
            if nb_chunk.step > own_step {
                stitch_edge_to_neighbor(&mut grid, &nb_chunk.height_grid, edge);
            }
        }
    }

    grid
}

// ── Geometry helpers ──────────────────────────────────────────────────────────

pub fn affected_chunks(center: WorldPos, radius: f32) -> (i32, i32, i32, i32) {
    let cs = chunk_size() as f64;
    let r = radius as f64;
    let wx = center.chunk.x as f64 * cs + center.local.x as f64;
    let wz = center.chunk.z as f64 * cs + center.local.z as f64;
    (
        ((wx - r) / cs).floor() as i32,
        ((wx + r) / cs).floor() as i32,
        ((wz - r) / cs).floor() as i32,
        ((wz + r) / cs).floor() as i32,
    )
}

fn polygon_chunk_bounds(polygon: &[WorldPos], falloff: f64, cs: f64) -> (i32, i32, i32, i32) {
    let mut min_wx = f64::MAX;
    let mut max_wx = f64::MIN;
    let mut min_wz = f64::MAX;
    let mut max_wz = f64::MIN;
    for p in polygon {
        let wx = p.chunk.x as f64 * cs + p.local.x as f64;
        let wz = p.chunk.z as f64 * cs + p.local.z as f64;
        min_wx = min_wx.min(wx);
        max_wx = max_wx.max(wx);
        min_wz = min_wz.min(wz);
        max_wz = max_wz.max(wz);
    }
    (
        ((min_wx - falloff) / cs).floor() as i32,
        ((max_wx + falloff) / cs).floor() as i32,
        ((min_wz - falloff) / cs).floor() as i32,
        ((max_wz + falloff) / cs).floor() as i32,
    )
}

/// Smoothstep falloff weight for Raise brush.
#[inline]
fn smooth_weight(d2: f32, r2: f32) -> f32 {
    let t = 1.0 - (d2 / r2);
    if t <= 0.0 {
        return 0.0;
    }
    t * t * (3.0 - 2.0 * t)
}

/// Flat-inside / smooth-outside weight used by FlatCircle and FlatPolyline.
#[inline]
fn flat_weight(dist: f32, inner: f32, falloff: f32) -> f32 {
    if dist <= inner {
        1.0
    } else {
        let t = (dist - inner) / falloff;
        let s = 1.0 - t;
        s * s * (3.0 - 2.0 * s)
    }
}

/// Ray-casting point-in-polygon + distance to nearest edge.
/// Returns `(inside, distance_to_polygon_boundary)`.
fn point_polygon_signed_dist(polygon: &[(f64, f64)], px: f64, pz: f64) -> (bool, f64) {
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

        // Ray-cast for inside test.
        if ((az > pz) != (bz > pz)) && (px < (bx - ax) * (pz - az) / (bz - az) + ax) {
            inside = !inside;
        }

        // Nearest point on segment.
        let dx = bx - ax;
        let dz = bz - az;
        let len_sq = dx * dx + dz * dz;
        let t = if len_sq < 1e-10 {
            0.0
        } else {
            (((px - ax) * dx + (pz - az) * dz) / len_sq).clamp(0.0, 1.0)
        };
        let qx = ax + t * dx;
        let qz = az + t * dz;
        let dist_sq = (px - qx) * (px - qx) + (pz - qz) * (pz - qz);
        min_dist_sq = min_dist_sq.min(dist_sq);
    }

    (inside, min_dist_sq.sqrt())
}

/// Nearest point on a polyline in world XZ + interpolated height.
/// Returns `(horizontal_distance, interpolated_height)`.
fn closest_point_on_polyline_world(wx: f64, wz: f64, polyline: &[WorldPos], cs: f64) -> (f32, f32) {
    let mut best_dist = f32::MAX;
    let mut best_h = 0.0_f32;

    for i in 0..polyline.len().saturating_sub(1) {
        let a = &polyline[i];
        let b = &polyline[i + 1];
        let ax = a.chunk.x as f64 * cs + a.local.x as f64;
        let az = a.chunk.z as f64 * cs + a.local.z as f64;
        let bx = b.chunk.x as f64 * cs + b.local.x as f64;
        let bz = b.chunk.z as f64 * cs + b.local.z as f64;

        let dx = bx - ax;
        let dz = bz - az;
        let len_sq = dx * dx + dz * dz;
        let t = if len_sq < 1e-10 {
            0.0
        } else {
            (((wx - ax) * dx + (wz - az) * dz) / len_sq).clamp(0.0, 1.0)
        };

        let qx = ax + t * dx;
        let qz = az + t * dz;
        let dist = (((wx - qx) * (wx - qx) + (wz - qz) * (wz - qz)) as f32).sqrt();

        if dist < best_dist {
            best_dist = dist;
            best_h =
                polyline[i].local.y + (polyline[i + 1].local.y - polyline[i].local.y) * t as f32;
        }
    }

    // Handle degenerate single-point case.
    if polyline.len() == 1 {
        let a = &polyline[0];
        let ax = a.chunk.x as f64 * cs + a.local.x as f64;
        let az = a.chunk.z as f64 * cs + a.local.z as f64;
        best_dist = (((wx - ax) * (wx - ax) + (wz - az) * (wz - az)) as f32).sqrt();
        best_h = polyline[0].local.y;
    }

    (best_dist, best_h)
}

// ── Mesh / grid utilities (unchanged) ────────────────────────────────────────

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

pub fn build_simple_grid_mesh(
    grid: &ChunkHeightGrid,
    coord: ChunkCoord,
    terrain_gen: &TerrainGenerator,
) -> (Vec<Vertex>, Vec<u32>) {
    let nx = grid.nx;
    let nz = grid.nz;
    let cell = grid.step_f32();
    let step = grid.step as usize;
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
            let m = terrain_gen.moisture(&pos, h);
            let c = terrain_gen.color(&pos, h, m);

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
            let i01 = (gx * nz + gz + 1) as u32;
            let i11 = ((gx + 1) * nz + gz + 1) as u32;
            indices.extend_from_slice(&[i00, i10, i01, i01, i10, i11]);
        }
    }

    (vertices, indices)
}

// ── LOD edge stitching (unchanged logic, now standalone) ──────────────────────

fn stitch_x_edge(grid: &mut ChunkHeightGrid, nb: &ChunkHeightGrid, our_gx: usize, nb_gx: usize) {
    let our_cell = grid.step_f32();
    let nb_cell = nb.step_f32();
    for our_gz in 0..grid.nz {
        let local_z = our_gz as f32 * our_cell;
        let nb_gz_f = local_z / nb_cell;
        let nb_gz_lo = (nb_gz_f.floor() as usize).min(nb.nz - 1);
        let nb_gz_hi = (nb_gz_lo + 1).min(nb.nz - 1);
        let t = (nb_gz_f - nb_gz_lo as f32).clamp(0.0, 1.0);
        let h0 = nb.heights[nb_gx * nb.nz + nb_gz_lo];
        let h1 = nb.heights[nb_gx * nb.nz + nb_gz_hi];
        grid.heights[our_gx * grid.nz + our_gz] = h0 + t * (h1 - h0);
    }
}

fn stitch_z_edge(grid: &mut ChunkHeightGrid, nb: &ChunkHeightGrid, our_gz: usize, nb_gz: usize) {
    let our_cell = grid.step_f32();
    let nb_cell = nb.step_f32();
    for our_gx in 0..grid.nx {
        let local_x = our_gx as f32 * our_cell;
        let nb_gx_f = local_x / nb_cell;
        let nb_gx_lo = (nb_gx_f.floor() as usize).min(nb.nx - 1);
        let nb_gx_hi = (nb_gx_lo + 1).min(nb.nx - 1);
        let t = (nb_gx_f - nb_gx_lo as f32).clamp(0.0, 1.0);
        let h0 = nb.heights[nb_gx_lo * nb.nz + nb_gz];
        let h1 = nb.heights[nb_gx_hi * nb.nz + nb_gz];
        grid.heights[our_gx * grid.nz + our_gz] = h0 + t * (h1 - h0);
    }
}

fn stitch_edge_to_neighbor(grid: &mut ChunkHeightGrid, nb: &ChunkHeightGrid, edge: Edge) {
    match edge {
        Edge::XNeg => stitch_x_edge(grid, nb, 0, nb.nx - 1),
        Edge::XPos => stitch_x_edge(grid, nb, grid.nx - 1, 0),
        Edge::ZNeg => stitch_z_edge(grid, nb, 0, nb.nz - 1),
        Edge::ZPos => stitch_z_edge(grid, nb, grid.nz - 1, 0),
    }
}

/// Apply every relevant edit to `grid` in-place for chunk `coord`.
///
/// The grid should contain the **base** (generator) heights before calling
/// this. Each edit is applied in registration order; the result is
/// deterministic and matches what was authored at edit time.
pub fn apply_edits_to_grid(
    edits: &Vec<TerrainEdit>,
    grid: &mut ChunkHeightGrid,
    coord: ChunkCoord,
) {
    let cs = chunk_size() as f64;
    let cell = grid.step_f32() as f64;
    let chunk_wx = coord.x as f64 * cs;
    let chunk_wz = coord.z as f64 * cs;

    for edit in edits {
        let (min_cx, max_cx, min_cz, max_cz) = edit.chunk_bounds();
        // Cheap AABB cull before doing any per-vertex work.
        if coord.x < min_cx || coord.x > max_cx || coord.z < min_cz || coord.z > max_cz {
            continue;
        }
        apply_edit_to_grid(edit, grid, chunk_wx, chunk_wz, cell, cs);
    }
}

pub fn extract_neighbor_edges(
    chunks: &HashMap<ChunkCoord, ChunkMeshLod>,
    coord: ChunkCoord,
) -> NeighborEdgeHeights {
    let mut edges = NeighborEdgeHeights::empty();

    // +X neighbor: extract its -X edge (first column, gx=0)
    if let Some(chunk) = chunks.get(&ChunkCoord::new(coord.x + 1, coord.z)) {
        let grid = &chunk.height_grid;
        let mut edge_heights = Vec::with_capacity(grid.nz);
        for gz in 0..grid.nz {
            edge_heights.push(grid.heights[gz]); // gx=0, varying gz
        }
        edges.pos_x = Some(edge_heights);
    }

    // -X neighbor: extract its +X edge (last column)
    if let Some(chunk) = chunks.get(&ChunkCoord::new(coord.x - 1, coord.z)) {
        let grid = &chunk.height_grid;
        let last_gx = grid.nx - 1;
        let mut edge_heights = Vec::with_capacity(grid.nz);
        for gz in 0..grid.nz {
            edge_heights.push(grid.heights[last_gx * grid.nz + gz]);
        }
        edges.neg_x = Some(edge_heights);
    }

    // +Z neighbor: extract its -Z edge (first row, gz=0)
    if let Some(chunk) = chunks.get(&ChunkCoord::new(coord.x, coord.z + 1)) {
        let grid = &chunk.height_grid;
        let mut edge_heights = Vec::with_capacity(grid.nx);
        for gx in 0..grid.nx {
            edge_heights.push(grid.heights[gx * grid.nz]); // gz=0, varying gx
        }
        edges.pos_z = Some(edge_heights);
    }

    // -Z neighbor: extract its +Z edge (last row)
    if let Some(chunk) = chunks.get(&ChunkCoord::new(coord.x, coord.z - 1)) {
        let grid = &chunk.height_grid;
        let last_gz = grid.nz - 1;
        let mut edge_heights = Vec::with_capacity(grid.nx);
        for gx in 0..grid.nx {
            edge_heights.push(grid.heights[gx * grid.nz + last_gz]);
        }
        edges.neg_z = Some(edge_heights);
    }

    edges
}

pub fn compute_best_fit_height_for_polygon(terrain: &Terrain, polygon: &[WorldPos]) -> f32 {
    let centroid = WorldPos::centroid(polygon);

    // Sample terrain at centroid and a few offset points to estimate slope
    let base = terrain.get_height_at(centroid, true);

    // Small cross pattern to estimate local slope
    let offset = 4.0f32;
    let samples = [
        centroid.add_vec2(Vec2::new(offset, 0.0)),
        centroid.add_vec2(Vec2::new(-offset, 0.0)),
        centroid.add_vec2(Vec2::new(0.0, offset)),
        centroid.add_vec2(Vec2::new(0.0, -offset)),
    ];

    let mut sum = base;
    let mut count = 1;

    for p in samples {
        let h = terrain.get_height_at(p, true);
        sum += h;
        count += 1;
    }

    sum / count as f32
}
