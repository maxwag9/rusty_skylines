// intersections.rs - Fixed Clipper2 API usage

use crate::helpers::positions::{ChunkSize, WorldPos};
use crate::renderer::gizmo::{DEBUG_DRAW_DURATION, Gizmo};
use crate::world::roads::road_editor::{polyline_cumulative_lengths, sample_polyline_at};
use crate::world::roads::road_helpers::*;
use crate::world::roads::road_mesh_manager::*;
use crate::world::roads::road_structs::*;
use crate::world::roads::roads::*;
use crate::world::terrain::terrain_subsystem::TerrainSubsystem;
use clipper2::{EndType, JoinType, Path, Paths};
use earcutr;
use glam::{Vec2, Vec3};
use std::collections::HashMap;
use std::f32::consts::TAU;

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_CORRIDOR_LENGTH: f32 = 30.0;

// ============================================================================
// Intersection Polygon
// ============================================================================

#[derive(Clone, Debug, Default)]
pub struct IntersectionPolygon {
    pub ring: Vec<WorldPos>,
    pub origin: WorldPos,
    pub chunk_size: ChunkSize,
}

impl IntersectionPolygon {
    pub fn new(ring: Vec<WorldPos>, origin: WorldPos, chunk_size: ChunkSize) -> Self {
        Self {
            ring,
            origin,
            chunk_size,
        }
    }

    pub fn empty(chunk_size: ChunkSize) -> Self {
        Self {
            ring: Vec::new(),
            origin: WorldPos::zero(),
            chunk_size,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.ring.len() < 3
    }

    fn world_pos_to_local_2d(&self, p: WorldPos) -> Vec2 {
        let rel = p.to_render_pos(self.origin, self.chunk_size);
        Vec2::new(rel.x, rel.z)
    }

    fn world_pos_from_local_2d(&self, v: Vec2, y: f32) -> WorldPos {
        self.origin
            .add_vec3(Vec3::new(v.x, y, v.y), self.chunk_size)
    }

    /// Point-in-polygon using winding number.
    pub fn contains_xz(&self, p: WorldPos) -> bool {
        if self.ring.len() < 3 {
            return false;
        }

        let test = self.world_pos_to_local_2d(p);
        let mut winding = 0i32;

        for i in 0..self.ring.len() {
            let a = self.world_pos_to_local_2d(self.ring[i]);
            let b = self.world_pos_to_local_2d(self.ring[(i + 1) % self.ring.len()]);

            if a.y <= test.y {
                if b.y > test.y && cross_2d(b - a, test - a) > 0.0 {
                    winding += 1;
                }
            } else if b.y <= test.y && cross_2d(b - a, test - a) < 0.0 {
                winding -= 1;
            }
        }

        winding != 0
    }

    pub fn first_intersection(&self, from: WorldPos, to: WorldPos) -> Option<(f32, WorldPos)> {
        let n = self.ring.len();
        if n < 3 {
            return None;
        }

        let p0 = self.world_pos_to_local_2d(from);
        let p1 = self.world_pos_to_local_2d(to);
        let d = p1 - p0;

        if d.length_squared() < 1e-10 {
            return None;
        }

        let mut best_t = f32::MAX;

        for i in 0..n {
            let a = self.world_pos_to_local_2d(self.ring[i]);
            let b = self.world_pos_to_local_2d(self.ring[(i + 1) % n]);
            let e = b - a;

            let denom = cross_2d(d, e);
            if denom.abs() < 1e-10 {
                continue;
            }

            let w = a - p0;
            let t = cross_2d(w, e) / denom;
            let u = cross_2d(w, d) / denom;

            // Allow t >= -epsilon to catch "start on boundary" cases
            if t >= -1e-6 && t < best_t && u >= -1e-6 && u <= 1.0 + 1e-6 {
                best_t = t.max(0.0);
            }
        }

        if best_t <= 1.0 {
            let hit_2d = p0 + d * best_t;
            let y = from.local.y + (to.local.y - from.local.y) * best_t;
            Some((best_t, self.world_pos_from_local_2d(hit_2d, y)))
        } else {
            None
        }
    }

    pub fn first_polyline_intersection(&self, polyline: &[WorldPos]) -> Option<(f32, WorldPos)> {
        if polyline.len() < 2 || self.ring.len() < 3 {
            return None;
        }

        let mut cumulative = 0.0f32;

        for i in 0..polyline.len() - 1 {
            let seg_start = polyline[i];
            let seg_end = polyline[i + 1];
            let seg_len = seg_start.distance_to(seg_end, self.chunk_size);

            if seg_len < 1e-6 {
                continue;
            }

            if let Some((t, hit)) = self.first_intersection(seg_start, seg_end) {
                return Some((cumulative + t * seg_len, hit));
            }

            cumulative += seg_len;
        }

        None
    }

    pub fn centroid(&self) -> WorldPos {
        if self.ring.is_empty() {
            return self.origin;
        }

        let mut sum = Vec2::ZERO;
        for p in &self.ring {
            sum += self.world_pos_to_local_2d(*p);
        }
        let avg = sum / self.ring.len() as f32;
        self.world_pos_from_local_2d(avg, self.origin.local.y)
    }

    pub fn signed_area(&self) -> f32 {
        let n = self.ring.len();
        if n < 3 {
            return 0.0;
        }

        let mut area = 0.0f32;
        for i in 0..n {
            let a = self.world_pos_to_local_2d(self.ring[i]);
            let b = self.world_pos_to_local_2d(self.ring[(i + 1) % n]);
            area += cross_2d(a, b);
        }
        area * 0.5
    }
}

fn cross_2d(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

// ============================================================================
// Local Polygon (for Clipper operations)
// ============================================================================

#[derive(Clone, Debug)]
struct LocalPolygon {
    points: Vec<Vec2>,
}

impl LocalPolygon {
    fn new(points: Vec<Vec2>) -> Self {
        Self { points }
    }

    /// Create a corridor rectangle from origin along a direction.
    fn corridor(direction: Vec2, half_width: f32, fwd_len: f32, back_len: f32) -> Self {
        let dir = direction.normalize_or_zero();
        if dir == Vec2::ZERO {
            return Self::new(Vec::new());
        }
        let perp = Vec2::new(-dir.y, dir.x);

        let a = -dir * back_len;
        let b = dir * fwd_len;

        let points = vec![
            a - perp * half_width,
            b - perp * half_width,
            b + perp * half_width,
            a + perp * half_width,
        ];
        Self::new(points)
    }

    /// Convert to clipper2 Paths format.
    fn to_clipper_paths(&self) -> Paths {
        if self.points.len() < 3 {
            return Vec::<Path>::new().into();
        }

        let coords: Vec<(f64, f64)> = self
            .points
            .iter()
            .map(|p| (p.x as f64, p.y as f64))
            .collect();

        vec![coords].into()
    }

    /// Create from clipper2 output path.
    fn from_clipper_coords(coords: Vec<(f64, f64)>) -> Self {
        let points = coords
            .into_iter()
            .map(|(x, y)| Vec2::new(x as f32, y as f32))
            .collect();
        Self::new(points)
    }
}

// ============================================================================
// Clipper2 Operations (Fixed API)
// ============================================================================

/// Compute signed area of polygon (positive for CCW, negative for CW)
fn polygon_signed_area(points: &[Vec2]) -> f32 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0f32;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        area += cross_2d(a, b);
    }
    area * 0.5
}

/// Union multiple polygons using Clipper2.
/// Returns only outer boundaries (filters out holes based on winding).
fn union_polygons(polygons: &[LocalPolygon]) -> Vec<LocalPolygon> {
    use clipper2::{FillRule, Paths, union};

    let valid: Vec<_> = polygons.iter().filter(|p| p.points.len() >= 3).collect();
    if valid.is_empty() {
        return vec![];
    }
    if valid.len() == 1 {
        return vec![valid[0].clone()];
    }

    let all_coords: Vec<Vec<(f64, f64)>> = valid
        .iter()
        .map(|p| {
            p.points
                .iter()
                .map(|pt| (pt.x as f64, pt.y as f64))
                .collect()
        })
        .collect();

    let subject: Paths = all_coords.into();
    let clip: Paths = Vec::<Vec<(f64, f64)>>::new().into();

    let result = match union(subject, clip, FillRule::NonZero) {
        Ok(r) => r,
        Err(_) => return vec![],
    };

    let output: Vec<Vec<(f64, f64)>> = result.into();

    let mut polys: Vec<LocalPolygon> = output
        .into_iter()
        .filter(|path| path.len() >= 3)
        .map(LocalPolygon::from_clipper_coords)
        .collect();

    // Optional: normalize all to CCW for your downstream code
    for p in &mut polys {
        if polygon_signed_area(&p.points) < 0.0 {
            p.points.reverse();
        }
    }

    polys
}

/// Offset a polygon outward (positive) or inward (negative).
fn offset_polygon(polygon: &LocalPolygon, offset: f32, round_corners: bool) -> Vec<LocalPolygon> {
    if polygon.points.len() < 3 || offset.abs() < 0.001 {
        return vec![polygon.clone()];
    }

    let coords: Vec<(f64, f64)> = polygon
        .points
        .iter()
        .map(|p| (p.x as f64, p.y as f64))
        .collect();

    let paths: Paths = vec![coords].into();

    let join_type = if round_corners {
        JoinType::Round
    } else {
        JoinType::Miter
    };

    // inflate(delta, join_type, end_type, miter_limit)
    let result = paths.inflate(offset as f64, join_type, EndType::Polygon, 2.0);

    let output: Vec<Vec<(f64, f64)>> = result.into();

    output
        .into_iter()
        .filter(|path| path.len() >= 3)
        .map(LocalPolygon::from_clipper_coords)
        .collect()
}

/// Simplify a polygon to remove redundant vertices.
fn simplify_clipper(polygon: &LocalPolygon, tolerance: f32) -> LocalPolygon {
    if polygon.points.len() < 4 || tolerance <= 0.0 {
        return polygon.clone();
    }

    let coords: Vec<(f64, f64)> = polygon
        .points
        .iter()
        .map(|p| (p.x as f64, p.y as f64))
        .collect();

    let paths: Paths = vec![coords].into();

    // simplify(tolerance, is_open)
    let result = paths.simplify(tolerance as f64, false);

    let output: Vec<Vec<(f64, f64)>> = result.into();

    output
        .into_iter()
        .next()
        .filter(|path| path.len() >= 3)
        .map(LocalPolygon::from_clipper_coords)
        .unwrap_or_else(|| polygon.clone())
}

// ============================================================================
// Intersection Geometry
// ============================================================================

#[derive(Clone, Debug)]
pub struct IntersectionGeometry {
    pub center: WorldPos,
    pub arms: Vec<Arm>,
    pub polygon: IntersectionPolygon,
    pub sidewalk_polygon: Option<IntersectionPolygon>,
}

#[derive(Clone, Debug, Default)]
pub struct IntersectionMeshResult {
    pub polygon: IntersectionPolygon,
}

// ============================================================================
// Build Parameters
// ============================================================================

#[derive(Clone, Debug)]
pub struct IntersectionBuildParams {
    pub corridor_length: f32,
    pub turn_samples: usize,
    pub lane_width_m: f32,
    pub turn_tightness: f32,
    pub sidewalk_width: f32,
    pub round_corners: bool,
    pub simplify_tolerance: f32,
}

impl Default for IntersectionBuildParams {
    fn default() -> Self {
        Self {
            corridor_length: DEFAULT_CORRIDOR_LENGTH,
            turn_samples: 12,
            lane_width_m: 3.5,
            turn_tightness: 1.0,
            sidewalk_width: 2.0,
            round_corners: true,
            simplify_tolerance: 0.1,
        }
    }
}

impl IntersectionBuildParams {
    pub fn from_style(style: &RoadStyleParams) -> Self {
        let road_type = style.road_type();
        Self {
            lane_width_m: road_type.lane_width,
            turn_tightness: style.turn_tightness(),
            sidewalk_width: road_type.sidewalk_width,
            ..Default::default()
        }
    }
}

// ============================================================================
// Main Intersection Building
// ============================================================================

pub fn build_intersection_at_node(
    storage: &mut RoadStorage,
    node_id: NodeId,
    params: &IntersectionBuildParams,
    recalc_clearance: bool,
    gizmo: &mut Gizmo,
) {
    let chunk_size = gizmo.chunk_size;

    if recalc_clearance {
        if let Some(geom) = compute_intersection_geometry(storage, node_id, params, chunk_size) {
            debug_draw_polygon(&geom.polygon, [0.0, 0.8, 0.2], gizmo);
            if let Some(ref sw) = geom.sidewalk_polygon {
                debug_draw_polygon(sw, [0.5, 0.5, 0.5], gizmo);
            }

            carve_lanes_with_polygon(storage, node_id, &geom, chunk_size, gizmo);
        }
    }

    storage.node_mut(node_id).clear_node_lanes();
    let node_lanes = build_node_lanes_for_intersection(storage, node_id, params, chunk_size);
    storage.node_mut(node_id).add_node_lanes(node_lanes);
}

fn debug_draw_polygon(poly: &IntersectionPolygon, color: [f32; 3], gizmo: &mut Gizmo) {
    if poly.ring.len() < 2 {
        return;
    }
    for i in 0..poly.ring.len() {
        let a = poly.ring[i];
        let b = poly.ring[(i + 1) % poly.ring.len()];
        gizmo.line(a, b, color, DEBUG_DRAW_DURATION);
    }
}

fn compute_intersection_geometry(
    storage: &RoadStorage,
    node_id: NodeId,
    params: &IntersectionBuildParams,
    chunk_size: ChunkSize,
) -> Option<IntersectionGeometry> {
    let node = storage.node(node_id)?;
    let center = node.position();

    let arms = gather_arms(storage, node_id, params, chunk_size);
    if arms.len() < 2 {
        return None;
    }

    // Extend back_len to ensure lane endpoints are strictly inside, not on boundary
    let back_len = params.corridor_length * 0.0;

    let corridors: Vec<LocalPolygon> = arms
        .iter()
        .map(|arm| {
            let dir_2d = Vec2::new(arm.direction().x, arm.direction().z);
            LocalPolygon::corridor(dir_2d, arm.half_width(), params.corridor_length, back_len)
        })
        .collect();

    let unioned = union_polygons(&corridors);
    if unioned.is_empty() {
        return None;
    }

    // Take the largest outer boundary
    let main_poly = unioned.into_iter().max_by(|a, b| {
        polygon_area(&a.points)
            .partial_cmp(&polygon_area(&b.points))
            .unwrap_or(std::cmp::Ordering::Equal)
    })?;

    let simplified = simplify_clipper(&main_poly, params.simplify_tolerance);

    let ring: Vec<WorldPos> = simplified
        .points
        .iter()
        .map(|v| center.add_vec3(Vec3::new(v.x, 0.0, v.y), chunk_size))
        .collect();

    let polygon = IntersectionPolygon::new(ring, center, chunk_size);

    let sidewalk_polygon = if params.sidewalk_width > 0.01 {
        let offset_polys = offset_polygon(&simplified, params.sidewalk_width, params.round_corners);
        offset_polys.into_iter().next().map(|p| {
            let ring: Vec<WorldPos> = p
                .points
                .iter()
                .map(|v| center.add_vec3(Vec3::new(v.x, 0.0, v.y), chunk_size))
                .collect();
            IntersectionPolygon::new(ring, center, chunk_size)
        })
    } else {
        None
    };

    Some(IntersectionGeometry {
        center,
        arms,
        polygon,
        sidewalk_polygon,
    })
}

fn polygon_area(points: &[Vec2]) -> f32 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0f32;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        area += cross_2d(a, b);
    }
    area.abs() * 0.5
}

fn gather_arms(
    storage: &RoadStorage,
    node_id: NodeId,
    params: &IntersectionBuildParams,
    chunk_size: ChunkSize,
) -> Vec<Arm> {
    let segment_ids = storage.enabled_segments_connected_to_node(node_id);
    let node = match storage.node(node_id) {
        Some(n) => n,
        None => return Vec::new(),
    };

    let mut arms: Vec<Arm> = segment_ids
        .into_iter()
        .filter_map(|seg_id| {
            let segment = storage.segment(seg_id);
            let points_to_node = segment.end() == node_id;

            let lane_ids: Vec<LaneId> = segment
                .lanes()
                .iter()
                .copied()
                .filter(|id| storage.lane(id).is_enabled())
                .collect();

            if lane_ids.is_empty() {
                return None;
            }

            let node_pos = node.position();
            let direction = compute_arm_direction(storage, &lane_ids, node_pos, chunk_size)?;

            let mut bearing = direction.z.atan2(direction.x);
            if bearing < 0.0 {
                bearing += TAU;
            }

            let lane_count = lane_ids.len();
            let half_width = lane_count as f32 * params.lane_width_m * 0.5 + params.sidewalk_width;

            let mut arm = Arm::new(seg_id, bearing, direction, half_width, points_to_node);

            for lane_id in &lane_ids {
                let lane = storage.lane(lane_id);
                let flows_to_node = lane.lane_index() > 0;
                if flows_to_node {
                    arm.add_incoming_lane(*lane_id);
                } else {
                    arm.add_outgoing_lane(*lane_id);
                }
            }

            arm.sort_lanes_by_index(storage);
            Some(arm)
        })
        .collect();

    arms.sort_by(|a, b| {
        a.bearing()
            .partial_cmp(&b.bearing())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    arms
}

// ============================================================================
// Lane Carving
// ============================================================================

fn carve_lanes_with_polygon(
    storage: &mut RoadStorage,
    node_id: NodeId,
    geom: &IntersectionGeometry,
    chunk_size: ChunkSize,
    gizmo: &mut Gizmo,
) {
    let mut edits: Vec<(LaneId, LaneGeometry)> = Vec::new();

    for arm in &geom.arms {
        let all_lanes: Vec<LaneId> = arm
            .incoming_lanes()
            .iter()
            .chain(arm.outgoing_lanes().iter())
            .copied()
            .collect();

        for lane_id in all_lanes {
            let lane = storage.lane(&lane_id);
            if !lane.is_enabled() {
                continue;
            }

            let pts = lane.polyline();
            if pts.len() < 2 {
                continue;
            }

            let node_pos = storage.node(node_id).unwrap().position();
            let Some(node_idx) = endpoint_index_near_node(pts, node_pos, chunk_size) else {
                continue;
            };

            let outward_poly: Vec<WorldPos> = if node_idx == 0 {
                pts.to_vec()
            } else {
                pts.iter().rev().copied().collect()
            };

            let Some((dist_to_boundary, hit)) =
                geom.polygon.first_polyline_intersection(&outward_poly)
            else {
                continue;
            };

            // REMOVED: dist_to_boundary >= 0.01 check - boundary cases are valid

            //gizmo.cross(hit, 0.2, [1.0, 0.0, 0.0], DEBUG_DRAW_DURATION);

            let new_pts = if node_idx == 0 {
                modify_polyline_start(pts, dist_to_boundary, chunk_size)
            } else {
                modify_polyline_end(pts, dist_to_boundary, chunk_size)
            };

            if let Some(new_pts) = new_pts {
                edits.push((lane_id, LaneGeometry::from_polyline(new_pts, chunk_size)));
            }
        }
    }

    for (lane_id, geom) in edits {
        storage.lane_mut(lane_id).replace_geometry(geom);
    }
}

// ============================================================================
// Triangulation with Earcut
// ============================================================================

/// Triangulate a simple polygon using earcut.
pub fn triangulate_polygon(ring: &[Vec2], base_index: u32, indices: &mut Vec<u32>) -> bool {
    if ring.len() < 3 {
        return false;
    }

    let coords: Vec<f64> = ring.iter().flat_map(|p| [p.x as f64, p.y as f64]).collect();

    let hole_indices: Vec<usize> = Vec::new();

    let tri_indices = earcutr::earcut(&coords, &hole_indices, 2).unwrap_or(Vec::new());

    if tri_indices.is_empty() {
        return false;
    }

    for idx in tri_indices {
        indices.push(base_index + idx as u32);
    }

    true
}

/// Triangulate a polygon with a hole (for sidewalk ring).
pub fn triangulate_polygon_with_hole(
    outer: &[Vec2],
    inner: &[Vec2],
    base_index: u32,
    indices: &mut Vec<u32>,
) -> bool {
    if outer.len() < 3 || inner.len() < 3 {
        return false;
    }

    let mut coords: Vec<f64> = Vec::with_capacity((outer.len() + inner.len()) * 2);

    for p in outer {
        coords.push(p.x as f64);
        coords.push(p.y as f64);
    }
    for p in inner {
        coords.push(p.x as f64);
        coords.push(p.y as f64);
    }

    let hole_indices = vec![outer.len()];

    let tri_indices = earcutr::earcut(&coords, &hole_indices, 2).unwrap_or(Vec::new());

    if tri_indices.is_empty() {
        return false;
    }

    for idx in tri_indices {
        indices.push(base_index + idx as u32);
    }

    true
}

// ============================================================================
// Intersection Mesh Builder
// ============================================================================

pub fn build_intersection_mesh(
    terrain: &TerrainSubsystem,
    _node_id: NodeId,
    node: &Node,
    storage: &RoadStorage,
    style: &RoadStyleParams,
    config: &MeshConfig,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
    gizmo: &mut Gizmo,
) -> IntersectionMeshResult {
    let center = node.position();
    let node_lanes = node.node_lanes();
    let chunk_size = terrain.chunk_size;

    if node_lanes.len() < 2 {
        return IntersectionMeshResult::default();
    }

    let road_type = style.road_type();
    let params = IntersectionBuildParams::from_style(style);

    let arms = gather_arms_from_node_lanes(node_lanes, storage, chunk_size, &params);
    if arms.len() < 2 {
        return IntersectionMeshResult::default();
    }

    // Extend back_len to ensure corridors overlap at center
    let back_len = params.corridor_length * 0.5;

    let corridors: Vec<LocalPolygon> = arms
        .iter()
        .map(|arm| {
            let dir_2d = Vec2::new(arm.direction().x, arm.direction().z);
            LocalPolygon::corridor(
                dir_2d,
                arm.half_width() * 1.2,
                params.corridor_length,
                back_len,
            )
        })
        .collect();

    let unioned = union_polygons(&corridors);

    let Some(main_poly) = unioned.into_iter().max_by(|a, b| {
        polygon_area(&a.points)
            .partial_cmp(&polygon_area(&b.points))
            .unwrap_or(std::cmp::Ordering::Equal)
    }) else {
        return IntersectionMeshResult::default();
    };

    let simplified = simplify_clipper(&main_poly, params.simplify_tolerance);
    if simplified.points.len() < 3 {
        return IntersectionMeshResult::default();
    }

    let asphalt_base = vertices.len() as u32;

    for p2d in &simplified.points {
        let mut pos = center.add_vec3(Vec3::new(p2d.x, 0.0, p2d.y), chunk_size);
        set_point_height_with_structure_type(terrain, road_type.structure(), &mut pos);
        pos.local.y += road_type.lane_height;

        let (u, v) = radial_uv(center, pos, chunk_size, config);
        vertices.push(road_vertex(
            pos,
            [0.0, 1.0, 0.0],
            road_type.lane_material_id,
            u,
            v,
        ));
    }

    triangulate_polygon(&simplified.points, asphalt_base, indices);

    if road_type.sidewalk_width > 0.01 {
        let offset_polys =
            offset_polygon(&simplified, road_type.sidewalk_width, params.round_corners);

        if let Some(outer_poly) = offset_polys.into_iter().next() {
            let outer_simplified = simplify_clipper(&outer_poly, params.simplify_tolerance);

            if outer_simplified.points.len() >= 3 {
                build_sidewalk_ring(
                    terrain,
                    center,
                    &simplified.points,
                    &outer_simplified.points,
                    road_type,
                    config,
                    vertices,
                    indices,
                );
            }
        }
    }

    let ring: Vec<WorldPos> = simplified
        .points
        .iter()
        .map(|v| center.add_vec3(Vec3::new(v.x, 0.0, v.y), chunk_size))
        .collect();

    IntersectionMeshResult {
        polygon: IntersectionPolygon::new(ring, center, chunk_size),
    }
}

fn gather_arms_from_node_lanes(
    node_lanes: &[NodeLane],
    storage: &RoadStorage,
    chunk_size: ChunkSize,
    params: &IntersectionBuildParams,
) -> Vec<Arm> {
    let mut segment_dirs: HashMap<SegmentId, (Vec3, f32)> = HashMap::new();

    for nl in node_lanes {
        if let Some(&LaneRef::Segment(lane_id, _)) = nl.merging().first() {
            let lane = storage.lane(&lane_id);
            let seg_id = lane.segment();

            let poly = nl.polyline();
            if poly.len() >= 2 {
                let dir = poly[0].delta_to(poly[1], chunk_size);
                let dir = Vec3::new(dir.x, 0.0, dir.z).normalize_or_zero();

                if dir != Vec3::ZERO {
                    let segment = storage.segment(seg_id);
                    let half_width = segment.lanes().len() as f32 * params.lane_width_m * 0.5
                        + params.sidewalk_width;
                    segment_dirs.insert(seg_id, (dir, half_width));
                }
            }
        }
    }

    segment_dirs
        .into_iter()
        .map(|(seg_id, (dir, half_width))| {
            let mut bearing = dir.z.atan2(dir.x);
            if bearing < 0.0 {
                bearing += TAU;
            }
            Arm::new(seg_id, bearing, dir, half_width, false)
        })
        .collect()
}

fn build_sidewalk_ring(
    terrain: &TerrainSubsystem,
    center: WorldPos,
    inner: &[Vec2],
    outer: &[Vec2],
    road_type: &RoadType,
    config: &MeshConfig,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    let chunk_size = terrain.chunk_size;
    let base = vertices.len() as u32;

    // Outer ring vertices
    for p2d in outer {
        let mut pos = center.add_vec3(Vec3::new(p2d.x, 0.0, p2d.y), chunk_size);
        set_point_height_with_structure_type(terrain, road_type.structure(), &mut pos);
        pos.local.y += road_type.sidewalk_height;

        let (u, v) = radial_uv(center, pos, chunk_size, config);
        vertices.push(road_vertex(
            pos,
            [0.0, 1.0, 0.0],
            road_type.sidewalk_material_id,
            u,
            v,
        ));
    }

    // Inner ring vertices (reversed for hole winding)
    let inner_reversed: Vec<Vec2> = inner.iter().rev().copied().collect();
    for p2d in &inner_reversed {
        let mut pos = center.add_vec3(Vec3::new(p2d.x, 0.0, p2d.y), chunk_size);
        set_point_height_with_structure_type(terrain, road_type.structure(), &mut pos);
        pos.local.y += road_type.sidewalk_height;

        let (u, v) = radial_uv(center, pos, chunk_size, config);
        vertices.push(road_vertex(
            pos,
            [0.0, 1.0, 0.0],
            road_type.sidewalk_material_id,
            u,
            v,
        ));
    }

    triangulate_polygon_with_hole(outer, &inner_reversed, base, indices);

    // Curb faces
    build_curb_faces(terrain, center, inner, road_type, config, vertices, indices);
}

fn build_curb_faces(
    terrain: &TerrainSubsystem,
    center: WorldPos,
    inner_ring: &[Vec2],
    road_type: &RoadType,
    config: &MeshConfig,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    let chunk_size = terrain.chunk_size;
    let curb_height = road_type.sidewalk_height - road_type.lane_height;

    if curb_height < 0.01 || inner_ring.len() < 3 {
        return;
    }

    for i in 0..inner_ring.len() {
        let p0 = inner_ring[i];
        let p1 = inner_ring[(i + 1) % inner_ring.len()];

        let mut pos0_low = center.add_vec3(Vec3::new(p0.x, 0.0, p0.y), chunk_size);
        let mut pos1_low = center.add_vec3(Vec3::new(p1.x, 0.0, p1.y), chunk_size);

        set_point_height_with_structure_type(terrain, road_type.structure(), &mut pos0_low);
        set_point_height_with_structure_type(terrain, road_type.structure(), &mut pos1_low);

        pos0_low.local.y += road_type.lane_height;
        pos1_low.local.y += road_type.lane_height;

        let mut pos0_high = pos0_low;
        let mut pos1_high = pos1_low;
        pos0_high.local.y += curb_height;
        pos1_high.local.y += curb_height;

        let edge = Vec2::new(p1.x - p0.x, p1.y - p0.y);
        let normal_2d = Vec2::new(edge.y, -edge.x).normalize_or_zero();
        let normal = [normal_2d.x, 0.0, normal_2d.y];

        let base = vertices.len() as u32;

        vertices.push(road_vertex(
            pos0_low,
            normal,
            road_type.sidewalk_material_id,
            0.0,
            0.0,
        ));
        vertices.push(road_vertex(
            pos1_low,
            normal,
            road_type.sidewalk_material_id,
            1.0,
            0.0,
        ));
        vertices.push(road_vertex(
            pos1_high,
            normal,
            road_type.sidewalk_material_id,
            1.0,
            1.0,
        ));
        vertices.push(road_vertex(
            pos0_high,
            normal,
            road_type.sidewalk_material_id,
            0.0,
            1.0,
        ));

        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

fn radial_uv(
    center: WorldPos,
    point: WorldPos,
    chunk_size: ChunkSize,
    config: &MeshConfig,
) -> (f32, f32) {
    let delta = center.delta_to(point, chunk_size);
    let dist = (delta.x * delta.x + delta.z * delta.z).sqrt();

    if dist <= 1e-6 {
        return (0.5, 0.5);
    }

    let nx = delta.x / dist;
    let nz = delta.z / dist;

    (
        0.5 + nx * dist * config.uv_scale_u,
        0.5 + nz * dist * config.uv_scale_v,
    )
}

pub fn road_vertex(world_pos: WorldPos, normal: [f32; 3], mat: u32, u: f32, v: f32) -> RoadVertex {
    RoadVertex {
        local_position: [world_pos.local.x, world_pos.local.y, world_pos.local.z],
        normal,
        uv: [u, v],
        material_id: mat,
        chunk_xz: [world_pos.chunk.x, world_pos.chunk.z],
    }
}

// ============================================================================
// Node Lane Building
// ============================================================================

fn build_node_lanes_for_intersection(
    storage: &RoadStorage,
    node_id: NodeId,
    params: &IntersectionBuildParams,
    chunk_size: ChunkSize,
) -> Vec<NodeLane> {
    let Some(node) = storage.node(node_id) else {
        return Vec::new();
    };

    let incoming = node.incoming_lanes();
    let outgoing = node.outgoing_lanes();

    let mut node_lanes = Vec::new();
    let lane_idx = storage.node_lane_count_for_node(node_id);

    for in_id in incoming {
        let in_lane = storage.lane(in_id);
        if !in_lane.is_enabled() {
            continue;
        }

        let in_pts = in_lane.polyline();
        if in_pts.len() < 2 {
            continue;
        }

        let in_segment = storage.segment(in_lane.segment());
        let in_node_idx = if in_segment.end() == node_id {
            in_pts.len() - 1
        } else {
            0
        };
        let in_pt = in_pts[in_node_idx];

        let Some(in_dir) = compute_lane_dir_at_node(in_pts, in_node_idx, true, chunk_size) else {
            continue;
        };

        for out_id in outgoing {
            if in_id == out_id {
                continue;
            }

            let out_lane = storage.lane(out_id);
            if !out_lane.is_enabled() || in_lane.segment() == out_lane.segment() {
                continue;
            }

            let out_pts = out_lane.polyline();
            if out_pts.len() < 2 {
                continue;
            }

            let out_segment = storage.segment(out_lane.segment());
            let out_node_idx = if out_segment.end() == node_id {
                out_pts.len() - 1
            } else {
                0
            };
            let out_pt = out_pts[out_node_idx];

            let Some(out_dir) = compute_lane_dir_at_node(out_pts, out_node_idx, false, chunk_size)
            else {
                continue;
            };

            if in_dir.dot(out_dir) < -0.8 {
                continue;
            }

            let chord = in_pt.distance_to(out_pt, chunk_size);
            let dot = in_dir.dot(out_dir);
            let tightness = compute_turn_tightness(chord, dot, params);

            let geom = generate_turn_geometry(
                in_pt,
                in_dir,
                out_pt,
                out_dir,
                params.turn_samples,
                tightness,
                chunk_size,
            );

            let nl = NodeLane::new(
                (lane_idx + node_lanes.len()) as NodeLaneId,
                vec![LaneRef::Segment(*in_id, in_node_idx as PolyIdx)],
                vec![LaneRef::Segment(*out_id, out_node_idx as PolyIdx)],
                geom,
                0.0,
                0.0,
                50.0,
                0,
            );

            node_lanes.push(nl);
        }
    }

    node_lanes
}

fn compute_lane_dir_at_node(
    pts: &[WorldPos],
    node_idx: usize,
    inward: bool,
    chunk_size: ChunkSize,
) -> Option<Vec3> {
    if pts.len() < 2 {
        return None;
    }

    let delta = if node_idx == 0 {
        if inward {
            pts[1].delta_to(pts[0], chunk_size)
        } else {
            pts[0].delta_to(pts[1], chunk_size)
        }
    } else {
        let n = pts.len();
        if inward {
            pts[n - 2].delta_to(pts[n - 1], chunk_size)
        } else {
            pts[n - 1].delta_to(pts[n - 2], chunk_size)
        }
    };

    let dir = Vec3::new(delta.x, 0.0, delta.z).normalize_or_zero();
    if dir == Vec3::ZERO { None } else { Some(dir) }
}

fn compute_turn_tightness(chord: f32, dot: f32, params: &IntersectionBuildParams) -> f32 {
    let base = params.turn_tightness;

    let chord_factor = if chord > 25.0 {
        1.3
    } else if chord < 8.0 {
        0.6
    } else {
        1.0
    };

    let angle_factor = if dot < 0.0 {
        0.7
    } else if dot > 0.8 {
        1.2
    } else {
        1.0
    };

    base * chord_factor * angle_factor
}

pub fn generate_turn_geometry(
    start: WorldPos,
    start_dir: Vec3,
    end: WorldPos,
    end_dir: Vec3,
    samples: usize,
    tightness: f32,
    chunk_size: ChunkSize,
) -> LaneGeometry {
    let dist = start.distance_to(end, chunk_size);

    if dist < 0.001 {
        return LaneGeometry::from_polyline(vec![start, end], chunk_size);
    }

    let control_len = dist * 0.4 * tightness;
    let ctrl1 = start.add_vec3(start_dir * control_len, chunk_size);
    let ctrl2 = end.add_vec3(-end_dir * control_len, chunk_size);

    let n = samples.clamp(4, 64);
    let points: Vec<WorldPos> = (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32;
            bezier3(start, ctrl1, ctrl2, end, t, chunk_size)
        })
        .collect();

    LaneGeometry::from_polyline(points, chunk_size)
}

fn bezier3(
    p0: WorldPos,
    p1: WorldPos,
    p2: WorldPos,
    p3: WorldPos,
    t: f32,
    chunk_size: ChunkSize,
) -> WorldPos {
    let t2 = t * t;
    let t3 = t2 * t;
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    let mt3 = mt2 * mt;

    let v1 = p1.to_render_pos(p0, chunk_size);
    let v2 = p2.to_render_pos(p0, chunk_size);
    let v3 = p3.to_render_pos(p0, chunk_size);

    let result = v1 * (3.0 * mt2 * t) + v2 * (3.0 * mt * t2) + v3 * t3;
    p0.add_vec3(result, chunk_size)
}

// ============================================================================
// Polyline Modification
// ============================================================================

pub fn modify_polyline_start(
    points: &[WorldPos],
    amount: f32,
    chunk_size: ChunkSize,
) -> Option<Vec<WorldPos>> {
    if points.len() < 2 || amount <= 0.0 {
        return Some(points.to_vec());
    }

    let lengths = polyline_cumulative_lengths(points, chunk_size);
    let total = *lengths.last().unwrap();

    if amount >= total - 0.01 {
        return None;
    }

    let (new_start, _) = sample_polyline_at(points, &lengths, amount, chunk_size);

    let mut keep_from = 1;
    while keep_from < lengths.len() && lengths[keep_from] <= amount {
        keep_from += 1;
    }

    let mut out = Vec::with_capacity(points.len() - keep_from + 2);
    out.push(new_start);
    out.extend_from_slice(&points[keep_from..]);

    if out.len() >= 2 { Some(out) } else { None }
}

pub fn modify_polyline_end(
    points: &[WorldPos],
    amount: f32,
    chunk_size: ChunkSize,
) -> Option<Vec<WorldPos>> {
    if points.len() < 2 || amount <= 0.0 {
        return Some(points.to_vec());
    }

    let lengths = polyline_cumulative_lengths(points, chunk_size);
    let total = *lengths.last().unwrap();

    if amount >= total - 0.01 {
        return None;
    }

    let target_len = total - amount;
    let (new_end, _) = sample_polyline_at(points, &lengths, target_len, chunk_size);

    let mut keep_to = lengths.len() - 1;
    while keep_to > 0 && lengths[keep_to] >= target_len {
        keep_to -= 1;
    }

    let mut out = Vec::with_capacity(keep_to + 2);
    out.extend_from_slice(&points[..=keep_to]);
    out.push(new_end);

    if out.len() >= 2 { Some(out) } else { None }
}

fn endpoint_index_near_node(
    pts: &[WorldPos],
    node_pos: WorldPos,
    chunk: ChunkSize,
) -> Option<usize> {
    if pts.len() < 2 {
        return None;
    }
    let d0 = pts[0].distance_to(node_pos, chunk);
    let d1 = pts[pts.len() - 1].distance_to(node_pos, chunk);
    Some(if d0 <= d1 { 0 } else { pts.len() - 1 })
}

fn lane_dir_away_from_node(pts: &[WorldPos], node_pos: WorldPos, chunk: ChunkSize) -> Option<Vec3> {
    let node_idx = endpoint_index_near_node(pts, node_pos, chunk)?;
    let d = if node_idx == 0 {
        pts[0].delta_to(pts[1], chunk) // away from node
    } else {
        let n = pts.len();
        pts[n - 1].delta_to(pts[n - 2], chunk) // away from node
    };
    let v = Vec3::new(d.x, 0.0, d.z).normalize_or_zero();
    (v != Vec3::ZERO).then_some(v)
}

fn compute_arm_direction(
    storage: &RoadStorage,
    lane_ids: &[LaneId],
    node_pos: WorldPos,
    chunk_size: ChunkSize,
) -> Option<Vec3> {
    let mut sum = Vec3::ZERO;
    let mut count = 0;

    for lane_id in lane_ids {
        let pts = storage.lane(lane_id).polyline();
        if let Some(dir) = lane_dir_away_from_node(pts, node_pos, chunk_size) {
            sum += dir;
            count += 1;
        }
    }

    if count == 0 {
        return None;
    }
    let avg = (sum / count as f32).normalize_or_zero();
    (avg != Vec3::ZERO).then_some(avg)
}
