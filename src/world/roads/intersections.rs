// intersections.rs - Fixed Clipper2 API usage

use crate::helpers::positions::{ChunkCoord, ChunkSize, LocalPos, WorldPos};
use crate::renderer::gizmo::{DEBUG_DRAW_DURATION, Gizmo};
use crate::world::roads::road_editor::{polyline_cumulative_lengths, sample_polyline_at};
use crate::world::roads::road_helpers::*;
use crate::world::roads::road_mesh_manager::*;
use crate::world::roads::road_structs::*;
use crate::world::roads::roads::*;
use crate::world::terrain::terrain_editing::TerrainEditor;
use crate::world::terrain::terrain_subsystem::TerrainSubsystem;
use clipper2::{EndType, JoinType, Path, Paths};
use earcutr;
use glam::{Vec2, Vec3, Vec3Swizzles};
use std::collections::HashSet;
use std::f32::consts::{FRAC_PI_2, TAU};

const DEFAULT_CORRIDOR_LENGTH: f32 = 6.0;
const MAX_CORRIDOR_LENGTH: f32 = 10000.0;
const MIN_ANGLE_FOR_DYNAMIC: f32 = 0.02;

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
    fn corridor(direction: Vec2, half_width: f32, fwd_len: f32) -> Self {
        let dir = direction.normalize_or_zero();
        if dir == Vec2::ZERO {
            return Self::new(Vec::new());
        }
        let perp = Vec2::new(-dir.y, dir.x);

        let a = -dir * 0.0;
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

fn normalize_angle_positive(angle: f32) -> f32 {
    let mut a = angle % TAU;
    if a < 0.0 {
        a += TAU;
    }
    a
}

fn compute_max_corridor_for_arm(arm: &Arm, storage: &RoadStorage, chunk_size: ChunkSize) -> f32 {
    const SAFETY_MARGIN: f32 = 0.5;

    let all_lanes: Vec<LaneId> = arm
        .incoming_lanes()
        .iter()
        .chain(arm.outgoing_lanes().iter())
        .copied()
        .collect();

    if all_lanes.is_empty() {
        return MAX_CORRIDOR_LENGTH;
    }

    let mut min_available = MAX_CORRIDOR_LENGTH;

    for lane_id in all_lanes {
        let lane = storage.lane(&lane_id);
        let pts = lane.polyline();

        if pts.len() < 3 {
            return SAFETY_MARGIN;
        }

        let mut total_len = 0.0f32;
        for i in 0..pts.len() - 1 {
            total_len += pts[i].distance_to(pts[i + 1], chunk_size);
        }

        let is_incoming = arm.incoming_lanes().contains(&lane_id);

        let preserve_len = if is_incoming {
            pts[0].distance_to(pts[1], chunk_size)
        } else {
            let n = pts.len();
            pts[n - 2].distance_to(pts[n - 1], chunk_size)
        };

        let available = (total_len - preserve_len - SAFETY_MARGIN).max(0.0);
        min_available = min_available.min(available);
    }

    min_available
}

fn compute_corridor_lengths(
    arms: &[Arm],
    sidewalk_width: f32,
    storage: &RoadStorage,
    chunk_size: ChunkSize,
) -> Vec<f32> {
    let n = arms.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        let max_from_lanes = compute_max_corridor_for_arm(&arms[0], storage, chunk_size);
        return vec![DEFAULT_CORRIDOR_LENGTH.min(max_from_lanes)];
    }

    arms.iter()
        .enumerate()
        .map(|(i, arm)| {
            let prev_idx = (i + n - 1) % n;
            let next_idx = (i + 1) % n;

            let angle_to_prev = normalize_angle_positive(arm.bearing() - arms[prev_idx].bearing());
            let angle_to_next = normalize_angle_positive(arms[next_idx].bearing() - arm.bearing());

            let mut corridor_len = DEFAULT_CORRIDOR_LENGTH;

            update_corridor_len(
                &mut corridor_len,
                angle_to_prev,
                arm,
                &arms[prev_idx],
                sidewalk_width,
            );

            update_corridor_len(
                &mut corridor_len,
                angle_to_next,
                arm,
                &arms[next_idx],
                sidewalk_width,
            );

            let max_from_lanes = compute_max_corridor_for_arm(arm, storage, chunk_size);
            corridor_len.min(max_from_lanes)
        })
        .collect()
}

fn update_corridor_len(
    corridor_len: &mut f32,
    angle: f32,
    arm: &Arm,
    other_arm: &Arm,
    sidewalk_width: f32,
) {
    if angle > MIN_ANGLE_FOR_DYNAMIC && angle < FRAC_PI_2 {
        let combined_width = arm.half_width() + other_arm.half_width() - sidewalk_width * 1.5;
        let len = combined_width / (angle * 0.5).tan().max(0.01);
        *corridor_len = corridor_len.max(len.min(MAX_CORRIDOR_LENGTH));
    }
}

fn create_corridor_end_clips(
    arms: &[Arm],
    corridor_lengths: &[f32],
    sidewalk_width: f32,
) -> Vec<LocalPolygon> {
    arms.iter()
        .zip(corridor_lengths.iter())
        .map(|(arm, &corridor_len)| {
            let dir = Vec2::new(arm.direction().x, arm.direction().z).normalize_or_zero();
            let perp = Vec2::new(-dir.y, dir.x);

            let half_width = arm.half_width() + sidewalk_width + 2.0;
            let clip_start = corridor_len - 0.05;
            let clip_end = corridor_len + sidewalk_width + 10.0;

            LocalPolygon::new(vec![
                dir * clip_start - perp * half_width,
                dir * clip_end - perp * half_width,
                dir * clip_end + perp * half_width,
                dir * clip_start + perp * half_width,
            ])
        })
        .collect()
}

// Clipper2 Stuff

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

fn difference_polygons(subject: &LocalPolygon, clips: &[LocalPolygon]) -> Vec<LocalPolygon> {
    use clipper2::{FillRule, difference};

    if subject.points.len() < 3 {
        return vec![];
    }

    let valid_clips: Vec<_> = clips.iter().filter(|c| c.points.len() >= 3).collect();
    if valid_clips.is_empty() {
        return vec![subject.clone()];
    }

    let subject_coords: Vec<(f64, f64)> = subject
        .points
        .iter()
        .map(|p| (p.x as f64, p.y as f64))
        .collect();

    let clip_coords: Vec<Vec<(f64, f64)>> = valid_clips
        .iter()
        .map(|c| c.points.iter().map(|p| (p.x as f64, p.y as f64)).collect())
        .collect();

    let subject_paths: Paths = vec![subject_coords].into();
    let clip_paths: Paths = clip_coords.into();

    match difference(subject_paths, clip_paths, FillRule::NonZero) {
        Ok(result) => {
            let output: Vec<Vec<(f64, f64)>> = result.into();
            output
                .into_iter()
                .filter(|path| path.len() >= 3)
                .map(LocalPolygon::from_clipper_coords)
                .collect()
        }
        Err(_) => vec![subject.clone()],
    }
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
    pub turn_samples: usize,
    pub turn_tightness: f32,
    pub round_corners: bool,
    pub simplify_tolerance: f32,
    pub road_type: RoadType,
}

impl Default for IntersectionBuildParams {
    fn default() -> Self {
        Self {
            turn_samples: 12,
            turn_tightness: 1.0,
            round_corners: true,
            simplify_tolerance: 0.1,
            road_type: Default::default(),
        }
    }
}

impl IntersectionBuildParams {
    pub fn from_style(style: &RoadStyleParams) -> Self {
        let road_type = style.road_type();
        Self {
            road_type: road_type.clone(),
            turn_tightness: style.turn_tightness(),
            ..Default::default()
        }
    }
}

// ============================================================================
// Main Intersection Building
// ============================================================================

pub fn build_intersection_at_node(
    terrain: &mut TerrainSubsystem,
    storage: &mut RoadStorage,
    node_id: NodeId,
    params: &IntersectionBuildParams,
    recalc_clearance: bool,
    gizmo: &mut Gizmo,
) -> HashSet<ChunkCoord> {
    let chunk_size = terrain.chunk_size;
    let mut affected_chunks = HashSet::new();

    if recalc_clearance {
        if let Some(geom) =
            compute_intersection_geometry(storage, node_id, params, chunk_size, gizmo)
        {
            debug_draw_polygon(&geom.polygon, [0.0, 0.8, 0.2], gizmo);
            if let Some(ref sw) = geom.sidewalk_polygon {
                debug_draw_polygon(sw, [0.5, 0.5, 0.5], gizmo);
            }

            carve_lanes_with_polygon(storage, node_id, &geom, chunk_size, gizmo);

            let flattening_polygon = if let Some(ref sw_poly) = geom.sidewalk_polygon {
                &sw_poly.ring
            } else {
                &geom.polygon.ring
            };

            if flattening_polygon.len() >= 3 {
                let center_height = compute_intersection_center_height(
                    terrain,
                    &geom.center,
                    flattening_polygon,
                    chunk_size,
                );

                let road_type = params.road_type;

                affected_chunks = terrain
                    .terrain_editor
                    .apply_intersection_polygon_flattening(
                        node_id,
                        flattening_polygon,
                        center_height,
                        -road_type.lane_height - 0.05,
                        params.road_type.sidewalk_width + 1.0,
                        chunk_size,
                        &terrain.chunks,
                    );
            }
        }
    }

    storage.node_mut(node_id).clear_node_lanes();
    let node_lanes =
        build_node_lanes_for_intersection(terrain, storage, node_id, params, chunk_size, gizmo);
    storage.node_mut(node_id).add_node_lanes(node_lanes);

    affected_chunks
}

pub fn remove_intersection_at_node(
    terrain_editor: &mut TerrainEditor,
    node_id: NodeId,
) -> HashSet<ChunkCoord> {
    terrain_editor.remove_intersection_flattening(node_id)
}

fn compute_intersection_center_height(
    terrain: &TerrainSubsystem,
    center: &WorldPos,
    polygon: &[WorldPos],
    chunk_size: ChunkSize,
) -> f32 {
    if polygon.is_empty() {
        return terrain.get_height_at(*center, true);
    }

    let mut sum = terrain.get_height_at(*center, true);
    let mut count = 1.0f32;

    let sample_count = polygon.len().min(8);
    let step = polygon.len() / sample_count;

    for i in (0..polygon.len()).step_by(step.max(1)) {
        sum += terrain.get_height_at(polygon[i], true);
        count += 1.0;
    }

    sum / count
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

fn create_corner_fillers(
    arms: &[Arm],
    half_widths: &[f32],
    _round_corners: bool,
) -> Vec<LocalPolygon> {
    use std::f32::consts::PI;
    let n = arms.len();

    if n < 2 || half_widths.len() != n {
        return vec![];
    }

    let mut fillers = Vec::new();

    for i in 0..n {
        let next_idx = (i + 1) % n;

        let angle_between = normalize_angle_positive(arms[next_idx].bearing() - arms[i].bearing());

        if angle_between > PI {
            let arm_i = &arms[i];
            let arm_next = &arms[next_idx];

            let dir_i = arm_i.direction().xz().normalize_or_zero();
            let dir_next = arm_next.direction().xz().normalize_or_zero();

            let perp_i = Vec2::new(-dir_i.y, dir_i.x);
            let perp_next = Vec2::new(-dir_next.y, dir_next.x);

            let hw_i = half_widths[i];
            let hw_next = half_widths[next_idx];

            let edge_i = perp_i * hw_i;
            let edge_next = -perp_next * hw_next;

            fillers.push(LocalPolygon::new(vec![Vec2::ZERO, edge_i, edge_next]));
        }
    }

    fillers
}

fn round_polygon_corners(polygon: &LocalPolygon, radius: f32) -> LocalPolygon {
    if polygon.points.len() < 3 || radius <= 0.001 {
        return polygon.clone();
    }

    let coords: Vec<(f64, f64)> = polygon
        .points
        .iter()
        .map(|p| (p.x as f64, p.y as f64))
        .collect();

    let paths: Paths = vec![coords].into();

    let shrunk = paths.inflate(-radius as f64, JoinType::Miter, EndType::Polygon, 2.0);
    let expanded = shrunk.inflate(radius as f64, JoinType::Round, EndType::Polygon, 2.0);

    let output: Vec<Vec<(f64, f64)>> = expanded.into();

    output
        .into_iter()
        .filter(|path| path.len() >= 3)
        .map(LocalPolygon::from_clipper_coords)
        .max_by(|a, b| {
            polygon_area(&a.points)
                .partial_cmp(&polygon_area(&b.points))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or_else(|| polygon.clone())
}
fn is_edge_at_corridor_entrance(
    edge_start: Vec2,
    edge_end: Vec2,
    arms: &[Arm],
    corridor_lengths: &[f32],
    tolerance: f32,
) -> bool {
    let edge_mid = (edge_start + edge_end) * 0.5;
    let edge_vec = edge_end - edge_start;
    let edge_len = edge_vec.length();

    if edge_len < 0.001 {
        return false;
    }

    let edge_dir = edge_vec / edge_len;

    for (arm, &corridor_len) in arms.iter().zip(corridor_lengths.iter()) {
        let arm_dir = arm.direction().xz().normalize_or_zero();

        let proj = edge_mid.dot(arm_dir);

        if (proj - corridor_len).abs() < tolerance {
            let dot = edge_dir.dot(arm_dir).abs();
            if dot < 0.5 {
                return true;
            }
        }
    }

    false
}

fn create_rounded_wedge(center: Vec2, start: Vec2, end: Vec2, _radius: f32) -> LocalPolygon {
    use std::f32::consts::{PI, TAU};

    let start_offset = start - center;
    let end_offset = end - center;

    let start_angle = start_offset.y.atan2(start_offset.x);
    let end_angle = end_offset.y.atan2(end_offset.x);

    let mut sweep = end_angle - start_angle;
    while sweep > PI {
        sweep -= TAU;
    }
    while sweep < -PI {
        sweep += TAU;
    }

    let start_radius = start_offset.length();
    let end_radius = end_offset.length();

    let avg_radius = (start_radius + end_radius) / 2.0;
    let arc_len = sweep.abs() * avg_radius;
    let steps = ((arc_len / 0.5) as usize).clamp(4, 48);

    let mut points = vec![center];
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let angle = start_angle + sweep * t;
        let radius = start_radius + (end_radius - start_radius) * t;
        points.push(center + Vec2::new(angle.cos() * radius, angle.sin() * radius));
    }

    LocalPolygon::new(points)
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

    for tri in tri_indices.chunks_exact(3) {
        let i0 = base_index + tri[0] as u32;
        let i1 = base_index + tri[1] as u32;
        let i2 = base_index + tri[2] as u32;

        // Flip winding
        indices.extend_from_slice(&[i0, i2, i1]);
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

    for tri in tri_indices.chunks_exact(3) {
        let i0 = base_index + tri[0] as u32;
        let i1 = base_index + tri[1] as u32;
        let i2 = base_index + tri[2] as u32;

        indices.extend_from_slice(&[i0, i2, i1]);
    }

    true
}

// Intersection Mesh Builder

fn build_polygon_curbs_filtered(
    terrain: &TerrainSubsystem,
    center: WorldPos,
    ring: &[Vec2],
    arms: &[Arm],
    corridor_lengths: &[f32],
    road_type: &RoadType,
    bottom_height: f32,
    top_height: f32,
    face_inward: bool,
    config: &MeshConfig,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    let chunk_size = terrain.chunk_size;
    let height_diff = top_height - bottom_height;

    if height_diff.abs() < 0.001 || ring.len() < 3 {
        return;
    }

    let n = ring.len();

    for i in 0..n {
        let next = (i + 1) % n;
        let a = ring[i];
        let b = ring[next];

        if is_edge_at_corridor_entrance(a, b, arms, corridor_lengths, 0.5) {
            continue;
        }

        let edge = b - a;
        let edge_len = edge.length();
        if edge_len < 0.001 {
            continue;
        }

        let edge_dir = edge / edge_len;
        let right_perp = Vec2::new(edge_dir.y, -edge_dir.x);
        let normal_2d = if face_inward { -right_perp } else { right_perp };
        let normal = Vec3::new(normal_2d.x, 0.0, normal_2d.y);

        let mut pos_a = center.add_vec3(Vec3::new(a.x, 0.0, a.y), chunk_size);
        let mut pos_b = center.add_vec3(Vec3::new(b.x, 0.0, b.y), chunk_size);

        set_point_height_with_structure_type(terrain, road_type.structure(), &mut pos_a, true);
        set_point_height_with_structure_type(terrain, road_type.structure(), &mut pos_b, true);

        let base = vertices.len() as u32;

        let mut p0 = pos_a;
        p0.local.y += bottom_height;
        let mut p1 = pos_b;
        p1.local.y += bottom_height;
        let mut p2 = pos_b;
        p2.local.y += top_height;
        let mut p3 = pos_a;
        p3.local.y += top_height;

        let u0 = 0.0;
        let u1 = edge_len * config.uv_scale_u;
        let v0 = 0.0;
        let v1 = height_diff.abs() * config.uv_scale_v;

        vertices.push(road_vertex(
            p0,
            normal.to_array(),
            road_type.sidewalk_material_id,
            u0,
            v0,
        ));
        vertices.push(road_vertex(
            p1,
            normal.to_array(),
            road_type.sidewalk_material_id,
            u1,
            v0,
        ));
        vertices.push(road_vertex(
            p2,
            normal.to_array(),
            road_type.sidewalk_material_id,
            u1,
            v1,
        ));
        vertices.push(road_vertex(
            p3,
            normal.to_array(),
            road_type.sidewalk_material_id,
            u0,
            v1,
        ));

        if face_inward {
            indices.push(base);
            indices.push(base + 1);
            indices.push(base + 2);
            indices.push(base);
            indices.push(base + 2);
            indices.push(base + 3);
        } else {
            indices.push(base);
            indices.push(base + 3);
            indices.push(base + 2);
            indices.push(base);
            indices.push(base + 2);
            indices.push(base + 1);
        }
    }
}

struct AdaptiveVertex {
    pos_2d: Vec2,
    height: f32,
}

fn adaptive_triangulate_polygon(
    terrain: &TerrainSubsystem,
    center: WorldPos,
    polygon_2d: &[Vec2],
    base_height_offset: f32,
    chunk_size: ChunkSize,
    structure_type: StructureType,
) -> (Vec<AdaptiveVertex>, Vec<u32>) {
    let mut verts: Vec<AdaptiveVertex> = polygon_2d
        .iter()
        .map(|p| {
            let mut world_pos = center.add_vec3(Vec3::new(p.x, 0.0, p.y), chunk_size);
            set_point_height_with_structure_type(terrain, structure_type, &mut world_pos, true);
            AdaptiveVertex {
                pos_2d: *p,
                height: world_pos.local.y + base_height_offset,
            }
        })
        .collect();

    let mut initial_indices = Vec::new();
    triangulate_polygon(polygon_2d, 0, &mut initial_indices);

    let mut final_indices = Vec::new();
    let mut tri_stack: Vec<(u32, u32, u32, u32)> = Vec::new();

    for chunk in initial_indices.chunks(3) {
        if chunk.len() == 3 {
            tri_stack.push((chunk[0], chunk[1], chunk[2], 0));
        }
    }

    const MAX_DEPTH: u32 = 3;
    const TERRAIN_MARGIN: f32 = 0.02;

    while let Some((i0, i1, i2, depth)) = tri_stack.pop() {
        if depth >= MAX_DEPTH {
            final_indices.extend_from_slice(&[i0, i1, i2]);
            continue;
        }

        let v0 = &verts[i0 as usize];
        let v1 = &verts[i1 as usize];
        let v2 = &verts[i2 as usize];

        let centroid_2d = Vec2::new(
            (v0.pos_2d.x + v1.pos_2d.x + v2.pos_2d.x) / 3.0,
            (v0.pos_2d.y + v1.pos_2d.y + v2.pos_2d.y) / 3.0,
        );

        let mesh_height_at_centroid = (v0.height + v1.height + v2.height) / 3.0;

        let mut centroid_world =
            center.add_vec3(Vec3::new(centroid_2d.x, 0.0, centroid_2d.y), chunk_size);
        set_point_height_with_structure_type(terrain, structure_type, &mut centroid_world, true);
        let terrain_height_at_centroid = centroid_world.local.y + base_height_offset;

        if terrain_height_at_centroid > mesh_height_at_centroid + TERRAIN_MARGIN {
            let new_idx = verts.len() as u32;
            verts.push(AdaptiveVertex {
                pos_2d: centroid_2d,
                height: terrain_height_at_centroid,
            });

            tri_stack.push((i0, i1, new_idx, depth + 1));
            tri_stack.push((i1, i2, new_idx, depth + 1));
            tri_stack.push((i2, i0, new_idx, depth + 1));
        } else {
            final_indices.extend_from_slice(&[i0, i1, i2]);
        }
    }

    (verts, final_indices)
}

fn emit_adaptive_mesh(
    center: WorldPos,
    adaptive_verts: &[AdaptiveVertex],
    adaptive_indices: &[u32],
    chunk_size: ChunkSize,
    config: &MeshConfig,
    material_id: u32,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    let base = vertices.len() as u32;

    for av in adaptive_verts {
        let pos = WorldPos {
            chunk: center.chunk,
            local: LocalPos::new(
                center.local.x + av.pos_2d.x,
                av.height,
                center.local.z + av.pos_2d.y,
            ),
        }
        .normalize(chunk_size);

        let (u, v) = radial_uv(center, pos, chunk_size, config);
        vertices.push(road_vertex(pos, [0.0, 1.0, 0.0], material_id, u, v));
    }

    for &idx in adaptive_indices {
        indices.push(base + idx);
    }
}

pub fn build_intersection_mesh(
    terrain: &TerrainSubsystem,
    node_id: NodeId,
    node: &Node,
    storage: &RoadStorage,
    style: &RoadStyleParams,
    config: &MeshConfig,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
    gizmo: &mut Gizmo,
) -> IntersectionMeshResult {
    let center = node.position();
    let chunk_size = terrain.chunk_size;

    let road_type = style.road_type();
    let params = IntersectionBuildParams::from_style(style);

    let arms = node.arms();
    if arms.len() < 2 {
        return IntersectionMeshResult::default();
    }

    let corridor_lengths =
        compute_corridor_lengths(&arms, params.road_type.sidewalk_width, storage, chunk_size);
    let corner_rounding_radius = if params.round_corners { 0.5 } else { 0.0 };

    let mut asphalt_corridors: Vec<LocalPolygon> = arms
        .iter()
        .zip(corridor_lengths.iter())
        .map(|(arm, &len)| {
            let dir_2d = arm.direction().xz();
            let asphalt_half_width = (arm.half_width() - params.road_type.sidewalk_width).max(0.5);
            LocalPolygon::corridor(dir_2d, asphalt_half_width, len)
        })
        .collect();

    let asphalt_hw: Vec<f32> = arms
        .iter()
        .map(|a| (a.half_width() - params.road_type.sidewalk_width).max(0.5))
        .collect();
    let asphalt_corners = create_corner_fillers(&arms, &asphalt_hw, false);
    asphalt_corridors.extend(asphalt_corners);

    let asphalt_union = union_polygons(&asphalt_corridors);

    let Some(asphalt_poly) = asphalt_union.into_iter().max_by(|a, b| {
        polygon_area(&a.points)
            .partial_cmp(&polygon_area(&b.points))
            .unwrap_or(std::cmp::Ordering::Equal)
    }) else {
        return IntersectionMeshResult::default();
    };

    let asphalt_rounded = if params.round_corners {
        round_polygon_corners(&asphalt_poly, corner_rounding_radius)
    } else {
        asphalt_poly
    };

    let asphalt_simplified = simplify_clipper(&asphalt_rounded, params.simplify_tolerance);
    if asphalt_simplified.points.len() < 3 {
        return IntersectionMeshResult::default();
    }

    let asphalt_centroid: Vec2 = asphalt_simplified.points.iter().copied().sum::<Vec2>()
        / asphalt_simplified.points.len() as f32;

    let asphalt_base = vertices.len() as u32;

    for p2d in &asphalt_simplified.points {
        let mut pos = center.add_vec3(Vec3::new(p2d.x, 0.0, p2d.y), chunk_size);
        set_point_height_with_structure_type(terrain, road_type.structure(), &mut pos, true);
        pos.local.y += road_type.lane_height;
        let local = *p2d - asphalt_centroid;
        let u = (local.x * config.uv_scale_u) + 0.5;
        let v = (local.y * config.uv_scale_v) + 0.5;
        vertices.push(road_vertex(
            pos,
            [0.0, 1.0, 0.0],
            road_type.lane_material_id,
            u,
            v,
        ));
    }

    triangulate_polygon(&asphalt_simplified.points, asphalt_base, indices);

    if road_type.sidewalk_width > 0.01 {
        let mut full_corridors: Vec<LocalPolygon> = arms
            .iter()
            .zip(corridor_lengths.iter())
            .map(|(arm, &len)| {
                let dir_2d = arm.direction().xz();
                LocalPolygon::corridor(dir_2d, arm.half_width(), len)
            })
            .collect();

        let full_hw: Vec<f32> = arms.iter().map(|a| a.half_width()).collect();
        let full_corners = create_corner_fillers(&arms, &full_hw, false);
        full_corridors.extend(full_corners);

        let full_union = union_polygons(&full_corridors);

        let full_poly = match full_union.into_iter().max_by(|a, b| {
            polygon_area(&a.points)
                .partial_cmp(&polygon_area(&b.points))
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Some(p) => p,
            None => return IntersectionMeshResult::default(),
        };

        let full_rounded = if params.round_corners {
            round_polygon_corners(&full_poly, corner_rounding_radius)
        } else {
            full_poly
        };

        let mut asphalt_corridors_for_sw: Vec<LocalPolygon> = arms
            .iter()
            .zip(corridor_lengths.iter())
            .map(|(arm, &len)| {
                let dir_2d = arm.direction().xz();
                let asphalt_half_width =
                    (arm.half_width() - params.road_type.sidewalk_width).max(0.5);
                LocalPolygon::corridor(dir_2d, asphalt_half_width, len)
            })
            .collect();

        let asphalt_corners_for_sw = create_corner_fillers(&arms, &asphalt_hw, false);
        asphalt_corridors_for_sw.extend(asphalt_corners_for_sw);

        let asphalt_union_for_sw = union_polygons(&asphalt_corridors_for_sw);
        let asphalt_poly_for_sw = asphalt_union_for_sw.into_iter().max_by(|a, b| {
            polygon_area(&a.points)
                .partial_cmp(&polygon_area(&b.points))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let asphalt_rounded_for_sw = if let Some(ap) = asphalt_poly_for_sw {
            if params.round_corners {
                round_polygon_corners(&ap, corner_rounding_radius)
            } else {
                ap
            }
        } else {
            asphalt_simplified.clone()
        };

        let end_clips =
            create_corridor_end_clips(&arms, &corridor_lengths, params.road_type.sidewalk_width);

        let sidewalk_polys = compute_sidewalk_ring_polygons(
            &[full_rounded.clone()],
            &[asphalt_rounded_for_sw],
            &end_clips,
        );

        for sidewalk_poly in &sidewalk_polys {
            if sidewalk_poly.points.len() >= 3 {
                let simplified = simplify_clipper(sidewalk_poly, params.simplify_tolerance);
                if simplified.points.len() >= 3 {
                    build_sidewalk_mesh(
                        terrain,
                        center,
                        &simplified.points,
                        road_type,
                        config,
                        vertices,
                        indices,
                    );
                }
            }
        }

        let mut inner_ccw = asphalt_simplified.points.clone();
        if polygon_signed_area(&inner_ccw) < 0.0 {
            inner_ccw.reverse();
        }
        build_polygon_curbs_filtered(
            terrain,
            center,
            &inner_ccw,
            &arms,
            &corridor_lengths,
            road_type,
            road_type.lane_height,
            road_type.sidewalk_height,
            true,
            config,
            vertices,
            indices,
        );

        let full_simplified = simplify_clipper(&full_rounded, params.simplify_tolerance);
        if full_simplified.points.len() >= 3 {
            let mut outer_ccw = full_simplified.points.clone();
            if polygon_signed_area(&outer_ccw) < 0.0 {
                outer_ccw.reverse();
            }
            build_polygon_curbs_filtered(
                terrain,
                center,
                &outer_ccw,
                &arms,
                &corridor_lengths,
                road_type,
                road_type.lane_height,
                road_type.sidewalk_height,
                false,
                config,
                vertices,
                indices,
            );
        }
    }

    let ring: Vec<WorldPos> = asphalt_simplified
        .points
        .iter()
        .map(|v| center.add_vec3(Vec3::new(v.x, 0.0, v.y), chunk_size))
        .collect();

    IntersectionMeshResult {
        polygon: IntersectionPolygon::new(ring, center, chunk_size),
    }
}

fn compute_intersection_geometry(
    storage: &RoadStorage,
    node_id: NodeId,
    params: &IntersectionBuildParams,
    chunk_size: ChunkSize,
    gizmo: &mut Gizmo,
) -> Option<IntersectionGeometry> {
    let node = storage.node(node_id)?;
    let center = node.position();

    let arms = node.arms();

    if arms.len() < 2 {
        return None;
    }

    let corridor_lengths =
        compute_corridor_lengths(&arms, params.road_type.sidewalk_width, storage, chunk_size);
    let corner_rounding_radius = if params.round_corners { 0.5 } else { 0.0 };

    let mut corridors: Vec<LocalPolygon> = arms
        .iter()
        .zip(corridor_lengths.iter())
        .map(|(arm, &len)| {
            let dir_2d = Vec2::new(arm.direction().x, arm.direction().z);
            LocalPolygon::corridor(dir_2d, arm.half_width(), len)
        })
        .collect();

    let full_hw: Vec<f32> = arms.iter().map(|a| a.half_width()).collect();
    let corner_fillers = create_corner_fillers(&arms, &full_hw, false);
    corridors.extend(corner_fillers);

    let unioned = union_polygons(&corridors);
    if unioned.is_empty() {
        return None;
    }

    let main_poly = unioned.into_iter().max_by(|a, b| {
        polygon_area(&a.points)
            .partial_cmp(&polygon_area(&b.points))
            .unwrap_or(std::cmp::Ordering::Equal)
    })?;

    let rounded = if params.round_corners {
        round_polygon_corners(&main_poly, corner_rounding_radius)
    } else {
        main_poly
    };

    let simplified = simplify_clipper(&rounded, params.simplify_tolerance);

    let ring: Vec<WorldPos> = simplified
        .points
        .iter()
        .map(|v| center.add_vec3(Vec3::new(v.x, 0.0, v.y), chunk_size))
        .collect();

    let polygon = IntersectionPolygon::new(ring, center, chunk_size);

    let sidewalk_polygon = if params.road_type.sidewalk_width > 0.01 {
        let mut asphalt_corridors: Vec<LocalPolygon> = arms
            .iter()
            .zip(corridor_lengths.iter())
            .map(|(arm, &len)| {
                let dir_2d = Vec2::new(arm.direction().x, arm.direction().z);
                let asphalt_half_width =
                    (arm.half_width() - params.road_type.sidewalk_width).max(0.5);
                LocalPolygon::corridor(dir_2d, asphalt_half_width, len)
            })
            .collect();

        let asphalt_hw: Vec<f32> = arms
            .iter()
            .map(|a| (a.half_width() - params.road_type.sidewalk_width).max(0.5))
            .collect();
        let asphalt_corners = create_corner_fillers(&arms, &asphalt_hw, false);
        asphalt_corridors.extend(asphalt_corners);

        let asphalt_union = union_polygons(&asphalt_corridors);
        asphalt_union
            .into_iter()
            .max_by(|a, b| {
                polygon_area(&a.points)
                    .partial_cmp(&polygon_area(&b.points))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| {
                let p = if params.round_corners {
                    round_polygon_corners(&p, corner_rounding_radius)
                } else {
                    p
                };
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
        arms: arms.to_vec(),
        polygon,
        sidewalk_polygon,
    })
}

fn compute_sidewalk_ring_polygons(
    full_corridors: &[LocalPolygon],
    asphalt_corridors: &[LocalPolygon],
    end_clips: &[LocalPolygon],
) -> Vec<LocalPolygon> {
    use clipper2::{FillRule, difference, union};

    let full_coords: Vec<Vec<(f64, f64)>> = full_corridors
        .iter()
        .filter(|p| p.points.len() >= 3)
        .map(|p| {
            p.points
                .iter()
                .map(|pt| (pt.x as f64, pt.y as f64))
                .collect()
        })
        .collect();

    if full_coords.is_empty() {
        return vec![];
    }

    let full_paths: Paths = full_coords.into();
    let empty: Paths = Vec::<Vec<(f64, f64)>>::new().into();

    let full_union = match union(full_paths, empty, FillRule::NonZero) {
        Ok(r) => r,
        Err(_) => return vec![],
    };

    let asphalt_coords: Vec<Vec<(f64, f64)>> = asphalt_corridors
        .iter()
        .filter(|p| p.points.len() >= 3)
        .map(|p| {
            p.points
                .iter()
                .map(|pt| (pt.x as f64, pt.y as f64))
                .collect()
        })
        .collect();

    if asphalt_coords.is_empty() {
        return vec![];
    }

    let asphalt_paths: Paths = asphalt_coords.into();

    let ring_result = match difference(full_union, asphalt_paths, FillRule::NonZero) {
        Ok(r) => r,
        Err(_) => return vec![],
    };

    let clip_coords: Vec<Vec<(f64, f64)>> = end_clips
        .iter()
        .filter(|p| p.points.len() >= 3)
        .map(|p| {
            p.points
                .iter()
                .map(|pt| (pt.x as f64, pt.y as f64))
                .collect()
        })
        .collect();

    let final_result = if clip_coords.is_empty() {
        ring_result
    } else {
        let clip_paths: Paths = clip_coords.into();
        match difference(ring_result, clip_paths, FillRule::NonZero) {
            Ok(r) => r,
            Err(_) => return vec![],
        }
    };

    let output: Vec<Vec<(f64, f64)>> = final_result.into();

    output
        .into_iter()
        .filter(|path| path.len() >= 3)
        .map(LocalPolygon::from_clipper_coords)
        .collect()
}

fn build_sidewalk_mesh(
    terrain: &TerrainSubsystem,
    center: WorldPos,
    ring: &[Vec2],
    road_type: &RoadType,
    config: &MeshConfig,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    let chunk_size = terrain.chunk_size;

    if ring.len() < 3 {
        return;
    }

    let centroid: Vec2 = ring.iter().copied().sum::<Vec2>() / ring.len() as f32;

    let base = vertices.len() as u32;

    for p2d in ring {
        let mut pos = center.add_vec3(Vec3::new(p2d.x, 0.0, p2d.y), chunk_size);
        set_point_height_with_structure_type(terrain, road_type.structure(), &mut pos, true);
        pos.local.y += road_type.sidewalk_height;

        let local = *p2d - centroid;
        let u = (local.x * config.uv_scale_u) + 0.5;
        let v = (local.y * config.uv_scale_v) + 0.5;
        vertices.push(road_vertex(
            pos,
            [0.0, 1.0, 0.0],
            road_type.sidewalk_material_id,
            u,
            v,
        ));
    }

    triangulate_polygon(ring, base, indices);
}

pub(crate) fn gather_arms(
    storage: &RoadStorage,
    node_id: NodeId,
    intersection_build_params: &IntersectionBuildParams,
    chunk_size: ChunkSize,
    gizmo: &mut Gizmo,
) -> Vec<Arm> {
    let segment_ids = storage.enabled_segments_connected_to_node(node_id);
    let node = match storage.node(node_id) {
        Some(n) => n,
        None => return Vec::new(),
    };

    let lane_width = intersection_build_params.road_type.lane_width;
    let sidewalk_width = intersection_build_params.road_type.sidewalk_width;

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

            let direction =
                segment_direction_at_node(segment, node_id, storage, chunk_size, gizmo)?;

            let mut bearing = direction.z.atan2(direction.x);
            if bearing < 0.0 {
                bearing += TAU;
            }

            let lane_count = lane_ids.len();
            let half_width = lane_count as f32 * lane_width * 0.5 + sidewalk_width;

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

fn gather_arms_from_node_lanes(
    node_id: NodeId,
    node_lanes: &[NodeLane],
    storage: &RoadStorage,
    chunk_size: ChunkSize,
    params: &IntersectionBuildParams,
    gizmo: &mut Gizmo,
) -> Vec<Arm> {
    let mut seen_segments: HashSet<SegmentId> = HashSet::new();

    for nl in node_lanes {
        for lane_ref in nl.merging() {
            if let LaneRef::Segment(lane_id, _) = lane_ref {
                let lane = storage.lane(lane_id);
                seen_segments.insert(lane.segment());
            }
        }
        for lane_ref in nl.merging() {
            if let LaneRef::Segment(lane_id, _) = lane_ref {
                let lane = storage.lane(lane_id);
                seen_segments.insert(lane.segment());
            }
        }
    }

    let mut arms: Vec<Arm> = seen_segments
        .into_iter()
        .filter_map(|seg_id| {
            let segment = storage.segment(seg_id);
            let direction =
                segment_direction_at_node(segment, node_id, storage, chunk_size, gizmo)?;

            let mut bearing = direction.z.atan2(direction.x);
            if bearing < 0.0 {
                bearing += TAU;
            }

            let lane_count = segment.lanes().len();
            let half_width = lane_count as f32 * params.road_type.lane_width * 0.5
                + params.road_type.sidewalk_width;

            let points_to_node = segment.end() == node_id;

            Some(Arm::new(
                seg_id,
                bearing,
                direction,
                half_width,
                points_to_node,
            ))
        })
        .collect();

    arms.sort_by(|a, b| {
        a.bearing()
            .partial_cmp(&b.bearing())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    arms
}

/// Compute bridge triangles for reflex angles (> 180°) between adjacent arms.
/// When the angle between consecutive arms exceeds 180°, the corridor rectangles
/// don't naturally meet at the outer corner, creating a gap. This function creates
/// triangles to fill those gaps.
fn compute_bridge_triangles(
    arms: &[Arm],
    corridor_lengths: &[f32],
    half_widths: &[f32],
) -> Vec<LocalPolygon> {
    use std::f32::consts::PI;

    let n = arms.len();
    if n < 2 {
        return vec![];
    }

    let mut bridges = Vec::new();

    for i in 0..n {
        let next_idx = (i + 1) % n;

        // Angle from this arm to next (going CCW in bearing order)
        let angle = normalize_angle_positive(arms[next_idx].bearing() - arms[i].bearing());

        // If angle > PI (180°), there's a gap at the outer corner
        if angle > PI {
            let dir = arms[i].direction().xz();
            let next_dir = arms[next_idx].direction().xz();
            let perp = Vec2::new(-dir.y, dir.x);
            let next_perp = Vec2::new(-next_dir.y, next_dir.x);

            // Arm i's far-right corner (facing the gap)
            let corner_i = dir * corridor_lengths[i] - perp * half_widths[i];

            // Arm i+1's far-left corner (facing the gap)
            let corner_next =
                next_dir * corridor_lengths[next_idx] + next_perp * half_widths[next_idx];

            // CCW winding for valid polygon: origin -> corner_next -> corner_i
            bridges.push(LocalPolygon::new(vec![Vec2::ZERO, corner_next, corner_i]));
        }
    }

    bridges
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
    terrain: &TerrainSubsystem,
    storage: &RoadStorage,
    node_id: NodeId,
    params: &IntersectionBuildParams,
    chunk_size: ChunkSize,
    gizmo: &mut Gizmo,
) -> Vec<NodeLane> {
    let Some(node) = storage.node(node_id) else {
        return Vec::new();
    };

    let node_pos = node.position();

    // Gather ALL lanes connected to this node and classify them geometrically
    let connected_segments = storage.enabled_segments_connected_to_node(node_id);

    let mut incoming_lanes: Vec<(LaneId, usize, WorldPos, Vec3)> = Vec::new(); // (id, node_idx, endpoint, direction_into_node)
    let mut outgoing_lanes: Vec<(LaneId, usize, WorldPos, Vec3)> = Vec::new(); // (id, node_idx, endpoint, direction_out_of_node)

    for seg_id in connected_segments {
        let segment = storage.segment(seg_id);
        let segment_starts_here = segment.start() == node_id;
        let segment_ends_here = segment.end() == node_id;

        for lane_id in segment.lanes() {
            let lane = storage.lane(lane_id);
            if !lane.is_enabled() {
                continue;
            }

            let pts = lane.polyline();
            if pts.len() < 2 {
                continue;
            }

            // Find which endpoint is actually at the node (geometrically)
            let Some(node_idx) = endpoint_index_near_node(pts, node_pos, chunk_size) else {
                continue;
            };

            let endpoint = pts[node_idx];

            // Determine if this lane is incoming or outgoing based on:
            // 1. Which end of the segment is at this node
            // 2. The lane_index convention (positive = toward segment end, negative = toward segment start)
            //
            // Actually, let's be more robust - determine from geometry + segment topology:
            let lane_idx = lane.lane_index();

            // In typical road setups:
            // - Positive lane indices flow in segment direction (start → end)
            // - Negative lane indices flow against segment direction (end → start)
            let flows_toward_end = lane_idx >= 0;

            let is_incoming = if segment_ends_here {
                // Segment ends at this node
                // Incoming if traffic flows toward segment end
                flows_toward_end
            } else if segment_starts_here {
                // Segment starts at this node
                // Incoming if traffic flows toward segment start (against segment direction)
                !flows_toward_end
            } else {
                continue; // Segment doesn't connect to this node??
            };

            // Compute direction based on actual geometry
            let (direction, endpoint) =
                compute_lane_tangent_at_node(pts, node_idx, is_incoming, chunk_size);

            let Some(direction) = direction else {
                continue;
            };

            if is_incoming {
                // Debug: green for incoming
                gizmo.cross(endpoint, 0.3, [0.0, 1.0, 0.0], DEBUG_DRAW_DURATION);
                gizmo.direction(endpoint, direction, [0.0, 0.8, 0.0], DEBUG_DRAW_DURATION);
                incoming_lanes.push((*lane_id, node_idx, endpoint, direction));
            } else {
                // Debug: red for outgoing
                gizmo.cross(endpoint, 0.3, [1.0, 0.0, 0.0], DEBUG_DRAW_DURATION);
                gizmo.direction(endpoint, direction, [0.8, 0.0, 0.0], DEBUG_DRAW_DURATION);
                outgoing_lanes.push((*lane_id, node_idx, endpoint, direction));
            }
        }
    }

    // Debug: print lane counts
    if incoming_lanes.is_empty() || outgoing_lanes.is_empty() {
        gizmo.cross(node_pos, 1.0, [1.0, 0.0, 1.0], DEBUG_DRAW_DURATION); // Magenta = missing in/out
        return Vec::new();
    }

    let mut node_lanes = Vec::new();
    let lane_idx_base = storage.node_lane_count_for_node(node_id);

    for (in_id, in_node_idx, in_pt, in_dir) in &incoming_lanes {
        let in_lane = storage.lane(in_id);

        for (out_id, out_node_idx, out_pt, out_dir) in &outgoing_lanes {
            // Skip same lane
            if in_id == out_id {
                continue;
            }

            // Skip same segment (no U-turns within same road)
            let out_lane = storage.lane(out_id);
            if in_lane.segment() == out_lane.segment() {
                continue;
            }

            // Angle-based filtering
            // in_dir points INTO the intersection (direction of incoming traffic)
            // out_dir points OUT OF the intersection (direction of outgoing traffic)
            //
            // For a straight-through: in_dir ≈ out_dir → dot ≈ 1
            // For 90° turn: dot ≈ 0
            // For U-turn (180°): dot ≈ -1
            let dot = in_dir.dot(*out_dir);

            // Filter out U-turns and very sharp turns
            if dot < -0.7 {
                continue;
            }

            // Compute turn geometry
            let chord = in_pt.distance_to(*out_pt, chunk_size);
            let tightness = compute_turn_tightness(chord, dot, params);

            let geom = generate_turn_geometry(
                terrain,
                *in_pt,
                *in_dir,
                *out_pt,
                *out_dir,
                params.turn_samples,
                tightness,
                chunk_size,
            );

            // Debug: draw the turn curve
            let turn_pts = &geom.points;
            for i in 0..turn_pts.len().saturating_sub(1) {
                gizmo.line(
                    turn_pts[i],
                    turn_pts[i + 1],
                    [1.0, 1.0, 0.0],
                    DEBUG_DRAW_DURATION,
                );
            }

            let nl = NodeLane::new(
                (lane_idx_base + node_lanes.len()) as NodeLaneId,
                vec![LaneRef::Segment(*in_id, *in_node_idx as PolyIdx)],
                vec![LaneRef::Segment(*out_id, *out_node_idx as PolyIdx)],
                geom,
                0.0,
                0.0,
                50.0,
                0,
            );

            node_lanes.push(nl);
        }
    }

    // Debug: orange if no node lanes generated
    if node_lanes.is_empty() {
        gizmo.cross(node_pos, 1.5, [1.0, 0.5, 0.0], DEBUG_DRAW_DURATION);
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
    terrain: &TerrainSubsystem,
    start: WorldPos,
    start_dir: Vec3, // Direction INTO intersection (incoming traffic direction)
    end: WorldPos,
    end_dir: Vec3, // Direction OUT OF intersection (outgoing traffic direction)
    samples: usize,
    tightness: f32,
    chunk_size: ChunkSize,
) -> LaneGeometry {
    let dist = start.distance_to(end, chunk_size);

    if dist < 0.001 {
        return LaneGeometry::from_polyline(vec![start, end], chunk_size);
    }

    // Control point distance scales with chord length and tightness
    // Longer control arms = smoother curves
    let control_len = dist * 0.4 * tightness;

    // ctrl1: Extend from start in the direction traffic is going (INTO intersection)
    //        This is start_dir itself
    let ctrl1 = start.add_vec3(start_dir * control_len, chunk_size);

    // ctrl2: Extend from end BACKWARD against traffic flow
    //        Traffic goes OUT in end_dir direction, so backward is -end_dir
    let ctrl2 = end.add_vec3(-end_dir * control_len, chunk_size);

    let n = samples.clamp(4, 64);
    let points: Vec<WorldPos> = (0..=n) // Note: 0..=n for n+1 points
        .map(|i| {
            let t = i as f32 / n as f32;
            let mut p = WorldPos::cubic_bezier_xz(start, ctrl1, ctrl2, end, t, chunk_size);
            p.local.y = terrain.get_height_at(p, true);
            p
        })
        .collect();

    LaneGeometry::from_polyline(points, chunk_size)
}

// Polyline Modification

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
fn segment_direction_at_node(
    segment: &Segment,
    node_id: NodeId,
    storage: &RoadStorage,
    chunk_size: ChunkSize,
    gizmo: &mut Gizmo,
) -> Option<Vec3> {
    let lane_id = *segment.lanes().first()?;
    let pts = storage.lane(&lane_id).polyline();

    if pts.len() < 2 {
        return None;
    }

    let (a, b) = if segment.start() == node_id {
        (pts[0], pts[1])
    } else {
        let n = pts.len();
        (pts[n - 1], pts[n - 2])
    };
    // gizmo.cross(a, 0.5, [1.0, 0.0, 0.0], DEBUG_DRAW_DURATION);
    // gizmo.cross(b, 0.6, [0.0, 1.0, 0.0], DEBUG_DRAW_DURATION);
    let d = a.delta_to(b, chunk_size);

    let v = Vec3::new(d.x, 0.0, d.z).normalize_or_zero();

    // gizmo.direction(a, v, [1.0, 1.0, 0.0], DEBUG_DRAW_DURATION);

    (v != Vec3::ZERO).then_some(v)
}

/// Compute the tangent direction of a lane at the node endpoint.
///
/// For incoming lanes: returns direction pointing INTO the intersection (traffic flow direction)
/// For outgoing lanes: returns direction pointing OUT OF the intersection (traffic flow direction)
fn compute_lane_tangent_at_node(
    pts: &[WorldPos],
    node_idx: usize,
    is_incoming: bool,
    chunk_size: ChunkSize,
) -> (Option<Vec3>, WorldPos) {
    if pts.len() < 2 {
        return (None, pts.get(0).copied().unwrap_or(WorldPos::zero()));
    }

    let endpoint = pts[node_idx];

    // For incoming: we want direction of traffic as it arrives at the node
    //   = direction from previous point toward the node endpoint
    // For outgoing: we want direction of traffic as it leaves the node
    //   = direction from node endpoint toward next point

    let direction = if node_idx == 0 {
        if is_incoming {
            // Traffic arrives at index 0 from index 1
            // Direction = from pts[1] toward pts[0]
            pts[1].delta_to(pts[0], chunk_size)
        } else {
            // Traffic leaves from index 0 toward index 1
            // Direction = from pts[0] toward pts[1]
            pts[0].delta_to(pts[1], chunk_size)
        }
    } else {
        let n = pts.len();
        if is_incoming {
            // Traffic arrives at index n-1 from index n-2
            // Direction = from pts[n-2] toward pts[n-1]
            pts[n - 2].delta_to(pts[n - 1], chunk_size)
        } else {
            // Traffic leaves from index n-1 toward index n-2
            // Direction = from pts[n-1] toward pts[n-2]
            pts[n - 1].delta_to(pts[n - 2], chunk_size)
        }
    };

    let dir = Vec3::new(direction.x, 0.0, direction.z).normalize_or_zero();

    if dir == Vec3::ZERO {
        (None, endpoint)
    } else {
        (Some(dir), endpoint)
    }
}
