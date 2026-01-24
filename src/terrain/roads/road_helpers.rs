use crate::positions::{ChunkCoord, ChunkSize, LocalPos, WorldPos};
use crate::renderer::gizmo::Gizmo;
use crate::renderer::world_renderer::TerrainRenderer;
use crate::terrain::roads::intersections::{IntersectionPolygon, OuterNodeLane};
use crate::terrain::roads::road_editor::{
    IntersectionBuildParams, polyline_cumulative_lengths, sample_polyline_at,
};
use crate::terrain::roads::road_mesh_manager::{CLEARANCE, ChunkId};
use crate::terrain::roads::road_structs::{NodeId, RoadStyleParams, SegmentId, StructureType};
use crate::terrain::roads::roads::{
    LaneRef, NodeLane, RoadCommand, RoadStorage, project_point_to_segment_xz,
};
use glam::{Vec2, Vec3, Vec3Swizzles};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::f32::consts::PI;

/// Offset an entire polyline by a fixed distance.
/// Positive offset = right side of travel direction.
/// Negative offset = left side of travel direction.
pub fn offset_polyline_f32(poly: &[WorldPos], offset: f32, chunk_size: ChunkSize) -> Vec<WorldPos> {
    if poly.len() < 2 {
        return poly.to_vec();
    }

    poly.iter()
        .enumerate()
        .map(|(i, &pt)| {
            let dir = polyline_direction_at(poly, i, chunk_size);
            let normal = dir.cross(Vec3::Y).normalize_or_zero();
            pt.add_vec3(normal * offset, chunk_size)
        })
        .collect()
}

pub fn select_outermost_lanes(
    node_lanes: &[NodeLane],
    storage: &RoadStorage,
    chunk_size: ChunkSize,
) -> Vec<OuterNodeLane> {
    let mut by_src: HashMap<SegmentId, Vec<(&NodeLane, i8, i8, SegmentId)>> = HashMap::new();

    for nl in node_lanes {
        let Some(&LaneRef::Segment(src_id, _)) = nl.merging().first() else {
            continue;
        };
        let Some(&LaneRef::Segment(dst_id, _)) = nl.splitting().first() else {
            continue;
        };

        let src = storage.lane(&src_id);
        let dst = storage.lane(&dst_id);

        if src.segment() == dst.segment() {
            continue;
        }

        by_src.entry(src.segment()).or_default().push((
            nl,
            src.lane_index().abs(),
            dst.lane_index().abs(),
            src.segment(), // Track segment ID
        ));
    }

    let mut result = Vec::new();

    for (_seg, entries) in by_src {
        if entries.is_empty() {
            continue;
        }

        let max_src = entries.iter().map(|&(_, s, _, _)| s).max().unwrap();
        let max_dst = entries.iter().map(|&(_, _, d, _)| d).max().unwrap();

        let winner = entries
            .into_iter()
            .filter(|&(_, s, d, _)| s == max_src && d == max_dst)
            .max_by(|(a, _, _, _), (b, _, _, _)| {
                let sa = right_turn_score(a.polyline(), chunk_size);
                let sb = right_turn_score(b.polyline(), chunk_size);
                sa.partial_cmp(&sb).unwrap()
            });

        if let Some((nl, _, _, seg_id)) = winner {
            result.push(OuterNodeLane {
                node_lane: nl.id(),
                outward_sign: 1,
                segment_id: seg_id,
            });
        }
    }

    result
}

/// Calculate a right turn score for a polyline.
/// Positive = right turn, negative = left turn.
/// Magnitude indicates turn sharpness.
pub fn right_turn_score(poly: &[WorldPos], chunk_size: ChunkSize) -> f32 {
    if poly.len() < 3 {
        return 0.0;
    }

    let dir_start = poly[1]
        .to_render_pos(poly[0], chunk_size)
        .normalize_or_zero();
    let dir_end = poly[poly.len() - 1]
        .to_render_pos(poly[poly.len() - 2], chunk_size)
        .normalize_or_zero();

    // Signed turn around Y axis
    let cross = dir_start.cross(dir_end);
    let dot = dir_start.dot(dir_end);

    // atan2 gives signed angle, negative = right turn
    let score = -cross.y.atan2(dot);
    if score.is_nan() { 0.0 } else { score }
}

/// Merges disjoint polylines into a single CCW ring sorted by angle around the center.
/// Ensures that individual segments also flow in the CCW direction.
pub fn merge_polylines_ccw(
    center: WorldPos,
    mut polylines: Vec<Vec<WorldPos>>,
    chunk_size: ChunkSize,
) -> Vec<WorldPos> {
    if polylines.is_empty() {
        return Vec::new();
    }

    // Helper to calculate angle in radians (-PI to PI) relative to center
    let get_angle = |p: WorldPos| -> f32 {
        let rel = p.to_render_pos(center, chunk_size);
        rel.z.atan2(rel.x)
    };

    // 1. Sort the chunks radially based on their endpoint
    polylines.sort_by(|a, b| {
        if a.is_empty() || b.is_empty() {
            return Ordering::Equal;
        }
        let pa = a.last().unwrap();
        let pb = b.last().unwrap();

        let angle_a = get_angle(*pa);
        let angle_b = get_angle(*pb);

        angle_a.partial_cmp(&angle_b).unwrap_or(Ordering::Equal)
    });

    let mut result_ring: Vec<WorldPos> = Vec::with_capacity(polylines.len() * 10);

    // 2. Process each polyline
    for poly in &mut polylines {
        if poly.len() < 2 {
            if !poly.is_empty() {
                result_ring.push(poly[0]);
            }
            continue;
        }

        let start = poly.first().unwrap();
        let end = poly.last().unwrap();

        let angle_start = get_angle(*start);
        let angle_end = get_angle(*end);

        // Calculate angular delta to determine direction
        let mut diff = angle_end - angle_start;

        // Normalize diff to -PI..PI to handle the wrap-around case
        while diff <= -PI {
            diff += 2.0 * PI;
        }
        while diff > PI {
            diff -= 2.0 * PI;
        }

        // If diff is negative, the polyline is running Clockwise.
        // We need to reverse it to flow CCW.
        if diff < 0.0 {
            poly.reverse();
        }

        // 3. Append to result
        // Skip duplicate vertices if ends touch exactly
        if let Some(&last_pt) = result_ring.last() {
            let first_new = &poly[0];
            let dist_sq = last_pt.distance_squared(*first_new, chunk_size);

            // If points are virtually identical, skip the first one
            if dist_sq < 0.01 {
                result_ring.extend_from_slice(&poly[1..]);
                continue;
            }
        }

        result_ring.extend_from_slice(poly);
    }

    result_ring
}

/// Creates a triangle fan connecting a central vertex (at `center_idx`)
/// to a ring of vertices starting at `ring_start`.
///
/// Layout in Vertex Buffer: [Center, Ring0, Ring1, Ring2, ... RingN]
/// `center_idx` = base
/// `ring_start` = base + 1
pub fn triangulate_center_fan(base_index: u32, ring_count: u32, indices: &mut Vec<u32>) {
    if ring_count < 2 {
        return;
    }

    let center = base_index;
    let ring_start = base_index + 1;

    for i in 0..ring_count {
        // Current point in ring
        let current = ring_start + i;

        // Next point in ring (wrap around to start)
        let next = ring_start + ((i + 1) % ring_count);

        // CCW Winding: Center -> Next -> Current
        // (Swap Next/Current if your engine uses CW culling, but usually standard is CCW)
        // Since our ring is sorted CCW, Center->Current->Next should be the correct order (Right-Hand Rule)
        indices.push(center);
        indices.push(next);
        indices.push(current);
    }
}
pub fn cross2(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

/// Returns t along a0->a1 where it intersects b0->b1 (if any). t in [0,1].
/// Handles proper intersections AND collinear overlap (returns the first overlap t).
pub fn seg_seg_intersection_t(a0: Vec2, a1: Vec2, b0: Vec2, b1: Vec2) -> Option<f32> {
    let r = a1 - a0;
    let s = b1 - b0;

    let denom = cross2(r, s);

    const EPS: f32 = 1e-6;
    const EPS_T: f32 = 1e-4;

    // Parallel (or collinear)
    if denom.abs() < EPS {
        // If not collinear, no intersection.
        let qp = b0 - a0;
        if cross2(qp, r).abs() >= EPS {
            return None;
        }

        // Collinear: project b endpoints onto a's param t and see if overlap exists.
        let rr = r.dot(r);
        if rr < EPS {
            return None; // a0==a1 degenerate
        }

        let t0 = (b0 - a0).dot(r) / rr;
        let t1 = (b1 - a0).dot(r) / rr;

        let lo = t0.min(t1).max(0.0);
        let hi = t0.max(t1).min(1.0);

        if lo <= hi {
            return Some(lo.clamp(0.0, 1.0));
        }
        return None;
    }

    let qp = b0 - a0;
    let t = cross2(qp, s) / denom;
    let u = cross2(qp, r) / denom;

    if t >= -EPS_T && t <= 1.0 + EPS_T && u >= -EPS_T && u <= 1.0 + EPS_T {
        Some(t.clamp(0.0, 1.0))
    } else {
        None
    }
}

/// Returns (distance_from_start, hit_point) for the *furthest* hit when
/// walking points[0]->points[1]->...
pub fn furthest_hit_distance_from_start_xz(points: &[Vec3], poly: &[Vec3]) -> Option<(f32, Vec3)> {
    if points.len() < 2 || poly.len() < 3 {
        return None;
    }

    let mut acc = 0.0f32;
    let mut best_dist: Option<f32> = None;
    let mut best_hit: Vec3 = Vec3::ZERO;

    for i in 0..points.len() - 1 {
        let a = points[i];
        let b = points[i + 1];
        let seg = b - a;
        let seg_len = seg.length();
        if seg_len < 1e-6 {
            continue;
        }

        let a0 = a.xz();
        let a1 = b.xz();

        // Find *latest* intersection on this segment (largest t) with ANY polygon edge
        let mut best_t_on_seg: Option<f32> = None;

        for j in 0..poly.len() {
            let p0 = poly[j].xz();
            let p1 = poly[(j + 1) % poly.len()].xz();

            if let Some(t) = seg_seg_intersection_t(a0, a1, p0, p1) {
                best_t_on_seg = Some(match best_t_on_seg {
                    None => t,
                    Some(bt) => bt.max(t), // furthest along this segment
                });
            }
        }

        if let Some(t) = best_t_on_seg {
            let hit = a + seg * t;
            let dist = acc + seg_len * t;

            match best_dist {
                None => {
                    best_dist = Some(dist);
                    best_hit = hit;
                }
                Some(cur) if dist > cur => {
                    best_dist = Some(dist);
                    best_hit = hit;
                }
                _ => {}
            }
        }

        acc += seg_len;
    }

    best_dist.map(|d| (d, best_hit))
}
/// 2D line segment intersection in XZ plane
/// Returns Some((t1, t2)) if segments intersect
pub fn line_segment_intersection_2d(
    ax1: f32,
    az1: f32,
    ax2: f32,
    az2: f32,
    bx1: f32,
    bz1: f32,
    bx2: f32,
    bz2: f32,
) -> Option<(f32, f32)> {
    let d1x = ax2 - ax1;
    let d1z = az2 - az1;
    let d2x = bx2 - bx1;
    let d2z = bz2 - bz1;

    let cross = d1x * d2z - d1z * d2x;
    if cross.abs() < 1e-10 {
        return None; // Parallel or coincident
    }

    let dx = bx1 - ax1;
    let dz = bz1 - az1;

    let t1 = (dx * d2z - dz * d2x) / cross;
    let t2 = (dx * d1z - dz * d1x) / cross;

    // Check if intersection is within both segments
    if t1 >= 0.0 && t1 <= 1.0 && t2 >= 0.0 && t2 <= 1.0 {
        Some((t1, t2))
    } else {
        None
    }
}

/// Subdivide a quadratic bezier to extract the section from t0 to t1.
/// Returns (new_p0, new_p1, new_p2) control points for the sub-curve.
pub fn subdivide_quadratic_bezier(
    p0: WorldPos,
    p1: WorldPos,
    p2: WorldPos,
    t0: f32,
    t1: f32,
    chunk_size: ChunkSize,
) -> (WorldPos, WorldPos, WorldPos) {
    /// Evaluate quadratic bezier at parameter t.
    fn eval_bezier(p0: WorldPos, p1: WorldPos, p2: WorldPos, t: f32, cs: ChunkSize) -> WorldPos {
        let v1 = p1.to_render_pos(p0, cs);
        let v2 = p2.to_render_pos(p0, cs);

        let omt = 1.0 - t;
        // B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
        let blend = v1 * (2.0 * omt * t) + v2 * (t * t);
        p0.add_vec3(blend, cs)
    }

    let new_p0 = eval_bezier(p0, p1, p2, t0, chunk_size);
    let new_p2 = eval_bezier(p0, p1, p2, t1, chunk_size);

    // Compute new control point to preserve curve shape
    // Derivative: B'(t) = 2(1-t)(P1-P0) + 2t(P2-P1)
    let dt = t1 - t0;
    let v01 = p1.to_render_pos(p0, chunk_size);
    let v12 = p2.to_render_pos(p1, chunk_size);
    let tangent_at_t0 = v01 * (1.0 - t0) + v12 * t0;
    let new_p1 = new_p0.add_vec3(tangent_at_t0 * dt, chunk_size);

    (new_p0, new_p1, new_p2)
}

/// Helper to push intersection for a node
pub fn push_intersection_for_node(
    cmds: &mut Vec<RoadCommand>,
    node_id: NodeId,
    road_style_params: &RoadStyleParams,
    chunk_id: ChunkId,
) {
    cmds.push(RoadCommand::MakeIntersection {
        node_id,
        params: IntersectionBuildParams::from_style(road_style_params),
        chunk_id,
        clear: true,
    });
}

pub fn clip_ribbon_edges_to_polygon(
    edges: &mut [(WorldPos, WorldPos)],
    poly: &IntersectionPolygon,
    at_start: bool,
    gizmo: &mut Gizmo,
) {
    let n = edges.len();
    if n < 2 || poly.ring.len() < 3 {
        return;
    }

    // idx_far = point deep in the lane (definitely outside polygon)
    // idx_near = point near intersection (the one we're clipping)
    let (idx_near, idx_far) = if at_start {
        (0, (n - 1).min(5)) // Use a point further back for stable direction
    } else {
        (n - 1, 0.max(n.saturating_sub(6)))
    };

    let (left_near, right_near) = edges[idx_near];
    let (left_far, right_far) = edges[idx_far];

    // Cast ray from FAR point through NEAR point, find first polygon hit
    let new_left = ray_to_polygon(left_far, left_near, poly, gizmo).unwrap_or(left_near);
    let new_right = ray_to_polygon(right_far, right_near, poly, gizmo).unwrap_or(right_near);

    edges[idx_near] = (new_left, new_right);
}

/// Cast ray from `from` through `through` and beyond, find first polygon intersection.
/// Ray extends far beyond 'through' to find distant intersections.
pub fn ray_to_polygon(
    from: WorldPos,
    through: WorldPos,
    poly: &IntersectionPolygon,
    gizmo: &mut Gizmo,
) -> Option<WorldPos> {
    let n = poly.ring.len();
    if n < 3 {
        return None;
    }

    // Direction from 'from' to 'through'
    let dir = through.to_render_pos(from, gizmo.chunk_size);
    let len_sq = dir.x * dir.x + dir.z * dir.z;

    if len_sq < 1e-6 {
        return None;
    }

    let len = len_sq.sqrt();
    let ray_dir = Vec3::new(dir.x / len, 0.0, dir.z / len);

    // Extend ray far beyond 'through' (2km)
    let max_dist = 2000.0;

    let mut best_t = f32::MAX;
    let mut best_hit: Option<WorldPos> = None;

    for i in 0..n {
        let a = poly.ring[i];
        let b = poly.ring[(i + 1) % n];

        // Edge vector relative to 'a'
        let edge = b.to_render_pos(a, gizmo.chunk_size);

        // Vector from 'from' to edge start
        let to_seg = a.to_render_pos(from, gizmo.chunk_size);

        let cross = ray_dir.x * edge.z - ray_dir.z * edge.x;
        if cross.abs() < 1e-10 {
            continue;
        }

        let t = (to_seg.x * edge.z - to_seg.z * edge.x) / cross;
        let u = (to_seg.x * ray_dir.z - to_seg.z * ray_dir.x) / cross;

        // t > 0: intersection is ahead of 'from'
        // u in [0,1]: intersection is on the polygon segment
        if t > 0.0 && t < max_dist && u >= 0.0 && u <= 1.0 && t < best_t {
            best_t = t;
            let offset = Vec3::new(ray_dir.x * t, through.local.y - from.local.y, ray_dir.z * t);
            best_hit = Some(from.add_vec3(offset, gizmo.chunk_size));
        }
    }

    if let Some(hit) = best_hit {
        gizmo.cross(hit, 10.0, [0.0, 0.0, 1.0], 50.0);
    }

    best_hit
}

/// Find intersection point of two segments (XZ plane).
pub fn segment_intersection_xz(
    a1: WorldPos,
    a2: WorldPos,
    b1: WorldPos,
    b2: WorldPos,
    chunk_size: ChunkSize,
) -> Option<(f32, f32)> {
    let d1 = a2.to_render_pos(a1, chunk_size);
    let d2 = b2.to_render_pos(b1, chunk_size);
    let d12 = b1.to_render_pos(a1, chunk_size);

    let cross = d1.x * d2.z - d1.z * d2.x;
    if cross.abs() < 1e-10 {
        return None;
    }

    let t = (d12.x * d2.z - d12.z * d2.x) / cross;
    let u = (d12.x * d1.z - d12.z * d1.x) / cross;

    if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
        Some((t, u))
    } else {
        None
    }
}

/// Find intersection between two line segments in XZ plane.
/// Returns (t, intersection_point) where t is the parameter along segment p1->p2.
pub fn segment_intersection_xz_with_t(
    p1: WorldPos,
    p2: WorldPos,
    p3: WorldPos,
    p4: WorldPos,
    chunk_size: ChunkSize,
) -> Option<(f32, WorldPos)> {
    // Compute everything relative to p1 for precision
    let d1 = p2.to_render_pos(p1, chunk_size);
    let d2 = p4.to_render_pos(p3, chunk_size);
    let d3 = p3.to_render_pos(p1, chunk_size);

    // Cross product in 2D (XZ plane)
    let cross = d1.x * d2.z - d1.z * d2.x;

    if cross.abs() < 1e-10 {
        return None; // Parallel or collinear
    }

    let t = (d3.x * d2.z - d3.z * d2.x) / cross;
    let u = (d3.x * d1.z - d3.z * d1.x) / cross;

    // Both parameters must be in [0, 1] for segments to intersect
    if t >= -1e-6 && t <= 1.0 + 1e-6 && u >= -1e-6 && u <= 1.0 + 1e-6 {
        let t_clamped = t.clamp(0.0, 1.0);

        // Interpolate Y as average of the two segments at intersection
        let y1 = p1.local.y + (p2.local.y - p1.local.y) * t_clamped;
        let y2 = p3.local.y + (p4.local.y - p3.local.y) * u.clamp(0.0, 1.0);
        let avg_y = (y1 + y2) * 0.5;

        // Compute intersection point
        let offset = Vec3::new(d1.x * t_clamped, avg_y - p1.local.y, d1.z * t_clamped);
        let point = p1.add_vec3(offset, chunk_size);

        Some((t_clamped, point))
    } else {
        None
    }
}
/// Find the intersection point of a ray with a polygon (closest to 'from').
pub fn ray_polygon_intersection(
    from: WorldPos,
    to: WorldPos,
    poly: &IntersectionPolygon,
    chunk_size: ChunkSize,
) -> Option<WorldPos> {
    let n = poly.ring.len();
    if n < 3 {
        return None;
    }

    let mut best_t: Option<f32> = None;

    // Ray direction relative to 'from'
    let d1 = to.to_render_pos(from, chunk_size);
    let d1_xz = glam::Vec2::new(d1.x, d1.z);

    for i in 0..n {
        let a = poly.ring[i];
        let b = poly.ring[(i + 1) % n];

        // Edge vector
        let edge = b.to_render_pos(a, chunk_size);
        let d2 = glam::Vec2::new(edge.x, edge.z);

        let denom = d1_xz.x * d2.y - d1_xz.y * d2.x;

        if denom.abs() < 1e-10 {
            continue;
        }

        // Vector from 'from' to edge start
        let d3 = a.to_render_pos(from, chunk_size);
        let d3_xz = glam::Vec2::new(d3.x, d3.z);

        let t = (d3_xz.x * d2.y - d3_xz.y * d2.x) / denom;
        let u = (d3_xz.x * d1_xz.y - d3_xz.y * d1_xz.x) / denom;

        // t must be positive (in direction of ray) and u must be on the edge segment
        if t > 1e-6 && t <= 1.0 && u >= 0.0 && u <= 1.0 {
            best_t = Some(match best_t {
                Some(bt) => bt.min(t),
                None => t,
            });
        }
    }

    best_t.map(|t| from.lerp(to, t, chunk_size))
}

pub fn set_point_height_with_structure_type(
    terrain_renderer: &TerrainRenderer,
    structure_type: StructureType,
    p: &mut WorldPos,
) {
    match structure_type {
        StructureType::Surface => {
            p.local.y = terrain_renderer.get_height_at(*p) + CLEARANCE;
        }
        StructureType::Bridge => {
            p.local.y = p.local.y.max(terrain_renderer.get_height_at(*p)) + CLEARANCE;
        }
        StructureType::Tunnel => {
            p.local.y = p.local.y + CLEARANCE;
        }
    }
}
/// More precise segment-chunk intersection test.
fn segment_intersects_chunk_precise(
    start: WorldPos,
    end: WorldPos,
    chunk: ChunkCoord,
    chunk_size: ChunkSize,
) -> bool {
    let cs = chunk_size as f32;

    // Chunk bounds as WorldPos
    let chunk_min = WorldPos::new(chunk, LocalPos::new(0.0, 0.0, 0.0));
    let chunk_max = WorldPos::new(chunk, LocalPos::new(cs, 0.0, cs));

    // Convert segment to chunk-local coordinates
    let a = start.to_render_pos(chunk_min, chunk_size);
    let b = end.to_render_pos(chunk_min, chunk_size);

    // 2D line-box intersection in XZ plane
    line_intersects_box_2d(
        Vec2::new(a.x, a.z),
        Vec2::new(b.x, b.z),
        Vec2::ZERO,
        Vec2::new(cs, cs),
    )
}

/// 2D line-box intersection test.
fn line_intersects_box_2d(a: Vec2, b: Vec2, box_min: Vec2, box_max: Vec2) -> bool {
    let d = b - a;
    let mut t_min = 0.0f32;
    let mut t_max = 1.0f32;

    for i in 0..2 {
        let (a_i, d_i, min_i, max_i) = match i {
            0 => (a.x, d.x, box_min.x, box_max.x),
            _ => (a.y, d.y, box_min.y, box_max.y),
        };

        if d_i.abs() < 1e-10 {
            if a_i < min_i || a_i > max_i {
                return false;
            }
        } else {
            let inv_d = 1.0 / d_i;
            let mut t1 = (min_i - a_i) * inv_d;
            let mut t2 = (max_i - a_i) * inv_d;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            t_min = t_min.max(t1);
            t_max = t_max.min(t2);
            if t_min > t_max {
                return false;
            }
        }
    }
    true
}

/// Resample a polyline to have approximately equal segment lengths.
pub fn resample_polyline(
    poly: &[WorldPos],
    target_segment_length: f32,
    chunk_size: ChunkSize,
) -> Vec<WorldPos> {
    if poly.len() < 2 {
        return poly.to_vec();
    }

    let lengths = polyline_cumulative_lengths(poly, chunk_size);
    let total_len = *lengths.last().unwrap();

    if total_len < target_segment_length {
        return poly.to_vec();
    }

    let num_segments = (total_len / target_segment_length).ceil() as usize;
    let actual_segment_len = total_len / num_segments as f32;

    let mut result = Vec::with_capacity(num_segments + 1);

    for i in 0..=num_segments {
        let t = i as f32 * actual_segment_len;
        let (pos, _) = sample_polyline_at(poly, &lengths, t, chunk_size);
        result.push(pos);
    }

    result
}

/// Smooth a polyline using Chaikin's algorithm.
pub fn smooth_polyline_chaikin(
    poly: &[WorldPos],
    iterations: usize,
    chunk_size: ChunkSize,
) -> Vec<WorldPos> {
    if poly.len() < 3 || iterations == 0 {
        return poly.to_vec();
    }

    let mut current = poly.to_vec();

    for _ in 0..iterations {
        let n = current.len();
        let mut next = Vec::with_capacity(n * 2);

        // Keep first point
        next.push(current[0]);

        for i in 0..n - 1 {
            let a = current[i];
            let b = current[i + 1];

            // Q = 0.75*A + 0.25*B
            let q = a.lerp(b, 0.25, chunk_size);
            // R = 0.25*A + 0.75*B
            let r = a.lerp(b, 0.75, chunk_size);

            next.push(q);
            next.push(r);
        }

        // Keep last point
        next.push(current[n - 1]);

        current = next;
    }

    current
}

/// Simplify a polyline using Douglas-Peucker algorithm.
pub fn simplify_polyline_dp(
    poly: &[WorldPos],
    tolerance: f32,
    chunk_size: ChunkSize,
) -> Vec<WorldPos> {
    if poly.len() < 3 {
        return poly.to_vec();
    }

    fn dp_recursive(
        poly: &[WorldPos],
        start: usize,
        end: usize,
        tolerance_sq: f32,
        keep: &mut Vec<bool>,
        chunk_size: ChunkSize,
    ) {
        if end <= start + 1 {
            return;
        }

        let mut max_dist_sq = 0.0f32;
        let mut max_idx = start;

        let line_start = poly[start];
        let line_end = poly[end];

        for i in start + 1..end {
            let (_, dist_sq) =
                project_point_to_segment_xz(poly[i], line_start, line_end, chunk_size);
            if dist_sq > max_dist_sq {
                max_dist_sq = dist_sq;
                max_idx = i;
            }
        }

        if max_dist_sq > tolerance_sq {
            keep[max_idx] = true;
            dp_recursive(poly, start, max_idx, tolerance_sq, keep, chunk_size);
            dp_recursive(poly, max_idx, end, tolerance_sq, keep, chunk_size);
        }
    }

    let n = poly.len();
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;

    dp_recursive(poly, 0, n - 1, tolerance * tolerance, &mut keep, chunk_size);

    poly.iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, &p)| p)
        .collect()
}

/// Reverse a polyline.
pub fn reverse_polyline(poly: &[WorldPos]) -> Vec<WorldPos> {
    poly.iter().copied().rev().collect()
}

/// Concatenate two polylines, removing duplicate junction point if present.
pub fn concat_polylines(a: &[WorldPos], b: &[WorldPos], chunk_size: ChunkSize) -> Vec<WorldPos> {
    if a.is_empty() {
        return b.to_vec();
    }
    if b.is_empty() {
        return a.to_vec();
    }

    let mut result = a.to_vec();

    // Check if last point of a is same as first point of b
    let dist = a
        .last()
        .unwrap()
        .distance_to(*b.first().unwrap(), chunk_size);

    if dist < 0.01 {
        // Skip first point of b (duplicate)
        result.extend_from_slice(&b[1..]);
    } else {
        result.extend_from_slice(b);
    }

    result
}

/// Compute robust direction with configurable minimum distance.
pub fn compute_robust_direction_with_min_dist(
    edge_line: &[WorldPos],
    at_start: bool,
    min_dist: f32,
    chunk_size: ChunkSize,
) -> Vec3 {
    let n = edge_line.len();
    if n < 2 {
        return Vec3::ZERO;
    }

    let target_idx = if at_start { 0 } else { n - 1 };
    let target = edge_line[target_idx];

    let min_dist_sq = min_dist * min_dist;

    // Try to find a point at least min_dist meters away for a reliable direction
    let indices: Box<dyn Iterator<Item = usize>> = if at_start {
        Box::new(1..n)
    } else {
        Box::new((0..n - 1).rev())
    };

    for i in indices {
        let delta = edge_line[i].to_render_pos(target, chunk_size);
        let len_sq = delta.x * delta.x + delta.z * delta.z; // XZ only

        if len_sq > min_dist_sq {
            let sign = if at_start { 1.0 } else { -1.0 };
            return Vec3::new(delta.x * sign, 0.0, delta.z * sign).normalize();
        }
    }

    // Fallback: use adjacent point
    let adj_idx = if at_start {
        1.min(n - 1)
    } else {
        (n - 1).saturating_sub(1)
    };

    let delta = edge_line[adj_idx].to_render_pos(target, chunk_size);
    let sign = if at_start { 1.0 } else { -1.0 };
    Vec3::new(delta.x * sign, 0.0, delta.z * sign).normalize_or_zero()
}

/// Get direction at a specific index along a polyline.
pub fn polyline_direction_at(poly: &[WorldPos], index: usize, chunk_size: ChunkSize) -> Vec3 {
    if poly.len() < 2 {
        return Vec3::ZERO;
    }

    let dir = if index + 1 < poly.len() {
        poly[index].delta_to(poly[index + 1], chunk_size)
    } else {
        poly[index - 1].delta_to(poly[index], chunk_size)
    };

    dir.normalize_or_zero()
}

/// Get tangent direction at a polyline point (average of incoming/outgoing).
pub fn polyline_tangent_at(poly: &[WorldPos], index: usize, chunk_size: ChunkSize) -> Vec3 {
    if poly.len() < 2 {
        return Vec3::ZERO;
    }

    let n = poly.len();

    if index == 0 {
        return poly[1]
            .to_render_pos(poly[0], chunk_size)
            .normalize_or_zero();
    }
    if index >= n - 1 {
        return poly[n - 1]
            .to_render_pos(poly[n - 2], chunk_size)
            .normalize_or_zero();
    }

    let d_prev = poly[index]
        .to_render_pos(poly[index - 1], chunk_size)
        .normalize_or_zero();
    let d_next = poly[index + 1]
        .to_render_pos(poly[index], chunk_size)
        .normalize_or_zero();

    ((d_prev + d_next) * 0.5).normalize_or_zero()
}
