use crate::renderer::gizmo::Gizmo;
use crate::terrain::roads::intersections::{IntersectionPolygon, OuterNodeLane};
use crate::terrain::roads::road_editor::IntersectionBuildParams;
use crate::terrain::roads::road_mesh_manager::ChunkId;
use crate::terrain::roads::road_structs::{NodeId, RoadStyleParams, SegmentId};
use crate::terrain::roads::roads::{LaneRef, NodeLane, RoadCommand, RoadStorage};
use glam::{Vec2, Vec3, Vec3Swizzles};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::f32::consts::PI;

/// Offset an entire polyline by a fixed distance
/// Positive offset = right side of travel direction
/// Negative offset = left side of travel direction
pub fn offset_polyline_f32(poly: &[Vec3], offset: f32) -> Vec<Vec3> {
    if poly.len() < 2 {
        return poly.to_vec();
    }

    let mut result = Vec::with_capacity(poly.len());

    for i in 0..poly.len() {
        let dir = if i + 1 < poly.len() {
            (poly[i + 1] - poly[i]).normalize()
        } else {
            (poly[i] - poly[i - 1]).normalize()
        };

        let normal = dir.cross(Vec3::Y).normalize();
        result.push(poly[i] + normal * offset);
    }

    result
}

pub fn select_outermost_lanes(
    node_lanes: &[NodeLane],
    storage: &RoadStorage,
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
                let sa = right_turn_score(&a.polyline());
                let sb = right_turn_score(&b.polyline());
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

pub fn right_turn_score(poly: &[Vec3]) -> f32 {
    if poly.len() < 3 {
        return 0.0;
    }

    let dir_start = (poly[1] - poly[0]).normalize();
    let dir_end = (poly[poly.len() - 1] - poly[poly.len() - 2]).normalize();

    // signed turn around Y axis
    let cross = dir_start.cross(dir_end);
    let dot = dir_start.dot(dir_end);

    // atan2 gives signed angle, negative = right turn
    let score = -cross.y.atan2(dot);
    if score.is_nan() { 0.0 } else { score }
}
pub fn right_normal(dir: Vec3) -> Vec3 {
    Vec3::new(dir.z, 0.0, -dir.x)
}

pub fn left_normal(dir: Vec3) -> Vec3 {
    Vec3::new(-dir.z, 0.0, dir.x)
}

/// Merges disjoint polylines into a single CCW ring sorted by angle around the center.
/// Ensures that individual segments also flow in the CCW direction.
pub fn merge_polylines_ccw(center: Vec3, mut polylines: Vec<Vec<Vec3>>) -> Vec<Vec3> {
    if polylines.is_empty() {
        return Vec::new();
    }

    // Helper to calculate angle in radians (-PI to PI)
    let get_angle = |p: Vec3| -> f32 { (p.z - center.z).atan2(p.x - center.x) };

    // 1. Sort the chunks radially based on their midpoint
    // This organizes the unconnected strips into a circle.
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

    let mut result_ring: Vec<Vec3> = Vec::with_capacity(polylines.len() * 10);

    // 2. Process each polyline
    for poly in &mut polylines {
        if poly.len() < 2 {
            continue;
        }

        let start = poly.first().unwrap();
        let end = poly.last().unwrap();

        let angle_start = get_angle(*start);
        let angle_end = get_angle(*end);

        // Calculate angular delta to determine direction
        let mut diff = angle_end - angle_start;

        // Normalize diff to -PI..PI to handle the wrap-around case (e.g. 179 deg to -179 deg)
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
        // Optional: Simple threshold to avoid duplicate vertices if ends touch exactly
        if let Some(last_pt) = result_ring.last() {
            let first_new = &poly[0];
            let dist_sq = (last_pt.x - first_new.x).powi(2) + (last_pt.z - first_new.z).powi(2);

            // If points are virtually identical, skip the first one to avoid zero-area triangles
            if dist_sq < 0.01 {
                result_ring.extend_from_slice(&poly[1..]);
                continue;
            }
        }

        result_ring.extend_from_slice(poly);
    }

    // 4. Close the loop (Optional depending on your mesh builder)
    // If your triangulation function expects the first point repeated at the end:
    /*
    if let Some(first) = result_ring.first().cloned() {
        // Only push if the last point isn't already the first point
        let last = result_ring.last().unwrap();
        let dist_sq = (last.x - first.x).powi(2) + (last.y - first.y).powi(2);
        if dist_sq > 0.01 {
            result_ring.push(first);
        }
    }
    */

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
pub fn trim_polyline_both_ends(points: &[Vec3], cut: usize) -> Vec<Vec3> {
    let len = points.len();

    // Need at least 2 points to be meaningful
    if len <= 2 {
        return points.to_vec();
    }

    // Maximum we are allowed to cut per side
    let max_cut = (len - 2) / 2;
    let cut = cut.min(max_cut);

    points[cut..(len - cut)].to_vec()
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

/// Subdivide a quadratic bezier to extract the section from t0 to t1
pub fn subdivide_quadratic_bezier(
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    t0: f32,
    t1: f32,
) -> (Vec3, Vec3, Vec3) {
    fn eval_bezier(p0: Vec3, p1: Vec3, p2: Vec3, t: f32) -> Vec3 {
        let omt = 1.0 - t;
        p0 * (omt * omt) + p1 * (2.0 * omt * t) + p2 * (t * t)
    }

    let new_p0 = eval_bezier(p0, p1, p2, t0);
    let new_p2 = eval_bezier(p0, p1, p2, t1);

    // Compute new control point to preserve curve shape
    let dt = t1 - t0;
    let tangent_at_t0 = (p1 - p0) * (1.0 - t0) + (p2 - p1) * t0;
    let new_p1 = new_p0 + tangent_at_t0 * dt;

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
    edges: &mut [(Vec3, Vec3)],
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

/// Cast ray from `from` through `through` and beyond, find first polygon intersection
fn ray_to_polygon(
    from: Vec3,
    through: Vec3,
    poly: &IntersectionPolygon,
    gizmo: &mut Gizmo,
) -> Option<Vec3> {
    let dx = through.x - from.x;
    let dz = through.z - from.z;
    let len = (dx * dx + dz * dz).sqrt();

    if len < 0.001 {
        return None;
    }

    // Extend ray far beyond 'through'
    let ray_end_x = from.x + dx / len * 2000.0;
    let ray_end_z = from.z + dz / len * 2000.0;

    let ray_dx = ray_end_x - from.x;
    let ray_dz = ray_end_z - from.z;

    let mut best_t = f32::MAX;
    let mut best_hit = None;

    for i in 0..poly.ring.len() {
        let a = poly.ring[i];
        let b = poly.ring[(i + 1) % poly.ring.len()];

        let seg_dx = b.x - a.x;
        let seg_dz = b.z - a.z;

        let cross = ray_dx * seg_dz - ray_dz * seg_dx;
        if cross.abs() < 1e-10 {
            continue;
        }

        let to_seg_x = a.x - from.x;
        let to_seg_z = a.z - from.z;

        let t = (to_seg_x * seg_dz - to_seg_z * seg_dx) / cross;
        let u = (to_seg_x * ray_dz - to_seg_z * ray_dx) / cross;

        // t > 0: intersection is ahead of 'from'
        // u in [0,1]: intersection is on the polygon segment
        if t > 0.0 && u >= 0.0 && u <= 1.0 && t < best_t {
            best_t = t;
            best_hit = Some(Vec3::new(
                from.x + ray_dx * t,
                through.y,
                from.z + ray_dz * t,
            ));
        }
    }
    if let Some(hit) = best_hit {
        // Uncomment to debug:
        println!("from={:?} through={:?} hit={:?}", from, through, hit);
        gizmo.draw_cross(hit, 10.0, [0.0, 0.0, 1.0], 50.0)
    }
    best_hit
}

/// points[0] = target (near intersection), points[1..] = going into lane
fn find_polygon_boundary_point(points: &[Vec3], poly: &IntersectionPolygon) -> Vec3 {
    if points.len() < 2 {
        return points.get(0).copied().unwrap_or(Vec3::ZERO);
    }

    let target = points[0];
    let target_inside = poly.contains_xz(target);

    if target_inside {
        // === TARGET INSIDE: Trace outward along edge to find exit ===
        for i in 0..points.len() - 1 {
            let a = points[i];
            let b = points[i + 1];

            if poly.contains_xz(a) && !poly.contains_xz(b) {
                if let Some(hit) = segment_poly_intersection(a, b, poly) {
                    return hit;
                }
            }
        }

        // Still inside at end of polyline - extend outward
        let last = points.len() - 1;
        let dir = (points[last] - points[0]).normalize_or_zero();
        let extended = points[last] + dir * 200.0;

        if let Some(hit) = segment_poly_intersection(points[last], extended, poly) {
            return hit;
        }
    } else {
        // === TARGET OUTSIDE: Extend toward intersection to find entry ===
        let dir = (points[0] - points[1]).normalize_or_zero();
        let extended = points[0] + dir * 200.0;

        if let Some(hit) = segment_poly_intersection(points[0], extended, poly) {
            return hit;
        }
    }

    target
}

fn segment_poly_intersection(a: Vec3, b: Vec3, poly: &IntersectionPolygon) -> Option<Vec3> {
    let mut best_t = f32::MAX;
    let mut hit = None;

    let dx = b.x - a.x;
    let dz = b.z - a.z;

    for i in 0..poly.ring.len() {
        let p1 = poly.ring[i];
        let p2 = poly.ring[(i + 1) % poly.ring.len()];

        let ex = p2.x - p1.x;
        let ez = p2.z - p1.z;

        let cross = dx * ez - dz * ex;
        if cross.abs() < 1e-10 {
            continue;
        }

        let fx = p1.x - a.x;
        let fz = p1.z - a.z;

        let t = (fx * ez - fz * ex) / cross;
        let u = (fx * dz - fz * dx) / cross;

        if t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 && t < best_t {
            best_t = t;
            hit = Some(Vec3::new(a.x + dx * t, a.y, a.z + dz * t));
        }
    }

    hit
}

fn find_edge_polygon_intersection(
    edge_line: &[Vec3],
    poly: &IntersectionPolygon,
    at_start: bool,
    gizmo: &mut Gizmo,
    debug_color: [f32; 3],
) -> Vec3 {
    let n = edge_line.len();
    if n < 2 {
        return edge_line.get(0).copied().unwrap_or(Vec3::ZERO);
    }

    let target_idx = if at_start { 0 } else { n - 1 };
    let target = edge_line[target_idx];

    // === METHOD 1: Check each edge segment for polygon crossing ===
    let segments: Box<dyn Iterator<Item = (usize, usize)>> = if at_start {
        Box::new((0..n - 1).map(|i| (i, i + 1)))
    } else {
        Box::new((0..n - 1).rev().map(|i| (i + 1, i)))
    };

    for (i, j) in segments {
        let a = edge_line[i];
        let b = edge_line[j];

        let a_inside = poly.contains_xz(a);
        let b_inside = poly.contains_xz(b);

        if a_inside != b_inside {
            // This segment crosses the boundary!
            if let Some(hit) = find_segment_polygon_hit(a, b, poly) {
                return hit;
            }
        }
    }

    // === METHOD 2: No crossing found, use ray projection ===
    // Compute direction from multiple points for robustness
    let dir = compute_robust_direction(edge_line, at_start);

    // DEBUG: Show the computed direction
    gizmo.render_arrow(target, target + dir * 10.0, [1.0, 1.0, 0.0], true, 50.0); // Yellow = direction

    // Cast ray from FAR inside the lane, through target, to FAR outside
    let ray_inside = target + dir * 200.0;
    let ray_outside = target - dir * 200.0;

    // Find ALL intersections and pick closest to target
    let mut best_hit: Option<Vec3> = None;
    let mut best_dist = f32::MAX;

    for i in 0..poly.ring.len() {
        let pa = poly.ring[i];
        let pb = poly.ring[(i + 1) % poly.ring.len()];

        if let Some(hit) = segment_intersection_xz(ray_outside, ray_inside, pa, pb) {
            let dist = (hit - target).length();

            if dist < best_dist {
                best_dist = dist;
                best_hit = Some(hit);
            }
        }
    }

    if let Some(hit) = best_hit {
        return hit;
    }

    // === METHOD 3: Nothing worked, return original ===
    gizmo.draw_cross(target, 5.5, [1.0, 0.0, 0.0], 10.0); // Red = failed
    target
}

fn compute_robust_direction(edge_line: &[Vec3], at_start: bool) -> Vec3 {
    let n = edge_line.len();

    let target_idx = if at_start { 0 } else { n - 1 };
    let target = edge_line[target_idx];

    // Try to find a point at least 2 meters away for a reliable direction
    let indices: Vec<usize> = if at_start {
        (1..n).collect()
    } else {
        (0..n - 1).rev().collect()
    };

    for i in indices {
        let delta = edge_line[i] - target;
        let len_sq = delta.x * delta.x + delta.z * delta.z; // XZ only
        if len_sq > 4.0 {
            // 2 meters squared
            return Vec3::new(delta.x, 0.0, delta.z).normalize();
        }
    }

    // Fallback: use adjacent point
    let adj_idx = if at_start {
        1.min(n - 1)
    } else {
        (n - 1).saturating_sub(1)
    };
    let delta = edge_line[adj_idx] - target;
    Vec3::new(delta.x, 0.0, delta.z).normalize_or_zero()
}

fn find_segment_polygon_hit(a: Vec3, b: Vec3, poly: &IntersectionPolygon) -> Option<Vec3> {
    let mut best_t = f32::MAX;
    let mut hit = None;

    for i in 0..poly.ring.len() {
        let pa = poly.ring[i];
        let pb = poly.ring[(i + 1) % poly.ring.len()];

        if let Some((t, point)) = segment_intersection_xz_with_t(a, b, pa, pb) {
            if t < best_t {
                best_t = t;
                hit = Some(point);
            }
        }
    }

    hit
}

fn segment_intersection_xz(p1: Vec3, p2: Vec3, p3: Vec3, p4: Vec3) -> Option<Vec3> {
    segment_intersection_xz_with_t(p1, p2, p3, p4).map(|(_, p)| p)
}

fn segment_intersection_xz_with_t(p1: Vec3, p2: Vec3, p3: Vec3, p4: Vec3) -> Option<(f32, Vec3)> {
    let d1x = p2.x - p1.x;
    let d1z = p2.z - p1.z;
    let d2x = p4.x - p3.x;
    let d2z = p4.z - p3.z;

    // Cross product in 2D (XZ plane)
    let cross = d1x * d2z - d1z * d2x;

    if cross.abs() < 1e-10 {
        return None; // Parallel or collinear
    }

    let d3x = p3.x - p1.x;
    let d3z = p3.z - p1.z;

    let t = (d3x * d2z - d3z * d2x) / cross;
    let u = (d3x * d1z - d3z * d1x) / cross;

    // Both parameters must be in [0, 1] for segments to intersect
    if t >= -1e-6 && t <= 1.0 + 1e-6 && u >= -1e-6 && u <= 1.0 + 1e-6 {
        let t_clamped = t.clamp(0.0, 1.0);
        let point = Vec3::new(
            p1.x + d1x * t_clamped,
            (p1.y + p2.y) * 0.5,
            p1.z + d1z * t_clamped,
        );
        Some((t_clamped, point))
    } else {
        None
    }
}
/// Find the intersection point of a ray with a polygon (closest to 'from')
pub fn ray_polygon_intersection(from: Vec3, to: Vec3, poly: &IntersectionPolygon) -> Option<Vec3> {
    let n = poly.ring.len();
    if n < 3 {
        return None;
    }

    let mut best_t: Option<f32> = None;

    let d1 = Vec2::new(to.x - from.x, to.z - from.z);

    for i in 0..n {
        let a = poly.ring[i];
        let b = poly.ring[(i + 1) % n];

        let d2 = Vec2::new(b.x - a.x, b.z - a.z);
        let denom = d1.x * d2.y - d1.y * d2.x;

        if denom.abs() < 1e-10 {
            continue;
        }

        let d3 = Vec2::new(a.x - from.x, a.z - from.z);
        let t = (d3.x * d2.y - d3.y * d2.x) / denom;
        let u = (d3.x * d1.y - d3.y * d1.x) / denom;

        // t must be positive (in direction of ray) and u must be on the edge segment
        if t > 1e-6 && t <= 1.0 && u >= 0.0 && u <= 1.0 {
            best_t = Some(match best_t {
                Some(bt) => bt.min(t),
                None => t,
            });
        }
    }

    best_t.map(|t| from.lerp(to, t))
}
