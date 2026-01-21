use crate::renderer::world_renderer::TerrainRenderer;
use crate::terrain::roads::intersections::{OuterNodeLane, road_vertex};
use crate::terrain::roads::road_editor::IntersectionBuildParams;
use crate::terrain::roads::road_mesh_manager::{ChunkId, RoadVertex};
use crate::terrain::roads::road_structs::{NodeId, RoadStyleParams, SegmentId};
use crate::terrain::roads::roads::{LaneRef, NodeLane, RoadCommand, RoadStorage};
use glam::{Vec2, Vec3, Vec3Swizzles};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::f32::consts::PI;

/// Offset direction for a polyline segment
fn compute_normal_at_point(p_prev: Vec3, p_next: Vec3) -> Vec3 {
    let tangent = (p_next - p_prev).normalize();
    // Cross with Y-up to get horizontal normal (pointing right of travel direction)
    tangent.cross(Vec3::Y).normalize()
}

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

pub fn min_polyline_distance(a: &[Vec3], b: &[Vec3]) -> f32 {
    let mut min = f32::MAX;

    for i in 0..a.len() - 1 {
        for j in 0..b.len() - 1 {
            let d = segment_segment_distance(a[i], a[i + 1], b[j], b[j + 1]);
            min = min.min(d);
        }
    }

    min
}
fn min_polyline_point_distance(poly: &[Vec3], p: Vec3) -> f32 {
    if poly.len() < 2 {
        return f32::MAX;
    }

    let mut min_d = f32::MAX;

    for i in 0..poly.len() - 1 {
        min_d = min_d.min(point_segment_distance(p, poly[i], poly[i + 1]));
    }

    min_d
}
pub fn right_normal(dir: Vec3) -> Vec3 {
    Vec3::new(dir.z, 0.0, -dir.x)
}

pub fn left_normal(dir: Vec3) -> Vec3 {
    Vec3::new(-dir.z, 0.0, dir.x)
}
pub fn intersect_rays_2d(p: Vec3, r: Vec3, q: Vec3, s: Vec3) -> Option<Vec3> {
    let p2 = p.xz();
    let r2 = r.xz();
    let q2 = q.xz();
    let s2 = s.xz();

    let cross = |a: Vec2, b: Vec2| a.x * b.y - a.y * b.x;

    let rxs = cross(r2, s2);
    if rxs.abs() < 1e-5 {
        return None;
    }

    let t = cross(q2 - p2, s2) / rxs;

    Some(p + r * t)
}

pub fn point_segment_distance(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let t = (p - a).dot(ab) / ab.dot(ab);

    if t <= 0.0 {
        (p - a).length()
    } else if t >= 1.0 {
        (p - b).length()
    } else {
        (p - (a + ab * t)).length()
    }
}
fn segment_segment_distance(a0: Vec3, a1: Vec3, b0: Vec3, b1: Vec3) -> f32 {
    let u = a1 - a0;
    let v = b1 - b0;
    let w = a0 - b0;

    let a = u.dot(u);
    let b = u.dot(v);
    let c = v.dot(v);
    let d = u.dot(w);
    let e = v.dot(w);

    let denom = a * c - b * b;
    let mut sc;
    let mut tc;

    if denom.abs() < 1e-6 {
        sc = 0.0;
        tc = if b > c { d / b } else { e / c };
    } else {
        sc = (b * e - c * d) / denom;
        tc = (a * e - b * d) / denom;
    }

    sc = sc.clamp(0.0, 1.0);
    tc = tc.clamp(0.0, 1.0);

    let p = a0 + u * sc;
    let q = b0 + v * tc;

    (p - q).length()
}

/// Get source segment from the lane's incoming merge reference
fn get_source_segment(lane: &NodeLane, storage: &RoadStorage) -> Option<SegmentId> {
    for merge_ref in lane.merging() {
        if let LaneRef::Segment(lane_id, _) = merge_ref {
            return Some(storage.lane(lane_id).segment());
        }
    }
    None
}

/// Get destination segment from the lane's outgoing connection reference
fn get_dest_segment(lane: &NodeLane, storage: &RoadStorage) -> Option<SegmentId> {
    for connect_ref in lane.merging() {
        if let LaneRef::Segment(lane_id, _) = connect_ref {
            return Some(storage.lane(lane_id).segment());
        }
    }
    None
}

/// Returns angular difference in range [0, PI]
fn angle_diff(a: f32, b: f32) -> f32 {
    use std::f32::consts::PI;
    let mut diff = (a - b).abs();
    if diff > PI {
        diff = 2.0 * PI - diff;
    }
    diff
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
        indices.push(current);
        indices.push(next);
    }
}

pub fn build_strip_polyline(
    terrain: &TerrainRenderer,
    inner: &[Vec3],
    outer: &[Vec3],
    height: f32,
    material_id: u32,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    assert!(inner.len() == outer.len() && inner.len() >= 2);
    let base = vertices.len() as u32;

    let n = inner.len();
    for (i, (&pi, &po)) in inner.iter().zip(outer.iter()).enumerate() {
        let h_in = terrain.get_height_at([pi.x, pi.z]);
        let h_out = terrain.get_height_at([po.x, po.z]);

        let u = i as f32 / (n - 1) as f32;

        vertices.push(road_vertex(pi.x, h_in + height, pi.z, material_id, u, 0.0));
        vertices.push(road_vertex(po.x, h_out + height, po.z, material_id, u, 1.0));
    }

    for i in 0..(n - 1) {
        let v0 = base + (i * 2) as u32;
        let v1 = base + (i * 2 + 1) as u32;
        let v2 = base + ((i + 1) * 2) as u32;
        let v3 = base + ((i + 1) * 2 + 1) as u32;

        // Ensure CCW winding when looking down Y+
        indices.extend_from_slice(&[v0, v1, v2]);
        indices.extend_from_slice(&[v2, v1, v3]);
    }
}
fn cross2(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

pub fn point_in_polygon_xz(p: Vec3, poly: &[Vec3]) -> bool {
    // Ray-casting in XZ plane.
    // poly is treated as closed (edge i -> (i+1)%n).
    let n = poly.len();
    if n < 3 {
        return false;
    }

    let x = p.x;
    let z = p.z;

    let mut inside = false;
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];

        let (x1, z1) = (a.x, a.z);
        let (x2, z2) = (b.x, b.z);

        // Check if the horizontal ray crosses edge (a,b)
        let intersects =
            ((z1 > z) != (z2 > z)) && (x < (x2 - x1) * (z - z1) / ((z2 - z1).max(1e-12)) + x1);

        if intersects {
            inside = !inside;
        }
    }
    inside
}
/// Returns intersections as (s_along_polyline_from_start, intersection_point_world)
pub fn polyline_polygon_intersections_xz(points: &[Vec3], poly: &[Vec3]) -> Vec<(f32, Vec3)> {
    let mut out = Vec::new();
    if points.len() < 2 || poly.len() < 3 {
        return out;
    }

    let mut s_acc = 0.0f32;

    for i in 0..(points.len() - 1) {
        let p0 = points[i];
        let p1 = points[i + 1];
        let a0 = p0.xz();
        let a1 = p1.xz();

        let seg_len = (p1 - p0).length();
        if seg_len < 1e-6 {
            continue;
        }

        for j in 0..poly.len() {
            let q0 = poly[j];
            let q1 = poly[(j + 1) % poly.len()];
            let b0 = q0.xz();
            let b1 = q1.xz();

            if let Some((hit2, t)) = seg_seg_intersection_2d(a0, a1, b0, b1) {
                // Lift back to 3D by interpolating the original 3D segment
                let hit3 = p0 + (p1 - p0) * t;
                let s_hit = s_acc + seg_len * t;

                out.push((s_hit, Vec3::new(hit2.x, hit3.y, hit2.y)));
            }
        }

        s_acc += seg_len;
    }

    // Dedup very-close hits (common when intersecting exactly at a polygon vertex)
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-3);

    out
}
pub fn segment_polygon_intersection_xz(a: Vec3, b: Vec3, poly: &[Vec3]) -> Option<(Vec3, f32)> {
    let n = poly.len();
    if n < 3 {
        return None;
    }

    let a0 = a.xz();
    let a1 = b.xz();

    let mut best_t: Option<f32> = None;
    let mut best_hit2: Vec2 = Vec2::ZERO;

    for i in 0..n {
        let q0 = poly[i].xz();
        let q1 = poly[(i + 1) % n].xz();

        if let Some((hit2, t)) = seg_seg_intersection_2d(a0, a1, q0, q1) {
            match best_t {
                None => {
                    best_t = Some(t);
                    best_hit2 = hit2;
                }
                Some(bt) if t < bt => {
                    best_t = Some(t);
                    best_hit2 = hit2;
                }
                _ => {}
            }
        }
    }

    let t = best_t?;
    let hit3 = a + (b - a) * t;
    Some((Vec3::new(best_hit2.x, hit3.y, best_hit2.y), t))
}
pub fn carve_polyline_outside_polygon(
    points: &[Vec3],
    cut_poly: &[Vec3],
    node_at_start: bool,
) -> Option<(Vec<Vec3>, Vec<Vec3>, Vec3)> {
    // returns (kept_points, removed_points_for_debug, boundary_point)

    if points.len() < 2 || cut_poly.len() < 3 {
        return None;
    }

    // Convenience closures
    let is_inside = |p: Vec3| point_in_polygon_xz(p, cut_poly);

    if node_at_start {
        // Remove from the START while inside; keep the first outside segment onward.
        if !is_inside(points[0]) {
            // Already outside -> nothing to carve
            return Some((points.to_vec(), Vec::new(), points[0]));
        }

        // Find first index that is outside
        let mut i_out = None;
        for i in 1..points.len() {
            if !is_inside(points[i]) {
                i_out = Some(i);
                break;
            }
        }
        let i_out = i_out?; // entire polyline inside -> obliterated

        // Crossing occurs between i_out-1 (inside) and i_out (outside)
        let a = points[i_out - 1];
        let b = points[i_out];
        let (hit, _t) = segment_polygon_intersection_xz(a, b, cut_poly)?;

        // removed: from start to hit
        let mut removed = Vec::new();
        removed.extend_from_slice(&points[..i_out]); // includes b(outside) but that's OK for debug; we’ll adjust below
        // kept: hit + remainder outside
        let mut kept = Vec::new();
        kept.push(hit);
        kept.extend_from_slice(&points[i_out..]);

        // Make removed nicer: show up to hit only
        removed.truncate(i_out); // points[..i_out] ends at inside point; fine
        removed.push(hit);

        Some((kept, removed, hit))
    } else {
        // Remove from the END while inside; keep up to the last outside point.
        let last = points[points.len() - 1];
        if !is_inside(last) {
            return Some((points.to_vec(), Vec::new(), last));
        }

        // Find last index that is outside (walking backward)
        let mut i_out = None;
        for i in (0..points.len() - 1).rev() {
            if !is_inside(points[i]) {
                i_out = Some(i);
                break;
            }
        }
        let i_out = i_out?; // entire polyline inside -> obliterated

        // Crossing between i_out (outside) and i_out+1 (inside)
        let a = points[i_out];
        let b = points[i_out + 1];
        let (hit, _t) = segment_polygon_intersection_xz(a, b, cut_poly)?;

        let mut kept = Vec::new();
        kept.extend_from_slice(&points[..=i_out]);
        kept.push(hit);

        let mut removed = Vec::new();
        removed.push(hit);
        removed.extend_from_slice(&points[i_out + 1..]);

        Some((kept, removed, hit))
    }
}
fn dist2_point_to_segment_2d(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
    let q = a + ab * t;
    (p - q).length_squared()
}

pub fn point_on_polygon_edge_xz(p: Vec3, poly: &[Vec3], eps: f32) -> bool {
    if poly.len() < 2 {
        return false;
    }
    let pp = p.xz();
    for i in 0..poly.len() {
        let a = poly[i].xz();
        let b = poly[(i + 1) % poly.len()].xz();
        if dist2_point_to_segment_2d(pp, a, b) <= eps * eps {
            return true;
        }
    }
    false
}

/// Ray-cast point-in-polygon in XZ.
/// IMPORTANT: points ON the edge are treated as OUTSIDE (so boundary points are kept).
pub fn point_in_polygon_xz_exclusive(p: Vec3, poly: &[Vec3]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }

    if point_on_polygon_edge_xz(p, poly, 1e-4) {
        return false;
    }

    let x = p.x;
    let z = p.z;

    let mut inside = false;
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        let (x1, z1) = (a.x, a.z);
        let (x2, z2) = (b.x, b.z);

        // does horizontal ray to +X cross edge?
        if ((z1 > z) != (z2 > z)) {
            let x_at_z = (x2 - x1) * (z - z1) / ((z2 - z1).max(1e-12)) + x1;
            if x < x_at_z {
                inside = !inside;
            }
        }
    }
    inside
}

/// 2D segment-segment intersection. Returns t on (a0->a1) in [0,1] if hit.
pub fn seg_seg_intersection_2d(a0: Vec2, a1: Vec2, b0: Vec2, b1: Vec2) -> Option<(Vec2, f32)> {
    let r = a1 - a0;
    let s = b1 - b0;
    let denom = cross2(r, s);

    const EPS: f32 = 1e-6;
    if denom.abs() < EPS {
        return None; // parallel
    }

    let qp = b0 - a0;
    let t = cross2(qp, s) / denom;
    let u = cross2(qp, r) / denom;

    if t >= -1e-4 && t <= 1.0 + 1e-4 && u >= -1e-4 && u <= 1.0 + 1e-4 {
        let tt = t.clamp(0.0, 1.0);
        let p = a0 + r * tt;
        return Some((p, tt));
    }
    None
}

/// Returns the FIRST intersection along segment a->b with the polygon boundary.
pub fn segment_polygon_first_intersection_xz(
    a: Vec3,
    b: Vec3,
    poly: &[Vec3],
) -> Option<(Vec3, f32)> {
    if poly.len() < 3 {
        return None;
    }

    let a0 = a.xz();
    let a1 = b.xz();

    let mut best_t: Option<f32> = None;
    let mut best_hit2 = Vec2::ZERO;

    for i in 0..poly.len() {
        let q0 = poly[i].xz();
        let q1 = poly[(i + 1) % poly.len()].xz();

        if let Some((hit2, t)) = seg_seg_intersection_2d(a0, a1, q0, q1) {
            match best_t {
                None => {
                    best_t = Some(t);
                    best_hit2 = hit2;
                }
                Some(bt) if t < bt => {
                    best_t = Some(t);
                    best_hit2 = hit2;
                }
                _ => {}
            }
        }
    }

    let t = best_t?;
    let y = (a + (b - a) * t).y;
    Some((Vec3::new(best_hit2.x, y, best_hit2.y), t))
}

/// Carve polyline against polygon: KEEP OUTSIDE, DELETE INSIDE.
/// Returns:
/// - outside pieces (can be multiple)
/// - inside pieces (debug)
/// - all intersection points used (debug)
pub fn carve_polyline_keep_outside_xz(
    points: &[Vec3],
    poly: &[Vec3],
) -> (Vec<Vec<Vec3>>, Vec<Vec<Vec3>>, Vec<Vec3>) {
    let mut outside_parts: Vec<Vec<Vec3>> = Vec::new();
    let mut inside_parts: Vec<Vec<Vec3>> = Vec::new();
    let mut hits: Vec<Vec3> = Vec::new();

    if points.len() < 2 || poly.len() < 3 {
        return (outside_parts, inside_parts, hits);
    }

    let inside = |p: Vec3| point_in_polygon_xz_exclusive(p, poly);

    let mut cur_out: Vec<Vec3> = Vec::new();
    let mut cur_in: Vec<Vec3> = Vec::new();

    let mut prev = points[0];
    let mut prev_inside = inside(prev);

    if prev_inside {
        cur_in.push(prev);
    } else {
        cur_out.push(prev);
    }

    for &next in &points[1..] {
        let next_inside = inside(next);

        match (prev_inside, next_inside) {
            (false, false) => {
                // outside -> outside
                cur_out.push(next);
            }
            (true, true) => {
                // inside -> inside
                cur_in.push(next);
            }
            (false, true) => {
                // entering polygon: outside -> inside
                if let Some((hit, _t)) = segment_polygon_first_intersection_xz(prev, next, poly) {
                    hits.push(hit);

                    cur_out.push(hit);
                    if cur_out.len() >= 2 {
                        outside_parts.push(std::mem::take(&mut cur_out));
                    }

                    cur_in.push(hit);
                    cur_in.push(next);
                } else {
                    // Shouldn't happen for simple polygon; keep debug visible
                    cur_out.push(next);
                    if cur_out.len() >= 2 {
                        outside_parts.push(std::mem::take(&mut cur_out));
                    }
                    cur_in.push(next);
                }
            }
            (true, false) => {
                // leaving polygon: inside -> outside
                if let Some((hit, _t)) = segment_polygon_first_intersection_xz(prev, next, poly) {
                    hits.push(hit);

                    cur_in.push(hit);
                    if cur_in.len() >= 2 {
                        inside_parts.push(std::mem::take(&mut cur_in));
                    }

                    cur_out.push(hit);
                    cur_out.push(next);
                } else {
                    cur_in.push(next);
                    if cur_in.len() >= 2 {
                        inside_parts.push(std::mem::take(&mut cur_in));
                    }
                    cur_out.push(next);
                }
            }
        }

        prev = next;
        prev_inside = next_inside;
    }

    if cur_out.len() >= 2 {
        outside_parts.push(cur_out);
    }
    if cur_in.len() >= 2 {
        inside_parts.push(cur_in);
    }

    (outside_parts, inside_parts, hits)
}

/// From multiple outside pieces, pick the one that continues AWAY from the node.
/// We do that by choosing the piece that contains the "far end" of the original polyline.
pub fn pick_outside_piece_away_from_node(
    outside_parts: Vec<Vec<Vec3>>,
    original: &[Vec3],
    node_at_start: bool,
) -> Option<Vec<Vec3>> {
    if outside_parts.is_empty() {
        return None;
    }
    if original.is_empty() {
        return None;
    }

    let far_end = if node_at_start {
        *original.last().unwrap()
    } else {
        original[0]
    };
    let eps2 = 1e-4f32 * 1e-4f32;

    // Prefer a part whose endpoint matches far_end (typical case)
    for part in &outside_parts {
        if (part[0] - far_end).length_squared() <= eps2
            || (part[part.len() - 1] - far_end).length_squared() <= eps2
        {
            return Some(part.clone());
        }
    }

    // Fallback: pick the part whose closest point to far_end is smallest (or longest if you prefer)
    let mut best_i = 0usize;
    let mut best_d2 = f32::INFINITY;

    for (i, part) in outside_parts.iter().enumerate() {
        for &p in part {
            let d2 = (p - far_end).length_squared();
            if d2 < best_d2 {
                best_d2 = d2;
                best_i = i;
            }
        }
    }

    Some(outside_parts[best_i].clone())
}
// Returns t along a0->a1 where it intersects b0->b1 (if any). t in [0,1]
pub fn seg_seg_intersection_t(a0: Vec2, a1: Vec2, b0: Vec2, b1: Vec2) -> Option<f32> {
    let r = a1 - a0;
    let s = b1 - b0;
    let denom = cross2(r, s);

    const EPS: f32 = 1e-6;
    if denom.abs() < EPS {
        return None; // parallel
    }

    let qp = b0 - a0;
    let t = cross2(qp, s) / denom;
    let u = cross2(qp, r) / denom;

    if t >= -1e-4 && t <= 1.0 + 1e-4 && u >= -1e-4 && u <= 1.0 + 1e-4 {
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
