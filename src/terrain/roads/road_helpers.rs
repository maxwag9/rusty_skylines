use crate::renderer::world_renderer::TerrainRenderer;
use crate::terrain::roads::intersections::{OuterNodeLane, road_vertex};
use crate::terrain::roads::road_mesh_manager::RoadVertex;
use crate::terrain::roads::road_structs::SegmentId;
use crate::terrain::roads::roads::{LaneRef, NodeLane, RoadStorage};
use glam::Vec3;
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
    let mut by_src: HashMap<SegmentId, Vec<(&NodeLane, i8, i8)>> = HashMap::new();

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
        ));
    }

    let mut result = Vec::new();

    for (_seg, entries) in by_src {
        if entries.is_empty() {
            continue;
        }

        let max_src = entries.iter().map(|&(_, s, _)| s).max().unwrap();
        let max_dst = entries.iter().map(|&(_, _, d)| d).max().unwrap();

        let winner = entries
            .into_iter()
            .filter(|&(_, s, d)| s == max_src && d == max_dst)
            .max_by(|(a, _, _), (b, _, _)| {
                let sa = right_turn_score(&a.polyline());
                let sb = right_turn_score(&b.polyline());
                sa.partial_cmp(&sb).unwrap()
            });

        if let Some((nl, _, _)) = winner {
            result.push(OuterNodeLane {
                node_lane: nl.id(),
                outward_sign: 1,
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
    -cross.y.atan2(dot)
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
