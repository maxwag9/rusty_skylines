use crate::positions::{ChunkSize, WorldPos};
use crate::renderer::gizmo::Gizmo;
use crate::renderer::world_renderer::TerrainRenderer;
use crate::terrain::roads::intersections::{IntersectionBuildParams, IntersectionPolygon};
use crate::terrain::roads::road_mesh_manager::{CLEARANCE, ChunkId};
use crate::terrain::roads::road_structs::{NodeId, RoadStyleParams, StructureType};
use crate::terrain::roads::roads::RoadCommand;
use glam::Vec3;
use std::cmp::Ordering;
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
        //gizmo.cross(hit, 10.0, [0.0, 0.0, 1.0], 50.0);
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

pub fn tangent_and_lateral(points: &[WorldPos], i: usize, chunk_size: ChunkSize) -> (Vec3, Vec3) {
    let tangent = if i + 1 < points.len() {
        points[i]
            .delta_to(points[i + 1], chunk_size)
            .normalize_or_zero()
    } else if i > 0 {
        points[i - 1]
            .delta_to(points[i], chunk_size)
            .normalize_or_zero()
    } else {
        Vec3::X
    };

    let lateral = Vec3::new(-tangent.z, 0.0, tangent.x).normalize_or_zero();
    (tangent, lateral)
}
