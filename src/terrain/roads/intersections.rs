use crate::renderer::gizmo::Gizmo;
use crate::renderer::world_renderer::TerrainRenderer;
use crate::terrain::roads::road_helpers::{
    build_strip_polyline, merge_polylines_ccw, offset_polyline_f32, select_outermost_lanes,
    triangulate_center_fan,
};
use crate::terrain::roads::road_mesh_manager::{
    CLEARANCE, ChunkId, MeshConfig, RoadVertex, chunk_x_range, chunk_z_range,
};
use crate::terrain::roads::road_structs::*;
use crate::terrain::roads::roads::*;
use glam::Vec3;
use std::collections::HashMap;

/// Samples per corner arc (more = smoother corners)
pub const CORNER_SAMPLES: usize = 5;

/// Laplacian smoothing passes on corner vertices
const SMOOTHING_PASSES: usize = 2;

/// Minimum extent from center to lane trim point
const MIN_EXTENT: f32 = 3.0;

/// Maximum extent (prevents blowup at acute angles)
const MAX_EXTENT: f32 = 25.0;

/// Minimum angle between approaches before clamping (radians, ~10°)
const MIN_ANGLE_RAD: f32 = 0.17;

/// Number of vertices to smooth after boundary connection
const BOUNDARY_SMOOTH_DEPTH: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PerimeterType {
    Opening,
    Corner,
}

#[derive(Clone, Debug)]
struct LaneApproachInfo {
    lane_id: LaneId,
    lane_index: i8,
    lateral_offset: f32,
}

#[derive(Clone, Debug)]
struct SegmentApproach {
    segment_id: SegmentId,
    lanes: Vec<LaneApproachInfo>,
    direction: Vec3,
    angle: f32,
    total_half_width: f32,
}

#[derive(Clone, Debug)]
pub struct LaneBoundaryInfo {
    pub trim_distance: f32,
    pub left: Vec3,
    pub right: Vec3,
    pub center: Vec3,
    pub direction: Vec3,
}

#[derive(Clone, Debug, Default)]
pub struct IntersectionMeshResult {
    pub lane_boundaries: LaneBoundaries,
}

/// Collected boundary info for a segment: (from_node_boundary, to_node_boundary)
pub type SegmentBoundaries =
    HashMap<SegmentId, (Option<LaneBoundaryInfo>, Option<LaneBoundaryInfo>)>;

/// Per-lane boundaries: lane_id -> (start_boundary, end_boundary)
pub type LaneBoundaries = HashMap<LaneId, (Option<LaneBoundaryInfo>, Option<LaneBoundaryInfo>)>;

// ============================================================================
// Intersection Mesh Builder
// ============================================================================
pub struct OuterNodeLane {
    pub(crate) node_lane: NodeLaneId,
    pub(crate) outward_sign: i8,
}
pub fn build_intersection_mesh(
    terrain: &TerrainRenderer,
    node_id: NodeId,
    node: &Node,
    storage: &RoadStorage,
    style: &RoadStyleParams,
    _config: &MeshConfig,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
    gizmo: &mut Gizmo,
) -> IntersectionMeshResult {
    let center = Vec3::from_array(node.position());

    // 1. Collect node lane polylines (towards node)
    let node_lanes = node.node_lanes();
    if node_lanes.len() < 2 {
        return IntersectionMeshResult::default();
    }

    // 2. Pick outermost lanes per approach
    let outer_lanes = select_outermost_lanes(node_lanes, storage);

    // 3. Offset outer lanes outward to form asphalt boundary polylines
    let mut asphalt_polylines = Vec::new();
    for lane in &outer_lanes {
        let offset = style.lane_width * 0.5 * lane.outward_sign as f32;
        //println!("node lane id: {}, length: {}", lane.node_lane, node_lanes.len());
        let node_lane = node.node_lane(lane.node_lane);
        //println!("Polyline length: {}", node_lane.polyline().len());
        gizmo.render_polyline(
            &node_lane
                .polyline()
                .iter()
                .map(|v| [v.x, v.y, v.z])
                .collect::<Vec<[f32; 3]>>(),
            [1.0, 0.0, 1.0],
            2.0,
            true,
        );

        let poly = offset_polyline_f32(&node_lane.polyline(), offset);
        asphalt_polylines.push(poly);
    }
    // for polyline in &asphalt_polylines {
    //     gizmo.render_polyline(
    //         &polyline.iter().map(|v| [v.x, v.y, v.z]).collect::<Vec<[f32; 3]>>(),
    //         [1.0, 0.0, 1.0],
    //         0.5,
    //         true,
    //     );
    // }

    // 4. Merge polylines into a single CCW ring
    let mut asphalt_ring = merge_polylines_ccw(center, asphalt_polylines);
    // gizmo.render_polyline(
    //     &asphalt_ring.iter().map(|v| [v.x, v.y, v.z]).collect::<Vec<[f32; 3]>>(),
    //     [0.0, 0.0, 1.0],
    //     0.5, true
    // );
    if asphalt_ring.len() < 3 {
        return IntersectionMeshResult::default();
    }

    // 5a. Push the Center Vertex (Pivot)
    // This prevents long sliver triangles and improves lighting/terrain conformity
    let asphalt_base = vertices.len() as u32;
    let center_h = terrain.get_height_at([center.x, center.z]);
    vertices.push(road_vertex(
        center.x,
        center_h + style.lane_height,
        center.z,
        style.lane_material_id,
        0.5, // UVs for center
        0.5,
    ));

    // 5b. Push Ring Vertices
    for p in asphalt_ring.iter().rev() {
        let h = terrain.get_height_at([p.x, p.z]);
        vertices.push(road_vertex(
            p.x,
            h + style.lane_height,
            p.z,
            style.lane_material_id,
            // Simple planar mapping or distance based UVs could go here
            (p.x - center.x) * 0.1,
            (p.z - center.z) * 0.1,
        ));
    }

    // 6. Triangulate (Triangle Fan around Center)
    triangulate_center_fan(asphalt_base, asphalt_ring.len() as u32, indices);

    // 6. Build sidewalks (simple offset)
    if style.sidewalk_width > 0.01 {
        for lane in &outer_lanes {
            let node_lane = node.node_lane(lane.node_lane);

            let asphalt_edge = offset_polyline_f32(
                &node_lane.polyline(),
                style.lane_width * 0.5 * lane.outward_sign as f32,
            );

            let sidewalk_outer = offset_polyline_f32(&asphalt_edge, style.sidewalk_width);

            build_strip_polyline(
                terrain,
                &asphalt_edge,
                &sidewalk_outer,
                style.sidewalk_height,
                style.sidewalk_material_id,
                vertices,
                indices,
            );
        }
    }

    // 7. Export exact lane boundary info for segments
    // let lane_boundaries = extract_lane_boundaries(
    //     node_id,
    //     &outer_lanes,
    //     &asphalt_ring,
    // );

    IntersectionMeshResult {
        lane_boundaries: LaneBoundaries::new(),
    }
}
// ============================================================================
// Helper Functions
// ============================================================================

fn lateral(dir: Vec3) -> Vec3 {
    Vec3::new(-dir.z, 0.0, dir.x)
}

fn positive_angle_diff(from: f32, to: f32) -> f32 {
    let mut diff = to - from;
    while diff < 0.0 {
        diff += std::f32::consts::TAU;
    }
    while diff >= std::f32::consts::TAU {
        diff -= std::f32::consts::TAU;
    }
    diff
}

fn arc_lerp(center: Vec3, a: Vec3, b: Vec3, t: f32) -> Vec3 {
    let ra = a - center;
    let rb = b - center;

    let angle_a = ra.z.atan2(ra.x);
    let angle_b = rb.z.atan2(rb.x);

    let mut delta = angle_b - angle_a;
    if delta < 0.0 {
        delta += std::f32::consts::TAU;
    }
    if delta > std::f32::consts::PI {
        delta -= std::f32::consts::TAU;
    }

    let angle = angle_a + delta * t;
    let radius = ra.length() * (1.0 - t) + rb.length() * t;

    center + Vec3::new(angle.cos() * radius, 0.0, angle.sin() * radius)
}

pub fn road_vertex(x: f32, y: f32, z: f32, mat: u32, u: f32, v: f32) -> RoadVertex {
    RoadVertex {
        position: [x, y + CLEARANCE, z],
        normal: [0.0, 1.0, 0.0],
        uv: [u, v],
        material_id: mat,
    }
}

fn interpolate_at_distance(geom: &LaneGeometry, target_dist: f32) -> Vec3 {
    for i in 1..geom.points.len() {
        if geom.lengths[i] >= target_dist {
            let prev_dist = geom.lengths[i - 1];
            let segment_len = geom.lengths[i] - prev_dist;
            if segment_len < 0.001 {
                return geom.points[i];
            }
            let t = (target_dist - prev_dist) / segment_len;
            return geom.points[i - 1].lerp(geom.points[i], t);
        }
    }
    *geom.points.last().unwrap_or(&Vec3::ZERO)
}

fn filter_by_chunk(points: &[(Vec3, f32)], chunk_filter: Option<ChunkId>) -> Vec<(Vec3, f32)> {
    match chunk_filter {
        None => points.to_vec(),
        Some(cid) => {
            let (min_x, max_x) = chunk_x_range(cid);
            let (min_z, max_z) = chunk_z_range(cid);

            let mut result = Vec::new();
            for (i, &(p, dist)) in points.iter().enumerate() {
                let in_chunk = p.x >= min_x && p.x < max_x && p.z >= min_z && p.z < max_z;

                let prev_in = i > 0 && {
                    let pp = points[i - 1].0;
                    pp.x >= min_x && pp.x < max_x && pp.z >= min_z && pp.z < max_z
                };

                let next_in = i + 1 < points.len() && {
                    let pn = points[i + 1].0;
                    pn.x >= min_x && pn.x < max_x && pn.z >= min_z && pn.z < max_z
                };

                if in_chunk || prev_in || next_in {
                    result.push((p, dist));
                }
            }
            result
        }
    }
}

fn compute_tangent(points: &[(Vec3, f32)], idx: usize) -> Vec3 {
    if points.len() < 2 {
        return Vec3::X;
    }

    if idx + 1 < points.len() {
        (points[idx + 1].0 - points[idx].0).normalize_or_zero()
    } else if idx > 0 {
        (points[idx].0 - points[idx - 1].0).normalize_or_zero()
    } else {
        Vec3::X
    }
}
