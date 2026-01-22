use crate::renderer::gizmo::Gizmo;
use crate::renderer::world_renderer::TerrainRenderer;
use crate::terrain::roads::road_helpers::{
    merge_polylines_ccw, offset_polyline_f32, select_outermost_lanes, triangulate_center_fan,
};
use crate::terrain::roads::road_mesh_manager::{
    CLEARANCE, ChunkId, MeshConfig, RoadVertex, build_ribbon_mesh, build_vertical_face,
    chunk_x_range, chunk_z_range,
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

// In your structures/types file

#[derive(Clone, Debug, Default)]
pub struct IntersectionMeshResult {
    /// Maps segment IDs to their boundary data at this intersection
    pub segment_boundaries: HashMap<SegmentId, SegmentBoundaryAtNode>,
}

#[derive(Clone, Debug, Default)]
pub struct SegmentBoundaryAtNode {
    /// Per-lane boundary edges (keyed by lane_index)
    pub lane_edges: HashMap<i8, LaneBoundaryEdge>,
    /// Outer left edge of the entire segment at this boundary
    pub outer_left: Vec3,
    /// Outer right edge of the entire segment at this boundary
    pub outer_right: Vec3,
    /// Direction pointing outward from intersection (into segment)
    pub outward_direction: Vec3,
}

#[derive(Clone, Debug, Default, Copy)]
pub struct LaneBoundaryEdge {
    pub left: Vec3,
    pub center: Vec3,
    pub right: Vec3,
}

/// Per-lane boundaries: lane_id -> (start_boundary, end_boundary)
pub type LaneBoundaries = HashMap<LaneId, (Option<LaneBoundaryInfo>, Option<LaneBoundaryInfo>)>;

// ============================================================================
// Intersection Mesh Builder
// ============================================================================
pub struct OuterNodeLane {
    pub node_lane: NodeLaneId,
    pub outward_sign: i8,
    pub segment_id: SegmentId,
}
pub fn build_intersection_mesh(
    terrain: &TerrainRenderer,
    node_id: NodeId,
    node: &Node,
    storage: &RoadStorage,
    style: &RoadStyleParams,
    config: &MeshConfig,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
    gizmo: &mut Gizmo,
) -> IntersectionMeshResult {
    let center = Vec3::from_array(node.position());
    let node_lanes = node.node_lanes();

    if node_lanes.len() < 2 {
        return IntersectionMeshResult::default();
    }

    let outer_lanes = select_outermost_lanes(node_lanes, storage);

    // Build polylines AND track segment ownership
    let mut asphalt_polylines = Vec::new();
    let mut polyline_segment_info: Vec<(SegmentId, i8)> = Vec::new(); // (segment_id, outward_sign)

    for lane in &outer_lanes {
        let offset = style.lane_width * 0.5 * lane.outward_sign as f32;
        let node_lane = node.node_lane(lane.node_lane);

        gizmo.render_polyline(
            &node_lane
                .polyline()
                .iter()
                .map(|v| [v.x, v.y, v.z])
                .collect::<Vec<_>>(),
            [1.0, 0.0, 1.0],
            4.0,
            true,
        );

        let poly = offset_polyline_f32(&node_lane.polyline(), offset);
        asphalt_polylines.push(poly);
        polyline_segment_info.push((lane.segment_id, lane.outward_sign));
    }

    let asphalt_ring = merge_polylines_ccw(center, asphalt_polylines.clone());

    if asphalt_ring.len() < 3 {
        return IntersectionMeshResult::default();
    }

    // === Build segment boundary data ===
    let segment_boundaries = extract_segment_boundaries(
        node_id,
        node,
        &outer_lanes,
        &asphalt_polylines,
        storage,
        style,
    );

    // === Build mesh vertices ===
    // base index of the center vertex
    let asphalt_base = vertices.len() as u32;

    // center vertex (UV at texture center)
    let center_h = terrain.get_height_at([center.x, center.z]);
    vertices.push(road_vertex(
        center.x,
        center_h + style.lane_height,
        center.z,
        style.lane_material_id,
        0.5, // u
        0.5, // v
    ));

    // radial uv mapping using separate U/V scales
    let radial_uv = |pt: Vec3| -> (f32, f32) {
        let dx = pt.x - center.x;
        let dz = pt.z - center.z;
        let dist = (dx * dx + dz * dz).sqrt();

        if dist <= 1e-6 {
            return (0.5, 0.5);
        }

        let nx = dx / dist;
        let nz = dz / dist;

        // map world radius to uv radius: different scales for U and V
        let ru = dist * config.uv_scale_u;
        let rv = dist * config.uv_scale_v;

        let u = 0.5 + nx * ru;
        let v = 0.5 + nz * rv;

        (u, v)
    };

    // push ring in the same order triangulator expects (CCW)
    for p in asphalt_ring.iter() {
        let h = terrain.get_height_at([p.x, p.z]);
        let (u, v) = radial_uv(*p);
        vertices.push(road_vertex(
            p.x,
            h + style.lane_height,
            p.z,
            style.lane_material_id,
            u,
            v,
        ));
    }

    triangulate_center_fan(asphalt_base, asphalt_ring.len() as u32, indices);

    // Sidewalks
    if style.sidewalk_width > 0.01 {
        for lane in &outer_lanes {
            let node_lane = node.node_lane(lane.node_lane);
            let asphalt_edge = offset_polyline_f32(
                &node_lane.polyline(),
                style.lane_width * 0.5 * lane.outward_sign as f32,
            );
            let sidewalk_outer = offset_polyline_f32(
                &node_lane.polyline(),
                style.lane_width * 0.5 * lane.outward_sign as f32 + style.sidewalk_width,
            );
            build_ribbon_mesh(
                terrain,
                node_lane.geometry(),
                style.sidewalk_width,
                style.sidewalk_height,
                style.lane_width * 0.5 + style.sidewalk_width * 0.5,
                style.sidewalk_material_id,
                None,
                (config.uv_scale_u, config.uv_scale_v),
                None,
                None,
                vertices,
                indices,
            );
            build_vertical_face(
                terrain,
                node_lane.geometry(),
                style.lane_width * 0.5,
                style.lane_height,
                style.sidewalk_height,
                style.sidewalk_material_id,
                None,
                (config.uv_scale_u, config.uv_scale_v),
                Some(-1f32),
                vertices,
                indices,
            );
            build_vertical_face(
                terrain,
                node_lane.geometry(),
                style.lane_width * 0.5 + style.sidewalk_width,
                style.lane_height,
                style.sidewalk_height,
                style.sidewalk_material_id,
                None,
                (config.uv_scale_u, config.uv_scale_v),
                Some(1f32),
                vertices,
                indices,
            );
        }
    }

    IntersectionMeshResult { segment_boundaries }
}

/// Extract boundary data for each segment at this intersection
fn extract_segment_boundaries(
    node_id: NodeId,
    node: &Node,
    outer_lanes: &[OuterNodeLane],
    asphalt_polylines: &[Vec<Vec3>],
    storage: &RoadStorage,
    style: &RoadStyleParams,
) -> HashMap<SegmentId, SegmentBoundaryAtNode> {
    let mut result: HashMap<SegmentId, SegmentBoundaryAtNode> = HashMap::new();

    // First pass: collect boundary positions from outer lane polylines
    for (i, outer_lane) in outer_lanes.iter().enumerate() {
        let seg_id = outer_lane.segment_id;
        let polyline = &asphalt_polylines[i];

        if polyline.len() < 2 {
            continue;
        }

        // The FIRST point of the polyline is at the segment end (away from intersection)
        // The LAST point is at the intersection boundary
        let boundary_point = polyline.last().copied().unwrap_or_default();
        let prev_point = polyline
            .get(polyline.len().saturating_sub(2))
            .copied()
            .unwrap_or(boundary_point);

        // Direction pointing outward from intersection (into the segment)
        let outward_dir = (prev_point - boundary_point).normalize_or_zero();

        let entry = result.entry(seg_id).or_default();
        entry.outward_direction = outward_dir;

        // This is an outer edge, determine if left or right based on outward_sign
        if outer_lane.outward_sign > 0 {
            entry.outer_right = boundary_point;
        } else {
            entry.outer_left = boundary_point;
        }
    }

    // Second pass: compute per-lane boundaries
    for lane_id in node
        .incoming_lanes()
        .iter()
        .chain(node.outgoing_lanes().iter())
    {
        let lane = storage.lane(lane_id);
        let seg_id = lane.segment();
        let lane_idx = lane.lane_index();

        if let Some(boundary) = result.get_mut(&seg_id) {
            // Interpolate lane position between outer_left and outer_right
            let segment = storage.segment(seg_id);
            let lane_count = segment.lanes().len() as f32;

            // Get lane's endpoint position
            let polyline = lane.polyline();
            let is_at_start = segment.start() == node_id;

            let (lane_center, direction) = if is_at_start {
                let pos = polyline.first().copied().unwrap_or_default();
                let dir = if polyline.len() >= 2 {
                    (polyline[1] - polyline[0]).normalize_or_zero()
                } else {
                    boundary.outward_direction
                };
                (pos, dir)
            } else {
                let pos = polyline.last().copied().unwrap_or_default();
                let dir = if polyline.len() >= 2 {
                    let n = polyline.len();
                    (polyline[n - 2] - polyline[n - 1]).normalize_or_zero()
                } else {
                    boundary.outward_direction
                };
                (pos, dir)
            };

            // Compute lateral vector (perpendicular in XZ plane)
            let lateral = Vec3::new(-direction.z, 0.0, direction.x).normalize_or_zero();
            let half_lane = style.lane_width * 0.5;

            boundary.lane_edges.insert(
                lane_idx,
                LaneBoundaryEdge {
                    left: lane_center - lateral * half_lane,
                    center: lane_center,
                    right: lane_center + lateral * half_lane,
                },
            );
        }
    }

    result
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
