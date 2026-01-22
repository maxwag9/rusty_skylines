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
use glam::{Vec2, Vec3};

/// Simple polygon for clipping segment meshes against intersections
#[derive(Clone, Debug, Default)]
pub struct IntersectionPolygon {
    pub ring: Vec<Vec3>, // CCW vertices in XZ plane
}

impl IntersectionPolygon {
    /// Point-in-polygon test (XZ plane, ray casting)
    pub fn contains_xz(&self, p: Vec3) -> bool {
        let n = self.ring.len();
        if n < 3 {
            return false;
        }

        let mut inside = false;
        let mut j = n - 1;

        for i in 0..n {
            let vi = self.ring[i];
            let vj = self.ring[j];

            if ((vi.z > p.z) != (vj.z > p.z))
                && (p.x < (vj.x - vi.x) * (p.z - vi.z) / (vj.z - vi.z) + vi.x)
            {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Find where line segment (from -> to) first intersects polygon boundary
    pub fn clip_to_edge(&self, from: Vec3, to: Vec3) -> Vec3 {
        let n = self.ring.len();
        let mut best_t = 1.0f32;

        for i in 0..n {
            let a = self.ring[i];
            let b = self.ring[(i + 1) % n];

            // Line-line intersection in XZ
            let d1 = Vec2::new(to.x - from.x, to.z - from.z);
            let d2 = Vec2::new(b.x - a.x, b.z - a.z);
            let denom = d1.x * d2.y - d1.y * d2.x;

            if denom.abs() < 1e-10 {
                continue;
            }

            let d3 = Vec2::new(a.x - from.x, a.z - from.z);
            let t = (d3.x * d2.y - d3.y * d2.x) / denom;
            let u = (d3.x * d1.y - d3.y * d1.x) / denom;

            if t > 0.0 && t < best_t && u >= 0.0 && u <= 1.0 {
                best_t = t;
            }
        }

        from.lerp(to, best_t)
    }
}
/// Helper: clip or extend a point to the polygon boundary using ray projection
pub fn clip_point_to_polygon(
    point: Vec3,
    lane_dir: Vec3, // Direction along lane AWAY from the intersection
    poly: &IntersectionPolygon,
    proj_dist: f32,
) -> Vec3 {
    if poly.contains_xz(point) {
        // Point is inside polygon - project outward along lane to find exit point
        let far_point = point + lane_dir * proj_dist;
        poly.clip_to_edge(far_point, point)
    } else {
        // Point is outside polygon - check if we need to EXTEND toward intersection
        let toward_intersection = point - lane_dir * proj_dist;
        if poly.contains_xz(toward_intersection) {
            // Lane doesn't reach polygon - extend to meet it
            poly.clip_to_edge(point, toward_intersection)
        } else {
            point
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct IntersectionMeshResult {
    pub polygon: IntersectionPolygon,
}

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
            10.0,
        );

        let poly = offset_polyline_f32(&node_lane.polyline(), offset);
        asphalt_polylines.push(poly);
        polyline_segment_info.push((lane.segment_id, lane.outward_sign));
    }

    let mut asphalt_ring = merge_polylines_ccw(center, asphalt_polylines.clone());

    if asphalt_ring.len() < 3 {
        return IntersectionMeshResult::default();
    }

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

            // Offset sign must match the lane's side
            let offset_sign = lane.outward_sign as f32;
            let sidewalk_offset =
                (style.lane_width * 0.5 + style.sidewalk_width * 0.5) * offset_sign;
            let curb_inner_offset = (style.lane_width * 0.5) * offset_sign;
            let curb_outer_offset = (style.lane_width * 0.5 + style.sidewalk_width) * offset_sign;

            build_ribbon_mesh(
                terrain,
                gizmo,
                node_lane.geometry(),
                style.sidewalk_width,
                style.sidewalk_height,
                sidewalk_offset, // Now correctly signed based on lane side
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
                curb_inner_offset,
                style.lane_height,
                style.sidewalk_height,
                style.sidewalk_material_id,
                None,
                (config.uv_scale_u, config.uv_scale_v),
                Some(-offset_sign), // Normal faces inward
                None,
                None,
                vertices,
                indices,
            );

            build_vertical_face(
                terrain,
                node_lane.geometry(),
                curb_outer_offset,
                style.lane_height,
                style.sidewalk_height,
                style.sidewalk_material_id,
                None,
                (config.uv_scale_u, config.uv_scale_v),
                Some(offset_sign), // Normal faces outward
                None,
                None,
                vertices,
                indices,
            );
        }
    }
    asphalt_ring.push(asphalt_ring.first().unwrap().clone());
    // Return the polygon for segment clipping
    IntersectionMeshResult {
        polygon: IntersectionPolygon { ring: asphalt_ring },
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
