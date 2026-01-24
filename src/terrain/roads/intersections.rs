use crate::positions::{ChunkSize, WorldPos};
use crate::renderer::gizmo::Gizmo;
use crate::renderer::world_renderer::TerrainRenderer;
use crate::terrain::roads::road_helpers::*;
use crate::terrain::roads::road_mesh_manager::*;
use crate::terrain::roads::road_structs::*;
use crate::terrain::roads::roads::*;
use glam::{Vec2, Vec3};

/// Simple polygon for clipping segment meshes against intersections.
#[derive(Clone, Debug, Default)]
pub struct IntersectionPolygon {
    pub ring: Vec<WorldPos>,
    pub(crate) chunk_size: ChunkSize,
}

impl IntersectionPolygon {
    pub fn new(ring: Vec<WorldPos>, chunk_size: ChunkSize) -> Self {
        Self { ring, chunk_size }
    }

    pub fn empty(chunk_size: ChunkSize) -> Self {
        Self {
            ring: Vec::new(),
            chunk_size,
        }
    }

    /// Point-in-polygon test (XZ plane, ray casting).
    pub fn contains_xz(&self, p: WorldPos) -> bool {
        let n = self.ring.len();
        if n < 3 {
            return false;
        }

        // Test relative to point for precision
        let mut inside = false;
        let mut j = n - 1;

        for i in 0..n {
            let vi = self.ring[i].to_render_pos(p, self.chunk_size);
            let vj = self.ring[j].to_render_pos(p, self.chunk_size);

            // Point is at origin in relative coords
            if (vi.z > 0.0) != (vj.z > 0.0) {
                let x_cross = vi.x + (0.0 - vi.z) / (vj.z - vi.z) * (vj.x - vi.x);
                if x_cross > 0.0 {
                    inside = !inside;
                }
            }
            j = i;
        }
        inside
    }

    /// Find where line segment (from -> to) first intersects polygon boundary.
    /// Returns the intersection point as WorldPos.
    pub fn clip_to_edge(&self, from: WorldPos, to: WorldPos) -> WorldPos {
        let n = self.ring.len();
        if n < 3 {
            return to;
        }

        let mut best_t = 1.0f32;

        // Compute segment direction relative to 'from'
        let d1 = to.to_render_pos(from, self.chunk_size);
        let d1_2d = Vec2::new(d1.x, d1.z);

        for i in 0..n {
            let a = self.ring[i];
            let b = self.ring[(i + 1) % n];

            // Edge relative to 'from'
            let a_rel = a.to_render_pos(from, self.chunk_size);
            let b_rel = b.to_render_pos(from, self.chunk_size);

            let d2 = Vec2::new(b_rel.x - a_rel.x, b_rel.z - a_rel.z);
            let denom = d1_2d.x * d2.y - d1_2d.y * d2.x;

            if denom.abs() < 1e-10 {
                continue;
            }

            let d3 = Vec2::new(a_rel.x, a_rel.z);
            let t = (d3.x * d2.y - d3.y * d2.x) / denom;
            let u = (d3.x * d1_2d.y - d3.y * d1_2d.x) / denom;

            if t > 0.0 && t < best_t && u >= 0.0 && u <= 1.0 {
                best_t = t;
            }
        }

        from.lerp(to, best_t, self.chunk_size)
    }

    /// Compute polygon centroid.
    pub fn centroid(&self) -> WorldPos {
        if self.ring.is_empty() {
            return WorldPos::zero();
        }

        let reference = self.ring[0];
        let mut sum = Vec3::ZERO;

        for v in &self.ring {
            sum += v.to_render_pos(reference, self.chunk_size);
        }

        let avg = sum / self.ring.len() as f32;
        reference.add_vec3(avg, self.chunk_size)
    }

    /// Compute polygon area (XZ plane, signed).
    pub fn signed_area_xz(&self) -> f32 {
        let n = self.ring.len();
        if n < 3 {
            return 0.0;
        }

        let reference = self.ring[0];
        let mut area = 0.0f32;

        for i in 0..n {
            let a = self.ring[i].to_render_pos(reference, self.chunk_size);
            let b = self.ring[(i + 1) % n].to_render_pos(reference, self.chunk_size);
            area += a.x * b.z - b.x * a.z;
        }

        area * 0.5
    }

    /// Check if polygon is CCW wound.
    pub fn is_ccw(&self) -> bool {
        self.signed_area_xz() > 0.0
    }
}
/// Helper: clip or extend a point to the polygon boundary using ray projection
pub fn clip_point_to_polygon(
    point: WorldPos,
    lane_dir: Vec3, // Direction along lane AWAY from the intersection
    poly: &IntersectionPolygon,
    proj_dist: f32,
) -> WorldPos {
    if poly.contains_xz(point) {
        // Point is inside polygon - project outward along lane to find exit point
        let far_point = point.add_vec3(lane_dir * proj_dist, poly.chunk_size);
        poly.clip_to_edge(far_point, point)
    } else {
        // Point is outside polygon - check if we need to EXTEND toward intersection
        let toward_intersection = point.sub_vec3(lane_dir * proj_dist, poly.chunk_size);
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
    let center = node.position(); //This is already WorldPos!!
    let node_lanes = node.node_lanes();

    if node_lanes.len() < 2 {
        return IntersectionMeshResult::default();
    }

    let outer_lanes = select_outermost_lanes(node_lanes, storage, terrain.chunk_size);

    // Build polylines AND track segment ownership
    let mut asphalt_polylines = Vec::new();
    let mut polyline_segment_info: Vec<(SegmentId, i8)> = Vec::new(); // (segment_id, outward_sign)

    for lane in &outer_lanes {
        let offset = style.lane_width * 0.5 * lane.outward_sign as f32;
        let node_lane = node.node_lane(lane.node_lane);

        gizmo.polyline(node_lane.polyline(), [1.0, 0.0, 1.0], 4.0, 10.0);

        let poly = offset_polyline_f32(node_lane.polyline(), offset, terrain.chunk_size);
        asphalt_polylines.push(poly);
        polyline_segment_info.push((lane.segment_id, lane.outward_sign));
    }

    let mut asphalt_ring = merge_polylines_ccw(center, asphalt_polylines, terrain.chunk_size);

    if asphalt_ring.len() < 3 {
        return IntersectionMeshResult::default();
    }

    // === Build mesh vertices ===
    // base index of the center vertex
    let asphalt_base = vertices.len() as u32;

    // center vertex (UV at texture center)
    let mut point = node.position();
    set_point_height_with_structure_type(terrain, style.road_type().structure(), &mut point);
    point.local.y += style.lane_height;
    vertices.push(road_vertex(
        point,
        [0.0, 1.0, 0.0],
        style.lane_material_id,
        0.5, // u
        0.5, // v
    ));

    // radial uv mapping using separate U/V scales
    let radial_uv = |pt: WorldPos| -> (f32, f32) {
        let delta = center.delta_to(pt, terrain.chunk_size);
        let dist = delta.length();

        if dist <= 1e-6 {
            return (0.5, 0.5);
        }

        let nx = delta.x / dist;
        let nz = delta.z / dist;

        let ru = dist * config.uv_scale_u;
        let rv = dist * config.uv_scale_v;

        let u = 0.5 + nx * ru;
        let v = 0.5 + nz * rv;

        (u, v)
    };

    // push ring in the same order triangulator expects (CCW)
    for p in asphalt_ring.iter() {
        let mut point = p.clone();
        set_point_height_with_structure_type(terrain, style.road_type().structure(), &mut point);
        point.local.y += style.lane_height;
        let (u, v) = radial_uv(*p);
        vertices.push(road_vertex(
            *p,
            [0.0, 1.0, 0.0],
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
                style,
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
                style,
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
                style,
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
        polygon: IntersectionPolygon {
            ring: asphalt_ring,
            chunk_size: terrain.chunk_size,
        },
    }
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
