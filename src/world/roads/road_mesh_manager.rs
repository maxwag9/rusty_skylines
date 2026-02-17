//# road_mesh_manager.rs
//! Road Mesh Manager for procedural lane-first citybuilder.
//!
//! Produces deterministic, chunked CPU mesh buffers from immutable road topology.
//! Refactored to be Lane-First: Geometry is derived directly from Lane centerlines.

use crate::helpers::positions::{ChunkCoord, ChunkSize, WorldPos};
use crate::renderer::gizmo::Gizmo;
use crate::world::roads::intersections::{
    IntersectionMeshResult, IntersectionPolygon, build_intersection_mesh, road_vertex,
};
use crate::world::roads::road_helpers::*;
use crate::world::roads::road_structs::*;
use crate::world::roads::roads::{LaneGeometry, Node, RoadStorage, Segment};
use crate::world::terrain::terrain_subsystem::TerrainSubsystem;
use glam::Vec3;
use std::collections::HashMap;
use std::f32::consts::{FRAC_PI_2, TAU};
use wgpu::{VertexAttribute, VertexFormat};

pub type ChunkId = u64;

// ============================================================================
// Constants & Configuration
// ============================================================================

pub const CLEARANCE: f32 = 0.08;
const NODE_ANGULAR_SEGMENTS: usize = 32;

// ============================================================================
// Chunking & ID Logic (Preserved)
// ============================================================================

#[inline]
fn zigzag_i32(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}
#[inline]
fn part1by1(n: u32) -> u64 {
    let mut x = n as u64;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x << 2)) & 0x3333333333333333;
    x = (x | (x << 1)) & 0x5555555555555555;
    x
}
#[inline(always)]
pub fn chunk_coord_to_id(cx: i32, cz: i32) -> ChunkId {
    part1by1(zigzag_i32(cx)) | (part1by1(zigzag_i32(cz)) << 1)
}
#[inline(always)]
pub fn world_pos_chunk_to_id(world_pos: WorldPos) -> ChunkId {
    part1by1(zigzag_i32(world_pos.chunk.x)) | (part1by1(zigzag_i32(world_pos.chunk.z)) << 1)
}
#[inline]
fn unzigzag_u32(v: u32) -> i32 {
    ((v >> 1) as i32) ^ -((v & 1) as i32)
}
#[inline]
fn compact1by1(x: u64) -> u32 {
    let mut x = x & 0x5555555555555555;
    x = (x | (x >> 1)) & 0x3333333333333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF0000FFFF;
    x = (x | (x >> 16)) & 0x00000000FFFFFFFF;
    x as u32
}
#[inline]
pub fn chunk_id_to_coord(id: ChunkId) -> ChunkCoord {
    ChunkCoord::new(
        unzigzag_u32(compact1by1(id)),
        unzigzag_u32(compact1by1(id >> 1)),
    )
}

// ============================================================================
// Vertex Format
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct RoadVertex {
    pub chunk_xz: [i32; 2],
    pub local_position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub material_id: u32,
}

impl RoadVertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<RoadVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // loc0 chunk_xz
                VertexAttribute {
                    shader_location: 0,
                    offset: 0,
                    format: VertexFormat::Sint32x2,
                },
                // @location(1) chunk-local position
                VertexAttribute {
                    shader_location: 1,
                    offset: 8,
                    format: VertexFormat::Float32x3,
                },
                // @location(2) normals
                VertexAttribute {
                    offset: 20,
                    shader_location: 2,
                    format: VertexFormat::Float32x3,
                },
                // @location(3) uv
                VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: VertexFormat::Float32x2,
                },
                // @location(4) material_id
                VertexAttribute {
                    offset: 40,
                    shader_location: 4,
                    format: VertexFormat::Uint32,
                },
            ],
        }
    }
}

// ============================================================================
// Mesh Output
// ============================================================================

#[derive(Clone, Debug)]
pub struct ChunkMesh {
    pub vertices: Vec<RoadVertex>,
    pub indices: Vec<u32>,
    pub topo_version: u64,
}

impl ChunkMesh {
    #![allow(dead_code)]
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            topo_version: 0,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct MeshConfig {
    pub uv_scale_u: f32,
    pub uv_scale_v: f32,
}

impl Default for MeshConfig {
    fn default() -> Self {
        Self {
            uv_scale_u: 1.0,
            uv_scale_v: 1.0,
        }
    }
}

// ============================================================================
// Mesh Builders (Lane First)
// ============================================================================

/// Mesh a segment, clipping endpoints against intersection polygons
fn mesh_segment_with_boundaries(
    terrain_renderer: &TerrainSubsystem,
    gizmo: &mut Gizmo,
    _seg_id: SegmentId,
    segment: &Segment,
    storage: &RoadStorage,
    style: &RoadStyleParams,
    config: &MeshConfig,
    chunk_filter: Option<ChunkId>,
    start_result: Option<&IntersectionMeshResult>,
    end_result: Option<&IntersectionMeshResult>,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    let lane_ids = segment.lanes();
    if lane_ids.is_empty() {
        return;
    }
    let road_type = style.road_type();
    // Extract clip polygons
    let start_clip = start_result
        .map(|r| &r.polygon)
        .filter(|p| p.ring.len() >= 3);
    let end_clip = end_result.map(|r| &r.polygon).filter(|p| p.ring.len() >= 3);
    if let Some(start_clip) = start_clip {
        //gizmo.polyline(&start_clip.ring, [0.2, 0.1, 0.8], 10.0, 25.0);
        for p in start_clip.ring.iter() {
            //gizmo.cross(*p, 0.3, [0.2, 0.1, 0.8], 25.0);
        }
    }
    if let Some(end_clip) = end_clip {
        //gizmo.polyline(&end_clip.ring, [0.8, 0.1, 0.2], 5.0, 25.0);
        for p in end_clip.ring.iter() {
            //gizmo.cross(*p, 0.2, [0.8, 0.1, 0.2], 25.0);
        }
    }
    let mut lane_data: Vec<(i8, LaneId)> = Vec::new();
    let mut min_lane: Option<(i8, LaneId)> = None;
    let mut max_lane: Option<(i8, LaneId)> = None;

    for &lane_id in segment.lanes() {
        let lane = storage.lane(&lane_id);
        if lane.is_disabled() {
            continue;
        }

        let lane_idx = lane.lane_index();

        lane_data.push((lane_idx, lane_id));

        min_lane = Some(min_lane.map_or((lane_idx, lane_id), |m| {
            if lane_idx < m.0 {
                (lane_idx, lane_id)
            } else {
                m
            }
        }));
        max_lane = Some(max_lane.map_or((lane_idx, lane_id), |m| {
            if lane_idx > m.0 {
                (lane_idx, lane_id)
            } else {
                m
            }
        }));
    }

    // 1. Draw lane surfaces
    for (_idx, lane_id) in &lane_data {
        let geom = storage.lane(&lane_id).geometry();
        build_ribbon_mesh(
            terrain_renderer,
            gizmo,
            style,
            geom,
            road_type.lane_width,
            road_type.lane_height,
            0.0,
            road_type.lane_material_id,
            chunk_filter,
            (config.uv_scale_u, config.uv_scale_v),
            start_clip,
            end_clip,
            vertices,
            indices,
        );
    }

    // 2. Draw sidewalks on outer edges
    if let Some((_idx, lane_id)) = min_lane {
        let geom = storage.lane(&lane_id).geometry();
        let offset = road_type.lane_width * 0.5 + road_type.sidewalk_width * 0.5;
        // gizmo.polyline(geom.points.as_slice(), [0.2, 1.0, 0.0], 10.0, 20.0);
        build_ribbon_mesh(
            terrain_renderer,
            gizmo,
            style,
            geom,
            road_type.sidewalk_width,
            road_type.sidewalk_height,
            offset,
            road_type.sidewalk_material_id,
            chunk_filter,
            (config.uv_scale_u, config.uv_scale_v),
            start_clip,
            end_clip,
            vertices,
            indices,
        );

        let inner_offset = road_type.lane_width * -0.5;
        build_vertical_face(
            terrain_renderer,
            style,
            geom,
            inner_offset,
            road_type.lane_height,
            road_type.sidewalk_height,
            0,
            chunk_filter,
            (config.uv_scale_u, config.uv_scale_v),
            Some(-1f32),
            None,
            None,
            vertices,
            indices,
        );

        let outer_offset = road_type.lane_width * -0.5 - road_type.sidewalk_width;
        build_vertical_face(
            terrain_renderer,
            style,
            geom,
            outer_offset,
            road_type.lane_height,
            road_type.sidewalk_height,
            0,
            chunk_filter,
            (config.uv_scale_u, config.uv_scale_v),
            Some(1f32),
            None,
            None,
            vertices,
            indices,
        );
    }

    if let Some((_idx, lane_id)) = max_lane {
        let geom = storage.lane(&lane_id).geometry();
        let offset = road_type.lane_width * 0.5 + road_type.sidewalk_width * 0.5;

        build_ribbon_mesh(
            terrain_renderer,
            gizmo,
            style,
            geom,
            road_type.sidewalk_width,
            road_type.sidewalk_height,
            offset,
            road_type.sidewalk_material_id,
            chunk_filter,
            (config.uv_scale_u, config.uv_scale_v),
            start_clip,
            end_clip,
            vertices,
            indices,
        );

        let inner_offset = road_type.lane_width * -0.5;
        build_vertical_face(
            terrain_renderer,
            style,
            geom,
            inner_offset,
            road_type.lane_height,
            road_type.sidewalk_height,
            road_type.sidewalk_material_id,
            chunk_filter,
            (config.uv_scale_u, config.uv_scale_v),
            Some(-1f32),
            None,
            None,
            vertices,
            indices,
        );

        let outer_offset = road_type.lane_width * -0.5 - road_type.sidewalk_width;
        build_vertical_face(
            terrain_renderer,
            style,
            geom,
            outer_offset,
            road_type.lane_height,
            road_type.sidewalk_height,
            road_type.sidewalk_material_id,
            chunk_filter,
            (config.uv_scale_u, config.uv_scale_v),
            Some(1f32),
            None,
            None,
            vertices,
            indices,
        );
    }

    // 3. Draw median if needed
    if min_lane.is_some() && max_lane.is_some() && road_type.median_width > 0.1 {
        if let Some((_lane_idx, lane_id)) = lane_data.iter().find(|(i, _)| *i == 1) {
            let geom = storage.lane(&lane_id).geometry();
            let offset = -road_type.lane_width * 0.5;

            build_ribbon_mesh(
                terrain_renderer,
                gizmo,
                style,
                geom,
                road_type.median_width,
                road_type.median_height,
                offset,
                road_type.median_material_id,
                chunk_filter,
                (config.uv_scale_u, config.uv_scale_v),
                start_clip,
                end_clip,
                vertices,
                indices,
            );

            let curb_offset_right = road_type.lane_width * 0.5 + road_type.median_width * 0.5;
            build_vertical_face(
                terrain_renderer,
                style,
                geom,
                curb_offset_right,
                road_type.lane_height,
                road_type.median_height,
                0,
                chunk_filter,
                (config.uv_scale_u, config.uv_scale_v),
                Some(-1.0),
                None,
                None,
                vertices,
                indices,
            );

            let curb_offset_left = road_type.lane_width * 0.5 - road_type.median_width * 0.5;
            build_vertical_face(
                terrain_renderer,
                style,
                geom,
                curb_offset_left,
                road_type.lane_height,
                road_type.median_height,
                0,
                chunk_filter,
                (config.uv_scale_u, config.uv_scale_v),
                Some(1.0),
                None,
                None,
                vertices,
                indices,
            );
        }
    }
}

fn draw_node_geometry(
    terrain_renderer: &TerrainSubsystem,
    gizmo: &mut Gizmo,
    node_pos: WorldPos,
    connected_lanes_info: &[(i8, f32)],
    cap_direction: Option<Vec3>,
    style: &RoadStyleParams,
    config: &MeshConfig,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    let chunk_size = terrain_renderer.chunk_size;
    let road_type = style.road_type();

    let mut max_radius = 2.0_f32;
    for (idx, width) in connected_lanes_info {
        max_radius = max_radius.max(idx.abs() as f32 * width + road_type.sidewalk_width);
    }

    let road_radius = max_radius - road_type.sidewalk_width;
    let sw_inner = road_radius;
    let sw_outer = max_radius;

    let (start_angle, end_angle) = if let Some(dir) = cap_direction {
        let forward = dir.normalize();
        let heading = forward.z.atan2(forward.x);
        (heading - FRAC_PI_2, heading + FRAC_PI_2)
    } else {
        (0.0_f32, TAU)
    };

    let segments = NODE_ANGULAR_SEGMENTS;

    let create_circular_geom = |radius: f32| -> LaneGeometry {
        let angle_span = end_angle - start_angle;
        let step = angle_span / segments as f32;

        let mut points = Vec::with_capacity(segments + 1);
        for i in 0..=segments {
            let angle = start_angle + i as f32 * step;
            let (sin_a, cos_a) = angle.sin_cos();
            let offset = Vec3::new(cos_a * radius, 0.0, sin_a * radius);
            points.push(node_pos.add_vec3(offset, chunk_size));
        }

        LaneGeometry::from_polyline(points, chunk_size)
    };

    let up_normal = [0.0_f32, 1.0, 0.0];

    let mut center_p = node_pos;
    set_point_height_with_structure_type(
        terrain_renderer,
        road_type.structure(),
        &mut center_p,
        true,
    );
    center_p.local.y += road_type.lane_height;
    let center_idx = vertices.len() as u32;

    vertices.push(road_vertex(
        center_p,
        up_normal,
        road_type.lane_material_id,
        0.5,
        0.5,
    ));

    let road_geom = create_circular_geom(road_radius);
    let road_ring_first = vertices.len() as u32;

    for i in 0..road_geom.points.len() {
        let mut outer_p = road_geom.points[i];
        set_point_height_with_structure_type(
            terrain_renderer,
            road_type.structure(),
            &mut outer_p,
            true,
        );
        outer_p.local.y += road_type.lane_height;

        let t = i as f32 / segments as f32;
        let angle = start_angle + t * (end_angle - start_angle);
        let (sin_a, cos_a) = angle.sin_cos();
        let uv_scale = road_radius / config.uv_scale_v;
        let u = 0.5 + cos_a * uv_scale;
        let v = 0.5 + sin_a * uv_scale;

        vertices.push(road_vertex(
            outer_p,
            up_normal,
            road_type.lane_material_id,
            u,
            v,
        ));
    }

    for i in 0..segments {
        let curr = road_ring_first + i as u32;
        let next = road_ring_first + (i + 1) as u32;
        emit_tri_for_top(indices, vertices, center_idx, curr, next);
    }

    let sidewalk_mid_radius = (sw_inner + sw_outer) / 2.0;
    let sidewalk_width = sw_outer - sw_inner;
    let sidewalk_geom = create_circular_geom(sidewalk_mid_radius);

    build_ribbon_mesh(
        terrain_renderer,
        gizmo,
        style,
        &sidewalk_geom,
        sidewalk_width,
        road_type.sidewalk_height,
        0.0,
        road_type.sidewalk_material_id,
        None,
        (1.0 / config.uv_scale_u, 1.0 / config.uv_scale_v),
        None,
        None,
        vertices,
        indices,
    );

    let inner_curb_geom = create_circular_geom(sw_inner);

    build_vertical_face(
        terrain_renderer,
        style,
        &inner_curb_geom,
        0.0,
        road_type.lane_height,
        road_type.sidewalk_height,
        road_type.sidewalk_material_id,
        None,
        (1.0 / config.uv_scale_u, 1.0 / config.uv_scale_v),
        Some(1.0),
        None,
        None,
        vertices,
        indices,
    );

    let outer_curb_geom = create_circular_geom(sw_outer);

    build_vertical_face(
        terrain_renderer,
        style,
        &outer_curb_geom,
        0.0,
        road_type.lane_height,
        road_type.sidewalk_height,
        road_type.sidewalk_material_id,
        None,
        (1.0 / config.uv_scale_u, 1.0 / config.uv_scale_v),
        Some(-1.0),
        None,
        None,
        vertices,
        indices,
    );
}

/// Builds a ribbon mesh for a single lane or strip.
pub fn build_ribbon_mesh(
    terrain_renderer: &TerrainSubsystem,
    gizmo: &mut Gizmo,
    style: &RoadStyleParams,
    lane_geom: &LaneGeometry,
    width: f32,
    height: f32,
    offset_from_center: f32,
    material_id: u32,
    chunk_filter: Option<ChunkId>,
    uv_config: (f32, f32),
    start_clip: Option<&IntersectionPolygon>,
    end_clip: Option<&IntersectionPolygon>,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    if lane_geom.points.len() < 2 {
        return;
    }
    let high_res_height = chunk_filter.is_some();
    let chunk_size = terrain_renderer.chunk_size;
    let half_width = width * 0.5;
    let base_vertex = vertices.len() as u32;

    // === Pre-calculate all edge positions in XZ ===
    let mut edges: Vec<(WorldPos, WorldPos)> = Vec::with_capacity(lane_geom.points.len());

    for i in 0..lane_geom.points.len() {
        let p = lane_geom.points[i];
        let (_, lateral) = tangent_and_lateral_right(&lane_geom.points, i, chunk_size);

        let center_pos = p.sub_vec3(lateral * offset_from_center, chunk_size);
        let left_pos = center_pos.sub_vec3(lateral * half_width, chunk_size);
        let right_pos = center_pos.add_vec3(lateral * half_width, chunk_size);

        edges.push((left_pos, right_pos));
    }

    if let Some(poly) = start_clip {
        clip_ribbon_edges_to_polygon(&mut edges, poly, true, gizmo);
    }

    if let Some(poly) = end_clip {
        clip_ribbon_edges_to_polygon(&mut edges, poly, false, gizmo);
    }

    // === Chunk filtering (unchanged) ===
    let mut included_indices = Vec::new();

    match chunk_filter {
        Some(cid) => {
            for i in 0..lane_geom.points.len() {
                let p = lane_geom.points[i];
                let in_chunk = world_pos_chunk_to_id(p) == cid;

                let prev_in = i > 0 && {
                    let pp = lane_geom.points[i - 1];
                    world_pos_chunk_to_id(pp) == cid
                };

                if in_chunk || prev_in {
                    included_indices.push(i);
                } else if i + 1 < lane_geom.points.len() {
                    let pn = lane_geom.points[i + 1];
                    let next_in = world_pos_chunk_to_id(pn) == cid;
                    if next_in {
                        included_indices.push(i);
                    }
                }
            }
        }
        None => {
            included_indices.extend(0..lane_geom.points.len());
        }
    }

    included_indices.sort_unstable();
    included_indices.dedup();

    if included_indices.len() < 2 {
        return;
    }

    // === Emit vertices using pre-clipped edges ===
    let mut point_idx_to_vert_idx = HashMap::new();
    let mut current_vert_idx = 0;

    for &i in &included_indices {
        let (mut left_pos, mut right_pos) = edges[i];

        set_point_height_with_structure_type(
            terrain_renderer,
            style.road_type().structure(),
            &mut left_pos,
            high_res_height,
        );
        set_point_height_with_structure_type(
            terrain_renderer,
            style.road_type().structure(),
            &mut right_pos,
            high_res_height,
        );

        left_pos.local.y += height;
        right_pos.local.y += height;

        // UV calculation: u based on arc length, v spans the width
        let u = lane_geom.lengths[i] * uv_config.0;
        let v_min = 0.0;
        let v_max = width * uv_config.1;

        vertices.push(road_vertex(
            left_pos,
            [0.0, 1.0, 0.0],
            material_id,
            u,
            v_min,
        ));

        vertices.push(road_vertex(
            right_pos,
            [0.0, 1.0, 0.0],
            material_id,
            u,
            v_max,
        ));

        point_idx_to_vert_idx.insert(i, current_vert_idx);
        current_vert_idx += 2;
    }

    // Generate indices
    for w in included_indices.windows(2) {
        let a = w[0];
        let b = w[1];

        let v_base = base_vertex + point_idx_to_vert_idx[&a];
        let v_next = base_vertex + point_idx_to_vert_idx[&b];

        indices.push(v_base + 1);
        indices.push(v_base);
        indices.push(v_next);

        indices.push(v_next + 1);
        indices.push(v_base + 1);
        indices.push(v_next);
    }
}

/// Builds vertical faces between two parallel strips (e.g., Curb).
/// Updated to accept start/end overrides to snap to intersection boundaries.
pub fn build_vertical_face(
    terrain_renderer: &TerrainSubsystem,
    style: &RoadStyleParams,
    ref_geom: &LaneGeometry,
    offset_lateral: f32,
    bottom_height: f32,
    top_height: f32,
    material_id: u32,
    chunk_filter: Option<ChunkId>,
    uv_config: (f32, f32),
    explicit_normal_sign: Option<f32>,
    start_override: Option<WorldPos>,
    end_override: Option<WorldPos>,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    if (top_height - bottom_height).abs() < 0.001 {
        return;
    }
    if ref_geom.points.len() < 2 {
        return;
    }
    let high_res_height = chunk_filter.is_some();
    let chunk_size = terrain_renderer.chunk_size;
    let base_vertex = vertices.len() as u32;
    let mut included_indices = Vec::new();
    let last_idx = ref_geom.points.len() - 1;

    match chunk_filter {
        Some(cid) => {
            for i in 0..ref_geom.points.len() {
                let p = ref_geom.points[i];
                if world_pos_chunk_to_id(p) == cid {
                    included_indices.push(i);
                } else if i + 1 < ref_geom.points.len() {
                    let pn = ref_geom.points[i + 1];
                    if world_pos_chunk_to_id(pn) == cid {
                        included_indices.push(i);
                    }
                }
            }
        }
        None => included_indices.extend(0..ref_geom.points.len()),
    }
    included_indices.sort_unstable();
    included_indices.dedup();

    if included_indices.len() < 2 {
        return;
    }

    let mut point_idx_to_vert_idx = HashMap::new();
    let mut current_vert_idx = 0;

    let normal_sign =
        explicit_normal_sign.unwrap_or_else(|| if offset_lateral > 0.0 { -1.0 } else { 1.0 });

    for &i in &included_indices {
        let p = ref_geom.points[i];

        // Check for override at endpoints
        let override_pos = if i == 0 {
            start_override
        } else if i == last_idx {
            end_override
        } else {
            None
        };

        let (face_pos, normal) = if let Some(ovr) = override_pos {
            let (_, lateral) = tangent_and_lateral_right(&ref_geom.points, i, chunk_size);
            (ovr, lateral * normal_sign)
        } else {
            let (_, lateral) = tangent_and_lateral_right(&ref_geom.points, i, chunk_size);
            let raw = p.add_vec3(lateral * offset_lateral, chunk_size);
            (raw, lateral * normal_sign)
        };

        let mut p_bottom = face_pos;
        set_point_height_with_structure_type(
            terrain_renderer,
            style.road_type().structure(),
            &mut p_bottom,
            high_res_height,
        );
        p_bottom.local.y += bottom_height;
        let mut p_top = face_pos;
        set_point_height_with_structure_type(
            terrain_renderer,
            style.road_type().structure(),
            &mut p_top,
            high_res_height,
        );
        p_top.local.y += top_height;
        let u = ref_geom.lengths[i] * uv_config.0;
        let v_h = (top_height - bottom_height).abs() * uv_config.1;

        // Bottom vertex
        vertices.push(road_vertex(
            p_bottom,
            normal.to_array(),
            material_id,
            u,
            0.0,
        ));

        // Top vertex
        vertices.push(road_vertex(p_top, normal.to_array(), material_id, u, v_h));

        point_idx_to_vert_idx.insert(i, current_vert_idx);
        current_vert_idx += 2;
    }

    // Index generation (same as before)
    for k in 0..included_indices.len() - 1 {
        let idx_curr = included_indices[k];
        let idx_next = included_indices[k + 1];
        if idx_next != idx_curr + 1 {
            continue;
        }

        let v_base = base_vertex + point_idx_to_vert_idx[&idx_curr];
        let v_next = base_vertex + point_idx_to_vert_idx[&idx_next];

        if normal_sign > 0.0 {
            indices.push(v_base);
            indices.push(v_next);
            indices.push(v_base + 1);
            indices.push(v_base + 1);
            indices.push(v_next);
            indices.push(v_next + 1);
        } else {
            indices.push(v_base);
            indices.push(v_base + 1);
            indices.push(v_next);
            indices.push(v_base + 1);
            indices.push(v_next + 1);
            indices.push(v_next);
        }
    }
}

// ============================================================================
// Node Meshing (Simplified Radial)
// ============================================================================

/// Shared logic to draw a node (intersection/end cap) with sidewalks and curbs.

// compute triangle normal (not normalized)
fn tri_normal(v0: Vec3, v1: Vec3, v2: Vec3) -> Vec3 {
    (v1 - v0).cross(v2 - v0)
}

// push triangle ensuring top-facing normals point up (normal.y > 0)
// triangles with normal.y < 0 will be flipped (swap i1 <-> i2)
fn emit_tri_for_top(indices: &mut Vec<u32>, vertices: &Vec<RoadVertex>, i0: u32, i1: u32, i2: u32) {
    let v0 = Vec3::from(vertices[i0 as usize].local_position);
    let v1 = Vec3::from(vertices[i1 as usize].local_position);
    let v2 = Vec3::from(vertices[i2 as usize].local_position);
    let n = tri_normal(v0, v1, v2);
    let ok = n.y >= 0.0;
    if ok {
        indices.push(i0);
        indices.push(i1);
        indices.push(i2);
    } else {
        indices.push(i2);
        indices.push(i1);
        indices.push(i0);
    }
}

// push triangle ensuring curb-face normals point outward from node_pos
// we check dot(normal, centroid_dir) >= 0 (normal points roughly away from center), flip if not
fn emit_tri_for_curb(
    indices: &mut Vec<u32>,
    vertices: &Vec<RoadVertex>,
    i0: u32,
    i1: u32,
    i2: u32,
    node_pos: WorldPos,
    chunk_size: ChunkSize,
) {
    let v0 = Vec3::from(vertices[i0 as usize].local_position);
    let v1 = Vec3::from(vertices[i1 as usize].local_position);
    let v2 = Vec3::from(vertices[i2 as usize].local_position);

    let n = tri_normal(v0, v1, v2);
    let centroid = (v0 + v1 + v2) / 3.0;

    let node_render = node_pos.to_render_pos(WorldPos::zero(), chunk_size);
    let centroid_dir = (centroid - node_render).normalize_or_zero();

    if n.dot(centroid_dir) >= 0.0 {
        indices.push(i0);
        indices.push(i1);
        indices.push(i2);
    } else {
        indices.push(i0);
        indices.push(i2);
        indices.push(i1);
    }
}

// Add this helper for inner curb (normal points INWARD toward center)
fn emit_tri_for_inner_curb(
    indices: &mut Vec<u32>,
    vertices: &Vec<RoadVertex>,
    i0: u32,
    i1: u32,
    i2: u32,
    node_pos: WorldPos,
    chunk_size: ChunkSize,
) {
    let v0 = Vec3::from(vertices[i0 as usize].local_position);
    let v1 = Vec3::from(vertices[i1 as usize].local_position);
    let v2 = Vec3::from(vertices[i2 as usize].local_position);

    let n = tri_normal(v0, v1, v2);

    let centroid = (v0 + v1 + v2) / 3.0;

    // Compute centroid_dir in render space relative to node_pos
    let node_render = node_pos.to_render_pos(WorldPos::zero(), chunk_size);
    let centroid_dir = (centroid - node_render).normalize_or_zero();

    // Inner curb: normal should point TOWARD center (dot < 0)
    if n.dot(centroid_dir) <= 0.0 {
        indices.push(i0);
        indices.push(i1);
        indices.push(i2);
    } else {
        indices.push(i0);
        indices.push(i2);
        indices.push(i1);
    }
}

// ============================================================================
// Road Mesh Manager (Refactored)
// ============================================================================

pub struct RoadMeshManager {
    chunk_cache: HashMap<ChunkId, ChunkMesh>,
    pub config: MeshConfig,
}

impl RoadMeshManager {
    pub fn new(config: MeshConfig) -> Self {
        Self {
            chunk_cache: HashMap::new(),
            config,
        }
    }

    pub fn get_chunk_mesh(&self, chunk_id: ChunkId) -> Option<&ChunkMesh> {
        self.chunk_cache.get(&chunk_id)
    }

    pub fn _invalidate_chunk(&mut self, chunk_id: ChunkId) {
        self.chunk_cache.remove(&chunk_id);
    }

    pub fn _clear_cache(&mut self) {
        self.chunk_cache.clear();
    }

    pub fn chunk_needs_update(
        &self,
        chunk_id: ChunkId,
        storage: &RoadStorage,
        chunk_size: ChunkSize,
    ) -> bool {
        match self.chunk_cache.get(&chunk_id) {
            None => true,
            Some(mesh) => mesh.topo_version != compute_topo_version(chunk_id, storage, chunk_size),
        }
    }

    /// Build mesh for a chunk (Some) or all geometry (None for previews)
    pub fn build_mesh(
        &self,
        terrain: &TerrainSubsystem,
        chunk_id: Option<ChunkId>,
        storage: &RoadStorage,
        style: &RoadStyleParams,
        gizmo: &mut Gizmo,
    ) -> ChunkMesh {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Store intersection results for segment meshing
        let mut intersection_results: HashMap<NodeId, IntersectionMeshResult> = HashMap::new();

        let node_ids: Vec<NodeId> = match chunk_id {
            Some(cid) => storage.nodes_in_chunk(cid),
            None => storage.get_active_node_ids(),
        };

        // === PASS 1: Build intersection meshes and collect boundary data ===
        for node_id in &node_ids {
            let node = match storage.node(*node_id) {
                Some(n) => n,
                None => continue,
            };

            let segment_count = storage.enabled_segment_count_connected_to_node(*node_id);

            if segment_count >= 2 {
                let result = build_intersection_mesh(
                    terrain,
                    *node_id,
                    node,
                    storage,
                    style,
                    &self.config,
                    &mut vertices,
                    &mut indices,
                    gizmo,
                );
                intersection_results.insert(*node_id, result);
            } else {
                // Dead end or single connection
                let center = node.position();
                let mut connected_lanes_info = Vec::new();

                for lane_id in node
                    .incoming_lanes()
                    .iter()
                    .chain(node.outgoing_lanes().iter())
                {
                    let lane = storage.lane(lane_id);
                    connected_lanes_info.push((lane.lane_index(), style.road_type().lane_width));
                }

                if connected_lanes_info.is_empty() {
                    continue;
                }

                let cap_direction =
                    compute_cap_direction(gizmo, *node_id, node, storage, terrain.chunk_size);

                draw_node_geometry(
                    terrain,
                    gizmo,
                    center,
                    connected_lanes_info.as_slice(),
                    cap_direction,
                    style,
                    &self.config,
                    &mut vertices,
                    &mut indices,
                );
            }
        }

        // === PASS 2: Build segment meshes using intersection boundary data ===
        let segment_ids: Vec<SegmentId> = match chunk_id {
            Some(cid) => {
                let mut ids =
                    storage.segment_ids_touching_chunk(chunk_id_to_coord(cid), terrain.chunk_size);
                ids.sort_unstable();
                ids
            }
            None => storage.get_active_segment_ids(),
        };

        for seg_id in segment_ids {
            let segment = storage.segment(seg_id);
            if !segment.enabled {
                continue;
            }

            // Get boundary data from start and end nodes
            let start_boundary = intersection_results.get(&segment.start());
            let end_boundary = intersection_results.get(&segment.end());

            mesh_segment_with_boundaries(
                terrain,
                gizmo,
                seg_id,
                segment,
                storage,
                style,
                &self.config,
                chunk_id,
                start_boundary,
                end_boundary,
                &mut vertices,
                &mut indices,
            );
        }

        ChunkMesh {
            vertices,
            indices,
            topo_version: chunk_id
                .map(|cid| compute_topo_version(cid, storage, terrain.chunk_size))
                .unwrap_or(0),
        }
    }
    pub fn update_chunk_mesh(
        &mut self,
        terrain: &TerrainSubsystem,
        chunk_id: ChunkId,
        storage: &RoadStorage,
        style: &RoadStyleParams,
        gizmo: &mut Gizmo,
    ) -> &ChunkMesh {
        let mesh = self.build_mesh(terrain, Some(chunk_id), storage, style, gizmo);
        self.chunk_cache.insert(chunk_id, mesh);
        self.chunk_cache.get(&chunk_id).unwrap()
    }

    /// Convenience wrapper for building preview meshes (no chunking)
    pub fn build_preview_mesh(
        &self,
        terrain: &TerrainSubsystem,
        preview_storage: &RoadStorage,
        style: &RoadStyleParams,
        gizmo: &mut Gizmo,
    ) -> ChunkMesh {
        self.build_mesh(terrain, None, preview_storage, style, gizmo)
    }
}

fn compute_cap_direction(
    gizmo: &mut Gizmo,
    node_id: NodeId,
    node: &Node,
    storage: &RoadStorage,
    chunk_size: ChunkSize,
) -> Option<Vec3> {
    let mut sum_dir = Vec3::ZERO;
    let mut lane_count = 0u32;

    for lane_id in node
        .incoming_lanes()
        .iter()
        .chain(node.outgoing_lanes().iter())
    {
        let lane = storage.lane(lane_id);
        let pts = &lane.geometry().points;

        let dir = if lane.from_node() == node_id {
            pts[1].direction_to(pts[0], chunk_size)
        } else {
            pts[pts.len() - 2].direction_to(pts[pts.len() - 1], chunk_size)
        };

        let normalized = dir.normalize_or_zero();
        if normalized.length_squared() > 0.01 {
            sum_dir += normalized;
            lane_count += 1;
        }
    }

    if lane_count > 0 {
        let avg_dir = sum_dir / lane_count as f32;
        let alignment = avg_dir.length();

        if alignment > 0.7 {
            Some(avg_dir.normalize_or_zero())
        } else {
            None
        }
    } else {
        None
    }
}
// ============================================================================
// Topo Version Hashing
// ============================================================================

const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;

pub(crate) fn compute_topo_version(
    chunk_id: ChunkId,
    storage: &RoadStorage,
    chunk_size: ChunkSize,
) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    let mut segs = storage.segment_ids_touching_chunk(chunk_id_to_coord(chunk_id), chunk_size);
    segs.sort_unstable();
    for seg_id in segs {
        hash ^= seg_id.0 as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        let seg = storage.segment(seg_id);
        if seg.enabled {
            hash ^= 1;
        } else {
            hash ^= 0;
        }
        hash = hash.wrapping_mul(FNV_PRIME);
        let (l, r) = storage.lane_counts_for_segment(seg);
        hash ^= l as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= r as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    let mut nodes = storage.nodes_in_chunk(chunk_id);
    nodes.sort_unstable();
    for node_id in nodes {
        hash ^= node_id.0 as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
