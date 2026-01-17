//# road_mesh_manager.rs
//! Road Mesh Manager for procedural lane-first citybuilder.
//!
//! Produces deterministic, chunked CPU mesh buffers from immutable road topology.
//! Refactored to be Lane-First: Geometry is derived directly from Lane centerlines.

use crate::renderer::world_renderer::TerrainRenderer;
use crate::terrain::roads::road_preview::RoadPreviewState;
use crate::terrain::roads::roads::{LaneGeometry, RoadStorage, SegmentId};
use glam::Vec3;
use std::collections::HashMap;

pub type ChunkId = u64;

// ============================================================================
// Constants & Configuration
// ============================================================================

const METERS_PER_LANE_POLYLINE_STEP: f32 = 2.0;
pub const CLEARANCE: f32 = 0.04;
const NODE_ANGULAR_SEGMENTS: usize = 32;

#[derive(Clone, Debug)]
pub struct RoadStyleParams {
    pub lane_width: f32,
    pub lane_height: f32,
    pub lane_material_id: u32,

    pub sidewalk_width: f32,
    pub sidewalk_height: f32,
    pub sidewalk_material_id: u32,

    pub median_width: f32,
    pub median_height: f32,
    pub median_material_id: u32,
}

impl Default for RoadStyleParams {
    fn default() -> Self {
        Self {
            lane_width: 2.5,
            lane_height: 0.0,
            lane_material_id: 2, // Asphalt
            sidewalk_width: 1.25,
            sidewalk_height: 0.15,
            sidewalk_material_id: 0, // Concrete or Pavement
            median_width: 0.25,
            median_height: 0.15,
            median_material_id: 0, // Concrete
        }
    }
}

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
pub fn chunk_id_to_coord(id: ChunkId) -> (i32, i32) {
    (
        unzigzag_u32(compact1by1(id)),
        unzigzag_u32(compact1by1(id >> 1)),
    )
}
pub fn chunk_x_range(chunk_id: ChunkId) -> (f32, f32) {
    let (cx, _) = chunk_id_to_coord(chunk_id);
    let min_x = cx as f32 * 64.0;
    (min_x, min_x + 64.0)
}
pub fn chunk_z_range(chunk_id: ChunkId) -> (f32, f32) {
    let (_, cz) = chunk_id_to_coord(chunk_id);
    let min_z = cz as f32 * 64.0;
    (min_z, min_z + 64.0)
}

// ============================================================================
// Vertex Format
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct RoadVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub material_id: u32,
}

impl RoadVertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<RoadVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Uint32,
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

/// Shared logic to draw a full segment (lanes, sidewalks, medians)
/// `lanes`: List of (Lane Index, Geometry) pairs.
fn draw_segment_geometry(
    terrain_renderer: &TerrainRenderer,
    lanes: &[(i8, LaneGeometry)],
    style: &RoadStyleParams,
    config: &MeshConfig,
    chunk_filter: Option<ChunkId>,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    if lanes.is_empty() {
        return;
    }

    let mut min_lane_idx = 0;
    let mut max_lane_idx = 0;
    let mut has_left = false;
    let mut has_right = false;

    // 1. Draw Lanes
    for (idx, geom) in lanes {
        let i = *idx;
        if i < min_lane_idx {
            min_lane_idx = i;
        }
        if i > max_lane_idx {
            max_lane_idx = i;
        }
        if i < 0 {
            has_left = true;
        }
        if i > 0 {
            has_right = true;
        }

        build_ribbon_mesh(
            terrain_renderer,
            geom,
            style.lane_width,
            style.lane_height,
            0.0,
            style.lane_material_id,
            chunk_filter,
            (config.uv_scale_u, config.uv_scale_v),
            vertices,
            indices,
        );
    }

    // 2. Draw Sidewalks
    if has_left {
        if let Some((_, geom)) = lanes.iter().find(|(i, _)| *i == min_lane_idx) {
            let offset = style.lane_width * 0.5 + style.sidewalk_width * 0.5;
            build_ribbon_mesh(
                terrain_renderer,
                geom,
                style.sidewalk_width,
                style.sidewalk_height,
                offset,
                style.sidewalk_material_id,
                chunk_filter,
                (config.uv_scale_u, config.uv_scale_v),
                vertices,
                indices,
            );
            build_vertical_face(
                terrain_renderer,
                geom,
                style.lane_width * 0.5,
                style.lane_height,
                style.sidewalk_height,
                0,
                chunk_filter,
                (config.uv_scale_u, config.uv_scale_v),
                None,
                vertices,
                indices,
            );
        }
    }

    if has_right {
        if let Some((_, geom)) = lanes.iter().find(|(i, _)| *i == max_lane_idx) {
            let offset = style.lane_width * 0.5 + style.sidewalk_width * 0.5;
            build_ribbon_mesh(
                terrain_renderer,
                geom,
                style.sidewalk_width,
                style.sidewalk_height,
                offset,
                style.sidewalk_material_id,
                chunk_filter,
                (config.uv_scale_u, config.uv_scale_v),
                vertices,
                indices,
            );
            build_vertical_face(
                terrain_renderer,
                geom,
                style.lane_width * 0.5,
                style.lane_height,
                style.sidewalk_height,
                0,
                chunk_filter,
                (config.uv_scale_u, config.uv_scale_v),
                None,
                vertices,
                indices,
            );
        }
    }

    // 3. Draw Median
    if has_left && has_right && style.median_width > 0.1 {
        // Use Lane 1 as reference
        if let Some((_, geom)) = lanes.iter().find(|(i, _)| *i == 1) {
            let offset = -style.lane_width * 0.5;

            build_ribbon_mesh(
                terrain_renderer,
                geom,
                style.median_width,
                style.median_height,
                offset,
                style.median_material_id,
                chunk_filter,
                (config.uv_scale_u, config.uv_scale_v),
                vertices,
                indices,
            );

            // Right Curb (Facing Lane 1)
            // Points to Lane 1 (+Lateral relative to Lane 1)
            let curb_offset_right = -style.lane_width * 0.5 + style.median_width * 0.5;
            build_vertical_face(
                terrain_renderer,
                geom,
                curb_offset_right,
                style.lane_height,
                style.median_height,
                0,
                chunk_filter,
                (config.uv_scale_u, config.uv_scale_v),
                Some(1.0),
                vertices,
                indices,
            );

            // Left Curb (Facing Lane -1)
            // Points to Lane -1 (-Lateral relative to Lane 1)
            let curb_offset_left = -style.lane_width * 0.5 - style.median_width * 0.5;
            build_vertical_face(
                terrain_renderer,
                geom,
                curb_offset_left,
                style.lane_height,
                style.median_height,
                0,
                chunk_filter,
                (config.uv_scale_u, config.uv_scale_v),
                Some(-1.0),
                vertices,
                indices,
            );
        }
    }
}

/// Builds a ribbon mesh for a single lane or strip.
fn build_ribbon_mesh(
    terrain_renderer: &TerrainRenderer,
    lane_geom: &LaneGeometry,
    width: f32,
    height: f32,
    offset_from_center: f32,
    material_id: u32,
    chunk_filter: Option<ChunkId>,
    uv_config: (f32, f32), // scale_u, scale_v
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    if lane_geom.points.len() < 2 {
        return;
    }

    let half_width = width * 0.5;
    let base_vertex = vertices.len() as u32;

    let mut included_indices = Vec::new();

    match chunk_filter {
        Some(cid) => {
            let (min_x, max_x) = chunk_x_range(cid);
            let (min_z, max_z) = chunk_z_range(cid);

            for i in 0..lane_geom.points.len() {
                let p = lane_geom.points[i];
                let in_chunk = p.x >= min_x && p.x < max_x && p.z >= min_z && p.z < max_z;

                let prev_in = i > 0 && {
                    let pp = lane_geom.points[i - 1];
                    pp.x >= min_x && pp.x < max_x && pp.z >= min_z && pp.z < max_z
                };

                if in_chunk || prev_in {
                    included_indices.push(i);
                } else if i + 1 < lane_geom.points.len() {
                    let pn = lane_geom.points[i + 1];
                    let next_in = pn.x >= min_x && pn.x < max_x && pn.z >= min_z && pn.z < max_z;
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

    let mut point_idx_to_vert_idx = HashMap::new();
    let mut current_vert_idx = 0;

    for &i in &included_indices {
        let p = lane_geom.points[i];

        let tangent = if i + 1 < lane_geom.points.len() {
            (lane_geom.points[i + 1] - p).normalize_or_zero()
        } else if i > 0 {
            (p - lane_geom.points[i - 1]).normalize_or_zero()
        } else {
            Vec3::X
        };

        // Lateral points LEFT relative to forward tangent (Y-up)
        // Tangent=(0,0,1) -> Lateral=(-1,0,0)
        let lateral = Vec3::new(-tangent.z, 0.0, tangent.x).normalize_or_zero();

        let center_pos = p + lateral * offset_from_center;

        // "Left" here means index 0 (v_base), which is center - lateral (Geometrically Right)
        // "Right" here means index 1 (v_base+1), which is center + lateral (Geometrically Left)
        let left_pos_raw = center_pos - lateral * half_width;
        let right_pos_raw = center_pos + lateral * half_width;

        let h_left = terrain_renderer.get_height_at([left_pos_raw.x, left_pos_raw.z]);
        let h_right = terrain_renderer.get_height_at([right_pos_raw.x, right_pos_raw.z]);

        let final_left = Vec3::new(left_pos_raw.x, h_left + height + CLEARANCE, left_pos_raw.z);
        let final_right = Vec3::new(
            right_pos_raw.x,
            h_right + height + CLEARANCE,
            right_pos_raw.z,
        );

        let normal = Vec3::Y;

        let u = lane_geom.lengths[i] * uv_config.0;
        let v_min = 0.0;
        let v_max = width * uv_config.1;

        vertices.push(RoadVertex {
            position: final_left.to_array(),
            normal: normal.to_array(),
            uv: [u, v_min],
            material_id,
        });

        vertices.push(RoadVertex {
            position: final_right.to_array(),
            normal: normal.to_array(),
            uv: [u, v_max],
            material_id,
        });

        point_idx_to_vert_idx.insert(i, current_vert_idx);
        current_vert_idx += 2;
    }

    // Generate indices (Corrected for CCW Face Back culling)
    for k in 0..included_indices.len() - 1 {
        let idx_curr = included_indices[k];
        let idx_next = included_indices[k + 1];

        if idx_next != idx_curr + 1 {
            continue;
        }

        let v_base = base_vertex + point_idx_to_vert_idx[&idx_curr];
        let v_next = base_vertex + point_idx_to_vert_idx[&idx_next];

        // Layout:
        // v_base (Geom Right)   ----- v_next (Geom Right)
        // v_base+1 (Geom Left)  ----- v_next+1 (Geom Left)
        //
        // Triangle 1: v_base -> v_base+1 -> v_next (Right -> Left -> NextRight) -> CCW Up
        indices.push(v_base);
        indices.push(v_base + 1);
        indices.push(v_next);

        // Triangle 2: v_next -> v_base+1 -> v_next+1 (NextRight -> Left -> NextLeft) -> CCW Up
        indices.push(v_next);
        indices.push(v_base + 1);
        indices.push(v_next + 1);
    }
}

/// Builds vertical faces between two parallel strips (e.g., Curb).
/// `explicit_normal_sign`: If Some(1.0), normal points along +Lateral. If Some(-1.0), along -Lateral.
/// If None, it infers based on offset (heuristic for simple curbs).
fn build_vertical_face(
    terrain_renderer: &TerrainRenderer,
    ref_geom: &LaneGeometry,
    offset_lateral: f32,
    bottom_height: f32,
    top_height: f32,
    material_id: u32,
    chunk_filter: Option<ChunkId>,
    uv_config: (f32, f32),
    explicit_normal_sign: Option<f32>, // <--- Added to fix median culling
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    if (top_height - bottom_height).abs() < 0.001 {
        return;
    }
    if ref_geom.points.len() < 2 {
        return;
    }

    let base_vertex = vertices.len() as u32;
    let mut included_indices = Vec::new();

    match chunk_filter {
        Some(cid) => {
            let (min_x, max_x) = chunk_x_range(cid);
            let (min_z, max_z) = chunk_z_range(cid);
            for i in 0..ref_geom.points.len() {
                let p = ref_geom.points[i];
                if p.x >= min_x && p.x < max_x && p.z >= min_z && p.z < max_z {
                    included_indices.push(i);
                } else if i + 1 < ref_geom.points.len() {
                    let pn = ref_geom.points[i + 1];
                    if pn.x >= min_x && pn.x < max_x && pn.z >= min_z && pn.z < max_z {
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

    // Determine face direction
    // If explicit is provided, use it. Otherwise fall back to offset-based heuristic.
    let normal_sign =
        explicit_normal_sign.unwrap_or_else(|| if offset_lateral > 0.0 { -1.0 } else { 1.0 });

    for &i in &included_indices {
        let p = ref_geom.points[i];
        let tangent = if i + 1 < ref_geom.points.len() {
            (ref_geom.points[i + 1] - p).normalize_or_zero()
        } else if i > 0 {
            (p - ref_geom.points[i - 1]).normalize_or_zero()
        } else {
            Vec3::X
        };

        let lateral = Vec3::new(-tangent.z, 0.0, tangent.x).normalize_or_zero();

        let face_pos_raw = p + lateral * offset_lateral;
        let terrain_y = terrain_renderer.get_height_at([face_pos_raw.x, face_pos_raw.z]);

        let normal = lateral * normal_sign;

        let u = ref_geom.lengths[i] * uv_config.0;
        let v_h = (top_height - bottom_height).abs() * uv_config.1;

        // Bottom vertex
        vertices.push(RoadVertex {
            position: [
                face_pos_raw.x,
                terrain_y + bottom_height + CLEARANCE,
                face_pos_raw.z,
            ],
            normal: normal.to_array(),
            uv: [u, 0.0],
            material_id,
        });

        // Top vertex
        vertices.push(RoadVertex {
            position: [
                face_pos_raw.x,
                terrain_y + top_height + CLEARANCE,
                face_pos_raw.z,
            ],
            normal: normal.to_array(),
            uv: [u, v_h],
            material_id,
        });

        point_idx_to_vert_idx.insert(i, current_vert_idx);
        current_vert_idx += 2;
    }

    for k in 0..included_indices.len() - 1 {
        let idx_curr = included_indices[k];
        let idx_next = included_indices[k + 1];
        if idx_next != idx_curr + 1 {
            continue;
        }

        let v_base = base_vertex + point_idx_to_vert_idx[&idx_curr];
        let v_next = base_vertex + point_idx_to_vert_idx[&idx_next];

        // Ensure CCW winding based on normal direction
        if normal_sign > 0.0 {
            // Normal is +Lateral.
            // Tri 1: B0 -> B1 -> T0
            indices.push(v_base);
            indices.push(v_next);
            indices.push(v_base + 1);
            // Tri 2: T0 -> B1 -> T1
            indices.push(v_base + 1);
            indices.push(v_next);
            indices.push(v_next + 1);
        } else {
            // Normal is -Lateral.
            // Tri 1: B0 -> T0 -> B1
            indices.push(v_base);
            indices.push(v_base + 1);
            indices.push(v_next);
            // Tri 2: T0 -> T1 -> B1
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
fn draw_node_geometry(
    terrain_renderer: &TerrainRenderer,
    node_pos: Vec3,
    connected_lanes_info: &[(i8, f32)], // (Lane Index, Lane Width)
    cap_direction: Option<Vec3>,        // Direction pointing AWAY from the node (for end caps)
    style: &RoadStyleParams,
    config: &MeshConfig, // Used for UV scaling
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    // 1. Calculate Radius
    let mut max_radius = 2.0;
    for (idx, width) in connected_lanes_info {
        // Approximate radius required to cover the widest lane offset
        let r = idx.abs() as f32 * width + style.sidewalk_width;
        if r > max_radius {
            max_radius = r;
        }
    }

    // Radii
    let road_radius = max_radius - style.sidewalk_width;
    let sidewalk_radius = max_radius;

    // 2. Determine Angular Range
    let mut start_angle = 0.0;
    let mut end_angle = std::f32::consts::TAU;
    let mut uv_rotation = 0.0;

    // Handle End Caps (Semicircles)
    if let Some(dir) = cap_direction {
        let heading = dir.z.atan2(dir.x);
        start_angle = heading - std::f32::consts::FRAC_PI_2;
        end_angle = heading + std::f32::consts::FRAC_PI_2;
        uv_rotation = heading; // Rotate UVs to align with road
    }

    // 3. Setup Heights
    // Fix Z-fighting: Raise node road surface slightly above segment clearance
    let node_clearance = CLEARANCE;
    let y_base = node_pos.y;

    // Heights
    let h_road = y_base + style.lane_height + node_clearance;
    let h_sw = y_base + style.sidewalk_height + node_clearance;

    // 4. Build Meshes (Fan/Ring)
    let center_idx = vertices.len() as u32;

    // Center Vertex (Road)
    vertices.push(RoadVertex {
        position: [node_pos.x, h_road, node_pos.z],
        normal: [0.0, 1.0, 0.0],
        uv: [0.0, 0.0],
        material_id: style.lane_material_id,
    });

    let step_angle = (end_angle - start_angle) / NODE_ANGULAR_SEGMENTS as f32;

    // We generate a "spoke" of vertices for each angle step, then connect them
    // Spoke indices relative to current loop iteration:
    // 0: Road Rim
    // 1: Sidewalk Inner (Top)
    // 2: Sidewalk Outer (Top)
    // 3: Sidewalk Outer (Bottom/Terrain)

    let first_spoke_idx = vertices.len() as u32;

    for i in 0..=NODE_ANGULAR_SEGMENTS {
        let angle = start_angle + i as f32 * step_angle;
        let (sin, cos) = angle.sin_cos();
        let dir = Vec3::new(cos, 0.0, sin);

        // UV Projection (Rotated)
        // Project world offset onto rotation basis to align texture with road
        let rel_x = cos;
        let rel_z = sin;
        let rot_cos = (-uv_rotation).cos();
        let rot_sin = (-uv_rotation).sin();
        let u_base = rel_x * rot_cos - rel_z * rot_sin;
        let v_base = rel_x * rot_sin + rel_z * rot_cos;

        // --- Positions ---
        let pos_road_rim = node_pos + dir * road_radius;
        let pos_sw_outer = node_pos + dir * sidewalk_radius;

        let terrain_road = terrain_renderer.get_height_at([pos_road_rim.x, pos_road_rim.z]);
        let terrain_sw = terrain_renderer.get_height_at([pos_sw_outer.x, pos_sw_outer.z]);

        // Adjust heights to follow terrain but maintain flat node if desired.
        // For intersections, usually flat is better, but edges should snap.
        // Here we clamp to base Y to prevent the node from burying.
        let y_road_rim = terrain_road.max(y_base) + style.lane_height + node_clearance;
        let y_sw_top = terrain_road.max(y_base) + style.sidewalk_height + node_clearance; // Use road pos for inner sw height to keep curb vertical
        let y_sw_outer_top = terrain_sw.max(y_base) + style.sidewalk_height + node_clearance;
        let y_sw_outer_btm = terrain_sw.max(y_base) + CLEARANCE; // Down to ground

        // 1. Road Rim Vertex
        vertices.push(RoadVertex {
            position: [pos_road_rim.x, y_road_rim, pos_road_rim.z],
            normal: [0.0, 1.0, 0.0],
            uv: [cos * road_radius, 0.0],
            material_id: style.lane_material_id,
        });

        // 2. Sidewalk Inner (Top of inner curb)
        vertices.push(RoadVertex {
            position: [pos_road_rim.x, y_sw_top, pos_road_rim.z],
            normal: [0.0, 1.0, 0.0],
            uv: [cos * road_radius, sin * road_radius],
            material_id: style.sidewalk_material_id,
        });

        // 3. Sidewalk Outer Top
        vertices.push(RoadVertex {
            position: [pos_sw_outer.x, y_sw_outer_top, pos_sw_outer.z],
            normal: [0.0, 1.0, 0.0],
            uv: [cos * sidewalk_radius, sin * sidewalk_radius],
            material_id: style.sidewalk_material_id,
        });

        // 4. Sidewalk Outer Bottom (Terrain)
        vertices.push(RoadVertex {
            position: [pos_sw_outer.x, y_sw_outer_btm, pos_sw_outer.z],
            normal: [dir.x, 0.0, dir.z], // Normal points out
            uv: [cos * road_radius, 0.0],
            material_id: style.sidewalk_material_id, // Or generic curb material
        });
    }

    // Connect Spokes
    // Spoke size = 4 vertices
    for i in 0..NODE_ANGULAR_SEGMENTS {
        let s1 = first_spoke_idx + (i as u32 * 4);
        let s2 = first_spoke_idx + ((i + 1) as u32 * 4);

        // Indices in spoke:
        // 0: Road Rim
        // 1: SW Inner
        // 2: SW Outer Top
        // 3: SW Outer Bottom

        // A. Road Fan (Center -> s1.0 -> s2.0)
        // CCW
        indices.push(center_idx);
        indices.push(s2); // Road Rim next
        indices.push(s1); // Road Rim curr

        // B. Inner Curb (Road Rim -> SW Inner)
        // Face points IN towards center.
        // Quad: s1.0, s1.1, s2.1, s2.0
        // Counter-clockwise facing center: s1.0 -> s2.0 -> s2.1 -> s1.1
        // Wait, normal points in (-dir).
        // Let's visualize: 0 is low, 1 is high.
        // We want face visible from road.
        // Tri 1: s1.0 -> s2.0 -> s1.1 (LowLeft -> LowRight -> HighLeft)
        indices.push(s1);
        indices.push(s2);
        indices.push(s1 + 1);
        // Tri 2: s2.0 -> s2.1 -> s1.1 (LowRight -> HighRight -> HighLeft)
        indices.push(s2);
        indices.push(s2 + 1);
        indices.push(s1 + 1);

        // C. Sidewalk Surface (Ring)
        // s1.1, s1.2, s2.2, s2.1
        // Upward facing. CCW.
        // Tri 1: s1.1 -> s2.1 -> s1.2
        indices.push(s1 + 1);
        indices.push(s2 + 1);
        indices.push(s1 + 2);
        // Tri 2: s2.1 -> s2.2 -> s1.2
        indices.push(s2 + 1);
        indices.push(s2 + 2);
        indices.push(s1 + 2);

        // D. Outer Curb (Sidewalk Outer Top -> Bottom)
        // Face points OUT.
        // s1.2, s1.3, s2.3, s2.2
        // Quad facing out.
        // Tri 1: s1.2 -> s2.2 -> s1.3 (TopLeft -> TopRight -> BtmLeft)
        indices.push(s1 + 2);
        indices.push(s2 + 2);
        indices.push(s1 + 3);
        // Tri 2: s2.2 -> s2.3 -> s1.3 (TopRight -> BtmRight -> BtmLeft)
        indices.push(s2 + 2);
        indices.push(s2 + 3);
        indices.push(s1 + 3);
    }
}

// ============================================================================
// Road Mesh Manager (Refactored)
// ============================================================================

pub struct RoadMeshManager {
    chunk_cache: HashMap<ChunkId, ChunkMesh>,
    // Cache LaneGeometry instead of Rings. Key: (SegmentId, LaneIndex)
    lane_geom_cache: HashMap<(SegmentId, i8), LaneGeometry>,
    pub config: MeshConfig,
    pub style: RoadStyleParams,
}

impl RoadMeshManager {
    pub fn new(config: MeshConfig) -> Self {
        Self {
            chunk_cache: HashMap::new(),
            lane_geom_cache: HashMap::new(),
            config,
            style: RoadStyleParams::default(),
        }
    }

    pub fn get_chunk_mesh(&self, chunk_id: ChunkId) -> Option<&ChunkMesh> {
        self.chunk_cache.get(&chunk_id)
    }

    pub fn invalidate_chunk(&mut self, chunk_id: ChunkId) {
        self.chunk_cache.remove(&chunk_id);
    }

    pub fn clear_cache(&mut self) {
        self.chunk_cache.clear();
        self.lane_geom_cache.clear();
    }

    pub fn chunk_needs_update(&self, chunk_id: ChunkId, storage: &RoadStorage) -> bool {
        match self.chunk_cache.get(&chunk_id) {
            None => true,
            Some(mesh) => mesh.topo_version != compute_topo_version(chunk_id, storage),
        }
    }
    pub fn update_chunk_mesh(
        &mut self,
        terrain_renderer: &TerrainRenderer,
        chunk_id: ChunkId,
        storage: &RoadStorage,
    ) -> &ChunkMesh {
        let mesh = self.build_chunk_mesh(terrain_renderer, chunk_id, storage);
        self.chunk_cache.insert(chunk_id, mesh);
        self.chunk_cache.get(&chunk_id).unwrap()
    }
    pub fn build_chunk_mesh(
        &mut self,
        terrain_renderer: &TerrainRenderer,
        chunk_id: ChunkId,
        storage: &RoadStorage,
    ) -> ChunkMesh {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // -------- Segments (really: lanes that belong to segments) --------
        let mut segment_ids = storage.segment_ids_touching_chunk(chunk_id);
        segment_ids.sort_unstable();

        for seg_id in segment_ids {
            let segment = storage.segment(seg_id);
            if !segment.enabled {
                continue;
            }

            let mut render_lanes = Vec::new();

            for &lane_id in segment.lanes() {
                let lane = storage.lane(lane_id);
                if !lane.is_enabled() {
                    continue;
                }

                render_lanes.push((lane.lane_index(), lane.geometry().clone()));
            }

            if render_lanes.is_empty() {
                continue;
            }

            draw_segment_geometry(
                terrain_renderer,
                &render_lanes,
                &self.style,
                &self.config,
                Some(chunk_id),
                &mut vertices,
                &mut indices,
            );
        }

        // -------- Nodes (lane-driven, not segment-driven) --------
        let node_ids = storage.nodes_in_chunk(chunk_id);
        for node_id in node_ids {
            let node = storage.node(node_id).unwrap();
            let center = Vec3::from_array(node.position());

            let mut connected_lanes_info = Vec::new();
            let mut cap_direction = None;

            let incoming = node.incoming_lanes();
            let outgoing = node.outgoing_lanes();

            // Cap logic: dead end
            if incoming.len() + outgoing.len() == 1 {
                let lane_id = incoming
                    .first()
                    .copied()
                    .or_else(|| outgoing.first().copied());

                if let Some(lid) = lane_id {
                    let lane = storage.lane(lid);
                    let dir = if lane.from_node() == node_id {
                        lane.geometry().points[1] - lane.geometry().points[0]
                    } else {
                        let pts = &lane.geometry().points;
                        pts[pts.len() - 2] - pts[pts.len() - 1]
                    };

                    cap_direction = Some(dir.normalize_or_zero());
                }
            }

            for &lane_id in incoming.iter().chain(outgoing.iter()) {
                let lane = storage.lane(lane_id);
                connected_lanes_info.push((lane.lane_index(), self.style.lane_width));
            }

            if connected_lanes_info.is_empty() {
                continue;
            }

            draw_node_geometry(
                terrain_renderer,
                center,
                &connected_lanes_info,
                cap_direction,
                &self.style,
                &self.config,
                &mut vertices,
                &mut indices,
            );
        }

        ChunkMesh {
            vertices,
            indices,
            topo_version: compute_topo_version(chunk_id, storage),
        }
    }

    pub fn build_preview_mesh(
        &self,
        terrain_renderer: &TerrainRenderer,
        preview_state: &RoadPreviewState,
    ) -> ChunkMesh {
        //BROKEN///
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // -------- Preview Segments --------
        for seg in &preview_state.segments {
            if !seg.is_valid {
                continue;
            }
            let mut render_lanes = Vec::new();
            if render_lanes.is_empty() {
                continue;
            }

            draw_segment_geometry(
                terrain_renderer,
                &render_lanes,
                &self.style,
                &self.config,
                None,
                &mut vertices,
                &mut indices,
            );
        }

        // -------- Preview Nodes --------
        for node in &preview_state.nodes {
            if !node.is_valid {
                continue;
            }

            let mut connected_lanes_info = Vec::new();
            let mut cap_direction = None;

            if node.lane_count() == 0 {
                connected_lanes_info.push((1, self.style.lane_width));
                connected_lanes_info.push((-1, self.style.lane_width));
            } else {
                for i in 0..node.lane_count() {
                    let idx = (i as i8) + 1;
                    connected_lanes_info.push((idx, self.style.lane_width));
                    connected_lanes_info.push((-idx, self.style.lane_width));
                }
            }

            draw_node_geometry(
                terrain_renderer,
                node.world_pos,
                &connected_lanes_info,
                cap_direction,
                &self.style,
                &self.config,
                &mut vertices,
                &mut indices,
            );
        }

        ChunkMesh {
            vertices,
            indices,
            topo_version: 0,
        }
    }
}

// ============================================================================
// Topo Version Hashing
// ============================================================================

const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;

fn compute_topo_version(chunk_id: ChunkId, storage: &RoadStorage) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    let mut segs = storage.segment_ids_touching_chunk(chunk_id);
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
