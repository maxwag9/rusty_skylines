//# road_mesh_manager.rs
//! Road Mesh Manager for procedural lane-first citybuilder.
//!
//! Produces deterministic, chunked CPU mesh buffers from immutable road topology.
//! Refactored to be Lane-First: Geometry is derived directly from Lane centerlines.

use crate::renderer::gizmo::Gizmo;
use crate::renderer::world_renderer::TerrainRenderer;
use crate::terrain::roads::intersections::{
    LaneBoundaries, LaneBoundaryInfo, build_intersection_mesh,
};
use crate::terrain::roads::road_structs::*;
use crate::terrain::roads::roads::{LaneGeometry, Node, RoadStorage};
use glam::Vec3;
use std::collections::HashMap;

pub type ChunkId = u64;

// ============================================================================
// Constants & Configuration
// ============================================================================

const METERS_PER_LANE_POLYLINE_STEP: f32 = 2.0;
pub const CLEARANCE: f32 = 0.04;
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
fn mesh_segment_geometry(
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

fn draw_node_geometry(
    terrain_renderer: &TerrainRenderer,
    node_pos: Vec3,
    connected_lanes_info: &[(i8, f32)],
    cap_direction: Option<Vec3>,
    style: &RoadStyleParams,
    config: &MeshConfig,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    // --- Compute radii ---
    let mut max_radius = 2.0_f32;
    for (idx, width) in connected_lanes_info {
        max_radius = max_radius.max(idx.abs() as f32 * width + style.sidewalk_width);
    }

    let road_radius = max_radius - style.sidewalk_width;
    let sw_inner = road_radius;
    let sw_outer = max_radius;

    // --- Angular setup ---
    let (start_angle, end_angle, is_cap) = if let Some(dir) = cap_direction {
        let forward = -dir.normalize();
        let heading = forward.z.atan2(forward.x);
        (
            heading - std::f32::consts::FRAC_PI_2,
            heading + std::f32::consts::FRAC_PI_2,
            true,
        )
    } else {
        (0.0_f32, std::f32::consts::TAU, false)
    };

    let segments = NODE_ANGULAR_SEGMENTS as usize;
    // For cap: we need segments+1 vertices to span segments quads over half circle
    // For full circle: we need segments vertices, last quad wraps to first
    let num_verts = if is_cap { segments + 1 } else { segments };
    let num_quads = segments;

    if num_verts < 2 {
        return;
    }

    let angle_span = end_angle - start_angle;
    let step_angle = angle_span / num_quads as f32;

    let y_base = node_pos.y;
    let up_normal = [0.0_f32, 1.0, 0.0];

    // Helper to get next vertex index (wraps for full circle, doesn't for cap)
    let next_idx = |i: usize| -> usize { if is_cap { i + 1 } else { (i + 1) % num_verts } };

    // =========================================================================
    // 1) ROAD SURFACE - Filled disk using triangle fan from center
    // =========================================================================

    let center_terrain = terrain_renderer.get_height_at([node_pos.x, node_pos.z]);
    let center_y = center_terrain.max(y_base) + style.lane_height + CLEARANCE;
    let center_idx = vertices.len() as u32;

    vertices.push(RoadVertex {
        position: [node_pos.x, center_y, node_pos.z],
        normal: up_normal,
        uv: [0.5, 0.5],
        material_id: style.lane_material_id,
    });

    let road_ring_first = vertices.len() as u32;
    for i in 0..num_verts {
        let angle = start_angle + i as f32 * step_angle;
        let (sin_a, cos_a) = angle.sin_cos();
        let dir = Vec3::new(cos_a, 0.0, sin_a);
        let pos = node_pos + dir * road_radius;

        let terrain_h = terrain_renderer.get_height_at([pos.x, pos.z]);
        let y = terrain_h.max(y_base) + style.lane_height + CLEARANCE;

        // Polar UVs for road disk
        let uv_scale = road_radius / config.uv_scale_v;
        let u = 0.5 + cos_a * uv_scale;
        let v = 0.5 + sin_a * uv_scale;

        vertices.push(RoadVertex {
            position: [pos.x, y, pos.z],
            normal: up_normal,
            uv: [u, v],
            material_id: style.lane_material_id,
        });
    }

    // Road fan triangles
    for i in 0..num_quads {
        let curr = road_ring_first + i as u32;
        let next = road_ring_first + next_idx(i) as u32;
        emit_tri_for_top(indices, vertices, center_idx, curr, next);
    }

    // =========================================================================
    // 2) SIDEWALK TOP - Annular ring from sw_inner to sw_outer
    // =========================================================================

    let sw_first_idx = vertices.len() as u32;
    let sw_stride = 2u32;

    for i in 0..num_verts {
        let angle = start_angle + i as f32 * step_angle;
        let (sin_a, cos_a) = angle.sin_cos();
        let dir = Vec3::new(cos_a, 0.0, sin_a);

        let pos_inner = node_pos + dir * sw_inner;
        let pos_outer = node_pos + dir * sw_outer;

        let terrain_inner = terrain_renderer.get_height_at([pos_inner.x, pos_inner.z]);
        let terrain_outer = terrain_renderer.get_height_at([pos_outer.x, pos_outer.z]);

        let y_inner = terrain_inner.max(y_base) + style.sidewalk_height + CLEARANCE;
        let y_outer = terrain_outer.max(y_base) + style.sidewalk_height + CLEARANCE;

        let arc_u = (angle - start_angle) * ((sw_inner + sw_outer) * 0.5) / config.uv_scale_u;

        // Inner edge of sidewalk
        vertices.push(RoadVertex {
            position: [pos_inner.x, y_inner, pos_inner.z],
            normal: up_normal,
            uv: [arc_u, 0.0],
            material_id: style.sidewalk_material_id,
        });

        // Outer edge of sidewalk
        vertices.push(RoadVertex {
            position: [pos_outer.x, y_outer, pos_outer.z],
            normal: up_normal,
            uv: [arc_u, style.sidewalk_width / config.uv_scale_v],
            material_id: style.sidewalk_material_id,
        });
    }

    // Sidewalk top triangles
    for i in 0..num_quads {
        let ni = next_idx(i);
        let inner_curr = sw_first_idx + (i as u32) * sw_stride;
        let outer_curr = inner_curr + 1;
        let inner_next = sw_first_idx + (ni as u32) * sw_stride;
        let outer_next = inner_next + 1;

        emit_tri_for_top(indices, vertices, inner_curr, outer_curr, inner_next);
        emit_tri_for_top(indices, vertices, inner_next, outer_curr, outer_next);
    }

    // =========================================================================
    // 3) INNER CURB - Vertical face at sw_inner (between road and sidewalk)
    //    Normal faces INWARD (toward center)
    // =========================================================================

    let inner_curb_first = vertices.len() as u32;
    let curb_stride = 2u32;

    for i in 0..num_verts {
        let angle = start_angle + i as f32 * step_angle;
        let (sin_a, cos_a) = angle.sin_cos();
        let dir = Vec3::new(cos_a, 0.0, sin_a);
        let pos = node_pos + dir * sw_inner;

        let terrain_h = terrain_renderer.get_height_at([pos.x, pos.z]);
        let y_top = terrain_h.max(y_base) + style.sidewalk_height + CLEARANCE;
        let y_bot = terrain_h.max(y_base) + style.lane_height + CLEARANCE;

        // Normal points INWARD (toward center)
        let inward_normal = [-dir.x, 0.0, -dir.z];

        let arc_u = (angle - start_angle) * sw_inner / config.uv_scale_u;
        let curb_height = style.sidewalk_height - style.lane_height;

        // Top of inner curb
        vertices.push(RoadVertex {
            position: [pos.x, y_top, pos.z],
            normal: inward_normal,
            uv: [arc_u, curb_height / config.uv_scale_v],
            material_id: style.sidewalk_material_id,
        });

        // Bottom of inner curb
        vertices.push(RoadVertex {
            position: [pos.x, y_bot, pos.z],
            normal: inward_normal,
            uv: [arc_u, 0.0],
            material_id: style.sidewalk_material_id,
        });
    }

    // Inner curb triangles (facing inward)
    for i in 0..num_quads {
        let ni = next_idx(i);
        let top_curr = inner_curb_first + (i as u32) * curb_stride;
        let bot_curr = top_curr + 1;
        let top_next = inner_curb_first + (ni as u32) * curb_stride;
        let bot_next = top_next + 1;

        emit_tri_for_inner_curb(indices, vertices, top_curr, bot_curr, top_next, node_pos);
        emit_tri_for_inner_curb(indices, vertices, top_next, bot_curr, bot_next, node_pos);
    }

    // =========================================================================
    // 4) OUTER CURB - Vertical face at sw_outer (sidewalk edge to ground)
    //    Normal faces OUTWARD (away from center)
    // =========================================================================

    let outer_curb_first = vertices.len() as u32;

    for i in 0..num_verts {
        let angle = start_angle + i as f32 * step_angle;
        let (sin_a, cos_a) = angle.sin_cos();
        let dir = Vec3::new(cos_a, 0.0, sin_a);
        let pos = node_pos + dir * sw_outer;

        let terrain_h = terrain_renderer.get_height_at([pos.x, pos.z]);
        let y_top = terrain_h.max(y_base) + style.sidewalk_height + CLEARANCE;
        let y_bot = terrain_h.max(y_base) + CLEARANCE;

        // Normal points OUTWARD (away from center)
        let outward_normal = [dir.x, 0.0, dir.z];

        let arc_u = (angle - start_angle) * sw_outer / config.uv_scale_u;

        // Top of outer curb
        vertices.push(RoadVertex {
            position: [pos.x, y_top, pos.z],
            normal: outward_normal,
            uv: [arc_u, style.sidewalk_height / config.uv_scale_v],
            material_id: style.sidewalk_material_id,
        });

        // Bottom of outer curb
        vertices.push(RoadVertex {
            position: [pos.x, y_bot, pos.z],
            normal: outward_normal,
            uv: [arc_u, 0.0],
            material_id: style.sidewalk_material_id,
        });
    }

    // Outer curb triangles (facing outward)
    for i in 0..num_quads {
        let ni = next_idx(i);
        let top_curr = outer_curb_first + (i as u32) * curb_stride;
        let bot_curr = top_curr + 1;
        let top_next = outer_curb_first + (ni as u32) * curb_stride;
        let bot_next = top_next + 1;

        emit_tri_for_curb(indices, vertices, top_curr, bot_curr, top_next, node_pos);
        emit_tri_for_curb(indices, vertices, top_next, bot_curr, bot_next, node_pos);
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
        // println!("Final Vertex Height Left: {}, Right: {}; Terrain Height Left: {}, Right: {}", final_left.y, final_right.y, h_left, h_right);
        point_idx_to_vert_idx.insert(i, current_vert_idx);
        current_vert_idx += 2;
    }

    // Generate indices (Corrected for CCW Face Back culling)
    for w in included_indices.windows(2) {
        let a = w[0];
        let b = w[1];

        let v_base = base_vertex + point_idx_to_vert_idx[&a];
        let v_next = base_vertex + point_idx_to_vert_idx[&b];

        indices.push(v_base);
        indices.push(v_base + 1);
        indices.push(v_next);

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

// compute triangle normal (not normalized)
fn tri_normal(v0: Vec3, v1: Vec3, v2: Vec3) -> Vec3 {
    (v1 - v0).cross(v2 - v0)
}

// push triangle ensuring top-facing normals point up (normal.y > 0)
// triangles with normal.y < 0 will be flipped (swap i1 <-> i2)
fn emit_tri_for_top(indices: &mut Vec<u32>, vertices: &Vec<RoadVertex>, i0: u32, i1: u32, i2: u32) {
    let v0 = Vec3::from(vertices[i0 as usize].position);
    let v1 = Vec3::from(vertices[i1 as usize].position);
    let v2 = Vec3::from(vertices[i2 as usize].position);
    let n = tri_normal(v0, v1, v2);
    let ok = n.y >= 0.0;
    if ok {
        indices.push(i0);
        indices.push(i1);
        indices.push(i2);
    } else {
        indices.push(i0);
        indices.push(i2);
        indices.push(i1);
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
    node_pos: Vec3,
) {
    let v0 = Vec3::from(vertices[i0 as usize].position);
    let v1 = Vec3::from(vertices[i1 as usize].position);
    let v2 = Vec3::from(vertices[i2 as usize].position);
    let n = tri_normal(v0, v1, v2);
    let centroid = (v0 + v1 + v2) / 3.0;
    let centroid_dir = (centroid - node_pos).normalize_or_zero();
    let dot = n.dot(centroid_dir);
    let ok = dot >= 0.0;
    if ok {
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
    node_pos: Vec3,
) {
    let v0 = Vec3::from(vertices[i0 as usize].position);
    let v1 = Vec3::from(vertices[i1 as usize].position);
    let v2 = Vec3::from(vertices[i2 as usize].position);
    let n = tri_normal(v0, v1, v2);
    let centroid = (v0 + v1 + v2) / 3.0;
    let centroid_dir = (centroid - node_pos).normalize_or_zero();
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

    pub fn invalidate_chunk(&mut self, chunk_id: ChunkId) {
        self.chunk_cache.remove(&chunk_id);
    }

    pub fn clear_cache(&mut self) {
        self.chunk_cache.clear();
    }

    pub fn chunk_needs_update(&self, chunk_id: ChunkId, storage: &RoadStorage) -> bool {
        match self.chunk_cache.get(&chunk_id) {
            None => true,
            Some(mesh) => mesh.topo_version != compute_topo_version(chunk_id, storage),
        }
    }

    /// Build mesh for a chunk (Some) or all geometry (None for previews)
    pub fn build_mesh(
        &self,
        terrain_renderer: &TerrainRenderer,
        chunk_id: Option<ChunkId>,
        storage: &RoadStorage,
        style: &RoadStyleParams,
        gizmo: &mut Gizmo,
    ) -> ChunkMesh {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // ════════════════════════════════════════════════════════════════════
        // Phase 1: Build all intersection meshes and collect lane boundaries
        // ═══════════════════════════════════════════════════════════ ═ ════

        let mut all_lane_boundaries: LaneBoundaries = HashMap::new();

        let node_ids: Vec<NodeId> = match chunk_id {
            Some(cid) => storage.nodes_in_chunk(cid),
            None => storage.get_active_node_ids(),
        };

        for node_id in &node_ids {
            let node = match storage.node(*node_id) {
                Some(n) => n,
                None => continue,
            };

            let segment_count = storage.enabled_segment_count_connected_to_node(*node_id);

            if segment_count >= 2 {
                // This is an intersection - build mesh and get boundaries
                let result = build_intersection_mesh(
                    terrain_renderer,
                    *node_id,
                    node,
                    storage,
                    style,
                    &self.config,
                    &mut vertices,
                    &mut indices,
                    gizmo,
                );

                // Merge lane boundaries into master map
                // for (lane_id, boundary) in result.lane_boundaries {
                //     let lane = storage.lane(&lane_id);
                //     let is_from_node = lane.from_node() == *node_id;
                //
                //     let entry = all_lane_boundaries.entry(lane_id).or_insert((None, None));
                //     if is_from_node {
                //         entry.0 = Some(boundary);
                //     } else {
                //         entry.1 = Some(boundary);
                //     }
                // }
            } else {
                // Dead end or single connection - use existing node geometry
                let center = Vec3::from_array(node.position());
                let mut connected_lanes_info = Vec::new();

                let incoming = node.incoming_lanes();
                let outgoing = node.outgoing_lanes();

                for lane_id in incoming.iter().chain(outgoing.iter()) {
                    let lane = storage.lane(lane_id);
                    connected_lanes_info.push((lane.lane_index(), style.lane_width));
                }

                if connected_lanes_info.is_empty() {
                    continue;
                }

                let cap_direction = compute_cap_direction(*node_id, node, storage);

                draw_node_geometry(
                    terrain_renderer,
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

        // ════════════════════════════════════════════════════════════════════
        // Phase 2: Build segment meshes with intersection trimming
        // ════════════════════════════════════════════════════════════════════

        let segment_ids: Vec<SegmentId> = match chunk_id {
            Some(cid) => {
                let mut ids = storage.segment_ids_touching_chunk(cid);
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

            // Collect lanes with their boundaries
            let mut lane_data: Vec<(
                LaneId,
                i8,
                LaneGeometry,
                Option<LaneBoundaryInfo>,
                Option<LaneBoundaryInfo>,
            )> = Vec::new();

            for lane_id in segment.lanes() {
                let lane = storage.lane(lane_id);
                if !lane.is_enabled() {
                    continue;
                }

                let boundaries = all_lane_boundaries.get(&lane_id);
                let (start_bound, end_bound) = match boundaries {
                    Some((s, e)) => (s.clone(), e.clone()),
                    None => (None, None),
                };

                lane_data.push((
                    *lane_id,
                    lane.lane_index(),
                    lane.geometry().clone(),
                    start_bound,
                    end_bound,
                ));
            }

            if lane_data.is_empty() {
                continue;
            }

            // draw_segment_geometry_with_trim(
            //     terrain_renderer,
            //     &lane_data,
            //     style,
            //     &self.config,
            //     chunk_id,
            //     &mut vertices,
            //     &mut indices,
            // );
        }

        ChunkMesh {
            vertices,
            indices,
            topo_version: chunk_id
                .map(|cid| compute_topo_version(cid, storage))
                .unwrap_or(0),
        }
    }

    pub fn update_chunk_mesh(
        &mut self,
        terrain_renderer: &TerrainRenderer,
        chunk_id: ChunkId,
        storage: &RoadStorage,
        style: &RoadStyleParams,
        gizmo: &mut Gizmo,
    ) -> &ChunkMesh {
        let mesh = self.build_mesh(terrain_renderer, Some(chunk_id), storage, style, gizmo);
        self.chunk_cache.insert(chunk_id, mesh);
        self.chunk_cache.get(&chunk_id).unwrap()
    }

    /// Convenience wrapper for building preview meshes (no chunking)
    pub fn build_preview_mesh(
        &self,
        terrain_renderer: &TerrainRenderer,
        preview_storage: &RoadStorage,
        style: &RoadStyleParams,
        gizmo: &mut Gizmo,
    ) -> ChunkMesh {
        self.build_mesh(terrain_renderer, None, preview_storage, style, gizmo)
    }
}

fn compute_cap_direction(node_id: NodeId, node: &Node, storage: &RoadStorage) -> Option<Vec3> {
    let cap_direction = {
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
                pts[1] - pts[0]
            } else {
                pts[pts.len() - 2] - pts[pts.len() - 1]
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
    };
    cap_direction
}
// ============================================================================
// Topo Version Hashing
// ============================================================================

const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;

pub(crate) fn compute_topo_version(chunk_id: ChunkId, storage: &RoadStorage) -> u64 {
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
