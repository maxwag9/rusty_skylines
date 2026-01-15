//# road_mesh_manager.rs
//! Road Mesh Manager for procedural lane-first citybuilder.
//!
//! Produces deterministic, chunked CPU mesh buffers from immutable road topology.
//! Supports: segments, nodes (sole/end/through/intersection), with bridge/intersection stubs.

use crate::renderer::world_renderer::TerrainRenderer;
use crate::terrain::roads::road_editor::{NodePreview, SegmentPreview};
use crate::terrain::roads::road_preview::RoadPreviewState;
use crate::terrain::roads::roads::{LaneId, Node, NodeId, RoadManager, Segment, SegmentId};
use std::collections::HashMap;

const N_SAMPLE: usize = 64;
const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;
pub const CLEARANCE: f32 = 0.04;
const NODE_ANGULAR_SEGMENTS: usize = 32;

pub type ChunkId = u64;

// ============================================================================
// Chunk ID encoding/decoding
// ============================================================================

#[inline]
fn zigzag_i32(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}

#[inline]
fn unzigzag_u32(v: u32) -> i32 {
    ((v >> 1) as i32) ^ -((v & 1) as i32)
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

#[inline(always)]
pub fn chunk_coord_to_id(cx: i32, cz: i32) -> ChunkId {
    part1by1(zigzag_i32(cx)) | (part1by1(zigzag_i32(cz)) << 1)
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

#[inline]
pub fn visible_chunks_to_chunk_ids(visible_i32: &[(i32, i32, i32)]) -> Vec<ChunkId> {
    visible_i32
        .iter()
        .map(|&(cx, cz, _)| chunk_coord_to_id(cx, cz))
        .collect()
}

// ============================================================================
// Vertex format
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
            array_stride: size_of::<RoadVertex>() as wgpu::BufferAddress,
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
// Cross-section definition
// ============================================================================

#[derive(Clone, Debug, PartialEq)]
pub struct CrossSectionRegion {
    pub width: f32,
    pub height: f32,
    pub material_id: u32,
}

#[derive(Clone, Debug)]
pub struct LateralStrip {
    pub left: f32,
    pub right: f32,
    pub material_id: u32,
    pub height: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CrossSection {
    pub regions: Vec<CrossSectionRegion>,
}

impl CrossSection {
    fn from_node_preview(node_preview: &NodePreview) -> CrossSection {
        let (left_lanes, right_lanes) = node_preview.lane_counts();
        let left_lanes = left_lanes.max(1);
        let right_lanes = right_lanes.max(1);
        let median = false;
        let sidewalk_left = left_lanes > 0;
        let sidewalk_right = true;
        build_cross_section(
            left_lanes,
            right_lanes,
            sidewalk_left,
            sidewalk_right,
            median,
            &CrossSectionParams::default(),
        )
    }

    pub fn from_segment(road_manager: &RoadManager, segment: &Segment) -> CrossSection {
        let (left_lanes, right_lanes) = road_manager.lane_counts_for_segment(segment);
        let lane_count = left_lanes + right_lanes;
        let median = lane_count > 6 && left_lanes != 0 && right_lanes != 0;
        let sidewalk_left = left_lanes <= 3;
        let sidewalk_right = right_lanes <= 3;
        build_cross_section(
            left_lanes,
            right_lanes,
            sidewalk_left,
            sidewalk_right,
            median,
            &CrossSectionParams::default(),
        )
    }

    fn from_preview_segment(preview_segment: &SegmentPreview) -> CrossSection {
        let (left_lanes, right_lanes) = preview_segment.lane_count_each_dir;
        let lane_count = left_lanes + right_lanes;
        let median = lane_count > 6 && left_lanes != 0 && right_lanes != 0;
        let sidewalk_left = left_lanes <= 3;
        let sidewalk_right = right_lanes <= 3;
        build_cross_section(
            left_lanes,
            right_lanes,
            sidewalk_left,
            sidewalk_right,
            median,
            &CrossSectionParams::default(),
        )
    }

    fn from_node(node: &Node, connections: usize, node_type: NodeType) -> CrossSection {
        let mut right_lanes = node.outgoing_lanes.len().max(1);
        let mut left_lanes = node.incoming_lanes.len();
        match node_type {
            NodeType::Sole => {
                right_lanes *= 2;
                left_lanes *= 2;
            }
            NodeType::End => {}
            NodeType::Through => {
                right_lanes = right_lanes / 2;
                left_lanes = left_lanes / 2
            }
            NodeType::Intersection => {}
        }
        let median = false;
        let sidewalk_left = left_lanes > 0;
        let sidewalk_right = true;

        build_cross_section(
            left_lanes,
            right_lanes,
            sidewalk_left,
            sidewalk_right,
            median,
            &CrossSectionParams::default(),
        )
    }

    pub fn from_node_geometry(geom: &NodeGeometry, params: &CrossSectionParams) -> Self {
        let left_lanes = geom._outgoing_lanes.len();
        let right_lanes = geom._incoming_lanes.len().max(1);

        let median = false;
        let sidewalk_left = left_lanes > 0;
        let sidewalk_right = true;

        build_cross_section(0, 1, sidewalk_left, sidewalk_right, median, params)
    }

    pub fn total_width(&self) -> f32 {
        self.regions.iter().map(|r| r.width).sum()
    }

    pub fn left_offset(&self) -> f32 {
        -self.total_width() * 0.5
    }

    pub fn half_width(&self) -> f32 {
        self.total_width() * 0.5
    }

    pub fn lateral_strips(&self) -> Vec<LateralStrip> {
        let mut strips = Vec::with_capacity(self.regions.len());
        let mut x = self.left_offset();
        for r in &self.regions {
            strips.push(LateralStrip {
                left: x,
                right: x + r.width,
                material_id: r.material_id,
                height: r.height,
            });
            x += r.width;
        }
        strips
    }
    pub fn right_lateral_strips(&self) -> Vec<LateralStrip> {
        let mut strips = Vec::with_capacity(self.regions.len());
        let mut x = self.left_offset();
        for r in &self.regions {
            if x > -0.001 {
                strips.push(LateralStrip {
                    left: x,
                    right: x + r.width,
                    material_id: r.material_id,
                    height: r.height,
                });
            }
            x += r.width;
        }
        strips
    }
}

pub struct CrossSectionParams {
    pub lane_width: f32,
    pub lane_height: f32,
    pub sidewalk_width: f32,
    pub sidewalk_height: f32,
    pub median_width: f32,
    pub median_height: f32,
}

impl Default for CrossSectionParams {
    fn default() -> Self {
        Self {
            lane_width: 2.5,
            lane_height: 0.0,
            sidewalk_width: 1.0,
            sidewalk_height: 0.1,
            median_width: 0.2,
            median_height: 0.1,
        }
    }
}

pub fn build_cross_section(
    left_lanes: usize,
    right_lanes: usize,
    sidewalk_left: bool,
    sidewalk_right: bool,
    median: bool,
    params: &CrossSectionParams,
) -> CrossSection {
    let mut regions = Vec::new();

    if sidewalk_left {
        regions.push(CrossSectionRegion {
            width: params.sidewalk_width,
            height: params.sidewalk_height,
            material_id: 0,
        });
    }

    for _ in 0..left_lanes {
        regions.push(CrossSectionRegion {
            width: params.lane_width,
            height: params.lane_height,
            material_id: 2,
        });
    }

    if median {
        regions.push(CrossSectionRegion {
            width: params.median_width,
            height: params.median_height,
            material_id: 0,
        });
    }

    for _ in 0..right_lanes {
        regions.push(CrossSectionRegion {
            width: params.lane_width,
            height: params.lane_height,
            material_id: 2,
        });
    }

    if sidewalk_right {
        regions.push(CrossSectionRegion {
            width: params.sidewalk_width,
            height: params.sidewalk_height,
            material_id: 0,
        });
    }

    CrossSection { regions }
}

// ============================================================================
// Horizontal profile (curves)
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HorizontalProfile {
    Linear,
    QuadraticBezier {
        control: [f32; 2],
    },
    CubicBezier {
        control1: [f32; 2],
        control2: [f32; 2],
    },
    Arc {
        radius: f32,
        large_arc: bool,
    },
}

// ============================================================================
// Node Type Classification
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeType {
    Sole,
    End,
    Through,
    Intersection,
}

impl NodeType {
    pub fn from_connection_count(count: usize) -> Self {
        match count {
            0 => NodeType::Sole,
            1 => NodeType::End,
            2 => NodeType::Through,
            _ => NodeType::Intersection,
        }
    }
}

// ============================================================================
// Segment Geometry - unified abstraction
// ============================================================================

#[derive(Clone, Debug)]
pub struct SegmentGeometry {
    pub start_xz: [f32; 2],
    pub end_xz: [f32; 2],
    pub horizontal_profile: HorizontalProfile,
    pub vertical_slope: f32,
}

impl SegmentGeometry {
    pub fn from_segment(
        segment_id: SegmentId,
        segment: &Segment,
        manager: &RoadManager,
    ) -> Option<Self> {
        let start = manager.node(segment.start())?;
        let end = manager.node(segment.end())?;
        let _ = segment_id;
        Some(Self {
            start_xz: [start.x, start.z],
            end_xz: [end.x, end.z],
            horizontal_profile: segment.horizontal_profile,
            vertical_slope: segment.vertical_profile.slope(),
        })
    }

    pub fn from_preview(preview: &SegmentPreview) -> Self {
        let horizontal_profile = match preview.control {
            Some(ctrl) => HorizontalProfile::QuadraticBezier {
                control: [ctrl.x, ctrl.z],
            },
            None => HorizontalProfile::Linear,
        };

        Self {
            start_xz: [preview.start.x, preview.start.z],
            end_xz: [preview.end.x, preview.end.z],
            horizontal_profile,
            vertical_slope: preview.end.y - preview.start.y,
        }
    }

    pub fn evaluate_xz(&self, t: f32) -> (f32, f32) {
        let p0 = self.start_xz;
        let p1 = self.end_xz;

        match self.horizontal_profile {
            HorizontalProfile::Linear => (lerp(p0[0], p1[0], t), lerp(p0[1], p1[1], t)),
            HorizontalProfile::QuadraticBezier { control } => {
                let omt = 1.0 - t;
                (
                    omt * omt * p0[0] + 2.0 * omt * t * control[0] + t * t * p1[0],
                    omt * omt * p0[1] + 2.0 * omt * t * control[1] + t * t * p1[1],
                )
            }
            HorizontalProfile::CubicBezier { control1, control2 } => {
                let omt = 1.0 - t;
                let omt2 = omt * omt;
                let omt3 = omt2 * omt;
                let t2 = t * t;
                let t3 = t2 * t;
                (
                    omt3 * p0[0]
                        + 3.0 * omt2 * t * control1[0]
                        + 3.0 * omt * t2 * control2[0]
                        + t3 * p1[0],
                    omt3 * p0[1]
                        + 3.0 * omt2 * t * control1[1]
                        + 3.0 * omt * t2 * control2[1]
                        + t3 * p1[1],
                )
            }
            HorizontalProfile::Arc { radius, large_arc } => {
                evaluate_arc_xz(p0, p1, radius, large_arc, t)
            }
        }
    }

    pub fn tangent_xz(&self, t: f32) -> [f32; 2] {
        let p0 = self.start_xz;
        let p1 = self.end_xz;

        match self.horizontal_profile {
            HorizontalProfile::Linear => vec2_normalize([p1[0] - p0[0], p1[1] - p0[1]]),
            HorizontalProfile::QuadraticBezier { control } => {
                let omt = 1.0 - t;
                let d0 = [control[0] - p0[0], control[1] - p0[1]];
                let d1 = [p1[0] - control[0], p1[1] - control[1]];
                vec2_normalize([
                    2.0 * omt * d0[0] + 2.0 * t * d1[0],
                    2.0 * omt * d0[1] + 2.0 * t * d1[1],
                ])
            }
            HorizontalProfile::CubicBezier { control1, control2 } => {
                let omt = 1.0 - t;
                let omt2 = omt * omt;
                let t2 = t * t;
                let d0 = [control1[0] - p0[0], control1[1] - p0[1]];
                let d1 = [control2[0] - control1[0], control2[1] - control1[1]];
                let d2 = [p1[0] - control2[0], p1[1] - control2[1]];
                vec2_normalize([
                    3.0 * omt2 * d0[0] + 6.0 * omt * t * d1[0] + 3.0 * t2 * d2[0],
                    3.0 * omt2 * d0[1] + 6.0 * omt * t * d1[1] + 3.0 * t2 * d2[1],
                ])
            }
            HorizontalProfile::Arc { .. } => {
                let dt = 0.0005;
                let (x0, z0) = self.evaluate_xz((t - dt).max(0.0));
                let (x1, z1) = self.evaluate_xz((t + dt).min(1.0));
                vec2_normalize([x1 - x0, z1 - z0])
            }
        }
    }

    pub fn length_estimate(&self) -> f32 {
        let dx = self.end_xz[0] - self.start_xz[0];
        let dz = self.end_xz[1] - self.start_xz[1];
        (dx * dx + dz * dz).sqrt()
    }
}

// ============================================================================
// Node Geometry - unified abstraction for all node types
// ============================================================================

#[derive(Clone, Debug)]
pub struct ConnectedSegmentInfo {
    pub direction_xz: [f32; 2],
    pub _tangent_xz: [f32; 2],
    pub segment_id: Option<SegmentId>,
    pub cross_section: CrossSection,
    pub node_is_start: bool,
}

#[derive(Clone, Debug)]
pub struct NodeGeometry {
    pub position: [f32; 3],
    pub connections: Vec<ConnectedSegmentInfo>,
    pub _incoming_lanes: Vec<LaneId>,
    pub _outgoing_lanes: Vec<LaneId>,
    pub is_preview: bool,
}

impl NodeGeometry {
    pub fn from_node(node_id: NodeId, manager: &RoadManager) -> Option<Self> {
        let node = manager.node(node_id)?;
        let connected_segment_ids = manager.segments_connected_to_node(node_id);

        let mut connections = Vec::new();
        for &seg_id in &connected_segment_ids {
            let segment = manager.segment(seg_id);
            let geom = SegmentGeometry::from_segment(seg_id, segment, manager)?;
            let cross_section = CrossSection::from_segment(manager, segment);

            let is_start = segment.start() == node_id;
            let (t_at_node, direction_sign) = if is_start { (0.0, 1.0) } else { (1.0, -1.0) };

            let tangent = geom.tangent_xz(t_at_node);
            let direction = [tangent[0] * direction_sign, tangent[1] * direction_sign];

            connections.push(ConnectedSegmentInfo {
                direction_xz: direction,
                _tangent_xz: tangent,
                segment_id: Some(seg_id),
                cross_section,
                node_is_start: is_start,
            });
        }

        sort_connections_by_angle(&mut connections);

        Some(Self {
            position: [node.x, node.y, node.z],
            connections,
            _incoming_lanes: node.incoming_lanes.clone(),
            _outgoing_lanes: node.outgoing_lanes.clone(),
            is_preview: false,
        })
    }

    pub fn from_preview(
        node_preview: &NodePreview,
        connected_previews: &[&SegmentPreview],
    ) -> Self {
        let mut connections = Vec::new();

        for seg_preview in connected_previews {
            if !seg_preview.is_valid {
                continue;
            }

            let geom = SegmentGeometry::from_preview(seg_preview);
            let cross_section = CrossSection::from_preview_segment(seg_preview);

            let is_start = (seg_preview.start - node_preview.world_pos).length() < 0.01;
            let (t_at_node, direction_sign) = if is_start { (0.0, 1.0) } else { (1.0, -1.0) };

            let tangent = geom.tangent_xz(t_at_node);
            let direction = [tangent[0] * direction_sign, tangent[1] * direction_sign];

            connections.push(ConnectedSegmentInfo {
                direction_xz: direction,
                _tangent_xz: tangent,
                segment_id: None,
                cross_section,
                node_is_start: is_start,
            });
        }
        connections.append(&mut node_preview.connected_segments.clone());
        sort_connections_by_angle(&mut connections);

        Self {
            position: node_preview.world_pos.to_array(),
            connections,
            _incoming_lanes: node_preview.incoming_lanes.clone(),
            _outgoing_lanes: node_preview.outgoing_lanes.clone(),
            is_preview: true,
        }
    }

    pub fn node_type(&self) -> NodeType {
        NodeType::from_connection_count(self.connections.len())
    }

    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Get the maximum half-width from all connected segments
    pub fn max_half_width(&self) -> f32 {
        self.connections
            .iter()
            .map(|c| c.cross_section.half_width())
            .fold(0.0f32, |a, b| a.max(b))
            .max(2.5) // Minimum reasonable size
    }
}

fn sort_connections_by_angle(connections: &mut Vec<ConnectedSegmentInfo>) {
    connections.sort_by(|a, b| {
        let angle_a = a.direction_xz[1].atan2(a.direction_xz[0]);
        let angle_b = b.direction_xz[1].atan2(b.direction_xz[0]);
        angle_a.partial_cmp(&angle_b).unwrap()
    });
}
// ============================================================================
// Ring and Arc-length sampling (for segments)
// ============================================================================

#[derive(Clone, Debug)]
pub struct Ring {
    pub t: f32,
    pub arc_length: f32,
    pub position: [f32; 3],
    pub tangent: [f32; 3],
    pub lateral: [f32; 2],
}

#[derive(Clone, Copy, Debug)]
struct ArcSample {
    t: f32,
    cumulative_length: f32,
}

fn estimate_arc_length(
    terrain_renderer: &TerrainRenderer,
    geom: &SegmentGeometry,
) -> (f32, Vec<ArcSample>) {
    let mut samples = Vec::with_capacity(N_SAMPLE + 1);
    let mut cumulative = 0.0f32;
    let (mut prev_x, mut prev_z) = geom.evaluate_xz(0.0);
    let mut prev_y = terrain_renderer.get_height_at([prev_x, prev_z]);

    samples.push(ArcSample {
        t: 0.0,
        cumulative_length: 0.0,
    });

    for i in 1..=N_SAMPLE {
        let t = i as f32 / N_SAMPLE as f32;
        let (x, z) = geom.evaluate_xz(t);
        let y = terrain_renderer.get_height_at([x, z]);

        let dx = x - prev_x;
        let dz = z - prev_z;
        let dy = y - prev_y;
        cumulative += (dx * dx + dy * dy + dz * dz).sqrt();

        samples.push(ArcSample {
            t,
            cumulative_length: cumulative,
        });
        prev_x = x;
        prev_z = z;
        prev_y = y;
    }

    (cumulative, samples)
}

fn arc_length_to_param(samples: &[ArcSample], target_arc: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let total = samples.last().unwrap().cumulative_length;
    if total < 1e-10 {
        return 0.0;
    }

    let target = target_arc.clamp(0.0, total);

    let idx = samples.partition_point(|s| s.cumulative_length < target);
    if idx == 0 {
        return samples[0].t;
    }
    if idx >= samples.len() {
        return samples.last().unwrap().t;
    }

    let s0 = &samples[idx - 1];
    let s1 = &samples[idx];
    let range = s1.cumulative_length - s0.cumulative_length;

    if range < 1e-10 {
        return s0.t;
    }
    lerp(s0.t, s1.t, (target - s0.cumulative_length) / range)
}

pub fn generate_rings(
    terrain_renderer: &TerrainRenderer,
    geom: &SegmentGeometry,
    max_edge_len: f32,
) -> Vec<Ring> {
    let (total_length, samples) = estimate_arc_length(terrain_renderer, geom);
    let n = ((total_length / max_edge_len).ceil() as usize).max(1);
    let mut rings = Vec::with_capacity(n + 1);

    for i in 0..=n {
        let arc_frac = i as f32 / n as f32;
        let arc_target = arc_frac * total_length;
        let t = arc_length_to_param(&samples, arc_target);

        let (x, z) = geom.evaluate_xz(t);
        let y = terrain_renderer.get_height_at([x, z]);

        let tangent_xz = geom.tangent_xz(t);
        let y_slope = if total_length > 1e-10 {
            geom.vertical_slope / total_length
        } else {
            0.0
        };
        let tangent = vec3_normalize([tangent_xz[0], y_slope, tangent_xz[1]]);
        let lateral = [-tangent_xz[1], tangent_xz[0]];

        rings.push(Ring {
            t,
            arc_length: arc_target,
            position: [x, y, z],
            tangent,
            lateral,
        });
    }

    rings
}

// ============================================================================
// Mesh output and config
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

    pub fn merge(&mut self, other: ChunkMesh) {
        let base = self.vertices.len() as u32;
        self.vertices.extend(other.vertices);
        self.indices.extend(other.indices.iter().map(|i| i + base));
    }
}

impl Default for ChunkMesh {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct MeshConfig {
    pub max_segment_edge_length_m: f32,
    pub uv_scale_u: f32,
    pub uv_scale_v: f32,
}

impl Default for MeshConfig {
    fn default() -> Self {
        Self {
            max_segment_edge_length_m: 2.0,
            uv_scale_u: 1.0,
            uv_scale_v: 1.0,
        }
    }
}

// ============================================================================
// Mesh Building - Segments
// ============================================================================

pub fn build_segment_mesh(
    terrain_renderer: &TerrainRenderer,
    rings: &[Ring],
    chunk_filter: Option<ChunkId>,
    cross_section: &CrossSection,
    uv_scale_u: f32,
    uv_scale_v: f32,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    if rings.len() < 2 {
        return;
    }

    let included: Vec<usize> = match chunk_filter {
        Some(chunk_id) => {
            let mut inc = Vec::new();
            for i in 0..rings.len() {
                let r = &rings[i];
                let in_chunk = ring_in_chunk(r, chunk_id);
                let adj_prev = i > 0 && quad_intersects_chunk(&rings[i - 1], r, chunk_id);
                let adj_next =
                    i + 1 < rings.len() && quad_intersects_chunk(r, &rings[i + 1], chunk_id);

                if in_chunk || adj_prev || adj_next {
                    if inc.last().copied() != Some(i) {
                        inc.push(i);
                    }
                }
            }
            inc
        }
        None => (0..rings.len()).collect(),
    };

    if included.len() < 2 {
        return;
    }

    let strips = cross_section.lateral_strips();
    let half_width = cross_section.half_width();

    // Horizontal strip surfaces
    for strip in &strips {
        let base_vertex = vertices.len() as u32;

        for &ring_idx in &included {
            let ring = &rings[ring_idx];
            let txz = glam::Vec3::new(ring.tangent[0], 0.0, ring.tangent[2]).normalize();
            let lateral = glam::Vec3::new(-txz.z, 0.0, txz.x);
            let normal = lateral.cross(txz).normalize();

            for &lat in &[strip.left, strip.right] {
                let terrain_y =
                    terrain_renderer.get_height_at([ring.position[0], ring.position[2]]);
                let pos = glam::Vec3::new(
                    ring.position[0],
                    terrain_y + strip.height + CLEARANCE,
                    ring.position[2],
                ) + lateral * lat;

                vertices.push(RoadVertex {
                    position: pos.to_array(),
                    normal: normal.to_array(),
                    uv: [
                        ring.arc_length * uv_scale_u,
                        (lat + half_width) * uv_scale_v,
                    ],
                    material_id: strip.material_id,
                });
            }
        }

        for i in 0..included.len() - 1 {
            let i0 = base_vertex + (i * 2) as u32;
            indices.extend_from_slice(&[i0, i0 + 1, i0 + 2, i0 + 1, i0 + 3, i0 + 2]);
        }
    }

    // Vertical curb faces
    for i in 0..strips.len() - 1 {
        let current = &strips[i];
        let next = &strips[i + 1];
        let height_diff = current.height - next.height;

        if height_diff.abs() < 0.0001 {
            continue;
        }

        let lat = current.right;
        let higher = current.height.max(next.height);
        let lower = current.height.min(next.height);
        let material_id = if current.height >= next.height {
            current.material_id
        } else {
            next.material_id
        };

        let base_vertex = vertices.len() as u32;

        for &ring_idx in &included {
            let ring = &rings[ring_idx];
            let txz = glam::Vec3::new(ring.tangent[0], 0.0, ring.tangent[2]).normalize();
            let lateral_dir = glam::Vec3::new(-txz.z, 0.0, txz.x);
            let normal = if height_diff > 0.0 {
                lateral_dir
            } else {
                -lateral_dir
            };

            let terrain_y = terrain_renderer.get_height_at([ring.position[0], ring.position[2]]);
            let base_pos =
                glam::Vec3::new(ring.position[0], terrain_y + CLEARANCE, ring.position[2])
                    + lateral_dir * lat;

            let u = ring.arc_length * uv_scale_u;
            let v_height = (higher - lower) * uv_scale_v;

            vertices.push(RoadVertex {
                position: (base_pos + glam::Vec3::Y * higher).to_array(),
                normal: normal.to_array(),
                uv: [u, 0.0],
                material_id,
            });
            vertices.push(RoadVertex {
                position: (base_pos + glam::Vec3::Y * lower).to_array(),
                normal: normal.to_array(),
                uv: [u, v_height],
                material_id,
            });
        }

        for j in 0..included.len() - 1 {
            let i0 = base_vertex + (j * 2) as u32;
            if height_diff > 0.0 {
                indices.extend_from_slice(&[i0, i0 + 1, i0 + 2, i0 + 2, i0 + 1, i0 + 3]);
            } else {
                indices.extend_from_slice(&[i0, i0 + 2, i0 + 1, i0 + 1, i0 + 2, i0 + 3]);
            }
        }
    }
}

// ============================================================================
// Mesh Building - Nodes (all types use quads only)
// ============================================================================

pub fn build_node_mesh(
    terrain_renderer: &TerrainRenderer,
    segment_ring_cache: &HashMap<SegmentId, Vec<Ring>>,
    node_geom: &NodeGeometry,
    cross_section: &CrossSection,
    uv_scale_u: f32,
    uv_scale_v: f32,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    let mut node_type = node_geom.node_type();
    if node_geom.is_preview && node_type == NodeType::End {
        node_type = NodeType::Sole;
    }
    match node_type {
        NodeType::Sole => {
            build_sole_node_mesh(
                terrain_renderer,
                node_geom,
                cross_section,
                uv_scale_u,
                uv_scale_v,
                vertices,
                indices,
            );
        }
        NodeType::End => {
            let Some(segment_id) = node_geom.connections[0].segment_id else {
                return;
            };
            let Some(rings) = segment_ring_cache.get(&segment_id) else {
                return;
            };
            let uv_offset: f32;
            if node_geom.connections[0].node_is_start {
                uv_offset = rings.first().unwrap().arc_length;
            } else {
                uv_offset = rings.last().unwrap().arc_length;
            }
            build_end_cap_mesh(
                terrain_renderer,
                node_geom,
                cross_section,
                uv_scale_u,
                uv_scale_v,
                uv_offset,
                vertices,
                indices,
            );
        }
        NodeType::Through => {
            let Some(segment_id) = node_geom.connections[0].segment_id else {
                return;
            };
            let Some(rings) = segment_ring_cache.get(&segment_id) else {
                return;
            };
            let uv_offset: f32;
            if node_geom.connections[0].node_is_start {
                uv_offset = rings.first().unwrap().arc_length;
            } else {
                uv_offset = rings.last().unwrap().arc_length;
            }
            build_through_node_mesh(
                terrain_renderer,
                node_geom,
                cross_section,
                uv_scale_u,
                uv_scale_v,
                uv_offset,
                vertices,
                indices,
            );
        }
        NodeType::Intersection => {
            build_intersection_mesh(
                terrain_renderer,
                node_geom,
                cross_section,
                uv_scale_u,
                uv_scale_v,
                vertices,
                indices,
            );
        }
    }
}

/// Sole node: full circular disk using quads only, matching segment cross-section exactly
fn build_sole_node_mesh(
    terrain_renderer: &TerrainRenderer,
    node_geom: &NodeGeometry,
    cross_section: &CrossSection,
    uv_scale_u: f32,
    uv_scale_v: f32,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    let center = glam::Vec3::from_array(node_geom.position);
    let strips = cross_section.right_lateral_strips();
    let half_width = cross_section.half_width();

    if strips.is_empty() || half_width < 0.001 {
        return;
    }

    // Generate angles for the full circle
    let angles: Vec<f32> = (0..=NODE_ANGULAR_SEGMENTS)
        .map(|i| (i as f32 / NODE_ANGULAR_SEGMENTS as f32) * std::f32::consts::TAU)
        .collect();

    // Precompute total circumference at reference radius
    let ref_r = half_width;
    let circumference = std::f32::consts::TAU * ref_r;
    let arc_start = angles[0];
    for strip in &strips {
        let inner_r = strip.left.max(0.0);
        let outer_r = strip.right.max(inner_r + 0.001);

        if outer_r <= inner_r {
            continue;
        }

        let base_vertex = vertices.len() as u32;

        for &angle in &angles {
            let delta_angle = angle - arc_start;

            // U: arc length at reference radius, NOT per-vertex radius
            let arc_len = delta_angle * ref_r;
            let u = arc_len * uv_scale_u;

            // V: identical to segment cross-section mapping
            let v_inner = (inner_r + half_width) * uv_scale_v;
            let v_outer = (outer_r + half_width) * uv_scale_v;

            let (cos_a, sin_a) = (angle.cos(), angle.sin());

            // Inner vertex
            let x_inner = center.x + inner_r * cos_a;
            let z_inner = center.z + inner_r * sin_a;
            let terrain_y_inner = terrain_renderer.get_height_at([x_inner, z_inner]);

            vertices.push(RoadVertex {
                position: [x_inner, terrain_y_inner + strip.height + CLEARANCE, z_inner],
                normal: [0.0, 1.0, 0.0],
                uv: [u, v_inner],
                material_id: strip.material_id,
            });

            // Outer vertex
            let x_outer = center.x + outer_r * cos_a;
            let z_outer = center.z + outer_r * sin_a;
            let terrain_y_outer = terrain_renderer.get_height_at([x_outer, z_outer]);

            vertices.push(RoadVertex {
                position: [x_outer, terrain_y_outer + strip.height + CLEARANCE, z_outer],
                normal: [0.0, 1.0, 0.0],
                uv: [u, v_outer],
                material_id: strip.material_id,
            });
        }

        for i in 0..NODE_ANGULAR_SEGMENTS {
            let i0 = base_vertex + (i * 2) as u32;
            indices.extend_from_slice(&[i0, i0 + 2, i0 + 1, i0 + 1, i0 + 2, i0 + 3]);
        }
    }

    // Build curb faces (probably needs similar UV love, but that's another party)
    build_circular_curb_faces(
        terrain_renderer,
        &center,
        &strips,
        half_width,
        &angles,
        uv_scale_u,
        uv_scale_v,
        vertices,
        indices,
    );
}

/// End cap: semicircle at segment end using quads only
// u_offset: arc_length (in world units) at the attachment ring where the cap starts
fn build_end_cap_mesh(
    terrain_renderer: &TerrainRenderer,
    node_geom: &NodeGeometry,
    cross_section: &CrossSection,
    uv_scale_u: f32,
    uv_scale_v: f32,
    u_offset: f32,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    if node_geom.connections.is_empty() {
        return;
    }

    let center = glam::Vec3::from_array(node_geom.position);
    let conn = &node_geom.connections[0];
    let strips = cross_section.lateral_strips();
    let half_width = cross_section.half_width();

    if strips.is_empty() || half_width < 0.001 {
        return;
    }

    let forward = glam::Vec2::new(conn.direction_xz[0], conn.direction_xz[1]);
    let base_angle = forward.y.atan2(forward.x);

    // Determine if we are expanding UVs forward or backward
    // If this node is the start of the segment, the cap goes "behind" the road (U decreases).
    // If this node is the end, the cap goes "after" the road (U increases).
    let uv_direction = if conn.node_is_start { -1.0 } else { 1.0 };

    let half_segments = NODE_ANGULAR_SEGMENTS / 2;
    // Generate angles for a semicircle (PI radians)
    // Adjust the offset range here if your mesh is appearing on the wrong side
    let angles: Vec<f32> = (0..=half_segments)
        .map(|i| {
            let t = i as f32 / half_segments as f32;
            base_angle + std::f32::consts::FRAC_PI_2 + t * std::f32::consts::PI
        })
        .collect();

    for strip in &strips {
        let inner_r = strip.left.max(0.0);
        let outer_r = strip.right.max(inner_r + 0.001);

        if outer_r <= inner_r {
            continue;
        }

        let base_vertex = vertices.len() as u32;

        // Use the first angle as the baseline for arc length = 0
        let arc_start_angle = angles[0];

        for &angle in &angles {
            // [FIX 1] Use OUTER radius for arc length calculation.
            // This prevents the "stretching" look on the outside edge.
            // The inner edge will look compressed (sharper), which is preferred over stretching.
            let arc_len_diff = outer_r * (angle - arc_start_angle).abs();

            // [FIX 2] Apply UV direction logic.
            // Ensures texture continuity from the straight road into the cap.
            let u = (u_offset + (arc_len_diff * uv_direction)) * uv_scale_u;

            // V Mapping
            let v_inner = (inner_r + half_width) * uv_scale_v;
            let v_outer = (outer_r + half_width) * uv_scale_v;

            let (cos_a, sin_a) = (angle.cos(), angle.sin());

            // Inner Vertex
            let x_inner = center.x + inner_r * cos_a;
            let z_inner = center.z + inner_r * sin_a;
            let terrain_y_inner = terrain_renderer.get_height_at([x_inner, z_inner]);
            let inner_pos =
                glam::Vec3::new(x_inner, terrain_y_inner + strip.height + CLEARANCE, z_inner);

            // Outer Vertex
            let x_outer = center.x + outer_r * cos_a;
            let z_outer = center.z + outer_r * sin_a;
            let terrain_y_outer = terrain_renderer.get_height_at([x_outer, z_outer]);
            let outer_pos =
                glam::Vec3::new(x_outer, terrain_y_outer + strip.height + CLEARANCE, z_outer);

            vertices.push(RoadVertex {
                position: inner_pos.to_array(),
                normal: [0.0, 1.0, 0.0],
                uv: [u, v_inner],
                material_id: strip.material_id,
            });

            vertices.push(RoadVertex {
                position: outer_pos.to_array(),
                normal: [0.0, 1.0, 0.0],
                uv: [u, v_outer],
                material_id: strip.material_id,
            });
        }

        for i in 0..half_segments {
            let i0 = base_vertex + (i * 2) as u32;
            indices.extend_from_slice(&[i0, i0 + 2, i0 + 1, i0 + 1, i0 + 2, i0 + 3]);
        }
    }

    // Curb logic (ensure you pass the same u_offset/scale/direction if curbs have texture)
    build_circular_curb_faces(
        terrain_renderer,
        &center,
        &strips,
        half_width,
        &angles,
        uv_scale_u,
        uv_scale_v,
        vertices,
        indices,
    );
}

/// Through node: smooth connection between two segments
fn build_through_node_mesh(
    terrain_renderer: &TerrainRenderer,
    node_geom: &NodeGeometry,
    cross_section: &CrossSection,
    uv_scale_u: f32,
    uv_scale_v: f32,
    u_offset: f32, // <--- ADDED: You must pass the arc length of the incoming segment here
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    if node_geom.connections.len() != 2 {
        return;
    }
    let center = glam::Vec3::from_array(node_geom.position);

    let conn0 = &node_geom.connections[0];
    let conn1 = &node_geom.connections[1];

    // Directions point AWAY from the node along each segment
    let dir0 = glam::Vec2::new(conn0.direction_xz[0], conn0.direction_xz[1]);
    let dir1 = glam::Vec2::new(conn1.direction_xz[0], conn1.direction_xz[1]);

    // Calculate angle between (should be close to 180° for straight through)
    let dot = dir0.dot(dir1);
    let angle_between = (-dot).acos();

    // If nearly straight (< 15 degrees deviation), no fill needed
    if angle_between < 0.0 {
        return;
    }

    let strips = cross_section.lateral_strips();
    let half_width = cross_section.half_width();

    if strips.is_empty() {
        return;
    }

    // Get angles for each direction
    let angle0 = dir0.y.atan2(dir0.x);
    let angle1 = dir1.y.atan2(dir1.x);

    // Determine which side needs the larger fill
    // Calculate the angular gap on each side
    let mut gap_angle_a = angle1 - angle0;
    let mut gap_angle_b = angle0 - angle1;

    // Normalize to [0, 2π)
    while gap_angle_a < 0.0 {
        gap_angle_a += std::f32::consts::TAU;
    }
    while gap_angle_a >= std::f32::consts::TAU {
        gap_angle_a -= std::f32::consts::TAU;
    }
    while gap_angle_b < 0.0 {
        gap_angle_b += std::f32::consts::TAU;
    }
    while gap_angle_b >= std::f32::consts::TAU {
        gap_angle_b -= std::f32::consts::TAU;
    }

    // Build the larger gap (the one > 180°, which is the "outer" curve of the turn)
    let (start_angle, sweep_angle) = if gap_angle_a > std::f32::consts::PI {
        // Gap A is the outer curve
        (
            angle0 + std::f32::consts::FRAC_PI_2,
            gap_angle_a - std::f32::consts::PI,
        )
    } else if gap_angle_b > std::f32::consts::PI {
        // Gap B is the outer curve
        (
            angle1 + std::f32::consts::FRAC_PI_2,
            gap_angle_b - std::f32::consts::PI,
        )
    } else {
        // Both gaps are less than 180°
        return;
    };
    // Number of segments for the curved fill
    let n_curve_segments =
        ((sweep_angle / std::f32::consts::TAU) * NODE_ANGULAR_SEGMENTS as f32).ceil() as usize;
    let n_curve_segments = n_curve_segments.max(4);

    // Generate angles for this arc
    let angles: Vec<f32> = (0..=n_curve_segments)
        .map(|i| start_angle + (i as f32 / n_curve_segments as f32) * sweep_angle)
        .collect();

    for strip in &strips {
        let inner_r = strip.left.max(0.0);
        let outer_r = strip.right.max(inner_r + 0.001);

        if outer_r <= inner_r {
            continue;
        }

        let base_vertex = vertices.len() as u32;
        let arc_start = angles[0];

        for &angle in &angles {
            // [FIX 1] Use Outer Radius for U-coord to prevent stretching
            // We take the absolute difference to ensure positive accumulation
            let arc_len = outer_r * (angle - arc_start).abs();

            // [FIX 2] Add to u_offset for seamless connection
            let u = (u_offset + arc_len) * uv_scale_u;

            // [FIX 3] Correct V mapping
            // Before you had `outer_r` for both. Now we map them to lateral position properly.
            let v_inner = (inner_r + half_width) * uv_scale_v;
            let v_outer = (outer_r + half_width) * uv_scale_v;

            let (cos_a, sin_a) = (angle.cos(), angle.sin());

            let x_inner = center.x + inner_r * cos_a;
            let z_inner = center.z + inner_r * sin_a;
            let terrain_y_inner = terrain_renderer.get_height_at([x_inner, z_inner]);
            let inner_pos =
                glam::Vec3::new(x_inner, terrain_y_inner + strip.height + CLEARANCE, z_inner);

            let x_outer = center.x + outer_r * cos_a;
            let z_outer = center.z + outer_r * sin_a;
            let terrain_y_outer = terrain_renderer.get_height_at([x_outer, z_outer]);
            let outer_pos =
                glam::Vec3::new(x_outer, terrain_y_outer + strip.height + CLEARANCE, z_outer);

            vertices.push(RoadVertex {
                position: inner_pos.to_array(),
                normal: [0.0, 1.0, 0.0],
                uv: [u, v_inner], // Uses shared U, correct V
                material_id: strip.material_id,
            });

            vertices.push(RoadVertex {
                position: outer_pos.to_array(),
                normal: [0.0, 1.0, 0.0],
                uv: [u, v_outer], // Uses shared U, correct V
                material_id: strip.material_id,
            });
        }

        for i in 0..n_curve_segments {
            let i0 = base_vertex + (i * 2) as u32;
            indices.extend_from_slice(&[i0, i0 + 2, i0 + 1, i0 + 1, i0 + 2, i0 + 3]);
        }
    }

    build_circular_curb_faces(
        terrain_renderer,
        &center,
        &strips,
        half_width,
        &angles,
        uv_scale_u,
        uv_scale_v,
        vertices,
        indices,
    );
}

/// Build vertical curb faces between concentric radial strips
fn build_circular_curb_faces(
    terrain_renderer: &TerrainRenderer,
    center: &glam::Vec3,
    strips: &[LateralStrip],
    half_width: f32,
    angles: &[f32],
    uv_scale_u: f32,
    uv_scale_v: f32,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    for i in 0..strips.len() - 1 {
        let current = &strips[i];
        let next = &strips[i + 1];
        let height_diff = current.height - next.height;

        if height_diff.abs() < 0.0001 {
            continue;
        }

        // Map boundary to radius using the same logic as the surface generation
        let radius = current.right.max(0.0);

        if radius < 0.001 {
            continue;
        }

        let higher = current.height.max(next.height);
        let lower = current.height.min(next.height);
        let material_id = if current.height >= next.height {
            current.material_id
        } else {
            next.material_id
        };

        let base_vertex = vertices.len() as u32;
        let arc_start = angles[0];
        for &angle in angles {
            let arc_len = radius * (angle - arc_start);
            let height = higher - lower;
            let (cos_a, sin_a) = (angle.cos(), angle.sin());
            let x = center.x + radius * cos_a;
            let z = center.z + radius * sin_a;
            let terrain_y = terrain_renderer.get_height_at([x, z]);
            let pos = glam::Vec3::new(x, terrain_y + CLEARANCE, z);

            // Determine normal direction based on which side is higher
            // We are moving from inner (current) to outer (next).
            // If current > next, we step DOWN, so the face points OUTWARDS (positive radius).
            // If current < next, we step UP, so the face points INWARDS (negative radius).
            let normal = if height_diff > 0.0 {
                [cos_a, 0.0, sin_a]
            } else {
                [-cos_a, 0.0, -sin_a]
            };

            vertices.push(RoadVertex {
                position: (pos + glam::Vec3::Y * higher).to_array(),
                normal,
                uv: [arc_len * uv_scale_u, height * uv_scale_v],
                material_id,
            });
            vertices.push(RoadVertex {
                position: (pos + glam::Vec3::Y * lower).to_array(),
                normal,
                uv: [arc_len * uv_scale_u, 0.0],
                material_id,
            });
        }

        let segment_count = angles.len() - 1;
        for j in 0..segment_count {
            let i0 = base_vertex + (j * 2) as u32;
            if height_diff > 0.0 {
                indices.extend_from_slice(&[i0, i0 + 2, i0 + 1, i0 + 1, i0 + 2, i0 + 3]);
            } else {
                indices.extend_from_slice(&[i0, i0 + 1, i0 + 2, i0 + 2, i0 + 1, i0 + 3]);
            }
        }
    }
}

/// Intersection mesh: Render as a filled circular node using the concentric quad strips
fn build_intersection_mesh(
    terrain_renderer: &TerrainRenderer,
    node_geom: &NodeGeometry,
    cross_section: &CrossSection,
    uv_scale_u: f32,
    uv_scale_v: f32,
    vertices: &mut Vec<RoadVertex>,
    indices: &mut Vec<u32>,
) {
    // Re-use the sole node mesh logic to create a detailed circular intersection
    // This ensures consistent materials (lanes, sidewalks) rather than a flat disk
    //build_sole_node_mesh(terrain_renderer, node_geom, cross_section, uv_scale, vertices, indices);
}

// ============================================================================
// Bridge Mesh - Stubs for future implementation
// ============================================================================

#[derive(Clone, Debug)]
pub struct BridgeConfig {
    pub pillar_spacing: f32,
    pub deck_thickness: f32,
    pub railing_height: f32,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            pillar_spacing: 20.0,
            deck_thickness: 0.5,
            railing_height: 1.2,
        }
    }
}

pub fn build_bridge_deck_mesh(
    _terrain_renderer: &TerrainRenderer,
    _rings: &[Ring],
    _cross_section: &CrossSection,
    _bridge_config: &BridgeConfig,
    _vertices: &mut Vec<RoadVertex>,
    _indices: &mut Vec<u32>,
) {
    // Stub
}

pub fn build_bridge_pillar_mesh(
    _terrain_renderer: &TerrainRenderer,
    _position: [f32; 3],
    _height: f32,
    _bridge_config: &BridgeConfig,
    _vertices: &mut Vec<RoadVertex>,
    _indices: &mut Vec<u32>,
) {
    // Stub
}

pub fn build_bridge_railing_mesh(
    _rings: &[Ring],
    _cross_section: &CrossSection,
    _bridge_config: &BridgeConfig,
    _vertices: &mut Vec<RoadVertex>,
    _indices: &mut Vec<u32>,
) {
    // Stub
}

// ============================================================================
// Road Mesh Manager
// ============================================================================

pub struct RoadMeshManager {
    chunk_cache: HashMap<ChunkId, ChunkMesh>,
    segment_ring_cache: HashMap<SegmentId, Vec<Ring>>,
    pub config: MeshConfig,
}

impl RoadMeshManager {
    pub fn new(config: MeshConfig) -> Self {
        Self {
            chunk_cache: HashMap::new(),
            segment_ring_cache: HashMap::new(),
            config,
        }
    }

    pub fn get_chunk_mesh(&self, chunk_id: ChunkId) -> Option<&ChunkMesh> {
        self.chunk_cache.get(&chunk_id)
    }

    pub fn invalidate_chunk(&mut self, chunk_id: ChunkId) {
        self.chunk_cache.remove(&chunk_id);
    }

    pub fn invalidate_segment(&mut self, seg_id: SegmentId) {
        self.segment_ring_cache.remove(&seg_id);
    }

    pub fn clear_cache(&mut self) {
        self.chunk_cache.clear();
        self.segment_ring_cache.clear();
    }

    pub fn chunk_needs_update(&self, chunk_id: ChunkId, manager: &RoadManager) -> bool {
        match self.chunk_cache.get(&chunk_id) {
            None => true,
            Some(mesh) => mesh.topo_version != compute_topo_version(chunk_id, manager),
        }
    }

    pub fn build_chunk_mesh(
        &mut self,
        terrain_renderer: &TerrainRenderer,
        chunk_id: ChunkId,
        manager: &RoadManager,
    ) -> ChunkMesh {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Build segment meshes
        let mut segment_ids = manager.segment_ids_touching_chunk(chunk_id);
        segment_ids.sort_unstable();

        for seg_id in segment_ids {
            let segment = manager.segment(seg_id);
            if !segment.enabled {
                continue;
            }

            let rings = self.segment_ring_cache.entry(seg_id).or_insert_with(|| {
                if let Some(geom) = SegmentGeometry::from_segment(seg_id, segment, manager) {
                    generate_rings(
                        terrain_renderer,
                        &geom,
                        self.config.max_segment_edge_length_m,
                    )
                } else {
                    Vec::new()
                }
            });
            let cross_section = CrossSection::from_segment(manager, segment);
            build_segment_mesh(
                terrain_renderer,
                rings,
                Some(chunk_id),
                &cross_section,
                self.config.uv_scale_u,
                self.config.uv_scale_v,
                &mut vertices,
                &mut indices,
            );
        }

        // Build node meshes for nodes in this chunk
        let node_ids = manager.nodes_in_chunk(chunk_id);
        for node_id in node_ids {
            if let Some(node_geom) = NodeGeometry::from_node(node_id, manager) {
                let Some(node) = manager.node(node_id) else {
                    continue;
                };
                let node_connections = manager.segment_count_connected_to_node(node_id);
                let cross_section =
                    CrossSection::from_node(node, node_connections, node_geom.node_type());
                build_node_mesh(
                    terrain_renderer,
                    &self.segment_ring_cache,
                    &node_geom,
                    &cross_section,
                    self.config.uv_scale_u,
                    self.config.uv_scale_v,
                    &mut vertices,
                    &mut indices,
                );
            }
        }

        ChunkMesh {
            vertices,
            indices,
            topo_version: compute_topo_version(chunk_id, manager),
        }
    }

    pub fn build_preview_mesh(
        &self,
        terrain_renderer: &TerrainRenderer,
        preview_state: &RoadPreviewState,
    ) -> ChunkMesh {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for segment_preview in &preview_state.segments {
            if !segment_preview.is_valid {
                continue;
            }

            let geom = SegmentGeometry::from_preview(segment_preview);
            let cross_section = CrossSection::from_preview_segment(segment_preview);
            let rings = generate_rings(
                terrain_renderer,
                &geom,
                self.config.max_segment_edge_length_m,
            );

            build_segment_mesh(
                terrain_renderer,
                &rings,
                None,
                &cross_section,
                self.config.uv_scale_u,
                self.config.uv_scale_v,
                &mut vertices,
                &mut indices,
            );
        }

        for node_preview in &preview_state.nodes {
            if !node_preview.is_valid {
                continue;
            }

            let connected_segments: Vec<&SegmentPreview> = preview_state
                .segments
                .iter()
                .filter(|s| {
                    s.is_valid
                        && ((s.start - node_preview.world_pos).length() < 0.01
                            || (s.end - node_preview.world_pos).length() < 0.01)
                })
                .collect();

            let node_geom = NodeGeometry::from_preview(node_preview, &connected_segments);
            let cross_section = CrossSection::from_node_preview(node_preview);
            build_node_mesh(
                terrain_renderer,
                &self.segment_ring_cache,
                &node_geom,
                &cross_section,
                self.config.uv_scale_u,
                self.config.uv_scale_v,
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

    pub fn update_chunk_mesh(
        &mut self,
        terrain_renderer: &TerrainRenderer,
        chunk_id: ChunkId,
        manager: &RoadManager,
    ) -> &ChunkMesh {
        let mesh = self.build_chunk_mesh(terrain_renderer, chunk_id, manager);
        self.chunk_cache.insert(chunk_id, mesh);
        self.chunk_cache.get(&chunk_id).unwrap()
    }
}

// ============================================================================
// Helper functions
// ============================================================================

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

#[inline]
fn vec2_normalize(v: [f32; 2]) -> [f32; 2] {
    let len = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if len < 1e-10 {
        [1.0, 0.0]
    } else {
        [v[0] / len, v[1] / len]
    }
}

#[inline]
fn vec3_normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        [0.0, 1.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

fn evaluate_arc_xz(p0: [f32; 2], p1: [f32; 2], radius: f32, large_arc: bool, t: f32) -> (f32, f32) {
    let chord = [p1[0] - p0[0], p1[1] - p0[1]];
    let chord_len = (chord[0] * chord[0] + chord[1] * chord[1]).sqrt();
    let abs_radius = radius.abs();

    if chord_len < 1e-10 || abs_radius < chord_len * 0.5 {
        return (lerp(p0[0], p1[0], t), lerp(p0[1], p1[1], t));
    }

    let mid = [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5];
    let chord_dir = [chord[0] / chord_len, chord[1] / chord_len];
    let perp = [-chord_dir[1], chord_dir[0]];

    let half_chord = chord_len * 0.5;
    let h = (abs_radius * abs_radius - half_chord * half_chord)
        .max(0.0)
        .sqrt();
    let sign = if large_arc {
        -radius.signum()
    } else {
        radius.signum()
    };
    let center = [mid[0] + perp[0] * h * sign, mid[1] + perp[1] * h * sign];

    let start_angle = (p0[1] - center[1]).atan2(p0[0] - center[0]);
    let end_angle = (p1[1] - center[1]).atan2(p1[0] - center[0]);

    let pi = std::f32::consts::PI;
    let mut delta = end_angle - start_angle;
    if large_arc {
        if delta.abs() < pi {
            delta += if delta > 0.0 { -2.0 * pi } else { 2.0 * pi };
        }
    } else if delta.abs() > pi {
        delta += if delta > 0.0 { -2.0 * pi } else { 2.0 * pi };
    }

    let angle = start_angle + t * delta;
    (
        center[0] + abs_radius * angle.cos(),
        center[1] + abs_radius * angle.sin(),
    )
}

#[inline]
fn ring_in_chunk(ring: &Ring, chunk_id: ChunkId) -> bool {
    let (min_x, max_x) = chunk_x_range(chunk_id);
    let (min_z, max_z) = chunk_z_range(chunk_id);
    ring.position[0] >= min_x
        && ring.position[0] < max_x
        && ring.position[2] >= min_z
        && ring.position[2] < max_z
}

#[inline]
fn quad_intersects_chunk(r0: &Ring, r1: &Ring, chunk_id: ChunkId) -> bool {
    let (min_x, max_x) = chunk_x_range(chunk_id);
    let (min_z, max_z) = chunk_z_range(chunk_id);

    let seg_min_x = r0.position[0].min(r1.position[0]);
    let seg_max_x = r0.position[0].max(r1.position[0]);
    let seg_min_z = r0.position[2].min(r1.position[2]);
    let seg_max_z = r0.position[2].max(r1.position[2]);

    seg_max_x >= min_x && seg_min_x < max_x && seg_max_z >= min_z && seg_min_z < max_z
}

fn compute_topo_version(chunk_id: ChunkId, manager: &RoadManager) -> u64 {
    let mut hash: u64 = FNV_OFFSET_BASIS;

    let mut segment_ids = manager.segment_ids_touching_chunk(chunk_id);
    segment_ids.sort_unstable();

    for seg_id in &segment_ids {
        let segment = manager.segment(*seg_id);
        hash ^= seg_id.raw() as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= segment.version as u64;
        hash = hash.wrapping_mul(FNV_PRIME);

        if let Some(start) = manager.node(segment.start()) {
            for &c in &[start.x, start.y, start.z] {
                hash ^= c.to_bits() as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
        if let Some(end) = manager.node(segment.end()) {
            for &c in &[end.x, end.y, end.z] {
                hash ^= c.to_bits() as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }
    }

    let node_ids = manager.nodes_in_chunk(chunk_id);
    for node_id in node_ids {
        hash ^= node_id.raw() as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        if let Some(node) = manager.node(node_id) {
            hash ^= node.version();
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }

    hash
}
