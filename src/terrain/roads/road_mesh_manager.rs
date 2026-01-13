//# road_mesh_manager.rs
//! Road Mesh Manager for procedural lane-first citybuilder.
//!
//! Produces deterministic, chunked CPU mesh buffers from immutable road topology.
//! Guarantees:
//! - Identical binary results across runs on same input
//! - Bitwise identical shared vertices at chunk seams
//! - World-space UVs with deterministic arc-length parameterization

use crate::renderer::world_renderer::TerrainRenderer;
use crate::terrain::roads::roads::{RoadManager, Segment, SegmentId};
use std::collections::HashMap;

/// Number of samples for arc-length estimation
const N_SAMPLE: usize = 64;
/// FNV-1a 64-bit offset basis
const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
/// FNV-1a prime multiplier
const FNV_PRIME: u64 = 1099511628211;

pub type ChunkId = u64;

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
    let ux = zigzag_i32(cx);
    let uz = zigzag_i32(cz);
    part1by1(ux) | (part1by1(uz) << 1)
}

#[inline]
pub fn visible_chunks_to_chunk_ids(visible_i32: &[(i32, i32, i32)]) -> Vec<ChunkId> {
    // Preserves input order (already sorted by dist²)
    // Fast: reserves exact capacity, no bounds checks in loop, inlined packing
    // Zero extra allocations beyond the output vec
    let mut ids = Vec::with_capacity(visible_i32.len());
    for &(cx, cz, _dist2) in visible_i32 {
        ids.push(chunk_coord_to_id(cx, cz));
    }
    ids
}

/// Inverse of zigzag encoding
#[inline]
fn unzigzag_u32(v: u32) -> i32 {
    ((v >> 1) as i32) ^ -((v & 1) as i32)
}

/// Inverse of part1by1 - compact every other bit
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

/// Decode Morton ChunkId back to (cx, cz) coordinates
#[inline]
pub fn chunk_id_to_coord(id: ChunkId) -> (i32, i32) {
    let ux = compact1by1(id);
    let uz = compact1by1(id >> 1);
    (unzigzag_u32(ux), unzigzag_u32(uz))
}

/// Fixed chunk X range using proper decoding
pub fn chunk_x_range(chunk_id: ChunkId) -> (f32, f32) {
    let (cx, _cz) = chunk_id_to_coord(chunk_id);
    let min_x = cx as f32 * 64.0;
    (min_x, min_x + 64.0)
}

/// You might also want chunk Z range (unused)
pub fn chunk_z_range(chunk_id: ChunkId) -> (f32, f32) {
    let (_cx, cz) = chunk_id_to_coord(chunk_id);
    let min_z = cz as f32 * 64.0;
    (min_z, min_z + 64.0)
}

/// Vertex format for road mesh. Material ID indexes texture array.
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

/// A single region within a road cross-section (e.g., sidewalk, curb, lane)
#[derive(Clone, Debug, PartialEq)]
pub struct CrossSectionRegion {
    pub width: f32,
    pub height: f32,
    pub material_id: u32,
}
pub struct LateralStrip {
    pub left: f32,
    pub right: f32,
    pub material_id: u32,
    pub height: f32,
}

/// Cross-section definition with multiple lateral regions.
/// Regions are ordered left-to-right.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossSection {
    pub regions: Vec<CrossSectionRegion>,
}

impl CrossSection {
    /// Total width of the cross-section in meters
    #[inline]
    pub fn total_width(&self) -> f32 {
        self.regions.iter().map(|r| r.width).sum()
    }

    /// Left offset from centerline (negative)
    #[inline]
    pub fn left_offset(&self) -> f32 {
        -self.total_width() * 0.5
    }

    /// Right offset from centerline (positive)
    #[inline]
    pub fn right_offset(&self) -> f32 {
        self.total_width() * 0.5
    }

    /// Map lateral offset to region. Returns (region_index, material_id, height).
    /// Deterministic: uses stable left-to-right scan.
    pub fn region_at_offset(&self, lateral: f32) -> (usize, u32, f32) {
        let mut cumulative = self.left_offset();
        for (i, region) in self.regions.iter().enumerate() {
            let next = cumulative + region.width;
            if lateral < next || i == self.regions.len() - 1 {
                return (i, region.material_id, region.height);
            }
            cumulative = next;
        }
        let last = self.regions.len().saturating_sub(1);
        let r = &self.regions[last];
        (last, r.material_id, r.height)
    }

    /// Generate lateral sample points for vertex generation.
    /// Returns Vec of (lateral_offset, material_id, height) in deterministic order.
    pub fn lateral_samples(&self) -> Vec<(f32, u32, f32)> {
        let mut samples = Vec::with_capacity(self.regions.len() + 1);
        let mut cumulative = self.left_offset();

        for region in &self.regions {
            samples.push((cumulative, region.material_id, region.height));
            cumulative += region.width;
        }
        if let Some(last) = self.regions.last() {
            samples.push((cumulative, last.material_id, last.height));
        }
        samples
    }
    pub fn lateral_strips(&self) -> Vec<LateralStrip> {
        let mut strips = Vec::with_capacity(self.regions.len());
        let mut x = self.left_offset();

        for r in &self.regions {
            let left = x;
            let right = x + r.width;
            strips.push(LateralStrip {
                left,
                right,
                material_id: r.material_id,
                height: r.height,
            });
            x = right;
        }

        strips
    }

    pub fn half_width(&self) -> f32 {
        self.left_offset().abs() + self.regions.iter().map(|r| r.width).sum::<f32>()
    }
}

/// Horizontal curve profile for segment centerline
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

// #[inline]
// pub fn evaluate(&self, t: f32) -> f32 {
//     self.start_z + t * (self.end_z - self.start_z)
// }

/// Output mesh for a single chunk
#[derive(Clone, Debug)]
pub struct ChunkMesh {
    pub vertices: Vec<RoadVertex>,
    pub indices: Vec<u32>,
    pub topo_version: u64,
}

/// Ring of vertices at a specific parameter along segment
#[derive(Clone, Debug)]
pub struct Ring {
    pub t: f32,
    pub arc_length: f32,
    pub position: [f32; 3],
    pub tangent: [f32; 3],
    pub lateral: [f32; 2],
}

/// Arc-length sample for parameterization table
#[derive(Clone, Copy, Debug)]
pub struct ArcSample {
    pub t: f32,
    pub cumulative_length: f32,
}

/// Configuration for mesh generation
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

/// Main Road Mesh Manager with per-chunk caching
pub struct RoadMeshManager {
    cache: HashMap<ChunkId, ChunkMesh>,
    config: MeshConfig,
    pub segment_ring_cache: HashMap<SegmentId, Vec<Ring>>,
}

impl RoadMeshManager {
    pub fn new(config: MeshConfig) -> Self {
        Self {
            cache: HashMap::new(),
            config,
            segment_ring_cache: Default::default(),
        }
    }

    pub fn rings_for_segment(
        &mut self,
        manager: &RoadManager,
        seg_id: SegmentId,
        segment: &Segment,
        max_edge_len: f32,
    ) -> &Vec<Ring> {
        self.segment_ring_cache
            .entry(seg_id)
            .or_insert_with(|| generate_rings_for_segment(segment, manager, max_edge_len))
    }

    pub fn get_chunk_mesh(&self, chunk_id: ChunkId) -> Option<&ChunkMesh> {
        self.cache.get(&chunk_id)
    }

    pub fn invalidate_chunk(&mut self, chunk_id: ChunkId) {
        self.cache.remove(&chunk_id);
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Check if chunk needs update by comparing topo_version
    pub fn chunk_needs_update(
        &self,
        chunk_id: ChunkId,
        cross_section: &CrossSection,
        manager: &RoadManager,
    ) -> bool {
        //if manager.segment_ids_touching_chunk(chunk_id).len() == 0 {return false}
        match self.cache.get(&chunk_id) {
            None => true,
            Some(mesh) => {
                let current = compute_topo_version(chunk_id, cross_section, manager);
                mesh.topo_version != current
            }
        }
    }

    /// Update chunk mesh and return reference to cached result
    pub fn update_chunk_mesh(
        &mut self,
        terrain_renderer: &TerrainRenderer,
        chunk_id: ChunkId,
        cross_section: &CrossSection,
        manager: &RoadManager,
    ) -> &ChunkMesh {
        let mesh = self.build_chunk_mesh(
            terrain_renderer,
            chunk_id,
            cross_section,
            manager,
            self.config.max_segment_edge_length_m,
            self.config.uv_scale_u,
            self.config.uv_scale_v,
        );
        self.cache.insert(chunk_id, mesh);
        self.cache.get(&chunk_id).unwrap()
    }

    pub fn config(&self) -> &MeshConfig {
        &self.config
    }
    /// Build chunk mesh from topology. Guarantees deterministic output and seam consistency.
    pub fn build_chunk_mesh(
        &mut self,
        terrain_renderer: &TerrainRenderer,
        chunk_id: ChunkId,
        cross_section: &CrossSection,
        manager: &RoadManager,
        max_edge_len: f32,
        uv_scale_u: f32,
        uv_scale_v: f32,
    ) -> ChunkMesh {
        let mut segment_ids = manager.segment_ids_touching_chunk(chunk_id);
        segment_ids.sort_unstable();

        let strips = cross_section.lateral_strips();
        let half_width = cross_section.half_width();

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for seg_id in segment_ids {
            let segment = manager.segment(seg_id);
            if !segment.enabled {
                continue;
            }

            let all_rings = self.rings_for_segment(manager, seg_id, segment, max_edge_len);
            if all_rings.len() < 2 {
                continue;
            }

            // Determine included rings
            let mut included = Vec::new();
            for i in 0..all_rings.len() {
                let r = &all_rings[i];
                let in_chunk = ring_in_chunk(r, chunk_id);
                let adj_prev = i > 0 && quad_intersects_chunk(&all_rings[i - 1], r, chunk_id);
                let adj_next = i + 1 < all_rings.len()
                    && quad_intersects_chunk(r, &all_rings[i + 1], chunk_id);

                if in_chunk || adj_prev || adj_next {
                    if included.last().copied() != Some(i) {
                        included.push(i);
                    }
                }
            }

            if included.len() < 2 {
                continue;
            }

            const CLEARANCE: f32 = 0.04_f32;

            // === HORIZONTAL STRIP SURFACES ===
            for strip in &strips {
                let base_vertex = vertices.len() as u32;

                // Vertices
                for &ring_idx in &included {
                    let ring = &all_rings[ring_idx];

                    let txz = glam::Vec3::new(ring.tangent[0], 0.0, ring.tangent[1]).normalize();
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

                        let u = ring.arc_length * uv_scale_u;
                        let v = (lat + half_width) * uv_scale_v;

                        vertices.push(RoadVertex {
                            position: pos.to_array(),
                            normal: normal.to_array(),
                            uv: [u, v],
                            material_id: strip.material_id,
                        });
                    }
                }

                // Indices
                let ring_count = included.len();
                for i in 0..ring_count - 1 {
                    let i0 = base_vertex + (i * 2) as u32;
                    let i1 = i0 + 1;
                    let i2 = i0 + 2;
                    let i3 = i0 + 3;

                    indices.extend_from_slice(&[i0, i1, i2, i1, i3, i2]);
                }
            }

            // === VERTICAL CONNECTING FACES (CURBS) ===
            // Connect strips with different heights using vertical faces
            for i in 0..strips.len() - 1 {
                let current_strip = &strips[i];
                let next_strip = &strips[i + 1];

                let height_diff = current_strip.height - next_strip.height;

                // Skip if heights are essentially equal
                if height_diff.abs() < 0.0001 {
                    continue;
                }

                // Lateral position where the two strips meet
                let lat = current_strip.right; // equals next_strip.left

                let higher_height = current_strip.height.max(next_strip.height);
                let lower_height = current_strip.height.min(next_strip.height);

                // Use material from the higher strip (your rule)
                let material_id = if current_strip.height >= next_strip.height {
                    current_strip.material_id
                } else {
                    next_strip.material_id
                };

                let base_vertex = vertices.len() as u32;

                // Generate vertices for the vertical face
                for &ring_idx in &included {
                    let ring = &all_rings[ring_idx];

                    let txz = glam::Vec3::new(ring.tangent[0], 0.0, ring.tangent[1]).normalize();
                    let lateral_dir = glam::Vec3::new(-txz.z, 0.0, txz.x);

                    // Normal points towards the lower side (so curb face is visible from road)
                    let normal = if height_diff > 0.0 {
                        lateral_dir // current/left is higher, normal points right
                    } else {
                        -lateral_dir // next/right is higher, normal points left
                    };

                    let terrain_y =
                        terrain_renderer.get_height_at([ring.position[0], ring.position[2]]);
                    let base_pos =
                        glam::Vec3::new(ring.position[0], terrain_y + CLEARANCE, ring.position[2])
                            + lateral_dir * lat;

                    let pos_high = base_pos + glam::Vec3::Y * higher_height;
                    let pos_low = base_pos + glam::Vec3::Y * lower_height;

                    let u = ring.arc_length * uv_scale_u;
                    let v_height = (higher_height - lower_height) * uv_scale_v;

                    // High vertex
                    vertices.push(RoadVertex {
                        position: pos_high.to_array(),
                        normal: normal.to_array(),
                        uv: [u, 0.0],
                        material_id,
                    });
                    // Low vertex
                    vertices.push(RoadVertex {
                        position: pos_low.to_array(),
                        normal: normal.to_array(),
                        uv: [u, v_height],
                        material_id,
                    });
                }

                // Generate indices for the vertical face
                let ring_count = included.len();
                for j in 0..ring_count - 1 {
                    let i0 = base_vertex + (j * 2) as u32; // ring j, high
                    let i1 = i0 + 1; // ring j, low
                    let i2 = i0 + 2; // ring j+1, high
                    let i3 = i0 + 3; // ring j+1, low

                    // Winding order based on which side is higher
                    // Face should be visible from the lower side
                    if height_diff > 0.0 {
                        // Left is higher, face visible from right
                        indices.extend_from_slice(&[i0, i1, i2, i2, i1, i3]);
                    } else {
                        // Right is higher, face visible from left
                        indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
                    }
                }
            }
        }

        let topo_version = compute_topo_version(chunk_id, cross_section, manager);

        ChunkMesh {
            vertices,
            indices,
            topo_version,
        }
    }
}

// ============================================================================
// Math helpers - inlined for performance
// ============================================================================

#[inline]
fn vec2_sub(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

#[inline]
fn vec2_add(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

#[inline]
fn vec2_scale(v: [f32; 2], s: f32) -> [f32; 2] {
    [v[0] * s, v[1] * s]
}

#[inline]
fn vec2_length(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

#[inline]
fn vec2_normalize(v: [f32; 2]) -> [f32; 2] {
    let len = vec2_length(v);
    if len < 1e-10 {
        [1.0, 0.0]
    } else {
        [v[0] / len, v[1] / len]
    }
}

#[inline]
fn vec2_perpendicular(v: [f32; 2]) -> [f32; 2] {
    [-v[1], v[0]]
}

#[inline]
fn vec3_length(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline]
fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn vec3_normalize(v: [f32; 3]) -> [f32; 3] {
    let len = vec3_length(v);
    if len < 1e-10 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

#[inline]
fn fold_f32_to_u64(f: f32) -> u64 {
    f.to_bits() as u64
}

// ============================================================================
// Curve evaluation
// ============================================================================

/// Evaluate horizontal XY position on segment at parameter t ∈ [0,1].
/// Deterministic: derives position only from node anchors and control points.
pub fn evaluate_horizontal_xz(segment: &Segment, t: f32, manager: &RoadManager) -> (f32, f32) {
    let Some(start) = manager.node(segment.start()) else {
        return (0.0, 0.0);
    };
    let Some(end) = manager.node(segment.end()) else {
        return (0.0, 0.0);
    };

    let p0 = [start.x, start.z];
    let p3 = [end.x, end.z];

    match segment.horizontal_profile {
        HorizontalProfile::Linear => (lerp(p0[0], p3[0], t), lerp(p0[1], p3[1], t)),
        HorizontalProfile::QuadraticBezier { control } => {
            let omt = 1.0 - t;
            let omt2 = omt * omt;
            let t2 = t * t;
            let x = omt2 * p0[0] + 2.0 * omt * t * control[0] + t2 * p3[0];
            let z = omt2 * p0[1] + 2.0 * omt * t * control[1] + t2 * p3[1];
            (x, z)
        }
        HorizontalProfile::CubicBezier { control1, control2 } => {
            let omt = 1.0 - t;
            let omt2 = omt * omt;
            let omt3 = omt2 * omt;
            let t2 = t * t;
            let t3 = t2 * t;
            let x = omt3 * p0[0]
                + 3.0 * omt2 * t * control1[0]
                + 3.0 * omt * t2 * control2[0]
                + t3 * p3[0];
            let z = omt3 * p0[1]
                + 3.0 * omt2 * t * control1[1]
                + 3.0 * omt * t2 * control2[1]
                + t3 * p3[1];
            (x, z)
        }
        HorizontalProfile::Arc { radius, large_arc } => {
            evaluate_arc_xz(p0, p3, radius, large_arc, t)
        }
    }
}

/// Evaluate arc profile. Falls back to linear if geometry is degenerate.
fn evaluate_arc_xz(p0: [f32; 2], p3: [f32; 2], radius: f32, large_arc: bool, t: f32) -> (f32, f32) {
    let chord = vec2_sub(p3, p0);
    let chord_len = vec2_length(chord);
    let abs_radius = radius.abs();

    if chord_len < 1e-10 || abs_radius < chord_len * 0.5 {
        return (lerp(p0[0], p3[0], t), lerp(p0[1], p3[1], t));
    }

    let mid = vec2_scale(vec2_add(p0, p3), 0.5);
    let chord_dir = vec2_normalize(chord);
    let perp = vec2_perpendicular(chord_dir);

    let half_chord = chord_len * 0.5;
    let h_sq = abs_radius * abs_radius - half_chord * half_chord;
    let h = if h_sq > 0.0 { h_sq.sqrt() } else { 0.0 };

    let sign = if large_arc {
        -radius.signum()
    } else {
        radius.signum()
    };
    let center = vec2_add(mid, vec2_scale(perp, h * sign));

    let to_start = vec2_sub(p0, center);
    let to_end = vec2_sub(p3, center);

    let start_angle = to_start[1].atan2(to_start[0]);
    let end_angle = to_end[1].atan2(to_end[0]);

    let mut delta = end_angle - start_angle;
    let pi = std::f32::consts::PI;

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

/// Compute tangent vector at parameter t. Uses analytic derivative where possible.
fn compute_tangent_xz(segment: &Segment, t: f32, manager: &RoadManager) -> [f32; 2] {
    let Some(start) = manager.node(segment.start()) else {
        return [0.0, 0.0];
    };
    let Some(end) = manager.node(segment.end()) else {
        return [0.0, 0.0];
    };

    let p0 = [start.x, start.z];
    let p3 = [end.x, end.z];

    match segment.horizontal_profile {
        HorizontalProfile::Linear => vec2_normalize(vec2_sub(p3, p0)),
        HorizontalProfile::QuadraticBezier { control } => {
            let omt = 1.0 - t;
            let d0 = vec2_sub(control, p0);
            let d1 = vec2_sub(p3, control);
            let dx = 2.0 * omt * d0[0] + 2.0 * t * d1[0];
            let dz = 2.0 * omt * d0[1] + 2.0 * t * d1[1];
            vec2_normalize([dx, dz])
        }
        HorizontalProfile::CubicBezier { control1, control2 } => {
            let omt = 1.0 - t;
            let omt2 = omt * omt;
            let t2 = t * t;
            let d0 = vec2_sub(control1, p0);
            let d1 = vec2_sub(control2, control1);
            let d2 = vec2_sub(p3, control2);
            let dx = 3.0 * omt2 * d0[0] + 6.0 * omt * t * d1[0] + 3.0 * t2 * d2[0];
            let dz = 3.0 * omt2 * d0[1] + 6.0 * omt * t * d1[1] + 3.0 * t2 * d2[1];
            vec2_normalize([dx, dz])
        }
        HorizontalProfile::Arc { .. } => {
            // Finite difference for arc
            let dt = 0.0005;
            let t0 = (t - dt).max(0.0);
            let t1 = (t + dt).min(1.0);
            let (x0, z0) = evaluate_horizontal_xz(segment, t0, manager);
            let (x1, z1) = evaluate_horizontal_xz(segment, t1, manager);
            vec2_normalize([x1 - x0, z1 - z0])
        }
    }
}

// ============================================================================
// Arc-length parameterization
// ============================================================================

/// Estimate total arc length and build sample table for inverse mapping.
/// Uses N_SAMPLE equally spaced t values for deterministic results.
pub fn estimate_arc_length(segment: &Segment, manager: &RoadManager) -> (f32, Vec<ArcSample>) {
    let mut samples = Vec::with_capacity(N_SAMPLE + 1);
    let mut cumulative = 0.0f32;
    let mut prev = evaluate_horizontal_xz(segment, 0.0, manager);
    let mut prev_y = segment.vertical_profile.evaluate(0.0);

    samples.push(ArcSample {
        t: 0.0,
        cumulative_length: 0.0,
    });

    for i in 1..=N_SAMPLE {
        let t = i as f32 / N_SAMPLE as f32;
        let (x, z) = evaluate_horizontal_xz(segment, t, manager);
        let y = segment.vertical_profile.evaluate(t);

        let dx = x - prev.0;
        let dz = z - prev.1;
        let dy = y - prev_y;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        cumulative += dist;
        samples.push(ArcSample {
            t,
            cumulative_length: cumulative,
        });

        prev = (x, z);
        prev_y = y;
    }

    (cumulative, samples)
}

/// Map arc-length to parametric t using binary search with linear interpolation.
/// Deterministic: uses same sample table and interpolation on all calls.
pub fn arc_length_to_param(samples: &[ArcSample], target_arc: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let total = samples.last().unwrap().cumulative_length;
    if total < 1e-10 {
        return 0.0;
    }

    let target = target_arc.clamp(0.0, total);

    // Binary search for first sample with cumulative >= target
    let mut lo = 0usize;
    let mut hi = samples.len();

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if samples[mid].cumulative_length < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    if lo == 0 {
        return samples[0].t;
    }
    if lo >= samples.len() {
        return samples.last().unwrap().t;
    }

    let s0 = &samples[lo - 1];
    let s1 = &samples[lo];
    let range = s1.cumulative_length - s0.cumulative_length;

    if range < 1e-10 {
        return s0.t;
    }

    let frac = (target - s0.cumulative_length) / range;
    lerp(s0.t, s1.t, frac)
}

// ============================================================================
// Ring generation
// ============================================================================

/// Compute surface normal from tangent and lateral.
/// Uses stable cross product to avoid flipping.
fn compute_surface_normal(tangent: [f32; 3], lateral: [f32; 2]) -> [f32; 3] {
    let lateral_3d = [lateral[0], 0.0, lateral[1]];
    let normal = vec3_cross(lateral_3d, tangent);
    let n = vec3_normalize(normal);
    // Ensure normal points generally upward for stability
    if n[2] < 0.0 { [-n[0], -n[1], -n[2]] } else { n }
}

/// Generate rings for a segment with deterministic arc-length subdivision.
pub fn generate_rings_for_segment(
    segment: &Segment,
    manager: &RoadManager,
    max_edge_len: f32,
) -> Vec<Ring> {
    let (total_length, samples) = estimate_arc_length(segment, manager);

    let n = ((total_length / max_edge_len).ceil() as usize).max(1);
    let mut rings = Vec::with_capacity(n + 1);

    for i in 0..=n {
        let arc_frac = i as f32 / n as f32;
        let arc_target = arc_frac * total_length;
        let t = arc_length_to_param(&samples, arc_target);

        let (x, z) = evaluate_horizontal_xz(segment, t, manager);
        let y = segment.vertical_profile.evaluate(t);

        let tangent_xz = compute_tangent_xz(segment, t, manager);
        let y_slope = if total_length > 1e-10 {
            segment.vertical_profile.slope() / total_length
        } else {
            0.0
        };
        let tangent = vec3_normalize([tangent_xz[0], tangent_xz[1], y_slope]);

        let lateral = vec2_perpendicular(tangent_xz);

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
// Chunk geometry
// ============================================================================

#[inline]
fn ring_in_chunk(ring: &Ring, chunk_id: ChunkId) -> bool {
    let (min_x, max_x) = chunk_x_range(chunk_id);
    ring.position[0] >= min_x && ring.position[0] < max_x
}

#[inline]
fn quad_intersects_chunk(r0: &Ring, r1: &Ring, chunk_id: ChunkId) -> bool {
    let (min_x, max_x) = chunk_x_range(chunk_id);
    let seg_min = r0.position[0].min(r1.position[0]);
    let seg_max = r0.position[0].max(r1.position[0]);
    seg_max >= min_x && seg_min < max_x
}

// ============================================================================
// Topology version hashing (FNV-1a)
// ============================================================================

/// Compute deterministic topo_version using FNV-1a 64-bit hash.
/// Incorporates segment IDs, versions, profile params, and node positions.
pub fn compute_topo_version(
    chunk_id: ChunkId,
    _cross_section: &CrossSection,
    manager: &RoadManager,
) -> u64 {
    let mut hash: u64 = FNV_OFFSET_BASIS;

    let mut segment_ids = manager.segment_ids_touching_chunk(chunk_id);
    segment_ids.sort_unstable(); // Deterministic ordering by raw ID

    for seg_id in segment_ids {
        let segment = manager.segment(seg_id); // else { continue };

        // Fold segment ID
        hash ^= seg_id.raw() as u64;
        hash = hash.wrapping_mul(FNV_PRIME);

        // Fold version
        hash ^= segment.version as u64;
        hash = hash.wrapping_mul(FNV_PRIME);

        // Fold horizontal profile
        match segment.horizontal_profile {
            HorizontalProfile::Linear => {
                hash ^= 0u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
            HorizontalProfile::QuadraticBezier { control } => {
                hash ^= 1u64;
                hash = hash.wrapping_mul(FNV_PRIME);
                hash ^= fold_f32_to_u64(control[0]);
                hash = hash.wrapping_mul(FNV_PRIME);
                hash ^= fold_f32_to_u64(control[1]);
                hash = hash.wrapping_mul(FNV_PRIME);
            }
            HorizontalProfile::CubicBezier { control1, control2 } => {
                hash ^= 2u64;
                hash = hash.wrapping_mul(FNV_PRIME);
                hash ^= fold_f32_to_u64(control1[0]);
                hash = hash.wrapping_mul(FNV_PRIME);
                hash ^= fold_f32_to_u64(control1[1]);
                hash = hash.wrapping_mul(FNV_PRIME);
                hash ^= fold_f32_to_u64(control2[0]);
                hash = hash.wrapping_mul(FNV_PRIME);
                hash ^= fold_f32_to_u64(control2[1]);
                hash = hash.wrapping_mul(FNV_PRIME);
            }
            HorizontalProfile::Arc { radius, large_arc } => {
                hash ^= 3u64;
                hash = hash.wrapping_mul(FNV_PRIME);
                hash ^= fold_f32_to_u64(radius);
                hash = hash.wrapping_mul(FNV_PRIME);
                hash ^= large_arc as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
            }
        }

        // Fold node positions
        let Some(start) = manager.node(segment.start()) else {
            return hash;
        };

        for &coord in [&start.x, &start.y, &start.z] {
            hash ^= fold_f32_to_u64(coord);
            hash = hash.wrapping_mul(FNV_PRIME);
        }

        let Some(end) = manager.node(segment.end()) else {
            return hash;
        };
        for &coord in [&end.x, &end.y, &end.z] {
            hash ^= fold_f32_to_u64(coord);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }

    hash
}
