//# road_mesh_manager.rs
//! Road Mesh Manager for procedural lane-first citybuilder.
//!
//! Produces deterministic, chunked CPU mesh buffers from immutable road topology.
//! Guarantees:
//! - Identical binary results across runs on same input
//! - Bitwise identical shared vertices at chunk seams
//! - World-space UVs with deterministic arc-length parameterization

use crate::terrain::roads::roads::{RoadManager, Segment};
use std::collections::HashMap;

/// Number of samples for arc-length estimation
const N_SAMPLE: usize = 64;
/// FNV-1a 64-bit offset basis
const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
/// FNV-1a prime multiplier
const FNV_PRIME: u64 = 1099511628211;
/// Chunk width in meters for X-axis based chunking
const CHUNK_WIDTH: f32 = 100.0;

pub type ChunkId = u64;

/// Vertex format for road mesh. Material ID indexes texture array.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct RoadVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub material_id: u32,
}

/// A single region within a road cross-section (e.g., sidewalk, curb, lane)
#[derive(Clone, Debug, PartialEq)]
pub struct CrossSectionRegion {
    pub width: f32,
    pub height: f32,
    pub material_id: u32,
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
}

impl RoadMeshManager {
    pub fn new(config: MeshConfig) -> Self {
        Self {
            cache: HashMap::new(),
            config,
        }
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
        chunk_id: ChunkId,
        cross_section: &CrossSection,
        manager: &RoadManager,
    ) -> &ChunkMesh {
        let mesh = build_chunk_mesh(
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
pub fn evaluate_horizontal_xy(segment: &Segment, t: f32, manager: &RoadManager) -> (f32, f32) {
    let start = manager.node(segment.start());
    let end = manager.node(segment.end());

    let p0 = [start.x, start.y];
    let p3 = [end.x, end.y];

    match segment.horizontal_profile {
        HorizontalProfile::Linear => (lerp(p0[0], p3[0], t), lerp(p0[1], p3[1], t)),
        HorizontalProfile::QuadraticBezier { control } => {
            let omt = 1.0 - t;
            let omt2 = omt * omt;
            let t2 = t * t;
            let x = omt2 * p0[0] + 2.0 * omt * t * control[0] + t2 * p3[0];
            let y = omt2 * p0[1] + 2.0 * omt * t * control[1] + t2 * p3[1];
            (x, y)
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
            let y = omt3 * p0[1]
                + 3.0 * omt2 * t * control1[1]
                + 3.0 * omt * t2 * control2[1]
                + t3 * p3[1];
            (x, y)
        }
        HorizontalProfile::Arc { radius, large_arc } => {
            evaluate_arc_xy(p0, p3, radius, large_arc, t)
        }
    }
}

/// Evaluate arc profile. Falls back to linear if geometry is degenerate.
fn evaluate_arc_xy(p0: [f32; 2], p3: [f32; 2], radius: f32, large_arc: bool, t: f32) -> (f32, f32) {
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
fn compute_tangent_xy(segment: &Segment, t: f32, manager: &RoadManager) -> [f32; 2] {
    let start = manager.node(segment.start());
    let end = manager.node(segment.end());

    let p0 = [start.x, start.y];
    let p3 = [end.x, end.y];

    match segment.horizontal_profile {
        HorizontalProfile::Linear => vec2_normalize(vec2_sub(p3, p0)),
        HorizontalProfile::QuadraticBezier { control } => {
            let omt = 1.0 - t;
            let d0 = vec2_sub(control, p0);
            let d1 = vec2_sub(p3, control);
            let dx = 2.0 * omt * d0[0] + 2.0 * t * d1[0];
            let dy = 2.0 * omt * d0[1] + 2.0 * t * d1[1];
            vec2_normalize([dx, dy])
        }
        HorizontalProfile::CubicBezier { control1, control2 } => {
            let omt = 1.0 - t;
            let omt2 = omt * omt;
            let t2 = t * t;
            let d0 = vec2_sub(control1, p0);
            let d1 = vec2_sub(control2, control1);
            let d2 = vec2_sub(p3, control2);
            let dx = 3.0 * omt2 * d0[0] + 6.0 * omt * t * d1[0] + 3.0 * t2 * d2[0];
            let dy = 3.0 * omt2 * d0[1] + 6.0 * omt * t * d1[1] + 3.0 * t2 * d2[1];
            vec2_normalize([dx, dy])
        }
        HorizontalProfile::Arc { .. } => {
            // Finite difference for arc
            let dt = 0.0005;
            let t0 = (t - dt).max(0.0);
            let t1 = (t + dt).min(1.0);
            let (x0, y0) = evaluate_horizontal_xy(segment, t0, manager);
            let (x1, y1) = evaluate_horizontal_xy(segment, t1, manager);
            vec2_normalize([x1 - x0, y1 - y0])
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
    let mut prev = evaluate_horizontal_xy(segment, 0.0, manager);
    let mut prev_z = segment.vertical_profile.evaluate(0.0);

    samples.push(ArcSample {
        t: 0.0,
        cumulative_length: 0.0,
    });

    for i in 1..=N_SAMPLE {
        let t = i as f32 / N_SAMPLE as f32;
        let (x, y) = evaluate_horizontal_xy(segment, t, manager);
        let z = segment.vertical_profile.evaluate(t);

        let dx = x - prev.0;
        let dy = y - prev.1;
        let dz = z - prev_z;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        cumulative += dist;
        samples.push(ArcSample {
            t,
            cumulative_length: cumulative,
        });

        prev = (x, y);
        prev_z = z;
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
    let lateral_3d = [lateral[0], lateral[1], 0.0];
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

        let (x, y) = evaluate_horizontal_xy(segment, t, manager);
        let z = segment.vertical_profile.evaluate(t);

        let tangent_xy = compute_tangent_xy(segment, t, manager);
        let z_slope = if total_length > 1e-10 {
            segment.vertical_profile.slope() / total_length
        } else {
            0.0
        };
        let tangent = vec3_normalize([tangent_xy[0], tangent_xy[1], z_slope]);

        let lateral = vec2_perpendicular(tangent_xy);

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

/// Deterministic chunk X range. Used for test/production parity.
pub fn chunk_x_range(chunk_id: ChunkId) -> (f32, f32) {
    let min_x = chunk_id as f32 * CHUNK_WIDTH;
    (min_x, min_x + CHUNK_WIDTH)
}

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
        let node = manager.node(segment.start());
        for &coord in [&node.x, &node.y, &node.z] {
            hash ^= fold_f32_to_u64(coord);
            hash = hash.wrapping_mul(FNV_PRIME);
        }

        let node = manager.node(segment.end());
        for &coord in [&node.x, &node.y, &node.z] {
            hash ^= fold_f32_to_u64(coord);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }

    hash
}

// ============================================================================
// Mesh building
// ============================================================================

/// Build chunk mesh from topology. Guarantees deterministic output and seam consistency.
pub fn build_chunk_mesh(
    chunk_id: ChunkId,
    cross_section: &CrossSection,
    manager: &RoadManager,
    max_edge_len: f32,
    uv_scale_u: f32,
    uv_scale_v: f32,
) -> ChunkMesh {
    let mut segment_ids = manager.segment_ids_touching_chunk(chunk_id);
    segment_ids.sort_unstable();

    let lateral_samples = cross_section.lateral_samples();
    let num_laterals = lateral_samples.len();

    // Pre-allocate with estimates
    let est_rings = 50;
    let est_cap = segment_ids.len() * est_rings * num_laterals;
    let mut vertices: Vec<RoadVertex> = Vec::with_capacity(est_cap);
    let mut indices: Vec<u32> = Vec::with_capacity(est_cap * 6);

    for seg_id in segment_ids {
        let segment = manager.segment(seg_id); //else { continue };
        if !segment.enabled {
            continue;
        }

        let all_rings = generate_rings_for_segment(segment, manager, max_edge_len);
        if all_rings.len() < 2 {
            continue;
        }

        // Determine which rings to include for this chunk
        let mut included: Vec<usize> = Vec::new();
        for (i, ring) in all_rings.iter().enumerate() {
            let in_chunk = ring_in_chunk(ring, chunk_id);
            let adj_prev = i > 0 && quad_intersects_chunk(&all_rings[i - 1], ring, chunk_id);
            let adj_next =
                i + 1 < all_rings.len() && quad_intersects_chunk(ring, &all_rings[i + 1], chunk_id);

            if in_chunk || adj_prev || adj_next {
                if included.last().copied() != Some(i) {
                    included.push(i);
                }
            }
        }

        if included.len() < 2 {
            continue;
        }

        let base_vertex = vertices.len() as u32;

        // Generate vertices for included rings
        for &ring_idx in &included {
            let ring = &all_rings[ring_idx];

            for &(lat_offset, mat_id, height) in &lateral_samples {
                let x = ring.position[0] + ring.lateral[0] * lat_offset;
                let y = ring.position[1] + ring.lateral[1] * lat_offset;
                let z = ring.position[2] + height;

                let normal = compute_surface_normal(ring.tangent, ring.lateral);

                let u = ring.arc_length * uv_scale_u;
                let v = lat_offset * uv_scale_v;

                vertices.push(RoadVertex {
                    position: [x, y, z],
                    normal,
                    uv: [u, v],
                    material_id: mat_id,
                });
            }
        }

        // Generate indices (CCW winding)
        let ring_count = included.len();
        for i in 0..(ring_count - 1) {
            for j in 0..(num_laterals - 1) {
                let v0 = base_vertex + (i * num_laterals + j) as u32;
                let v1 = base_vertex + (i * num_laterals + j + 1) as u32;
                let v2 = base_vertex + ((i + 1) * num_laterals + j) as u32;
                let v3 = base_vertex + ((i + 1) * num_laterals + j + 1) as u32;

                // First triangle
                indices.push(v0);
                indices.push(v2);
                indices.push(v1);

                // Second triangle
                indices.push(v1);
                indices.push(v2);
                indices.push(v3);
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

// ============================================================================
// Tests
// ============================================================================

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use std::collections::HashMap;
//     use crate::terrain::roads::roads::{NodeId, SegmentId};
//     fn create_simple_cross_section() -> CrossSection {
//         CrossSection {
//             regions: vec![CrossSectionRegion {
//                 width: 10.0,
//                 height: 0.0,
//                 material_id: 0,
//             }],
//         }
//     }
//
//     fn create_multi_region_cross_section() -> CrossSection {
//         CrossSection {
//             regions: vec![
//                 CrossSectionRegion {
//                     width: 2.0,
//                     height: 0.15,
//                     material_id: 1, // sidewalk
//                 },
//                 CrossSectionRegion {
//                     width: 0.5,
//                     height: 0.1,
//                     material_id: 2, // curb
//                 },
//                 CrossSectionRegion {
//                     width: 7.0,
//                     height: 0.0,
//                     material_id: 0, // road
//                 },
//                 CrossSectionRegion {
//                     width: 0.5,
//                     height: 0.1,
//                     material_id: 2, // curb
//                 },
//                 CrossSectionRegion {
//                     width: 2.0,
//                     height: 0.15,
//                     material_id: 1, // sidewalk
//                 },
//             ],
//         }
//     }
//
//     // ============================================================================
//     // CrossSection Tests
//     // ============================================================================
//
//     #[test]
//     fn test_cross_section_total_width_single_region() {
//         let cs = CrossSection {
//             regions: vec![CrossSectionRegion {
//                 width: 10.0,
//                 height: 0.0,
//                 material_id: 0,
//             }],
//         };
//         assert!((cs.total_width() - 10.0).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_cross_section_total_width_multiple_regions() {
//         let cs = create_multi_region_cross_section();
//         let expected = 2.0 + 0.5 + 7.0 + 0.5 + 2.0;
//         assert!((cs.total_width() - expected).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_cross_section_total_width_empty() {
//         let cs = CrossSection { regions: vec![] };
//         assert!((cs.total_width() - 0.0).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_cross_section_left_right_offsets_symmetric() {
//         let cs = create_simple_cross_section();
//         assert!((cs.left_offset() + cs.right_offset()).abs() < 1e-6);
//         assert!((cs.left_offset() - (-5.0)).abs() < 1e-6);
//         assert!((cs.right_offset() - 5.0).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_cross_section_region_at_offset_center() {
//         let cs = create_simple_cross_section();
//         let (idx, mat, height) = cs.region_at_offset(0.0);
//         assert_eq!(idx, 0);
//         assert_eq!(mat, 0);
//         assert!((height - 0.0).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_cross_section_region_at_offset_left_edge() {
//         let cs = create_multi_region_cross_section();
//         let left = cs.left_offset();
//         let (idx, mat, height) = cs.region_at_offset(left + 0.001);
//         assert_eq!(idx, 0);
//         assert_eq!(mat, 1); // sidewalk
//         assert!((height - 0.15).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_cross_section_region_at_offset_right_edge() {
//         let cs = create_multi_region_cross_section();
//         let right = cs.right_offset();
//         let (idx, mat, _) = cs.region_at_offset(right - 0.001);
//         assert_eq!(idx, 4); // last region
//         assert_eq!(mat, 1); // sidewalk
//     }
//
//     #[test]
//     fn test_cross_section_region_at_offset_boundary() {
//         let cs = create_multi_region_cross_section();
//         // At boundary between sidewalk (2.0) and curb (0.5)
//         let boundary = cs.left_offset() + 2.0;
//         let (idx, mat, _) = cs.region_at_offset(boundary);
//         assert_eq!(idx, 1); // curb
//         assert_eq!(mat, 2);
//     }
//
//     #[test]
//     fn test_cross_section_lateral_samples_count() {
//         let cs = create_multi_region_cross_section();
//         let samples = cs.lateral_samples();
//         // Should be regions.len() + 1
//         assert_eq!(samples.len(), 6);
//     }
//
//     #[test]
//     fn test_cross_section_lateral_samples_order() {
//         let cs = create_multi_region_cross_section();
//         let samples = cs.lateral_samples();
//
//         // Verify samples are in ascending order
//         for i in 1..samples.len() {
//             assert!(
//                 samples[i].0 > samples[i - 1].0,
//                 "Samples should be in ascending lateral order"
//             );
//         }
//     }
//
//     #[test]
//     fn test_cross_section_lateral_samples_span() {
//         let cs = create_simple_cross_section();
//         let samples = cs.lateral_samples();
//
//         assert!((samples.first().unwrap().0 - cs.left_offset()).abs() < 1e-6);
//         assert!((samples.last().unwrap().0 - cs.right_offset()).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_cross_section_lateral_samples_deterministic() {
//         let cs = create_multi_region_cross_section();
//         let samples1 = cs.lateral_samples();
//         let samples2 = cs.lateral_samples();
//
//         assert_eq!(samples1.len(), samples2.len());
//         for (s1, s2) in samples1.iter().zip(samples2.iter()) {
//             assert_eq!(s1.0.to_bits(), s2.0.to_bits());
//             assert_eq!(s1.1, s2.1);
//             assert_eq!(s1.2.to_bits(), s2.2.to_bits());
//         }
//     }
//
//     // ============================================================================
//     // VerticalProfile Tests
//     // ============================================================================
//
//     #[test]
//     fn test_vertical_profile_evaluate_endpoints() {
//         let vp = VerticalProfile {
//             start_z: 10.0,
//             end_z: 20.0,
//         };
//         assert!((vp.evaluate(0.0) - 10.0).abs() < 1e-6);
//         assert!((vp.evaluate(1.0) - 20.0).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_vertical_profile_evaluate_midpoint() {
//         let vp = VerticalProfile {
//             start_z: 0.0,
//             end_z: 100.0,
//         };
//         assert!((vp.evaluate(0.5) - 50.0).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_vertical_profile_evaluate_negative() {
//         let vp = VerticalProfile {
//             start_z: 50.0,
//             end_z: -50.0,
//         };
//         assert!((vp.evaluate(0.5) - 0.0).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_vertical_profile_slope_positive() {
//         let vp = VerticalProfile {
//             start_z: 0.0,
//             end_z: 10.0,
//         };
//         assert!((vp.slope() - 10.0).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_vertical_profile_slope_negative() {
//         let vp = VerticalProfile {
//             start_z: 10.0,
//             end_z: 0.0,
//         };
//         assert!((vp.slope() - (-10.0)).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_vertical_profile_slope_zero() {
//         let vp = VerticalProfile {
//             start_z: 5.0,
//             end_z: 5.0,
//         };
//         assert!((vp.slope() - 0.0).abs() < 1e-6);
//     }
//
//     // ============================================================================
//     // Curve Evaluation Tests
//     // ============================================================================
//
//     #[test]
//     fn test_evaluate_horizontal_linear_endpoints() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [100.0, 0.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let segment = manager.segment(SegmentId(1)).unwrap();
//
//         let (x0, y0) = evaluate_horizontal_xy(segment, 0.0, &manager);
//         assert!((x0 - 0.0).abs() < 1e-6);
//         assert!((y0 - 0.0).abs() < 1e-6);
//
//         let (x1, y1) = evaluate_horizontal_xy(segment, 1.0, &manager);
//         assert!((x1 - 100.0).abs() < 1e-6);
//         assert!((y1 - 0.0).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_evaluate_horizontal_linear_midpoint() {
//         let mut manager = RoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [100.0, 50.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let segment = manager.segment(SegmentId(1)).unwrap();
//         let (x, y) = evaluate_horizontal_xy(segment, 0.5, &manager);
//         assert!((x - 50.0).abs() < 1e-6);
//         assert!((y - 25.0).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_evaluate_horizontal_quadratic_bezier_endpoints() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [100.0, 0.0, 0.0]);
//
//         let segment = Segment {
//             id: SegmentId(1),
//             start_node: NodeId(1),
//             end_node: NodeId(2),
//             horizontal_profile: HorizontalProfile::QuadraticBezier {
//                 control: [50.0, 50.0],
//             },
//             vertical_profile: VerticalProfile {
//                 start_z: 0.0,
//                 end_z: 0.0,
//             },
//             version: 1,
//             enabled: true,
//         };
//         manager.add_segment(segment);
//
//         let seg = manager.segment(SegmentId(1)).unwrap();
//
//         let (x0, y0) = evaluate_horizontal_xy(seg, 0.0, &manager);
//         assert!((x0 - 0.0).abs() < 1e-6);
//         assert!((y0 - 0.0).abs() < 1e-6);
//
//         let (x1, y1) = evaluate_horizontal_xy(seg, 1.0, &manager);
//         assert!((x1 - 100.0).abs() < 1e-6);
//         assert!((y1 - 0.0).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_evaluate_horizontal_quadratic_bezier_midpoint() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [100.0, 0.0, 0.0]);
//
//         let segment = Segment {
//             id: SegmentId(1),
//             start_node: NodeId(1),
//             end_node: NodeId(2),
//             horizontal_profile: HorizontalProfile::QuadraticBezier {
//                 control: [50.0, 50.0],
//             },
//             vertical_profile: VerticalProfile {
//                 start_z: 0.0,
//                 end_z: 0.0,
//             },
//             version: 1,
//             enabled: true,
//         };
//         manager.add_segment(segment);
//
//         let seg = manager.segment(SegmentId(1)).unwrap();
//         let (x, y) = evaluate_horizontal_xy(seg, 0.5, &manager);
//         // At t=0.5: (1-t)^2*p0 + 2*(1-t)*t*c + t^2*p3
//         // = 0.25*0 + 0.5*50 + 0.25*100 = 50
//         // y = 0.25*0 + 0.5*50 + 0.25*0 = 25
//         assert!((x - 50.0).abs() < 1e-6);
//         assert!((y - 25.0).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_evaluate_horizontal_cubic_bezier_endpoints() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [100.0, 0.0, 0.0]);
//
//         let segment = Segment {
//             id: SegmentId(1),
//             start_node: NodeId(1),
//             end_node: NodeId(2),
//             horizontal_profile: HorizontalProfile::CubicBezier {
//                 control1: [25.0, 50.0],
//                 control2: [75.0, 50.0],
//             },
//             vertical_profile: VerticalProfile {
//                 start_z: 0.0,
//                 end_z: 0.0,
//             },
//             version: 1,
//             enabled: true,
//         };
//         manager.add_segment(segment);
//
//         let seg = manager.segment(SegmentId(1)).unwrap();
//
//         let (x0, y0) = evaluate_horizontal_xy(seg, 0.0, &manager);
//         assert!((x0 - 0.0).abs() < 1e-6);
//         assert!((y0 - 0.0).abs() < 1e-6);
//
//         let (x1, y1) = evaluate_horizontal_xy(seg, 1.0, &manager);
//         assert!((x1 - 100.0).abs() < 1e-6);
//         assert!((y1 - 0.0).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_evaluate_horizontal_arc_endpoints() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [100.0, 0.0, 0.0]);
//
//         let segment = Segment {
//             id: SegmentId(1),
//             start_node: NodeId(1),
//             end_node: NodeId(2),
//             horizontal_profile: HorizontalProfile::Arc {
//                 radius: 100.0,
//                 large_arc: false,
//             },
//             vertical_profile: VerticalProfile {
//                 start_z: 0.0,
//                 end_z: 0.0,
//             },
//             version: 1,
//             enabled: true,
//         };
//         manager.add_segment(segment);
//
//         let seg = manager.segment(SegmentId(1)).unwrap();
//
//         let (x0, y0) = evaluate_horizontal_xy(seg, 0.0, &manager);
//         assert!((x0 - 0.0).abs() < 1e-3);
//         assert!((y0 - 0.0).abs() < 1e-3);
//
//         let (x1, y1) = evaluate_horizontal_xy(seg, 1.0, &manager);
//         assert!((x1 - 100.0).abs() < 1e-3);
//         assert!((y1 - 0.0).abs() < 1e-3);
//     }
//
//     #[test]
//     fn test_evaluate_horizontal_arc_midpoint_bulges() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [100.0, 0.0, 0.0]);
//
//         let segment = Segment {
//             id: SegmentId(1),
//             start_node: NodeId(1),
//             end_node: NodeId(2),
//             horizontal_profile: HorizontalProfile::Arc {
//                 radius: 100.0,
//                 large_arc: false,
//             },
//             vertical_profile: VerticalProfile {
//                 start_z: 0.0,
//                 end_z: 0.0,
//             },
//             version: 1,
//             enabled: true,
//         };
//         manager.add_segment(segment);
//
//         let seg = manager.segment(SegmentId(1)).unwrap();
//         let (_, y) = evaluate_horizontal_xy(seg, 0.5, &manager);
//         // Arc should bulge away from the chord
//         assert!(y.abs() > 1.0, "Arc midpoint should bulge from chord");
//     }
//
//     #[test]
//     fn test_evaluate_horizontal_arc_degenerate_fallback_to_linear() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [100.0, 0.0, 0.0]);
//
//         // Radius smaller than half chord length - degenerate
//         let segment = Segment {
//             id: SegmentId(1),
//             start_node: NodeId(1),
//             end_node: NodeId(2),
//             horizontal_profile: HorizontalProfile::Arc {
//                 radius: 10.0, // Too small for chord of 100
//                 large_arc: false,
//             },
//             vertical_profile: VerticalProfile {
//                 start_z: 0.0,
//                 end_z: 0.0,
//             },
//             version: 1,
//             enabled: true,
//         };
//         manager.add_segment(segment);
//
//         let seg = manager.segment(SegmentId(1)).unwrap();
//         let (x, y) = evaluate_horizontal_xy(seg, 0.5, &manager);
//         // Should fall back to linear
//         assert!((x - 50.0).abs() < 1e-3);
//         assert!((y - 0.0).abs() < 1e-3);
//     }
//
//     #[test]
//     fn test_compute_tangent_linear() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [100.0, 0.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let segment = manager.segment(SegmentId(1)).unwrap();
//         let tangent = compute_tangent_xy(segment, 0.5, &manager);
//         assert!((tangent[0] - 1.0).abs() < 1e-6);
//         assert!((tangent[1] - 0.0).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_compute_tangent_diagonal() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [100.0, 100.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let segment = manager.segment(SegmentId(1)).unwrap();
//         let tangent = compute_tangent_xy(segment, 0.5, &manager);
//         let expected = 1.0 / 2.0f32.sqrt();
//         assert!((tangent[0] - expected).abs() < 1e-6);
//         assert!((tangent[1] - expected).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_compute_tangent_bezier_varies_along_curve() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [100.0, 0.0, 0.0]);
//
//         let segment = Segment {
//             id: SegmentId(1),
//             start_node: NodeId(1),
//             end_node: NodeId(2),
//             horizontal_profile: HorizontalProfile::QuadraticBezier {
//                 control: [50.0, 100.0],
//             },
//             vertical_profile: VerticalProfile {
//                 start_z: 0.0,
//                 end_z: 0.0,
//             },
//             version: 1,
//             enabled: true,
//         };
//         manager.add_segment(segment);
//
//         let seg = manager.segment(SegmentId(1)).unwrap();
//         let t0 = compute_tangent_xy(seg, 0.0, &manager);
//         let t1 = compute_tangent_xy(seg, 1.0, &manager);
//
//         // Tangent at start should point toward control point
//         assert!(t0[1] > 0.0, "Start tangent should have positive y");
//         // Tangent at end should point away from control point
//         assert!(t1[1] < 0.0, "End tangent should have negative y");
//     }
//
//     // ============================================================================
//     // Arc Length Tests
//     // ============================================================================
//
//     #[test]
//     fn test_estimate_arc_length_linear() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [100.0, 0.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let segment = manager.segment(SegmentId(1)).unwrap();
//         let (total, samples) = estimate_arc_length(segment, &manager);
//
//         assert!((total - 100.0).abs() < 0.1);
//         assert_eq!(samples.len(), N_SAMPLE + 1);
//     }
//
//     #[test]
//     fn test_estimate_arc_length_with_elevation() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [30.0, 0.0, 0.0]);
//
//         let segment = Segment {
//             id: SegmentId(1),
//             start_node: NodeId(1),
//             end_node: NodeId(2),
//             horizontal_profile: HorizontalProfile::Linear,
//             vertical_profile: VerticalProfile {
//                 start_z: 0.0,
//                 end_z: 40.0, // 3-4-5 triangle: sqrt(30^2 + 40^2) = 50
//             },
//             version: 1,
//             enabled: true,
//         };
//         manager.add_segment(segment);
//
//         let seg = manager.segment(SegmentId(1)).unwrap();
//         let (total, _) = estimate_arc_length(seg, &manager);
//
//         assert!((total - 50.0).abs() < 0.5);
//     }
//
//     #[test]
//     fn test_estimate_arc_length_samples_monotonic() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [100.0, 50.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let segment = manager.segment(SegmentId(1)).unwrap();
//         let (_, samples) = estimate_arc_length(segment, &manager);
//
//         for i in 1..samples.len() {
//             assert!(
//                 samples[i].cumulative_length >= samples[i - 1].cumulative_length,
//                 "Arc length samples must be monotonically increasing"
//             );
//             assert!(
//                 samples[i].t >= samples[i - 1].t,
//                 "Parameter t must be monotonically increasing"
//             );
//         }
//     }
//
//     #[test]
//     fn test_estimate_arc_length_curved_longer_than_chord() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [100.0, 0.0, 0.0]);
//
//         let segment = Segment {
//             id: SegmentId(1),
//             start_node: NodeId(1),
//             end_node: NodeId(2),
//             horizontal_profile: HorizontalProfile::QuadraticBezier {
//                 control: [50.0, 100.0],
//             },
//             vertical_profile: VerticalProfile {
//                 start_z: 0.0,
//                 end_z: 0.0,
//             },
//             version: 1,
//             enabled: true,
//         };
//         manager.add_segment(segment);
//
//         let seg = manager.segment(SegmentId(1)).unwrap();
//         let (total, _) = estimate_arc_length(seg, &manager);
//
//         // Curved path should be longer than straight-line distance of 100
//         assert!(total > 100.0);
//     }
//
//     #[test]
//     fn test_arc_length_to_param_endpoints() {
//         let samples = vec![
//             ArcSample {
//                 t: 0.0,
//                 cumulative_length: 0.0,
//             },
//             ArcSample {
//                 t: 0.5,
//                 cumulative_length: 50.0,
//             },
//             ArcSample {
//                 t: 1.0,
//                 cumulative_length: 100.0,
//             },
//         ];
//
//         assert!((arc_length_to_param(&samples, 0.0) - 0.0).abs() < 1e-6);
//         assert!((arc_length_to_param(&samples, 100.0) - 1.0).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_arc_length_to_param_midpoint() {
//         let samples = vec![
//             ArcSample {
//                 t: 0.0,
//                 cumulative_length: 0.0,
//             },
//             ArcSample {
//                 t: 0.5,
//                 cumulative_length: 50.0,
//             },
//             ArcSample {
//                 t: 1.0,
//                 cumulative_length: 100.0,
//             },
//         ];
//
//         assert!((arc_length_to_param(&samples, 50.0) - 0.5).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_arc_length_to_param_interpolation() {
//         let samples = vec![
//             ArcSample {
//                 t: 0.0,
//                 cumulative_length: 0.0,
//             },
//             ArcSample {
//                 t: 0.5,
//                 cumulative_length: 50.0,
//             },
//             ArcSample {
//                 t: 1.0,
//                 cumulative_length: 100.0,
//             },
//         ];
//
//         let t = arc_length_to_param(&samples, 25.0);
//         assert!((t - 0.25).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_arc_length_to_param_clamping() {
//         let samples = vec![
//             ArcSample {
//                 t: 0.0,
//                 cumulative_length: 0.0,
//             },
//             ArcSample {
//                 t: 1.0,
//                 cumulative_length: 100.0,
//             },
//         ];
//
//         // Negative arc length should clamp to 0
//         assert!((arc_length_to_param(&samples, -50.0) - 0.0).abs() < 1e-6);
//         // Arc length beyond total should clamp to 1
//         assert!((arc_length_to_param(&samples, 150.0) - 1.0).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_arc_length_to_param_empty() {
//         let samples: Vec<ArcSample> = vec![];
//         assert!((arc_length_to_param(&samples, 50.0) - 0.0).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_arc_length_to_param_zero_length() {
//         let samples = vec![
//             ArcSample {
//                 t: 0.0,
//                 cumulative_length: 0.0,
//             },
//             ArcSample {
//                 t: 1.0,
//                 cumulative_length: 0.0,
//             },
//         ];
//         assert!((arc_length_to_param(&samples, 0.0) - 0.0).abs() < 1e-6);
//     }
//
//     // ============================================================================
//     // Ring Generation Tests
//     // ============================================================================
//
//     #[test]
//     fn test_generate_rings_minimum_two() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [1.0, 0.0, 0.0]); // Very short segment
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let segment = manager.segment(SegmentId(1)).unwrap();
//         let rings = generate_rings_for_segment(segment, &manager, 100.0); // Large edge length
//
//         assert!(rings.len() >= 2, "Must have at least start and end rings");
//     }
//
//     #[test]
//     fn test_generate_rings_subdivision() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [100.0, 0.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let segment = manager.segment(SegmentId(1)).unwrap();
//         let rings = generate_rings_for_segment(segment, &manager, 10.0);
//
//         // 100m segment with 10m max edge should have ~11 rings
//         assert!(rings.len() >= 10);
//     }
//
//     #[test]
//     fn test_generate_rings_arc_length_spacing() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [100.0, 0.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let segment = manager.segment(SegmentId(1)).unwrap();
//         let rings = generate_rings_for_segment(segment, &manager, 10.0);
//
//         // Check arc lengths are evenly spaced
//         for i in 1..rings.len() {
//             let spacing = rings[i].arc_length - rings[i - 1].arc_length;
//             let expected_spacing = 100.0 / (rings.len() - 1) as f32;
//             assert!(
//                 (spacing - expected_spacing).abs() < 1.0,
//                 "Ring arc lengths should be evenly spaced"
//             );
//         }
//     }
//
//     #[test]
//     fn test_generate_rings_position_matches_arc_length() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [100.0, 0.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let segment = manager.segment(SegmentId(1)).unwrap();
//         let rings = generate_rings_for_segment(segment, &manager, 10.0);
//
//         // First ring at origin
//         assert!((rings[0].position[0] - 0.0).abs() < 1e-3);
//
//         // Last ring at end
//         let last = rings.last().unwrap();
//         assert!((last.position[0] - 100.0).abs() < 1e-3);
//     }
//
//     #[test]
//     fn test_generate_rings_tangent_normalized() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [100.0, 50.0, 25.0]);
//
//         let segment = Segment {
//             id: SegmentId(1),
//             start_node: NodeId(1),
//             end_node: NodeId(2),
//             horizontal_profile: HorizontalProfile::Linear,
//             vertical_profile: VerticalProfile {
//                 start_z: 0.0,
//                 end_z: 25.0,
//             },
//             version: 1,
//             enabled: true,
//         };
//         manager.add_segment(segment);
//
//         let seg = manager.segment(SegmentId(1)).unwrap();
//         let rings = generate_rings_for_segment(seg, &manager, 10.0);
//
//         for ring in &rings {
//             let len = vec3_length(ring.tangent);
//             assert!(
//                 (len - 1.0).abs() < 1e-5,
//                 "Tangent must be unit length"
//             );
//         }
//     }
//
//     #[test]
//     fn test_generate_rings_lateral_perpendicular_to_tangent() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [100.0, 50.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let segment = manager.segment(SegmentId(1)).unwrap();
//         let rings = generate_rings_for_segment(segment, &manager, 10.0);
//
//         for ring in &rings {
//             let dot = ring.tangent[0] * ring.lateral[0] + ring.tangent[1] * ring.lateral[1];
//             assert!(
//                 dot.abs() < 1e-5,
//                 "Lateral must be perpendicular to tangent in XY"
//             );
//         }
//     }
//
//     // ============================================================================
//     // Chunk Range Tests
//     // ============================================================================
//
//     #[test]
//     fn test_chunk_x_range() {
//         let (min, max) = chunk_x_range(0);
//         assert!((min - 0.0).abs() < 1e-6);
//         assert!((max - CHUNK_WIDTH).abs() < 1e-6);
//
//         let (min, max) = chunk_x_range(5);
//         assert!((min - 500.0).abs() < 1e-6);
//         assert!((max - 600.0).abs() < 1e-6);
//     }
//
//     #[test]
//     fn test_ring_in_chunk() {
//         let ring_in = Ring {
//             t: 0.5,
//             arc_length: 50.0,
//             position: [50.0, 0.0, 0.0],
//             tangent: [1.0, 0.0, 0.0],
//             lateral: [0.0, 1.0],
//         };
//
//         let ring_out = Ring {
//             t: 0.5,
//             arc_length: 50.0,
//             position: [150.0, 0.0, 0.0],
//             tangent: [1.0, 0.0, 0.0],
//             lateral: [0.0, 1.0],
//         };
//
//         assert!(ring_in_chunk(&ring_in, 0));
//         assert!(!ring_in_chunk(&ring_out, 0));
//         assert!(ring_in_chunk(&ring_out, 1));
//     }
//
//     #[test]
//     fn test_ring_in_chunk_boundary() {
//         // Exactly at chunk boundary
//         let ring_at_boundary = Ring {
//             t: 1.0,
//             arc_length: 100.0,
//             position: [100.0, 0.0, 0.0], // Exactly at x=100
//             tangent: [1.0, 0.0, 0.0],
//             lateral: [0.0, 1.0],
//         };
//
//         // x >= min_x && x < max_x, so x=100 is NOT in chunk 0 (range [0, 100))
//         assert!(!ring_in_chunk(&ring_at_boundary, 0));
//         // But it IS in chunk 1 (range [100, 200))
//         assert!(ring_in_chunk(&ring_at_boundary, 1));
//     }
//
//     #[test]
//     fn test_quad_intersects_chunk() {
//         let r0 = Ring {
//             t: 0.0,
//             arc_length: 0.0,
//             position: [90.0, 0.0, 0.0],
//             tangent: [1.0, 0.0, 0.0],
//             lateral: [0.0, 1.0],
//         };
//         let r1 = Ring {
//             t: 0.1,
//             arc_length: 10.0,
//             position: [110.0, 0.0, 0.0],
//             tangent: [1.0, 0.0, 0.0],
//             lateral: [0.0, 1.0],
//         };
//
//         // Quad from 90 to 110 should intersect both chunk 0 and chunk 1
//         assert!(quad_intersects_chunk(&r0, &r1, 0));
//         assert!(quad_intersects_chunk(&r0, &r1, 1));
//
//         // But not chunk 2
//         assert!(!quad_intersects_chunk(&r0, &r1, 2));
//     }
//
//     // ============================================================================
//     // Topology Version Tests
//     // ============================================================================
//
//     #[test]
//     fn test_compute_topo_version_deterministic() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [50.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [150.0, 0.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let cs = create_simple_cross_section();
//
//         let v1 = compute_topo_version(0, &cs, &manager);
//         let v2 = compute_topo_version(0, &cs, &manager);
//
//         assert_eq!(v1, v2, "Topo version must be deterministic");
//     }
//
//     #[test]
//     fn test_compute_topo_version_changes_with_segment_version() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [50.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [150.0, 0.0, 0.0]);
//
//         let segment = Segment {
//             id: SegmentId(1),
//             start_node: NodeId(1),
//             end_node: NodeId(2),
//             horizontal_profile: HorizontalProfile::Linear,
//             vertical_profile: VerticalProfile {
//                 start_z: 0.0,
//                 end_z: 0.0,
//             },
//             version: 1,
//             enabled: true,
//         };
//         manager.add_segment(segment);
//
//         let cs = create_simple_cross_section();
//         let v1 = compute_topo_version(0, &cs, &manager);
//
//         // Update segment version
//         let segment = Segment {
//             id: SegmentId(1),
//             start_node: NodeId(1),
//             end_node: NodeId(2),
//             horizontal_profile: HorizontalProfile::Linear,
//             vertical_profile: VerticalProfile {
//                 start_z: 0.0,
//                 end_z: 0.0,
//             },
//             version: 2,
//             enabled: true,
//         };
//         manager.add_segment(segment);
//
//         let v2 = compute_topo_version(0, &cs, &manager);
//
//         assert_ne!(v1, v2, "Version change should update topo hash");
//     }
//
//     #[test]
//     fn test_compute_topo_version_changes_with_position() {
//         let mut manager1 = TestRoadManager::new();
//         manager1.add_node(NodeId(1), [50.0, 0.0, 0.0]);
//         manager1.add_node(NodeId(2), [150.0, 0.0, 0.0]);
//         manager1.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let mut manager2 = TestRoadManager::new();
//         manager2.add_node(NodeId(1), [50.0, 1.0, 0.0]); // Different Y
//         manager2.add_node(NodeId(2), [150.0, 0.0, 0.0]);
//         manager2.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let cs = create_simple_cross_section();
//
//         let v1 = compute_topo_version(0, &cs, &manager1);
//         let v2 = compute_topo_version(0, &cs, &manager2);
//
//         assert_ne!(v1, v2, "Position change should update topo hash");
//     }
//
//     #[test]
//     fn test_compute_topo_version_changes_with_profile() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [50.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [150.0, 0.0, 0.0]);
//
//         let segment1 = Segment {
//             id: SegmentId(1),
//             start_node: NodeId(1),
//             end_node: NodeId(2),
//             horizontal_profile: HorizontalProfile::Linear,
//             vertical_profile: VerticalProfile {
//                 start_z: 0.0,
//                 end_z: 0.0,
//             },
//             version: 1,
//             enabled: true,
//         };
//
//         let segment2 = Segment {
//             id: SegmentId(1),
//             start_node: NodeId(1),
//             end_node: NodeId(2),
//             horizontal_profile: HorizontalProfile::QuadraticBezier {
//                 control: [100.0, 50.0],
//             },
//             vertical_profile: VerticalProfile {
//                 start_z: 0.0,
//                 end_z: 0.0,
//             },
//             version: 1,
//             enabled: true,
//         };
//
//         let cs = create_simple_cross_section();
//
//         manager.add_segment(segment1);
//         let v1 = compute_topo_version(0, &cs, &manager);
//
//         manager.add_segment(segment2);
//         let v2 = compute_topo_version(0, &cs, &manager);
//
//         assert_ne!(v1, v2, "Profile change should update topo hash");
//     }
//
//     #[test]
//     fn test_compute_topo_version_empty_chunk() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [500.0, 0.0, 0.0]); // Far from chunk 0
//         manager.add_node(NodeId(2), [600.0, 0.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let cs = create_simple_cross_section();
//
//         // Chunk 0 has no segments touching it
//         let v = compute_topo_version(0, &cs, &manager);
//         // Should still produce a valid hash (the offset basis)
//         assert_eq!(v, FNV_OFFSET_BASIS);
//     }
//
//     // ============================================================================
//     // Mesh Building Tests
//     // ============================================================================
//
//     #[test]
//     fn test_build_chunk_mesh_empty() {
//         let manager = TestRoadManager::new();
//         let cs = create_simple_cross_section();
//
//         let mesh = build_chunk_mesh(0, &cs, &manager, 2.0, 1.0, 1.0);
//
//         assert!(mesh.vertices.is_empty());
//         assert!(mesh.indices.is_empty());
//     }
//
//     #[test]
//     fn test_build_chunk_mesh_simple_segment() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [50.0, 0.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let cs = create_simple_cross_section();
//
//         let mesh = build_chunk_mesh(0, &cs, &manager, 10.0, 1.0, 1.0);
//
//         assert!(!mesh.vertices.is_empty());
//         assert!(!mesh.indices.is_empty());
//         // Indices should be multiples of 3 (triangles)
//         assert_eq!(mesh.indices.len() % 3, 0);
//     }
//
//     #[test]
//     fn test_build_chunk_mesh_indices_valid() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [50.0, 0.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let cs = create_simple_cross_section();
//
//         let mesh = build_chunk_mesh(0, &cs, &manager, 10.0, 1.0, 1.0);
//
//         let max_idx = mesh.vertices.len() as u32;
//         for idx in &mesh.indices {
//             assert!(
//                 *idx < max_idx,
//                 "Index {} out of bounds (max {})",
//                 idx,
//                 max_idx
//             );
//         }
//     }
//
//     #[test]
//     fn test_build_chunk_mesh_disabled_segment_excluded() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [50.0, 0.0, 0.0]);
//
//         let segment = Segment {
//             id: SegmentId(1),
//             start_node: NodeId(1),
//             end_node: NodeId(2),
//             horizontal_profile: HorizontalProfile::Linear,
//             vertical_profile: VerticalProfile {
//                 start_z: 0.0,
//                 end_z: 0.0,
//             },
//             version: 1,
//             enabled: false, // Disabled!
//         };
//         manager.add_segment(segment);
//
//         let cs = create_simple_cross_section();
//
//         let mesh = build_chunk_mesh(0, &cs, &manager, 10.0, 1.0, 1.0);
//
//         assert!(mesh.vertices.is_empty());
//         assert!(mesh.indices.is_empty());
//     }
//
//     #[test]
//     fn test_build_chunk_mesh_vertex_positions_in_expected_range() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [50.0, 0.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let cs = create_simple_cross_section(); // 10m wide, centered
//
//         let mesh = build_chunk_mesh(0, &cs, &manager, 10.0, 1.0, 1.0);
//
//         for v in &mesh.vertices {
//             // X should be along segment
//             assert!(v.position[0] >= -1.0 && v.position[0] <= 51.0);
//             // Y should be within cross-section width (±5m)
//             assert!(v.position[1] >= -6.0 && v.position[1] <= 6.0);
//         }
//     }
//
//     #[test]
//     fn test_build_chunk_mesh_normals_normalized() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [50.0, 0.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let cs = create_simple_cross_section();
//
//         let mesh = build_chunk_mesh(0, &cs, &manager, 10.0, 1.0, 1.0);
//
//         for v in &mesh.vertices {
//             let len = vec3_length(v.normal);
//             assert!(
//                 (len - 1.0).abs() < 1e-3,
//                 "Normal must be unit length"
//             );
//         }
//     }
//
//     #[test]
//     fn test_build_chunk_mesh_uv_world_space() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [50.0, 0.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let cs = create_simple_cross_section();
//
//         let mesh = build_chunk_mesh(0, &cs, &manager, 10.0, 1.0, 1.0);
//
//         // U should increase along segment (arc length based)
//         // V should vary with lateral offset
//         let mut found_varying_u = false;
//         let mut found_varying_v = false;
//
//         if mesh.vertices.len() > 1 {
//             let first_u = mesh.vertices[0].uv[0];
//             let first_v = mesh.vertices[0].uv[1];
//
//             for v in &mesh.vertices {
//                 if (v.uv[0] - first_u).abs() > 1e-6 {
//                     found_varying_u = true;
//                 }
//                 if (v.uv[1] - first_v).abs() > 1e-6 {
//                     found_varying_v = true;
//                 }
//             }
//         }
//
//         assert!(found_varying_u, "U coordinate should vary along segment");
//         assert!(found_varying_v, "V coordinate should vary laterally");
//     }
//
//     #[test]
//     fn test_build_chunk_mesh_material_ids() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [50.0, 0.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let cs = create_multi_region_cross_section();
//
//         let mesh = build_chunk_mesh(0, &cs, &manager, 10.0, 1.0, 1.0);
//
//         let mut material_ids: Vec<u32> = mesh.vertices.iter().map(|v| v.material_id).collect();
//         material_ids.sort();
//         material_ids.dedup();
//
//         // Should have materials 0, 1, 2 from cross-section
//         assert!(material_ids.contains(&0));
//         assert!(material_ids.contains(&1));
//         assert!(material_ids.contains(&2));
//     }
//
//     #[test]
//     fn test_build_chunk_mesh_multiple_segments() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [30.0, 0.0, 0.0]);
//         manager.add_node(NodeId(3), [0.0, 20.0, 0.0]);
//         manager.add_node(NodeId(4), [30.0, 20.0, 0.0]);
//
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//         manager.create_linear_segment(SegmentId(2), NodeId(3), NodeId(4));
//
//         let cs = create_simple_cross_section();
//
//         let mesh = build_chunk_mesh(0, &cs, &manager, 10.0, 1.0, 1.0);
//
//         // Should have vertices from both segments
//         let mut y_values: Vec<f32> = mesh.vertices.iter().map(|v| v.position[1]).collect();
//         y_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
//
//         // Should span from negative (cross-section width) to around 20 + width
//         assert!(
//             y_values.first().unwrap() < &5.0,
//             "Should include lower segment area"
//         );
//         assert!(
//             y_values.last().unwrap() > &15.0,
//             "Should include upper segment area"
//         );
//     }
//
//     #[test]
//     fn test_build_chunk_mesh_deterministic() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [50.0, 0.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let cs = create_simple_cross_section();
//
//         let mesh1 = build_chunk_mesh(0, &cs, &manager, 10.0, 1.0, 1.0);
//         let mesh2 = build_chunk_mesh(0, &cs, &manager, 10.0, 1.0, 1.0);
//
//         assert_eq!(mesh1.vertices.len(), mesh2.vertices.len());
//         assert_eq!(mesh1.indices.len(), mesh2.indices.len());
//         assert_eq!(mesh1.topo_version, mesh2.topo_version);
//
//         for (v1, v2) in mesh1.vertices.iter().zip(mesh2.vertices.iter()) {
//             assert_eq!(v1.position[0].to_bits(), v2.position[0].to_bits());
//             assert_eq!(v1.position[1].to_bits(), v2.position[1].to_bits());
//             assert_eq!(v1.position[2].to_bits(), v2.position[2].to_bits());
//             assert_eq!(v1.uv[0].to_bits(), v2.uv[0].to_bits());
//             assert_eq!(v1.uv[1].to_bits(), v2.uv[1].to_bits());
//             assert_eq!(v1.material_id, v2.material_id);
//         }
//
//         for (i1, i2) in mesh1.indices.iter().zip(mesh2.indices.iter()) {
//             assert_eq!(i1, i2);
//         }
//     }
//
//     #[test]
//     fn test_build_chunk_mesh_segment_spanning_chunks() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [50.0, 0.0, 0.0]); // Chunk 0
//         manager.add_node(NodeId(2), [150.0, 0.0, 0.0]); // Chunk 1
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let cs = create_simple_cross_section();
//
//         let mesh0 = build_chunk_mesh(0, &cs, &manager, 10.0, 1.0, 1.0);
//         let mesh1 = build_chunk_mesh(1, &cs, &manager, 10.0, 1.0, 1.0);
//
//         assert!(!mesh0.vertices.is_empty(), "Chunk 0 should have geometry");
//         assert!(!mesh1.vertices.is_empty(), "Chunk 1 should have geometry");
//     }
//
//     #[test]
//     fn test_build_chunk_mesh_seam_vertices_match() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [50.0, 0.0, 0.0]); // Chunk 0
//         manager.add_node(NodeId(2), [150.0, 0.0, 0.0]); // Chunk 1
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let cs = create_simple_cross_section();
//
//         let mesh0 = build_chunk_mesh(0, &cs, &manager, 10.0, 1.0, 1.0);
//         let mesh1 = build_chunk_mesh(1, &cs, &manager, 10.0, 1.0, 1.0);
//
//         // Find vertices near x=100 (chunk boundary)
//         let seam0: Vec<_> = mesh0
//             .vertices
//             .iter()
//             .filter(|v| (v.position[0] - 100.0).abs() < 0.1)
//             .collect();
//
//         let seam1: Vec<_> = mesh1
//             .vertices
//             .iter()
//             .filter(|v| (v.position[0] - 100.0).abs() < 0.1)
//             .collect();
//
//         // Both chunks should have seam vertices
//         assert!(!seam0.is_empty(), "Chunk 0 should have seam vertices");
//         assert!(!seam1.is_empty(), "Chunk 1 should have seam vertices");
//
//         // Seam vertices should have identical positions (bitwise)
//         for v0 in &seam0 {
//             let matching = seam1.iter().find(|v1| {
//                 v0.position[0].to_bits() == v1.position[0].to_bits()
//                     && v0.position[1].to_bits() == v1.position[1].to_bits()
//                     && v0.position[2].to_bits() == v1.position[2].to_bits()
//             });
//             assert!(
//                 matching.is_some(),
//                 "Seam vertex {:?} should have matching vertex in other chunk",
//                 v0.position
//             );
//         }
//     }
//
//     // ============================================================================
//     // RoadMeshManager Tests
//     // ============================================================================
//
//     #[test]
//     fn test_road_mesh_manager_new() {
//         let config = MeshConfig::default();
//         let rmm = RoadMeshManager::new(config);
//
//         assert!(rmm.get_chunk_mesh(0).is_none());
//     }
//
//     #[test]
//     fn test_road_mesh_manager_cache_miss() {
//         let config = MeshConfig::default();
//         let rmm = RoadMeshManager::new(config);
//
//         assert!(rmm.get_chunk_mesh(0).is_none());
//         assert!(rmm.get_chunk_mesh(1).is_none());
//     }
//
//     #[test]
//     fn test_road_mesh_manager_update_and_retrieve() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [50.0, 0.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let cs = create_simple_cross_section();
//         let config = MeshConfig::default();
//         let mut rmm = RoadMeshManager::new(config);
//
//         rmm.update_chunk_mesh(0, &cs, &manager);
//
//         let mesh = rmm.get_chunk_mesh(0);
//         assert!(mesh.is_some());
//         assert!(!mesh.unwrap().vertices.is_empty());
//     }
//
//     #[test]
//     fn test_road_mesh_manager_invalidate() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [50.0, 0.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let cs = create_simple_cross_section();
//         let config = MeshConfig::default();
//         let mut rmm = RoadMeshManager::new(config);
//
//         rmm.update_chunk_mesh(0, &cs, &manager);
//         assert!(rmm.get_chunk_mesh(0).is_some());
//
//         rmm.invalidate_chunk(0);
//         assert!(rmm.get_chunk_mesh(0).is_none());
//     }
//
//     #[test]
//     fn test_road_mesh_manager_clear_cache() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [150.0, 0.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let cs = create_simple_cross_section();
//         let config = MeshConfig::default();
//         let mut rmm = RoadMeshManager::new(config);
//
//         rmm.update_chunk_mesh(0, &cs, &manager);
//         rmm.update_chunk_mesh(1, &cs, &manager);
//
//         assert!(rmm.get_chunk_mesh(0).is_some());
//         assert!(rmm.get_chunk_mesh(1).is_some());
//
//         rmm.clear_cache();
//
//         assert!(rmm.get_chunk_mesh(0).is_none());
//         assert!(rmm.get_chunk_mesh(1).is_none());
//     }
//
//     #[test]
//     fn test_road_mesh_manager_chunk_needs_update_no_cache() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [50.0, 0.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let cs = create_simple_cross_section();
//         let config = MeshConfig::default();
//         let rmm = RoadMeshManager::new(config);
//
//         assert!(rmm.chunk_needs_update(0, &cs, &manager));
//     }
//
//     #[test]
//     fn test_road_mesh_manager_chunk_needs_update_cached() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [50.0, 0.0, 0.0]);
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let cs = create_simple_cross_section();
//         let config = MeshConfig::default();
//         let mut rmm = RoadMeshManager::new(config);
//
//         rmm.update_chunk_mesh(0, &cs, &manager);
//
//         assert!(!rmm.chunk_needs_update(0, &cs, &manager));
//     }
//
//     #[test]
//     fn test_road_mesh_manager_chunk_needs_update_after_change() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [50.0, 0.0, 0.0]);
//
//         let segment = Segment {
//             id: SegmentId(1),
//             start_node: NodeId(1),
//             end_node: NodeId(2),
//             horizontal_profile: HorizontalProfile::Linear,
//             vertical_profile: VerticalProfile {
//                 start_z: 0.0,
//                 end_z: 0.0,
//             },
//             version: 1,
//             enabled: true,
//         };
//         manager.add_segment(segment);
//
//         let cs = create_simple_cross_section();
//         let config = MeshConfig::default();
//         let mut rmm = RoadMeshManager::new(config);
//
//         rmm.update_chunk_mesh(0, &cs, &manager);
//         assert!(!rmm.chunk_needs_update(0, &cs, &manager));
//
//         // Bump segment version
//         let segment = Segment {
//             id: SegmentId(1),
//             start_node: NodeId(1),
//             end_node: NodeId(2),
//             horizontal_profile: HorizontalProfile::Linear,
//             vertical_profile: VerticalProfile {
//                 start_z: 0.0,
//                 end_z: 0.0,
//             },
//             version: 2,
//             enabled: true,
//         };
//         manager.add_segment(segment);
//
//         assert!(rmm.chunk_needs_update(0, &cs, &manager));
//     }
//
//     #[test]
//     fn test_road_mesh_manager_config_access() {
//         let config = MeshConfig {
//             max_segment_edge_length_m: 5.0,
//             uv_scale_u: 2.0,
//             uv_scale_v: 0.5,
//         };
//         let rmm = RoadMeshManager::new(config.clone());
//
//         assert!((rmm.config().max_segment_edge_length_m - 5.0).abs() < 1e-6);
//         assert!((rmm.config().uv_scale_u - 2.0).abs() < 1e-6);
//         assert!((rmm.config().uv_scale_v - 0.5).abs() < 1e-6);
//     }
//
//     // ============================================================================
//     // MeshConfig Tests
//     // ============================================================================
//
//     #[test]
//     fn test_mesh_config_default() {
//         let config = MeshConfig::default();
//         assert!((config.max_segment_edge_length_m - 2.0).abs() < 1e-6);
//         assert!((config.uv_scale_u - 1.0).abs() < 1e-6);
//         assert!((config.uv_scale_v - 1.0).abs() < 1e-6);
//     }
//
//     // ============================================================================
//     // Edge Cases and Robustness Tests
//     // ============================================================================
//
//     #[test]
//     fn test_zero_length_segment() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [50.0, 50.0, 0.0]);
//         manager.add_node(NodeId(2), [50.0, 50.0, 0.0]); // Same position
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let cs = create_simple_cross_section();
//
//         // Should not crash
//         let mesh = build_chunk_mesh(0, &cs, &manager, 10.0, 1.0, 1.0);
//
//         // May produce no geometry or degenerate geometry
//         assert_eq!(mesh.indices.len() % 3, 0);
//     }
//
//     #[test]
//     fn test_very_short_segment() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [50.0, 50.0, 0.0]);
//         manager.add_node(NodeId(2), [50.001, 50.0, 0.0]); // 1mm segment
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let cs = create_simple_cross_section();
//
//         let mesh = build_chunk_mesh(0, &cs, &manager, 10.0, 1.0, 1.0);
//
//         // Should produce valid (possibly minimal) geometry
//         assert_eq!(mesh.indices.len() % 3, 0);
//     }
//
//     #[test]
//     fn test_very_long_segment() {
//         let mut manager = TestRoadManager::new();
//         manager.add_node(NodeId(1), [0.0, 0.0, 0.0]);
//         manager.add_node(NodeId(2), [10000.0, 0.0, 0.0]); // 10km segment
//         manager.create_linear_segment(SegmentId(1), NodeId(1), NodeId(2));
//
//         let cs = create_simple_cross_section();
//
//         // Only get chunk 0
//         let mesh = build_chunk_mesh(0, &cs, &manager, 10.0, 1.0, 1.0);
//
//         // Should only include geometry for chunk 0 (0-100m)
//         for v in &mesh.vertices {
//             assert!(
//                 v.position[0] >= -10.0 && v.position[0] <= 110.0,
//                 "Vertex x={} should be within or near chunk 0",
//                 v.position[0]
//             );
//         }
//     }
// }
