//! road_preview.rs
//! Preview mesh generation for road editor - temporary geometry rendered with blue tint.

use crate::renderer::world_renderer::TerrainRenderer;
use crate::terrain::roads::road_editor::{
    LanePreview, NodePreview, PreviewError, RoadEditorCommand, SegmentPreview, SnapPreview,
};
use crate::terrain::roads::road_mesh_manager::{CrossSection, RoadVertex};
use glam::Vec3;
use wgpu::util::DeviceExt;
use wgpu::{Device, Queue};
// ============================================================================
// Road Appearance Uniform (for tinting)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RoadAppearanceUniform {
    pub tint: [f32; 4],
}

impl RoadAppearanceUniform {
    /// Normal roads: no tint
    pub fn normal() -> Self {
        Self {
            tint: [1.0, 1.0, 1.0, 1.0],
        }
    }

    /// Preview roads: subtle blue tint
    pub fn preview() -> Self {
        Self {
            tint: [0.8, 0.8, 2.1, 0.9],
        }
    }

    /// Error preview: reddish tint for invalid placements
    pub fn preview_error() -> Self {
        Self {
            tint: [1.3, 0.6, 0.6, 1.0],
        }
    }
}

// ============================================================================
// Preview State - captures preview commands each frame
// ============================================================================

#[derive(Default, Debug, Clone)]
pub struct RoadPreviewState {
    pub segments: Vec<SegmentPreview>,
    pub snap: Option<SnapPreview>,
    pub node: Option<NodePreview>,
    pub lanes: Vec<LanePreview>,
    pub error: Option<PreviewError>,
}

impl RoadPreviewState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest preview commands from the editor. Called each frame.
    /// Clears previous state and captures new previews.
    pub fn ingest(&mut self, cmds: &[RoadEditorCommand]) {
        // Reset all state
        self.segments.clear();
        self.snap = None;
        self.node = None;
        self.lanes.clear();
        self.error = None;

        for cmd in cmds {
            match cmd {
                RoadEditorCommand::PreviewClear => {
                    self.segments.clear();
                    self.snap = None;
                    self.node = None;
                    self.lanes.clear();
                    self.error = None;
                }
                RoadEditorCommand::PreviewSegment(seg) => {
                    if seg.estimated_length > 1.0 {
                        self.segments.push(seg.clone());
                    }
                }
                RoadEditorCommand::PreviewSnap(snap) => {
                    self.snap = Some(snap.clone());
                }
                RoadEditorCommand::PreviewNode(node) => {
                    self.node = Some(node.clone());
                }
                RoadEditorCommand::PreviewLane(lane) => {
                    self.lanes.push(lane.clone());
                }
                RoadEditorCommand::PreviewError(err) => {
                    self.error = Some(err.clone());
                }
                RoadEditorCommand::Road(_) => {
                    // Real topology commands - ignore for preview state
                }
            }
        }
    }

    pub fn has_segments(&self) -> bool {
        !self.segments.is_empty()
    }

    /// Returns true if any segment is marked invalid
    pub fn has_invalid_segment(&self) -> bool {
        self.segments.iter().any(|s| !s.is_valid)
    }
}

// ============================================================================
// Preview CPU Mesh
// ============================================================================

#[derive(Default, Debug, Clone)]
pub struct PreviewCpuMesh {
    pub vertices: Vec<RoadVertex>,
    pub indices: Vec<u32>,
}

impl PreviewCpuMesh {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }
}

// ============================================================================
// Preview Ring (internal sampling structure)
// ============================================================================

#[derive(Clone, Debug)]
struct PreviewRing {
    position: [f32; 3],
    tangent: [f32; 3],
    arc_length: f32,
}

// ============================================================================
// Preview Mesh Builder
// ============================================================================

/// Sample a polyline at regular intervals, computing tangents and arc lengths.
fn sample_polyline_to_rings(polyline: &[Vec3], max_step: f32) -> Vec<PreviewRing> {
    if polyline.len() < 2 {
        return Vec::new();
    }

    let mut rings = Vec::new();
    let mut arc_length = 0.0f32;

    // First point with forward tangent
    let first_tangent = (polyline[1] - polyline[0]).normalize_or_zero();
    rings.push(PreviewRing {
        position: polyline[0].to_array(),
        tangent: first_tangent.to_array(),
        arc_length: 0.0,
    });

    for i in 0..polyline.len() - 1 {
        let p0 = polyline[i];
        let p1 = polyline[i + 1];
        let segment_vec = p1 - p0;
        let segment_len = segment_vec.length();

        if segment_len < 0.001 {
            continue;
        }

        let tangent = segment_vec / segment_len;

        // Subdivide long segments
        let num_steps = ((segment_len / max_step).ceil() as usize).max(1);
        let step_len = segment_len / num_steps as f32;

        for s in 1..=num_steps {
            let t = s as f32 / num_steps as f32;
            let pos = p0 + segment_vec * t;
            arc_length += step_len;

            // Smooth tangent at segment boundaries
            let ring_tangent = if i + 1 < polyline.len() - 1 && s == num_steps {
                let next_dir = (polyline[i + 2] - polyline[i + 1]).normalize_or_zero();
                ((tangent + next_dir) * 0.5).normalize_or_zero()
            } else {
                tangent
            };

            rings.push(PreviewRing {
                position: pos.to_array(),
                tangent: ring_tangent.to_array(),
                arc_length,
            });
        }
    }

    rings
}

/// Build preview mesh from preview state using the same cross-section as real roads.
pub fn build_preview_mesh(
    terrain_renderer: &TerrainRenderer,
    preview_state: &RoadPreviewState,
    cross_section: &CrossSection,
) -> PreviewCpuMesh {
    let mut mesh = PreviewCpuMesh::new();

    if preview_state.segments.is_empty() {
        return mesh;
    }

    const MAX_STEP_LENGTH: f32 = 2.0;
    const CLEARANCE: f32 = 0.04;

    let strips = cross_section.lateral_strips();
    let half_width = cross_section.half_width();

    for segment in &preview_state.segments {
        if segment.polyline.len() < 2 {
            continue;
        }

        let rings = sample_polyline_to_rings(&segment.polyline, MAX_STEP_LENGTH);
        if rings.len() < 2 {
            continue;
        }

        // === HORIZONTAL STRIP SURFACES ===
        for strip in &strips {
            let strip_base = mesh.vertices.len() as u32;

            // Generate vertices for this strip
            for ring in &rings {
                // Tangent in XZ plane
                let txz = Vec3::new(ring.tangent[0], 0.0, ring.tangent[2]).normalize_or_zero();
                let lateral = Vec3::new(-txz.z, 0.0, txz.x);
                let normal = lateral.cross(txz).normalize_or_zero();

                for &lat in &[strip.left, strip.right] {
                    let terrain_y =
                        terrain_renderer.get_height_at([ring.position[0], ring.position[2]]);

                    let pos = Vec3::new(
                        ring.position[0],
                        terrain_y + strip.height + CLEARANCE,
                        ring.position[2],
                    ) + lateral * lat;

                    let u = ring.arc_length;
                    let v = lat + half_width;

                    mesh.vertices.push(RoadVertex {
                        position: pos.to_array(),
                        normal: normal.to_array(),
                        uv: [u, v],
                        material_id: strip.material_id,
                    });
                }
            }

            // Generate indices for strip quads
            let ring_count = rings.len();
            for i in 0..ring_count - 1 {
                let i0 = strip_base + (i * 2) as u32;
                let i1 = i0 + 1;
                let i2 = i0 + 2;
                let i3 = i0 + 3;

                mesh.indices.extend_from_slice(&[i0, i1, i2, i1, i3, i2]);
            }
        }

        // === VERTICAL CURB FACES ===
        for i in 0..strips.len().saturating_sub(1) {
            let current_strip = &strips[i];
            let next_strip = &strips[i + 1];
            let height_diff = current_strip.height - next_strip.height;

            if height_diff.abs() < 0.0001 {
                continue;
            }

            let lat = current_strip.right;
            let higher_height = current_strip.height.max(next_strip.height);
            let lower_height = current_strip.height.min(next_strip.height);

            let material_id = if current_strip.height >= next_strip.height {
                current_strip.material_id
            } else {
                next_strip.material_id
            };

            let curb_base = mesh.vertices.len() as u32;

            for ring in &rings {
                let txz = Vec3::new(ring.tangent[0], 0.0, ring.tangent[2]).normalize_or_zero();
                let lateral_dir = Vec3::new(-txz.z, 0.0, txz.x);

                let normal = if height_diff > 0.0 {
                    lateral_dir
                } else {
                    -lateral_dir
                };

                let terrain_y =
                    terrain_renderer.get_height_at([ring.position[0], ring.position[2]]);
                let base_pos = Vec3::new(ring.position[0], terrain_y + CLEARANCE, ring.position[2])
                    + lateral_dir * lat;

                let pos_high = base_pos + Vec3::Y * higher_height;
                let pos_low = base_pos + Vec3::Y * lower_height;

                let u = ring.arc_length;
                let v_height = higher_height - lower_height;

                // High vertex
                mesh.vertices.push(RoadVertex {
                    position: pos_high.to_array(),
                    normal: normal.to_array(),
                    uv: [u, 0.0],
                    material_id,
                });
                // Low vertex
                mesh.vertices.push(RoadVertex {
                    position: pos_low.to_array(),
                    normal: normal.to_array(),
                    uv: [u, v_height],
                    material_id,
                });
            }

            // Generate curb face indices
            let ring_count = rings.len();
            for j in 0..ring_count - 1 {
                let i0 = curb_base + (j * 2) as u32;
                let i1 = i0 + 1;
                let i2 = i0 + 2;
                let i3 = i0 + 3;

                if height_diff > 0.0 {
                    mesh.indices.extend_from_slice(&[i0, i1, i2, i2, i1, i3]);
                } else {
                    mesh.indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
                }
            }
        }
    }

    mesh
}

// ============================================================================
// Preview GPU Mesh
// ============================================================================

pub struct PreviewGpuMesh {
    pub vb: Option<wgpu::Buffer>,
    pub ib: Option<wgpu::Buffer>,
    pub index_count: u32,
}

impl Default for PreviewGpuMesh {
    fn default() -> Self {
        Self::new()
    }
}

impl PreviewGpuMesh {
    pub fn new() -> Self {
        Self {
            vb: None,
            ib: None,
            index_count: 0,
        }
    }

    /// Upload CPU mesh to GPU. Recreates buffers each frame (fine for small preview meshes).
    pub fn upload(&mut self, device: &Device, mesh: &PreviewCpuMesh) {
        if mesh.is_empty() {
            self.vb = None;
            self.ib = None;
            self.index_count = 0;
            return;
        }

        self.vb = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Road Preview VB"),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
        );

        self.ib = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Road Preview IB"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
        );

        self.index_count = mesh.indices.len() as u32;
    }

    pub fn is_empty(&self) -> bool {
        self.index_count == 0
    }
}

// ============================================================================
// Road Appearance GPU Resources
// ============================================================================

pub struct RoadAppearanceGpu {
    pub normal_buffer: wgpu::Buffer,
    pub preview_buffer: wgpu::Buffer,
}

impl RoadAppearanceGpu {
    pub fn new(device: &Device) -> Self {
        let preview_uniform = RoadAppearanceUniform::preview();
        let normal_uniform = RoadAppearanceUniform::normal();

        let preview_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Road Appearance Preview"),
            contents: bytemuck::cast_slice(&[preview_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let normal_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Road Appearance Normal"),
            contents: bytemuck::cast_slice(&[normal_uniform]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        Self {
            preview_buffer,
            normal_buffer,
        }
    }
    pub fn update_preview_buffer(
        &mut self,
        device: &Device,
        queue: &Queue,
        preview_state: &RoadPreviewState,
    ) {
        let mut new_preview = RoadAppearanceUniform::normal();
        if preview_state.error.is_some() {
            new_preview = RoadAppearanceUniform::preview_error()
        } else if !preview_state.segments.is_empty() {
            new_preview = RoadAppearanceUniform::preview()
        }

        queue.write_buffer(&self.preview_buffer, 0, bytemuck::bytes_of(&new_preview));
    }
}
