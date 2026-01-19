//! road_preview.rs
//! Preview mesh generation for road editor - temporary geometry rendered with blue tint.
//! Uses *exactly* the same geometry generation code as normal roads for perfect vertex-for-vertex compatibility.
//! Node preview for isolated nodes is a closed circular road loop (separate ring generation, same meshing).
//! Multiple preview segments are supported (e.g. for showing connected existing roads during placement).
use crate::terrain::roads::road_mesh_manager::ChunkMesh;
use crate::terrain::roads::road_structs::*;
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
    pub fn normal() -> Self {
        Self {
            tint: [1.0, 1.0, 1.0, 1.0],
        }
    }
    pub fn preview() -> Self {
        Self {
            tint: [0.8, 0.8, 2.1, 0.9],
        }
    }
    pub fn preview_error() -> Self {
        Self {
            tint: [1.3, 0.6, 0.6, 1.0],
        }
    }
}

// ============================================================================
// Preview State
// ============================================================================
#[derive(Default, Debug, Clone)]
pub struct RoadPreviewState {
    pub snap: Option<SnapPreview>,
    pub error: Option<PreviewError>,
}
impl RoadPreviewState {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn ingest(&mut self, cmds: &[RoadEditorCommand]) {
        self.snap = None;
        self.error = None;
        for cmd in cmds {
            match cmd {
                RoadEditorCommand::PreviewClear => {
                    self.snap = None;
                    self.error = None;
                }
                RoadEditorCommand::PreviewSnap(snap) => {
                    self.snap = Some(snap.clone());
                }
                RoadEditorCommand::PreviewError(err) => {
                    self.error = Some(err.clone());
                }
                _ => {}
            }
        }
    }
}

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
    pub fn upload(&mut self, device: &Device, mesh: &ChunkMesh) {
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
    pub fn update_preview_buffer(&mut self, queue: &Queue, preview_state: &RoadPreviewState) {
        let new_preview: RoadAppearanceUniform;
        if preview_state.error.is_some() {
            new_preview = RoadAppearanceUniform::preview_error();
        } else {
            new_preview = RoadAppearanceUniform::preview();
        }
        queue.write_buffer(&self.preview_buffer, 0, bytemuck::bytes_of(&new_preview));
    }
}
