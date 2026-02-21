//! road_mesh_renderer.rs
use crate::world::roads::road_editor::RoadEditor;
use crate::world::roads::road_mesh_manager::{ChunkId, MeshConfig, RoadMeshManager};
use crate::world::roads::roads::{
    RoadManager, apply_commands_world, apply_preview_commands_world, collect_affected_chunks,
};
use crate::world::terrain::terrain_subsystem::Terrain;

use crate::renderer::gizmo::gizmo::Gizmo;
use crate::resources::Time;
use crate::simulation::Ticker;
use crate::ui::input::Input;
use crate::world::camera::Camera;
use crate::world::cars::car_subsystem::CarSubsystem;
use crate::world::cars::partitions::PartitionManager;
use crate::world::roads::road_preview::{PreviewGpuMesh, RoadAppearanceGpu, RoadPreviewState};
use crate::world::roads::road_structs::RoadEditorCommand;
use std::collections::HashMap;
use wgpu::util::DeviceExt;
use wgpu::{Device, Queue};

pub struct ChunkGpuMesh {
    pub vertex: wgpu::Buffer,
    pub index: wgpu::Buffer,
    pub index_count: u32,
    pub topo_version: u64,
}
pub struct RoadRenderSubsystem {
    pub mesh_manager: RoadMeshManager,

    pub chunk_gpu: HashMap<ChunkId, ChunkGpuMesh>,
    pub visible_draw_list: Vec<ChunkId>,

    pub preview_state: RoadPreviewState,
    pub preview_gpu: PreviewGpuMesh,
    pub road_appearance: RoadAppearanceGpu,
}
impl RoadRenderSubsystem {
    pub fn new(device: &Device) -> Self {
        Self {
            mesh_manager: RoadMeshManager::new(MeshConfig::default()),
            chunk_gpu: Default::default(),
            visible_draw_list: vec![],
            preview_state: RoadPreviewState::new(),
            preview_gpu: PreviewGpuMesh::new(),
            road_appearance: RoadAppearanceGpu::new(device),
        }
    }

    /// Render-only update: processes commands for preview/mesh, rebuilds chunk meshes, uploads to GPU.
    pub fn update(
        &mut self,
        terrain: &mut Terrain,
        road_subsystem: &RoadSubsystem,
        device: &Device,
        queue: &Queue,
        camera: &Camera,
        gizmo: &mut Gizmo,
    ) {
        // --- Preview mesh ---
        // Rebuild preview mesh from the (already-mutated) preview_roads
        let preview_mesh = self.mesh_manager.build_preview_mesh(
            terrain,
            &road_subsystem.road_manager.preview_roads,
            &road_subsystem.road_editor.style,
            gizmo,
        );
        self.preview_gpu.upload(device, &preview_mesh);

        // Preview appearance state
        self.preview_state
            .ingest(road_subsystem.road_commands.as_slice());
        self.road_appearance
            .update_preview_buffer(queue, &self.preview_state, camera.orbit_radius);

        // --- Chunk meshes for committed roads ---
        // Rebuild any dirty chunk meshes from commands
        let affected_chunks = collect_affected_chunks(road_subsystem.road_commands.as_slice());
        for chunk_id in &affected_chunks {
            self.mesh_manager.update_chunk_mesh(
                terrain,
                *chunk_id,
                &road_subsystem.road_manager.roads,
                &road_subsystem.road_editor.style,
                gizmo,
            );
        }

        // --- Visible draw list ---
        self.visible_draw_list.clear();

        for v in &terrain.visible {
            let chunk_id = v.id;

            let needs_rebuild = self.mesh_manager.chunk_needs_update(
                chunk_id,
                &road_subsystem.road_manager.roads,
                terrain.chunk_size,
            );

            let mesh = if needs_rebuild {
                self.mesh_manager.update_chunk_mesh(
                    terrain,
                    chunk_id,
                    &road_subsystem.road_manager.roads,
                    &road_subsystem.road_editor.style,
                    gizmo,
                )
            } else {
                match self.mesh_manager.get_chunk_mesh(chunk_id) {
                    Some(m) => m,
                    None => continue,
                }
            };

            if mesh.indices.is_empty() || mesh.vertices.is_empty() {
                self.chunk_gpu.remove(&chunk_id);
                continue;
            }

            let needs_gpu_upload = match self.chunk_gpu.get(&chunk_id) {
                Some(gpu) => gpu.topo_version != mesh.topo_version,
                None => true,
            };

            if needs_gpu_upload {
                let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Road Chunk VB"),
                    contents: bytemuck::cast_slice(&mesh.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

                let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Road Chunk IB"),
                    contents: bytemuck::cast_slice(&mesh.indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

                self.chunk_gpu.insert(
                    chunk_id,
                    ChunkGpuMesh {
                        vertex: vb,
                        index: ib,
                        index_count: mesh.indices.len() as u32,
                        topo_version: mesh.topo_version,
                    },
                );
            }

            if self.chunk_gpu.contains_key(&chunk_id) {
                self.visible_draw_list.push(chunk_id);
            }
        }
    }
}

pub struct RoadSubsystem {
    pub road_manager: RoadManager,
    pub road_editor: RoadEditor,
    pub partition_manager: PartitionManager,
    road_commands: Vec<RoadEditorCommand>,
    pub tick_20hz: Ticker,
}
impl RoadSubsystem {
    pub fn new() -> Self {
        Self {
            tick_20hz: Ticker::new(20.0),
            road_manager: RoadManager::new(),
            road_editor: RoadEditor::new(),
            partition_manager: PartitionManager::new(),
            road_commands: vec![],
        }
    }
    /// World-only update: runs the editor, applies commands to road storage and car subsystem.
    /// Returns the commands so the render subsystem can process previews/meshes.
    pub fn update(
        &mut self,
        terrain: &mut Terrain,
        car_subsystem: &mut CarSubsystem,
        input: &mut Input,
        time: &Time,
        gizmo: &mut Gizmo,
    ) {
        self.road_commands = self
            .road_editor
            .update(&self.road_manager, terrain, input, gizmo);

        // Apply preview commands to preview_roads (world-side storage mutation only, no mesh)
        apply_preview_commands_world(
            terrain,
            &self.road_editor.style,
            &mut self.road_manager.preview_roads,
            &self.road_manager.roads,
            car_subsystem,
            &self.road_commands,
            gizmo,
        );

        // Apply real commands to roads storage
        if !self.road_commands.is_empty() {
            apply_commands_world(
                terrain,
                &mut self.road_manager.roads,
                car_subsystem,
                gizmo,
                &self.road_commands,
            );
        }
        self.partition_manager
            .rebuild_all(&self.road_manager.roads, terrain.chunk_size);
    }
}
pub const _WGSL_SDF_TEXT_OVERLAY: &str = r#"
@group(1) @binding(0) var glyph_atlas: texture_2d_array<f32>;
@group(1) @binding(1) var glyph_sampler: sampler;

const SDF_EDGE_VALUE: f32 = 0.5;
const SDF_SMOOTHING: f32 = 0.1;
const TEXT_BLEND_FACTOR: f32 = 1.0;
const TEXT_COLOR: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);

fn sample_sdf_glyph(text_uv: vec2<f32>, glyph_layer: u32) -> f32 {
    let distance = textureSample(glyph_atlas, glyph_sampler, text_uv, i32(glyph_layer)).r;
    let alpha = smoothstep(SDF_EDGE_VALUE - SDF_SMOOTHING, SDF_EDGE_VALUE + SDF_SMOOTHING, distance);
    return alpha;
}

fn blend_sdf_text_additive(
    base_color: vec4<f32>,
    text_uv: vec2<f32>,
    glyph_layer: u32,
    text_color: vec3<f32>
) -> vec4<f32> {
    let text_alpha = sample_sdf_glyph(text_uv, glyph_layer);
    let text_contribution = text_color * text_alpha * TEXT_BLEND_FACTOR;
    return vec4<f32>(base_color.rgb + text_contribution, base_color.a);
}

fn apply_text_overlay(base_color: vec4<f32>, text_uv: vec2<f32>, glyph_layer: u32) -> vec4<f32> {
    return blend_sdf_text_additive(base_color, text_uv, glyph_layer, TEXT_COLOR);
}
"#;
