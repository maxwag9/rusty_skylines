//! road_mesh_renderer.rs
use crate::renderer::world_renderer::{PickedPoint, TerrainRenderer};
use crate::terrain::roads::road_editor::RoadEditor;
use crate::terrain::roads::road_mesh_manager::{ChunkId, MeshConfig, RoadMeshManager};
use crate::terrain::roads::roads::{RoadManager, apply_commands, apply_preview_commands};

use crate::components::camera::Camera;
use crate::renderer::gizmo::Gizmo;
use crate::terrain::roads::road_preview::{PreviewGpuMesh, RoadAppearanceGpu, RoadPreviewState};
use crate::terrain::roads::road_structs::RoadStyleParams;
use crate::ui::input::InputState;
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

    pub style: RoadStyleParams,
    pub chunk_gpu: HashMap<ChunkId, ChunkGpuMesh>,
    pub visible_draw_list: Vec<ChunkId>,

    pub road_manager: RoadManager,
    pub road_editor: RoadEditor,

    pub preview_state: RoadPreviewState,
    pub preview_gpu: PreviewGpuMesh,
    pub road_appearance: RoadAppearanceGpu,
}
impl RoadRenderSubsystem {
    pub fn new(device: &Device) -> Self {
        Self {
            mesh_manager: RoadMeshManager::new(MeshConfig::default()),
            style: RoadStyleParams::default(),
            chunk_gpu: Default::default(),
            visible_draw_list: vec![],
            road_manager: RoadManager::new(),
            road_editor: RoadEditor::new(),
            preview_state: RoadPreviewState::new(),
            preview_gpu: PreviewGpuMesh::new(),
            road_appearance: RoadAppearanceGpu::new(device),
        }
    }

    /// Called each frame with visible chunk IDs
    pub fn update(
        &mut self,
        terrain_renderer: &TerrainRenderer,
        device: &Device,
        queue: &Queue,
        input: &mut InputState,
        picked_point: &Option<PickedPoint>,
        camera: &Camera,
        gizmo: &mut Gizmo,
    ) {
        // Get commands from road editor
        let road_commands = self.road_editor.update(
            &self.road_manager,
            terrain_renderer,
            &mut self.style,
            input,
            picked_point,
        );
        // Apply preview commands to create preview geometry
        apply_preview_commands(
            terrain_renderer,
            &mut self.mesh_manager,
            &mut self.road_manager.preview_roads, // preview RoadStorage
            &self.road_manager.roads,
            &self.style,
            gizmo,
            &road_commands,
        );
        self.preview_state.ingest(&road_commands);
        self.road_appearance
            .update_preview_buffer(queue, &self.preview_state, camera.orbit_radius);
        // Apply real topology commands
        if !road_commands.is_empty() {
            //println!("{:?}", road_commands);
            let _results = apply_commands(
                terrain_renderer,
                &mut self.mesh_manager,
                &mut self.road_manager.roads,
                &self.style,
                false,
                gizmo,
                road_commands,
            );
        }

        let preview_mesh = self.mesh_manager.build_preview_mesh(
            terrain_renderer,
            &self.road_manager.preview_roads,
            &self.style,
            gizmo,
        );

        self.preview_gpu.upload(device, &preview_mesh);

        // === Existing chunk mesh update logic ===
        self.visible_draw_list.clear();

        for v in &terrain_renderer.visible {
            let chunk_id = v.id;

            let needs_rebuild = self.mesh_manager.chunk_needs_update(
                chunk_id,
                &self.road_manager.roads,
                terrain_renderer.chunk_size,
            );

            let mesh = if needs_rebuild {
                self.mesh_manager.update_chunk_mesh(
                    terrain_renderer,
                    chunk_id,
                    &self.road_manager.roads,
                    &self.style,
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
