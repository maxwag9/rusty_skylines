//! road_mesh_renderer.rs
use crate::world::roads::road_editor::RoadEditor;
use crate::world::roads::road_mesh_manager::{ChunkId, MeshConfig, RoadMeshManager};
use crate::world::roads::roads::{
    RoadManager, apply_road_commands_preview, apply_road_commands_real, collect_affected_chunks,
};
use crate::world::terrain::terrain_subsystem::Terrain;

use crate::data::Settings;
use crate::renderer::gizmo::gizmo::Gizmo;
use crate::resources::Time;
use crate::simulation::Ticker;
use crate::ui::input::Input;
use crate::world::camera::Camera;
use crate::world::cars::car_subsystem::Cars;
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

    pub preview_gpu: PreviewGpuMesh,
    pub road_appearance: RoadAppearanceGpu,
}
impl RoadRenderSubsystem {
    pub fn new(device: &Device) -> Self {
        Self {
            mesh_manager: RoadMeshManager::new(MeshConfig::default()),
            chunk_gpu: Default::default(),
            visible_draw_list: vec![],
            preview_gpu: PreviewGpuMesh::new(),
            road_appearance: RoadAppearanceGpu::new(device),
        }
    }

    /// Render-only update: processes commands for preview/mesh, rebuilds chunk meshes, uploads to GPU.
    pub fn update(
        &mut self,
        terrain: &mut Terrain,
        roads: &mut Roads,
        device: &Device,
        queue: &Queue,
        camera: &Camera,
        gizmo: &mut Gizmo,
    ) {
        // --- Preview mesh ---
        // Rebuild preview mesh from the (already-mutated) preview_roads
        if roads.preview_state.has_changed {
            let preview_mesh = self.mesh_manager.build_preview_mesh(
                terrain,
                &roads.road_manager,
                &roads.road_editor.style,
                gizmo,
            );
            self.preview_gpu.upload(device, &preview_mesh);
        }

        // Preview appearance state
        self.road_appearance.update_preview_buffer(
            queue,
            &roads.preview_state,
            camera.orbit_radius,
        );

        // --- Chunk meshes for committed roads ---
        // Rebuild any dirty chunk meshes from commands
        let affected_chunks = collect_affected_chunks(roads.road_commands.as_slice());
        for chunk_id in &affected_chunks {
            self.mesh_manager.update_chunk_mesh(
                terrain,
                *chunk_id,
                &roads.road_manager,
                &roads.road_editor.style,
                gizmo,
            );
        }

        // --- Visible draw list ---
        self.visible_draw_list.clear();
        let mut chunk_rebuild_ids = std::mem::take(&mut roads.road_editor.pending_chunk_rebuilds);

        for v in &terrain.visible {
            let chunk_id = v.id;

            let needs_rebuild = self
                .mesh_manager
                .chunk_needs_update(chunk_id, &roads.road_manager.roads);
            if needs_rebuild {
                chunk_rebuild_ids.push(chunk_id)
            } else {
                let mesh = match self.mesh_manager.get_chunk_mesh(chunk_id) {
                    Some(m) => m,
                    None => continue,
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
            };
        }

        for chunk_id in chunk_rebuild_ids {
            let mesh = self.mesh_manager.update_chunk_mesh(
                terrain,
                chunk_id,
                &roads.road_manager,
                &roads.road_editor.style,
                gizmo,
            );

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

pub struct Roads {
    pub road_manager: RoadManager,
    pub road_editor: RoadEditor,
    pub partition_manager: PartitionManager,
    pub road_commands: Vec<RoadEditorCommand>,
    pub tick_20hz: Ticker,
    pub preview_state: RoadPreviewState,
}
impl Roads {
    pub fn new() -> Self {
        Self {
            tick_20hz: Ticker::new(20.0),
            road_manager: RoadManager::new(),
            road_editor: RoadEditor::new(),
            partition_manager: PartitionManager::new(),
            road_commands: vec![],
            preview_state: RoadPreviewState::new(),
        }
    }
    /// World-only update: runs the editor, applies commands to road storage and car subsystem.
    /// Returns the commands so the render subsystem can process previews/meshes.
    pub fn update(
        &mut self,
        terrain: &mut Terrain,
        cars: &mut Cars,
        input: &mut Input,
        _time: &Time,
        settings: &Settings,
        gizmo: &mut Gizmo,
    ) {
        self.road_commands = self
            .road_editor
            .update(&mut self.road_manager, terrain, input, gizmo);
        self.preview_state.ingest(self.road_commands.as_slice());
        // Apply preview commands to preview_roads (world-side storage mutation only, no mesh)
        if self.preview_state.has_changed {
            apply_road_commands_preview(terrain, self, cars, settings, gizmo);
        }

        // Apply real commands to roads storage
        if !self.road_commands.is_empty() {
            apply_road_commands_real(
                terrain,
                &mut self.road_manager,
                cars,
                settings,
                gizmo,
                &self.road_commands,
            );
        }
        self.partition_manager.rebuild_all(&self.road_manager.roads);
    }
}
