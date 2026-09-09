use crate::renderer::gizmo::gizmo::Gizmo;
use crate::renderer::props::Props;
use crate::world::buildings::building_mesher::BuildingMeshManager;
use crate::world::buildings::buildings::Buildings;
use crate::world::buildings::zoning::Zoning;
use crate::world::camera::Camera;
use crate::world::cars::parking::ParkingStorage;
use crate::world::roads::road_mesh_manager::ChunkId;
use crate::world::roads::road_subsystem::ChunkGpuMesh;
use crate::world::terrain::terrain_subsystem::Terrain;
use std::collections::HashMap;
use wgpu::util::DeviceExt;
use wgpu::{Device, Queue};
use wgpu_render_manager::renderer::RenderManager;

pub struct BuildingRenderer {
    pub mesh_manager: BuildingMeshManager,
    pub chunk_gpu: HashMap<ChunkId, ChunkGpuMesh>,
}
impl BuildingRenderer {
    pub fn new(device: &Device) -> Self {
        Self {
            mesh_manager: BuildingMeshManager::new(),
            chunk_gpu: Default::default(),
        }
    }

    /// Render-only update: processes commands for preview/mesh, rebuilds chunk meshes, uploads to GPU.
    pub fn update(
        &mut self,
        render_manager: &mut RenderManager,
        terrain: &mut Terrain,
        props: &mut Props,
        buildings: &mut Buildings,
        zoning: &mut Zoning,
        parking_storage: &mut ParkingStorage,
        device: &Device,
        queue: &Queue,
        camera: &Camera,
        gizmo: &mut Gizmo,
    ) {
        let visible_chunk_ids: Vec<ChunkId> = terrain.visible.iter().map(|v| v.id).collect(); // At least 8KB cloned at 32 render distance... Every frame

        for chunk_id in visible_chunk_ids {
            let needs_rebuild = self.mesh_manager.chunk_needs_update(chunk_id, buildings);

            let mesh = if needs_rebuild {
                self.mesh_manager.update_chunk_mesh(
                    render_manager,
                    terrain,
                    props,
                    chunk_id,
                    buildings,
                    zoning,
                    parking_storage,
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
                    label: Some("Building Chunk VB"),
                    contents: bytemuck::cast_slice(&mesh.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

                let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Building Chunk IB"),
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
        }
    }
}
