use crate::helpers::modpack::ModManager;
use crate::renderer::gizmo::gizmo::Gizmo;
use crate::renderer::props::Props;
use crate::world::buildings::building_mesher::{BuildingMeshBuilder, BuildingMeshManager};
use crate::world::buildings::buildings::{Building, BuildingId, Buildings};
use crate::world::buildings::zoning::{Lot, Zoning};
use crate::world::camera::Camera;
use crate::world::cars::parking::ParkingStorage;
use crate::world::roads::road_mesh_manager::ChunkCoord;
use crate::world::roads::road_subsystem::ChunkGpuMesh;
use crate::world::terrain::terrain_subsystem::{PreviewBuilding, PreviewBuildingModel, Terrain};
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::mem::take;
use wgpu::util::DeviceExt;
use wgpu::{Device, Queue};
use wgpu_render_manager::renderer::RenderManager;

pub struct BuildingRenderer {
    pub mesh_manager: BuildingMeshManager,
    pub chunk_gpu: HashMap<ChunkCoord, ChunkGpuMesh>,
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
        mod_manager: &ModManager,
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
        let visible_chunk_coords: Vec<ChunkCoord> =
            terrain.visible.iter().map(|v| v.chunk_coord).collect(); // At least 8KB cloned at 32 render distance... Every frame

        for chunk_coord in visible_chunk_coords {
            let needs_rebuild = self.mesh_manager.chunk_needs_update(chunk_coord, buildings);

            let mesh = if needs_rebuild {
                self.mesh_manager.update_chunk_mesh(
                    render_manager,
                    mod_manager,
                    terrain,
                    props,
                    chunk_coord,
                    buildings,
                    zoning,
                    parking_storage,
                    gizmo,
                )
            } else {
                match self.mesh_manager.get_chunk_mesh(chunk_coord) {
                    Some(m) => m,
                    None => continue,
                }
            };

            if mesh.indices.is_empty() || mesh.vertices.is_empty() {
                self.chunk_gpu.remove(&chunk_coord);
                continue;
            }

            let needs_gpu_upload = match self.chunk_gpu.get(&chunk_coord) {
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
                    chunk_coord,
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

    // pub fn update_preview(&mut self, pb: &mut PreviewBuilding, render_manager: &mut RenderManager, terrain: &mut Terrain, buildings: &mut Buildings, zoning: &mut Zoning, props: &mut Props, parking_storage: &mut ParkingStorage, gizmo: &mut Gizmo) {
    //     if pb.cached_model.is_none() {
    //         if let Some(lot) = zoning.zoning_storage.get_mut_lot(pb.lot_id) {
    //             if let Some(building_id) = lot.building_id {
    //                 let mesh = &mut BuildingMeshBuilder::default();
    //                 self.mesh_manager.build_mesh_for_building(mesh, render_manager, terrain, buildings, false, building_id, lot, props, parking_storage, gizmo);
    //
    //             }
    //
    //         }
    //
    //     }
    //
    // }
    pub fn render_preview(
        &mut self,
        render_manager: &mut RenderManager,
        mod_manager: &ModManager,
        terrain: &mut Terrain,
        props: &mut Props,
        parking_storage: &mut ParkingStorage,
        buildings: &mut Buildings,
        zoning: &mut Zoning,
        gizmo: &mut Gizmo,
        pb: &mut PreviewBuilding,
    ) {
        let Some(lot) = zoning.zoning_storage.get_mut_lot(pb.lot_id) else {
            return;
        };
        let building_id = pb.building_id;
        let Some(building) = buildings.storage.get(building_id) else {
            return;
        };
        let signature = building_mesh_signature(lot, building);

        if let Some(model) = pb.cached_model.as_ref() {
            if model.signature != signature {
                self.rebuild_preview_cache(
                    pb,
                    signature,
                    true,
                    render_manager,
                    mod_manager,
                    terrain,
                    buildings,
                    building_id,
                    lot,
                    props,
                    parking_storage,
                    gizmo,
                );
            } else if model.old_pos != pb.new_pos || model.old_dir != pb.new_dir {
                self.rebuild_preview_cache(
                    pb,
                    signature,
                    false,
                    render_manager,
                    mod_manager,
                    terrain,
                    buildings,
                    building_id,
                    lot,
                    props,
                    parking_storage,
                    gizmo,
                );
            };
        } else {
            self.rebuild_preview_cache(
                pb,
                signature,
                true,
                render_manager,
                mod_manager,
                terrain,
                buildings,
                building_id,
                lot,
                props,
                parking_storage,
                gizmo,
            );
        }
    }

    pub fn rebuild_preview_cache(
        &mut self,
        pb: &mut PreviewBuilding,
        signature: u64,
        regenerate_mesh: bool,
        render_manager: &mut RenderManager,
        mod_manager: &ModManager,
        terrain: &mut Terrain,
        buildings: &mut Buildings,
        building_id: BuildingId,
        lot: &mut Lot,
        props: &mut Props,
        parking_storage: &mut ParkingStorage,
        gizmo: &mut Gizmo,
    ) {
        let mesh = if regenerate_mesh {
            let mut mesh = BuildingMeshBuilder::default();
            self.mesh_manager.build_mesh_for_building(
                &mut mesh,
                render_manager,
                mod_manager,
                terrain,
                buildings,
                false,
                building_id,
                lot,
                props,
                parking_storage,
                gizmo,
            );
            mesh
        } else {
            // Just move the props/trees
            let Some(building) = buildings.storage.get(building_id) else {
                return;
            };
            let Some(bm) = pb.cached_model.as_mut() else {
                return;
            };
            props.move_multiple_offset(
                building.prop_instance_ids.as_slice(),
                bm.old_pos,
                bm.old_dir,
                pb.new_pos,
                pb.new_dir,
            );
            bm.mesh
                .move_rotate(bm.old_pos, bm.old_dir, pb.new_pos, pb.new_dir);
            take(&mut bm.mesh)
        };

        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            pb.cached_model = None;
            return;
        }

        let vertex =
            render_manager
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Preview Building VB"),
                    contents: bytemuck::cast_slice(&mesh.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        let index = render_manager
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Preview Building IB"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        let index_count = mesh.indices.len() as u32;

        pb.cached_model = Some(PreviewBuildingModel {
            vertex,
            index,
            mesh,
            signature,
            old_pos: pb.new_pos,
            old_dir: pb.new_dir,
        });
    }
}
pub fn building_mesh_signature(lot: &Lot, building: &Building) -> u64 {
    let hasher = &mut DefaultHasher::default();
    //lot.bounds_version.hash(hasher);
    //WorldPos::area(lot.bounds.as_slice()).to_bits().hash(hasher);
    lot.bounds.as_slice().hash(hasher);
    building.level.hash(hasher);
    building.design_source.hash(hasher);
    hasher.finish()
}
