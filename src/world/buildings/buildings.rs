use crate::helpers::positions::{ChunkCoord, WorldPos};
use crate::renderer::gizmo::gizmo::Gizmo;
use crate::ui::input::Input;
use crate::ui::variables::Variables;
use crate::world::buildings::zoning::{LotId, Zoning, ZoningType};
use crate::world::camera::Camera;
use crate::world::cars::car_structs::{ChunkDistance, SimTime};
use crate::world::roads::road_mesh_manager::{ChunkId, RoadMeshManager, chunk_id_to_coord};
use crate::world::roads::road_structs::SegmentId;
use crate::world::roads::road_subsystem::{ChunkGpuMesh, Roads};
use crate::world::terrain::terrain_subsystem::Terrain;
use rayon::iter::IntoParallelRefMutIterator;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::{Deref, DerefMut};
use std::slice::{Iter, IterMut};
use wgpu::util::DeviceExt;
use wgpu::{Device, Queue, VertexAttribute, VertexFormat};
use wgpu_render_manager::generator::{TextureKey, TextureParams};
use wgpu_render_manager::renderer::RenderManager;

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Color(pub [f32; 3]);

impl Color {
    fn white() -> Color {
        Color([1.0, 1.0, 1.0])
    }
}

impl Deref for Color {
    type Target = [f32; 3];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Color {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
pub type BuildingId = u32;

#[derive(Serialize, Deserialize, Clone, Default)]
pub enum RoofType {
    /// Just a flat Roof
    #[default]
    Flat,
    /// Roof with just one side inclined, deg
    Angled(f32),
    /// Roof with two sides at an equal but opposite incline, deg
    Triangle(f32),
}
impl Hash for RoofType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            RoofType::Flat => {
                0u8.hash(state);
            }
            RoofType::Angled(v) => {
                1u8.hash(state);
                v.to_bits().hash(state);
            }
            RoofType::Triangle(v) => {
                2u8.hash(state);
                v.to_bits().hash(state);
            }
        }
    }
}
#[derive(Serialize, Deserialize, Clone, Default, Hash)]
pub enum RoofMaterial {
    #[default]
    Shingles,
    Metal,
    Custom(TextureKey),
}
#[derive(Serialize, Deserialize, Clone, Default, Hash)]
pub struct MiscBuildingParams {
    pub window_material_accent: WallMaterial,
    pub solar_modules: bool,
    pub antenna: bool,
}
#[derive(Serialize, Deserialize, Clone, Default, Hash)]
pub struct BasementParams {}
#[derive(Serialize, Deserialize, Clone, Hash)]
pub enum WallMaterial {
    Paint(Color),
    Metal,
    Custom(TextureKey),
}
impl Default for WallMaterial {
    fn default() -> Self {
        WallMaterial::Paint(Color::white())
    }
}
impl Hash for Color {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self[0].to_bits().hash(state);
        self[1].to_bits().hash(state);
        self[2].to_bits().hash(state);
    }
}
#[derive(Serialize, Deserialize, Clone, Default, Hash)]
pub enum GardenLook {
    #[default]
    Normal,
    Overgrown,
}
#[derive(Serialize, Deserialize, Clone, Default, Hash)]
pub struct GardenParams {
    pub look: GardenLook,
}
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct BuildingParams {
    pub roof: RoofType,
    pub roof_material: RoofMaterial,
    pub wall_material: WallMaterial,
    pub height: f32,
    pub basement: BasementParams,
    pub garden: GardenParams,
    pub miscellaneous: MiscBuildingParams,
}
impl Hash for BuildingParams {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.roof.hash(state);
        self.roof_material.hash(state);
        self.wall_material.hash(state);
        self.height.to_bits().hash(state); // <- important
        self.basement.hash(state);
        self.garden.hash(state);
        self.miscellaneous.hash(state);
    }
}
#[derive(Serialize, Deserialize, Clone, Default, Hash)]
pub struct BuildingParamsLevels {
    pub level0: BuildingParams,
    pub level1: BuildingParams,
    pub level2: BuildingParams,
    pub level3: BuildingParams,
    pub level4: BuildingParams,
    pub level5: BuildingParams,
}
#[derive(Serialize, Deserialize, Clone, Default, Hash)]
pub struct Building {
    pub id: BuildingId,
    pub position: WorldPos,
    pub segment_id: SegmentId,
    pub lot_id: LotId,
    pub building_params: BuildingParamsLevels,
}

#[derive(Clone, Default)]
pub struct Buildings {
    pub storage: BuildingStorage,
    pub zoning: Zoning,
}

impl Buildings {
    pub fn new() -> Buildings {
        Self {
            storage: BuildingStorage::new(),
            zoning: Zoning::new(),
        }
    }
    pub fn update(
        &mut self,
        camera: &Camera,
        terrain: &Terrain,
        roads: &Roads,
        road_mesh_manager: &RoadMeshManager,
        input: &mut Input,
        gizmo: &mut Gizmo,
        variables: &Variables,
    ) {
        self.zoning.update(
            camera,
            terrain,
            roads,
            road_mesh_manager,
            input,
            gizmo,
            variables,
        );
    }
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct BuildingStorage {
    pub building_chunk_storage: BuildingChunkStorage,
    buildings: Vec<Option<Building>>,
    free_list: Vec<BuildingId>,
    // Reverse mapping so I can remove from chunks in O(1) when destroying
    building_locations: HashMap<BuildingId, ChunkCoord>,
    center_chunk: ChunkCoord,
}

impl BuildingStorage {
    pub fn update(&mut self, target_chunk: ChunkCoord) {
        self.center_chunk = target_chunk;

        self.update_building_chunk_distances()
    }
    pub fn move_building_between_chunks(
        &mut self,
        from: ChunkCoord,
        to: ChunkCoord,
        building_id: BuildingId,
    ) {
        self.building_chunk_storage
            .remove_building(from, building_id);
        let building_chunk_distance = ChunkDistance::from_chunk_positions(self.center_chunk, to);
        self.building_chunk_storage
            .add_building(to, building_chunk_distance, building_id);
    }
    pub fn iter_buildings(&self) -> Iter<'_, Option<Building>> {
        self.buildings.iter()
    }
    pub fn iter_mut_buildings(&mut self) -> IterMut<'_, Option<Building>> {
        self.buildings.iter_mut()
    }
    /// Returns a parallel mutable iterator over building slots.
    /// Each slot is independent, so this is safe for rayon.
    pub fn par_iter_mut_buildings(&mut self) -> rayon::slice::IterMut<'_, Option<Building>> {
        self.buildings.par_iter_mut()
    }
    pub fn update_building_chunk_distances(&mut self) {
        let (moved, removed) = self
            .building_chunk_storage
            .update_all_distances(self.center_chunk);

        if !removed.is_empty() {
            //println!("Removed {} empty building chunks", removed.len());
        }
        if moved > 0 {
            //println!("Moved {} chunks between distance tiers", moved);
        }
    }
    pub fn new() -> Self {
        let mut buildings: Vec<Option<Building>> = Vec::new();
        buildings.push(None); // reserve index 0 — never use it for a real building, because building 0 doesn't get RTX shadows.
        Self {
            building_chunk_storage: BuildingChunkStorage::new(),
            buildings,
            free_list: Vec::new(),
            building_locations: HashMap::new(),
            center_chunk: ChunkCoord::zero(),
        }
    }

    pub fn spawn(
        &mut self,
        chunk_coord: ChunkCoord,
        zoning_type: ZoningType,
        building_chunk_distance: ChunkDistance,
        mut building: Building,
    ) -> BuildingId {
        let building_id = if let Some(reused_id) = self.free_list.pop() {
            // Reuse slot - III know it's None because it's in free_list
            building.id = reused_id;
            self.buildings[reused_id as usize] = Some(building);
            reused_id
        } else {
            let new_id = self.buildings.len() as u32;
            building.id = new_id;
            self.buildings.push(Some(building));
            new_id
        };

        // Add to chunk storage and record location
        self.building_chunk_storage
            .add_building(chunk_coord, building_chunk_distance, building_id);
        self.building_locations.insert(building_id, chunk_coord);
        println!("Created building: {}", building_id);
        building_id
    }

    pub fn despawn(&mut self, id: BuildingId) {
        if id == 0 {
            // index 0 is reserved, ignore attempts to despawn it
            return;
        }
        if self
            .buildings
            .get(id as usize)
            .and_then(|opt| opt.as_ref())
            .is_some()
        {
            // Remove from chunk first using reverse lookup
            if let Some(chunk_coord) = self.building_locations.remove(&id) {
                self.building_chunk_storage.remove_building(chunk_coord, id);
            }

            // Actually free the slot
            self.buildings[id as usize] = None;
            self.free_list.push(id);
        }
    }

    pub fn building_count(&self) -> usize {
        self.buildings.len() - self.free_list.len() - 1
    }

    #[inline]
    pub fn get(&self, id: BuildingId) -> Option<&Building> {
        self.buildings.get(id as usize)?.as_ref()
    }

    #[inline]
    pub fn get_mut(&mut self, id: BuildingId) -> Option<&mut Building> {
        self.buildings.get_mut(id as usize)?.as_mut()
    }
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct BuildingChunk {
    pub distance: ChunkDistance,
    pub building_ids: Vec<BuildingId>,
    pub last_update_time: SimTime,
}
impl BuildingChunk {
    pub fn new(distance: ChunkDistance, building_ids: Vec<BuildingId>) -> Self {
        Self {
            distance,
            building_ids,
            last_update_time: 0.0,
        }
    }
    pub fn empty(distance: ChunkDistance) -> Self {
        Self {
            distance,
            building_ids: Vec::new(),
            last_update_time: 0.0,
        }
    }
}
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct BuildingChunkStorage {
    close: HashMap<ChunkCoord, BuildingChunk>,
    medium: HashMap<ChunkCoord, BuildingChunk>,
    far: HashMap<ChunkCoord, BuildingChunk>,
}

impl BuildingChunkStorage {
    pub fn new() -> Self {
        Self {
            close: HashMap::new(),
            medium: HashMap::new(),
            far: HashMap::new(),
        }
    }

    #[inline]
    pub fn close(&self) -> &HashMap<ChunkCoord, BuildingChunk> {
        &self.close
    }

    #[inline]
    pub fn close_mut(&mut self) -> &mut HashMap<ChunkCoord, BuildingChunk> {
        &mut self.close
    }

    #[inline]
    pub fn medium(&self) -> &HashMap<ChunkCoord, BuildingChunk> {
        &self.medium
    }

    #[inline]
    pub fn medium_mut(&mut self) -> &mut HashMap<ChunkCoord, BuildingChunk> {
        &mut self.medium
    }

    #[inline]
    pub fn far(&self) -> &HashMap<ChunkCoord, BuildingChunk> {
        &self.far
    }

    #[inline]
    pub fn far_mut(&mut self) -> &mut HashMap<ChunkCoord, BuildingChunk> {
        &mut self.far
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    #[inline]
    fn map_for_distance(&self, dist: &ChunkDistance) -> &HashMap<ChunkCoord, BuildingChunk> {
        match dist {
            ChunkDistance::Close => &self.close,
            ChunkDistance::Medium => &self.medium,
            ChunkDistance::Far => &self.far,
        }
    }

    #[inline]
    fn map_for_distance_mut(
        &mut self,
        dist: &ChunkDistance,
    ) -> &mut HashMap<ChunkCoord, BuildingChunk> {
        match dist {
            ChunkDistance::Close => &mut self.close,
            ChunkDistance::Medium => &mut self.medium,
            ChunkDistance::Far => &mut self.far,
        }
    }

    /// Find which distance tier a building chunk is in (if it exists)
    #[inline]
    pub fn find_distance(&self, coord: &ChunkCoord) -> Option<ChunkDistance> {
        if self.close.contains_key(coord) {
            Some(ChunkDistance::Close)
        } else if self.medium.contains_key(coord) {
            Some(ChunkDistance::Medium)
        } else if self.far.contains_key(coord) {
            Some(ChunkDistance::Far)
        } else {
            None
        }
    }

    // ========================================================================
    // Lookup by coord (searches all tiers, still O(1))
    // ========================================================================

    #[inline]
    pub fn get(&self, coord: &ChunkCoord) -> Option<&BuildingChunk> {
        self.close
            .get(coord)
            .or_else(|| self.medium.get(coord))
            .or_else(|| self.far.get(coord))
    }

    #[inline]
    pub fn get_mut(&mut self, coord: &ChunkCoord) -> Option<&mut BuildingChunk> {
        if let Some(chunk) = self.close.get_mut(coord) {
            return Some(chunk);
        }
        if let Some(chunk) = self.medium.get_mut(coord) {
            return Some(chunk);
        }
        self.far.get_mut(coord)
    }

    #[inline]
    pub fn contains(&self, coord: &ChunkCoord) -> bool {
        self.close.contains_key(coord)
            || self.medium.contains_key(coord)
            || self.far.contains_key(coord)
    }

    pub fn remove(&mut self, coord: &ChunkCoord) -> Option<BuildingChunk> {
        self.close
            .remove(coord)
            .or_else(|| self.medium.remove(coord))
            .or_else(|| self.far.remove(coord))
    }

    // ========================================================================
    // Iteration over all chunks
    // ========================================================================

    pub fn iter(&self) -> impl Iterator<Item = (&ChunkCoord, &BuildingChunk)> {
        self.close
            .iter()
            .chain(self.medium.iter())
            .chain(self.far.iter())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&ChunkCoord, &mut BuildingChunk)> {
        self.close
            .iter_mut()
            .chain(self.medium.iter_mut())
            .chain(self.far.iter_mut())
    }

    pub fn values(&self) -> impl Iterator<Item = &BuildingChunk> {
        self.close
            .values()
            .chain(self.medium.values())
            .chain(self.far.values())
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut BuildingChunk> {
        self.close
            .values_mut()
            .chain(self.medium.values_mut())
            .chain(self.far.values_mut())
    }

    pub fn keys(&self) -> impl Iterator<Item = &ChunkCoord> {
        self.close
            .keys()
            .chain(self.medium.keys())
            .chain(self.far.keys())
    }

    // ========================================================================
    // Fast distance-specific building accessors - TRUE O(1) for close buildings!
    // ========================================================================

    /// O(close_chunk_count) - only iterates close chunks
    pub fn close_buildings(&self) -> Vec<BuildingId> {
        self.close
            .values()
            .flat_map(|chunk| chunk.building_ids.iter().copied())
            .collect()
    }

    /// Iterator version - zero allocation
    pub fn close_building_ids(&self) -> impl Iterator<Item = BuildingId> + '_ {
        self.close
            .values()
            .flat_map(|chunk| chunk.building_ids.iter().copied())
    }

    /// O(medium_chunk_count)
    pub fn medium_buildings(&self) -> Vec<BuildingId> {
        self.medium
            .values()
            .flat_map(|chunk| chunk.building_ids.iter().copied())
            .collect()
    }

    pub fn medium_building_ids(&self) -> impl Iterator<Item = BuildingId> + '_ {
        self.medium
            .values()
            .flat_map(|chunk| chunk.building_ids.iter().copied())
    }

    /// O(far_chunk_count)
    pub fn far_buildings(&self) -> Vec<BuildingId> {
        self.far
            .values()
            .flat_map(|chunk| chunk.building_ids.iter().copied())
            .collect()
    }

    pub fn far_building_ids(&self) -> impl Iterator<Item = BuildingId> + '_ {
        self.far
            .values()
            .flat_map(|chunk| chunk.building_ids.iter().copied())
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    pub fn close_chunk_count(&self) -> usize {
        self.close.len()
    }

    pub fn medium_chunk_count(&self) -> usize {
        self.medium.len()
    }

    pub fn far_chunk_count(&self) -> usize {
        self.far.len()
    }

    pub fn total_chunk_count(&self) -> usize {
        self.close.len() + self.medium.len() + self.far.len()
    }

    pub fn close_building_count(&self) -> usize {
        self.close.values().map(|c| c.building_ids.len()).sum()
    }

    pub fn medium_building_count(&self) -> usize {
        self.medium.values().map(|c| c.building_ids.len()).sum()
    }

    pub fn far_building_count(&self) -> usize {
        self.far.values().map(|c| c.building_ids.len()).sum()
    }

    pub fn add_building(
        &mut self,
        chunk_coord: ChunkCoord,
        dist: ChunkDistance,
        building_id: BuildingId,
    ) {
        let map = self.map_for_distance_mut(&dist);
        let chunk = map
            .entry(chunk_coord)
            .or_insert_with(|| BuildingChunk::empty(dist.clone()));

        debug_assert_eq!(
            chunk.distance, dist,
            "ChunkDistance mismatch when adding to existing chunk"
        );

        chunk.building_ids.push(building_id);
    }

    pub fn remove_building(&mut self, chunk_coord: ChunkCoord, building_id: BuildingId) {
        // Try each tier - only one can contain the chunk
        let maps: [&mut HashMap<ChunkCoord, BuildingChunk>; 3] =
            [&mut self.close, &mut self.medium, &mut self.far];

        for map in maps {
            if let Some(chunk) = map.get_mut(&chunk_coord) {
                chunk.building_ids.retain(|&x| x != building_id);
                if chunk.building_ids.is_empty() {
                    map.remove(&chunk_coord);
                }
                return;
            }
        }
    }

    /// Move a chunk to a different distance tier. Returns true if moved.
    pub fn update_chunk_distance(
        &mut self,
        coord: ChunkCoord,
        new_distance: ChunkDistance,
    ) -> bool {
        // Find current tier
        let current = if self.close.contains_key(&coord) {
            ChunkDistance::Close
        } else if self.medium.contains_key(&coord) {
            ChunkDistance::Medium
        } else if self.far.contains_key(&coord) {
            ChunkDistance::Far
        } else {
            return false;
        };

        // Already in correct tier
        if current == new_distance {
            return false;
        }

        // Remove from old tier
        let mut chunk = match current {
            ChunkDistance::Close => self.close.remove(&coord),
            ChunkDistance::Medium => self.medium.remove(&coord),
            ChunkDistance::Far => self.far.remove(&coord),
        }
        .expect("Chunk must exist, we just checked");

        // Update distance marker and insert into new tier
        chunk.distance = new_distance.clone();
        let new_map = self.map_for_distance_mut(&new_distance);
        new_map.insert(coord, chunk);

        true
    }

    /// Bulk update all chunk distances. Returns number of chunks moved.
    pub fn update_all_distances(&mut self, center_chunk: ChunkCoord) -> (usize, Vec<ChunkCoord>) {
        let mut moved = 0;
        let mut to_remove: Vec<ChunkCoord> = Vec::new();

        // Collect all coords and their new distances
        let updates: Vec<(ChunkCoord, ChunkDistance, bool)> = self
            .iter()
            .map(|(coord, chunk)| {
                let is_empty = chunk.building_ids.is_empty();
                let dist2 = center_chunk.dist2(coord);
                let new_dist = ChunkDistance::from_dist2(dist2);
                (*coord, new_dist, is_empty)
            })
            .collect();

        // Apply updates
        for (coord, new_dist, is_empty) in updates {
            if is_empty {
                to_remove.push(coord);
            } else if self.update_chunk_distance(coord, new_dist) {
                moved += 1;
            }
        }

        // Remove empty chunks
        for coord in &to_remove {
            self.remove(coord);
        }

        (moved, to_remove)
    }
}

fn calculate_unique_params(total_buildings: usize, max_unique: usize) -> usize {
    let k = 50.0;
    let alpha = 0.6;
    let n_buildings = total_buildings as f64;
    let params = n_buildings * (k / n_buildings).powf(alpha);
    params.clamp(1.0, max_unique as f64).round() as usize
}

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
        buildings: &Buildings,
        device: &Device,
        queue: &Queue,
        camera: &Camera,
        gizmo: &mut Gizmo,
    ) {
        for v in &terrain.visible {
            let chunk_id = v.id;
            let needs_rebuild = self.mesh_manager.chunk_needs_update(chunk_id, buildings);

            let mesh = if needs_rebuild {
                self.mesh_manager.update_chunk_mesh(
                    render_manager,
                    terrain,
                    chunk_id,
                    buildings,
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

pub struct BuildingMeshManager {
    chunk_cache: HashMap<ChunkId, BuildingChunkMesh>,
}

impl BuildingMeshManager {
    pub fn new() -> Self {
        Self {
            chunk_cache: HashMap::new(),
        }
    }

    pub fn get_chunk_mesh(&self, chunk_id: ChunkId) -> Option<&BuildingChunkMesh> {
        self.chunk_cache.get(&chunk_id)
    }

    pub fn _invalidate_chunk(&mut self, chunk_id: ChunkId) {
        self.chunk_cache.remove(&chunk_id);
    }

    pub fn _clear_cache(&mut self) {
        self.chunk_cache.clear();
    }

    pub fn chunk_needs_update(&self, chunk_id: ChunkId, buildings: &Buildings) -> bool {
        match self.chunk_cache.get(&chunk_id) {
            None => true,
            Some(mesh) => {
                mesh.topo_version != compute_building_chunk_topo_version(chunk_id, buildings)
            }
        }
    }

    /// Build mesh for a chunk
    /// Build mesh for a chunk
    pub fn build_mesh_for_chunk(
        &mut self,
        render_manager: &mut RenderManager,
        terrain: &Terrain,
        cid: ChunkId,
        buildings: &Buildings,
        gizmo: &mut Gizmo,
    ) -> BuildingChunkMesh {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let lot_ids: Vec<LotId> = buildings.zoning.zoning_storage.lots_in_chunk(cid);
        if !lot_ids.is_empty() {
            println!("Building chunk mesh!");
        }

        for lot in lot_ids
            .iter()
            .flat_map(|id| buildings.zoning.zoning_storage.get_lot(*id))
        {
            let Some(building) = buildings.storage.get(lot.building_id) else {
                continue;
            };

            let wall_key = match &building.building_params.level0.wall_material {
                WallMaterial::Paint(_) => TextureKey::new("goo", TextureParams::default(), 512),
                WallMaterial::Metal => TextureKey::new("goo", TextureParams::default(), 512),
                WallMaterial::Custom(key) => key.clone(),
            };

            let material_ids = render_manager.ensure_textures(&[wall_key]);
            let material_id = material_ids[0];

            let points = &lot.bounds;
            let n = points.len();
            if n < 2 {
                continue;
            }

            let wall_height = 20.0;

            // Build vertical walls around the perimeter.
            // Works for any polygon size.
            for i in 0..n {
                let j = (i + 1) % n;

                let p0 = &points[i];
                let p1 = &points[j];

                let base_index = vertices.len() as u32;

                let a0 = [p0.local.x, p0.local.y, p0.local.z];
                let b0 = [p1.local.x, p1.local.y, p1.local.z];
                let b1 = [p1.local.x, p1.local.y + wall_height, p1.local.z];
                let a1 = [p0.local.x, p0.local.y + wall_height, p0.local.z];

                // Edge direction in XZ
                let dx = p1.local.x - p0.local.x;
                let dz = p1.local.z - p0.local.z;

                // Outward-ish wall normal. If winding is reversed, this flips.
                let len = (dx * dx + dz * dz).sqrt().max(0.0001);
                let normal = [dz / len, 0.0, -dx / len];

                vertices.push(BuildingVertex {
                    chunk_xz: [p0.chunk.x, p0.chunk.z],
                    local_position: a0,
                    normal,
                    uv: [0.0, 0.0],
                    color: [70.2, 80.0, 70.2, 1.0],
                    material_id,
                });
                vertices.push(BuildingVertex {
                    chunk_xz: [p1.chunk.x, p1.chunk.z],
                    local_position: b0,
                    normal,
                    uv: [1.0, 0.0],
                    color: [70.2, 80.0, 70.2, 1.0],
                    material_id,
                });
                vertices.push(BuildingVertex {
                    chunk_xz: [p1.chunk.x, p1.chunk.z],
                    local_position: b1,
                    normal,
                    uv: [1.0, 1.0],
                    color: [70.2, 80.0, 70.2, 1.0],
                    material_id,
                });
                vertices.push(BuildingVertex {
                    chunk_xz: [p0.chunk.x, p0.chunk.z],
                    local_position: a1,
                    normal,
                    uv: [0.0, 1.0],
                    color: [70.2, 80.0, 70.2, 1.0],
                    material_id,
                });

                // Two triangles per wall quad
                indices.extend_from_slice(&[
                    base_index,
                    base_index + 2,
                    base_index + 1,
                    base_index,
                    base_index + 3,
                    base_index + 2,
                ]);
            }
            let roof_base = vertices.len() as u32;

            // push top vertices (already elevated)
            for p in points {
                vertices.push(BuildingVertex {
                    chunk_xz: [p.chunk.x, p.chunk.z],
                    local_position: [p.local.x, p.local.y + wall_height, p.local.z],
                    normal: [0.0, 1.0, 0.0],
                    uv: [p.local.x, p.local.z], // lazy planar mapping
                    color: [70.2, 50.0, 70.2, 1.0],
                    material_id,
                });
            }

            // triangle fan
            for i in 1..(n - 1) {
                indices.extend_from_slice(&[
                    roof_base,
                    roof_base + (i as u32 + 1),
                    roof_base + i as u32,
                ]);
            }
        }

        BuildingChunkMesh {
            vertices,
            indices,
            topo_version: compute_building_chunk_topo_version(cid, buildings),
        }
    }
    pub fn update_chunk_mesh(
        &mut self,
        render_manager: &mut RenderManager,
        terrain: &Terrain,
        chunk_id: ChunkId,
        buildings: &Buildings,
        gizmo: &mut Gizmo,
    ) -> &BuildingChunkMesh {
        let mesh = self.build_mesh_for_chunk(render_manager, terrain, chunk_id, buildings, gizmo);
        self.chunk_cache.insert(chunk_id, mesh);
        self.chunk_cache.get(&chunk_id).unwrap()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BuildingVertex {
    pub chunk_xz: [i32; 2],
    pub local_position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    pub material_id: u32,
}

impl BuildingVertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // loc0 chunk_xz
                VertexAttribute {
                    shader_location: 0,
                    offset: 0,
                    format: VertexFormat::Sint32x2,
                },
                // @location(1) chunk-local position
                VertexAttribute {
                    shader_location: 1,
                    offset: 8,
                    format: VertexFormat::Float32x3,
                },
                // @location(2) normals
                VertexAttribute {
                    offset: 20,
                    shader_location: 2,
                    format: VertexFormat::Float32x3,
                },
                // @location(3) uv
                VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: VertexFormat::Float32x2,
                },
                // @location(4) color
                VertexAttribute {
                    offset: 40,
                    shader_location: 4,
                    format: VertexFormat::Float32x4,
                },
                // @location(5) material_id
                VertexAttribute {
                    offset: 56,
                    shader_location: 5,
                    format: VertexFormat::Uint32,
                },
            ],
        }
    }
}
#[derive(Clone, Debug)]
pub struct BuildingChunkMesh {
    pub vertices: Vec<BuildingVertex>,
    pub indices: Vec<u32>,
    pub topo_version: u64,
}

impl BuildingChunkMesh {
    #![allow(dead_code)]
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
}
#[derive(Clone, Debug)]
pub struct BuildingMeshIndex {
    pub building_id: BuildingId,
    pub indices_start: u32,
    pub indices_count: u32,
}
fn compute_building_chunk_topo_version(chunk_id: ChunkId, buildings: &Buildings) -> u64 {
    if let Some(building_chunk) = buildings
        .storage
        .building_chunk_storage
        .get(&chunk_id_to_coord(chunk_id))
    {
        let hasher = &mut DefaultHasher::default();
        for building in building_chunk
            .building_ids
            .iter()
            .flat_map(|id| buildings.storage.get(*id))
        {
            building.hash(hasher);
        }
        return hasher.finish();
    }
    0
}
