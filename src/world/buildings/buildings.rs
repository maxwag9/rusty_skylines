use crate::helpers::positions::{ChunkCoord, WorldPos};
use crate::renderer::gizmo::gizmo::Gizmo;
use crate::ui::input::Input;
use crate::ui::parser::Value;
use crate::ui::variables::Variables;
use crate::world::buildings::zoning::{Lot, LotId, Tile, TileType, Zoning, ZoningType};
use crate::world::camera::Camera;
use crate::world::cars::car_structs::{ChunkDistance, SimTime};
use crate::world::roads::road_mesh_manager::{ChunkId, RoadMeshManager, chunk_id_to_coord};
use crate::world::roads::road_structs::SegmentId;
use crate::world::roads::road_subsystem::{ChunkGpuMesh, Roads};
use crate::world::statisticals::demands::ZoningDemand;
use crate::world::terrain::terrain_subsystem::Terrain;
use glam::Vec2;
use rayon::iter::IntoParallelRefMutIterator;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::{Deref, DerefMut};
use std::slice::{Iter, IterMut};
use wgpu::util::DeviceExt;
use wgpu::{Device, Queue, VertexAttribute, VertexFormat};
use wgpu_render_manager::generator::{TextureKey, TextureParams};
use wgpu_render_manager::renderer::RenderManager;

#[derive(Debug, Copy, Clone, Serialize, Deserialize, Default, Hash)]
pub enum BuildingUsage {
    #[default]
    Residential,
    Commercial,
    Industrial,
    Office,
}
impl BuildingUsage {
    pub fn from_value(value: &Value) -> Self {
        match value {
            Value::String(s) => match s.to_lowercase().as_str() {
                "residential" => BuildingUsage::Residential,
                "commercial" => BuildingUsage::Commercial,
                "industrial" => BuildingUsage::Industrial,
                "office" => BuildingUsage::Office,
                _ => BuildingUsage::Residential,
            },
            _ => BuildingUsage::Residential,
        }
    }
    pub fn from_zoning_type(zoning_type: &ZoningType) -> Self {
        match zoning_type {
            ZoningType::Residential => BuildingUsage::Residential,
            ZoningType::Commercial => BuildingUsage::Commercial,
            ZoningType::Industrial => BuildingUsage::Industrial,
            ZoningType::Office => BuildingUsage::Office,
            _ => BuildingUsage::Residential,
        }
    }
}
impl Display for BuildingUsage {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildingUsage::Residential => write!(f, "residential"),
            BuildingUsage::Commercial => write!(f, "commercial"),
            BuildingUsage::Industrial => write!(f, "industrial"),
            BuildingUsage::Office => write!(f, "office"),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Color(pub [f32; 4]);

impl Color {
    fn white() -> Color {
        Color([1.0, 1.0, 1.0, 1.0])
    }
}

impl Deref for Color {
    type Target = [f32; 4];
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
    /// Roof with just one side inclined, rad
    Angled { pitch: f32, direction_rad: f32 },
    /// Roof with two sides at an equal but opposite incline, deg
    Triangle(f32),
}
impl Hash for RoofType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            RoofType::Flat => {
                0u8.hash(state);
            }
            RoofType::Angled {
                pitch,
                direction_rad,
            } => {
                1u8.hash(state);
                pitch.to_bits().hash(state);
                direction_rad.to_bits().hash(state);
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
    pub usage: BuildingUsage,
}
#[derive(Serialize, Deserialize, Clone, Default, Hash)]
pub struct BasementParams {}
#[derive(Serialize, Deserialize, Clone, Hash)]
pub enum WallMaterial {
    Paint(Color),
    Custom(TextureKey),
}
impl Default for WallMaterial {
    fn default() -> Self {
        WallMaterial::Paint(Color::white())
    }
}
#[derive(Serialize, Deserialize, Clone, Hash)]
pub enum DrivewayMaterial {
    Bricks,
    Custom(TextureKey),
}
impl Default for DrivewayMaterial {
    fn default() -> Self {
        Self::Bricks
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
    pub driveway_material: DrivewayMaterial,
    pub story_height: f32,
    pub num_stories: u16,
    pub basement: BasementParams,
    pub garden: GardenParams,
    pub miscellaneous: MiscBuildingParams,
}
impl BuildingParams {
    pub fn max_people(&self, one_story_area: f64) -> u32 {
        let total_area = one_story_area * self.num_stories as f64;

        let area_per_person = match self.miscellaneous.usage {
            BuildingUsage::Residential => 40.0,
            BuildingUsage::Commercial => 10.0,
            BuildingUsage::Industrial => 10.0,
            BuildingUsage::Office => 15.0,
        };

        (total_area / area_per_person) as u32
    }
}
impl Hash for BuildingParams {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.roof.hash(state);
        self.roof_material.hash(state);
        self.wall_material.hash(state);
        self.story_height.to_bits().hash(state); // <- important
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
pub enum BuildingLevel {
    #[default]
    Level0,
    Level1,
    Level2,
    Level3,
    Level4,
    Level5,
}
#[derive(Serialize, Deserialize, Clone, Default, Hash)]
pub struct Building {
    pub id: BuildingId,
    pub position: WorldPos,
    pub segment_id: SegmentId,
    pub lot_id: LotId,
    pub level: BuildingLevel,
    pub building_params: BuildingParamsLevels,
}
impl Building {
    pub fn current_level_params(&self) -> &BuildingParams {
        match self.level {
            BuildingLevel::Level0 => &self.building_params.level0,
            BuildingLevel::Level1 => &self.building_params.level1,
            BuildingLevel::Level2 => &self.building_params.level2,
            BuildingLevel::Level3 => &self.building_params.level3,
            BuildingLevel::Level4 => &self.building_params.level4,
            BuildingLevel::Level5 => &self.building_params.level5,
        }
    }
}
#[derive(Clone, Default)]
pub struct Buildings {
    pub storage: BuildingStorage,
}

impl Buildings {
    pub fn new() -> Buildings {
        Self {
            storage: BuildingStorage::new(),
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
        buildings: &mut Buildings,
        zoning: &mut Zoning,
        chunk_coord: ChunkCoord,
        zoning_type: ZoningType,
        building_chunk_distance: ChunkDistance,
        mut building: Building,
    ) -> BuildingId {
        let storage = &mut buildings.storage;
        let building_id = if let Some(reused_id) = storage.free_list.pop() {
            // Reuse slot - III know it's None because it's in free_list
            building.id = reused_id;
            storage.buildings[reused_id as usize] = Some(building);
            reused_id
        } else {
            let new_id = storage.buildings.len() as u32;
            building.id = new_id;
            storage.buildings.push(Some(building));
            new_id
        };

        // Add to chunk storage and record location
        storage.building_chunk_storage.add_building(
            chunk_coord,
            building_chunk_distance,
            building_id,
        );
        storage.building_locations.insert(building_id, chunk_coord);
        ZoningDemand::spawn_building(buildings, zoning, building_id);
        println!("Created building: {}", building_id);
        building_id
    }

    pub fn despawn(buildings: &mut Buildings, zoning: &mut Zoning, id: BuildingId) {
        if id == 0 {
            // index 0 is reserved, ignore attempts to despawn it
            return;
        }
        let storage = &mut buildings.storage;
        if storage
            .buildings
            .get(id as usize)
            .and_then(|opt| opt.as_ref())
            .is_some()
        {
            // Remove from chunk first using reverse lookup
            if let Some(chunk_coord) = storage.building_locations.remove(&id) {
                storage
                    .building_chunk_storage
                    .remove_building(chunk_coord, id);
            }

            // Actually free the slot
            storage.buildings[id as usize] = None;
            storage.free_list.push(id);
            ZoningDemand::despawn_building(buildings, zoning, id);
        }
    }

    pub fn building_count(&self) -> usize {
        self.buildings.len() - self.free_list.len() - 1
    }

    #[inline]
    pub fn get<I>(&self, id: I) -> Option<&Building>
    where
        I: Into<Option<BuildingId>>,
    {
        let id = id.into()?;
        self.buildings.get(id as usize)?.as_ref()
    }

    #[inline]
    pub fn get_mut<I>(&mut self, id: I) -> Option<&mut Building>
    where
        I: Into<Option<BuildingId>>,
    {
        let id = id.into()?;
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
        zoning: &mut Zoning,
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
                    zoning,
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
    pub fn build_mesh_for_chunk(
        &mut self,
        render_manager: &mut RenderManager,
        terrain: &Terrain,
        cid: ChunkId,
        buildings: &Buildings,
        zoning: &mut Zoning,
        gizmo: &mut Gizmo,
    ) -> BuildingChunkMesh {
        let mut mesh = BuildingMeshBuilder {
            vertices: Vec::new(),
            indices: Vec::new(),
        };

        let lot_ids: Vec<LotId> = zoning.zoning_storage.lots_in_chunk(cid);

        for lot in lot_ids
            .iter()
            .flat_map(|id| zoning.zoning_storage.get_lot(*id))
        {
            let Some(building) = buildings.storage.get(lot.building_id) else {
                continue;
            };

            let level = building.current_level_params();
            let story_height = level.story_height;
            let num_stories = level.num_stories;
            let wall_height = story_height * num_stories as f32;

            let wall_key = match &level.wall_material {
                WallMaterial::Paint(color) => TextureKey::new(
                    "paint",
                    TextureParams::default()
                        .with_primary_color(**color)
                        .with_secondary_color([0.0, 0.0, 0.0, 1.0]),
                    512,
                ),
                WallMaterial::Custom(key) => key.clone(),
            };
            let roof_key = match &level.roof_material {
                RoofMaterial::Shingles => TextureKey::new(
                    "shingles",
                    TextureParams::default().with_primary_color([0.4, 0.15, 0.05, 1.0]),
                    512,
                ),
                RoofMaterial::Metal => TextureKey::new(
                    "metal_roof",
                    TextureParams::default().with_primary_color([0.2, 0.2, 0.2, 1.0]),
                    512,
                ),
                RoofMaterial::Custom(key) => key.clone(),
            };
            let window_key = match &level.miscellaneous.window_material_accent {
                WallMaterial::Paint(color) => TextureKey::new(
                    "paint",
                    TextureParams::default().with_primary_color(**color),
                    512,
                ),
                WallMaterial::Custom(key) => key.clone(),
            };
            let driveway_key = match &level.driveway_material {
                DrivewayMaterial::Bricks => TextureKey::new(
                    "driveway_bricks",
                    TextureParams::default()
                        .with_primary_color([0.18, 0.18, 0.22, 1.0])
                        .with_secondary_color([0.01, 0.01, 0.01, 1.0])
                        .with_roughness(0.0)
                        .with_scale(5.0),
                    512,
                ),
                DrivewayMaterial::Custom(key) => key.clone(),
            };
            let mat_ids =
                render_manager.ensure_textures(&[wall_key, roof_key, window_key, driveway_key]);
            let wall_id = mat_ids[0];
            let roof_id = mat_ids[1];
            let window_id = mat_ids[2];
            let driveway_id = mat_ids[3];

            let border = &lot.bounds;
            let n = border.len();
            if n < 3 {
                continue;
            }
            let tiles = lot.get_tiles();
            for ((x, z), tile) in tiles.iter() {
                let direction = lot.entrance.1;

                let forward = Vec2::new(direction.x, direction.z).normalize();
                let right = Vec2::new(forward.y, -forward.x);
                let mut tile_origin = lot
                    .entrance
                    .0
                    .add_vec2(right * (*x as f32 + 0.5) + forward * (*z as f32 + 0.5));
                tile_origin.local.y = terrain.get_height_at(tile_origin, true);
                match tile {
                    Tile::Square(tile_type) => {
                        let zero_height = lot.entrance.0.local.y;
                        let corners_local =
                            [(0.0_f32, 0.0_f32), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];

                        let mut points: Vec<WorldPos> = corners_local
                            .iter()
                            .map(|(cx, cz)| {
                                let mut corner = lot.entrance.0.add_vec2(
                                    right * (*x as f32 + cx) + forward * (*z as f32 + cz),
                                );
                                corner.local.y = terrain.get_height_at(corner, true);
                                corner
                            })
                            .collect();

                        // Close the loop
                        points.push(points[0]);

                        //gizmo.polyline(points.as_slice(), [0.0, 0.0, 0.0, 1.0], 0.0, 0.1, 100.0);
                        match tile_type {
                            TileType::Grass => {}
                            TileType::Tree => {}
                            TileType::Garden => {}
                            TileType::House => {
                                let roof_y =
                                    zero_height + level.num_stories as f32 * level.story_height;
                                mesh.push_walled_square(
                                    lot,
                                    zero_height,
                                    roof_y,
                                    right,
                                    forward,
                                    (*x, *z),
                                    wall_id,
                                );
                                mesh.push_square(lot, roof_y, right, forward, (*x, *z), roof_id);
                            }
                            TileType::HouseBalcony => {}
                            TileType::HouseEntrance => {}
                            TileType::LotEntrance => {}
                            TileType::Garage => {}
                            TileType::Driveway => {
                                mesh.push_square(
                                    lot,
                                    zero_height,
                                    right,
                                    forward,
                                    (*x, *z),
                                    driveway_id,
                                );
                            }
                        }
                    }
                    Tile::Polygon(tile_type, points) => {
                        // Draw the polygon outline
                        for i in 0..points.len() {
                            let a = points[i];
                            let b = points[(i + 1) % points.len()];
                            gizmo.line(a, b, [0.2, 1.0, 0.2, 1.0], 0.0, 10.0);
                        }
                    }
                }
            }
        }

        BuildingChunkMesh {
            vertices: mesh.vertices,
            indices: mesh.indices,
            topo_version: compute_building_chunk_topo_version(cid, buildings),
        }
    }
    pub fn update_chunk_mesh(
        &mut self,
        render_manager: &mut RenderManager,
        terrain: &Terrain,
        chunk_id: ChunkId,
        buildings: &Buildings,
        zoning: &mut Zoning,
        gizmo: &mut Gizmo,
    ) -> &BuildingChunkMesh {
        let mesh =
            self.build_mesh_for_chunk(render_manager, terrain, chunk_id, buildings, zoning, gizmo);
        self.chunk_cache.insert(chunk_id, mesh);
        self.chunk_cache.get(&chunk_id).unwrap()
    }
}
struct BuildingMeshBuilder {
    vertices: Vec<BuildingVertex>,
    indices: Vec<u32>,
}
impl BuildingMeshBuilder {
    /// Heights are ABSOLUTE!!
    fn push_square(
        &mut self,
        lot: &Lot,
        height: f32,
        right: Vec2,
        forward: Vec2,
        tile_pos: (i32, i32),
        material_id: u32,
    ) {
        let corners_local = [(0.0_f32, 0.0_f32), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];

        let points: Vec<WorldPos> = corners_local
            .iter()
            .map(|(cx, cz)| {
                let mut corner = lot.entrance.0.add_vec2(
                    right * (tile_pos.0 as f32 + cx) + forward * (tile_pos.1 as f32 + cz),
                );

                corner.local.y = height;
                corner
            })
            .collect();

        let base_index = self.vertices.len() as u32;

        // 0
        self.vertices.push(BuildingVertex {
            chunk_xz: points[0].chunk.as_slice(),
            local_position: points[0].local.as_slice(),
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            material_id,
        });

        // 1
        self.vertices.push(BuildingVertex {
            chunk_xz: points[1].chunk.as_slice(),
            local_position: points[1].local.as_slice(),
            normal: [0.0, 1.0, 0.0],
            uv: [1.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            material_id,
        });

        // 2
        self.vertices.push(BuildingVertex {
            chunk_xz: points[2].chunk.as_slice(),
            local_position: points[2].local.as_slice(),
            normal: [0.0, 1.0, 0.0],
            uv: [1.0, 1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            material_id,
        });

        // 3
        self.vertices.push(BuildingVertex {
            chunk_xz: points[3].chunk.as_slice(),
            local_position: points[3].local.as_slice(),
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            material_id,
        });

        // CCW triangles, back-face culled
        // top-right
        self.indices
            .extend_from_slice(&[base_index + 0, base_index + 3, base_index + 1]);

        // bottom-left
        self.indices
            .extend_from_slice(&[base_index + 3, base_index + 2, base_index + 1]);
    }

    /// Heights are ABSOLUTE!!
    fn push_walled_square(
        &mut self,
        lot: &Lot,
        start_height: f32,
        end_height: f32,
        right: Vec2,
        forward: Vec2,
        tile_pos: (i32, i32),
        material_id: u32,
    ) {
        let (bottom_y, top_y) = if start_height <= end_height {
            (start_height, end_height)
        } else {
            (end_height, start_height)
        };

        let corner_world = |cx: f32, cz: f32, y: f32| -> WorldPos {
            let mut corner = lot
                .entrance
                .0
                .add_vec2(right * (tile_pos.0 as f32 + cx) + forward * (tile_pos.1 as f32 + cz));
            corner.local.y = y;
            corner
        };

        let p00b = corner_world(0.0, 0.0, bottom_y);
        let p10b = corner_world(1.0, 0.0, bottom_y);
        let p11b = corner_world(1.0, 1.0, bottom_y);
        let p01b = corner_world(0.0, 1.0, bottom_y);

        let p00t = corner_world(0.0, 0.0, top_y);
        let p10t = corner_world(1.0, 0.0, top_y);
        let p11t = corner_world(1.0, 1.0, top_y);
        let p01t = corner_world(0.0, 1.0, top_y);

        let n_south = [-forward.x, 0.0, -forward.y];
        let n_east = [right.x, 0.0, right.y];
        let n_north = [forward.x, 0.0, forward.y];
        let n_west = [-right.x, 0.0, -right.y];

        let mut push_quad =
            |a: WorldPos, b: WorldPos, c: WorldPos, d: WorldPos, normal: [f32; 3]| {
                let base = self.vertices.len() as u32;

                let mut push_v = |p: WorldPos, uv: [f32; 2]| {
                    self.vertices.push(BuildingVertex {
                        chunk_xz: p.chunk.as_slice(),
                        local_position: p.local.as_slice(),
                        normal,
                        uv,
                        color: [1.0, 1.0, 1.0, 1.0],
                        material_id,
                    });
                };

                push_v(a, [0.0, 0.0]);
                push_v(b, [0.0, 1.0]);
                push_v(c, [1.0, 1.0]);
                push_v(d, [1.0, 0.0]);

                self.indices.extend_from_slice(&[
                    base + 0,
                    base + 1,
                    base + 2,
                    base + 2,
                    base + 3,
                    base + 0,
                ]);
            };

        // Near / south side
        push_quad(p00b, p00t, p10t, p10b, n_south);

        // Right / east side
        push_quad(p10b, p10t, p11t, p11b, n_east);

        // Far / north side
        push_quad(p01b, p11b, p11t, p01t, n_north);

        // Left / west side
        push_quad(p00b, p01b, p01t, p00t, n_west);
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
