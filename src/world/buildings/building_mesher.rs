use crate::helpers::positions::{ChunkCoord, LocalPos, WorldPos};
use crate::renderer::gizmo::gizmo::Gizmo;
use crate::renderer::props::{PropInstance, PropInstanceId, Props};
use crate::renderer::textures::material_keys::terrain_material_keys;
use crate::world::buildings::buildings::{BuildingId, Buildings, MiscBuildingParams, RoofType};
use crate::world::buildings::zoning::{Lot, LotEntrance, LotId, Tile, TilePos, TileType, Zoning};
use crate::world::cars::parking::ParkingStorage;
use crate::world::terrain::terrain_editing::TerrainEditSource;
use crate::world::terrain::terrain_subsystem::Terrain;
use glam::{Vec2, Vec3};
use revision::revisioned;
use serde::Deserialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::{Deref, DerefMut};
use wgpu::{VertexAttribute, VertexFormat};
use wgpu_render_manager::generator::{MipmapMode, TextureKey, TextureParams};
use wgpu_render_manager::renderer::RenderManager;
#[derive(Debug, Clone, Default)]
#[revisioned(revision = 1)]
pub struct Color(pub [f32; 4]);
impl<'de> Deserialize<'de> for Color {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum ColorData {
            Rgb([f32; 3]),
            Rgba([f32; 4]),
        }

        match ColorData::deserialize(deserializer)? {
            ColorData::Rgb([r, g, b]) => Ok(Self([r, g, b, 1.0])),
            ColorData::Rgba(v) => Ok(Self(v)),
        }
    }
}
impl Color {
    pub fn white() -> Color {
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
impl Hash for Color {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self[0].to_bits().hash(state);
        self[1].to_bits().hash(state);
        self[2].to_bits().hash(state);
        self[3].to_bits().hash(state);
    }
}
pub struct BuildingMeshManager {
    chunk_cache: HashMap<ChunkCoord, BuildingChunkMesh>,
}

impl BuildingMeshManager {
    // pub fn build_mesh_for_preview(&self) -> BuildingChunkMesh {
    //
    // }
    pub fn build_mesh_for_building(
        &self,
        mesh: &mut BuildingMeshBuilder,
        render_manager: &mut RenderManager,
        terrain: &mut Terrain,
        buildings: &mut Buildings,
        flatten_terrain: bool,
        building_id: BuildingId,
        lot: &mut Lot,
        props: &mut Props,
        parking_storage: &mut ParkingStorage,
        gizmo: &mut Gizmo,
    ) {
        let border = lot.bounds.as_slice();
        if border.len() < 3 {
            return;
        }

        let Some(building) = buildings.storage.get_mut(building_id) else {
            return;
        };
        props.remove_instances(building.prop_instance_ids.as_slice());
        building.prop_instance_ids.clear();
        if let Some(edit_id) = building.edit_id {
            terrain.terrain_editor.remove_edit(edit_id);
        }
        if flatten_terrain {
            building.edit_id = Some(terrain.terrain_editor.push_flat_polygon(
                lot.bounds.clone(),
                -0.01,
                3.0,
                TerrainEditSource::Building(building_id),
            ));
        }

        let Some(level) = building.current_level_params(&buildings.catalog) else {
            return;
        };

        let mut prop_instance_ids = vec![];
        let entrance = &lot.entrance;

        let wall_key = level.wall_material.texture_key();
        let roof_side_key = TextureKey::new(
            "wood_slab",
            TextureParams::default()
                .with_primary_color([0.62, 0.42, 0.22, 1.0])
                .with_secondary_color([0.30, 0.18, 0.10, 1.0])
                .with_scale(0.5)
                .with_roughness(0.4),
            512,
            MipmapMode::Generate,
        );
        let roof_key = level.roof_material.texture_key();
        let window_key = level.miscellaneous.window_material_accent.texture_key();
        let driveway_key = level.driveway_material.texture_key();

        let mut grass_key = terrain_material_keys().remove(0);
        grass_key.resolution = 512;
        grass_key.params.scale = 20.0;

        let mut garden_key = grass_key.clone();
        garden_key.params.color_primary[1] *= 1.2;

        let mut notex_key = TextureKey::notex();
        notex_key.resolution = 512;

        let mat_ids = render_manager.ensure_textures(&[
            wall_key,
            roof_key,
            window_key,
            driveway_key,
            grass_key,
            garden_key,
            notex_key,
            roof_side_key,
        ]);

        let wall_id = mat_ids[0];
        let roof_id = mat_ids[1];
        let window_id = mat_ids[2];
        let driveway_id = mat_ids[3];
        let grass_id = mat_ids[4];
        let garden_id = mat_ids[5];
        let notex_id = mat_ids[6];
        let roof_side_id = mat_ids[7];

        let direction = entrance.dir;
        let forward = Vec2::new(direction.x, direction.z).normalize();
        let right = Vec2::new(forward.y, -forward.x);
        let zero_height = entrance.pos.local.y;

        // if lot.layout.is_none() {
        let lot_layout = lot.generate_layout(parking_storage);
        lot.layout = Some(lot_layout);
        //}

        let layout = lot.layout.as_mut().unwrap();
        let tiles = &layout.tiles;

        let mut roof_tiles: Vec<RoofTile> = Vec::new();
        let mut house_tiles: HashSet<TilePos> = HashSet::new();
        let mut garage_tiles: HashSet<TilePos> = HashSet::new();
        let mut grass_tiles: HashSet<TilePos> = HashSet::new();
        let mut garden_tiles: HashSet<TilePos> = HashSet::new();
        let mut driveway_tiles: HashSet<TilePos> = HashSet::new();
        let mut notex_tiles: HashSet<TilePos> = HashSet::new();

        for (&pos, tile) in tiles.iter() {
            match tile {
                Tile::Square(tile_type) => match tile_type {
                    TileType::Grass => {
                        grass_tiles.insert(pos);
                    }
                    TileType::Tree => {
                        garden_tiles.insert(pos);
                        let center_gx = pos.x as f32 + 0.5;
                        let center_gz = pos.z as f32 + 0.5;
                        let mut center = entrance
                            .pos
                            .add_vec2(right * center_gx + forward * center_gz);
                        center.local.y = zero_height;
                        let prop_instance_id = props.place_prop(
                            "oak",
                            PropInstance {
                                id: None,
                                archetype_id: None,
                                pos: center,
                                scale: rand::random_range(0.8..1.5),
                                rotation_y_rad: rand::random_range(0.0..5.0),
                                seed: rand::random(),
                                color: [1.0, 1.0, 1.0, 1.0],
                                wind_strength: 0.2,
                                variant: 0,
                                generated: false,
                            },
                        );
                        prop_instance_ids.push(prop_instance_id);
                    }
                    TileType::Garden => {
                        garden_tiles.insert(pos);
                    }
                    TileType::House => {
                        house_tiles.insert(pos);
                        let roof_y = zero_height + level.num_stories as f32 * level.story_height;
                        roof_tiles.push(RoofTile {
                            tile_pos: pos,
                            base_y: roof_y,
                        });
                    }
                    TileType::HouseBalcony | TileType::HouseEntrance | TileType::LotEntrance => {
                        notex_tiles.insert(pos);
                    }
                    TileType::Garage => {
                        if let Some(garage) = &level.garage {
                            garage_tiles.insert(pos);
                            let roof_y =
                                zero_height + garage.story_height * garage.num_stories as f32;
                            roof_tiles.push(RoofTile {
                                tile_pos: pos,
                                base_y: roof_y,
                            });
                        }
                    }
                    TileType::Driveway => {
                        driveway_tiles.insert(pos);
                    }
                },
                Tile::Polygon(_, points) => {
                    for i in 0..points.len() {
                        let a = points[i];
                        let b = points[(i + 1) % points.len()];
                        gizmo.line(a, b, [0.2, 1.0, 0.2, 1.0], 0.0, 10.0);
                    }
                }
            }
        }

        mesh.push_merged_ground(
            entrance,
            right,
            forward,
            &greedy_merge_tiles(&grass_tiles),
            zero_height,
            grass_id,
            GROUND_UV_SCALE,
        );
        mesh.push_merged_ground(
            entrance,
            right,
            forward,
            &greedy_merge_tiles(&garden_tiles),
            zero_height,
            garden_id,
            GROUND_UV_SCALE,
        );
        mesh.push_merged_ground(
            entrance,
            right,
            forward,
            &greedy_merge_tiles(&driveway_tiles),
            zero_height,
            driveway_id,
            GROUND_UV_SCALE,
        );
        mesh.push_merged_ground(
            entrance,
            right,
            forward,
            &greedy_merge_tiles(&notex_tiles),
            zero_height,
            notex_id,
            GROUND_UV_SCALE,
        );

        let house_top = zero_height + level.num_stories as f32 * level.story_height;
        mesh.emit_walls(
            entrance,
            right,
            forward,
            tiles,
            &house_tiles,
            zero_height,
            house_top,
            wall_id,
            WALL_UV_SCALE,
        );

        if let Some(garage) = &level.garage {
            let garage_top = zero_height + garage.story_height * garage.num_stories as f32;
            mesh.emit_walls(
                entrance,
                right,
                forward,
                tiles,
                &garage_tiles,
                zero_height,
                garage_top,
                wall_id,
                WALL_UV_SCALE,
            );
        }

        let components = group_roof_components(&level.roof, &roof_tiles);
        mesh.emit_roof(entrance, right, forward, &components, roof_id, roof_side_id);

        for accessory in active_roof_accessories(&level.miscellaneous) {
            for component in &components {
                let ctx = RoofAccessoryContext {
                    entrance,
                    right,
                    forward,
                    component,
                };
                prop_instance_ids.extend(accessory.place(&ctx, props));
            }
        }

        building.prop_instance_ids = prop_instance_ids;
    }
}

impl BuildingMeshManager {
    pub fn new() -> Self {
        Self {
            chunk_cache: HashMap::new(),
        }
    }

    pub fn get_chunk_mesh(&self, chunk_coord: ChunkCoord) -> Option<&BuildingChunkMesh> {
        self.chunk_cache.get(&chunk_coord)
    }

    pub fn invalidate_chunk(&mut self, chunk_coord: ChunkCoord) {
        self.chunk_cache.remove(&chunk_coord);
    }

    pub fn _clear_cache(&mut self) {
        self.chunk_cache.clear();
    }

    pub fn chunk_needs_update(&self, chunk_coord: ChunkCoord, buildings: &Buildings) -> bool {
        match self.chunk_cache.get(&chunk_coord) {
            None => true,
            Some(mesh) => {
                mesh.topo_version != compute_building_chunk_topo_version(chunk_coord, buildings)
            }
        }
    }

    /// Build mesh for a chunk
    pub fn build_mesh_for_chunk(
        &mut self,
        render_manager: &mut RenderManager,
        terrain: &mut Terrain,
        props: &mut Props,
        chunk_coord: ChunkCoord,
        buildings: &mut Buildings,
        zoning: &mut Zoning,
        parking_storage: &mut ParkingStorage,
        gizmo: &mut Gizmo,
    ) -> BuildingChunkMesh {
        let mut mesh = BuildingMeshBuilder {
            vertices: Vec::new(),
            indices: Vec::new(),
        };

        let lot_ids: Vec<LotId> = zoning.zoning_storage.lots_in_chunk(chunk_coord);

        for id in lot_ids.iter() {
            let Some(lot) = zoning.zoning_storage.get_mut_lot(*id) else {
                continue;
            };
            let Some(building) = buildings.storage.get_mut(lot.building_id) else {
                continue;
            };
            if building.misc.is_preview {
                continue;
            };
            if let Some(bmesh) = building.misc.mesh.clone() {
                mesh.extend_foreign(bmesh);
                building.edit_id = Some(terrain.terrain_editor.push_flat_polygon(
                    lot.bounds.clone(),
                    -0.01,
                    3.0,
                    TerrainEditSource::Building(building.id),
                ));
                continue;
            };
            if let Some(edit_id) = building.edit_id {
                terrain.terrain_editor.remove_edit(edit_id);
            }
            props.remove_instances(building.prop_instance_ids.as_slice());
            let bounds: Vec<_> = lot
                .bounds
                .iter()
                .cloned()
                .map(|mut p| {
                    p.local.y = lot.entrance.pos.local.y;
                    p
                })
                .collect();
            lot.bounds = bounds;
            let building_id = building.id;
            self.build_mesh_for_building(
                &mut mesh,
                render_manager,
                terrain,
                buildings,
                true,
                building_id,
                lot,
                props,
                parking_storage,
                gizmo,
            );
        }

        BuildingChunkMesh {
            vertices: mesh.vertices,
            indices: mesh.indices,
            topo_version: compute_building_chunk_topo_version(chunk_coord, buildings),
        }
    }
    pub fn update_chunk_mesh(
        &mut self,
        render_manager: &mut RenderManager,
        terrain: &mut Terrain,
        props: &mut Props,
        chunk_coord: ChunkCoord,
        buildings: &mut Buildings,
        zoning: &mut Zoning,
        parking_storage: &mut ParkingStorage,
        gizmo: &mut Gizmo,
    ) -> &BuildingChunkMesh {
        let mesh = self.build_mesh_for_chunk(
            render_manager,
            terrain,
            props,
            chunk_coord,
            buildings,
            zoning,
            parking_storage,
            gizmo,
        );
        self.chunk_cache.insert(chunk_coord, mesh);
        self.chunk_cache.get(&chunk_coord).unwrap()
    }
}
#[revisioned(revision = 1)]
#[derive(Debug, Clone, Default)]
pub struct BuildingMeshBuilder {
    pub vertices: Vec<BuildingVertex>,
    pub indices: Vec<u32>,
}
impl BuildingMeshBuilder {
    pub fn move_rotate(
        &mut self,
        old_center_pos: WorldPos,
        old_dir: Vec3,
        new_center_pos: WorldPos,
        new_dir: Vec3,
    ) {
        let old_xz = Vec2::new(old_dir.x, old_dir.z);
        let new_xz = Vec2::new(new_dir.x, new_dir.z);

        let delta_angle = if old_xz.length_squared() < 1e-10 || new_xz.length_squared() < 1e-10 {
            0.0f32
        } else {
            let old_angle = old_xz.y.atan2(old_xz.x);
            let new_angle = new_xz.y.atan2(new_xz.x);
            new_angle - old_angle
        };

        let (sin_a, cos_a) = delta_angle.sin_cos();

        for vert in &mut self.vertices {
            let vert_pos = WorldPos::new(
                ChunkCoord::new(vert.chunk_xz[0], vert.chunk_xz[1]),
                LocalPos::new(
                    vert.local_position[0],
                    vert.local_position[1],
                    vert.local_position[2],
                ),
            );

            let offset = old_center_pos.delta_to(vert_pos);

            let rotated_offset = Vec3::new(
                offset.x * cos_a - offset.z * sin_a,
                offset.y,
                offset.x * sin_a + offset.z * cos_a,
            );

            let new_pos = new_center_pos.add_vec3(rotated_offset);

            vert.chunk_xz = [new_pos.chunk.x, new_pos.chunk.z];
            vert.local_position = [new_pos.local.x, new_pos.local.y, new_pos.local.z];

            let n = vert.normal;
            vert.normal = [
                n[0] * cos_a - n[2] * sin_a,
                n[1],
                n[0] * sin_a + n[2] * cos_a,
            ];
        }
    }

    pub fn extend_foreign(&mut self, other: BuildingMeshBuilder) {
        let vertex_offset = self.vertices.len() as u32;

        self.vertices.extend(other.vertices);
        self.indices
            .extend(other.indices.into_iter().map(|index| index + vertex_offset));
    }
    /// Heights are ABSOLUTE!!

    fn grid_point_world(
        entrance: &LotEntrance,
        right: Vec2,
        forward: Vec2,
        gx: f32,
        gz: f32,
        y: f32,
    ) -> WorldPos {
        let mut p = entrance.pos.add_vec2(right * gx + forward * gz);
        p.local.y = y;
        p
    }

    fn grid_point_local(right: Vec2, forward: Vec2, gx: f32, gz: f32, y: f32) -> Vec3 {
        Vec3::new(
            right.x * gx + forward.x * gz,
            y,
            right.y * gx + forward.y * gz,
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct RoofTile {
    tile_pos: TilePos,
    base_y: f32,
}

#[revisioned(revision = 1)]
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

#[derive(Clone, Debug)]
pub struct BuildingMeshIndex {
    pub building_id: BuildingId,
    pub indices_start: u32,
    pub indices_count: u32,
}
fn compute_building_chunk_topo_version(chunk_coord: ChunkCoord, buildings: &Buildings) -> u64 {
    if let Some(building_chunk) = buildings.storage.building_chunk_storage.get(&chunk_coord) {
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

pub trait TexturedMaterial {
    fn texture_key(&self) -> TextureKey;
}

impl TexturedMaterial for WallMaterial {
    fn texture_key(&self) -> TextureKey {
        match self {
            WallMaterial::Paint(color) => TextureKey::new(
                "paint",
                TextureParams::default()
                    .with_primary_color(**color)
                    .with_secondary_color([0.0, 0.0, 0.0, 1.0]),
                512,
                MipmapMode::Generate,
            ),
            WallMaterial::Stucco(color) => TextureKey::new(
                "stucco",
                TextureParams::default()
                    .with_primary_color(**color)
                    .with_roughness(0.8)
                    .with_scale(2.0),
                512,
                MipmapMode::Generate,
            ),
            WallMaterial::WoodSiding(color) => TextureKey::new(
                "wood_siding",
                TextureParams::default()
                    .with_primary_color(**color)
                    .with_secondary_color([0.25, 0.16, 0.08, 1.0])
                    .with_scale(1.5),
                512,
                MipmapMode::Generate,
            ),
            WallMaterial::Glass(color) => TextureKey::new(
                "glass",
                TextureParams::default()
                    .with_primary_color(**color)
                    .with_roughness(0.05),
                512,
                MipmapMode::Generate,
            ),
            WallMaterial::Custom(key) => key.clone(),
        }
    }
}

impl TexturedMaterial for RoofMaterial {
    fn texture_key(&self) -> TextureKey {
        match self {
            RoofMaterial::Shingles => TextureKey::new(
                "shingles",
                TextureParams::default().with_primary_color([0.4, 0.15, 0.05, 1.0]),
                512,
                MipmapMode::Generate,
            ),
            RoofMaterial::Metal => TextureKey::new(
                "metal_roof",
                TextureParams::default().with_primary_color([0.2, 0.2, 0.2, 1.0]),
                512,
                MipmapMode::Generate,
            ),
            RoofMaterial::Tile => TextureKey::new(
                "roof_tile",
                TextureParams::default()
                    .with_primary_color([0.55, 0.2, 0.12, 1.0])
                    .with_scale(1.2),
                512,
                MipmapMode::Generate,
            ),
            RoofMaterial::Slate => TextureKey::new(
                "slate",
                TextureParams::default()
                    .with_primary_color([0.22, 0.24, 0.27, 1.0])
                    .with_roughness(0.3),
                512,
                MipmapMode::Generate,
            ),
            RoofMaterial::Custom(key) => key.clone(),
        }
    }
}

impl TexturedMaterial for DrivewayMaterial {
    fn texture_key(&self) -> TextureKey {
        match self {
            DrivewayMaterial::Bricks => TextureKey::new(
                "driveway_bricks",
                TextureParams::default()
                    .with_primary_color([0.18, 0.18, 0.22, 1.0])
                    .with_secondary_color([0.01, 0.01, 0.01, 1.0])
                    .with_roughness(0.0)
                    .with_scale(5.0),
                512,
                MipmapMode::Generate,
            ),
            DrivewayMaterial::Concrete => TextureKey::new(
                "concrete",
                TextureParams::default()
                    .with_primary_color([0.6, 0.6, 0.58, 1.0])
                    .with_roughness(0.6)
                    .with_scale(3.0),
                512,
                MipmapMode::Generate,
            ),
            DrivewayMaterial::Gravel => TextureKey::new(
                "gravel",
                TextureParams::default()
                    .with_primary_color([0.45, 0.42, 0.38, 1.0])
                    .with_roughness(0.9)
                    .with_scale(8.0),
                512,
                MipmapMode::Generate,
            ),
            DrivewayMaterial::Custom(key) => key.clone(),
        }
    }
}

#[derive(Debug, Clone, Hash, Deserialize)]
#[revisioned(revision = 1)]
pub enum WallMaterial {
    Paint(Color),
    Stucco(Color),
    WoodSiding(Color),
    Glass(Color),
    Custom(TextureKey),
}
impl Default for WallMaterial {
    fn default() -> Self {
        WallMaterial::Paint(Color::white())
    }
}

#[derive(Debug, Clone, Default, Hash, Deserialize)]
#[revisioned(revision = 1)]
pub enum RoofMaterial {
    #[default]
    Shingles,
    Metal,
    Tile,
    Slate,
    Custom(TextureKey),
}

#[derive(Debug, Clone, Hash, Deserialize, Default)]
#[revisioned(revision = 1)]
pub enum DrivewayMaterial {
    #[default]
    Bricks,
    Concrete,
    Gravel,
    Custom(TextureKey),
}

const GROUND_UV_SCALE: f32 = 1.0;
const WALL_UV_SCALE: f32 = 1.0;
const ROOF_UV_SCALE: f32 = 1.0;

fn planar_uv(normal: Vec3, gx: f32, gz: f32, gy: f32, scale: f32) -> [f32; 2] {
    let ax = normal.x.abs();
    let ay = normal.y.abs();
    let az = normal.z.abs();

    if ay >= ax && ay >= az {
        [gx * scale, gz * scale]
    } else if ax >= az {
        [gz * scale, gy * scale]
    } else {
        [gx * scale, gy * scale]
    }
}

#[derive(Clone, Copy, Debug)]
struct TileRect {
    min_x: i16,
    min_z: i16,
    max_x: i16,
    max_z: i16,
}

fn greedy_merge_tiles(cells: &HashSet<TilePos>) -> Vec<TileRect> {
    let mut ordered: Vec<TilePos> = cells.iter().copied().collect();
    ordered.sort_by_key(|p| (p.z, p.x));

    let mut visited: HashSet<TilePos> = HashSet::new();
    let mut rects = Vec::new();

    for start in ordered {
        if visited.contains(&start) {
            continue;
        }

        let mut width = 1;
        while cells.contains(&TilePos::new(start.x + width, start.z))
            && !visited.contains(&TilePos::new(start.x + width, start.z))
        {
            width += 1;
        }

        let mut depth = 1;
        loop {
            let row_clear = (0..width).all(|dx| {
                let p = TilePos::new(start.x + dx, start.z + depth);
                cells.contains(&p) && !visited.contains(&p)
            });
            if !row_clear {
                break;
            }
            depth += 1;
        }

        for dz in 0..depth {
            for dx in 0..width {
                visited.insert(TilePos::new(start.x + dx, start.z + dz));
            }
        }

        rects.push(TileRect {
            min_x: start.x,
            min_z: start.z,
            max_x: start.x + width,
            max_z: start.z + depth,
        });
    }

    rects
}
impl BuildingMeshBuilder {
    fn push_merged_ground(
        &mut self,
        entrance: &LotEntrance,
        right: Vec2,
        forward: Vec2,
        rects: &[TileRect],
        height: f32,
        material_id: u32,
        uv_scale: f32,
    ) {
        for rect in rects {
            let x0 = rect.min_x as f32;
            let x1 = rect.max_x as f32;
            let z0 = rect.min_z as f32;
            let z1 = rect.max_z as f32;

            let corners = [(x0, z0), (x1, z0), (x1, z1), (x0, z1)];
            let uvs = [
                [x0 * uv_scale, z0 * uv_scale],
                [x1 * uv_scale, z0 * uv_scale],
                [x1 * uv_scale, z1 * uv_scale],
                [x0 * uv_scale, z1 * uv_scale],
            ];

            let base = self.vertices.len() as u32;
            for ((cx, cz), uv) in corners.into_iter().zip(uvs.into_iter()) {
                let mut wp = entrance.pos.add_vec2(right * cx + forward * cz);
                wp.local.y = height;
                self.vertices.push(BuildingVertex {
                    chunk_xz: wp.chunk.as_slice(),
                    local_position: wp.local.as_slice(),
                    normal: [0.0, 1.0, 0.0],
                    uv,
                    color: [1.0, 1.0, 1.0, 1.0],
                    material_id,
                });
            }

            self.indices.extend_from_slice(&[
                base,
                base + 3,
                base + 1,
                base + 3,
                base + 2,
                base + 1,
            ]);
        }
    }
}
enum WallSide {
    South,
    East,
    North,
    West,
}

fn is_blocking_tile(tiles: &HashMap<TilePos, Tile>, pos: TilePos) -> bool {
    matches!(
        tiles.get(&pos),
        Some(Tile::Square(TileType::House)) | Some(Tile::Square(TileType::Garage))
    )
}

fn merge_open_runs(is_open: impl Fn(i16) -> bool, from: i16, to_exclusive: i16) -> Vec<(i16, i16)> {
    let mut runs = Vec::new();
    let mut i = from;
    while i < to_exclusive {
        if is_open(i) {
            let start = i;
            while i < to_exclusive && is_open(i) {
                i += 1;
            }
            runs.push((start, i));
        } else {
            i += 1;
        }
    }
    runs
}

fn merge_wall_faces(
    tiles: &HashMap<TilePos, Tile>,
    member_of: &HashSet<TilePos>,
) -> Vec<(WallSide, i16, i16, i16)> {
    let mut faces = Vec::new();

    let mut by_z: HashMap<i16, Vec<i16>> = HashMap::new();
    let mut by_x: HashMap<i16, Vec<i16>> = HashMap::new();
    for pos in member_of {
        by_z.entry(pos.z).or_default().push(pos.x);
        by_x.entry(pos.x).or_default().push(pos.z);
    }

    for (&z, xs) in by_z.iter() {
        let mut xs = xs.clone();
        xs.sort_unstable();
        let lo = xs[0];
        let hi = xs[xs.len() - 1] + 1;

        for (start, end) in merge_open_runs(
            |x| {
                member_of.contains(&TilePos::new(x, z))
                    && !is_blocking_tile(tiles, TilePos::new(x, z - 1))
            },
            lo,
            hi,
        ) {
            faces.push((WallSide::South, z, start, end));
        }

        for (start, end) in merge_open_runs(
            |x| {
                member_of.contains(&TilePos::new(x, z))
                    && !is_blocking_tile(tiles, TilePos::new(x, z + 1))
            },
            lo,
            hi,
        ) {
            faces.push((WallSide::North, z, start, end));
        }
    }

    for (&x, zs) in by_x.iter() {
        let mut zs = zs.clone();
        zs.sort_unstable();
        let lo = zs[0];
        let hi = zs[zs.len() - 1] + 1;

        for (start, end) in merge_open_runs(
            |z| {
                member_of.contains(&TilePos::new(x, z))
                    && !is_blocking_tile(tiles, TilePos::new(x - 1, z))
            },
            lo,
            hi,
        ) {
            faces.push((WallSide::West, x, start, end));
        }

        for (start, end) in merge_open_runs(
            |z| {
                member_of.contains(&TilePos::new(x, z))
                    && !is_blocking_tile(tiles, TilePos::new(x + 1, z))
            },
            lo,
            hi,
        ) {
            faces.push((WallSide::East, x, start, end));
        }
    }

    faces
}

impl BuildingMeshBuilder {
    fn push_wall_face(
        &mut self,
        entrance: &LotEntrance,
        right: Vec2,
        forward: Vec2,
        side: WallSide,
        line: i16,
        start: i16,
        end: i16,
        bottom_y: f32,
        top_y: f32,
        material_id: u32,
        uv_scale: f32,
    ) {
        let start_f = start as f32;
        let end_f = end as f32;
        let line_f = line as f32;

        let (a, b, c, d, normal) = match side {
            WallSide::South => (
                (start_f, line_f, bottom_y),
                (start_f, line_f, top_y),
                (end_f, line_f, top_y),
                (end_f, line_f, bottom_y),
                [-forward.x, 0.0, -forward.y],
            ),
            WallSide::East => (
                (line_f + 1.0, start_f, bottom_y),
                (line_f + 1.0, start_f, top_y),
                (line_f + 1.0, end_f, top_y),
                (line_f + 1.0, end_f, bottom_y),
                [right.x, 0.0, right.y],
            ),
            WallSide::North => (
                (start_f, line_f + 1.0, bottom_y),
                (end_f, line_f + 1.0, bottom_y),
                (end_f, line_f + 1.0, top_y),
                (start_f, line_f + 1.0, top_y),
                [forward.x, 0.0, forward.y],
            ),
            WallSide::West => (
                (line_f, start_f, bottom_y),
                (line_f, end_f, bottom_y),
                (line_f, end_f, top_y),
                (line_f, start_f, top_y),
                [-right.x, 0.0, -right.y],
            ),
        };

        let run_length = (end_f - start_f).abs();
        let height = (top_y - bottom_y).abs();

        let base = self.vertices.len() as u32;
        let corners = [a, b, c, d];
        let uvs = match side {
            WallSide::South | WallSide::East => [
                [0.0, 0.0],
                [0.0, height * uv_scale],
                [run_length * uv_scale, height * uv_scale],
                [run_length * uv_scale, 0.0],
            ],
            WallSide::North | WallSide::West => [
                [0.0, 0.0],
                [run_length * uv_scale, 0.0],
                [run_length * uv_scale, height * uv_scale],
                [0.0, height * uv_scale],
            ],
        };

        for (corner, uv) in corners.into_iter().zip(uvs.into_iter()) {
            let mut wp = entrance.pos.add_vec2(right * corner.0 + forward * corner.1);
            wp.local.y = corner.2;
            self.vertices.push(BuildingVertex {
                chunk_xz: wp.chunk.as_slice(),
                local_position: wp.local.as_slice(),
                normal,
                uv,
                color: [1.0, 1.0, 1.0, 1.0],
                material_id,
            });
        }

        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    }

    fn emit_walls(
        &mut self,
        entrance: &LotEntrance,
        right: Vec2,
        forward: Vec2,
        tiles: &HashMap<TilePos, Tile>,
        member_of: &HashSet<TilePos>,
        bottom_y: f32,
        top_y: f32,
        material_id: u32,
        uv_scale: f32,
    ) {
        for (side, line, start, end) in merge_wall_faces(tiles, member_of) {
            self.push_wall_face(
                entrance,
                right,
                forward,
                side,
                line,
                start,
                end,
                bottom_y,
                top_y,
                material_id,
                uv_scale,
            );
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub enum RidgeAxis {
    AlongX,
    AlongZ,
}

#[derive(Clone, Copy, Debug)]
pub enum RoofSurfaceSampler {
    Flat {
        base_y: f32,
    },
    Angled {
        base_y: f32,
        rise_per_unit: f32,
        dir: Vec2,
        min_proj: f32,
    },
    Ridge {
        base_y: f32,
        rise_per_unit: f32,
        axis: RidgeAxis,
        ridge_pos: f32,
        span_min: f32,
        span_max: f32,
        peak_y: f32,
    },
}

impl RoofSurfaceSampler {
    fn build(roof: &RoofType, base_y: f32, min_x: f32, max_x: f32, min_z: f32, max_z: f32) -> Self {
        match roof {
            RoofType::Flat => RoofSurfaceSampler::Flat { base_y },
            RoofType::Angled {
                pitch,
                direction_rad,
            } => {
                let rise_per_unit = pitch.tan();
                let dir = Vec2::new(direction_rad.cos(), direction_rad.sin());
                let corners = [
                    (min_x, min_z),
                    (max_x, min_z),
                    (max_x, max_z),
                    (min_x, max_z),
                ];
                let min_proj = corners
                    .iter()
                    .map(|&(x, z)| Vec2::new(x, z).dot(dir))
                    .fold(f32::INFINITY, f32::min);
                RoofSurfaceSampler::Angled {
                    base_y,
                    rise_per_unit,
                    dir,
                    min_proj,
                }
            }
            RoofType::Triangle(pitch_deg) => {
                let rise_per_unit = pitch_deg.to_radians().tan();
                let width = max_x - min_x;
                let depth = max_z - min_z;
                if width <= depth {
                    let ridge_pos = (min_x + max_x) * 0.5;
                    let peak_y = base_y + (ridge_pos - min_x) * rise_per_unit;
                    RoofSurfaceSampler::Ridge {
                        base_y,
                        rise_per_unit,
                        axis: RidgeAxis::AlongZ,
                        ridge_pos,
                        span_min: min_x,
                        span_max: max_x,
                        peak_y,
                    }
                } else {
                    let ridge_pos = (min_z + max_z) * 0.5;
                    let peak_y = base_y + (ridge_pos - min_z) * rise_per_unit;
                    RoofSurfaceSampler::Ridge {
                        base_y,
                        rise_per_unit,
                        axis: RidgeAxis::AlongX,
                        ridge_pos,
                        span_min: min_z,
                        span_max: max_z,
                        peak_y,
                    }
                }
            }
        }
    }

    fn height_at(&self, gx: f32, gz: f32) -> f32 {
        match self {
            RoofSurfaceSampler::Flat { base_y } => *base_y,
            RoofSurfaceSampler::Angled {
                base_y,
                rise_per_unit,
                dir,
                min_proj,
            } => {
                let proj = Vec2::new(gx, gz).dot(*dir);
                base_y + (proj - min_proj) * rise_per_unit
            }
            RoofSurfaceSampler::Ridge {
                base_y,
                rise_per_unit,
                axis,
                ridge_pos,
                span_min,
                span_max,
                ..
            } => {
                let coord = match axis {
                    RidgeAxis::AlongZ => gx,
                    RidgeAxis::AlongX => gz,
                };
                if coord <= *ridge_pos {
                    base_y + (coord - span_min) * rise_per_unit
                } else {
                    base_y + (span_max - coord) * rise_per_unit
                }
            }
        }
    }

    fn normal_at(&self, gx: f32, gz: f32) -> Vec3 {
        match self {
            RoofSurfaceSampler::Flat { .. } => Vec3::Y,
            RoofSurfaceSampler::Angled {
                rise_per_unit, dir, ..
            } => Vec3::new(-dir.x * rise_per_unit, 1.0, -dir.y * rise_per_unit).normalize(),
            RoofSurfaceSampler::Ridge {
                rise_per_unit,
                axis,
                ridge_pos,
                ..
            } => {
                let coord = match axis {
                    RidgeAxis::AlongZ => gx,
                    RidgeAxis::AlongX => gz,
                };
                let slope_sign = if coord <= *ridge_pos { 1.0 } else { -1.0 };
                match axis {
                    RidgeAxis::AlongZ => {
                        Vec3::new(-slope_sign * rise_per_unit, 1.0, 0.0).normalize()
                    }
                    RidgeAxis::AlongX => {
                        Vec3::new(0.0, 1.0, -slope_sign * rise_per_unit).normalize()
                    }
                }
            }
        }
    }

    fn ridge_split(&self, x0: f32, x1: f32, z0: f32, z1: f32) -> Option<f32> {
        if let RoofSurfaceSampler::Ridge {
            axis, ridge_pos, ..
        } = self
        {
            let eps = 1e-4;
            let (c0, c1) = match axis {
                RidgeAxis::AlongZ => (x0, x1),
                RidgeAxis::AlongX => (z0, z1),
            };
            if *ridge_pos > c0 + eps && *ridge_pos < c1 - eps {
                return Some(*ridge_pos);
            }
        }
        None
    }
}

pub struct RoofComponent {
    pub tiles: HashSet<TilePos>,
    pub min_x: f32,
    pub max_x: f32,
    pub min_z: f32,
    pub max_z: f32,
    pub sampler: RoofSurfaceSampler,
}

fn group_roof_components(roof: &RoofType, roof_tiles: &[RoofTile]) -> Vec<RoofComponent> {
    let tile_heights: HashMap<TilePos, f32> =
        roof_tiles.iter().map(|t| (t.tile_pos, t.base_y)).collect();
    let mut visited: HashSet<TilePos> = HashSet::new();
    let mut components = Vec::new();

    for (&start, &base_y) in tile_heights.iter() {
        if visited.contains(&start) {
            continue;
        }

        let wanted_bits = base_y.to_bits();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);
        let mut member_tiles: HashSet<TilePos> = HashSet::new();

        while let Some(pos) = queue.pop_front() {
            member_tiles.insert(pos);
            for neighbor in pos.get_neighbors_plus() {
                if visited.contains(&neighbor) {
                    continue;
                }
                if let Some(&neighbor_y) = tile_heights.get(&neighbor) {
                    if neighbor_y.to_bits() == wanted_bits {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        let mut min_x = f32::INFINITY;
        let mut min_z = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_z = f32::NEG_INFINITY;
        for &TilePos { x, z } in &member_tiles {
            min_x = min_x.min(x as f32);
            min_z = min_z.min(z as f32);
            max_x = max_x.max(x as f32 + 1.0);
            max_z = max_z.max(z as f32 + 1.0);
        }

        let sampler = RoofSurfaceSampler::build(roof, base_y, min_x, max_x, min_z, max_z);
        components.push(RoofComponent {
            tiles: member_tiles,
            min_x,
            max_x,
            min_z,
            max_z,
            sampler,
        });
    }

    components
}

#[derive(Clone, Copy)]
struct TileOpenness {
    south: bool,
    east: bool,
    north: bool,
    west: bool,
}

impl TileOpenness {
    fn any_open(&self) -> bool {
        self.south || self.east || self.north || self.west
    }
}

fn tile_openness(component: &HashSet<TilePos>, x: i16, z: i16) -> TileOpenness {
    TileOpenness {
        south: !component.contains(&TilePos::new(x, z - 1)),
        east: !component.contains(&TilePos::new(x + 1, z)),
        north: !component.contains(&TilePos::new(x, z + 1)),
        west: !component.contains(&TilePos::new(x - 1, z)),
    }
}

fn overhung_corners(
    x0: f32,
    x1: f32,
    z0: f32,
    z1: f32,
    openness: TileOpenness,
    overhang: f32,
) -> [(f32, f32); 4] {
    let west = if openness.west { overhang } else { 0.0 };
    let east = if openness.east { overhang } else { 0.0 };
    let south = if openness.south { overhang } else { 0.0 };
    let north = if openness.north { overhang } else { 0.0 };
    [
        (x0 - west, z0 - south),
        (x1 + east, z0 - south),
        (x1 + east, z1 + north),
        (x0 - west, z1 + north),
    ]
}
impl BuildingMeshBuilder {
    fn push_triangle_points(
        &mut self,
        entrance: &LotEntrance,
        right: Vec2,
        forward: Vec2,
        a: (f32, f32, f32),
        b: (f32, f32, f32),
        c: (f32, f32, f32),
        desired_normal: Option<Vec3>,
        material_id: u32,
        uv_scale: f32,
    ) {
        let mut pts = [a, b, c];

        let calc_normal = |pts: &[(f32, f32, f32); 3]| -> Vec3 {
            let pa = Self::grid_point_local(right, forward, pts[0].0, pts[0].1, pts[0].2);
            let pb = Self::grid_point_local(right, forward, pts[1].0, pts[1].1, pts[1].2);
            let pc = Self::grid_point_local(right, forward, pts[2].0, pts[2].1, pts[2].2);
            (pb - pa).cross(pc - pa)
        };

        let mut normal = calc_normal(&pts);
        if normal.length_squared() < 1e-6 {
            return;
        }

        if let Some(wanted) = desired_normal {
            let wanted = if wanted.length_squared() > 1e-6 {
                wanted.normalize()
            } else {
                Vec3::Y
            };
            if normal.dot(wanted) < 0.0 {
                pts.swap(1, 2);
                normal = calc_normal(&pts);
                if normal.length_squared() < 1e-6 {
                    return;
                }
            }
        }

        let normal = normal.normalize();
        let base_index = self.vertices.len() as u32;

        for pt in pts {
            let wp = Self::grid_point_world(entrance, right, forward, pt.0, pt.1, pt.2);
            let uv = planar_uv(normal, pt.0, pt.1, pt.2, uv_scale);
            self.vertices.push(BuildingVertex {
                chunk_xz: wp.chunk.as_slice(),
                local_position: wp.local.as_slice(),
                normal: [normal.x, normal.y, normal.z],
                uv,
                color: [1.0, 1.0, 1.0, 1.0],
                material_id,
            });
        }

        self.indices
            .extend_from_slice(&[base_index, base_index + 1, base_index + 2]);
    }

    fn push_quad_points(
        &mut self,
        entrance: &LotEntrance,
        right: Vec2,
        forward: Vec2,
        a: (f32, f32, f32),
        b: (f32, f32, f32),
        c: (f32, f32, f32),
        d: (f32, f32, f32),
        desired_normal: Option<Vec3>,
        material_id: u32,
        uv_scale: f32,
    ) {
        self.push_triangle_points(
            entrance,
            right,
            forward,
            a,
            d,
            b,
            desired_normal,
            material_id,
            uv_scale,
        );
        self.push_triangle_points(
            entrance,
            right,
            forward,
            d,
            c,
            b,
            desired_normal,
            material_id,
            uv_scale,
        );
    }

    fn push_edge_skirt(
        &mut self,
        entrance: &LotEntrance,
        right: Vec2,
        forward: Vec2,
        start: (f32, f32),
        end: (f32, f32),
        start_top_y: f32,
        end_top_y: f32,
        thickness: f32,
        desired_normal: Vec3,
        material_id: u32,
        uv_scale: f32,
    ) {
        if thickness <= 1e-4 {
            return;
        }

        let a = (start.0, start.1, start_top_y - thickness);
        let b = (end.0, end.1, end_top_y - thickness);
        let c = (end.0, end.1, end_top_y);
        let d = (start.0, start.1, start_top_y);

        self.push_quad_points(
            entrance,
            right,
            forward,
            a,
            b,
            c,
            d,
            Some(desired_normal),
            material_id,
            uv_scale,
        );
    }

    fn push_thick_roof_quad(
        &mut self,
        entrance: &LotEntrance,
        right: Vec2,
        forward: Vec2,
        a: (f32, f32, f32),
        b: (f32, f32, f32),
        c: (f32, f32, f32),
        d: (f32, f32, f32),
        normal: Vec3,
        thickness: f32,
        material_id: u32,
        uv_scale: f32,
    ) {
        self.push_quad_points(
            entrance,
            right,
            forward,
            a,
            b,
            c,
            d,
            Some(normal),
            material_id,
            uv_scale,
        );
        self.push_quad_points(
            entrance,
            right,
            forward,
            (d.0, d.1, d.2 - thickness),
            (c.0, c.1, c.2 - thickness),
            (b.0, b.1, b.2 - thickness),
            (a.0, a.1, a.2 - thickness),
            Some(-normal),
            material_id,
            uv_scale,
        );
    }
}
impl BuildingMeshBuilder {
    fn emit_roof_patch(
        &mut self,
        entrance: &LotEntrance,
        right: Vec2,
        forward: Vec2,
        sampler: &RoofSurfaceSampler,
        corners_xy: &[(f32, f32); 4],
        ridge_split: Option<f32>,
        thickness: f32,
        material_id: u32,
    ) {
        let [sw_xy, se_xy, ne_xy, nw_xy] = *corners_xy;
        let up = Vec3::Y;
        let lift = |(gx, gz): (f32, f32)| (gx, gz, sampler.height_at(gx, gz));

        match (ridge_split, sampler) {
            (
                Some(ridge),
                RoofSurfaceSampler::Ridge {
                    axis: RidgeAxis::AlongZ,
                    peak_y,
                    ..
                },
            ) => {
                let south_ridge = (ridge, sw_xy.1, *peak_y);
                let north_ridge = (ridge, nw_xy.1, *peak_y);
                let south_ridge_e = (ridge, se_xy.1, *peak_y);
                let north_ridge_e = (ridge, ne_xy.1, *peak_y);

                self.push_thick_roof_quad(
                    entrance,
                    right,
                    forward,
                    lift(sw_xy),
                    south_ridge,
                    north_ridge,
                    lift(nw_xy),
                    up,
                    thickness,
                    material_id,
                    ROOF_UV_SCALE,
                );
                self.push_thick_roof_quad(
                    entrance,
                    right,
                    forward,
                    south_ridge_e,
                    lift(se_xy),
                    lift(ne_xy),
                    north_ridge_e,
                    up,
                    thickness,
                    material_id,
                    ROOF_UV_SCALE,
                );
            }
            (
                Some(ridge),
                RoofSurfaceSampler::Ridge {
                    axis: RidgeAxis::AlongX,
                    peak_y,
                    ..
                },
            ) => {
                let west_ridge = (sw_xy.0, ridge, *peak_y);
                let east_ridge = (se_xy.0, ridge, *peak_y);
                let west_ridge_n = (nw_xy.0, ridge, *peak_y);
                let east_ridge_n = (ne_xy.0, ridge, *peak_y);

                self.push_thick_roof_quad(
                    entrance,
                    right,
                    forward,
                    lift(sw_xy),
                    lift(se_xy),
                    east_ridge,
                    west_ridge,
                    up,
                    thickness,
                    material_id,
                    ROOF_UV_SCALE,
                );
                self.push_thick_roof_quad(
                    entrance,
                    right,
                    forward,
                    west_ridge_n,
                    east_ridge_n,
                    lift(ne_xy),
                    lift(nw_xy),
                    up,
                    thickness,
                    material_id,
                    ROOF_UV_SCALE,
                );
            }
            _ => {
                self.push_thick_roof_quad(
                    entrance,
                    right,
                    forward,
                    lift(sw_xy),
                    lift(se_xy),
                    lift(ne_xy),
                    lift(nw_xy),
                    up,
                    thickness,
                    material_id,
                    ROOF_UV_SCALE,
                );
            }
        }
    }

    fn emit_roof_skirts(
        &mut self,
        entrance: &LotEntrance,
        right: Vec2,
        forward: Vec2,
        sampler: &RoofSurfaceSampler,
        corners_xy: [(f32, f32); 4],
        openness: TileOpenness,
        ridge_split: Option<f32>,
        thickness: f32,
        wall_id: u32,
    ) {
        let [sw_xy, se_xy, ne_xy, nw_xy] = corners_xy;
        let sw = (sw_xy.0, sw_xy.1, sampler.height_at(sw_xy.0, sw_xy.1));
        let se = (se_xy.0, se_xy.1, sampler.height_at(se_xy.0, se_xy.1));
        let ne = (ne_xy.0, ne_xy.1, sampler.height_at(ne_xy.0, ne_xy.1));
        let nw = (nw_xy.0, nw_xy.1, sampler.height_at(nw_xy.0, nw_xy.1));

        let east = Vec3::new(right.x, 0.0, right.y);
        let west = -east;
        let north = Vec3::new(forward.x, 0.0, forward.y);
        let south = -north;

        let split = match (ridge_split, sampler) {
            (Some(ridge), RoofSurfaceSampler::Ridge { axis, peak_y, .. }) => {
                Some((*axis, ridge, *peak_y))
            }
            _ => None,
        };

        if openness.south {
            if let Some((RidgeAxis::AlongZ, ridge, peak_y)) = split {
                self.push_edge_skirt(
                    entrance,
                    right,
                    forward,
                    (sw.0, sw.1),
                    (ridge, sw.1),
                    sw.2,
                    peak_y,
                    thickness,
                    south,
                    wall_id,
                    ROOF_UV_SCALE,
                );
                self.push_edge_skirt(
                    entrance,
                    right,
                    forward,
                    (ridge, se.1),
                    (se.0, se.1),
                    peak_y,
                    se.2,
                    thickness,
                    south,
                    wall_id,
                    ROOF_UV_SCALE,
                );
            } else {
                self.push_edge_skirt(
                    entrance,
                    right,
                    forward,
                    (sw.0, sw.1),
                    (se.0, se.1),
                    sw.2,
                    se.2,
                    thickness,
                    south,
                    wall_id,
                    ROOF_UV_SCALE,
                );
            }
        }

        if openness.north {
            if let Some((RidgeAxis::AlongZ, ridge, peak_y)) = split {
                self.push_edge_skirt(
                    entrance,
                    right,
                    forward,
                    (nw.0, nw.1),
                    (ridge, nw.1),
                    nw.2,
                    peak_y,
                    thickness,
                    north,
                    wall_id,
                    ROOF_UV_SCALE,
                );
                self.push_edge_skirt(
                    entrance,
                    right,
                    forward,
                    (ridge, ne.1),
                    (ne.0, ne.1),
                    peak_y,
                    ne.2,
                    thickness,
                    north,
                    wall_id,
                    ROOF_UV_SCALE,
                );
            } else {
                self.push_edge_skirt(
                    entrance,
                    right,
                    forward,
                    (nw.0, nw.1),
                    (ne.0, ne.1),
                    nw.2,
                    ne.2,
                    thickness,
                    north,
                    wall_id,
                    ROOF_UV_SCALE,
                );
            }
        }

        if openness.west {
            if let Some((RidgeAxis::AlongX, ridge, peak_y)) = split {
                self.push_edge_skirt(
                    entrance,
                    right,
                    forward,
                    (sw.0, sw.1),
                    (sw.0, ridge),
                    sw.2,
                    peak_y,
                    thickness,
                    west,
                    wall_id,
                    ROOF_UV_SCALE,
                );
                self.push_edge_skirt(
                    entrance,
                    right,
                    forward,
                    (nw.0, ridge),
                    (nw.0, nw.1),
                    peak_y,
                    nw.2,
                    thickness,
                    west,
                    wall_id,
                    ROOF_UV_SCALE,
                );
            } else {
                self.push_edge_skirt(
                    entrance,
                    right,
                    forward,
                    (sw.0, sw.1),
                    (nw.0, nw.1),
                    sw.2,
                    nw.2,
                    thickness,
                    west,
                    wall_id,
                    ROOF_UV_SCALE,
                );
            }
        }

        if openness.east {
            if let Some((RidgeAxis::AlongX, ridge, peak_y)) = split {
                self.push_edge_skirt(
                    entrance,
                    right,
                    forward,
                    (se.0, se.1),
                    (se.0, ridge),
                    se.2,
                    peak_y,
                    thickness,
                    east,
                    wall_id,
                    ROOF_UV_SCALE,
                );
                self.push_edge_skirt(
                    entrance,
                    right,
                    forward,
                    (ne.0, ridge),
                    (ne.0, ne.1),
                    peak_y,
                    ne.2,
                    thickness,
                    east,
                    wall_id,
                    ROOF_UV_SCALE,
                );
            } else {
                self.push_edge_skirt(
                    entrance,
                    right,
                    forward,
                    (se.0, se.1),
                    (ne.0, ne.1),
                    se.2,
                    ne.2,
                    thickness,
                    east,
                    wall_id,
                    ROOF_UV_SCALE,
                );
            }
        }
    }

    fn emit_roof_component(
        &mut self,
        entrance: &LotEntrance,
        right: Vec2,
        forward: Vec2,
        component: &RoofComponent,
        roof_id: u32,
        wall_id: u32,
    ) {
        let roof_thickness = 0.25_f32;
        let overhang = 0.35_f32;

        let can_merge_interior = matches!(
            component.sampler,
            RoofSurfaceSampler::Flat { .. } | RoofSurfaceSampler::Angled { .. }
        );

        let interior: HashSet<TilePos> = if can_merge_interior {
            component
                .tiles
                .iter()
                .copied()
                .filter(|pos| !tile_openness(&component.tiles, pos.x, pos.z).any_open())
                .collect()
        } else {
            HashSet::new()
        };

        for rect in greedy_merge_tiles(&interior) {
            let x0 = rect.min_x as f32;
            let x1 = rect.max_x as f32;
            let z0 = rect.min_z as f32;
            let z1 = rect.max_z as f32;
            let corners = [(x0, z0), (x1, z0), (x1, z1), (x0, z1)];
            self.emit_roof_patch(
                entrance,
                right,
                forward,
                &component.sampler,
                &corners,
                None,
                roof_thickness,
                roof_id,
            );
        }

        for &pos in component.tiles.iter() {
            if interior.contains(&pos) {
                continue;
            }

            let openness = tile_openness(&component.tiles, pos.x, pos.z);
            let x0 = pos.x as f32;
            let x1 = x0 + 1.0;
            let z0 = pos.z as f32;
            let z1 = z0 + 1.0;

            let corners = overhung_corners(x0, x1, z0, z1, openness, overhang);
            let split = component.sampler.ridge_split(x0, x1, z0, z1);

            self.emit_roof_patch(
                entrance,
                right,
                forward,
                &component.sampler,
                &corners,
                split,
                roof_thickness,
                roof_id,
            );

            if openness.any_open() {
                self.emit_roof_skirts(
                    entrance,
                    right,
                    forward,
                    &component.sampler,
                    corners,
                    openness,
                    split,
                    roof_thickness,
                    wall_id,
                );
            }
        }
    }

    fn emit_roof(
        &mut self,
        entrance: &LotEntrance,
        right: Vec2,
        forward: Vec2,
        components: &[RoofComponent],
        roof_id: u32,
        wall_id: u32,
    ) {
        for component in components {
            self.emit_roof_component(entrance, right, forward, component, roof_id, wall_id);
        }
    }
}
pub struct RoofAccessoryContext<'a> {
    pub entrance: &'a LotEntrance,
    pub right: Vec2,
    pub forward: Vec2,
    pub component: &'a RoofComponent,
}

pub trait RoofAccessory {
    fn place(&self, ctx: &RoofAccessoryContext, props: &mut Props) -> Vec<PropInstanceId>;
}

pub struct SolarPanelAccessory {
    pub spacing: i16,
    pub lift: f32,
}

impl Default for SolarPanelAccessory {
    fn default() -> Self {
        Self {
            spacing: 1,
            lift: 0.05,
        }
    }
}

impl RoofAccessory for SolarPanelAccessory {
    fn place(&self, ctx: &RoofAccessoryContext, props: &mut Props) -> Vec<PropInstanceId> {
        let mut ids = Vec::new();
        if !matches!(
            ctx.component.sampler,
            RoofSurfaceSampler::Flat { .. } | RoofSurfaceSampler::Angled { .. }
        ) {
            return ids;
        }

        let min_x = ctx.component.min_x.ceil() as i16;
        let max_x = ctx.component.max_x.floor() as i16;
        let min_z = ctx.component.min_z.ceil() as i16;
        let max_z = ctx.component.max_z.floor() as i16;

        let mut x = min_x;
        while x < max_x {
            let mut z = min_z;
            while z < max_z {
                let center = TilePos::new(x, z);
                if ctx.component.tiles.contains(&center) {
                    let gx = x as f32 + 0.5;
                    let gz = z as f32 + 0.5;
                    let height = ctx.component.sampler.height_at(gx, gz) + self.lift;
                    let normal = ctx.component.sampler.normal_at(gx, gz);

                    let mut pos = ctx.entrance.pos.add_vec2(ctx.right * gx + ctx.forward * gz);
                    pos.local.y = height;

                    let id = props.place_prop(
                        "solar_panel",
                        PropInstance {
                            id: None,
                            archetype_id: None,
                            pos,
                            scale: 1.0,
                            rotation_y_rad: normal.z.atan2(normal.x),
                            seed: rand::random(),
                            color: [1.0, 1.0, 1.0, 1.0],
                            wind_strength: 0.0,
                            variant: 0,
                            generated: false,
                        },
                    );
                    ids.push(id);
                }
                z += self.spacing + 1;
            }
            x += self.spacing + 1;
        }

        ids
    }
}

pub struct AntennaAccessory {
    pub lift: f32,
}

impl Default for AntennaAccessory {
    fn default() -> Self {
        Self { lift: 0.0 }
    }
}

impl RoofAccessory for AntennaAccessory {
    fn place(&self, ctx: &RoofAccessoryContext, props: &mut Props) -> Vec<PropInstanceId> {
        let gx = (ctx.component.min_x + ctx.component.max_x) * 0.5;
        let gz = (ctx.component.min_z + ctx.component.max_z) * 0.5;
        let height = ctx.component.sampler.height_at(gx, gz) + self.lift;

        let mut pos = ctx.entrance.pos.add_vec2(ctx.right * gx + ctx.forward * gz);
        pos.local.y = height;

        let id = props.place_prop(
            "antenna",
            PropInstance {
                id: None,
                archetype_id: None,
                pos,
                scale: 1.0,
                rotation_y_rad: 0.0,
                seed: rand::random(),
                color: [1.0, 1.0, 1.0, 1.0],
                wind_strength: 0.0,
                variant: 0,
                generated: false,
            },
        );

        vec![id]
    }
}

pub fn active_roof_accessories(misc: &MiscBuildingParams) -> Vec<Box<dyn RoofAccessory>> {
    let mut accessories: Vec<Box<dyn RoofAccessory>> = Vec::new();
    if misc.solar_modules {
        accessories.push(Box::new(SolarPanelAccessory::default()));
    }
    if misc.antenna {
        accessories.push(Box::new(AntennaAccessory::default()));
    }
    accessories
}
