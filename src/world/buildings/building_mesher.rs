use crate::helpers::positions::WorldPos;
use crate::renderer::gizmo::gizmo::Gizmo;
use crate::renderer::props::{PropInstance, Props};
use crate::renderer::textures::material_keys::terrain_material_keys;
use crate::world::buildings::buildings::{
    BuildingId, Buildings, DrivewayMaterial, RoofMaterial, RoofType, WallMaterial,
};
use crate::world::buildings::zoning::{Lot, LotId, Tile, TilePos, TileType, Zoning};
use crate::world::cars::parking::ParkingStorage;
use crate::world::roads::road_mesh_manager::{ChunkId, chunk_id_to_coord};
use crate::world::terrain::terrain_editing::TerrainEditSource;
use crate::world::terrain::terrain_subsystem::Terrain;
use glam::{Vec2, Vec3};
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use wgpu::{VertexAttribute, VertexFormat};
use wgpu_render_manager::generator::{MipmapMode, TextureKey, TextureParams};
use wgpu_render_manager::renderer::RenderManager;

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
        terrain: &mut Terrain,
        props: &mut Props,
        cid: ChunkId,
        buildings: &mut Buildings,
        zoning: &mut Zoning,
        parking_storage: &mut ParkingStorage,
        gizmo: &mut Gizmo,
    ) -> BuildingChunkMesh {
        let mut mesh = BuildingMeshBuilder {
            vertices: Vec::new(),
            indices: Vec::new(),
        };

        let lot_ids: Vec<LotId> = zoning.zoning_storage.lots_in_chunk(cid);

        for id in lot_ids.iter() {
            let Some(lot) = zoning.zoning_storage.get_mut_lot(*id) else {
                continue;
            };
            let Some(building) = buildings.storage.get_mut(lot.building_id) else {
                continue;
            };
            if let Some(edit_id) = building.edit_id {
                terrain.terrain_editor.remove_edit(edit_id);
            }
            props.remove_instances(lot.center.chunk, building.prop_instance_ids.as_slice());
            let bounds: Vec<_> = lot
                .bounds
                .iter()
                .cloned()
                .map(|mut p| {
                    p.local.y = lot.entrance.pos.local.y;
                    p
                })
                .collect();

            building.edit_id = Some(terrain.terrain_editor.push_flat_polygon(
                bounds,
                -0.01,
                3.0,
                TerrainEditSource::Building(building.id),
            ));
            let mut prop_instance_ids = vec![];
            let Some(level) = building.current_level_params(&buildings.catalog) else {
                continue;
            };
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
                    MipmapMode::Generate,
                ),
                WallMaterial::Custom(key) => key.clone(),
            };
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
            let roof_key = match &level.roof_material {
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
                RoofMaterial::Custom(key) => key.clone(),
            };
            let window_key = match &level.miscellaneous.window_material_accent {
                WallMaterial::Paint(color) => TextureKey::new(
                    "paint",
                    TextureParams::default().with_primary_color(**color),
                    512,
                    MipmapMode::Generate,
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
                    MipmapMode::Generate,
                ),
                DrivewayMaterial::Custom(key) => key.clone(),
            };
            let mut grass_key = terrain_material_keys().remove(0);
            grass_key.resolution = 512; // MADNATORY! (Intentional misspelling)
            // This comment is officially useless, BUT no one can stop me, this is MY game, my territory, so I will piss on it like a dog to mark my territory.
            grass_key.params.scale = 20.0;

            let mut garden_key = grass_key.clone();
            garden_key.params.color_primary[1] *= 1.2; // Greener, because I want to.
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
            //println!("{:?}", mat_ids);
            let border = &lot.bounds;
            let n = border.len();
            if n < 3 {
                continue;
            }

            let direction = lot.entrance.dir;
            let forward = Vec2::new(direction.x, direction.z).normalize();
            let right = Vec2::new(forward.y, -forward.x);
            let zero_height = lot.entrance.pos.local.y;

            let mut roof_tiles: Vec<RoofTile> = Vec::new();
            if lot.layout.is_none() {
                //println!("making layout {}", building.id);
                let lot_layout = lot.generate_layout(parking_storage);
                lot.layout = Some(lot_layout);
            }

            let Some(layout) = lot.layout.as_ref() else {
                continue;
            };
            let tiles = &layout.tiles;
            for (&TilePos { x, z }, tile) in tiles.iter() {
                let direction = lot.entrance.dir;

                let forward = Vec2::new(direction.x, direction.z).normalize();
                let right = Vec2::new(forward.y, -forward.x);
                let mut tile_origin = lot
                    .entrance
                    .pos
                    .add_vec2(right * (x as f32 + 0.5) + forward * (z as f32 + 0.5));
                tile_origin.local.y = terrain.get_height_at(tile_origin, true);
                let is_house_tile = |tp: Option<&Tile>| {
                    matches!(tp, Some(Tile::Square(TileType::House)))
                        || matches!(tp, Some(Tile::Square(TileType::Garage)))
                };

                let south_open = !is_house_tile(tiles.get(&TilePos::new(x, z - 1)));
                let east_open = !is_house_tile(tiles.get(&TilePos::new(x + 1, z)));
                let north_open = !is_house_tile(tiles.get(&TilePos::new(x, z + 1)));
                let west_open = !is_house_tile(tiles.get(&TilePos::new(x - 1, z)));
                match tile {
                    Tile::Square(tile_type) => {
                        let zero_height = lot.entrance.pos.local.y;
                        let corners_local =
                            [(0.0_f32, 0.0_f32), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];

                        let mut points: Vec<WorldPos> = corners_local
                            .iter()
                            .map(|(cx, cz)| {
                                let mut corner = lot
                                    .entrance
                                    .pos
                                    .add_vec2(right * (x as f32 + cx) + forward * (z as f32 + cz));
                                corner.local.y = terrain.get_height_at(corner, true);
                                corner
                            })
                            .collect();

                        // Close the loop
                        points.push(points[0]);

                        //gizmo.polyline(points.as_slice(), [0.0, 0.0, 0.0, 1.0], 0.0, 0.1, 100.0);
                        match tile_type {
                            TileType::Grass => {
                                mesh.push_square(
                                    lot,
                                    zero_height,
                                    right,
                                    forward,
                                    TilePos::new(x, z),
                                    grass_id,
                                );
                            }
                            TileType::Tree => {
                                mesh.push_square(
                                    lot,
                                    zero_height,
                                    right,
                                    forward,
                                    TilePos::new(x, z),
                                    garden_id,
                                );
                                let mut center = lot.entrance.pos.add_vec2(
                                    right * (x as f32 + 0.5) + forward * (z as f32 + 0.5),
                                );
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
                                mesh.push_square(
                                    lot,
                                    zero_height,
                                    right,
                                    forward,
                                    TilePos::new(x, z),
                                    garden_id,
                                );
                            }
                            TileType::House => {
                                let roof_y =
                                    zero_height + level.num_stories as f32 * level.story_height;

                                mesh.push_walled_square(
                                    lot,
                                    zero_height,
                                    roof_y,
                                    right,
                                    forward,
                                    TilePos::new(x, z),
                                    wall_id,
                                    south_open,
                                    east_open,
                                    north_open,
                                    west_open,
                                );

                                roof_tiles.push(RoofTile {
                                    tile_pos: TilePos::new(x, z),
                                    base_y: roof_y,
                                });
                            }
                            TileType::HouseBalcony => {
                                mesh.push_square(
                                    lot,
                                    zero_height,
                                    right,
                                    forward,
                                    TilePos::new(x, z),
                                    notex_id,
                                );
                            }
                            TileType::HouseEntrance => {
                                mesh.push_square(
                                    lot,
                                    zero_height,
                                    right,
                                    forward,
                                    TilePos::new(x, z),
                                    notex_id,
                                );
                            }
                            TileType::LotEntrance => {
                                mesh.push_square(
                                    lot,
                                    zero_height,
                                    right,
                                    forward,
                                    TilePos::new(x, z),
                                    notex_id,
                                );
                            }
                            TileType::Garage => {
                                if let Some(garage) = &level.garage {
                                    let roof_y = zero_height
                                        + garage.story_height * garage.num_stories as f32;

                                    mesh.push_walled_square(
                                        lot,
                                        zero_height,
                                        roof_y,
                                        right,
                                        forward,
                                        TilePos::new(x, z),
                                        wall_id,
                                        south_open,
                                        east_open,
                                        north_open,
                                        west_open,
                                    );

                                    roof_tiles.push(RoofTile {
                                        tile_pos: TilePos::new(x, z),
                                        base_y: roof_y,
                                    });
                                }
                            }
                            TileType::Driveway => {
                                mesh.push_square(
                                    lot,
                                    zero_height,
                                    right,
                                    forward,
                                    TilePos::new(x, z),
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
            mesh.construct_roof(
                lot,
                right,
                forward,
                &level.roof,
                &roof_tiles,
                roof_id,
                roof_side_id,
            );
            building.prop_instance_ids = prop_instance_ids;
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
        terrain: &mut Terrain,
        props: &mut Props,
        chunk_id: ChunkId,
        buildings: &mut Buildings,
        zoning: &mut Zoning,
        parking_storage: &mut ParkingStorage,
        gizmo: &mut Gizmo,
    ) -> &BuildingChunkMesh {
        let mesh = self.build_mesh_for_chunk(
            render_manager,
            terrain,
            props,
            chunk_id,
            buildings,
            zoning,
            parking_storage,
            gizmo,
        );
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
        tile_pos: TilePos,
        material_id: u32,
    ) {
        let corners_local = [(0.0_f32, 0.0_f32), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];

        let points: Vec<WorldPos> = corners_local
            .iter()
            .map(|(cx, cz)| {
                let mut corner = lot.entrance.pos.add_vec2(
                    right * (tile_pos.x as f32 + cx) + forward * (tile_pos.z as f32 + cz),
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
        tile_pos: TilePos,
        material_id: u32,
        south_open: bool,
        east_open: bool,
        north_open: bool,
        west_open: bool,
    ) {
        let (bottom_y, top_y) = if start_height <= end_height {
            (start_height, end_height)
        } else {
            (end_height, start_height)
        };

        let corner_world = |cx: f32, cz: f32, y: f32| -> WorldPos {
            let mut corner = lot
                .entrance
                .pos
                .add_vec2(right * (tile_pos.x as f32 + cx) + forward * (tile_pos.z as f32 + cz));
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

        if south_open {
            push_quad(p00b, p00t, p10t, p10b, n_south);
        }
        if east_open {
            push_quad(p10b, p10t, p11t, p11b, n_east);
        }
        if north_open {
            push_quad(p01b, p11b, p11t, p01t, n_north);
        }
        if west_open {
            push_quad(p00b, p01b, p01t, p00t, n_west);
        }
    }

    fn grid_point_world(
        lot: &Lot,
        right: Vec2,
        forward: Vec2,
        gx: f32,
        gz: f32,
        y: f32,
    ) -> WorldPos {
        let mut p = lot.entrance.pos.add_vec2(right * gx + forward * gz);
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

    fn push_triangle_points(
        &mut self,
        lot: &Lot,
        right: Vec2,
        forward: Vec2,
        a: (f32, f32, f32),
        b: (f32, f32, f32),
        c: (f32, f32, f32),
        desired_normal: Option<Vec3>,
        material_id: u32,
        uv_a: [f32; 2],
        uv_b: [f32; 2],
        uv_c: [f32; 2],
    ) {
        let mut pts = [a, b, c];
        let mut uvs = [uv_a, uv_b, uv_c];

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
                uvs.swap(1, 2);
                normal = calc_normal(&pts);
                if normal.length_squared() < 1e-6 {
                    return;
                }
            }
        }

        let normal = normal.normalize();
        let base_index = self.vertices.len() as u32;

        for (pt, uv) in pts.into_iter().zip(uvs.into_iter()) {
            let wp = Self::grid_point_world(lot, right, forward, pt.0, pt.1, pt.2);
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

    /// Quad points are expected in perimeter order:
    /// a = southwest-ish, b = southeast-ish, c = northeast-ish, d = northwest-ish
    fn push_quad_points(
        &mut self,
        lot: &Lot,
        right: Vec2,
        forward: Vec2,
        a: (f32, f32, f32),
        b: (f32, f32, f32),
        c: (f32, f32, f32),
        d: (f32, f32, f32),
        desired_normal: Option<Vec3>,
        material_id: u32,
    ) {
        // Same split pattern as your push_square: (a, d, b) and (d, c, b)
        self.push_triangle_points(
            lot,
            right,
            forward,
            a,
            d,
            b,
            desired_normal,
            material_id,
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        );

        self.push_triangle_points(
            lot,
            right,
            forward,
            d,
            c,
            b,
            desired_normal,
            material_id,
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        );
    }

    fn push_edge_skirt(
        &mut self,
        lot: &Lot,
        right: Vec2,
        forward: Vec2,
        start: (f32, f32),
        end: (f32, f32),
        start_top_y: f32,
        end_top_y: f32,
        thickness: f32,
        desired_normal: Vec3,
        material_id: u32,
    ) {
        if thickness <= 1e-4 {
            return;
        }

        let a = (start.0, start.1, start_top_y - thickness);
        let b = (end.0, end.1, end_top_y - thickness);
        let c = (end.0, end.1, end_top_y);
        let d = (start.0, start.1, start_top_y);

        self.push_quad_points(
            lot,
            right,
            forward,
            a,
            b,
            c,
            d,
            Some(desired_normal),
            material_id,
        );
    }

    fn construct_roof(
        &mut self,
        lot: &Lot,
        right: Vec2,
        forward: Vec2,
        roof: &RoofType,
        roof_tiles: &[RoofTile],
        roof_id: u32,
        wall_id: u32,
    ) {
        if roof_tiles.is_empty() {
            return;
        }

        let roof_thickness = 0.25_f32;
        let overhang = 0.35_f32;

        let up = Vec3::Y;
        let east = Vec3::new(right.x, 0.0, right.y);
        let west = -east;
        let north = Vec3::new(forward.x, 0.0, forward.y);
        let south = -north;

        let tile_map: HashMap<TilePos, f32> =
            roof_tiles.iter().map(|t| (t.tile_pos, t.base_y)).collect();

        let mut visited: HashSet<TilePos> = HashSet::new();

        macro_rules! emit_thick_quad {
            ($a:expr, $b:expr, $c:expr, $d:expr, $normal:expr, $material:expr) => {{
                let a = $a;
                let b = $b;
                let c = $c;
                let d = $d;

                self.push_quad_points(lot, right, forward, a, b, c, d, Some($normal), $material);

                self.push_quad_points(
                    lot,
                    right,
                    forward,
                    (d.0, d.1, d.2 - roof_thickness),
                    (c.0, c.1, c.2 - roof_thickness),
                    (b.0, b.1, b.2 - roof_thickness),
                    (a.0, a.1, a.2 - roof_thickness),
                    Some(-$normal),
                    $material,
                );
            }};
        }

        for (&start, &base_y) in tile_map.iter() {
            if visited.contains(&start) {
                continue;
            }

            let wanted_bits = base_y.to_bits();
            let mut queue = VecDeque::new();
            queue.push_back(start);
            visited.insert(start);

            let mut component = Vec::new();

            while let Some(pos) = queue.pop_front() {
                component.push(pos);

                let neighbors = pos.get_neighbors_plus();

                for n in neighbors {
                    if visited.contains(&n) {
                        continue;
                    }

                    if let Some(&neighbor_y) = tile_map.get(&n) {
                        if neighbor_y.to_bits() == wanted_bits {
                            visited.insert(n);
                            queue.push_back(n);
                        }
                    }
                }
            }

            let component_set: HashSet<TilePos> = component.iter().copied().collect();

            let mut min_x = f32::INFINITY;
            let mut min_z = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut max_z = f32::NEG_INFINITY;

            for &TilePos { x, z } in &component {
                min_x = min_x.min(x as f32);
                min_z = min_z.min(z as f32);
                max_x = max_x.max(x as f32 + 1.0);
                max_z = max_z.max(z as f32 + 1.0);
            }

            match roof {
                RoofType::Flat => {
                    for &TilePos { x, z } in &component {
                        let x0 = x as f32;
                        let x1 = x0 + 1.0;
                        let z0 = z as f32;
                        let z1 = z0 + 1.0;

                        let west_missing = !component_set.contains(&TilePos::new(x - 1, z));
                        let east_missing = !component_set.contains(&TilePos::new(x + 1, z));
                        let south_missing = !component_set.contains(&TilePos::new(x, z - 1));
                        let north_missing = !component_set.contains(&TilePos::new(x, z + 1));

                        let sw = (
                            x0 - if west_missing { overhang } else { 0.0 },
                            z0 - if south_missing { overhang } else { 0.0 },
                            base_y,
                        );
                        let se = (
                            x1 + if east_missing { overhang } else { 0.0 },
                            z0 - if south_missing { overhang } else { 0.0 },
                            base_y,
                        );
                        let ne = (
                            x1 + if east_missing { overhang } else { 0.0 },
                            z1 + if north_missing { overhang } else { 0.0 },
                            base_y,
                        );
                        let nw = (
                            x0 - if west_missing { overhang } else { 0.0 },
                            z1 + if north_missing { overhang } else { 0.0 },
                            base_y,
                        );

                        emit_thick_quad!(sw, se, ne, nw, up, roof_id);

                        if south_missing {
                            self.push_edge_skirt(
                                lot,
                                right,
                                forward,
                                (sw.0, sw.1),
                                (se.0, se.1),
                                base_y,
                                base_y,
                                roof_thickness,
                                south,
                                wall_id,
                            );
                        }
                        if east_missing {
                            self.push_edge_skirt(
                                lot,
                                right,
                                forward,
                                (se.0, se.1),
                                (ne.0, ne.1),
                                base_y,
                                base_y,
                                roof_thickness,
                                east,
                                wall_id,
                            );
                        }
                        if north_missing {
                            self.push_edge_skirt(
                                lot,
                                right,
                                forward,
                                (nw.0, nw.1),
                                (ne.0, ne.1),
                                base_y,
                                base_y,
                                roof_thickness,
                                north,
                                wall_id,
                            );
                        }
                        if west_missing {
                            self.push_edge_skirt(
                                lot,
                                right,
                                forward,
                                (sw.0, sw.1),
                                (nw.0, nw.1),
                                base_y,
                                base_y,
                                roof_thickness,
                                west,
                                wall_id,
                            );
                        }
                    }
                }

                RoofType::Angled {
                    pitch,
                    direction_rad,
                } => {
                    let rise_per_unit = pitch.tan();
                    let dir = Vec2::new(direction_rad.cos(), direction_rad.sin());

                    let mut min_proj = f32::INFINITY;
                    for &TilePos { x, z } in &component {
                        let x0 = x as f32;
                        let x1 = x0 + 1.0;
                        let z0 = z as f32;
                        let z1 = z0 + 1.0;

                        let west_missing = !component_set.contains(&TilePos::new(x - 1, z));
                        let east_missing = !component_set.contains(&TilePos::new(x + 1, z));
                        let south_missing = !component_set.contains(&TilePos::new(x, z - 1));
                        let north_missing = !component_set.contains(&TilePos::new(x, z + 1));

                        let corners = [
                            (
                                x0 - if west_missing { overhang } else { 0.0 },
                                z0 - if south_missing { overhang } else { 0.0 },
                            ),
                            (
                                x1 + if east_missing { overhang } else { 0.0 },
                                z0 - if south_missing { overhang } else { 0.0 },
                            ),
                            (
                                x1 + if east_missing { overhang } else { 0.0 },
                                z1 + if north_missing { overhang } else { 0.0 },
                            ),
                            (
                                x0 - if west_missing { overhang } else { 0.0 },
                                z1 + if north_missing { overhang } else { 0.0 },
                            ),
                        ];

                        for (gx, gz) in corners {
                            min_proj = min_proj.min(Vec2::new(gx, gz).dot(dir));
                        }
                    }

                    let height_at = |gx: f32, gz: f32| -> f32 {
                        let proj = Vec2::new(gx, gz).dot(dir);
                        base_y + (proj - min_proj) * rise_per_unit
                    };

                    for &TilePos { x, z } in &component {
                        let x0 = x as f32;
                        let x1 = x0 + 1.0;
                        let z0 = z as f32;
                        let z1 = z0 + 1.0;

                        let west_missing = !component_set.contains(&TilePos::new(x - 1, z));
                        let east_missing = !component_set.contains(&TilePos::new(x + 1, z));
                        let south_missing = !component_set.contains(&TilePos::new(x, z - 1));
                        let north_missing = !component_set.contains(&TilePos::new(x, z + 1));

                        let sw_xy = (
                            x0 - if west_missing { overhang } else { 0.0 },
                            z0 - if south_missing { overhang } else { 0.0 },
                        );
                        let se_xy = (
                            x1 + if east_missing { overhang } else { 0.0 },
                            z0 - if south_missing { overhang } else { 0.0 },
                        );
                        let ne_xy = (
                            x1 + if east_missing { overhang } else { 0.0 },
                            z1 + if north_missing { overhang } else { 0.0 },
                        );
                        let nw_xy = (
                            x0 - if west_missing { overhang } else { 0.0 },
                            z1 + if north_missing { overhang } else { 0.0 },
                        );

                        let sw = (sw_xy.0, sw_xy.1, height_at(sw_xy.0, sw_xy.1));
                        let se = (se_xy.0, se_xy.1, height_at(se_xy.0, se_xy.1));
                        let ne = (ne_xy.0, ne_xy.1, height_at(ne_xy.0, ne_xy.1));
                        let nw = (nw_xy.0, nw_xy.1, height_at(nw_xy.0, nw_xy.1));

                        emit_thick_quad!(sw, se, ne, nw, up, roof_id);

                        if south_missing {
                            self.push_edge_skirt(
                                lot,
                                right,
                                forward,
                                (sw.0, sw.1),
                                (se.0, se.1),
                                sw.2,
                                se.2,
                                roof_thickness,
                                south,
                                wall_id,
                            );
                        }
                        if east_missing {
                            self.push_edge_skirt(
                                lot,
                                right,
                                forward,
                                (se.0, se.1),
                                (ne.0, ne.1),
                                se.2,
                                ne.2,
                                roof_thickness,
                                east,
                                wall_id,
                            );
                        }
                        if north_missing {
                            self.push_edge_skirt(
                                lot,
                                right,
                                forward,
                                (nw.0, nw.1),
                                (ne.0, ne.1),
                                nw.2,
                                ne.2,
                                roof_thickness,
                                north,
                                wall_id,
                            );
                        }
                        if west_missing {
                            self.push_edge_skirt(
                                lot,
                                right,
                                forward,
                                (sw.0, sw.1),
                                (nw.0, nw.1),
                                sw.2,
                                nw.2,
                                roof_thickness,
                                west,
                                wall_id,
                            );
                        }
                    }
                }

                RoofType::Triangle(pitch_deg) => {
                    let rise_per_unit = pitch_deg.to_radians().tan();
                    let width = max_x - min_x;
                    let depth = max_z - min_z;
                    let eps = 1e-4;

                    let slope_across_x = width <= depth;

                    if slope_across_x {
                        let ridge_x = (min_x + max_x) * 0.5;
                        let peak_y = base_y + (ridge_x - min_x) * rise_per_unit;

                        let height_x = |gx: f32| -> f32 {
                            if gx <= ridge_x {
                                base_y + (gx - min_x) * rise_per_unit
                            } else {
                                base_y + (max_x - gx) * rise_per_unit
                            }
                        };

                        for &TilePos { x, z } in &component {
                            let x0 = x as f32;
                            let x1 = x0 + 1.0;
                            let z0 = z as f32;
                            let z1 = z0 + 1.0;

                            let west_missing = !component_set.contains(&TilePos::new(x - 1, z));
                            let east_missing = !component_set.contains(&TilePos::new(x + 1, z));
                            let south_missing = !component_set.contains(&TilePos::new(x, z - 1));
                            let north_missing = !component_set.contains(&TilePos::new(x, z + 1));

                            let sw_xy = (
                                x0 - if west_missing { overhang } else { 0.0 },
                                z0 - if south_missing { overhang } else { 0.0 },
                            );
                            let se_xy = (
                                x1 + if east_missing { overhang } else { 0.0 },
                                z0 - if south_missing { overhang } else { 0.0 },
                            );
                            let ne_xy = (
                                x1 + if east_missing { overhang } else { 0.0 },
                                z1 + if north_missing { overhang } else { 0.0 },
                            );
                            let nw_xy = (
                                x0 - if west_missing { overhang } else { 0.0 },
                                z1 + if north_missing { overhang } else { 0.0 },
                            );

                            let sw = (sw_xy.0, sw_xy.1, height_x(sw_xy.0));
                            let se = (se_xy.0, se_xy.1, height_x(se_xy.0));
                            let ne = (ne_xy.0, ne_xy.1, height_x(ne_xy.0));
                            let nw = (nw_xy.0, nw_xy.1, height_x(nw_xy.0));

                            if ridge_x > x0 + eps && ridge_x < x1 - eps {
                                let south_ridge = (ridge_x, sw.1, peak_y);
                                let north_ridge = (ridge_x, nw.1, peak_y);
                                let south_ridge_e = (ridge_x, se.1, peak_y);
                                let north_ridge_e = (ridge_x, ne.1, peak_y);

                                emit_thick_quad!(sw, south_ridge, north_ridge, nw, up, roof_id);
                                emit_thick_quad!(south_ridge_e, se, ne, north_ridge_e, up, roof_id);
                            } else {
                                emit_thick_quad!(sw, se, ne, nw, up, roof_id);
                            }

                            if south_missing {
                                if ridge_x > x0 + eps && ridge_x < x1 - eps {
                                    self.push_edge_skirt(
                                        lot,
                                        right,
                                        forward,
                                        (sw.0, sw.1),
                                        (ridge_x, sw.1),
                                        sw.2,
                                        peak_y,
                                        roof_thickness,
                                        south,
                                        wall_id,
                                    );
                                    self.push_edge_skirt(
                                        lot,
                                        right,
                                        forward,
                                        (ridge_x, se.1),
                                        (se.0, se.1),
                                        peak_y,
                                        se.2,
                                        roof_thickness,
                                        south,
                                        wall_id,
                                    );
                                } else {
                                    self.push_edge_skirt(
                                        lot,
                                        right,
                                        forward,
                                        (sw.0, sw.1),
                                        (se.0, se.1),
                                        sw.2,
                                        se.2,
                                        roof_thickness,
                                        south,
                                        wall_id,
                                    );
                                }
                            }

                            if north_missing {
                                if ridge_x > x0 + eps && ridge_x < x1 - eps {
                                    self.push_edge_skirt(
                                        lot,
                                        right,
                                        forward,
                                        (nw.0, nw.1),
                                        (ridge_x, nw.1),
                                        nw.2,
                                        peak_y,
                                        roof_thickness,
                                        north,
                                        wall_id,
                                    );
                                    self.push_edge_skirt(
                                        lot,
                                        right,
                                        forward,
                                        (ridge_x, ne.1),
                                        (ne.0, ne.1),
                                        peak_y,
                                        ne.2,
                                        roof_thickness,
                                        north,
                                        wall_id,
                                    );
                                } else {
                                    self.push_edge_skirt(
                                        lot,
                                        right,
                                        forward,
                                        (nw.0, nw.1),
                                        (ne.0, ne.1),
                                        nw.2,
                                        ne.2,
                                        roof_thickness,
                                        north,
                                        wall_id,
                                    );
                                }
                            }

                            if west_missing {
                                self.push_edge_skirt(
                                    lot,
                                    right,
                                    forward,
                                    (sw.0, sw.1),
                                    (nw.0, nw.1),
                                    sw.2,
                                    nw.2,
                                    roof_thickness,
                                    west,
                                    wall_id,
                                );
                            }

                            if east_missing {
                                self.push_edge_skirt(
                                    lot,
                                    right,
                                    forward,
                                    (se.0, se.1),
                                    (ne.0, ne.1),
                                    se.2,
                                    ne.2,
                                    roof_thickness,
                                    east,
                                    wall_id,
                                );
                            }
                        }
                    } else {
                        let ridge_z = (min_z + max_z) * 0.5;
                        let peak_y = base_y + (ridge_z - min_z) * rise_per_unit;

                        let height_z = |gz: f32| -> f32 {
                            if gz <= ridge_z {
                                base_y + (gz - min_z) * rise_per_unit
                            } else {
                                base_y + (max_z - gz) * rise_per_unit
                            }
                        };

                        for &TilePos { x, z } in &component {
                            let x0 = x as f32;
                            let x1 = x as f32 + 1.0;
                            let z0 = z as f32;
                            let z1 = z as f32 + 1.0;

                            let west_missing = !component_set.contains(&TilePos::new(x - 1, z));
                            let east_missing = !component_set.contains(&TilePos::new(x + 1, z));
                            let south_missing = !component_set.contains(&TilePos::new(x, z - 1));
                            let north_missing = !component_set.contains(&TilePos::new(x, z + 1));

                            let sw_xy = (
                                x0 - if west_missing { overhang } else { 0.0 },
                                z0 - if south_missing { overhang } else { 0.0 },
                            );
                            let se_xy = (
                                x1 + if east_missing { overhang } else { 0.0 },
                                z0 - if south_missing { overhang } else { 0.0 },
                            );
                            let ne_xy = (
                                x1 + if east_missing { overhang } else { 0.0 },
                                z1 + if north_missing { overhang } else { 0.0 },
                            );
                            let nw_xy = (
                                x0 - if west_missing { overhang } else { 0.0 },
                                z1 + if north_missing { overhang } else { 0.0 },
                            );

                            let sw = (sw_xy.0, sw_xy.1, height_z(sw_xy.1));
                            let se = (se_xy.0, se_xy.1, height_z(se_xy.1));
                            let ne = (ne_xy.0, ne_xy.1, height_z(ne_xy.1));
                            let nw = (nw_xy.0, nw_xy.1, height_z(nw_xy.1));

                            if ridge_z > z0 + eps && ridge_z < z1 - eps {
                                let west_ridge = (sw.0, ridge_z, peak_y);
                                let east_ridge = (se.0, ridge_z, peak_y);
                                let west_ridge_n = (nw.0, ridge_z, peak_y);
                                let east_ridge_n = (ne.0, ridge_z, peak_y);

                                emit_thick_quad!(sw, se, east_ridge, west_ridge, up, roof_id);
                                emit_thick_quad!(west_ridge_n, east_ridge_n, ne, nw, up, roof_id);
                            } else {
                                emit_thick_quad!(sw, se, ne, nw, up, roof_id);
                            }

                            if south_missing {
                                self.push_edge_skirt(
                                    lot,
                                    right,
                                    forward,
                                    (sw.0, sw.1),
                                    (se.0, se.1),
                                    sw.2,
                                    se.2,
                                    roof_thickness,
                                    south,
                                    wall_id,
                                );
                            }

                            if north_missing {
                                self.push_edge_skirt(
                                    lot,
                                    right,
                                    forward,
                                    (nw.0, nw.1),
                                    (ne.0, ne.1),
                                    nw.2,
                                    ne.2,
                                    roof_thickness,
                                    north,
                                    wall_id,
                                );
                            }

                            if west_missing {
                                if ridge_z > z0 + eps && ridge_z < z1 - eps {
                                    self.push_edge_skirt(
                                        lot,
                                        right,
                                        forward,
                                        (sw.0, sw.1),
                                        (sw.0, ridge_z),
                                        sw.2,
                                        peak_y,
                                        roof_thickness,
                                        west,
                                        wall_id,
                                    );
                                    self.push_edge_skirt(
                                        lot,
                                        right,
                                        forward,
                                        (nw.0, ridge_z),
                                        (nw.0, nw.1),
                                        peak_y,
                                        nw.2,
                                        roof_thickness,
                                        west,
                                        wall_id,
                                    );
                                } else {
                                    self.push_edge_skirt(
                                        lot,
                                        right,
                                        forward,
                                        (sw.0, sw.1),
                                        (nw.0, nw.1),
                                        sw.2,
                                        nw.2,
                                        roof_thickness,
                                        west,
                                        wall_id,
                                    );
                                }
                            }

                            if east_missing {
                                if ridge_z > z0 + eps && ridge_z < z1 - eps {
                                    self.push_edge_skirt(
                                        lot,
                                        right,
                                        forward,
                                        (se.0, se.1),
                                        (se.0, ridge_z),
                                        se.2,
                                        peak_y,
                                        roof_thickness,
                                        east,
                                        wall_id,
                                    );
                                    self.push_edge_skirt(
                                        lot,
                                        right,
                                        forward,
                                        (ne.0, ridge_z),
                                        (ne.0, ne.1),
                                        peak_y,
                                        ne.2,
                                        roof_thickness,
                                        east,
                                        wall_id,
                                    );
                                } else {
                                    self.push_edge_skirt(
                                        lot,
                                        right,
                                        forward,
                                        (se.0, se.1),
                                        (ne.0, ne.1),
                                        se.2,
                                        ne.2,
                                        roof_thickness,
                                        east,
                                        wall_id,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RoofTile {
    tile_pos: TilePos,
    base_y: f32,
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
