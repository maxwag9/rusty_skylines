pub(crate) use crate::helpers::implementations::RevisionedSmallVec;
use crate::helpers::implementations::SerializableVec3;
use crate::helpers::modpack::ModManager;
use crate::helpers::positions::{ChunkCoord, WorldPos};
use crate::renderer::gizmo::gizmo::Gizmo;
use crate::renderer::props::{PropInstanceId, Props};
use crate::ui::input::Input;
use crate::ui::parser::Value;
use crate::ui::variables::Variables;
use crate::world::buildings::building_mesher::{
    BuildingMeshBuilder, BuildingMeshManager, DrivewayMaterial, RoofMaterial, WallMaterial,
};
use crate::world::buildings::building_renderer::BuildingRenderer;
use crate::world::buildings::lot_fitting::{
    BUILDING_SNAP_RADIUS, LotPoint, close_polygon, fit_lot_to_neighbors, gather_closest_lot_point,
};
use crate::world::buildings::utilities::utilities::UtilityUsage;
use crate::world::buildings::utilities::utilities_network::{
    EndpointType, UtilityEndpoint, UtilityEndpointId, UtilityNetwork,
};
use crate::world::buildings::zoning::{Lot, LotEntrance, LotId, Zoning, ZoningType};
use crate::world::camera::Camera;
use crate::world::cars::car_structs::{ChunkDistance, SimTime};
use crate::world::cars::partitions::{PartitionId, PartitionManager};
use crate::world::roads::road_mesh_manager::RoadMeshManager;
use crate::world::roads::road_structs::SegmentId;
use crate::world::roads::road_subsystem::Roads;
use crate::world::sound::sound::Sounds;
use crate::world::statisticals::demands::{JobOccupancy, ZoningDemand};
use crate::world::statisticals::demography::{Groups, LifeStage, Person, WORKHORSE_AGE_RANGE};
use crate::world::terrain::terrain_editing::{EditId, TerrainEditor};
use crate::world::terrain::terrain_subsystem::{Cursor, CursorMode, PreviewBuilding, Terrain};
use glam::{Vec2, Vec3};
use rand::RngExt;
use rand::rngs::ThreadRng;
use rand_distr::num_traits::Zero;
use rayon::iter::IntoParallelRefMutIterator;
use revision::revisioned;
use serde::Deserialize;
use smallvec::SmallVec;
use std::collections::{BTreeMap, HashMap};
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::mem::take;
use std::slice::{Iter, IterMut};
use tracing::error;
use wgpu_render_manager::renderer::RenderManager;

#[derive(Debug, Copy, Clone, Default, Hash, Deserialize)]
#[revisioned(revision = 1)]
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
    pub fn from_zoning_type(zoning_type: ZoningType) -> Self {
        match zoning_type {
            ZoningType::Residential => BuildingUsage::Residential,
            ZoningType::Commercial => BuildingUsage::Commercial,
            ZoningType::Industrial => BuildingUsage::Industrial,
            ZoningType::Office => BuildingUsage::Office,
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

pub type BuildingId = u32;

#[derive(Debug, Clone, Default, Deserialize)]
#[revisioned(revision = 1)]
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

#[derive(Debug, Clone, Default, Hash, Deserialize)]
#[revisioned(revision = 1)]
pub struct MiscBuildingParams {
    #[serde(default)]
    pub window_material_accent: WallMaterial,
    #[serde(default)]
    pub solar_modules: bool,
    #[serde(default)]
    pub antenna: bool,
    #[serde(default)]
    pub usage: BuildingUsage,
}
#[derive(Debug, Clone, Default, Hash, Deserialize)]
#[revisioned(revision = 1)]
pub struct BasementParams {}

#[derive(Debug, Clone, Default, Hash, Deserialize)]
#[revisioned(revision = 1)]
pub enum GardenLook {
    #[default]
    Normal,
    Overgrown,
}
#[derive(Debug, Clone, Default, Hash, Deserialize)]
#[revisioned(revision = 1)]
pub struct GardenParams {
    #[serde(default)]
    pub look: GardenLook,
}
#[derive(Debug, Clone, Default, Deserialize)]
#[revisioned(revision = 1)]
pub struct GarageParams {
    #[serde(default)]
    pub story_height: f32,
    #[serde(default)]
    pub num_stories: u16,
}
impl Hash for GarageParams {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.story_height.to_bits().hash(state);
        self.num_stories.hash(state);
    }
}
#[derive(Debug, Clone, Default, Deserialize)]
#[revisioned(revision = 1)]
pub struct BuildingParams {
    pub roof: RoofType,
    #[serde(default)]
    pub roof_material: RoofMaterial,
    #[serde(default)]
    pub wall_material: WallMaterial,
    #[serde(default)]
    pub driveway_material: DrivewayMaterial,
    #[serde(default)]
    pub story_height: f32,
    #[serde(default)]
    pub num_stories: u16,
    #[serde(default)]
    pub basement: BasementParams,
    #[serde(default)]
    pub garden: GardenParams,
    #[serde(default)]
    pub garage: Option<GarageParams>,
    #[serde(default)]
    pub miscellaneous: MiscBuildingParams,
    #[serde(default)]
    pub utilities: HashMap<String, UtilityUsage>,
    #[serde(default)]
    pub utilities_scale_per_square_meter: bool,
}
impl BuildingParams {
    pub fn apply_changes(&mut self, changes: BuildingParamsChanges) {
        if let Some(v) = changes.roof {
            self.roof = v;
        }

        if let Some(v) = changes.roof_material {
            self.roof_material = v;
        }

        if let Some(v) = changes.wall_material {
            self.wall_material = v;
        }

        if let Some(v) = changes.driveway_material {
            self.driveway_material = v;
        }

        if let Some(v) = changes.story_height {
            self.story_height = v.raw();
        }

        if let Some(v) = changes.num_stories {
            self.num_stories = v;
        }

        if let Some(v) = changes.basement {
            self.basement = v;
        }

        if let Some(v) = changes.garden {
            self.garden = v;
        }

        if let Some(v) = changes.garage {
            self.garage = v;
        }

        if let Some(v) = changes.miscellaneous {
            self.miscellaneous = v;
        }
        if let Some(v) = changes.utilities {
            self.utilities = v.into_iter().collect();
        }
    }

    pub fn with_changes(&self, changes: BuildingParamsChanges) -> Self {
        let mut result = self.clone();
        result.apply_changes(changes);
        result
    }

    pub fn utility_usages(
        &self,
        utility_network: &UtilityNetwork,
        one_story_area: f64,
    ) -> Vec<UtilityUsage> {
        let mut usages = Vec::with_capacity(utility_network.utilities.len());

        for utility in &utility_network.utilities {
            let mut usage = self
                .utilities
                .get(&utility.name)
                .copied()
                .unwrap_or_default();

            if self.utilities_scale_per_square_meter {
                usage.consumption *= one_story_area as f32;
                usage.production *= one_story_area as f32;
            }

            usages.push(usage);
        }

        usages
    }
    //{
    //
    //             }; THESE ARE HOLY! IF THESE ARE REMOVED; MY GAME WILL FLOP!!!! Shut up, Astrology teacher!
}
impl BuildingParams {
    pub fn max_people(&self, one_story_area: f64) -> u32 {
        let total_area = one_story_area * self.num_stories as f64;

        let area_per_person = match self.miscellaneous.usage {
            BuildingUsage::Residential => 40.0,
            BuildingUsage::Commercial => 20.0,
            BuildingUsage::Industrial => 20.0,
            BuildingUsage::Office => 30.0,
        };

        (total_area / area_per_person) as u32
    }
}
impl Hash for BuildingParams {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.roof.hash(state);
        self.roof_material.hash(state);
        self.wall_material.hash(state);
        self.driveway_material.hash(state);
        self.story_height.to_bits().hash(state); // <- important
        self.basement.hash(state);
        self.garden.hash(state);
        self.garage.hash(state);
        self.miscellaneous.hash(state);
    }
}

#[derive(Clone, Default, Hash)]
pub struct BuildingParamsLevels {
    pub level0: BuildingParams,
    pub changes: BTreeMap<u8, BuildingParamsChanges>,
    cache: [BuildingParams; 6],
}

#[derive(Deserialize)]
struct BuildingParamsLevelsDeserialize {
    level0: BuildingParams,
    changes: BTreeMap<u8, BuildingParamsChanges>,
}

impl<'de> Deserialize<'de> for BuildingParamsLevels {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let data = BuildingParamsLevelsDeserialize::deserialize(deserializer)?;

        let mut result = Self {
            level0: data.level0,
            changes: data.changes,
            cache: Default::default(),
        };

        result.build_cache();

        Ok(result)
    }
}
impl BuildingParamsLevels {
    pub fn build_cache(&mut self) {
        self.cache[0] = self.level0.clone();

        for level in 1..=5 {
            self.cache[level] = self.cache[level - 1].clone();

            if let Some(changes) = self.changes.get(&(level as u8)) {
                self.cache[level].apply_changes(changes.clone());
            }
        }
    }
    pub fn get_level(&self, level: u8) -> &BuildingParams {
        &self.cache[level as usize]
    }
}
#[derive(Clone, Default, Hash, Deserialize)]
pub struct BuildingParamsChanges {
    pub roof: Option<RoofType>,
    pub roof_material: Option<RoofMaterial>,
    pub wall_material: Option<WallMaterial>,
    pub driveway_material: Option<DrivewayMaterial>,
    pub story_height: Option<HashableF32>,
    pub num_stories: Option<u16>,
    pub basement: Option<BasementParams>,
    pub garden: Option<GardenParams>,
    pub garage: Option<Option<GarageParams>>,
    pub miscellaneous: Option<MiscBuildingParams>,
    pub utilities: Option<BTreeMap<String, UtilityUsage>>,
}
#[derive(Clone, Default, Deserialize)]
pub struct HashableF32(pub f32);
impl Hash for HashableF32 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}
impl HashableF32 {
    pub fn raw(&self) -> f32 {
        self.0
    }
}
#[derive(Debug, Copy, Clone, Default, Hash)]
#[revisioned(revision = 1)]
pub enum BuildingLevel {
    #[default]
    Level0,
    Level1,
    Level2,
    Level3,
    Level4,
    Level5,
}
impl BuildingLevel {
    pub fn to_u8(self) -> u8 {
        match self {
            BuildingLevel::Level0 => 0,
            BuildingLevel::Level1 => 1,
            BuildingLevel::Level2 => 2,
            BuildingLevel::Level3 => 3,
            BuildingLevel::Level4 => 4,
            BuildingLevel::Level5 => 5,
        }
    }
}
#[derive(Clone, Debug)]
pub enum BuildingComplaint {
    NotEnoughWorkers,
    NotEnoughCustomers,
    NotEnoughJobs,
}

#[derive(Clone, Debug, Default)]
#[revisioned(revision = 1)]
pub struct BuildingOccupancy {
    pub workers: u32,
    pub residential_capacity: u32,
    pub jobs_capacity: u32,
    pub employed_tenants: u32,
    pub groups: Groups,
}

impl BuildingOccupancy {
    pub fn fill_rate_workplace(&self) -> f32 {
        if self.jobs_capacity == 0 {
            return 1.0;
        }
        (self.workers / self.jobs_capacity) as f32
    }
    pub fn fill_rate_residential(&self) -> f32 {
        if self.residential_capacity == 0 {
            return 0.0;
        }
        (self.groups.whole_population() / self.residential_capacity) as f32
    }
    pub fn fill_rate(&self, zoning_type: ZoningType) -> f32 {
        if zoning_type.is_workplace() {
            if self.jobs_capacity == 0 {
                return 0.0;
            }
            (self.workers / self.jobs_capacity) as f32
        } else {
            if self.residential_capacity == 0 {
                return 0.0;
            }
            (self.groups.whole_population() / self.residential_capacity) as f32
        }
    }
    pub fn employment(&self) -> f32 {
        let workhorses = self.groups.workhorse_population();
        if workhorses.is_zero() {
            // I am interesting
            f32::zero()
        } else {
            self.employed_tenants as f32 / workhorses as f32
        }
    }
    pub fn unemployed(&self) -> u32 {
        let workhorses = self.groups.workhorse_population();
        if workhorses.is_zero() {
            u32::zero()
        } else {
            self.employed_tenants / workhorses
        }
    }
    pub fn random_employable_age(&self) -> usize {
        let mut rng = ThreadRng::default();

        let total = self.groups.workhorse_population();

        if total == 0 {
            return *WORKHORSE_AGE_RANGE.start();
        }

        let mut roll = rng.random_range(0..total);

        for age in WORKHORSE_AGE_RANGE {
            let count = self.groups.get_age(age);
            if roll < count {
                return age;
            }
            roll -= count;
        }

        *WORKHORSE_AGE_RANGE.end()
    }
    pub fn add_worker(&mut self) {}
    // commercial_attractiveness < 0  →  more capacity than workers, so footfall is low too
    // residential_attractiveness < 0 →  job_pull < housing_deficit, people are leaving
    pub fn complaint(
        &self,
        zoning_type: ZoningType,
        demand: &ZoningDemand,
        job_occupancy: &JobOccupancy,
    ) -> Option<BuildingComplaint> {
        match zoning_type {
            ZoningType::Residential => {
                if self.employment() < 0.7 && self.fill_rate_residential() < 0.4 {
                    //  TODO
                    return Some(BuildingComplaint::NotEnoughJobs);
                }
                None
            }
            ZoningType::Commercial => {
                if self.jobs_capacity == 0 {
                    return None;
                }
                if self.fill_rate_workplace() < 0.4 {
                    if demand.commercial_attractiveness < -0.2 {
                        return Some(BuildingComplaint::NotEnoughCustomers);
                    }
                    return Some(BuildingComplaint::NotEnoughWorkers);
                }
                None
            }
            ZoningType::Industrial | ZoningType::Office => {
                if self.jobs_capacity == 0 {
                    return None;
                }
                if self.fill_rate_workplace() < 0.3 {
                    return Some(BuildingComplaint::NotEnoughWorkers);
                }
                None
            }
        }
    }
}

#[derive(Debug, Clone, Hash)]
#[revisioned(revision = 1)]
pub enum BuildingDesignSource {
    BuildingParams {
        levels: RevisionedSmallVec<BuildingParams, 6>,
    },
    Design(String),
}
impl Default for BuildingDesignSource {
    fn default() -> Self {
        BuildingDesignSource::BuildingParams {
            levels: RevisionedSmallVec(SmallVec::from_vec(vec![BuildingParams::default(); 6])),
        }
    }
}
#[revisioned(revision = 1)]
#[derive(Debug, Clone, Default)]
pub struct BuildingMisc {
    pub is_preview: bool,
    pub mesh: Option<BuildingMeshBuilder>,
}
#[revisioned(revision = 1)]
#[derive(Debug, Clone, Default)]
pub struct Building {
    pub id: BuildingId,
    pub pos: WorldPos,
    pub segment_id: Option<SegmentId>,
    pub lot_id: LotId,
    pub level: BuildingLevel,

    pub design_source: BuildingDesignSource,
    pub edit_id: Option<EditId>,
    pub prop_instance_ids: Vec<PropInstanceId>,
    pub utility_endpoint_id: Option<UtilityEndpointId>,
    pub occupancy: BuildingOccupancy,
    pub misc: BuildingMisc,
}
impl Building {
    pub fn current_level_params<'a>(
        &'a self,
        buildings_catalog: &'a BuildingsCatalog,
    ) -> Option<&'a BuildingParams> {
        let idx = self.level.to_u8() as usize;
        match &self.design_source {
            BuildingDesignSource::BuildingParams { levels } => levels.0.get(idx),
            BuildingDesignSource::Design(design_name) => {
                let bd = buildings_catalog.get_design(design_name.as_str(), self.level);
                //println!("{} {:#?}", design_name, bd);
                bd
            }
        }
    }
}

#[derive(Clone, Default)]
pub struct Buildings {
    pub storage: BuildingStorage,
    pub partitions: PartitionManager,
    pub catalog: BuildingsCatalog,
    pub utilities: UtilityNetwork,
}

impl Buildings {
    pub fn start_preview(
        &mut self,
        cursor: &mut Cursor,
        building_name: String,
        pos: WorldPos,
        zoning: &mut Zoning,
        variables: &mut Variables,
        roads: &Roads,
        road_mesh_manager: &RoadMeshManager,
    ) {
        let snap = gather_closest_lot_point(roads, road_mesh_manager, pos)
            .filter(|point| point.dist <= BUILDING_SNAP_RADIUS);

        let fit = snap.as_ref().and_then(|snap| {
            fit_lot_to_neighbors(&zoning.zoning_storage, road_mesh_manager, snap, None)
        });

        let (bounds, center, entrance, dir, segment_id) = if let Some(fit) = fit {
            let center = WorldPos::centroid(&fit.bounds[..fit.bounds.len() - 1]);
            let entrance = LotEntrance::new(fit.snap_point.pos, fit.snap_point.lateral);
            let dir = fit.snap_point.tangent;

            (
                fit.bounds,
                center,
                entrance,
                dir,
                Some(fit.snap_point.segment_id),
            )
        } else {
            let center = pos;
            let half_size = 30.0;

            let mut corners = vec![
                center.add_vec3(Vec3::new(-half_size, 0.0, -half_size)),
                center.add_vec3(Vec3::new(half_size, 0.0, -half_size)),
                center.add_vec3(Vec3::new(half_size, 0.0, half_size)),
                center.add_vec3(Vec3::new(-half_size, 0.0, half_size)),
            ];

            close_polygon(&mut corners);

            let entrance = LotEntrance {
                pos: center.add_vec3(Vec3::new(0.0, 0.0, -half_size)),
                dir: SerializableVec3::from_vec3(Vec3::new(1.0, 0.0, 0.0)),
            };

            (corners, center, entrance, Vec3::new(1.0, 0.0, 0.0), None)
        };

        let lot = Lot {
            id: 696,
            bounds,
            bounds_version: 0,
            center,
            entrance: entrance.clone(),
            layout: None,
            zoning_type: Some(ZoningType::Commercial),
            segment_id: Default::default(),
            district_id: 0,
            building_id: None,
            land_value: 0.0,
        };
        let lot_id = zoning.zoning_storage.spawn_lot(lot);
        let building = Building {
            id: 0,
            pos: center,
            segment_id: Default::default(),
            lot_id,
            level: Default::default(),
            design_source: BuildingDesignSource::Design(building_name.clone()),
            edit_id: None,
            prop_instance_ids: vec![],
            utility_endpoint_id: None,
            occupancy: Default::default(),
            misc: BuildingMisc {
                is_preview: true,
                mesh: None,
            },
        };
        let building_id = BuildingStorage::spawn(self, zoning, building);
        let lot = zoning.zoning_storage.get_mut_lot(lot_id).unwrap();
        lot.building_id = Some(building_id);

        let preview = PreviewBuilding {
            name: building_name,
            cached_model: None,
            new_pos: center,
            new_dir: dir,
            lot_id,
            building_id,
        };
        cursor.preview_building = Some(preview);
        cursor.mode = CursorMode::Buildings;
        variables.set_string("cursor_mode", "Buildings");
    }

    pub fn cancel_preview(
        &mut self,
        cursor: &mut Cursor,
        zoning: &mut Zoning,
        terrain_editor: &mut TerrainEditor,
        props: &mut Props,
        variables: &mut Variables,
    ) {
        if let Some(pb) = cursor.preview_building.take() {
            zoning.zoning_storage.despawn_lot(pb.lot_id);
            BuildingStorage::despawn(self, zoning, terrain_editor, props, pb.building_id);
            cursor.mode = CursorMode::None;
            variables.set_string("cursor_mode", "None");
        }
    }
    pub fn update_preview(
        &mut self,
        pb: &mut PreviewBuilding,
        pos: WorldPos,
        zoning: &mut Zoning,
        road_mesh_manager: &RoadMeshManager,
        snap: Option<LotPoint>,
    ) {
        let fit = snap.as_ref().and_then(|snap| {
            fit_lot_to_neighbors(
                &zoning.zoning_storage,
                road_mesh_manager,
                snap,
                Some(pb.lot_id),
            )
        });

        if let Some(fit) = fit {
            let center = WorldPos::centroid(&fit.bounds[..fit.bounds.len() - 1]);

            zoning.zoning_storage.update_lot_geometry(
                pb.lot_id,
                LotEntrance::new(fit.snap_point.pos, fit.snap_point.lateral),
                fit.bounds,
                center,
            );

            pb.new_dir = fit.snap_point.tangent;

            if let Some(building) = self.storage.get_mut(pb.building_id) {
                building.pos = center;
            }

            pb.new_pos = center;
            return;
        }

        let center = pos;
        let half_size = 30.0;

        const BASE_FORWARD: Vec3 = Vec3::new(0.0, 0.0, -1.0);

        let dir_xz = Vec2::new(pb.new_dir.x, pb.new_dir.z);

        let (sin_a, cos_a) = if dir_xz.length_squared() < 1e-10 {
            (0.0f32, 1.0f32)
        } else {
            let base_angle = BASE_FORWARD.z.atan2(BASE_FORWARD.x);
            let target_angle = dir_xz.y.atan2(dir_xz.x);
            (target_angle - base_angle).sin_cos()
        };

        let rotate_xz = |v: Vec3| -> Vec3 {
            Vec3::new(v.x * cos_a - v.z * sin_a, v.y, v.x * sin_a + v.z * cos_a)
        };

        let mut corners = vec![
            center.add_vec3(rotate_xz(Vec3::new(-half_size, 0.0, -half_size))),
            center.add_vec3(rotate_xz(Vec3::new(half_size, 0.0, -half_size))),
            center.add_vec3(rotate_xz(Vec3::new(half_size, 0.0, half_size))),
            center.add_vec3(rotate_xz(Vec3::new(-half_size, 0.0, half_size))),
        ];

        close_polygon(&mut corners);

        let entrance = LotEntrance {
            pos: center.add_vec3(rotate_xz(Vec3::new(0.0, 0.0, -half_size))),
            dir: SerializableVec3::from_vec3(rotate_xz(Vec3::new(1.0, 0.0, 0.0))),
        };

        zoning
            .zoning_storage
            .update_lot_geometry(pb.lot_id, entrance, corners, center);

        if let Some(building) = self.storage.get_mut(pb.building_id) {
            building.pos = center;
        }

        pb.new_pos = center;
    }
    pub fn finish_preview(
        &mut self,
        cursor: &mut Cursor,
        zoning: &mut Zoning,
        terrain_editor: &mut TerrainEditor,
        props: &mut Props,
        variables: &mut Variables,
        sounds: &mut Sounds,
        building_mesh_manager: &mut BuildingMeshManager,
    ) {
        if let Some(mut pb) = cursor.preview_building.take() {
            if let Some(building) = self.storage.get_mut(pb.building_id) {
                building.misc.is_preview = false;
                if let Some(model) = pb.cached_model.take() {
                    building.misc.mesh = Some(model.mesh);
                }
                building_mesh_manager.invalidate_chunk(building.pos.chunk);
            }
        }
        cursor.mode = CursorMode::None;
        variables.set_string("cursor_mode", "None");
        sounds.queue_sfx("PlaceBuilding");
    }
}

impl Buildings {
    pub fn new(mod_manager: &ModManager) -> Buildings {
        Self {
            storage: BuildingStorage::new(),
            partitions: PartitionManager::new(),
            catalog: BuildingsCatalog::new(mod_manager),
            utilities: UtilityNetwork::new(mod_manager),
        }
    }

    pub fn update(
        &mut self,
        camera: &Camera,
        terrain: &mut Terrain,
        roads: &mut Roads,
        road_mesh_manager: &RoadMeshManager,
        input: &mut Input,
        gizmo: &mut Gizmo,
        variables: &mut Variables,
        building_renderer: &mut BuildingRenderer,
        render_manager: &mut RenderManager,
        mod_manager: &ModManager,
        props: &mut Props,
        zoning: &mut Zoning,
        sounds: &mut Sounds,
    ) {
        if terrain.cursor.mode != CursorMode::Buildings {
            self.cancel_preview(
                &mut terrain.cursor,
                zoning,
                &mut terrain.terrain_editor,
                props,
                variables,
            );
            return;
        }

        if input.action_pressed_once("Cancel Building") {
            self.cancel_preview(
                &mut terrain.cursor,
                zoning,
                &mut terrain.terrain_editor,
                props,
                variables,
            );
            return;
        }

        let place_requested = input.action_pressed_once("Place Building");

        if let Some(mut pb) = terrain.cursor.preview_building.take() {
            let pos = terrain.last_picked.as_ref().map(|p| p.pos);

            let snap = pos.and_then(|pos| {
                let point = gather_closest_lot_point(roads, road_mesh_manager, pos)?;

                (point.dist <= BUILDING_SNAP_RADIUS).then_some(point)
            });

            self.rotate_preview(input, &mut pb, snap.is_some());

            let zoning_snap_point = snap.clone();

            if let Some(pos) = pos {
                self.update_preview(&mut pb, pos, zoning, road_mesh_manager, snap);
            }

            let preview_valid = if let Some(picked) = terrain.last_picked.clone() {
                zoning.run_lot_zoning(
                    terrain,
                    self,
                    roads,
                    road_mesh_manager,
                    input,
                    variables,
                    zoning_snap_point,
                    &picked,
                    ZoningType::Commercial,
                    props,
                    gizmo,
                    Some(pb.lot_id),
                )
            } else {
                false
            };

            building_renderer.render_preview(
                render_manager,
                mod_manager,
                terrain,
                props,
                &mut roads.parking,
                self,
                zoning,
                gizmo,
                &mut pb,
            );

            terrain.cursor.preview_building = Some(pb);

            if place_requested && preview_valid {
                self.finish_preview(
                    &mut terrain.cursor,
                    zoning,
                    &mut terrain.terrain_editor,
                    props,
                    variables,
                    sounds,
                    &mut building_renderer.mesh_manager,
                );
            }
        }
    }

    fn rotate_preview(&self, input: &mut Input, pb: &mut PreviewBuilding, snapped: bool) {
        if snapped {
            return;
        }

        const SENSITIVITY: f32 = 0.02;
        let delta_angle = if input.mouse.buttons.right.pressed {
            -input.mouse.delta.x * SENSITIVITY
        } else if input.action_repeat("Rotate building +10°") {
            10.0_f32.to_radians()
        } else if input.action_repeat("Rotate building -10°") {
            -10.0_f32.to_radians()
        } else {
            return;
        };

        if delta_angle == 0.0 {
            return;
        }

        let (sin_a, cos_a) = delta_angle.sin_cos();

        let new_x = pb.new_dir.x * cos_a - pb.new_dir.z * sin_a;
        let new_z = pb.new_dir.x * sin_a + pb.new_dir.z * cos_a;

        pb.new_dir = Vec3::new(new_x, pb.new_dir.y, new_z).normalize_or_zero();
    }
}

#[derive(Clone, Default)]
#[revisioned(revision = 1)]
pub struct BuildingStorage {
    pub building_chunk_storage: BuildingChunkStorage,
    buildings: Vec<Option<Building>>,
    free_list: Vec<BuildingId>,
    // Reverse mapping so I can remove from chunks in O(1) when destroying
    building_locations: HashMap<BuildingId, ChunkCoord>,
    center_chunk: ChunkCoord,
    building_to_partition: Vec<Option<PartitionId>>,
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
        Self {
            building_chunk_storage: BuildingChunkStorage::new(),
            buildings: Vec::new(),
            free_list: Vec::new(),
            building_locations: HashMap::new(),
            center_chunk: ChunkCoord::zero(),
            building_to_partition: Vec::new(),
        }
    }

    pub fn spawn(
        buildings: &mut Buildings,
        zoning: &mut Zoning,
        mut building: Building,
    ) -> BuildingId {
        let storage = &mut buildings.storage;
        let entrance_pos = zoning
            .zoning_storage
            .get_lot(building.lot_id)
            .map_or(building.pos, |l| l.entrance.pos);
        let chunk_coord = entrance_pos.chunk;

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
            ChunkDistance::from_chunk_positions(storage.center_chunk, chunk_coord),
            building_id,
        );
        storage.building_locations.insert(building_id, chunk_coord);
        let partition_id = PartitionManager::add_building(buildings, building_id, entrance_pos);
        let storage = &mut buildings.storage;
        storage.set_partition_of_building(building_id, partition_id);

        //println!("Created building: {}", building_id);
        building_id
    }

    pub fn despawn<I>(
        buildings: &mut Buildings,
        zoning: &mut Zoning,
        terrain_editor: &mut TerrainEditor,
        props: &mut Props,
        id: I,
    ) where
        I: Into<Option<BuildingId>>,
    {
        let Some(id) = id.into() else {
            return;
        };

        // Check building exists before proceeding
        if buildings
            .storage
            .buildings
            .get(id as usize)
            .and_then(|opt| opt.as_ref())
            .is_none()
        {
            return;
        }

        // Remove from chunk first using reverse lookup
        if let Some(chunk_coord) = buildings.storage.building_locations.remove(&id) {
            buildings
                .storage
                .building_chunk_storage
                .remove_building(chunk_coord, id);
        }
        if let Some(edit_id) = buildings.storage.buildings[id as usize]
            .as_ref()
            .and_then(|b| b.edit_id)
        {
            terrain_editor.remove_edit(edit_id);
        }
        if let Some(prop_ids) = buildings.storage.buildings[id as usize]
            .as_mut()
            .map(|b| take(&mut b.prop_instance_ids))
        {
            for prop_id in prop_ids {
                props.remove_instance(prop_id);
            }
        }

        PartitionManager::remove_building(buildings, id);

        // Actually free the slot
        let storage = &mut buildings.storage;
        storage.free_list.push(id);
        let Some(building) = storage.buildings[id as usize].take() else {
            return;
        };
        if let Some(endpoint_id) = building.utility_endpoint_id {
            buildings.utilities.remove_endpoint(endpoint_id);
        }
        // Remove jobs
        let Some(district) = zoning
            .zoning_storage
            .get_lot(building.lot_id)
            .and_then(|lot| zoning.zoning_storage.get_district(lot.district_id))
        else {
            return;
        };

        let mut rng = ThreadRng::default();
        for _ in 0..building.occupancy.employed_tenants {
            let age = building.occupancy.random_employable_age();
            let person = Person {
                education_level: district
                    .zoning_demand
                    .demography
                    .education
                    .get_citizen_education(LifeStage::from_int(age)),
                age: age as u8,
            };

            let Some(workplace_id) = zoning.zoning_storage.get_work_place(
                building.pos,
                &buildings.storage,
                person,
                &mut rng,
            ) else {
                continue; // Continue or break?
            };

            if let Some(workplace) = buildings.storage.get_mut(workplace_id) {
                workplace.occupancy.workers -= 1;
            }
        }
    }
    pub fn update_utility_endpoint(
        buildings: &mut Buildings,
        zoning: &Zoning,
        building_id: BuildingId,
    ) {
        let Some(building) = buildings.storage.get_mut(building_id) else {
            return;
        };
        let Some(lot) = zoning.zoning_storage.get_lot(building.lot_id) else {
            return;
        };
        let Some(segment_id) = lot.segment_id else {
            return;
        };
        let Some(building_params) = building.current_level_params(&buildings.catalog) else {
            return;
        };
        let utility_usages =
            building_params.utility_usages(&buildings.utilities, lot.floor_area_or_zero());

        let endpoint = UtilityEndpoint {
            endpoint_type: EndpointType::Building {
                building_id,
                segment_id,
            },
            utility_usages,
        };
        if let Some(endpoint_id) = building.utility_endpoint_id {
            buildings.utilities.update_endpoint(endpoint_id, endpoint);
        } else {
            buildings.utilities.add_endpoint(endpoint);
        }
    }
    pub fn building_count(&self) -> usize {
        self.buildings.len() - self.free_list.len()
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

    #[inline(always)]
    pub fn get_partition_of_building(&self, building: BuildingId) -> Option<PartitionId> {
        unsafe {
            *self.building_to_partition.get_unchecked(building as usize) // Hope-based unsafe usage, first ever unsafe usage.
            // Should have better performance because no bound checking is being done, good for car signfinding!
        }
    }

    #[inline]
    fn ensure_partition_slot(&mut self, building_id: BuildingId) {
        let idx = building_id as usize;
        if self.building_to_partition.len() <= idx {
            self.building_to_partition.resize(idx + 1, None);
        }
    }

    #[inline]
    pub fn set_partition_of_building(
        &mut self,
        building_id: BuildingId,
        partition_id: PartitionId,
    ) {
        self.ensure_partition_slot(building_id);
        self.building_to_partition[building_id as usize] = Some(partition_id);
    }
    #[inline]
    pub fn clear_partition_of_building(&mut self, building_id: BuildingId) {
        if let Some(slot) = self.building_to_partition.get_mut(building_id as usize) {
            *slot = None;
        }
    }
}

#[derive(Clone, Default)]
#[revisioned(revision = 1)]
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
#[derive(Clone, Default)]
#[revisioned(revision = 1)]
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
            .map(|(&coord, chunk)| {
                let is_empty = chunk.building_ids.is_empty();
                let dist2 = center_chunk.dist2(coord);
                let new_dist = ChunkDistance::from_dist2(dist2);
                (coord, new_dist, is_empty)
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

impl Hash for Building {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if self.misc.is_preview {
            return;
        }
        self.id.hash(state);
        self.pos.hash(state);
        self.segment_id.hash(state);
        self.lot_id.hash(state);
        self.level.hash(state);
        self.design_source.hash(state);
        // self.edit_id.hash(state);
        // self.prop_instance_ids.hash(state);
        // I DO NOT hash occupancy cuz I update it all the time!
    }
}
#[derive(Clone, Default, Deserialize)]
pub struct PloppableBuilding {
    pub name: String,
    pub branch: String,
    pub tab: String,
    pub building_levels: BuildingParamsLevels,
}
#[derive(Clone, Default)]
pub struct BuildingsCatalog {
    pub ploppables: HashMap<String, PloppableBuilding>,
}

impl BuildingsCatalog {
    pub fn new(mod_manager: &ModManager) -> Self {
        let mut catalog = Self {
            ploppables: HashMap::new(),
        };
        catalog.load(mod_manager);
        catalog
    }
    fn load(&mut self, mod_manager: &ModManager) {
        let mut ploppables = HashMap::new();

        let paths = mod_manager.building_paths(); // I TRUST you will give me only Files and they are Yaml.!!!!!

        for path in paths {
            let name = match path.file_stem().and_then(|stem| stem.to_str()) {
                Some(name) => name.to_owned(),
                None => {
                    error!(
                        "[Buildings] Failed to determine Building name from file '{}'. Filename is not valid UTF-8.",
                        path.display()
                    );
                    continue;
                }
            };

            let contents = match std::fs::read_to_string(&path) {
                Ok(contents) => contents,
                Err(err) => {
                    error!(
                        "[Buildings] Failed to read Building file '{}'. Error: {}",
                        path.display(),
                        err
                    );
                    continue;
                }
            };

            let building = match serde_yaml::from_str::<PloppableBuilding>(&contents) {
                Ok(building) => building,
                Err(err) => {
                    error!(
                        "[Buildings] Failed to deserialize Building file '{}'. Expected a valid PloppableBuilding YAML. Error: {}",
                        path.display(),
                        err
                    );
                    continue;
                }
            };

            let building_name = building.name.clone();

            if ploppables.insert(building_name.clone(), building).is_some() {
                tracing::warn!(
                    "[Buildings] Warning: Duplicate Building name '{}'. The Building from file '{}' overwrote the previously loaded Building with the same name.",
                    building_name,
                    path.display()
                );
            }
        }
        println!("[Buildings] Loaded {} Building files", ploppables.len());
        //println!("{:?}", ploppables.keys());
        self.ploppables = ploppables;
    }
    pub fn get_design(&self, design_name: &str, level: BuildingLevel) -> Option<&BuildingParams> {
        self.ploppables
            .get(design_name)
            .map(|pb| pb.building_levels.get_level(level.to_u8()))
    }
}
