use crate::helpers::paths::saves_dir;
use crate::helpers::positions::{ChunkCoord, ChunkSize, WorldPos};
use crate::renderer::props::{Props, SavePropChunk};
use crate::world::buildings::buildings::{BuildingStorage, Buildings};
use crate::world::buildings::zoning::ZoningStorage;
use crate::world::camera::{Camera, CameraController};
use crate::world::cars::partitions::PartitionManager;
use crate::world::roads::road_subsystem::Roads;
use crate::world::roads::roads::{RoadStorage, RoadTypes};
use crate::world::terrain::terrain_editing::PersistedChunk;
use crate::world::terrain::terrain_subsystem::Terrain;
use sanitize_filename::sanitize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Error, Write};
use std::path::PathBuf;
use std::time::SystemTime;
use std::{fs, mem};
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter, EnumString};

#[derive(Debug)]
pub enum LoadResult {
    Success(SaveVersion),
    FileNotFound(PathBuf),
    WrongExtension(PathBuf),
    PathError(Error),
    FileNonExistent(String),
    CantGetExtension,
    CantGetData(Error),
    CantDecompress(Error),
    CantDecodeData(postcard::Error),
}
#[derive(Debug)]
pub enum SaveResult {
    Success,
    NotAFile,

    CantWriteFile(Error),
    CantCreateDir(Error),
    CantCompress(Error),
    CantEncodeData(postcard::Error),
    CantGetExtension(String),
    WrongExtension(String),
    EmptySaveName,
    DowngradeError(String),
}

const SAVE_MAGIC: &str = "RSS1";

fn sanitize_header_value(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\r' | '\n' => ' ',
            _ => c,
        })
        .collect()
}

fn build_save_header(save: &SaveState) -> String {
    format!(
        "{magic}\n\
name={name}\n\
version={version}\n\
timestamp_unix={timestamp}\n\
chunk_size={chunk_size:?}\n\
compression=zstd\n\
serialization=postcard\n\
payload=compressed_binary\n\
\n",
        magic = SAVE_MAGIC,
        name = sanitize_header_value(&save.name),
        version = sanitize_header_value(&save.version.to_string()),
        timestamp = save.timestamp_unix,
        chunk_size = save.chunk_size,
    )
}

fn find_header_end(data: &[u8]) -> Option<usize> {
    if let Some(pos) = data.windows(4).position(|w| w == b"\r\n\r\n") {
        return Some(pos + 4);
    }
    if let Some(pos) = data.windows(2).position(|w| w == b"\n\n") {
        return Some(pos + 2);
    }
    None
}
pub struct GameState {
    pub current_save: SaveState,
}
impl GameState {
    pub fn new() -> Self {
        Self {
            current_save: SaveState::default(),
        }
    }

    pub fn load(
        &mut self,
        save_name: &str,
        camera: &mut Camera,
        camera_controller: &mut CameraController,
        roads: &mut Roads,
        terrain: &mut Terrain,
        props: &mut Props,
        buildings: &mut Buildings,
    ) -> LoadResult {
        let safe_name = sanitize(save_name);
        let path = saves_dir().join(format!("{}.rss", safe_name));
        let detected_version: Option<SaveVersion>;
        match path.try_exists() {
            Ok(true) => {
                if let Some(ext) = path.extension() {
                    if ext != "rss" {
                        return LoadResult::WrongExtension(path);
                    }
                } else {
                    return LoadResult::CantGetExtension;
                }

                let data = match fs::read(&path) {
                    Ok(d) => d,
                    Err(e) => return LoadResult::CantGetData(e),
                };

                let (header_bytes, payload) = if let Some(end) = find_header_end(&data) {
                    (&data[..end], &data[end..])
                } else {
                    (b"" as &[u8], &data[..])
                };

                fn parse_version_from_header(header: &str) -> Option<SaveVersion> {
                    header
                        .lines()
                        .find_map(|l| l.strip_prefix("version="))
                        .and_then(|v| v.trim().parse().ok())
                }
                detected_version =
                    parse_version_from_header(&String::from_utf8_lossy(header_bytes));

                let decompressed = match zstd::decode_all(payload) {
                    Ok(d) => d,
                    Err(e) => return LoadResult::CantDecompress(e),
                };
                let save_state = match detected_version {
                    // Old format: needs migration
                    Some(SaveVersion::Alpha1_6_16a) => {
                        match postcard::from_bytes::<SaveStateAlpha1_6_16a>(&decompressed) {
                            Ok(old) => old.upgrade_to_latest(),
                            Err(e) => return LoadResult::CantDecodeData(e),
                        }
                    }
                    // Current format or unknown: deserialize directly as SaveState
                    Some(SaveVersion::Alpha1_7_2a) | None => {
                        match postcard::from_bytes::<SaveState>(&decompressed) {
                            Ok(s) => s,
                            Err(e) => return LoadResult::CantDecodeData(e),
                        }
                    }
                };

                self.current_save = save_state;
                self.current_save.name = save_name.to_string();
            }
            Ok(false) => {
                self.current_save = SaveState::default();
                self.current_save.name = save_name.to_string();
                return LoadResult::FileNonExistent(
                    path.to_str().unwrap_or("unknown path").to_string(),
                );
            }
            Err(e) => return LoadResult::PathError(e),
        }

        self.current_save
            .load(camera, camera_controller, roads, terrain, props, buildings);
        LoadResult::Success(detected_version.unwrap_or(SaveVersion::current()))
    }

    pub fn save_as_version(
        &mut self,
        camera: &Camera,
        roads: &Roads,
        terrain: &Terrain,
        props: &Props,
        buildings: &Buildings,
        target_version: Option<SaveVersion>,
    ) -> SaveResult {
        let safe_name = sanitize(&self.current_save.name);
        if safe_name.is_empty() {
            return SaveResult::EmptySaveName;
        }

        let path = saves_dir().join(format!("{}.rss", safe_name));
        if path.is_dir() {
            return SaveResult::NotAFile;
        }

        if let Some(ext) = path.extension() {
            if ext != "rss" {
                return SaveResult::WrongExtension(ext.to_str().unwrap_or("unknown").to_string());
            }
        } else {
            return SaveResult::CantGetExtension(
                path.to_str().unwrap_or("unknown path").to_string(),
            );
        }

        self.current_save
            .save(camera, roads, terrain, props, buildings);

        let serialized = match target_version {
            Some(ver) => match SaveStateVersioned::from_latest(self.current_save.clone(), ver) {
                Ok(versioned) => match versioned {
                    SaveStateVersioned::Alpha1_6_16a(v) => match postcard::to_stdvec(&v) {
                        Ok(d) => d,
                        Err(e) => return SaveResult::CantEncodeData(e),
                    },
                    SaveStateVersioned::Alpha1_7_2a(v) => match postcard::to_stdvec(&v) {
                        Ok(d) => d,
                        Err(e) => return SaveResult::CantEncodeData(e),
                    },
                },
                Err(e) => return SaveResult::DowngradeError(e),
            },
            None => match postcard::to_stdvec(&self.current_save) {
                Ok(d) => d,
                Err(e) => return SaveResult::CantEncodeData(e),
            },
        };

        let compressed = match zstd::encode_all(&serialized[..], 10) {
            Ok(d) => d,
            Err(e) => return SaveResult::CantCompress(e),
        };

        if let Some(parent) = path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                return SaveResult::CantCreateDir(e);
            }
        }

        let mut file = match File::create(path) {
            Ok(f) => f,
            Err(e) => return SaveResult::CantWriteFile(e),
        };

        let header = build_save_header(&self.current_save);
        if let Err(e) = file.write_all(header.as_bytes()) {
            return SaveResult::CantWriteFile(e);
        }
        if let Err(e) = file.write_all(&compressed) {
            return SaveResult::CantWriteFile(e);
        }

        SaveResult::Success
    }

    pub fn save(
        &mut self,
        camera: &Camera,
        roads: &Roads,
        terrain: &Terrain,
        props: &Props,
        buildings: &Buildings,
    ) -> SaveResult {
        self.save_as_version(camera, roads, terrain, props, buildings, None)
    }
}

impl Default for GameState {
    fn default() -> Self {
        let mut save = Self::new();
        save.current_save = SaveState::new(128);
        save
    }
}

pub trait UpgradeToLatest {
    fn upgrade_to_latest(self) -> SaveState;
}

pub trait DowngradeFrom<T> {
    fn downgrade_from(from: T) -> Self;
}

macro_rules! define_migrations {
    (
        latest = $latest_variant:ident($latest:ty);
        enum $enum_name:ident {
            $(
                $variant:ident($from_ty:ty) => $next_variant:ident($to_ty:ty) {
                    upgrade: {
                        copy: [ $( $up_copy:ident ),* $(,)? ]
                        $(, default:   [ $( $up_default_field:ident : $up_default_ty:ty ),* $(,)? ] )?
                        $(, transform: [ $( $up_xform_field:ident : $up_xform_closure:expr ),* $(,)? ] )?
                        $(,)?
                    }
                    $(, downgrade: {
                        copy: [ $( $down_copy:ident ),* $(,)? ]
                        $(, default:   [ $( $down_default_field:ident : $down_default_ty:ty ),* $(,)? ] )?
                        $(, transform: [ $( $down_xform_field:ident : $down_xform_closure:expr ),* $(,)? ] )?
                        $(,)?
                    })?
                    $(,)?
                }
            ),+ $(,)?
        }
    ) => {
        #[derive(Serialize, Deserialize)]
        pub enum $enum_name {
            $( $variant($from_ty), )+
            $latest_variant($latest),
        }

        $(
            impl From<$from_ty> for $to_ty {
                fn from(v: $from_ty) -> Self {
                    $($(
                        let $up_xform_field = { let v_ref = &v; ($up_xform_closure)(v_ref) };
                    )*)?
                    Self {
                        $( $up_copy: v.$up_copy, )*
                        $( $( $up_default_field: <$up_default_ty>::default(), )* )?
                        $( $( $up_xform_field, )* )?
                    }
                }
            }

            impl UpgradeToLatest for $from_ty {
                fn upgrade_to_latest(self) -> SaveState {
                    <$to_ty>::from(self).upgrade_to_latest()
                }
            }

            $(
                impl DowngradeFrom<$to_ty> for $from_ty {
                    fn downgrade_from(v: $to_ty) -> Self {
                        $($(
                            let $down_xform_field = { let v_ref = &v; ($down_xform_closure)(v_ref) };
                        )*)?
                        Self {
                            $( $down_copy: v.$down_copy, )*
                            $( $( $down_default_field: <$down_default_ty>::default(), )* )?
                            $( $( $down_xform_field, )* )?
                        }
                    }
                }
            )?
        )+

        impl UpgradeToLatest for $latest {
            fn upgrade_to_latest(self) -> SaveState { self }
        }

        impl $enum_name {
            pub fn into_latest(self) -> $latest {
                match self {
                    $( Self::$variant(v) => v.upgrade_to_latest(), )+
                    Self::$latest_variant(v) => v,
                }
            }
        }
    };
}

define_migrations! {
    latest = Alpha1_7_2a(SaveState);
    enum SaveStateVersioned {
        Alpha1_6_16a(SaveStateAlpha1_6_16a) => Alpha1_7_2a(SaveState) {
            upgrade: {
                copy: [
                    name, chunk_size, timestamp_unix,
                    player_pos, player_yaw, player_pitch,
                    terrain_edits, roads, props,
                ],
                default:   [zones: ZoningStorage, buildings: BuildingStorage, partitions: PartitionManager, road_types: RoadTypes],
                transform: [version: |_v: &SaveStateAlpha1_6_16a| SaveVersion::current()],
            },
            downgrade: {
                copy: [
                    name, chunk_size, timestamp_unix,
                    player_pos, player_yaw, player_pitch,
                    terrain_edits, roads, props,
                ],
                transform: [version: |v: &SaveState| v.version.to_string()],
                // zones is dropped
            },
        },
    }
}

impl SaveStateVersioned {
    pub fn from_latest(latest: SaveState, target_version: SaveVersion) -> Result<Self, String> {
        match target_version {
            SaveVersion::Alpha1_7_2a => Ok(Self::Alpha1_7_2a(latest)),
            SaveVersion::Alpha1_6_16a => Ok(Self::Alpha1_6_16a(
                SaveStateAlpha1_6_16a::downgrade_from(latest),
            )),
        }
    }
}

#[derive(Serialize, Deserialize, Default)]
pub struct SaveStateAlpha1_6_16a {
    pub name: String,
    pub chunk_size: ChunkSize,
    pub version: String,
    pub timestamp_unix: u128,
    pub player_pos: WorldPos,
    pub player_yaw: f32,
    pub player_pitch: f32,
    pub terrain_edits: HashMap<ChunkCoord, PersistedChunk>,
    pub roads: RoadStorage,
    pub props: Vec<SavePropChunk>,
}

#[derive(
    Serialize,
    Deserialize,
    Default,
    EnumString,
    EnumIter,
    PartialOrd,
    Ord,
    Display,
    PartialEq,
    Eq,
    Clone,
    Debug,
)]
pub enum SaveVersion {
    #[default]
    #[strum(serialize = "Alpha v1.6.16")]
    Alpha1_6_16a,
    Alpha1_7_2a,
}
impl SaveVersion {
    pub fn current() -> SaveVersion {
        SaveVersion::iter().max().unwrap()
    }
}
#[derive(Serialize, Deserialize, Default, Clone)]
pub struct SaveState {
    pub name: String,
    pub chunk_size: ChunkSize,
    pub version: SaveVersion,
    pub timestamp_unix: u128,
    pub player_pos: WorldPos,
    pub player_yaw: f32,
    pub player_pitch: f32,
    pub terrain_edits: HashMap<ChunkCoord, PersistedChunk>,
    pub roads: RoadStorage,
    pub road_types: RoadTypes,
    pub partitions: PartitionManager,
    pub props: Vec<SavePropChunk>,
    #[serde(default)]
    pub zones: ZoningStorage,
    pub buildings: BuildingStorage,
}
impl SaveState {
    pub fn new(chunk_size: ChunkSize) -> Self {
        Self {
            chunk_size,
            ..Default::default()
        }
    }
    pub fn load(
        &mut self,
        camera: &mut Camera,
        camera_controller: &mut CameraController,
        roads: &mut Roads,
        terrain: &mut Terrain,
        props: &mut Props,
        buildings: &mut Buildings,
    ) {
        if self.chunk_size == 0 {
            self.chunk_size = default_chunk_size();
        }

        camera.chunk_size = self.chunk_size;
        camera.target = self.player_pos;
        camera.yaw = self.player_yaw;
        camera.pitch = self.player_pitch;

        camera_controller.target_yaw = self.player_yaw;
        camera_controller.target_pitch = self.player_pitch;

        terrain
            .terrain_editor
            .load_edits_from_hashmap(mem::take(&mut self.terrain_edits));

        roads.road_manager.roads = mem::take(&mut self.roads);

        props.load_props(mem::take(&mut self.props));

        buildings.zoning.zoning_storage = mem::take(&mut self.zones);
        buildings.storage = mem::take(&mut self.buildings);
        roads.partition_manager = mem::take(&mut self.partitions);
        roads.road_manager.road_types = mem::take(&mut self.road_types);
    }
    pub fn save(
        &mut self,
        camera: &Camera,
        roads: &Roads,
        terrain: &Terrain,
        props: &Props,
        buildings: &Buildings,
    ) {
        self.chunk_size = if camera.chunk_size == 0 {
            default_chunk_size()
        } else {
            camera.chunk_size
        };
        self.version = SaveVersion::current();
        self.timestamp_unix = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);

        self.player_pos = camera.target;
        self.player_yaw = camera.yaw;
        self.player_pitch = camera.pitch;

        self.terrain_edits = terrain.terrain_editor.get_edits();
        self.roads = roads.road_manager.roads.clone();
        self.props = props.get_props();
        self.zones = buildings.zoning.zoning_storage.clone();
        self.buildings = buildings.storage.clone();
        self.partitions = roads.partition_manager.clone();
        self.road_types = roads.road_manager.road_types.clone();
    }
}

fn default_chunk_size() -> ChunkSize {
    128
}
