use crate::helpers::paths::saves_dir;
use crate::helpers::positions::{ChunkSize, WorldPos, chunk_size, set_chunk_size};
use crate::renderer::props::{Props, SavedProps};
use crate::world::buildings::buildings::BuildingStorage;
use crate::world::buildings::zoning::ZoningStorage;
use crate::world::cars::partitions::PartitionManager;
use crate::world::roads::roads::{RoadStorage, RoadTypes};
use crate::world::statisticals::CityState;
use crate::world::terrain::terrain_editing::TerrainEdit;
use crate::world::world::World;
use revision::revisioned;
use sanitize_filename::sanitize;
use serde::{Deserialize, Serialize};
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
    CantDecodeData(revision::Error),
    EmptyName,
    CantCreateSave(SaveResult),
}

#[derive(Debug)]
pub enum SaveResult {
    Success,
    NotAFile,

    CantWriteFile(Error),
    CantCreateDir(Error),
    CantCompress(Error),
    CantEncodeData(revision::Error),
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
payload=compressed_binary\n\
all_of_this_crap_is_for_your_enjoyment\n\
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

    pub fn load(&mut self, save_name: &str, world: &mut World, props: &mut Props) -> LoadResult {
        let safe_name = sanitize(save_name);
        if safe_name.is_empty() {
            return LoadResult::EmptyName;
        }
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
                let save_state = match revision::from_slice::<SaveState>(&decompressed) {
                    Ok(s) => s,
                    Err(e) => return LoadResult::CantDecodeData(e),
                };

                self.current_save = save_state;
                self.current_save.name = save_name.to_string();
            }
            Ok(false) => {
                self.current_save = SaveState::default();
                self.current_save.name = save_name.to_string();

                match self.save(world, props) {
                    SaveResult::Success => {
                        // Load the save I just created.
                        return self.load(save_name, world, props);
                    }
                    e => {
                        return LoadResult::CantCreateSave(e);
                    }
                }
            }
            Err(e) => return LoadResult::PathError(e),
        }

        self.current_save.load(world, props);
        LoadResult::Success(detected_version.unwrap_or(SaveVersion::current()))
    }

    pub fn save(&mut self, world: &World, props: &Props) -> SaveResult {
        let safe_name = sanitize(&self.current_save.name).to_string();
        if safe_name.is_empty() {
            return SaveResult::EmptySaveName;
        }
        // if safe_name.is_empty() {
        //     let base_name = "New World";
        //     safe_name = base_name.to_string();
        //
        //     let mut path = saves_dir().join(format!("{}.rss", safe_name));
        //
        //     if path.exists() {
        //         let mut i = 0;
        //
        //         loop {
        //             let candidate = format!("{base_name} {i}");
        //             path = saves_dir().join(format!("{}.rss", candidate));
        //
        //             if !path.exists() {
        //                 safe_name = candidate;
        //                 break;
        //             }
        //
        //             i += 1;
        //         }
        //     }
        // }

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

        self.current_save.save(world, props);

        let serialized = match revision::to_vec(&self.current_save) {
            Ok(d) => d,
            Err(e) => return SaveResult::CantEncodeData(e),
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
}

impl Default for GameState {
    fn default() -> Self {
        let mut save = Self::new();
        save.current_save = SaveState::new();
        save
    }
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
#[revisioned(revision = 1)]
pub enum SaveVersion {
    #[default]
    AlphaV1_8_3a,
}
impl SaveVersion {
    pub fn current() -> SaveVersion {
        SaveVersion::iter().max().unwrap()
    }
}
#[revisioned(revision = 1)]
#[derive(Default, Clone)]
pub struct SaveState {
    pub name: String,
    pub chunk_size: ChunkSize,
    pub version: SaveVersion,
    pub timestamp_unix: u128,
    pub player_pos: WorldPos,
    pub player_yaw: f32,
    pub player_pitch: f32,
    pub terrain_edits: Vec<TerrainEdit>,
    pub roads: RoadStorage,
    pub road_types: RoadTypes,
    pub partitions: PartitionManager,
    pub props: SavedProps,
    pub zones: ZoningStorage,
    pub buildings: BuildingStorage,
    pub city_state: CityState,
}
impl SaveState {
    pub fn new() -> Self {
        Self {
            chunk_size: chunk_size(),
            ..Default::default()
        }
    }
    pub fn load(&mut self, world: &mut World, props: &mut Props) {
        if self.chunk_size == 0 {
            self.chunk_size = default_chunk_size();
        }
        let camera = &mut world.world_state.camera;
        let camera_controller = &mut world.world_state.cam_controller;
        let terrain = &mut world.terrain;
        let roads = &mut world.roads;
        let zoning = &mut world.zoning;
        let buildings = &mut world.buildings;
        set_chunk_size(self.chunk_size);
        camera.target = self.player_pos;
        camera.yaw = self.player_yaw;
        camera.pitch = self.player_pitch;

        camera_controller.target_yaw = self.player_yaw;
        camera_controller.target_pitch = self.player_pitch;

        terrain
            .terrain_editor
            .load_edits_from_vec(mem::take(&mut self.terrain_edits));

        roads.road_manager.roads = mem::take(&mut self.roads);

        props.load_props(mem::take(&mut self.props));

        zoning.zoning_storage = mem::take(&mut self.zones);
        buildings.storage = mem::take(&mut self.buildings);
        buildings.partitions = mem::take(&mut self.partitions);
        roads.road_manager.road_types = mem::take(&mut self.road_types);
        world.city_state = mem::take(&mut self.city_state);
        //variables.set_i64("lanes_left", terrain.cursor.road_type.unwrap().lanes_each_direction().0 as i64);
    }
    pub fn save(&mut self, world: &World, props: &Props) {
        self.chunk_size = if chunk_size() == 0 {
            default_chunk_size()
        } else {
            chunk_size()
        };
        self.version = SaveVersion::current();
        self.timestamp_unix = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let camera = &world.world_state.camera;
        let camera_controller = &world.world_state.cam_controller;
        let terrain = &world.terrain;
        let roads = &world.roads;
        let zoning = &world.zoning;
        let buildings = &world.buildings;
        self.player_pos = camera.target;
        self.player_yaw = camera.yaw;
        self.player_pitch = camera.pitch;

        self.terrain_edits = terrain.terrain_editor.get_edits_for_save().to_vec();
        self.roads = roads.road_manager.roads.clone();
        self.props = props.get_props();
        self.zones = zoning.zoning_storage.clone();
        self.buildings = buildings.storage.clone();
        self.partitions = buildings.partitions.clone();
        self.road_types = roads.road_manager.road_types.clone();
        self.city_state = world.city_state.clone();
    }
}

fn default_chunk_size() -> ChunkSize {
    128
}
