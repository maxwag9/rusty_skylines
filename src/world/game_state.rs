use crate::data::Settings;
use crate::helpers::paths::saves_dir;
use crate::helpers::positions::{ChunkSize, WorldPos, chunk_size, set_chunk_size};
use crate::renderer::props::{Props, SavedProps};
use crate::ui::parser::Value;
use crate::ui::variables::Variables;
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
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{fs, mem};
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumIter, EnumString};

#[derive(Display, Debug, Clone)]
pub enum LoadResult {
    Success(SaveVersion),
    FileNotFound(PathBuf),
    WrongExtension(PathBuf),
    PathError(String),
    FileNonExistent(String),
    CantGetExtension,
    CantGetData(String),
    CantDecompress(String),
    CantDecodeData(String),
    EmptyName,
    CantCreateSave(SaveResult),
}
// impl fmt::Display for LoadResult {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         match self {
//             LoadResult::Success(version) => {
//                 write!(f, "Success({})", version)
//             }
//
//             LoadResult::FileNotFound(path) => {
//                 write!(f, "FileNotFound({})", path.display())
//             }
//
//             LoadResult::WrongExtension(path) => {
//                 write!(f, "WrongExtension({})", path.display())
//             }
//
//             LoadResult::PathError(err) => {
//                 write!(f, "PathError({})", err)
//             }
//
//             LoadResult::FileNonExistent(name) => {
//                 write!(f, "FileNonExistent({})", name)
//             }
//
//             LoadResult::CantGetExtension => {
//                 write!(f, "CantGetExtension")
//             }
//
//             LoadResult::CantGetData(err) => {
//                 write!(f, "CantGetData({})", err)
//             }
//
//             LoadResult::CantDecompress(err) => {
//                 write!(f, "CantDecompress({})", err)
//             }
//
//             LoadResult::CantDecodeData(err) => {
//                 write!(f, "CantDecodeData({})", err)
//             }
//
//             LoadResult::EmptyName => {
//                 write!(f, "EmptyName")
//             }
//
//             LoadResult::CantCreateSave(result) => {
//                 write!(f, "CantCreateSave({})", result)
//             }
//         }
//     }
// }
#[derive(Display, Debug, Clone)]
pub enum SaveResult {
    Success,
    NotAFile,
    SaveAlreadyExists(String),
    CantWriteFile(String),
    CantCreateDir(String),
    CantCompress(String),
    CantEncodeData(String),
    CantGetExtension(String),
    WrongExtension(String),
    EmptySaveName,
    DowngradeError(String),
    TriedToSaveEmptySave,
    PathError(String),
}

// impl fmt::Display for SaveResult {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         match self {
//             SaveResult::Success => {
//                 write!(f, "Success")
//             }
//
//             SaveResult::NotAFile => {
//                 write!(f, "NotAFile")
//             }
//
//             SaveResult::CantWriteFile(err) => {
//                 write!(f, "CantWriteFile({})", err)
//             }
//
//             SaveResult::CantCreateDir(err) => {
//                 write!(f, "CantCreateDir({})", err)
//             }
//
//             SaveResult::CantCompress(err) => {
//                 write!(f, "CantCompress({})", err)
//             }
//
//             SaveResult::CantEncodeData(err) => {
//                 write!(f, "CantEncodeData({})", err)
//             }
//
//             SaveResult::CantGetExtension(ext) => {
//                 write!(f, "CantGetExtension({})", ext)
//             }
//
//             SaveResult::WrongExtension(ext) => {
//                 write!(f, "WrongExtension({})", ext)
//             }
//
//             SaveResult::EmptySaveName => {
//                 write!(f, "EmptySaveName")
//             }
//
//             SaveResult::DowngradeError(msg) => {
//                 write!(f, "DowngradeError({})", msg)
//             }
//         }
//     }
// }

const SAVE_MAGIC: &str = "RSS1";

#[derive(Debug, Clone)]
pub struct SaveHeader {
    pub name: String,
    pub version: SaveVersion,
    pub timestamp_unix: u128,
    pub chunk_size: ChunkSize,
    pub difficulty: String,
}
impl Default for SaveHeader {
    fn default() -> Self {
        SaveHeader {
            name: String::new(),
            version: SaveVersion::current(),
            timestamp_unix: 0,
            chunk_size: default_chunk_size(),
            difficulty: "Easy".to_string(),
        }
    }
}
macro_rules! parse_save_header {
    ($bytes:expr, $($field:ident : $ty:ty),* $(,)?) => {{
        let text = String::from_utf8_lossy($bytes);

        let mut header = SaveHeader::default();

        for line in text.lines() {
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };

            match key {
                $(
                    stringify!($field) => {
                        if let Ok(parsed) = value.trim().parse::<$ty>() {
                            header.$field = parsed;
                        }
                    }
                )*
                _ => {}
            }
        }

        header
    }};
}

fn sanitize_header_value(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\r' | '\n' => ' ',
            _ => c,
        })
        .collect()
}

fn build_save_header(save: &SaveInfo) -> String {
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
        version = sanitize_header_value(&save.load_result.to_string()),
        timestamp = save.timestamp_unix,
        chunk_size = save.chunk_size
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
pub struct NewSavePackage {
    pub name: String,
    pub difficulty: String,
}
#[derive(Default)]
pub struct GameState {
    pub current_save_info: Option<SaveInfo>,
}
impl GameState {
    pub fn new() -> Self {
        Self {
            current_save_info: None,
        }
    }

    pub fn create_save(
        &mut self,
        world: &mut World,
        props: &Props,
        settings: &Settings,
        variables: &mut Variables,
        new_save_package: NewSavePackage,
    ) -> SaveResult {
        let save_name = make_safe_save_name(new_save_package.name.as_str());

        if save_name.is_empty() {
            return SaveResult::EmptySaveName;
        }

        let path = saves_dir().join(format!("{save_name}.rss"));

        // Do not silently overwrite an existing save.
        match path.try_exists() {
            Ok(true) => {
                return SaveResult::SaveAlreadyExists(save_name);
            }
            Ok(false) => {}
            Err(e) => {
                return SaveResult::PathError(e.to_string());
            }
        }

        // This is now the active save.
        self.current_save_info = Some(SaveInfo {
            name: save_name.clone(),
            difficulty: new_save_package.difficulty,
            ..SaveInfo::default()
        });

        let result = self.save(world, props, settings, variables, false);

        match result {
            SaveResult::Success => {
                println!("Created World '{}'", save_name);
            }

            ref e => {
                eprintln!("Failed to create World '{}': {:?}", save_name, e);

                self.current_save_info = None;
            }
        }

        result
    }

    pub fn load(
        &mut self,
        save_name: &str,
        world: &mut World,
        props: &mut Props,
        settings: &Settings,
        variables: &mut Variables,
    ) -> LoadResult {
        let save_name = make_safe_save_name(save_name);
        if save_name.is_empty() {
            return LoadResult::EmptyName;
        }
        let path = saves_dir().join(format!("{}.rss", save_name));
        let detected_version: SaveVersion;
        let mut load_save: SaveState;
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
                    Err(e) => return LoadResult::CantGetData(e.to_string()),
                };

                let (header_bytes, payload) = if let Some(end) = find_header_end(&data) {
                    (&data[..end], &data[end..])
                } else {
                    (b"" as &[u8], &data[..])
                };

                let header = parse_the_save_header(header_bytes);
                detected_version = header.version.clone();

                let decompressed = match zstd::decode_all(payload) {
                    Ok(d) => d,
                    Err(e) => return LoadResult::CantDecompress(e.to_string()),
                };
                let save_state = match revision::from_slice::<SaveState>(&decompressed) {
                    Ok(s) => s,
                    Err(e) => return LoadResult::CantDecodeData(e.to_string()),
                };

                load_save = save_state;
                load_save.name = save_name.to_string();
            }
            Ok(false) => {
                self.current_save_info = Some(SaveInfo::default());
                self.current_save_info
                    .as_mut()
                    .map(|s| s.name = save_name.to_string());

                match self.save(world, props, settings, variables, false) {
                    SaveResult::Success => {
                        // Load the save I just created.
                        return self.load(save_name.as_str(), world, props, settings, variables);
                    }
                    e => {
                        self.current_save_info = None;
                        return LoadResult::CantCreateSave(e);
                    }
                }
            }
            Err(e) => return LoadResult::PathError(e.to_string()),
        }

        load_save.load(world, props);

        let success = LoadResult::Success(detected_version);
        self.current_save_info = Some(SaveInfo {
            name: load_save.name.clone(),
            load_result: success.clone(),
            timestamp_unix: load_save.timestamp_unix,
            chunk_size: load_save.chunk_size,
            difficulty: if load_save.difficulty.is_empty() {
                "Easy".to_string()
            } else {
                load_save.difficulty
            },
        });
        success
    }

    pub fn save(
        &mut self,
        world: &mut World,
        props: &Props,
        settings: &Settings,
        variables: &mut Variables,
        and_exit: bool,
    ) -> SaveResult {
        let Some(current_save_info) = self.current_save_info.as_ref() else {
            return SaveResult::TriedToSaveEmptySave;
        };
        let safe_name = make_safe_save_name(current_save_info.name.as_str());
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
        let mut save_save = SaveState::default();
        save_save.difficulty = current_save_info.difficulty.clone();
        save_save.save(world, props);

        let serialized = match revision::to_vec(&save_save) {
            Ok(d) => d,
            Err(e) => return SaveResult::CantEncodeData(e.to_string()),
        };

        let compressed = match zstd::encode_all(&serialized[..], 10) {
            Ok(d) => d,
            Err(e) => return SaveResult::CantCompress(e.to_string()),
        };

        if let Some(parent) = path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                return SaveResult::CantCreateDir(e.to_string());
            }
        }

        let mut file = match File::create(path) {
            Ok(f) => f,
            Err(e) => return SaveResult::CantWriteFile(e.to_string()),
        };

        let header = build_save_header(self.current_save_info.as_ref().unwrap());
        if let Err(e) = file.write_all(header.as_bytes()) {
            return SaveResult::CantWriteFile(e.to_string());
        }
        if let Err(e) = file.write_all(&compressed) {
            return SaveResult::CantWriteFile(e.to_string());
        }
        if and_exit {
            self.current_save_info = None;
            world.recreate(settings, props);
        }
        // If exiting the save after saving, set to false, if not exiting after saving, then set to true, because The player is inside the save of course!
        SaveResult::Success
    }
}

pub fn make_safe_save_name(name: &str) -> String {
    let cleaned = sanitize(name.trim());

    cleaned
        .chars()
        .filter(|&c| !matches!(c, '{' | '}' | '.'))
        .collect::<String>()
        .trim()
        .trim_matches('.')
        .to_string()
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
    Copy,
    Debug,
)]
#[revisioned(revision = 2)]
pub enum SaveVersion {
    #[default]
    #[revision(start = 2)]
    AlphaV1_8_6a,
    AlphaV1_8_3a,
}
impl SaveVersion {
    pub fn current() -> SaveVersion {
        SaveVersion::iter().max().unwrap()
    }
}
#[revisioned(revision = 3)]
#[derive(Default, Clone)]
pub struct SaveState {
    pub name: String,
    pub chunk_size: ChunkSize,
    pub version: SaveVersion,
    pub timestamp_unix: u128,
    pub player_pos: WorldPos,
    pub player_yaw: f32,
    pub player_pitch: f32,
    #[revision(start = 2)]
    pub total_game_time: f64,
    #[revision(start = 3)]
    pub difficulty: String,
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
        world.time.total_game_time = self.total_game_time;
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
        self.total_game_time = world.time.total_game_time;
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

#[derive(Clone)]
pub struct SaveInfo {
    pub name: String,
    pub load_result: LoadResult,
    pub timestamp_unix: u128,
    pub chunk_size: ChunkSize,
    pub difficulty: String,
}
impl SaveInfo {
    pub fn to_values(self) -> Vec<Value> {
        vec![
            Value::String(self.name),
            Value::String(format!("{:?}", self.load_result)),
            Value::I64(self.timestamp_unix as i64),
            Value::I64(self.chunk_size as i64),
            Value::String(self.difficulty),
        ]
    }
}
impl Default for SaveInfo {
    fn default() -> Self {
        SaveInfo {
            name: "Default Save".to_string(),
            load_result: LoadResult::Success(SaveVersion::current()),
            timestamp_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0),
            chunk_size: default_chunk_size(),
            difficulty: "Easy".to_string(),
        }
    }
}
pub fn get_available_saves() -> Vec<SaveInfo> {
    let mut saves = Vec::new();

    let dir = saves_dir();

    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return saves,
    };

    for entry in entries.flatten() {
        let path = entry.path();

        // Only .rss files
        if path.extension().and_then(|e| e.to_str()) != Some("rss") {
            continue;
        }

        let fallback_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown")
            .to_string();

        let data = match fs::read(&path) {
            Ok(d) => d,
            Err(e) => {
                saves.push(SaveInfo {
                    name: fallback_name,
                    load_result: LoadResult::CantGetData(e.to_string()),
                    ..Default::default()
                });
                continue;
            }
        };

        let (header_bytes, payload) = if let Some(end) = find_header_end(&data) {
            (&data[..end], &data[end..])
        } else {
            (b"" as &[u8], &data[..])
        };

        let header = parse_the_save_header(header_bytes);

        let save_name = if header.name.is_empty() {
            fallback_name
        } else {
            header.name
        };

        let version = header.version;
        let timestamp = header.timestamp_unix;

        // Validate actual save data
        let result = match zstd::decode_all(payload) {
            Ok(decoded) => match revision::from_slice::<SaveState>(&decoded) {
                Ok(_) => LoadResult::Success(version.clone()),
                Err(e) => LoadResult::CantDecodeData(e.to_string()),
            },
            Err(e) => LoadResult::CantDecompress(e.to_string()),
        };

        saves.push(SaveInfo {
            name: save_name,
            load_result: result,
            timestamp_unix: timestamp,
            chunk_size: header.chunk_size,
            difficulty: header.difficulty,
        });
    }

    // newest saves first
    saves.sort_by(|a, b| b.timestamp_unix.cmp(&a.timestamp_unix));

    saves
}

fn parse_the_save_header(data: &[u8]) -> SaveHeader {
    parse_save_header!(
        data,
        timestamp_unix: u128,
        version: SaveVersion,
        chunk_size: ChunkSize
    )
}
