use crate::data::Settings;
use crate::helpers::modpack::ModManager;
use crate::helpers::paths::saves_dir;
use crate::resources::Time;
use crate::ui::parser::Value;
use crate::ui::variables::Variables;
use crate::world::buildings::buildings::BuildingStorage;
use crate::world::buildings::zoning::ZoningStorage;
use crate::world::cars::partitions::PartitionManager;
use crate::world::props::props::{Props, SavedProps};
use crate::world::roads::roads::{RoadStorage, RoadTypes, RoadTypesData};
use crate::world::statisticals::CityState;
use crate::world::terrain::terrain_editing::TerrainEdit;
use crate::world::world::World;
use revision::revisioned;
use rusty_skylines_utils::positions::{ChunkSize, WorldPos, set_chunk_size};
use sanitize_filename::sanitize;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{fs, mem};
use std::collections::BTreeMap;
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
                        let parsed: $ty = value.trim().parse().unwrap_or_default();
                        header.$field = parsed;
                    }
                )*
                _ => {}
            }
        }

        header
    }};
}
const SAVE_MAGIC: &str = "RSS1";

#[derive(Debug, Clone)]
pub struct SaveHeader {
    pub name: String,
    pub version: SaveVersion,
    pub timestamp_unix: u128,
    pub chunk_size: ChunkSize,
    pub difficulty: String,
    pub terrain_preset_name: String
}
impl Default for SaveHeader {
    fn default() -> Self {
        SaveHeader {
            name: String::new(),
            version: SaveVersion::current(),
            timestamp_unix: 0,
            chunk_size: default_chunk_size(),
            difficulty: "Easy".to_string(),
            terrain_preset_name: "Default".to_string(),
        }
    }
}
fn parse_the_save_header(data: &[u8]) -> SaveHeader {
    parse_save_header!(
        data,
        name: String,
        version: SaveVersion,
        timestamp_unix: u128,
        chunk_size: ChunkSize,
        difficulty: String,
        terrain_preset_name: String,
    )
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
    let chunk_size = if save.chunk_size == 0 {
        default_chunk_size()
    } else {
        save.chunk_size
    };
    let timestamp_unix = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!(
        "{magic}\n\
name={name}\n\
version={version}\n\
timestamp_unix={timestamp}\n\
chunk_size={chunk_size:?}\n\
difficulty={difficulty}\n\
terrain_preset_name={terrain_preset_name}\n\
compression=zstd\n\
payload=compressed_binary\n\
all_of_this_crap_is_for_your_enjoyment\n\
\n",
        magic = SAVE_MAGIC,
        name = sanitize_header_value(&save.name),
        version = sanitize_header_value(SaveVersion::current().to_string().as_str()),
        timestamp = timestamp_unix,
        chunk_size = chunk_size,
        difficulty = sanitize_header_value(&save.difficulty),
        terrain_preset_name = sanitize_header_value(&save.terrain_preset_name)
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
    pub preset_name: String,
}
#[derive(Default)]
pub struct GameState {
    pub current_save_info: Option<SaveInfo>,
    pub mod_manager: ModManager,
}

impl GameState {
    pub fn new() -> Self {
        Self {
            current_save_info: None,
            mod_manager: ModManager::new(),
        }
    }
    pub fn new_off() -> Self {
        Self {
            current_save_info: None,
            mod_manager: ModManager::new_off(),
        }
    }
    pub fn create_save(
        &mut self,
        world: &mut World,
        settings: &Settings,
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
            terrain_preset_name: new_save_package.preset_name,
            ..SaveInfo::default()
        });

        let result = self.save(world, settings, false);

        match result {
            SaveResult::Success => {
                println!("[Saves] Created World '{}'", save_name);
            }

            ref e => {
                eprintln!("[Saves] Failed to create World '{}': {:?}", save_name, e);

                self.current_save_info = None;
            }
        }

        result
    }

    pub fn load(
        &mut self,
        save_name: &str,
        world: &mut World,
        settings: &Settings
    ) -> LoadResult {
        let save_name = make_safe_save_name(save_name);
        if save_name.is_empty() {
            return LoadResult::EmptyName;
        }
        let path = saves_dir().join(format!("{}.rss", save_name));
        let mut header: SaveHeader;
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

                header = parse_the_save_header(header_bytes);

                let decompressed = match zstd::decode_all(payload) {
                    Ok(d) => d,
                    Err(e) => return LoadResult::CantDecompress(e.to_string()),
                };
                let save_state = match revision::from_slice::<SaveState>(&decompressed) {
                    Ok(s) => s,
                    Err(e) => return LoadResult::CantDecodeData(e.to_string()),
                };

                load_save = save_state;
            }
            Ok(false) => {
                self.current_save_info = Some(SaveInfo::default());
                self.current_save_info
                    .as_mut()
                    .map(|s| s.name = save_name.to_string());

                match self.save(world, settings, false) {
                    SaveResult::Success => {
                        // Load the save I just created.
                        return self.load(save_name.as_str(), world, settings);
                    }
                    e => {
                        self.current_save_info = None;
                        return LoadResult::CantCreateSave(e);
                    }
                }
            }
            Err(e) => return LoadResult::PathError(e.to_string()),
        }
        if header.chunk_size == 0 {
            header.chunk_size = default_chunk_size();
        }
        set_chunk_size(header.chunk_size);
        load_save.load(world, &self.mod_manager, &header);

        let success = LoadResult::Success(header.version);
        self.current_save_info = Some(SaveInfo {
            name: header.name.clone(),
            load_result: success.clone(),
            timestamp_unix: header.timestamp_unix,
            chunk_size: header.chunk_size,
            difficulty: if header.difficulty.is_empty() {
                "Easy".to_string()
            } else {
                header.difficulty
            },
            terrain_preset_name: header.terrain_preset_name,
        });
        success
    }

    pub fn save(
        &mut self,
        world: &mut World,
        settings: &Settings,
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
        save_save.save(world);

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

        let header = build_save_header(current_save_info);
        if let Err(e) = file.write_all(header.as_bytes()) {
            return SaveResult::CantWriteFile(e.to_string());
        }
        if let Err(e) = file.write_all(&compressed) {
            return SaveResult::CantWriteFile(e.to_string());
        }
        if and_exit {
            self.current_save_info = None;
            world.recreate(settings, &self.mod_manager);
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
#[revisioned(revision = 3)]
pub enum SaveVersion {
    #[default]
    #[revision(start = 3)]
    AlphaV1_8_7a,
    #[revision(start = 2)]
    AlphaV1_8_6a,
    AlphaV1_8_3a,
}
impl SaveVersion {
    pub fn current() -> SaveVersion {
        SaveVersion::iter().min().unwrap()
    }
}

#[revisioned(revision = 5)]
#[derive(Default, Clone)]
pub struct SaveState {
    #[revision(end = 4, convert_fn = "dud")]
    pub name: String,
    #[revision(end = 4, convert_fn = "dud")]
    pub chunk_size: ChunkSize,
    #[revision(end = 4, convert_fn = "dud")]
    pub version: SaveVersion,
    #[revision(end = 4, convert_fn = "dud")]
    pub timestamp_unix: u128,
    pub player_pos: WorldPos,
    pub player_yaw: f32,
    pub player_pitch: f32,
    #[revision(start = 2)]
    pub total_game_time: f64,
    #[revision(start = 3, end = 4, convert_fn = "dud")]
    pub difficulty: String,
    pub terrain_edits: Vec<TerrainEdit>,
    pub roads: RoadStorage,
    #[revision(start = 5)]
    pub road_types: RoadTypesData,
    pub partitions: PartitionManager,
    pub props: SavedProps,
    pub zones: ZoningStorage,
    pub buildings: BuildingStorage,
    pub city_state: CityState,
}
impl SaveState {
    fn dud<T>(
        &mut self,
        _revision: u16,
        _value: T,
    ) -> Result<(), revision::Error> {
        Ok(())
    }
    pub fn new(time: &Time) -> Self {
        Self {
            total_game_time: time.get_spring(),
            ..Default::default()
        }
    }
    pub fn load(&mut self, world: &mut World, mod_manager: &ModManager, header: &SaveHeader) {
        let camera = &mut world.world_state.camera;
        let camera_controller = &mut world.world_state.cam_controller;
        let terrain = &mut world.terrain;
        let roads = &mut world.roads;
        let zoning = &mut world.zoning;
        let buildings = &mut world.buildings;
        world.time.total_game_time = self.total_game_time;

        camera.target = self.player_pos;
        camera.yaw = self.player_yaw;
        camera.pitch = self.player_pitch;

        camera_controller.target_yaw = self.player_yaw;
        camera_controller.target_pitch = self.player_pitch;

        terrain.terrain_editor.load_edits_from_vec(mem::take(&mut self.terrain_edits));
        //let preset_path = mod_manager.resource_path(format!("simulation/terrain/presets/{}.yaml", header.terrain_preset).as_str());
        terrain.current_preset_name = header.terrain_preset_name.clone(); //preset_path.and_then(|p| Terrain::compile_preset_handled(p)).unwrap_or_default();
        terrain.recreate_generator(&world.props, mod_manager);
        roads.road_manager.roads = mem::take(&mut self.roads);

        world.props.load_props(mem::take(&mut self.props), mod_manager);

        zoning.zoning_storage = mem::take(&mut self.zones);
        buildings.storage = mem::take(&mut self.buildings);
        buildings.partitions = mem::take(&mut self.partitions);
        roads.road_manager.road_types.save_data = mem::take(&mut self.road_types);
        world.city_state = mem::take(&mut self.city_state);
        //variables.set_i64("lanes_left", terrain.cursor.road_type.unwrap().lanes_each_direction().0 as i64);
    }
    pub fn save(&mut self, world: &World) {
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
        self.props = world.props.get_props();
        self.zones = zoning.zoning_storage.clone();
        self.buildings = buildings.storage.clone();
        self.partitions = buildings.partitions.clone();
        self.road_types = roads.road_manager.road_types.data.clone();
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
    pub terrain_preset_name: String,
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
            terrain_preset_name: "Default".to_string(),
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
            terrain_preset_name: header.terrain_preset_name,
        });
    }

    // newest saves first
    saves.sort_by(|a, b| b.timestamp_unix.cmp(&a.timestamp_unix));

    saves
}


