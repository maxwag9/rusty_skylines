use crate::helpers::paths::saves_dir;
use crate::helpers::positions::{ChunkCoord, ChunkSize, WorldPos};
use crate::renderer::props::{Props, SavePropChunk};
use crate::world::camera::{Camera, CameraController};
use crate::world::roads::road_subsystem::Roads;
use crate::world::roads::roads::RoadStorage;
use crate::world::terrain::terrain_editing::PersistedChunk;
use crate::world::terrain::terrain_subsystem::Terrain;
use sanitize_filename::sanitize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{Error, Write};
use std::path::PathBuf;
use std::time::SystemTime;

#[derive(Debug)]
pub enum LoadResult {
    Success,
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
        version = sanitize_header_value(&save.version),
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
    ) -> LoadResult {
        let safe_name = sanitize(save_name);
        let path = saves_dir().join(format!("{}.rss", safe_name));

        match path.try_exists() {
            Ok(true) => {
                if let Some(extension) = path.extension() {
                    if extension != "rss" {
                        return LoadResult::WrongExtension(path);
                    }
                } else {
                    return LoadResult::CantGetExtension;
                }

                let data = match fs::read(&path) {
                    Ok(data) => data,
                    Err(e) => return LoadResult::CantGetData(e),
                };

                let payload = if let Some(header_end) = find_header_end(&data) {
                    &data[header_end..]
                } else {
                    &data[..]
                };

                let decompressed = match zstd::decode_all(payload) {
                    Ok(decompressed) => decompressed,
                    Err(e) => return LoadResult::CantDecompress(e),
                };

                let save_state: SaveState = match postcard::from_bytes(&decompressed) {
                    Ok(meta) => meta,
                    Err(e) => return LoadResult::CantDecodeData(e),
                };

                self.current_save = save_state;
                self.current_save.name = save_name.to_string();
            }
            Ok(false) => {
                self.current_save = SaveState::default();
                self.current_save.name = save_name.to_string();
                return LoadResult::FileNonExistent(
                    path.to_str()
                        .unwrap_or("Unable to tell you which path was used")
                        .to_string(),
                );
            }
            Err(e) => {
                return LoadResult::PathError(e);
            }
        }

        self.current_save
            .load(camera, camera_controller, roads, terrain, props);
        LoadResult::Success
    }

    pub fn save(
        &mut self,
        camera: &Camera,
        roads: &Roads,
        terrain: &Terrain,
        props: &Props,
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
                return SaveResult::WrongExtension(
                    ext.to_str()
                        .unwrap_or("Unable to tell you which wrong extension was used")
                        .to_string(),
                );
            }
        } else {
            return SaveResult::CantGetExtension(
                path.to_str()
                    .unwrap_or("Unable to tell you which path was used")
                    .to_string(),
            );
        }

        self.current_save.save(camera, roads, terrain, props);

        let serialized = match postcard::to_stdvec(&self.current_save) {
            Ok(data) => data,
            Err(e) => return SaveResult::CantEncodeData(e),
        };

        let compressed = match zstd::encode_all(&serialized[..], 10) {
            Ok(data) => data,
            Err(e) => return SaveResult::CantCompress(e),
        };

        if let Some(parent) = path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                return SaveResult::CantCreateDir(e);
            }
        }

        let mut file = match File::create(path) {
            Ok(file) => file,
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
        save.current_save = SaveState::new(128);
        save
    }
}

#[derive(Serialize, Deserialize)]
pub struct SaveState {
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
impl Default for SaveState {
    fn default() -> Self {
        Self {
            name: "New World".to_string(),
            chunk_size: default_chunk_size(),
            version: current_save_version(),
            timestamp_unix: 0,
            player_pos: WorldPos::zero(),
            player_yaw: 0f32,
            player_pitch: 0f32,
            terrain_edits: HashMap::new(),
            roads: RoadStorage::default(),
            props: vec![],
        }
    }
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
            .load_edits_from_hashmap(self.terrain_edits.clone());
        roads.road_manager.roads = self.roads.clone();
        props.load_props(self.props.clone());
    }
    pub fn save(&mut self, camera: &Camera, roads: &Roads, terrain: &Terrain, props: &Props) {
        self.chunk_size = if camera.chunk_size == 0 {
            default_chunk_size()
        } else {
            camera.chunk_size
        };
        self.version = current_save_version();
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
    }
}

fn default_chunk_size() -> ChunkSize {
    128
}
fn current_save_version() -> String {
    "Alpha v1.6.16".to_string()
}
