use crate::helpers::positions::{ChunkSize, WorldPos};
use crate::renderer::pipelines::ToneMappingState;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};
use wgpu::*;

pub trait CycleNext {
    fn next(&self) -> Self;
}

/// Mode switch: Strict attempts to deserialize the file as normal JSON into GuiLayout.
/// Bent ignores JSON structure and deterministically synthesizes a GuiLayout from the file bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BendMode {
    Strict,
    Bent,
    #[serde(other)]
    Unknown,
}
impl Default for BendMode {
    fn default() -> Self {
        BendMode::Strict
    }
}
impl CycleNext for BendMode {
    fn next(&self) -> Self {
        match self {
            BendMode::Strict => BendMode::Bent,
            BendMode::Bent => BendMode::Strict,
            BendMode::Unknown => BendMode::Strict,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")] // "fifo", "mailbox", ...
pub enum PresentModeSetting {
    Immediate,
    Mailbox,
    Fifo,
    #[serde(other)]
    Unknown,
}

impl PresentModeSetting {
    pub fn to_wgpu(self) -> PresentMode {
        match self {
            PresentModeSetting::Immediate => PresentMode::Immediate,
            PresentModeSetting::Mailbox => PresentMode::Mailbox,
            PresentModeSetting::Fifo => PresentMode::Fifo,
            _ => PresentMode::Mailbox,
        }
    }
}

impl Default for PresentModeSetting {
    fn default() -> Self {
        PresentModeSetting::Mailbox
    }
}
impl CycleNext for PresentModeSetting {
    fn next(&self) -> Self {
        match self {
            PresentModeSetting::Immediate => PresentModeSetting::Mailbox,
            PresentModeSetting::Mailbox => PresentModeSetting::Fifo,
            PresentModeSetting::Fifo => PresentModeSetting::Immediate,
            PresentModeSetting::Unknown => PresentModeSetting::Mailbox,
        }
    }
}

fn default_chunk_size() -> ChunkSize {
    128
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum DebugViewState {
    Off,
    Normals,
    Depth,
    GtaoRaw,
    GtaoBlurred,
    RTRaw,
    Motion,
}
impl Default for DebugViewState {
    fn default() -> Self {
        Self::Off
    }
}
impl DebugViewState {
    pub fn next(&self) -> Self {
        use DebugViewState::*;
        match self {
            Off => Normals,
            Normals => Depth,
            Depth => GtaoRaw,
            GtaoRaw => GtaoBlurred,
            GtaoBlurred => RTRaw,
            RTRaw => Motion,
            Motion => Off,
        }
    }
}
impl CycleNext for DebugViewState {
    fn next(&self) -> Self {
        DebugViewState::next(self)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum InternalMenu {
    None,
    MainMenu,
}
impl Default for InternalMenu {
    fn default() -> Self {
        Self::MainMenu
    }
}
impl CycleNext for InternalMenu {
    fn next(&self) -> Self {
        match self {
            InternalMenu::None => InternalMenu::MainMenu,
            InternalMenu::MainMenu => InternalMenu::None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum LodCenterType {
    Eye,
    Target,
}
impl Default for LodCenterType {
    fn default() -> Self {
        Self::Target
    }
}
impl CycleNext for LodCenterType {
    fn next(&self) -> Self {
        match self {
            LodCenterType::Eye => LodCenterType::Target,
            LodCenterType::Target => LodCenterType::Eye,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub enum ShadowType {
    OFF,
    CSM,
    RT,
}
impl Default for ShadowType {
    fn default() -> Self {
        Self::CSM
    }
}
impl ShadowType {
    pub fn next(&self) -> Self {
        use crate::data::ShadowType::*;
        match self {
            OFF => CSM,
            CSM => RT,
            RT => OFF,
        }
    }
}
impl CycleNext for ShadowType {
    fn next(&self) -> Self {
        ShadowType::next(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SettingKind {
    Bool,
    Cycle,
    Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum SettingValue {
    F32(f32),
    F64(f64),
    U32(u32),
    Bool(bool),
    Vec4([f32; 4]),
    ChunkSize(ChunkSize),
    ToneMappingState(ToneMappingState),
    WorldPos(WorldPos),
    PresentModeSetting(PresentModeSetting),
    BendMode(BendMode),
    ShadowType(ShadowType),
    DebugViewState(DebugViewState),
    InternalMenu(InternalMenu),
    LodCenterType(LodCenterType),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SettingOp {
    Toggle,
    CycleNext,
    Set(SettingValue),
}

macro_rules! apply_setting_arm {
    (Bool, $s:ident, $field:ident, $val:ident, $op:ident) => {
        match $op {
            SettingOp::Toggle => $s.$field = !$s.$field,
            SettingOp::Set(SettingValue::$val(v)) => $s.$field = v,
            _ => {}
        }
    };
    (Cycle, $s:ident, $field:ident, $val:ident, $op:ident) => {
        match $op {
            SettingOp::CycleNext => $s.$field = CycleNext::next(&$s.$field),
            SettingOp::Set(SettingValue::$val(v)) => $s.$field = v,
            _ => {}
        }
    };
    (Value, $s:ident, $field:ident, $val:ident, $op:ident) => {
        if let SettingOp::Set(SettingValue::$val(v)) = $op {
            $s.$field = v;
        }
    };
}

macro_rules! define_settings {
    ($(
        $key:ident => $field:ident : $ty:ty = $default:expr ; $kind:ident ; $val:ident
    ),* $(,)?) => {
        #[derive(Debug, Deserialize, Serialize, Clone)]
        #[serde(default)]
        pub struct Settings {
            $(pub $field: $ty,)*
        }

        impl Default for Settings {
            fn default() -> Self {
                Self {
                    $($field: $default,)*
                }
            }
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
        #[serde(rename_all = "snake_case")]
        pub enum SettingKey {
            $($key,)*
        }

        impl SettingKey {
            pub fn kind(self) -> SettingKind {
                match self {
                    $(SettingKey::$key => SettingKind::$kind,)*
                }
            }

            pub fn read(self, s: &Settings) -> SettingValue {
                match self {
                    $(SettingKey::$key => SettingValue::$val(s.$field.clone()),)*
                }
            }

            pub fn apply(self, s: &mut Settings, op: SettingOp) {
                match self {
                    $(SettingKey::$key => apply_setting_arm!($kind, s, $field, $val, op),)*
                }
            }
        }

        impl Settings {
            pub fn read_setting(&self, key: SettingKey) -> SettingValue {
                key.read(self)
            }

            pub fn apply_setting(&mut self, key: SettingKey, op: SettingOp) {
                key.apply(self, op)
            }
        }
    };
}

define_settings! {
    TargetFps => target_fps: f32 = 60.0; Value; F32,
    TargetTps => target_tps: f32 = 60.0; Value; F32,
    PresentMode => present_mode: PresentModeSetting = PresentModeSetting::Mailbox; Cycle; PresentModeSetting,
    EditorMode => editor_mode: bool = false; Bool; Bool,
    OverrideMode => override_mode: bool = false; Bool; Bool,
    ShowGui => show_gui: bool = true; Bool; Bool,
    BackgroundColor => background_color: [f32; 4] = [0.0, 0.0, 0.0, 1.0]; Value; Vec4,
    TotalGameTime => total_game_time: f64 = 0.0; Value; F64,
    WorldGenerationBenchmarkMode => world_generation_benchmark_mode: bool = false; Bool; Bool,
    BendMode => bend_mode: BendMode = BendMode::Strict; Cycle; BendMode,
    ShowWorld => show_world: bool = true; Bool; Bool,
    AlwaysDay => always_day: bool = false; Bool; Bool,
    MsaaSamples => msaa_samples: u32 = 4; Value; U32,
    ShadowMapSize => shadow_map_size: u32 = 4096; Value; U32,
    ShadowType => shadow_type: ShadowType = ShadowType::default(); Cycle; ShadowType,
    GtaoEnabled => gtao_enabled: bool = true; Bool; Bool,
    ZoomSpeed => zoom_speed: f32 = 10.0; Value; F32,
    RenderLanesGizmo => render_lanes_gizmo: bool = false; Bool; Bool,
    RenderPartitionsGizmo => render_partitions_gizmo: bool = false; Bool; Bool,
    RenderChunkBounds => render_chunk_bounds: bool = false; Bool; Bool,
    ChunkSize => chunk_size: ChunkSize = default_chunk_size(); Value; ChunkSize,
    TonemappingState => tonemapping_state: ToneMappingState = ToneMappingState::default(); Value; ToneMappingState,
    DebugViewState => debug_view_state: DebugViewState = DebugViewState::Off; Cycle; DebugViewState,
    StartingMenu => starting_menu: InternalMenu = InternalMenu::default(); Cycle; InternalMenu,
    LodCenter => lod_center: LodCenterType = LodCenterType::default(); Cycle; LodCenterType,
    ReversedDepthZ => reversed_depth_z: bool = true; Bool; Bool,
    ShowFog => show_fog: bool = true; Bool; Bool,
    PlayerPos => player_pos: WorldPos = WorldPos::default(); Value; WorldPos,
    DriveCar => drive_car: bool = false; Bool; Bool,
    RenderRtGizmo => render_rt_gizmo: bool = false; Bool; Bool,
    Noclip => noclip: bool = false; Bool; Bool,
}

impl Settings {
    pub(crate) fn is_gtao_prep_off(&self) -> bool {
        !self.gtao_enabled
            && (self.shadow_type == ShadowType::OFF || self.shadow_type == ShadowType::CSM)
            && !self.show_fog
    }
}

impl Settings {
    pub fn load<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref();

        let mut settings = match fs::read_to_string(path) {
            Ok(content) => match toml::from_str::<Settings>(&content) {
                Ok(settings) => settings,
                Err(err) => {
                    eprintln!("Error parsing {:?}: {err}", path);
                    Self::default()
                }
            },
            Err(_) => {
                eprintln!("No settings file found, creating default {:?}", path);
                let default = Self::default();
                if let Ok(toml_str) = toml::to_string_pretty(&default) {
                    let _ = fs::write(path, toml_str);
                }
                default
            }
        };
        match settings.starting_menu {
            InternalMenu::None => {}
            InternalMenu::MainMenu => settings.show_world = false,
        }
        settings
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let path = path.as_ref();

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let toml_str = toml::to_string_pretty(self)?;
        fs::write(path, toml_str)?;

        Ok(())
    }
}
