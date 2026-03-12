use crate::helpers::positions::{ChunkSize, WorldPos};
use crate::renderer::pipelines::ToneMappingState;
use crate::ui::variables::UiValue;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};
use wgpu::*;

macro_rules! impl_cycle {
    ($ty:ty : $($variant:expr),+ $(,)?) => {
        impl Cycle for $ty {
            fn next(&self) -> Self {
                let variants: &[$ty] = &[$($variant),+];
                let idx = variants.iter()
                    .position(|v| std::mem::discriminant(v) == std::mem::discriminant(self))
                    .unwrap_or(0);
                variants[(idx + 1) % variants.len()].clone()
            }

            fn prev(&self) -> Self {
                let variants: &[$ty] = &[$($variant),+];
                let len = variants.len();
                let idx = variants.iter()
                    .position(|v| std::mem::discriminant(v) == std::mem::discriminant(self))
                    .unwrap_or(0);
                variants[(idx + len - 1) % len].clone()
            }
        }
    };
}
pub trait Cycle {
    fn next(&self) -> Self;
    fn prev(&self) -> Self;
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
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

impl_cycle!(BendMode: BendMode::Strict, BendMode::Bent);

impl_cycle!(PresentModeSetting:
    PresentModeSetting::Immediate,
    PresentModeSetting::Mailbox,
    PresentModeSetting::Fifo
);

impl_cycle!(DebugViewState:
    DebugViewState::Off,
    DebugViewState::Normals,
    DebugViewState::Depth,
    DebugViewState::GtaoRaw,
    DebugViewState::GtaoBlurred,
    DebugViewState::RTRaw,
    DebugViewState::Motion
);

impl_cycle!(InternalMenu: InternalMenu::None, InternalMenu::MainMenu);

impl_cycle!(LodCenterType: LodCenterType::Eye, LodCenterType::Target);

impl_cycle!(ShadowType: ShadowType::OFF, ShadowType::CSM, ShadowType::RT);

// ============ Simplified SettingValue ============

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
    Bool(bool),
    U16(u16),
    U32(u32),
    F32(f32),
    F64(f64),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    /// Serialized enum/struct as TOML string
    Enum(String),
}
impl SettingValue {
    pub fn to_ui_value(&self) -> UiValue {
        match self {
            SettingValue::Bool(v) => UiValue::Bool(*v),
            SettingValue::U16(v) => UiValue::I64(*v as i64),
            SettingValue::U32(v) => UiValue::I64(*v as i64),
            SettingValue::F32(v) => UiValue::F64(*v as f64),
            SettingValue::F64(v) => UiValue::F64(*v),
            SettingValue::Vec2(v) => UiValue::String(format!("[{}, {}]", v[0], v[1])),
            SettingValue::Vec3(v) => UiValue::String(format!("[{}, {}, {}]", v[0], v[1], v[2])),
            SettingValue::Vec4(v) => {
                UiValue::String(format!("[{}, {}, {}, {}]", v[0], v[1], v[2], v[3]))
            }
            SettingValue::Enum(v) => UiValue::String(v.clone()),
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            SettingValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
    pub fn multiply(&self, factor: impl Into<f64>) -> Option<SettingValue> {
        let f = factor.into();
        match self {
            SettingValue::Bool(_) => None,
            SettingValue::U16(v) => {
                let res = ((*v as u64) as f64 * f).round();
                let res = res.max(u16::MIN as f64).min(u16::MAX as f64);
                Some(SettingValue::U16(res as u16))
            }
            SettingValue::U32(v) => {
                let res = ((*v as u64) as f64 * f).round();
                let res = res.max(u32::MIN as f64).min(u32::MAX as f64);
                Some(SettingValue::U32(res as u32))
            }
            SettingValue::F32(v) => Some(SettingValue::F32(((*v as f64) * f) as f32)),
            SettingValue::F64(v) => Some(SettingValue::F64(*v * f)),
            SettingValue::Vec2(v) => Some(SettingValue::Vec2([
                ((v[0] as f64) * f) as f32,
                ((v[1] as f64) * f) as f32,
            ])),
            SettingValue::Vec3(v) => Some(SettingValue::Vec3([
                ((v[0] as f64) * f) as f32,
                ((v[1] as f64) * f) as f32,
                ((v[2] as f64) * f) as f32,
            ])),
            SettingValue::Vec4(v) => Some(SettingValue::Vec4([
                ((v[0] as f64) * f) as f32,
                ((v[1] as f64) * f) as f32,
                ((v[2] as f64) * f) as f32,
                ((v[3] as f64) * f) as f32,
            ])),
            SettingValue::Enum(_) => None,
        }
    }

    pub fn clamp_range(&self, min: impl Into<f64>, max: impl Into<f64>) -> Option<SettingValue> {
        let a = min.into();
        let b = max.into();
        let lo = a.min(b);
        let hi = a.max(b);
        match self {
            SettingValue::Bool(_) => None,
            SettingValue::U16(v) => {
                let val = (*v as u64) as f64;
                let clamped = val.max(lo).min(hi).round();
                let clamped = clamped.max(u16::MIN as f64).min(u16::MAX as f64);
                Some(SettingValue::U16(clamped as u16))
            }
            SettingValue::U32(v) => {
                let val = (*v as u64) as f64;
                let clamped = val.max(lo).min(hi).round();
                let clamped = clamped.max(u32::MIN as f64).min(u32::MAX as f64);
                Some(SettingValue::U32(clamped as u32))
            }
            SettingValue::F32(v) => {
                let val = *v as f64;
                let clamped = val.max(lo).min(hi);
                Some(SettingValue::F32(clamped as f32))
            }
            SettingValue::F64(v) => {
                let clamped = (*v).max(lo).min(hi);
                Some(SettingValue::F64(clamped))
            }
            SettingValue::Vec2(v) => Some(SettingValue::Vec2([
                (*v.get(0).unwrap() as f64).max(lo).min(hi) as f32,
                (*v.get(1).unwrap() as f64).max(lo).min(hi) as f32,
            ])),
            SettingValue::Vec3(v) => Some(SettingValue::Vec3([
                (*v.get(0).unwrap() as f64).max(lo).min(hi) as f32,
                (*v.get(1).unwrap() as f64).max(lo).min(hi) as f32,
                (*v.get(2).unwrap() as f64).max(lo).min(hi) as f32,
            ])),
            SettingValue::Vec4(v) => Some(SettingValue::Vec4([
                (*v.get(0).unwrap() as f64).max(lo).min(hi) as f32,
                (*v.get(1).unwrap() as f64).max(lo).min(hi) as f32,
                (*v.get(2).unwrap() as f64).max(lo).min(hi) as f32,
                (*v.get(3).unwrap() as f64).max(lo).min(hi) as f32,
            ])),
            SettingValue::Enum(_) => None,
        }
    }

    pub fn add(&self, amount: impl Into<f64>) -> Option<SettingValue> {
        let a = amount.into();
        match self {
            SettingValue::Bool(_) => None,
            SettingValue::U16(v) => {
                let res = ((*v as u64) as f64 + a).round();
                let res = res.max(0.0).min(u16::MAX as f64);
                Some(SettingValue::U16(res as u16))
            }
            SettingValue::U32(v) => {
                let res = ((*v as u64) as f64 + a).round();
                let res = res.max(0.0).min(u32::MAX as f64);
                Some(SettingValue::U32(res as u32))
            }
            SettingValue::F32(v) => Some(SettingValue::F32(((*v as f64) + a) as f32)),
            SettingValue::F64(v) => Some(SettingValue::F64(*v + a)),
            SettingValue::Vec2(v) => Some(SettingValue::Vec2([
                ((*v.get(0).unwrap() as f64) + a) as f32,
                ((*v.get(1).unwrap() as f64) + a) as f32,
            ])),
            SettingValue::Vec3(v) => Some(SettingValue::Vec3([
                ((*v.get(0).unwrap() as f64) + a) as f32,
                ((*v.get(1).unwrap() as f64) + a) as f32,
                ((*v.get(2).unwrap() as f64) + a) as f32,
            ])),
            SettingValue::Vec4(v) => Some(SettingValue::Vec4([
                ((*v.get(0).unwrap() as f64) + a) as f32,
                ((*v.get(1).unwrap() as f64) + a) as f32,
                ((*v.get(2).unwrap() as f64) + a) as f32,
                ((*v.get(3).unwrap() as f64) + a) as f32,
            ])),
            SettingValue::Enum(_) => None,
        }
    }

    pub fn subtract(&self, amount: impl Into<f64>) -> Option<SettingValue> {
        self.add(-amount.into())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SettingOp {
    Toggle,
    CycleNext,
    CyclePrev,
    Set(SettingValue),
}

// ============ Conversion Trait ============

// Helper for parsing enums from command strings
fn parse_enum_from_str<T: serde::de::DeserializeOwned>(s: &str) -> Option<T> {
    #[derive(Deserialize)]
    struct Wrapper<T> {
        value: T,
    }
    // Try with quotes (for simple enum variants)
    let toml_str = format!("value = \"{}\"", s);
    if let Ok(w) = toml::from_str::<Wrapper<T>>(&toml_str) {
        return Some(w.value);
    }
    // Try raw (for complex values)
    let toml_str = format!("value = {}", s);
    toml::from_str::<Wrapper<T>>(&toml_str)
        .ok()
        .map(|w| w.value)
}

// ============ Updated Conversion Trait ============

pub trait SettingConvert: Sized + Clone {
    fn to_setting_value(&self) -> SettingValue;
    fn from_setting_value(value: SettingValue) -> Option<Self>;
    fn from_ui_value(arg: &UiValue) -> Option<Self>;
}

// Primitives
impl SettingConvert for bool {
    fn to_setting_value(&self) -> SettingValue {
        SettingValue::Bool(*self)
    }
    fn from_setting_value(value: SettingValue) -> Option<Self> {
        match value {
            SettingValue::Bool(v) => Some(v),
            _ => None,
        }
    }
    fn from_ui_value(arg: &UiValue) -> Option<Self> {
        Some(arg.as_bool())
    }
}

impl SettingConvert for u16 {
    fn to_setting_value(&self) -> SettingValue {
        SettingValue::U16(*self)
    }
    fn from_setting_value(value: SettingValue) -> Option<Self> {
        match value {
            SettingValue::U16(v) => Some(v),
            _ => None,
        }
    }
    fn from_ui_value(arg: &UiValue) -> Option<Self> {
        arg.as_int().map(|i| i as u16)
    }
}

impl SettingConvert for u32 {
    fn to_setting_value(&self) -> SettingValue {
        SettingValue::U32(*self)
    }
    fn from_setting_value(value: SettingValue) -> Option<Self> {
        match value {
            SettingValue::U32(v) => Some(v),
            _ => None,
        }
    }
    fn from_ui_value(arg: &UiValue) -> Option<Self> {
        arg.as_int().map(|i| i as u32)
    }
}

impl SettingConvert for f32 {
    fn to_setting_value(&self) -> SettingValue {
        SettingValue::F32(*self)
    }
    fn from_setting_value(value: SettingValue) -> Option<Self> {
        match value {
            SettingValue::F32(v) => Some(v),
            _ => None,
        }
    }
    fn from_ui_value(arg: &UiValue) -> Option<Self> {
        match arg {
            UiValue::F64(v) => Some(*v as f32),
            UiValue::I64(v) => Some(*v as f32),
            other => None,
        }
    }
}

impl SettingConvert for f64 {
    fn to_setting_value(&self) -> SettingValue {
        SettingValue::F64(*self)
    }
    fn from_setting_value(value: SettingValue) -> Option<Self> {
        match value {
            SettingValue::F64(v) => Some(v),
            _ => None,
        }
    }
    fn from_ui_value(arg: &UiValue) -> Option<Self> {
        arg.as_float().map(|f| f)
    }
}

// Vectors - not easily settable from single CommandArg
impl SettingConvert for [f32; 2] {
    fn to_setting_value(&self) -> SettingValue {
        SettingValue::Vec2(*self)
    }
    fn from_setting_value(value: SettingValue) -> Option<Self> {
        match value {
            SettingValue::Vec2(v) => Some(v),
            _ => None,
        }
    }
    fn from_ui_value(_arg: &UiValue) -> Option<Self> {
        None
    }
}

impl SettingConvert for [f32; 3] {
    fn to_setting_value(&self) -> SettingValue {
        SettingValue::Vec3(*self)
    }
    fn from_setting_value(value: SettingValue) -> Option<Self> {
        match value {
            SettingValue::Vec3(v) => Some(v),
            _ => None,
        }
    }
    fn from_ui_value(_arg: &UiValue) -> Option<Self> {
        None
    }
}

impl SettingConvert for [f32; 4] {
    fn to_setting_value(&self) -> SettingValue {
        SettingValue::Vec4(*self)
    }
    fn from_setting_value(value: SettingValue) -> Option<Self> {
        match value {
            SettingValue::Vec4(v) => Some(v),
            _ => None,
        }
    }
    fn from_ui_value(_arg: &UiValue) -> Option<Self> {
        None
    }
}

// Enums and complex types
macro_rules! impl_setting_convert_enum {
    ($($ty:ty),* $(,)?) => {
        $(
            impl SettingConvert for $ty {
                fn to_setting_value(&self) -> SettingValue {
                    SettingValue::Enum(toml::to_string(self).unwrap_or_default())
                }
                fn from_setting_value(value: SettingValue) -> Option<Self> {
                    match value {
                        SettingValue::Enum(s) => toml::from_str(&s).ok(),
                        _ => None,
                    }
                }
                fn from_ui_value(arg: &UiValue) -> Option<Self> {
                    arg.as_str().and_then(parse_enum_from_str)
                }
            }
        )*
    };
}

impl_setting_convert_enum!(
    PresentModeSetting,
    BendMode,
    ShadowType,
    DebugViewState,
    InternalMenu,
    LodCenterType,
    ToneMappingState,
    WorldPos,
);

// ============ Settings Macros ============

macro_rules! apply_setting_arm {
    (Bool, $s:ident, $field:ident, $op:ident) => {
        match $op {
            SettingOp::Toggle => $s.$field = !$s.$field,
            SettingOp::Set(v) => {
                if let Some(val) = SettingConvert::from_setting_value(v) {
                    $s.$field = val;
                }
            }
            _ => {}
        }
    };
    (Cycle, $s:ident, $field:ident, $op:ident) => {
        match $op {
            SettingOp::CycleNext => $s.$field = Cycle::next(&$s.$field),
            SettingOp::CyclePrev => $s.$field = Cycle::prev(&$s.$field),
            SettingOp::Set(v) => {
                if let Some(val) = SettingConvert::from_setting_value(v) {
                    $s.$field = val;
                }
            }
            _ => {}
        }
    };
    (Value, $s:ident, $field:ident, $op:ident) => {
        if let SettingOp::Set(v) = $op {
            if let Some(val) = SettingConvert::from_setting_value(v) {
                $s.$field = val;
            }
        }
    };
}

macro_rules! define_settings {
    ($(
        $key:ident => $field:ident : $ty:ty = $default:expr ; $kind:ident
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
            /// Parse a snake_case string into a SettingKey
            pub fn from_str(s: &str) -> Option<Self> {
                match s {
                    $(stringify!($field) => Some(SettingKey::$key),)*
                    _ => None,
                }
            }

            pub fn kind(self) -> SettingKind {
                match self {
                    $(SettingKey::$key => SettingKind::$kind,)*
                }
            }

            pub fn read(self, s: &Settings) -> SettingValue {
                match self {
                    $(SettingKey::$key => SettingConvert::to_setting_value(&s.$field),)*
                }
            }

            pub fn apply(self, s: &mut Settings, op: SettingOp) {
                match self {
                    $(SettingKey::$key => apply_setting_arm!($kind, s, $field, op),)*
                }
            }

            /// Convert a CommandArg to the appropriate SettingValue for this key
            pub fn parse_command_arg(self, arg: &UiValue) -> Option<SettingValue> {
                match self {
                    $(SettingKey::$key => {
                        <$ty as SettingConvert>::from_ui_value(arg)
                            .map(|v| v.to_setting_value())
                    },)*
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

// ============ Settings Definition (simplified - no $val!) ============

fn default_chunk_size() -> ChunkSize {
    128
}

define_settings! {
    TargetFps => target_fps: f32 = 60.0; Value,
    TargetTps => target_tps: f32 = 60.0; Value,
    PresentMode => present_mode: PresentModeSetting = PresentModeSetting::Mailbox; Cycle,
    EditorMode => editor_mode: bool = false; Bool,
    OverrideMode => override_mode: bool = false; Bool,
    ShowGui => show_gui: bool = true; Bool,
    BackgroundColor => background_color: [f32; 4] = [0.0, 0.0, 0.0, 1.0]; Value,
    TotalGameTime => total_game_time: f64 = 0.0; Value,
    WorldGenerationBenchmarkMode => world_generation_benchmark_mode: bool = false; Bool,
    BendMode => bend_mode: BendMode = BendMode::Strict; Cycle,
    ShowWorld => show_world: bool = true; Bool,
    AlwaysDay => always_day: bool = false; Bool,
    MsaaSamples => msaa_samples: u32 = 4; Value,
    ShadowMapSize => shadow_map_size: u32 = 4096; Value,
    ShadowType => shadow_type: ShadowType = ShadowType::default(); Cycle,
    GtaoEnabled => gtao_enabled: bool = true; Bool,
    ZoomSpeed => zoom_speed: f32 = 10.0; Value,
    RenderLanesGizmo => render_lanes_gizmo: bool = false; Bool,
    RenderPartitionsGizmo => render_partitions_gizmo: bool = false; Bool,
    RenderChunkBounds => render_chunk_bounds: bool = false; Bool,
    ChunkSize => chunk_size: ChunkSize = default_chunk_size(); Value,
    TonemappingState => tonemapping_state: ToneMappingState = ToneMappingState::default(); Value,
    DebugViewState => debug_view_state: DebugViewState = DebugViewState::Off; Cycle,
    StartingMenu => starting_menu: InternalMenu = InternalMenu::default(); Cycle,
    LodCenter => lod_center: LodCenterType = LodCenterType::default(); Cycle,
    ReversedDepthZ => reversed_depth_z: bool = true; Bool,
    ShowFog => show_fog: bool = true; Bool,
    PlayerPos => player_pos: WorldPos = WorldPos::default(); Value,
    DriveCar => drive_car: bool = false; Bool,
    RenderRtGizmo => render_rt_gizmo: bool = false; Bool,
    Noclip => noclip: bool = false; Bool,
}

impl Settings {
    pub(crate) fn is_gtao_prep_off(&self) -> bool {
        !self.gtao_enabled
            && (self.shadow_type == ShadowType::OFF || self.shadow_type == ShadowType::CSM)
            && !self.show_fog
    }

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
        if let InternalMenu::MainMenu = settings.starting_menu {
            //settings.show_world = false;
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
