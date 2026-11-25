use glam::Vec3;
use revision::{
    DeserializeRevisioned, Revisioned, SerializeRevisioned, SkipRevisioned, revisioned,
};
use smallvec::SmallVec;
use std::io::{Read, Write};

#[revisioned(revision = 1)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SerializableVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
impl SerializableVec3 {
    pub fn as_vec3(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
    pub fn from_vec3(vec3: Vec3) -> SerializableVec3 {
        SerializableVec3 {
            x: vec3.x,
            y: vec3.y,
            z: vec3.z,
        }
    }
}
impl From<Vec3> for SerializableVec3 {
    fn from(v: Vec3) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

impl From<SerializableVec3> for Vec3 {
    fn from(v: SerializableVec3) -> Self {
        Vec3::new(v.x, v.y, v.z)
    }
}
impl PartialEq for SerializableVec3 {
    fn eq(&self, other: &Self) -> bool {
        self.x.to_bits() == other.x.to_bits()
            && self.y.to_bits() == other.y.to_bits()
            && self.z.to_bits() == other.z.to_bits()
    }
}

impl Eq for SerializableVec3 {}

impl std::hash::Hash for SerializableVec3 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.x.to_bits().hash(state);
        self.y.to_bits().hash(state);
        self.z.to_bits().hash(state);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RevisionedSmallVec<T, const N: usize>(pub SmallVec<T, N>);

impl<T, const N: usize> Revisioned for RevisionedSmallVec<T, N>
where
    T: Revisioned,
{
    fn revision() -> u16 {
        1
    }
}

impl<T, const N: usize> SerializeRevisioned for RevisionedSmallVec<T, N>
where
    T: SerializeRevisioned + 'static,
{
    #[inline]
    fn serialize_revisioned<W: Write>(&self, writer: &mut W) -> Result<(), revision::Error> {
        let len = self.0.len();

        len.serialize_revisioned(writer)?;

        for value in &self.0 {
            value.serialize_revisioned(writer)?;
        }

        Ok(())
    }
}

impl<T, const N: usize> DeserializeRevisioned for RevisionedSmallVec<T, N>
where
    T: DeserializeRevisioned + 'static,
{
    #[inline]
    fn deserialize_revisioned<R: Read>(reader: &mut R) -> Result<Self, revision::Error> {
        let len = usize::deserialize_revisioned(reader)?;

        let mut values = SmallVec::<T, N>::with_capacity(len);

        for _ in 0..len {
            values.push(T::deserialize_revisioned(reader)?);
        }

        Ok(Self(values))
    }
}
impl<T, const N: usize> SkipRevisioned for RevisionedSmallVec<T, N>
where
    T: SkipRevisioned,
{
    #[inline]
    fn skip_revisioned<R: Read>(reader: &mut R) -> Result<(), revision::Error> {
        let len = usize::deserialize_revisioned(reader)?;

        for _ in 0..len {
            T::skip_revisioned(reader)?;
        }

        Ok(())
    }
}

use std::ffi::OsString;
use std::sync::mpsc::{Receiver, Sender};
use image::{
    ColorType, ImageDecoder, ImageError, ImageResult,
    error::{DecodingError, ImageFormatHint},
    hooks::register_decoding_hook,
};
use resvg::{tiny_skia, usvg};

struct SvgDecoder {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

impl ImageDecoder for SvgDecoder {
    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn color_type(&self) -> ColorType {
        ColorType::Rgba8
    }

    fn total_bytes(&self) -> u64 {
        self.rgba.len() as u64
    }

    fn read_image(self, buf: &mut [u8]) -> ImageResult<()> {
        if buf.len() != self.rgba.len() {
            return Err(ImageError::Decoding(DecodingError::new(
                ImageFormatHint::Name("SVG".into()),
                "output buffer has incorrect size",
            )));
        }

        buf.copy_from_slice(&self.rgba);
        Ok(())
    }

    fn read_image_boxed(self: Box<Self>, buf: &mut [u8]) -> ImageResult<()> {
        (*self).read_image(buf)
    }
}

pub fn register_svg_decoding_hook() {
    register_decoding_hook(
        OsString::from("svg"),
        Box::new(|mut reader| {
            let mut data = Vec::new();
            reader.read_to_end(&mut data)?;

            let options = usvg::Options::default();

            let tree = usvg::Tree::from_data(&data, &options).map_err(|err| {
                ImageError::Decoding(DecodingError::new(ImageFormatHint::Name("SVG".into()), err))
            })?;

            let size = tree.size().to_int_size();
            let mut pixmap =
                tiny_skia::Pixmap::new(size.width(), size.height()).ok_or_else(|| {
                    ImageError::Decoding(DecodingError::new(
                        ImageFormatHint::Name("SVG".into()),
                        "failed to allocate SVG raster buffer",
                    ))
                })?;

            resvg::render(
                &tree,
                tiny_skia::Transform::identity(),
                &mut pixmap.as_mut(),
            );

            Ok(Box::new(SvgDecoder {
                width: size.width(),
                height: size.height(),
                rgba: pixmap.take_demultiplied(),
            }))
        }),
    );
}


use std::sync::{mpsc, OnceLock};
use std::time::Duration;
use all_smi::{AllSmi, device::GpuInfo as AllSmiGpuInfo};
use nvml_wrapper::enum_wrappers::device::Sampling;
use nvml_wrapper::enums::device::SampleValue;
use nvml_wrapper::Nvml;
use tracing::error;
use wgpu::AdapterInfo;
use crate::resources::{format_memory, Resources, GPU_TELEMETRY};
use crate::ui::parser::Value;
use crate::ui::variables::Variables;
fn get_rendering_gpu_static_info(
    adapter_info: &AdapterInfo,
    all_smi_gpu: &AllSmiGpuInfo,
    nvidia_identity: Option<&NvidiaGpuIdentity>,
) -> RenderingGpuStaticInfo {
    let vendor = GpuVendor::from_vendor_id(adapter_info.vendor);

    let pci_bus_id = {
        let id = adapter_info.device_pci_bus_id.trim();

        if id.is_empty() {
            None
        } else {
            Some(id.to_owned())
        }
    };

    let core_count = match vendor {
        GpuVendor::Nvidia => {
            nvidia_identity.and_then(|identity| identity.core_count)
        }
        GpuVendor::Apple => all_smi_gpu.gpu_core_count,
        _ => None,
    };

    let memory_bus_width = get_gpu_memory_bus_width(
        vendor,
        pci_bus_id.as_deref(),
    );

    RenderingGpuStaticInfo {
        name: adapter_info.name.clone(),
        vendor,
        pci_bus_id,
        core_count,
        memory_bus_width,
    }
}
fn get_gpu_memory_bus_width(
    vendor: GpuVendor,
    pci_bus_id: Option<&str>,
) -> Option<u16> {
    let pci_bus_id = pci_bus_id?;

    match vendor {
        GpuVendor::Nvidia => get_nvidia_memory_bus_width(pci_bus_id),
        GpuVendor::Amd => get_amd_memory_bus_width(pci_bus_id),
        GpuVendor::Intel => get_intel_memory_bus_width(pci_bus_id),
        GpuVendor::Apple | GpuVendor::Unknown => None,
    }
}
fn get_nvidia_memory_bus_width(pci_bus_id: &str) -> Option<u16> {
    let nvml = nvml_instance()?;
    let device = nvml.device_by_pci_bus_id(pci_bus_id).ok()?;
    let width = device.memory_bus_width().ok()?;

    u16::try_from(width).ok().filter(|&width| width > 0)
}

#[cfg(target_os = "linux")]
fn get_amd_memory_bus_width(pci_bus_id: &str) -> Option<u16> {
    use libdrm_amdgpu_sys::AMDGPU::GPU_INFO;
    use libdrm_amdgpu_sys::LibDrmAmdgpu;
    use std::fs::File;

    let render_node = format!("/dev/dri/by-path/pci-{pci_bus_id}-render");
    let file = File::open(render_node).ok()?;
    let libdrm = LibDrmAmdgpu::new().ok()?;
    let (device, _, _) = libdrm.init_device_handle_with_fd(file).ok()?;
    let info = device.device_info().ok()?;

    u16::try_from(info.vram_bit_width())
        .ok()
        .filter(|&width| width > 0)
}

#[cfg(not(target_os = "linux"))]
fn get_amd_memory_bus_width(_: &str) -> Option<u16> {
    None
}

fn get_intel_memory_bus_width(pci_bus_id: &str) -> Option<u16> {
    use std::process::Command;

    let output = Command::new("xpu-smi")
        .args([
            "query",
            "--json",
            "--device",
            pci_bus_id,
            "--display",
            "MEMORY",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let json: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;

    let gpu = json
        .get("gpus")?
        .as_array()?
        .iter()
        .find(|gpu| {
            gpu.get("pci_bus_id")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|id| id.eq_ignore_ascii_case(pci_bus_id))
        })?;

    let width = gpu.pointer("/memory_info/memory_bus_width")?;

    let width = width
        .as_u64()
        .or_else(|| width.as_str()?.parse::<u64>().ok())?;

    u16::try_from(width).ok().filter(|&width| width > 0)
}

pub fn parse_pci_bus_id(id: &str) -> Option<(u32, u32, u32, u32)> {
    let mut parts = id.trim().split([':', '.']);

    let domain = u32::from_str_radix(parts.next()?, 16).ok()?;
    let bus = u32::from_str_radix(parts.next()?, 16).ok()?;
    let device = u32::from_str_radix(parts.next()?, 16).ok()?;
    let function = u32::from_str_radix(parts.next()?, 16).ok()?;

    if parts.next().is_some() {
        return None;
    }

    Some((domain, bus, device, function))
}
#[derive(Debug, Clone)]
struct NvidiaGpuIdentity {
    uuid: String,
    core_count: Option<u32>,
}

fn get_nvidia_device_identity(
    adapter_info: &AdapterInfo,
) -> Option<NvidiaGpuIdentity> {
    if adapter_info.vendor != 0x10DE {
        return None;
    }

    let expected_pci_id = parse_pci_bus_id(&adapter_info.device_pci_bus_id)?;
    let nvml = nvml_instance()?;
    let device_count = nvml.device_count().ok()?;

    for index in 0..device_count {
        let Ok(device) = nvml.device_by_index(index) else {
            continue;
        };

        let Ok(pci_info) = device.pci_info() else {
            continue;
        };

        if parse_pci_bus_id(&pci_info.bus_id) != Some(expected_pci_id) {
            continue;
        }

        return Some(NvidiaGpuIdentity {
            uuid: device.uuid().ok()?,
            core_count: device.num_cores().ok(),
        });
    }

    None
}
fn all_smi_gpu_pci_id(gpu: &AllSmiGpuInfo) -> Option<(u32, u32, u32, u32)> {
    let pci_id = [
        "PCI Bus ID",
        "PCI Bus",
        "pci_bus_id",
        "pcie_address",
    ]
        .iter()
        .find_map(|key| gpu.detail.get(*key))?;

    parse_pci_bus_id(pci_id)
}
fn find_matching_gpu<'a>(
    gpus: &'a [AllSmiGpuInfo],
    adapter_info: &AdapterInfo,
    nvidia_uuid: Option<&str>,
) -> Option<&'a AllSmiGpuInfo> {
    if let Some(uuid) = nvidia_uuid {
        if let Some(gpu) = gpus.iter().find(|gpu| {
            gpu.device_type.eq_ignore_ascii_case("GPU") && gpu.uuid == uuid
        }) {
            return Some(gpu);
        }
    }

    let adapter_pci_id = parse_pci_bus_id(&adapter_info.device_pci_bus_id);

    if let Some(adapter_pci_id) = adapter_pci_id {
        let matches: Vec<_> = gpus
            .iter()
            .filter(|gpu| {
                gpu.device_type.eq_ignore_ascii_case("GPU")
                    && all_smi_gpu_pci_id(gpu) == Some(adapter_pci_id)
            })
            .collect();

        match matches.as_slice() {
            [gpu] => return Some(*gpu),
            [] => {}
            _ => return None,
        }
    }

    let adapter_name = adapter_info.name.trim();

    let matches: Vec<_> = gpus
        .iter()
        .filter(|gpu| {
            gpu.device_type.eq_ignore_ascii_case("GPU")
                && gpu.name.trim().eq_ignore_ascii_case(adapter_name)
        })
        .collect();

    match matches.as_slice() {
        [gpu] => Some(*gpu),
        _ => None,
    }
}
fn percentage_to_u8(value: f64) -> Option<u8> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }

    Some(value.round().clamp(0.0, 100.0) as u8)
}

fn watts_to_mw(value: f64) -> Option<u32> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }

    let milliwatts = (value * 1000.0).round();

    if milliwatts > u32::MAX as f64 {
        return None;
    }

    Some(milliwatts as u32)
}

fn detail_u8(gpu: &AllSmiGpuInfo, key: &str) -> Option<u8> {
    gpu.detail.get(key)?.parse::<u8>().ok()
}

fn detail_watts(gpu: &AllSmiGpuInfo, key: &str) -> Option<f64> {
    let value = gpu.detail.get(key)?.parse::<f64>().ok()?;

    if value.is_finite() && value >= 0.0 {
        Some(value)
    } else {
        None
    }
}
fn convert_gpu_telemetry(
    gpu: &AllSmiGpuInfo,
    nvidia: bool,
) -> GpuTelemetry {
    let load_pct = gpu
        .utilization_reading()
        .and_then(percentage_to_u8);

    let temperature_c = gpu
        .temperature_reading()
        .and_then(|value| i32::try_from(value).ok());

    let memory_total_bytes = (gpu.total_memory > 0).then_some(gpu.total_memory);

    let memory_used_bytes = memory_total_bytes.map(|_| gpu.used_memory);

    let memory_free_bytes = memory_total_bytes
        .and_then(|total| total.checked_sub(gpu.used_memory));

    let power_watts = gpu
        .power_consumption_reading()
        .filter(|value| value.is_finite() && *value >= 0.0)
        .filter(|value| !nvidia || *value > 0.0);

    let power_mw = power_watts.and_then(watts_to_mw);

    let power_limit_watts = detail_watts(gpu, "power_limit_current");

    let power_limit_mw = power_limit_watts.and_then(watts_to_mw);

    let power_usage_pct = match (power_mw, power_limit_mw) {
        (Some(power), Some(limit)) if limit > 0 => {
            percentage_to_u8(power as f64 / limit as f64 * 100.0)
        }
        _ => None,
    };

    let memory = match (memory_used_bytes, memory_total_bytes) {
        (Some(used), Some(total)) => {
            format!("{} / {}", format_memory(used), format_memory(total))
        }
        (Some(used), None) => format!("{} used", format_memory(used)),
        (None, Some(total)) => format!("Total: {}", format_memory(total)),
        (None, None) => "Unavailable".to_owned(),
    };

    let pcie_gen = detail_u8(gpu, "pcie_gen_current");
    let pcie_width = detail_u8(gpu, "pcie_width_current");

    GpuTelemetry {
        load_pct,
        temperature_c,
        memory_used_bytes,
        memory_total_bytes,
        memory_free_bytes,
        memory_utilization_pct: None,
        power_mw,
        power_limit_mw,
        power_usage_pct,
        fan_speed_pct: None,
        fan_rpm: gpu.fan_speed_rpm,
        current_pcie_width: pcie_width,
        current_pcie_gen: pcie_gen,
        compute_utilization: None,
        graphics_utilization: None,

        load: load_pct
            .map(|value| format!("{value}%"))
            .unwrap_or_else(|| "N/A".to_owned()),

        temperature: temperature_c
            .map(|value| format!("{value} °C"))
            .unwrap_or_else(|| "N/A".to_owned()),

        memory,

        power: power_watts
            .map(|value| format!("{value:.1} W"))
            .unwrap_or_else(|| "N/A".to_owned()),
    }
}
fn start_gpu_telemetry_worker(
    smi: AllSmi,
    mut gpu_info: AllSmiGpuInfo,
    nvidia: bool,
    pci_bus_id: Option<String>,
    sender: Sender<GpuTelemetry>,
) {
    let spawn_result = std::thread::Builder::new()
        .name("gpu-telemetry".to_owned())
        .spawn(move || {
            loop {
                if smi.refresh_gpu(&mut gpu_info) {
                    let mut telemetry = convert_gpu_telemetry(&gpu_info, nvidia);

                    if nvidia {
                        if let Some(pci_bus_id) = pci_bus_id.as_deref() {
                            telemetry.fan_speed_pct = get_nvidia_fan_speed_pct(pci_bus_id);
                            telemetry.fan_rpm = get_nvidia_fan_rpm(pci_bus_id);
                        }
                    }
                    if sender.send(telemetry).is_err() {
                        break;
                    }
                }

                std::thread::sleep(Duration::from_millis(500));
            }
        });

    if let Err(error) = spawn_result {
        eprintln!("Failed to start GPU telemetry thread: {error}");
    }
}
pub fn gpu_telemetry(
    adapter_info: &AdapterInfo,
    variables: &mut Variables,
) -> Receiver<GpuTelemetry> {
    variables.set_string("GPU_NAME", adapter_info.name.as_str());
    variables.set_string("GPU_VENDOR", "Unknown");
    variables.set_string("GPU_LOAD", "N/A");
    variables.set_string("GPU_TEMP_DISPLAY", "N/A");
    variables.set_string("GPU_MEMORY", "N/A");
    variables.set_string("GPU_POWER_DISPLAY", "N/A");

    variables.set_var("GPU_TEMP", Value::None);
    variables.set_var("GPU_POWER", Value::None);

    variables.set_bool("GPU_INTEGRATED", false);
    variables.set_bool("INEXACT_GPU_POWER", false);

    let (gpu_telemetry_tx, gpu_telemetry_rx) = mpsc::channel();

    let smi = match AllSmi::new() {
        Ok(smi) => smi,
        Err(error) => {
            error!("[GPU Info] AllSmi initialization failed: {}", error);
            return gpu_telemetry_rx;
        }
    };

    let nvidia_identity = get_nvidia_device_identity(adapter_info);

    let nvidia_uuid = nvidia_identity
        .as_ref()
        .map(|identity| identity.uuid.as_str());

    let gpus = smi.get_gpu_info();

    let Some(gpu_info) = find_matching_gpu(
        &gpus,
        adapter_info,
        nvidia_uuid,
    ).cloned() else {
        error!(
        "[GPU Info] No matching AllSmi GPU found for {}",
        adapter_info.name
    );
        return gpu_telemetry_rx;
    };

    let vendor = match adapter_info.vendor {
        0x10DE => "NVIDIA",
        0x1002 | 0x1022 => "AMD",
        0x8086 => "Intel",
        0x106B => "Apple",
        _ => "Unknown",
    };

    let core_name = match adapter_info.vendor {
        0x10DE => "CUDA Cores",
        0x1002 | 0x1022 => "Stream Processors",
        0x8086 => "Execution Units",
        0x106B => "GPU Cores",
        _ => "GPU Cores",
    };

    let integrated = gpu_info
        .detail
        .get("Interconnect")
        .is_some_and(|value| value.eq_ignore_ascii_case("Integrated"));

    let static_info = get_rendering_gpu_static_info(
        adapter_info,
        &gpu_info,
        nvidia_identity.as_ref(),
    );

    variables.set_string(
        "GPU_FULL_GOVERNMENT_NAME",
        static_info.name.as_str(),
    );

    variables.set_string(
        "GPU_CORE_NAME",
        static_info.vendor.core_name(),
    );

    variables.set_var(
        "GPU_CORE_COUNT",
        static_info.core_count.map_or(
            Value::None,
            |v| Value::I64(v as i64),
        ),
    );

    variables.set_var(
        "GPU_MEMORY_BUS_WIDTH",
        static_info.memory_bus_width.map_or(
            Value::None,
            |v| Value::I64(v as i64),
        ),
    );
    variables.set_string("GPU_NAME", gpu_info.name.as_str());
    variables.set_string("GPU_VENDOR", vendor);

    variables.set_bool("GPU_INTEGRATED", integrated);

    // error!(
    //     "[GPU Info] Selected {} ({}) UUID={} PCI={}",
    //     gpu_info.name,
    //     vendor,
    //     gpu_info.uuid,
    //     adapter_info.device_pci_bus_id
    // );

    start_gpu_telemetry_worker(
        smi,
        gpu_info,
        adapter_info.vendor == 0x10DE,
        Some(adapter_info.device_pci_bus_id.clone()),
        gpu_telemetry_tx,
    );

    gpu_telemetry_rx
}
pub fn nvml_instance() -> Option<&'static Nvml> {
    static NVML: OnceLock<Option<Nvml>> = OnceLock::new();
    NVML.get_or_init(|| Nvml::init().ok()).as_ref()
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Unknown,
}

impl GpuVendor {
    pub fn from_vendor_id(vendor_id: u32) -> Self {
        match vendor_id {
            0x10DE => Self::Nvidia,
            0x1002 | 0x1022 => Self::Amd,
            0x8086 => Self::Intel,
            0x106B => Self::Apple,
            _ => Self::Unknown,
        }
    }

    pub fn core_name(self) -> &'static str {
        match self {
            Self::Nvidia => "CUDA Cores",
            Self::Amd => "Stream Processors",
            Self::Intel => "Execution Units",
            Self::Apple => "GPU Cores",
            Self::Unknown => "GPU Cores",
        }
    }
}

#[derive(Debug, Clone)]
pub struct RenderingGpuStaticInfo {
    pub name: String,
    pub vendor: GpuVendor,
    pub pci_bus_id: Option<String>,
    pub core_count: Option<u32>,
    pub memory_bus_width: Option<u16>,
}

pub fn update_gpu_variables(resources: &mut Resources) {
    let Resources {
        ui,
        render_core,
        gpu_rx,
        gpu_telemetry,
        adapter,
        ..
    } = resources;

    let gpu_pci_bus_id = adapter.get_info().device_pci_bus_id;

    let proc_textures_total_bytes =
        render_core.render_manager.generator().gpu_memory_usage_bytes();

    let proc_textures_uncompressed_bytes =
        render_core.render_manager.generator().gpu_memory_usage_uncompressed_bytes();

    let variables = &mut ui.variables;

    let mut count = 0;

    while let Ok(gpu) = gpu_rx.try_recv() {
        if count >= 10 {
            error!(
                "GPU try_recv() had more than 10 values stored. Was there a >5 second long lagspike?"
            );
            break;
        }

        variables.set_string("GPU_LOAD", gpu.load.as_str());
        variables.set_string("GPU_TEMP_DISPLAY", gpu.temperature.as_str());
        variables.set_string("GPU_MEMORY", gpu.memory.as_str());

        variables.set_var(
            "GPU_PCIE_GEN",
            gpu.current_pcie_gen.map_or(Value::None, |v| Value::I64(v as i64)),
        );

        variables.set_var(
            "GPU_PCIE_LANES",
            gpu.current_pcie_width.map_or(Value::None, |v| Value::I64(v as i64)),
        );

        variables.set_var(
            "GPU_LOAD_PCT",
            gpu.load_pct.map_or(Value::None, |v| Value::I64(v as i64)),
        );

        variables.set_var(
            "GPU_TEMP",
            gpu.temperature_c.map_or(Value::None, |v| Value::I64(v as i64)),
        );

        variables.set_var(
            "USED_VRAM",
            gpu.memory_used_bytes.map_or(Value::None, |v| Value::I64(v as i64)),
        );

        variables.set_var(
            "TOTAL_VRAM",
            gpu.memory_total_bytes.map_or(Value::None, |v| Value::I64(v as i64)),
        );

        variables.set_var(
            "FREE_VRAM",
            gpu.memory_free_bytes.map_or(Value::None, |v| Value::I64(v as i64)),
        );

        variables.set_var(
            "GPU_MEMORY_UTILIZATION_PCT",
            gpu.memory_utilization_pct.map_or(Value::None, |v| Value::I64(v as i64)),
        );

        let power = if let Some(power) = gpu.power_mw {
            variables.set_bool("INEXACT_GPU_POWER", false);
            Some(power as f64 * 0.001)
        } else if let Some(power) = get_gpu_power_from_samples(&gpu_pci_bus_id) {
            variables.set_bool("INEXACT_GPU_POWER", true);
            Some(power)
        } else {
            variables.set_bool("INEXACT_GPU_POWER", false);
            None
        };

        if let Some(power) = power {
            variables.set_f64("GPU_POWER", power);
            variables.set_string("GPU_POWER_DISPLAY", &format!("{power:.1} W"));
        } else {
            variables.set_var("GPU_POWER", Value::None);
            variables.set_string("GPU_POWER_DISPLAY", "N/A");
        }

        variables.set_var(
            "GPU_POWER_LIMIT",
            gpu.power_limit_mw.map_or(
                Value::None,
                |v| Value::I64((v as f64 * 0.001) as i64),
            ),
        );

        variables.set_var(
            "GPU_POWER_USAGE_PCT",
            gpu.power_usage_pct.map_or(Value::None, |v| Value::I64(v as i64)),
        );

        variables.set_var(
            "GPU_FAN_SPEED_PCT",
            gpu.fan_speed_pct.map_or(Value::None, |v| Value::I64(v as i64)),
        );

        variables.set_var(
            "GPU_FAN_RPM",
            gpu.fan_rpm.map_or(Value::None, |v| Value::I64(v as i64)),
        );

        if let Ok(mut telemetry) = GPU_TELEMETRY.write() {
            *telemetry = gpu.clone();
        }

        *gpu_telemetry = gpu;

        count += 1;
    }

    variables.set_i64(
        "PROC_TEXTURES_TOTAL_BYTES",
        proc_textures_total_bytes as i64,
    );

    variables.set_i64(
        "PROC_TEXTURES_TOTAL_BYTES_HYPOTHETICAL_UNCOMPRESSED",
        proc_textures_uncompressed_bytes as i64,
    );
}
fn get_gpu_power_from_samples(pci_bus_id: &str) -> Option<f64> {
    let expected_pci_id = parse_pci_bus_id(pci_bus_id)?;
    let nvml = nvml_instance()?;
    let device_count = nvml.device_count().ok()?;

    for index in 0..device_count {
        let Ok(device) = nvml.device_by_index(index) else {
            continue;
        };

        let Ok(pci_info) = device.pci_info() else {
            continue;
        };

        if parse_pci_bus_id(&pci_info.bus_id) != Some(expected_pci_id) {
            continue;
        }

        let samples = device.samples(Sampling::Power, None).ok()?;

        let values: Vec<f64> = samples
            .iter()
            .filter_map(|sample| match &sample.value {
                SampleValue::U32(v) => Some(*v as f64),
                SampleValue::U64(v) => Some(*v as f64),
                SampleValue::I64(v) if *v >= 0 => Some(*v as f64),
                SampleValue::F64(v) if v.is_finite() && *v >= 0.0 => Some(*v),
                _ => None,
            })
            .collect();

        if values.is_empty() {
            return None;
        }

        return Some(
            values.iter().sum::<f64>() / values.len() as f64 / 1000.0
        );
    }

    None
}
#[derive(Default, Clone)]
pub struct GpuTelemetry {
    pub load_pct: Option<u8>,
    pub temperature_c: Option<i32>,
    pub memory_used_bytes: Option<u64>,
    pub memory_total_bytes: Option<u64>,
    pub memory_free_bytes: Option<u64>,
    pub memory_utilization_pct: Option<u8>,
    pub power_mw: Option<u32>,
    pub power_limit_mw: Option<u32>,
    pub power_usage_pct: Option<u8>,
    pub fan_speed_pct: Option<u8>,
    pub fan_rpm: Option<u32>,

    pub load: String,
    pub temperature: String,
    pub memory: String,
    pub power: String,
    pub current_pcie_width: Option<u8>,
    pub current_pcie_gen: Option<u8>,
    pub compute_utilization: Option<u8>,
    pub graphics_utilization: Option<u8>,
}
fn get_nvidia_fan_speed_pct(pci_bus_id: &str) -> Option<u8> {
    let nvml = nvml_instance()?;
    let target_pci = parse_pci_bus_id(pci_bus_id)?;
    let device_count = nvml.device_count().ok()?;

    for index in 0..device_count {
        let device = nvml.device_by_index(index).ok()?;
        let device_pci = device.pci_info().ok()?;

        if parse_pci_bus_id(&device_pci.bus_id) == Some(target_pci) {
            return device
                .fan_speed(0)
                .ok()
                .and_then(|speed| u8::try_from(speed).ok());
        }
    }

    None
}
fn get_nvidia_fan_rpm(pci_bus_id: &str) -> Option<u32> {
    let nvml = nvml_instance()?;
    let target_pci = parse_pci_bus_id(pci_bus_id)?;
    let device_count = nvml.device_count().ok()?;

    for index in 0..device_count {
        let device = nvml.device_by_index(index).ok()?;
        let device_pci = device.pci_info().ok()?;

        if parse_pci_bus_id(&device_pci.bus_id) == Some(target_pci) {
            return device.fan_speed_rpm(0).ok();
        }
    }

    None
}