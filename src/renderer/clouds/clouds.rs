use crate::data::Cycle;
use crate::ui::parser::Value;
use std::time::Instant;
use strum_macros::Display;
use wgpu::{
    AddressMode, Buffer, BufferDescriptor, BufferUsages, Device, Extent3d, FilterMode, Queue,
    Sampler, SamplerDescriptor, SurfaceConfiguration, Texture, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};
use wgpu_render_manager::renderer::RenderManager;

use crate::helpers::modpack::ModManager;
use crate::helpers::positions::WorldPos;
use crate::impl_cycle;
use crate::renderer::clouds::cloud_terrain_height::CloudTerrainHeightMap;
use crate::renderer::clouds::noise::CloudNoiseResources;
use crate::world::terrain::terrain_subsystem::Terrain;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CloudQuality {
    Performance,
    #[default]
    High,
    Ultra,
}

impl CloudQuality {
    fn resolution_divisor(self) -> u32 {
        match self {
            Self::Performance | Self::High => 4,
            Self::Ultra => 2,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Display)]
#[repr(u32)]
pub enum CloudDebugMode {
    #[default]
    Lit = 0,
    Density = 1,
    DirectLight = 2,
    AmbientLight = 3,
    Transmittance = 4,
}
impl_cycle!(CloudDebugMode: CloudDebugMode::Lit, CloudDebugMode::Density, CloudDebugMode::DirectLight, CloudDebugMode::AmbientLight, CloudDebugMode::Transmittance);
#[derive(Clone, Copy, Debug)]
pub struct CloudSettings {
    pub enabled: bool,
    pub quality: CloudQuality,
    pub coverage: f32,
    pub density: f32,
    pub detail_strength: f32,
    pub wind: [f32; 2],
    pub debug_mode: CloudDebugMode,
}

impl Default for CloudSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            quality: CloudQuality::High,
            coverage: 0.4,
            density: 1.0,
            detail_strength: 1.0,
            wind: [80.0, 50.0],
            debug_mode: CloudDebugMode::Lit,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CloudUniforms {
    pub screen_size: [f32; 2],
    pub time: f32,
    pub coverage: f32,
    pub density: f32,
    pub detail_strength: f32,
    pub wind: [f32; 2],
    pub camera_height: f32,
    pub enabled: u32,
    pub frame_index: u32,
    pub history_valid: u32,
    pub terrain_map_origin_relative: [f32; 2],
    pub padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CloudTemporalUniforms {
    previous_time: f32,
    history_valid: u32,
    padding: [u32; 2],
}

struct CloudTarget {
    _texture: Texture,
    view: TextureView,
}

impl CloudTarget {
    fn new(device: &Device, width: u32, height: u32, label: &'static str) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&TextureViewDescriptor::default());

        Self {
            _texture: texture,
            view,
        }
    }
}

pub struct CloudSubsystem {
    pub cloud_noise: CloudNoiseResources,
    pub settings: CloudSettings,
    terrain_height_map: CloudTerrainHeightMap,
    pub uniform_buffer: Buffer,
    pub sampler: Sampler,
    trace: CloudTarget,
    trace_depth: CloudTarget,
    history: [CloudTarget; 2],
    trace_size: [u32; 2],
    history_index: usize,
    history_valid: bool,
    previous_time: f32,
    temporal_uniform_buffer: Buffer,
}

impl CloudSubsystem {
    pub fn new(
        device: &Device,
        config: &SurfaceConfiguration,
        queue: &Queue,
        render_manager: &mut RenderManager,
        mod_manager: &ModManager,
    ) -> Self {
        let start = Instant::now();
        let cloud_noise = CloudNoiseResources::new(device, queue, render_manager, mod_manager);
        let duration = start.elapsed();
        println!("[Clouds] Cloud Noise Initialization Time: {:?}", duration);

        let settings = CloudSettings::default();
        let trace_size = Self::target_size(config, settings.quality);

        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Cloud Uniforms"),
            size: std::mem::size_of::<CloudUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let temporal_uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Cloud Temporal Uniforms"),
            size: std::mem::size_of::<CloudTemporalUniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("Cloud Linear Sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let trace = CloudTarget::new(device, trace_size[0], trace_size[1], "Cloud Trace");

        let trace_depth =
            CloudTarget::new(device, trace_size[0], trace_size[1], "Cloud Trace Depth");

        let history = [
            CloudTarget::new(device, trace_size[0], trace_size[1], "Cloud History A"),
            CloudTarget::new(device, trace_size[0], trace_size[1], "Cloud History B"),
        ];
        let terrain_height_map = CloudTerrainHeightMap::new(device);
        Self {
            cloud_noise,
            settings,
            terrain_height_map,
            uniform_buffer,
            sampler,
            trace,
            trace_depth,
            history,
            trace_size,
            history_index: 0,
            history_valid: false,
            previous_time: 0.0,
            temporal_uniform_buffer,
        }
    }

    fn target_size(config: &SurfaceConfiguration, quality: CloudQuality) -> [u32; 2] {
        let divisor = quality.resolution_divisor();
        [
            config.width.div_ceil(divisor).max(1),
            config.height.div_ceil(divisor).max(1),
        ]
    }

    pub fn resize(&mut self, device: &Device, config: &SurfaceConfiguration) {
        let target_size = Self::target_size(config, self.settings.quality);
        if target_size == self.trace_size {
            return;
        }

        self.trace = CloudTarget::new(device, target_size[0], target_size[1], "Cloud Trace");
        self.trace_depth =
            CloudTarget::new(device, target_size[0], target_size[1], "Cloud Trace Depth");

        self.history = [
            CloudTarget::new(device, target_size[0], target_size[1], "Cloud History A"),
            CloudTarget::new(device, target_size[0], target_size[1], "Cloud History B"),
        ];

        self.trace_size = target_size;
        self.history_index = 0;
        self.previous_time = 0.0;
        self.invalidate_history();
    }

    pub fn set_quality(
        &mut self,
        device: &Device,
        config: &SurfaceConfiguration,
        quality: CloudQuality,
    ) {
        if self.settings.quality == quality {
            return;
        }

        self.settings.quality = quality;
        self.invalidate_history();
        self.resize(device, config);
    }

    pub fn update_uniforms(
        &mut self,
        queue: &Queue,
        time: f32,
        frame_index: u32,
        camera_height: f32,
        terrain_map_origin_relative: [f32; 2],
        terrain_map_valid: bool,
    ) {
        let history_valid = self.history_valid && self.settings.enabled;
        let uniforms = CloudUniforms {
            screen_size: [self.trace_size[0] as f32, self.trace_size[1] as f32],
            time,
            coverage: self.settings.coverage.clamp(0.0, 1.0),
            density: self.settings.density.max(0.0),
            detail_strength: self.settings.detail_strength.clamp(0.0, 1.0),
            wind: self.settings.wind,
            camera_height,
            enabled: self.settings.enabled as u32,
            frame_index,
            history_valid: history_valid as u32,
            terrain_map_origin_relative,
            padding: [self.settings.debug_mode as u32, terrain_map_valid as u32],
        };

        let temporal_uniforms = CloudTemporalUniforms {
            previous_time: self.previous_time,
            history_valid: history_valid as u32,
            padding: [0; 2],
        };

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        queue.write_buffer(
            &self.temporal_uniform_buffer,
            0,
            bytemuck::bytes_of(&temporal_uniforms),
        );

        self.previous_time = time;
        if !self.settings.enabled {
            self.invalidate_history();
        }
    }
    pub fn set_debug_mode(&mut self, debug_mode: CloudDebugMode) {
        if self.settings.debug_mode != debug_mode {
            self.settings.debug_mode = debug_mode;
            self.invalidate_history();
        }
    }
    pub fn trace_view(&self) -> &TextureView {
        &self.trace.view
    }

    pub fn trace_depth_view(&self) -> &TextureView {
        &self.trace_depth.view
    }

    pub fn history_read_view(&self) -> &TextureView {
        &self.history[self.history_index].view
    }

    pub fn history_write_view(&self) -> &TextureView {
        &self.history[1 - self.history_index].view
    }

    pub fn temporal_uniform_buffer(&self) -> &Buffer {
        &self.temporal_uniform_buffer
    }

    pub fn trace_size(&self) -> [u32; 2] {
        self.trace_size
    }

    pub fn history_valid(&self) -> bool {
        self.history_valid
    }

    pub fn finish_temporal_frame(&mut self) {
        self.history_index = 1 - self.history_index;
        self.history_valid = true;
    }

    pub fn invalidate_history(&mut self) {
        self.history_valid = false;
    }

    pub fn update_terrain_height_map(
        &mut self,
        queue: &Queue,
        terrain: &Terrain,
        camera_world: WorldPos,
    ) {
        if !self.settings.enabled {
            return;
        }
        self.terrain_height_map.update(queue, terrain, camera_world);
    }

    pub fn terrain_height_view(&self) -> &TextureView {
        self.terrain_height_map.view()
    }

    pub fn terrain_height_origin(&self, camera_world: WorldPos) -> ([f32; 2], bool) {
        self.terrain_height_map.origin_relative_to(camera_world)
    }
}
