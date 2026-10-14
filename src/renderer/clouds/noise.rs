use crate::helpers::modpack::ModManager;
use std::path::Path;
use wgpu::util::DeviceExt;
use wgpu::{
    AddressMode, Buffer, BufferUsages, CommandEncoderDescriptor, Device, Extent3d, FilterMode,
    MipmapFilterMode, Queue, Sampler, SamplerDescriptor, Texture, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};
use wgpu_render_manager::compute_system::{BufferSet, ComputePipelineOptions};
use wgpu_render_manager::renderer::RenderManager;

const WEATHER_SIZE: u32 = 512;
const SHAPE_SIZE: u32 = 128;
const DETAIL_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CloudNoiseParams {
    size: u32,
    seed: u32,
    frequency: u32,
    padding: u32,
}

struct CloudTexture {
    _texture: Texture,
    sampled: TextureView,
    mip_views: Vec<TextureView>,
}

impl CloudTexture {
    fn new(device: &Device, label: &'static str, size: u32, dimension: TextureDimension) -> Self {
        let mip_count = u32::BITS - size.leading_zeros();

        let texture = device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: if dimension == TextureDimension::D3 {
                    size
                } else {
                    1
                },
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let sampled = texture.create_view(&TextureViewDescriptor::default());

        let mip_views = (0..mip_count)
            .map(|level| {
                texture.create_view(&TextureViewDescriptor {
                    label: Some(label),
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        Self {
            _texture: texture,
            sampled,
            mip_views,
        }
    }
}

pub struct CloudNoiseResources {
    weather_texture: CloudTexture,
    shape_texture: CloudTexture,
    detail_texture: CloudTexture,
    pub sampler: Sampler,
}

impl CloudNoiseResources {
    pub fn new(
        device: &Device,
        queue: &Queue,
        render_manager: &mut RenderManager,
        mod_manager: &ModManager,
    ) -> Self {
        let weather_texture = CloudTexture::new(
            device,
            "Cloud Weather Map",
            WEATHER_SIZE,
            TextureDimension::D2,
        );

        let shape_texture = CloudTexture::new(
            device,
            "Cloud Shape Noise",
            SHAPE_SIZE,
            TextureDimension::D3,
        );

        let detail_texture = CloudTexture::new(
            device,
            "Cloud Detail Noise",
            DETAIL_SIZE,
            TextureDimension::D3,
        );

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("Cloud Noise Repeat Sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            ..Default::default()
        });

        let weather_shader = mod_manager
            .resource_path("shaders/compute/cloud_weather_generate.wgsl")
            .expect("Missing cloud_weather_generate.wgsl");

        let volume_shader = mod_manager
            .resource_path("shaders/compute/cloud_volume_generate.wgsl")
            .expect("Missing cloud_volume_generate.wgsl");

        let weather_mip_shader = mod_manager
            .resource_path("shaders/compute/cloud_weather_mip.wgsl")
            .expect("Missing cloud_weather_mip.wgsl");

        let volume_mip_shader = mod_manager
            .resource_path("shaders/compute/cloud_volume_mip.wgsl")
            .expect("Missing cloud_volume_mip.wgsl");

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Cloud Noise Generation"),
        });

        let weather_params = Self::make_params(WEATHER_SIZE, 20261010, 4);

        render_manager.compute(
            Some(&mut encoder),
            "cloud_weather_generate",
            vec![],
            vec![&weather_texture.mip_views[0]],
            weather_shader,
            ComputePipelineOptions {
                dispatch_size: [WEATHER_SIZE.div_ceil(8), WEATHER_SIZE.div_ceil(8), 1],
            },
            &[BufferSet::from_uniform(&Self::make_params_buffer(
                device,
                &weather_params,
            ))],
        );

        let shape_params = Self::make_params(SHAPE_SIZE, 1337, 4);
        let shape_params_buffer = Self::make_params_buffer(device, &shape_params);

        render_manager.compute(
            Some(&mut encoder),
            "cloud_volume_generate",
            vec![],
            vec![&shape_texture.mip_views[0]],
            volume_shader,
            ComputePipelineOptions {
                dispatch_size: [
                    SHAPE_SIZE.div_ceil(4),
                    SHAPE_SIZE.div_ceil(4),
                    SHAPE_SIZE.div_ceil(4),
                ],
            },
            &[BufferSet::from_uniform(&shape_params_buffer)],
        );

        let detail_params = Self::make_params(DETAIL_SIZE, 7331, 8);
        let detail_params_buffer = Self::make_params_buffer(device, &detail_params);

        render_manager.compute(
            Some(&mut encoder),
            "cloud_detail_generate",
            vec![],
            vec![&detail_texture.mip_views[0]],
            volume_shader,
            ComputePipelineOptions {
                dispatch_size: [
                    DETAIL_SIZE.div_ceil(4),
                    DETAIL_SIZE.div_ceil(4),
                    DETAIL_SIZE.div_ceil(4),
                ],
            },
            &[BufferSet::from_uniform(&detail_params_buffer)],
        );

        Self::generate_weather_mips(
            render_manager,
            &mut encoder,
            &weather_texture,
            weather_mip_shader,
        );

        Self::generate_volume_mips(
            render_manager,
            &mut encoder,
            &shape_texture,
            volume_mip_shader,
        );

        Self::generate_volume_mips(
            render_manager,
            &mut encoder,
            &detail_texture,
            volume_mip_shader,
        );

        queue.submit(Some(encoder.finish()));

        Self {
            weather_texture,
            shape_texture,
            detail_texture,
            sampler,
        }
    }

    fn make_params(size: u32, seed: u32, frequency: u32) -> CloudNoiseParams {
        CloudNoiseParams {
            size,
            seed,
            frequency,
            padding: 0,
        }
    }

    fn make_params_buffer(device: &Device, params: &CloudNoiseParams) -> Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cloud Noise Parameters"),
            contents: bytemuck::bytes_of(params),
            usage: BufferUsages::UNIFORM,
        })
    }

    fn generate_weather_mips(
        render_manager: &mut RenderManager,
        encoder: &mut wgpu::CommandEncoder,
        texture: &CloudTexture,
        shader: &Path,
    ) {
        let mut size = WEATHER_SIZE;

        for level in 1..texture.mip_views.len() {
            size = (size / 2).max(1);

            render_manager.compute(
                Some(encoder),
                "cloud_weather_mip",
                vec![&texture.mip_views[level - 1]],
                vec![&texture.mip_views[level]],
                shader,
                ComputePipelineOptions {
                    dispatch_size: [size.div_ceil(8), size.div_ceil(8), 1],
                },
                &[],
            );
        }
    }

    fn generate_volume_mips(
        render_manager: &mut RenderManager,
        encoder: &mut wgpu::CommandEncoder,
        texture: &CloudTexture,
        shader: &Path,
    ) {
        let mut size = texture.mip_views.len() as u32;
        size = 1 << (size - 1);

        for level in 1..texture.mip_views.len() {
            size = (size / 2).max(1);

            render_manager.compute(
                Some(encoder),
                "cloud_volume_mip",
                vec![&texture.mip_views[level - 1]],
                vec![&texture.mip_views[level]],
                shader,
                ComputePipelineOptions {
                    dispatch_size: [size.div_ceil(4), size.div_ceil(4), size.div_ceil(4)],
                },
                &[],
            );
        }
    }

    pub fn weather_map(&self) -> &TextureView {
        &self.weather_texture.sampled
    }

    pub fn shape_noise(&self) -> &TextureView {
        &self.shape_texture.sampled
    }

    pub fn detail_noise(&self) -> &TextureView {
        &self.detail_texture.sampled
    }
}
