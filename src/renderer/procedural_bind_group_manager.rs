use crate::data::Settings;
use crate::renderer::procedural_texture_manager::TextureCacheKey;
use std::collections::HashMap;
use wgpu::{
    AddressMode, BindGroup, BindGroupLayout, BindingResource, BindingType, CompareFunction, Device,
    FilterMode, MipmapFilterMode, Sampler, ShaderStages, TextureSampleType, TextureView,
    TextureViewDimension,
};
// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// pub enum MaterialKind {
//     Diffuse,
//     Normal,
//     Roughness,
//     Metallic,
//     AmbientOcclusion,
//     Emissive,
// }

pub struct MaterialBindGroupManager {
    device: Device,
    sampler: Sampler,
    shadow_sampler: Sampler,
    shadows_off_sampler: Sampler,
    layout_cache: HashMap<(usize, bool, bool), BindGroupLayout>,
    bind_group_cache: HashMap<(Vec<TextureCacheKey>, bool, bool, bool), BindGroup>,
}

impl MaterialBindGroupManager {
    pub fn new(device: Device) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            lod_max_clamp: f32::MAX,
            anisotropy_clamp: 16,
            ..Default::default()
        });
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Nearest,
            compare: Some(CompareFunction::LessEqual),
            ..Default::default()
        });
        let shadows_off_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadows Off Sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Nearest,
            compare: Some(CompareFunction::Always), // <--- Mission CRITICAL
            ..Default::default()
        });
        Self {
            device,
            sampler,
            shadow_sampler,
            shadows_off_sampler,
            layout_cache: HashMap::new(),
            bind_group_cache: HashMap::new(),
        }
    }

    pub fn clear_bind_groups(&mut self) {
        self.bind_group_cache.clear();
    }
    pub fn get_layout(
        &mut self,
        material_count: usize,
        shadow_pass: bool,
        fullscreen_pass: bool,
    ) -> &BindGroupLayout {
        if !self
            .layout_cache
            .contains_key(&(material_count, shadow_pass, fullscreen_pass))
        {
            let layout = self.create_layout(material_count, shadow_pass, fullscreen_pass);
            self.layout_cache
                .insert((material_count, shadow_pass, fullscreen_pass), layout);
        }

        self.layout_cache
            .get(&(material_count, shadow_pass, fullscreen_pass))
            .unwrap()
    }

    pub fn request_bind_group(
        &mut self,
        materials: &Vec<TextureCacheKey>,
        views: Vec<&TextureView>,
        shadow_array_view: &TextureView,
        shadow_pass: bool,
        fullscreen_pass: bool,
        settings: &Settings,
    ) -> &BindGroup {
        // Fullscreen tonemap: we expect exactly 1 source texture view (resolved HDR)
        if fullscreen_pass {
            assert_eq!(views.len(), 1);
        } else {
            assert_eq!(materials.len(), views.len());
        }

        let material_count = if fullscreen_pass { 1 } else { materials.len() };

        let key = (
            materials.to_vec(),
            shadow_pass,
            fullscreen_pass,
            settings.shadows_enabled,
        );

        if !self
            .layout_cache
            .contains_key(&(material_count, shadow_pass, fullscreen_pass))
        {
            let layout = self.create_layout(material_count, shadow_pass, fullscreen_pass);
            self.layout_cache
                .insert((material_count, shadow_pass, fullscreen_pass), layout);
        }

        if !self.bind_group_cache.contains_key(&key) {
            let bind_group = self.build_bind_group(
                material_count,
                views,
                shadow_array_view,
                shadow_pass,
                fullscreen_pass,
                settings,
            );
            self.bind_group_cache.insert(key.clone(), bind_group);
        }

        self.bind_group_cache.get(&key).unwrap()
    }

    fn create_layout(
        &self,
        material_count: usize,
        shadow_pass: bool,
        fullscreen_pass: bool,
    ) -> BindGroupLayout {
        let mut entries = Vec::new();

        // Shadow pass: no material bindings
        if shadow_pass {
            // leave empty
        }
        // Fullscreen pass (tonemap): fixed bindings:
        // @group(0) @binding(0) texture_2d<f32>
        // @group(0) @binding(1) sampler
        else if fullscreen_pass {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        // Normal forward/gbuffer/etc: your existing material + shadow bindings
        else {
            for i in 0..material_count {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                });
            }

            entries.push(wgpu::BindGroupLayoutEntry {
                binding: material_count as u32,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });

            entries.push(wgpu::BindGroupLayoutEntry {
                binding: (material_count + 1) as u32,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Depth,
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            });

            entries.push(wgpu::BindGroupLayoutEntry {
                binding: (material_count + 2) as u32,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            });
        }

        self.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Material Bind Group Layout"),
                entries: &entries,
            })
    }

    fn build_bind_group(
        &self,
        material_count: usize,
        views: Vec<&TextureView>,
        shadow_array_view: &TextureView,
        shadow_pass: bool,
        fullscreen_pass: bool,
        settings: &Settings,
    ) -> BindGroup {
        let layout = self
            .layout_cache
            .get(&(material_count, shadow_pass, fullscreen_pass))
            .unwrap();

        let mut entries: Vec<wgpu::BindGroupEntry> = Vec::new();

        if shadow_pass {
            // no entries
        } else if fullscreen_pass {
            // binding(0) = hdr_tex, binding(1) = hdr_sampler
            entries.push(wgpu::BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(views[0]),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(&self.sampler),
            });
        } else {
            for (i, view) in views.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: BindingResource::TextureView(view),
                });
            }

            entries.push(wgpu::BindGroupEntry {
                binding: material_count as u32,
                resource: BindingResource::Sampler(&self.sampler),
            });

            entries.push(wgpu::BindGroupEntry {
                binding: (material_count + 1) as u32,
                resource: BindingResource::TextureView(shadow_array_view),
            });

            entries.push(wgpu::BindGroupEntry {
                binding: (material_count + 2) as u32,
                resource: BindingResource::Sampler(if settings.shadows_enabled {
                    &self.shadow_sampler
                } else {
                    &self.shadows_off_sampler
                }),
            });
        }

        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Material Bind Group"),
            layout,
            entries: &entries,
        })
    }
}
