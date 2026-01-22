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
    layout_cache: HashMap<(usize, bool), BindGroupLayout>,
    bind_group_cache: HashMap<(Vec<TextureCacheKey>, bool), BindGroup>,
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
            compare: Some(CompareFunction::LessEqual), // <--- CRITICAL
            ..Default::default()
        });
        Self {
            device,
            sampler,
            shadow_sampler,
            layout_cache: HashMap::new(),
            bind_group_cache: HashMap::new(),
        }
    }

    pub fn get_layout(&mut self, material_count: usize, shadow_pass: bool) -> &BindGroupLayout {
        if !self
            .layout_cache
            .contains_key(&(material_count, shadow_pass))
        {
            let layout = self.create_layout(material_count, shadow_pass);
            self.layout_cache
                .insert((material_count, shadow_pass), layout);
        }
        self.layout_cache
            .get(&(material_count, shadow_pass))
            .unwrap()
    }

    pub fn request_bind_group(
        &mut self,
        materials: &Vec<TextureCacheKey>,
        views: Vec<&TextureView>,
        shadow_array_view: &TextureView,
        shadow_pass: bool,
    ) -> &BindGroup {
        assert_eq!(materials.len(), views.len());

        let key = (materials.to_vec(), shadow_pass);
        let material_count = materials.len();

        if !self
            .layout_cache
            .contains_key(&(material_count, shadow_pass))
        {
            let layout = self.create_layout(material_count, shadow_pass);
            self.layout_cache
                .insert((material_count, shadow_pass), layout);
        }

        if !self.bind_group_cache.contains_key(&key) {
            let bind_group =
                self.build_bind_group(material_count, views, shadow_array_view, shadow_pass);
            self.bind_group_cache.insert(key.clone(), bind_group);
        }

        self.bind_group_cache.get(&key).unwrap()
    }

    fn create_layout(&self, material_count: usize, shadow_pass: bool) -> BindGroupLayout {
        let mut entries = Vec::new();
        if !shadow_pass {
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

            // Binding N: Standard Sampler
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: material_count as u32,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
            // Binding N+1: The Shadow Map Texture
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
            // Binding N+2: Shadow Sampler (Comparison)
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
    ) -> BindGroup {
        let layout = self
            .layout_cache
            .get(&(material_count, shadow_pass))
            .unwrap();
        let mut entries: Vec<wgpu::BindGroupEntry> = Vec::new();

        if !shadow_pass {
            // Bind material textures
            for (i, view) in views.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: BindingResource::TextureView(view),
                });
            }

            // Bind Standard Sampler
            entries.push(wgpu::BindGroupEntry {
                binding: material_count as u32,
                resource: BindingResource::Sampler(&self.sampler),
            });

            // Bind Shadow Map Texture
            entries.push(wgpu::BindGroupEntry {
                binding: (material_count + 1) as u32,
                resource: BindingResource::TextureView(shadow_array_view),
            });

            // Bind Shadow Comparison Sampler
            entries.push(wgpu::BindGroupEntry {
                binding: (material_count + 2) as u32,
                resource: BindingResource::Sampler(&self.shadow_sampler),
            });
        }

        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Road Material Bind Group"),
            layout,
            entries: &entries,
        })
    }
}
