use crate::renderer::procedural_texture_manager::TextureCacheKey;
use std::collections::HashMap;
use wgpu::{
    AddressMode, BindGroup, BindGroupLayout, BindingResource, BindingType, Device, FilterMode,
    MipmapFilterMode, Sampler, ShaderStages, TextureSampleType, TextureView, TextureViewDimension,
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
    layout_cache: HashMap<usize, BindGroupLayout>,
    bind_group_cache: HashMap<Vec<TextureCacheKey>, BindGroup>,
}

impl MaterialBindGroupManager {
    pub fn new(device: Device) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Nearest,
            lod_max_clamp: 0.0,
            anisotropy_clamp: 1,
            ..Default::default()
        });

        Self {
            device,
            sampler,
            layout_cache: HashMap::new(),
            bind_group_cache: HashMap::new(),
        }
    }

    pub fn get_layout(&mut self, material_count: usize) -> &BindGroupLayout {
        if !self.layout_cache.contains_key(&material_count) {
            let layout = self.create_layout(material_count);
            self.layout_cache.insert(material_count, layout);
        }
        self.layout_cache.get(&material_count).unwrap()
    }

    pub fn request_bind_group(
        &mut self,
        materials: &Vec<TextureCacheKey>,
        views: Vec<&TextureView>,
    ) -> &BindGroup {
        assert_eq!(materials.len(), views.len());

        let key = materials.to_vec();
        let material_count = materials.len();

        if !self.layout_cache.contains_key(&material_count) {
            let layout = self.create_layout(material_count);
            self.layout_cache.insert(material_count, layout);
        }

        if !self.bind_group_cache.contains_key(&key) {
            let bind_group = self.build_bind_group(material_count, views);
            self.bind_group_cache.insert(key.clone(), bind_group);
        }

        self.bind_group_cache.get(&key).unwrap()
    }

    fn create_layout(&self, material_count: usize) -> BindGroupLayout {
        let mut entries = Vec::with_capacity(material_count + 1);

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

        self.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &entries,
            })
    }

    fn build_bind_group(&self, material_count: usize, views: Vec<&TextureView>) -> BindGroup {
        let layout = self.layout_cache.get(&material_count).unwrap();

        let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(views.len() + 1);

        for (i, view) in views.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: BindingResource::TextureView(view),
            });
        }

        entries.push(wgpu::BindGroupEntry {
            binding: views.len() as u32,
            resource: BindingResource::Sampler(&self.sampler),
        });

        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &entries,
        })
    }
}
