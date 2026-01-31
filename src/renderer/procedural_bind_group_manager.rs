use crate::data::Settings;
use crate::renderer::textures::procedural_texture_manager::TextureCacheKey;
use std::collections::HashMap;
use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, CompareFunction,
    Device, FilterMode, MipmapFilterMode, Sampler, SamplerBindingType, ShaderStages,
    TextureSampleType, TextureView, TextureViewDimension,
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
#[derive(Eq, Hash, PartialEq, Clone, Copy, Debug)]
pub enum FullscreenPassType {
    /// Regular material path (N textures + sampler + shadow map)
    None,
    /// 1 color texture + sampler (tonemapping for example)
    Normal,
    /// 1 depth texture (no sampler; can be MSAA or not)
    Fog,
    // /// 1 depth texture, 1 normal texture (no sampler; can be MSAA or not)
    SsaoGen,
    SsaoBlur,
    SsaoApply,
}

#[derive(Eq, Hash, PartialEq, Clone, Copy, Debug)]
struct FullscreenLayoutKey {
    pass: FullscreenPassType,
    input_multisampled: bool,
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct MaterialBindGroupKey {
    materials: Vec<TextureCacheKey>,
    view_ptrs: Vec<usize>,
    shadow_pass: bool,
    fullscreen: FullscreenLayoutKey,
    shadows_enabled: bool,
}
pub struct MaterialBindGroupManager {
    device: Device,
    sampler: Sampler,
    shadow_sampler: Sampler,
    shadow_sampler_reversed_z: Sampler,
    shadows_off_sampler: Sampler,

    layout_cache: HashMap<(usize, bool, FullscreenLayoutKey), BindGroupLayout>,
    bind_group_cache: HashMap<MaterialBindGroupKey, BindGroup>,
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
        let shadow_sampler_reversed_z = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow Sampler (Reversed-Z)"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Nearest,
            compare: Some(CompareFunction::GreaterEqual), // <-- important
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
            shadow_sampler_reversed_z,
            shadows_off_sampler,
            layout_cache: HashMap::new(),
            bind_group_cache: HashMap::new(),
        }
    }
    pub fn clear_bind_groups(&mut self) {
        self.bind_group_cache.clear();
    }

    fn fullscreen_key(
        fullscreen_pass: FullscreenPassType,
        msaa_samples_for_input: u32,
    ) -> FullscreenLayoutKey {
        let input_multisampled = matches!(
            fullscreen_pass,
            FullscreenPassType::Fog | FullscreenPassType::SsaoGen | FullscreenPassType::SsaoBlur
        ) && msaa_samples_for_input > 1;

        FullscreenLayoutKey {
            pass: fullscreen_pass,
            input_multisampled,
        }
    }

    pub fn get_layout(
        &mut self,
        material_count: usize,
        shadow_pass: bool,
        fullscreen_pass: FullscreenPassType,
        msaa_samples: u32,
    ) -> &BindGroupLayout {
        let fk = Self::fullscreen_key(fullscreen_pass, msaa_samples);

        if !self
            .layout_cache
            .contains_key(&(material_count, shadow_pass, fk))
        {
            let layout = self.create_layout(material_count, shadow_pass, fk);
            self.layout_cache
                .insert((material_count, shadow_pass, fk), layout);
        }

        self.layout_cache
            .get(&(material_count, shadow_pass, fk))
            .unwrap()
    }

    pub fn request_bind_group(
        &mut self,
        materials: &Vec<TextureCacheKey>,
        views: Vec<&TextureView>,
        shadow_array_view: &TextureView,
        shadow_pass: bool,
        fullscreen_pass: FullscreenPassType,
        msaa_samples: u32,
        settings: &Settings,
    ) -> &BindGroup {
        let fk = Self::fullscreen_key(fullscreen_pass, msaa_samples);

        // Validate + determine layout material_count
        let material_count = match fullscreen_pass {
            FullscreenPassType::None => {
                assert_eq!(materials.len(), views.len());
                materials.len()
            }
            FullscreenPassType::Normal | FullscreenPassType::Fog => {
                assert_eq!(views.len(), 1);
                1
            }
            FullscreenPassType::SsaoGen => {
                assert_eq!(views.len(), 2);
                2
            }
            FullscreenPassType::SsaoBlur => {
                assert_eq!(views.len(), 3);
                3
            }
            FullscreenPassType::SsaoApply => {
                assert_eq!(views.len(), 1);
                1
            }
        };

        // Ensure layout exists
        if !self
            .layout_cache
            .contains_key(&(material_count, shadow_pass, fk))
        {
            let layout = self.create_layout(material_count, shadow_pass, fk);
            self.layout_cache
                .insert((material_count, shadow_pass, fk), layout);
        }

        let view_ptrs: Vec<usize> = views
            .iter()
            .map(|v| (*v as *const TextureView) as usize)
            .collect();

        let key = MaterialBindGroupKey {
            materials: materials.to_vec(),
            view_ptrs,
            shadow_pass,
            fullscreen: fk,
            shadows_enabled: settings.shadows_enabled,
        };

        if !self.bind_group_cache.contains_key(&key) {
            let bind_group = self.build_bind_group(
                material_count,
                views,
                shadow_array_view,
                shadow_pass,
                fk,
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
        fullscreen: FullscreenLayoutKey,
    ) -> BindGroupLayout {
        let mut entries = Vec::new();

        if shadow_pass {
            // shadow pass: intentionally empty
        } else {
            match fullscreen.pass {
                FullscreenPassType::Normal => {
                    // 1 color texture + sampler
                    entries.push(BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    });
                    entries.push(BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    });
                }

                FullscreenPassType::Fog => {
                    // 1 depth texture, MSAA depending on input
                    entries.push(BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Depth,
                            view_dimension: TextureViewDimension::D2,
                            multisampled: fullscreen.input_multisampled,
                        },
                        count: None,
                    });
                }
                FullscreenPassType::SsaoGen => {
                    // depth (MSAA depending on fk)
                    entries.push(BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Depth,
                            view_dimension: TextureViewDimension::D2,
                            multisampled: fullscreen.input_multisampled,
                        },
                        count: None,
                    });

                    // normals (resolved normals are single-sample)
                    entries.push(BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    });
                }

                FullscreenPassType::SsaoBlur => {
                    // ao input (single-sample)
                    entries.push(BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    });

                    // depth (MSAA depending on fk)
                    entries.push(BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Depth,
                            view_dimension: TextureViewDimension::D2,
                            multisampled: fullscreen.input_multisampled,
                        },
                        count: None,
                    });

                    // normals (single-sample resolved)
                    entries.push(BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    });
                }

                FullscreenPassType::SsaoApply => {
                    // blurred AO input (single-sample)
                    entries.push(BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    });
                }
                FullscreenPassType::None => {
                    // normal material path
                    for i in 0..material_count {
                        entries.push(BindGroupLayoutEntry {
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

                    entries.push(BindGroupLayoutEntry {
                        binding: material_count as u32,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    });

                    entries.push(BindGroupLayoutEntry {
                        binding: (material_count + 1) as u32,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Depth,
                            view_dimension: TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    });

                    entries.push(BindGroupLayoutEntry {
                        binding: (material_count + 2) as u32,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::Sampler(SamplerBindingType::Comparison),
                        count: None,
                    });
                }
            }
        }

        self.device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
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
        fullscreen: FullscreenLayoutKey,
        settings: &Settings,
    ) -> BindGroup {
        let layout = self
            .layout_cache
            .get(&(material_count, shadow_pass, fullscreen))
            .unwrap();

        let mut entries = Vec::new();

        if shadow_pass {
            // shadow pass: intentionally empty
        } else {
            match fullscreen.pass {
                FullscreenPassType::Normal => {
                    entries.push(BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(views[0]),
                    });
                    entries.push(BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&self.sampler),
                    });
                }

                FullscreenPassType::Fog => {
                    entries.push(BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(views[0]),
                    });
                }
                FullscreenPassType::SsaoGen => {
                    entries.push(BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(views[0]),
                    });
                    entries.push(BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(views[1]),
                    });
                }

                FullscreenPassType::SsaoBlur => {
                    entries.push(BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(views[0]),
                    }); // ao
                    entries.push(BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(views[1]),
                    }); // depth
                    entries.push(BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(views[2]),
                    }); // normals
                }

                FullscreenPassType::SsaoApply => {
                    entries.push(BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(views[0]),
                    });
                }
                FullscreenPassType::None => {
                    for (i, view) in views.iter().enumerate() {
                        entries.push(BindGroupEntry {
                            binding: i as u32,
                            resource: BindingResource::TextureView(view),
                        });
                    }

                    entries.push(BindGroupEntry {
                        binding: material_count as u32,
                        resource: BindingResource::Sampler(&self.sampler),
                    });

                    entries.push(BindGroupEntry {
                        binding: (material_count + 1) as u32,
                        resource: BindingResource::TextureView(shadow_array_view),
                    });

                    let shadow_samp = if !settings.shadows_enabled {
                        &self.shadows_off_sampler
                    } else if settings.reversed_depth_z {
                        &self.shadow_sampler_reversed_z
                    } else {
                        &self.shadow_sampler
                    };

                    entries.push(BindGroupEntry {
                        binding: (material_count + 2) as u32,
                        resource: BindingResource::Sampler(shadow_samp),
                    });
                }
            }
        }

        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Material Bind Group"),
            layout,
            entries: &entries,
        })
    }
}
