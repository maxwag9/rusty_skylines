// compute_system.rs

use std::collections::HashMap;
use std::fs;
use wgpu::*;

/// Options for compute dispatch
pub struct ComputePipelineOptions {
    pub dispatch_size: [u32; 3],
}

impl Default for ComputePipelineOptions {
    fn default() -> Self {
        Self {
            dispatch_size: [1, 1, 1],
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct PipelineKey {
    shader_path: String,
    input_specs: Vec<(TextureFormat, u32)>, // (format, sample_count)
    output_formats: Vec<TextureFormat>,
    uniform_count: usize,
}

struct CachedPipeline {
    pipeline: ComputePipeline,
    bind_group_layouts: [BindGroupLayout; 3],
}

pub struct ComputeSystem {
    device: Device,
    queue: Queue,
    pipeline_cache: HashMap<PipelineKey, CachedPipeline>,
    default_sampler: Sampler,
}

impl ComputeSystem {
    pub fn new(device: Device, queue: Queue) -> Self {
        let default_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("compute_sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            ..Default::default()
        });

        Self {
            device,
            queue,
            pipeline_cache: HashMap::new(),
            default_sampler,
        }
    }

    /// Run a compute pass.
    ///
    /// Bind group layout:
    /// - group(0): input textures (binding 0..n) + sampler (binding n)
    /// - group(1): output storage textures (binding 0..m)
    /// - group(2): uniform buffers (binding 0..k)
    ///
    /// Empty vectors result in empty bind groups for those slots.
    /// MSAA textures are automatically detected via sample_count.
    pub fn compute(
        &mut self,
        label: &str,
        input_views: Vec<&TextureView>,
        output_views: Vec<&TextureView>,
        shader_path: &str,
        options: ComputePipelineOptions,
        uniforms: &[&Buffer],
    ) {
        let input_specs: Vec<_> = input_views
            .iter()
            .map(|v| {
                let tex = v.texture();
                (tex.format(), tex.sample_count())
            })
            .collect();

        let output_formats: Vec<_> = output_views.iter().map(|v| v.texture().format()).collect();

        let key = PipelineKey {
            shader_path: shader_path.to_string(),
            input_specs: input_specs.clone(),
            output_formats: output_formats.clone(),
            uniform_count: uniforms.len(),
        };

        // Get or create cached pipeline
        if !self.pipeline_cache.contains_key(&key) {
            let cached =
                self.create_pipeline(shader_path, &input_specs, &output_formats, uniforms.len());
            self.pipeline_cache.insert(key.clone(), cached);
        }
        let cached = self.pipeline_cache.get(&key).unwrap();

        // Create bind groups
        let input_bg = self.create_input_bind_group(&cached.bind_group_layouts[0], &input_views);
        let output_bg = self.create_output_bind_group(&cached.bind_group_layouts[1], &output_views);
        let uniform_bg = self.create_uniform_bind_group(&cached.bind_group_layouts[2], uniforms);

        // Execute
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: Some(label) });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });

            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &input_bg, &[]);
            pass.set_bind_group(1, &output_bg, &[]);
            pass.set_bind_group(2, &uniform_bg, &[]);
            pass.dispatch_workgroups(
                options.dispatch_size[0],
                options.dispatch_size[1],
                options.dispatch_size[2],
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    fn create_pipeline(
        &self,
        shader_path: &str,
        input_specs: &[(TextureFormat, u32)],
        output_formats: &[TextureFormat],
        uniform_count: usize,
    ) -> CachedPipeline {
        let shader_source = fs::read_to_string(shader_path)
            .unwrap_or_else(|_| panic!("Failed to read shader: {}", shader_path));

        let shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some(shader_path),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        // Group 0: inputs + sampler
        let mut input_entries: Vec<BindGroupLayoutEntry> = input_specs
            .iter()
            .enumerate()
            .map(|(i, (format, sample_count))| {
                let multisampled = *sample_count > 1;
                let sample_type = if format.has_depth_aspect() {
                    TextureSampleType::Depth
                } else {
                    TextureSampleType::Float {
                        filterable: !multisampled,
                    }
                };

                BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type,
                        view_dimension: TextureViewDimension::D2,
                        multisampled,
                    },
                    count: None,
                }
            })
            .collect();

        if !input_specs.is_empty() {
            input_entries.push(BindGroupLayoutEntry {
                binding: input_specs.len() as u32,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            });
        }

        let input_layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("compute_input_layout"),
                entries: &input_entries,
            });

        // Group 1: outputs
        let output_entries: Vec<BindGroupLayoutEntry> = output_formats
            .iter()
            .enumerate()
            .map(|(i, format)| BindGroupLayoutEntry {
                binding: i as u32,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: *format,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            })
            .collect();

        let output_layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("compute_output_layout"),
                entries: &output_entries,
            });

        // Group 2: uniforms
        let uniform_entries: Vec<BindGroupLayoutEntry> = (0..uniform_count)
            .map(|i| BindGroupLayoutEntry {
                binding: i as u32,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();

        let uniform_layout = self
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("compute_uniform_layout"),
                entries: &uniform_entries,
            });

        let bind_group_layouts = [input_layout, output_layout, uniform_layout];

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("compute_pipeline_layout"),
                bind_group_layouts: &bind_group_layouts.iter().collect::<Vec<_>>(),
                immediate_size: 0,
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(shader_path),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        CachedPipeline {
            pipeline,
            bind_group_layouts,
        }
    }

    fn create_input_bind_group(
        &self,
        layout: &BindGroupLayout,
        views: &[&TextureView],
    ) -> BindGroup {
        let mut entries: Vec<BindGroupEntry> = views
            .iter()
            .enumerate()
            .map(|(i, view)| BindGroupEntry {
                binding: i as u32,
                resource: BindingResource::TextureView(view),
            })
            .collect();

        if !views.is_empty() {
            entries.push(BindGroupEntry {
                binding: views.len() as u32,
                resource: BindingResource::Sampler(&self.default_sampler),
            });
        }

        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("compute_input_bg"),
            layout,
            entries: &entries,
        })
    }

    fn create_output_bind_group(
        &self,
        layout: &BindGroupLayout,
        views: &[&TextureView],
    ) -> BindGroup {
        let entries: Vec<BindGroupEntry> = views
            .iter()
            .enumerate()
            .map(|(i, view)| BindGroupEntry {
                binding: i as u32,
                resource: BindingResource::TextureView(view),
            })
            .collect();

        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("compute_output_bg"),
            layout,
            entries: &entries,
        })
    }

    fn create_uniform_bind_group(
        &self,
        layout: &BindGroupLayout,
        uniforms: &[&Buffer],
    ) -> BindGroup {
        let entries: Vec<BindGroupEntry> = uniforms
            .iter()
            .enumerate()
            .map(|(i, buffer)| BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();

        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("compute_uniform_bg"),
            layout,
            entries: &entries,
        })
    }

    /// Clear the pipeline cache (useful if shaders changed on disk)
    pub fn invalidate_cache(&mut self) {
        self.pipeline_cache.clear();
    }
}
