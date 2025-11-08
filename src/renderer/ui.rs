use crate::renderer::helper::{make_pipeline, make_uniform_layout, rebuild_layer_cache};
use crate::renderer::ui_editor::{RuntimeLayer, UiButtonLoader};
use crate::resources::TimeSystem;
use crate::vertex::{MiscButtonSettings, UiButtonText, UiVertex, UiVertexPoly, UiVertexText};
use fontdue::Font;
use rect_packer::DensePacker;
use std::collections::HashMap;
use util::DeviceExt;
use wgpu::*;
use winit::dpi::PhysicalSize;

const PAD: i32 = 1;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScreenUniform {
    pub size: [f32; 2],
    pub time: f32,
    pub enable_dither: u32, // use 0 = off, 1 = on
}

#[derive(Clone, Copy)]
pub struct GlyphUv {
    pub u0: f32,
    pub v0: f32,
    pub u1: f32,
    pub v1: f32,
    pub advance: f32,
    pub w: f32,
    pub h: f32, // px
    pub bearing_x: f32,
    pub bearing_y: f32,
}

pub struct TextAtlas {
    pub tex: Texture, // R8Unorm
    pub view: TextureView,
    pub sampler: Sampler,
    pub size: (u32, u32),
    pub glyphs: HashMap<(char, u16), GlyphUv>, // (char, px_size) -> uv+metrics
    pub line_height: f32,
}

pub struct UiRenderer {
    pub vertex_buffer: Buffer,
    pub uniform_bind_group: BindGroup,
    pub num_vertices: u32,
    circle_pipeline: RenderPipeline,
    circle_outline_pipeline: RenderPipeline,
    polygon_pipeline: RenderPipeline,
    overlay_polygon_pipeline: RenderPipeline,
    pub circles: Vec<CircleParams>,
    pub quad_buffer: Buffer,
    glow_pipeline: RenderPipeline,
    pub(crate) uniform_buffer: Buffer,
    device: Device,
    circle_layout: BindGroupLayout,
    pub text_atlas: Option<TextAtlas>,
    pub text_pipeline: RenderPipeline,
    pub text_layout: BindGroupLayout,
    pub text_bind_group: Option<BindGroup>,
    pub text_vertex_buffer: Buffer,
    pub text_vertex_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CircleParams {
    pub center_radius_border: [f32; 4], // cx, cy, radius, border
    pub fill_color: [f32; 4],
    pub border_color: [f32; 4],
    pub glow_color: [f32; 4],
    pub glow_misc: [f32; 4], // glow_size, glow_pulse_speed, glow_pulse_intensity
    pub misc: [f32; 4],      // active, touched_time, is_touched, id_hash
}

impl Default for CircleParams {
    fn default() -> Self {
        Self {
            center_radius_border: [0.0, 0.0, 0.0, 0.0],
            fill_color: [0.0; 4],
            border_color: [0.0; 4],
            glow_color: [0.0; 4],
            glow_misc: [0.0; 4],
            misc: [0.0; 4], // active, touched_time, is_touched, id_hash
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CircleOutlineParams {
    pub(crate) center_radius_border: [f32; 4], // cx, cy, radius, thickness
    pub(crate) dash_color: [f32; 4],
    pub(crate) dash_misc: [f32; 4], // (dash_len, dash_spacing, dash_roundness, speed)
    pub(crate) misc: [f32; 4],      // active, touched_time, is_touched, id_hash
    pub sub_dash_color: [f32; 4],
    pub sub_dash_misc: [f32; 4], // (dash_len, dash_spacing, dash_roundness, speed)
}
impl Default for CircleOutlineParams {
    fn default() -> Self {
        Self {
            center_radius_border: [0.0; 4], // cx, cy, radius, thickness
            dash_color: [0.0, 0.2, 0.7, 0.8],
            dash_misc: [2.0, 1.0, 1.0, 2.0], // (dash_len, dash_spacing, dash_roundness, speed)
            sub_dash_color: [0.3, 0.4, 0.5, 0.9],
            sub_dash_misc: [2.0, 1.0, 1.0, -2.0], // (sub_dash_len, sub_dash_spacing, sub_dash_roundness, sub_speed)
            misc: [1.0, 0.0, 0.0, 0.0],           // active, touched_time, is_touched, id_hash
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PolygonOutlineParams {
    center_radius_border: [f32; 4], // cx, cy, radius, thickness
    dash_color: [f32; 4],
    dash_misc: [f32; 4], // (dash_len, dash_spacing, dash_roundness, speed)
    misc: [f32; 4],      // active, touched_time, is_touched, id_hash
}
impl Default for PolygonOutlineParams {
    fn default() -> Self {
        Self {
            center_radius_border: [0.0; 4], // cx, cy, radius, thickness
            dash_color: [0.0; 4],
            dash_misc: [0.0; 4], // (dash_len, dash_spacing, dash_roundness, speed)
            misc: [0.0; 4],      // active, touched_time, is_touched, id_hash
        }
    }
}

pub struct TextParams {
    pub pos: [f32; 2],
    pub px: u16,
    pub color: [f32; 4],
    pub id_hash: f32,
    pub misc: [f32; 4], // [active, touched_time, is_down, id_hash]
    pub text: String,
}

impl UiRenderer {
    pub fn new(device: &Device, format: TextureFormat, size: PhysicalSize<u32>) -> Self {
        let quad_vertices = [
            UiVertexPoly {
                pos: [-1.0, -1.0],
                _pad: [1.0; 2],
                color: [1.0; 4],
                misc: [1.0, 0.0, 0.0, 0.0],
            },
            UiVertexPoly {
                pos: [1.0, -1.0],
                _pad: [1.0; 2],
                color: [1.0; 4],
                misc: [1.0, 0.0, 0.0, 0.0],
            },
            UiVertexPoly {
                pos: [-1.0, 1.0],
                _pad: [1.0; 2],
                color: [1.0; 4],
                misc: [1.0, 0.0, 0.0, 0.0],
            },
            UiVertexPoly {
                pos: [1.0, 1.0],
                _pad: [1.0; 2],
                color: [1.0; 4],
                misc: [1.0, 0.0, 0.0, 0.0],
            },
        ];
        let quad_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("UI Quad VB"),
            contents: bytemuck::cast_slice(&quad_vertices),
            usage: BufferUsages::VERTEX,
        });
        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("UI VB"),
            size: 1024 * 1024, // 1MB buffer
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = make_uniform_layout(device, "UI Bind Layout");

        let screen_uniform = ScreenUniform {
            size: [size.width as f32, size.height as f32],
            time: 0.0,
            enable_dither: 1,
        };
        let screen_data = bytemuck::bytes_of(&screen_uniform);

        let uniform_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("UI Uniforms"),
            contents: screen_data,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let uniform_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("UI Bind Group"),
            layout: &layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let text_shader = device.create_shader_module(include_wgsl!("shaders/text.wgsl"));
        let text_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Text Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let text_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Text Pipeline Layout"),
            bind_group_layouts: &[&layout, &text_layout],
            push_constant_ranges: &[],
        });
        let text_pipeline = make_pipeline(
            device,
            "UI Text Pipeline",
            &text_pipeline_layout,
            &text_shader,
            "vs_main",
            "fs_main",
            &[UiVertexText::desc()],
            format,
            Some(BlendState::ALPHA_BLENDING),
            PrimitiveTopology::TriangleList,
        );

        let text_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UI Text VB"),
            size: 256 * 1024,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // println!("CircleParams size: {}", std::mem::size_of::<CircleParams>());
        let circles = vec![CircleParams::default()];
        let circle_shader = device.create_shader_module(include_wgsl!("shaders/ui_circle.wgsl"));
        let circle_outline_shader =
            device.create_shader_module(include_wgsl!("shaders/ui_circle_outline.wgsl"));
        let circle_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Circle Layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let circle_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Circle Pipeline Layout"),
            bind_group_layouts: &[&layout, &circle_layout],
            push_constant_ranges: &[],
        });

        let circle_pipeline = make_pipeline(
            device,
            "UI Circle Pipeline",
            &circle_pipeline_layout,
            &circle_shader,
            "vs_main",
            "fs_main",
            &[UiVertexPoly::desc()],
            format,
            Some(BlendState::ALPHA_BLENDING),
            PrimitiveTopology::TriangleStrip,
        );
        let circle_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Circle Layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let circle_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Circle Layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let circle_outline_pipeline = make_pipeline(
            device,
            "UI Overlay Circle Pipeline",
            &circle_pipeline_layout,
            &circle_outline_shader,
            "vs_main",
            "fs_main",
            &[UiVertexPoly::desc()],
            format,
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            PrimitiveTopology::TriangleStrip,
        );

        let polygon_shader = device.create_shader_module(include_wgsl!("shaders/ui_polygon.wgsl"));
        let polygon_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("UI Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let polygon_pipeline = make_pipeline(
            device,
            "UI Polygon Pipeline",
            &polygon_pipeline_layout,
            &polygon_shader,
            "vs_main",
            "fs_main",
            &[UiVertexPoly::desc()],
            format,
            Some(BlendState::ALPHA_BLENDING),
            PrimitiveTopology::TriangleStrip,
        );
        let overlay_polygon_pipeline = make_pipeline(
            device,
            "UI Polygon Pipeline",
            &polygon_pipeline_layout,
            &polygon_shader,
            "vs_main",
            "fs_main",
            &[UiVertexPoly::desc()],
            format,
            None,
            PrimitiveTopology::TriangleStrip,
        );

        let glow_shader = device.create_shader_module(include_wgsl!("shaders/ui_circle_glow.wgsl"));
        let glow_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("UI Glow Pipeline Layout"),
            bind_group_layouts: &[&layout, &circle_layout],
            push_constant_ranges: &[],
        });
        let additive_blend = BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
        };

        let glow_pipeline = make_pipeline(
            device,
            "UI Glow Pipeline",
            &glow_pipeline_layout,
            &glow_shader,
            "vs_main",
            "fs_main",
            &[UiVertexPoly::desc()],
            format,
            Some(additive_blend),
            PrimitiveTopology::TriangleStrip,
        );

        Self {
            vertex_buffer,
            uniform_bind_group,
            num_vertices: 0,
            circle_pipeline,
            circle_outline_pipeline,
            polygon_pipeline,
            overlay_polygon_pipeline,
            circles,
            quad_buffer,
            glow_pipeline,
            uniform_buffer,
            device: device.clone(),
            circle_layout,
            text_atlas: None,
            text_pipeline,
            text_layout,
            text_bind_group: None,
            text_vertex_buffer,
            text_vertex_count: 0,
        }
    }

    pub fn render<'a>(
        &mut self,
        pass: &mut RenderPass<'a>,
        ui: &mut UiButtonLoader,
        queue: &Queue,
        time: &TimeSystem,
        size: (f32, f32),
    ) {
        let new_uniform = ScreenUniform {
            size: [size.0, size.1],
            time: time.total_time,
            enable_dither: 1,
        };

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&new_uniform));

        self.update_fps_layer(ui, time);
        let dirty_layers: Vec<usize> = ui
            .layers
            .iter()
            .enumerate()
            .filter(|(_, l)| l.active && l.dirty)
            .map(|(i, _)| i)
            .collect();

        for i in dirty_layers {
            let layer = &mut ui.layers[i];
            let runtime = &ui.ui_runtime;
            rebuild_layer_cache(layer, runtime);
            self.upload_layer(queue, layer);
        }

        //println!("{}", ui.layers.len());
        for layer in ui.layers.iter().filter(|l| l.active) {
            // println!("{}", layer.order);
            // circle
            if layer.gpu.circle_count > 0 {
                let circle_bg = self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some(&format!("{}_circle_bg", layer.name)),
                    layout: &self.circle_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: layer.gpu.circle_ssbo.as_ref().unwrap().as_entire_binding(),
                    }],
                });
                pass.set_pipeline(&self.circle_pipeline);

                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                pass.set_bind_group(1, &circle_bg, &[]);
                pass.set_vertex_buffer(0, self.quad_buffer.slice(..));
                pass.draw(0..4, 0..layer.gpu.circle_count);

                pass.set_pipeline(&self.glow_pipeline);
                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                pass.set_bind_group(1, &circle_bg, &[]);
                pass.set_vertex_buffer(0, self.quad_buffer.slice(..));
                pass.draw(0..4, 0..layer.gpu.circle_count);
            }

            if layer.gpu.circle_outline_count > 0 {
                let circle_outline_bg = self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some(&format!("{}_circle_outline_bg", layer.name)),
                    layout: &self.circle_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: layer
                            .gpu
                            .circle_outline_ssbo
                            .as_ref()
                            .unwrap()
                            .as_entire_binding(),
                    }],
                });

                pass.set_pipeline(&self.circle_outline_pipeline);
                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                pass.set_bind_group(1, &circle_outline_bg, &[]);
                pass.set_vertex_buffer(0, self.quad_buffer.slice(..));
                pass.draw(0..4, 0..layer.gpu.circle_outline_count);
            }

            // polygons
            if layer.gpu.poly_count > 0 {
                pass.set_pipeline(&self.polygon_pipeline);
                pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                pass.set_vertex_buffer(0, layer.gpu.poly_vbo.as_ref().unwrap().slice(..));
                pass.draw(0..layer.gpu.poly_count, 0..1);
            }

            // text
            if layer.gpu.text_count > 0 {
                if let (Some(bind), Some(_)) = (&self.text_bind_group, &self.text_atlas) {
                    pass.set_pipeline(&self.text_pipeline);
                    pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                    pass.set_bind_group(1, bind, &[]);
                    pass.set_vertex_buffer(0, layer.gpu.text_vbo.as_ref().unwrap().slice(..));
                    pass.draw(0..layer.gpu.text_count, 0..1);
                }
            }
        }
    }

    pub fn build_text_atlas(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ttf_bytes: &[u8],
        px_sizes: &[u16], // e.g. &[18, 24]
        atlas_w: u32,
        atlas_h: u32, // e.g. 1024, 1024
    ) -> Result<(), Box<dyn std::error::Error>> {
        // 1) Load font
        let font = Font::from_bytes(ttf_bytes, fontdue::FontSettings::default())
            .map_err(|e| format!("Failed to load font: {e}"))?;

        // 2) Prepare packer & CPU atlas (R8, 1 byte/px)
        let mut packer = DensePacker::new(atlas_w as i32, atlas_h as i32);
        let mut cpu_atlas = vec![0u8; (atlas_w * atlas_h) as usize];

        // ASCII HUD charset
        let charset: Vec<char> = (32u8..127).map(|c| c as char).collect();
        let mut glyphs: HashMap<(char, u16), GlyphUv> = HashMap::new();
        let mut max_line_h = 0.0f32;

        // 3) Rasterize, pack, blit
        for &px in px_sizes {
            for ch in &charset {
                let (metrics, bitmap) = font.rasterize(*ch, px as f32);
                if metrics.width == 0 || metrics.height == 0 {
                    continue;
                }

                // pack rect for this glyph
                let rect = packer
                    .pack(
                        (metrics.width as i32) + 2 * PAD,
                        (metrics.height as i32) + 2 * PAD,
                        false,
                    )
                    .ok_or("Atlas full")?;
                let gx = rect.x + PAD;
                let gy = rect.y + PAD;
                // copy bitmap rows into the CPU atlas
                for row in 0..metrics.height {
                    let src_start = row * metrics.width;
                    let src_end = src_start + metrics.width;

                    let dst_y = (gy as usize) + (row as usize);
                    let dst_x = gx as usize;
                    let dst_index = dst_y * (atlas_w as usize) + dst_x;

                    cpu_atlas[dst_index..dst_index + (metrics.width as usize)]
                        .copy_from_slice(&bitmap[src_start..src_end]);
                }

                // UVs in [0,1]
                let u0 = (gx as f32) / atlas_w as f32;
                let v0 = (gy as f32) / atlas_h as f32;
                let u1 = (gx + metrics.width as i32) as f32 / atlas_w as f32;
                let v1 = (gy + metrics.height as i32) as f32 / atlas_h as f32;

                // Store glyph info; NOTE: flip ymin sign for baseline math later
                glyphs.insert(
                    (*ch, px),
                    GlyphUv {
                        u0,
                        v0,
                        u1,
                        v1,
                        advance: metrics.advance_width,
                        w: metrics.width as f32,
                        h: metrics.height as f32,
                        bearing_x: metrics.xmin as f32,
                        bearing_y: (metrics.ymin + metrics.height as i32) as f32,
                    },
                );

                max_line_h = max_line_h.max(metrics.height as f32);
            }
        }

        // 4) Create GPU texture and upload
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Text Atlas"),
            size: Extent3d {
                width: atlas_w,
                height: atlas_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // For your wgpu build, bytes_per_row expects a plain u32
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &cpu_atlas,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas_w), // R8 => 1 byte/px * atlas_w
                rows_per_image: Some(atlas_h),
            },
            Extent3d {
                width: atlas_w,
                height: atlas_h,
                depth_or_array_layers: 1,
            },
        );

        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // 5) Bind group for the text pipeline
        let text_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Text Atlas Bind Group"),
            layout: &self.text_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // 6) Save into renderer state
        self.text_atlas = Some(TextAtlas {
            tex,
            view,
            sampler,
            size: (atlas_w, atlas_h),
            glyphs,
            line_height: max_line_h,
        });
        self.text_bind_group = Some(text_bind_group);

        Ok(())
    }

    fn upload_layer(&self, queue: &wgpu::Queue, layer: &mut RuntimeLayer) {
        // ---- circles (SSBO) ----
        let circle_len = layer.cache.circle_params.len() as u32;
        if circle_len > 0 {
            let bytes = bytemuck::cast_slice(&layer.cache.circle_params);
            let need_new = layer
                .gpu
                .circle_ssbo
                .as_ref()
                .map(|b| b.size() < bytes.len() as u64)
                .unwrap_or(true);
            if need_new {
                layer.gpu.circle_ssbo = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("{}_circle_ssbo", layer.name)),
                    size: bytes.len() as u64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            queue.write_buffer(layer.gpu.circle_ssbo.as_ref().unwrap(), 0, bytes);
        }
        layer.gpu.circle_count = circle_len;

        let circle_outline_len = layer.cache.circle_outline_params.len() as u32;
        if circle_outline_len > 0 {
            let bytes = bytemuck::cast_slice(&layer.cache.circle_outline_params);
            let need_new = layer
                .gpu
                .circle_outline_ssbo
                .as_ref()
                .map(|b| b.size() < bytes.len() as u64)
                .unwrap_or(true);
            if need_new {
                layer.gpu.circle_outline_ssbo =
                    Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("{}_circle_outline_ssbo", layer.name)),
                        size: bytes.len() as u64,
                        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));
            }
            queue.write_buffer(layer.gpu.circle_outline_ssbo.as_ref().unwrap(), 0, bytes);
        }
        layer.gpu.circle_outline_count = circle_outline_len;

        // ---- polygons (VBO) : rects + tris + polys concatenated ----
        let mut poly_vertices: Vec<UiVertexPoly> = Vec::with_capacity(
            layer.cache.rect_vertices.len()
                + layer.cache.triangle_vertices.len()
                + layer.cache.polygon_vertices.len(),
        );
        poly_vertices.extend_from_slice(&layer.cache.rect_vertices);
        poly_vertices.extend_from_slice(&layer.cache.triangle_vertices);
        poly_vertices.extend_from_slice(&layer.cache.polygon_vertices);

        let poly_count = poly_vertices.len() as u32;
        if poly_count > 0 {
            let bytes = bytemuck::cast_slice(&poly_vertices);
            let need_new = layer
                .gpu
                .poly_vbo
                .as_ref()
                .map(|b| b.size() < bytes.len() as u64)
                .unwrap_or(true);
            if need_new {
                layer.gpu.poly_vbo = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("{}_poly_vbo", layer.name)),
                    size: bytes.len() as u64,
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            queue.write_buffer(layer.gpu.poly_vbo.as_ref().unwrap(), 0, bytes);
        }
        layer.gpu.poly_count = poly_count;

        // ---- text (VBO) : build glyphs for this layer only ----
        let mut text_vertices: Vec<UiVertexText> = Vec::new();
        if let Some(atlas) = &self.text_atlas {
            for tp in &layer.cache.texts {
                let mut pen_x = tp.pos[0];
                let baseline_y = tp.pos[1] + atlas.line_height;
                for ch in tp.text.chars() {
                    if let Some(g) = atlas.glyphs.get(&(ch, tp.px)) {
                        let x0 = (pen_x + g.bearing_x).round();
                        let y0 = (baseline_y - g.bearing_y).round();
                        let x1 = x0 + g.w;
                        let y1 = y0 + g.h;

                        let col = tp.color;
                        text_vertices.extend_from_slice(&[
                            UiVertexText {
                                pos: [x0, y0],
                                uv: [g.u0, g.v0],
                                color: col,
                            },
                            UiVertexText {
                                pos: [x1, y0],
                                uv: [g.u1, g.v0],
                                color: col,
                            },
                            UiVertexText {
                                pos: [x1, y1],
                                uv: [g.u1, g.v1],
                                color: col,
                            },
                            UiVertexText {
                                pos: [x0, y0],
                                uv: [g.u0, g.v0],
                                color: col,
                            },
                            UiVertexText {
                                pos: [x1, y1],
                                uv: [g.u1, g.v1],
                                color: col,
                            },
                            UiVertexText {
                                pos: [x0, y1],
                                uv: [g.u0, g.v1],
                                color: col,
                            },
                        ]);
                        pen_x += g.advance;
                    }
                }
            }
        }
        let text_bytes = bytemuck::cast_slice(&text_vertices);
        if !text_vertices.is_empty() {
            let need_new = layer
                .gpu
                .text_vbo
                .as_ref()
                .map(|b| b.size() < text_bytes.len() as u64)
                .unwrap_or(true);
            if need_new {
                layer.gpu.text_vbo = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("{}_text_vbo", layer.name)),
                    size: text_bytes.len() as u64,
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            queue.write_buffer(layer.gpu.text_vbo.as_ref().unwrap(), 0, text_bytes);
        }
        layer.gpu.text_count = (text_vertices.len() as u32);
    }

    fn update_fps_layer(&mut self, ui: &mut UiButtonLoader, t: &TimeSystem) {
        let name = "debug_fps";
        if let Some(layer) = ui.layers.iter_mut().find(|l| l.name == name) {
            let s = format!(
                "FPS: {:.1} | Frame: {:.2}ms | Sim: {:.2}ms",
                t.render_fps,
                t.render_dt * 1000.0,
                t.sim_dt * 1000.0
            );

            // ensure it has one text
            if layer.texts.is_empty() {
                layer.texts.push(UiButtonText {
                    id: Some("fps_text".into()),
                    x: 150.0,
                    y: 60.0,
                    stretch_x: 0.0,
                    stretch_y: 0.0,
                    top_left_vertex: UiVertex {
                        pos: [0.0, 0.0],
                        color: [1.0; 4],
                        roundness: 0.0,
                    },
                    bottom_left_vertex: UiVertex {
                        pos: [0.0, 0.0],
                        color: [1.0; 4],
                        roundness: 0.0,
                    },
                    top_right_vertex: UiVertex {
                        pos: [0.0, 0.0],
                        color: [1.0; 4],
                        roundness: 0.0,
                    },
                    bottom_right_vertex: UiVertex {
                        pos: [0.0, 0.0],
                        color: [1.0; 4],
                        roundness: 0.0,
                    },
                    px: 24,
                    color: [1.0; 4],
                    text: s.clone(),
                    misc: MiscButtonSettings {
                        active: true,
                        touched_time: 0.0,
                        is_touched: false,
                    },
                });
            } else {
                layer.texts[0].text = s;
            }
            layer.dirty = true; // only this layer rebuilds each frame
        }
    }
}
