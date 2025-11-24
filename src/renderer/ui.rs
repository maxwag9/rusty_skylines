use crate::renderer::ui_editor::{RuntimeLayer, UiButtonLoader};
use crate::renderer::ui_pipelines::UiPipelines;
use crate::resources::{MouseState, TimeSystem};
use crate::vertex::{UiVertexPoly, UiVertexText};
use fontdue::Font;
use rect_packer::DensePacker;
use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;
use wgpu::*;
use winit::dpi::PhysicalSize;

const PAD: i32 = 1;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScreenUniform {
    pub size: [f32; 2],
    pub time: f32,
    pub enable_dither: u32, // use 0 = off, 1 = on
    pub mouse: [f32; 2],    // position!
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

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CircleParams {
    pub center_radius_border: [f32; 4], // cx, cy, radius, border
    pub fade: [f32; 4],
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
            fade: [0.0; 4],
            fill_color: [0.0; 4],
            border_color: [0.0; 4],
            glow_color: [0.0; 4],
            glow_misc: [0.0; 4],
            misc: [0.0; 4], // active, touched_time, is_touched, id_hash
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OutlineParams {
    pub(crate) mode: f32,          // 0.0 = circle, 1.0 = polygon
    pub(crate) vertex_offset: u32, // polygon vertices start index (ignored for circle)
    pub(crate) vertex_count: u32,  // polygon vertex count (ignored for circle)
    pub(crate) _pad0: u32,

    pub(crate) shape_data: [f32; 4], // (cx, cy, radius, thickness_factor)
    pub(crate) dash_color: [f32; 4],
    pub(crate) dash_misc: [f32; 4], // (dash_len, dash_spacing, dash_roundness, speed)
    pub(crate) sub_dash_color: [f32; 4],
    pub(crate) sub_dash_misc: [f32; 4], // (sub_dash_len, sub_dash_spacing, sub_roundness, sub_speed)
    pub(crate) misc: [f32; 4],          // active, touched_time, is_touched, id_hash
}

impl Default for OutlineParams {
    fn default() -> Self {
        Self {
            mode: 0.0,
            vertex_offset: 0,
            vertex_count: 0,
            _pad0: 1,

            shape_data: [600.0, 600.0, 60.0, 10.0],
            dash_color: [0.0, 0.2, 0.7, 0.8],
            dash_misc: [2.0, 1.0, 1.0, 2.0], // (dash_len, dash_spacing, dash_roundness, speed)
            sub_dash_color: [0.3, 0.4, 0.5, 0.9],
            sub_dash_misc: [2.0, 1.0, 1.0, -2.0], // (sub_dash_len, sub_dash_spacing, sub_dash_roundness, sub_speed)
            misc: [1.0, 0.0, 0.0, 0.0],           // active, touched_time, is_touched, id_hash
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HandleParams {
    pub center_radius_mode: [f32; 4], // cx, cy, is_circle, ?
    pub(crate) handle_color: [f32; 4],
    pub(crate) handle_misc: [f32; 4], // (handle_len, handle_width, handle_roundness, ?)
    pub sub_handle_color: [f32; 4], // color of the center line drawn on top of the center of the normal handle
    pub sub_handle_misc: [f32; 4],  // (sub_handle_len, sub_handle_width, sub_handle_roundness, ?)
    pub(crate) misc: [f32; 4],      // active, touched_time, is_touched, id_hash
}
impl Default for HandleParams {
    fn default() -> Self {
        Self {
            center_radius_mode: [0.0; 4], // cx, cy, is_circle, ?
            handle_color: [0.0, 0.2, 0.7, 0.8],
            handle_misc: [2.0, 1.0, 1.0, 2.0], // (dash_len, dash_spacing, dash_roundness, speed)
            sub_handle_color: [0.3, 0.4, 0.5, 0.9],
            sub_handle_misc: [2.0, 1.0, 1.0, -2.0], // (sub_dash_len, sub_dash_spacing, sub_dash_roundness, sub_speed)
            misc: [1.0, 0.0, 0.0, 0.0],             // active, touched_time, is_touched, id_hash
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

#[derive(Debug)]
pub struct TextParams {
    pub pos: [f32; 2],
    pub px: u16,
    pub color: [f32; 4],
    pub id_hash: f32,
    pub misc: [f32; 4], // [active, touched_time, is_down, id_hash]
    pub text: String,
    pub natural_width: f32,
    pub natural_height: f32,
    pub id: Option<String>,
}

pub struct DrawCmd<'a> {
    pub z: i32,
    pub pipeline: &'a wgpu::RenderPipeline,
    pub bind_group0: &'a wgpu::BindGroup,
    pub bind_group1: Option<wgpu::BindGroup>,
    pub vertex_buffer: Option<&'a wgpu::Buffer>,
    pub vertex_range: Range<u32>,
    pub instance_range: Range<u32>,
}

pub struct UiRenderer {
    pub pipelines: UiPipelines,

    device: Device,
}

impl UiRenderer {
    pub fn new(
        device: &Device,
        format: TextureFormat,
        size: PhysicalSize<u32>,
        msaa_samples: u32,
        shader_dir: &Path,
    ) -> anyhow::Result<Self> {
        let pipelines = UiPipelines::new(device, format, msaa_samples, size, shader_dir)?;

        Ok(Self {
            pipelines,
            device: device.clone(),
        })
    }

    pub fn reload_shaders(&mut self) -> anyhow::Result<()> {
        self.pipelines.reload_shaders()?;
        Ok(())
    }

    pub fn render<'a>(
        &mut self,
        pass: &mut wgpu::RenderPass<'a>,
        ui: &mut UiButtonLoader,
        queue: &wgpu::Queue,
        time: &TimeSystem,
        size: (f32, f32),
        mouse: &MouseState,
    ) {
        let new_uniform = ScreenUniform {
            size: [size.0, size.1],
            time: time.total_time,
            enable_dither: 1,
            mouse: mouse.pos.to_array(),
        };

        queue.write_buffer(
            &self.pipelines.uniform_buffer,
            0,
            bytemuck::bytes_of(&new_uniform),
        );
        ui.update_dynamic_texts();

        ui.sync_console_ui();

        for (menu_name, menu) in ui.menus.iter_mut().filter(|(_, menu)| menu.active) {
            let dirty_indices: Vec<usize> = menu
                .layers
                .iter()
                .enumerate()
                .filter(|(_, l)| l.active && l.dirty)
                .map(|(i, _)| i)
                .collect();

            for idx in dirty_indices {
                menu.rebuild_layer_cache_index(idx, &ui.ui_runtime);
                let layer = &mut menu.layers[idx];
                self.upload_layer(queue, layer);
            }

            for layer in menu.layers.iter().filter(|l| l.active) {
                let mut cmds: Vec<DrawCmd> = Vec::new();
                if layer.gpu.circle_count > 0 {
                    let circle_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.circle_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: layer.gpu.circle_ssbo.as_ref().unwrap().as_entire_binding(),
                        }],
                    });

                    let mut zlist: Vec<(i32, u32)> = Vec::new();
                    for (i, c) in layer.circles.iter().enumerate() {
                        zlist.push((c.z_index, i as u32));
                    }
                    zlist.sort_by_key(|v| v.0);

                    for (_, idx) in zlist.iter() {
                        let z = layer.circles[*idx as usize].z_index;

                        cmds.push(DrawCmd {
                            z,
                            pipeline: &self.pipelines.circle_pipeline,
                            bind_group0: &self.pipelines.uniform_bind_group,
                            bind_group1: Some(circle_bg.clone()),
                            vertex_buffer: Some(&self.pipelines.quad_buffer),
                            vertex_range: 0..4,
                            instance_range: *idx..*idx + 1,
                        });

                        cmds.push(DrawCmd {
                            z,
                            pipeline: &self.pipelines.glow_pipeline,
                            bind_group0: &self.pipelines.uniform_bind_group,
                            bind_group1: Some(circle_bg.clone()),
                            vertex_buffer: Some(&self.pipelines.quad_buffer),
                            vertex_range: 0..4,
                            instance_range: *idx..*idx + 1,
                        });
                    }
                }

                if layer.gpu.handle_count > 0 {
                    let handle_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.handle_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: layer.gpu.handle_ssbo.as_ref().unwrap().as_entire_binding(),
                        }],
                    });

                    let mut zlist: Vec<(i32, u32)> = Vec::new();
                    for (i, h) in layer.handles.iter().enumerate() {
                        zlist.push((h.z_index, i as u32));
                    }
                    zlist.sort_by_key(|v| v.0);

                    for (_, idx) in zlist.iter() {
                        cmds.push(DrawCmd {
                            z: layer.handles[*idx as usize].z_index,
                            pipeline: &self.pipelines.handle_pipeline,
                            bind_group0: &self.pipelines.uniform_bind_group,
                            bind_group1: Some(handle_bg.clone()),
                            vertex_buffer: Some(&self.pipelines.handle_quad_buffer),
                            vertex_range: 0..4,
                            instance_range: *idx..*idx + 1,
                        });
                    }
                }

                if layer.gpu.poly_count > 0 {
                    let mut offset = 0u32;
                    if let Some(poly_vbo) = &layer.gpu.poly_vbo {
                        for p in layer.polygons.iter() {
                            let count = (p.tri_count * 3) as u32;

                            cmds.push(DrawCmd {
                                z: p.z_index,
                                pipeline: &self.pipelines.polygon_pipeline,
                                bind_group0: &self.pipelines.uniform_bind_group,
                                bind_group1: None,
                                vertex_buffer: Some(poly_vbo),
                                vertex_range: offset..offset + count,
                                instance_range: 0..1,
                            });

                            offset += count;
                        }
                    }
                }

                if layer.gpu.outline_count > 0 {
                    let outline_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.outline_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: layer
                                    .gpu
                                    .outline_shapes_ssbo
                                    .as_ref()
                                    .unwrap()
                                    .as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: layer
                                    .gpu
                                    .outline_poly_vertices_ssbo
                                    .as_ref()
                                    .unwrap()
                                    .as_entire_binding(),
                            },
                        ],
                    });

                    for (i, o) in layer.outlines.iter().enumerate() {
                        cmds.push(DrawCmd {
                            z: o.z_index,
                            pipeline: &self.pipelines.outline_pipeline,
                            bind_group0: &self.pipelines.uniform_bind_group,
                            bind_group1: Some(outline_bg.clone()),
                            vertex_buffer: Some(&self.pipelines.quad_buffer),
                            vertex_range: 0..4,
                            instance_range: i as u32..i as u32 + 1,
                        });
                    }
                }

                if layer.gpu.text_count > 0 {
                    if let (Some(text_bg), Some(_)) =
                        (&self.pipelines.text_bind_group, &self.pipelines.text_atlas)
                    {
                        if let Some(text_vbo) = &layer.gpu.text_vbo {
                            let z = layer.texts.iter().map(|t| t.z_index).max().unwrap_or(0);

                            cmds.push(DrawCmd {
                                z,
                                pipeline: &self.pipelines.text_pipeline,
                                bind_group0: &self.pipelines.uniform_bind_group,
                                bind_group1: Some(text_bg.clone()),
                                vertex_buffer: Some(text_vbo),
                                vertex_range: 0..layer.gpu.text_count,
                                instance_range: 0..1,
                            });
                        }
                    }
                }

                cmds.sort_by_key(|c| c.z);

                for c in cmds {
                    pass.set_pipeline(c.pipeline);
                    pass.set_bind_group(0, c.bind_group0, &[]);
                    if let Some(bg1) = &c.bind_group1 {
                        pass.set_bind_group(1, bg1, &[]);
                    }
                    if let Some(vb) = c.vertex_buffer {
                        pass.set_vertex_buffer(0, vb.slice(..));
                    }
                    pass.draw(c.vertex_range.clone(), c.instance_range.clone());
                }
            }
        }
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

        // ---- 1. ShapeParams SSBO ----
        let outline_len = layer.cache.outline_params.len() as u32;
        if outline_len > 0 {
            let bytes = bytemuck::cast_slice(&layer.cache.outline_params);
            let need_new = layer
                .gpu
                .outline_shapes_ssbo
                .as_ref()
                .map(|b| b.size() < bytes.len() as u64)
                .unwrap_or(true);

            if need_new {
                layer.gpu.outline_shapes_ssbo =
                    Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("{}_outline_shapes_ssbo", layer.name)),
                        size: bytes.len() as u64,
                        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));
            }

            queue.write_buffer(layer.gpu.outline_shapes_ssbo.as_ref().unwrap(), 0, bytes);
        }
        layer.gpu.outline_count = outline_len;

        // ---- 2. Polygon vertex buffer (vec2<f32>) ----
        let poly_verts = &layer.cache.outline_poly_vertices;
        let poly_vcount = poly_verts.len() as u32;

        if poly_vcount > 0 {
            // upload real polygon vertices
            let bytes = bytemuck::cast_slice(poly_verts);

            let need_new = layer
                .gpu
                .outline_poly_vertices_ssbo
                .as_ref()
                .map(|b| b.size() < bytes.len() as u64)
                .unwrap_or(true);

            if need_new {
                layer.gpu.outline_poly_vertices_ssbo =
                    Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("{}_outline_poly_ssbo", layer.name)),
                        size: bytes.len() as u64,
                        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));
            }

            queue.write_buffer(
                layer.gpu.outline_poly_vertices_ssbo.as_ref().unwrap(),
                0,
                bytes,
            );
        } else {
            // No polygon outlines → still must provide SOME buffer to satisfy wgpu layout
            if layer.gpu.outline_poly_vertices_ssbo.is_none() {
                layer.gpu.outline_poly_vertices_ssbo =
                    Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!("{}_outline_poly_dummy", layer.name)),
                        size: 16, // one vec2<f32> worth of space
                        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));
            }
        }

        let handle_len = layer.cache.handle_params.len() as u32;
        if handle_len > 0 {
            let bytes = bytemuck::cast_slice(&layer.cache.handle_params);
            let need_new = layer
                .gpu
                .handle_ssbo
                .as_ref()
                .map(|b| b.size() < bytes.len() as u64)
                .unwrap_or(true);
            if need_new {
                layer.gpu.handle_ssbo = Some(self.device.create_buffer(&BufferDescriptor {
                    label: Some(&format!("{}_handle_ssbo", layer.name)),
                    size: bytes.len() as u64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            queue.write_buffer(layer.gpu.handle_ssbo.as_ref().unwrap(), 0, bytes);
        }
        layer.gpu.handle_count = handle_len;

        // ---- polygons (VBO) : polys concatenated ----
        let mut poly_vertices: Vec<UiVertexPoly> =
            Vec::with_capacity(layer.cache.polygon_vertices.len());
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
        if let Some(atlas) = &self.pipelines.text_atlas {
            for tp in &mut layer.cache.texts {
                let mut min_x = f32::MAX;
                let mut min_y = f32::MAX;
                let mut max_x = f32::MIN;
                let mut max_y = f32::MIN;

                let mut pen_x = tp.pos[0];
                let baseline_y = tp.pos[1] + atlas.line_height;

                for ch in tp.text.chars() {
                    if let Some(g) = atlas.glyphs.get(&(ch, tp.px)) {
                        let x0 = (pen_x + g.bearing_x).round();
                        let y0 = (baseline_y - g.bearing_y).round();
                        let x1 = x0 + g.w;
                        let y1 = y0 + g.h;

                        // bounding box
                        min_x = min_x.min(x0);
                        min_y = min_y.min(y0);
                        max_x = max_x.max(x1);
                        max_y = max_y.max(y1);

                        // push triangles
                        text_vertices.extend_from_slice(&[
                            UiVertexText {
                                pos: [x0, y0],
                                uv: [g.u0, g.v0],
                                color: tp.color,
                            },
                            UiVertexText {
                                pos: [x1, y0],
                                uv: [g.u1, g.v0],
                                color: tp.color,
                            },
                            UiVertexText {
                                pos: [x1, y1],
                                uv: [g.u1, g.v1],
                                color: tp.color,
                            },
                            UiVertexText {
                                pos: [x0, y0],
                                uv: [g.u0, g.v0],
                                color: tp.color,
                            },
                            UiVertexText {
                                pos: [x1, y1],
                                uv: [g.u1, g.v1],
                                color: tp.color,
                            },
                            UiVertexText {
                                pos: [x0, y1],
                                uv: [g.u0, g.v1],
                                color: tp.color,
                            },
                        ]);

                        pen_x += g.advance;
                    }
                }

                tp.natural_width = (max_x - min_x).max(0.0);
                tp.natural_height = (max_y - min_y).max(0.0);

                // ---- write back into runtime UiButtonText ----
                if let Some(ref text_id) = tp.id {
                    for original_text in &mut layer.texts {
                        if original_text.id.as_ref() == Some(text_id) {
                            original_text.natural_width = tp.natural_width;
                            original_text.natural_height = tp.natural_height;
                            break;
                        }
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

                if metrics.width == 0 && metrics.height == 0 {
                    // this is needed so space characters work
                    glyphs.insert(
                        (*ch, px),
                        GlyphUv {
                            u0: 0.0,
                            v0: 0.0,
                            u1: 0.0,
                            v1: 0.0,
                            advance: metrics.advance_width,
                            w: 0.0,
                            h: 0.0,
                            bearing_x: 0.0,
                            bearing_y: 0.0,
                        },
                    );
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
            layout: &self.pipelines.text_layout,
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
        self.pipelines.text_atlas = Some(TextAtlas {
            tex,
            view,
            sampler,
            size: (atlas_w, atlas_h),
            glyphs,
            line_height: max_line_h,
        });
        self.pipelines.text_bind_group = Some(text_bind_group);

        Ok(())
    }
}
