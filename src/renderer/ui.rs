use crate::renderer::ui_pipelines::UiPipelines;
use crate::resources::TimeSystem;
use crate::ui::input::MouseState;
use crate::ui::ui_editor::{UiButtonLoader, UiRuntime};
use crate::ui::vertex::{
    PolygonEdgeGpu, PolygonInfoGpu, RuntimeLayer, UiButtonText, UiVertexPoly, UiVertexText,
};
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
    pub fill_color: [f32; 4],
    pub inside_border_color: [f32; 4],
    pub border_color: [f32; 4],
    pub glow_color: [f32; 4],
    pub glow_misc: [f32; 4], // glow_size, glow_speed, glow_intensity
    pub misc: [f32; 4],      // active, touched_time, is_touched, id_hash

    pub fade: f32,  // 0..1 for fade effect
    pub style: u32, // 0 = normal, 1 = hue circle, 2 = SV, etc.
    pub inside_border_thickness: f32,
    pub _pad: u32, // padding to maintain 16-byte alignment
}

impl Default for CircleParams {
    fn default() -> Self {
        Self {
            center_radius_border: [0.0, 0.0, 0.0, 0.0],

            fill_color: [0.0; 4],
            inside_border_color: [0.0; 4],
            border_color: [0.0; 4],
            glow_color: [0.0; 4],
            glow_misc: [0.0; 4],
            misc: [0.0; 4], // active, touched_time, is_touched, id_hash
            fade: 0.0,
            style: 0,
            inside_border_thickness: 0.0,
            _pad: 1,
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
    pub caret: usize,
    pub glyph_bounds: Vec<(f32, f32)>,
}

pub struct DrawCmd<'a> {
    pub z: i32,
    pub pipeline: &'a RenderPipeline,
    pub bind_group0: &'a BindGroup,
    pub bind_group1: Option<BindGroup>,
    pub vertex_buffer: Option<&'a Buffer>,
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
        pass: &mut RenderPass<'a>,
        ui: &mut UiButtonLoader,
        queue: &Queue,
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
                .filter(|(_, l)| l.active && l.dirty.any())
                .map(|(i, _)| i)
                .collect();

            for idx in dirty_indices {
                menu.rebuild_layer_cache_index(idx, &ui.ui_runtime);
                let layer = &mut menu.layers[idx];
                self.upload_layer(queue, layer, time, &ui.ui_runtime, menu_name);
            }

            for layer in menu.layers.iter().filter(|l| l.active) {
                let mut cmds: Vec<DrawCmd> = Vec::new();
                if layer.gpu.circle_count > 0 {
                    let circle_bg = self.device.create_bind_group(&BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.circle_layout,
                        entries: &[BindGroupEntry {
                            binding: 0,
                            resource: layer.gpu.circle_ssbo.as_ref().unwrap().as_entire_binding(),
                        }],
                    });

                    for idx in self.sorted_indices_by_z(&layer.circles, |c| c.z_index) {
                        let z = layer.circles[idx].z_index;

                        cmds.push(DrawCmd {
                            z,
                            pipeline: &self.pipelines.circle_pipeline,
                            bind_group0: &self.pipelines.uniform_bind_group,
                            bind_group1: Some(circle_bg.clone()),
                            vertex_buffer: Some(&self.pipelines.quad_buffer),
                            vertex_range: 0..4,
                            instance_range: idx as u32..idx as u32 + 1,
                        });

                        cmds.push(DrawCmd {
                            z,
                            pipeline: &self.pipelines.glow_pipeline,
                            bind_group0: &self.pipelines.uniform_bind_group,
                            bind_group1: Some(circle_bg.clone()),
                            vertex_buffer: Some(&self.pipelines.quad_buffer),
                            vertex_range: 0..4,
                            instance_range: idx as u32..idx as u32 + 1,
                        });
                    }
                }

                if layer.gpu.handle_count > 0 {
                    let handle_bg = self.device.create_bind_group(&BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.handle_layout,
                        entries: &[BindGroupEntry {
                            binding: 0,
                            resource: layer.gpu.handle_ssbo.as_ref().unwrap().as_entire_binding(),
                        }],
                    });

                    for idx in self.sorted_indices_by_z(&layer.handles, |h| h.z_index) {
                        cmds.push(DrawCmd {
                            z: layer.handles[idx].z_index,
                            pipeline: &self.pipelines.handle_pipeline,
                            bind_group0: &self.pipelines.uniform_bind_group,
                            bind_group1: Some(handle_bg.clone()),
                            vertex_buffer: Some(&self.pipelines.handle_quad_buffer),
                            vertex_range: 0..4,
                            instance_range: idx as u32..idx as u32 + 1,
                        });
                    }
                }

                if layer.gpu.poly_count > 0 {
                    let mut offset = 0u32;

                    let poly_bg = if let (Some(info), Some(edges)) =
                        (&layer.gpu.poly_info_ssbo, &layer.gpu.poly_edge_ssbo)
                    {
                        Some(self.device.create_bind_group(&BindGroupDescriptor {
                            label: None,
                            layout: &self.pipelines.polygon_layout,
                            entries: &[
                                BindGroupEntry {
                                    binding: 0,
                                    resource: info.as_entire_binding(),
                                },
                                BindGroupEntry {
                                    binding: 1,
                                    resource: edges.as_entire_binding(),
                                },
                            ],
                        }))
                    } else {
                        None
                    };

                    if let Some(poly_vbo) = &layer.gpu.poly_vbo {
                        for p in layer.polygons.iter() {
                            let count = (p.tri_count * 3) as u32;

                            cmds.push(DrawCmd {
                                z: p.z_index,
                                pipeline: &self.pipelines.polygon_pipeline,
                                bind_group0: &self.pipelines.uniform_bind_group,
                                bind_group1: poly_bg.clone(),
                                vertex_buffer: Some(poly_vbo),
                                vertex_range: offset..offset + count,
                                instance_range: 0..1,
                            });

                            offset += count;
                        }
                    }
                }

                if layer.gpu.outline_count > 0 {
                    let outline_bg = self.device.create_bind_group(&BindGroupDescriptor {
                        label: None,
                        layout: &self.pipelines.outline_layout,
                        entries: &[
                            BindGroupEntry {
                                binding: 0,
                                resource: layer
                                    .gpu
                                    .outline_shapes_ssbo
                                    .as_ref()
                                    .unwrap()
                                    .as_entire_binding(),
                            },
                            BindGroupEntry {
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

    fn write_storage_buffer(
        &self,
        queue: &Queue,
        target: &mut Option<Buffer>,
        label: &str,
        usage: BufferUsages,
        bytes: &[u8],
    ) {
        if bytes.is_empty() {
            return;
        }

        let needs_new = target
            .as_ref()
            .map(|b| b.size() < bytes.len() as u64)
            .unwrap_or(true);

        if needs_new {
            *target = Some(self.device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: bytes.len() as u64,
                usage: usage | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        if let Some(buf) = target {
            queue.write_buffer(buf, 0, bytes);
        }
    }

    fn sorted_indices_by_z<T>(&self, items: &[T], z: impl Fn(&T) -> i32) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..items.len()).collect();
        indices.sort_by_key(|&i| z(&items[i]));
        indices
    }

    fn upload_layer(
        &self,
        queue: &Queue,
        layer: &mut RuntimeLayer,
        time_system: &TimeSystem,
        ui_runtime: &UiRuntime,
        menu_name: &String,
    ) {
        // ---- circles (SSBO) ----
        let circle_len = layer.cache.circle_params.len() as u32;
        let circle_bytes = bytemuck::cast_slice(&layer.cache.circle_params);
        self.write_storage_buffer(
            queue,
            &mut layer.gpu.circle_ssbo,
            &format!("{}_circle_ssbo", layer.name),
            BufferUsages::STORAGE,
            circle_bytes,
        );
        layer.gpu.circle_count = circle_len;

        // ---- 1. ShapeParams SSBO ----
        let outline_len = layer.cache.outline_params.len() as u32;
        let outline_bytes = bytemuck::cast_slice(&layer.cache.outline_params);
        self.write_storage_buffer(
            queue,
            &mut layer.gpu.outline_shapes_ssbo,
            &format!("{}_outline_shapes_ssbo", layer.name),
            BufferUsages::STORAGE,
            outline_bytes,
        );
        layer.gpu.outline_count = outline_len;

        // ---- 2. Polygon vertex buffer (vec2<f32>) ----
        let poly_verts = &layer.cache.outline_poly_vertices;
        let poly_vcount = poly_verts.len() as u32;

        if poly_vcount > 0 {
            // upload real polygon vertices
            let bytes = bytemuck::cast_slice(poly_verts);
            self.write_storage_buffer(
                queue,
                &mut layer.gpu.outline_poly_vertices_ssbo,
                &format!("{}_outline_poly_ssbo", layer.name),
                BufferUsages::STORAGE,
                bytes,
            );
        } else {
            // No polygon outlines → still must provide SOME buffer to satisfy wgpu layout
            if layer.gpu.outline_poly_vertices_ssbo.is_none() {
                layer.gpu.outline_poly_vertices_ssbo =
                    Some(self.device.create_buffer(&BufferDescriptor {
                        label: Some(&format!("{}_outline_poly_dummy", layer.name)),
                        size: 16, // one vec2<f32> worth of space
                        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));
            }
        }

        let handle_len = layer.cache.handle_params.len() as u32;
        let handle_bytes = bytemuck::cast_slice(&layer.cache.handle_params);
        self.write_storage_buffer(
            queue,
            &mut layer.gpu.handle_ssbo,
            &format!("{}_handle_ssbo", layer.name),
            BufferUsages::STORAGE,
            handle_bytes,
        );
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
                layer.gpu.poly_vbo = Some(self.device.create_buffer(&BufferDescriptor {
                    label: Some(&format!("{}_poly_vbo", layer.name)),
                    size: bytes.len() as u64,
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            queue.write_buffer(layer.gpu.poly_vbo.as_ref().unwrap(), 0, bytes);
        }
        layer.gpu.poly_count = poly_count;

        // ---- polygon infos and edges (SSBOs) ----
        let mut infos: Vec<PolygonInfoGpu> = Vec::with_capacity(layer.polygons.len());
        let mut edges: Vec<PolygonEdgeGpu> = Vec::new();

        for poly in &layer.polygons {
            let edge_offset = edges.len() as u32;
            let mut edge_count = 0u32;

            let n = poly.vertices.len();
            if n >= 2 {
                for i in 0..n {
                    let a = poly.vertices[i].pos;
                    let b = poly.vertices[(i + 1) % n].pos;
                    edges.push(PolygonEdgeGpu { p0: a, p1: b });
                    edge_count += 1;
                }
            }

            infos.push(PolygonInfoGpu {
                edge_offset,
                edge_count,
                _pad0: [0, 0],
            });
        }

        if !infos.is_empty() {
            let bytes = bytemuck::cast_slice(&infos);
            let need_new = layer
                .gpu
                .poly_info_ssbo
                .as_ref()
                .map(|b| b.size() < bytes.len() as u64)
                .unwrap_or(true);
            if need_new {
                layer.gpu.poly_info_ssbo = Some(self.device.create_buffer(&BufferDescriptor {
                    label: Some(&format!("{}_poly_info_ssbo", layer.name)),
                    size: bytes.len() as u64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            queue.write_buffer(layer.gpu.poly_info_ssbo.as_ref().unwrap(), 0, bytes);
        } else {
            layer.gpu.poly_info_ssbo = None;
        }

        if !edges.is_empty() {
            let bytes = bytemuck::cast_slice(&edges);
            let need_new = layer
                .gpu
                .poly_edge_ssbo
                .as_ref()
                .map(|b| b.size() < bytes.len() as u64)
                .unwrap_or(true);
            if need_new {
                layer.gpu.poly_edge_ssbo = Some(self.device.create_buffer(&BufferDescriptor {
                    label: Some(&format!("{}_poly_edge_ssbo", layer.name)),
                    size: bytes.len() as u64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            queue.write_buffer(layer.gpu.poly_edge_ssbo.as_ref().unwrap(), 0, bytes);
        } else {
            layer.gpu.poly_edge_ssbo = None;
        }

        // ---- text (VBO) : build glyphs for this layer only ----
        let mut text_vertices: Vec<UiVertexText> = Vec::new();

        if let Some(atlas) = &self.pipelines.text_atlas {
            for tp in &mut layer.cache.texts {
                let pad = 4.0; // same as your selection outline pad

                // tp is TextParams (render-time)
                // must find corresponding original UiButtonText to get caret
                let mut maybe_caret = None;

                if let Some(ref text_id) = tp.id {
                    for original in &layer.texts {
                        if original.id.as_ref() == Some(text_id) {
                            maybe_caret = Some(original.caret);

                            break;
                        }
                    }
                }

                let caret_index = maybe_caret.unwrap_or(tp.text.len());
                tp.glyph_bounds.clear();

                // ---- measure + render glyphs ----
                let mut min_x = f32::MAX;
                let mut min_y = f32::MAX;
                let mut max_x = f32::MIN;
                let mut max_y = f32::MIN;

                let mut pen_x = tp.pos[0];
                let base_y = tp.pos[1] + atlas.line_height;

                // caret position tracker
                let mut caret_x = pen_x;

                let mut char_i = 0;

                let mut original_text: Option<&mut UiButtonText> = None;

                if let Some(ref text_id) = tp.id {
                    for o in &mut layer.texts {
                        if o.id.as_ref() == Some(text_id) {
                            original_text = Some(o);
                            break;
                        }
                    }
                }

                if let Some(orig) = &mut original_text {
                    orig.glyph_bounds.clear();
                }

                for ch in tp.text.chars() {
                    if let Some(g) = atlas.glyphs.get(&(ch, tp.px)) {
                        // compute rounded glyph quad first
                        let x0 = (pen_x + g.bearing_x).round();
                        let y0 = (base_y - g.bearing_y).round();
                        let x1 = x0 + g.w;
                        let y1 = y0 + g.h;

                        // *** NEW CARET LOGIC ***
                        // caret BEFORE this character uses x0, not pen_x
                        if char_i == caret_index {
                            caret_x = x0;
                        }

                        if let Some(orig) = &mut original_text {
                            orig.glyph_bounds.push((x0, x1));
                        }

                        tp.glyph_bounds.push((x0, x1));

                        // bounding box
                        min_x = min_x.min(x0);
                        min_y = min_y.min(y0);
                        max_x = max_x.max(x1);
                        max_y = max_y.max(y1);

                        // triangles
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

                    char_i += 1;
                }

                // caret at end → use last glyph’s right edge
                if caret_index == tp.text.len() {
                    if let Some((_, last_x1)) = tp.glyph_bounds.last() {
                        caret_x = *last_x1;
                    } else {
                        caret_x = tp.pos[0];
                    }
                }

                // ---- bounding box for the whole text ----
                if max_x < min_x {
                    tp.natural_width = 0.0;
                    tp.natural_height = atlas.line_height;
                } else {
                    let padded_min_x = min_x - pad;
                    let padded_min_y = min_y - pad;
                    let padded_max_x = max_x + pad;
                    let padded_max_y = max_y + pad;

                    tp.natural_width = padded_max_x - padded_min_x;
                    tp.natural_height = padded_max_y - padded_min_y;
                }

                let mut being_edited = false;
                let mut being_hovered = false;
                // ---- write back natural_width/natural_height to UiButtonText ----
                if let Some(orig) = &mut original_text {
                    orig.natural_width = tp.natural_width;
                    orig.natural_height = tp.natural_height;
                    being_edited = orig.being_edited;
                    being_hovered = orig.being_hovered;
                    orig.being_hovered = false;
                    orig.just_unhovered = true;
                }

                // ---- check if this text is selected ----
                let mut is_selected = false;
                if let Some(ref text_id) = tp.id {
                    let sel = &ui_runtime.selected_ui_element_primary;
                    if sel.active
                        && sel.element_id == *text_id
                        && sel.layer_name == layer.name
                        && sel.menu_name == *menu_name
                    {
                        is_selected = true;
                    }
                }

                if let Some(orig) = &mut original_text {
                    if orig.has_selection {
                        let (l, r) = if orig.sel_start < orig.sel_end {
                            (orig.sel_start, orig.sel_end)
                        } else {
                            (orig.sel_end, orig.sel_start)
                        };

                        if l < r && l < orig.glyph_bounds.len() {
                            let x_start = orig.glyph_bounds[l].0;
                            let x_end = if r == 0 {
                                x_start
                            } else if r - 1 < orig.glyph_bounds.len() {
                                orig.glyph_bounds[r - 1].1
                            } else {
                                orig.glyph_bounds.last().unwrap().1
                            };

                            let y0 = min_y - 2.0;
                            let y1 = max_y + 2.0;

                            let col = [0.3, 0.5, 1.0, 0.35];
                            let uv = [-1.0, -1.0];

                            text_vertices.extend_from_slice(&[
                                UiVertexText {
                                    pos: [x_start, y0],
                                    uv,
                                    color: col,
                                },
                                UiVertexText {
                                    pos: [x_end, y0],
                                    uv,
                                    color: col,
                                },
                                UiVertexText {
                                    pos: [x_end, y1],
                                    uv,
                                    color: col,
                                },
                                UiVertexText {
                                    pos: [x_start, y0],
                                    uv,
                                    color: col,
                                },
                                UiVertexText {
                                    pos: [x_end, y1],
                                    uv,
                                    color: col,
                                },
                                UiVertexText {
                                    pos: [x_start, y1],
                                    uv,
                                    color: col,
                                },
                            ]);
                        }
                    }
                }

                // ---- editor_mode outline (light rectangle) ----
                if ui_runtime.editor_mode && !being_edited && !is_selected {
                    // rectangle bounds from natural dimensions
                    let x0 = min_x - pad;
                    let y0 = min_y - pad;
                    let x1 = max_x + pad;
                    let y1 = max_y + pad;

                    // very soft color, hover brightens it
                    let base_alpha = if being_hovered { 0.30 } else { 0.01 };
                    let col = [0.9, 0.9, 1.0, base_alpha];
                    let uv = [-1.0, -1.0];

                    // outline thickness
                    let t = 1.5;

                    let mut push_quad = |xa: f32, ya: f32, xb: f32, yb: f32| {
                        text_vertices.extend_from_slice(&[
                            UiVertexText {
                                pos: [xa, ya],
                                uv,
                                color: col,
                            },
                            UiVertexText {
                                pos: [xb, ya],
                                uv,
                                color: col,
                            },
                            UiVertexText {
                                pos: [xb, yb],
                                uv,
                                color: col,
                            },
                            UiVertexText {
                                pos: [xa, ya],
                                uv,
                                color: col,
                            },
                            UiVertexText {
                                pos: [xb, yb],
                                uv,
                                color: col,
                            },
                            UiVertexText {
                                pos: [xa, yb],
                                uv,
                                color: col,
                            },
                        ]);
                    };

                    // top
                    push_quad(x0, y0, x1, y0 + t);
                    // bottom
                    push_quad(x0, y1 - t, x1, y1);
                    // left
                    push_quad(x0, y0, x0 + t, y1);
                    // right
                    push_quad(x1 - t, y0, x1, y1);
                }

                // ---- corner brackets (NOT editing, but selected) ----
                if is_selected && !being_edited {
                    // base size / padding
                    let base_len = 6.0;
                    let base_pad = 4.0;
                    let thick = 2.0;

                    // hover → push brackets further out
                    let hover_factor = if being_hovered { 1.6 } else { 1.0 };

                    let br = base_len * hover_factor;
                    let pad = base_pad * hover_factor;

                    let x0 = min_x - pad;
                    let y0 = min_y - pad;
                    let x1 = max_x + pad;
                    let y1 = max_y + pad;

                    let c = [1.0, 0.85, 0.2, 1.0]; // gold-ish
                    let uv = [-1.0, -1.0];

                    // small helper to push a quad
                    let mut push_quad = |xa: f32, ya: f32, xb: f32, yb: f32| {
                        text_vertices.extend_from_slice(&[
                            UiVertexText {
                                pos: [xa, ya],
                                uv,
                                color: c,
                            },
                            UiVertexText {
                                pos: [xb, ya],
                                uv,
                                color: c,
                            },
                            UiVertexText {
                                pos: [xb, yb],
                                uv,
                                color: c,
                            },
                            UiVertexText {
                                pos: [xa, ya],
                                uv,
                                color: c,
                            },
                            UiVertexText {
                                pos: [xb, yb],
                                uv,
                                color: c,
                            },
                            UiVertexText {
                                pos: [xa, yb],
                                uv,
                                color: c,
                            },
                        ]);
                    };

                    // top-left
                    push_quad(x0, y0, x0 + br, y0 + thick); // horizontal
                    push_quad(x0, y0, x0 + thick, y0 + br); // vertical

                    // top-right
                    push_quad(x1 - br, y0, x1, y0 + thick);
                    push_quad(x1 - thick, y0, x1, y0 + br);

                    // bottom-left
                    push_quad(x0, y1 - thick, x0 + br, y1);
                    push_quad(x0, y1 - br, x0 + thick, y1);

                    // bottom-right
                    push_quad(x1 - br, y1 - thick, x1, y1);
                    push_quad(x1 - thick, y1 - br, x1, y1);
                }

                if being_edited {
                    let caret_width = 2.0;
                    let caret_offset_y = 4.0;

                    let x0 = caret_x;
                    let y0 = tp.pos[1] + caret_offset_y;
                    let x1 = caret_x + caret_width;
                    let y1 = tp.pos[1] + atlas.line_height + caret_offset_y;

                    let t = time_system.total_time * 3.0; // adjust speed here
                    let blink_alpha = 0.5 + 0.5 * t.cos(); // smooth
                    let caret_alpha = blink_alpha.clamp(0.0, 1.0);

                    let caret_color = [1.0, 1.0, 1.0, caret_alpha];

                    let uv = [-1.0, -1.0];

                    text_vertices.extend_from_slice(&[
                        UiVertexText {
                            pos: [x0, y0],
                            uv,
                            color: caret_color,
                        },
                        UiVertexText {
                            pos: [x1, y0],
                            uv,
                            color: caret_color,
                        },
                        UiVertexText {
                            pos: [x1, y1],
                            uv,
                            color: caret_color,
                        },
                        UiVertexText {
                            pos: [x0, y0],
                            uv,
                            color: caret_color,
                        },
                        UiVertexText {
                            pos: [x1, y1],
                            uv,
                            color: caret_color,
                        },
                        UiVertexText {
                            pos: [x0, y1],
                            uv,
                            color: caret_color,
                        },
                    ]);
                }
            }
        }

        // upload text_vertices into the VBO...

        let text_bytes = bytemuck::cast_slice(&text_vertices);
        if !text_vertices.is_empty() {
            let need_new = layer
                .gpu
                .text_vbo
                .as_ref()
                .map(|b| b.size() < text_bytes.len() as u64)
                .unwrap_or(true);
            if need_new {
                layer.gpu.text_vbo = Some(self.device.create_buffer(&BufferDescriptor {
                    label: Some(&format!("{}_text_vbo", layer.name)),
                    size: text_bytes.len() as u64,
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            queue.write_buffer(layer.gpu.text_vbo.as_ref().unwrap(), 0, text_bytes);
        }
        layer.gpu.text_count = text_vertices.len() as u32;
    }

    fn build_text_vertices(
        &self,
        layer: &mut RuntimeLayer,
        time_system: &TimeSystem,
        ui_runtime: &UiRuntime,
        menu_name: &String,
    ) -> Vec<UiVertexText> {
        let mut text_vertices: Vec<UiVertexText> = Vec::new();

        if let Some(atlas) = &self.pipelines.text_atlas {
            for tp in &mut layer.cache.texts {
                let pad = 4.0; // same as your selection outline pad

                // tp is TextParams (render-time)
                // must find corresponding original UiButtonText to get caret
                let mut maybe_caret = None;

                if let Some(ref text_id) = tp.id {
                    for original in &layer.texts {
                        if original.id.as_ref() == Some(text_id) {
                            maybe_caret = Some(original.caret);

                            break;
                        }
                    }
                }

                let caret_index = maybe_caret.unwrap_or(tp.text.len());
                tp.glyph_bounds.clear();

                // ---- measure + render glyphs ----
                let mut min_x = f32::MAX;
                let mut min_y = f32::MAX;
                let mut max_x = f32::MIN;
                let mut max_y = f32::MIN;

                let mut pen_x = tp.pos[0];
                let base_y = tp.pos[1] + atlas.line_height;

                // caret position tracker
                let mut caret_x = pen_x;

                let mut char_i = 0;

                let mut original_text: Option<&mut UiButtonText> = None;

                if let Some(ref text_id) = tp.id {
                    for o in &mut layer.texts {
                        if o.id.as_ref() == Some(text_id) {
                            original_text = Some(o);
                            break;
                        }
                    }
                }

                if let Some(orig) = &mut original_text {
                    orig.glyph_bounds.clear();
                }

                for ch in tp.text.chars() {
                    if let Some(g) = atlas.glyphs.get(&(ch, tp.px)) {
                        // compute rounded glyph quad first
                        let x0 = (pen_x + g.bearing_x).round();
                        let y0 = (base_y - g.bearing_y).round();
                        let x1 = x0 + g.w;
                        let y1 = y0 + g.h;

                        // *** NEW CARET LOGIC ***
                        // caret BEFORE this character uses x0, not pen_x
                        if char_i == caret_index {
                            caret_x = x0;
                        }

                        if let Some(orig) = &mut original_text {
                            orig.glyph_bounds.push((x0, x1));
                        }

                        tp.glyph_bounds.push((x0, x1));

                        // bounding box
                        min_x = min_x.min(x0);
                        min_y = min_y.min(y0);
                        max_x = max_x.max(x1);
                        max_y = max_y.max(y1);

                        // triangles
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

                    char_i += 1;
                }

                // caret at end → use last glyph’s right edge
                if caret_index == tp.text.len() {
                    if let Some((_, last_x1)) = tp.glyph_bounds.last() {
                        caret_x = *last_x1;
                    } else {
                        caret_x = tp.pos[0];
                    }
                }

                // ---- bounding box for the whole text ----
                if max_x < min_x {
                    tp.natural_width = 0.0;
                    tp.natural_height = atlas.line_height;
                } else {
                    let padded_min_x = min_x - pad;
                    let padded_min_y = min_y - pad;
                    let padded_max_x = max_x + pad;
                    let padded_max_y = max_y + pad;

                    tp.natural_width = padded_max_x - padded_min_x;
                    tp.natural_height = padded_max_y - padded_min_y;
                }

                let mut being_edited = false;
                let mut being_hovered = false;
                // ---- write back natural_width/natural_height to UiButtonText ----
                if let Some(orig) = &mut original_text {
                    orig.natural_width = tp.natural_width;
                    orig.natural_height = tp.natural_height;
                    being_edited = orig.being_edited;
                    being_hovered = orig.being_hovered;
                    orig.being_hovered = false;
                    orig.just_unhovered = true;
                }

                // ---- check if this text is selected ----
                let mut is_selected = false;
                if let Some(ref text_id) = tp.id {
                    let sel = &ui_runtime.selected_ui_element_primary;
                    if sel.active
                        && sel.element_id == *text_id
                        && sel.layer_name == layer.name
                        && sel.menu_name == *menu_name
                    {
                        is_selected = true;
                    }
                }

                if let Some(orig) = &mut original_text {
                    if orig.has_selection {
                        let (l, r) = if orig.sel_start < orig.sel_end {
                            (orig.sel_start, orig.sel_end)
                        } else {
                            (orig.sel_end, orig.sel_start)
                        };

                        if l < r && l < orig.glyph_bounds.len() {
                            let x_start = orig.glyph_bounds[l].0;
                            let x_end = if r == 0 {
                                x_start
                            } else if r - 1 < orig.glyph_bounds.len() {
                                orig.glyph_bounds[r - 1].1
                            } else {
                                orig.glyph_bounds.last().unwrap().1
                            };

                            let y0 = min_y - 2.0;
                            let y1 = max_y + 2.0;

                            let col = [0.3, 0.5, 1.0, 0.35];
                            let uv = [-1.0, -1.0];

                            text_vertices.extend_from_slice(&[
                                UiVertexText {
                                    pos: [x_start, y0],
                                    uv,
                                    color: col,
                                },
                                UiVertexText {
                                    pos: [x_end, y0],
                                    uv,
                                    color: col,
                                },
                                UiVertexText {
                                    pos: [x_end, y1],
                                    uv,
                                    color: col,
                                },
                                UiVertexText {
                                    pos: [x_start, y0],
                                    uv,
                                    color: col,
                                },
                                UiVertexText {
                                    pos: [x_end, y1],
                                    uv,
                                    color: col,
                                },
                                UiVertexText {
                                    pos: [x_start, y1],
                                    uv,
                                    color: col,
                                },
                            ]);
                        }
                    }
                }

                // ---- editor_mode outline (light rectangle) ----
                if ui_runtime.editor_mode && !being_edited && !is_selected {
                    // rectangle bounds from natural dimensions
                    let x0 = min_x - pad;
                    let y0 = min_y - pad;
                    let x1 = max_x + pad;
                    let y1 = max_y + pad;

                    // very soft color, hover brightens it
                    let base_alpha = if being_hovered { 0.30 } else { 0.01 };
                    let col = [0.9, 0.9, 1.0, base_alpha];
                    let uv = [-1.0, -1.0];

                    // outline thickness
                    let t = 1.5;

                    let mut push_quad = |xa: f32, ya: f32, xb: f32, yb: f32| {
                        text_vertices.extend_from_slice(&[
                            UiVertexText {
                                pos: [xa, ya],
                                uv,
                                color: col,
                            },
                            UiVertexText {
                                pos: [xb, ya],
                                uv,
                                color: col,
                            },
                            UiVertexText {
                                pos: [xb, yb],
                                uv,
                                color: col,
                            },
                            UiVertexText {
                                pos: [xa, ya],
                                uv,
                                color: col,
                            },
                            UiVertexText {
                                pos: [xb, yb],
                                uv,
                                color: col,
                            },
                            UiVertexText {
                                pos: [xa, yb],
                                uv,
                                color: col,
                            },
                        ]);
                    };

                    // top
                    push_quad(x0, y0, x1, y0 + t);
                    // bottom
                    push_quad(x0, y1 - t, x1, y1);
                    // left
                    push_quad(x0, y0, x0 + t, y1);
                    // right
                    push_quad(x1 - t, y0, x1, y1);
                }

                // ---- corner brackets (NOT editing, but selected) ----
                if is_selected && !being_edited {
                    // base size / padding
                    let base_len = 6.0;
                    let base_pad = 4.0;
                    let thick = 2.0;

                    // hover → push brackets further out
                    let hover_factor = if being_hovered { 1.6 } else { 1.0 };

                    let br = base_len * hover_factor;
                    let pad = base_pad * hover_factor;

                    let x0 = min_x - pad;
                    let y0 = min_y - pad;
                    let x1 = max_x + pad;
                    let y1 = max_y + pad;

                    let c = [1.0, 0.85, 0.2, 1.0]; // gold-ish
                    let uv = [-1.0, -1.0];

                    // small helper to push a quad
                    let mut push_quad = |xa: f32, ya: f32, xb: f32, yb: f32| {
                        text_vertices.extend_from_slice(&[
                            UiVertexText {
                                pos: [xa, ya],
                                uv,
                                color: c,
                            },
                            UiVertexText {
                                pos: [xb, ya],
                                uv,
                                color: c,
                            },
                            UiVertexText {
                                pos: [xb, yb],
                                uv,
                                color: c,
                            },
                            UiVertexText {
                                pos: [xa, ya],
                                uv,
                                color: c,
                            },
                            UiVertexText {
                                pos: [xb, yb],
                                uv,
                                color: c,
                            },
                            UiVertexText {
                                pos: [xa, yb],
                                uv,
                                color: c,
                            },
                        ]);
                    };

                    // top-left
                    push_quad(x0, y0, x0 + br, y0 + thick); // horizontal
                    push_quad(x0, y0, x0 + thick, y0 + br); // vertical

                    // top-right
                    push_quad(x1 - br, y0, x1, y0 + thick);
                    push_quad(x1 - thick, y0, x1, y0 + br);

                    // bottom-left
                    push_quad(x0, y1 - thick, x0 + br, y1);
                    push_quad(x0, y1 - br, x0 + thick, y1);

                    // bottom-right
                    push_quad(x1 - br, y1 - thick, x1, y1);
                    push_quad(x1 - thick, y1 - br, x1, y1);
                }

                if being_edited {
                    let caret_width = 2.0;
                    let caret_offset_y = 4.0;

                    let x0 = caret_x;
                    let y0 = tp.pos[1] + caret_offset_y;
                    let x1 = caret_x + caret_width;
                    let y1 = tp.pos[1] + atlas.line_height + caret_offset_y;

                    let t = time_system.total_time * 3.0; // adjust speed here
                    let blink_alpha = 0.5 + 0.5 * t.cos(); // smooth
                    let caret_alpha = blink_alpha.clamp(0.0, 1.0);

                    let caret_color = [1.0, 1.0, 1.0, caret_alpha];

                    let uv = [-1.0, -1.0];

                    text_vertices.extend_from_slice(&[
                        UiVertexText {
                            pos: [x0, y0],
                            uv,
                            color: caret_color,
                        },
                        UiVertexText {
                            pos: [x1, y0],
                            uv,
                            color: caret_color,
                        },
                        UiVertexText {
                            pos: [x1, y1],
                            uv,
                            color: caret_color,
                        },
                        UiVertexText {
                            pos: [x0, y0],
                            uv,
                            color: caret_color,
                        },
                        UiVertexText {
                            pos: [x1, y1],
                            uv,
                            color: caret_color,
                        },
                        UiVertexText {
                            pos: [x0, y1],
                            uv,
                            color: caret_color,
                        },
                    ]);
                }
            }
        }
        text_vertices
    }

    pub fn build_text_atlas(
        &mut self,
        device: &Device,
        queue: &Queue,
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
        let tex = device.create_texture(&TextureDescriptor {
            label: Some("Text Atlas"),
            size: Extent3d {
                width: atlas_w,
                height: atlas_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // For your wgpu build, bytes_per_row expects a plain u32
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &cpu_atlas,
            TexelCopyBufferLayout {
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

        let view = tex.create_view(&TextureViewDescriptor::default());
        let sampler = device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });

        // 5) Bind group for the text pipeline
        let text_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Text Atlas Bind Group"),
            layout: &self.pipelines.text_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
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
