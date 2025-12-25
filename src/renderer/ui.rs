use crate::renderer::ui_pipelines::UiPipelines;
use crate::renderer::ui_text::{
    Anchor, anchor_to_top_left, glyphs_to_vertices, render_corner_brackets, render_editor_caret,
    render_editor_outline, render_selection,
};
use crate::resources::TimeSystem;
use crate::ui::input::MouseState;
use crate::ui::ui_editor::{UiButtonLoader, UiRuntime};
use crate::ui::vertex::{
    PolygonEdgeGpu, PolygonInfoGpu, RuntimeLayer, UiButtonText, UiVertexPoly, UiVertexText,
};
use std::ops::Range;
use std::path::Path;
use wgpu::*;
use winit::dpi::PhysicalSize;

pub const PAD: i32 = 1;

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
    pub width: f32,
    pub height: f32, // px
    pub bearing_x: f32,
    pub bearing_y: f32,
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
    pub anchor: Option<Anchor>,
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
        if !ui.ui_runtime.show_gui {
            return;
        }
        let new_uniform = ScreenUniform {
            size: [size.0, size.1],
            time: time.total_time as f32,
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
                    if let Some(text_vbo) = &layer.gpu.text_vbo {
                        let z = layer.texts.iter().map(|t| t.z_index).max().unwrap_or(0);

                        cmds.push(DrawCmd {
                            z,
                            pipeline: &self.pipelines.text_pipeline,
                            bind_group0: &self.pipelines.uniform_bind_group,
                            bind_group1: Some(self.pipelines.text_bind_group.clone()),
                            vertex_buffer: Some(text_vbo),
                            vertex_range: 0..layer.gpu.text_count,
                            instance_range: 0..1,
                        });
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
        &mut self,
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
        let text_vertices =
            self.build_text_vertices(layer, time_system, ui_runtime, menu_name, queue);

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

    pub fn build_text_vertices(
        &mut self,
        layer: &mut RuntimeLayer,
        time_system: &TimeSystem,
        ui_runtime: &UiRuntime,
        menu_name: &String,
        queue: &Queue,
    ) -> Vec<UiVertexText> {
        let mut text_vertices: Vec<UiVertexText> = Vec::new();

        for tp in &mut layer.cache.texts {
            // ensure atlas has this size
            if !self.pipelines.text_atlas.metrics.contains_key(&tp.px) {
                self.pipelines
                    .text_atlas
                    .ensure_px_size(&self.device, queue, tp.px)
                    .expect("failed to ensure text atlas size");
                // quick sanity: atlas must have some pixels
                debug_assert!(
                    self.pipelines.text_atlas.cpu_atlas.iter().any(|&b| b != 0),
                    "text atlas empty after rasterize"
                );
                self.rebuild_text_bind_group()
            }

            let pad = 4.0; // same as your selection outline pad

            // find caret from original UiButtonText if present
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

            let metrics = &self.pipelines.text_atlas.metrics[&tp.px];
            let used_pos = anchor_to_top_left(
                tp.anchor.unwrap_or(Anchor::TopLeft),
                tp.pos,
                0.0,
                tp.natural_height,
            );

            let text_top = used_pos[1] + (tp.natural_height * 0.5);
            let baseline_y = text_top + metrics.ascent;
            let mut pen_x = used_pos[0];

            // find mutable original text if needed (for writing glyph bounds back)
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

            // caret position tracker
            let mut caret_x = pen_x;
            let mut char_i = 0;

            glyphs_to_vertices(
                &self.pipelines,
                &mut text_vertices,
                tp, // TextParams mutable ref
                &mut char_i,
                caret_index,
                baseline_y,
                &mut pen_x,
                &mut caret_x,
                &mut original_text,
                &mut min_x,
                &mut min_y,
                &mut max_x,
                &mut max_y,
            );

            // caret at end → last glyph right edge or pen_x
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
                tp.natural_height = metrics.line_height + 2.0 * pad;
            } else {
                let padded_min_x = min_x - pad;
                let padded_max_x = max_x + pad;

                tp.natural_width = padded_max_x - padded_min_x;
                tp.natural_height = metrics.line_height + 2.0 * pad;
            }

            // ---- write back natural_width/natural_height to UiButtonText ----
            let mut being_edited = false;
            let mut being_hovered = false;
            let mut is_input_box = false;
            if let Some(orig) = &mut original_text {
                orig.natural_width = tp.natural_width;
                orig.natural_height = tp.natural_height;
                orig.ascent = metrics.ascent; // <-- use per-size ascent
                being_edited = orig.being_edited;
                being_hovered = orig.being_hovered;
                orig.being_hovered = false;
                orig.just_unhovered = true;
                is_input_box = orig.input_box
            }

            // ---- selection / editor outlines / brackets etc ----
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

            if let Some(orig) = original_text {
                render_selection(orig, min_y, max_y, &mut text_vertices);
            }

            // editor outline
            if ui_runtime.editor_mode && !being_edited && !is_selected {
                render_editor_outline(
                    min_x,
                    min_y,
                    max_x,
                    max_y,
                    &mut text_vertices,
                    pad,
                    being_hovered,
                );
            }

            // corner brackets
            if is_selected && !being_edited {
                render_corner_brackets(
                    min_x,
                    min_y,
                    max_x,
                    max_y,
                    &mut text_vertices,
                    being_hovered,
                );
            }

            // caret rendering when editing
            if being_edited || (is_input_box && is_selected) {
                render_editor_caret(tp, caret_x, &mut text_vertices, metrics, time_system);
            }
        } // for tp

        text_vertices
    }

    pub fn rebuild_text_bind_group(&mut self) {
        let atlas = &self.pipelines.text_atlas;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Text Atlas Bind Group"),
            layout: &self.pipelines.text_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&atlas.sampler),
                },
            ],
        });

        self.pipelines.text_bind_group = bind_group;
    }
}
