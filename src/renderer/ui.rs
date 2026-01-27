use crate::paths::shader_dir;
use crate::renderer::core::color_target;
use crate::renderer::pipelines::Pipelines;
use crate::renderer::procedural_render_manager::{PipelineOptions, RenderManager};
use crate::renderer::ui_pipelines::UiPipelines;
use crate::renderer::ui_text_rendering::Anchor;
use crate::renderer::ui_upload::*;
use crate::resources::{InputState, TimeSystem};
use crate::ui::ui_editor::UiButtonLoader;
use crate::ui::ui_touch_manager::UiTouchManager;
use crate::ui::vertex::{
    PolygonEdgeGpu, PolygonInfoGpu, RuntimeLayer, UiButtonPolygon, UiElement, UiVertexPoly,
    UiVertexText,
};
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
    pub inside_border_thickness_percentage: f32,
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
            inside_border_thickness_percentage: 0.0,
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

            shape_data: [600.0, 600.0, 60.0, 0.1],
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

#[derive(Debug, Clone)]
pub struct TextParams {
    pub pos: [f32; 2],
    pub px: u16,
    pub color: [f32; 4],
    pub id_hash: f32,
    pub misc: [f32; 4], // [active, touched_time, is_down, id_hash]
    pub text: String,
    pub natural_width: f32,
    pub natural_height: f32,
    pub id: String,
    pub caret: usize,
    pub glyph_bounds: Vec<(f32, f32)>,
    pub anchor: Option<Anchor>,
}

impl Default for TextParams {
    fn default() -> Self {
        Self {
            pos: [0.0, 0.0],
            px: 0,
            color: [0.0, 0.0, 0.0, 0.0],
            id_hash: 0.0,
            misc: [0.0; 4], // active, touched_time, is_touched, id_hash

            text: "".to_string(),
            natural_width: 20.0,
            natural_height: 10.0,
            id: "None".to_string(),
            caret: 0,
            glyph_bounds: vec![],
            anchor: None,
        }
    }
}

pub struct UiRenderer {
    pub pipelines: UiPipelines,

    pub(crate) device: Device,
}

impl UiRenderer {
    pub fn new(
        device: &Device,
        format: TextureFormat,
        size: PhysicalSize<u32>,
        msaa_samples: u32,
    ) -> anyhow::Result<Self> {
        let pipelines = UiPipelines::new(device, format, msaa_samples, size)?;

        Ok(Self {
            pipelines,
            device: device.clone(),
        })
    }

    pub fn update(
        &mut self,
        ui_loader: &mut UiButtonLoader,
        time: &TimeSystem,
        input_state: &InputState,
        queue: &Queue,
        size: &PhysicalSize<u32>,
    ) {
        let new_uniform = ScreenUniform {
            size: [size.width as f32, size.height as f32],
            time: time.total_time as f32,
            enable_dither: 1,
            mouse: input_state.mouse.pos.to_array(),
        };

        queue.write_buffer(
            &self.pipelines.uniform_buffer,
            0,
            bytemuck::bytes_of(&new_uniform),
        );
        for (menu_name, menu) in ui_loader.menus.iter_mut().filter(|(_, menu)| menu.active) {
            let dirty_indices: Vec<usize> = menu
                .layers
                .iter()
                .enumerate()
                .filter(|(_, l)| l.active && l.dirty.any())
                .map(|(i, _)| i)
                .collect();

            for idx in dirty_indices {
                menu.rebuild_layer_cache_index(idx, &ui_loader.touch_manager.runtimes);
                let layer = &mut menu.layers[idx];
                self.upload_layer(queue, layer, &ui_loader.touch_manager, time, menu_name);
            }
        }
    }

    pub fn reload_shaders(&mut self) -> anyhow::Result<()> {
        self.pipelines.reload_shaders()?;
        Ok(())
    }

    pub fn render<'a>(
        &self,
        render_manager: &mut RenderManager,
        pass: &mut RenderPass<'a>,
        ui: &mut UiButtonLoader,
        pipelines: &Pipelines,
    ) {
        if !ui.touch_manager.options.show_gui {
            return;
        }

        ui.update_dynamic_texts();

        let mut layers_to_render: Vec<&RuntimeLayer> = Vec::new();
        for (_, menu) in ui.menus.iter().filter(|(_, m)| m.active) {
            for layer in menu.layers.iter().filter(|l| l.active) {
                layers_to_render.push(layer);
            }
        }
        layers_to_render.sort_by_key(|l| l.order);

        let depth_stencil = None;
        let msaa = self.pipelines.msaa_samples;

        for layer in layers_to_render {
            // Build per-layer bind groups (same as before)
            let circle_bg = if layer.gpu.circle_count > 0 {
                Some(self.device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &self.pipelines.circle_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: layer.gpu.circle_ssbo.as_ref().unwrap().as_entire_binding(),
                    }],
                }))
            } else {
                None
            };

            let handle_bg = if layer.gpu.handle_count > 0 {
                Some(self.device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &self.pipelines.handle_layout,
                    entries: &[BindGroupEntry {
                        binding: 0,
                        resource: layer.gpu.handle_ssbo.as_ref().unwrap().as_entire_binding(),
                    }],
                }))
            } else {
                None
            };

            let poly_bg = if layer.gpu.poly_count > 0 {
                if let (Some(info), Some(edges)) =
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
                }
            } else {
                None
            };

            let outline_bg = if layer.gpu.outline_count > 0 {
                Some(
                    self.device.create_bind_group(&BindGroupDescriptor {
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
                    }),
                )
            } else {
                None
            };

            let mut circle_idx: u32 = 0;
            let mut handle_idx: u32 = 0;
            let mut outline_idx: u32 = 0;
            let mut poly_vtx_offset: u32 = 0;

            for element in &layer.elements {
                match element {
                    UiElement::Circle(_) => {
                        if let Some(bg1) = &circle_bg {
                            // Circle
                            let targets = color_target(pipelines, Some(BlendState::ALPHA_BLENDING));
                            let options = PipelineOptions {
                                topology: PrimitiveTopology::TriangleStrip,
                                msaa_samples: msaa,
                                depth_stencil: depth_stencil.clone(),
                                vertex_layouts: vec![UiVertexPoly::desc()],
                                cull_mode: None,
                                shadow_pass: false,
                                fullscreen_pass: false,
                                targets,
                            };
                            render_manager.render_with_bind_groups(
                                "UI Circle",
                                &shader_dir().join("ui_circle.wgsl"),
                                options,
                                &[
                                    &self.pipelines.uniform_layout,
                                    &self.pipelines.circle_layout,
                                ],
                                &[&self.pipelines.uniform_bind_group, bg1],
                                pass,
                            );
                            pass.set_vertex_buffer(0, self.pipelines.quad_buffer.slice(..));
                            pass.draw(0..4, circle_idx..circle_idx + 1);

                            // Glow
                            let targets =
                                color_target(pipelines, Some(self.pipelines.additive_blend));
                            let options = PipelineOptions {
                                topology: PrimitiveTopology::TriangleStrip,
                                msaa_samples: msaa,
                                depth_stencil: depth_stencil.clone(),
                                vertex_layouts: vec![UiVertexPoly::desc()],
                                cull_mode: None,
                                shadow_pass: false,
                                fullscreen_pass: false,
                                targets,
                            };
                            render_manager.render_with_bind_groups(
                                "UI Glow",
                                &shader_dir().join("ui_circle_glow.wgsl"),
                                options,
                                &[
                                    &self.pipelines.uniform_layout,
                                    &self.pipelines.circle_layout,
                                ],
                                &[&self.pipelines.uniform_bind_group, bg1],
                                pass,
                            );
                            pass.set_vertex_buffer(0, self.pipelines.quad_buffer.slice(..));
                            pass.draw(0..4, circle_idx..circle_idx + 1);
                        }
                        circle_idx += 1;
                    }

                    UiElement::Handle(_) => {
                        if let Some(bg1) = &handle_bg {
                            let targets = color_target(pipelines, self.pipelines.good_blend);
                            let options = PipelineOptions {
                                topology: PrimitiveTopology::TriangleStrip,
                                msaa_samples: msaa,
                                depth_stencil: depth_stencil.clone(),
                                vertex_layouts: vec![UiVertexPoly::desc()],
                                cull_mode: None,
                                shadow_pass: false,
                                fullscreen_pass: false,
                                targets,
                            };
                            render_manager.render_with_bind_groups(
                                "UI Handle",
                                &shader_dir().join("ui_handle.wgsl"),
                                options,
                                &[
                                    &self.pipelines.uniform_layout,
                                    &self.pipelines.handle_layout,
                                ],
                                &[&self.pipelines.uniform_bind_group, bg1],
                                pass,
                            );
                            pass.set_vertex_buffer(0, self.pipelines.handle_quad_buffer.slice(..));
                            pass.draw(0..4, handle_idx..handle_idx + 1);
                        }
                        handle_idx += 1;
                    }

                    UiElement::Polygon(poly) => {
                        let count = poly.tri_count.saturating_mul(3);
                        let targets = color_target(pipelines, Some(BlendState::ALPHA_BLENDING));
                        if let (Some(bg1), Some(vbo)) = (&poly_bg, &layer.gpu.poly_vbo) {
                            let options = PipelineOptions {
                                topology: PrimitiveTopology::TriangleStrip,
                                msaa_samples: msaa,
                                depth_stencil: depth_stencil.clone(),
                                vertex_layouts: vec![UiVertexPoly::desc()],
                                cull_mode: None,
                                shadow_pass: false,
                                fullscreen_pass: false,
                                targets,
                            };
                            render_manager.render_with_bind_groups(
                                "UI Polygon",
                                &shader_dir().join("ui_polygon.wgsl"),
                                options,
                                &[
                                    &self.pipelines.uniform_layout,
                                    &self.pipelines.polygon_layout,
                                ],
                                &[&self.pipelines.uniform_bind_group, bg1],
                                pass,
                            );
                            pass.set_vertex_buffer(0, vbo.slice(..));
                            pass.draw(poly_vtx_offset..poly_vtx_offset + count, 0..1);
                        }
                        poly_vtx_offset = poly_vtx_offset.saturating_add(count);
                    }

                    UiElement::Outline(_) => {
                        if let Some(bg1) = &outline_bg {
                            let targets = color_target(pipelines, self.pipelines.good_blend);
                            let options = PipelineOptions {
                                topology: PrimitiveTopology::TriangleStrip,
                                msaa_samples: msaa,
                                depth_stencil: depth_stencil.clone(),
                                vertex_layouts: vec![UiVertexPoly::desc()],
                                cull_mode: None,
                                shadow_pass: false,
                                fullscreen_pass: false,
                                targets,
                            };
                            render_manager.render_with_bind_groups(
                                "UI Outline",
                                &shader_dir().join("ui_shape_outline.wgsl"),
                                options,
                                &[
                                    &self.pipelines.uniform_layout,
                                    &self.pipelines.outline_layout,
                                ],
                                &[&self.pipelines.uniform_bind_group, bg1],
                                pass,
                            );
                            pass.set_vertex_buffer(0, self.pipelines.quad_buffer.slice(..));
                            pass.draw(0..4, outline_idx..outline_idx + 1);
                        }
                        outline_idx += 1;
                    }

                    UiElement::Text(_) => {
                        // handled after other elements for proper layering
                    }
                }
            }

            // Text on top
            if layer.gpu.text_count > 0 {
                if let Some(text_vbo) = &layer.gpu.text_vbo {
                    let targets = color_target(pipelines, Some(BlendState::ALPHA_BLENDING));
                    let options = PipelineOptions {
                        topology: PrimitiveTopology::TriangleList,
                        msaa_samples: msaa,
                        depth_stencil: depth_stencil.clone(),
                        vertex_layouts: vec![UiVertexText::desc()],
                        cull_mode: None,
                        shadow_pass: false,
                        fullscreen_pass: false,
                        targets,
                    };
                    render_manager.render_with_bind_groups(
                        "UI Text",
                        &shader_dir().join("text.wgsl"),
                        options,
                        &[&self.pipelines.uniform_layout, &self.pipelines.text_layout],
                        &[
                            &self.pipelines.uniform_bind_group,
                            &self.pipelines.text_bind_group,
                        ],
                        pass,
                    );
                    pass.set_vertex_buffer(0, text_vbo.slice(..));
                    pass.draw(0..layer.gpu.text_count, 0..1);
                }
            }
        }
    }

    pub(crate) fn write_storage_buffer(
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

    fn upload_layer(
        &mut self,
        queue: &Queue,
        layer: &mut RuntimeLayer,
        touch_manager: &UiTouchManager,
        time_system: &TimeSystem,
        menu_name: &String,
    ) {
        upload_circles(self, queue, layer);
        upload_outlines(self, queue, layer);
        upload_handles(self, queue, layer);
        upload_polygons(self, queue, layer);
        upload_text(self, queue, layer, time_system, touch_manager, menu_name);
    }

    pub fn rebuild_text_bind_group(&mut self) {
        let atlas = &self.pipelines.text_atlas;

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Text Atlas Bind Group"),
            layout: &self.pipelines.text_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&atlas.view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&atlas.sampler),
                },
            ],
        });

        self.pipelines.text_bind_group = bind_group;
    }
}

pub(crate) fn make_poly_ssbo(
    edges: &mut Vec<PolygonEdgeGpu>,
    poly: &UiButtonPolygon,
    infos: &mut Vec<PolygonInfoGpu>,
) {
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

pub(crate) fn upload_poly_vbo(
    ui_renderer: &mut UiRenderer,
    poly_vertices: Vec<UiVertexPoly>,
    layer: &mut RuntimeLayer,
    queue: &Queue,
) {
    let bytes = bytemuck::cast_slice(&poly_vertices);
    let need_new = layer
        .gpu
        .poly_vbo
        .as_ref()
        .map(|b| b.size() < bytes.len() as u64)
        .unwrap_or(true);
    if need_new {
        layer.gpu.poly_vbo = Some(ui_renderer.device.create_buffer(&BufferDescriptor {
            label: Some(&format!("{}_poly_vbo", layer.name)),
            size: bytes.len() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }
    queue.write_buffer(layer.gpu.poly_vbo.as_ref().unwrap(), 0, bytes);
}
