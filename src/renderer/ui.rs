use crate::data::Settings;
use crate::helpers::paths::{data_dir, shader_dir};
use crate::renderer::pipelines::{COLOR_FORMAT, Pipelines};
use crate::renderer::render_core::{create_color_attachment_clear, create_color_attachment_load};
use crate::renderer::render_passes::{color_target, color_target_ui};
use crate::renderer::ui_pipelines::{Background, UI_DEPTH_FORMAT, UiPipelines, multisample_state};
use crate::renderer::ui_text_rendering::{Anchor, anchor_to};
use crate::renderer::ui_upload::*;
use crate::resources::Time;
use crate::ui::input::Input;
use crate::ui::ui_editor::Ui;
use crate::ui::ui_touch_manager::UiTouchManager;
use crate::ui::vertex::{
    PolygonEdgeGpu, PolygonInfoGpu, RuntimeLayer, UiButtonPolygon, UiButtonText, UiElement,
    UiVertexPoly, UiVertexText,
};
use glyphon::{
    Cache, FontSystem, Resolution, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer,
    Viewport, fontdb,
};
use std::fs;
use wgpu::*;
use wgpu_render_manager::pipelines::{FragmentOption, PipelineOptions};
use wgpu_render_manager::renderer::RenderManager;
use wgpu_text::glyph_brush::ab_glyph::FontArc;
use winit::dpi::PhysicalSize;
const UI_COMPARE_FUNCTION: CompareFunction = CompareFunction::Always; // TODO: I have to make layer order more explicit and stuff and handle colliding layer orders and APs! Less function is correct. Back is 1.0 front is 0.0 depth.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScreenUniform {
    pub size: [f32; 2],
    pub time: f32,
    pub enable_dither: u32, // use 0 = off, 1 = on
    pub mouse: [f32; 2],    // position!
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
    pub depth: f32,
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
            depth: 0.0,
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OutlineParams {
    pub(crate) mode: f32,          // 0.0 = circle, 1.0 = polygon
    pub(crate) vertex_offset: u32, // polygon vertices start index (ignored for circle)
    pub(crate) vertex_count: u32,  // polygon vertex count (ignored for circle)
    pub(crate) depth: f32,

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
            depth: 1.0,

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
    pub depth: f32,
    pub _pad0: [f32; 3],
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
            depth: 0.0,
            _pad0: [0.0; 3],
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
    pub depth: f32,
    pub _pad0: [f32; 3],
}
impl Default for PolygonOutlineParams {
    fn default() -> Self {
        Self {
            center_radius_border: [0.0; 4], // cx, cy, radius, thickness
            dash_color: [0.0; 4],
            dash_misc: [0.0; 4], // (dash_len, dash_spacing, dash_roundness, speed)
            misc: [0.0; 4],      // active, touched_time, is_touched, id_hash
            depth: 0.0,
            _pad0: [0.0; 3],
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextParams {
    pub pos: [f32; 2],
    pub pt: f32,
    pub color: [f32; 4],
    pub id_hash: f32,
    pub misc: [f32; 4], // [active, touched_time, is_down, id_hash]
    pub text: String,
    pub width: f32,
    pub height: f32,
    pub id: String,
    pub caret: usize,
    pub anchor: Anchor,
    pub depth: f32,
}

impl Default for TextParams {
    fn default() -> Self {
        Self {
            pos: [0.0, 0.0],
            pt: 14.0,
            color: [0.0, 0.0, 0.0, 0.0],
            id_hash: 0.0,
            misc: [0.0; 4], // active, touched_time, is_touched, id_hash

            text: "".to_string(),
            width: 20.0,
            height: 10.0,
            id: "None".to_string(),
            caret: 0,
            anchor: Anchor::default(),
            depth: 0.0,
        }
    }
}

pub struct UiRenderer {
    pub pipelines: UiPipelines,

    pub font_system: FontSystem,
    pub swash_cache: SwashCache,
    pub text_atlas: TextAtlas,
    pub text_renderer: TextRenderer,
    pub viewport: Viewport,

    pub device: Device,
    pub font_arc: FontArc,
}

impl UiRenderer {
    pub fn new(
        device: &Device,
        queue: &Queue,
        config: &SurfaceConfiguration,
        size: PhysicalSize<u32>,
        msaa_samples: u32,
    ) -> anyhow::Result<Self> {
        let pipelines = UiPipelines::new(device, config, msaa_samples, size)?;

        let mut font_system = FontSystem::new();

        let swash_cache = SwashCache::new();
        let cache = Cache::new(&device);
        let mut text_atlas = TextAtlas::new(device, queue, &cache, COLOR_FORMAT);
        let text_renderer = TextRenderer::new(
            &mut text_atlas,
            device,
            MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            Some(DepthStencilState {
                format: UI_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(UI_COMPARE_FUNCTION),
                stencil: Default::default(),
                bias: Default::default(),
            }),
        );

        let mut viewport = Viewport::new(device, &cache);

        viewport.update(
            queue,
            Resolution {
                width: config.width,
                height: config.height,
            },
        );
        let use_system_font = true;
        let system_font_family = "Noto Sans";
        let font_data = if use_system_font {
            let face_id = font_system
                .db()
                .faces()
                .find(|face| {
                    face.families
                        .iter()
                        .any(|(family, _)| family.eq_ignore_ascii_case(system_font_family))
                })
                .map(|face| face.id)
                .ok_or_else(|| anyhow::anyhow!("System font '{}' not found", system_font_family))?;

            let face = font_system
                .db()
                .face(face_id)
                .ok_or_else(|| anyhow::anyhow!("Failed to retrieve system font face"))?;

            match &face.source {
                fontdb::Source::Binary(data) => data.as_ref().as_ref().to_vec(),
                fontdb::Source::File(path) => fs::read(path)?,
                fontdb::Source::SharedFile(path, _) => fs::read(path)?,
            }
        } else {
            let dir = data_dir("ui_data/ttf");

            let font_path = fs::read_dir(dir)?
                .filter_map(Result::ok)
                .find(|e| e.path().extension().map(|x| x == "ttf").unwrap_or(false))
                .ok_or_else(|| anyhow::anyhow!("No TTF found"))?
                .path();

            fs::read(font_path)?
        };

        let font_arc = FontArc::try_from_vec(font_data.clone()).expect("Failed to load font data");

        if !use_system_font {
            font_system.db_mut().load_font_data(font_data.clone());
        }

        Ok(Self {
            pipelines,

            font_system,
            swash_cache,
            text_atlas,
            text_renderer,
            viewport,
            font_arc,

            device: device.clone(),
        })
    }

    pub fn update(
        &mut self,
        ui_loader: &mut Ui,
        time: &Time,
        input_state: &Input,
        queue: &Queue,
        window_size: PhysicalSize<u32>,
        settings: &Settings,
    ) {
        let new_uniform = ScreenUniform {
            size: [window_size.width as f32, window_size.height as f32],
            time: time.total_time as f32,
            enable_dither: 1,
            mouse: input_state.mouse.pos.to_array(),
        };
        queue.write_buffer(
            &self.pipelines.uniform_buffer,
            0,
            bytemuck::bytes_of(&new_uniform),
        );
        let bg_uniform = Background {
            primary_color: settings.background_color.map(|c| c + 0.01f32),
            secondary_color: settings.background_color.map(|c| c + 0.05f32),
            block_size: 30f32,
            warp_strength: 0.02,
            warp_radius: 0.10,
            time_scale: 0.03,
            wave_strength: 0.002,
            _padding: [0.0; 3],
        };
        queue.write_buffer(
            &self.pipelines.background_buffer,
            0,
            bytemuck::bytes_of(&bg_uniform),
        );

        for (menu_name, menu) in ui_loader.menus.iter_mut().filter(|(_, menu)| menu.active) {
            let dirty_indices: Vec<usize> = menu
                .layers
                .iter()
                .enumerate()
                .filter(|(_, l)| l.active && l.dirty.any())
                .map(|(i, _)| i)
                .collect();

            let mut ap_layers = vec![];

            for idx in dirty_indices {
                ap_layers.append(&mut menu.rebuild_layer_cache_index(
                    settings,
                    &ui_loader.variables,
                    &mut self.font_system,
                    idx,
                    &ui_loader.touch_manager.runtimes,
                    &ui_loader.aps,
                    window_size,
                ));

                let layer = &mut menu.layers[idx];
                self.upload_layer(queue, layer, &ui_loader.touch_manager, time, menu_name);
            }

            for ap_layer in ap_layers {
                menu.layers.push(ap_layer);
                let Some(layer) = menu.layers.last_mut() else {
                    continue;
                };
                self.upload_layer(queue, layer, &ui_loader.touch_manager, time, menu_name);
                layer.dirty.mark_all();
            }
        }
    }

    pub fn render(
        &mut self,
        render_manager: &mut RenderManager,
        encoder: &mut CommandEncoder,
        queue: &Queue,
        ui: &mut Ui,
        pipelines: &Pipelines,
        settings: &Settings,
    ) {
        self.draw_background(render_manager, encoder, pipelines, settings);

        let color_attachment = create_color_attachment_clear(&pipelines.resolved.ui);
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Main Pass (UI Elements)"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &self.pipelines.depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        if !ui.touch_manager.options.show_gui {
            return;
        }

        ui.update_dynamic_texts(settings);

        let mut layers_to_render: Vec<&RuntimeLayer> = Vec::new();
        for (_, menu) in ui.menus.iter().filter(|(_, m)| m.active) {
            for layer in menu.layers.iter().filter(|l| l.active) {
                layers_to_render.push(layer);
            }
        }
        layers_to_render.sort_by_key(|l| l.order);

        let mut pending_text: Vec<(TextArea<'_>, f32)> = Vec::new();

        for layer in layers_to_render {
            let circle_bg = self.circle_bind_group(layer);
            let handle_bg = self.handle_bind_group(layer);
            let poly_bg = self.polygon_bind_group(layer);
            let outline_bg = self.outline_bind_group(layer);
            let rect_bg = self.rect_bind_group(layer);

            let mut circle_idx: u32 = 0;
            let mut handle_idx: u32 = 0;
            let mut outline_idx: u32 = 0;
            let mut poly_vtx_offset: u32 = 0;
            let mut rect_idx: u32 = 0;

            for (element_idx, element) in layer.elements.iter().enumerate() {
                if !element.is_active() {
                    continue;
                }
                match element {
                    UiElement::Circle(c) => {
                        if let Some(bg1) = circle_bg.as_ref() {
                            self.draw_circle(render_manager, pipelines, &mut pass, bg1, circle_idx);
                        }

                        circle_idx += 1;
                    }
                    UiElement::Handle(h) => {
                        if let Some(bg1) = handle_bg.as_ref() {
                            self.draw_handle(render_manager, pipelines, &mut pass, bg1, handle_idx);
                        }

                        handle_idx += 1;
                    }
                    UiElement::Polygon(poly) => {
                        if let (Some(bg1), Some(vbo)) = (poly_bg.as_ref(), &layer.gpu.poly_vbo) {
                            let count = poly.tri_count.saturating_mul(3);
                            let start = poly_vtx_offset;
                            poly_vtx_offset += count;
                            self.draw_polygon(
                                render_manager,
                                pipelines,
                                &mut pass,
                                bg1,
                                vbo,
                                start,
                                count,
                            );
                        }
                    }
                    UiElement::Outline(o) => {
                        if let Some(bg1) = outline_bg.as_ref() {
                            self.draw_outline(
                                render_manager,
                                pipelines,
                                &mut pass,
                                bg1,
                                outline_idx,
                            );
                        }

                        outline_idx += 1;
                    }
                    UiElement::Text(t) => {
                        if let Some(texts) =
                            self.make_text_areas(t, layer.order as usize, element_idx)
                        {
                            pending_text.extend(texts);
                        }
                    }
                    UiElement::Rect(rect) => {
                        if let Some(bg1) = rect_bg.as_ref() {
                            self.draw_rect(
                                render_manager,
                                pipelines,
                                &mut pass,
                                bg1,
                                rect_idx,
                                rect.blur > 0.0,
                            );
                        }

                        rect_idx += 1;
                    }
                    UiElement::Advanced(_) => {}
                }
            }
            if let Some(text_misc_vbo) = layer.gpu.text_misc_vbo.as_ref() {
                let targets = color_target_ui(pipelines, Some(BlendState::ALPHA_BLENDING));
                let options = &PipelineOptions {
                    topology: PrimitiveTopology::TriangleList,
                    multisample_state: multisample_state(1),
                    depth_stencil: Some(DepthStencilState {
                        format: UI_DEPTH_FORMAT,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(UI_COMPARE_FUNCTION),
                        stencil: StencilState::default(),
                        bias: DepthBiasState::default(),
                    }),
                    vertex_layouts: vec![Some(UiVertexText::desc())],
                    fragment: FragmentOption::Default { targets },
                    ..Default::default()
                };

                render_manager.render(
                    &[],
                    &shader_dir().join("ui_triangles.wgsl"),
                    options,
                    &[&self.pipelines.uniform_buffer],
                    &mut pass,
                );
                pass.set_vertex_buffer(0, text_misc_vbo.slice(..));
                pass.draw(0..layer.gpu.text_misc_vertex_count, 0..1);
            }
        }

        self.flush_text_batch(&mut pass, queue, &mut pending_text);
    }

    pub fn write_storage_buffer(
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
        time_system: &Time,
        menu_name: &String,
    ) {
        upload_circles(self, queue, layer);
        upload_outlines(self, queue, layer);
        upload_handles(self, queue, layer);
        upload_polygons(self, queue, layer);
        upload_rects(self, queue, layer);
        upload_text(self, queue, layer, time_system, touch_manager, menu_name);
    }
}

impl UiRenderer {
    fn f32_to_u8(v: f32) -> u8 {
        (v.clamp(0.0, 1.0) * 255.0).round() as u8
    }

    fn color_from_rgba(color: [f32; 4]) -> glyphon::Color {
        glyphon::Color::rgba(
            Self::f32_to_u8(color[0]),
            Self::f32_to_u8(color[1]),
            Self::f32_to_u8(color[2]),
            Self::f32_to_u8(color[3]),
        )
    }

    fn draw_background(
        &self,
        render_manager: &mut RenderManager,
        encoder: &mut CommandEncoder,
        pipelines: &Pipelines,
        settings: &Settings,
    ) {
        if !settings.editor_mode {
            return;
        }

        let color_attachment = create_color_attachment_load(
            &pipelines.msaa.hdr,
            &pipelines.resolved.hdr,
            settings.msaa_samples,
        );

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Main Pass (UI Editor Background)"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        let background_shader = &shader_dir().join("ui_background.wgsl");
        let targets = color_target(pipelines, Some(BlendState::ALPHA_BLENDING));
        let options = &PipelineOptions {
            topology: PrimitiveTopology::TriangleList,
            multisample_state: multisample_state(settings.msaa_samples),
            depth_stencil: None,
            vertex_layouts: vec![],
            cull_mode: None,
            fragment: FragmentOption::Default { targets },
            ..Default::default()
        };

        render_manager.render(
            &[],
            background_shader,
            options,
            &[
                &self.pipelines.uniform_buffer,
                &self.pipelines.background_buffer,
            ],
            &mut pass,
        );
        pass.draw(0..3, 0..1);
        pass.forget_lifetime();
    }

    fn circle_bind_group(&self, layer: &RuntimeLayer) -> Option<BindGroup> {
        if layer.gpu.circle_count == 0 {
            return None;
        }

        Some(self.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.circle_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: layer.gpu.circle_ssbo.as_ref()?.as_entire_binding(),
            }],
        }))
    }

    fn handle_bind_group(&self, layer: &RuntimeLayer) -> Option<BindGroup> {
        if layer.gpu.handle_count == 0 {
            return None;
        }

        Some(self.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.handle_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: layer.gpu.handle_ssbo.as_ref()?.as_entire_binding(),
            }],
        }))
    }

    fn polygon_bind_group(&self, layer: &RuntimeLayer) -> Option<BindGroup> {
        if layer.gpu.poly_count == 0 {
            return None;
        }

        let (Some(info), Some(edges)) = (&layer.gpu.poly_info_ssbo, &layer.gpu.poly_edge_ssbo)
        else {
            return None;
        };

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
    }

    fn outline_bind_group(&self, layer: &RuntimeLayer) -> Option<BindGroup> {
        if layer.gpu.outline_count == 0 {
            return None;
        }

        Some(
            self.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &self.pipelines.outline_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: layer.gpu.outline_shapes_ssbo.as_ref()?.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: layer
                            .gpu
                            .outline_poly_vertices_ssbo
                            .as_ref()?
                            .as_entire_binding(),
                    },
                ],
            }),
        )
    }

    fn rect_bind_group(&self, layer: &RuntimeLayer) -> Option<BindGroup> {
        let ssbo = layer.gpu.rect_ssbo.as_ref()?;

        Some(self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("rect_bind_group"),
            layout: &self.pipelines.rect_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: ssbo.as_entire_binding(),
            }],
        }))
    }

    fn draw_circle(
        &self,
        render_manager: &mut RenderManager,
        pipelines: &Pipelines,
        pass: &mut RenderPass<'_>,
        circle_bg: &BindGroup,
        this_idx: u32,
    ) {
        let targets = color_target_ui(pipelines, Some(self.pipelines.additive_blend));
        let options = &PipelineOptions {
            topology: PrimitiveTopology::TriangleStrip,
            multisample_state: multisample_state(1),
            depth_stencil: Some(DepthStencilState {
                format: UI_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(UI_COMPARE_FUNCTION),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            vertex_layouts: vec![Some(UiVertexPoly::desc())],
            fragment: FragmentOption::Default { targets },
            ..Default::default()
        };

        render_manager.render_with_layouts(
            &shader_dir().join("ui_circle_glow.wgsl"),
            &[
                &self.pipelines.uniform_layout,
                &self.pipelines.circle_layout,
            ],
            &[&self.pipelines.uniform_bind_group, circle_bg],
            options,
            pass,
        );
        pass.set_vertex_buffer(0, self.pipelines.quad_buffer.slice(..));
        pass.draw(0..4, this_idx..this_idx + 1);

        let targets = color_target_ui(pipelines, Some(BlendState::ALPHA_BLENDING));
        let options = &PipelineOptions {
            topology: PrimitiveTopology::TriangleStrip,
            multisample_state: multisample_state(1),
            depth_stencil: Some(DepthStencilState {
                format: UI_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(UI_COMPARE_FUNCTION),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            vertex_layouts: vec![Some(UiVertexPoly::desc())],
            fragment: FragmentOption::Default { targets },
            ..Default::default()
        };

        render_manager.render_with_layouts(
            &shader_dir().join("ui_circle.wgsl"),
            &[
                &self.pipelines.uniform_layout,
                &self.pipelines.circle_layout,
            ],
            &[&self.pipelines.uniform_bind_group, circle_bg],
            options,
            pass,
        );
        pass.set_vertex_buffer(0, self.pipelines.quad_buffer.slice(..));
        pass.draw(0..4, this_idx..this_idx + 1);
    }

    fn draw_handle(
        &self,
        render_manager: &mut RenderManager,
        pipelines: &Pipelines,
        pass: &mut RenderPass<'_>,
        handle_bg: &BindGroup,
        this_idx: u32,
    ) {
        let targets = color_target_ui(pipelines, self.pipelines.good_blend);
        let options = &PipelineOptions {
            topology: PrimitiveTopology::TriangleStrip,
            multisample_state: multisample_state(1),
            depth_stencil: Some(DepthStencilState {
                format: UI_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(UI_COMPARE_FUNCTION),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            vertex_layouts: vec![Some(UiVertexPoly::desc())],
            fragment: FragmentOption::Default { targets },
            ..Default::default()
        };

        render_manager.render_with_layouts(
            &shader_dir().join("ui_handle.wgsl"),
            &[
                &self.pipelines.uniform_layout,
                &self.pipelines.handle_layout,
            ],
            &[&self.pipelines.uniform_bind_group, handle_bg],
            options,
            pass,
        );
        pass.set_vertex_buffer(0, self.pipelines.handle_quad_buffer.slice(..));
        pass.draw(0..4, this_idx..this_idx + 1);
    }

    fn draw_polygon(
        &self,
        render_manager: &mut RenderManager,
        pipelines: &Pipelines,
        pass: &mut RenderPass<'_>,
        poly_bg: &BindGroup,
        vbo: &wgpu::Buffer,
        start: u32,
        count: u32,
    ) {
        let targets = color_target_ui(pipelines, Some(BlendState::ALPHA_BLENDING));
        let options = &PipelineOptions {
            topology: PrimitiveTopology::TriangleStrip,
            multisample_state: multisample_state(1),
            depth_stencil: Some(DepthStencilState {
                format: UI_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(UI_COMPARE_FUNCTION),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            vertex_layouts: vec![Some(UiVertexPoly::desc())],
            fragment: FragmentOption::Default { targets },
            ..Default::default()
        };

        render_manager.render_with_layouts(
            &shader_dir().join("ui_polygon.wgsl"),
            &[
                &self.pipelines.uniform_layout,
                &self.pipelines.polygon_layout,
            ],
            &[&self.pipelines.uniform_bind_group, poly_bg],
            options,
            pass,
        );
        pass.set_vertex_buffer(0, vbo.slice(..));
        pass.draw(start..start + count, 0..1);
    }

    fn draw_outline(
        &self,
        render_manager: &mut RenderManager,
        pipelines: &Pipelines,
        pass: &mut RenderPass<'_>,
        outline_bg: &BindGroup,
        this_idx: u32,
    ) {
        let targets = color_target_ui(pipelines, self.pipelines.good_blend);
        let options = &PipelineOptions {
            topology: PrimitiveTopology::TriangleStrip,
            multisample_state: multisample_state(1),
            depth_stencil: Some(DepthStencilState {
                format: UI_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(UI_COMPARE_FUNCTION),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            vertex_layouts: vec![Some(UiVertexPoly::desc())],
            fragment: FragmentOption::Default { targets },
            ..Default::default()
        };

        render_manager.render_with_layouts(
            &shader_dir().join("ui_shape_outline.wgsl"),
            &[
                &self.pipelines.uniform_layout,
                &self.pipelines.outline_layout,
            ],
            &[&self.pipelines.uniform_bind_group, outline_bg],
            options,
            pass,
        );
        pass.set_vertex_buffer(0, self.pipelines.quad_buffer.slice(..));
        pass.draw(0..4, this_idx..this_idx + 1);
    }

    fn draw_rect(
        &self,
        render_manager: &mut RenderManager,
        pipelines: &Pipelines,
        pass: &mut RenderPass<'_>,
        rect_bg: &BindGroup,
        this_idx: u32,
        needs_blur: bool,
    ) {
        let targets = color_target_ui(pipelines, Some(self.pipelines.additive_blend));
        let options = &PipelineOptions {
            topology: PrimitiveTopology::TriangleStrip,
            multisample_state: multisample_state(1),
            depth_stencil: Some(DepthStencilState {
                format: UI_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(UI_COMPARE_FUNCTION),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            vertex_layouts: vec![Some(UiVertexPoly::desc())],
            fragment: FragmentOption::Default { targets },
            ..Default::default()
        };

        render_manager.render_with_layouts(
            &shader_dir().join("ui_rect_glow.wgsl"),
            &[&self.pipelines.uniform_layout, &self.pipelines.rect_layout],
            &[&self.pipelines.uniform_bind_group, rect_bg],
            options,
            pass,
        );
        pass.set_vertex_buffer(0, self.pipelines.quad_buffer.slice(..));
        pass.draw(0..4, this_idx..this_idx + 1);
        if needs_blur {
            let targets = color_target_ui(pipelines, Some(self.pipelines.additive_blend));
            let options = &PipelineOptions {
                topology: PrimitiveTopology::TriangleStrip,
                multisample_state: multisample_state(1),
                depth_stencil: Some(DepthStencilState {
                    format: UI_DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(UI_COMPARE_FUNCTION),
                    stencil: StencilState::default(),
                    bias: DepthBiasState::default(),
                }),
                vertex_layouts: vec![Some(UiVertexPoly::desc())],
                fragment: FragmentOption::Default { targets },
                ..Default::default()
            };

            render_manager.render_with_layouts_and_textures(
                &[&pipelines.resolved.hdr],
                &shader_dir().join("ui_rect_blur.wgsl"),
                &[&self.pipelines.rect_layout],
                &[rect_bg],
                options,
                &[&self.pipelines.uniform_buffer],
                pass,
            );
            pass.set_vertex_buffer(0, self.pipelines.quad_buffer.slice(..));
            pass.draw(0..4, this_idx..this_idx + 1);
        }

        let good_blend = BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
        };

        let targets = color_target_ui(pipelines, Some(good_blend));
        let options = &PipelineOptions {
            topology: PrimitiveTopology::TriangleStrip,
            multisample_state: multisample_state(1),
            depth_stencil: Some(DepthStencilState {
                format: UI_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(UI_COMPARE_FUNCTION),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            vertex_layouts: vec![Some(UiVertexPoly::desc())],
            fragment: FragmentOption::Default { targets },
            ..Default::default()
        };

        render_manager.render_with_layouts(
            &shader_dir().join("ui_rect.wgsl"),
            &[&self.pipelines.uniform_layout, &self.pipelines.rect_layout],
            &[&self.pipelines.uniform_bind_group, rect_bg],
            options,
            pass,
        );
        pass.set_vertex_buffer(0, self.pipelines.quad_buffer.slice(..));
        pass.draw(0..4, this_idx..this_idx + 1);
    }

    fn make_text_areas<'a>(
        &self,
        text: &'a UiButtonText,
        layer_order: usize,
        element_idx: usize,
    ) -> Option<Vec<(TextArea<'a>, f32)>> {
        let cache = text.cache.as_ref()?;

        let width = cache.width.max(1.0);
        let height = cache.height.max(1.0);
        let top_left = anchor_to(cache.anchor, cache.pos, width, height);
        let (left, top) = (top_left[0], top_left[1]);

        let depth = depth_for(layer_order, element_idx);

        let border_size = 1.0; // TODO.:Fo ork... uhh... FORK Glyphon and add text border support in the shader and Rust neatly in the text buffer or whatever.    SHIT! I forked Glyphon and I must now fork cosmic-text too!!
        let bounds = TextBounds {
            left: (left - border_size - 2.0).floor() as i32,
            top: (top - border_size - 2.0).floor() as i32,
            right: (left + width + border_size + 2.0).ceil() as i32,
            bottom: (top + height + border_size + 2.0).ceil() as i32,
        };

        let offsets = [
            (-border_size, -border_size),
            (0.0, -border_size),
            (border_size, -border_size),
            (-border_size, 0.0),
            (border_size, 0.0),
            (-border_size, border_size),
            (0.0, border_size),
            (border_size, border_size),
        ];

        let mut result = Vec::with_capacity(9);

        for (dx, dy) in offsets {
            result.push((
                TextArea {
                    buffer: &text.buffer,
                    left: left + dx,
                    top: top + dy,
                    scale: 1.0,
                    bounds,
                    default_color: glyphon::Color::rgba(0, 0, 0, 255),
                    custom_glyphs: &[],
                },
                depth,
            ));
        }

        result.push((
            TextArea {
                buffer: &text.buffer,
                left,
                top,
                scale: 1.0,
                bounds,
                default_color: Self::color_from_rgba(cache.color),
                custom_glyphs: &[],
            },
            depth,
        ));

        Some(result)
    }

    fn flush_text_batch<'a>(
        &mut self,
        pass: &mut RenderPass<'a>,
        queue: &Queue,
        batch: &mut Vec<(TextArea<'a>, f32)>,
    ) {
        if batch.is_empty() {
            return;
        }

        let depths: Vec<f32> = batch.iter().map(|(_, depth)| *depth).collect();

        let areas = batch.drain(..).map(|(area, _)| area);

        if let Err(e) = self.text_renderer.prepare_with_depth_and_custom(
            &self.device,
            queue,
            &mut self.font_system,
            &mut self.text_atlas,
            &self.viewport,
            areas,
            &mut self.swash_cache,
            |index| depths[index],
            |_| None,
        ) {
            println!("{}", e);
            return;
        }

        if let Err(e) = self
            .text_renderer
            .render(&self.text_atlas, &self.viewport, pass)
        {
            println!("{}", e);
        }
    }
}

pub fn make_poly_ssbo(
    edges: &mut Vec<PolygonEdgeGpu>,
    poly: &UiButtonPolygon,
    infos: &mut Vec<PolygonInfoGpu>,
) {
    let edge_offset = edges.len() as u32;
    let mut edge_count = 0u32;

    let n = poly.scaled_vertices().len();
    if n >= 2 {
        for i in 0..n {
            let a = poly.scaled_vertices()[i].pos;
            let b = poly.scaled_vertices()[(i + 1) % n].pos;
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

pub fn upload_poly_vbo(
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
