use crate::renderer::ui::{
    CircleParams, HandleParams, OutlineParams, UiRenderer, make_poly_ssbo, upload_poly_vbo,
};
use crate::renderer::ui_text_rendering::{
    Anchor, anchor_to, render_corner_brackets, render_editor_caret, render_editor_outline,
    render_selection,
};
use crate::resources::Time;
use crate::ui::ui_touch_manager::{ElementRef, UiTouchManager};
use crate::ui::vertex::*;
use std::collections::HashMap;
use wgpu::{BufferDescriptor, BufferUsages, Queue};
use wgpu_text::glyph_brush::ab_glyph::Font;
use wgpu_text::glyph_brush::{
    BuiltInLineBreaker, Layout, OwnedSection, Section, Text, VerticalAlign,
};

pub fn upload_circles(ui_renderer: &mut UiRenderer, queue: &Queue, layer: &mut RuntimeLayer) {
    let circle_params: Vec<CircleParams> = layer.cache.iter_circles().cloned().collect();

    let circle_len = circle_params.len() as u32;
    let circle_bytes = bytemuck::cast_slice(&circle_params);
    ui_renderer.write_storage_buffer(
        queue,
        &mut layer.gpu.circle_ssbo,
        &format!("{}_circle_ssbo", layer.name),
        BufferUsages::STORAGE,
        circle_bytes,
    );
    layer.gpu.circle_count = circle_len;
}

pub fn upload_outlines(ui_renderer: &mut UiRenderer, queue: &Queue, layer: &mut RuntimeLayer) {
    // 1. ShapeParams SSBO
    let outline_params: Vec<OutlineParams> = layer.cache.iter_outlines().cloned().collect();

    let outline_len = outline_params.len() as u32;
    let outline_bytes = bytemuck::cast_slice(&outline_params);
    ui_renderer.write_storage_buffer(
        queue,
        &mut layer.gpu.outline_shapes_ssbo,
        &format!("{}_outline_shapes_ssbo", layer.name),
        BufferUsages::STORAGE,
        outline_bytes,
    );
    layer.gpu.outline_count = outline_len;

    // 2. Polygon vertex buffer (vec2<f32>)
    let poly_verts = &layer.cache.outline_poly_vertices;
    let poly_vcount = poly_verts.len() as u32;

    if poly_vcount > 0 {
        let bytes = bytemuck::cast_slice(poly_verts);
        ui_renderer.write_storage_buffer(
            queue,
            &mut layer.gpu.outline_poly_vertices_ssbo,
            &format!("{}_outline_poly_ssbo", layer.name),
            BufferUsages::STORAGE,
            bytes,
        );
    } else {
        // No polygon outlines → must provide dummy buffer for wgpu layout
        if layer.gpu.outline_poly_vertices_ssbo.is_none() {
            layer.gpu.outline_poly_vertices_ssbo =
                Some(ui_renderer.device.create_buffer(&BufferDescriptor {
                    label: Some(&format!("{}_outline_poly_dummy", layer.name)),
                    size: 48,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
        }
    }
}

pub fn upload_handles(ui_renderer: &mut UiRenderer, queue: &Queue, layer: &mut RuntimeLayer) {
    let handle_params: Vec<HandleParams> = layer.cache.iter_handles().cloned().collect();

    let handle_len = handle_params.len() as u32;
    let handle_bytes = bytemuck::cast_slice(&handle_params);
    ui_renderer.write_storage_buffer(
        queue,
        &mut layer.gpu.handle_ssbo,
        &format!("{}_handle_ssbo", layer.name),
        BufferUsages::STORAGE,
        handle_bytes,
    );
    layer.gpu.handle_count = handle_len;
}

pub fn upload_polygons(ui_renderer: &mut UiRenderer, queue: &Queue, layer: &mut RuntimeLayer) {
    // 1. Polygons (VBO)
    let mut poly_vertices: Vec<UiVertexPoly> = Vec::new();

    for poly in layer.cache.iter_polygons() {
        poly_vertices.extend_from_slice(poly);
    }

    let poly_count = poly_vertices.len() as u32;
    if poly_count > 0 {
        // Uses the provided helper function
        upload_poly_vbo(ui_renderer, poly_vertices, layer, queue);
    }
    layer.gpu.poly_count = poly_count;

    // 2. Polygon infos and edges (SSBOs)
    let mut infos: Vec<PolygonInfoGpu> = Vec::new();
    let mut edges: Vec<PolygonEdgeGpu> = Vec::new();

    for poly in layer.iter_polygons() {
        infos.reserve(1);
        make_poly_ssbo(&mut edges, poly, &mut infos);
    }

    upload_poly_metadata_ssbos(ui_renderer, queue, layer, &infos, &edges);
}
pub fn upload_rects(ui_renderer: &mut UiRenderer, queue: &Queue, layer: &mut RuntimeLayer) {
    let rects: Vec<RectGpu> = layer
        .cache
        .elements
        .iter()
        .filter_map(|e| {
            if let UiElementCache::Rect(r) = e {
                Some(*r)
            } else {
                None
            }
        })
        .collect();

    let rect_count = rects.len() as u32;

    if rect_count > 0 {
        let bytes = bytemuck::cast_slice(&rects);
        let need_new = layer
            .gpu
            .rect_ssbo
            .as_ref()
            .map(|b| b.size() < bytes.len() as u64)
            .unwrap_or(true);

        if need_new {
            layer.gpu.rect_ssbo = Some(ui_renderer.device.create_buffer(&BufferDescriptor {
                label: Some(&format!("{}_rect_ssbo", layer.name)),
                size: bytes.len() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        queue.write_buffer(layer.gpu.rect_ssbo.as_ref().unwrap(), 0, bytes);
    } else {
        layer.gpu.rect_ssbo = None;
    }

    layer.gpu.rect_count = rect_count;
}
pub fn upload_poly_metadata_ssbos(
    ui_renderer: &mut UiRenderer,
    queue: &Queue,
    layer: &mut RuntimeLayer,
    infos: &[PolygonInfoGpu],
    edges: &[PolygonEdgeGpu],
) {
    // Info SSBO
    if !infos.is_empty() {
        let bytes = bytemuck::cast_slice(infos);
        let need_new = layer
            .gpu
            .poly_info_ssbo
            .as_ref()
            .map(|b| b.size() < bytes.len() as u64)
            .unwrap_or(true);

        if need_new {
            layer.gpu.poly_info_ssbo = Some(ui_renderer.device.create_buffer(&BufferDescriptor {
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

    // Edge SSBO
    if !edges.is_empty() {
        let bytes = bytemuck::cast_slice(edges);
        let need_new = layer
            .gpu
            .poly_edge_ssbo
            .as_ref()
            .map(|b| b.size() < bytes.len() as u64)
            .unwrap_or(true);

        if need_new {
            layer.gpu.poly_edge_ssbo = Some(ui_renderer.device.create_buffer(&BufferDescriptor {
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
}

pub fn upload_text(
    ui_renderer: &mut UiRenderer,
    queue: &Queue,
    layer: &mut RuntimeLayer,
    time_system: &Time,
    touch_manager: &UiTouchManager,
    menu_name: &String,
) {
    let estimated = layer.cache.elements.len();
    let mut sections: Vec<OwnedSection> = Vec::with_capacity(estimated);
    let mut bounds_map: HashMap<String, (f32, f32)> = HashMap::with_capacity(estimated);
    let mut vertices: Vec<UiVertexText> = Vec::new();

    for tp in layer
        .cache
        .elements
        .iter_mut()
        .filter_map(UiElementCache::as_text_mut)
    {
        //println!("cached text layer: {}, name: {}", layer.name, tp.id);
        let layout = Layout::default()
            .v_align(VerticalAlign::Top)
            .line_breaker(BuiltInLineBreaker::AnyCharLineBreaker);
        let text = tp.text.replace("\\n", "\n");

        let Some(font) = ui_renderer.brush.fonts().first() else {
            continue;
        };
        let Some(px) = font.pt_to_px_scale(tp.pt) else {
            continue;
        };
        let s = Section::default().with_layout(layout).add_text(
            Text::new(&text)
                .with_scale(px) // px -> font size in pixels
                .with_color(tp.color),
        );
        //println!("SECTION cached text layer: {}, name: {}, SECTION: {:?}", layer.name, tp.id, s);
        if let Some(rect) = ui_renderer.brush.glyph_bounds(&s) {
            let (w, h) = (rect.width(), rect.height());
            bounds_map.insert(tp.id.clone(), (w, h));

            tp.width = w;
            tp.height = h;
            let pos = anchor_to(tp.anchor.unwrap_or(Anchor::Center), tp.pos, w, h);
            let s = s.with_screen_position((pos[0], pos[1]));
            sections.push(s.to_owned());
        } else {
            let s = s.with_screen_position(tp.pos);
            sections.push(s.to_owned());
        }
    }

    layer.gpu.text_sections = sections;
    let layer_name = layer.name.clone();
    for t in layer.iter_texts_mut() {
        if let Some(&(w, h)) = bounds_map.get(&t.id) {
            t.width = w;
            t.height = h;
        }
        let mut is_selected = false;
        if touch_manager.selection.is_selected(&ElementRef::new(
            menu_name,
            layer_name.as_str(),
            t.id.as_str(),
            ElementKind::Text,
        )) {
            is_selected = true;
        }
        let pos = anchor_to(
            t.anchor.unwrap_or(Anchor::Center),
            [t.x, t.y],
            t.width,
            t.height,
        );
        if is_selected && !t.being_edited {
            render_corner_brackets(
                pos[0],
                pos[1],
                pos[0] + t.width,
                pos[1] + t.height,
                &mut vertices,
                t.being_hovered,
            );
        }

        render_selection(t, &mut vertices);

        // editor outline
        if touch_manager.editor.enabled && !t.being_edited && !is_selected {
            let pad = 2.0;
            render_editor_outline(
                pos[0],
                pos[1],
                pos[0] + t.width,
                pos[1] + t.height,
                &mut vertices,
                pad,
                t.being_hovered,
            );
        }
        if t.being_edited || (t.input_box && is_selected) {
            render_editor_caret(ui_renderer, t, &mut vertices, time_system);
        }
    }

    if !vertices.is_empty() {
        let bytes = bytemuck::cast_slice(&vertices);
        let need_new = layer
            .gpu
            .text_misc_vbo
            .as_ref()
            .map(|b| b.size() < bytes.len() as u64)
            .unwrap_or(true);

        if need_new {
            layer.gpu.text_misc_vbo = Some(ui_renderer.device.create_buffer(&BufferDescriptor {
                label: Some(&format!("{} Text Miscellaneous Stuff VBO", layer.name)),
                size: bytes.len() as u64,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        layer.gpu.text_misc_vertex_count = vertices.len() as u32;
        queue.write_buffer(layer.gpu.text_misc_vbo.as_ref().unwrap(), 0, bytes);
    } else {
        layer.gpu.text_misc_vbo = None;
    }
}
