use crate::renderer::ui::{
    CircleParams, HandleParams, OutlineParams, UiRenderer, make_poly_ssbo, upload_poly_vbo,
};
use crate::renderer::ui_text_rendering::{
    anchor_to, render_corner_brackets, render_editor_caret, render_editor_outline, render_selection,
};
use crate::resources::Time;
use crate::ui::ui_touch_manager::{ElementRef, UiTouchManager};
use crate::ui::vertex::*;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue};

fn write_or_clear_buffer(
    device: &Device,
    queue: &Queue,
    buffer: &mut Option<Buffer>,
    label: String,
    usage: BufferUsages,
    bytes: &[u8],
) {
    if bytes.is_empty() {
        *buffer = None;
        return;
    }

    let need_new = buffer
        .as_ref()
        .map(|b| b.size() < bytes.len() as u64)
        .unwrap_or(true);

    if need_new {
        *buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some(&label),
            size: bytes.len() as u64,
            usage,
            mapped_at_creation: false,
        }));
    }

    queue.write_buffer(buffer.as_ref().unwrap(), 0, bytes);
}

pub fn upload_circles(ui_renderer: &mut UiRenderer, queue: &Queue, layer: &mut RuntimeLayer) {
    let circle_params: Vec<CircleParams> = layer
        .elements
        .iter()
        .enumerate()
        .filter_map(|(element_idx, element)| match element {
            UiElement::Circle(c) => c.cache.as_ref().map(|cache| {
                let mut value = *cache;
                value.depth = depth_for(layer.order as usize, element_idx);
                value
            }),
            _ => None,
        })
        .collect();

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
    let outline_params: Vec<OutlineParams> = layer
        .elements
        .iter()
        .enumerate()
        .filter_map(|(element_idx, element)| match element {
            UiElement::Outline(o) => o.cache.as_ref().map(|cache| {
                let mut value = *cache;
                value.depth = depth_for(layer.order as usize, element_idx);
                value
            }),
            _ => None,
        })
        .collect();

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

    let poly_verts = &layer.outline_poly_vertices;
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
    } else if layer.gpu.outline_poly_vertices_ssbo.is_none() {
        layer.gpu.outline_poly_vertices_ssbo =
            Some(ui_renderer.device.create_buffer(&BufferDescriptor {
                label: Some(&format!("{}_outline_poly_dummy", layer.name)),
                size: 48,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
    }
}

pub fn upload_handles(ui_renderer: &mut UiRenderer, queue: &Queue, layer: &mut RuntimeLayer) {
    let handle_params: Vec<HandleParams> = layer
        .elements
        .iter()
        .enumerate()
        .filter_map(|(element_idx, element)| match element {
            UiElement::Handle(h) => h.cache.as_ref().map(|cache| {
                let mut value = *cache;
                value.depth = depth_for(layer.order as usize, element_idx);
                value
            }),
            _ => None,
        })
        .collect();

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
    let mut poly_vertices: Vec<UiVertexPoly> = Vec::new();

    for (element_idx, element) in layer.elements.iter().enumerate() {
        if let UiElement::Polygon(poly) = element {
            if let Some(cached) = poly.cache.as_ref() {
                let mut cached = cached.clone();
                let depth = depth_for(layer.order as usize, element_idx);
                for v in &mut cached {
                    v.depth = depth;
                }
                poly_vertices.extend_from_slice(&cached);
            }
        }
    }

    let poly_count = poly_vertices.len() as u32;

    if poly_count > 0 {
        upload_poly_vbo(ui_renderer, poly_vertices, layer, queue);
    } else {
        layer.gpu.poly_vbo = None;
    }

    layer.gpu.poly_count = poly_count;

    let mut infos: Vec<PolygonInfoGpu> = Vec::new();
    let mut edges: Vec<PolygonEdgeGpu> = Vec::new();

    for element in &layer.elements {
        if let UiElement::Polygon(poly) = element {
            make_poly_ssbo(&mut edges, poly, &mut infos);
        }
    }

    upload_poly_metadata_ssbos(ui_renderer, queue, layer, &infos, &edges);
}

pub fn upload_rects(ui_renderer: &mut UiRenderer, queue: &Queue, layer: &mut RuntimeLayer) {
    let rects: Vec<RectParams> = layer
        .elements
        .iter()
        .enumerate()
        .filter_map(|(element_idx, element)| match element {
            UiElement::Rect(r) => r.cache.as_ref().map(|cache| {
                let mut value = *cache;
                value.depth = depth_for(layer.order as usize, element_idx);
                value
            }),
            _ => None,
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
    let mut vertices: Vec<UiVertexText> = Vec::new();
    let layer_name = layer.name.clone();

    for (element_idx, element) in layer.elements.iter_mut().enumerate() {
        let UiElement::Text(t) = element else {
            continue;
        };

        let Some(cache) = t.cache.as_mut() else {
            continue;
        };

        let depth = depth_for(layer.order as usize, element_idx);

        t.width = cache.width;
        t.height = cache.height;

        let pos = anchor_to(t.anchor, [t.x, t.y], t.width, t.height);

        let is_selected = touch_manager.selection.is_selected(&ElementRef::new(
            menu_name,
            layer_name.as_str(),
            t.id.as_str(),
            ElementKind::Text,
        ));

        let start = vertices.len();

        if is_selected && !t.being_edited {
            render_corner_brackets(
                pos[0],
                pos[1],
                pos[0] + t.width,
                pos[1] + t.height,
                &mut vertices,
                t.being_hovered,
                depth,
            );
            set_text_vertex_depth(&mut vertices[start..], depth);
        }

        let start = vertices.len();
        render_selection(t, &mut vertices, depth);
        set_text_vertex_depth(&mut vertices[start..], depth);

        if touch_manager.editor.enabled && !t.being_edited && !is_selected {
            let pad = 2.0;
            let start = vertices.len();
            render_editor_outline(
                pos[0],
                pos[1],
                pos[0] + t.width,
                pos[1] + t.height,
                &mut vertices,
                pad,
                t.being_hovered,
                depth,
            );
            set_text_vertex_depth(&mut vertices[start..], depth);
        }

        if t.being_edited {
            //|| (t.input_box && is_selected) {
            let start = vertices.len();
            render_editor_caret(ui_renderer, t, &mut vertices, time_system, depth);
            set_text_vertex_depth(&mut vertices[start..], depth);
        }
        let Some(cache) = t.cache.as_mut() else {
            continue;
        };
        cache.depth = depth;
        cache.pos = [t.x, t.y];
        cache.width = t.width;
        cache.height = t.height;
        cache.anchor = t.anchor;
        cache.color = t.color;
        cache.text = t.text.clone();
        cache.id = t.id.clone();
        cache.pt = t.pt;
    }

    let vertex_count = vertices.len() as u32;

    if vertex_count > 0 {
        let bytes = bytemuck::cast_slice(&vertices);
        let need_new = layer
            .gpu
            .text_misc_vbo
            .as_ref()
            .map(|b| b.size() < bytes.len() as u64)
            .unwrap_or(true);

        if need_new {
            layer.gpu.text_misc_vbo = Some(ui_renderer.device.create_buffer(&BufferDescriptor {
                label: Some(&format!("{}_text_misc_vbo", layer.name)),
                size: bytes.len() as u64,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        layer.gpu.text_misc_vertex_count = vertex_count;
        queue.write_buffer(layer.gpu.text_misc_vbo.as_ref().unwrap(), 0, bytes);
    } else {
        layer.gpu.text_misc_vbo = None;
        layer.gpu.text_misc_vertex_count = 0;
    }
}
pub fn depth_for(layer_order: usize, element_idx: usize) -> f32 {
    1.0 - (layer_order as f32 * 0.0001 + element_idx as f32 * 0.000001)
}

pub fn set_text_vertex_depth(vertices: &mut [UiVertexText], depth: f32) {
    for v in vertices {
        v.depth = depth;
    }
}
