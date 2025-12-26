use crate::renderer::ui::UiRenderer;
use crate::renderer::ui_text::{
    Anchor, anchor_to_top_left, glyphs_to_vertices, render_corner_brackets, render_editor_caret,
    render_editor_outline, render_selection,
};
use crate::resources::TimeSystem;
use crate::ui::ui_editor::{
    PolygonEdgeGpu, PolygonInfoGpu, RuntimeLayer, UiButtonText, UiRuntime, UiVertexPoly,
    UiVertexText,
};
use wgpu::{BufferDescriptor, BufferUsages, Queue};

pub fn upload_circles(ui_renderer: &mut UiRenderer, queue: &Queue, layer: &mut RuntimeLayer) {
    let circle_len = layer.cache.circle_params.len() as u32;
    let circle_bytes = bytemuck::cast_slice(&layer.cache.circle_params);
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
    let outline_len = layer.cache.outline_params.len() as u32;
    let outline_bytes = bytemuck::cast_slice(&layer.cache.outline_params);
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
                    size: 16, // one vec2<f32>
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
        }
    }
}

pub fn upload_handles(ui_renderer: &mut UiRenderer, queue: &Queue, layer: &mut RuntimeLayer) {
    let handle_len = layer.cache.handle_params.len() as u32;
    let handle_bytes = bytemuck::cast_slice(&layer.cache.handle_params);
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
    let mut poly_vertices: Vec<UiVertexPoly> =
        Vec::with_capacity(layer.cache.polygon_vertices.len());
    poly_vertices.extend_from_slice(&layer.cache.polygon_vertices);

    let poly_count = poly_vertices.len() as u32;
    if poly_count > 0 {
        // Uses the provided helper function
        crate::renderer::ui::upload_poly_vbo(ui_renderer, poly_vertices, layer, queue);
    }
    layer.gpu.poly_count = poly_count;

    // 2. Polygon infos and edges (SSBOs)
    let mut infos: Vec<PolygonInfoGpu> = Vec::with_capacity(layer.polygons.len());
    let mut edges: Vec<PolygonEdgeGpu> = Vec::new();

    for poly in &layer.polygons {
        // Uses the provided helper function
        crate::renderer::ui::make_poly_ssbo(&mut edges, poly, &mut infos);
    }

    upload_poly_metadata_ssbos(ui_renderer, queue, layer, &infos, &edges);
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
    time_system: &TimeSystem,
    ui_runtime: &UiRuntime,
    menu_name: &String,
) {
    let text_vertices = build_text_vertices(
        ui_renderer,
        layer,
        time_system,
        ui_runtime,
        menu_name,
        queue,
    );
    let text_bytes = bytemuck::cast_slice(&text_vertices);

    if !text_vertices.is_empty() {
        let need_new = layer
            .gpu
            .text_vbo
            .as_ref()
            .map(|b| b.size() < text_bytes.len() as u64)
            .unwrap_or(true);

        if need_new {
            layer.gpu.text_vbo = Some(ui_renderer.device.create_buffer(&BufferDescriptor {
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
    ui_renderer: &mut UiRenderer,
    layer: &mut RuntimeLayer,
    time_system: &TimeSystem,
    ui_runtime: &UiRuntime,
    menu_name: &String,
    queue: &Queue,
) -> Vec<UiVertexText> {
    let mut text_vertices: Vec<UiVertexText> = Vec::new();

    for tp in &mut layer.cache.texts {
        // ensure atlas has this size
        if !ui_renderer
            .pipelines
            .text_atlas
            .metrics
            .contains_key(&tp.px)
        {
            ui_renderer
                .pipelines
                .text_atlas
                .ensure_px_size(&ui_renderer.device, queue, tp.px)
                .expect("failed to ensure text atlas size");
            // quick sanity: atlas must have some pixels
            debug_assert!(
                ui_renderer
                    .pipelines
                    .text_atlas
                    .cpu_atlas
                    .iter()
                    .any(|&b| b != 0),
                "text atlas empty after rasterize"
            );
            ui_renderer.rebuild_text_bind_group()
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

        let metrics = &ui_renderer.pipelines.text_atlas.metrics[&tp.px];
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
            &ui_renderer.pipelines,
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
