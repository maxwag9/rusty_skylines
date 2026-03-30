use crate::renderer::ui::{CircleParams, HandleParams, OutlineParams, TextParams};
use crate::renderer::ui_text_rendering::{Anchor, anchor_to};
use crate::ui::actions::style_to_u32;
use crate::ui::helper::triangulate_polygon;
use crate::ui::ui_editor::Ui;
use crate::ui::ui_runtime::UiRuntimes;
use crate::ui::ui_touch_manager::ElementRef;
use crate::ui::vertex::*;
use bytemuck::Zeroable;
use wgpu_text::TextBrush;
use wgpu_text::glyph_brush::ab_glyph::{Font, Point, Rect};

pub fn rebuild_text_cache(
    brush: &TextBrush,
    layer: &mut RuntimeLayer,
    rebuilt: &mut LayerDirty,
    runtime: &UiRuntimes,
) {
    for (element, cache_element) in layer
        .elements
        .iter_mut()
        .zip(layer.cache.elements.iter_mut())
    {
        if !element.is_active() {
            continue;
        }
        if let (UiElement::Text(t), UiElementCache::Text(cached)) = (element, cache_element) {
            let (rt, hash) = runtime_info(runtime, &t.id);
            let Some(font) = brush.fonts().first() else {
                return;
            };

            let mut glyph_bounds: Vec<Rect> = Vec::new();
            let mut char_spans: Vec<std::ops::Range<usize>> = Vec::new();

            let Some(px_scale) = font.pt_to_px_scale(t.pt) else {
                return;
            };
            let scale_factor = px_scale.y / font.height_unscaled();
            let line_height = px_scale.y;
            let pos = anchor_to(
                t.anchor.unwrap_or(Anchor::Center),
                [t.x, t.y],
                t.width,
                t.height,
            );
            let mut cursor_x = pos[0];
            let mut cursor_y = pos[1];

            let chars: Vec<(usize, char)> = t.text.char_indices().collect();
            let mut i = 0;

            while i < chars.len() {
                let (byte_start, ch) = chars[i];

                // Handle real control chars and escape sequences
                let (render_char, span_chars) = match ch {
                    '\n' | '\t' | '\r' => (ch, 1),
                    '\\' if i + 1 < chars.len() => match chars[i + 1].1 {
                        'n' => ('\n', 2),
                        't' => ('\t', 2),
                        'r' => ('\r', 2),
                        '\\' => ('\\', 2),
                        _ => (ch, 1),
                    },
                    _ => (ch, 1),
                };

                let byte_end = if span_chars == 2 {
                    chars[i + 1].0 + chars[i + 1].1.len_utf8()
                } else {
                    byte_start + ch.len_utf8()
                };

                match render_char {
                    '\n' => {
                        glyph_bounds.push(Rect {
                            min: Point {
                                x: cursor_x,
                                y: cursor_y,
                            },
                            max: Point {
                                x: cursor_x + 4.0,
                                y: cursor_y + line_height,
                            },
                        });
                        char_spans.push(byte_start..byte_end);
                        cursor_x = pos[0];
                        cursor_y += line_height;
                    }
                    '\t' => {
                        let space_advance =
                            font.h_advance_unscaled(font.glyph_id(' ')) * scale_factor;
                        let tab_width = space_advance * 4.0;
                        glyph_bounds.push(Rect {
                            min: Point {
                                x: cursor_x,
                                y: cursor_y,
                            },
                            max: Point {
                                x: cursor_x + tab_width,
                                y: cursor_y + line_height,
                            },
                        });
                        char_spans.push(byte_start..byte_end);
                        cursor_x += tab_width;
                    }
                    '\r' => {
                        glyph_bounds.push(Rect {
                            min: Point {
                                x: cursor_x,
                                y: cursor_y,
                            },
                            max: Point {
                                x: cursor_x,
                                y: cursor_y + line_height,
                            },
                        });
                        char_spans.push(byte_start..byte_end);
                        cursor_x = pos[0];
                    }
                    _ => {
                        let glyph_id = font.glyph_id(render_char);
                        let advance = font.h_advance_unscaled(glyph_id) * scale_factor;
                        glyph_bounds.push(Rect {
                            min: Point {
                                x: cursor_x,
                                y: cursor_y,
                            },
                            max: Point {
                                x: cursor_x + advance,
                                y: cursor_y + line_height,
                            },
                        });
                        char_spans.push(byte_start..byte_end);
                        cursor_x += advance;
                    }
                }

                i += span_chars;
            }

            t.glyph_bounds = glyph_bounds;
            t.char_spans = char_spans;

            *cached = TextParams {
                pos: [t.x, t.y],
                pt: t.pt,
                color: t.color,
                id_hash: hash,
                misc: [
                    f32::from(t.misc.active),
                    rt.touched_time,
                    f32::from(rt.is_down),
                    hash,
                ],
                text: t.text.clone(),
                width: t.width,
                height: t.height,
                id: t.id.clone(),
                caret: t.char_spans.len(),
                anchor: t.anchor,
            };
        }
    }
    rebuilt.mark_texts();
}
pub fn rebuild_circle_cache(
    layer: &mut RuntimeLayer,
    rebuilt: &mut LayerDirty,
    runtime: &UiRuntimes,
) {
    for (element, cache_element) in layer.elements.iter().zip(layer.cache.elements.iter_mut()) {
        if !element.is_active() {
            continue;
        }
        if let (UiElement::Circle(c), UiElementCache::Circle(cached)) = (element, cache_element) {
            let (rt, hash) = runtime_info(runtime, &c.id);

            *cached = CircleParams {
                center_radius_border: [c.x, c.y, c.radius, c.border_thickness_percentage],
                fill_color: c.fill_color,
                inside_border_color: c.inside_border_color,
                border_color: c.border_color,
                glow_color: c.glow_color,
                glow_misc: [
                    c.glow_misc.glow_size,
                    c.glow_misc.glow_speed,
                    c.glow_misc.glow_intensity,
                    1.0,
                ],
                misc: [
                    f32::from(c.misc.active),
                    rt.touched_time,
                    f32::from(rt.is_down),
                    hash,
                ],
                fade: c.fade,
                style: style_to_u32(&c.style),
                inside_border_thickness_percentage: c.inside_border_thickness_percentage,
                _pad: 1,
            };
        }
    }

    rebuilt.mark_circles();
}

fn find_polygon_by_id<'a>(
    id: &Option<ElementRef>,
    before: &'a [RuntimeLayer],
    after: &'a [RuntimeLayer],
) -> Option<&'a UiButtonPolygon> {
    let target = id.as_ref()?;

    for layer in before.iter().chain(after.iter()) {
        for element in &layer.elements {
            //panic!("Yay");
            if let UiElement::Polygon(p) = element {
                if p.id == target.id {
                    return Some(p);
                }
            }
        }
    }
    None
}

pub fn rebuild_outline_cache(
    layer: &mut RuntimeLayer,
    before: &[RuntimeLayer],
    after: &[RuntimeLayer],
    rebuilt: &mut LayerDirty,
    runtime: &UiRuntimes,
) {
    layer.cache.outline_poly_vertices.clear();

    for (element, cache_element) in layer
        .elements
        .iter_mut()
        .zip(layer.cache.elements.iter_mut())
    {
        if !element.is_active() {
            continue;
        }
        if let (UiElement::Outline(o), UiElementCache::Outline(cached)) = (element, cache_element) {
            if o.mode == 1.0 {
                if let Some(poly) = find_polygon_by_id(&o.parent, before, after) {
                    o.vertex_offset = layer.cache.outline_poly_vertices.len() as u32;
                    o.vertex_count = poly.scaled_vertices().len() as u32;

                    for v in &poly.scaled_vertices() {
                        layer.cache.outline_poly_vertices.push([v.pos[0], v.pos[1]]);
                    }
                }
            }

            let (rt, hash) = runtime_info(runtime, &o.id);

            *cached = OutlineParams {
                mode: o.mode,
                vertex_offset: o.vertex_offset,
                vertex_count: o.vertex_count,
                _pad0: 0,
                shape_data: [
                    o.shape_data.x,
                    o.shape_data.y,
                    o.shape_data.radius,
                    o.shape_data.border_thickness,
                ],
                dash_color: o.dash_color,
                dash_misc: [
                    o.dash_misc.dash_len,
                    o.dash_misc.dash_spacing,
                    o.dash_misc.dash_roundness,
                    o.dash_misc.dash_speed,
                ],
                sub_dash_color: o.sub_dash_color,
                sub_dash_misc: [
                    o.sub_dash_misc.dash_len,
                    o.sub_dash_misc.dash_spacing,
                    o.sub_dash_misc.dash_roundness,
                    o.sub_dash_misc.dash_speed,
                ],
                misc: [
                    f32::from(o.misc.active),
                    rt.touched_time,
                    f32::from(rt.is_down),
                    hash,
                ],
            };
        }
    }

    rebuilt.mark_outlines();
}

pub fn rebuild_handle_cache(
    layer: &mut RuntimeLayer,
    rebuilt: &mut LayerDirty,
    runtime: &UiRuntimes,
) {
    for (element, cache_element) in layer.elements.iter().zip(layer.cache.elements.iter_mut()) {
        if !element.is_active() {
            continue;
        }
        if let (UiElement::Handle(h), UiElementCache::Handle(cached)) = (element, cache_element) {
            let (rt, hash) = runtime_info(runtime, &h.id);

            *cached = HandleParams {
                center_radius_mode: [h.x, h.y, h.radius, 1.0],
                handle_color: h.handle_color,
                handle_misc: [
                    h.handle_misc.handle_len,
                    h.handle_misc.handle_width,
                    h.handle_misc.handle_roundness,
                    h.handle_misc.handle_speed,
                ],
                sub_handle_color: h.sub_handle_color,
                sub_handle_misc: [
                    h.sub_handle_misc.handle_len,
                    h.sub_handle_misc.handle_width,
                    h.sub_handle_misc.handle_roundness,
                    h.sub_handle_misc.handle_speed,
                ],
                misc: [
                    f32::from(h.misc.active),
                    rt.touched_time,
                    f32::from(rt.is_down),
                    hash,
                ],
            };
        }
    }

    rebuilt.mark_handles();
}

pub fn rebuild_polygon_cache(
    layer: &mut RuntimeLayer,
    rebuilt: &mut LayerDirty,
    runtime: &UiRuntimes,
) {
    let mut poly_index = 0;

    for (element, cache_element) in layer
        .elements
        .iter_mut()
        .zip(layer.cache.elements.iter_mut())
    {
        if !element.is_active() {
            continue;
        }
        if let (UiElement::Polygon(poly), UiElementCache::Polygon(cached_vertices)) =
            (element, cache_element)
        {
            let (rt, hash) = runtime_info(runtime, &poly.id);

            let misc = [
                f32::from(poly.misc.active),
                rt.touched_time,
                f32::from(rt.is_down),
                hash,
            ];

            let poly_index_f = poly_index as f32;
            poly.update_scaled_vertices();
            let tris = triangulate_polygon(&poly.scaled_vertices());
            poly.tri_count = tris.len() as u32 / 3;

            cached_vertices.clear();
            for v in &tris {
                cached_vertices.push(UiVertexPoly {
                    pos: v.pos,
                    data: [v.roundness, poly_index_f],
                    color: v.color,
                    misc,
                });
            }

            poly_index += 1;
        }
    }

    rebuilt.mark_polygons();
}
pub fn rebuild_rect_cache(
    layer: &mut RuntimeLayer,
    rebuilt: &mut LayerDirty,
    runtime: &UiRuntimes,
) {
    for (element, cache_element) in layer.elements.iter().zip(layer.cache.elements.iter_mut()) {
        if !element.is_active() {
            continue;
        }
        if let (UiElement::Rect(rect), UiElementCache::Rect(cached)) = (element, cache_element) {
            let (rt, hash) = runtime_info(runtime, &rect.id);

            let min_dim = rect.w.min(rect.h);
            let border = rect.border_thickness_percentage * min_dim;

            *cached = RectGpu {
                center: [rect.x, rect.y],
                half_size: [rect.w * 0.5, rect.h * 0.5],
                color: rect.color,
                border_color: rect.border_color,
                roundness: rect.roundness,
                border_thickness: border,
                rotation: -rect.rotation.to_radians(), // NEGATIVE so that it feels intuitive for us mere mortal humans who have a preference for positive rotation to go CW instead of CCW!
                fade: rect.fade,
                misc: [
                    f32::from(rect.misc.active),
                    rt.touched_time,
                    f32::from(rt.is_down),
                    hash,
                ],
            };
        }
    }

    rebuilt.mark_rects();
}
pub fn runtime_info(runtime: &UiRuntimes, id: &String) -> (ButtonRuntime, f32) {
    let runtime = runtime.get(id);

    let hash = if id.is_empty() {
        f32::MAX
    } else {
        Ui::hash_id(id)
    };

    (runtime, hash)
}

pub fn init_cache_structure(layer: &mut RuntimeLayer) {
    // Only rebuild structure if lengths don't match
    if layer.cache.elements.len() == layer.elements.len() {
        return;
    }

    layer.cache.elements.clear();

    for element in &layer.elements {
        layer.cache.elements.push(match element {
            UiElement::Text(_) => UiElementCache::Text(TextParams::default()),
            UiElement::Circle(_) => UiElementCache::Circle(CircleParams::default()),
            UiElement::Polygon(_) => UiElementCache::Polygon(Vec::new()),
            UiElement::Outline(_) => UiElementCache::Outline(OutlineParams::default()),
            UiElement::Handle(_) => UiElementCache::Handle(HandleParams::default()),
            UiElement::Rect(_) => UiElementCache::Rect(RectGpu::zeroed()),
            UiElement::Advanced(_) => continue,
        });
    }
}
