use crate::renderer::ui::{CircleParams, HandleParams, OutlineParams, TextParams};
use crate::ui::actions::style_to_u32;
use crate::ui::helper::triangulate_polygon;
use crate::ui::text_formatting::{
    build_rich_spans, collect_obfuscated_replacements, format_text, replace_formatted_characters,
};
use crate::ui::ui_editor::Ui;
use crate::ui::ui_runtime::UiRuntimes;
use crate::ui::ui_touch_manager::ElementRef;
use crate::ui::vertex::*;
use bytemuck::Zeroable;
use sluggrs_skylines::FontSystem;
use sluggrs_skylines::cosmic_text::{Attrs, Metrics, Shaping, Weight};
use unicode_segmentation::UnicodeSegmentation;

pub fn rebuild_text_cache(
    font_system: &mut FontSystem,
    layer: &mut RuntimeLayer,
    rebuilt: &mut LayerDirty,
    runtime: &mut UiRuntimes,
) {
    for element in &mut layer.elements {
        if !element.is_active() {
            continue;
        }

        if let UiElement::Text(t) = element {
            let (rt, hash) = runtime_info(runtime, &t.id);
            let metrics = Metrics::new(t.pt, t.pt * 1.2);
            t.buffer.set_metrics(metrics);

            let attrs = Attrs::new();

            let visible_graphemes;

            if t.being_edited {
                t.buffer
                    .set_text(t.text.as_str(), &attrs, Shaping::Basic, None);
                visible_graphemes = t.text.graphemes(true).count();
            } else {
                let formatted_source = t.text.replace("\\n", "\n");
                let formatted = format_text(&formatted_source);

                visible_graphemes = formatted.text.graphemes(true).count();

                if formatted.text.is_empty() {
                    t.buffer.set_text("", &attrs, Shaping::Basic, None);
                } else {
                    let shaping = if formatted.text.is_ascii() {
                        Shaping::Basic
                    } else {
                        Shaping::Advanced
                    };

                    t.buffer.set_rich_text(
                        build_rich_spans(&formatted, &attrs),
                        &attrs,
                        shaping,
                        None,
                    );

                    t.buffer.shape_until_scroll(font_system, false);

                    if formatted.spans.iter().any(|span| span.style.obfuscated) {
                        let replacements = collect_obfuscated_replacements(
                            font_system,
                            &t.buffer,
                            &formatted,
                            runtime,
                        );
                        // eprintln!(
                        //     "obfuscated spans: {}, cached fonts: {}, replacements: {}",
                        //     formatted.spans.iter().filter(|s| s.style.obfuscated).count(),
                        //     runtime.obfuscation_fonts.len(),
                        //     replacements.len(),
                        // );
                        if !replacements.is_empty() {
                            let obfuscated =
                                replace_formatted_characters(&formatted, &replacements);

                            t.buffer.set_rich_text(
                                build_rich_spans(&obfuscated, &attrs),
                                &attrs,
                                shaping,
                                None,
                            );

                            t.buffer.shape_until_scroll(font_system, false);
                        }
                    }
                }
            }

            t.buffer.shape_until_scroll(font_system, false);
            let mut text_width: f32 = 0.0;
            let mut text_height: f32 = 0.0;

            for run in t.buffer.layout_runs() {
                text_width = text_width.max(run.line_w);
                text_height = text_height.max(run.line_top + run.line_height);
                for glyph in run.glyphs.iter() {
                    let font = font_system.get_font(glyph.font_id, Weight::NORMAL);
                }
            }
            //println!("{}", text_width);
            let cache = t.cache.get_or_insert_with(TextParams::default);

            *cache = TextParams {
                pos: [t.x, t.y],
                pt: t.pt,
                color: t.color,
                rotation: t.rotation,
                id_hash: hash,
                misc: [
                    if t.misc.active { 1.0 } else { 0.0 },
                    rt.touched_time,
                    if rt.is_down { 1.0 } else { 0.0 },
                    hash,
                ],
                text: t.text.clone(),
                width: text_width,
                height: text_height,
                id: t.id.clone(),
                caret: t.caret.min(visible_graphemes),
                anchor: t.anchor,
                depth: 0.0,
                decorations: t.decorations.clone(),
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
    for element in &mut layer.elements {
        if !element.is_active() {
            continue;
        }

        if let UiElement::Circle(c) = element {
            let (rt, hash) = runtime_info(runtime, &c.id);

            let cache = c.cache.get_or_insert_with(CircleParams::default);

            *cache = CircleParams {
                center_radius_border: [c.x, c.y, c.radius, c.border_thickness],
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
                    if c.misc.active { 1.0 } else { 0.0 },
                    rt.touched_time,
                    if rt.is_down { 1.0 } else { 0.0 },
                    hash,
                ],
                fade: c.fade,
                style: style_to_u32(&c.style),
                inside_border_thickness: c.inside_border_thickness,
                depth: 0.0,
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
    layer.outline_poly_vertices.clear();

    for element in &mut layer.elements {
        if !element.is_active() {
            continue;
        }

        if let UiElement::Outline(o) = element {
            if o.mode == 1.0 {
                if let Some(poly) = find_polygon_by_id(&o.parent, before, after) {
                    let scaled = poly.scaled_vertices();
                    o.vertex_offset = layer.outline_poly_vertices.len() as u32;
                    o.vertex_count = scaled.len() as u32;

                    for v in scaled {
                        layer.outline_poly_vertices.push([v.pos[0], v.pos[1]]);
                    }
                }
            }

            let (rt, hash) = runtime_info(runtime, &o.id);
            let cache = o.cache.get_or_insert_with(OutlineParams::default);

            *cache = OutlineParams {
                mode: o.mode,
                vertex_offset: o.vertex_offset,
                vertex_count: o.vertex_count,
                depth: 1.0,
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
                    if o.misc.active { 1.0 } else { 0.0 },
                    rt.touched_time,
                    if rt.is_down { 1.0 } else { 0.0 },
                    hash,
                ],
            }
        }
    }

    rebuilt.mark_outlines();
}

pub fn rebuild_handle_cache(
    layer: &mut RuntimeLayer,
    rebuilt: &mut LayerDirty,
    runtime: &UiRuntimes,
) {
    for element in &mut layer.elements {
        if !element.is_active() {
            continue;
        }

        if let UiElement::Handle(h) = element {
            let (rt, hash) = runtime_info(runtime, &h.id);
            let cache = h.cache.get_or_insert_with(HandleParams::default);

            *cache = HandleParams {
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
                    if h.misc.active { 1.0 } else { 0.0 },
                    rt.touched_time,
                    if rt.is_down { 1.0 } else { 0.0 },
                    hash,
                ],
                depth: 0.0,
                _pad0: [0.0; 3],
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

    for element in &mut layer.elements {
        if !element.is_active() {
            continue;
        }

        if let UiElement::Polygon(poly) = element {
            let (rt, hash) = runtime_info(runtime, &poly.id);

            let misc = [
                if poly.misc.active { 1.0 } else { 0.0 },
                rt.touched_time,
                if rt.is_down { 1.0 } else { 0.0 },
                hash,
            ];

            let poly_index_f = poly_index as f32;
            poly.update_scaled_vertices();
            let tris = triangulate_polygon(&poly.scaled_vertices());
            poly.tri_count = tris.len() as u32 / 3;

            let cached_vertices = poly.cache.get_or_insert_with(Vec::new);
            cached_vertices.clear();

            for v in &tris {
                cached_vertices.push(UiVertexPoly {
                    pos: v.pos,
                    data: [v.roundness, poly_index_f],
                    color: v.color,
                    misc,
                    depth: 0.0,
                    _pad0: Default::default(),
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
    for element in &mut layer.elements {
        if !element.is_active() {
            continue;
        }

        if let UiElement::Rect(rect) = element {
            let (rt, hash) = runtime_info(runtime, &rect.id);
            let cache = rect.cache.get_or_insert_with(RectParams::zeroed);

            *cache = RectParams {
                center: [rect.x, rect.y],
                half_size: [rect.w * 0.5, rect.h * 0.5],
                color: rect.color,
                border_color: rect.border_color,
                roundness: rect.roundness,
                border_thickness: rect.border_thickness,
                rotation: -rect.rotation.to_radians(),
                fade: rect.fade,
                blur: rect.blur,
                glow_color: rect.glow_color,
                glow_misc: [
                    rect.glow_misc.glow_size,
                    rect.glow_misc.glow_speed,
                    rect.glow_misc.glow_intensity,
                    1.0,
                ],
                misc: [
                    if rect.misc.active { 1.0 } else { 0.0 },
                    rt.touched_time,
                    if rt.is_down { 1.0 } else { 0.0 },
                    hash,
                ],
                _pad0: [0.0; 2],
                depth: 0.0,
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
