use crate::renderer::ui::{CircleParams, HandleParams, OutlineParams, TextParams};
use crate::ui::actions::style_to_u32;
use crate::ui::helper::triangulate_polygon;
use crate::ui::ui_editor::UiButtonLoader;
use crate::ui::ui_runtime::UiRuntime;
use crate::ui::vertex::*;

pub fn rebuild_text_cache(layer: &mut RuntimeLayer, rebuilt: &mut LayerDirty, runtime: &UiRuntime) {
    layer.cache.texts.clear();

    for t in &layer.texts {
        let (rt, hash) = runtime_info(runtime, &t.id);

        layer.cache.texts.push(TextParams {
            pos: [t.x, t.y],
            px: t.px,
            color: t.color,
            id_hash: hash,
            misc: [
                f32::from(t.misc.active),
                rt.touched_time,
                f32::from(rt.is_down),
                hash,
            ],
            text: t.text.clone(),

            natural_width: t.natural_width,
            natural_height: t.natural_height,
            id: t.id.clone(),
            caret: t.text.len(),
            glyph_bounds: vec![],
            anchor: t.anchor,
        });
    }

    rebuilt.mark_texts();
}
pub fn rebuild_circle_cache(
    layer: &mut RuntimeLayer,
    rebuilt: &mut LayerDirty,
    runtime: &UiRuntime,
) {
    layer.cache.circle_params.clear();

    for c in &layer.circles {
        let (rt, hash) = runtime_info(runtime, &c.id);

        layer.cache.circle_params.push(CircleParams {
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
                f32::from(c.misc.active),
                rt.touched_time,
                f32::from(rt.is_down),
                hash,
            ],
            fade: c.fade,
            style: style_to_u32(&c.style),
            inside_border_thickness: c.inside_border_thickness,
            _pad: 1,
        });
    }

    rebuilt.mark_circles();
}
fn find_polygon_by_id<'a>(
    id: &Option<String>,
    before: &'a [RuntimeLayer],
    after: &'a [RuntimeLayer],
) -> Option<&'a UiButtonPolygon> {
    let target = id.as_ref()?;

    for layer in before.iter().chain(after.iter()) {
        for p in &layer.polygons {
            if p.id.as_ref() == Some(target) {
                return Some(p);
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
    runtime: &UiRuntime,
) {
    layer.cache.outline_params.clear();
    layer.cache.outline_poly_vertices.clear();

    for o in &mut layer.outlines {
        if o.mode == 1.0 {
            if let Some(poly) = find_polygon_by_id(&o.parent_id, before, after) {
                o.vertex_offset = layer.cache.outline_poly_vertices.len() as u32;
                o.vertex_count = poly.vertices.len() as u32;

                for v in &poly.vertices {
                    layer.cache.outline_poly_vertices.push([v.pos[0], v.pos[1]]);
                }
            }
        }

        let (rt, hash) = runtime_info(runtime, &o.id);

        layer.cache.outline_params.push(OutlineParams {
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
        });
    }

    rebuilt.mark_outlines();
}
pub fn rebuild_handle_cache(
    layer: &mut RuntimeLayer,
    rebuilt: &mut LayerDirty,
    runtime: &UiRuntime,
) {
    layer.cache.handle_params.clear();

    for h in &layer.handles {
        let (rt, hash) = runtime_info(runtime, &h.id);

        layer.cache.handle_params.push(HandleParams {
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
        });
    }

    rebuilt.mark_handles();
}
pub fn rebuild_polygon_cache(
    layer: &mut RuntimeLayer,
    rebuilt: &mut LayerDirty,
    runtime: &UiRuntime,
) {
    layer.cache.polygon_vertices.clear();

    for (poly_index, poly) in layer.polygons.iter_mut().enumerate() {
        let (rt, hash) = runtime_info(runtime, &poly.id);

        let misc = [
            f32::from(poly.misc.active),
            rt.touched_time,
            f32::from(rt.is_down),
            hash,
        ];

        let poly_index_f = poly_index as f32;
        let tris = triangulate_polygon(&mut poly.vertices);
        poly.tri_count = tris.len() as u32 / 3;

        for v in &tris {
            layer.cache.polygon_vertices.push(UiVertexPoly {
                pos: v.pos,
                data: [v.roundness, poly_index_f],
                color: v.color,
                misc,
            });
        }
    }

    rebuilt.mark_polygons();
}

pub fn runtime_info(runtime: &UiRuntime, id: &Option<String>) -> (ButtonRuntime, f32) {
    let id_str = id.as_deref().unwrap_or("");
    let runtime = runtime.get(id_str);

    let hash = if id_str.is_empty() {
        f32::MAX
    } else {
        UiButtonLoader::hash_id(id_str)
    };

    (runtime, hash)
}
