use crate::renderer::ui::{CircleParams, HandleParams, OutlineParams, TextParams};
use crate::ui::actions::style_to_u32;
use crate::ui::helper::triangulate_polygon;
use crate::ui::ui_editor::UiButtonLoader;
use crate::ui::ui_runtime::UiRuntime;
use crate::ui::ui_touch_manager::ElementRef;
use crate::ui::vertex::*;

pub fn rebuild_text_cache(layer: &mut RuntimeLayer, rebuilt: &mut LayerDirty, runtime: &UiRuntime) {
    for (element, cache_element) in layer.elements.iter().zip(layer.cache.elements.iter_mut()) {
        if let (UiElement::Text(t), UiElementCache::Text(cached)) = (element, cache_element) {
            let (rt, hash) = runtime_info(runtime, &t.id);

            *cached = TextParams {
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
            };
        }
    }

    rebuilt.mark_texts();
}

pub fn rebuild_circle_cache(
    layer: &mut RuntimeLayer,
    rebuilt: &mut LayerDirty,
    runtime: &UiRuntime,
) {
    for (element, cache_element) in layer.elements.iter().zip(layer.cache.elements.iter_mut()) {
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
    runtime: &UiRuntime,
) {
    layer.cache.outline_poly_vertices.clear();

    for (element, cache_element) in layer
        .elements
        .iter_mut()
        .zip(layer.cache.elements.iter_mut())
    {
        if let (UiElement::Outline(o), UiElementCache::Outline(cached)) = (element, cache_element) {
            if o.mode == 1.0 {
                if let Some(poly) = find_polygon_by_id(&o.parent, before, after) {
                    o.vertex_offset = layer.cache.outline_poly_vertices.len() as u32;
                    o.vertex_count = poly.vertices.len() as u32;

                    for v in &poly.vertices {
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
    runtime: &UiRuntime,
) {
    for (element, cache_element) in layer.elements.iter().zip(layer.cache.elements.iter_mut()) {
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
    runtime: &UiRuntime,
) {
    let mut poly_index = 0;

    for (element, cache_element) in layer
        .elements
        .iter_mut()
        .zip(layer.cache.elements.iter_mut())
    {
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
            let tris = triangulate_polygon(&mut poly.vertices);
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

pub fn runtime_info(runtime: &UiRuntime, id: &String) -> (ButtonRuntime, f32) {
    let runtime = runtime.get(id);

    let hash = if id.is_empty() {
        f32::MAX
    } else {
        UiButtonLoader::hash_id(id)
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
        });
    }
}
