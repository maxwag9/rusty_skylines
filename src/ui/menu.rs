use crate::renderer::ui::{CircleParams, HandleParams, OutlineParams, TextParams};
use crate::ui::actions::style_to_u32;
use crate::ui::helper::triangulate_polygon;
use crate::ui::ui_editor::{
    ElementKind, LayerDirty, RuntimeLayer, UiButtonLoader, UiButtonPolygon, UiVariableRegistry,
    UiVertex, UiVertexPoly,
};
use crate::ui::uiruntime::UiRuntime;

#[derive(Debug)]
pub struct Menu {
    pub layers: Vec<RuntimeLayer>,
    pub active: bool,
}

impl Menu {
    pub fn rebuild_layer_cache_index(&mut self, layer_index: usize, runtime: &UiRuntime) {
        // Split so we can have &mut to this layer and & to all others.
        let (before, rest) = self.layers.split_at_mut(layer_index);
        let (layer, after) = rest.split_first_mut().unwrap();
        let l = layer;

        let dirty = l.dirty;
        if !dirty.any() {
            return;
        }

        let outlines_dirty = dirty.outlines || dirty.polygons;
        let mut rebuilt = LayerDirty::none();

        let runtime_info = |id: &Option<String>| {
            let id_str = id.as_deref().unwrap_or("");
            let runtime = runtime.get(id_str);
            let hash = if id_str.is_empty() {
                f32::MAX
            } else {
                UiButtonLoader::hash_id(id_str)
            };

            (runtime, hash)
        };

        // ------- TEXTS -------
        if dirty.texts {
            l.cache.texts.clear();

            for t in &l.texts {
                let (rt, hash) = runtime_info(&t.id);

                l.cache.texts.push(TextParams {
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
                });
            }

            rebuilt.mark_texts();
        }

        // ------- CIRCLES -------
        if dirty.circles {
            l.cache.circle_params.clear();

            for c in &l.circles {
                let (rt, hash) = runtime_info(&c.id);

                l.cache.circle_params.push(CircleParams {
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

        // Local helper that searches ALL OTHER layers (before + after) for a polygon id
        fn find_polygon_by_id<'a>(
            id: &Option<String>,
            before: &'a [RuntimeLayer],
            after: &'a [RuntimeLayer],
        ) -> Option<&'a UiButtonPolygon> {
            let target = match id {
                Some(s) => s,
                None => return None,
            };

            for layer in before.iter().chain(after.iter()) {
                for p in &layer.polygons {
                    if let Some(pid) = &p.id {
                        if pid == target {
                            return Some(p);
                        }
                    }
                }
            }
            None
        }

        // ------- OUTLINES -------
        if outlines_dirty {
            l.cache.outline_params.clear();
            l.cache.outline_poly_vertices.clear();

            for o in &mut l.outlines {
                if o.mode == 1.0 {
                    if let Some(poly) = find_polygon_by_id(&o.parent_id, before, after) {
                        o.vertex_offset = l.cache.outline_poly_vertices.len() as u32;

                        for v in &poly.vertices {
                            l.cache.outline_poly_vertices.push([v.pos[0], v.pos[1]]);
                        }

                        o.vertex_count = poly.vertices.len() as u32;
                    }
                }

                let (rt, hash) = runtime_info(&o.id);

                l.cache.outline_params.push(OutlineParams {
                    mode: o.mode,
                    vertex_offset: o.vertex_offset,
                    vertex_count: o.vertex_count,
                    _pad0: 0, // u32 padding, must be integer
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

        // ------- HANDLES -------
        if dirty.handles {
            l.cache.handle_params.clear();

            for h in &l.handles {
                let (rt, hash) = runtime_info(&h.id);

                l.cache.handle_params.push(HandleParams {
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

        // Common builder for vertex-emitting shapes
        let push_with_misc = |v: &UiVertex,
                              roundness_px: f32,
                              poly_index_f: f32,
                              misc: [f32; 4],
                              out: &mut Vec<UiVertexPoly>| {
            out.push(UiVertexPoly {
                pos: v.pos,
                data: [roundness_px, poly_index_f],
                color: v.color,
                misc,
            });
        };

        // ------- POLYGONS (N verts) -------
        if dirty.polygons {
            l.cache.polygon_vertices.clear();

            for (poly_index, poly) in l.polygons.iter_mut().enumerate() {
                let (rt, hash) = runtime_info(&poly.id);

                let misc = [
                    f32::from(poly.misc.active),
                    rt.touched_time,
                    f32::from(rt.is_down),
                    hash,
                ];

                let poly_index_f = poly_index as f32;

                let tris = triangulate_polygon(&mut poly.vertices);
                poly.tri_count = tris.len() as u32 / 3;

                for vertex in &tris {
                    let roundness_px = vertex.roundness;
                    push_with_misc(
                        vertex,
                        roundness_px,
                        poly_index_f,
                        misc,
                        &mut l.cache.polygon_vertices,
                    );
                }
            }

            rebuilt.mark_polygons();
        }

        l.dirty.clear(rebuilt);
    }

    pub fn sort_layers(&mut self) {
        self.layers.sort_by_key(|l| l.order);
    }

    pub fn bump_layer_order(
        &mut self,
        layer_name: &str,
        delta: i32,
        variables: &mut UiVariableRegistry,
    ) {
        for layer in &mut self.layers {
            if layer.name == layer_name {
                let new = layer.order as i32 + delta;
                layer.order = new.max(0) as u32;
                variables.set_i32("selected_layer.order", layer.order as i32);
                return;
            }
        }
    }

    pub fn change_element_color(
        &mut self,
        layer_name: &str,
        element_id: &str,
        element_type: ElementKind,
        new_color: [f32; 4],
    ) -> bool {
        let layer = match self
            .layers
            .iter_mut()
            .find(|l| l.active && l.saveable && l.name == layer_name)
        {
            Some(l) => l,
            None => return false,
        };

        match element_type {
            ElementKind::Polygon => {
                if let Some(p) = layer
                    .polygons
                    .iter_mut()
                    .find(|p| p.id.as_deref() == Some(element_id))
                {
                    for v in p.vertices.iter_mut() {
                        v.color = new_color;
                    }

                    layer.dirty.mark_polygons();
                    return true;
                }
            }

            ElementKind::Circle => {
                if let Some(c) = layer
                    .circles
                    .iter_mut()
                    .find(|c| c.id.as_deref() == Some(element_id))
                {
                    c.fill_color = new_color.into();
                    layer.dirty.mark_circles();
                    return true;
                }
            }

            ElementKind::Text => {
                if let Some(t) = layer
                    .texts
                    .iter_mut()
                    .find(|t| t.id.as_deref() == Some(element_id))
                {
                    t.color = new_color;
                    layer.dirty.mark_texts();
                    return true;
                }
            }

            ElementKind::Outline => {}
            ElementKind::Handle => {}
            ElementKind::None => {}
        }

        false
    }
}

pub fn get_selected_element_color(loader: &UiButtonLoader) -> Option<[f32; 4]> {
    let selected = &loader.ui_runtime.selected_ui_element_primary;

    if !selected.active || selected.action_name == "Drag Hue Point" {
        return None;
    }

    // Find the menu
    let menu = loader.menus.get(&selected.menu_name)?;
    // Find the layer
    let layer = menu
        .layers
        .iter()
        .find(|l| l.active && l.saveable && l.name == selected.layer_name)?;

    // Match element type
    match selected.element_type {
        ElementKind::Polygon => {
            let poly = layer
                .polygons
                .iter()
                .find(|p| p.id.as_deref() == Some(&selected.element_id))?;

            // take color from first vertex (they are all the same in your system)
            poly.vertices.get(0).map(|v| v.color)
        }

        ElementKind::Circle => {
            let circle = layer
                .circles
                .iter()
                .find(|c| c.id.as_deref() == Some(&selected.element_id))?;

            Some(circle.fill_color.into())
        }

        ElementKind::Text => {
            let text = layer
                .texts
                .iter()
                .find(|t| t.id.as_deref() == Some(&selected.element_id))?;

            Some(text.color)
        }

        ElementKind::Outline => None,
        ElementKind::Handle => None,
        ElementKind::None => None,
    }
}
