// ========================================================================
// ELEMENT POSITION/SIZE/COLOR SETTERS (for undo system)
// ========================================================================

use crate::ui::input::Mouse;
use crate::ui::menu::Menu;
use crate::ui::parser::Value;
use crate::ui::ui_edit_manager::ColorComponent;
use crate::ui::ui_touch_manager::ElementRef;
use crate::ui::variables::Variables;
use crate::ui::vertex::*;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};

pub fn set_element_position(
    menus: &mut HashMap<String, Menu>,
    element_ref: &ElementRef,
    pos: [f32; 2],
) -> Option<[f32; 2]> {
    let Some(menu) = menus.get_mut(&element_ref.menu) else {
        return None;
    };
    let Some(layer) = menu.layers.iter_mut().find(|l| l.name == element_ref.layer) else {
        return None;
    };
    let mut before: Option<[f32; 2]> = None;
    match element_ref.kind {
        ElementKind::Circle => {
            if let Some(c) = layer
                .elements
                .iter_mut()
                .filter_map(UiElement::as_circle_mut)
                .find(|c| c.id == element_ref.id)
            {
                before = Some([c.x, c.y]);
                c.x = pos[0];
                c.y = pos[1];
                layer.dirty.mark_circles();
            }
            before
        }
        ElementKind::Text => {
            if let Some(t) = layer
                .elements
                .iter_mut()
                .filter_map(UiElement::as_text_mut)
                .find(|t| t.id == element_ref.id)
            {
                before = Some([t.x, t.y]);
                t.x = pos[0];
                t.y = pos[1];
                layer.dirty.mark_texts();
            }
            before
        }
        ElementKind::Polygon => {
            if let Some(p) = layer
                .elements
                .iter_mut()
                .filter_map(UiElement::as_polygon_mut)
                .find(|p| p.id == element_ref.id)
            {
                before = Some(p.center());
                p.x = pos[0];
                p.y = pos[1];
                layer.dirty.mark_polygons();
            }
            before
        }
        ElementKind::Handle => {
            if let Some(h) = layer
                .elements
                .iter_mut()
                .filter_map(UiElement::as_handle_mut)
                .find(|h| h.id == element_ref.id)
            {
                before = Some([h.x, h.y]);
                h.x = pos[0];
                h.y = pos[1];
                layer.dirty.mark_handles();
            }
            before
        }
        ElementKind::Rect => {
            if let Some(r) = layer
                .elements
                .iter_mut()
                .filter_map(UiElement::as_rect_mut)
                .find(|r| r.id == element_ref.id)
            {
                before = Some([r.x, r.y]);
                r.x = pos[0];
                r.y = pos[1];
                layer.dirty.mark_rects();
            }
            before
        }
        _ => before,
    }
}

/// Size properties that can be changed
#[derive(Clone, Debug, PartialEq)]
pub enum SizeProperty {
    Radius(f32),
    Pt(f32),
    Width(f32),
    Height(f32),
    Rect([f32; 2]),
    PolygonScale(f32),
    AdvancedPrimitiveScale(f32),
}
impl SizeProperty {
    pub fn radius(&self) -> Option<f32> {
        match self {
            SizeProperty::Radius(r) => Some(*r),
            _ => None,
        }
    }
    pub fn pt(&self) -> Option<f32> {
        match self {
            SizeProperty::Pt(pt) => Some(*pt),
            _ => None,
        }
    }
    pub fn size2(&self) -> Option<[f32; 2]> {
        match self {
            SizeProperty::Rect(size) => Some(*size),
            _ => None,
        }
    }

    pub fn value_size2(&self) -> Option<Value> {
        self.size2().map(|r| {
            Value::Array(
                r.iter()
                    .map(|s| Value::F64(*s as f64))
                    .collect::<Vec<Value>>(),
            )
        })
    }

    pub fn scale_by(&self, scale: f32) -> Self {
        let mut scaled = self.clone();
        match &mut scaled {
            SizeProperty::Radius(r) => {
                *r *= scale;
            }
            SizeProperty::Pt(pt) => {
                *pt *= scale;
            }
            SizeProperty::Width(w) => {
                *w *= scale;
            }
            SizeProperty::Height(h) => {
                *h *= scale;
            }
            SizeProperty::Rect(rect) => {
                rect[0] *= scale;
                rect[1] *= scale;
            }
            SizeProperty::PolygonScale(ps) => {
                *ps *= scale;
            }
            SizeProperty::AdvancedPrimitiveScale(ps) => {
                *ps *= scale;
            }
        }
        scaled
    }
}

impl Display for SizeProperty {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        format!("{:#?}", self).to_lowercase().fmt(f)
    }
}
pub fn set_element_size(
    menus: &mut HashMap<String, Menu>,
    element_ref: &ElementRef,
    size: &SizeProperty,
) -> Option<SizeProperty> {
    let Some(menu) = menus.get_mut(&element_ref.menu) else {
        return None;
    };
    let Some(layer) = menu.layers.iter_mut().find(|l| l.name == element_ref.layer) else {
        return None;
    };

    for element in &mut layer.elements {
        match element {
            // PRIMARY ELEMENT
            UiElement::Circle(c)
                if element_ref.kind == ElementKind::Circle && c.id == element_ref.id =>
            {
                match size {
                    SizeProperty::Radius(r) => {
                        c.radius = *r;
                    }
                    _ => {}
                }

                layer.dirty.mark_circles();
                return Some(element.size());
            }

            UiElement::Text(t)
                if element_ref.kind == ElementKind::Text && t.id == element_ref.id =>
            {
                match size {
                    SizeProperty::Pt(pt) => {
                        t.pt = pt.max(4.0);
                    }
                    _ => {}
                }

                layer.dirty.mark_texts();
                return Some(element.size());
            }

            // DEPENDENTS (always applied immediately)
            UiElement::Handle(h) if matches!(h.parent.as_ref(), Some(p) if p.id == element_ref.id) =>
            {
                match size {
                    SizeProperty::Radius(r) => {
                        h.radius = *r;
                    }
                    _ => {}
                }
                layer.dirty.mark_handles();
                return Some(element.size());
            }

            UiElement::Outline(o) if matches!(o.parent.as_ref(), Some(p) if p.id == element_ref.id) =>
            {
                match size {
                    SizeProperty::Radius(r) => {
                        o.shape_data.radius = *r;
                    }
                    _ => {}
                }
                layer.dirty.mark_outlines();
                return Some(element.size());
            }
            UiElement::Polygon(p)
                if element_ref.kind == ElementKind::Polygon && p.id == element_ref.id =>
            {
                match size {
                    SizeProperty::PolygonScale(s) => {
                        p.scale = *s;
                        p.scale_by(*s);
                    }
                    _ => {}
                }

                layer.dirty.mark_polygons();
                return Some(element.size());
            }
            UiElement::Rect(r)
                if element_ref.kind == ElementKind::Rect && r.id == element_ref.id =>
            {
                match size {
                    SizeProperty::Width(w) => {
                        r.w = *w;
                    }
                    SizeProperty::Height(h) => {
                        r.h = *h;
                    }
                    SizeProperty::Rect(rect) => {
                        r.w = rect[0];
                        r.h = rect[1];
                    }
                    _ => {}
                }
                layer.dirty.mark_rects();
                return Some(element.size());
            }
            _ => {}
        }
    }
    None
}

pub fn set_element_color(
    menus: &mut HashMap<String, Menu>,
    element_ref: &ElementRef,
    property: &ColorComponent,
    color: [f32; 4],
) -> Option<[f32; 4]> {
    let Some(menu) = menus.get_mut(&element_ref.menu) else {
        return None;
    };
    let Some(layer) = menu.layers.iter_mut().find(|l| l.name == element_ref.layer) else {
        return None;
    };
    let mut previous_color: Option<[f32; 4]> = None;

    match element_ref.kind {
        ElementKind::Circle => {
            if let Some(c) = layer
                .elements
                .iter_mut()
                .filter_map(UiElement::as_circle_mut)
                .find(|c| c.id == element_ref.id)
            {
                match property {
                    ColorComponent::Fill => {
                        previous_color = Some(c.fill_color);
                        c.fill_color = color;
                    }
                    ColorComponent::Border => {
                        previous_color = Some(c.border_color);
                        c.border_color = color;
                    }
                    ColorComponent::InsideBorder => {
                        previous_color = Some(c.inside_border_color);
                        c.inside_border_color = color;
                    }
                    ColorComponent::Glow => {
                        previous_color = Some(c.glow_color);
                        c.glow_color = color;
                    }
                    _ => {}
                }
                layer.dirty.mark_circles();
            }
        }

        ElementKind::Text => {
            if let Some(t) = layer
                .elements
                .iter_mut()
                .filter_map(UiElement::as_text_mut)
                .find(|t| t.id == element_ref.id)
            {
                if matches!(property, ColorComponent::Fill) {
                    previous_color = Some(t.color);
                    t.color = color;
                    layer.dirty.mark_texts();
                }
            }
        }

        ElementKind::Rect => {
            if let Some(r) = layer
                .elements
                .iter_mut()
                .filter_map(UiElement::as_rect_mut)
                .find(|r| r.id == element_ref.id)
            {
                if matches!(property, ColorComponent::Fill) {
                    previous_color = Some(r.color);
                    r.color = color;
                    layer.dirty.mark_rects();
                }
            }
        }

        ElementKind::Polygon => {
            if let Some(p) = layer
                .elements
                .iter_mut()
                .filter_map(UiElement::as_polygon_mut)
                .find(|p| p.id == element_ref.id)
            {
                match property {
                    ColorComponent::Fill => {
                        // Take color from first vertex as "previous"
                        if let Some(first) = p.unscaled_vertices.first() {
                            previous_color = Some(first.color);
                        }

                        for v in &mut p.unscaled_vertices {
                            v.color = color;
                        }

                        p.invalidate_scaled_vertices_cache();
                        layer.dirty.mark_rects();
                    }

                    ColorComponent::VertexIndex(i) => {
                        if let Some(vertex) = p.unscaled_vertices.get_mut(*i as usize) {
                            previous_color = Some(vertex.color);
                            vertex.color = color;

                            p.invalidate_scaled_vertices_cache();
                            layer.dirty.mark_rects();
                        }
                    }

                    _ => {}
                }
            }
        }

        _ => {}
    }

    previous_color
}

pub fn get_element_color(
    menus: &HashMap<String, Menu>,
    element_ref: &ElementRef,
    property: &ColorComponent,
) -> Option<[f32; 4]> {
    let menu = menus.get(&element_ref.menu)?;
    let layer = menu.layers.iter().find(|l| l.name == element_ref.layer)?;

    match element_ref.kind {
        ElementKind::Circle => {
            let c = layer
                .elements
                .iter()
                .filter_map(UiElement::as_circle)
                .find(|c| c.id == element_ref.id)?;

            match property {
                ColorComponent::Fill => Some(c.fill_color),
                ColorComponent::Border => Some(c.border_color),
                ColorComponent::InsideBorder => Some(c.inside_border_color),
                ColorComponent::Glow => Some(c.glow_color),
                _ => None,
            }
        }

        ElementKind::Text => {
            let t = layer
                .elements
                .iter()
                .filter_map(UiElement::as_text)
                .find(|t| t.id == element_ref.id)?;

            match property {
                ColorComponent::Fill => Some(t.color),
                _ => None,
            }
        }

        ElementKind::Rect => {
            let r = layer
                .elements
                .iter()
                .filter_map(UiElement::as_rect)
                .find(|r| r.id == element_ref.id)?;

            match property {
                ColorComponent::Fill => Some(r.color),
                _ => None,
            }
        }

        ElementKind::Polygon => {
            let p = layer
                .elements
                .iter()
                .filter_map(UiElement::as_polygon)
                .find(|p| p.id == element_ref.id)?;

            match property {
                ColorComponent::Fill => p.unscaled_vertices.first().map(|v| v.color),

                ColorComponent::VertexIndex(i) => {
                    p.unscaled_vertices.get(*i as usize).map(|v| v.color)
                }

                _ => None,
            }
        }

        _ => None,
    }
}

pub fn set_vertex_position(
    menus: &mut HashMap<String, Menu>,
    element_ref: &ElementRef,
    vertex_index: usize,
    pos: [f32; 2],
) -> Option<[f32; 2]> {
    let Some(menu) = menus.get_mut(&element_ref.menu) else {
        return None;
    };
    let Some(layer) = menu.layers.iter_mut().find(|l| l.name == element_ref.layer) else {
        return None;
    };
    let mut before = None;
    if let Some(p) = layer
        .elements
        .iter_mut()
        .filter_map(UiElement::as_polygon_mut)
        .find(|p| p.id == element_ref.id)
    {
        if let Some(v) = p.unscaled_vertices.get_mut(vertex_index) {
            before = Some(v.pos);
            v.pos = pos;
            p.invalidate_scaled_vertices_cache();
            layer.dirty.mark_polygons();
        }
    }
    before
}

pub fn bump_layer_order(
    menus: &mut HashMap<String, Menu>,
    variables: &mut Variables,
    menu_name: &str,
    layer_name: &str,
    delta: i32,
) {
    let Some(menu) = menus.get_mut(menu_name) else {
        return;
    };

    menu.bump_layer_order(layer_name, delta, variables);
}

pub fn set_text_content(
    menus: &mut HashMap<String, Menu>,
    menu_name: &str,
    layer_name: &str,
    id: &str,
    text: &str,
    template: &str,
    caret: usize,
) {
    let Some(menu) = menus.get_mut(menu_name) else {
        return;
    };
    let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer_name) else {
        return;
    };

    if let Some(t) = layer
        .elements
        .iter_mut()
        .filter_map(UiElement::as_text_mut)
        .find(|t| t.id == id)
    {
        t.text = text.to_string();
        t.template = template.to_string();
        t.caret = caret;
        layer.dirty.mark_texts();
    }
}

pub fn change_z_index(
    menus: &mut HashMap<String, Menu>,
    menu_name: &str,
    layer_name: &str,
    id: &str,
    delta: i32,
) {
    let Some(menu) = menus.get_mut(menu_name) else {
        return;
    };
    let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer_name) else {
        return;
    };

    layer.bump_element_z(id, delta);
    layer.dirty.mark_all();
}

pub fn replace_circle(
    menus: &mut HashMap<String, Menu>,
    menu_name: &str,
    layer_name: &str,
    new_state: &UiButtonCircle,
) {
    let Some(menu) = menus.get_mut(menu_name) else {
        return;
    };
    let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer_name) else {
        return;
    };

    if let Some(c) = layer
        .elements
        .iter_mut()
        .filter_map(UiElement::as_circle_mut)
        .find(|c| c.id == new_state.id)
    {
        *c = new_state.clone();
        layer.dirty.mark_circles();
    }
}

pub fn replace_text(
    menus: &mut HashMap<String, Menu>,
    menu_name: &str,
    layer_name: &str,
    new_state: &UiButtonText,
) {
    let Some(menu) = menus.get_mut(menu_name) else {
        return;
    };
    let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer_name) else {
        return;
    };

    if let Some(t) = layer
        .elements
        .iter_mut()
        .filter_map(UiElement::as_text_mut)
        .find(|t| t.id == new_state.id)
    {
        *t = new_state.clone();
        layer.dirty.mark_texts();
    }
}

pub fn replace_polygon(
    menus: &mut HashMap<String, Menu>,
    menu_name: &str,
    layer_name: &str,
    new_state: &UiButtonPolygon,
) {
    let Some(menu) = menus.get_mut(menu_name) else {
        return;
    };
    let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer_name) else {
        return;
    };

    if let Some(p) = layer
        .elements
        .iter_mut()
        .filter_map(UiElement::as_polygon_mut)
        .find(|p| p.id == new_state.id)
    {
        *p = new_state.clone();
        layer.dirty.mark_polygons();
    }
}

/// Delete element - takes element reference
pub fn delete_element(
    menus: &mut HashMap<String, Menu>,
    element: &ElementRef,
) -> Result<UiElement, String> {
    let menu = menus
        .get_mut(&element.menu)
        .ok_or_else(|| format!("Menu '{}' doesn't exist", element.menu))?;
    let layer = menu
        .layers
        .iter_mut()
        .find(|l| l.name == element.layer)
        .ok_or_else(|| {
            format!(
                "Layer '{}' not found in menu '{}'",
                element.layer, element.menu
            )
        })?;

    if let Some(id) = layer.elements.iter().position(|e| e.id() == element.id) {
        let removed = layer.elements.remove(id);
        match removed {
            UiElement::Circle(_) => layer.dirty.mark_circles(),
            UiElement::Text(_) => layer.dirty.mark_texts(),
            UiElement::Polygon(_) => layer.dirty.mark_polygons(),
            UiElement::Handle(_) => layer.dirty.mark_handles(),
            UiElement::Outline(_) => layer.dirty.mark_outlines(),
            UiElement::Rect(_) => layer.dirty.mark_rects(),
            UiElement::Advanced(_) => layer.dirty.mark_advanced_primitives(),
        }
        Ok(removed)
    } else {
        Err(format!(
            "Element '{}' not found in layer '{}'",
            element.id, element.layer
        ))
    }
}

/// Create a new element
pub fn create_element(
    menus: &mut HashMap<String, Menu>,
    menu_name: &str,
    layer_name: &str,
    element: UiElement,
    mouse: &Mouse,
) -> Result<(), String> {
    let menu = menus
        .get_mut(menu_name)
        .ok_or_else(|| format!("Menu '{}' doesn't exist", menu_name))?;
    let layer = menu
        .layers
        .iter_mut()
        .find(|l| l.name == layer_name)
        .ok_or_else(|| format!("Layer '{}' not found in menu '{}'", layer_name, menu_name))?;

    // Push element veryyy simple
    match &element {
        UiElement::Text(_) => layer.dirty.mark_texts(),
        UiElement::Circle(_) => layer.dirty.mark_circles(),
        UiElement::Outline(_) => layer.dirty.mark_outlines(),
        UiElement::Handle(_) => layer.dirty.mark_handles(),
        UiElement::Polygon(_) => layer.dirty.mark_polygons(),
        UiElement::Rect(_) => layer.dirty.mark_rects(),
        UiElement::Advanced(ap) => layer.dirty.mark_advanced_primitives(),
    }
    layer.elements.push(element);

    Ok(())
}
