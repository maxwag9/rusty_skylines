// ========================================================================
// ELEMENT POSITION/SIZE/COLOR SETTERS (for undo system)
// ========================================================================

use crate::ui::input::MouseState;
use crate::ui::menu::Menu;
use crate::ui::ui_edit_manager::ColorProperty;
use crate::ui::vertex::*;
use std::collections::HashMap;

pub fn set_element_position(
    menus: &mut HashMap<String, Menu>,
    menu_name: &str,
    layer_name: &str,
    id: &str,
    kind: ElementKind,
    pos: (f32, f32),
) {
    let Some(menu) = menus.get_mut(menu_name) else {
        return;
    };
    let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer_name) else {
        return;
    };

    match kind {
        ElementKind::Circle => {
            if let Some(c) = layer
                .elements
                .iter_mut()
                .filter_map(UiElement::as_circle_mut)
                .find(|c| c.id.as_deref() == Some(id))
            {
                c.x = pos.0;
                c.y = pos.1;
                layer.dirty.mark_circles();
            }
        }
        ElementKind::Text => {
            if let Some(t) = layer
                .elements
                .iter_mut()
                .filter_map(UiElement::as_text_mut)
                .find(|t| t.id.as_deref() == Some(id))
            {
                t.x = pos.0;
                t.y = pos.1;
                layer.dirty.mark_texts();
            }
        }
        ElementKind::Polygon => {
            if let Some(p) = layer
                .elements
                .iter_mut()
                .filter_map(UiElement::as_polygon_mut)
                .find(|p| p.id.as_deref() == Some(id))
            {
                let (cx, cy) = p.center();
                let dx = pos.0 - cx;
                let dy = pos.1 - cy;
                for v in &mut p.vertices {
                    v.pos[0] += dx;
                    v.pos[1] += dy;
                }
                layer.dirty.mark_polygons();
            }
        }
        ElementKind::Handle => {
            if let Some(h) = layer
                .elements
                .iter_mut()
                .filter_map(UiElement::as_handle_mut)
                .find(|h| h.id.as_deref() == Some(id))
            {
                h.x = pos.0;
                h.y = pos.1;
                layer.dirty.mark_handles();
            }
        }
        _ => {}
    }
}

pub fn set_element_size(
    menus: &mut HashMap<String, Menu>,
    menu_name: &str,
    layer_name: &str,
    id: &str,
    kind: ElementKind,
    size: f32,
) {
    let Some(menu) = menus.get_mut(menu_name) else {
        return;
    };
    let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer_name) else {
        return;
    };

    let size = size.max(2.0);

    for element in &mut layer.elements {
        match element {
            // PRIMARY ELEMENT
            UiElement::Circle(c) if kind == ElementKind::Circle && c.id.as_deref() == Some(id) => {
                c.radius = size;
                layer.dirty.mark_circles();
            }

            UiElement::Text(t) if kind == ElementKind::Text && t.id.as_deref() == Some(id) => {
                t.px = size.max(4.0) as u16;
                layer.dirty.mark_texts();
            }

            // DEPENDENTS (always applied immediately)
            UiElement::Handle(h) if h.parent_id.as_deref() == Some(id) => {
                h.radius = size;
                layer.dirty.mark_handles();
            }

            UiElement::Outline(o) if o.parent_id.as_deref() == Some(id) => {
                o.shape_data.radius = size;
                layer.dirty.mark_outlines();
            }

            _ => {}
        }
    }
}

pub fn set_element_color(
    menus: &mut HashMap<String, Menu>,
    menu_name: &str,
    layer_name: &str,
    id: &str,
    kind: ElementKind,
    property: &ColorProperty,
    color: [f32; 4],
) {
    let Some(menu) = menus.get_mut(menu_name) else {
        return;
    };
    let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer_name) else {
        return;
    };

    match kind {
        ElementKind::Circle => {
            if let Some(c) = layer
                .elements
                .iter_mut()
                .filter_map(UiElement::as_circle_mut)
                .find(|c| c.id.as_deref() == Some(id))
            {
                match property {
                    ColorProperty::Fill => c.fill_color = color,
                    ColorProperty::Border => c.border_color = color,
                    ColorProperty::InsideBorder => c.inside_border_color = color,
                    ColorProperty::Glow => c.glow_color = color,
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
                .find(|t| t.id.as_deref() == Some(id))
            {
                if matches!(property, ColorProperty::TextColor) {
                    t.color = color;
                    layer.dirty.mark_texts();
                }
            }
        }
        _ => {}
    }
}

pub fn set_vertex_position(
    menus: &mut HashMap<String, Menu>,
    menu_name: &str,
    layer_name: &str,
    id: &str,
    vertex_index: usize,
    pos: [f32; 2],
) {
    let Some(menu) = menus.get_mut(menu_name) else {
        return;
    };
    let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer_name) else {
        return;
    };

    if let Some(poly) = layer
        .elements
        .iter_mut()
        .filter_map(UiElement::as_polygon_mut)
        .find(|p| p.id.as_deref() == Some(id))
    {
        if let Some(v) = poly.vertices.get_mut(vertex_index) {
            v.pos = pos;
            layer.dirty.mark_polygons();
        }
    }
}

pub fn set_layer_order(
    menus: &mut HashMap<String, Menu>,
    menu_name: &str,
    layer_name: &str,
    order: u32,
) {
    let Some(menu) = menus.get_mut(menu_name) else {
        return;
    };
    if let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer_name) {
        layer.order = order;
    }
    menu.layers.sort_by_key(|l| l.order);
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
        .find(|t| t.id.as_deref() == Some(id))
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
    menu_name: &str,
    layer_name: &str,
    element: &UiElement,
) {
    let Some(menu) = menus.get_mut(menu_name) else {
        return;
    };
    let Some(layer) = menu.layers.iter_mut().find(|l| l.name == layer_name) else {
        return;
    };

    let element_id = element.id();

    if let Some(pos) = layer.elements.iter().position(|e| e.id() == element_id) {
        let removed = layer.elements.remove(pos);
        match removed {
            UiElement::Circle(_) => layer.dirty.mark_circles(),
            UiElement::Text(_) => layer.dirty.mark_texts(),
            UiElement::Polygon(_) => layer.dirty.mark_polygons(),
            UiElement::Handle(_) => layer.dirty.mark_handles(),
            UiElement::Outline(_) => layer.dirty.mark_outlines(),
        }
    }
}

/// Create a new element
pub fn create_element(
    menus: &mut HashMap<String, Menu>,
    menu_name: &str,
    layer_name: &str,
    mut element: UiElement,
    mouse: &MouseState,
) -> Result<(), String> {
    let menu = menus
        .get_mut(menu_name)
        .ok_or_else(|| format!("Menu '{}' doesn't exist", menu_name))?;
    let layer = menu
        .layers
        .iter_mut()
        .find(|l| l.name == layer_name)
        .ok_or_else(|| format!("Layer '{}' not found in menu '{}'", layer_name, menu_name))?;

    let id = mouse.pos.x as u32 - mouse.pos.y as u32;

    // 3. Apply mouse positioning (editor placement)
    match &mut element {
        UiElement::Text(t) => {
            t.x = mouse.pos.x;
            t.y = mouse.pos.y;
            t.id = Option::from(id.to_string());
        }
        UiElement::Circle(c) => {
            c.x = mouse.pos.x;
            c.y = mouse.pos.y;
            c.id = Option::from(id.to_string());
        }
        UiElement::Outline(o) => {
            o.id = Option::from(id.to_string());
        }
        UiElement::Handle(h) => {
            h.x = mouse.pos.x;
            h.y = mouse.pos.y;
            h.id = Option::from(id.to_string());
        }
        UiElement::Polygon(p) => {
            p.id = Option::from(id.to_string());
        }
    }

    // Push element veryyy simple
    match &element {
        UiElement::Text(_) => layer.dirty.mark_texts(),
        UiElement::Circle(_) => layer.dirty.mark_circles(),
        UiElement::Outline(_) => layer.dirty.mark_outlines(),
        UiElement::Handle(_) => layer.dirty.mark_handles(),
        UiElement::Polygon(_) => layer.dirty.mark_polygons(),
    }
    layer.elements.push(element);

    Ok(())
}
