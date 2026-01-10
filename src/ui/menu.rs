use crate::ui::cache::*;
use crate::ui::ui_editor::UiButtonLoader;
use crate::ui::ui_runtime::UiRuntimes;
use crate::ui::variables::UiVariableRegistry;
use crate::ui::vertex::*;

#[derive(Debug)]
pub struct Menu {
    pub layers: Vec<RuntimeLayer>,
    pub active: bool,
}

impl Menu {
    pub fn sort_layers(&mut self) {
        self.layers.sort_by_key(|l| l.order);
    }
    pub fn get_element(&self, layer_name: &str, element_id: &str) -> Option<UiElement> {
        let layer = self.layers.iter().find(|l| l.name == layer_name)?;

        layer.elements.iter().find_map(|e| match e {
            UiElement::Circle(c) if c.id == element_id => Some(UiElement::Circle(c.clone())),

            UiElement::Text(t) if t.id == element_id => Some(UiElement::Text(t.clone())),

            UiElement::Polygon(p) if p.id == element_id => Some(UiElement::Polygon(p.clone())),

            UiElement::Handle(h) if h.id == element_id => Some(UiElement::Handle(h.clone())),

            UiElement::Outline(o) if o.id == element_id => Some(UiElement::Outline(o.clone())),

            _ => None,
        })
    }

    pub fn rebuild_layer_cache_index(&mut self, layer_index: usize, runtime: &UiRuntimes) {
        let (before, rest) = self.layers.split_at_mut(layer_index);
        let (layer, after) = rest.split_first_mut().unwrap();

        let dirty = layer.dirty;
        if !dirty.any() {
            return;
        }

        let outlines_dirty = dirty.outlines || dirty.polygons;
        let mut rebuilt = LayerDirty::none();

        init_cache_structure(layer);

        if dirty.texts {
            rebuild_text_cache(layer, &mut rebuilt, runtime);
        }

        if dirty.circles {
            rebuild_circle_cache(layer, &mut rebuilt, runtime);
        }

        if outlines_dirty {
            rebuild_outline_cache(layer, before, after, &mut rebuilt, runtime);
        }

        if dirty.handles {
            rebuild_handle_cache(layer, &mut rebuilt, runtime);
        }

        if dirty.polygons {
            rebuild_polygon_cache(layer, &mut rebuilt, runtime);
        }

        layer.dirty.clear(rebuilt);
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
                    .elements
                    .iter_mut()
                    .filter_map(UiElement::as_polygon_mut)
                    .find(|p| p.id == element_id)
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
                    .elements
                    .iter_mut()
                    .filter_map(UiElement::as_circle_mut)
                    .find(|c| c.id == element_id)
                {
                    c.fill_color = new_color.into();
                    layer.dirty.mark_circles();
                    return true;
                }
            }

            ElementKind::Text => {
                if let Some(t) = layer
                    .elements
                    .iter_mut()
                    .filter_map(UiElement::as_text_mut)
                    .find(|t| t.id == element_id)
                {
                    t.color = new_color;
                    layer.dirty.mark_texts();
                    return true;
                }
            }

            ElementKind::Outline | ElementKind::Handle | ElementKind::None => {}
        }

        false
    }
}

pub fn get_selected_element_color(loader: &UiButtonLoader) -> Option<[f32; 4]> {
    let Some(sel) = &loader.touch_manager.selection.primary else {
        return None;
    };

    if loader.touch_manager.selection.primary_action == Some("Drag Hue Point".to_string()) {
        return None;
    }

    // Find the menu
    let menu = loader.menus.get(&sel.menu)?;
    // Find the layer
    let layer = menu
        .layers
        .iter()
        .find(|l| l.active && l.saveable && l.name == sel.layer)?;

    // Match element type
    match sel.kind {
        ElementKind::Polygon => {
            let poly = layer.iter_polygons().find(|p| p.id == sel.id)?;

            // take color from first vertex (but they are not all the same!!)
            poly.vertices.get(0).map(|v| v.color)
        }

        ElementKind::Circle => {
            let circle = layer.iter_circles().find(|c| c.id == sel.id)?;

            Some(circle.fill_color.into())
        }

        ElementKind::Text => {
            let text = layer.iter_texts().find(|t| t.id == sel.id)?;

            Some(text.color)
        }

        ElementKind::Outline => None,
        ElementKind::Handle => None,
        ElementKind::None => None,
    }
}
