use crate::ui::cache::*;
use crate::ui::ui_editor::UiButtonLoader;
use crate::ui::ui_runtime::UiRuntime;
use crate::ui::variables::UiVariableRegistry;
use crate::ui::vertex::*;

#[derive(Debug)]
pub struct Menu {
    pub layers: Vec<RuntimeLayer>,
    pub active: bool,
}

impl Menu {
    pub fn get_element(&mut self, layer_name: &str, element_id: &str) -> Option<UiElement> {
        let layer = self.layers.iter().find(|l| l.name == layer_name)?;

        if let Some(c) = layer
            .circles
            .iter()
            .find(|c| c.id.as_deref() == Some(element_id))
        {
            return Some(UiElement::Circle(c.clone()));
        }
        if let Some(t) = layer
            .texts
            .iter()
            .find(|t| t.id.as_deref() == Some(element_id))
        {
            return Some(UiElement::Text(t.clone()));
        }
        if let Some(p) = layer
            .polygons
            .iter()
            .find(|p| p.id.as_deref() == Some(element_id))
        {
            return Some(UiElement::Polygon(p.clone()));
        }
        if let Some(h) = layer
            .handles
            .iter()
            .find(|h| h.id.as_deref() == Some(element_id))
        {
            return Some(UiElement::Handle(h.clone()));
        }
        if let Some(o) = layer
            .outlines
            .iter()
            .find(|o| o.id.as_deref() == Some(element_id))
        {
            return Some(UiElement::Outline(o.clone()));
        }

        None
    }
    pub fn rebuild_layer_cache_index(&mut self, layer_index: usize, runtime: &UiRuntime) {
        let (before, rest) = self.layers.split_at_mut(layer_index);
        let (layer, after) = rest.split_first_mut().unwrap();

        let dirty = layer.dirty;
        if !dirty.any() {
            return;
        }

        let outlines_dirty = dirty.outlines || dirty.polygons;
        let mut rebuilt = LayerDirty::none();

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
