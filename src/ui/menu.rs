use crate::data::Settings;
use crate::ui::cache::*;
use crate::ui::ui_runtime::UiRuntimes;
use crate::ui::variables::Variables;
use crate::ui::vertex::*;
use std::collections::HashMap;
use wgpu_text::TextBrush;
use winit::dpi::PhysicalSize;

#[derive(Debug)]
pub struct Menu {
    pub layers: Vec<RuntimeLayer>,
    pub active: bool,
}

impl Menu {
    pub fn iter_all_elements(&self) -> impl Iterator<Item = &UiElement> {
        self.layers.iter().flat_map(|l| l.iter_all())
    }
    pub fn iter_all_elements_mut(&mut self) -> impl Iterator<Item = &mut UiElement> {
        self.layers.iter_mut().flat_map(|l| l.iter_all_mut())
    }
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

    pub fn rebuild_layer_cache_index(
        &mut self,
        settings: &Settings,
        brush: &TextBrush,
        layer_index: usize,
        runtime: &UiRuntimes,
        aps: &HashMap<String, UiLayerYaml>,
        window_size: PhysicalSize<u32>,
    ) -> Vec<RuntimeLayer> {
        let mut ap_layers = vec![];
        let (before, rest) = self.layers.split_at_mut(layer_index);
        let (layer, after) = rest.split_first_mut().unwrap();

        let dirty = layer.dirty;
        if !dirty.any() || !layer.active {
            return ap_layers;
        }

        let outlines_dirty = dirty.outlines || dirty.polygons || dirty.rects || dirty.circles;
        let mut rebuilt = LayerDirty::none();

        init_cache_structure(layer);

        if dirty.aps {
            // Do not remove the ap references, references must stay. Only if they are temporary though...
            let mut ids_to_remove = vec![];
            for (idx, element) in layer.elements.iter().enumerate() {
                match element {
                    UiElement::Advanced(ap) => {
                        if !before
                            .iter()
                            .chain(std::iter::once(&*layer))
                            .chain(after.iter())
                            .any(|l| l.name == ap.id)
                        {
                            let layer =
                                ap.clone().to_layer(settings, aps, layer.order, window_size);
                            ap_layers.push(layer);
                        }
                        if ap.is_temporary {
                            ids_to_remove.push(idx);
                        }
                    }
                    _ => continue,
                }
            }
            ids_to_remove.sort_unstable();
            ids_to_remove.into_iter().rev().for_each(|i| {
                let _ = layer.elements.remove(i);
            });
        }

        if dirty.texts {
            rebuild_text_cache(brush, layer, &mut rebuilt, runtime);
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

        if dirty.rects {
            rebuild_rect_cache(layer, &mut rebuilt, runtime);
        }

        layer.dirty.clear(rebuilt);
        ap_layers
    }

    pub fn bump_layer_order(&mut self, layer_name: &str, delta: i32, variables: &mut Variables) {
        for layer in &mut self.layers {
            if layer.name == layer_name {
                let new = layer.order as i32 + delta;
                layer.order = new.max(0) as u32;
                variables.set_i64("selected_layer.order", layer.order as i32);
                return;
            }
        }
        self.layers.sort_by_key(|l| l.order);
    }
}
