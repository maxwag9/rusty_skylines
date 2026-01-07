use crate::ui::menu::Menu;
use crate::ui::ui_touch_manager::ElementRef;
use crate::ui::vertex::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SelectedUiElement {
    pub menu_name: String,
    pub layer_name: String,
    pub element_id: String,
    pub active: bool,
    pub just_deselected: bool,
    pub dragging: bool,
    pub just_selected: bool,
    pub action_name: String,
    pub input_box: bool,
}

impl SelectedUiElement {
    pub(crate) fn default() -> SelectedUiElement {
        Self {
            menu_name: "no menu".to_string(),
            layer_name: "no layer".to_string(),
            element_id: "no element".to_string(),
            active: false,
            just_deselected: true,
            dragging: false,
            just_selected: false,
            action_name: "None".to_string(),
            input_box: false,
        }
    }
    pub fn element_type(&self, menus: &HashMap<String, Menu>) -> ElementKind {
        menus
            .get(&self.menu_name)
            .and_then(|menu| menu.layers.iter().find(|l| l.name == self.layer_name))
            .and_then(|layer| layer.iter_all().find(|e| e.id() == &self.element_id))
            .map(ElementKind::from)
            .unwrap_or(ElementKind::None)
    }
}

/// Manages UI element selection state
#[derive(Clone, Debug, Default)]
pub struct SelectionManager {
    /// Primary selected element
    pub primary: Option<ElementRef>,
    /// Additional selected elements (multi-select)
    pub secondary: Vec<ElementRef>,
    /// Anchor point for box selection
    pub box_select_anchor: Option<(f32, f32)>,
    /// Whether selection changed this frame
    pub selection_changed: bool,
    /// Whether a selection was just made
    pub just_selected: bool,
    /// Whether a deselection just happened
    pub just_deselected: bool,
    /// Action associated with primary selection
    pub primary_action: Option<String>,
    /// Whether primary is an input box
    pub primary_is_input_box: bool,
}

impl SelectionManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Select a single element, clearing any existing selection
    pub fn select(&mut self, element: ElementRef, action: Option<String>, is_input_box: bool) {
        self.just_deselected = self.primary.is_some();
        self.primary = Some(element);
        self.secondary.clear();
        self.primary_action = action;
        self.primary_is_input_box = is_input_box;
        self.selection_changed = true;
        self.just_selected = true;
    }

    /// Add element to selection (multi-select)
    pub fn add_to_selection(&mut self, element: ElementRef) {
        // Don't add duplicates
        if self.primary.as_ref() == Some(&element) {
            return;
        }
        if self.secondary.contains(&element) {
            return;
        }

        if self.primary.is_none() {
            self.primary = Some(element);
        } else {
            self.secondary.push(element);
        }
        self.selection_changed = true;
        self.just_selected = true;
    }

    /// Toggle element in selection
    pub fn toggle_selection(&mut self, element: ElementRef) {
        if self.primary.as_ref() == Some(&element) {
            // Deselect primary, promote first secondary if any
            self.primary = self.secondary.pop();
            self.selection_changed = true;
            self.just_deselected = true;
        } else if let Some(pos) = self.secondary.iter().position(|e| e == &element) {
            // Remove from secondary
            self.secondary.remove(pos);
            self.selection_changed = true;
            self.just_deselected = true;
        } else {
            // Add to selection
            self.add_to_selection(element);
        }
    }

    /// Move primary to secondary and select new primary
    pub fn move_primary_to_multi(&mut self, new_primary: ElementRef, action: Option<String>) {
        if let Some(old_primary) = self.primary.take() {
            if old_primary != new_primary {
                self.secondary.push(old_primary);
            }
        }
        self.primary = Some(new_primary);
        self.primary_action = action;
        self.selection_changed = true;
        self.just_selected = true;
    }

    /// Clear all selection
    pub fn deselect_all(&mut self, menus: &mut HashMap<String, Menu>) {
        for (_, menu) in menus.iter_mut() {
            for layer in menu.layers.iter_mut() {
                for text in layer.elements.iter_mut().filter_map(UiElement::as_text_mut) {
                    text.being_edited = false;
                    text.clear_selection();
                }
            }
        }
        self.just_deselected = self.primary.is_some() || !self.secondary.is_empty();
        self.primary = None;
        self.secondary.clear();
        self.primary_action = None;
        self.primary_is_input_box = false;
        self.selection_changed = true;
    }

    /// Set selection from box select results
    pub fn set_from_box(&mut self, elements: Vec<ElementRef>, menus: &mut HashMap<String, Menu>) {
        self.deselect_all(menus);
        if let Some((first, rest)) = elements.split_first() {
            self.primary = Some(first.clone());
            self.secondary = rest.to_vec();
            self.selection_changed = true;
            self.just_selected = !elements.is_empty();
        }
    }

    /// Check if element is selected (primary or secondary)
    pub fn is_selected(&self, element: &ElementRef) -> bool {
        self.primary.as_ref() == Some(element) || self.secondary.contains(element)
    }

    /// Check if element is primary selection
    pub fn is_primary(&self, element: &ElementRef) -> bool {
        self.primary.as_ref() == Some(element)
    }

    /// Get all selected elements
    pub fn all_selected(&self) -> impl Iterator<Item = &ElementRef> {
        self.primary.iter().chain(self.secondary.iter())
    }

    /// Get count of selected elements
    pub fn count(&self) -> usize {
        self.primary.iter().count() + self.secondary.len()
    }

    /// Reset frame-specific flags
    pub fn reset_frame_flags(&mut self) {
        self.selection_changed = false;
        self.just_selected = false;
        self.just_deselected = false;
    }

    /// Begin box selection
    pub fn begin_box_select(&mut self, start: (f32, f32)) {
        self.box_select_anchor = Some(start);
    }

    /// End box selection
    pub fn end_box_select(&mut self) -> Option<(f32, f32)> {
        self.box_select_anchor.take()
    }

    /// Check if box selecting
    pub fn is_box_selecting(&self) -> bool {
        self.box_select_anchor.is_some()
    }

    /// Convert to SelectedUiElement for compatibility
    pub fn to_selected_ui_element(&self) -> SelectedUiElement {
        match &self.primary {
            Some(element) => SelectedUiElement {
                active: true,
                menu_name: element.menu.clone(),
                layer_name: element.layer.clone(),
                element_id: element.id.clone(),
                just_selected: self.just_selected,
                just_deselected: self.just_deselected,
                dragging: false,
                action_name: self.primary_action.clone().unwrap_or_default(),
                input_box: self.primary_is_input_box,
            },
            None => SelectedUiElement::default(),
        }
    }
}
