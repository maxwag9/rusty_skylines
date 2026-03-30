use crate::ui::menu::Menu;
use crate::ui::ui_touch_manager::ElementRef;
use crate::ui::vertex::*;
use std::collections::HashMap;

/// Manages UI element selection state
#[derive(Clone, Debug, Default)]
pub struct SelectionManager {
    pub selected: Vec<ElementRef>,
    pub active_tool: Option<ElementRef>,
    /// Anchor point for box selection
    pub box_select_anchor: Option<[f32; 2]>,
    /// Whether selection changed this frame
    pub selection_changed: bool,
    /// Whether a selection was just made
    pub just_selected: bool,
    /// Whether a deselection just happened
    pub just_deselected: bool,
}

impl SelectionManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn select_from_overwrite(
        &mut self,
        menus: &HashMap<String, Menu>,
        selected: &Vec<ElementRef>,
        active_tool: &Option<ElementRef>,
    ) {
        self.selected = selected.clone();
        self.active_tool = active_tool.clone();
        self.just_selected = true;
        self.selection_changed = true;
        self.just_deselected = false;
    }
    /// Add element to selection (multi-select)
    pub fn add_to_selection(&mut self, element: ElementRef) {
        self.selected.push(element);

        self.selection_changed = true;
        self.just_selected = true;
    }

    pub fn toggle_selection(&mut self, element: ElementRef) {
        if let Some(pos) = self.selected.iter().position(|e| e == &element) {
            self.selected.swap_remove(pos);
            self.active_tool = None;
            self.just_deselected = true;
        } else {
            self.selected.push(element);
            self.just_selected = true;
            self.just_deselected = false;
        }

        self.selection_changed = true;
    }

    /// Clear all selections and add this element to selection.
    pub fn select_single(&mut self, element: ElementRef) {
        self.selected.clear();
        self.selected.push(element);
        self.active_tool = None;
        self.selection_changed = true;
        self.just_selected = true;
        self.just_deselected = false;
    }

    /// Clear all selections
    pub fn deselect_all(&mut self, menus: &mut HashMap<String, Menu>) {
        for (_, menu) in menus.iter_mut() {
            for layer in menu.layers.iter_mut() {
                for text in layer.elements.iter_mut().filter_map(UiElement::as_text_mut) {
                    text.being_edited = false;
                    text.clear_selection();
                }
            }
        }
        self.just_deselected = !self.selected.is_empty() || self.active_tool.is_some();
        self.selected.clear();
        self.active_tool = None;
        self.selection_changed = true;
    }

    /// Set selection from box select results
    pub fn set_from_box(&mut self, elements: Vec<ElementRef>, menus: &mut HashMap<String, Menu>) {
        self.deselect_all(menus);
        self.just_selected = !elements.is_empty();
        self.selected = elements;
        self.active_tool = None;
        self.selection_changed = true;
    }

    /// Check if element is selected
    pub fn is_selected(&self, element: &ElementRef) -> bool {
        self.selected.contains(element)
    }

    /// Get count of selected elements
    pub fn count(&self) -> usize {
        self.selected.len()
    }

    /// Reset frame-specific flags
    pub fn reset_frame_flags(&mut self) {
        self.selection_changed = false;
        self.just_selected = false;
        self.just_deselected = false;
    }

    /// Begin box selection
    pub fn begin_box_select(&mut self, start: [f32; 2]) {
        self.box_select_anchor = Some(start);
    }

    /// End box selection
    pub fn end_box_select(&mut self) -> Option<[f32; 2]> {
        self.box_select_anchor.take()
    }

    /// Check if box selecting
    pub fn is_box_selecting(&self) -> bool {
        self.box_select_anchor.is_some()
    }
}
