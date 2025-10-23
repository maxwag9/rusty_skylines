use egui::{Align2, Area, Id, Vec2};

/// Helper for creating anchored floating panels.
pub fn anchored_panel(id: Id, anchor: Align2, offset: Vec2) -> Area {
    Area::new(id).anchor(anchor, offset).interactable(true)
}
