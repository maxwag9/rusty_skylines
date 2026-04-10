use crate::helpers::positions::WorldPos;
use crate::renderer::gizmo::gizmo::Gizmo;
use crate::ui::input::Input;
use crate::world::roads::road_subsystem::Roads;
use crate::world::terrain::terrain_subsystem::{CursorMode, Terrain};
use std::collections::HashMap;

struct ZoningState {}
pub struct Zone {
    pub points: Vec<WorldPos>,
}
pub struct Zoning {
    zoning_state: Option<ZoningState>,
    zones: Vec<Zone>,
}

impl Zoning {
    pub fn new() -> Self {
        Self { zoning_state: None }
    }
    pub fn update(
        &mut self,
        terrain: &Terrain,
        roads: &Roads,
        input: &mut Input,
        gizmo: &mut Gizmo,
    ) {
        if !terrain.cursor.mode.eq(&CursorMode::Zoning) {
            return;
        }
        if let Some(picked) = &terrain.last_picked {
            gizmo.circle(picked.pos, 0.5, [0.3, 0.5, 0.9], 0.0);
            if let Some((closest_point, _, _)) = roads
                .road_manager
                .roads
                .closest_point_to(&picked.pos, terrain.chunk_size)
            {
                gizmo.circle(closest_point, 2.5, [0.7, 0.5, 0.9], 0.0);
                gizmo.line(closest_point, picked.pos, [0.7, 0.8, 0.9], 0.0);
            }
            for zone in self.zones.iter() {
                gizmo.area(zone.points);
                gizmo.polyline(zone.points.as_slice(), [0.3, 0.5, 0.9], 0.0, 0.0);
            }
        }

        if input.action_pressed_once("Place Zoning Point") {
            self.zoning_state = Some(ZoningState {})
        }
        if let Some(state) = &mut self.zoning_state {}
    }
}
