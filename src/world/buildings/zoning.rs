use crate::helpers::positions::{ChunkCoord, ChunkSize, WorldPos};
use crate::renderer::gizmo::gizmo::Gizmo;
use crate::ui::input::Input;
use crate::world::roads::road_subsystem::Roads;
use crate::world::terrain::terrain_subsystem::{CursorMode, Terrain};
use rayon::iter::IntoParallelRefMutIterator;
use std::slice::IterMut;

const SNAP_RADIUS: f64 = 2.5;
const EPS: f64 = 0.0001;

struct ZoningState {
    pub zone_id: ZoneId,
}
pub struct Zone {
    pub id: ZoneId,
    pub points: Vec<WorldPos>,
}

impl Zone {
    pub fn new(points: Vec<WorldPos>) -> Zone {
        Zone { id: 69420, points }
    }
    pub fn add_point(&mut self, point: WorldPos) {
        self.points.push(point);
    }
}
#[derive(Clone, Copy, Debug)]
enum PlacePos {
    Free(WorldPos, f64),
    RoadSnap(WorldPos, f64),
    CurrentZoneFirstPoint(WorldPos, f64),
    CurrentZonePoint(WorldPos, f64),
    CurrentZoneLastPoint(WorldPos, f64),
    OtherZonePoint(WorldPos, f64),
}

impl PlacePos {
    fn pos(self) -> WorldPos {
        match self {
            PlacePos::RoadSnap(pos, _)
            | PlacePos::Free(pos, _)
            | PlacePos::CurrentZoneFirstPoint(pos, _)
            | PlacePos::CurrentZonePoint(pos, _)
            | PlacePos::CurrentZoneLastPoint(pos, _)
            | PlacePos::OtherZonePoint(pos, _) => pos,
        }
    }

    fn dist(self) -> f64 {
        match self {
            PlacePos::Free(_, dist) => dist,
            PlacePos::RoadSnap(_, dist)
            | PlacePos::CurrentZoneFirstPoint(_, dist)
            | PlacePos::CurrentZonePoint(_, dist)
            | PlacePos::CurrentZoneLastPoint(_, dist)
            | PlacePos::OtherZonePoint(_, dist) => dist,
        }
    }
}
pub struct Zoning {
    zoning_state: Option<ZoningState>,
    zoning_storage: ZoningStorage,
}

impl Zoning {
    pub fn new() -> Self {
        Self {
            zoning_state: None,
            zoning_storage: ZoningStorage::new(),
        }
    }
    pub fn update(
        &mut self,
        terrain: &Terrain,
        roads: &Roads,
        input: &mut Input,
        gizmo: &mut Gizmo,
    ) {
        if terrain.cursor.mode != CursorMode::Zoning {
            return;
        }

        let Some(picked) = terrain.last_picked.as_ref() else {
            return;
        };

        let active_zone_id = self.zoning_state.as_ref().map(|state| state.zone_id);

        let mut best_place: PlacePos = PlacePos::Free(picked.pos, 0.0);

        let mut consider = |candidate: PlacePos| {
            let replace = match best_place {
                PlacePos::Free(_, _) => true,
                _ => candidate.dist() < best_place.dist(),
            };

            if replace {
                best_place = candidate;
            }
        };

        if let Some((pos, dist, _)) = roads
            .road_manager
            .roads
            .closest_point_to(&picked.pos, terrain.chunk_size)
            .filter(|(_, dist, _)| *dist <= SNAP_RADIUS)
        {
            consider(PlacePos::RoadSnap(pos, dist));
        }

        if let Some(zone_id) = active_zone_id {
            if let Some(zone) = self.zoning_storage.get(zone_id) {
                let len = zone.points.len();

                for (i, point) in zone.points.iter().copied().enumerate() {
                    let dist = point.distance_to(picked.pos, terrain.chunk_size);

                    if dist > SNAP_RADIUS {
                        continue;
                    }

                    let candidate = if i == 0 {
                        PlacePos::CurrentZoneFirstPoint(point, dist)
                    } else if i + 1 == len {
                        PlacePos::CurrentZoneLastPoint(point, dist)
                    } else {
                        PlacePos::CurrentZonePoint(point, dist)
                    };

                    consider(candidate);
                }
            }
        }

        for zone in self.zoning_storage.iter_zones() {
            if Some(zone.id) == active_zone_id {
                continue;
            }

            for point in zone.points.iter().copied() {
                let dist = point.distance_to(picked.pos, terrain.chunk_size);

                if dist <= SNAP_RADIUS {
                    consider(PlacePos::OtherZonePoint(point, dist));
                }
            }
        }

        let can_place =
            self.can_place_zoning_point(gizmo, best_place, active_zone_id, terrain.chunk_size);

        let preview_color = if can_place {
            [0.3, 0.5, 0.9]
        } else {
            [0.9, 0.2, 0.2]
        };

        gizmo.circle(best_place.pos(), 0.5, preview_color, 0.0, 0.0);

        if let Some(last_pos) = active_zone_id
            .and_then(|zone_id| self.zoning_storage.get(zone_id))
            .and_then(|zone| zone.points.last().copied())
        {
            gizmo.line(best_place.pos(), last_pos, preview_color, 0.0, 0.0);
        }

        if input.action_pressed_once("Place Zoning Point") && can_place {
            if let Some(zoning_state) = &self.zoning_state {
                let zone_id = zoning_state.zone_id;
                match best_place {
                    PlacePos::Free(pos, _) => {
                        if let Some(zone) = self.zoning_storage.get_mut(zone_id) {
                            zone.add_point(pos);
                        }
                    }
                    PlacePos::CurrentZoneFirstPoint(_, _) => {
                        if let Some(zone) = self.zoning_storage.get(zone_id) {
                            if zone.points.len() >= 3 {
                                println!("Closed zoning loop");
                                self.zoning_state = None;
                            }
                        }
                    }

                    PlacePos::RoadSnap(pos, _) => {
                        if let Some(zone) = self.zoning_storage.get_mut(zone_id) {
                            zone.add_point(pos);
                        }
                    }

                    PlacePos::CurrentZonePoint(_, _) => {}
                    PlacePos::CurrentZoneLastPoint(_, _) => {}
                    PlacePos::OtherZonePoint(_, _) => {}
                }
            } else {
                let zone_id = self.zoning_storage.spawn(Zone::new(vec![best_place.pos()]));
                self.zoning_state = Some(ZoningState { zone_id });
            }
        }

        for zone in self.zoning_storage.iter_zones() {
            if Some(zone.id) != active_zone_id {
                gizmo.area(zone.points.as_slice(), [0.4, 0.6, 0.9], 0.0);
            }

            gizmo.polyline(zone.points.as_slice(), [0.3, 0.5, 0.9], 0.0, 0.2, 0.0);
        }
    }

    fn can_place_zoning_point(
        &self,
        gizmo: &mut Gizmo,
        place_pos: PlacePos,
        active_zone_id: Option<ZoneId>,
        chunk_size: ChunkSize,
    ) -> bool {
        let Some(zone_id) = active_zone_id else {
            return true;
        };
        let Some(zone) = self.zoning_storage.get(zone_id) else {
            return false;
        };

        let Some(&last) = zone.points.last() else {
            return false;
        };
        gizmo.line(last, place_pos.pos(), [0.0, 1.0, 0.0], 0.5, 0.0);
        if self.segment_intersects_any_zone(place_pos.pos(), last, active_zone_id, chunk_size) {
            return false;
        };
        match place_pos {
            PlacePos::Free(_, _) => true,
            PlacePos::RoadSnap(_, _) => true,
            PlacePos::CurrentZoneFirstPoint(_, _) => zone.points.len() >= 3,
            PlacePos::CurrentZonePoint(_, _) => false,
            PlacePos::CurrentZoneLastPoint(_, _) => false,
            PlacePos::OtherZonePoint(_, _) => false,
        }
    }

    fn segment_intersects_any_zone(
        &self,
        a: WorldPos,
        b: WorldPos,
        active_zone_id: Option<ZoneId>,
        chunk_size: ChunkSize,
    ) -> bool {
        for zone in self.zoning_storage.iter_zones() {
            let points = &zone.points;
            if points.len() < 2 {
                continue;
            }

            for edge in points.windows(2) {
                let c = edge[0];
                let d = edge[1];

                if segment_intersection_xz(a, b, c, d, chunk_size).is_some() {
                    println!("First one");
                    return true;
                }
            }

            if Some(zone.id) != active_zone_id && points.len() >= 3 {
                let c = *points.last().unwrap();
                let d = points[0];

                if segment_intersection_xz(a, b, c, d, chunk_size).is_some() {
                    println!("Second one");
                    return true;
                }
            }
        }

        false
    }
}

// fn same_pos(a: WorldPos, b: WorldPos) -> bool {
//     a
// }

pub type ZoneId = u32;

pub struct ZoningStorage {
    zones: Vec<Option<Zone>>,
    free_list: Vec<ZoneId>,
    chunk_size: ChunkSize,
    center_chunk: ChunkCoord,
}

impl ZoningStorage {
    pub fn update_target_and_chunk_size(
        &mut self,
        target_chunk: ChunkCoord,
        chunk_size: ChunkSize,
    ) {
        self.center_chunk = target_chunk;
        self.chunk_size = chunk_size;
    }

    pub fn iter_zones(&self) -> impl Iterator<Item = &Zone> {
        self.zones.iter().filter_map(|z| z.as_ref())
    }
    pub fn iter_mut_zones(&mut self) -> IterMut<'_, Option<Zone>> {
        self.zones.iter_mut()
    }
    /// Returns a parallel mutable iterator over building slots.
    /// Each slot is independent, so this is safe for rayon.
    pub fn par_iter_mut_zones(&mut self) -> rayon::slice::IterMut<'_, Option<Zone>> {
        self.zones.par_iter_mut()
    }
    pub fn new() -> Self {
        let mut zones: Vec<Option<Zone>> = Vec::new();
        zones.push(None); // reserve index 0 — for no reason
        Self {
            zones,
            free_list: Vec::new(),
            chunk_size: 128,
            center_chunk: ChunkCoord::zero(),
        }
    }

    pub fn spawn(&mut self, mut zone: Zone) -> ZoneId {
        let zone_id = if let Some(reused_id) = self.free_list.pop() {
            // Reuse slot - III know it's None because it's in free_list
            zone.id = reused_id;
            self.zones[reused_id as usize] = Some(zone);
            reused_id
        } else {
            let new_id = self.zones.len() as u32;
            zone.id = new_id;
            self.zones.push(Some(zone));
            new_id
        };

        zone_id
    }

    pub fn despawn(&mut self, id: ZoneId) {
        if id == 0 {
            // index 0 is reserved, ignore attempts to despawn it
            return;
        }
        if self
            .zones
            .get(id as usize)
            .and_then(|opt| opt.as_ref())
            .is_some()
        {
            // Actually free the slot
            self.zones[id as usize] = None;
            self.free_list.push(id);
        }
    }

    pub fn building_count(&self) -> usize {
        self.zones.len() - self.free_list.len() - 1
    }

    #[inline]
    pub fn get(&self, id: ZoneId) -> Option<&Zone> {
        self.zones.get(id as usize)?.as_ref()
    }

    #[inline]
    pub fn get_mut(&mut self, id: ZoneId) -> Option<&mut Zone> {
        self.zones.get_mut(id as usize)?.as_mut()
    }
}

/// Find intersection point of two segments (XZ plane).
/// Find intersection point of two segments (XZ plane).
/// Returns (t, u) ∈ [0,1]² where the segments cross, or None if parallel/non-intersecting.
pub fn segment_intersection_xz(
    a1: WorldPos,
    a2: WorldPos,
    b1: WorldPos,
    b2: WorldPos,
    chunk_size: ChunkSize,
) -> Option<(f32, f32)> {
    // Direction vectors (XZ)
    let d1x = a1.dx(a2, chunk_size);
    let d1z = a1.dz(a2, chunk_size);

    let d2x = b1.dx(b2, chunk_size);
    let d2z = b1.dz(b2, chunk_size);

    // Vector from a1 → b1
    let d12x = a1.dx(b1, chunk_size);
    let d12z = a1.dz(b1, chunk_size);

    // 2-D cross product of d1 × d2
    let cross = d1x * d2z - d1z * d2x;
    if cross.abs() < 1e-10 {
        return None; // Parallel or coincident
    }

    let t = (d12x * d2z - d12z * d2x) / cross;
    let u = (d12x * d1z - d12z * d1x) / cross;

    const ENDPOINT_EPS: f32 = 1e-5;

    if t > ENDPOINT_EPS && t < 1.0 - ENDPOINT_EPS && u > ENDPOINT_EPS && u < 1.0 - ENDPOINT_EPS {
        Some((t, u))
    } else {
        None
    }
}
