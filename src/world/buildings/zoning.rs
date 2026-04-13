use crate::helpers::positions::{ChunkCoord, ChunkSize, LocalPos, WorldPos};
use crate::renderer::gizmo::gizmo::Gizmo;
use crate::ui::input::Input;
use crate::ui::parser::Value;
use crate::ui::variables::Variables;
use crate::world::roads::road_subsystem::Roads;
use crate::world::terrain::terrain_subsystem::{CursorMode, Terrain};
use rayon::iter::IntoParallelRefMutIterator;
use std::fmt::{Display, Formatter};
use std::slice::IterMut;

const SNAP_RADIUS: f64 = 2.5;
const EPS: f64 = 0.0001;

struct ZoningState {
    pub zone_id: ZoneId,
}

#[derive(Debug, Copy, Clone)]
pub enum ZoneType {
    None,
    Residential,
    Commercial,
    Industrial,
    Office,
}
impl ZoneType {
    pub(crate) fn from_value(value: &Value) -> Self {
        match value {
            Value::String(s) => match s.to_lowercase().as_str() {
                "none" => ZoneType::None,
                "residential" => ZoneType::Residential,
                "commercial" => ZoneType::Commercial,
                "industrial" => ZoneType::Industrial,
                "office" => ZoneType::Office,
                _ => ZoneType::None,
            },
            _ => ZoneType::None,
        }
    }
}
impl Display for ZoneType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ZoneType::None => write!(f, "none"),
            ZoneType::Residential => write!(f, "residential"),
            ZoneType::Commercial => write!(f, "commercial"),
            ZoneType::Industrial => write!(f, "industrial"),
            ZoneType::Office => write!(f, "office"),
        }
    }
}
pub struct Zone {
    pub id: ZoneId,
    pub points: Vec<WorldPos>,
    pub zone_type: ZoneType,
}

impl Zone {
    pub fn new(points: Vec<WorldPos>, zone_type: ZoneType) -> Zone {
        Zone {
            id: 69420,
            points,
            zone_type,
        }
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
        variables: &Variables,
    ) {
        if terrain.cursor.mode != CursorMode::Zoning {
            return;
        }
        let new_zone_type = &terrain.cursor.zone_type;
        let active_zone_id = self.zoning_state.as_ref().map(|state| state.zone_id);
        for zone in self.zoning_storage.iter_zones() {
            if Some(zone.id) != active_zone_id {
                draw_zone_area(zone, variables, gizmo, None, None);
            }

            gizmo.polyline(
                zone.points.as_slice(),
                [0.07, 0.28, 0.5, 1.0],
                0.0,
                0.2,
                0.0,
            );
        }
        let Some(picked) = terrain.last_picked.as_ref() else {
            return;
        };

        let mut best_place: PlacePos = PlacePos::Free(picked.pos, 0.0);

        let mut consider = |candidate: PlacePos| {
            use PlacePos::*;

            let priority = |p: &PlacePos| match p {
                CurrentZoneFirstPoint(_, _) => 5,
                CurrentZoneLastPoint(_, _) => 4,
                CurrentZonePoint(_, _) => 3,
                OtherZonePoint(_, _) => 2,
                RoadSnap(_, _) => 1,
                Free(_, _) => 0,
            };

            let best_pri = priority(&best_place);
            let cand_pri = priority(&candidate);
            let best_dist = best_place.dist();
            let cand_dist = candidate.dist();

            let should_replace = match (
                best_pri.cmp(&cand_pri),
                best_dist >= SNAP_RADIUS,
                cand_dist < SNAP_RADIUS,
            ) {
                (std::cmp::Ordering::Less, _, true) => true, // Promote if candidate in radius
                (std::cmp::Ordering::Greater, true, _) => true, // Demote if best outside radius
                (std::cmp::Ordering::Equal, _, _) => cand_dist < best_dist, // Same priority: closer wins
                _ => false,
            };

            if should_replace {
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
                    println!("Considering: {:?}", candidate);
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
            [0.1, 0.3, 0.7, 1.0]
        } else {
            [0.7, 0.1, 0.1, 1.0]
        };
        let mut inside_zone_id: Option<ZoneId> = None;
        if can_place && matches!(best_place, PlacePos::Free(_, _)) {
            for zone in self.zoning_storage.iter_zones() {
                if Some(zone.id) == active_zone_id {
                    continue;
                }
                if point_inside_polygon(best_place.pos(), &zone.points, gizmo.chunk_size, false) {
                    // best_pos is inside this zone
                    inside_zone_id.replace(zone.id);
                }
            }
        }

        let canceling = input.action_pressed_once("Cancel");
        if canceling {
            if let Some(zoning_state) = &self.zoning_state {
                let zone_id = zoning_state.zone_id;
                if let Some(zone) = self.zoning_storage.get_mut(zone_id) {
                    zone.points.pop();
                    if zone.points.is_empty() {
                        self.zoning_storage.despawn(zone_id);
                        self.zoning_state = None;
                    }
                }
            }
        }
        if let Some(id) = inside_zone_id {
            if let Some(zone) = self.zoning_storage.get_mut(id) {
                draw_zone_area(
                    zone,
                    variables,
                    gizmo,
                    Some([1.1, 1.1, 1.1, 1.0]),
                    Some(new_zone_type),
                );
                if input.action_pressed_once("Place Zoning Point") {
                    zone.zone_type = *new_zone_type;
                }
            }
        } else {
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
                        PlacePos::CurrentZoneFirstPoint(pos, _) => {
                            if let Some(zone) = self.zoning_storage.get_mut(zone_id) {
                                if zone.points.len() >= 3 {
                                    zone.add_point(pos);
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
                        PlacePos::OtherZonePoint(pos, _) => {
                            if let Some(zone) = self.zoning_storage.get_mut(zone_id) {
                                zone.add_point(pos);
                            }
                        }
                    }
                } else {
                    let zone_id = self
                        .zoning_storage
                        .spawn(Zone::new(vec![best_place.pos()], *new_zone_type));
                    self.zoning_state = Some(ZoningState { zone_id });
                }
            }
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
        gizmo.line(last, place_pos.pos(), [0.0, 1.0, 0.0, 1.0], 0.5, 0.0);
        if self.segment_intersects_any_zone(place_pos.pos(), last, active_zone_id, chunk_size) {
            return false;
        };
        println!("{:?}", place_pos);
        match place_pos {
            PlacePos::Free(_, _) => true,
            PlacePos::RoadSnap(_, _) => true,
            PlacePos::CurrentZoneFirstPoint(_, _) => {
                let point_amount_requirement = zone.points.len() >= 3;

                let enclosing_another_area = self
                    .zoning_storage
                    .iter_zones()
                    .filter(|other_zone| other_zone.id != zone.id)
                    .any(|other_zone| {
                        other_zone.points.iter().any(|point| {
                            point_inside_polygon(*point, &zone.points, gizmo.chunk_size, true)
                        })
                    });

                point_amount_requirement && !enclosing_another_area
            }
            PlacePos::CurrentZonePoint(_, _) => false,
            PlacePos::CurrentZoneLastPoint(_, _) => false,
            PlacePos::OtherZonePoint(_, _) => true,
        }
    }

    fn segment_intersects_any_zone(
        &self,
        a: WorldPos,
        b: WorldPos,
        active_zone_id: Option<ZoneId>,
        chunk_size: ChunkSize,
    ) -> bool {
        const CLEARANCE: f64 = 0.0; // 1 meter clearance

        for zone in self.zoning_storage.iter_zones() {
            let points = &zone.points;
            if points.len() < 2 {
                continue;
            }

            for edge in points.windows(2) {
                let c = edge[0];
                let d = edge[1];

                if segment_intersection_xz(a, b, c, d, chunk_size, CLEARANCE).is_some() {
                    //println!("first one, {}", a.local.x);
                    return true;
                }
            }

            if Some(zone.id) != active_zone_id && points.len() >= 3 {
                let c = *points.last().unwrap();
                let d = points[0];

                if segment_intersection_xz(a, b, c, d, chunk_size, CLEARANCE).is_some() {
                    //println!("second one");
                    return true;
                }
            }
        }

        false
    }
}

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

/// Calculates the minimum distance from a point `p` to the line segment defined by `a` and `b`.
/// Returns the distance in world units (squared distance for performance).
pub fn point_to_segment_distance_sq(
    p: WorldPos,
    a: WorldPos,
    b: WorldPos,
    chunk_size: ChunkSize,
) -> f32 {
    // 1. Convert relevant points to render coordinates (f32)
    let p_render = p.to_render_pos(p, chunk_size);
    let a_render = a.to_render_pos(a, chunk_size);
    let b_render = b.to_render_pos(b, chunk_size);

    // 2. Vector from A to B
    let ab_x = b_render.x - a_render.x;
    let ab_z = b_render.z - a_render.z;

    // 3. Vector from A to P
    let ap_x = p_render.x - a_render.x;
    let ap_z = p_render.z - a_render.z;

    // 4. Calculate the squared length of the segment AB
    let len_sq = ab_x * ab_x + ab_z * ab_z;

    // Handle degenerate case: segment is a point
    if len_sq < 1e-10 {
        return ap_x * ap_x + ap_z * ap_z;
    }

    // 5. Project AP onto AB to find the closest point on the infinite line
    // t represents how far along the segment (0.0 to 1.0) the closest point is
    let t = (ap_x * ab_x + ap_z * ab_z) / len_sq;

    // 6. Clamp t to the segment bounds [0.0, 1.0]
    // If t < 0, closest point is A. If t > 1, closest point is B.
    let t_clamped = t.max(0.0).min(1.0);

    // 7. Calculate the coordinates of the closest point on the segment
    let closest_x = a_render.x + t_clamped * ab_x;
    let closest_z = a_render.z + t_clamped * ab_z;

    // 8. Calculate squared distance from P to the closest point
    let dx = p_render.x - closest_x;
    let dz = p_render.z - closest_z;

    dx * dx + dz * dz
}

/// Wrapper to return the actual distance (not squared).
pub fn point_to_segment_distance(
    p: WorldPos,
    a: WorldPos,
    b: WorldPos,
    chunk_size: ChunkSize,
) -> f32 {
    point_to_segment_distance_sq(p, a, b, chunk_size).sqrt()
}
/// Find intersection point of two segments (XZ plane).
/// Returns the intersection point in WorldPos, or None if parallel/non-intersecting.
///
/// `tolerance`: If provided, segments must intersect with at least this much
/// clearance from their endpoints (in world units). Use this to allow segments
/// to get close without triggering an intersection.
pub fn segment_intersection_xz(
    a1: WorldPos,
    a2: WorldPos,
    b1: WorldPos,
    b2: WorldPos,
    chunk_size: ChunkSize,
    tolerance: f64,
) -> Option<WorldPos> {
    let d1 = a2.sub_world_pos(a1, chunk_size);
    let d2 = b2.sub_world_pos(b1, chunk_size);
    let d12 = b1.sub_world_pos(a1, chunk_size);

    // ✅ Convert to full world coordinates
    let d1_x = d1.chunk.x as f64 * chunk_size as f64 + d1.local.x as f64;
    let d1_z = d1.chunk.z as f64 * chunk_size as f64 + d1.local.z as f64;
    let d2_x = d2.chunk.x as f64 * chunk_size as f64 + d2.local.x as f64;
    let d2_z = d2.chunk.z as f64 * chunk_size as f64 + d2.local.z as f64;
    let d12_x = d12.chunk.x as f64 * chunk_size as f64 + d12.local.x as f64;
    let d12_z = d12.chunk.z as f64 * chunk_size as f64 + d12.local.z as f64;

    let cross = d1_x * d2_z - d1_z * d2_x;

    if cross.abs() < 1e-10 {
        return None;
    }

    let t = (d12_x * d2_z - d12_z * d2_x) / cross;
    let u = (d12_x * d1_z - d12_z * d1_x) / cross;

    // Convert tolerance directly to parametric epsilon
    let eps = if tolerance == 0.0 {
        0.0 // Actually respect zero tolerance!
    } else {
        let len_a = (d1_x * d1_x + d1_z * d1_z).sqrt();
        let len_b = (d2_x * d2_x + d2_z * d2_z).sqrt();
        // Use MINIMUM or average, not MAXIMUM!
        let eps_a = tolerance / len_a.max(1e-6);
        let eps_b = tolerance / len_b.max(1e-6);
        eps_a.min(eps_b) // Most conservative
    };

    if t > eps && t < 1.0 - eps && u > eps && u < 1.0 - eps {
        let intersection = WorldPos {
            chunk: ChunkCoord::new(a1.chunk.x + d1.chunk.x, a1.chunk.z + d1.chunk.z),
            local: LocalPos::new(
                a1.local.x + t as f32 * d1.local.x,
                a1.local.y + t as f32 * d1.local.y,
                a1.local.z + t as f32 * d1.local.z,
            ),
        }
        .normalize(chunk_size);

        Some(intersection)
    } else {
        None
    }
}

pub fn point_inside_polygon(
    p: WorldPos,
    poly: &[WorldPos],
    chunk_size: ChunkSize,
    wrapped: bool,
) -> bool {
    let n = poly.len();
    if n < 2 {
        return false;
    }

    let p2 = p.to_render_pos(p, chunk_size); // (0,0,0)
    let mut inside = false;

    let edge_count = if wrapped { n } else { n - 1 };

    for i in 0..edge_count {
        let a = poly[i].to_render_pos(p, chunk_size);
        let b = if i + 1 < n {
            poly[i + 1].to_render_pos(p, chunk_size)
        } else {
            // only happens when wrapped == true and i == n-1
            poly[0].to_render_pos(p, chunk_size)
        };

        let ax = a.x;
        let az = a.z;
        let bx = b.x;
        let bz = b.z;

        // On-edge check
        let cross = (bx - ax) * (p2.z - az) - (bz - az) * (p2.x - ax);
        let dot = (p2.x - ax) * (p2.x - bx) + (p2.z - az) * (p2.z - bz);
        if cross.abs() < 1e-6 && dot <= 0.0 {
            return true;
        }

        // Ray cast (+X)
        let intersects =
            ((az > p2.z) != (bz > p2.z)) && (p2.x < (bx - ax) * (p2.z - az) / (bz - az) + ax);

        if intersects {
            inside = !inside;
        }
    }

    inside
}

fn draw_zone_area(
    zone: &Zone,
    variables: &Variables,
    gizmo: &mut Gizmo,
    color_multiplier: Option<[f32; 4]>,
    predicted_zone_type: Option<&ZoneType>,
) {
    let zone_type = predicted_zone_type.unwrap_or(&zone.zone_type);

    let key = match zone_type {
        ZoneType::None => "none_zone_color",
        ZoneType::Residential => "residential_zone_color",
        ZoneType::Commercial => "commercial_zone_color",
        ZoneType::Industrial => "industrial_zone_color",
        ZoneType::Office => "office_zone_color",
    };

    if let Some(mut c) = variables.get(key).unwrap_or(&Value::Null).as_color4() {
        if let Some(m) = color_multiplier {
            for i in 0..4 {
                c[i] *= m[i];
            }
        }
        gizmo.area_textured(zone.points.as_slice(), c, 0.0);
    }
}
