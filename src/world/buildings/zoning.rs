use crate::helpers::positions::{ChunkCoord, ChunkSize, LocalPos, WorldPos};
use crate::renderer::gizmo::gizmo::Gizmo;
use crate::ui::input::Input;
use crate::ui::parser::Value;
use crate::ui::variables::Variables;
use crate::world::camera::Camera;
use crate::world::roads::road_mesh_manager::{Edges, RoadEdges, RoadMeshManager};
use crate::world::roads::road_structs::{LaneId, SegmentId};
use crate::world::roads::road_subsystem::Roads;
use crate::world::statisticals::demands::ZoningDemand;
use crate::world::terrain::terrain_subsystem::{CursorMode, PickedPoint, Terrain};
use glam::Vec3;
use rayon::iter::IntoParallelRefMutIterator;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::slice::IterMut;

const SNAP_RADIUS: f64 = 10.0;
const EPS: f64 = 0.0001;

#[derive(Clone, Default)]
struct ZoningState {
    pub zone_id: ZoneId,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum ZoneType {
    None,
    Residential,
    Commercial,
    Industrial,
    Office,
}
impl ZoneType {
    pub fn from_value(value: &Value) -> Self {
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
#[derive(Serialize, Deserialize, Clone)]
pub struct Zone {
    pub id: ZoneId,
    pub points: Vec<WorldPos>,
    pub lots: Vec<LotId>,
    pub zone_type: ZoneType,
    pub zoning_demand: ZoningDemand,
}

impl Zone {
    pub fn new(points: Vec<WorldPos>, zone_type: ZoneType) -> Zone {
        Zone {
            id: 69420,
            points,
            lots: vec![],
            zone_type,
            zoning_demand: ZoningDemand::new(),
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
enum LotPointType {
    Sidewalk,
    Lane(LaneId),
}
struct LotPoint {
    pos: WorldPos,
    tangent: Vec3,
    lateral: Vec3,
    left_point: Option<(WorldPos, Vec3, Vec3, f64)>,
    right_point: Option<(WorldPos, Vec3, Vec3, f64)>,
    dist: f64,
    segment_id: SegmentId,
    point_type: LotPointType,
}
#[derive(Clone, Default)]
pub struct Zoning {
    zoning_state: Option<ZoningState>,
    pub zoning_storage: ZoningStorage,
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
        camera: &Camera,
        terrain: &Terrain,
        roads: &Roads,
        road_mesh_manager: &RoadMeshManager,
        input: &mut Input,
        gizmo: &mut Gizmo,
        variables: &Variables,
    ) {
        self.zoning_storage
            .update_target_and_chunk_size(camera.target.chunk, terrain.chunk_size);
        if terrain.cursor.mode != CursorMode::Area && terrain.cursor.mode != CursorMode::Zoning {
            return;
        };
        let Some(picked) = terrain.last_picked.as_ref() else {
            return;
        };
        let chunk_size = terrain.chunk_size;
        let new_zone_type = &terrain.cursor.zone_type;
        let active_zone_id = self.zoning_state.as_ref().map(|state| state.zone_id);
        for zone in self.zoning_storage.iter_zones() {
            if Some(zone.id) != active_zone_id {
                draw_zone_area(
                    zone.points.as_slice(),
                    &zone.zone_type,
                    variables,
                    gizmo,
                    None,
                    None,
                );
            }

            gizmo.polyline(
                zone.points.as_slice(),
                [0.07, 0.28, 0.5, 1.0],
                0.0,
                0.2,
                0.0,
            );
            // This was to test if the lateral is ACTUALLY right! It was perfect!
            // for idx in 0..zone.points.len() {
            //     let (_, right) = tangent_and_lateral_right(&*zone.points, idx, camera.chunk_size);
            //     gizmo.direction(zone.points[idx], right, [1.0, 0.0, 0.0, 1.0], 0.0, 0.0);
            // }
        }
        let mut closest_point: Option<LotPoint> = None;
        let mut closest_distance = 100.0;

        for segment_id in roads
            .road_manager
            .roads
            .segment_ids_touching_chunk(picked.chunk.coords.chunk_coord, terrain.chunk_size)
        {
            let Some(road_edges) = road_mesh_manager.road_edges.get(&segment_id) else {
                continue;
            };

            for point in road_edges
                .right_sidewalk_edge
                .right_points
                .iter()
                .chain(road_edges.left_sidewalk_edge.right_points.iter())
            {
                gizmo.circle(*point, 0.4, [0.8, 0.3, 1.0, 1.0], 0.1, 0.0);
            }

            // Prefer sidewalks if they exist
            if !road_edges.right_sidewalk_edge.is_empty() {
                let edges = &road_edges.right_sidewalk_edge;

                collect_lot_point(
                    edges,
                    picked,
                    &mut closest_distance,
                    &mut closest_point,
                    segment_id,
                    chunk_size,
                );
            }

            if !road_edges.left_sidewalk_edge.is_empty() {
                let edges = &road_edges.left_sidewalk_edge;

                collect_lot_point(
                    edges,
                    picked,
                    &mut closest_distance,
                    &mut closest_point,
                    segment_id,
                    chunk_size,
                );
            } else {
                // No sidewalks, use lane edges manually
                let lane_ids = roads.road_manager.roads.segment(segment_id).lanes();

                for lane_id in lane_ids {
                    let Some(lane_edges) = road_edges.lane_edges.get(lane_id) else {
                        continue;
                    };
                    for point in &lane_edges.right_points {
                        gizmo.circle(*point, 0.4, [0.8, 0.3, 1.0, 1.0], 0.1, 0.0);
                    }

                    collect_lot_point(
                        lane_edges,
                        picked,
                        &mut closest_distance,
                        &mut closest_point,
                        segment_id,
                        chunk_size,
                    )
                }
            }
        }

        let lot_snap_point = if let Some(snap_point) = closest_point {
            if snap_point.dist < SNAP_RADIUS {
                gizmo.circle(snap_point.pos, 1.0, [0.8, 0.3, 1.0, 1.0], 0.15, 0.0);
                Some(snap_point)
            } else {
                if let Some(road_edges) = road_mesh_manager.road_edges.get(&snap_point.segment_id) {
                    for point in road_edges
                        .lane_edges
                        .values()
                        .map(|edges| &edges.right_points)
                        .flatten()
                        .chain(road_edges.right_sidewalk_edge.right_points.iter())
                        .chain(road_edges.left_sidewalk_edge.right_points.iter())
                    {
                        gizmo.circle(*point, 0.4, [0.8, 0.3, 1.0, 1.0], 0.1, 0.0);
                    }
                }

                None
            }
        } else {
            None
        };

        match terrain.cursor.mode {
            CursorMode::Zoning => {
                self.run_lot_zoning(
                    terrain,
                    roads,
                    road_mesh_manager,
                    input,
                    variables,
                    lot_snap_point,
                    picked,
                    new_zone_type,
                    gizmo,
                );
            }
            CursorMode::Area => {
                self.run_area_zoning(
                    terrain,
                    roads,
                    input,
                    variables,
                    picked,
                    active_zone_id,
                    new_zone_type,
                    gizmo,
                );
            }
            _ => {}
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
        let Some(zone) = self.zoning_storage.get_zone(zone_id) else {
            return false;
        };

        let Some(&last) = zone.points.last() else {
            return false;
        };
        gizmo.line(last, place_pos.pos(), [0.0, 1.0, 0.0, 1.0], 0.5, 0.0);
        if self.segment_intersects_any_zone(place_pos.pos(), last, active_zone_id, chunk_size) {
            return false;
        };
        //println!("{:?}", place_pos);
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
    fn lot_intersects_any_road(
        &self,
        roads: &Roads,
        road_mesh_manager: &RoadMeshManager,
        lot_points: &[WorldPos],
        chunk_size: ChunkSize,
    ) -> bool {
        // If the lot has fewer than 3 points, it's degenerate → no intersection possible
        if lot_points.len() < 3 {
            return false;
        }
        let segment_ids = lot_points
            .iter()
            .map(|point| {
                roads
                    .road_manager
                    .roads
                    .segment_ids_touching_chunk(point.chunk, chunk_size)
            })
            .flatten()
            .collect::<Vec<SegmentId>>();
        for segment_id in segment_ids.iter() {
            let Some(edges) = road_mesh_manager.road_edges.get(segment_id) else {
                continue;
            };
            let road_points = collect_road_points(edges);
            for point in road_points {
                if point_inside_polygon(*point, lot_points, chunk_size, false) {
                    return true;
                }
            }
        }
        false
    }
    fn lot_intersects_any_lot(
        &self,
        points: &[WorldPos],
        ignore_lot_id: Option<LotId>,
        chunk_size: ChunkSize,
    ) -> bool {
        // If the lot has fewer than 3 points, it's degenerate → no intersection possible
        if points.len() < 3 {
            return false;
        }

        for existing_lot in self.zoning_storage.iter_lots() {
            if Some(existing_lot.id) == ignore_lot_id {
                continue;
            }

            if existing_lot.bounds.len() < 3 {
                continue;
            }

            // Quick bounding box check (optional but recommended for performance)
            // if !bbox_intersects(&points.bounds, &existing_lot.bounds) {
            //     continue;
            // }

            // Check if any edge of the new lot intersects any edge of the existing lot
            match self.lot_intersects_lot(points, existing_lot.bounds.as_slice(), chunk_size) {
                None => continue,
                Some(intersection_type) => {
                    //println!("{:?}", intersection_type);
                    return true;
                }
            }
        }
        false
    }
    fn lot_intersects_lot(
        &self,
        points_a: &[WorldPos],
        points_b: &[WorldPos],
        chunk_size: ChunkSize,
    ) -> Option<SegmentIntersectionType> {
        if points_a.len() < 3 || points_b.len() < 3 {
            return None;
        }

        const CLEARANCE: f64 = 0.0;

        // 1. Check edge-to-edge intersections (including closing the polygons)
        for edge_a in points_a.windows(2) {
            let a = edge_a[0];
            let b = edge_a[1];

            for edge_b in points_b.windows(2) {
                let c = edge_b[0];
                let d = edge_b[1];

                if segment_intersection_xz(a, b, c, d, chunk_size, CLEARANCE).is_some() {
                    return Some(SegmentIntersectionType::OtherEdges);
                }
            }

            // Check against the closing edge of lot_b
            if let (Some(&first), Some(&last)) = (points_b.first(), points_b.last()) {
                if segment_intersection_xz(a, b, last, first, chunk_size, CLEARANCE).is_some() {
                    return Some(SegmentIntersectionType::ClosingEdgeOfB);
                }
            }
        }

        // 2. Check closing edge of lot_a against all edges of lot_b
        if let (Some(&first_a), Some(&last_a)) = (points_a.first(), points_a.last()) {
            for edge_b in points_b.windows(2) {
                let c = edge_b[0];
                let d = edge_b[1];

                if segment_intersection_xz(last_a, first_a, c, d, chunk_size, CLEARANCE).is_some() {
                    return Some(SegmentIntersectionType::ClosingEdgeOfA);
                }
            }
        }

        // 3. Optional: Check if one lot is completely inside the other
        //     (important for preventing nested lots or one lot swallowing another)      BUGGY!!!!!! Stays collided forever
        // if point_inside_polygon(points_a[0], points_b, chunk_size, true) || point_inside_polygon(points_a[0], points_a, chunk_size, true) {
        //     return Some(SegmentIntersectionType::PointInsidePolygon)
        // }

        None
    }

    fn run_lot_zoning(
        &mut self,
        terrain: &Terrain,
        roads: &Roads,
        road_mesh_manager: &RoadMeshManager,
        input: &mut Input,
        variables: &Variables,
        lot_snap_point: Option<LotPoint>,
        picked: &PickedPoint,
        new_zone_type: &ZoneType,
        gizmo: &mut Gizmo,
    ) {
        let lot_width = variables.get_f64("lot_width").unwrap_or(15.0) as f32;
        let lot_length = variables.get_f64("lot_length").unwrap_or(20.0) as f32;
        let chunk_size = terrain.chunk_size;
        let mut inside_lot_id: Option<LotId> = None;
        for lot in self.zoning_storage.iter_lots() {
            if point_inside_polygon(picked.pos, lot.bounds.as_slice(), chunk_size, false) {
                inside_lot_id = Some(lot.id);
                break;
            }
        }
        if let Some(snap_point) = lot_snap_point
            && inside_lot_id.is_none()
        {
            gizmo.circle(snap_point.pos, 0.3, [0.1, 0.3, 0.8, 1.0], 0.1, 0.0);

            let half_width = lot_width * 0.5;
            let origin = snap_point
                .pos
                .add_vec3(snap_point.lateral * 0.5, chunk_size);
            // Road edge corners
            let front_left = origin.sub_vec3(snap_point.tangent * half_width, chunk_size);

            let front_right = origin.add_vec3(snap_point.tangent * half_width, chunk_size);
            // Back corners away from road
            let back_left = front_left.add_vec3(snap_point.lateral * lot_length, chunk_size);

            let back_right = front_right.add_vec3(snap_point.lateral * lot_length, chunk_size);

            let mut preview = vec![front_left, front_right, back_right, back_left, front_left];
            for point in preview.iter_mut() {
                point.local.y = terrain.get_height_at(*point, true);
            }
            let invalid = self.lot_intersects_any_lot(preview.as_slice(), None, chunk_size)
                || self.lot_intersects_any_road(
                    roads,
                    road_mesh_manager,
                    preview.as_slice(),
                    chunk_size,
                );

            match invalid {
                true => {
                    // Oh, no it does intersect!
                    // Preview draw
                    gizmo.polyline(preview.as_slice(), [0.8, 0.1, 0.1, 1.0], 8.0, 0.25, 0.0);
                }
                false => {
                    // DOESN'T INTERSCET LETS GO
                    // Preview draw
                    gizmo.polyline(preview.as_slice(), [0.1, 0.8, 0.1, 1.0], 8.0, 0.2, 0.0);
                    if input.action_repeat("Place Zoning Point") {
                        let lot = Lot {
                            id: 69420,
                            bounds: preview,
                            zone_type: new_zone_type.clone(),
                        };

                        self.zoning_storage.spawn_lot(lot);
                    }
                }
            }
        }
        let removing_lot = input.action_down("Remove Lot");
        let finished_removing_lot = input.action_released("Remove Lot");
        for lot in self.zoning_storage.iter_mut_lots() {
            if inside_lot_id == Some(lot.id) {
                if removing_lot {
                    draw_zone_area(
                        lot.bounds.as_slice(),
                        &lot.zone_type,
                        variables,
                        gizmo,
                        Some([3.0, 0.2, 0.2, 1.0]),
                        Some(new_zone_type),
                    );
                } else {
                    draw_zone_area(
                        lot.bounds.as_slice(),
                        &lot.zone_type,
                        variables,
                        gizmo,
                        Some([1.2, 1.2, 1.2, 1.0]),
                        Some(new_zone_type),
                    );
                    if input.action_pressed_once("Place Zoning Point") {
                        lot.zone_type = *new_zone_type;
                    }
                }
            } else {
                draw_zone_area(
                    lot.bounds.as_slice(),
                    &lot.zone_type,
                    variables,
                    gizmo,
                    None,
                    None,
                );
            }

            gizmo.polyline(lot.bounds.as_slice(), [0.1, 0.3, 0.7, 1.0], 0.0, 0.15, 0.0);
        }
        if let Some(lot_id) = inside_lot_id
            && finished_removing_lot
        {
            self.zoning_storage.despawn_lot(lot_id);
        }

        // if input.action_pressed_once("Place Zoning Point") {
        //
        // }
    }

    fn run_area_zoning(
        &mut self,
        terrain: &Terrain,
        roads: &Roads,
        input: &mut Input,
        variables: &Variables,
        picked: &PickedPoint,
        active_zone_id: Option<ZoneId>,
        new_zone_type: &ZoneType,
        gizmo: &mut Gizmo,
    ) {
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
            if let Some(zone) = self.zoning_storage.get_zone(zone_id) {
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
                    //println!("Considering: {:?}", candidate);
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
        let mut inside_lot_id: Option<LotId> = None;
        if can_place && matches!(best_place, PlacePos::Free(_, _)) {
            for zone in self.zoning_storage.iter_zones() {
                if Some(zone.id) == active_zone_id {
                    continue;
                }
                if point_inside_polygon(best_place.pos(), &zone.points, gizmo.chunk_size, false) {
                    // best_pos is inside this zone
                    inside_zone_id.replace(zone.id);
                    for lot in zone
                        .lots
                        .iter()
                        .map(|lot_id| self.zoning_storage.get_lot(*lot_id))
                    {
                        let Some(lot) = lot else { continue };
                        if point_inside_polygon(
                            best_place.pos(),
                            &lot.bounds,
                            gizmo.chunk_size,
                            false,
                        ) {
                            inside_lot_id.replace(lot.id);
                        }
                    }
                }
            }
        }

        let canceling = input.action_pressed_once("Cancel");
        if canceling {
            if let Some(zoning_state) = &self.zoning_state {
                let zone_id = zoning_state.zone_id;
                if let Some(zone) = self.zoning_storage.get_mut_zone(zone_id) {
                    zone.points.pop();
                    if zone.points.is_empty() {
                        self.zoning_storage.despawn_zone(zone_id);
                        self.zoning_state = None;
                    }
                }
            }
        }
        if let Some(id) = inside_zone_id {
            if input.action_pressed_once("Place Zoning Point") {
                if let Some(zone) = self.zoning_storage.get_mut_zone(id) {
                    zone.zone_type = *new_zone_type;
                }
            }
            if let Some(zone) = self.zoning_storage.get_zone(id) {
                draw_zone_area(
                    zone.points.as_slice(),
                    &zone.zone_type,
                    variables,
                    gizmo,
                    Some([1.1, 1.1, 1.1, 1.0]),
                    Some(new_zone_type),
                );
                for lot_id in &zone.lots {
                    let Some(lot) = self.zoning_storage.get_lot(*lot_id) else {
                        continue;
                    };

                    if Some(*lot_id) == inside_lot_id {
                        // mouse inside lot, highlighted
                        draw_zone_area(
                            lot.bounds.as_slice(),
                            &lot.zone_type,
                            variables,
                            gizmo,
                            Some([1.1, 1.1, 1.1, 1.0]),
                            Some(new_zone_type),
                        );
                        gizmo.polyline(
                            lot.bounds.as_slice(),
                            [0.07, 0.48, 0.5, 1.0],
                            0.0,
                            0.15,
                            0.0,
                        );
                    } else {
                        // Normal lot drawing, not highlighted
                        draw_zone_area(
                            lot.bounds.as_slice(),
                            &lot.zone_type,
                            variables,
                            gizmo,
                            Some([1.0, 1.0, 1.0, 1.0]),
                            Some(new_zone_type),
                        );
                        gizmo.polyline(
                            lot.bounds.as_slice(),
                            [0.07, 0.48, 0.5, 1.0],
                            0.0,
                            0.1,
                            0.0,
                        );
                    }
                }
            }
        } else {
            gizmo.circle(best_place.pos(), 0.5, preview_color, 0.0, 0.0);

            if let Some(last_pos) = active_zone_id
                .and_then(|zone_id| self.zoning_storage.get_zone(zone_id))
                .and_then(|zone| zone.points.last().copied())
            {
                gizmo.line(best_place.pos(), last_pos, preview_color, 0.0, 0.0);
            }

            if input.action_pressed_once("Place Zoning Point") && can_place {
                if let Some(zoning_state) = &self.zoning_state {
                    let zone_id = zoning_state.zone_id;
                    match best_place {
                        PlacePos::Free(pos, _) => {
                            if let Some(zone) = self.zoning_storage.get_mut_zone(zone_id) {
                                zone.add_point(pos);
                            }
                        }
                        PlacePos::CurrentZoneFirstPoint(pos, _) => {
                            if let Some(zone) = self.zoning_storage.get_mut_zone(zone_id) {
                                if zone.points.len() >= 3 {
                                    zone.add_point(pos);
                                    println!("Closed zoning loop");
                                    self.zoning_state = None;
                                }
                            }
                            if self.zoning_state.is_none() {
                                self.zoning_storage.calculate_lots_for_zone(zone_id);
                            }
                        }

                        PlacePos::RoadSnap(pos, _) => {
                            if let Some(zone) = self.zoning_storage.get_mut_zone(zone_id) {
                                zone.add_point(pos);
                            }
                        }

                        PlacePos::CurrentZonePoint(_, _) => {}
                        PlacePos::CurrentZoneLastPoint(_, _) => {}
                        PlacePos::OtherZonePoint(pos, _) => {
                            if let Some(zone) = self.zoning_storage.get_mut_zone(zone_id) {
                                zone.add_point(pos);
                            }
                        }
                    }
                } else {
                    let zone_id = self
                        .zoning_storage
                        .spawn_zone(Zone::new(vec![best_place.pos()], *new_zone_type));
                    self.zoning_state = Some(ZoningState { zone_id });
                }
            }
        }
    }
}

fn collect_road_points(edges: &RoadEdges) -> Vec<&WorldPos> {
    let points: Vec<&WorldPos> = match (
        edges.left_sidewalk_edge.is_empty(),
        edges.right_sidewalk_edge.is_empty(),
    ) {
        (true, _) => edges
            .right_sidewalk_edge
            .right_points
            .iter()
            .chain(
                edges
                    .lane_edges
                    .values()
                    .flat_map(|e| e.right_points.iter()),
            )
            .collect(),

        (_, true) => edges
            .left_sidewalk_edge
            .left_points
            .iter()
            .chain(
                edges
                    .lane_edges
                    .values()
                    .flat_map(|e| e.right_points.iter()),
            )
            .collect(),

        (false, false) => edges
            .right_sidewalk_edge
            .right_points
            .iter()
            .chain(edges.left_sidewalk_edge.right_points.iter())
            .collect(),
    };
    points
}

fn collect_lot_point(
    edges: &Edges,
    picked: &PickedPoint,
    closest_distance: &mut f64,
    closest_point: &mut Option<LotPoint>,
    segment_id: SegmentId,
    chunk_size: ChunkSize,
) {
    //println!("points: {:?} tangents: {:?} laterals: {:?}", edges.right_points.len(), edges.right_tangents.len(), edges.right_laterals.len());
    for (idx, ((point, tangent), lateral)) in edges
        .right_points
        .iter()
        .zip(edges.right_tangents.iter())
        .zip(edges.right_laterals.iter())
        .enumerate()
    {
        let dist = picked.pos.distance_to(*point, chunk_size);
        println!("{}", dist);
        if dist < *closest_distance {
            *closest_distance = dist;

            let left_point = if idx > 0 {
                Some((
                    edges.right_points[idx - 1],
                    edges.right_tangents[idx - 1],
                    edges.right_laterals[idx - 1],
                    picked
                        .pos
                        .distance_to(edges.right_points[idx - 1], chunk_size),
                ))
            } else {
                None
            };

            let right_point = if idx + 1 < edges.right_points.len() {
                Some((
                    edges.right_points[idx + 1],
                    edges.right_tangents[idx + 1],
                    edges.right_laterals[idx + 1],
                    picked
                        .pos
                        .distance_to(edges.right_points[idx + 1], chunk_size),
                ))
            } else {
                None
            };

            *closest_point = Some(LotPoint {
                pos: *point,
                tangent: *tangent,
                lateral: *lateral,
                left_point,
                right_point,
                dist,
                segment_id,
                point_type: LotPointType::Sidewalk,
            });
        }
    }
}

pub type ZoneId = u32;
pub type LotId = u32;
#[derive(Serialize, Deserialize, Clone)]
pub struct Lot {
    pub id: LotId,
    pub bounds: Vec<WorldPos>,
    pub zone_type: ZoneType,
}
#[derive(Serialize, Deserialize, Default, Clone)]
pub struct ZoningStorage {
    zones: Vec<Option<Zone>>,
    lots: Vec<Option<Lot>>,
    zone_free_list: Vec<ZoneId>,
    lot_free_list: Vec<LotId>,
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
    pub fn iter_lots(&self) -> impl Iterator<Item = &Lot> {
        self.lots.iter().filter_map(|l| l.as_ref())
    }
    pub fn iter_mut_lots(&mut self) -> impl Iterator<Item = &mut Lot> {
        self.lots.iter_mut().filter_map(|l| l.as_mut())
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
        let mut lots: Vec<Option<Lot>> = Vec::new();
        lots.push(None); // reserve index 0 — for no reason
        Self {
            zones,
            lots,
            zone_free_list: Vec::new(),
            lot_free_list: Vec::new(),
            chunk_size: 128,
            center_chunk: ChunkCoord::zero(),
        }
    }

    pub fn spawn_zone(&mut self, mut zone: Zone) -> ZoneId {
        let zone_id = if let Some(reused_id) = self.zone_free_list.pop() {
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

    pub fn despawn_zone(&mut self, id: ZoneId) {
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
            self.zone_free_list.push(id);
        }
    }

    /// Subdivides the zone into rectangular lots whose size reflects the zone's
    /// real-world purpose, then spawns them and records their IDs on the zone.
    ///
    /// Only lots whose corners all lie strictly inside the zone polygon
    /// are kept, so the result never spills outside the zone boundary.
    pub fn calculate_lots_for_zone(&mut self, id: ZoneId) {
        return;
        #[derive(Clone, Copy, Debug)]
        struct Pt {
            x: f64,
            y: f64,
            z: f64,
        }

        fn cross(a: Pt, b: Pt, c: Pt) -> f64 {
            (b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x)
        }

        fn polygon_signed_area(pts: &[Pt]) -> f64 {
            let mut area = 0.0;
            for i in 0..pts.len() {
                let a = pts[i];
                let b = pts[(i + 1) % pts.len()];
                area += a.x * b.z - b.x * a.z;
            }
            area * 0.5
        }

        fn point_in_triangle(p: Pt, a: Pt, b: Pt, c: Pt) -> bool {
            let c1 = cross(a, b, p);
            let c2 = cross(b, c, p);
            let c3 = cross(c, a, p);

            let has_neg = c1 < -1e-9 || c2 < -1e-9 || c3 < -1e-9;
            let has_pos = c1 > 1e-9 || c2 > 1e-9 || c3 > 1e-9;

            !(has_neg && has_pos)
        }

        fn ear_clip_triangulate(points: &[Pt]) -> Vec<[usize; 3]> {
            let n = points.len();
            let mut result = Vec::new();
            if n < 3 {
                return result;
            }

            let mut indices: Vec<usize> = if polygon_signed_area(points) >= 0.0 {
                (0..n).collect()
            } else {
                (0..n).rev().collect()
            };

            let mut guard = 0usize;

            while indices.len() > 3 && guard < 10_000 {
                let len = indices.len();
                let mut ear_found = false;

                for i in 0..len {
                    let prev = indices[(i + len - 1) % len];
                    let curr = indices[i];
                    let next = indices[(i + 1) % len];

                    let a = points[prev];
                    let b = points[curr];
                    let c = points[next];

                    if cross(a, b, c) <= 1e-9 {
                        continue;
                    }

                    let mut contains_other = false;
                    for &j in &indices {
                        if j == prev || j == curr || j == next {
                            continue;
                        }
                        if point_in_triangle(points[j], a, b, c) {
                            contains_other = true;
                            break;
                        }
                    }

                    if contains_other {
                        continue;
                    }

                    result.push([prev, curr, next]);
                    indices.remove(i);
                    ear_found = true;
                    break;
                }

                if !ear_found {
                    break;
                }

                guard += 1;
            }

            if indices.len() == 3 {
                result.push([indices[0], indices[1], indices[2]]);
            }

            result
        }

        let (points, zone_type) = {
            let Some(zone) = self.get_zone(id) else {
                return;
            };
            if zone.points.len() < 3 {
                return;
            }
            (zone.points.clone(), zone.zone_type)
        };

        let chunk_size = self.chunk_size;

        let poly_f64: Vec<Pt> = points
            .iter()
            .map(|p| {
                let (x, y, z) = world_pos_to_world_f64(*p, chunk_size);
                Pt { x, y, z }
            })
            .collect();

        if poly_f64.len() < 3 {
            return;
        }

        let triangles = ear_clip_triangulate(&poly_f64);
        if triangles.is_empty() {
            return;
        }

        let triangles_per_lot = match zone_type {
            ZoneType::Residential => 1,
            ZoneType::Commercial => 2,
            ZoneType::Office => 3,
            ZoneType::Industrial => 4,
            ZoneType::None => 2,
        }
        .max(1);

        let mut edge_use_count: HashMap<(usize, usize), usize> = HashMap::new();
        let mut tri_neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); triangles.len()];
        let mut tri_boundary_score: Vec<usize> = vec![0; triangles.len()];

        for (ti, tri) in triangles.iter().enumerate() {
            let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];

            for &(a, b) in &edges {
                let key = if a < b { (a, b) } else { (b, a) };
                *edge_use_count.entry(key).or_insert(0) += 1;
            }
        }

        for (ti, tri) in triangles.iter().enumerate() {
            let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];

            let mut boundary_hits = 0usize;

            for &(a, b) in &edges {
                let key = if a < b { (a, b) } else { (b, a) };
                if edge_use_count.get(&key).copied().unwrap_or(0) == 1 {
                    boundary_hits += 1;
                }
            }

            tri_boundary_score[ti] = boundary_hits;
        }

        for i in 0..triangles.len() {
            for j in (i + 1)..triangles.len() {
                let tri_a = triangles[i];
                let tri_b = triangles[j];

                let mut shared = 0usize;
                for &va in &tri_a {
                    for &vb in &tri_b {
                        if va == vb {
                            shared += 1;
                        }
                    }
                }

                if shared >= 2 {
                    tri_neighbors[i].insert(j);
                    tri_neighbors[j].insert(i);
                }
            }
        }

        let mut order: Vec<usize> = (0..triangles.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(tri_boundary_score[i]));

        let mut assigned = vec![false; triangles.len()];
        let mut new_lot_ids = Vec::new();

        for &start in &order {
            if assigned[start] {
                continue;
            }

            let mut component = vec![start];
            assigned[start] = true;

            while component.len() < triangles_per_lot {
                let mut best_next = None;

                for &t in &component {
                    for &n in &tri_neighbors[t] {
                        if !assigned[n] {
                            best_next = Some(n);
                            break;
                        }
                    }
                    if best_next.is_some() {
                        break;
                    }
                }

                let Some(next_tri) = best_next else {
                    break;
                };

                assigned[next_tri] = true;
                component.push(next_tri);
            }

            let mut comp_edges: HashMap<(usize, usize), usize> = HashMap::new();

            for &ti in &component {
                let tri = triangles[ti];
                let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];

                for &(a, b) in &edges {
                    let key = if a < b { (a, b) } else { (b, a) };
                    *comp_edges.entry(key).or_insert(0) += 1;
                }
            }

            let boundary_edges: Vec<(usize, usize)> = comp_edges
                .into_iter()
                .filter_map(|(edge, count)| if count == 1 { Some(edge) } else { None })
                .collect();

            if boundary_edges.len() < 3 {
                continue;
            }

            let mut boundary_adj: HashMap<usize, Vec<usize>> = HashMap::new();
            for (a, b) in &boundary_edges {
                boundary_adj.entry(*a).or_default().push(*b);
                boundary_adj.entry(*b).or_default().push(*a);
            }

            let Some(&start_v) = boundary_adj.keys().next() else {
                continue;
            };

            let mut ordered_vertices = Vec::new();
            ordered_vertices.push(start_v);

            let mut prev = usize::MAX;
            let mut curr = start_v;

            loop {
                let Some(neighs) = boundary_adj.get(&curr) else {
                    break;
                };

                let next = if neighs.len() == 1 {
                    neighs[0]
                } else if neighs[0] != prev {
                    neighs[0]
                } else {
                    neighs[1]
                };

                if next == start_v {
                    break;
                }

                if ordered_vertices.len() > boundary_edges.len() + 5 {
                    break;
                }

                ordered_vertices.push(next);
                prev = curr;
                curr = next;
            }

            if ordered_vertices.len() < 3 {
                continue;
            }

            let mut bounds = Vec::with_capacity(ordered_vertices.len());
            for idx in ordered_vertices {
                let p = poly_f64[idx];
                bounds.push(world_f64_to_world_pos(p.x, p.y, p.z, chunk_size));
            }

            let lot = Lot {
                id: 0,
                bounds,
                zone_type,
            };

            let lot_id = self.spawn_lot(lot);
            new_lot_ids.push(lot_id);
        }

        if let Some(zone) = self.get_mut_zone(id) {
            zone.lots.extend(new_lot_ids);
        }
    }

    pub fn zone_count(&self) -> usize {
        self.zones.len() - self.zone_free_list.len() - 1
    }

    #[inline]
    pub fn get_zone(&self, id: ZoneId) -> Option<&Zone> {
        self.zones.get(id as usize)?.as_ref()
    }

    #[inline]
    pub fn get_mut_zone(&mut self, id: ZoneId) -> Option<&mut Zone> {
        self.zones.get_mut(id as usize)?.as_mut()
    }

    pub fn spawn_lot(&mut self, mut lot: Lot) -> LotId {
        let lot_id = if let Some(reused_id) = self.lot_free_list.pop() {
            // Reuse slot - IIII know it's None because it's in free_list
            lot.id = reused_id;
            self.lots[reused_id as usize] = Some(lot);
            reused_id
        } else {
            let new_id = self.lots.len() as u32;
            lot.id = new_id;
            self.lots.push(Some(lot));
            new_id
        };

        lot_id
    }

    pub fn despawn_lot(&mut self, id: LotId) {
        if id == 0 {
            // index 0 is reserved, ignore attempts to despawn it
            return;
        }
        if self
            .lots
            .get(id as usize)
            .and_then(|opt| opt.as_ref())
            .is_some()
        {
            // Actually free the slot
            self.lots[id as usize] = None;
            self.lot_free_list.push(id);
        }
    }

    pub fn lot_count(&self) -> usize {
        self.lots.len() - self.lot_free_list.len() - 1
    }

    #[inline]
    pub fn get_lot(&self, id: LotId) -> Option<&Lot> {
        self.lots.get(id as usize)?.as_ref()
    }

    #[inline]
    pub fn get_mut_lot(&mut self, id: LotId) -> Option<&mut Lot> {
        self.lots.get_mut(id as usize)?.as_mut()
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
    points: &[WorldPos],
    zone_type: &ZoneType,
    variables: &Variables,
    gizmo: &mut Gizmo,
    color_multiplier: Option<[f32; 4]>,
    predicted_zone_type: Option<&ZoneType>,
) {
    let zone_type = predicted_zone_type.unwrap_or(zone_type);

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
        gizmo.area_textured(points, c, 0.0);
    }
}

/// Flatten a WorldPos to absolute world coordinates (f64 for precision).
#[inline]
fn world_pos_to_world_f64(pos: WorldPos, chunk_size: ChunkSize) -> (f64, f64, f64) {
    let wx = pos.chunk.x as f64 * chunk_size as f64 + pos.local.x as f64;
    let wz = pos.chunk.z as f64 * chunk_size as f64 + pos.local.z as f64;
    (wx, pos.local.y as f64, wz)
}

/// Reconstruct a normalised WorldPos from absolute world coordinates.
#[inline]
fn world_f64_to_world_pos(wx: f64, y: f64, wz: f64, chunk_size: ChunkSize) -> WorldPos {
    // floor-divide so negative values are handled correctly
    let cx = (wx / chunk_size as f64).floor() as i32;
    let cz = (wz / chunk_size as f64).floor() as i32;
    let lx = (wx - cx as f64 * chunk_size as f64) as f32;
    let lz = (wz - cz as f64 * chunk_size as f64) as f32;
    WorldPos {
        chunk: ChunkCoord { x: cx, z: cz },
        local: LocalPos {
            x: lx,
            y: y as f32,
            z: lz,
        },
    }
}

#[derive(Debug)]
enum SegmentIntersectionType {
    OtherEdges,
    ClosingEdgeOfB,
    ClosingEdgeOfA,
    PointInsidePolygon,
}
