use crate::helpers::positions::{ChunkCoord, ChunkSize, LocalPos, WorldPos};
use crate::renderer::gizmo::gizmo::Gizmo;
use crate::resources::Time;
use crate::simulation::Ticker;
use crate::ui::input::Input;
use crate::ui::parser::Value;
use crate::ui::variables::Variables;
use crate::world::camera::Camera;
use crate::world::roads::road_mesh_manager::{Edges, RoadEdgeStorage, RoadEdges, RoadMeshManager};
use crate::world::roads::road_structs::{LaneId, SegmentId};
use crate::world::roads::road_subsystem::Roads;
use crate::world::statisticals::demands::ZoningDemand;
use crate::world::terrain::terrain_subsystem::{CursorMode, PickedPoint, Terrain};
use glam::Vec3;
use rand::RngExt;
use rand::rngs::ThreadRng;
use rayon::iter::IntoParallelRefMutIterator;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::slice::IterMut;

const SNAP_RADIUS: f64 = 10.0;
const EPS: f64 = 0.0001;

#[derive(Serialize, Deserialize, Clone)]
pub enum DistrictType {
    AutomaticallyMade,
    PlayerMade,
}
#[derive(Serialize, Deserialize, Clone)]
pub struct District {
    pub id: DistrictId,
    pub name: String,
    pub district_type: DistrictType,
    pub center: WorldPos,
    points: Vec<WorldPos>,
    pub lot_ids: Vec<LotId>,
    pub zoning_demand: ZoningDemand,
}
impl District {
    pub fn new(
        name: String,
        points: Vec<WorldPos>,
        district_type: DistrictType,
        chunk_size: ChunkSize,
    ) -> District {
        let center = WorldPos::centroid(&points, chunk_size);
        Self {
            id: 3346243577,
            name,
            district_type,
            center,
            points,
            lot_ids: vec![],
            zoning_demand: ZoningDemand::new(),
        }
    }
    #[inline]
    pub fn add_point(&mut self, point: WorldPos, chunk_size: ChunkSize) {
        self.points.push(point);
        self.center = WorldPos::centroid(&self.points, chunk_size);
    }
    pub fn update(
        time: &Time,
        chunk_size: ChunkSize,
        zoning_storage: &mut ZoningStorage,
        district_id: DistrictId,
        road_edge_storage: &RoadEdgeStorage,
    ) -> Option<District> {
        let Some(district) = zoning_storage.get_mut_district(district_id) else {
            return None;
        };
        district.zoning_demand.update_demands(time);
        if !matches!(district.district_type, DistrictType::PlayerMade)
            && district.should_split(chunk_size)
        {
            return Self::split(zoning_storage, district_id, road_edge_storage, chunk_size);
        }
        None
    }
    fn should_split(&self, chunk_size: ChunkSize) -> bool {
        if self.points.len() < 3 {
            return false;
        }
        if self.lot_ids.len() > 500 {
            return true;
        }
        let area = WorldPos::area(self.points.as_slice(), chunk_size);
        //println!("Area: {}", area);
        if area > 500_000.0 {
            return true;
        }
        false
    }

    /// Split the district along the road geometry of one of its boundary segments.
    ///
    /// Strategy:
    ///   1. Among all road segments referenced by our lots, find the one whose
    ///      right-sidewalk edge actually bisects the district (crosses the boundary
    ///      ≥ 2 times) and whose midpoint is closest to our centroid.
    ///   2. Find the first and last boundary crossings of that edge polyline.
    ///   3. Cut the district polygon at those two crossing points, keeping the
    ///      road-edge vertices in between as the seam.
    ///   4. Assign each lot to whichever half its centroid falls in.
    ///   5. Mutate `self` to be the first half; return the second half.
    ///
    /// Everything stays in WorldPos — dx/dz for geometry, delta_to for offsets.
    pub fn split(
        zoning_storage: &mut ZoningStorage,
        district_id: DistrictId,
        road_edge_storage: &RoadEdgeStorage,
        chunk_size: ChunkSize,
    ) -> Option<District> {
        let district = zoning_storage.get_district(district_id)?;
        if district.points.len() < 6 {
            return None;
        }

        // ── 1. Find the best bisecting road edge ───────────────────────────
        let centroid = WorldPos::centroid(&district.points, chunk_size);

        let mut best_line: Option<Vec<WorldPos>> = None;
        let mut best_dist = f64::MAX;
        let mut seen_segments = std::collections::HashSet::new();

        for &lot_id in &district.lot_ids {
            let Some(lot) = zoning_storage.get_lot(lot_id) else {
                continue;
            };
            if !seen_segments.insert(lot.segment_id) {
                continue;
            }
            let Some(road_edges) = road_edge_storage.get(&lot.segment_id) else {
                continue;
            };

            // "Right is the outside" — use the outer sidewalk edge as the cut line.
            // It runs parallel to (and outside) the road, making it a clean seam.
            let candidate = &road_edges.right_sidewalk_edge.right_points;
            if candidate.len() < 2 {
                continue;
            }

            // Require the candidate to actually cross our boundary at least twice,
            // otherwise it doesn't bisect us and is useless as a split line.
            if boundary_crossing_count(candidate, &district.points, chunk_size) < 2 {
                continue;
            }

            // Among valid candidates, prefer the one closest to our centroid.
            let mid = candidate[candidate.len() / 2];
            let dist = centroid.distance_to(mid, chunk_size);
            if dist < best_dist {
                best_dist = dist;
                best_line = Some(candidate.clone());
            }
        }

        let split_line = best_line?;

        // ── 2. Collect all crossings of split_line with our boundary ────────
        let boundary = &district.points;
        let n = boundary.len();

        struct Crossing {
            boundary_edge: usize, // which edge of the district polygon
            t_boundary: f32,      // parameter [0,1] on that boundary edge
            split_pos: f32,       // si + t_split: ordering key along split_line
            point: WorldPos,      // the exact crossing point
        }

        let mut crossings: Vec<Crossing> = Vec::new();

        for si in 0..split_line.len() - 1 {
            let s0 = split_line[si];
            let s1 = split_line[si + 1];
            for bi in 0..n {
                let b0 = boundary[bi];
                let b1 = boundary[(bi + 1) % n];
                if let Some((t_b, t_s)) = segment_xz_intersect(b0, b1, s0, s1, chunk_size) {
                    crossings.push(Crossing {
                        boundary_edge: bi,
                        t_boundary: t_b,
                        split_pos: si as f32 + t_s,
                        point: lerp_on_segment(b0, b1, t_b, chunk_size),
                    });
                }
            }
        }

        if crossings.len() < 2 {
            return None;
        }

        // Order by position along the split_line so entry comes before exit.
        crossings.sort_by(|a, b| {
            a.split_pos
                .partial_cmp(&b.split_pos)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let entry = &crossings[0];
        let exit = &crossings[crossings.len() - 1];

        // ── 3. Build the seam (road-edge vertices between the two crossings) ─
        //
        //  cut = [entry.point, split_line[⌈entry.split_pos⌉ .. ⌊exit.split_pos⌋], exit.point]
        //
        let mut cut: Vec<WorldPos> = Vec::new();
        cut.push(entry.point);
        let first_vi = entry.split_pos.ceil() as usize;
        let last_vi = exit.split_pos.floor() as usize;
        for vi in first_vi..=last_vi {
            if vi < split_line.len() {
                cut.push(split_line[vi]);
            }
        }
        cut.push(exit.point);

        let entry_bi = entry.boundary_edge;
        let exit_bi = exit.boundary_edge;

        // ── 4. Build the two sub-polygons ────────────────────────────────────
        //
        //  poly_a  entry.point
        //          → boundary vertices going FORWARD from entry_bi+1 to exit_bi
        //          → exit.point
        //          → cut interior REVERSED back to entry.point      (seam)
        //
        //  poly_b  exit.point
        //          → boundary vertices going FORWARD from exit_bi+1 to entry_bi
        //          → entry.point
        //          → cut interior FORWARD back to exit.point        (seam)
        //
        let build_half = |start_bi: usize, end_bi: usize, start_pt: WorldPos, end_pt: WorldPos| {
            let mut poly = Vec::<WorldPos>::new();
            poly.push(start_pt);
            let mut i = (start_bi + 1) % n;
            let stop = (end_bi + 1) % n;
            let mut steps = 0;
            while i != stop && steps < n {
                poly.push(boundary[i]);
                i = (i + 1) % n;
                steps += 1;
            }
            poly.push(end_pt);
            poly
        };

        let mut poly_a = build_half(entry_bi, exit_bi, entry.point, exit.point);
        // Seal poly_a: walk the cut seam backwards (skipping duplicated endpoints)
        if cut.len() > 2 {
            for p in cut[1..cut.len() - 1].iter().rev() {
                poly_a.push(*p);
            }
        }

        let mut poly_b = build_half(exit_bi, entry_bi, exit.point, entry.point);
        // Seal poly_b: walk the cut seam forwards
        if cut.len() > 2 {
            for p in cut[1..cut.len() - 1].iter() {
                poly_b.push(*p);
            }
        }

        if poly_a.len() < 3 || poly_b.len() < 3 {
            return None;
        }

        // ── 5. Assign lots: centroid-in-polygon test ─────────────────────────
        let mut lots_a: Vec<LotId> = Vec::new();
        let mut lots_b: Vec<LotId> = Vec::new();

        for &lot_id in &district.lot_ids {
            let Some(lot) = zoning_storage.get_lot(lot_id) else {
                lots_b.push(lot_id); // keep unresolved lots in the new district
                continue;
            };
            let lot_centroid = WorldPos::centroid(&lot.bounds, chunk_size);
            if point_in_polygon_xz(lot_centroid, &poly_a, chunk_size) {
                lots_a.push(lot_id);
            } else {
                lots_b.push(lot_id);
            }
        }

        // build and return the new district
        let district = zoning_storage.get_mut_district(district_id)?;
        district.points = poly_a;
        district.lot_ids = lots_a;
        district.district_type = DistrictType::AutomaticallyMade;

        let mut rng = ThreadRng::default();
        let name = generate_district_name(&mut rng);
        let mut new_district =
            District::new(name, poly_b, DistrictType::AutomaticallyMade, chunk_size);
        new_district.lot_ids = lots_b;
        new_district.zoning_demand = ZoningDemand::new();

        Some(new_district)
    }
}

#[derive(Clone, Default)]
struct ZoningState {
    pub district_id: DistrictId,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum ZoningType {
    None,
    Residential,
    Commercial,
    Industrial,
    Office,
}
impl ZoningType {
    pub fn from_value(value: &Value) -> Self {
        match value {
            Value::String(s) => match s.to_lowercase().as_str() {
                "none" => ZoningType::None,
                "residential" => ZoningType::Residential,
                "commercial" => ZoningType::Commercial,
                "industrial" => ZoningType::Industrial,
                "office" => ZoningType::Office,
                _ => ZoningType::None,
            },
            _ => ZoningType::None,
        }
    }
}
impl Display for ZoningType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ZoningType::None => write!(f, "none"),
            ZoningType::Residential => write!(f, "residential"),
            ZoningType::Commercial => write!(f, "commercial"),
            ZoningType::Industrial => write!(f, "industrial"),
            ZoningType::Office => write!(f, "office"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum PlacePos {
    Free(WorldPos, f64),
    RoadSnap(WorldPos, f64),
    CurrentDistrictFirstPoint(WorldPos, f64),
    CurrentDistrictPoint(WorldPos, f64),
    CurrentDistrictLastPoint(WorldPos, f64),
    OtherDistrictPoint(WorldPos, f64),
}

impl PlacePos {
    fn pos(self) -> WorldPos {
        match self {
            PlacePos::RoadSnap(pos, _)
            | PlacePos::Free(pos, _)
            | PlacePos::CurrentDistrictFirstPoint(pos, _)
            | PlacePos::CurrentDistrictPoint(pos, _)
            | PlacePos::CurrentDistrictLastPoint(pos, _)
            | PlacePos::OtherDistrictPoint(pos, _) => pos,
        }
    }

    fn dist(self) -> f64 {
        match self {
            PlacePos::Free(_, dist) => dist,
            PlacePos::RoadSnap(_, dist)
            | PlacePos::CurrentDistrictFirstPoint(_, dist)
            | PlacePos::CurrentDistrictPoint(_, dist)
            | PlacePos::CurrentDistrictLastPoint(_, dist)
            | PlacePos::OtherDistrictPoint(_, dist) => dist,
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
    pub ticker: Ticker,
    zoning_state: Option<ZoningState>,
    pub zoning_storage: ZoningStorage,
}

impl Zoning {
    pub fn new() -> Self {
        Self {
            ticker: Ticker::new(0.01),
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
        let new_district_type = &terrain.cursor.zoning_type;
        let active_district_id = self.zoning_state.as_ref().map(|state| state.district_id);
        for district in self.zoning_storage.iter_districts() {
            if Some(district.id) != active_district_id {
                draw_district_area(
                    district.points.as_slice(),
                    &ZoningType::None,
                    variables,
                    gizmo,
                    None,
                    None,
                );
            }

            gizmo.polyline(
                district.points.as_slice(),
                [0.07, 0.28, 0.5, 1.0],
                0.0,
                0.2,
                0.0,
            );

            gizmo.text(
                district.name.clone(),
                district.center,
                3.0,
                [0.05, 0.05, 0.05, 1.0],
                None,
                0.0,
                0.0,
            )

            // This was to test if the lateral is ACTUALLY right! It was perfect!
            // for idx in 0..district.points.len() {
            //     let (_, right) = tangent_and_lateral_right(&*district.points, idx, camera.chunk_size);
            //     gizmo.direction(district.points[idx], right, [1.0, 0.0, 0.0, 1.0], 0.0, 0.0);
            // }
        }
        // After this, if the mouse is over the sky or UI, stuff doesn't get rendered cuz there is no picked point ofc!
        let Some(picked) = terrain.last_picked.as_ref() else {
            return;
        };
        let chunk_size = terrain.chunk_size;

        let mut closest_point: Option<LotPoint> = None;
        let mut closest_distance = 100.0;

        for segment_id in roads
            .road_manager
            .roads
            .segment_ids_touching_chunk(picked.chunk.coords.chunk_coord, terrain.chunk_size)
        {
            let Some(road_edges) = road_mesh_manager.road_edge_storage.get(&segment_id) else {
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
                if let Some(road_edges) = road_mesh_manager
                    .road_edge_storage
                    .get(&snap_point.segment_id)
                {
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
                    new_district_type,
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
                    active_district_id,
                    new_district_type,
                    gizmo,
                );
            }
            _ => {}
        }
    }
    pub fn update_districts(
        &mut self,
        time: &Time,
        chunk_size: ChunkSize,
        road_edge_storage: &RoadEdgeStorage,
    ) {
        if !self.ticker.tick(time.target_sim_dt) {
            return;
        }
        println!("updating district");
        for district_id in self.zoning_storage.district_ids() {
            District::update(
                time,
                chunk_size,
                &mut self.zoning_storage,
                district_id,
                road_edge_storage,
            );
        }
    }
    fn can_place_zoning_point(
        &self,
        gizmo: &mut Gizmo,
        place_pos: PlacePos,
        active_district_id: Option<DistrictId>,
        chunk_size: ChunkSize,
    ) -> bool {
        let Some(district_id) = active_district_id else {
            return true;
        };
        let Some(district) = self.zoning_storage.get_district(district_id) else {
            return false;
        };

        let Some(&last) = district.points.last() else {
            return false;
        };
        gizmo.line(last, place_pos.pos(), [0.0, 1.0, 0.0, 1.0], 0.5, 0.0);
        if self.segment_intersects_any_district(
            place_pos.pos(),
            last,
            active_district_id,
            chunk_size,
        ) {
            return false;
        };
        //println!("{:?}", place_pos);
        match place_pos {
            PlacePos::Free(_, _) => true,
            PlacePos::RoadSnap(_, _) => true,
            PlacePos::CurrentDistrictFirstPoint(_, _) => {
                let point_amount_requirement = district.points.len() >= 3;

                let enclosing_another_area = self
                    .zoning_storage
                    .iter_districts()
                    .filter(|other_district| other_district.id != district.id)
                    .any(|other_district| {
                        other_district.points.iter().any(|point| {
                            point_in_polygon_xz(*point, &district.points, gizmo.chunk_size)
                        })
                    });

                point_amount_requirement && !enclosing_another_area
            }
            PlacePos::CurrentDistrictPoint(_, _) => false,
            PlacePos::CurrentDistrictLastPoint(_, _) => false,
            PlacePos::OtherDistrictPoint(_, _) => true,
        }
    }

    fn segment_intersects_any_district(
        &self,
        a: WorldPos,
        b: WorldPos,
        active_district_id: Option<DistrictId>,
        chunk_size: ChunkSize,
    ) -> bool {
        const CLEARANCE: f64 = 0.0; // 1 meter clearance

        for district in self.zoning_storage.iter_districts() {
            let points = &district.points;
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

            if Some(district.id) != active_district_id && points.len() >= 3 {
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
            let Some(edges) = road_mesh_manager.road_edge_storage.get(segment_id) else {
                continue;
            };
            let road_points = collect_road_points(edges);
            for point in road_points {
                if point_in_polygon_xz(*point, lot_points, chunk_size) {
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
        new_zoning_type: &ZoningType,
        gizmo: &mut Gizmo,
    ) {
        let lot_width = variables.get_f64("lot_width").unwrap_or(15.0) as f32;
        let lot_length = variables.get_f64("lot_length").unwrap_or(20.0) as f32;
        let chunk_size = terrain.chunk_size;
        let mut inside_lot_id: Option<LotId> = None;
        for lot in self.zoning_storage.iter_lots() {
            if point_in_polygon_xz(picked.pos, lot.bounds.as_slice(), chunk_size) {
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
                            id: 6945220,
                            bounds: preview,
                            zoning_type: new_zoning_type.clone(),
                            segment_id: snap_point.segment_id,
                            district_id: 6378186,
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
                    draw_district_area(
                        lot.bounds.as_slice(),
                        &lot.zoning_type,
                        variables,
                        gizmo,
                        Some([3.0, 0.2, 0.2, 1.0]),
                        Some(new_zoning_type),
                    );
                } else {
                    draw_district_area(
                        lot.bounds.as_slice(),
                        &lot.zoning_type,
                        variables,
                        gizmo,
                        Some([1.2, 1.2, 1.2, 1.0]),
                        Some(new_zoning_type),
                    );
                    if input.action_pressed_once("Place Zoning Point") {
                        lot.zoning_type = *new_zoning_type;
                    }
                }
            } else {
                draw_district_area(
                    lot.bounds.as_slice(),
                    &lot.zoning_type,
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
        active_district_id: Option<DistrictId>,
        new_district_type: &ZoningType,
        gizmo: &mut Gizmo,
    ) {
        let chunk_size = terrain.chunk_size;
        let mut best_place: PlacePos = PlacePos::Free(picked.pos, 0.0);

        let mut consider = |candidate: PlacePos| {
            use PlacePos::*;

            let priority = |p: &PlacePos| match p {
                CurrentDistrictFirstPoint(_, _) => 5,
                CurrentDistrictLastPoint(_, _) => 4,
                CurrentDistrictPoint(_, _) => 3,
                OtherDistrictPoint(_, _) => 2,
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

        if let Some(district_id) = active_district_id {
            if let Some(district) = self.zoning_storage.get_district(district_id) {
                let len = district.points.len();

                for (i, point) in district.points.iter().copied().enumerate() {
                    let dist = point.distance_to(picked.pos, terrain.chunk_size);

                    if dist > SNAP_RADIUS {
                        continue;
                    }

                    let candidate = if i == 0 {
                        PlacePos::CurrentDistrictFirstPoint(point, dist)
                    } else if i + 1 == len {
                        PlacePos::CurrentDistrictLastPoint(point, dist)
                    } else {
                        PlacePos::CurrentDistrictPoint(point, dist)
                    };
                    //println!("Considering: {:?}", candidate);
                    consider(candidate);
                }
            }
        }

        for district in self.zoning_storage.iter_districts() {
            if Some(district.id) == active_district_id {
                continue;
            }

            for point in district.points.iter().copied() {
                let dist = point.distance_to(picked.pos, terrain.chunk_size);

                if dist <= SNAP_RADIUS {
                    consider(PlacePos::OtherDistrictPoint(point, dist));
                }
            }
        }

        let can_place =
            self.can_place_zoning_point(gizmo, best_place, active_district_id, terrain.chunk_size);

        let preview_color = if can_place {
            [0.1, 0.3, 0.7, 1.0]
        } else {
            [0.7, 0.1, 0.1, 1.0]
        };
        let mut inside_district_id: Option<DistrictId> = None;
        let inside_lot_id: Option<LotId> = None;
        if can_place && matches!(best_place, PlacePos::Free(_, _)) {
            for district in self.zoning_storage.iter_districts() {
                if Some(district.id) == active_district_id {
                    continue;
                }
                if point_in_polygon_xz(best_place.pos(), &district.points, gizmo.chunk_size) {
                    // best_pos is inside this district
                    inside_district_id.replace(district.id);
                    // for lot in district
                    //     .lots
                    //     .iter()
                    //     .map(|lot_id| self.zoning_storage.get_lot(*lot_id))
                    // {
                    //     let Some(lot) = lot else { continue };
                    //     if point_inside_polygon(
                    //         best_place.pos(),
                    //         &lot.bounds,
                    //         gizmo.chunk_size,
                    //         false,
                    //     ) {
                    //         inside_lot_id.replace(lot.id);
                    //     }
                    // }
                }
            }
        }

        let canceling = input.action_pressed_once("Cancel");
        if canceling {
            if let Some(zoning_state) = &self.zoning_state {
                let district_id = zoning_state.district_id;
                if let Some(district) = self.zoning_storage.get_mut_district(district_id) {
                    district.points.pop();
                    if district.points.is_empty() {
                        self.zoning_storage.despawn_district(district_id);
                        self.zoning_state = None;
                    }
                }
            }
        }
        if let Some(id) = inside_district_id {
            if input.action_pressed_once("Place Zoning Point") {
                if let Some(district) = self.zoning_storage.get_mut_district(id) {
                    //district.district_type = *new_district_type; // Update zoning type of district, but district don't actually have a zoning type so whatever...
                }
            }
            if let Some(district) = self.zoning_storage.get_district(id) {
                draw_district_area(
                    district.points.as_slice(),
                    &ZoningType::None,
                    variables,
                    gizmo,
                    Some([1.1, 1.1, 1.1, 1.0]),
                    Some(new_district_type),
                );
                // for lot_id in &district.lots {
                //     let Some(lot) = self.zoning_storage.get_lot(*lot_id) else {
                //         continue;
                //     };
                //
                //     if Some(*lot_id) == inside_lot_id {
                //         // mouse inside lot, highlighted
                //         draw_district_area(
                //             lot.bounds.as_slice(),
                //             &lot.district_type,
                //             variables,
                //             gizmo,
                //             Some([1.1, 1.1, 1.1, 1.0]),
                //             Some(new_district_type),
                //         );
                //         gizmo.polyline(
                //             lot.bounds.as_slice(),
                //             [0.07, 0.48, 0.5, 1.0],
                //             0.0,
                //             0.15,
                //             0.0,
                //         );
                //     } else {
                //         // Normal lot drawing, not highlighted
                //         draw_district_area(
                //             lot.bounds.as_slice(),
                //             &lot.district_type,
                //             variables,
                //             gizmo,
                //             Some([1.0, 1.0, 1.0, 1.0]),
                //             Some(new_district_type),
                //         );
                //         gizmo.polyline(
                //             lot.bounds.as_slice(),
                //             [0.07, 0.48, 0.5, 1.0],
                //             0.0,
                //             0.1,
                //             0.0,
                //         );
                //     }
                // }
            }
        } else {
            gizmo.circle(best_place.pos(), 0.5, preview_color, 0.0, 0.0);

            if let Some(last_pos) = active_district_id
                .and_then(|district_id| self.zoning_storage.get_district(district_id))
                .and_then(|district| district.points.last().copied())
            {
                gizmo.line(best_place.pos(), last_pos, preview_color, 0.0, 0.0);
            }

            if input.action_pressed_once("Place Zoning Point") && can_place {
                if let Some(zoning_state) = &self.zoning_state {
                    let district_id = zoning_state.district_id;
                    match best_place {
                        PlacePos::Free(pos, _) => {
                            if let Some(district) =
                                self.zoning_storage.get_mut_district(district_id)
                            {
                                district.add_point(pos, chunk_size);
                            }
                        }
                        PlacePos::CurrentDistrictFirstPoint(pos, _) => {
                            if let Some(district) =
                                self.zoning_storage.get_mut_district(district_id)
                            {
                                if district.points.len() >= 3 {
                                    district.add_point(pos, chunk_size);
                                    println!("Closed zoning loop");
                                    self.zoning_state = None;
                                }
                            }
                            if self.zoning_state.is_none() {
                                //self.zoning_storage.calculate_lots_for_district(district_id);
                                // ↑ Doesn't exist, is bullshit anyway now, because Districts just have LotIds which reference Lots which are the actual lots, pretty independent and that's good.
                            }
                        }

                        PlacePos::RoadSnap(pos, _) => {
                            if let Some(district) =
                                self.zoning_storage.get_mut_district(district_id)
                            {
                                district.add_point(pos, chunk_size);
                            }
                        }

                        PlacePos::CurrentDistrictPoint(_, _) => {}
                        PlacePos::CurrentDistrictLastPoint(_, _) => {}
                        PlacePos::OtherDistrictPoint(pos, _) => {
                            if let Some(district) =
                                self.zoning_storage.get_mut_district(district_id)
                            {
                                district.add_point(pos, chunk_size);
                            }
                        }
                    }
                } else {
                    let mut rng = ThreadRng::default();
                    let name = generate_district_name(&mut rng);
                    let district_id = self.zoning_storage.spawn_district(District::new(
                        name,
                        vec![best_place.pos()],
                        DistrictType::PlayerMade,
                        chunk_size,
                    ));
                    self.zoning_state = Some(ZoningState { district_id });
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
        //println!("{}", dist);
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

pub type DistrictId = u32;
pub type LotId = u32;
#[derive(Serialize, Deserialize, Clone)]
pub struct Lot {
    pub id: LotId,
    pub bounds: Vec<WorldPos>,
    pub zoning_type: ZoningType,
    pub segment_id: SegmentId,
    pub district_id: DistrictId,
}
#[derive(Serialize, Deserialize, Default, Clone)]
pub struct ZoningStorage {
    districts: Vec<Option<District>>,
    lots: Vec<Option<Lot>>,
    district_free_list: Vec<DistrictId>,
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
    pub fn district_ids(&self) -> Vec<DistrictId> {
        self.districts
            .iter()
            .flatten()
            .map(|d| d.id)
            .collect::<Vec<DistrictId>>()
    }
    pub fn iter_districts(&self) -> impl Iterator<Item = &District> {
        self.districts.iter().filter_map(|z| z.as_ref())
    }
    pub fn iter_lots(&self) -> impl Iterator<Item = &Lot> {
        self.lots.iter().filter_map(|l| l.as_ref())
    }
    pub fn iter_mut_lots(&mut self) -> impl Iterator<Item = &mut Lot> {
        self.lots.iter_mut().filter_map(|l| l.as_mut())
    }
    pub fn iter_mut_districts(&mut self) -> IterMut<'_, Option<District>> {
        self.districts.iter_mut()
    }
    /// Returns a parallel mutable iterator over building slots.
    /// Each slot is independent, so this is safe for rayon.
    pub fn par_iter_mut_districts(&mut self) -> rayon::slice::IterMut<'_, Option<District>> {
        self.districts.par_iter_mut()
    }
    pub fn new() -> Self {
        let mut districts: Vec<Option<District>> = Vec::new();
        districts.push(None); // reserve index 0 — for no reason
        let mut lots: Vec<Option<Lot>> = Vec::new();
        lots.push(None); // reserve index 0 — for no reason
        Self {
            districts: districts,
            lots,
            district_free_list: Vec::new(),
            lot_free_list: Vec::new(),
            chunk_size: 128,
            center_chunk: ChunkCoord::zero(),
        }
    }

    pub fn spawn_district(&mut self, mut district: District) -> DistrictId {
        let district_id = if let Some(reused_id) = self.district_free_list.pop() {
            // Reuse slot - III know it's None because it's in free_list
            district.id = reused_id;
            self.districts[reused_id as usize] = Some(district);
            reused_id
        } else {
            let new_id = self.districts.len() as u32;
            district.id = new_id;
            self.districts.push(Some(district));
            new_id
        };

        district_id
    }

    pub fn despawn_district(&mut self, id: DistrictId) {
        if id == 0 {
            // index 0 is reserved, ignore attempts to despawn it
            return;
        }
        let mut lots_to_despawn = Vec::new();
        if self
            .districts
            .get(id as usize)
            .and_then(|opt| opt.as_ref())
            .is_some()
        {
            if let Some(district) = self.districts.get(id as usize) {
                if let Some(district) = district {
                    lots_to_despawn = district.lot_ids.clone();
                }
            }
            // Actually free the slot
            self.districts[id as usize] = None;
            self.district_free_list.push(id);
        }
        for lot_id in lots_to_despawn {
            self.despawn_lot(lot_id);
        }
    }

    pub fn district_count(&self) -> usize {
        self.districts.len() - self.district_free_list.len() - 1
    }

    #[inline]
    pub fn get_district(&self, id: DistrictId) -> Option<&District> {
        self.districts.get(id as usize)?.as_ref()
    }

    #[inline]
    pub fn get_mut_district(&mut self, id: DistrictId) -> Option<&mut District> {
        self.districts.get_mut(id as usize)?.as_mut()
    }

    pub fn spawn_lot(&mut self, mut lot: Lot) -> LotId {
        let lot_id = if let Some(reused_id) = self.lot_free_list.pop() {
            lot.id = reused_id;
            self.lots[reused_id as usize] = Some(lot);
            reused_id
        } else {
            let new_id = self.lots.len() as u32;
            lot.id = new_id;
            self.lots.push(Some(lot));
            new_id
        };

        // Clone bounds out — breaks the borrow conflict with iter_districts()
        let bounds: Vec<WorldPos> = self.lots[lot_id as usize].as_ref().unwrap().bounds.clone();

        let mut best: Option<(DistrictId, u32)> = None;

        for district in self.iter_districts() {
            let count = bounds
                .iter()
                .filter(|&&pt| point_in_polygon_xz(pt, &district.points, self.chunk_size))
                .count() as u32;

            if count > 0 && best.map_or(true, |(_, best_count)| count > best_count) {
                best = Some((district.id, count));
            }
        }

        if let Some((district_id, _)) = best {
            if let Some(lot) = self.lots[lot_id as usize].as_mut() {
                lot.district_id = district_id;
            }
            if let Some(district) = self.get_mut_district(district_id) {
                district.lot_ids.push(lot_id);
            }
        } else {
            let mut rng = ThreadRng::default();
            let name = generate_district_name(&mut rng);
            let mut district = District::new(
                name,
                bounds,
                DistrictType::AutomaticallyMade,
                self.chunk_size,
            );
            district.lot_ids.push(lot_id);
            let district_id = self.spawn_district(district);
            if let Some(lot) = self.lots[lot_id as usize].as_mut() {
                lot.district_id = district_id;
            }
        }

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
            if let Some(lot) = self.lots.get(id as usize) {
                if let Some(lot) = lot {
                    if let Some(district) = self.districts.get_mut(lot.district_id as usize) {
                        if let Some(district) = district {
                            district.lot_ids.retain(|id| id != &lot.id);
                        }
                    }
                }
            }
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

fn draw_district_area(
    points: &[WorldPos],
    district_type: &ZoningType,
    variables: &Variables,
    gizmo: &mut Gizmo,
    color_multiplier: Option<[f32; 4]>,
    predicted_district_type: Option<&ZoningType>,
) {
    let district_type = predicted_district_type.unwrap_or(district_type);

    let key = match district_type {
        ZoningType::None => "none_zone_color",
        ZoningType::Residential => "residential_zone_color",
        ZoningType::Commercial => "commercial_zone_color",
        ZoningType::Industrial => "industrial_zone_color",
        ZoningType::Office => "office_zone_color",
    };

    if let Some(mut c) = variables.get(key).unwrap_or(&Value::Null).as_color4() {
        if let Some(m) = color_multiplier {
            for i in 0..4 {
                c[i] *= m[i];
            }
        }
        gizmo.area(points, c, 0.0);
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

/// Parametric XZ intersection of segment a→b with segment c→d.
/// Returns (t along a→b, u along c→d), both ∈ [0, 1].
/// Uses only WorldPos::dx / dz — no raw world coordinates.
fn segment_xz_intersect(
    a: WorldPos,
    b: WorldPos,
    c: WorldPos,
    d: WorldPos,
    chunk_size: ChunkSize,
) -> Option<(f32, f32)> {
    let r_x = a.dx(b, chunk_size);
    let r_z = a.dz(b, chunk_size);
    let s_x = c.dx(d, chunk_size);
    let s_z = c.dz(d, chunk_size);

    // 2-D cross product r × s
    let denom = r_x * s_z - r_z * s_x;
    if denom.abs() < 1e-5 {
        return None; // parallel / collinear
    }

    let ac_x = a.dx(c, chunk_size);
    let ac_z = a.dz(c, chunk_size);

    // Standard parametric intersection formulas
    let t = (ac_x * s_z - ac_z * s_x) / denom;
    let u = (ac_x * r_z - ac_z * r_x) / denom;

    if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
        Some((t, u))
    } else {
        None
    }
}

/// Point at parameter `t` ∈ [0, 1] along segment a→b.
#[inline]
fn lerp_on_segment(a: WorldPos, b: WorldPos, t: f32, chunk_size: ChunkSize) -> WorldPos {
    a.add_vec3(a.delta_to(b, chunk_size) * t, chunk_size)
}

/// Ray-casting point-in-polygon test, XZ plane only.
/// Ray direction is +X from `point`; uses only dx/dz offsets.
/// WorldPos Native!!!
pub fn point_in_polygon_xz(point: WorldPos, polygon: &[WorldPos], chunk_size: ChunkSize) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }

    let mut inside = false;
    let mut j = n - 1;

    for i in 0..n {
        // Vertex positions relative to `point` — dx/dz keep everything WorldPos-native
        let xi = point.dx(polygon[i], chunk_size);
        let zi = point.dz(polygon[i], chunk_size);
        let xj = point.dx(polygon[j], chunk_size);
        let zj = point.dz(polygon[j], chunk_size);

        // Does edge j→i cross z = 0 to the right of the origin?
        if (zi > 0.0) != (zj > 0.0) {
            let x_cross = xj + (xi - xj) * (-zj) / (zi - zj);
            if x_cross > 0.0 {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// How many times does a polyline cross a closed polygon boundary?
/// Used to score candidate split lines before committing.
fn boundary_crossing_count(
    polyline: &[WorldPos],
    boundary: &[WorldPos],
    chunk_size: ChunkSize,
) -> usize {
    let n = boundary.len();
    let mut count = 0;
    for si in 0..polyline.len().saturating_sub(1) {
        for bi in 0..n {
            if segment_xz_intersect(
                boundary[bi],
                boundary[(bi + 1) % n],
                polyline[si],
                polyline[si + 1],
                chunk_size,
            )
            .is_some()
            {
                count += 1;
            }
        }
    }
    count
}

fn generate_district_name(rng: &mut impl rand::Rng) -> String {
    let prefixes = [
        "North", "South", "East", "West", "New", "Old", "Upper", "Lower",
    ];
    let cores = [
        "Oak", "River", "Stone", "Linden", "Brook", "Hill", "Maple", "Iron",
    ];
    let suffixes = ["District", "Heights", "Quarter", "Park", "Gardens", "Zone"];

    let use_prefix = rng.random_bool(0.4);
    let prefix = if use_prefix {
        Some(prefixes[rng.random_range(0..prefixes.len())])
    } else {
        None
    };

    let core = cores[rng.random_range(0..cores.len())];
    let suffix = suffixes[rng.random_range(0..suffixes.len())];

    match prefix {
        Some(p) => format!("{} {} {}", p, core, suffix),
        None => format!("{} {}", core, suffix),
    }
}
