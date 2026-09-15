use crate::helpers::positions::WorldPos;
use crate::world::buildings::zoning::{Lot, LotId, ZoningStorage};
use crate::world::roads::road_mesh_manager::{Edges, RoadEdges, RoadMeshManager};
use crate::world::roads::road_structs::{LaneId, SegmentId};
use crate::world::roads::road_subsystem::Roads;
use glam::{Vec2, Vec3};

pub const LOT_MIN_FRONTAGE: f64 = 5.0;
pub const LOT_PREFERRED_FRONTAGE: f64 = 12.0;
pub const LOT_MAX_FRONTAGE: f64 = 25.0;

pub const LOT_MIN_DEPTH: f64 = 8.0;
pub const LOT_PREFERRED_DEPTH: f64 = 20.0;
pub const LOT_MAX_DEPTH: f64 = 40.0;

pub const LOT_FRONT_OFFSET: f64 = 0.5;
pub const LOT_NEIGHBOR_DISTANCE: f64 = 18.0;
pub const BUILDING_SNAP_RADIUS: f64 = 10.0;
pub const LOT_SIDE_DEPTH_RANGE: f64 = 45.0;

#[derive(Clone)]
pub struct LotFit {
    pub bounds: Vec<WorldPos>,
    pub width: f64,
    pub depth: f64,
    pub snap_point: LotPoint,
}
pub fn collect_road_points(edges: &RoadEdges) -> Vec<&WorldPos> {
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

#[derive(Clone)]
pub struct LotPoint {
    pub pos: WorldPos,
    pub tangent: Vec3,
    pub lateral: Vec3,
    pub left_point: Option<(WorldPos, Vec3, Vec3, f64)>,
    pub right_point: Option<(WorldPos, Vec3, Vec3, f64)>,
    pub dist: f64,
    pub segment_id: SegmentId,
    pub point_type: LotPointType,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SidewalkSide {
    Left,
    Right,
}

#[derive(Clone)]
pub enum LotPointType {
    Sidewalk(SidewalkSide),
    Lane(LaneId),
}

fn lot_local(origin: WorldPos, tangent: Vec3, lateral: Vec3, point: WorldPos) -> Vec2 {
    let tangent = Vec2::new(tangent.x, tangent.z).normalize_or_zero();
    let lateral = Vec2::new(lateral.x, lateral.z).normalize_or_zero();

    origin.delta_xz(point, tangent, lateral)
}

fn polygon_max_x_in_z_range(
    bounds: &[WorldPos],
    origin: WorldPos,
    tangent: Vec3,
    lateral: Vec3,
    z_min: f64,
    z_max: f64,
) -> Option<f64> {
    if bounds.len() < 2 {
        return None;
    }

    let mut result = None;

    for i in 0..bounds.len() {
        let a = lot_local(origin, tangent, lateral, bounds[i]);
        let b = lot_local(origin, tangent, lateral, bounds[(i + 1) % bounds.len()]);

        let consider = |x: f64, result: &mut Option<f64>| {
            *result = Some(result.map_or(x, |v| v.max(x)));
        };

        if a.y >= z_min as f32 && a.y <= z_max as f32 {
            consider(a.x as f64, &mut result);
        }

        if b.y >= z_min as f32 && b.y <= z_max as f32 {
            consider(b.x as f64, &mut result);
        }

        let dz = b.y - a.y;

        if dz.abs() > 1e-8 {
            for z in [z_min, z_max] {
                let t = (z as f32 - a.y) / dz;

                if t >= 0.0 && t <= 1.0 {
                    let x = a.x + (b.x - a.x) * t;
                    consider(x as f64, &mut result);
                }
            }
        }
    }

    result
}

fn polygon_min_x_in_z_range(
    bounds: &[WorldPos],
    origin: WorldPos,
    tangent: Vec3,
    lateral: Vec3,
    z_min: f64,
    z_max: f64,
) -> Option<f64> {
    if bounds.len() < 2 {
        return None;
    }

    let mut result = None;

    for i in 0..bounds.len() {
        let a = lot_local(origin, tangent, lateral, bounds[i]);
        let b = lot_local(origin, tangent, lateral, bounds[(i + 1) % bounds.len()]);

        let consider = |x: f64, result: &mut Option<f64>| {
            *result = Some(result.map_or(x, |v| v.min(x)));
        };

        if a.y >= z_min as f32 && a.y <= z_max as f32 {
            consider(a.x as f64, &mut result);
        }

        if b.y >= z_min as f32 && b.y <= z_max as f32 {
            consider(b.x as f64, &mut result);
        }

        let dz = b.y - a.y;

        if dz.abs() > 1e-8 {
            for z in [z_min, z_max] {
                let t = (z as f32 - a.y) / dz;

                if t >= 0.0 && t <= 1.0 {
                    let x = a.x + (b.x - a.x) * t;
                    consider(x as f64, &mut result);
                }
            }
        }
    }

    result
}

fn polygon_min_z_in_x_range(
    bounds: &[WorldPos],
    origin: WorldPos,
    tangent: Vec3,
    lateral: Vec3,
    x_min: f64,
    x_max: f64,
) -> Option<f64> {
    if bounds.len() < 2 {
        return None;
    }

    let mut result = None;

    for i in 0..bounds.len() {
        let a = lot_local(origin, tangent, lateral, bounds[i]);
        let b = lot_local(origin, tangent, lateral, bounds[(i + 1) % bounds.len()]);

        let consider = |z: f64, result: &mut Option<f64>| {
            *result = Some(result.map_or(z, |v| v.min(z)));
        };

        if a.x as f64 >= x_min && a.x as f64 <= x_max {
            consider(a.y as f64, &mut result);
        }

        if b.x as f64 >= x_min && b.x as f64 <= x_max {
            consider(b.y as f64, &mut result);
        }

        let dx = b.x - a.x;

        if dx.abs() > 1e-8 {
            for x in [x_min, x_max] {
                let t = (x as f32 - a.x) / dx;

                if t >= 0.0 && t <= 1.0 {
                    let z = a.y + (b.y - a.y) * t;
                    consider(z as f64, &mut result);
                }
            }
        }
    }

    result
}

pub fn collect_lot_point(
    edges: &Edges,
    target: &WorldPos,
    closest_distance: &mut f64,
    closest_point: &mut Option<LotPoint>,
    segment_id: SegmentId,
    point_type: LotPointType,
) {
    for (idx, ((point, tangent), lateral)) in edges
        .right_points
        .iter()
        .zip(edges.right_tangents.iter())
        .zip(edges.right_laterals.iter())
        .enumerate()
    {
        let dist = target.distance_to(*point);

        if dist < *closest_distance {
            *closest_distance = dist;

            let left_point = if idx > 0 {
                Some((
                    edges.right_points[idx - 1],
                    edges.right_tangents[idx - 1],
                    edges.right_laterals[idx - 1],
                    target.distance_to(edges.right_points[idx - 1]),
                ))
            } else {
                None
            };

            let right_point = if idx + 1 < edges.right_points.len() {
                Some((
                    edges.right_points[idx + 1],
                    edges.right_tangents[idx + 1],
                    edges.right_laterals[idx + 1],
                    target.distance_to(edges.right_points[idx + 1]),
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
                point_type: point_type.clone(),
            });
        }
    }
}

pub fn gather_closest_lot_point(
    roads: &Roads,
    road_mesh_manager: &RoadMeshManager,
    target: WorldPos,
) -> Option<LotPoint> {
    let mut closest_distance = f64::INFINITY;
    let mut closest_point = None;

    for segment_id in roads
        .road_manager
        .roads
        .segment_ids_touching_chunks(&target.chunk.get_chunks_3x3())
    {
        let Some(road_edges) = road_mesh_manager.road_edge_storage.get(&segment_id) else {
            continue;
        };

        if !road_edges.right_sidewalk_edge.is_empty() {
            collect_lot_point(
                &road_edges.right_sidewalk_edge,
                &target,
                &mut closest_distance,
                &mut closest_point,
                segment_id,
                LotPointType::Sidewalk(SidewalkSide::Right),
            );
        }

        if !road_edges.left_sidewalk_edge.is_empty() {
            collect_lot_point(
                &road_edges.left_sidewalk_edge,
                &target,
                &mut closest_distance,
                &mut closest_point,
                segment_id,
                LotPointType::Sidewalk(SidewalkSide::Left),
            );
        } else {
            for lane_id in roads.road_manager.roads.segment(segment_id).lanes() {
                let Some(lane_edges) = road_edges.lane_edges.get(lane_id) else {
                    continue;
                };

                collect_lot_point(
                    lane_edges,
                    &target,
                    &mut closest_distance,
                    &mut closest_point,
                    segment_id,
                    LotPointType::Lane(lane_id.clone()),
                );
            }
        }
    }

    closest_point
}

pub fn close_polygon(points: &mut Vec<WorldPos>) {
    if points.len() < 3 {
        return;
    }

    let first = points[0];
    let last = points[points.len() - 1];

    if first.distance_to(last) > 1e-6 {
        points.push(first);
    }
}

fn lot_world(origin: WorldPos, tangent: Vec3, lateral: Vec3, local: Vec2) -> WorldPos {
    let tangent = Vec2::new(tangent.x, tangent.z).normalize_or_zero();
    let lateral = Vec2::new(lateral.x, lateral.z).normalize_or_zero();

    origin
        .add_vec3(Vec3::new(tangent.x, 0.0, tangent.y) * local.x)
        .add_vec3(Vec3::new(lateral.x, 0.0, lateral.y) * local.y)
}

fn edges_for_point<'a>(
    road_mesh_manager: &'a RoadMeshManager,
    segment_id: SegmentId,
    point_type: &LotPointType,
) -> Option<&'a Edges> {
    let road_edges = road_mesh_manager.road_edge_storage.get(&segment_id)?;

    match point_type {
        LotPointType::Sidewalk(SidewalkSide::Right) => {
            (!road_edges.right_sidewalk_edge.is_empty()).then(|| &road_edges.right_sidewalk_edge)
        }
        LotPointType::Sidewalk(SidewalkSide::Left) => {
            (!road_edges.left_sidewalk_edge.is_empty()).then(|| &road_edges.left_sidewalk_edge)
        }
        LotPointType::Lane(lane_id) => road_edges.lane_edges.get(lane_id),
    }
}

fn sample_frontage_curve(
    edge: &Edges,
    origin: WorldPos,
    tangent: Vec3,
    lateral: Vec3,
    left_x: f64,
    right_x: f64,
) -> Vec<(WorldPos, Vec3)> {
    let mut locals: Vec<(f64, f64, Vec3)> = edge
        .right_points
        .iter()
        .zip(edge.right_laterals.iter())
        .map(|(point, point_lateral)| {
            let local = lot_local(origin, tangent, lateral, *point);
            (local.x as f64, local.y as f64, *point_lateral)
        })
        .collect();

    if locals.len() < 2 {
        return Vec::new();
    }

    if locals.first().unwrap().0 > locals.last().unwrap().0 {
        locals.reverse();
    }

    let mut curve = Vec::new();

    for window in locals.windows(2) {
        let (xa, za, la) = window[0];
        let (xb, zb, lb) = window[1];

        if xb < left_x || xa > right_x {
            continue;
        }

        let dx = xb - xa;

        if xa < left_x && xb >= left_x {
            let t = if dx.abs() > 1e-8 {
                (left_x - xa) / dx
            } else {
                0.0
            };
            let z = za + (zb - za) * t;
            let lat = la.lerp(lb, t as f32).normalize_or_zero();
            curve.push((
                lot_world(origin, tangent, lateral, Vec2::new(left_x as f32, z as f32)),
                lat,
            ));
        }

        if xa >= left_x && xa <= right_x {
            curve.push((
                lot_world(origin, tangent, lateral, Vec2::new(xa as f32, za as f32)),
                la,
            ));
        }

        if xb >= left_x && xb <= right_x {
            curve.push((
                lot_world(origin, tangent, lateral, Vec2::new(xb as f32, zb as f32)),
                lb,
            ));
        } else if xa <= right_x && xb > right_x {
            let t = if dx.abs() > 1e-8 {
                (right_x - xa) / dx
            } else {
                1.0
            };
            let z = za + (zb - za) * t;
            let lat = la.lerp(lb, t as f32).normalize_or_zero();
            curve.push((
                lot_world(
                    origin,
                    tangent,
                    lateral,
                    Vec2::new(right_x as f32, z as f32),
                ),
                lat,
            ));
        }
    }

    curve.dedup_by(|a, b| a.0.distance_to(b.0) < 1e-4);

    curve
}

pub fn fit_lot_to_neighbors(
    zoning_storage: &ZoningStorage,
    road_mesh_manager: &RoadMeshManager,
    snap_point: &LotPoint,
    ignore_lot_id: Option<LotId>,
) -> Option<LotFit> {
    let origin = snap_point.pos;
    let front_z = LOT_FRONT_OFFSET;

    let mut left_boundary: Option<f64> = None;
    let mut right_boundary: Option<f64> = None;

    let mut nearby_lots = Vec::new();

    for lot in zoning_storage.iter_lots() {
        if Some(lot.id) == ignore_lot_id {
            continue;
        }

        if lot.bounds.len() < 4 {
            continue;
        }

        if origin.polygon_distance_squared(lot.bounds.as_slice())
            > LOT_NEIGHBOR_DISTANCE * LOT_NEIGHBOR_DISTANCE
        {
            continue;
        }

        nearby_lots.push(lot);
    }

    for lot in &nearby_lots {
        let open_bounds = &lot.bounds[..lot.bounds.len() - 1];
        let center = WorldPos::centroid(open_bounds);
        let local_center = lot_local(origin, snap_point.tangent, snap_point.lateral, center);

        if lot.segment_id != Some(snap_point.segment_id) {
            continue;
        }

        if (local_center.y as f64) < -LOT_FRONT_OFFSET {
            continue;
        };

        if (local_center.y as f64) > LOT_SIDE_DEPTH_RANGE {
            continue;
        };

        if local_center.x < 0.0 {
            if let Some(x) = polygon_max_x_in_z_range(
                lot.bounds.as_slice(),
                origin,
                snap_point.tangent,
                snap_point.lateral,
                front_z - 2.0,
                front_z + LOT_MAX_DEPTH,
            ) {
                left_boundary = Some(left_boundary.map_or(x, |v: f64| v.max(x)));
            }
        } else if local_center.x > 0.0 {
            if let Some(x) = polygon_min_x_in_z_range(
                lot.bounds.as_slice(),
                origin,
                snap_point.tangent,
                snap_point.lateral,
                front_z - 2.0,
                front_z + LOT_MAX_DEPTH,
            ) {
                right_boundary = Some(right_boundary.map_or(x, |v: f64| v.min(x)));
            }
        }
    }

    let (left_x, right_x) = match (left_boundary, right_boundary) {
        (Some(left), Some(right)) => {
            let gap = right - left;

            if gap < LOT_MIN_FRONTAGE {
                return None;
            }

            if gap <= LOT_MAX_FRONTAGE {
                (left, right)
            } else {
                let center = (left + right) * 0.5;
                (
                    center - LOT_MAX_FRONTAGE * 0.5,
                    center + LOT_MAX_FRONTAGE * 0.5,
                )
            }
        }

        (Some(left), None) => {
            let width = varied_dimension(
                origin,
                1,
                LOT_PREFERRED_FRONTAGE,
                LOT_MIN_FRONTAGE,
                LOT_MAX_FRONTAGE,
            );
            (left, left + width)
        }

        (None, Some(right)) => {
            let width = varied_dimension(
                origin,
                1,
                LOT_PREFERRED_FRONTAGE,
                LOT_MIN_FRONTAGE,
                LOT_MAX_FRONTAGE,
            );
            (right - width, right)
        }

        (None, None) => {
            let width = varied_dimension(
                origin,
                1,
                LOT_PREFERRED_FRONTAGE,
                LOT_MIN_FRONTAGE,
                LOT_MAX_FRONTAGE,
            );
            (-width * 0.5, width * 0.5)
        }
    };

    let frontage = right_x - left_x;

    if frontage < LOT_MIN_FRONTAGE || frontage > LOT_MAX_FRONTAGE {
        return None;
    }

    let fallback_depth =
        varied_dimension(origin, 2, LOT_PREFERRED_DEPTH, LOT_MIN_DEPTH, LOT_MAX_DEPTH);

    let straight_available = depth_at_x(
        &nearby_lots,
        origin,
        snap_point.tangent,
        snap_point.lateral,
        front_z,
        left_x,
        right_x,
    );

    if let Some(available) = straight_available {
        if available < LOT_MIN_DEPTH {
            return None;
        }
    }

    let depth = straight_available
        .map(|available| available.min(LOT_MAX_DEPTH))
        .unwrap_or(fallback_depth);

    if depth < LOT_MIN_DEPTH || depth > LOT_MAX_DEPTH {
        return None;
    }

    let center_x = (left_x + right_x) * 0.5;

    let edge = edges_for_point(
        road_mesh_manager,
        snap_point.segment_id,
        &snap_point.point_type,
    );

    let curve = edge
        .map(|edge| {
            sample_frontage_curve(
                edge,
                origin,
                snap_point.tangent,
                snap_point.lateral,
                left_x,
                right_x,
            )
        })
        .filter(|curve| curve.len() >= 2);

    let mut bounds = if let Some(curve) = curve {
        let half_band = 0.75;
        let mut invalid = false;

        let front_curve: Vec<(WorldPos, Vec3, f64)> = curve
            .into_iter()
            .map(|(point, lat)| {
                let local_x =
                    lot_local(origin, snap_point.tangent, snap_point.lateral, point).x as f64;

                let point_depth = match depth_at_x(
                    &nearby_lots,
                    origin,
                    snap_point.tangent,
                    snap_point.lateral,
                    front_z,
                    local_x - half_band,
                    local_x + half_band,
                ) {
                    Some(available) => {
                        if available < LOT_MIN_DEPTH {
                            invalid = true;
                        }

                        available.clamp(LOT_MIN_DEPTH, LOT_MAX_DEPTH)
                    }
                    None => depth,
                };

                (
                    point.add_vec3(lat * LOT_FRONT_OFFSET as f32),
                    lat,
                    point_depth,
                )
            })
            .collect();

        if invalid {
            return None;
        }

        let mut points: Vec<WorldPos> = front_curve.iter().map(|(point, _, _)| *point).collect();

        for (point, lat, point_depth) in front_curve.iter().rev() {
            points.push(point.add_vec3(*lat * *point_depth as f32));
        }

        points
    } else {
        let front_center = origin
            .add_vec3(snap_point.tangent * center_x as f32)
            .add_vec3(snap_point.lateral * LOT_FRONT_OFFSET as f32);

        let front_left = front_center.sub_vec3(snap_point.tangent * (frontage * 0.5) as f32);
        let front_right = front_center.add_vec3(snap_point.tangent * (frontage * 0.5) as f32);
        let back_left = front_left.add_vec3(snap_point.lateral * depth as f32);
        let back_right = front_right.add_vec3(snap_point.lateral * depth as f32);

        vec![front_left, front_right, back_right, back_left]
    };

    close_polygon(&mut bounds);

    let entrance_pos = origin
        .add_vec3(snap_point.tangent * center_x as f32)
        .add_vec3(snap_point.lateral * LOT_FRONT_OFFSET as f32);

    let fitted_snap = LotPoint {
        pos: entrance_pos,
        tangent: snap_point.tangent,
        lateral: snap_point.lateral,
        left_point: snap_point.left_point.clone(),
        right_point: snap_point.right_point.clone(),
        dist: snap_point.dist,
        segment_id: snap_point.segment_id,
        point_type: snap_point.point_type.clone(),
    };

    Some(LotFit {
        bounds,
        width: frontage,
        depth,
        snap_point: fitted_snap,
    })
}

fn hash_to_unit(mut x: u64) -> f64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

fn position_seed(pos: WorldPos, salt: u64) -> u64 {
    let qx =
        (pos.chunk.x as i64).wrapping_mul(1_000_003) ^ (pos.local.x as f64 * 8.0).round() as i64;
    let qz =
        (pos.chunk.z as i64).wrapping_mul(1_000_033) ^ (pos.local.z as f64 * 8.0).round() as i64;

    let mut seed = (qx as u64).wrapping_mul(0x9E3779B97F4A7C15);
    seed ^= (qz as u64).wrapping_mul(0xC2B2AE3D27D4EB4F);
    seed ^= salt.wrapping_mul(0x165667B19E3779F9);
    seed
}

fn varied_dimension(pos: WorldPos, salt: u64, preferred: f64, min: f64, max: f64) -> f64 {
    let t = hash_to_unit(position_seed(pos, salt));
    (preferred * (0.7 + 0.6 * t)).clamp(min, max)
}

fn depth_at_x(
    nearby_lots: &[&Lot],
    origin: WorldPos,
    tangent: Vec3,
    lateral: Vec3,
    front_z: f64,
    x_min: f64,
    x_max: f64,
) -> Option<f64> {
    let mut back_z: Option<f64> = None;

    for lot in nearby_lots {
        let open_bounds = &lot.bounds[..lot.bounds.len() - 1];
        let center = WorldPos::centroid(open_bounds);
        let local_center = lot_local(origin, tangent, lateral, center);

        if local_center.y as f64 <= front_z + LOT_MIN_DEPTH {
            continue;
        }

        if let Some(z) = polygon_min_z_in_x_range(
            lot.bounds.as_slice(),
            origin,
            tangent,
            lateral,
            x_min,
            x_max,
        ) {
            if z > front_z {
                back_z = Some(back_z.map_or(z, |v: f64| v.min(z)));
            }
        }
    }

    back_z.map(|z| z - front_z)
}
