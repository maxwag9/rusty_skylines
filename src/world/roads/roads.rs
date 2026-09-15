#![allow(dead_code, unused_variables)]
//! roads.rs - Lane-first, 3D, topology-only road system for chunked citybuilder
//!
//! This module provides the canonical road topology and command API.
//! All operations are deterministic and suitable for simulation replay.
//!
//! # Invariants
//! - IDs are monotonically increasing and never reused
//! - Topology objects are deleted, not deleted, for undo/redo support
//! - Lanes are first-class graph edges; segments are grouping/metadata
//! - Every node is an intersection with attachable traffic controls
//! - Mutable operations must occur outside simulation ticks

use crate::data::Settings;
use crate::helpers::implementations::SerializableVec3;
use crate::helpers::positions::{ChunkCoord, WorldPos};
use crate::renderer::gizmo::gizmo::Gizmo;
use crate::systems::systems::RoadDestroyType;
use crate::world::buildings::buildings::Buildings;
use crate::world::buildings::zoning::{DistrictId, ZoningStorage};
use crate::world::cars::car_subsystem::Cars;
use crate::world::cars::parking::ParkingSpotId;
use crate::world::cars::partitions::Destination;
use crate::world::cars::signfinding::{BreadCrumb, BreadCrumbDestination};
use crate::world::roads::intersections::{
    IntersectionBuildParams, build_intersection_at_node, gather_arms,
};
use crate::world::roads::road_editor::offset_polyline;
use crate::world::roads::road_mesh_manager::RoadMeshManager;
use crate::world::roads::road_structs::*;
use crate::world::roads::road_subsystem::Roads;
use crate::world::sound::sound::Sounds;
use crate::world::terrain::chunk_builder::ChunkMeshLod;
use crate::world::terrain::terrain_gen::TerrainGenerator;
use crate::world::terrain::terrain_subsystem::Terrain;
use glam::{Vec3, Vec3Swizzles};
use revision::revisioned;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::{PI, TAU};
use std::hash::{Hash, Hasher};
use std::mem::replace;
use xxhash_rust::xxh3::xxh3_64;

pub const METERS_PER_LANE_POLYLINE_STEP: f64 = 2.0;

type PartitionId = u32;
/// One physical "leg" of an intersection, a direction you can come from or go to.
/// Arms are sorted by bearing angle (clockwise from north, or whatever convention).
#[derive(Debug, Clone)]
#[revisioned(revision = 1)]
pub struct Arm {
    segment_id: SegmentId,
    /// Bearing angle in radians [0, 2π), CCW from +X axis
    bearing: f32,
    /// Direction vector pointing AWAY from node center (normalized)
    direction: SerializableVec3,
    /// Half-width of the road at this arm (lanes + sidewalk)
    half_width: f32,
    /// Length of this arm/corridor
    pub corridor_length: f32,

    incoming_lanes: Vec<LaneId>,
    outgoing_lanes: Vec<LaneId>,

    /// Dynamic: learned travel times to segments, updated by cars reporting back.
    bread_crumbs: Vec<BreadCrumb>,

    /// Current congestion estimate (0.0 = free flow, 1.0 = gridlocked)
    congestion: f32,
}
impl Arm {
    pub fn new(
        segment: SegmentId,
        bearing: f32,
        direction: Vec3,
        half_width: f32,
        corridor_length: f32,
    ) -> Self {
        Self {
            segment_id: segment,
            bearing,
            direction: SerializableVec3::from(direction.normalize_or_zero()),
            half_width,
            corridor_length,
            incoming_lanes: Vec::new(),
            outgoing_lanes: Vec::new(),
            bread_crumbs: Vec::new(),
            congestion: 0.0,
        }
    }
    pub fn update_travel_time(
        // TODO: Update when a car finishes its trip!
        &mut self,
        distr_id: DistrictId,
        part_id: PartitionId,
        seg_id: SegmentId,
        duration: f32,
    ) {
        let mut district: Option<&mut EMA> = None;
        let mut partition: Option<&mut EMA> = None;
        let mut segment: Option<&mut EMA> = None;

        for crumb in self.bread_crumbs.iter_mut() {
            match crumb.bread_crumb_destination {
                BreadCrumbDestination::Coarse {
                    district_id,
                    partition_id,
                } => {
                    if partition.is_none() {
                        // TODO: Check logic
                        if district_id == distr_id {
                            district = Some(&mut crumb.ema);

                            if partition_id == part_id {
                                district = None;
                                partition = Some(&mut crumb.ema);
                            }
                        }
                    }
                }

                BreadCrumbDestination::Fine { segment_id } if segment_id == seg_id => {
                    segment = Some(&mut crumb.ema);
                }

                _ => {}
            }
        }
        district.map(|ema| ema.update(duration));
        partition.map(|ema| ema.update(duration));
        segment.map(|ema| ema.update(duration)); // TODO: Check if they get moved?!
    }
    #[inline]
    pub fn travel_time_for_destination(&self, destination: Destination) -> Option<EMA> {
        let times = self.emas_for_destination(destination)?;
        times.segment.or(times.partition).or(times.district)
    }

    pub fn travel_time_for_destination_f32(&self, destination: Destination) -> f32 {
        self.travel_time_for_destination(destination)
            .map(|tt| tt.get())
            .unwrap_or(f32::MAX)
    }

    pub fn emas_for_destination(
        &self,
        destination_type: Destination,
    ) -> Option<DestinationTravelTimes> {
        let Some((distr_id, part_id, seg_id, _)) = destination_type.as_building() else {
            return None;
        };
        let mut result = DestinationTravelTimes::default();

        for crumb in self.bread_crumbs.iter() {
            match crumb.bread_crumb_destination {
                BreadCrumbDestination::Coarse {
                    district_id,
                    partition_id,
                } => {
                    if result.partition.is_none() {
                        if district_id == distr_id {
                            result.district = Some(crumb.ema.clone());

                            if partition_id == part_id {
                                result.district = None;
                                result.partition = Some(crumb.ema.clone());
                            }
                        }
                    }
                }

                BreadCrumbDestination::Fine { segment_id } if Some(segment_id) == seg_id => {
                    result.segment = Some(crumb.ema.clone());
                }

                _ => {}
            }
        }
        Some(result)
    }

    // === Getters ===
    pub fn corridor_length(&self) -> f32 {
        self.corridor_length
    }
    pub fn segment(&self) -> SegmentId {
        self.segment_id
    }

    pub fn bearing(&self) -> f32 {
        self.bearing
    }

    pub fn direction(&self) -> Vec3 {
        self.direction.as_vec3()
    }

    pub fn half_width(&self) -> f32 {
        self.half_width
    }

    pub fn incoming_lanes(&self) -> &[LaneId] {
        &self.incoming_lanes
    }

    pub fn outgoing_lanes(&self) -> &[LaneId] {
        &self.outgoing_lanes
    }

    pub fn congestion(&self) -> f32 {
        self.congestion
    }

    // === Lane Management ===

    pub fn add_incoming_lane(&mut self, lane_id: LaneId) {
        if !self.incoming_lanes.contains(&lane_id) {
            self.incoming_lanes.push(lane_id);
        }
    }

    pub fn add_outgoing_lane(&mut self, lane_id: LaneId) {
        if !self.outgoing_lanes.contains(&lane_id) {
            self.outgoing_lanes.push(lane_id);
        }
    }

    pub fn clear_lanes(&mut self) {
        self.incoming_lanes.clear();
        self.outgoing_lanes.clear();
    }

    /// Sort lanes by lane index (rightmost first for proper turn ordering)
    pub fn sort_lanes_by_index(&mut self, storage: &RoadStorage) {
        self.incoming_lanes.sort_by(|a, b| {
            let idx_a = storage.lane(*a).lane_index();
            let idx_b = storage.lane(*b).lane_index();
            idx_b.cmp(&idx_a) // Descending (rightmost first)
        });

        self.outgoing_lanes.sort_by(|a, b| {
            let idx_a = storage.lane(*a).lane_index();
            let idx_b = storage.lane(*b).lane_index();
            idx_a.cmp(&idx_b) // Ascending (rightmost first for outgoing)
        });
    }

    pub fn clear_travel_times(&mut self) {
        self.bread_crumbs.clear();
    }

    // === Congestion ===

    pub fn update_congestion(&mut self, new_value: f32) {
        // Smooth congestion updates
        const CONGESTION_ALPHA: f32 = 0.2;
        self.congestion = CONGESTION_ALPHA * new_value.clamp(0.0, 1.0)
            + (1.0 - CONGESTION_ALPHA) * self.congestion;
    }

    pub fn set_congestion(&mut self, value: f32) {
        self.congestion = value.clamp(0.0, 1.0);
    }

    // === Geometry Helpers ===

    /// Get perpendicular vector pointing to the RIGHT of this arm's direction
    pub fn right_perpendicular(&self) -> Vec3 {
        // Rotate direction 90° clockwise in XZ plane: (x, z) -> (z, -x)
        Vec3::new(self.direction.z, 0.0, -self.direction.x)
    }

    /// Get perpendicular vector pointing to the LEFT of this arm's direction
    pub fn left_perpendicular(&self) -> Vec3 {
        // Rotate direction 90° counter-clockwise in XZ plane: (x, z) -> (-z, x)
        Vec3::new(-self.direction.z, 0.0, self.direction.x)
    }

    /// Get the position of the right edge at a given distance from center
    pub fn right_edge_at(&self, center: WorldPos, distance: f32) -> WorldPos {
        let offset =
            self.direction.as_vec3() * distance + self.right_perpendicular() * self.half_width;
        center.add_vec3(offset)
    }

    /// Get the position of the left edge at a given distance from center
    pub fn left_edge_at(&self, center: WorldPos, distance: f32) -> WorldPos {
        let offset =
            self.direction.as_vec3() * distance + self.left_perpendicular() * self.half_width;
        center.add_vec3(offset)
    }

    #[inline]
    pub fn road_type<'a>(
        &self,
        storage: &RoadStorage,
        road_types: &'a RoadTypes,
    ) -> Option<&'a RoadType> {
        let segment = storage.segment(self.segment_id);
        let Some(road_type) = road_types.get_road_type(segment.road_type_id) else {
            return None;
        };
        Some(road_type)
    }
}

// pub fn travel_time_averaged_from_emas(
//     emas: Vec<&EMA>
// ) -> Option<EMA> {
//     let mut total_weight = 0.0f32;
//     let mut weighted_sum = 0.0f32;
//
//     for ema in emas {
//         let w = ema.sample_count().max(1) as f32;
//         weighted_sum += ema.get() * w;
//         total_weight += w;
//     }
//
//     if total_weight == 0.0 {
//         return None;
//     }
//     let value = weighted_sum/total_weight;
//     Some(EMA::with_state(
//         value, // I have to choose the smoothing so that it's good!!!!
//         total_weight as u16 // trust indicator
//     ))
// }
/// Intersection anchor point in 3D space.
/// Every node is an intersection with attachable traffic controls.
#[derive(Default, Debug, Clone)]
#[revisioned(revision = 1)]
pub struct Node {
    pos: WorldPos,
    /// Sorted by bearing, clockwise
    arms: Vec<Arm>,
    node_lanes: Vec<NodeLane>,
    incoming_lanes: Vec<LaneId>,
    outgoing_lanes: Vec<LaneId>,
    attached_controls: Vec<AttachedControl>,
    next_control_id: u32,

    car_spawning_rate: f32,
}

impl Node {
    pub fn new(pos: WorldPos) -> Self {
        Self {
            pos,
            arms: Vec::with_capacity(2),
            node_lanes: Vec::new(),
            incoming_lanes: Vec::new(),
            outgoing_lanes: Vec::new(),
            attached_controls: Vec::new(),
            next_control_id: 0,
            car_spawning_rate: 0.0,
        }
    }

    // TODO: No way found should activate compass mode (wait no... Not here, but whoever calls this function should do it!)
    pub fn ranked_arms_for_address(
        &self,
        buildings: &Buildings,
        zoning: &ZoningStorage,
        destination_type: Destination,
    ) -> Vec<(&Arm, f32)> {
        let mut ranked: Vec<(&Arm, f32)> = self
            .arms
            .iter()
            .map(|arm| (arm, arm.travel_time_for_destination_f32(destination_type)))
            .collect();

        ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
        ranked
    }
    #[inline]
    pub fn arm_for_segment(&self, segment_id: SegmentId) -> Option<&Arm> {
        self.arms.iter().find(|&arm| arm.segment_id == segment_id)
    }
    #[inline]
    pub fn arm_for_segment_mut(&mut self, segment_id: SegmentId) -> Option<&mut Arm> {
        self.arms
            .iter_mut()
            .find(|arm| arm.segment_id == segment_id)
    }

    #[inline]
    pub fn pos(&self) -> WorldPos {
        self.pos
    }

    #[inline]
    pub fn chunk_coord(&self) -> ChunkCoord {
        self.pos.chunk
    }

    /// Every node is an intersection by design.
    #[inline]
    pub fn node_lanes(&self) -> &[NodeLane] {
        &self.node_lanes
    }
    /// Every segment is kind of an arm by design.
    #[inline]
    pub fn arms(&self) -> &[Arm] {
        &self.arms
    }
    #[inline]
    pub fn add_node_lanes<I>(&mut self, lanes: I)
    where
        I: IntoIterator<Item = NodeLane>,
    {
        self.node_lanes.extend(lanes);
    }

    #[inline]
    pub fn clear_node_lanes(&mut self) {
        self.node_lanes.clear()
    }
    #[inline]
    pub fn incoming_lanes(&self) -> &[LaneId] {
        &self.incoming_lanes
    }

    #[inline]
    pub fn outgoing_lanes(&self) -> &[LaneId] {
        &self.outgoing_lanes
    }
    #[inline]
    pub fn replace_incoming_lanes(&mut self, lanes: Vec<LaneId>) {
        self.incoming_lanes = lanes;
    }
    #[inline]
    pub fn replace_outgoing_lanes(&mut self, lanes: Vec<LaneId>) {
        self.outgoing_lanes = lanes;
    }
    #[inline]
    pub fn lanes(&self) -> impl Iterator<Item = &LaneId> {
        self.incoming_lanes.iter().chain(self.outgoing_lanes.iter())
    }
    #[inline]
    pub fn node_lane(&self, node_lane_id: NodeLaneId) -> Option<&NodeLane> {
        self.node_lanes.get(node_lane_id as usize)
    }
    #[inline]
    pub fn _attached_controls(&self) -> &[AttachedControl] {
        &self.attached_controls
    }
    /// Returns true if the node has any active traffic control.
    #[inline]
    pub fn _has_active_control(&self) -> bool {
        self.attached_controls
            .iter()
            .any(|c| c.enabled && !matches!(c.control, TrafficControl::None))
    }

    /// Returns the count of connected lanes (incoming + outgoing).
    #[inline]
    pub fn connection_count(&self) -> usize {
        self.incoming_lanes.len() + self.outgoing_lanes.len()
    }
    #[inline]
    pub fn car_spawning_rate(&self) -> f32 {
        self.car_spawning_rate
    }
    pub fn update_heights(
        &mut self,
        chunks: &mut HashMap<ChunkCoord, ChunkMeshLod>,
        terrain_gen: &TerrainGenerator,
    ) {
        for nodelane in self.node_lanes.iter_mut() {
            nodelane.update_heights(chunks, terrain_gen);
        }
        self.pos.local.y = Terrain::get_height_at_explicit(chunks, terrain_gen, self.pos, true);
    }
}

/// Road segment connecting two nodes, containing multiple lanes.
/// Segments are grouping/metadata; lanes are the first-class graph edges.
#[derive(Debug, Clone, Default)]
#[revisioned(revision = 1)]
pub struct Segment {
    pub start: NodeId,
    pub end: NodeId,
    pub lanes: Vec<LaneId>,
    pub centerline: LaneGeometry,
    pub structure: StructureType,
    pub version: u32,
    pub road_type_id: RoadTypeId, // The ONLY place this is stored btw, intersections ask segments!
    pub parking_spots: Vec<ParkingSpotId>,
}

impl Segment {
    fn new(start: NodeId, end: NodeId, structure: StructureType, road_type_id: RoadTypeId) -> Self {
        Self {
            start,
            end,
            lanes: Vec::new(),
            centerline: LaneGeometry::default(),
            structure,
            version: 0,
            road_type_id,
            parking_spots: vec![],
        }
    }

    #[inline]
    pub fn start(&self) -> NodeId {
        self.start
    }

    #[inline]
    pub fn end(&self) -> NodeId {
        self.end
    }
    #[inline]
    pub fn nodes(&self) -> Vec<NodeId> {
        vec![self.start, self.end]
    }
    #[inline]
    pub fn other_node(&self, node_id: NodeId) -> Option<NodeId> {
        if node_id == self.start {
            Some(self.end)
        } else if node_id == self.end {
            Some(self.start)
        } else {
            None
        }
    }

    #[inline]
    pub fn lanes(&self) -> &[LaneId] {
        &self.lanes
    }

    #[inline]
    pub fn structure(&self) -> StructureType {
        self.structure
    }

    #[inline]
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Returns lane count in each direction (forward, backward).
    pub fn lane_counts(&self, storage: &RoadStorage) -> LaneDirectionCounts {
        let mut counts = LaneDirectionCounts::default();
        for lane_id in &self.lanes {
            let lane = storage.lane(*lane_id);
            if lane.from_node() == self.start {
                counts.forward += 1;
            } else {
                counts.backward += 1;
            }
        }
        counts
    }
}

/// Directed lane edge connecting two nodes within a segment.
/// Lanes are the primary graph edges for pathfinding and simulation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[revisioned(revision = 1)]
pub struct Lane {
    from: NodeId,
    to: NodeId,
    segment: SegmentId,
    lane_index: i8, // signed, relative to segment centerline
    speed_limit: f32,
    capacity: u32,
    vehicle_mask: u32,
    geometry: LaneGeometry,
}

impl Lane {
    pub fn new(
        from: NodeId,
        to: NodeId,
        segment: SegmentId,
        lane_index: i8, // signed, relative to segment centerline
        speed_limit: f32,
        capacity: u32,
        vehicle_mask: u32,
        geometry: LaneGeometry,
    ) -> Self {
        Self {
            from,
            to,
            segment,
            lane_index,
            speed_limit,
            capacity,
            vehicle_mask,
            geometry,
        }
    }

    #[allow(clippy::wrong_self_convention)]
    #[inline]
    pub fn from_node(&self) -> NodeId {
        self.from
    }

    #[inline]
    pub fn to_node(&self) -> NodeId {
        self.to
    }

    #[inline]
    pub fn segment(&self) -> SegmentId {
        self.segment
    }
    #[inline]
    pub fn lane_index(&self) -> i8 {
        self.lane_index
    }
    #[inline]
    pub fn speed_limit(&self) -> f32 {
        self.speed_limit
    }

    #[inline]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    #[inline]
    pub fn vehicle_mask(&self) -> u32 {
        self.vehicle_mask
    }

    /// Returns true if the lane allows the given vehicle type.
    #[inline]
    pub fn allows_vehicle(&self, vehicle_type: u32) -> bool {
        (self.vehicle_mask & vehicle_type) != 0
    }
    #[inline]
    pub fn geometry(&self) -> &LaneGeometry {
        &self.geometry
    }
    #[inline]
    pub fn polyline(&self) -> &Vec<WorldPos> {
        &self.geometry.points
    }
    #[inline]
    pub fn replace_geometry(&mut self, geometry: LaneGeometry) {
        self.geometry = geometry;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[revisioned(revision = 1)]
pub enum LaneRef {
    Lane(LaneId, PolyIdx),
    NodeLane(NodeId, NodeLaneId, PolyIdx),
}
impl LaneRef {
    pub fn as_lane(&self) -> Option<(LaneId, PolyIdx)> {
        match self {
            LaneRef::Lane(lane_id, poly_idx) => Some((*lane_id, *poly_idx)),
            LaneRef::NodeLane(_, _, _) => None,
        }
    }
}
impl Hash for NodeLane {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        // TODO: Investigate the lack of hashes
    }
}
/// Directed lane edge connecting two segments within a node or connecting NodeLanes with each other.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
#[revisioned(revision = 1)]
pub struct NodeLane {
    id: NodeLaneId,
    merging: Vec<LaneRef>,
    splitting: Vec<LaneRef>,
    geometry: LaneGeometry,

    speed_limit: f32,
    vehicle_mask: u32,
}

impl NodeLane {
    pub fn new(
        id: NodeLaneId,
        merging: Vec<LaneRef>,
        splitting: Vec<LaneRef>,
        geometry: LaneGeometry,
        speed_limit: f32,
        vehicle_mask: u32,
    ) -> Self {
        Self {
            id,
            merging,
            splitting,
            geometry,
            speed_limit,
            vehicle_mask,
        }
    }

    #[allow(clippy::wrong_self_convention)]
    #[inline]
    pub fn id(&self) -> NodeLaneId {
        self.id
    }
    #[inline]
    pub fn splitting(&self) -> &Vec<LaneRef> {
        &self.splitting
    }
    #[inline]
    pub fn merging(&self) -> &Vec<LaneRef> {
        &self.merging
    }
    #[inline]
    pub fn polyline(&self) -> &Vec<WorldPos> {
        &self.geometry.points
    }
    #[inline]
    pub fn geometry(&self) -> &LaneGeometry {
        &self.geometry
    }
    #[inline]
    pub fn total_length(&self) -> f64 {
        self.geometry.total_len
    }

    #[inline]
    pub fn speed_limit(&self) -> f32 {
        self.speed_limit
    }

    /// Returns true if the lane allows the given vehicle type.
    #[inline]
    pub fn allows_vehicle(&self, vehicle_type: u32) -> bool {
        (self.vehicle_mask & vehicle_type) != 0
    }
    pub fn update_heights(
        &mut self,
        chunks: &mut HashMap<ChunkCoord, ChunkMeshLod>,
        terrain_gen: &TerrainGenerator,
    ) {
        self.geometry.update_heights(chunks, terrain_gen);
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[revisioned(revision = 1)]
pub struct LaneGeometry {
    pub points: Vec<WorldPos>, // polyline
    pub lengths: Vec<f64>,     // cumulative arc length
    pub total_len: f64,
}

impl LaneGeometry {
    pub fn from_polyline(points: Vec<WorldPos>) -> Self {
        debug_assert!(points.len() >= 2);

        let mut lengths = Vec::with_capacity(points.len());
        let mut total_len = 0.0;

        lengths.push(0.0);

        for i in 1..points.len() {
            total_len += points[i].distance_to(points[i - 1]);
            lengths.push(total_len);
        }

        LaneGeometry {
            points,
            lengths,
            total_len,
        }
    }

    pub fn closest_point_to_xz(&self, pos: WorldPos) -> LaneProjection {
        debug_assert!(self.points.len() >= 2);

        let mut best_pos = self.points[0];
        let mut best_dist_sq = f64::INFINITY;
        let mut best_t = 0.0;
        let mut best_tangent = Vec3::ZERO;
        let mut best_poly_idx = 0 as PolyIdx;

        if self.total_len <= 1e-9 {
            return LaneProjection {
                position: best_pos,
                distance: best_pos.delta_to(pos).xz().length() as f64,
                tangent: Vec3::ZERO,
                t: 0.0,
                poly_idx: best_poly_idx,
            };
        }

        for i in 0..self.points.len() - 1 {
            let a = self.points[i];
            let b = self.points[i + 1];

            let ab = a.delta_to(b);
            let ap = a.delta_to(pos);

            let dx = ab.x as f64;
            let dz = ab.z as f64;
            let len_sq = dx * dx + dz * dz;

            if len_sq <= 1e-12 {
                continue;
            }

            let t = ((ap.x as f64 * dx) + (ap.z as f64 * dz)) / len_sq;
            let t = t.clamp(0.0, 1.0);

            let candidate = a.add_vec3(ab * t as f32);
            let diff = candidate.delta_to(pos);

            let dist_sq = diff.x as f64 * diff.x as f64 + diff.z as f64 * diff.z as f64;

            if dist_sq < best_dist_sq {
                let segment_len = self.lengths[i + 1] - self.lengths[i];
                let distance_along = self.lengths[i] + segment_len * t;

                best_pos = candidate;
                best_dist_sq = dist_sq;
                best_t = distance_along / self.total_len;
                best_tangent = Vec3::new(ab.x, 0.0, ab.z).normalize_or_zero();
                best_poly_idx = i as PolyIdx;
            }
        }

        LaneProjection {
            position: best_pos,
            distance: best_dist_sq.sqrt(),
            tangent: best_tangent,
            t: best_t,
            poly_idx: best_poly_idx,
        }
    }

    pub fn sample_at_t(&self, t: f64) -> Option<WorldPos> {
        if self.points.len() < 2 {
            return None;
        }

        let t = t.clamp(0.0, 1.0);

        if t <= 0.0 {
            return Some(self.points[0]);
        }

        if t >= 1.0 {
            return Some(*self.points.last().unwrap());
        }

        if self.total_len <= 1e-9 {
            return Some(self.points[0]);
        }

        let target = t * self.total_len;

        let mut i = 1;
        while i < self.lengths.len() && self.lengths[i] < target {
            i += 1;
        }

        let i1 = i.min(self.points.len() - 1);
        let i0 = i1 - 1;

        let a_len = self.lengths[i0];
        let b_len = self.lengths[i1];

        let local_t = if b_len > a_len {
            (target - a_len) / (b_len - a_len)
        } else {
            0.0
        };

        Some(self.points[i0].lerp(self.points[i1], local_t))
    }

    pub fn tangent_at_t(&self, t: f64) -> Vec3 {
        if self.points.len() < 2 || self.total_len <= 0.0 {
            return Vec3::ZERO;
        }

        let t = t.clamp(0.0, 1.0);

        if t <= 0.0 {
            return self.points[0].delta_to(self.points[1]).normalize_or_zero();
        }

        if t >= 1.0 {
            let n = self.points.len();
            return self.points[n - 2]
                .delta_to(self.points[n - 1])
                .normalize_or_zero();
        }

        let target = t * self.total_len;

        let mut i = 1;
        while i < self.lengths.len() && self.lengths[i] < target {
            i += 1;
        }

        let i0 = i - 1;
        let i1 = i.min(self.points.len() - 1);

        self.points[i0]
            .delta_to(self.points[i1])
            .normalize_or_zero()
    }

    pub fn update_heights(
        &mut self,
        chunks: &mut HashMap<ChunkCoord, ChunkMeshLod>,
        terrain_gen: &TerrainGenerator,
    ) {
        for point in self.points.iter_mut() {
            point.local.y = Terrain::get_height_at_explicit(chunks, terrain_gen, *point, true);
        }
        *self = Self::from_polyline(self.points.clone());
    }
}

// RoadManager

pub type RoadRegionId = u32;

#[derive(Serialize, Deserialize, Clone)]
#[revisioned(revision = 1)]
pub struct RoadRegion {
    nodes: Vec<NodeId>,
}

impl RoadRegion {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn node_ids(&self) -> &[NodeId] {
        &self.nodes
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[derive(Clone)]
#[revisioned(revision = 1)]
pub struct RoadStorage {
    pub nodes: Vec<Option<Node>>,
    nodes_free_list: Vec<NodeId>,
    pub segments: Vec<Option<Segment>>,
    segments_free_list: Vec<SegmentId>,
    pub lanes: Vec<Option<Lane>>,
    lanes_free_list: Vec<LaneId>,
    node_to_region: Vec<RoadRegionId>,
    regions: Vec<RoadRegion>,
    free_regions: Vec<RoadRegionId>,
    active_region_count: usize,
}

impl Default for RoadStorage {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            nodes_free_list: vec![],
            segments: Vec::new(),
            segments_free_list: vec![],
            lanes: Vec::new(),
            lanes_free_list: vec![],
            node_to_region: Vec::new(),
            regions: Vec::new(),
            free_regions: Vec::new(),
            active_region_count: 0,
        }
    }
}

impl RoadStorage {
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.segments.clear();
        self.lanes.clear();
        self.node_to_region.clear();
        self.regions.clear();
        self.free_regions.clear();
        self.active_region_count = 0;
    }

    pub fn add_node(&mut self, id: NodeId, world_pos: WorldPos) {
        self.nodes[id.index()] = Some(Node::new(world_pos));

        let region_id = if let Some(reused_id) = self.free_regions.pop() {
            self.regions[reused_id as usize].nodes.push(id);
            reused_id
        } else {
            let new_id = self.regions.len() as RoadRegionId;
            let mut region = RoadRegion::new();
            region.nodes.push(id);
            self.regions.push(region);
            new_id
        };

        self.node_to_region.push(region_id);
        self.active_region_count += 1;
    }

    pub fn add_segment(
        &mut self,
        id: SegmentId,
        start: NodeId,
        end: NodeId,
        structure: StructureType,
        road_type_id: RoadTypeId,
    ) {
        let segment = Segment::new(start, end, structure, road_type_id);
        self.segments[id.index()] = Some(segment);

        let region_a = self.node_to_region[start.index()];
        let region_b = self.node_to_region[end.index()];

        if region_a != region_b {
            self.merge_regions(region_a, region_b);
        }
    }

    pub fn alloc_node_id(&mut self) -> NodeId {
        let id = if let Some(reused_id) = self.nodes_free_list.pop() {
            reused_id.raw() as usize
        } else {
            self.nodes.len()
        };

        if id < self.nodes.len() {
            debug_assert!(self.nodes[id].is_none());
            self.nodes[id] = Some(Node::new(WorldPos::zero()));
        } else {
            debug_assert_eq!(id, self.nodes.len());
            self.nodes.push(Some(Node::new(WorldPos::zero())));
        }

        NodeId::new(id as u32)
    }
    pub fn alloc_segment_id(&mut self) -> SegmentId {
        let id = if let Some(reused_id) = self.segments_free_list.pop() {
            reused_id.raw() as usize
        } else {
            self.segments.len()
        };

        if id < self.segments.len() {
            debug_assert!(self.segments[id].is_none());
            self.segments[id] = Some(Segment::default());
        } else {
            debug_assert_eq!(id, self.segments.len());
            self.segments.push(Some(Segment::default()));
        }

        SegmentId::new(id as u32)
    }

    pub fn alloc_lane_id(&mut self) -> LaneId {
        let id = if let Some(reused_id) = self.lanes_free_list.pop() {
            reused_id.raw() as usize
        } else {
            self.lanes.len()
        };

        if id < self.lanes.len() {
            debug_assert!(self.lanes[id].is_none());
            self.lanes[id] = Some(Lane::default());
        } else {
            debug_assert_eq!(id, self.lanes.len());
            self.lanes.push(Some(Lane::default()));
        }

        LaneId::new(id as u32)
    }

    pub fn segment_of_lane(&self, lane_id: LaneId) -> SegmentId {
        self.lane(lane_id).segment()
    }

    fn merge_regions(&mut self, a: RoadRegionId, b: RoadRegionId) {
        // let len_a = self.regions[a as usize].nodes.len();
        // let len_b = self.regions[b as usize].nodes.len();
        //
        // let (smaller, larger) = if len_a <= len_b { (a, b) } else { (b, a) };
        //
        // let nodes_to_move = std::mem::take(&mut self.regions[smaller as usize].nodes);
        //
        // for &node_idx in &nodes_to_move {
        //     self.node_to_region[node_idx.index()] = larger;
        // }
        //
        // self.regions[larger as usize].nodes.extend(nodes_to_move);
        // self.free_regions.push(smaller);
        // self.active_region_count -= 1;
    }

    /// Returns the current region ID for a node.
    ///
    /// # Stability
    ///
    /// Region IDs become stale after merges. If you call `add_segment` connecting
    /// two nodes in different regions, the smaller region is merged into the larger.
    /// Any previously-obtained ID for the smaller region now points to an empty slot.
    /// Re-query after any connectivity changes if freshness matters.
    #[inline]
    pub fn region_for_node(&self, node_id: NodeId) -> RoadRegionId {
        self.node_to_region[node_id.index()]
    }
    /// Returns an iterator over all active (non-empty) regions with their IDs.
    ///
    /// Active regions contain at least one node. Empty regions resulting from
    /// prior merge operations are skipped. Region IDs remain stable until the
    /// next merge operation occurs.
    pub fn iter_active_regions(&self) -> impl Iterator<Item = (RoadRegionId, &RoadRegion)> {
        self.regions
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.is_empty())
            .map(|(i, r)| (i as RoadRegionId, r))
    }
    /// Returns the region, which may be empty if it was merged into another.
    #[inline]
    pub fn get_region(&self, region_id: RoadRegionId) -> &RoadRegion {
        &self.regions[region_id as usize]
    }

    #[inline]
    pub fn are_nodes_connected(&self, a: NodeId, b: NodeId) -> bool {
        self.node_to_region[a.index()] == self.node_to_region[b.index()]
    }

    #[inline]
    pub fn nodes_in_region(&self, region_id: RoadRegionId) -> &[NodeId] {
        &self.regions[region_id as usize].nodes
    }

    #[inline]
    pub fn is_region_active(&self, region_id: RoadRegionId) -> bool {
        (region_id as usize) < self.regions.len() && !self.regions[region_id as usize].is_empty()
    }

    #[inline]
    pub fn active_region_count(&self) -> usize {
        self.active_region_count
    }

    #[inline]
    pub fn total_region_slots(&self) -> usize {
        self.regions.len()
    }

    #[inline]
    pub fn free_region_slot_count(&self) -> usize {
        self.free_regions.len()
    }

    #[inline]
    pub fn node(&self, id: NodeId) -> &Node {
        self.nodes[id.index()].as_ref().unwrap_or_else(|| {
            panic!(
                "Report to maxwag9 on Github pls: NodeId {} couldn't be unpacked",
                id.raw()
            )
        })
    }
    #[inline]
    pub fn node_safe(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.index()).and_then(|n| n.as_ref())
    }
    #[inline]
    pub fn node_exists(&self, id: NodeId) -> bool {
        self.nodes.get(id.index()).is_some()
    }
    #[inline]
    pub fn node_mut(&mut self, id: NodeId) -> &mut Node {
        self.nodes[id.0 as usize].as_mut().unwrap_or_else(|| {
            panic!(
                "Report to maxwag9 on Github pls: NodeId {} couldn't be unpacked mutably",
                id.raw()
            )
        })
    }
    // #[inline]
    // pub fn node_mut_safe(&mut self, id: NodeId) -> Option<&mut Node> {
    //     self.nodes.get_mut(id.0 as usize).and_then(|n| n.as_mut())
    // }
    /// deletes the node and all segments/lanes touching it
    pub fn delete_node(&mut self, id: NodeId, road_types: &RoadTypes, gizmo: &mut Gizmo) {
        let impact = self.impact_of_deleting_node(id);
        self.apply_impact(&impact);

        for node_id in impact.nodes_needing_regen {
            let arms = gather_arms(self, road_types, node_id, gizmo);
            let node = self.node_mut(node_id);
            node.arms = arms;
        }
    }

    #[inline]
    pub fn iter_nodes(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.nodes
            .iter()
            .enumerate()
            .flat_map(|(i, n)| n.as_ref().map(|n| (NodeId::new(i as u32), n)))
    }

    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn lane_counts_for_segment(&self, segment: &Segment) -> LaneDirectionCounts {
        segment.lane_counts(self)
    }

    pub fn segments_connected_to_node(&self, node_id: NodeId) -> Vec<SegmentId> {
        let node = self.node(node_id);

        let mut segments = Vec::new();

        for &lane_id in node.incoming_lanes().iter().chain(node.outgoing_lanes()) {
            let lane = self.lane(lane_id);
            let seg = lane.segment();
            let segment = self.segment(seg);
            if !segments.contains(&seg) {
                segments.push(seg);
            }
        }

        segments
    }

    pub fn segment_count_connected_to_node(&self, node_id: NodeId) -> usize {
        let node = self.node(node_id);

        let mut count = 0;
        let mut seen = Vec::new();

        for &lane_id in node.incoming_lanes().iter().chain(node.outgoing_lanes()) {
            let lane = self.lane(lane_id);
            let seg = lane.segment();
            let segment = self.segment(seg);
            if !seen.contains(&seg) {
                seen.push(seg);
                count += 1;
            }
        }

        count
    }

    pub fn get_active_segment_ids(&self) -> impl Iterator<Item = SegmentId> + '_ {
        self.segments
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_some())
            .map(|(idx, _)| SegmentId::new(idx as u32))
    }

    pub fn get_active_node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.is_some())
            .map(|(idx, _)| NodeId::new(idx as u32))
    }

    #[inline]
    pub fn segment(&self, id: SegmentId) -> &Segment {
        self.segments[id.0 as usize]
            .as_ref()
            .unwrap_or_else(|| panic!("Report to maxwag9 on Github pls and explain how this happened: SegmentId {} couldn't be unpacked mutably", id.raw()))
    }
    #[inline]
    pub fn segment_mut(&mut self, id: SegmentId) -> &mut Segment {
        self.segments[id.0 as usize].as_mut().unwrap()
    }
    #[inline]
    pub fn segment_safe(&self, id: SegmentId) -> Option<&Segment> {
        self.segments.get(id.index()).and_then(|s| s.as_ref())
    }
    // #[inline]
    // pub fn segment_safe_mut(&mut self, id: SegmentId) -> Option<&mut Segment> {
    //     self.segments.get_mut(id.index()).and_then(|s| s.as_mut())
    // }

    /// deletes the segment and all its lanes
    pub fn delete_segment(
        &mut self,
        id: SegmentId,
        road_types: &RoadTypes,
        gizmo: &mut Gizmo,
        remove_orphan_nodes: bool,
    ) {
        let impact = self.impact_of_deleting_segment(id, remove_orphan_nodes);
        self.apply_impact(&impact);
        for node_id in impact.nodes_needing_regen {
            let arms = gather_arms(self, road_types, node_id, gizmo);
            let node = self.node_mut(node_id);
            node.arms = arms;
        }
    }

    #[inline]
    pub fn iter_segments(&self) -> impl Iterator<Item = (SegmentId, &Segment)> {
        self.segments
            .iter()
            .enumerate()
            .flat_map(|(i, s)| s.as_ref().map(|s| (SegmentId::new(i as u32), s)))
    }

    #[inline]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    pub fn segment_ids_touching_chunk(&self, chunk_coord: ChunkCoord) -> Vec<SegmentId> {
        self.iter_segments()
            .filter_map(|(idx, seg)| {
                let touches = if seg.lanes.is_empty() {
                    let start = self.node(seg.start()).pos();
                    let end = self.node(seg.end()).pos();
                    chunk_coord.line_touches_chunk_precise(start, end)
                } else {
                    seg.lanes.iter().any(|&lane_id| {
                        chunk_coord.polyline_touches_chunk(&self.lane(lane_id).geometry().points)
                    })
                };
                touches.then_some(idx)
            })
            .collect()
    }

    #[inline]
    pub fn segment_ids_touching_chunks(&self, chunks: &Vec<ChunkCoord>) -> Vec<SegmentId> {
        chunks
            .iter()
            .map(|chunk_coord| self.segment_ids_touching_chunk(*chunk_coord))
            .flatten()
            .collect()
    }
    #[inline]
    pub fn node_lane_count_for_node(&self, id: NodeId) -> usize {
        self.node(id).node_lanes.len()
    }

    pub fn add_lane(
        &mut self,
        id: LaneId,
        from: NodeId,
        to: NodeId,
        segment: SegmentId,
        lane_index: i8,
        geometry: LaneGeometry,
        speed_limit: f32,
        capacity: u32,
        vehicle_mask: u32,
    ) {
        if let Some(seg) = self.segment_safe(segment) {
            debug_assert!(
                (from == seg.start && to == seg.end) || (from == seg.end && to == seg.start),
                "Lane endpoints must match segment endpoints"
            );
        }

        let lane = Lane::new(
            from,
            to,
            segment,
            lane_index,
            speed_limit,
            capacity,
            vehicle_mask,
            geometry,
        );

        self.lanes[id.index()] = Some(lane);

        let segment = self.segment_mut(segment);
        segment.lanes.push(id);

        let node = self.node_mut(from);
        node.outgoing_lanes.push(id);

        let node = self.node_mut(to);
        node.incoming_lanes.push(id);
    }
    #[inline]
    pub fn lane_exists(&self, id: LaneId) -> bool {
        self.lanes.get(id.index()).is_some()
    }
    #[inline]
    pub fn lane(&self, id: LaneId) -> &Lane {
        self.lanes[id.index()]
            .as_ref()
            .unwrap_or_else(|| panic!("Report to maxwag9 on Github pls and explain how this happened: LaneId {} couldn't be unpacked", id.raw()))
    }
    #[inline]
    pub fn lane_safe(&self, id: LaneId) -> Option<&Lane> {
        self.lanes.get(id.index()).and_then(|l| l.as_ref())
    }
    // #[inline]
    // pub fn lane_mut(&mut self, id: LaneId) -> Option<&mut Lane> {
    //     self.lanes.get_mut(id.index()).and_then(|l| l.as_mut())
    // }
    #[inline]
    pub fn lane_mut(&mut self, id: LaneId) -> &mut Lane {
        self.lanes[id.index()].as_mut().unwrap()
    }

    /// deletes a single lane, then cascades
    pub fn delete_lane(
        &mut self,
        id: LaneId,
        road_types: &RoadTypes,
        gizmo: &mut Gizmo,
        delete_orphans: bool,
    ) {
        let impact = self.impact_of_deleting_lane(id, delete_orphans);
        let (from, to) = {
            let lane = self.lane(id);
            (lane.from_node(), lane.to_node())
        };

        self.apply_impact(&impact);

        if !impact.nodes.contains(&from) {
            if let Some(node) = self.nodes[from.index()].as_mut() {
                node.outgoing_lanes.retain(|lane_id| *lane_id != id);
            }
        }

        if !impact.nodes.contains(&to) {
            if let Some(node) = self.nodes[to.index()].as_mut() {
                node.incoming_lanes.retain(|lane_id| *lane_id != id);
            }
        }

        for node_id in impact.nodes_needing_regen {
            let arms = gather_arms(self, road_types, node_id, gizmo);
            let node = self.node_mut(node_id);
            node.arms = arms;
        }
    }

    #[inline]
    pub fn iter_lanes(&self) -> impl Iterator<Item = (LaneId, &Lane)> {
        self.lanes
            .iter()
            .enumerate()
            .flat_map(|(i, l)| l.as_ref().map(|l| (LaneId::new(i as u32), l)))
    }

    #[inline]
    pub fn lane_count(&self) -> usize {
        self.lanes.len()
    }

    pub fn attach_control(&mut self, node_id: NodeId, control: TrafficControl) -> ControlId {
        let node = self.node_mut(node_id);
        let id = ControlId::new(node.next_control_id);
        node.next_control_id += 1;
        node.attached_controls.push(AttachedControl {
            id,
            control,
            enabled: true,
        });
        id
    }

    pub fn delete_control(&mut self, node_id: NodeId, control_id: ControlId) {
        let node = self.node_mut(node_id);
        if let Some(ctrl) = node
            .attached_controls
            .iter_mut()
            .find(|c| c.id == control_id)
        {
            ctrl.enabled = false;
        }
    }

    pub fn enable_control(&mut self, node_id: NodeId, control_id: ControlId) {
        let node = self.node_mut(node_id);
        if let Some(ctrl) = node
            .attached_controls
            .iter_mut()
            .find(|c| c.id == control_id)
        {
            ctrl.enabled = true;
        }
    }

    pub fn add_node_lane(
        &mut self,
        node_id: NodeId,
        merging: Vec<LaneRef>,
        splitting: Vec<LaneRef>,
        geometry: LaneGeometry,
        speed_limit: f32,
        vehicle_mask: u32,
    ) -> NodeLaneId {
        debug_assert!(!merging.is_empty());
        debug_assert!(!splitting.is_empty());

        let node = self.node_mut(node_id);
        let id = node.node_lanes.len() as NodeLaneId;

        node.node_lanes.push(NodeLane {
            id,
            merging,
            splitting,
            geometry,
            speed_limit,
            vehicle_mask,
        });

        id
    }

    pub fn upgrade_segment<F>(
        &mut self,
        old_segment: SegmentId,
        add_new: F,
        road_types: &RoadTypes,
        gizmo: &mut Gizmo,
    ) -> Vec<SegmentId>
    where
        F: FnOnce(&mut Self),
    {
        let segment_count_before = self.segments.len();

        self.delete_segment(old_segment, road_types, gizmo, false);

        add_new(self);

        (segment_count_before..self.segments.len())
            .map(|i| SegmentId::new(i as u32))
            .collect()
    }

    pub fn nodes_in_chunk(&self, chunk_coord: ChunkCoord) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(move |(idx, node)| {
                let node = node.as_ref()?;
                (node.chunk_coord() == chunk_coord).then(|| NodeId::new(idx as u32))
            })
    }
    pub fn iter_node_ids_optionally_chunked(
        &self,
        chunk_coord: Option<ChunkCoord>,
    ) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(move |(idx, node)| {
                let node = node.as_ref()?;

                if chunk_coord.is_none_or(|cid| node.chunk_coord() == cid) {
                    Some(NodeId::new(idx as u32))
                } else {
                    None
                }
            })
    }
    /// TODO: Expensive! →øþø→↓
    pub fn closest_lane_point_to(&self, pos: WorldPos) -> Option<LaneProjection> {
        let mut best: Option<LaneProjection> = None;
        for segment_id in self.segment_ids_touching_chunk(pos.chunk) {
            let segment = self.segment(segment_id);
            for &lane_id in segment.lanes() {
                let lane = self.lane(lane_id);
                let projection = lane.geometry.closest_point_to_xz(pos);
                if best
                    .as_ref()
                    .map_or(true, |b| projection.distance < b.distance)
                {
                    best = Some(projection);
                }
            }
        }
        best
    }

    pub fn impact_of_deleting_node(&self, id: NodeId) -> DeleteImpact {
        let node = self.node(id);

        let mut deleted_nodes = vec![id];
        let mut deleted_segments = Vec::new();
        let mut deleted_lanes = Vec::new();
        let mut nodes_needing_regen = Vec::new();

        let mut seen_segs: Vec<SegmentId> = Vec::new();
        for &lane_id in node.incoming_lanes().iter().chain(node.outgoing_lanes()) {
            let seg_id = self.segment_of_lane(lane_id);
            if !seen_segs.contains(&seg_id) {
                seen_segs.push(seg_id);
            }
        }

        let mut other_nodes: Vec<NodeId> = Vec::new();
        for &seg_id in &seen_segs {
            deleted_segments.push(seg_id);

            let Some(seg) = &self.segments[seg_id.index()] else {
                continue;
            };
            for &lane_id in &seg.lanes {
                deleted_lanes.push(lane_id);
            }

            let other = if seg.start == id { seg.end } else { seg.start };
            if !other_nodes.contains(&other) {
                other_nodes.push(other);
            }
        }

        for other in other_nodes {
            let other_node = self.node(other);
            let other_still_connected = other_node
                .incoming_lanes()
                .iter()
                .chain(other_node.outgoing_lanes())
                .any(|&lid| !deleted_lanes.contains(&lid));

            if other_still_connected {
                if !nodes_needing_regen.contains(&other) {
                    nodes_needing_regen.push(other);
                }
            } else if !deleted_nodes.contains(&other) {
                deleted_nodes.push(other);
            }
        }

        DeleteImpact {
            nodes: deleted_nodes,
            segments: deleted_segments,
            lanes: deleted_lanes,
            nodes_needing_regen,
        }
    }

    pub fn impact_of_deleting_segment(
        &self,
        id: SegmentId,
        remove_orphan_nodes: bool,
    ) -> DeleteImpact {
        let seg = self.segment(id);

        let lane_ids: Vec<LaneId> = seg.lanes.clone();
        let endpoints = if remove_orphan_nodes {
            vec![seg.start, seg.end]
        } else {
            vec![]
        };

        let mut deleted_nodes = Vec::new();
        let mut nodes_needing_regen = Vec::new();

        for &node_id in &endpoints {
            let Some(node) = self.node_safe(node_id) else {
                continue;
            };
            let still_connected = node
                .incoming_lanes()
                .iter()
                .chain(node.outgoing_lanes())
                .any(|&lid| !lane_ids.contains(&lid));

            if still_connected {
                if !nodes_needing_regen.contains(&node_id) {
                    nodes_needing_regen.push(node_id);
                }
            } else if !deleted_nodes.contains(&node_id) {
                deleted_nodes.push(node_id);
            }
        }

        DeleteImpact {
            nodes: deleted_nodes,
            segments: vec![id],
            lanes: lane_ids,
            nodes_needing_regen,
        }
    }

    pub fn impact_of_deleting_lane(&self, id: LaneId, delete_orphans: bool) -> DeleteImpact {
        let lane = self.lane(id);

        let from_id = lane.from;
        let to_id = lane.to;
        let seg_id = lane.segment;

        // Segment dies if this is its last enabled lane.
        let segment = self.segment(seg_id);
        let segment_also_dies = segment
            .lanes
            .iter()
            .all(|&lid| lid == id || !self.lane_exists(lid))
            && delete_orphans;

        let mut deleted_nodes = Vec::new();
        let mut nodes_needing_regen = Vec::new();

        if delete_orphans {
            for &node_id in &[from_id, to_id] {
                let node = self.node(node_id);
                let still_connected = node
                    .incoming_lanes()
                    .iter()
                    .chain(node.outgoing_lanes())
                    .any(|&lid| lid != id && self.lane_exists(lid));

                if still_connected {
                    if !nodes_needing_regen.contains(&node_id) {
                        nodes_needing_regen.push(node_id);
                    }
                } else if !deleted_nodes.contains(&node_id) {
                    deleted_nodes.push(node_id);
                }
            }
        }

        DeleteImpact {
            nodes: deleted_nodes,
            segments: if segment_also_dies {
                vec![seg_id]
            } else {
                vec![]
            },
            lanes: vec![id],
            nodes_needing_regen,
        }
    }
    fn apply_impact(&mut self, impact: &DeleteImpact) {
        for &node_id in &impact.nodes {
            self.nodes[node_id.index()] = None;
        }
        for &seg_id in &impact.segments {
            self.segments[seg_id.index()] = None;
        }
        for &lane_id in &impact.lanes {
            self.lanes[lane_id.index()] = None;
        }
    }

    pub fn lane_ref_closest_to_pos(&self, pos: WorldPos) -> Option<LaneRef> {
        let chunk = pos.chunk;

        // Stage 1: only current chunk
        if let Some(r) = self.search_lane_ref_in_chunks(pos, std::iter::once(chunk)) {
            return Some(r);
        }

        // Stage 2: broader search (3x3)
        let broader_chunks = chunk.get_chunks_3x3();
        self.search_lane_ref_in_chunks(pos, broader_chunks)
    }

    fn search_lane_ref_in_chunks<I>(&self, pos: WorldPos, chunks: I) -> Option<LaneRef>
    where
        I: IntoIterator<Item = ChunkCoord>,
    {
        let mut best_dist = f64::INFINITY;
        let mut best_ref = None;

        let mut segment_ids = Vec::new();

        for chunk in chunks {
            segment_ids.extend(self.segment_ids_touching_chunk(chunk));
        }

        for &seg_id in segment_ids.iter() {
            let segment = self.segment(seg_id);

            for &lane_id in segment.lanes.iter() {
                let lane = self.lane(lane_id);
                let projection = lane.geometry().closest_point_to_xz(pos);

                if projection.distance < best_dist {
                    best_dist = projection.distance;
                    best_ref = Some(LaneRef::Lane(lane_id, projection.poly_idx));
                }
            }
        }

        for node_id in segment_ids
            .iter()
            .flat_map(|&seg_id| self.segment(seg_id).nodes())
        {
            let node = self.node(node_id);

            for node_lane in node.node_lanes() {
                let projection = node_lane.geometry().closest_point_to_xz(pos);

                if projection.distance < best_dist {
                    best_dist = projection.distance;
                    best_ref = Some(LaneRef::NodeLane(
                        node_id,
                        node_lane.id,
                        projection.poly_idx,
                    ));
                }
            }
        }

        best_ref
    }
    pub fn segmentlane_ref_closest_to_pos(&self, pos: WorldPos) -> Option<(LaneId, PolyIdx)> {
        let chunk = pos.chunk;

        // Stage 1: only current chunk
        if let Some(r) = self.search_segmentlane_ref_in_chunks(pos, std::iter::once(chunk)) {
            return Some(r);
        }

        // Stage 2: broader search (3x3)
        let broader_chunks = chunk.get_chunks_3x3();
        self.search_segmentlane_ref_in_chunks(pos, broader_chunks)
    }
    fn search_segmentlane_ref_in_chunks<I>(
        &self,
        pos: WorldPos,
        chunks: I,
    ) -> Option<(LaneId, PolyIdx)>
    where
        I: IntoIterator<Item = ChunkCoord>,
    {
        let mut best_dist = f64::INFINITY;
        let mut best_ref = None;

        let mut segment_ids = Vec::new();

        for chunk in chunks {
            segment_ids.extend(self.segment_ids_touching_chunk(chunk));
        }

        for &seg_id in segment_ids.iter() {
            let segment = self.segment(seg_id);

            for &lane_id in segment.lanes.iter() {
                let lane = self.lane(lane_id);
                let projection = lane.geometry().closest_point_to_xz(pos);

                if projection.distance < best_dist {
                    best_dist = projection.distance;
                    best_ref = Some((lane_id, projection.poly_idx));
                }
            }
        }

        best_ref
    }

    // ONLY in preview!!
    pub fn add_raw(
        &mut self,
        nodes: Vec<(NodeId, Node)>,
        segments: Vec<(SegmentId, Segment)>,
        lanes: Vec<(LaneId, Lane)>,
        nodes_needing_regen: Vec<(NodeId, Node)>,
    ) {
        // ── validated remap tables ────────────────────────────────────────────────
        // Each table only contains IDs that will actually be inserted.
        // Anything absent = dangling = will be dropped by filter_map later.

        // Nodes: always valid, they are the roots.
        let remap_node: HashMap<NodeId, NodeId> = nodes
            .iter()
            .chain(nodes_needing_regen.iter())
            .enumerate()
            .map(|(i, (old, _))| (*old, self.alloc_node_id())) // Fuck. Allocation needs to fill or else it will recycle the same shit idx
            .collect();

        // Segments: valid only if both endpoints exist in the node batch.
        let remap_seg: HashMap<SegmentId, SegmentId> = {
            let mut map = HashMap::new();
            for (old_id, seg) in &segments {
                if remap_node.contains_key(&seg.start) && remap_node.contains_key(&seg.end) {
                    map.insert(*old_id, self.alloc_segment_id());
                }
            }
            map
        };

        // Lanes: valid only if from, to, AND parent segment are all in the batch.
        let remap_lane: HashMap<LaneId, LaneId> = {
            let mut map = HashMap::new();
            for (old_id, lane) in &lanes {
                if remap_node.contains_key(&lane.from)
                    && remap_node.contains_key(&lane.to)
                    && remap_seg.contains_key(&lane.segment)
                {
                    map.insert(*old_id, self.alloc_lane_id());
                }
            }
            map
        };

        // ── insert nodes ──────────────────────────────────────────────────────────
        for (_, mut node) in nodes.into_iter().chain(nodes_needing_regen.into_iter()) {
            // OMG so stupid! I forgot to add it here!!
            node.incoming_lanes = node
                .incoming_lanes
                .iter()
                .filter_map(|id| remap_lane.get(id).copied())
                .collect();
            node.outgoing_lanes = node
                .outgoing_lanes
                .iter()
                .filter_map(|id| remap_lane.get(id).copied())
                .collect();

            node.arms
                .retain(|arm| remap_seg.contains_key(&arm.segment_id));
            for arm in &mut node.arms {
                arm.segment_id = remap_seg[&arm.segment_id];
                arm.incoming_lanes = arm
                    .incoming_lanes
                    .iter()
                    .filter_map(|id| remap_lane.get(id).copied())
                    .collect();
                arm.outgoing_lanes = arm
                    .outgoing_lanes
                    .iter()
                    .filter_map(|id| remap_lane.get(id).copied())
                    .collect();
            }

            node.node_lanes.retain(|nl| {
                let ref_ok = |lr: &LaneRef| match lr {
                    LaneRef::Lane(lid, _) => remap_lane.contains_key(lid),
                    LaneRef::NodeLane(_, _, _) => true,
                };
                nl.merging.iter().all(ref_ok) && nl.splitting.iter().all(ref_ok)
            });
            for nl in &mut node.node_lanes {
                for lr in nl.merging.iter_mut().chain(nl.splitting.iter_mut()) {
                    if let LaneRef::Lane(lid, _) = lr {
                        if let Some(&new_id) = remap_lane.get(lid) {
                            *lid = new_id;
                        }
                    }
                }
            }

            let node_idx = self.alloc_node_id();
            self.nodes[node_idx.index()] = Some(node);

            let region_id = if let Some(reused) = self.free_regions.pop() {
                self.regions[reused as usize].nodes.push(node_idx);
                reused
            } else {
                let new_id = self.regions.len() as RoadRegionId;
                let mut r = RoadRegion::new();
                r.nodes.push(node_idx);
                self.regions.push(r);
                new_id
            };
            self.node_to_region.push(region_id);
            self.active_region_count += 1;
        }

        // ── insert segments (skip any that failed validation) ─────────────────────
        for (old_id, mut seg) in segments {
            if !remap_seg.contains_key(&old_id) {
                continue;
            }

            seg.start = remap_node[&seg.start()];
            seg.end = remap_node[&seg.end()];
            seg.lanes = seg
                .lanes
                .iter()
                .filter_map(|id| remap_lane.get(id).copied())
                .collect();

            self.segments[remap_seg[&old_id].index()] = Some(seg);

            let idx = SegmentId((self.segments.len() - 1) as u32);
            let ra = self.node_to_region[self.segment(idx).start.index()];
            let rb = self.node_to_region[self.segment(idx).end.index()];
            if ra != rb {
                self.merge_regions(ra, rb);
            }
        }

        // ── insert lanes (skip any that failed validation) ────────────────────────
        for (old_id, mut lane) in lanes {
            if !remap_lane.contains_key(&old_id) {
                continue;
            }

            lane.from = remap_node[&lane.from];
            lane.to = remap_node[&lane.to];
            lane.segment = remap_seg[&lane.segment];
            self.lanes[remap_lane[&old_id].index()] = Some(lane);
        }
    }

    // When the terrain changes, so must the roads.
    // pub fn update_heights_in_chunk(
    //     &mut self,
    //     chunks: &mut HashMap<ChunkCoord, ChunkMeshLod>,
    //     terrain_gen: &TerrainGenerator,
    //     coord: ChunkCoord
    // ) {
    //     let seg_ids: Vec<SegmentId> = self.segment_ids_touching_chunk(coord);
    //
    //     for seg_id in seg_ids {
    //         let seg = &self.segments[seg_id.0 as usize];
    //         for lane_id in &seg.lanes {
    //             let lane = &mut self.lanes[lane_id.0 as usize];
    //             lane.geometry.update_heights(chunks, terrain_gen);
    //         }
    //         let node = &mut self.nodes[seg.start.0 as usize];
    //         node.update_heights(chunks, terrain_gen);
    //     }
    // }
}

#[derive(Clone, Serialize, Deserialize, Default)]
#[revisioned(revision = 1)]
pub struct RoadTypes {
    road_types: HashMap<RoadTypeId, RoadType>,
}
impl RoadTypes {
    pub fn new() -> Self {
        Self {
            road_types: HashMap::new(),
        }
    }
    pub fn get_road_type(&self, id: RoadTypeId) -> Option<&RoadType> {
        self.road_types.get(&id)
    }
    pub fn change_road_type(&mut self, road_type: RoadType) -> RoadTypeId {
        let bytes = postcard::to_stdvec(&road_type).unwrap_or_default();
        let key = xxh3_64(&bytes) as u32;
        self.road_types.insert(key, road_type);
        key
    }

    pub fn add_road_type(&mut self, road_type: &RoadType) -> RoadTypeId {
        let bytes = postcard::to_stdvec(road_type).unwrap_or_default();
        let key = xxh3_64(&bytes) as u32;
        if !self.road_types.contains_key(&key) {
            self.road_types.insert(key, road_type.clone());
        }
        key
    }
}
/// Global road topology manager with append-only storage.
///
/// # Thread Safety
/// RoadManager is `Send + Sync` for read-only access during simulation.
/// Mutable operations must be serialized and occur outside simulation ticks.
///
/// # Determinism
/// All ID allocation is monotonic and deterministic.
/// Iteration order is stable and matches insertion order.
pub struct RoadManager {
    pub roads: RoadStorage,
    pub preview_roads: RoadStorage,
    pub road_types: RoadTypes,
}

// Safety: RoadManager uses no interior mutability.
// All mutable access is explicitly controlled by the caller.
impl RoadManager {
    /// Creates an empty road topology.
    pub fn new() -> Self {
        Self {
            roads: RoadStorage::default(),
            preview_roads: RoadStorage::default(),
            road_types: RoadTypes::new(),
        }
    }
}

impl Default for RoadManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Sample position along a lane at parameter t in [0,1].
/// Returns the WorldPos on the lane.
#[inline]
pub fn sample_lane_position(lane_geometry: &LaneGeometry, t: f64) -> Option<WorldPos> {
    lane_geometry.sample_at_t(t)
}
pub fn nearest_lane_to_point(storage: &RoadStorage, point: WorldPos) -> Option<LaneId> {
    let mut best_id = None;
    let mut best_dist_sq = f64::INFINITY;

    for (id, lane) in storage.iter_lanes() {
        let Some((_, dist_sq)) = project_point_to_lane_xz(lane, point, storage) else {
            continue;
        };

        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best_id = Some(id);
        }
    }

    best_id
}

/// Project a WorldPos onto a lane and returns (t, distance_squared).
/// t is the parameter [0,1] along the lane; dist_sq is squared XZ distance.
#[inline]
pub fn project_point_to_lane_xz(
    lane: &Lane,
    point: WorldPos,
    _storage: &RoadStorage,
) -> Option<(f64, f64)> {
    let projection = lane.geometry().closest_point_to_xz(point);
    Some((projection.t, projection.distance * projection.distance))
}

/// Commands for deterministic road topology modification.
///
/// Commands are applied atomically and can be serialized for replay.
/// The engine should queue commands and apply them between simulation ticks.
///
/// # Invariants
/// - Commands must be applied in deterministic order
/// - AddNode/AddSegment/AddLane commands produce monotonically increasing IDs
/// - delete commands are idempotent
/// - UpgradeSegmentBegin must be followed by UpgradeSegmentEnd
#[derive(Debug, Clone)]
pub enum RoadCommand {
    /// Add a new intersection node.
    AddNode {
        id: NodeId,
        world_pos: WorldPos,
    },
    AddNodeFull {
        id: NodeId,
        node: Node,
    },
    /// Add a new road segment.
    AddSegment {
        id: SegmentId,
        start: NodeId,
        end: NodeId,
        structure: StructureType,
        chunk_coord: ChunkCoord,
        road_type_id: RoadTypeId,
    },
    /// Add a new lane to a segment.
    AddLane {
        id: LaneId,
        from: NodeId,
        to: NodeId,
        segment: SegmentId,
        lane_index: i8,
        geometry: LaneGeometry,
        speed_limit: f32,
        capacity: u32,
        vehicle_mask: u32,
        chunk_coord: ChunkCoord,
    },
    /// Add a new lane to a segment.
    AddNodeLane {
        node_id: NodeId,
        merging: Vec<LaneRef>,
        splitting: Vec<LaneRef>,
        geometry: LaneGeometry,
        speed_limit: f32,
        vehicle_mask: u32,
        chunk_coord: ChunkCoord,
    },
    AddRaw {
        nodes: Vec<(NodeId, Node)>,
        segments: Vec<(SegmentId, Segment)>,
        lanes: Vec<(LaneId, Lane)>,
        nodes_needing_regen: Vec<(NodeId, Node)>,
    },
    ClearNodeLanes {
        node_id: NodeId,
        chunk_coord: ChunkCoord,
    },
    /// delete a node.
    DeleteNode {
        node_id: NodeId,
        chunk_coord: ChunkCoord,
    },
    /// delete a segment and its lanes.
    DeleteSegment {
        chunk_coord: ChunkCoord,
        segment_id: SegmentId,
        remove_orphan_nodes: bool,
    },
    /// delete a lane.
    DeleteLane {
        chunk_coord: ChunkCoord,
        lane_id: LaneId,
        delete_orphans: bool,
    },
    /// Attach a traffic control to a node.
    AttachControl {
        node_id: NodeId,
        chunk_coord: ChunkCoord,
        control: TrafficControl,
    },
    /// delete a traffic control.
    DeleteControl {
        node_id: NodeId,
        chunk_coord: ChunkCoord,
        control_id: ControlId,
    },
    /// Enable a traffic control.
    EnableControl {
        node_id: NodeId,
        chunk_coord: ChunkCoord,
        control_id: ControlId,
    },
    /// Procedurally rebuild node lanes using *current* incoming/outgoing segment lanes.
    MakeIntersection {
        node_id: NodeId,
        intersection_params: IntersectionBuildParams,
        chunk_coord: ChunkCoord,
        recalc_clearance: bool,
    },
    /// Begin segment upgrade (deletes old segment).
    UpgradeSegmentBegin {
        old_segment: SegmentId,
        chunk_coord: ChunkCoord,
    },
    /// End segment upgrade (records new segment IDs for replay).
    UpgradeSegmentEnd {
        new_segments: Vec<SegmentId>,
        chunk_coord: ChunkCoord,
    },
    ReplaceNode {
        old_node_id: NodeId,
        new_node: Node,
        chunk_coord: ChunkCoord,
    },
}
impl RoadCommand {
    /// Returns the chunk ID affected by this command, if one is explicitly stored.
    /// For `AddNode`, the chunk is derived from `world_pos` and must be computed separately.
    pub fn chunk_coord(&self) -> ChunkCoord {
        match self {
            RoadCommand::AddNode { world_pos, .. } => world_pos.chunk,
            RoadCommand::AddSegment { chunk_coord, .. } => *chunk_coord,
            RoadCommand::AddLane { chunk_coord, .. } => *chunk_coord,
            RoadCommand::AddNodeLane { chunk_coord, .. } => *chunk_coord,
            RoadCommand::AddRaw { .. } => ChunkCoord::zero(),
            RoadCommand::ClearNodeLanes { chunk_coord, .. } => *chunk_coord,
            RoadCommand::DeleteNode { chunk_coord, .. } => *chunk_coord,
            RoadCommand::DeleteSegment { chunk_coord, .. } => *chunk_coord,
            RoadCommand::DeleteLane { chunk_coord, .. } => *chunk_coord,
            RoadCommand::AttachControl { chunk_coord, .. } => *chunk_coord,
            RoadCommand::DeleteControl { chunk_coord, .. } => *chunk_coord,
            RoadCommand::EnableControl { chunk_coord, .. } => *chunk_coord,
            RoadCommand::MakeIntersection { chunk_coord, .. } => *chunk_coord,
            RoadCommand::UpgradeSegmentBegin { chunk_coord, .. } => *chunk_coord,
            RoadCommand::UpgradeSegmentEnd { chunk_coord, .. } => *chunk_coord,
            RoadCommand::ReplaceNode { chunk_coord, .. } => *chunk_coord,
            RoadCommand::AddNodeFull { node, .. } => node.chunk_coord(),
        }
    }
}

/// Result of applying a command.
#[derive(Debug, Clone)]
pub enum CommandResult {
    /// Node was created with this ID.
    NodeCreated(ChunkCoord, NodeId),
    /// Segment was created with this ID.
    SegmentCreated(ChunkCoord, SegmentId),
    /// Lane was created with this ID.
    LaneCreated(ChunkCoord, LaneId),
    /// NodeLane was created with this ID.
    NodeLaneCreated(ChunkCoord, NodeLaneId),
    /// Control was attached with this ID.
    ControlAttached(ChunkCoord, ControlId),
    /// Command applied with no new IDs.
    Ok,
    /// Command failed (invalid reference).
    InvalidReference,
}

/// Applies only the world-state mutations from real (non-preview) commands.
/// No mesh rebuilding — that's the render subsystem's job.
pub fn apply_road_commands_real(
    terrain: &mut Terrain,
    road_manager: &mut RoadManager,
    car_subsystem: &mut Cars,
    settings: &Settings,
    gizmo: &mut Gizmo,
    commands: &[RoadEditorCommand],
) {
    for cmd in commands {
        if let RoadEditorCommand::Road(road_command) = cmd {
            apply_road_command(
                terrain,
                road_manager,
                car_subsystem,
                settings,
                gizmo,
                road_command,
                false,
            );
        }
    }
}

/// Applies world-state mutations for preview commands (populates preview_storage).
pub fn apply_road_commands_preview(
    terrain: &mut Terrain,
    roads: &mut Roads,
    car_subsystem: &mut Cars,
    settings: &Settings,
    gizmo: &mut Gizmo,
    sounds: &mut Sounds,
) {
    roads.road_manager.preview_roads.clear();

    // 1) Apply explicit Road commands to preview storage
    // for cmd in &roads.road_commands {
    //     if let RoadEditorCommand::Road(road_command) = cmd {
    //         apply_road_command(
    //             terrain,
    //             &mut roads.road_manager,
    //             car_subsystem,
    //             settings,
    //             gizmo,
    //             road_command,
    //             true,
    //         );
    //     }
    // }

    // 2) Collect preview inputs
    let mut node_previews: Vec<&NodePreview> = Vec::new();
    let mut crossing_previews: Vec<&CrossingPoint> = Vec::new();
    let mut segment_preview: Option<&SegmentPreview> = None;
    let mut destruction_preview: Option<&RoadDestroyType> = None;
    let mut snap_preview: Option<&SnapPreview> = None;

    for cmd in &roads.road_commands {
        match cmd {
            RoadEditorCommand::PreviewNode(n) => node_previews.push(n),
            RoadEditorCommand::PreviewSegment(s) => segment_preview = Some(s),
            RoadEditorCommand::PreviewCrossing(c) => crossing_previews.push(c),
            RoadEditorCommand::PreviewDestruction(d) => destruction_preview = Some(d),
            RoadEditorCommand::PreviewClear => return,
            RoadEditorCommand::PreviewSnap(s) => snap_preview = Some(s),
            _ => {}
        }
    }
    let mut road_commands: Vec<RoadCommand> = Vec::new();

    // 3) Crossing preview
    if !crossing_previews.is_empty() {
        road_commands.extend(generate_intersection_preview(
            terrain,
            &mut roads.road_manager,
            &roads.road_editor.style,
            &crossing_previews,
        ));
    }

    // 4) Segment preview
    if let Some(seg) = segment_preview {
        if (roads.road_editor.last_road_length - seg.estimated_length).abs() > 4.0
            && !roads.road_editor.style.is_idle()
        {
            roads.road_editor.last_road_length = seg.estimated_length;
            sounds.queue_sfx("DragRoad");
        }
        if seg.is_valid {
            road_commands.extend(generate_segment_preview(
                terrain,
                &mut roads.road_manager,
                &roads.road_editor.style,
                seg,
            ));
        } else {
            road_commands.extend(generate_invalid_segment_preview(
                terrain,
                &mut roads.road_manager,
                &roads.road_editor.style,
                seg,
            ));
        }
    }

    // 5) Hover nodes
    if segment_preview.is_none() && !node_previews.is_empty() {
        road_commands.extend(generate_hover_preview(
            terrain,
            &mut roads.road_manager,
            &roads.road_editor.style,
            &node_previews,
        ));
    }
    if let Some(road_destroy_type) = destruction_preview {
        road_commands.extend(generate_destruction_preview(
            terrain,
            &roads.road_manager,
            road_destroy_type,
        ));
    }
    if let Some(snap_preview) = snap_preview {
        //gizmo.snap_preview(snap_preview);
        match snap_preview.kind {
            SnapKind::Free => {}
            SnapKind::Node { id } => {}
            SnapKind::Lane { lane_id, .. } => {
                let lane_id = roads.road_manager.roads.lane(lane_id).segment();
                // let segment = roads.road_manager.roads.segment(lane.segment());
                // let segment_preview = SegmentPreview {
                //     start: roads.road_manager.roads.node(lane.from).pos,
                //     end: roads.road_manager.roads.node(lane.to).pos,
                //     polyline: lane.polyline().clone(),
                //     ..Default::default()
                // };
                road_commands.extend(generate_segment_preview_segment(
                    terrain,
                    &mut roads.road_manager,
                    lane_id,
                ));
            }
        }
    }
    // 6) Apply generated preview commands to preview storage (world-only)
    for cmd in road_commands {
        apply_road_command(
            terrain,
            &mut roads.road_manager,
            car_subsystem,
            settings,
            gizmo,
            &cmd,
            true,
        );
    }
}

/// Single command application — world state only, no mesh rebuild.
pub fn apply_road_command(
    terrain: &mut Terrain,
    road_manager: &mut RoadManager,
    cars: &mut Cars,
    settings: &Settings,
    gizmo: &mut Gizmo,
    road_command: &RoadCommand,
    is_preview: bool,
) -> CommandResult {
    let storage = if is_preview {
        &mut road_manager.preview_roads
    } else {
        &mut road_manager.roads
    };
    let road_types = &road_manager.road_types;
    match road_command {
        RoadCommand::AddNode { id, world_pos } => {
            //println!("{:?} {}", id, is_preview);
            storage.add_node(*id, *world_pos);
            if !is_preview {
                cars.add_spawning_node(*id);
            }
            let chunk_coord = world_pos.chunk;
            CommandResult::NodeCreated(chunk_coord, *id)
        }
        RoadCommand::AddSegment {
            id,
            start,
            end,
            structure,
            chunk_coord,
            road_type_id,
        } => {
            if start.raw() as usize >= storage.node_count()
                || end.raw() as usize >= storage.node_count()
            {
                return CommandResult::InvalidReference;
            }
            storage.add_segment(*id, *start, *end, structure.clone(), *road_type_id);
            CommandResult::SegmentCreated(*chunk_coord, *id)
        }
        RoadCommand::AddLane {
            id,
            from,
            to,
            segment,
            lane_index,
            geometry,
            speed_limit,
            capacity,
            vehicle_mask,
            chunk_coord,
        } => {
            if from.raw() as usize >= storage.node_count()
                || to.raw() as usize >= storage.node_count()
                || segment.raw() as usize >= storage.segment_count()
            {
                return CommandResult::InvalidReference;
            }
            storage.add_lane(
                *id,
                *from,
                *to,
                *segment,
                *lane_index,
                geometry.clone(),
                *speed_limit,
                *capacity,
                *vehicle_mask,
            );
            CommandResult::LaneCreated(*chunk_coord, *id)
        }
        RoadCommand::AddNodeLane {
            node_id,
            merging,
            splitting,
            geometry,
            speed_limit,
            vehicle_mask,
            chunk_coord,
        } => {
            let id = storage.add_node_lane(
                *node_id,
                merging.clone(),
                splitting.clone(),
                geometry.clone(),
                *speed_limit,
                *vehicle_mask,
            );
            CommandResult::NodeLaneCreated(*chunk_coord, id)
        }
        RoadCommand::AddRaw {
            nodes,
            segments,
            lanes,
            nodes_needing_regen,
        } => {
            storage.add_raw(
                nodes.clone(),
                segments.clone(),
                lanes.clone(),
                nodes_needing_regen.clone(),
            );
            CommandResult::Ok
        }
        RoadCommand::ClearNodeLanes {
            node_id,
            chunk_coord,
        } => {
            let node = storage.node_mut(*node_id);
            node.node_lanes.clear();
            CommandResult::Ok
        }
        RoadCommand::DeleteNode {
            node_id,
            chunk_coord,
        } => {
            if node_id.raw() as usize >= storage.node_count() {
                return CommandResult::InvalidReference;
            }
            storage.delete_node(*node_id, road_types, gizmo);
            if !is_preview {
                cars.remove_spawning_node(*node_id);
            }
            CommandResult::Ok
        }
        RoadCommand::DeleteSegment {
            segment_id,
            chunk_coord,
            remove_orphan_nodes,
        } => {
            if segment_id.raw() as usize >= storage.segment_count() {
                return CommandResult::InvalidReference;
            }
            storage.delete_segment(*segment_id, road_types, gizmo, *remove_orphan_nodes);
            CommandResult::Ok
        }
        RoadCommand::DeleteLane {
            lane_id,
            chunk_coord,
            delete_orphans,
        } => {
            if lane_id.raw() as usize >= storage.lane_count() {
                return CommandResult::InvalidReference;
            }
            storage.delete_lane(*lane_id, road_types, gizmo, *delete_orphans);
            CommandResult::Ok
        }
        RoadCommand::AttachControl {
            node_id,
            chunk_coord,
            control,
        } => {
            if node_id.raw() as usize >= storage.node_count() {
                return CommandResult::InvalidReference;
            }
            let id = storage.attach_control(*node_id, control.clone());
            CommandResult::ControlAttached(*chunk_coord, id)
        }
        RoadCommand::DeleteControl {
            node_id,
            control_id,
            chunk_coord,
        } => {
            if node_id.raw() as usize >= storage.node_count() {
                return CommandResult::InvalidReference;
            }
            storage.delete_control(*node_id, *control_id);
            CommandResult::Ok
        }
        RoadCommand::EnableControl {
            node_id,
            control_id,
            chunk_coord,
        } => {
            if node_id.raw() as usize >= storage.node_count() {
                return CommandResult::InvalidReference;
            }
            storage.enable_control(*node_id, *control_id);
            CommandResult::Ok
        }
        RoadCommand::MakeIntersection {
            node_id,
            intersection_params,
            chunk_coord,
            recalc_clearance: clear,
        } => {
            if node_id.raw() as usize >= storage.node_count() {
                return CommandResult::InvalidReference;
            }
            let arms = gather_arms(storage, road_types, *node_id, gizmo);

            storage.node_mut(*node_id).arms = arms;

            build_intersection_at_node(
                terrain,
                storage,
                road_types,
                *node_id,
                intersection_params,
                *clear,
                settings,
                gizmo,
            );
            // Twice is correct
            let arms = gather_arms(storage, road_types, *node_id, gizmo);
            //println!("node {:?} exists: {}", node_id, storage.node(*node_id).is_some());
            storage.node_mut(*node_id).arms = arms;
            CommandResult::Ok
        }
        RoadCommand::UpgradeSegmentBegin {
            old_segment,
            chunk_coord,
        } => {
            storage.delete_segment(*old_segment, road_types, gizmo, false);
            CommandResult::Ok
        }
        RoadCommand::UpgradeSegmentEnd { chunk_coord, .. } => CommandResult::Ok,
        RoadCommand::ReplaceNode {
            chunk_coord,
            old_node_id,
            new_node,
        } => {
            let node = storage.node_mut(*old_node_id);
            let _ = replace(node, new_node.clone());
            CommandResult::Ok
        }
        RoadCommand::AddNodeFull {
            // TODO: Region tracking maybe
            id,
            node,
        } => {
            storage.nodes[id.index()] = Some(node.clone());
            //storage.add_node()
            CommandResult::Ok
        }
    }
}

/// Extracts all chunk IDs affected by real (non-preview) commands,
/// so the render subsystem knows which chunks to rebuild meshes for.
pub fn collect_affected_chunks(commands: &[RoadEditorCommand]) -> Vec<ChunkCoord> {
    let mut chunks = Vec::new();
    for cmd in commands {
        if let RoadEditorCommand::Road(road_cmd) = cmd {
            let chunk_coord = road_cmd.chunk_coord();
            chunks.push(chunk_coord);
        }
    }
    chunks.dedup();
    chunks
}

/// Applies a command to the road manager deterministically.
/// Returns the result of the operation.
///
/// # Panics
/// Panics only on programmer errors (debug assertions).
/// Applies a command to the road manager deterministically and updates the mesh.
pub fn apply_command(
    terrain: &mut Terrain,
    road_mesh_manager: &mut RoadMeshManager,
    car_subsystem: &mut Cars,
    road_manager: &mut RoadManager,
    road_style_params: &RoadStyleParams,
    command: RoadEditorCommand,
    is_preview: bool,
    settings: &Settings,
    gizmo: &mut Gizmo,
) -> CommandResult {
    let storage = if is_preview {
        &mut road_manager.preview_roads
    } else {
        &mut road_manager.roads
    };
    let road_types = &road_manager.road_types;
    match command {
        RoadEditorCommand::Road(road_command) => {
            // store the chunk ID here if an operation succeeds
            let mut affected_chunk: Option<ChunkCoord> = None;
            let result = match road_command {
                RoadCommand::AddNode { id, world_pos } => {
                    //println!("{:?} {}", id, is_preview);
                    storage.add_node(id, world_pos);
                    gather_arms(storage, road_types, id, gizmo);
                    if !is_preview {
                        car_subsystem.add_spawning_node(id);
                    }
                    let chunk_coord = world_pos.chunk;
                    affected_chunk = Some(chunk_coord);
                    CommandResult::NodeCreated(chunk_coord, id)
                }
                RoadCommand::AddSegment {
                    id,
                    start,
                    end,
                    structure,
                    chunk_coord,
                    road_type_id,
                } => {
                    if start.raw() as usize >= storage.node_count()
                        || end.raw() as usize >= storage.node_count()
                    {
                        return CommandResult::InvalidReference;
                    }

                    storage.add_segment(id, start, end, structure, road_type_id);
                    affected_chunk = Some(chunk_coord);
                    CommandResult::SegmentCreated(chunk_coord, id)
                }
                RoadCommand::AddLane {
                    id,
                    from,
                    to,
                    segment,
                    lane_index,
                    geometry,
                    speed_limit,
                    capacity,
                    vehicle_mask,
                    chunk_coord,
                } => {
                    if from.raw() as usize >= storage.node_count()
                        || to.raw() as usize >= storage.node_count()
                        || segment.raw() as usize >= storage.segment_count()
                    {
                        return CommandResult::InvalidReference;
                    }
                    storage.add_lane(
                        id,
                        from,
                        to,
                        segment,
                        lane_index,
                        geometry,
                        speed_limit,
                        capacity,
                        vehicle_mask,
                    );
                    affected_chunk = Some(chunk_coord);
                    CommandResult::LaneCreated(chunk_coord, id)
                }
                RoadCommand::AddNodeLane {
                    node_id,
                    merging,
                    splitting,
                    geometry,
                    speed_limit,
                    vehicle_mask,
                    chunk_coord,
                } => {
                    let id = storage.add_node_lane(
                        node_id,
                        merging,
                        splitting,
                        geometry,
                        speed_limit,
                        vehicle_mask,
                    );
                    affected_chunk = Some(chunk_coord);
                    CommandResult::NodeLaneCreated(chunk_coord, id)
                }
                RoadCommand::AddRaw {
                    nodes,
                    segments,
                    lanes,
                    nodes_needing_regen,
                } => {
                    storage.add_raw(nodes, segments, lanes, nodes_needing_regen);
                    CommandResult::Ok
                }
                RoadCommand::ClearNodeLanes {
                    node_id,
                    chunk_coord,
                } => {
                    let node = storage.node_mut(node_id);
                    node.node_lanes.clear();
                    affected_chunk = Some(chunk_coord);
                    CommandResult::Ok
                }
                RoadCommand::DeleteNode {
                    node_id,
                    chunk_coord,
                } => {
                    storage.delete_node(node_id, road_types, gizmo);
                    affected_chunk = Some(chunk_coord);
                    CommandResult::Ok
                }
                RoadCommand::DeleteSegment {
                    segment_id,
                    chunk_coord,
                    remove_orphan_nodes,
                } => {
                    storage.delete_segment(segment_id, road_types, gizmo, remove_orphan_nodes);
                    affected_chunk = Some(chunk_coord);
                    CommandResult::Ok
                }
                RoadCommand::DeleteLane {
                    lane_id,
                    chunk_coord,
                    delete_orphans,
                } => {
                    storage.delete_lane(lane_id, road_types, gizmo, delete_orphans);
                    affected_chunk = Some(chunk_coord);
                    CommandResult::Ok
                }
                RoadCommand::AttachControl {
                    node_id,
                    chunk_coord,
                    control,
                } => {
                    let id = storage.attach_control(node_id, control.clone());
                    affected_chunk = Some(chunk_coord);
                    CommandResult::ControlAttached(chunk_coord, id)
                }
                RoadCommand::DeleteControl {
                    node_id,
                    control_id,
                    chunk_coord,
                } => {
                    storage.delete_control(node_id, control_id);
                    affected_chunk = Some(chunk_coord);
                    CommandResult::Ok
                }
                RoadCommand::EnableControl {
                    node_id,
                    control_id,
                    chunk_coord,
                } => {
                    storage.enable_control(node_id, control_id);
                    affected_chunk = Some(chunk_coord);
                    CommandResult::Ok
                }
                RoadCommand::MakeIntersection {
                    node_id,
                    intersection_params: params,
                    chunk_coord,
                    recalc_clearance: clear,
                } => {
                    let arms = gather_arms(storage, road_types, node_id, gizmo);

                    storage.node_mut(node_id).arms = arms;

                    build_intersection_at_node(
                        terrain, storage, road_types, node_id, &params, clear, settings, gizmo,
                    );
                    // Twice is correct
                    let arms = gather_arms(storage, road_types, node_id, gizmo);

                    storage.node_mut(node_id).arms = arms;
                    affected_chunk = Some(chunk_coord);
                    CommandResult::Ok
                }
                RoadCommand::UpgradeSegmentBegin {
                    old_segment,
                    chunk_coord,
                } => {
                    storage.delete_segment(old_segment, road_types, gizmo, false);
                    affected_chunk = Some(chunk_coord);
                    CommandResult::Ok
                }
                RoadCommand::UpgradeSegmentEnd { chunk_coord, .. } => {
                    affected_chunk = Some(chunk_coord);
                    CommandResult::Ok
                }
                RoadCommand::ReplaceNode {
                    chunk_coord,
                    old_node_id,
                    new_node,
                } => {
                    let node = storage.node_mut(old_node_id);
                    let _ = replace(node, new_node);
                    affected_chunk = Some(chunk_coord);
                    CommandResult::Ok
                }
                RoadCommand::AddNodeFull { id, node } => {
                    affected_chunk = Some(node.chunk_coord());
                    storage.nodes[id.index()] = Some(node);
                    CommandResult::Ok
                }
            };

            // Only update the mesh if this is not a preview and the command succeeded (chunk ID was set)
            if !is_preview {
                if let Some(chunk_coord) = affected_chunk {
                    road_mesh_manager.update_chunk_mesh(
                        terrain,
                        chunk_coord,
                        road_manager,
                        road_style_params,
                        gizmo,
                    );
                }
            }

            result
        }
        _ => CommandResult::Ok,
    }
}

fn generate_intersection_preview(
    terrain_renderer: &Terrain,
    road_manager: &mut RoadManager,
    road_style_params: &RoadStyleParams,
    crossings: &[&CrossingPoint],
) -> Vec<RoadCommand> {
    let mut commands = Vec::new();

    for &crossing in crossings {
        generate_node_with_stub_preview(
            terrain_renderer,
            road_manager,
            road_style_params,
            crossing.pos,
            &mut commands,
        );
    }

    commands
}

/// Generate preview for hover state - creates node with stub lanes so it renders
fn generate_hover_preview(
    terrain_renderer: &Terrain,
    road_manager: &mut RoadManager,
    road_style_params: &RoadStyleParams,
    node_previews: &[&NodePreview],
) -> Vec<RoadCommand> {
    let mut commands = Vec::new();

    for node in node_previews {
        generate_node_with_stub_preview(
            terrain_renderer,
            road_manager,
            road_style_params,
            node.world_pos,
            &mut commands,
        );
    }

    commands
}

fn generate_destruction_preview(
    terrain_renderer: &Terrain,
    road_manager: &RoadManager,
    road_destroy_type: &RoadDestroyType,
) -> Vec<RoadCommand> {
    let mut commands = Vec::new();
    let storage = &road_manager.roads;
    let impact = match road_destroy_type {
        RoadDestroyType::Segment(id) => storage.impact_of_deleting_segment(*id, true),
        RoadDestroyType::Node(id) => storage.impact_of_deleting_node(*id),
    };

    commands.push(RoadCommand::AddRaw {
        nodes: impact
            .nodes
            .into_iter()
            .map(|node_id| (node_id, road_manager.roads.node(node_id).clone()))
            .collect(),
        segments: impact
            .segments
            .into_iter()
            .map(|id| (id, storage.segment(id).clone()))
            .collect(),
        lanes: impact
            .lanes
            .into_iter()
            .map(|id| (id, storage.lane(id).clone()))
            .collect(),
        nodes_needing_regen: impact
            .nodes_needing_regen
            .into_iter()
            .map(|node_id| (node_id, storage.node(node_id).clone()))
            .collect(),
    });
    commands
}

/// Generate preview for invalid segment - shows both endpoints with stubs
fn generate_invalid_segment_preview(
    terrain_renderer: &Terrain,
    road_manager: &mut RoadManager,
    road_style_params: &RoadStyleParams,
    preview: &SegmentPreview,
) -> Vec<RoadCommand> {
    let mut commands = Vec::new();

    // Start node with stub pointing toward end
    generate_node_with_stub_preview(
        terrain_renderer,
        road_manager,
        road_style_params,
        preview.start,
        &mut commands,
    );

    let reason = preview.reason_invalid.clone().unwrap();
    if !matches!(reason, PreviewError::TooShort) {
        // End node with stub pointing toward start
        generate_node_with_stub_preview(
            terrain_renderer,
            road_manager,
            road_style_params,
            preview.end,
            &mut commands,
        );
    }

    commands
}

/// Generate full segment preview with both nodes and all lanes
fn generate_segment_preview(
    terrain_renderer: &Terrain,
    road_manager: &mut RoadManager,
    road_style_params: &RoadStyleParams,
    preview: &SegmentPreview,
) -> Vec<RoadCommand> {
    let mut commands = Vec::new();
    let Some(road_type) = road_style_params.road_type(&road_manager.road_types) else {
        return Vec::new();
    };
    let storage = &mut road_manager.preview_roads;

    let lane_defs = compute_lane_geometries(terrain_renderer, road_type, &preview.polyline);

    let start_node_id = storage.alloc_node_id();
    commands.push(RoadCommand::AddNode {
        id: start_node_id,
        world_pos: preview.start,
    });

    let end_node_id = storage.alloc_node_id();
    commands.push(RoadCommand::AddNode {
        id: end_node_id,
        world_pos: preview.end,
    });

    let segment_id = storage.alloc_segment_id();
    commands.push(RoadCommand::AddSegment {
        id: segment_id,
        start: start_node_id,
        end: end_node_id,
        structure: road_type.structure(),
        chunk_coord: preview.start.chunk,
        road_type_id: road_style_params.road_type_id(),
    });

    let speed = road_type.speed_limit();
    let capacity = road_type.capacity();
    let mask = road_type.vehicle_mask();

    for lane_def in lane_defs {
        let lane_id = storage.alloc_lane_id();

        let (from, to) = if lane_def.is_forward {
            (start_node_id, end_node_id)
        } else {
            (end_node_id, start_node_id)
        };

        commands.push(RoadCommand::AddLane {
            id: lane_id,
            from,
            to,
            segment: segment_id,
            lane_index: lane_def.lane_index,
            geometry: lane_def.geometry,
            speed_limit: speed,
            capacity,
            vehicle_mask: mask,
            chunk_coord: preview.start.chunk,
        });
    }

    commands
}
fn generate_segment_preview_segment(
    terrain_renderer: &Terrain,
    road_manager: &mut RoadManager,
    segment_id: SegmentId,
) -> Vec<RoadCommand> {
    let mut commands = Vec::new();
    // let Some(road_type) = road_style_params.road_type(&road_manager.road_types) else {
    //     return Vec::new();
    // };
    let ref_segment = road_manager.roads.segment(segment_id);
    let (start, end) = (
        road_manager.roads.node(ref_segment.start).pos,
        road_manager.roads.node(ref_segment.end).pos,
    );

    let storage = &mut road_manager.preview_roads;

    let start_node_id = storage.alloc_node_id();
    commands.push(RoadCommand::AddNode {
        id: start_node_id,
        world_pos: start,
    });

    let end_node_id = storage.alloc_node_id();
    commands.push(RoadCommand::AddNode {
        id: end_node_id,
        world_pos: end,
    });

    let segment_id = storage.alloc_segment_id();
    commands.push(RoadCommand::AddSegment {
        id: segment_id,
        start: start_node_id,
        end: end_node_id,
        structure: ref_segment.structure(),
        chunk_coord: start.chunk,
        road_type_id: ref_segment.road_type_id,
    });

    for lane_id in ref_segment.lanes.iter() {
        let lane = road_manager.roads.lane(*lane_id);
        let lane_id = storage.alloc_lane_id();

        let (from, to) = if lane.from == ref_segment.start {
            (start_node_id, end_node_id)
        } else {
            (end_node_id, start_node_id)
        };

        commands.push(RoadCommand::AddLane {
            id: lane_id,
            from,
            to,
            segment: segment_id,
            lane_index: lane.lane_index,
            geometry: lane.geometry.clone(),
            speed_limit: lane.speed_limit(),
            capacity: lane.capacity(),
            vehicle_mask: lane.vehicle_mask(),
            chunk_coord: start.chunk,
        });
    }

    commands
}
/// Creates a node with a short stub segment and lanes so it renders properly
fn generate_node_with_stub_preview(
    terrain_renderer: &Terrain,
    road_manager: &mut RoadManager,
    road_style_params: &RoadStyleParams,
    position: WorldPos,
    commands: &mut Vec<RoadCommand>,
) {
    let Some(road_type) = road_style_params.road_type(&road_manager.road_types) else {
        return;
    };
    let storage = &mut road_manager.preview_roads;
    // Main node at position
    let main_node_id = storage.alloc_node_id();
    commands.push(RoadCommand::AddNode {
        id: main_node_id,
        world_pos: position,
    });

    // Segment connecting them
    let segment_id = storage.alloc_segment_id();
    commands.push(RoadCommand::AddSegment {
        id: segment_id,
        start: main_node_id,
        end: main_node_id,
        structure: road_type.structure(),
        chunk_coord: position.chunk,
        road_type_id: road_style_params.road_type_id(),
    });

    // Compute and add lanes
    let centerline = vec![position, position];
    let lane_defs = compute_lane_geometries(terrain_renderer, road_type, &centerline);

    let speed = road_type.speed_limit();
    let capacity = road_type.capacity();
    let mask = road_type.vehicle_mask();

    for lane_def in lane_defs {
        let lane_id = storage.alloc_lane_id();

        let (from, to) = (main_node_id, main_node_id);

        commands.push(RoadCommand::AddLane {
            id: lane_id,
            from,
            to,
            segment: segment_id,
            lane_index: lane_def.lane_index,
            geometry: lane_def.geometry,
            speed_limit: speed,
            capacity,
            vehicle_mask: mask,
            chunk_coord: position.chunk,
        });
    }
}

/// Pre-computed lane definition
struct LaneDefinition {
    lane_index: i8,
    is_forward: bool,
    geometry: LaneGeometry,
}

/// Compute all lane geometries from a centerline polyline
fn compute_lane_geometries(
    terrain_renderer: &Terrain,
    road_type: &RoadType,
    centerline: &[WorldPos],
) -> Vec<LaneDefinition> {
    let mut lanes = Vec::new();
    let (left_count, right_count) = road_type.lanes_each_direction();
    let lane_width = road_type.lane_width;

    // Forward lanes (right side: travel from start to end)
    for i in 0..right_count {
        let lane_index = (i as i8).saturating_add(1);
        let polyline = offset_polyline(
            terrain_renderer,
            centerline,
            lane_index,
            lane_width,
            road_type.structure,
        );
        let geometry = LaneGeometry::from_polyline(polyline);

        lanes.push(LaneDefinition {
            lane_index,
            is_forward: true,
            geometry,
        });
    }

    // Backward lanes (left side: travel from end to start)
    for i in 0..left_count {
        let lane_index = -((i as i8) + 1);
        let mut polyline = offset_polyline(
            terrain_renderer,
            centerline,
            lane_index,
            lane_width,
            road_type.structure,
        );
        polyline.reverse();
        let geometry = LaneGeometry::from_polyline(polyline);

        lanes.push(LaneDefinition {
            lane_index,
            is_forward: false,
            geometry,
        });
    }

    lanes
}

#[derive(Debug, Clone)]
pub enum TurnType {
    Straight,
    Right,
    Left,
    UTurn,
    SharpRight,
    SharpLeft,
}
/// Wraps any angle into [0, 2π)
fn normalize_angle(angle: f32) -> f32 {
    let a = angle % TAU;
    if a < 0.0 { a + TAU } else { a }
}
fn classify_turn(from_arm: &Arm, to_arm: &Arm, arm_count: usize) -> TurnType {
    let angle_diff = normalize_angle(to_arm.bearing - from_arm.bearing);
    // With arms sorted, you can also just use index distance:
    // adjacent arm to the right = right turn, opposite = straight, etc.
    match angle_diff {
        a if a < 0.3 => TurnType::UTurn,
        a if a < PI * 0.6 => TurnType::SharpRight,
        a if a < PI * 0.85 => TurnType::Right,
        a if a < PI * 1.15 => TurnType::Straight,
        a if a < PI * 1.4 => TurnType::Left,
        a if a < PI * 1.7 => TurnType::SharpLeft,
        _ => TurnType::UTurn,
    }
}
fn turn_cost(turn: TurnType) -> f32 {
    match turn {
        TurnType::Straight => 0.0,
        TurnType::Right => 2.0, // seconds
        TurnType::Left => 5.0,  // wait for gap
        TurnType::UTurn => 12.0,
        TurnType::SharpRight => 3.0,
        TurnType::SharpLeft => 7.0,
    }
}
// fn has_priority_over(&self, other_arm_idx: usize, my_arm_idx: usize) -> bool {
//     // In clockwise-sorted arms, the arm to your right
//     // is the previous index (wrapping)
//     let right_of_me = (my_arm_idx + arms.len() - 1) % arms.len();
//     other_arm_idx == right_of_me
// }

const BREAD_CRUMB_ALPHA: f32 = 0.1;

/// Tracks a running average that forgets old data exponentially.
/// Recent reports matter more than ancient ones.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[revisioned(revision = 1)]
pub struct EMA {
    duration: f32,
    count: u16, // how many samples we've seen (useful for "is this trustworthy?")
}

impl EMA {
    pub fn new() -> Self {
        Self {
            duration: 0.0,
            count: 0,
        }
    }

    /// Seed with an initial estimate (e.g., Euclidean distance / speed limit)
    /// so the first cars aren't completely blind.
    pub fn with_initial(initial_duration: f32) -> Self {
        Self {
            duration: initial_duration,
            count: 1,
        }
    }

    pub fn with_state(value: f32, count: u16) -> Self {
        Self {
            duration: value,
            count,
        }
    }

    /// A car reports a new observed travel time.
    pub fn update(&mut self, sample: f32) {
        // TODO: Update the EMAs! Fade out with time!
        if self.count == 0 {
            // First sample: just accept it wholesale
            self.duration = sample;
        } else {
            self.duration = BREAD_CRUMB_ALPHA * sample + (1.0 - BREAD_CRUMB_ALPHA) * self.duration;
        }
        self.count = self.count.saturating_add(1);
    }

    /// Current best estimate of travel time.
    pub fn get(&self) -> f32 {
        self.duration
    }

    /// How many reports this is based on.
    /// Cars might trust high-count averages more than low-count ones.
    pub fn sample_count(&self) -> u16 {
        self.count
    }

    /// Is this estimate based on enough data to be meaningful?
    pub fn is_reliable(&self, min_samples: u16) -> bool {
        self.count >= min_samples
    }
}

/// What WOULD be affected by a delete operation!!
#[derive(Debug, Default)]
pub struct DeleteImpact {
    pub nodes: Vec<NodeId>,               // fully isolated, will be deleted
    pub segments: Vec<SegmentId>,         // will be deleted
    pub lanes: Vec<LaneId>,               // will be deleted
    pub nodes_needing_regen: Vec<NodeId>, // still have connections, arms must be rebuilt
}
