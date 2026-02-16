#![allow(dead_code, unused_variables)]
//! roads.rs - Lane-first, 3D, topology-only road system for chunked citybuilder
//!
//! This module provides the canonical road topology and command API.
//! All operations are deterministic and suitable for simulation replay.
//!
//! # Invariants
//! - IDs are monotonically increasing and never reused
//! - Topology objects are disabled, not deleted, for undo/redo support
//! - Lanes are first-class graph edges; segments are grouping/metadata
//! - Every node is an intersection with attachable traffic controls
//! - Mutable operations must occur outside simulation ticks

// ============================================================================
// ID Newtypes
// ============================================================================

use crate::helpers::positions::{ChunkCoord, ChunkSize, LocalPos, WorldPos};
use crate::renderer::gizmo::Gizmo;
use crate::world::cars::car_subsystem::CarSubsystem;
use crate::world::roads::intersections::{
    IntersectionBuildParams, build_intersection_at_node, gather_arms,
};
use crate::world::roads::road_editor::offset_polyline;
use crate::world::roads::road_mesh_manager::{
    ChunkId, RoadMeshManager, chunk_coord_to_id, world_pos_chunk_to_id,
};
use crate::world::roads::road_structs::*;
use crate::world::terrain::terrain_subsystem::TerrainSubsystem;
use glam::{Vec2, Vec3};
use std::collections::HashMap;
use std::f32::consts::{PI, TAU};

pub const METERS_PER_LANE_POLYLINE_STEP: f32 = 2.0;

type PartitionId = u32;
/// One physical "leg" of an intersection — a direction you can come from or go to.
/// Arms are sorted by bearing angle (clockwise from north, or whatever convention).
#[derive(Debug, Clone)]
pub struct Arm {
    segment: SegmentId,
    /// Bearing angle in radians [0, 2π), CCW from +X axis
    bearing: f32,
    /// Direction vector pointing AWAY from node center (normalized)
    direction: Vec3,
    /// Half-width of the road at this arm (lanes + sidewalk)
    half_width: f32,
    /// Whether the segment points toward this node (end == node_id)
    points_to_node: bool,

    incoming_lanes: Vec<LaneId>,
    outgoing_lanes: Vec<LaneId>,

    // === Sign-based routing data ===
    /// Static: partitions reachable by leaving through this arm's outgoing lanes.
    /// Built once at map construction, rebuilt on topology change.
    reachable_partitions: Vec<PartitionId>,

    /// Dynamic: learned travel times to partitions, updated by cars reporting back.
    travel_times: HashMap<PartitionId, ExponentialMovingAverage>,

    /// Current congestion estimate (0.0 = free flow, 1.0 = gridlocked)
    congestion: f32,
}
impl Arm {
    pub fn new(
        segment: SegmentId,
        bearing: f32,
        direction: Vec3,
        half_width: f32,
        points_to_node: bool,
    ) -> Self {
        Self {
            segment,
            bearing,
            direction: direction.normalize_or_zero(),
            half_width,
            points_to_node,
            incoming_lanes: Vec::new(),
            outgoing_lanes: Vec::new(),
            reachable_partitions: Vec::new(),
            travel_times: HashMap::new(),
            congestion: 0.0,
        }
    }

    // === Getters ===

    pub fn segment(&self) -> SegmentId {
        self.segment
    }

    pub fn bearing(&self) -> f32 {
        self.bearing
    }

    pub fn direction(&self) -> Vec3 {
        self.direction
    }

    pub fn half_width(&self) -> f32 {
        self.half_width
    }

    pub fn points_to_node(&self) -> bool {
        self.points_to_node
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

    pub fn reachable_partitions(&self) -> &[PartitionId] {
        &self.reachable_partitions
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
            let idx_a = storage.lane(a).lane_index();
            let idx_b = storage.lane(b).lane_index();
            idx_b.cmp(&idx_a) // Descending (rightmost first)
        });

        self.outgoing_lanes.sort_by(|a, b| {
            let idx_a = storage.lane(a).lane_index();
            let idx_b = storage.lane(b).lane_index();
            idx_a.cmp(&idx_b) // Ascending (rightmost first for outgoing)
        });
    }

    // === Routing Data ===

    pub fn set_reachable_partitions(&mut self, partitions: Vec<PartitionId>) {
        self.reachable_partitions = partitions;
    }

    pub fn add_reachable_partition(&mut self, partition: PartitionId) {
        if !self.reachable_partitions.contains(&partition) {
            self.reachable_partitions.push(partition);
        }
    }

    pub fn travel_time_to(&self, partition: PartitionId) -> Option<f32> {
        self.travel_times.get(&partition).map(|ema| ema.get())
    }

    pub fn update_travel_time(&mut self, partition: PartitionId, time: f32) {
        self.travel_times
            .entry(partition)
            .or_insert_with(|| ExponentialMovingAverage::new(0.1))
            .update(time);
    }

    pub fn clear_travel_times(&mut self) {
        self.travel_times.clear();
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
    pub fn right_edge_at(
        &self,
        center: WorldPos,
        distance: f32,
        chunk_size: ChunkSize,
    ) -> WorldPos {
        let offset = self.direction * distance + self.right_perpendicular() * self.half_width;
        center.add_vec3(offset, chunk_size)
    }

    /// Get the position of the left edge at a given distance from center
    pub fn left_edge_at(&self, center: WorldPos, distance: f32, chunk_size: ChunkSize) -> WorldPos {
        let offset = self.direction * distance + self.left_perpendicular() * self.half_width;
        center.add_vec3(offset, chunk_size)
    }
}
/// Intersection anchor point in 3D space.
/// Every node is an intersection with attachable traffic controls.
#[derive(Debug, Clone)]
pub struct Node {
    pos: WorldPos,
    enabled: bool,
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
            enabled: true,
            arms: Vec::with_capacity(2),
            node_lanes: Vec::new(),
            incoming_lanes: Vec::new(),
            outgoing_lanes: Vec::new(),
            attached_controls: Vec::new(),
            next_control_id: 0,
            car_spawning_rate: 10.0,
        }
    }

    pub fn _version(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis

        #[inline(always)]
        fn mix(h: &mut u64, v: u64) {
            *h ^= v;
            *h = h.wrapping_mul(0x100000001b3);
        }

        mix(&mut h, self.pos.local.x.to_bits() as u64);
        mix(&mut h, self.pos.local.y.to_bits() as u64);
        mix(&mut h, self.pos.local.z.to_bits() as u64);

        mix(&mut h, self.chunk_id() as u64);
        mix(&mut h, self.enabled as u64);
        mix(&mut h, self.next_control_id as u64);

        mix(&mut h, self.attached_controls.len() as u64);
        mix(&mut h, self.incoming_lanes.len() as u64);
        mix(&mut h, self.outgoing_lanes.len() as u64);

        h
    }

    #[inline]
    pub fn position(&self) -> WorldPos {
        self.pos
    }

    #[inline]
    pub fn chunk_id(&self) -> ChunkId {
        chunk_coord_to_id(self.pos.chunk.x, self.pos.chunk.z)
    }

    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
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
    pub fn node_lane(&self, node_lane_id: NodeLaneId) -> &NodeLane {
        &self.node_lanes.get(node_lane_id as usize).unwrap()
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
}

// ============================================================================
// Segment
// ============================================================================

/// Road segment connecting two nodes, containing multiple lanes.
/// Segments are grouping/metadata; lanes are the first-class graph edges.
#[derive(Debug, Clone)]
pub struct Segment {
    pub start: NodeId,
    pub end: NodeId,
    pub enabled: bool,
    pub lanes: Vec<LaneId>,
    pub structure: StructureType,
    pub version: u32,
}

impl Segment {
    fn new(start: NodeId, end: NodeId, structure: StructureType) -> Self {
        Self {
            start,
            end,
            enabled: true,
            lanes: Vec::new(),
            structure,
            version: 0,
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
    pub fn is_enabled(&self) -> bool {
        self.enabled
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
    pub fn lane_counts(&self, storage: &RoadStorage) -> (usize, usize) {
        let mut forward = 0;
        let mut backward = 0;
        for lane_id in &self.lanes {
            let lane = storage.lane(lane_id);
            if lane.from_node() == self.start {
                forward += 1;
            } else {
                backward += 1;
            }
        }
        (forward, backward)
    }
}

// ============================================================================
// Lane
// ============================================================================

/// Directed lane edge connecting two nodes within a segment.
/// Lanes are the primary graph edges for pathfinding and simulation.
#[derive(Debug, Clone)]
pub struct Lane {
    from: NodeId,
    to: NodeId,
    segment: SegmentId,
    lane_index: i8, // signed, relative to segment centerline
    enabled: bool,
    speed_limit: f32,
    capacity: u32,
    vehicle_mask: u32,
    base_cost: f32,
    dynamic_cost: f32,
    geometry: LaneGeometry,
}

impl Lane {
    pub(crate) fn new(
        from: NodeId,
        to: NodeId,
        segment: SegmentId,
        lane_index: i8, // signed, relative to segment centerline
        speed_limit: f32,
        capacity: u32,
        vehicle_mask: u32,
        base_cost: f32,
        geometry: LaneGeometry,
    ) -> Self {
        Self {
            from,
            to,
            segment,
            lane_index,
            enabled: true,
            speed_limit,
            capacity,
            vehicle_mask,
            base_cost,
            dynamic_cost: 0.0,
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
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    #[inline]
    pub fn is_disabled(&self) -> bool {
        !self.enabled
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

    #[inline]
    pub fn base_cost(&self) -> f32 {
        self.base_cost
    }

    #[inline]
    pub fn dynamic_cost(&self) -> f32 {
        self.dynamic_cost
    }

    #[inline]
    pub fn total_cost(&self) -> f32 {
        self.base_cost + self.dynamic_cost
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
    pub fn replace_base_cost(&mut self, base_cost: f32) {
        self.base_cost = base_cost;
    }
    #[inline]
    pub fn replace_geometry(&mut self, geometry: LaneGeometry) {
        self.geometry = geometry;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LaneRef {
    Segment(LaneId, PolyIdx),
    NodeLane(NodeLaneId, PolyIdx),
}

/// Directed lane edge connecting two segments within a node.
/// NodeLanes are the primary graph edges for pathfinding and simulation.
#[derive(Default, Debug, Clone)]
pub struct NodeLane {
    id: NodeLaneId,
    merging: Vec<LaneRef>,
    splitting: Vec<LaneRef>,
    geometry: LaneGeometry,
    // Cached costs for pathfinding
    base_cost: f32,
    dynamic_cost: f32,
    enabled: bool,
    speed_limit: f32,
    vehicle_mask: u32,
}

impl NodeLane {
    pub(crate) fn new(
        id: NodeLaneId,
        merging: Vec<LaneRef>,
        splitting: Vec<LaneRef>,
        geometry: LaneGeometry,
        // Cached costs for pathfinding
        base_cost: f32,
        dynamic_cost: f32,
        speed_limit: f32,
        vehicle_mask: u32,
    ) -> Self {
        Self {
            id,
            merging,
            splitting,
            geometry,
            base_cost,
            dynamic_cost,
            enabled: true,
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
    pub fn total_length(&self) -> f32 {
        self.geometry.total_len
    }
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    #[inline]
    pub fn disable(&mut self) {
        self.enabled = false;
    }
    #[inline]
    pub fn enable(&mut self) {
        self.enabled = true;
    }
    #[inline]
    pub fn speed_limit(&self) -> f32 {
        self.speed_limit
    }

    #[inline]
    pub fn base_cost(&self) -> f32 {
        self.base_cost
    }

    #[inline]
    pub fn dynamic_cost(&self) -> f32 {
        self.dynamic_cost
    }

    #[inline]
    pub fn total_cost(&self) -> f32 {
        self.base_cost + self.dynamic_cost
    }

    /// Returns true if the lane allows the given vehicle type.
    #[inline]
    pub fn allows_vehicle(&self, vehicle_type: u32) -> bool {
        (self.vehicle_mask & vehicle_type) != 0
    }
}
#[derive(Clone, Debug, Default)]
pub struct LaneGeometry {
    pub points: Vec<WorldPos>, // polyline
    pub lengths: Vec<f32>,     // cumulative arc length
    pub total_len: f32,
}

impl LaneGeometry {
    pub fn from_polyline(points: Vec<WorldPos>, chunk_size: ChunkSize) -> Self {
        debug_assert!(points.len() >= 2);

        let mut lengths = Vec::with_capacity(points.len());
        let mut total_len = 0.0;

        lengths.push(0.0);

        for i in 1..points.len() {
            total_len += points[i].distance_to(points[i - 1], chunk_size);
            lengths.push(total_len);
        }

        LaneGeometry {
            points,
            lengths,
            total_len,
        }
    }
}

// ============================================================================
// RoadManager
// ============================================================================

pub struct RoadStorage {
    pub nodes: Vec<Node>,
    pub segments: Vec<Segment>,
    pub lanes: Vec<Lane>,
}
impl Default for RoadStorage {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            segments: Vec::new(),
            lanes: Vec::new(),
        }
    }
}
impl RoadStorage {
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.segments.clear();
        self.lanes.clear();
    }
    // ------------------------------------------------------------------------
    // Node operations
    // ------------------------------------------------------------------------

    /// Adds a new intersection node at the specified position.
    /// Returns the stable, monotonically increasing ID.
    pub fn add_node(&mut self, world_pos: WorldPos) -> NodeId {
        let id = NodeId::new(self.nodes.len() as u32);
        self.nodes.push(Node::new(world_pos));
        id
    }

    /// Returns a reference to the node with the given ID.
    #[inline]
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.0 as usize)
    }

    /// Returns a mutable reference to the node.
    /// Only valid during command application, not during simulation.
    #[inline]
    pub fn node_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id.0 as usize]
    }

    /// Disables a node. Does not affect connected lanes/segments.
    pub fn disable_node(&mut self, id: NodeId) {
        self.nodes[id.0 as usize].enabled = false;
    }

    /// Re-enables a previously disabled node.
    pub fn enable_node(&mut self, id: NodeId) {
        self.nodes[id.0 as usize].enabled = true;
    }

    /// Returns an iterator over all nodes in insertion order.
    #[inline]
    pub fn iter_nodes(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (NodeId::new(i as u32), n))
    }

    /// Returns an iterator over enabled nodes only.
    #[inline]
    pub fn iter_enabled_nodes(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.iter_nodes().filter(|(_, n)| n.is_enabled())
    }

    /// Returns the total number of nodes (including disabled).
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn lane_counts_for_segment(&self, segment: &Segment) -> (usize, usize) {
        let mut left_lanes = 0;
        let mut right_lanes = 0;
        let seg_node_a_id = segment.start();
        let seg_node_b_id = segment.end();
        for lane_id in segment.lanes.iter() {
            let lane = self.lane(lane_id);
            if lane.from == seg_node_a_id && lane.to == seg_node_b_id {
                right_lanes += 1;
            } else {
                left_lanes += 1;
            }
        }
        (left_lanes, right_lanes)
    }
    pub fn enabled_segments_connected_to_node(&self, node_id: NodeId) -> Vec<SegmentId> {
        let Some(node) = self.node(node_id) else {
            return Vec::new();
        };

        let mut segments = Vec::new();

        for &lane_id in node.incoming_lanes().iter().chain(node.outgoing_lanes()) {
            let lane = &self.lanes[lane_id.0 as usize];
            let seg = lane.segment();
            let segment = self.segment(seg);
            if segment.is_enabled() {
                // de-dup without HashSet (cheap, small N…)
                if !segments.contains(&seg) {
                    segments.push(seg);
                }
            }
        }

        segments
    }
    pub fn enabled_segment_count_connected_to_node(&self, node_id: NodeId) -> usize {
        let Some(node) = self.node(node_id) else {
            return 0;
        };

        let mut count = 0;
        let mut seen = Vec::new();

        for &lane_id in node.incoming_lanes().iter().chain(node.outgoing_lanes()) {
            let seg = self.lanes[lane_id.0 as usize].segment();
            let segment = self.segment(seg);
            if segment.is_enabled() {
                if !seen.contains(&seg) {
                    seen.push(seg);
                    count += 1;
                }
            }
        }

        count
    }

    pub fn get_active_segment_ids(&self) -> Vec<SegmentId> {
        self.segments
            .iter()
            .enumerate()
            .filter(|(_, s)| s.enabled)
            .map(|(id, _)| SegmentId(id as u32))
            .collect()
    }
    pub fn get_active_node_ids(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.enabled)
            .map(|(id, _)| NodeId(id as u32))
            .collect()
    }
    // ------------------------------------------------------------------------
    // Segment operations
    // ------------------------------------------------------------------------

    /// Adds a new road segment connecting two nodes.
    pub fn add_segment(
        &mut self,
        start: NodeId,
        end: NodeId,
        structure: StructureType,
    ) -> SegmentId {
        let id = SegmentId::new(self.segments.len() as u32);
        self.segments.push(Segment::new(start, end, structure));
        id
    }

    /// Returns a reference to the segment with the given ID.
    #[inline]
    pub fn segment(&self, id: SegmentId) -> &Segment {
        &self.segments[id.0 as usize]
    }

    /// Returns a mutable reference to the segment.
    #[inline]
    pub fn segment_mut(&mut self, id: SegmentId) -> &mut Segment {
        &mut self.segments[id.0 as usize]
    }

    /// Disables a segment and all its lanes. Increments segment version.
    pub fn disable_segment(&mut self, id: SegmentId) {
        let segment = &mut self.segments[id.0 as usize];
        segment.enabled = false;
        segment.version += 1;

        // Collect lane IDs first to avoid borrow issues
        let lane_ids: Vec<LaneId> = segment.lanes.clone();
        for lane_id in lane_ids {
            self.lanes[lane_id.0 as usize].enabled = false;
        }
    }

    /// Re-enables a segment (lanes must be re-enabled separately if desired).
    pub fn enable_segment(&mut self, id: SegmentId) {
        self.segments[id.0 as usize].enabled = true;
    }

    /// Returns an iterator over all segments in insertion order.
    #[inline]
    pub fn iter_segments(&self) -> impl Iterator<Item = (SegmentId, &Segment)> {
        self.segments
            .iter()
            .enumerate()
            .map(|(i, s)| (SegmentId::new(i as u32), s))
    }

    /// Returns an iterator over enabled segments only.
    #[inline]
    pub fn iter_enabled_segments(&self) -> impl Iterator<Item = (SegmentId, &Segment)> {
        self.iter_segments().filter(|(_, s)| s.is_enabled())
    }

    /// Returns the total number of segments (including disabled).
    #[inline]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Find segments that potentially touch a chunk (using bounding box test).
    pub fn segment_ids_touching_chunk(
        &self,
        chunk_coord: ChunkCoord,
        chunk_size: ChunkSize,
    ) -> Vec<SegmentId> {
        self.segments
            .iter()
            .enumerate()
            .filter_map(|(idx, seg)| {
                if !seg.enabled {
                    return None;
                }

                let start = self.nodes.get(seg.start.raw() as usize)?;
                let end = self.nodes.get(seg.end.raw() as usize)?;

                let start_pos = start.position();
                let end_pos = end.position();

                // Check if segment's bounding box overlaps this chunk
                if segment_touches_chunk_precise(start_pos, end_pos, chunk_coord, chunk_size) {
                    Some(SegmentId::new(idx as u32))
                } else {
                    None
                }
            })
            .collect()
    }

    #[inline]
    pub fn node_lane_count_for_node(&self, id: NodeId) -> usize {
        self.nodes[id.raw() as usize].node_lanes.len()
    }
    // ------------------------------------------------------------------------
    // Lane operations
    // ------------------------------------------------------------------------

    /// Adds a new lane to a segment.
    /// The lane direction is from->to; must match segment endpoints.
    pub fn add_lane(
        &mut self,
        from: NodeId,
        to: NodeId,
        segment: SegmentId,
        lane_index: i8,
        geometry: LaneGeometry,
        speed_limit: f32,
        capacity: u32,
        vehicle_mask: u32,
        base_cost: f32,
    ) -> LaneId {
        let seg = &self.segments[segment.0 as usize];
        debug_assert!(
            (from == seg.start && to == seg.end) || (from == seg.end && to == seg.start),
            "Lane endpoints must match segment endpoints"
        );

        let id = LaneId::new(self.lanes.len() as u32);

        self.lanes.push(Lane::new(
            from,
            to,
            segment,
            lane_index,
            speed_limit,
            capacity,
            vehicle_mask,
            base_cost,
            geometry,
        ));

        self.segments[segment.0 as usize].lanes.push(id);
        self.nodes[from.0 as usize].outgoing_lanes.push(id);
        self.nodes[to.0 as usize].incoming_lanes.push(id);

        id
    }

    /// Returns a reference to the lane with the given ID.
    #[inline]
    pub fn lane(&self, id: &LaneId) -> &Lane {
        &self.lanes[id.0 as usize]
    }
    #[inline]
    pub fn lane_exists(&self, id: &LaneId) -> bool {
        self.lanes.get(id.raw() as usize).is_some()
    }
    /// Returns a mutable reference to the lane.
    #[inline]
    pub fn lane_mut(&mut self, id: LaneId) -> &mut Lane {
        &mut self.lanes[id.0 as usize]
    }

    /// Disables a lane.
    pub fn disable_lane(&mut self, id: LaneId) {
        self.lanes[id.0 as usize].enabled = false;
    }

    /// Re-enables a lane.
    pub fn enable_lane(&mut self, id: LaneId) {
        self.lanes[id.0 as usize].enabled = true;
    }

    /// Returns an iterator over all lanes in insertion order.
    #[inline]
    pub fn iter_lanes(&self) -> impl Iterator<Item = (LaneId, &Lane)> {
        self.lanes
            .iter()
            .enumerate()
            .map(|(i, l)| (LaneId::new(i as u32), l))
    }

    /// Returns an iterator over enabled lanes only.
    #[inline]
    pub fn iter_enabled_lanes(&self) -> impl Iterator<Item = (LaneId, &Lane)> {
        self.iter_lanes().filter(|(_, l)| l.is_enabled())
    }

    /// Returns the total number of lanes (including disabled).
    #[inline]
    pub fn lane_count(&self) -> usize {
        self.lanes.len()
    }

    // ------------------------------------------------------------------------
    // Traffic control operations
    // ------------------------------------------------------------------------

    /// Attaches a traffic control to a node intersection.
    /// Returns the stable control ID for later modification/removal.
    pub fn attach_control(&mut self, node_id: NodeId, control: TrafficControl) -> ControlId {
        let node = &mut self.nodes[node_id.0 as usize];
        let id = ControlId::new(node.next_control_id);
        node.next_control_id += 1;
        node.attached_controls.push(AttachedControl {
            id,
            control,
            enabled: true,
        });
        id
    }

    /// Disables a traffic control (soft remove for undo support).
    pub fn disable_control(&mut self, node_id: NodeId, control_id: ControlId) {
        let node = &mut self.nodes[node_id.0 as usize];
        if let Some(ctrl) = node
            .attached_controls
            .iter_mut()
            .find(|c| c.id == control_id)
        {
            ctrl.enabled = false;
        }
    }

    /// Re-enables a traffic control.
    pub fn enable_control(&mut self, node_id: NodeId, control_id: ControlId) {
        let node = &mut self.nodes[node_id.0 as usize];
        if let Some(ctrl) = node
            .attached_controls
            .iter_mut()
            .find(|c| c.id == control_id)
        {
            ctrl.enabled = true;
        }
    }
    pub(crate) fn add_node_lane(
        &mut self,
        node_id: NodeId,
        merging: Vec<LaneRef>,
        splitting: Vec<LaneRef>,
        geometry: LaneGeometry,
        base_cost: f32,
        dynamic_cost: f32,
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
            base_cost,
            dynamic_cost,
            enabled: true,
            speed_limit,
            vehicle_mask,
        });

        id
    }

    // ------------------------------------------------------------------------
    // Upgrade operations
    // ------------------------------------------------------------------------

    /// Upgrades a segment by disabling it and allowing the caller to add replacements.
    /// Returns the IDs of newly added segments during the upgrade.
    ///
    /// Pattern: disable old, then add new segments/lanes via callback.
    /// Ensures operation is deterministic and versioned.
    pub fn upgrade_segment<F>(&mut self, old_segment: SegmentId, add_new: F) -> Vec<SegmentId>
    where
        F: FnOnce(&mut Self),
    {
        let segment_count_before = self.segments.len();

        // Disable old segment and its lanes
        self.disable_segment(old_segment);

        // Let caller add new segments and lanes
        add_new(self);

        // Collect newly added segment IDs
        (segment_count_before..self.segments.len())
            .map(|i| SegmentId::new(i as u32))
            .collect()
    }

    /// Returns nodes belonging to a specific chunk.
    pub fn nodes_in_chunk(&self, chunk_id: ChunkId) -> Vec<NodeId> {
        self.iter_nodes()
            .filter(|(_, n)| n.is_enabled() && n.chunk_id() == chunk_id)
            .map(|(id, _)| id)
            .collect()
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
}

// Safety: RoadManager uses no interior mutability.
// All mutable access is explicitly controlled by the caller.
impl RoadManager {
    /// Creates an empty road topology.
    pub fn new() -> Self {
        Self {
            roads: RoadStorage::default(),
            preview_roads: RoadStorage::default(),
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
pub fn sample_lane_position(
    lane: &Lane,
    t: f32,
    storage: &RoadStorage,
    chunk_size: ChunkSize,
) -> Option<WorldPos> {
    let from = storage.node(lane.from_node())?;
    let to = storage.node(lane.to_node())?;

    let from_pos = from.position();
    let to_pos = to.position();

    Some(from_pos.lerp(to_pos, t, chunk_size))
}

/// Project a WorldPos onto a lane and returns (t, distance_squared).
/// t is the parameter [0,1] along the lane; dist_sq is squared XZ distance.
#[inline]
pub fn project_point_to_lane_xz(
    lane: &Lane,
    point: WorldPos,
    storage: &RoadStorage,
    chunk_size: ChunkSize,
) -> Option<(f32, f32)> {
    let from = storage.node(lane.from_node())?;
    let to = storage.node(lane.to_node())?;

    let from_pos = from.position();
    let to_pos = to.position();

    Some(project_point_to_segment_xz(
        point, from_pos, to_pos, chunk_size,
    ))
}

/// Project a point onto a line segment (XZ plane).
/// Returns (t_clamped, distance_squared).
#[inline]
pub fn project_point_to_segment_xz(
    point: WorldPos,
    seg_start: WorldPos,
    seg_end: WorldPos,
    chunk_size: ChunkSize,
) -> (f32, f32) {
    // Compute everything relative to seg_start for precision
    let d = seg_end.to_render_pos(seg_start, chunk_size);
    let p = point.to_render_pos(seg_start, chunk_size);

    let dx = d.x;
    let dz = d.z;
    let len_sq = dx * dx + dz * dz;

    if len_sq < 1e-10 {
        // Degenerate segment
        return (0.0, p.x * p.x + p.z * p.z);
    }

    // Project point onto line
    let t = (p.x * dx + p.z * dz) / len_sq;
    let t_clamped = t.clamp(0.0, 1.0);

    // Compute the closest point on segment (relative to seg_start)
    let cx = t_clamped * dx;
    let cz = t_clamped * dz;

    let dist_sq = (p.x - cx) * (p.x - cx) + (p.z - cz) * (p.z - cz);

    (t_clamped, dist_sq)
}

/// Finds the nearest enabled lane to a 3D point (brute force).
/// Returns None if no enabled lanes exist.
///
/// Note: For production use, build chunk-local spatial indexes.
/// This function is O(n) in the number of lanes.
pub fn nearest_lane_to_point(
    storage: &RoadStorage,
    point: WorldPos,
    chunk_size: ChunkSize,
) -> Option<LaneId> {
    let mut best_id: Option<LaneId> = None;
    let mut best_dist_sq = f32::MAX;

    for (id, lane) in storage.iter_lanes() {
        if !lane.is_enabled() {
            continue;
        }

        let (_, dist_sq) = project_point_to_lane_xz(lane, point, storage, chunk_size)?;
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best_id = Some(id);
        }
    }

    best_id
}

// ============================================================================
// Chunk State
// ============================================================================

/// Per-lane runtime state managed by chunk simulation.
#[derive(Debug, Clone, Default)]
pub struct LaneRuntimeState {
    /// Estimated occupancy fraction [0, 1].
    pub occupancy_estimate: f32,
    /// True if lane is temporarily blocked.
    pub blocked: bool,
    /// Dynamic cost modifier added to lane.dynamic_cost.
    pub dynamic_cost_modifier: f32,
}

/// Per-chunk road simulation state.
/// Owned by chunk simulation code, not by RoadManager.
#[derive(Debug, Clone)]
pub struct RoadChunkState {
    chunk_id: ChunkId,
    /// Lane states stored in deterministic order (by LaneId).
    lane_states: Vec<(LaneId, LaneRuntimeState)>,
    /// Last simulation update time in seconds.
    pub last_update_time_seconds: f64,
    /// Deferred tick interval for this chunk.
    pub deferred_tick_seconds: f32,
    /// Per-chunk PRNG seed for deterministic simulation.
    pub prng_seed: u64,
}

impl RoadChunkState {
    /// Creates a new chunk state for the given lanes.
    pub fn new(chunk_id: u64, lanes: &[LaneId], prng_seed: u64) -> Self {
        let mut lane_states: Vec<_> = lanes
            .iter()
            .map(|&id| (id, LaneRuntimeState::default()))
            .collect();
        // Sort by lane ID for deterministic iteration
        lane_states.sort_by_key(|(id, _)| id.raw());

        Self {
            chunk_id,
            lane_states,
            last_update_time_seconds: 0.0,
            deferred_tick_seconds: 2.0,
            prng_seed,
        }
    }

    /// Returns the chunk ID this state belongs to.
    #[inline]
    pub fn chunk_id(&self) -> u64 {
        self.chunk_id
    }

    /// Returns the runtime state for a lane, if tracked.
    pub fn lane_state(&self, id: LaneId) -> Option<&LaneRuntimeState> {
        self.lane_states
            .binary_search_by_key(&id.raw(), |(lid, _)| lid.raw())
            .ok()
            .map(|idx| &self.lane_states[idx].1)
    }

    /// Returns mutable runtime state for a lane, if tracked.
    pub fn lane_state_mut(&mut self, id: LaneId) -> Option<&mut LaneRuntimeState> {
        self.lane_states
            .binary_search_by_key(&id.raw(), |(lid, _)| lid.raw())
            .ok()
            .map(|idx| &mut self.lane_states[idx].1)
    }

    /// Iterates over all lane states in deterministic order.
    pub fn iter_lane_states(&self) -> impl Iterator<Item = (LaneId, &LaneRuntimeState)> {
        self.lane_states.iter().map(|(id, state)| (*id, state))
    }

    /// Iterates mutably over all lane states in deterministic order.
    pub fn iter_lane_states_mut(
        &mut self,
    ) -> impl Iterator<Item = (LaneId, &mut LaneRuntimeState)> {
        self.lane_states.iter_mut().map(|(id, state)| (*id, state))
    }

    /// Adds a lane to tracking (used when lanes are added to chunk).
    pub fn add_lane(&mut self, id: LaneId) {
        let idx = self
            .lane_states
            .binary_search_by_key(&id.raw(), |(lid, _)| lid.raw())
            .unwrap_or_else(|i| i);
        self.lane_states
            .insert(idx, (id, LaneRuntimeState::default()));
    }

    /// Marks a lane as removed from active tracking (keeps entry for replay).
    pub fn mark_lane_disabled(&mut self, id: LaneId) {
        if let Some(state) = self.lane_state_mut(id) {
            state.blocked = true;
        }
    }

    /// Returns the number of tracked lanes.
    #[inline]
    pub fn lane_count(&self) -> usize {
        self.lane_states.len()
    }
}

// ============================================================================
// Boundary Summary
// ============================================================================

/// Summary of road state at chunk boundaries for cross-chunk coordination.
#[derive(Debug, Clone, Default)]
pub struct ChunkBoundarySummary {
    /// Lanes that cross into this chunk from other chunks.
    pub incoming_cross_chunk_lanes: Vec<LaneId>,
    /// Lanes that exit this chunk into other chunks.
    pub outgoing_cross_chunk_lanes: Vec<LaneId>,
    /// Aggregated flow rates at boundary (lane_id, vehicles_per_second).
    pub boundary_flow_rates: Vec<(LaneId, f32)>,
}

impl ChunkBoundarySummary {
    /// Creates a boundary summary for a chunk.
    pub fn compute(storage: &RoadStorage, chunk_id: ChunkId) -> Self {
        let mut incoming = Vec::new();
        let mut outgoing = Vec::new();

        for (id, lane) in storage.iter_enabled_lanes() {
            let Some(from) = storage.node(lane.from_node()) else {
                continue;
            };
            let Some(to) = storage.node(lane.to_node()) else {
                continue;
            };
            let from_chunk = from.chunk_id();
            let to_chunk = to.chunk_id();

            if from_chunk != chunk_id && to_chunk == chunk_id {
                incoming.push(id);
            } else if from_chunk == chunk_id && to_chunk != chunk_id {
                outgoing.push(id);
            }
        }

        // Sort for determinism
        incoming.sort_by_key(|id| id.raw());
        outgoing.sort_by_key(|id| id.raw());

        Self {
            incoming_cross_chunk_lanes: incoming,
            outgoing_cross_chunk_lanes: outgoing,
            boundary_flow_rates: Vec::new(),
        }
    }
}

// ============================================================================
// Command System
// ============================================================================

/// Commands for deterministic road topology modification.
///
/// Commands are applied atomically and can be serialized for replay.
/// The engine should queue commands and apply them between simulation ticks.
///
/// # Invariants
/// - Commands must be applied in deterministic order
/// - AddNode/AddSegment/AddLane commands produce monotonically increasing IDs
/// - Disable commands are idempotent
/// - UpgradeSegmentBegin must be followed by UpgradeSegmentEnd
#[derive(Debug, Clone)]
pub enum RoadCommand {
    /// Add a new intersection node.
    AddNode {
        world_pos: WorldPos,
    },
    /// Add a new road segment.
    AddSegment {
        start: NodeId,
        end: NodeId,
        structure: StructureType,
        chunk_id: ChunkId,
    },
    /// Add a new lane to a segment.
    AddLane {
        from: NodeId,
        to: NodeId,
        segment: SegmentId,
        lane_index: i8,
        geometry: LaneGeometry,
        speed_limit: f32,
        capacity: u32,
        vehicle_mask: u32,
        base_cost: f32,
        chunk_id: ChunkId,
    },
    /// Add a new lane to a segment.
    AddNodeLane {
        node_id: NodeId,
        merging: Vec<LaneRef>,
        splitting: Vec<LaneRef>,
        geometry: LaneGeometry,
        // Cached costs for pathfinding
        speed_limit: f32,
        vehicle_mask: u32,
        base_cost: f32,
        dynamic_cost: f32,
        chunk_id: ChunkId,
    },
    ClearNodeLanes {
        node_id: NodeId,
        chunk_id: ChunkId,
    },
    /// Disable a node.
    DisableNode {
        node_id: NodeId,
        chunk_id: ChunkId,
    },
    /// Enable a node.
    EnableNode {
        node_id: NodeId,
        chunk_id: ChunkId,
    },
    /// Disable a segment and its lanes.
    DisableSegment {
        segment_id: SegmentId,
        chunk_id: ChunkId,
    },
    /// Enable a segment.
    EnableSegment {
        segment_id: SegmentId,
        chunk_id: ChunkId,
    },
    /// Disable a lane.
    DisableLane {
        lane_id: LaneId,
        chunk_id: ChunkId,
    },
    /// Enable a lane.
    EnableLane {
        lane_id: LaneId,
        chunk_id: ChunkId,
    },
    /// Attach a traffic control to a node.
    AttachControl {
        node_id: NodeId,
        chunk_id: ChunkId,
        control: TrafficControl,
    },
    /// Disable a traffic control.
    DisableControl {
        node_id: NodeId,
        chunk_id: ChunkId,
        control_id: ControlId,
    },
    /// Enable a traffic control.
    EnableControl {
        node_id: NodeId,
        chunk_id: ChunkId,
        control_id: ControlId,
    },
    /// Procedurally rebuild node lanes using *current* incoming/outgoing segment lanes.
    MakeIntersection {
        node_id: NodeId,
        params: IntersectionBuildParams,
        chunk_id: ChunkId,
        clear: bool,
    },
    /// Begin segment upgrade (disables old segment).
    UpgradeSegmentBegin {
        old_segment: SegmentId,
        chunk_id: ChunkId,
    },
    /// End segment upgrade (records new segment IDs for replay).
    UpgradeSegmentEnd {
        new_segments: Vec<SegmentId>,
        chunk_id: ChunkId,
    },
}
impl RoadCommand {
    /// Returns the chunk ID affected by this command, if one is explicitly stored.
    /// For `AddNode`, the chunk is derived from `world_pos` and must be computed separately.
    pub fn chunk_id(&self) -> ChunkId {
        match self {
            RoadCommand::AddNode { world_pos } => world_pos_chunk_to_id(*world_pos),
            RoadCommand::AddSegment { chunk_id, .. } => *chunk_id,
            RoadCommand::AddLane { chunk_id, .. } => *chunk_id,
            RoadCommand::AddNodeLane { chunk_id, .. } => *chunk_id,
            RoadCommand::ClearNodeLanes { chunk_id, .. } => *chunk_id,
            RoadCommand::DisableNode { chunk_id, .. } => *chunk_id,
            RoadCommand::EnableNode { chunk_id, .. } => *chunk_id,
            RoadCommand::DisableSegment { chunk_id, .. } => *chunk_id,
            RoadCommand::EnableSegment { chunk_id, .. } => *chunk_id,
            RoadCommand::DisableLane { chunk_id, .. } => *chunk_id,
            RoadCommand::EnableLane { chunk_id, .. } => *chunk_id,
            RoadCommand::AttachControl { chunk_id, .. } => *chunk_id,
            RoadCommand::DisableControl { chunk_id, .. } => *chunk_id,
            RoadCommand::EnableControl { chunk_id, .. } => *chunk_id,
            RoadCommand::MakeIntersection { chunk_id, .. } => *chunk_id,
            RoadCommand::UpgradeSegmentBegin { chunk_id, .. } => *chunk_id,
            RoadCommand::UpgradeSegmentEnd { chunk_id, .. } => *chunk_id,
        }
    }
}

/// Result of applying a command.
#[derive(Debug, Clone)]
pub enum CommandResult {
    /// Node was created with this ID.
    NodeCreated(ChunkId, NodeId),
    /// Segment was created with this ID.
    SegmentCreated(ChunkId, SegmentId),
    /// Lane was created with this ID.
    LaneCreated(ChunkId, LaneId),
    /// NodeLane was created with this ID.
    NodeLaneCreated(ChunkId, NodeLaneId),
    /// Control was attached with this ID.
    ControlAttached(ChunkId, ControlId),
    /// Command applied with no new IDs.
    Ok,
    /// Command failed (invalid reference).
    InvalidReference,
}

/// Applies only the world-state mutations from real (non-preview) commands.
/// No mesh rebuilding — that's the render subsystem's job.
pub fn apply_commands_world(
    storage: &mut RoadStorage,
    car_subsystem: &mut CarSubsystem,
    gizmo: &mut Gizmo,
    commands: &[RoadEditorCommand],
) {
    for cmd in commands {
        if let RoadEditorCommand::Road(road_command) = cmd {
            apply_command_world(storage, car_subsystem, gizmo, road_command, false);
        }
    }
}

/// Applies world-state mutations for preview commands (populates preview_storage).
pub fn apply_preview_commands_world(
    terrain_renderer: &TerrainSubsystem,
    road_style_params: &RoadStyleParams,
    preview_storage: &mut RoadStorage,
    real_storage: &RoadStorage,
    car_subsystem: &mut CarSubsystem,
    commands: &[RoadEditorCommand],
    gizmo: &mut Gizmo,
) {
    preview_storage.clear();

    // 1) Apply explicit Road commands to preview storage
    for cmd in commands {
        if let RoadEditorCommand::Road(road_command) = cmd {
            apply_command_world(preview_storage, car_subsystem, gizmo, road_command, true);
        }
    }

    // 2) Collect preview inputs
    let mut node_previews: Vec<&NodePreview> = Vec::new();
    let mut crossing_previews: Vec<&CrossingPoint> = Vec::new();
    let mut segment_preview: Option<&SegmentPreview> = None;

    for cmd in commands {
        match cmd {
            RoadEditorCommand::PreviewNode(n) => node_previews.push(n),
            RoadEditorCommand::PreviewSegment(s) => segment_preview = Some(s),
            RoadEditorCommand::PreviewCrossing(c) => crossing_previews.push(c),
            RoadEditorCommand::PreviewClear => return,
            _ => {}
        }
    }

    let mut allocator = PreviewIdAllocator::new();
    let mut road_commands: Vec<RoadCommand> = Vec::new();

    // 3) Crossing preview
    if !crossing_previews.is_empty() {
        road_commands.extend(generate_intersection_preview(
            terrain_renderer,
            preview_storage,
            real_storage,
            &mut allocator,
            road_style_params,
            &crossing_previews,
        ));
    }

    // 4) Segment preview
    if let Some(seg) = segment_preview {
        if seg.is_valid {
            road_commands.extend(generate_segment_preview(
                terrain_renderer,
                &mut allocator,
                road_style_params,
                seg,
            ));
        } else {
            road_commands.extend(generate_invalid_segment_preview(
                terrain_renderer,
                &mut allocator,
                road_style_params,
                seg,
            ));
        }
    }

    // 5) Hover nodes
    if segment_preview.is_none() && !node_previews.is_empty() {
        road_commands.extend(generate_hover_preview(
            terrain_renderer,
            &mut allocator,
            road_style_params,
            &node_previews,
        ));
    }

    // 6) Apply generated preview commands to preview storage (world-only)
    for cmd in road_commands {
        apply_command_world(preview_storage, car_subsystem, gizmo, &cmd, true);
    }
}

/// Single command application — world state only, no mesh rebuild.
pub fn apply_command_world(
    storage: &mut RoadStorage,
    car_subsystem: &mut CarSubsystem,
    gizmo: &mut Gizmo,
    road_command: &RoadCommand,
    is_preview: bool,
) -> CommandResult {
    match road_command {
        RoadCommand::AddNode { world_pos } => {
            let id = storage.add_node(*world_pos);
            if !is_preview {
                car_subsystem.add_spawning_node(id);
            }
            let chunk_id = world_pos_chunk_to_id(*world_pos);
            CommandResult::NodeCreated(chunk_id, id)
        }
        RoadCommand::AddSegment {
            start,
            end,
            structure,
            chunk_id,
        } => {
            if start.raw() as usize >= storage.node_count()
                || end.raw() as usize >= storage.node_count()
            {
                return CommandResult::InvalidReference;
            }
            let id = storage.add_segment(*start, *end, structure.clone());
            CommandResult::SegmentCreated(*chunk_id, id)
        }
        RoadCommand::AddLane {
            from,
            to,
            segment,
            lane_index,
            geometry,
            speed_limit,
            capacity,
            vehicle_mask,
            base_cost,
            chunk_id,
        } => {
            if from.raw() as usize >= storage.node_count()
                || to.raw() as usize >= storage.node_count()
                || segment.raw() as usize >= storage.segment_count()
            {
                return CommandResult::InvalidReference;
            }
            let id = storage.add_lane(
                *from,
                *to,
                *segment,
                *lane_index,
                geometry.clone(),
                *speed_limit,
                *capacity,
                *vehicle_mask,
                *base_cost,
            );
            CommandResult::LaneCreated(*chunk_id, id)
        }
        RoadCommand::AddNodeLane {
            node_id,
            merging,
            splitting,
            geometry,
            speed_limit,
            vehicle_mask,
            base_cost,
            dynamic_cost,
            chunk_id,
        } => {
            let id = storage.add_node_lane(
                *node_id,
                merging.clone(),
                splitting.clone(),
                geometry.clone(),
                *base_cost,
                *speed_limit,
                *dynamic_cost,
                *vehicle_mask,
            );
            CommandResult::NodeLaneCreated(*chunk_id, id)
        }
        RoadCommand::ClearNodeLanes { node_id, chunk_id } => {
            let node = storage.node_mut(*node_id);
            node.node_lanes.clear();
            CommandResult::Ok
        }
        RoadCommand::DisableNode { node_id, chunk_id } => {
            if node_id.raw() as usize >= storage.node_count() {
                return CommandResult::InvalidReference;
            }
            storage.disable_node(*node_id);
            CommandResult::Ok
        }
        RoadCommand::EnableNode { node_id, chunk_id } => {
            if node_id.raw() as usize >= storage.node_count() {
                return CommandResult::InvalidReference;
            }
            storage.enable_node(*node_id);
            CommandResult::Ok
        }
        RoadCommand::DisableSegment {
            segment_id,
            chunk_id,
        } => {
            if segment_id.raw() as usize >= storage.segment_count() {
                return CommandResult::InvalidReference;
            }
            storage.disable_segment(*segment_id);
            CommandResult::Ok
        }
        RoadCommand::EnableSegment {
            segment_id,
            chunk_id,
        } => {
            if segment_id.raw() as usize >= storage.segment_count() {
                return CommandResult::InvalidReference;
            }
            storage.enable_segment(*segment_id);
            CommandResult::Ok
        }
        RoadCommand::DisableLane { lane_id, chunk_id } => {
            if lane_id.raw() as usize >= storage.lane_count() {
                return CommandResult::InvalidReference;
            }
            storage.disable_lane(*lane_id);
            CommandResult::Ok
        }
        RoadCommand::EnableLane { lane_id, chunk_id } => {
            if lane_id.raw() as usize >= storage.lane_count() {
                return CommandResult::InvalidReference;
            }
            storage.enable_lane(*lane_id);
            CommandResult::Ok
        }
        RoadCommand::AttachControl {
            node_id,
            chunk_id,
            control,
        } => {
            if node_id.raw() as usize >= storage.node_count() {
                return CommandResult::InvalidReference;
            }
            let id = storage.attach_control(*node_id, control.clone());
            CommandResult::ControlAttached(*chunk_id, id)
        }
        RoadCommand::DisableControl {
            node_id,
            control_id,
            chunk_id,
        } => {
            if node_id.raw() as usize >= storage.node_count() {
                return CommandResult::InvalidReference;
            }
            storage.disable_control(*node_id, *control_id);
            CommandResult::Ok
        }
        RoadCommand::EnableControl {
            node_id,
            control_id,
            chunk_id,
        } => {
            if node_id.raw() as usize >= storage.node_count() {
                return CommandResult::InvalidReference;
            }
            storage.enable_control(*node_id, *control_id);
            CommandResult::Ok
        }
        RoadCommand::MakeIntersection {
            node_id,
            params,
            chunk_id,
            clear,
        } => {
            if node_id.raw() as usize >= storage.node_count() {
                return CommandResult::InvalidReference;
            }
            let arms = gather_arms(storage, *node_id, params, gizmo.chunk_size, gizmo);

            storage.node_mut(*node_id).arms = arms;
            build_intersection_at_node(storage, *node_id, params, *clear, gizmo);
            CommandResult::Ok
        }
        RoadCommand::UpgradeSegmentBegin {
            old_segment,
            chunk_id,
        } => {
            if old_segment.raw() as usize >= storage.segment_count() {
                return CommandResult::InvalidReference;
            }
            storage.disable_segment(*old_segment);
            CommandResult::Ok
        }
        RoadCommand::UpgradeSegmentEnd { chunk_id, .. } => CommandResult::Ok,
    }
}

/// Extracts all chunk IDs affected by real (non-preview) commands,
/// so the render subsystem knows which chunks to rebuild meshes for.
pub fn collect_affected_chunks(commands: &[RoadEditorCommand]) -> Vec<ChunkId> {
    let mut chunks = Vec::new();
    for cmd in commands {
        if let RoadEditorCommand::Road(road_cmd) = cmd {
            let chunk_id = road_cmd.chunk_id();
            chunks.push(chunk_id);
        }
    }
    chunks.sort();
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
    terrain_renderer: &TerrainSubsystem,
    road_mesh_manager: &mut RoadMeshManager,
    car_subsystem: &mut CarSubsystem,
    storage: &mut RoadStorage,
    road_style_params: &RoadStyleParams,
    command: RoadEditorCommand,
    is_preview: bool,
    gizmo: &mut Gizmo,
) -> CommandResult {
    match command {
        RoadEditorCommand::Road(road_command) => {
            // We store the chunk ID here if an operation succeeds
            let affected_chunk: Option<ChunkId>;

            let result = match road_command {
                RoadCommand::AddNode { world_pos } => {
                    let id = storage.add_node(world_pos);
                    gather_arms(
                        storage,
                        id,
                        &IntersectionBuildParams::from_style(road_style_params),
                        terrain_renderer.chunk_size,
                        gizmo,
                    );
                    if !is_preview {
                        car_subsystem.add_spawning_node(id);
                    }
                    let chunk_id = world_pos_chunk_to_id(world_pos);
                    affected_chunk = Some(chunk_id);
                    CommandResult::NodeCreated(chunk_id, id)
                }
                RoadCommand::AddSegment {
                    start,
                    end,
                    structure,
                    chunk_id,
                } => {
                    if start.raw() as usize >= storage.node_count()
                        || end.raw() as usize >= storage.node_count()
                    {
                        return CommandResult::InvalidReference;
                    }
                    let id = storage.add_segment(start, end, structure);
                    affected_chunk = Some(chunk_id);
                    CommandResult::SegmentCreated(chunk_id, id)
                }
                RoadCommand::AddLane {
                    from,
                    to,
                    segment,
                    lane_index,
                    geometry,
                    speed_limit,
                    capacity,
                    vehicle_mask,
                    base_cost,
                    chunk_id,
                } => {
                    if from.raw() as usize >= storage.node_count()
                        || to.raw() as usize >= storage.node_count()
                        || segment.raw() as usize >= storage.segment_count()
                    {
                        return CommandResult::InvalidReference;
                    }
                    let id = storage.add_lane(
                        from,
                        to,
                        segment,
                        lane_index,
                        geometry,
                        speed_limit,
                        capacity,
                        vehicle_mask,
                        base_cost,
                    );
                    affected_chunk = Some(chunk_id);
                    CommandResult::LaneCreated(chunk_id, id)
                }
                RoadCommand::AddNodeLane {
                    node_id,
                    merging,
                    splitting,
                    geometry,
                    speed_limit,
                    vehicle_mask,
                    base_cost,
                    dynamic_cost,
                    chunk_id,
                } => {
                    let id = storage.add_node_lane(
                        node_id,
                        merging,
                        splitting,
                        geometry,
                        base_cost,
                        speed_limit,
                        dynamic_cost,
                        vehicle_mask,
                    );
                    affected_chunk = Some(chunk_id);
                    CommandResult::NodeLaneCreated(chunk_id, id)
                }
                RoadCommand::ClearNodeLanes { node_id, chunk_id } => {
                    let node = storage.node_mut(node_id);
                    node.node_lanes.clear();
                    affected_chunk = Some(chunk_id);
                    CommandResult::Ok
                }
                RoadCommand::DisableNode { node_id, chunk_id } => {
                    if node_id.raw() as usize >= storage.node_count() {
                        return CommandResult::InvalidReference;
                    }
                    storage.disable_node(node_id);
                    affected_chunk = Some(chunk_id);
                    CommandResult::Ok
                }
                RoadCommand::EnableNode { node_id, chunk_id } => {
                    if node_id.raw() as usize >= storage.node_count() {
                        return CommandResult::InvalidReference;
                    }
                    storage.enable_node(node_id);
                    affected_chunk = Some(chunk_id);
                    CommandResult::Ok
                }
                RoadCommand::DisableSegment {
                    segment_id,
                    chunk_id,
                } => {
                    if segment_id.raw() as usize >= storage.segment_count() {
                        return CommandResult::InvalidReference;
                    }
                    storage.disable_segment(segment_id);
                    affected_chunk = Some(chunk_id);
                    CommandResult::Ok
                }
                RoadCommand::EnableSegment {
                    segment_id,
                    chunk_id,
                } => {
                    if segment_id.raw() as usize >= storage.segment_count() {
                        return CommandResult::InvalidReference;
                    }
                    storage.enable_segment(segment_id);
                    affected_chunk = Some(chunk_id);
                    CommandResult::Ok
                }
                RoadCommand::DisableLane { lane_id, chunk_id } => {
                    if lane_id.raw() as usize >= storage.lane_count() {
                        return CommandResult::InvalidReference;
                    }
                    storage.disable_lane(lane_id);
                    affected_chunk = Some(chunk_id);
                    CommandResult::Ok
                }
                RoadCommand::EnableLane { lane_id, chunk_id } => {
                    if lane_id.raw() as usize >= storage.lane_count() {
                        return CommandResult::InvalidReference;
                    }
                    storage.enable_lane(lane_id);
                    affected_chunk = Some(chunk_id);
                    CommandResult::Ok
                }
                RoadCommand::AttachControl {
                    node_id,
                    chunk_id,
                    control,
                } => {
                    if node_id.raw() as usize >= storage.node_count() {
                        return CommandResult::InvalidReference;
                    }
                    let id = storage.attach_control(node_id, control.clone());
                    affected_chunk = Some(chunk_id);
                    CommandResult::ControlAttached(chunk_id, id)
                }
                RoadCommand::DisableControl {
                    node_id,
                    control_id,
                    chunk_id,
                } => {
                    if node_id.raw() as usize >= storage.node_count() {
                        return CommandResult::InvalidReference;
                    }
                    storage.disable_control(node_id, control_id);
                    affected_chunk = Some(chunk_id);
                    CommandResult::Ok
                }
                RoadCommand::EnableControl {
                    node_id,
                    control_id,
                    chunk_id,
                } => {
                    if node_id.raw() as usize >= storage.node_count() {
                        return CommandResult::InvalidReference;
                    }
                    storage.enable_control(node_id, control_id);
                    affected_chunk = Some(chunk_id);
                    CommandResult::Ok
                }
                RoadCommand::MakeIntersection {
                    node_id,
                    params,
                    chunk_id,
                    clear,
                } => {
                    if node_id.raw() as usize >= storage.node_count() {
                        return CommandResult::InvalidReference;
                    }
                    let arms = gather_arms(storage, node_id, &params, gizmo.chunk_size, gizmo);

                    storage.node_mut(node_id).arms = arms;
                    build_intersection_at_node(storage, node_id, &params, clear, gizmo);

                    affected_chunk = Some(chunk_id);
                    CommandResult::Ok
                }
                RoadCommand::UpgradeSegmentBegin {
                    old_segment,
                    chunk_id,
                } => {
                    if old_segment.raw() as usize >= storage.segment_count() {
                        return CommandResult::InvalidReference;
                    }
                    storage.disable_segment(old_segment);
                    affected_chunk = Some(chunk_id);
                    CommandResult::Ok
                }
                RoadCommand::UpgradeSegmentEnd { chunk_id, .. } => {
                    affected_chunk = Some(chunk_id);
                    CommandResult::Ok
                }
            };

            // Only update the mesh if this is not a preview and the command succeeded (chunk ID was set)
            if !is_preview {
                if let Some(chunk_id) = affected_chunk {
                    road_mesh_manager.update_chunk_mesh(
                        terrain_renderer,
                        chunk_id,
                        storage,
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

/// Applies a batch of commands in order, ensuring deterministic execution.
pub fn apply_commands(
    terrain_renderer: &TerrainSubsystem,
    road_mesh_manager: &mut RoadMeshManager,
    storage: &mut RoadStorage,
    car_subsystem: &mut CarSubsystem,
    road_style_params: &RoadStyleParams,
    is_preview: bool,
    gizmo: &mut Gizmo,
    commands: Vec<RoadEditorCommand>,
) -> Vec<CommandResult> {
    commands
        .into_iter()
        .map(|cmd| {
            apply_command(
                terrain_renderer,
                road_mesh_manager,
                car_subsystem,
                storage,
                road_style_params,
                cmd,
                is_preview,
                gizmo,
            )
        })
        .collect()
}

/// Processes preview commands and generates preview road geometry.
/// Called every frame after RoadEditor::update().
pub fn apply_preview_commands(
    terrain_renderer: &TerrainSubsystem,
    road_mesh_manager: &mut RoadMeshManager,
    preview_storage: &mut RoadStorage,
    real_storage: &RoadStorage,
    car_subsystem: &mut CarSubsystem,
    road_style_params: &RoadStyleParams,
    gizmo: &mut Gizmo,
    commands: &[RoadEditorCommand],
) {
    preview_storage.clear();

    // 1) Apply explicit Road commands immediately
    for cmd in commands {
        if let RoadEditorCommand::Road(_) = cmd {
            apply_command(
                terrain_renderer,
                road_mesh_manager,
                car_subsystem,
                preview_storage,
                road_style_params,
                cmd.clone(),
                true,
                gizmo,
            );
        }
    }

    // 2) Collect preview inputs
    let mut node_previews: Vec<&NodePreview> = Vec::new();
    let mut crossing_previews: Vec<&CrossingPoint> = Vec::new();
    let mut segment_preview: Option<&SegmentPreview> = None;

    for cmd in commands {
        match cmd {
            RoadEditorCommand::PreviewNode(n) => node_previews.push(n),
            RoadEditorCommand::PreviewSegment(s) => segment_preview = Some(s),
            RoadEditorCommand::PreviewCrossing(c) => crossing_previews.push(c),
            RoadEditorCommand::PreviewClear => return,
            _ => {}
        }
    }

    let mut allocator = PreviewIdAllocator::new();
    let mut road_commands: Vec<RoadCommand> = Vec::new();

    // 3) Crossing preview (independent)
    if !crossing_previews.is_empty() {
        road_commands.extend(generate_intersection_preview(
            terrain_renderer,
            preview_storage,
            real_storage,
            &mut allocator,
            road_style_params,
            &crossing_previews,
        ));
    }

    // 4) Segment preview (independent)
    if let Some(seg) = segment_preview {
        if seg.is_valid {
            road_commands.extend(generate_segment_preview(
                terrain_renderer,
                &mut allocator,
                road_style_params,
                seg,
            ));
        } else {
            road_commands.extend(generate_invalid_segment_preview(
                terrain_renderer,
                &mut allocator,
                road_style_params,
                seg,
            ));
        }
    }

    // 5) Hover nodes (independent, lower priority visually)
    if segment_preview.is_none() && !node_previews.is_empty() {
        road_commands.extend(generate_hover_preview(
            terrain_renderer,
            &mut allocator,
            road_style_params,
            &node_previews,
        ));
    }

    // 6) Apply all generated preview commands
    for cmd in road_commands {
        apply_command(
            terrain_renderer,
            road_mesh_manager,
            car_subsystem,
            preview_storage,
            road_style_params,
            RoadEditorCommand::Road(cmd),
            true,
            gizmo,
        );
    }
}

fn generate_intersection_preview(
    terrain_renderer: &TerrainSubsystem,
    preview_storage: &mut RoadStorage,
    real_storage: &RoadStorage,
    allocator: &mut PreviewIdAllocator,
    road_style_params: &RoadStyleParams,
    crossings: &[&CrossingPoint],
) -> Vec<RoadCommand> {
    let commands = Vec::new();
    for &crossing in crossings {
        match crossing.kind {
            CrossingKind::ExistingNode(n) => {
                // copy_real_node_to_preview(n, preview_storage, real_storage);
                // println!("hi");
                // commands.push(RoadCommand::MakeIntersection {
                //     node_id: n,
                //     chunk_id: 0,
                //     params: IntersectionBuildParams::from_style(road_style_params),
                //     clear: true,
                // });
            }
            CrossingKind::LaneCrossing { .. } => {}
        }
    }

    commands
}

struct PreviewIdAllocator {
    next_node: u32,
    next_segment: u32,
    next_lane: u32,
}

impl PreviewIdAllocator {
    fn new() -> Self {
        Self {
            next_node: 0,
            next_segment: 0,
            next_lane: 0,
        }
    }

    fn alloc_node(&mut self) -> NodeId {
        let id = NodeId::new(self.next_node);
        self.next_node += 1;
        id
    }

    fn alloc_segment(&mut self) -> SegmentId {
        let id = SegmentId::new(self.next_segment);
        self.next_segment += 1;
        id
    }

    fn alloc_lane(&mut self) -> LaneId {
        let id = LaneId::new(self.next_lane);
        self.next_lane += 1;
        id
    }
}

/// Generate preview for hover state - creates node with stub lanes so it renders
fn generate_hover_preview(
    terrain_renderer: &TerrainSubsystem,
    allocator: &mut PreviewIdAllocator,
    road_style_params: &RoadStyleParams,
    node_previews: &[&NodePreview],
) -> Vec<RoadCommand> {
    let mut commands = Vec::new();

    for node in node_previews {
        generate_node_with_stub(
            terrain_renderer,
            allocator,
            road_style_params,
            node.world_pos,
            &mut commands,
        );
    }

    commands
}
/// Generate preview for invalid segment - shows both endpoints with stubs
fn generate_invalid_segment_preview(
    terrain_renderer: &TerrainSubsystem,
    allocator: &mut PreviewIdAllocator,
    road_style_params: &RoadStyleParams,
    preview: &SegmentPreview,
) -> Vec<RoadCommand> {
    let mut commands = Vec::new();

    // Start node with stub pointing toward end
    generate_node_with_stub(
        terrain_renderer,
        allocator,
        road_style_params,
        preview.start,
        &mut commands,
    );

    let reason = preview.reason_invalid.clone().unwrap();
    if !matches!(reason, PreviewError::TooShort) {
        // End node with stub pointing toward start
        generate_node_with_stub(
            terrain_renderer,
            allocator,
            road_style_params,
            preview.end,
            &mut commands,
        );
    }

    commands
}

/// Generate full segment preview with both nodes and all lanes
fn generate_segment_preview(
    terrain_renderer: &TerrainSubsystem,
    allocator: &mut PreviewIdAllocator,
    road_style_params: &RoadStyleParams,
    preview: &SegmentPreview,
) -> Vec<RoadCommand> {
    let mut commands = Vec::new();

    // ========================================
    // STEP 1: Compute lane geometries (LANE-FIRST)
    // ========================================
    let lane_defs = compute_lane_geometries(terrain_renderer, road_style_params, &preview.polyline);

    // ========================================
    // STEP 2: Create nodes
    // ========================================
    let start_node_id = allocator.alloc_node();
    commands.push(RoadCommand::AddNode {
        world_pos: preview.start,
    });

    let end_node_id = allocator.alloc_node();
    commands.push(RoadCommand::AddNode {
        world_pos: preview.end,
    });

    // ========================================
    // STEP 3: Create segment
    // ========================================
    let segment_id = allocator.alloc_segment();
    commands.push(RoadCommand::AddSegment {
        start: start_node_id,
        end: end_node_id,
        structure: preview.road_type.structure(),
        chunk_id: 0,
    });

    // ========================================
    // STEP 4: Create lanes from pre-computed geometries
    // ========================================
    let speed = preview.road_type.speed_limit();
    let capacity = preview.road_type.capacity();
    let mask = preview.road_type.vehicle_mask();

    for lane_def in lane_defs {
        allocator.alloc_lane();

        let (from, to) = if lane_def.is_forward {
            (start_node_id, end_node_id)
        } else {
            (end_node_id, start_node_id)
        };

        commands.push(RoadCommand::AddLane {
            from,
            to,
            segment: segment_id,
            lane_index: lane_def.lane_index,
            geometry: lane_def.geometry,
            speed_limit: speed,
            capacity,
            vehicle_mask: mask,
            base_cost: lane_def.base_cost,
            chunk_id: 0,
        });
    }

    commands
}

/// Creates a node with a short stub segment and lanes so it renders properly
fn generate_node_with_stub(
    terrain_renderer: &TerrainSubsystem,
    allocator: &mut PreviewIdAllocator,
    road_style_params: &RoadStyleParams,
    position: WorldPos,
    commands: &mut Vec<RoadCommand>,
) {
    let road_type = road_style_params.road_type();

    // Main node at position
    let main_node_id = allocator.alloc_node();
    commands.push(RoadCommand::AddNode {
        world_pos: position,
    });

    // Segment connecting them
    let segment_id = allocator.alloc_segment();
    commands.push(RoadCommand::AddSegment {
        start: main_node_id,
        end: main_node_id,
        structure: road_type.structure(),
        chunk_id: 0,
    });

    // Compute and add lanes
    let centerline = vec![position, position];
    let lane_defs = compute_lane_geometries(terrain_renderer, road_style_params, &centerline);

    let speed = road_type.speed_limit();
    let capacity = road_type.capacity();
    let mask = road_type.vehicle_mask();

    for lane_def in lane_defs {
        allocator.alloc_lane();

        let (from, to) = (main_node_id, main_node_id);

        commands.push(RoadCommand::AddLane {
            from,
            to,
            segment: segment_id,
            lane_index: lane_def.lane_index,
            geometry: lane_def.geometry,
            speed_limit: speed,
            capacity,
            vehicle_mask: mask,
            base_cost: lane_def.base_cost,
            chunk_id: 0,
        });
    }
}

/// Pre-computed lane definition
struct LaneDefinition {
    lane_index: i8,
    is_forward: bool,
    geometry: LaneGeometry,
    base_cost: f32,
}

/// Compute all lane geometries from a centerline polyline
fn compute_lane_geometries(
    terrain_renderer: &TerrainSubsystem,
    road_style_params: &RoadStyleParams,
    centerline: &[WorldPos],
) -> Vec<LaneDefinition> {
    let mut lanes = Vec::new();
    let (left_count, right_count) = road_style_params.road_type().lanes_each_direction();
    let lane_width = road_style_params.road_type().lane_width;

    // Forward lanes (right side: travel from start to end)
    for i in 0..right_count {
        let lane_index = (i as i8) + 1;
        let polyline = offset_polyline(
            terrain_renderer,
            centerline,
            lane_index,
            lane_width,
            road_style_params.road_type().structure,
        );
        let geometry = LaneGeometry::from_polyline(polyline, terrain_renderer.chunk_size);
        let base_cost = geometry.total_len.max(0.1);

        lanes.push(LaneDefinition {
            lane_index,
            is_forward: true,
            geometry,
            base_cost,
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
            road_style_params.road_type().structure,
        );
        polyline.reverse();
        let geometry = LaneGeometry::from_polyline(polyline, terrain_renderer.chunk_size);
        let base_cost = geometry.total_len.max(0.1);

        lanes.push(LaneDefinition {
            lane_index,
            is_forward: false,
            geometry,
            base_cost,
        });
    }

    lanes
}

/// Cubic Bézier interpolation.
pub(crate) fn bezier3(
    p0: WorldPos,
    p1: WorldPos,
    p2: WorldPos,
    p3: WorldPos,
    t: f32,
    cs: ChunkSize,
) -> WorldPos {
    let t2 = t * t;
    let t3 = t2 * t;
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    let mt3 = mt2 * mt;

    // Convert to Vec3 relative to p0 for calculation
    let v1 = p1.to_render_pos(p0, cs);
    let v2 = p2.to_render_pos(p0, cs);
    let v3 = p3.to_render_pos(p0, cs);

    let result = v1 * (3.0 * mt2 * t) + v2 * (3.0 * mt * t2) + v3 * t3;
    p0.add_vec3(result, cs)
}
/// More precise segment-chunk intersection test.
fn segment_touches_chunk_precise(
    start: WorldPos,
    end: WorldPos,
    chunk: ChunkCoord,
    chunk_size: ChunkSize,
) -> bool {
    let cs = chunk_size as f32;

    // Chunk bounds as WorldPos
    let chunk_min = WorldPos::new(chunk, LocalPos::new(0.0, 0.0, 0.0));
    let chunk_max = WorldPos::new(chunk, LocalPos::new(cs, 0.0, cs));

    // Convert segment to chunk-local coordinates
    let a = start.to_render_pos(chunk_min, chunk_size);
    let b = end.to_render_pos(chunk_min, chunk_size);

    // 2D line-box intersection in XZ plane
    line_intersects_box_2d(
        Vec2::new(a.x, a.z),
        Vec2::new(b.x, b.z),
        Vec2::ZERO,
        Vec2::new(cs, cs),
    )
}

/// 2D line-box intersection test.
fn line_intersects_box_2d(a: Vec2, b: Vec2, box_min: Vec2, box_max: Vec2) -> bool {
    let d = b - a;
    let mut t_min = 0.0f32;
    let mut t_max = 1.0f32;

    for i in 0..2 {
        let (a_i, d_i, min_i, max_i) = match i {
            0 => (a.x, d.x, box_min.x, box_max.x),
            _ => (a.y, d.y, box_min.y, box_max.y),
        };

        if d_i.abs() < 1e-10 {
            if a_i < min_i || a_i > max_i {
                return false;
            }
        } else {
            let inv_d = 1.0 / d_i;
            let mut t1 = (min_i - a_i) * inv_d;
            let mut t2 = (max_i - a_i) * inv_d;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            t_min = t_min.max(t1);
            t_max = t_max.min(t2);
            if t_min > t_max {
                return false;
            }
        }
    }
    true
}
#[derive(Debug, Clone)]
pub(crate) enum TurnType {
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

/// Tracks a running average that forgets old data exponentially.
/// Recent reports matter more than ancient ones.
#[derive(Debug, Clone)]
pub struct ExponentialMovingAverage {
    value: f32,
    alpha: f32, // smoothing factor: 0.0 = never update, 1.0 = only latest
    count: u32, // how many samples we've seen (useful for "is this trustworthy?")
}

impl ExponentialMovingAverage {
    pub fn new(alpha: f32) -> Self {
        Self {
            value: 0.0,
            alpha,
            count: 0,
        }
    }

    /// Seed with an initial estimate (e.g., Euclidean distance / speed limit)
    /// so the first cars aren't completely blind.
    pub fn with_initial(alpha: f32, initial: f32) -> Self {
        Self {
            value: initial,
            alpha,
            count: 1,
        }
    }

    /// A car reports a new observed travel time.
    pub fn update(&mut self, sample: f32) {
        if self.count == 0 {
            // First sample: just accept it wholesale
            self.value = sample;
        } else {
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value;
        }
        self.count = self.count.saturating_add(1);
    }

    /// Current best estimate of travel time.
    pub fn get(&self) -> f32 {
        self.value
    }

    /// How many reports this is based on.
    /// Cars might trust high-count averages more than low-count ones.
    pub fn sample_count(&self) -> u32 {
        self.count
    }

    /// Is this estimate based on enough data to be meaningful?
    pub fn is_reliable(&self, min_samples: u32) -> bool {
        self.count >= min_samples
    }
}
