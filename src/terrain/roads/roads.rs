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

use crate::renderer::world_renderer::TerrainRenderer;
use crate::terrain::roads::road_editor::{
    LanePreview, NodePreview, RoadEditorCommand, SegmentPreview, SnapPreview, offset_polyline,
};
use crate::terrain::roads::road_mesh_manager::{
    ChunkId, RoadMeshManager, RoadStyleParams, chunk_x_range,
};
use glam::Vec3;

pub const METERS_PER_LANE_POLYLINE_STEP: f32 = 2.0;
struct LaneTurn {
    from: LaneId,
    to: LaneId,
    cost: f32,
    allowed: bool,
}

/// Stable, monotonically increasing node identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct NodeId(pub u32);

impl NodeId {
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    #[inline]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl From<u32> for NodeId {
    #[inline]
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<NodeId> for u32 {
    #[inline]
    fn from(id: NodeId) -> Self {
        id.0
    }
}

/// Stable, monotonically increasing segment identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct SegmentId(pub u32);

impl SegmentId {
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    #[inline]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl From<u32> for SegmentId {
    #[inline]
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<SegmentId> for u32 {
    #[inline]
    fn from(id: SegmentId) -> Self {
        id.0
    }
}

/// Stable, monotonically increasing lane identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct LaneId(u32);

impl LaneId {
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    #[inline]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl From<u32> for LaneId {
    #[inline]
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<LaneId> for u32 {
    #[inline]
    fn from(id: LaneId) -> Self {
        id.0
    }
}

pub type NodeLaneId = u32;
pub type PolyIdx = u16;

/// Stable identifier for traffic control attachments within a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ControlId(u32);

impl ControlId {
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    #[inline]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

// ============================================================================
// Core Enums
// ============================================================================

/// Physical structure type of a road segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StructureType {
    Surface,
    Bridge,
    Tunnel,
}

impl Default for StructureType {
    fn default() -> Self {
        Self::Surface
    }
}

/// Traffic signal configuration (no ticking logic).
#[derive(Debug, Clone, PartialEq)]
pub struct TrafficSignal {
    /// Duration of each phase in seconds.
    pub phase_durations: Vec<f32>,
    /// Current active phase index (updated by simulation, not by this module).
    pub current_phase: u32,
    /// Offset from global cycle start in seconds.
    pub cycle_offset: f32,
}

impl TrafficSignal {
    pub fn new(phase_durations: Vec<f32>) -> Self {
        Self {
            phase_durations,
            current_phase: 0,
            cycle_offset: 0.0,
        }
    }

    #[inline]
    pub fn total_cycle_duration(&self) -> f32 {
        self.phase_durations.iter().sum()
    }
}

/// Traffic control device attached to a node intersection.
#[derive(Debug, Clone, PartialEq)]
pub enum TrafficControl {
    None,
    Signal(TrafficSignal),
    StopSign,
    Yield,
}

impl Default for TrafficControl {
    fn default() -> Self {
        Self::None
    }
}

/// Attached control with stable ID for removal/modification.
#[derive(Debug, Clone)]
pub struct AttachedControl {
    id: ControlId,
    control: TrafficControl,
    enabled: bool,
}

impl AttachedControl {
    #[inline]
    pub fn id(&self) -> ControlId {
        self.id
    }

    #[inline]
    pub fn control(&self) -> &TrafficControl {
        &self.control
    }

    #[inline]
    pub fn control_mut(&mut self) -> &mut TrafficControl {
        &mut self.control
    }

    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

#[derive(Clone, Copy)]
pub struct RoadPreset {
    pub forward_lanes: usize,
    pub backward_lanes: usize,
    pub speed_limit: f32,
    pub capacity_per_lane: u32,
    pub vehicle_mask: u32,
    pub base_cost: f32,
    pub structure: StructureType,
}

impl Default for RoadPreset {
    fn default() -> Self {
        Self {
            // Default: basic 2-lane road (1 lane each direction) – feels like Cities: Skylines small roads
            forward_lanes: 1,
            backward_lanes: 1,
            speed_limit: 50.0,
            capacity_per_lane: 20,
            vehicle_mask: 0xFF,
            base_cost: 1.0,
            structure: StructureType::Surface,
        }
    }
}

// Editing state – stored in RoadManager
#[derive(Default)]
struct EditingState {
    active: bool,
    preset: RoadPreset,
    chain: Vec<NodeId>,              // Nodes placed in the current road chain
    placed_segments: Vec<SegmentId>, // Segments created in the current road (for undo/cancel)
}
// ============================================================================
// Node
// ============================================================================

/// Intersection anchor point in 3D space.
/// Every node is an intersection with attachable traffic controls.
#[derive(Debug, Clone)]
pub struct Node {
    x: f32,
    y: f32,
    z: f32,
    chunk_id: ChunkId,
    enabled: bool,
    node_lanes: Vec<NodeLane>,
    incoming_lanes: Vec<LaneId>,
    outgoing_lanes: Vec<LaneId>,
    attached_controls: Vec<AttachedControl>,
    next_control_id: u32,
}

impl Node {
    pub fn new(x: f32, y: f32, z: f32, chunk_id: ChunkId) -> Self {
        Self {
            x,
            y,
            z,
            chunk_id,
            enabled: true,
            node_lanes: Vec::new(),
            incoming_lanes: Vec::new(),
            outgoing_lanes: Vec::new(),
            attached_controls: Vec::new(),
            next_control_id: 0,
        }
    }

    pub(crate) fn version(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis

        #[inline(always)]
        fn mix(h: &mut u64, v: u64) {
            *h ^= v;
            *h = h.wrapping_mul(0x100000001b3);
        }

        mix(&mut h, self.x.to_bits() as u64);
        mix(&mut h, self.y.to_bits() as u64);
        mix(&mut h, self.z.to_bits() as u64);

        mix(&mut h, self.chunk_id as u64);
        mix(&mut h, self.enabled as u64);
        mix(&mut h, self.next_control_id as u64);

        mix(&mut h, self.attached_controls.len() as u64);
        mix(&mut h, self.incoming_lanes.len() as u64);
        mix(&mut h, self.outgoing_lanes.len() as u64);

        h
    }

    #[inline]
    pub fn x(&self) -> f32 {
        self.x
    }

    #[inline]
    pub fn y(&self) -> f32 {
        self.y
    }

    #[inline]
    pub fn z(&self) -> f32 {
        self.z
    }

    #[inline]
    pub fn position(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    #[inline]
    pub fn chunk_id(&self) -> ChunkId {
        self.chunk_id
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

    #[inline]
    pub fn incoming_lanes(&self) -> &[LaneId] {
        &self.incoming_lanes
    }

    #[inline]
    pub fn outgoing_lanes(&self) -> &[LaneId] {
        &self.outgoing_lanes
    }

    #[inline]
    pub fn attached_controls(&self) -> &[AttachedControl] {
        &self.attached_controls
    }
    /// Returns true if the node has any active traffic control.
    #[inline]
    pub fn has_active_control(&self) -> bool {
        self.attached_controls
            .iter()
            .any(|c| c.enabled && !matches!(c.control, TrafficControl::None))
    }

    /// Returns the count of connected lanes (incoming + outgoing).
    #[inline]
    pub fn connection_count(&self) -> usize {
        self.incoming_lanes.len() + self.outgoing_lanes.len()
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
        for &lane_id in &self.lanes {
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
    pub(crate) fn geometry(&self) -> &LaneGeometry {
        &self.geometry
    }
    #[inline]
    pub fn polyline(&self) -> &Vec<Vec3> {
        &self.geometry.points
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LaneRef {
    Segment(LaneId, PolyIdx),
    NodeLane(NodeLaneId, PolyIdx),
}

/// Directed lane edge connecting two segments within a node.
/// NodeLanes are the primary graph edges for pathfinding and simulation.
#[derive(Debug, Clone)]
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
    pub fn polyline(&self) -> &Vec<Vec3> {
        &self.geometry.points
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
#[derive(Clone, Debug)]
pub struct LaneGeometry {
    pub points: Vec<Vec3>, // polyline
    pub lengths: Vec<f32>, // cumulative arc length
    pub total_len: f32,
}

impl LaneGeometry {
    pub fn from_polyline(points: Vec<Vec3>) -> Self {
        debug_assert!(points.len() >= 2);

        let mut lengths = Vec::with_capacity(points.len());
        let mut total_len = 0.0;

        lengths.push(0.0);

        for i in 1..points.len() {
            total_len += points[i].distance(points[i - 1]);
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
    pub fn add_node(&mut self, x: f32, y: f32, z: f32, chunk_id: ChunkId) -> NodeId {
        let id = NodeId::new(self.nodes.len() as u32);
        self.nodes.push(Node::new(x, y, z, chunk_id));
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
            let lane = self.lane(*lane_id);
            if lane.from == seg_node_a_id && lane.to == seg_node_b_id {
                right_lanes += 1;
            } else {
                left_lanes += 1;
            }
        }
        (left_lanes, right_lanes)
    }
    pub fn segments_connected_to_node(&self, node_id: NodeId) -> Vec<SegmentId> {
        let Some(node) = self.node(node_id) else {
            return Vec::new();
        };

        let mut segments = Vec::new();

        for &lane_id in node.incoming_lanes().iter().chain(node.outgoing_lanes()) {
            let lane = &self.lanes[lane_id.0 as usize];
            let seg = lane.segment();

            // de-dup without HashSet (cheap, small N…)
            if !segments.contains(&seg) {
                segments.push(seg);
            }
        }

        segments
    }
    pub fn segment_count_connected_to_node(&self, node_id: NodeId) -> usize {
        let Some(node) = self.node(node_id) else {
            return 0;
        };

        let mut count = 0;
        let mut seen = Vec::new();

        for &lane_id in node.incoming_lanes().iter().chain(node.outgoing_lanes()) {
            let seg = self.lanes[lane_id.0 as usize].segment();
            if !seen.contains(&seg) {
                seen.push(seg);
                count += 1;
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

    pub fn segment_ids_touching_chunk(&self, chunk_id: ChunkId) -> Vec<SegmentId> {
        let (min_x, max_x) = chunk_x_range(chunk_id);

        self.segments
            .iter()
            .enumerate()
            .filter(|(idx, seg)| {
                if !seg.enabled {
                    return false;
                }

                // Assuming NodeId indices are valid and within bounds
                let start = self.nodes.get(seg.start.raw() as usize);
                let end = self.nodes.get(seg.end.raw() as usize);

                match (start, end) {
                    (Some(start), Some(end)) => {
                        let seg_min_x = f32::min(start.x, end.x);
                        let seg_max_x = f32::max(start.x, end.x);

                        seg_max_x >= min_x && seg_min_x < max_x
                    }
                    _ => false,
                }
            })
            .map(|(idx, _)| SegmentId::new(idx as u32)) // Index == raw SegmentId because of append-only monotonic allocation
            .collect()
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
    pub fn lane(&self, id: LaneId) -> &Lane {
        &self.lanes[id.0 as usize]
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

// ============================================================================
// Spatial Helpers
// ============================================================================

/// Computes the 3D length of a lane using linear distance between node anchors.
#[inline]
pub fn lane_length(lane: &Lane, storage: &RoadStorage) -> f32 {
    let Some(from) = storage.node(lane.from_node()) else {
        return 0.0;
    };
    let Some(to) = storage.node(lane.to_node()) else {
        return 0.0;
    };
    let dx = to.x() - from.x();
    let dy = to.y() - from.y();
    let dz = to.z() - from.z();
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Computes the 2D (XY) length of a lane.
#[inline]
pub fn lane_length_2d(lane: &Lane, storage: &RoadStorage) -> f32 {
    let Some(from) = storage.node(lane.from_node()) else {
        return 0.0;
    };
    let Some(to) = storage.node(lane.to_node()) else {
        return 0.0;
    };
    let dx = to.x() - from.x();
    let dz = to.z() - from.z();
    (dx * dx + dz * dz).sqrt()
}

/// Samples the height (z) along a lane at parameter t in [0,1].
/// Uses the segment's vertical profile for interpolation.
// #[inline]
// pub fn sample_lane_height(lane: &Lane, t: f32, manager: &RoadManager) -> f32 {
//     let segment = manager.segment(lane.segment());
//     let Some(from) = manager.node(lane.from_node()) else {
//         return 0.0;
//     };
//     let Some(to) = manager.node(lane.to_node()) else {
//         return 0.0;
//     };
//
//     // Determine if lane direction matches segment direction
//     let reversed = lane.from_node() != segment.start();
//     let t_seg = if reversed { 1.0 - t } else { t };
//     from.y()
// }

/// Samples the 3D position along a lane at parameter t in [0,1].
#[inline]
pub fn sample_lane_position(lane: &Lane, t: f32, storage: &RoadStorage) -> (f32, f32) {
    let Some(from) = storage.node(lane.from_node()) else {
        return (0.0, 0.0);
    };
    let Some(to) = storage.node(lane.to_node()) else {
        return (0.0, 0.0);
    };

    let x = from.x() + (to.x() - from.x()) * t;
    let z = from.z() + (to.z() - from.z()) * t;

    (x, z)
}

/// Projects a 2D point onto a lane and returns (t, distance_squared).
/// t is the parameter [0,1] along the lane; dist_sq is squared XY distance.
#[inline]
pub fn project_point_to_lane_xz(lane: &Lane, x: f32, z: f32, storage: &RoadStorage) -> (f32, f32) {
    let Some(from) = storage.node(lane.from_node()) else {
        return (0.0, 0.0);
    };
    let Some(to) = storage.node(lane.to_node()) else {
        return (0.0, 0.0);
    };

    let ax = from.x();
    let az = from.z();
    let bx = to.x();
    let bz = to.z();

    let dx = bx - ax;
    let dz = bz - az;
    let len_sq = dx * dx + dz * dz;

    if len_sq < 1e-10 {
        // Degenerate segment (zero length)
        let px = x - ax;
        let pz = z - az;
        return (0.0, px * px + pz * pz);
    }

    // Project point onto line
    let t = ((x - ax) * dx + (z - az) * dz) / len_sq;
    let t_clamped = t.clamp(0.0, 1.0);

    // Compute closest point on segment
    let cx = ax + t_clamped * dx;
    let cz = az + t_clamped * dz;

    let dist_sq = (x - cx) * (x - cx) + (z - cz) * (z - cz);

    (t_clamped, dist_sq)
}

/// Finds the nearest enabled lane to a 3D point (brute force).
/// Returns None if no enabled lanes exist.
///
/// Note: For production use, build chunk-local spatial indexes.
/// This function is O(n) in the number of lanes.
pub fn nearest_lane_to_point(storage: &RoadStorage, x: f32, _y: f32, z: f32) -> Option<LaneId> {
    let mut best_id: Option<LaneId> = None;
    let mut best_dist_sq = f32::MAX;

    for (id, lane) in storage.iter_lanes() {
        if !lane.is_enabled() {
            continue;
        }

        let (_, dist_sq) = project_point_to_lane_xz(lane, x, z, storage);
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best_id = Some(id);
        }
    }

    best_id
}

/// Finds the k nearest enabled lanes to a point (brute force).
/// Returns lanes sorted by distance, closest first.
pub fn k_nearest_lanes_to_point(
    storage: &RoadStorage,
    x: f32,
    z: f32,
    k: usize,
) -> Vec<(LaneId, f32)> {
    let mut candidates: Vec<(LaneId, f32)> = Vec::with_capacity(storage.lane_count());

    for (id, lane) in storage.iter_lanes() {
        if !lane.is_enabled() {
            continue;
        }
        let (_, dist_sq) = project_point_to_lane_xz(lane, x, z, storage);
        candidates.push((id, dist_sq));
    }

    // Sort by distance (deterministic: stable sort by dist, then by id for ties)
    candidates.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.raw().cmp(&b.0.raw()))
    });

    candidates.truncate(k);
    candidates
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
        x: f32,
        y: f32,
        z: f32,
        chunk_id: ChunkId,
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
        id: NodeLaneId,
        merging: Vec<LaneRef>,
        splitting: Vec<LaneRef>,
        geometry: LaneGeometry,
        // Cached costs for pathfinding
        length_m: f32,
        lane_width: f32,
        speed_limit: f32,
        capacity: u32,
        vehicle_mask: u32,
        base_cost: f32,
        dynamic_cost: f32,
        chunk_id: ChunkId,
    },
    /// Disable a node.
    DisableNode { node_id: NodeId, chunk_id: ChunkId },
    /// Enable a node.
    EnableNode { node_id: NodeId, chunk_id: ChunkId },
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
    DisableLane { lane_id: LaneId, chunk_id: ChunkId },
    /// Enable a lane.
    EnableLane { lane_id: LaneId, chunk_id: ChunkId },
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

/// Applies a command to the road manager deterministically.
/// Returns the result of the operation.
///
/// # Panics
/// Panics only on programmer errors (debug assertions).
/// Applies a command to the road manager deterministically and updates the mesh.
pub fn apply_command(
    terrain_renderer: &TerrainRenderer,
    road_mesh_manager: &mut RoadMeshManager,
    storage: &mut RoadStorage,
    command: RoadEditorCommand,
    is_preview: bool,
) -> CommandResult {
    match command {
        RoadEditorCommand::Road(road_command) => {
            // We store the chunk ID here if an operation succeeds
            let affected_chunk: Option<ChunkId>;

            let result = match road_command {
                RoadCommand::AddNode { x, y, z, chunk_id } => {
                    let id = storage.add_node(x, y, z, chunk_id);
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
                    ..
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
                    road_mesh_manager.update_chunk_mesh(terrain_renderer, chunk_id, storage);
                }
            }

            result
        }
        _ => CommandResult::Ok,
    }
}

/// Applies a batch of commands in order, ensuring deterministic execution.
pub fn apply_commands(
    terrain_renderer: &TerrainRenderer,
    road_mesh_manager: &mut RoadMeshManager,
    storage: &mut RoadStorage,
    is_preview: bool,
    commands: Vec<RoadEditorCommand>,
) -> Vec<CommandResult> {
    commands
        .into_iter()
        .map(|cmd| {
            apply_command(
                terrain_renderer,
                road_mesh_manager,
                storage,
                cmd,
                is_preview,
            )
        })
        .collect()
}

/// Processes preview commands and generates preview road geometry.
/// Called every frame after RoadEditor::update().
pub fn apply_preview_commands(
    terrain_renderer: &TerrainRenderer,
    road_mesh_manager: &mut RoadMeshManager,
    preview_storage: &mut RoadStorage,
    commands: &[RoadEditorCommand],
) {
    let road_style_params: &RoadStyleParams = &road_mesh_manager.style;
    preview_storage.clear();

    // Collect all preview data from this frame
    let mut snap_preview: Option<&SnapPreview> = None;
    let mut node_previews: Vec<&NodePreview> = Vec::new();
    let mut lane_preview: Option<&LanePreview> = None;
    let mut segment_preview: Option<&SegmentPreview> = None;

    for cmd in commands {
        match cmd {
            RoadEditorCommand::PreviewSnap(s) => snap_preview = Some(s),
            RoadEditorCommand::PreviewNode(n) => node_previews.push(n),
            RoadEditorCommand::PreviewLane(l) => lane_preview = Some(l),
            RoadEditorCommand::PreviewSegment(s) => segment_preview = Some(s),
            RoadEditorCommand::PreviewClear => {
                // Already cleared above
                return;
            }
            _ => {}
        }
    }

    let chunk_id: ChunkId = 0;
    let mut allocator = PreviewIdAllocator::new();

    // Generate commands based on preview state
    let road_commands = match segment_preview {
        Some(seg) if seg.is_valid => {
            // Full segment preview: create nodes, segment, and lanes
            generate_segment_preview(
                terrain_renderer,
                &mut allocator,
                road_style_params,
                seg,
                &node_previews,
                chunk_id,
            )
        }
        Some(_seg) => {
            // Invalid segment - show nodes only to indicate why it's invalid
            generate_nodes_only(&mut allocator, &node_previews, chunk_id)
        }
        None => {
            // No segment being drawn - show hover state
            generate_hover_preview(&mut allocator, &node_previews, lane_preview, chunk_id)
        }
    };

    // Apply all generated commands to create preview geometry
    for cmd in road_commands {
        apply_command(
            terrain_renderer,
            road_mesh_manager,
            preview_storage,
            RoadEditorCommand::Road(cmd),
            true,
        );
    }
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

/// Generate preview for hover state (no segment being drawn)
fn generate_hover_preview(
    allocator: &mut PreviewIdAllocator,
    node_previews: &[&NodePreview],
    lane_preview: Option<&LanePreview>,
    chunk_id: ChunkId,
) -> Vec<RoadCommand> {
    let mut commands = Vec::new();

    // Create preview nodes for all node previews
    for node in node_previews {
        allocator.alloc_node();
        commands.push(RoadCommand::AddNode {
            x: node.world_pos.x,
            y: node.world_pos.y,
            z: node.world_pos.z,
            chunk_id,
        });
    }

    // If hovering over a lane (potential split point), we could show that lane highlighted
    // The lane already exists in real storage, so we don't recreate it in preview
    // But we could create a marker/indicator

    commands
}

/// Generate preview nodes only (for invalid segment states)
fn generate_nodes_only(
    allocator: &mut PreviewIdAllocator,
    node_previews: &[&NodePreview],
    chunk_id: ChunkId,
) -> Vec<RoadCommand> {
    let mut commands = Vec::new();

    for node in node_previews {
        allocator.alloc_node();
        commands.push(RoadCommand::AddNode {
            x: node.world_pos.x,
            y: node.world_pos.y,
            z: node.world_pos.z,
            chunk_id,
        });
    }

    commands
}

/// Generate full segment preview with nodes and lanes
/// Lane-first: we compute lane geometries first, then wrap them in the required structure
fn generate_segment_preview(
    terrain_renderer: &TerrainRenderer,
    allocator: &mut PreviewIdAllocator,
    road_style_params: &RoadStyleParams,
    preview: &SegmentPreview,
    node_previews: &[&NodePreview],
    chunk_id: ChunkId,
) -> Vec<RoadCommand> {
    let mut commands = Vec::new();

    // ========================================
    // STEP 1: Compute lane geometries (LANE-FIRST)
    // ========================================
    let lane_geometries = compute_lane_geometries(terrain_renderer, road_style_params, preview);

    // ========================================
    // STEP 2: Create nodes (derived from lane endpoints)
    // ========================================
    let start_node_id = allocator.alloc_node();
    commands.push(RoadCommand::AddNode {
        x: preview.start.x,
        y: preview.start.y,
        z: preview.start.z,
        chunk_id,
    });

    let end_node_id = allocator.alloc_node();
    commands.push(RoadCommand::AddNode {
        x: preview.end.x,
        y: preview.end.y,
        z: preview.end.z,
        chunk_id,
    });

    // ========================================
    // STEP 3: Create segment (container for lanes)
    // ========================================
    let segment_id = allocator.alloc_segment();
    commands.push(RoadCommand::AddSegment {
        start: start_node_id,
        end: end_node_id,
        structure: preview.road_type.structure(),
        chunk_id,
    });

    // ========================================
    // STEP 4: Create lanes from pre-computed geometries
    // ========================================
    let speed = preview.road_type.speed_limit();
    let capacity = preview.road_type.capacity();
    let mask = preview.road_type.vehicle_mask();

    for lane_def in lane_geometries {
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
            chunk_id,
        });
    }

    commands
}

/// Pre-computed lane definition
struct LaneDefinition {
    lane_index: i8,
    is_forward: bool,
    geometry: LaneGeometry,
    base_cost: f32,
}

/// Compute all lane geometries from the segment centerline (LANE-FIRST)
fn compute_lane_geometries(
    terrain_renderer: &TerrainRenderer,
    road_style_params: &RoadStyleParams,
    preview: &SegmentPreview,
) -> Vec<LaneDefinition> {
    let mut lanes = Vec::new();
    let (left_count, right_count) = preview.lane_count_each_dir;
    let lane_width = road_style_params.lane_width;

    // Forward lanes (right side: travel from start to end)
    for i in 0..right_count {
        let lane_index = (i as i8) + 1;
        let polyline = offset_polyline(terrain_renderer, &preview.polyline, lane_index, lane_width);
        let geometry = LaneGeometry::from_polyline(polyline);
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
        let mut polyline =
            offset_polyline(terrain_renderer, &preview.polyline, lane_index, lane_width);
        polyline.reverse(); // Reverse for opposite direction
        let geometry = LaneGeometry::from_polyline(polyline);
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

#[inline]
fn bezier2(a: Vec3, b: Vec3, c: Vec3, t: f32) -> Vec3 {
    let u = 1.0 - t;
    a * (u * u) + b * (2.0 * u * t) + c * (t * t)
}

#[inline]
fn bezier3(a: Vec3, b: Vec3, c: Vec3, d: Vec3, t: f32) -> Vec3 {
    let u = 1.0 - t;
    a * (u * u * u) + b * (3.0 * u * u * t) + c * (3.0 * u * t * t) + d * (t * t * t)
}
