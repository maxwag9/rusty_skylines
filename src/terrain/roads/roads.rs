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

use crate::hsv::lerp;
use crate::terrain::roads::road_mesh_manager::{ChunkId, HorizontalProfile, chunk_x_range};

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

/// Vertical elevation profile between segment endpoints.
/// Linear interpolation is used for Flat and Linear variants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VerticalProfile {
    /// Constant elevation matching start node z.
    Flat,
    /// Linear interpolation between explicit z values.
    Linear { start_z: f32, end_z: f32 },
    /// Reference to external profile data (splines, etc.).
    Custom { profile_id: u32 },
}

impl VerticalProfile {
    pub(crate) fn evaluate(&self, t: f32) -> f32 {
        match self {
            VerticalProfile::Flat => 0.0,
            VerticalProfile::Linear { start_z, end_z } => lerp(*start_z, *end_z, t),
            VerticalProfile::Custom { .. } => 0.0,
        }
    }
    #[inline]
    pub fn slope(&self) -> f32 {
        match self {
            VerticalProfile::Flat => 0.0,
            VerticalProfile::Linear { start_z, end_z } => end_z - start_z,
            VerticalProfile::Custom { .. } => 0.0,
        }
    }
}

impl Default for VerticalProfile {
    fn default() -> Self {
        Self::Flat
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

// ============================================================================
// Node
// ============================================================================

/// Intersection anchor point in 3D space.
/// Every node is an intersection with attachable traffic controls.
#[derive(Debug, Clone)]
pub struct Node {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub chunk_id: ChunkId,
    pub enabled: bool,
    pub attached_controls: Vec<AttachedControl>,
    pub incoming_lanes: Vec<LaneId>,
    pub outgoing_lanes: Vec<LaneId>,
    pub next_control_id: u32,
}

impl Node {
    fn new(x: f32, y: f32, z: f32, chunk_id: ChunkId) -> Self {
        Self {
            x,
            y,
            z,
            chunk_id,
            enabled: true,
            attached_controls: Vec::new(),
            incoming_lanes: Vec::new(),
            outgoing_lanes: Vec::new(),
            next_control_id: 0,
        }
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
    pub fn position(&self) -> (f32, f32, f32) {
        (self.x, self.y, self.z)
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
    pub fn is_intersection(&self) -> bool {
        true
    }

    #[inline]
    pub fn attached_controls(&self) -> &[AttachedControl] {
        &self.attached_controls
    }

    #[inline]
    pub fn incoming_lanes(&self) -> &[LaneId] {
        &self.incoming_lanes
    }

    #[inline]
    pub fn outgoing_lanes(&self) -> &[LaneId] {
        &self.outgoing_lanes
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
    pub horizontal_profile: HorizontalProfile,
    pub(crate) vertical_profile: VerticalProfile,
    pub(crate) version: u32,
}

impl Segment {
    fn new(
        start: NodeId,
        end: NodeId,
        structure: StructureType,
        horizontal_profile: HorizontalProfile,
        vertical_profile: VerticalProfile,
    ) -> Self {
        Self {
            start,
            end,
            enabled: true,
            lanes: Vec::new(),
            structure,
            horizontal_profile,
            vertical_profile,
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
    pub fn vertical_profile(&self) -> &VerticalProfile {
        &self.vertical_profile
    }

    #[inline]
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Returns lane count in each direction (forward, backward).
    pub fn lane_counts(&self, manager: &RoadManager) -> (usize, usize) {
        let mut forward = 0;
        let mut backward = 0;
        for &lane_id in &self.lanes {
            let lane = manager.lane(lane_id);
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
    enabled: bool,
    speed_limit: f32,
    capacity: u32,
    vehicle_mask: u32,
    base_cost: f32,
    dynamic_cost: f32,
}

impl Lane {
    fn new(
        from: NodeId,
        to: NodeId,
        segment: SegmentId,
        speed_limit: f32,
        capacity: u32,
        vehicle_mask: u32,
        base_cost: f32,
    ) -> Self {
        Self {
            from,
            to,
            segment,
            enabled: true,
            speed_limit,
            capacity,
            vehicle_mask,
            base_cost,
            dynamic_cost: 0.0,
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
}

// ============================================================================
// RoadManager
// ============================================================================

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
    nodes: Vec<Node>,
    segments: Vec<Segment>,
    lanes: Vec<Lane>,
}

// Safety: RoadManager uses no interior mutability.
// All mutable access is explicitly controlled by the caller.

impl RoadManager {
    /// Creates an empty road topology.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            segments: Vec::new(),
            lanes: Vec::new(),
        }
    }

    /// Creates a road topology with pre-allocated capacity.
    pub fn with_capacity(nodes: usize, segments: usize, lanes: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(nodes),
            segments: Vec::with_capacity(segments),
            lanes: Vec::with_capacity(lanes),
        }
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
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
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

    // ------------------------------------------------------------------------
    // Segment operations
    // ------------------------------------------------------------------------

    /// Adds a new road segment connecting two nodes.
    pub fn add_segment(
        &mut self,
        start: NodeId,
        end: NodeId,
        structure: StructureType,
        horizontal_profile: HorizontalProfile,
        vertical_profile: VerticalProfile,
    ) -> SegmentId {
        let id = SegmentId::new(self.segments.len() as u32);
        self.segments.push(Segment::new(
            start,
            end,
            structure,
            horizontal_profile,
            vertical_profile,
        ));
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
        speed_limit: f32,
        capacity: u32,
        vehicle_mask: u32,
        base_cost: f32,
    ) -> LaneId {
        // Validate lane connects segment endpoints
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
            speed_limit,
            capacity,
            vehicle_mask,
            base_cost,
        ));

        // Register lane with segment
        self.segments[segment.0 as usize].lanes.push(id);

        // Register lane with nodes
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

    // ------------------------------------------------------------------------
    // Chunk view
    // ------------------------------------------------------------------------

    /// Creates a lightweight read-only view of lanes for a specific chunk.
    /// Returns lanes whose from or to nodes belong to the specified chunk.
    ///
    /// Note: For production use, build chunk-local spatial indexes.
    pub fn lane_view_for_chunk(&self, chunk_id: ChunkId) -> LaneGraphView<'_> {
        let mut lane_ids = Vec::new();

        for (id, lane) in self.iter_lanes() {
            if !lane.is_enabled() {
                continue;
            }
            let from_chunk = self.node(lane.from_node()).chunk_id();
            let to_chunk = self.node(lane.to_node()).chunk_id();
            if from_chunk == chunk_id || to_chunk == chunk_id {
                lane_ids.push(id);
            }
        }

        LaneGraphView {
            manager: self,
            lane_ids,
        }
    }

    /// Returns nodes belonging to a specific chunk.
    pub fn nodes_in_chunk(&self, chunk_id: ChunkId) -> Vec<NodeId> {
        self.iter_nodes()
            .filter(|(_, n)| n.is_enabled() && n.chunk_id() == chunk_id)
            .map(|(id, _)| id)
            .collect()
    }
}

impl Default for RoadManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// LaneGraphView
// ============================================================================

/// Zero-copy read-only view of lanes within a chunk.
/// Created by `RoadManager::lane_view_for_chunk`.
pub struct LaneGraphView<'a> {
    manager: &'a RoadManager,
    lane_ids: Vec<LaneId>,
}

impl<'a> LaneGraphView<'a> {
    /// Returns the lane IDs in this view.
    #[inline]
    pub fn lane_ids(&self) -> &[LaneId] {
        &self.lane_ids
    }

    /// Returns a lane by ID.
    #[inline]
    pub fn lane(&self, id: LaneId) -> &Lane {
        self.manager.lane(id)
    }

    /// Returns a node by ID.
    #[inline]
    pub fn node(&self, id: NodeId) -> &Node {
        self.manager.node(id)
    }

    /// Returns a segment by ID.
    #[inline]
    pub fn segment(&self, id: SegmentId) -> &Segment {
        self.manager.segment(id)
    }

    /// Iterates over lanes in this view.
    #[inline]
    pub fn iter_lanes(&self) -> impl Iterator<Item = (LaneId, &Lane)> {
        self.lane_ids.iter().map(|&id| (id, self.manager.lane(id)))
    }

    /// Returns the number of lanes in this view.
    #[inline]
    pub fn len(&self) -> usize {
        self.lane_ids.len()
    }

    /// Returns true if this view contains no lanes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lane_ids.is_empty()
    }

    /// Returns the underlying manager reference.
    #[inline]
    pub fn manager(&self) -> &RoadManager {
        self.manager
    }
}

// ============================================================================
// Spatial Helpers
// ============================================================================

/// Computes the 3D length of a lane using linear distance between node anchors.
#[inline]
pub fn lane_length(lane: &Lane, manager: &RoadManager) -> f32 {
    let from = manager.node(lane.from_node());
    let to = manager.node(lane.to_node());
    let dx = to.x() - from.x();
    let dy = to.y() - from.y();
    let dz = to.z() - from.z();
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Computes the 2D (XY) length of a lane.
#[inline]
pub fn lane_length_2d(lane: &Lane, manager: &RoadManager) -> f32 {
    let from = manager.node(lane.from_node());
    let to = manager.node(lane.to_node());
    let dx = to.x() - from.x();
    let dy = to.y() - from.y();
    (dx * dx + dy * dy).sqrt()
}

/// Samples the height (z) along a lane at parameter t in [0,1].
/// Uses the segment's vertical profile for interpolation.
#[inline]
pub fn sample_lane_height(lane: &Lane, t: f32, manager: &RoadManager) -> f32 {
    let segment = manager.segment(lane.segment());
    let from = manager.node(lane.from_node());
    let to = manager.node(lane.to_node());

    // Determine if lane direction matches segment direction
    let reversed = lane.from_node() != segment.start();
    let t_seg = if reversed { 1.0 - t } else { t };

    match segment.vertical_profile() {
        VerticalProfile::Flat => from.z(),
        VerticalProfile::Linear { start_z, end_z } => start_z + (end_z - start_z) * t_seg,
        VerticalProfile::Custom { .. } => {
            // Custom profiles resolved elsewhere; fall back to linear node z
            from.z() + (to.z() - from.z()) * t
        }
    }
}

/// Samples the 3D position along a lane at parameter t in [0,1].
#[inline]
pub fn sample_lane_position(lane: &Lane, t: f32, manager: &RoadManager) -> (f32, f32, f32) {
    let from = manager.node(lane.from_node());
    let to = manager.node(lane.to_node());

    let x = from.x() + (to.x() - from.x()) * t;
    let y = from.y() + (to.y() - from.y()) * t;
    let z = sample_lane_height(lane, t, manager);

    (x, y, z)
}

/// Projects a 2D point onto a lane and returns (t, distance_squared).
/// t is the parameter [0,1] along the lane; dist_sq is squared XY distance.
#[inline]
pub fn project_point_to_lane_xy(lane: &Lane, x: f32, y: f32, manager: &RoadManager) -> (f32, f32) {
    let from = manager.node(lane.from_node());
    let to = manager.node(lane.to_node());

    let ax = from.x();
    let ay = from.y();
    let bx = to.x();
    let by = to.y();

    let dx = bx - ax;
    let dy = by - ay;
    let len_sq = dx * dx + dy * dy;

    if len_sq < 1e-10 {
        // Degenerate segment (zero length)
        let px = x - ax;
        let py = y - ay;
        return (0.0, px * px + py * py);
    }

    // Project point onto line
    let t = ((x - ax) * dx + (y - ay) * dy) / len_sq;
    let t_clamped = t.clamp(0.0, 1.0);

    // Compute closest point on segment
    let cx = ax + t_clamped * dx;
    let cy = ay + t_clamped * dy;

    let dist_sq = (x - cx) * (x - cx) + (y - cy) * (y - cy);

    (t_clamped, dist_sq)
}

/// Finds the nearest enabled lane to a 3D point (brute force).
/// Returns None if no enabled lanes exist.
///
/// Note: For production use, build chunk-local spatial indexes.
/// This function is O(n) in the number of lanes.
pub fn nearest_lane_to_point(manager: &RoadManager, x: f32, y: f32, _z: f32) -> Option<LaneId> {
    let mut best_id: Option<LaneId> = None;
    let mut best_dist_sq = f32::MAX;

    for (id, lane) in manager.iter_lanes() {
        if !lane.is_enabled() {
            continue;
        }

        let (_, dist_sq) = project_point_to_lane_xy(lane, x, y, manager);
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
    manager: &RoadManager,
    x: f32,
    y: f32,
    k: usize,
) -> Vec<(LaneId, f32)> {
    let mut candidates: Vec<(LaneId, f32)> = Vec::with_capacity(manager.lane_count());

    for (id, lane) in manager.iter_lanes() {
        if !lane.is_enabled() {
            continue;
        }
        let (_, dist_sq) = project_point_to_lane_xy(lane, x, y, manager);
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
    pub fn compute(manager: &RoadManager, chunk_id: ChunkId) -> Self {
        let mut incoming = Vec::new();
        let mut outgoing = Vec::new();

        for (id, lane) in manager.iter_enabled_lanes() {
            let from_chunk = manager.node(lane.from_node()).chunk_id();
            let to_chunk = manager.node(lane.to_node()).chunk_id();

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
pub enum Command {
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
        horizontal_profile: HorizontalProfile,
        vertical_profile: VerticalProfile,
    },
    /// Add a new lane to a segment.
    AddLane {
        from: NodeId,
        to: NodeId,
        segment: SegmentId,
        speed_limit: f32,
        capacity: u32,
        vehicle_mask: u32,
        base_cost: f32,
    },
    /// Disable a node.
    DisableNode { node_id: NodeId },
    /// Enable a node.
    EnableNode { node_id: NodeId },
    /// Disable a segment and its lanes.
    DisableSegment { segment_id: SegmentId },
    /// Enable a segment.
    EnableSegment { segment_id: SegmentId },
    /// Disable a lane.
    DisableLane { lane_id: LaneId },
    /// Enable a lane.
    EnableLane { lane_id: LaneId },
    /// Attach a traffic control to a node.
    AttachControl {
        node_id: NodeId,
        control: TrafficControl,
    },
    /// Disable a traffic control.
    DisableControl {
        node_id: NodeId,
        control_id: ControlId,
    },
    /// Enable a traffic control.
    EnableControl {
        node_id: NodeId,
        control_id: ControlId,
    },
    /// Begin segment upgrade (disables old segment).
    UpgradeSegmentBegin { old_segment: SegmentId },
    /// End segment upgrade (records new segment IDs for replay).
    UpgradeSegmentEnd { new_segments: Vec<SegmentId> },
}

/// Result of applying a command.
#[derive(Debug, Clone)]
pub enum CommandResult {
    /// Node was created with this ID.
    NodeCreated(NodeId),
    /// Segment was created with this ID.
    SegmentCreated(SegmentId),
    /// Lane was created with this ID.
    LaneCreated(LaneId),
    /// Control was attached with this ID.
    ControlAttached(ControlId),
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
pub fn apply_command(manager: &mut RoadManager, command: &Command) -> CommandResult {
    match command {
        Command::AddNode { x, y, z, chunk_id } => {
            let id = manager.add_node(*x, *y, *z, *chunk_id);
            CommandResult::NodeCreated(id)
        }
        Command::AddSegment {
            start,
            end,
            structure,
            horizontal_profile,
            vertical_profile,
        } => {
            if start.raw() as usize >= manager.node_count()
                || end.raw() as usize >= manager.node_count()
            {
                return CommandResult::InvalidReference;
            }
            let id = manager.add_segment(
                *start,
                *end,
                *structure,
                *horizontal_profile,
                *vertical_profile,
            );
            CommandResult::SegmentCreated(id)
        }
        Command::AddLane {
            from,
            to,
            segment,
            speed_limit,
            capacity,
            vehicle_mask,
            base_cost,
        } => {
            if from.raw() as usize >= manager.node_count()
                || to.raw() as usize >= manager.node_count()
                || segment.raw() as usize >= manager.segment_count()
            {
                return CommandResult::InvalidReference;
            }
            let id = manager.add_lane(
                *from,
                *to,
                *segment,
                *speed_limit,
                *capacity,
                *vehicle_mask,
                *base_cost,
            );
            CommandResult::LaneCreated(id)
        }
        Command::DisableNode { node_id } => {
            if node_id.raw() as usize >= manager.node_count() {
                return CommandResult::InvalidReference;
            }
            manager.disable_node(*node_id);
            CommandResult::Ok
        }
        Command::EnableNode { node_id } => {
            if node_id.raw() as usize >= manager.node_count() {
                return CommandResult::InvalidReference;
            }
            manager.enable_node(*node_id);
            CommandResult::Ok
        }
        Command::DisableSegment { segment_id } => {
            if segment_id.raw() as usize >= manager.segment_count() {
                return CommandResult::InvalidReference;
            }
            manager.disable_segment(*segment_id);
            CommandResult::Ok
        }
        Command::EnableSegment { segment_id } => {
            if segment_id.raw() as usize >= manager.segment_count() {
                return CommandResult::InvalidReference;
            }
            manager.enable_segment(*segment_id);
            CommandResult::Ok
        }
        Command::DisableLane { lane_id } => {
            if lane_id.raw() as usize >= manager.lane_count() {
                return CommandResult::InvalidReference;
            }
            manager.disable_lane(*lane_id);
            CommandResult::Ok
        }
        Command::EnableLane { lane_id } => {
            if lane_id.raw() as usize >= manager.lane_count() {
                return CommandResult::InvalidReference;
            }
            manager.enable_lane(*lane_id);
            CommandResult::Ok
        }
        Command::AttachControl { node_id, control } => {
            if node_id.raw() as usize >= manager.node_count() {
                return CommandResult::InvalidReference;
            }
            let id = manager.attach_control(*node_id, control.clone());
            CommandResult::ControlAttached(id)
        }
        Command::DisableControl {
            node_id,
            control_id,
        } => {
            if node_id.raw() as usize >= manager.node_count() {
                return CommandResult::InvalidReference;
            }
            manager.disable_control(*node_id, *control_id);
            CommandResult::Ok
        }
        Command::EnableControl {
            node_id,
            control_id,
        } => {
            if node_id.raw() as usize >= manager.node_count() {
                return CommandResult::InvalidReference;
            }
            manager.enable_control(*node_id, *control_id);
            CommandResult::Ok
        }
        Command::UpgradeSegmentBegin { old_segment } => {
            if old_segment.raw() as usize >= manager.segment_count() {
                return CommandResult::InvalidReference;
            }
            manager.disable_segment(*old_segment);
            CommandResult::Ok
        }
        Command::UpgradeSegmentEnd { .. } => {
            // Recording only; no action needed
            CommandResult::Ok
        }
    }
}

/// Applies a batch of commands in order, ensuring deterministic execution.
pub fn apply_commands(manager: &mut RoadManager, commands: &[Command]) -> Vec<CommandResult> {
    commands
        .iter()
        .map(|cmd| apply_command(manager, cmd))
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_nodes_segments_lanes() {
        let mut manager = RoadManager::new();

        // Add two nodes
        let node_a = manager.add_node(0.0, 0.0, 0.0, 1);
        let node_b = manager.add_node(100.0, 0.0, 0.0, 1);

        assert_eq!(node_a.raw(), 0);
        assert_eq!(node_b.raw(), 1);
        assert!(manager.node(node_a).is_enabled());
        assert!(manager.node(node_b).is_enabled());

        // Add segment
        let segment = manager.add_segment(
            node_a,
            node_b,
            StructureType::Surface,
            HorizontalProfile::Linear,
            VerticalProfile::Flat,
        );
        assert_eq!(segment.raw(), 0);

        // Add two opposite lanes
        let lane_ab = manager.add_lane(node_a, node_b, segment, 50.0, 10, 0xFF, 1.0);
        let lane_ba = manager.add_lane(node_b, node_a, segment, 50.0, 10, 0xFF, 1.0);

        assert_eq!(lane_ab.raw(), 0);
        assert_eq!(lane_ba.raw(), 1);

        // Check lanes are present
        assert_eq!(manager.segment(segment).lanes().len(), 2);
        assert!(manager.segment(segment).lanes().contains(&lane_ab));
        assert!(manager.segment(segment).lanes().contains(&lane_ba));

        // Check node connections
        assert!(manager.node(node_a).outgoing_lanes().contains(&lane_ab));
        assert!(manager.node(node_a).incoming_lanes().contains(&lane_ba));
        assert!(manager.node(node_b).outgoing_lanes().contains(&lane_ba));
        assert!(manager.node(node_b).incoming_lanes().contains(&lane_ab));
    }

    #[test]
    fn test_attach_traffic_signal() {
        let mut manager = RoadManager::new();
        let node = manager.add_node(0.0, 0.0, 0.0, 1);

        // Node is always an intersection
        assert!(manager.node(node).is_intersection());

        // Attach traffic signal
        let signal = TrafficSignal::new(vec![30.0, 5.0, 25.0, 5.0]);
        let control_id = manager.attach_control(node, TrafficControl::Signal(signal));

        assert_eq!(control_id.raw(), 0);
        assert!(manager.node(node).has_active_control());
        assert_eq!(manager.node(node).attached_controls().len(), 1);

        // Disable control
        manager.disable_control(node, control_id);
        assert!(!manager.node(node).has_active_control());

        // Control still exists (append-only)
        assert_eq!(manager.node(node).attached_controls().len(), 1);
        assert!(!manager.node(node).attached_controls()[0].is_enabled());
    }

    #[test]
    fn test_disable_lane() {
        let mut manager = RoadManager::new();
        let node_a = manager.add_node(0.0, 0.0, 0.0, 1);
        let node_b = manager.add_node(100.0, 0.0, 0.0, 1);
        let segment = manager.add_segment(
            node_a,
            node_b,
            StructureType::Surface,
            HorizontalProfile::Linear,
            VerticalProfile::Flat,
        );
        let lane = manager.add_lane(node_a, node_b, segment, 50.0, 10, 0xFF, 1.0);

        assert!(manager.lane(lane).is_enabled());

        manager.disable_lane(lane);
        assert!(!manager.lane(lane).is_enabled());

        // Lane still addressable
        assert_eq!(manager.lane(lane).from_node(), node_a);
        assert_eq!(manager.lane(lane).to_node(), node_b);
    }

    #[test]
    fn test_upgrade_segment() {
        let mut manager = RoadManager::new();

        // Initial setup: two nodes, one segment, one lane
        let node_a = manager.add_node(0.0, 0.0, 0.0, 1);
        let node_b = manager.add_node(100.0, 0.0, 0.0, 1);
        let old_segment = manager.add_segment(
            node_a,
            node_b,
            StructureType::Surface,
            HorizontalProfile::Linear,
            VerticalProfile::Flat,
        );
        let old_lane = manager.add_lane(node_a, node_b, old_segment, 30.0, 5, 0xFF, 1.0);

        assert_eq!(manager.segment(old_segment).version(), 0);

        // Upgrade: disable old, add new with more lanes
        let new_segments = manager.upgrade_segment(old_segment, |mgr| {
            let new_seg = mgr.add_segment(
                node_a,
                node_b,
                StructureType::Surface,
                HorizontalProfile::Linear,
                VerticalProfile::Flat,
            );
            mgr.add_lane(node_a, node_b, new_seg, 50.0, 10, 0xFF, 1.0);
            mgr.add_lane(node_b, node_a, new_seg, 50.0, 10, 0xFF, 1.0);
        });

        // Old segment and lane disabled
        assert!(!manager.segment(old_segment).is_enabled());
        assert!(!manager.lane(old_lane).is_enabled());
        assert_eq!(manager.segment(old_segment).version(), 1);

        // New segment created
        assert_eq!(new_segments.len(), 1);
        let new_segment = new_segments[0];
        assert!(manager.segment(new_segment).is_enabled());
        assert_eq!(manager.segment(new_segment).lanes().len(), 2);

        // IDs are monotonic
        assert!(new_segment.raw() > old_segment.raw());
    }

    #[test]
    fn test_monotonic_ids() {
        let mut manager = RoadManager::new();

        let n1 = manager.add_node(0.0, 0.0, 0.0, 1);
        let n2 = manager.add_node(1.0, 0.0, 0.0, 1);
        let n3 = manager.add_node(2.0, 0.0, 0.0, 1);

        assert_eq!(n1.raw(), 0);
        assert_eq!(n2.raw(), 1);
        assert_eq!(n3.raw(), 2);

        let s1 = manager.add_segment(
            n1,
            n2,
            StructureType::Surface,
            HorizontalProfile::Linear,
            VerticalProfile::Flat,
        );
        let s2 = manager.add_segment(
            n2,
            n3,
            StructureType::Surface,
            HorizontalProfile::Linear,
            VerticalProfile::Flat,
        );

        assert_eq!(s1.raw(), 0);
        assert_eq!(s2.raw(), 1);

        let l1 = manager.add_lane(n1, n2, s1, 50.0, 10, 0xFF, 1.0);
        let l2 = manager.add_lane(n2, n3, s2, 50.0, 10, 0xFF, 1.0);

        assert_eq!(l1.raw(), 0);
        assert_eq!(l2.raw(), 1);

        // Disable and add more - IDs continue to increment
        manager.disable_node(n1);
        let n4 = manager.add_node(3.0, 0.0, 0.0, 1);
        assert_eq!(n4.raw(), 3);
    }

    #[test]
    fn test_lane_length() {
        let mut manager = RoadManager::new();
        let node_a = manager.add_node(0.0, 0.0, 0.0, 1);
        let node_b = manager.add_node(3.0, 4.0, 0.0, 1);
        let segment = manager.add_segment(
            node_a,
            node_b,
            StructureType::Surface,
            HorizontalProfile::Linear,
            VerticalProfile::Flat,
        );
        let lane = manager.add_lane(node_a, node_b, segment, 50.0, 10, 0xFF, 1.0);

        let length = lane_length(manager.lane(lane), &manager);
        assert!((length - 5.0).abs() < 0.001); // 3-4-5 triangle

        let length_2d = lane_length_2d(manager.lane(lane), &manager);
        assert!((length_2d - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_lane_length_3d() {
        let mut manager = RoadManager::new();
        let node_a = manager.add_node(0.0, 0.0, 0.0, 1);
        let node_b = manager.add_node(0.0, 0.0, 10.0, 1);
        let segment = manager.add_segment(
            node_a,
            node_b,
            StructureType::Bridge,
            HorizontalProfile::Linear,
            VerticalProfile::Flat,
        );
        let lane = manager.add_lane(node_a, node_b, segment, 50.0, 10, 0xFF, 1.0);

        let length = lane_length(manager.lane(lane), &manager);
        assert!((length - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_sample_lane_height_flat() {
        let mut manager = RoadManager::new();
        let node_a = manager.add_node(0.0, 0.0, 5.0, 1);
        let node_b = manager.add_node(100.0, 0.0, 5.0, 1);
        let segment = manager.add_segment(
            node_a,
            node_b,
            StructureType::Surface,
            HorizontalProfile::Linear,
            VerticalProfile::Flat,
        );
        let lane = manager.add_lane(node_a, node_b, segment, 50.0, 10, 0xFF, 1.0);

        let z0 = sample_lane_height(manager.lane(lane), 0.0, &manager);
        let z_mid = sample_lane_height(manager.lane(lane), 0.5, &manager);
        let z1 = sample_lane_height(manager.lane(lane), 1.0, &manager);

        assert!((z0 - 5.0).abs() < 0.001);
        assert!((z_mid - 5.0).abs() < 0.001);
        assert!((z1 - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_sample_lane_height_linear() {
        let mut manager = RoadManager::new();
        let node_a = manager.add_node(0.0, 0.0, 0.0, 1);
        let node_b = manager.add_node(100.0, 0.0, 10.0, 1);
        let segment = manager.add_segment(
            node_a,
            node_b,
            StructureType::Bridge,
            HorizontalProfile::Linear,
            VerticalProfile::Linear {
                start_z: 0.0,
                end_z: 10.0,
            },
        );
        let lane = manager.add_lane(node_a, node_b, segment, 50.0, 10, 0xFF, 1.0);

        let z0 = sample_lane_height(manager.lane(lane), 0.0, &manager);
        let z_mid = sample_lane_height(manager.lane(lane), 0.5, &manager);
        let z1 = sample_lane_height(manager.lane(lane), 1.0, &manager);

        assert!((z0 - 0.0).abs() < 0.001);
        assert!((z_mid - 5.0).abs() < 0.001);
        assert!((z1 - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_sample_lane_height_linear_reversed() {
        let mut manager = RoadManager::new();
        let node_a = manager.add_node(0.0, 0.0, 0.0, 1);
        let node_b = manager.add_node(100.0, 0.0, 10.0, 1);
        let segment = manager.add_segment(
            node_a,
            node_b,
            StructureType::Bridge,
            HorizontalProfile::Linear,
            VerticalProfile::Linear {
                start_z: 0.0,
                end_z: 10.0,
            },
        );
        // Lane goes B -> A (reversed)
        let lane = manager.add_lane(node_b, node_a, segment, 50.0, 10, 0xFF, 1.0);

        let z0 = sample_lane_height(manager.lane(lane), 0.0, &manager);
        let z_mid = sample_lane_height(manager.lane(lane), 0.5, &manager);
        let z1 = sample_lane_height(manager.lane(lane), 1.0, &manager);

        // At t=0 (start of lane = node_b), should be at end_z
        assert!((z0 - 10.0).abs() < 0.001);
        assert!((z_mid - 5.0).abs() < 0.001);
        assert!((z1 - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_project_point_to_lane() {
        let mut manager = RoadManager::new();
        let node_a = manager.add_node(0.0, 0.0, 0.0, 1);
        let node_b = manager.add_node(100.0, 0.0, 0.0, 1);
        let segment = manager.add_segment(
            node_a,
            node_b,
            StructureType::Surface,
            HorizontalProfile::Linear,
            VerticalProfile::Flat,
        );
        let lane = manager.add_lane(node_a, node_b, segment, 50.0, 10, 0xFF, 1.0);

        // Point on lane
        let (t, dist_sq) = project_point_to_lane_xy(manager.lane(lane), 50.0, 0.0, &manager);
        assert!((t - 0.5).abs() < 0.001);
        assert!(dist_sq < 0.001);

        // Point off lane
        let (t, dist_sq) = project_point_to_lane_xy(manager.lane(lane), 50.0, 10.0, &manager);
        assert!((t - 0.5).abs() < 0.001);
        assert!((dist_sq - 100.0).abs() < 0.001);

        // Point before start
        let (t, dist_sq) = project_point_to_lane_xy(manager.lane(lane), -10.0, 0.0, &manager);
        assert!((t - 0.0).abs() < 0.001);
        assert!((dist_sq - 100.0).abs() < 0.001);

        // Point after end
        let (t, dist_sq) = project_point_to_lane_xy(manager.lane(lane), 110.0, 0.0, &manager);
        assert!((t - 1.0).abs() < 0.001);
        assert!((dist_sq - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_nearest_lane_to_point() {
        let mut manager = RoadManager::new();
        let node_a = manager.add_node(0.0, 0.0, 0.0, 1);
        let node_b = manager.add_node(100.0, 0.0, 0.0, 1);
        let node_c = manager.add_node(0.0, 100.0, 0.0, 1);

        let seg_ab = manager.add_segment(
            node_a,
            node_b,
            StructureType::Surface,
            HorizontalProfile::Linear,
            VerticalProfile::Flat,
        );
        let seg_ac = manager.add_segment(
            node_a,
            node_c,
            StructureType::Surface,
            HorizontalProfile::Linear,
            VerticalProfile::Flat,
        );

        let lane_ab = manager.add_lane(node_a, node_b, seg_ab, 50.0, 10, 0xFF, 1.0);
        let lane_ac = manager.add_lane(node_a, node_c, seg_ac, 50.0, 10, 0xFF, 1.0);

        // Point closer to lane_ab
        let nearest = nearest_lane_to_point(&manager, 50.0, 5.0, 0.0);
        assert_eq!(nearest, Some(lane_ab));

        // Point closer to lane_ac
        let nearest = nearest_lane_to_point(&manager, 5.0, 50.0, 0.0);
        assert_eq!(nearest, Some(lane_ac));

        // Point at origin - equidistant, should return lower ID (deterministic)
        let nearest = nearest_lane_to_point(&manager, 0.0, 0.0, 0.0);
        assert!(nearest.is_some());
    }

    #[test]
    fn test_lane_view_for_chunk() {
        let mut manager = RoadManager::new();

        // Chunk 1 nodes
        let n1a = manager.add_node(0.0, 0.0, 0.0, 1);
        let n1b = manager.add_node(10.0, 0.0, 0.0, 1);

        // Chunk 2 nodes
        let n2a = manager.add_node(100.0, 0.0, 0.0, 2);
        let n2b = manager.add_node(110.0, 0.0, 0.0, 2);

        // Chunk 1 internal segment
        let seg1 = manager.add_segment(
            n1a,
            n1b,
            StructureType::Surface,
            HorizontalProfile::Linear,
            VerticalProfile::Flat,
        );
        let lane1 = manager.add_lane(n1a, n1b, seg1, 50.0, 10, 0xFF, 1.0);

        // Chunk 2 internal segment
        let seg2 = manager.add_segment(
            n2a,
            n2b,
            StructureType::Surface,
            HorizontalProfile::Linear,
            VerticalProfile::Flat,
        );
        let lane2 = manager.add_lane(n2a, n2b, seg2, 50.0, 10, 0xFF, 1.0);

        // Cross-chunk segment (1 -> 2)
        let seg_cross = manager.add_segment(
            n1b,
            n2a,
            StructureType::Bridge,
            HorizontalProfile::Linear,
            VerticalProfile::Flat,
        );
        let lane_cross = manager.add_lane(n1b, n2a, seg_cross, 50.0, 10, 0xFF, 1.0);

        // View for chunk 1
        let view1 = manager.lane_view_for_chunk(1);
        assert!(view1.lane_ids().contains(&lane1));
        assert!(view1.lane_ids().contains(&lane_cross)); // Cross-chunk lane touches chunk 1
        assert!(!view1.lane_ids().contains(&lane2));

        // View for chunk 2
        let view2 = manager.lane_view_for_chunk(2);
        assert!(view2.lane_ids().contains(&lane2));
        assert!(view2.lane_ids().contains(&lane_cross)); // Cross-chunk lane touches chunk 2
        assert!(!view2.lane_ids().contains(&lane1));

        // View for non-existent chunk
        let view3 = manager.lane_view_for_chunk(999);
        assert!(view3.is_empty());
    }

    #[test]
    fn test_chunk_boundary_summary() {
        let mut manager = RoadManager::new();

        let n1 = manager.add_node(0.0, 0.0, 0.0, 1);
        let n2 = manager.add_node(50.0, 0.0, 0.0, 2);

        let seg = manager.add_segment(
            n1,
            n2,
            StructureType::Surface,
            HorizontalProfile::Linear,
            VerticalProfile::Flat,
        );
        let _lane_out = manager.add_lane(n1, n2, seg, 50.0, 10, 0xFF, 1.0);
        let _lane_in = manager.add_lane(n2, n1, seg, 50.0, 10, 0xFF, 1.0);

        let summary1 = ChunkBoundarySummary::compute(&manager, 1);
        assert_eq!(summary1.outgoing_cross_chunk_lanes.len(), 1);
        assert_eq!(summary1.incoming_cross_chunk_lanes.len(), 1);

        let summary2 = ChunkBoundarySummary::compute(&manager, 2);
        assert_eq!(summary2.outgoing_cross_chunk_lanes.len(), 1);
        assert_eq!(summary2.incoming_cross_chunk_lanes.len(), 1);
    }

    #[test]
    fn test_road_chunk_state() {
        let mut manager = RoadManager::new();
        let node_a = manager.add_node(0.0, 0.0, 0.0, 1);
        let node_b = manager.add_node(100.0, 0.0, 0.0, 1);
        let segment = manager.add_segment(
            node_a,
            node_b,
            StructureType::Surface,
            HorizontalProfile::Linear,
            VerticalProfile::Flat,
        );
        let lane = manager.add_lane(node_a, node_b, segment, 50.0, 10, 0xFF, 1.0);

        let mut chunk_state = RoadChunkState::new(1, &[lane], 12345);

        assert_eq!(chunk_state.chunk_id(), 1);
        assert_eq!(chunk_state.prng_seed, 12345);
        assert!((chunk_state.deferred_tick_seconds - 2.0).abs() < 0.001);

        // Modify lane state
        if let Some(state) = chunk_state.lane_state_mut(lane) {
            state.occupancy_estimate = 0.5;
            state.dynamic_cost_modifier = 10.0;
        }

        assert!((chunk_state.lane_state(lane).unwrap().occupancy_estimate - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_command_application() {
        let mut manager = RoadManager::new();

        // Add nodes via commands
        let result = apply_command(
            &mut manager,
            &Command::AddNode {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                chunk_id: 1,
            },
        );
        let node_a = match result {
            CommandResult::NodeCreated(id) => id,
            _ => panic!("Expected NodeCreated"),
        };

        let result = apply_command(
            &mut manager,
            &Command::AddNode {
                x: 100.0,
                y: 0.0,
                z: 0.0,
                chunk_id: 1,
            },
        );
        let node_b = match result {
            CommandResult::NodeCreated(id) => id,
            _ => panic!("Expected NodeCreated"),
        };

        // Add segment
        let result = apply_command(
            &mut manager,
            &Command::AddSegment {
                start: node_a,
                end: node_b,
                structure: StructureType::Surface,
                horizontal_profile: HorizontalProfile::Linear,
                vertical_profile: VerticalProfile::Flat,
            },
        );
        let segment = match result {
            CommandResult::SegmentCreated(id) => id,
            _ => panic!("Expected SegmentCreated"),
        };

        // Add lane
        let result = apply_command(
            &mut manager,
            &Command::AddLane {
                from: node_a,
                to: node_b,
                segment,
                speed_limit: 50.0,
                capacity: 10,
                vehicle_mask: 0xFF,
                base_cost: 1.0,
            },
        );
        let lane = match result {
            CommandResult::LaneCreated(id) => id,
            _ => panic!("Expected LaneCreated"),
        };

        assert!(manager.lane(lane).is_enabled());

        // Disable via command
        apply_command(&mut manager, &Command::DisableLane { lane_id: lane });
        assert!(!manager.lane(lane).is_enabled());

        // Invalid reference
        let result = apply_command(
            &mut manager,
            &Command::DisableNode {
                node_id: NodeId::new(999),
            },
        );
        assert!(matches!(result, CommandResult::InvalidReference));
    }

    #[test]
    fn test_deterministic_iteration() {
        let mut manager1 = RoadManager::new();
        let mut manager2 = RoadManager::new();

        // Build identical topologies
        for i in 0..10 {
            manager1.add_node(i as f32, 0.0, 0.0, 1);
            manager2.add_node(i as f32, 0.0, 0.0, 1);
        }

        // Iteration order must be identical
        let ids1: Vec<_> = manager1.iter_nodes().map(|(id, _)| id).collect();
        let ids2: Vec<_> = manager2.iter_nodes().map(|(id, _)| id).collect();

        assert_eq!(ids1, ids2);

        // Values must be identical
        for (id, node) in manager1.iter_nodes() {
            let node2 = manager2.node(id);
            assert_eq!(node.x(), node2.x());
        }
    }

    #[test]
    fn test_sample_lane_position() {
        let mut manager = RoadManager::new();
        let node_a = manager.add_node(0.0, 0.0, 0.0, 1);
        let node_b = manager.add_node(100.0, 100.0, 50.0, 1);
        let segment = manager.add_segment(
            node_a,
            node_b,
            StructureType::Bridge,
            HorizontalProfile::Linear,
            VerticalProfile::Linear {
                start_z: 0.0,
                end_z: 50.0,
            },
        );
        let lane = manager.add_lane(node_a, node_b, segment, 50.0, 10, 0xFF, 1.0);

        let (x, y, z) = sample_lane_position(manager.lane(lane), 0.0, &manager);
        assert!((x - 0.0).abs() < 0.001);
        assert!((y - 0.0).abs() < 0.001);
        assert!((z - 0.0).abs() < 0.001);

        let (x, y, z) = sample_lane_position(manager.lane(lane), 0.5, &manager);
        assert!((x - 50.0).abs() < 0.001);
        assert!((y - 50.0).abs() < 0.001);
        assert!((z - 25.0).abs() < 0.001);

        let (x, y, z) = sample_lane_position(manager.lane(lane), 1.0, &manager);
        assert!((x - 100.0).abs() < 0.001);
        assert!((y - 100.0).abs() < 0.001);
        assert!((z - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_traffic_signal() {
        let signal = TrafficSignal::new(vec![30.0, 5.0, 25.0, 5.0]);
        assert!((signal.total_cycle_duration() - 65.0).abs() < 0.001);
        assert_eq!(signal.current_phase, 0);
    }

    #[test]
    fn test_segment_lane_counts() {
        let mut manager = RoadManager::new();
        let node_a = manager.add_node(0.0, 0.0, 0.0, 1);
        let node_b = manager.add_node(100.0, 0.0, 0.0, 1);
        let segment = manager.add_segment(
            node_a,
            node_b,
            StructureType::Surface,
            HorizontalProfile::Linear,
            VerticalProfile::Flat,
        );

        // Add 2 forward, 1 backward
        manager.add_lane(node_a, node_b, segment, 50.0, 10, 0xFF, 1.0);
        manager.add_lane(node_a, node_b, segment, 50.0, 10, 0xFF, 1.0);
        manager.add_lane(node_b, node_a, segment, 50.0, 10, 0xFF, 1.0);

        let (forward, backward) = manager.segment(segment).lane_counts(&manager);
        assert_eq!(forward, 2);
        assert_eq!(backward, 1);
    }

    #[test]
    fn test_k_nearest_lanes() {
        let mut manager = RoadManager::new();

        let n1 = manager.add_node(0.0, 0.0, 0.0, 1);
        let n2 = manager.add_node(10.0, 0.0, 0.0, 1);
        let n3 = manager.add_node(20.0, 0.0, 0.0, 1);
        let n4 = manager.add_node(30.0, 0.0, 0.0, 1);

        let s1 = manager.add_segment(
            n1,
            n2,
            StructureType::Surface,
            HorizontalProfile::Linear,
            VerticalProfile::Flat,
        );
        let s2 = manager.add_segment(
            n2,
            n3,
            StructureType::Surface,
            HorizontalProfile::Linear,
            VerticalProfile::Flat,
        );
        let s3 = manager.add_segment(
            n3,
            n4,
            StructureType::Surface,
            HorizontalProfile::Linear,
            VerticalProfile::Flat,
        );

        let l1 = manager.add_lane(n1, n2, s1, 50.0, 10, 0xFF, 1.0);
        let l2 = manager.add_lane(n2, n3, s2, 50.0, 10, 0xFF, 1.0);
        let l3 = manager.add_lane(n3, n4, s3, 50.0, 10, 0xFF, 1.0);

        // Query point at x=5, closest to l1
        let nearest = k_nearest_lanes_to_point(&manager, 5.0, 0.0, 2);
        assert_eq!(nearest.len(), 2);
        assert_eq!(nearest[0].0, l1);
        assert_eq!(nearest[1].0, l2);

        // Query with k > lanes
        let all = k_nearest_lanes_to_point(&manager, 15.0, 0.0, 10);
        assert_eq!(all.len(), 3);

        // Verify deterministic ordering for ties
        let _ordered: Vec<_> = all.iter().map(|(id, _)| id.raw()).collect();
    }

    #[test]
    fn test_vehicle_mask() {
        let mut manager = RoadManager::new();
        let node_a = manager.add_node(0.0, 0.0, 0.0, 1);
        let node_b = manager.add_node(100.0, 0.0, 0.0, 1);
        let segment = manager.add_segment(
            node_a,
            node_b,
            StructureType::Surface,
            HorizontalProfile::Linear,
            VerticalProfile::Flat,
        );

        // Lane allows only types 0x01 and 0x02
        let lane = manager.add_lane(node_a, node_b, segment, 50.0, 10, 0x03, 1.0);

        assert!(manager.lane(lane).allows_vehicle(0x01));
        assert!(manager.lane(lane).allows_vehicle(0x02));
        assert!(manager.lane(lane).allows_vehicle(0x03));
        assert!(!manager.lane(lane).allows_vehicle(0x04));
        assert!(!manager.lane(lane).allows_vehicle(0x08));
    }
}
