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
use crate::terrain::roads::road_editor::RoadEditorCommand;
use crate::terrain::roads::road_mesh_manager::{
    ChunkId, HorizontalProfile, RoadMeshManager, chunk_x_range,
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
    Flat,
    EndPoints { start_y: f32, end_y: f32 },
}

impl VerticalProfile {
    #[inline]
    pub fn slope(&self) -> f32 {
        match self {
            VerticalProfile::Flat => 0.0,
            VerticalProfile::EndPoints { start_y, end_y } => end_y - start_y,
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
    pub vertical_profile: VerticalProfile,
    pub version: u32,
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
    pub fn polyline(&self) -> &Vec<Vec3> {
        &self.geometry.points
    }
}

#[derive(Clone, Debug)]
pub struct LaneGeometry {
    pub points: Vec<Vec3>, // polyline
    pub lengths: Vec<f32>, // cumulative arc length
    pub total_len: f32,
}

impl LaneGeometry {
    pub fn from_segment(
        terrain_renderer: &TerrainRenderer,
        road_manager: &RoadManager,
        segment: &Segment,
        lane_index: i8,
        lane_width: f32,
    ) -> LaneGeometry {
        let initial_offset = lane_width * 0.5;
        let (og_start_pos, og_end_pos) = if lane_index > 0 {
            (
                Vec3::from_array(road_manager.node(segment.start).unwrap().position()),
                Vec3::from_array(road_manager.node(segment.end).unwrap().position()),
            )
        } else {
            (
                Vec3::from_array(road_manager.node(segment.end).unwrap().position()), // flipped, because left side!
                Vec3::from_array(road_manager.node(segment.start).unwrap().position()),
            ) // flipped
        };

        let estimated_len = estimate_segment_length(segment, og_start_pos, og_end_pos);

        let samples =
            ((estimated_len / METERS_PER_LANE_POLYLINE_STEP).ceil() as usize).clamp(2, 512);

        let mut points = Vec::with_capacity(samples + 1);
        let mut lengths = Vec::with_capacity(samples + 1);

        // Direction in XZ
        let dir = Vec3::new(
            og_end_pos.x - og_start_pos.x,
            0.0,
            og_end_pos.z - og_start_pos.z,
        );

        let len_xz = (dir.x * dir.x + dir.z * dir.z).sqrt().max(0.0001);
        let dir_xz = Vec3::new(dir.x / len_xz, 0.0, dir.z / len_xz);

        let right = Vec3::new(dir_xz.z, 0.0, -dir_xz.x);

        let lateral = right * (lane_index.abs() as f32 * lane_width - initial_offset);
        println!("{} {} {}", lateral, lane_index, lane_width);
        let mut total_len = 0.0;
        lengths.push(0.0);

        for i in 0..=samples {
            let t = i as f32 / samples as f32;
            let start_pos = og_start_pos - lateral;
            let end_pos = og_end_pos - lateral;
            let mut p: Vec3 = match segment.horizontal_profile {
                HorizontalProfile::Linear => start_pos.lerp(end_pos, t),

                HorizontalProfile::QuadraticBezier { control } => {
                    let c = Vec3::new(control[0], start_pos.y, control[1]);
                    bezier2(start_pos, c, end_pos, t)
                }

                HorizontalProfile::CubicBezier { control1, control2 } => {
                    let c1 = Vec3::new(control1[0], start_pos.y, control1[1]);
                    let c2 = Vec3::new(control2[0], start_pos.y, control2[1]);
                    bezier3(start_pos, c1, c2, end_pos, t)
                }

                HorizontalProfile::Arc { .. } => start_pos.lerp(end_pos, t),
            };

            p.y = p.y.max(terrain_renderer.get_height_at([p.x, p.z]));

            if let Some(&prev) = points.last() {
                total_len += p.distance(prev);
                lengths.push(total_len);
            }

            points.push(p);
        }

        LaneGeometry {
            points,
            lengths,
            total_len,
        }
    }
}

fn estimate_segment_length(segment: &Segment, start: Vec3, end: Vec3) -> f32 {
    const ESTIMATE_SAMPLES: usize = 8;

    let mut len = 0.0;
    let mut prev = start;

    for i in 1..=ESTIMATE_SAMPLES {
        let t = i as f32 / ESTIMATE_SAMPLES as f32;

        let p = match segment.horizontal_profile {
            HorizontalProfile::Linear => start.lerp(end, t),

            HorizontalProfile::QuadraticBezier { control } => {
                let c = Vec3::new(control[0], start.y, control[1]);
                bezier2(start, c, end, t)
            }

            HorizontalProfile::CubicBezier { control1, control2 } => {
                let c1 = Vec3::new(control1[0], start.y, control1[1]);
                let c2 = Vec3::new(control2[0], start.y, control2[1]);
                bezier3(start, c1, c2, end, t)
            }

            HorizontalProfile::Arc { .. } => start.lerp(end, t),
        };

        len += (p - prev).length();
        prev = p;
    }

    len
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
    pub nodes: Vec<Node>,
    pub segments: Vec<Segment>,
    pub lanes: Vec<Lane>,
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
        terrain_renderer: &TerrainRenderer,
        from: NodeId,
        to: NodeId,
        segment: SegmentId,
        lane_index: i8, // signed, relative to segment centerline, 0 is prohibited!
        lane_width: f32,
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
        let geometry =
            LaneGeometry::from_segment(terrain_renderer, &self, seg, lane_index, lane_width);
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
            let Some(from_chunk_node) = self.node(lane.from_node()) else {
                continue;
            };
            let Some(to_chunk_node) = self.node(lane.to_node()) else {
                continue;
            };
            let from_chunk = from_chunk_node.chunk_id();
            let to_chunk = to_chunk_node.chunk_id();
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
    pub fn node(&self, id: NodeId) -> Option<&Node> {
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
    let Some(from) = manager.node(lane.from_node()) else {
        return 0.0;
    };
    let Some(to) = manager.node(lane.to_node()) else {
        return 0.0;
    };
    let dx = to.x() - from.x();
    let dy = to.y() - from.y();
    let dz = to.z() - from.z();
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Computes the 2D (XY) length of a lane.
#[inline]
pub fn lane_length_2d(lane: &Lane, manager: &RoadManager) -> f32 {
    let Some(from) = manager.node(lane.from_node()) else {
        return 0.0;
    };
    let Some(to) = manager.node(lane.to_node()) else {
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
pub fn sample_lane_position(lane: &Lane, t: f32, manager: &RoadManager) -> (f32, f32) {
    let Some(from) = manager.node(lane.from_node()) else {
        return (0.0, 0.0);
    };
    let Some(to) = manager.node(lane.to_node()) else {
        return (0.0, 0.0);
    };

    let x = from.x() + (to.x() - from.x()) * t;
    let z = from.z() + (to.z() - from.z()) * t;

    (x, z)
}

/// Projects a 2D point onto a lane and returns (t, distance_squared).
/// t is the parameter [0,1] along the lane; dist_sq is squared XY distance.
#[inline]
pub fn project_point_to_lane_xz(lane: &Lane, x: f32, z: f32, manager: &RoadManager) -> (f32, f32) {
    let Some(from) = manager.node(lane.from_node()) else {
        return (0.0, 0.0);
    };
    let Some(to) = manager.node(lane.to_node()) else {
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
pub fn nearest_lane_to_point(manager: &RoadManager, x: f32, _y: f32, z: f32) -> Option<LaneId> {
    let mut best_id: Option<LaneId> = None;
    let mut best_dist_sq = f32::MAX;

    for (id, lane) in manager.iter_lanes() {
        if !lane.is_enabled() {
            continue;
        }

        let (_, dist_sq) = project_point_to_lane_xz(lane, x, z, manager);
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
    z: f32,
    k: usize,
) -> Vec<(LaneId, f32)> {
    let mut candidates: Vec<(LaneId, f32)> = Vec::with_capacity(manager.lane_count());

    for (id, lane) in manager.iter_lanes() {
        if !lane.is_enabled() {
            continue;
        }
        let (_, dist_sq) = project_point_to_lane_xz(lane, x, z, manager);
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
            let Some(from) = manager.node(lane.from_node()) else {
                continue;
            };
            let Some(to) = manager.node(lane.to_node()) else {
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
        horizontal_profile: HorizontalProfile,
        vertical_profile: VerticalProfile,
        chunk_id: ChunkId,
    },
    /// Add a new lane to a segment.
    AddLane {
        from: NodeId,
        to: NodeId,
        segment: SegmentId,
        lane_index: i8,
        lane_width: f32,
        speed_limit: f32,
        capacity: u32,
        vehicle_mask: u32,
        base_cost: f32,
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
pub fn apply_command(
    terrain_renderer: &TerrainRenderer,
    road_mesh_manager: &mut RoadMeshManager,
    manager: &mut RoadManager,
    command: &RoadEditorCommand,
) -> CommandResult {
    match command {
        RoadEditorCommand::Road(road_command) => {
            match road_command {
                RoadCommand::AddNode { x, y, z, chunk_id } => {
                    let id = manager.add_node(*x, *y, *z, *chunk_id);
                    road_mesh_manager.update_chunk_mesh(terrain_renderer, *chunk_id, manager);
                    CommandResult::NodeCreated(*chunk_id, id)
                }
                RoadCommand::AddSegment {
                    start,
                    end,
                    structure,
                    horizontal_profile,
                    vertical_profile,
                    chunk_id,
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
                    road_mesh_manager.update_chunk_mesh(terrain_renderer, *chunk_id, manager);
                    CommandResult::SegmentCreated(*chunk_id, id)
                }
                RoadCommand::AddLane {
                    from,
                    to,
                    segment,
                    lane_index,
                    lane_width,
                    speed_limit,
                    capacity,
                    vehicle_mask,
                    base_cost,
                    chunk_id,
                } => {
                    if from.raw() as usize >= manager.node_count()
                        || to.raw() as usize >= manager.node_count()
                        || segment.raw() as usize >= manager.segment_count()
                    {
                        return CommandResult::InvalidReference;
                    }
                    let id = manager.add_lane(
                        terrain_renderer,
                        *from,
                        *to,
                        *segment,
                        *lane_index,
                        *lane_width,
                        *speed_limit,
                        *capacity,
                        *vehicle_mask,
                        *base_cost,
                    );
                    road_mesh_manager.update_chunk_mesh(terrain_renderer, *chunk_id, manager);
                    CommandResult::LaneCreated(*chunk_id, id)
                }
                RoadCommand::DisableNode { node_id, chunk_id } => {
                    if node_id.raw() as usize >= manager.node_count() {
                        return CommandResult::InvalidReference;
                    }
                    manager.disable_node(*node_id);
                    road_mesh_manager.update_chunk_mesh(terrain_renderer, *chunk_id, manager);
                    CommandResult::Ok
                }
                RoadCommand::EnableNode { node_id, chunk_id } => {
                    if node_id.raw() as usize >= manager.node_count() {
                        return CommandResult::InvalidReference;
                    }
                    manager.enable_node(*node_id);
                    road_mesh_manager.update_chunk_mesh(
                        terrain_renderer,
                        *chunk_id,
                        &Default::default(),
                    );
                    CommandResult::Ok
                }
                RoadCommand::DisableSegment {
                    segment_id,
                    chunk_id,
                } => {
                    if segment_id.raw() as usize >= manager.segment_count() {
                        return CommandResult::InvalidReference;
                    }
                    manager.disable_segment(*segment_id);
                    road_mesh_manager.update_chunk_mesh(terrain_renderer, *chunk_id, manager);
                    CommandResult::Ok
                }
                RoadCommand::EnableSegment {
                    segment_id,
                    chunk_id,
                } => {
                    if segment_id.raw() as usize >= manager.segment_count() {
                        return CommandResult::InvalidReference;
                    }
                    manager.enable_segment(*segment_id);
                    road_mesh_manager.update_chunk_mesh(terrain_renderer, *chunk_id, manager);
                    CommandResult::Ok
                }
                RoadCommand::DisableLane { lane_id, chunk_id } => {
                    if lane_id.raw() as usize >= manager.lane_count() {
                        return CommandResult::InvalidReference;
                    }
                    manager.disable_lane(*lane_id);
                    road_mesh_manager.update_chunk_mesh(terrain_renderer, *chunk_id, manager);
                    CommandResult::Ok
                }
                RoadCommand::EnableLane { lane_id, chunk_id } => {
                    if lane_id.raw() as usize >= manager.lane_count() {
                        return CommandResult::InvalidReference;
                    }
                    manager.enable_lane(*lane_id);
                    road_mesh_manager.update_chunk_mesh(terrain_renderer, *chunk_id, manager);
                    CommandResult::Ok
                }
                RoadCommand::AttachControl {
                    node_id,
                    chunk_id,
                    control,
                } => {
                    if node_id.raw() as usize >= manager.node_count() {
                        return CommandResult::InvalidReference;
                    }
                    let id = manager.attach_control(*node_id, control.clone());
                    road_mesh_manager.update_chunk_mesh(terrain_renderer, *chunk_id, manager);
                    CommandResult::ControlAttached(*chunk_id, id)
                }
                RoadCommand::DisableControl {
                    node_id,
                    control_id,
                    chunk_id,
                } => {
                    if node_id.raw() as usize >= manager.node_count() {
                        return CommandResult::InvalidReference;
                    }
                    manager.disable_control(*node_id, *control_id);
                    road_mesh_manager.update_chunk_mesh(terrain_renderer, *chunk_id, manager);
                    CommandResult::Ok
                }
                RoadCommand::EnableControl {
                    node_id,
                    control_id,
                    chunk_id,
                } => {
                    if node_id.raw() as usize >= manager.node_count() {
                        return CommandResult::InvalidReference;
                    }
                    manager.enable_control(*node_id, *control_id);
                    road_mesh_manager.update_chunk_mesh(terrain_renderer, *chunk_id, manager);
                    CommandResult::Ok
                }
                RoadCommand::UpgradeSegmentBegin {
                    old_segment,
                    chunk_id,
                } => {
                    if old_segment.raw() as usize >= manager.segment_count() {
                        return CommandResult::InvalidReference;
                    }
                    manager.disable_segment(*old_segment);
                    road_mesh_manager.update_chunk_mesh(terrain_renderer, *chunk_id, manager);
                    CommandResult::Ok
                }
                RoadCommand::UpgradeSegmentEnd { chunk_id, .. } => {
                    // Recording only; no action needed... maybe...
                    road_mesh_manager.update_chunk_mesh(terrain_renderer, *chunk_id, manager);
                    CommandResult::Ok
                }
            }
        }
        RoadEditorCommand::PreviewClear => CommandResult::Ok,
        RoadEditorCommand::PreviewSnap(_) => CommandResult::Ok,
        RoadEditorCommand::PreviewNode(_) => CommandResult::Ok,
        RoadEditorCommand::PreviewLane(_) => CommandResult::Ok,
        RoadEditorCommand::PreviewSegment(_) => CommandResult::Ok,
        RoadEditorCommand::PreviewError(_) => CommandResult::Ok,
    }
}

/// Applies a batch of commands in order, ensuring deterministic execution.
pub fn apply_commands(
    terrain_renderer: &TerrainRenderer,
    road_mesh_manager: &mut RoadMeshManager,
    road_manager: &mut RoadManager,
    commands: &Vec<RoadEditorCommand>,
) -> Vec<CommandResult> {
    let results = commands
        .iter()
        .map(|cmd| apply_command(terrain_renderer, road_mesh_manager, road_manager, cmd))
        .collect();
    results
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
