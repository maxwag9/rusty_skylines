#![allow(dead_code)]

use crate::helpers::positions::WorldPos;
use crate::terrain::roads::roads::{RoadCommand, RoadStorage};
use glam::Vec3;

#[derive(Clone, Debug)]
pub struct RoadStyleParams {
    state: EditorState,
    mode: BuildMode,
    road_type: RoadType,
}

impl RoadStyleParams {
    pub fn turn_tightness(&self) -> f32 {
        self.road_type.turn_tightness
    }
    pub fn _set_mode(&mut self, mode: BuildMode) {
        if self.mode != mode {
            self.mode = mode;
            self.state = EditorState::Idle;
        }
    }
    pub fn set_road_type(&mut self, ty: RoadType) {
        self.road_type = ty;
    }
    pub fn set_state(&mut self, editor_state: EditorState) {
        self.state = editor_state;
    }
    pub fn mode(&self) -> BuildMode {
        self.mode
    }
    pub fn road_type(&self) -> &RoadType {
        &self.road_type
    }
    pub fn state(&self) -> &EditorState {
        &self.state
    }
    pub fn is_idle(&self) -> bool {
        matches!(self.state, EditorState::Idle)
    }
    pub fn _cancel(&mut self) {
        self.set_to_idle();
    }
    pub fn set_to_idle(&mut self) {
        self.state = EditorState::Idle;
    }
}

impl Default for RoadStyleParams {
    fn default() -> Self {
        Self {
            state: EditorState::Idle,
            mode: BuildMode::Straight,
            road_type: RoadType::default(),
        }
    }
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
pub struct LaneId(pub(crate) u32);

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
    pub(crate) id: ControlId,
    pub(crate) control: TrafficControl,
    pub(crate) enabled: bool,
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

pub type LeftLaneCount = usize;
pub type RightLaneCount = usize;
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoadType {
    pub name: &'static str,

    pub lanes_each_direction: (LeftLaneCount, RightLaneCount),

    pub lane_width: f32,
    pub lane_height: f32,
    pub lane_material_id: u32,

    pub sidewalk_width: f32,
    pub sidewalk_height: f32,
    pub sidewalk_material_id: u32,

    pub median_width: f32,
    pub median_height: f32,
    pub median_material_id: u32,

    pub speed_limit: f32,
    pub vehicle_mask: u32,
    pub structure: StructureType,
    pub turn_tightness: f32,
}
impl Default for RoadType {
    fn default() -> Self {
        Self {
            name: "Small Road",

            lanes_each_direction: (1, 1),

            lane_width: 3.1,
            lane_height: 0.0,
            lane_material_id: 2, // asphalt

            sidewalk_width: 1.40,
            sidewalk_height: 0.15,
            sidewalk_material_id: 0, // concrete

            median_width: 0.0,
            median_height: 0.15,
            median_material_id: 0, // concrete

            speed_limit: 16.7,
            vehicle_mask: 1,
            structure: StructureType::Surface,
            turn_tightness: 1.0,
        }
    }
}

impl RoadType {
    pub fn total_lanes(&self) -> usize {
        self.lanes_each_direction.0 + self.lanes_each_direction.1
    }

    pub fn capacity(&self) -> u32 {
        let lanes = self.total_lanes() as f32;
        let base_per_lane = 900.0;
        let speed_factor = (self.speed_limit / 13.9).clamp(0.5, 3.0);

        (lanes * base_per_lane * speed_factor) as u32
    }
    pub fn speed_limit(&self) -> f32 {
        self.speed_limit
    }

    pub fn vehicle_mask(&self) -> u32 {
        self.vehicle_mask
    }

    pub fn structure(&self) -> StructureType {
        self.structure
    }

    pub fn lanes_each_direction(&self) -> (LeftLaneCount, RightLaneCount) {
        self.lanes_each_direction
    }

    pub fn name(&self) -> &'static str {
        self.name
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildMode {
    Straight,
    Curved,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SnapKind {
    Free,
    Node { id: NodeId },
    Lane { lane_id: LaneId, t: f32 },
}

#[derive(Debug, Clone, Copy)]
pub struct SnapResult {
    pub world_pos: WorldPos,
    pub kind: SnapKind,
    pub distance: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodePreviewResult {
    NewNode,
    MergedIntoExisting(NodeId),
    SplitIntersection,
}

#[derive(Debug, Clone)]
pub struct SnapPreview {
    pub world_pos: WorldPos,
    pub kind: SnapKind,
    pub distance: f32,
}

#[derive(Debug, Clone)]
pub struct NodePreview {
    pub world_pos: WorldPos,
    pub result: NodePreviewResult,
    pub is_valid: bool,
    pub incoming_lanes: Vec<(LaneId, Vec3)>,
    pub outgoing_lanes: Vec<(LaneId, Vec3)>,
}

impl NodePreview {
    pub(crate) fn lane_counts(&self) -> (usize, usize) {
        (self.incoming_lanes.len(), self.outgoing_lanes.len())
    }
    pub(crate) fn lane_count(&self) -> usize {
        self.incoming_lanes.len() + self.outgoing_lanes.len()
    }
}

#[derive(Debug, Clone)]
pub struct LanePreview {
    pub lane_id: LaneId,
    pub projected_t: f32,
    pub projected_point: WorldPos,
    pub sample_points: Vec<WorldPos>,
}

#[derive(Debug, Clone)]
pub struct SplitInfo {
    pub lane_id: LaneId,
    pub t: f32,
    pub split_pos: WorldPos,
}

#[derive(Debug, Clone)]
pub struct SegmentPreview {
    pub road_type: RoadType,
    pub mode: BuildMode,
    pub is_valid: bool,
    pub reason_invalid: Option<PreviewError>,
    pub start: WorldPos,
    pub end: WorldPos,
    pub control: Option<WorldPos>,
    pub polyline: Vec<WorldPos>,
    pub would_split_start: Option<SplitInfo>,
    pub would_split_end: Option<SplitInfo>,
    pub would_merge_start: Option<NodeId>,
    pub would_merge_end: Option<NodeId>,
    pub lane_count_each_dir: (LeftLaneCount, RightLaneCount),
    pub estimated_length: f32,
    pub crossing_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PreviewError {
    NoPickedPoint,
    TooShort,
    SameNode,
    MissingNodeData,
    MissingSegmentData,
    MissingLaneData,
    InvalidSnap,
}

#[derive(Debug, Clone)]
pub enum RoadEditorCommand {
    Road(RoadCommand),
    PreviewClear,
    PreviewSnap(SnapPreview),
    PreviewNode(NodePreview),
    PreviewLane(LanePreview),
    PreviewSegment(SegmentPreview),
    PreviewError(PreviewError),
    PreviewCrossing(CrossingPoint),
}

#[derive(Debug, Clone)]
pub enum PlannedNode {
    Existing(NodeId),
    New {
        pos: WorldPos,
    },
    Split {
        lane_id: LaneId,
        t: f32,
        pos: WorldPos,
    },
}

impl PlannedNode {
    pub fn position(&self, storage: &RoadStorage) -> Option<WorldPos> {
        match self {
            PlannedNode::Existing(id) => {
                let node = storage.node(*id)?;
                Some(node.position())
            }
            PlannedNode::New { pos } => Some(*pos),
            PlannedNode::Split { pos, .. } => Some(*pos),
        }
    }

    pub(crate) fn merged_node_id(&self) -> Option<NodeId> {
        match self {
            PlannedNode::Existing(id) => Some(*id),
            _ => None,
        }
    }

    pub(crate) fn split_info(&self) -> Option<SplitInfo> {
        match self {
            PlannedNode::Split { lane_id, t, pos } => Some(SplitInfo {
                lane_id: *lane_id,
                t: *t,
                split_pos: *pos,
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Anchor {
    pub snap: SnapResult,
    pub planned_node: PlannedNode,
}

#[derive(Debug, Clone)]
pub enum EditorState {
    Idle,
    StraightPickEnd { start: Anchor },
    CurvePickControl { start: Anchor },
    CurvePickEnd { start: Anchor, control: WorldPos },
}

pub struct IdAllocator {
    next_node: u32,
    next_segment: u32,
    next_lane: u32,
}

impl IdAllocator {
    pub fn new() -> Self {
        Self {
            next_node: 0,
            next_segment: 0,
            next_lane: 0,
        }
    }
    pub fn update(&mut self, real_roads_storage: &RoadStorage) {
        self.next_node = real_roads_storage.nodes.len() as u32;
        self.next_segment = real_roads_storage.segments.len() as u32;
        self.next_lane = real_roads_storage.lanes.len() as u32;
    }
    pub fn alloc_node(&mut self) -> NodeId {
        let id = NodeId::new(self.next_node);
        self.next_node += 1;
        id
    }

    pub fn alloc_segment(&mut self) -> SegmentId {
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

#[derive(Debug, Clone)]
pub struct CrossingPoint {
    /// Position along the new road from 0.0 (start) to 1.0 (end)
    pub(crate) t: f32,
    /// World position of the crossing
    pub(crate) world_pos: WorldPos,
    /// What we're crossing
    pub(crate) kind: CrossingKind,
}

#[derive(Debug, Clone)]
pub enum CrossingKind {
    /// Crossing through an existing node
    ExistingNode(NodeId),
    /// Crossing through an existing lane segment
    LaneCrossing { lane_id: LaneId, lane_t: f32 },
}

#[derive(Clone)]
pub struct ResolvedWaypoint {
    pub(crate) node_id: NodeId,
    pub(crate) pos: WorldPos,
    pub(crate) t: f32,
}
#[derive(Clone, Debug)]
pub struct IntersectionArm {
    pub(crate) segment_id: SegmentId,
    pub(crate) points_to_node: bool,
    pub(crate) angle: f32,
    pub(crate) direction: Vec3,
    pub(crate) half_width: f32,
    pub(crate) lane_ids: Vec<LaneId>,
}
