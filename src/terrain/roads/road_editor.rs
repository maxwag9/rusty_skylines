use crate::renderer::world_renderer::{PickedPoint, TerrainRenderer};
use crate::resources::InputState;
use crate::terrain::roads::road_mesh_manager::{CLEARANCE, ChunkId};
use crate::terrain::roads::roads::{
    Lane, LaneGeometry, LaneId, NodeId, RoadCommand, RoadManager, RoadStorage, SegmentId,
    StructureType, nearest_lane_to_point, project_point_to_lane_xz, sample_lane_position,
};
use glam::Vec3;

const NODE_SNAP_RADIUS: f32 = 8.0;
const LANE_SNAP_RADIUS: f32 = 8.0;
const ENDPOINT_T_EPS: f32 = 0.02;
const MIN_SEGMENT_LENGTH: f32 = 1.0;
const DEFAULT_LANE_WIDTH: f32 = 3.5;

// ============================================================================
// Types & Structs
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoadType {
    SmallRoad,
    MediumRoad,
    Highway,
}

impl RoadType {
    pub fn speed_limit(self) -> f32 {
        match self {
            RoadType::SmallRoad => 13.9,
            RoadType::MediumRoad => 16.7,
            RoadType::Highway => 33.3,
        }
    }

    pub fn capacity(self) -> u32 {
        match self {
            RoadType::SmallRoad => 2000,
            RoadType::MediumRoad => 3000,
            RoadType::Highway => 4000,
        }
    }

    pub fn vehicle_mask(self) -> u32 {
        1
    }

    pub fn structure(self) -> StructureType {
        StructureType::Surface
    }

    pub fn lanes_each_direction(self) -> (usize, usize) {
        match self {
            RoadType::SmallRoad => (1, 1),
            RoadType::MediumRoad => (2, 2),
            RoadType::Highway => (3, 3),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            RoadType::SmallRoad => "Small Road",
            RoadType::MediumRoad => "Medium Road",
            RoadType::Highway => "Highway",
        }
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
    pub world_pos: Vec3,
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
    pub world_pos: Vec3,
    pub kind: SnapKind,
    pub distance: f32,
}

/// Simplified connection info for previews, replacing the old MeshManager version
#[derive(Debug, Clone)]
pub struct ConnectedSegmentInfo {
    pub direction_xz: [f32; 2],
    pub node_is_start: bool,
    pub segment_id: Option<SegmentId>,
}

#[derive(Debug, Clone)]
pub struct NodePreview {
    pub world_pos: Vec3,
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
    pub projected_point: Vec3,
    pub sample_points: Vec<Vec3>,
}

#[derive(Debug, Clone)]
pub struct SplitInfo {
    pub lane_id: LaneId,
    pub t: f32,
    pub split_pos: Vec3,
}

#[derive(Debug, Clone)]
pub struct SegmentPreview {
    pub road_type: RoadType,
    pub mode: BuildMode,
    pub is_valid: bool,
    pub reason_invalid: Option<PreviewError>,
    pub start: Vec3,
    pub end: Vec3,
    pub control: Option<Vec3>,
    pub polyline: Vec<Vec3>,
    pub would_split_start: Option<SplitInfo>,
    pub would_split_end: Option<SplitInfo>,
    pub would_merge_start: Option<NodeId>,
    pub would_merge_end: Option<NodeId>,
    pub lane_count_each_dir: (usize, usize),
    pub estimated_length: f32,
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
}

#[derive(Debug, Clone)]
enum PlannedNode {
    Existing(NodeId),
    New { pos: Vec3 },
    Split { lane_id: LaneId, t: f32, pos: Vec3 },
}

impl PlannedNode {
    fn position(&self, storage: &RoadStorage) -> Option<Vec3> {
        match self {
            PlannedNode::Existing(id) => {
                let node = storage.node(*id)?;
                Some(Vec3::new(node.x(), node.y(), node.z()))
            }
            PlannedNode::New { pos } => Some(*pos),
            PlannedNode::Split { pos, .. } => Some(*pos),
        }
    }

    fn merged_node_id(&self) -> Option<NodeId> {
        match self {
            PlannedNode::Existing(id) => Some(*id),
            _ => None,
        }
    }

    fn split_info(&self) -> Option<SplitInfo> {
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
struct Anchor {
    snap: SnapResult,
    planned_node: PlannedNode,
}

#[derive(Debug, Clone)]
enum EditorState {
    Idle,
    StraightPickEnd { start: Anchor },
    CurvePickControl { start: Anchor },
    CurvePickEnd { start: Anchor, control: Vec3 },
}

struct IdAllocator {
    next_node: u32,
    next_segment: u32,
    next_lane: u32,
}

impl IdAllocator {
    fn new() -> Self {
        Self {
            next_node: 0,
            next_segment: 0,
            next_lane: 0,
        }
    }
    pub(crate) fn update(&mut self, real_roads_storage: &RoadStorage) {
        self.next_node = real_roads_storage.nodes.len() as u32;
        self.next_segment = real_roads_storage.segments.len() as u32;
        self.next_lane = real_roads_storage.lanes.len() as u32;
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

// ============================================================================
// Road Editor Implementation
// ============================================================================

pub struct RoadEditor {
    state: EditorState,
    mode: BuildMode,
    road_type: RoadType,
    allocator: IdAllocator,
}

impl RoadEditor {
    pub fn new() -> Self {
        Self {
            state: EditorState::Idle,
            mode: BuildMode::Straight,
            road_type: RoadType::SmallRoad,
            allocator: IdAllocator::new(),
        }
    }

    pub fn set_mode(&mut self, mode: BuildMode) {
        if self.mode != mode {
            self.mode = mode;
            self.state = EditorState::Idle;
        }
    }

    pub fn set_road_type(&mut self, ty: RoadType) {
        self.road_type = ty;
    }

    pub fn mode(&self) -> BuildMode {
        self.mode
    }

    pub fn road_type(&self) -> RoadType {
        self.road_type
    }

    pub fn is_idle(&self) -> bool {
        matches!(self.state, EditorState::Idle)
    }

    pub fn cancel(&mut self) {
        self.state = EditorState::Idle;
    }

    pub fn update(
        &mut self,
        road_manager: &RoadManager,
        terrain_renderer: &TerrainRenderer,
        input: &mut InputState,
        picked_point: &Option<PickedPoint>,
    ) -> Vec<RoadEditorCommand> {
        self.allocator.update(&road_manager.preview_roads);
        let storage = &road_manager.roads;
        let mut output = Vec::new();

        if input.action_pressed_once("Cancel") {
            self.state = EditorState::Idle;
            output.push(RoadEditorCommand::PreviewClear);
            return output;
        }

        let Some(picked) = picked_point else {
            output.push(RoadEditorCommand::PreviewError(PreviewError::NoPickedPoint));
            output.push(RoadEditorCommand::PreviewClear);
            return output;
        };

        let chunk_id = picked.chunk.id;
        let snap = self.find_best_snap(storage, terrain_renderer, picked.pos);

        output.push(RoadEditorCommand::PreviewSnap(SnapPreview {
            world_pos: snap.world_pos,
            kind: snap.kind,
            distance: snap.distance,
        }));

        if let SnapKind::Lane { lane_id, t } = snap.kind {
            if let Some(lane_preview) =
                self.build_lane_preview(storage, terrain_renderer, lane_id, t)
            {
                output.push(RoadEditorCommand::PreviewLane(lane_preview));
            }
        }

        let place_pressed = input.action_pressed_once("Place Road Node");

        match self.state.clone() {
            EditorState::Idle => {
                self.handle_idle(storage, &snap, place_pressed, &mut output);
            }
            EditorState::StraightPickEnd { start } => {
                self.handle_straight_pick_end(
                    storage,
                    &start,
                    &snap,
                    place_pressed,
                    chunk_id,
                    &mut output,
                );
            }
            EditorState::CurvePickControl { start } => {
                self.handle_curve_pick_control(storage, &start, &snap, place_pressed, &mut output);
            }
            EditorState::CurvePickEnd { start, control } => {
                self.handle_curve_pick_end(
                    storage,
                    &start,
                    control,
                    &snap,
                    place_pressed,
                    chunk_id,
                    &mut output,
                );
            }
        }

        output
    }

    fn handle_idle(
        &mut self,
        storage: &RoadStorage,
        snap: &SnapResult,
        place_pressed: bool,
        output: &mut Vec<RoadEditorCommand>,
    ) {
        let node_preview = self.build_node_preview_from_snap(storage, snap);
        output.push(RoadEditorCommand::PreviewNode(node_preview));

        if place_pressed {
            let anchor = self.build_anchor_from_snap(snap);
            match self.mode {
                BuildMode::Straight => {
                    self.state = EditorState::StraightPickEnd { start: anchor };
                }
                BuildMode::Curved => {
                    self.state = EditorState::CurvePickControl { start: anchor };
                }
            }
        }
    }

    fn handle_straight_pick_end(
        &mut self,
        storage: &RoadStorage,
        start: &Anchor,
        snap: &SnapResult,
        place_pressed: bool,
        chunk_id: ChunkId,
        output: &mut Vec<RoadEditorCommand>,
    ) {
        let Some(start_pos) = start.planned_node.position(storage) else {
            output.push(RoadEditorCommand::PreviewError(
                PreviewError::MissingNodeData,
            ));
            self.state = EditorState::Idle;
            return;
        };

        let end_anchor = self.build_anchor_from_snap(snap);
        let end_pos = snap.world_pos;
        let polyline = vec![start_pos, end_pos];
        let estimated_length = (end_pos - start_pos).length();

        let (is_valid, reason) = self.validate_placement(storage, start, &end_anchor);

        let seg_preview = SegmentPreview {
            road_type: self.road_type,
            mode: self.mode,
            is_valid,
            reason_invalid: reason,
            start: start_pos,
            end: end_pos,
            control: None,
            polyline,
            would_split_start: start.planned_node.split_info(),
            would_split_end: end_anchor.planned_node.split_info(),
            would_merge_start: start.planned_node.merged_node_id(),
            would_merge_end: end_anchor.planned_node.merged_node_id(),
            lane_count_each_dir: self.road_type.lanes_each_direction(),
            estimated_length,
        };
        output.push(RoadEditorCommand::PreviewSegment(seg_preview.clone()));

        let node_preview = self.build_node_preview_from_snap(storage, snap);
        output.push(RoadEditorCommand::PreviewNode(node_preview));

        if place_pressed && is_valid {
            let road_cmds = self.commit_road(storage, start, &end_anchor, None, chunk_id, output);
            for cmd in road_cmds {
                output.push(RoadEditorCommand::Road(cmd));
            }
            self.state = EditorState::Idle;
        } else if place_pressed {
            if let Some(err) = seg_preview.reason_invalid {
                output.push(RoadEditorCommand::PreviewError(err));
            }
        }
    }

    fn handle_curve_pick_control(
        &mut self,
        storage: &RoadStorage,
        start: &Anchor,
        snap: &SnapResult,
        place_pressed: bool,
        output: &mut Vec<RoadEditorCommand>,
    ) {
        let Some(start_pos) = start.planned_node.position(storage) else {
            output.push(RoadEditorCommand::PreviewError(
                PreviewError::MissingNodeData,
            ));
            self.state = EditorState::Idle;
            return;
        };

        let control_pos = snap.world_pos;
        let polyline = vec![start_pos, control_pos];
        let estimated_length = (control_pos - start_pos).length();

        let seg_preview = SegmentPreview {
            road_type: self.road_type,
            mode: self.mode,
            is_valid: true,
            reason_invalid: None,
            start: start_pos,
            end: control_pos,
            control: Some(control_pos),
            polyline,
            would_split_start: start.planned_node.split_info(),
            would_split_end: None,
            would_merge_start: start.planned_node.merged_node_id(),
            would_merge_end: None,
            lane_count_each_dir: self.road_type.lanes_each_direction(),
            estimated_length,
        };
        output.push(RoadEditorCommand::PreviewSegment(seg_preview));

        if place_pressed {
            self.state = EditorState::CurvePickEnd {
                start: start.clone(),
                control: control_pos,
            };
        }
    }

    fn handle_curve_pick_end(
        &mut self,
        storage: &RoadStorage,
        start: &Anchor,
        control: Vec3,
        snap: &SnapResult,
        place_pressed: bool,
        chunk_id: ChunkId,
        output: &mut Vec<RoadEditorCommand>,
    ) {
        let Some(start_pos) = start.planned_node.position(storage) else {
            output.push(RoadEditorCommand::PreviewError(
                PreviewError::MissingNodeData,
            ));
            self.state = EditorState::Idle;
            return;
        };

        let end_anchor = self.build_anchor_from_snap(snap);
        let end_pos = snap.world_pos;

        let estimated_length = estimate_bezier_arc_length(start_pos, control, end_pos);
        let segment_count = compute_curve_segment_count(estimated_length);
        let polyline = sample_quadratic_bezier(start_pos, control, end_pos, segment_count);

        let (is_valid, reason) = self.validate_placement(storage, start, &end_anchor);

        let seg_preview = SegmentPreview {
            road_type: self.road_type,
            mode: self.mode,
            is_valid,
            reason_invalid: reason,
            start: start_pos,
            end: end_pos,
            control: Some(control),
            polyline: polyline.clone(),
            would_split_start: start.planned_node.split_info(),
            would_split_end: end_anchor.planned_node.split_info(),
            would_merge_start: start.planned_node.merged_node_id(),
            would_merge_end: end_anchor.planned_node.merged_node_id(),
            lane_count_each_dir: self.road_type.lanes_each_direction(),
            estimated_length,
        };
        output.push(RoadEditorCommand::PreviewSegment(seg_preview.clone()));

        let node_preview = self.build_node_preview_from_snap(storage, snap);
        output.push(RoadEditorCommand::PreviewNode(node_preview));

        if place_pressed && is_valid {
            let road_cmds =
                self.commit_road(storage, start, &end_anchor, Some(control), chunk_id, output);
            for cmd in road_cmds {
                output.push(RoadEditorCommand::Road(cmd));
            }
            self.state = EditorState::Idle;
        } else if place_pressed {
            if let Some(err) = seg_preview.reason_invalid {
                output.push(RoadEditorCommand::PreviewError(err));
            }
        }
    }

    fn find_best_snap(
        &self,
        storage: &RoadStorage,
        terrain_renderer: &TerrainRenderer,
        pos: Vec3,
    ) -> SnapResult {
        if let Some((node_id, node_pos, dist)) = self.find_nearest_node(storage, pos) {
            return SnapResult {
                world_pos: node_pos,
                kind: SnapKind::Node { id: node_id },
                distance: dist,
            };
        }

        if let Some((lane_id, t, projected_pos, dist)) =
            self.find_nearest_lane_snap(storage, terrain_renderer, pos)
        {
            return SnapResult {
                world_pos: projected_pos,
                kind: SnapKind::Lane { lane_id, t },
                distance: dist,
            };
        }

        SnapResult {
            world_pos: pos,
            kind: SnapKind::Free,
            distance: 0.0,
        }
    }

    fn find_nearest_node(&self, storage: &RoadStorage, pos: Vec3) -> Option<(NodeId, Vec3, f32)> {
        let mut best: Option<(NodeId, Vec3, f32)> = None;

        for (id, node) in storage.iter_enabled_nodes() {
            let node_pos = Vec3::new(node.x(), node.y(), node.z());
            let dx = node.x() - pos.x;
            let dz = node.z() - pos.z;
            let dist = (dx * dx + dz * dz).sqrt();

            if dist < NODE_SNAP_RADIUS {
                if best.is_none() || dist < best.as_ref().unwrap().2 {
                    best = Some((id, node_pos, dist));
                }
            }
        }

        best
    }

    fn find_nearest_lane_snap(
        &self,
        storage: &RoadStorage,
        terrain_renderer: &TerrainRenderer,
        pos: Vec3,
    ) -> Option<(LaneId, f32, Vec3, f32)> {
        let lane_id = nearest_lane_to_point(storage, pos.x, pos.y, pos.z)?;
        let lane = storage.lane(lane_id);
        let (t, dist_sq) = project_point_to_lane_xz(lane, pos.x, pos.z, storage);
        let dist = dist_sq.sqrt();

        if dist >= LANE_SNAP_RADIUS {
            return None;
        }

        if t < ENDPOINT_T_EPS {
            let node_id = lane.from_node();
            let node = storage.node(node_id)?;
            return Some((lane_id, 0.0, Vec3::new(node.x(), node.y(), node.z()), dist));
        }

        if t > 1.0 - ENDPOINT_T_EPS {
            let node_id = lane.to_node();
            let node = storage.node(node_id)?;
            return Some((lane_id, 1.0, Vec3::new(node.x(), node.y(), node.z()), dist));
        }

        let (px, pz) = sample_lane_position(lane, t, storage);
        let py = terrain_renderer.get_height_at([px, pz]) + CLEARANCE;
        Some((lane_id, t, Vec3::new(px, py, pz), dist))
    }

    fn build_anchor_from_snap(&self, snap: &SnapResult) -> Anchor {
        let planned_node = match snap.kind {
            SnapKind::Node { id } => PlannedNode::Existing(id),
            SnapKind::Lane { lane_id, t } => {
                if t < ENDPOINT_T_EPS || t > 1.0 - ENDPOINT_T_EPS {
                    PlannedNode::New {
                        pos: snap.world_pos,
                    }
                } else {
                    PlannedNode::Split {
                        lane_id,
                        t,
                        pos: snap.world_pos,
                    }
                }
            }
            SnapKind::Free => PlannedNode::New {
                pos: snap.world_pos,
            },
        };

        Anchor {
            snap: *snap,
            planned_node,
        }
    }

    fn build_node_preview_from_snap(
        &self,
        storage: &RoadStorage,
        snap: &SnapResult,
    ) -> NodePreview {
        let (result, incoming_lanes, outgoing_lanes) = match snap.kind {
            SnapKind::Node { id } => {
                let mut in_lanes = Vec::new();
                let mut out_lanes = Vec::new();

                let node = storage.node(id).unwrap();

                for &lane_id in node.incoming_lanes() {
                    let lane = storage.lane(lane_id);
                    let dir = lane_direction_at_node(lane, id);
                    in_lanes.push((lane_id, dir));
                }

                for &lane_id in node.outgoing_lanes() {
                    let lane = storage.lane(lane_id);
                    let dir = lane_direction_at_node(lane, id);
                    out_lanes.push((lane_id, dir));
                }

                (
                    NodePreviewResult::MergedIntoExisting(id),
                    in_lanes,
                    out_lanes,
                )
            }

            SnapKind::Lane { lane_id, t } => {
                if t < ENDPOINT_T_EPS || t > 1.0 - ENDPOINT_T_EPS {
                    (NodePreviewResult::NewNode, Vec::new(), Vec::new())
                } else {
                    let lane = storage.lane(lane_id);
                    let dir = lane.geometry().points[1] - lane.geometry().points[0];

                    (
                        NodePreviewResult::SplitIntersection,
                        vec![(lane_id, -dir.normalize())],
                        vec![(lane_id, dir.normalize())],
                    )
                }
            }

            SnapKind::Free => (NodePreviewResult::NewNode, Vec::new(), Vec::new()),
        };

        NodePreview {
            world_pos: snap.world_pos,
            result,
            is_valid: true,
            incoming_lanes,
            outgoing_lanes,
        }
    }

    fn build_lane_preview(
        &self,
        storage: &RoadStorage,
        terrain_renderer: &TerrainRenderer,
        lane_id: LaneId,
        t: f32,
    ) -> Option<LanePreview> {
        let lane = storage.lane(lane_id);
        let (px, pz) = sample_lane_position(lane, t, storage);

        let sample_count = 11;
        let mut sample_points = Vec::with_capacity(sample_count);
        for i in 0..sample_count {
            let sample_t = i as f32 / (sample_count - 1) as f32;
            let (sx, sz) = sample_lane_position(lane, sample_t, storage);
            let sy = terrain_renderer.get_height_at([sx, sz]) + CLEARANCE;
            sample_points.push(Vec3::new(sx, sy, sz));
        }
        let py = terrain_renderer.get_height_at([px, pz]) + CLEARANCE;
        Some(LanePreview {
            lane_id,
            projected_t: t,
            projected_point: Vec3::new(px, py, pz),
            sample_points,
        })
    }

    fn validate_placement(
        &self,
        storage: &RoadStorage,
        start: &Anchor,
        end: &Anchor,
    ) -> (bool, Option<PreviewError>) {
        let Some(start_pos) = start.planned_node.position(storage) else {
            return (false, Some(PreviewError::MissingNodeData));
        };

        let Some(end_pos) = end.planned_node.position(storage) else {
            return (false, Some(PreviewError::MissingNodeData));
        };

        if let (PlannedNode::Existing(start_id), PlannedNode::Existing(end_id)) =
            (&start.planned_node, &end.planned_node)
        {
            if start_id == end_id {
                return (false, Some(PreviewError::SameNode));
            }
        }

        let dx = end_pos.x - start_pos.x;
        let dz = end_pos.z - start_pos.z;
        let length_xz = (dx * dx + dz * dz).sqrt();

        if length_xz < MIN_SEGMENT_LENGTH {
            return (false, Some(PreviewError::TooShort));
        }

        (true, None)
    }

    fn commit_road(
        &mut self,
        storage: &RoadStorage,
        start: &Anchor,
        end: &Anchor,
        control: Option<Vec3>, // None = straight, Some = curve
        chunk_id: ChunkId,
        output: &mut Vec<RoadEditorCommand>,
    ) -> Vec<RoadCommand> {
        let mut cmds = Vec::new();

        // Resolve anchors
        let Some((start_id, start_pos)) =
            self.resolve_anchor(storage, start, chunk_id, &mut cmds, output)
        else {
            return Vec::new();
        };

        let Some((end_id, end_pos)) =
            self.resolve_anchor(storage, end, chunk_id, &mut cmds, output)
        else {
            return cmds;
        };

        if start_id == end_id {
            output.push(RoadEditorCommand::PreviewError(PreviewError::SameNode));
            return Vec::new();
        }

        // Build centerline polyline (EDITOR ONLY)
        let centerline: Vec<Vec3> = match control {
            Some(c) => {
                let est_len = estimate_bezier_arc_length(start_pos, c, end_pos);
                let samples = compute_curve_segment_count(est_len);
                sample_quadratic_bezier(start_pos, c, end_pos, samples)
            }
            None => vec![start_pos, end_pos],
        };

        if centerline.len() < 2 {
            return cmds;
        }

        // Allocate segment (topology only)
        let segment_id = self.allocator.alloc_segment();
        cmds.push(RoadCommand::AddSegment {
            start: start_id,
            end: end_id,
            structure: self.road_type.structure(),
            chunk_id,
        });

        // Emit lanes from centerline
        self.emit_lanes_from_centerline(
            &mut cmds,
            segment_id,
            start_id,
            end_id,
            &centerline,
            chunk_id,
        );

        cmds
    }

    fn resolve_anchor(
        &mut self,
        storage: &RoadStorage,
        anchor: &Anchor,
        chunk_id: ChunkId,
        cmds: &mut Vec<RoadCommand>,
        output: &mut Vec<RoadEditorCommand>,
    ) -> Option<(NodeId, Vec3)> {
        match &anchor.planned_node {
            PlannedNode::Existing(id) => {
                let node = storage.node(*id)?;
                Some((*id, Vec3::new(node.x(), node.y(), node.z())))
            }
            PlannedNode::New { pos } => {
                let node_id = self.allocator.alloc_node();
                cmds.push(RoadCommand::AddNode {
                    x: pos.x,
                    y: pos.y,
                    z: pos.z,
                    chunk_id,
                });
                Some((node_id, *pos))
            }
            PlannedNode::Split { lane_id, pos, .. } => {
                let result = self.split(storage, *lane_id, *pos, chunk_id, output)?;
                cmds.extend(result.0);
                Some((result.1, *pos))
            }
        }
    }

    fn split(
        &mut self,
        storage: &RoadStorage,
        lane_id: LaneId,
        split_pos: Vec3,
        chunk_id: ChunkId,
        output: &mut Vec<RoadEditorCommand>,
    ) -> Option<(Vec<RoadCommand>, NodeId)> {
        let lane = storage.lane(lane_id);
        let old_segment_id = lane.segment();
        let old_segment = storage.segment(old_segment_id);

        let a_id = old_segment.start();
        let b_id = old_segment.end();

        let mut cmds = Vec::new();

        cmds.push(RoadCommand::DisableSegment {
            segment_id: old_segment_id,
            chunk_id,
        });

        let new_node_id = self.allocator.alloc_node();
        cmds.push(RoadCommand::AddNode {
            x: split_pos.x,
            y: split_pos.y,
            z: split_pos.z,
            chunk_id,
        });

        let seg1_id = self.allocator.alloc_segment();
        let seg2_id = self.allocator.alloc_segment();

        cmds.push(RoadCommand::AddSegment {
            start: a_id,
            end: new_node_id,
            structure: old_segment.structure(),
            chunk_id,
        });

        cmds.push(RoadCommand::AddSegment {
            start: new_node_id,
            end: b_id,
            structure: old_segment.structure(),
            chunk_id,
        });

        for &old_lane_id in old_segment.lanes() {
            let old_lane = storage.lane(old_lane_id);

            let (geom1, geom2) = split_lane_geometry(old_lane.geometry(), split_pos);

            let total_len = old_lane.geometry().total_len.max(0.001);
            let cost1 = old_lane.base_cost() * (geom1.total_len / total_len);
            let cost2 = old_lane.base_cost() * (geom2.total_len / total_len);

            if old_lane.from_node() == a_id {
                cmds.push(RoadCommand::AddLane {
                    from: a_id,
                    to: new_node_id,
                    segment: seg1_id,
                    lane_index: old_lane.lane_index(),
                    geometry: geom1,
                    speed_limit: old_lane.speed_limit(),
                    capacity: old_lane.capacity(),
                    vehicle_mask: old_lane.vehicle_mask(),
                    base_cost: cost1,
                    chunk_id,
                });

                cmds.push(RoadCommand::AddLane {
                    from: new_node_id,
                    to: b_id,
                    segment: seg2_id,
                    lane_index: old_lane.lane_index(),
                    geometry: geom2,
                    speed_limit: old_lane.speed_limit(),
                    capacity: old_lane.capacity(),
                    vehicle_mask: old_lane.vehicle_mask(),
                    base_cost: cost2,
                    chunk_id,
                });
            } else {
                cmds.push(RoadCommand::AddLane {
                    from: b_id,
                    to: new_node_id,
                    segment: seg2_id,
                    lane_index: old_lane.lane_index(),
                    geometry: geom2,
                    speed_limit: old_lane.speed_limit(),
                    capacity: old_lane.capacity(),
                    vehicle_mask: old_lane.vehicle_mask(),
                    base_cost: cost2,
                    chunk_id,
                });

                cmds.push(RoadCommand::AddLane {
                    from: new_node_id,
                    to: a_id,
                    segment: seg1_id,
                    lane_index: old_lane.lane_index(),
                    geometry: geom1,
                    speed_limit: old_lane.speed_limit(),
                    capacity: old_lane.capacity(),
                    vehicle_mask: old_lane.vehicle_mask(),
                    base_cost: cost1,
                    chunk_id,
                });
            }
        }

        Some((cmds, new_node_id))
    }

    fn emit_lanes_from_centerline(
        &self,
        cmds: &mut Vec<RoadCommand>,
        segment: SegmentId,
        start: NodeId,
        end: NodeId,
        centerline: &[Vec3],
        chunk_id: ChunkId,
    ) {
        let (left_lanes, right_lanes) = self.road_type.lanes_each_direction();
        let speed = self.road_type.speed_limit();
        let capacity = self.road_type.capacity();
        let mask = self.road_type.vehicle_mask();
        let lane_width = DEFAULT_LANE_WIDTH;

        // Right side lanes (start -> end)
        for i in 0..right_lanes {
            let lane_index = (i as i8) + 1;
            let poly = offset_polyline(centerline, lane_index, lane_width);
            let geom = LaneGeometry::from_polyline(poly);
            let base_cost = geom.total_len.max(0.1);
            cmds.push(RoadCommand::AddLane {
                from: start,
                to: end,
                segment,
                lane_index,
                geometry: geom,
                speed_limit: speed,
                capacity,
                vehicle_mask: mask,
                base_cost,
                chunk_id,
            });
        }

        // Left side lanes (end -> start, reversed geometry)
        for i in 0..left_lanes {
            let lane_index = -((i as i8) + 1);
            let mut poly = offset_polyline(centerline, lane_index.abs(), lane_width);
            poly.reverse();
            let geom = LaneGeometry::from_polyline(poly);
            let base_cost = geom.total_len.max(0.1);
            cmds.push(RoadCommand::AddLane {
                from: end,
                to: start,
                segment,
                lane_index,
                geometry: geom,
                speed_limit: speed,
                capacity,
                vehicle_mask: mask,
                base_cost,
                chunk_id,
            });
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn split_lane_geometry(geom: &LaneGeometry, split_pos: Vec3) -> (LaneGeometry, LaneGeometry) {
    // Find the closest segment on polyline
    let mut best_i = 0;
    let mut best_t = 0.0;
    let mut best_dist = f32::MAX;

    for i in 0..geom.points.len() - 1 {
        let a = geom.points[i];
        let b = geom.points[i + 1];

        let ab = b - a;
        let t = ((split_pos - a).dot(ab) / ab.dot(ab)).clamp(0.0, 1.0);
        let p = a + ab * t;

        let d = p.distance(split_pos);
        if d < best_dist {
            best_dist = d;
            best_i = i;
            best_t = t;
        }
    }

    let a = geom.points[best_i];
    let b = geom.points[best_i + 1];
    let split_point = a + (b - a) * best_t;

    let mut pts1 = Vec::new();
    let mut pts2 = Vec::new();

    pts1.extend_from_slice(&geom.points[..=best_i]);
    pts1.push(split_point);

    pts2.push(split_point);
    pts2.extend_from_slice(&geom.points[best_i + 1..]);

    (
        LaneGeometry::from_polyline(pts1),
        LaneGeometry::from_polyline(pts2),
    )
}

fn lane_direction_at_node(lane: &Lane, node: NodeId) -> Vec3 {
    let pts = &lane.geometry().points;

    if lane.from_node() == node {
        (pts[1] - pts[0]).normalize()
    } else {
        let n = pts.len();
        (pts[n - 2] - pts[n - 1]).normalize()
    }
}

fn sample_quadratic_bezier(p0: Vec3, p1: Vec3, p2: Vec3, segments: usize) -> Vec<Vec3> {
    let segments = segments.max(1);
    let mut points = Vec::with_capacity(segments + 1);

    for i in 0..=segments {
        let t = i as f32 / segments as f32;
        let one_minus_t = 1.0 - t;
        let point = p0 * (one_minus_t * one_minus_t) + p1 * (2.0 * one_minus_t * t) + p2 * (t * t);
        points.push(point);
    }
    points
}

fn estimate_bezier_arc_length(p0: Vec3, p1: Vec3, p2: Vec3) -> f32 {
    let samples = sample_quadratic_bezier(p0, p1, p2, 16);
    polyline_length(&samples)
}

fn polyline_length(points: &[Vec3]) -> f32 {
    let mut length = 0.0;
    for i in 1..points.len() {
        length += (points[i] - points[i - 1]).length();
    }
    length
}

fn compute_curve_segment_count(arc_length: f32) -> usize {
    ((arc_length / 6.0).ceil() as usize).clamp(4, 32)
}

fn gather_node_lanes(storage: &RoadStorage, node_id: NodeId) -> (Vec<LaneId>, Vec<LaneId>) {
    let Some(node) = storage.node(node_id) else {
        return (Vec::new(), Vec::new());
    };
    (
        node.incoming_lanes().to_vec(),
        node.outgoing_lanes().to_vec(),
    )
}

fn gather_split_lanes(manager: &RoadManager, lane_id: LaneId) -> (Vec<LaneId>, Vec<LaneId>) {
    // At a split point: from → new_node is incoming, new_node → to is outgoing
    (vec![lane_id], vec![lane_id])
}

fn offset_polyline(center: &[Vec3], lane_index: i8, lane_width: f32) -> Vec<Vec3> {
    let offset = (lane_index as f32 - 0.5) * lane_width;
    let mut out = Vec::with_capacity(center.len());

    for i in 0..center.len() {
        let dir = if i + 1 < center.len() {
            center[i + 1] - center[i]
        } else {
            center[i] - center[i - 1]
        };

        let dir_xz = Vec3::new(dir.x, 0.0, dir.z).normalize_or_zero();
        let right = Vec3::new(dir_xz.z, 0.0, -dir_xz.x);

        out.push(center[i] + right * offset);
    }

    out
}
