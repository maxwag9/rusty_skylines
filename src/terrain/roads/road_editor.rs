use crate::renderer::world_renderer::PickedPoint;
use crate::resources::InputState;
use crate::terrain::roads::road_mesh_manager::{
    ChunkId, ConnectedSegmentInfo, CrossSection, HorizontalProfile, SegmentGeometry,
};
use crate::terrain::roads::roads::{
    LaneId, NodeId, RoadCommand, RoadManager, SegmentId, StructureType, VerticalProfile,
    nearest_lane_to_point, project_point_to_lane_xz, sample_lane_position,
};
use glam::Vec3;

const NODE_SNAP_RADIUS: f32 = 8.0;
const LANE_SNAP_RADIUS: f32 = 8.0;
const ENDPOINT_T_EPS: f32 = 0.02;
const MIN_SEGMENT_LENGTH: f32 = 1.0;

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

#[derive(Debug, Clone)]
pub struct NodePreview {
    pub world_pos: Vec3,
    pub result: NodePreviewResult,
    pub is_valid: bool,
    pub connected_segments: Vec<ConnectedSegmentInfo>,
    pub incoming_lanes: Vec<LaneId>,
    pub outgoing_lanes: Vec<LaneId>,
}

impl NodePreview {
    pub(crate) fn lane_counts(&self) -> (usize, usize) {
        (self.incoming_lanes.len(), self.outgoing_lanes.len())
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
    fn position(&self, manager: &RoadManager) -> Option<Vec3> {
        match self {
            PlannedNode::Existing(id) => {
                let node = manager.node(*id)?;
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
}

impl IdAllocator {
    fn new(manager: &RoadManager) -> Self {
        Self {
            next_node: manager.nodes.len() as u32,
            next_segment: manager.segments.len() as u32,
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
}

pub struct RoadEditor {
    state: EditorState,
    mode: BuildMode,
    road_type: RoadType,
}

impl Default for RoadEditor {
    fn default() -> Self {
        Self::new()
    }
}

impl RoadEditor {
    pub fn new() -> Self {
        Self {
            state: EditorState::Idle,
            mode: BuildMode::Straight,
            road_type: RoadType::SmallRoad,
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
        manager: &RoadManager,
        input: &mut InputState,
        picked_point: &Option<PickedPoint>,
    ) -> Vec<RoadEditorCommand> {
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
        let snap = self.find_best_snap(manager, picked.pos);

        output.push(RoadEditorCommand::PreviewSnap(SnapPreview {
            world_pos: snap.world_pos,
            kind: snap.kind,
            distance: snap.distance,
        }));

        if let SnapKind::Lane { lane_id, t } = snap.kind {
            if let Some(lane_preview) = self.build_lane_preview(manager, lane_id, t) {
                output.push(RoadEditorCommand::PreviewLane(lane_preview));
            }
        }

        let place_pressed = input.action_pressed_once("Place Road Node");

        match self.state.clone() {
            EditorState::Idle => {
                self.handle_idle(manager, &snap, place_pressed, &mut output);
            }
            EditorState::StraightPickEnd { start } => {
                self.handle_straight_pick_end(
                    manager,
                    &start,
                    &snap,
                    place_pressed,
                    chunk_id,
                    &mut output,
                );
            }
            EditorState::CurvePickControl { start } => {
                self.handle_curve_pick_control(manager, &start, &snap, place_pressed, &mut output);
            }
            EditorState::CurvePickEnd { start, control } => {
                self.handle_curve_pick_end(
                    manager,
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
        manager: &RoadManager,
        snap: &SnapResult,
        place_pressed: bool,
        output: &mut Vec<RoadEditorCommand>,
    ) {
        let node_preview = self.build_node_preview_from_snap(manager, snap);
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
        manager: &RoadManager,
        start: &Anchor,
        snap: &SnapResult,
        place_pressed: bool,
        chunk_id: ChunkId,
        output: &mut Vec<RoadEditorCommand>,
    ) {
        let Some(start_pos) = start.planned_node.position(manager) else {
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

        let (is_valid, reason) = self.validate_placement(manager, start, &end_anchor);

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

        let node_preview = self.build_node_preview_from_snap(manager, snap);
        output.push(RoadEditorCommand::PreviewNode(node_preview));

        if place_pressed && is_valid {
            let road_cmds =
                self.commit_straight_road(manager, start, &end_anchor, chunk_id, output);
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
        manager: &RoadManager,
        start: &Anchor,
        snap: &SnapResult,
        place_pressed: bool,
        output: &mut Vec<RoadEditorCommand>,
    ) {
        let Some(start_pos) = start.planned_node.position(manager) else {
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
        manager: &RoadManager,
        start: &Anchor,
        control: Vec3,
        snap: &SnapResult,
        place_pressed: bool,
        chunk_id: ChunkId,
        output: &mut Vec<RoadEditorCommand>,
    ) {
        let Some(start_pos) = start.planned_node.position(manager) else {
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

        let (is_valid, reason) = self.validate_placement(manager, start, &end_anchor);

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

        let node_preview = self.build_node_preview_from_snap(manager, snap);
        output.push(RoadEditorCommand::PreviewNode(node_preview));

        if place_pressed && is_valid {
            let road_cmds =
                self.commit_curved_road(manager, start, control, &end_anchor, chunk_id, output);
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

    fn find_best_snap(&self, manager: &RoadManager, pos: Vec3) -> SnapResult {
        if let Some((node_id, node_pos, dist)) = self.find_nearest_node(manager, pos) {
            return SnapResult {
                world_pos: node_pos,
                kind: SnapKind::Node { id: node_id },
                distance: dist,
            };
        }

        if let Some((lane_id, t, projected_pos, dist)) = self.find_nearest_lane_snap(manager, pos) {
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

    fn find_nearest_node(&self, manager: &RoadManager, pos: Vec3) -> Option<(NodeId, Vec3, f32)> {
        let mut best: Option<(NodeId, Vec3, f32)> = None;

        for (id, node) in manager.iter_enabled_nodes() {
            let node_pos = Vec3::new(node.x, node.y, node.z);
            let dx = node.x - pos.x;
            let dz = node.z - pos.z;
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
        manager: &RoadManager,
        pos: Vec3,
    ) -> Option<(LaneId, f32, Vec3, f32)> {
        let lane_id = nearest_lane_to_point(manager, pos.x, pos.y, pos.z)?;
        let lane = manager.lane(lane_id);
        let (t, dist_sq) = project_point_to_lane_xz(lane, pos.x, pos.z, manager);
        let dist = dist_sq.sqrt();

        if dist >= LANE_SNAP_RADIUS {
            return None;
        }

        if t < ENDPOINT_T_EPS {
            let node_id = lane.from_node();
            let node = manager.node(node_id)?;
            return Some((lane_id, 0.0, Vec3::new(node.x(), node.y(), node.z()), dist));
        }

        if t > 1.0 - ENDPOINT_T_EPS {
            let node_id = lane.to_node();
            let node = manager.node(node_id)?;
            return Some((lane_id, 1.0, Vec3::new(node.x(), node.y(), node.z()), dist));
        }

        let (px, py, pz) = sample_lane_position(lane, t, manager);
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
        manager: &RoadManager,
        snap: &SnapResult,
    ) -> NodePreview {
        let (result, incoming_lanes, outgoing_lanes, connected_segments) = match snap.kind {
            SnapKind::Node { id: node_id } => {
                let (in_lanes, out_lanes) = gather_node_lanes(manager, node_id);
                let connected = manager.segments_connected_to_node(node_id);
                let mut connections = Vec::new();
                for segment_id in connected {
                    let segment = manager.segment(segment_id);
                    let Some(geom) = SegmentGeometry::from_segment(segment_id, segment, manager)
                    else {
                        continue;
                    };
                    let cross_section = CrossSection::from_segment(manager, segment);

                    let is_start = segment.start() == node_id;
                    let (t_at_node, direction_sign) =
                        if is_start { (0.0, 1.0) } else { (1.0, -1.0) };

                    let tangent = geom.tangent_xz(t_at_node);
                    let direction = [tangent[0] * direction_sign, tangent[1] * direction_sign];

                    connections.push(ConnectedSegmentInfo {
                        direction_xz: direction,
                        tangent_xz: tangent,
                        segment_id: None,
                        cross_section,
                        node_is_start: is_start,
                    });
                }
                (
                    NodePreviewResult::MergedIntoExisting(node_id),
                    in_lanes,
                    out_lanes,
                    connections,
                )
            }

            SnapKind::Lane { lane_id, t } => {
                if t < ENDPOINT_T_EPS || t > 1.0 - ENDPOINT_T_EPS {
                    // Endpoint snap behaves like free placement
                    (
                        NodePreviewResult::NewNode,
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                    )
                } else {
                    let (in_lanes, out_lanes) = gather_split_lanes(manager, lane_id);
                    (
                        NodePreviewResult::SplitIntersection,
                        in_lanes,
                        out_lanes,
                        Vec::new(),
                    )
                }
            }

            SnapKind::Free => (
                NodePreviewResult::NewNode,
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ),
        };

        NodePreview {
            world_pos: snap.world_pos,
            result,
            is_valid: true,
            connected_segments,
            incoming_lanes,
            outgoing_lanes,
        }
    }

    fn build_lane_preview(
        &self,
        manager: &RoadManager,
        lane_id: LaneId,
        t: f32,
    ) -> Option<LanePreview> {
        let lane = manager.lane(lane_id);
        let (px, py, pz) = sample_lane_position(lane, t, manager);

        let sample_count = 11;
        let mut sample_points = Vec::with_capacity(sample_count);
        for i in 0..sample_count {
            let sample_t = i as f32 / (sample_count - 1) as f32;
            let (sx, sy, sz) = sample_lane_position(lane, sample_t, manager);
            sample_points.push(Vec3::new(sx, sy, sz));
        }

        Some(LanePreview {
            lane_id,
            projected_t: t,
            projected_point: Vec3::new(px, py, pz),
            sample_points,
        })
    }

    fn validate_placement(
        &self,
        manager: &RoadManager,
        start: &Anchor,
        end: &Anchor,
    ) -> (bool, Option<PreviewError>) {
        let Some(start_pos) = start.planned_node.position(manager) else {
            return (false, Some(PreviewError::MissingNodeData));
        };

        let Some(end_pos) = end.planned_node.position(manager) else {
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

    fn commit_straight_road(
        &self,
        manager: &RoadManager,
        start: &Anchor,
        end: &Anchor,
        chunk_id: ChunkId,
        output: &mut Vec<RoadEditorCommand>,
    ) -> Vec<RoadCommand> {
        let mut cmds = Vec::new();
        let mut allocator = IdAllocator::new(manager);

        let Some((start_id, start_pos)) =
            self.resolve_anchor(manager, start, chunk_id, &mut cmds, &mut allocator, output)
        else {
            return Vec::new();
        };

        let Some((end_id, end_pos)) =
            self.resolve_anchor(manager, end, chunk_id, &mut cmds, &mut allocator, output)
        else {
            return cmds;
        };

        if start_id == end_id {
            output.push(RoadEditorCommand::PreviewError(PreviewError::SameNode));
            return Vec::new();
        }

        let dx = end_pos.x - start_pos.x;
        let dz = end_pos.z - start_pos.z;
        let length_xz = (dx * dx + dz * dz).sqrt();

        if length_xz < MIN_SEGMENT_LENGTH {
            output.push(RoadEditorCommand::PreviewError(PreviewError::TooShort));
            return Vec::new();
        }

        let segment_id = allocator.alloc_segment();
        cmds.push(RoadCommand::AddSegment {
            start: start_id,
            end: end_id,
            structure: self.road_type.structure(),
            horizontal_profile: HorizontalProfile::Linear,
            vertical_profile: VerticalProfile::EndPoints {
                start_y: start_pos.y,
                end_y: end_pos.y,
            },
            chunk_id,
        });

        self.emit_lanes_for_segment(&mut cmds, start_id, end_id, segment_id, length_xz, chunk_id);

        cmds
    }

    fn commit_curved_road(
        &self,
        manager: &RoadManager,
        start: &Anchor,
        control: Vec3,
        end: &Anchor,
        chunk_id: ChunkId,
        output: &mut Vec<RoadEditorCommand>,
    ) -> Vec<RoadCommand> {
        let mut cmds = Vec::new();
        let mut allocator = IdAllocator::new(manager);

        let Some((start_id, start_pos)) =
            self.resolve_anchor(manager, start, chunk_id, &mut cmds, &mut allocator, output)
        else {
            return Vec::new();
        };

        let Some((end_id, end_pos)) =
            self.resolve_anchor(manager, end, chunk_id, &mut cmds, &mut allocator, output)
        else {
            return cmds;
        };

        if start_id == end_id {
            output.push(RoadEditorCommand::PreviewError(PreviewError::SameNode));
            return Vec::new();
        }

        let estimated_length = estimate_bezier_arc_length(start_pos, control, end_pos);
        let segment_count = compute_curve_segment_count(estimated_length);
        let polyline = sample_quadratic_bezier(start_pos, control, end_pos, segment_count);

        if polyline.len() < 2 {
            return cmds;
        }

        let mut node_ids: Vec<NodeId> = Vec::with_capacity(polyline.len());
        node_ids.push(start_id);

        for i in 1..polyline.len() - 1 {
            let pos = polyline[i];
            let node_id = allocator.alloc_node();
            cmds.push(RoadCommand::AddNode {
                x: pos.x,
                y: pos.y,
                z: pos.z,
                chunk_id,
            });
            node_ids.push(node_id);
        }

        node_ids.push(end_id);

        for i in 0..node_ids.len() - 1 {
            let seg_start_id = node_ids[i];
            let seg_end_id = node_ids[i + 1];
            let seg_start_pos = polyline[i];
            let seg_end_pos = polyline[i + 1];

            let segment_id = allocator.alloc_segment();

            cmds.push(RoadCommand::AddSegment {
                start: seg_start_id,
                end: seg_end_id,
                structure: self.road_type.structure(),
                horizontal_profile: HorizontalProfile::Linear,
                vertical_profile: VerticalProfile::EndPoints {
                    start_y: seg_start_pos.y,
                    end_y: seg_end_pos.y,
                },
                chunk_id,
            });

            let dx = seg_end_pos.x - seg_start_pos.x;
            let dz = seg_end_pos.z - seg_start_pos.z;
            let seg_length = (dx * dx + dz * dz).sqrt().max(0.1);

            self.emit_lanes_for_segment(
                &mut cmds,
                seg_start_id,
                seg_end_id,
                segment_id,
                seg_length,
                chunk_id,
            );
        }

        cmds
    }

    fn resolve_anchor(
        &self,
        manager: &RoadManager,
        anchor: &Anchor,
        chunk_id: ChunkId,
        cmds: &mut Vec<RoadCommand>,
        allocator: &mut IdAllocator,
        output: &mut Vec<RoadEditorCommand>,
    ) -> Option<(NodeId, Vec3)> {
        match &anchor.planned_node {
            PlannedNode::Existing(id) => {
                let node = manager.node(*id)?;
                Some((*id, Vec3::new(node.x(), node.y(), node.z())))
            }
            PlannedNode::New { pos } => {
                let node_id = allocator.alloc_node();
                cmds.push(RoadCommand::AddNode {
                    x: pos.x,
                    y: pos.y,
                    z: pos.z,
                    chunk_id,
                });
                Some((node_id, *pos))
            }
            PlannedNode::Split { lane_id, pos, .. } => {
                let result =
                    self.plan_split(manager, *lane_id, *pos, chunk_id, allocator, output)?;
                cmds.extend(result.0);
                Some((result.1, *pos))
            }
        }
    }

    fn plan_split(
        &self,
        manager: &RoadManager,
        lane_id: LaneId,
        split_pos: Vec3,
        chunk_id: ChunkId,
        allocator: &mut IdAllocator,
        output: &mut Vec<RoadEditorCommand>,
    ) -> Option<(Vec<RoadCommand>, NodeId)> {
        let lane = manager.lane(lane_id);
        let old_segment_id = lane.segment();
        let old_segment = manager.segment(old_segment_id);

        let a_id = old_segment.start();
        let b_id = old_segment.end();

        let a_node = manager.node(a_id);
        let b_node = manager.node(b_id);

        if a_node.is_none() || b_node.is_none() {
            output.push(RoadEditorCommand::PreviewError(
                PreviewError::MissingNodeData,
            ));
            return None;
        }

        let a_node = a_node.unwrap();
        let b_node = b_node.unwrap();

        let mut cmds = Vec::new();

        cmds.push(RoadCommand::DisableSegment {
            segment_id: old_segment_id,
            chunk_id,
        });

        let new_node_id = allocator.alloc_node();
        cmds.push(RoadCommand::AddNode {
            x: split_pos.x,
            y: split_pos.y,
            z: split_pos.z,
            chunk_id,
        });

        let seg1_id = allocator.alloc_segment();
        let seg2_id = allocator.alloc_segment();

        let structure = old_segment.structure();
        let y_a = a_node.y();
        let y_b = b_node.y();

        cmds.push(RoadCommand::AddSegment {
            start: a_id,
            end: new_node_id,
            structure,
            horizontal_profile: HorizontalProfile::Linear,
            vertical_profile: VerticalProfile::EndPoints {
                start_y: y_a,
                end_y: split_pos.y,
            },
            chunk_id,
        });

        cmds.push(RoadCommand::AddSegment {
            start: new_node_id,
            end: b_id,
            structure,
            horizontal_profile: HorizontalProfile::Linear,
            vertical_profile: VerticalProfile::EndPoints {
                start_y: split_pos.y,
                end_y: y_b,
            },
            chunk_id,
        });

        for &old_lane_id in old_segment.lanes() {
            let old_lane = manager.lane(old_lane_id);
            let speed = old_lane.speed_limit();
            let cap = old_lane.capacity();
            let mask = old_lane.vehicle_mask();
            let old_cost = old_lane.base_cost();

            let a_pos = Vec3::new(a_node.x(), a_node.y(), a_node.z());
            let b_pos = Vec3::new(b_node.x(), b_node.y(), b_node.z());

            let total_len = ((b_pos.x - a_pos.x).powi(2) + (b_pos.z - a_pos.z).powi(2)).sqrt();
            let len1 = ((split_pos.x - a_pos.x).powi(2) + (split_pos.z - a_pos.z).powi(2)).sqrt();
            let len2 = ((b_pos.x - split_pos.x).powi(2) + (b_pos.z - split_pos.z).powi(2)).sqrt();

            let ratio1 = if total_len > 0.01 {
                len1 / total_len
            } else {
                0.5
            };
            let ratio2 = if total_len > 0.01 {
                len2 / total_len
            } else {
                0.5
            };

            let cost1 = old_cost * ratio1;
            let cost2 = old_cost * ratio2;

            if old_lane.from_node() == a_id {
                cmds.push(RoadCommand::AddLane {
                    from: a_id,
                    to: new_node_id,
                    segment: seg1_id,
                    speed_limit: speed,
                    capacity: cap,
                    vehicle_mask: mask,
                    base_cost: cost1,
                    chunk_id,
                });
                cmds.push(RoadCommand::AddLane {
                    from: new_node_id,
                    to: b_id,
                    segment: seg2_id,
                    speed_limit: speed,
                    capacity: cap,
                    vehicle_mask: mask,
                    base_cost: cost2,
                    chunk_id,
                });
            } else {
                cmds.push(RoadCommand::AddLane {
                    from: b_id,
                    to: new_node_id,
                    segment: seg2_id,
                    speed_limit: speed,
                    capacity: cap,
                    vehicle_mask: mask,
                    base_cost: cost2,
                    chunk_id,
                });
                cmds.push(RoadCommand::AddLane {
                    from: new_node_id,
                    to: a_id,
                    segment: seg1_id,
                    speed_limit: speed,
                    capacity: cap,
                    vehicle_mask: mask,
                    base_cost: cost1,
                    chunk_id,
                });
            }
        }

        Some((cmds, new_node_id))
    }

    fn emit_lanes_for_segment(
        &self,
        cmds: &mut Vec<RoadCommand>,
        start: NodeId,
        end: NodeId,
        segment: SegmentId,
        base_cost: f32,
        chunk_id: ChunkId,
    ) {
        let (left_lanes, right_lanes) = self.road_type.lanes_each_direction();
        let speed = self.road_type.speed_limit();
        let capacity = self.road_type.capacity();
        let mask = self.road_type.vehicle_mask();
        let cost = base_cost.max(0.1);

        for _ in 0..right_lanes {
            cmds.push(RoadCommand::AddLane {
                from: start,
                to: end,
                segment,
                speed_limit: speed,
                capacity,
                vehicle_mask: mask,
                base_cost: cost,
                chunk_id,
            });
        }

        for _ in 0..left_lanes {
            cmds.push(RoadCommand::AddLane {
                from: end,
                to: start,
                segment,
                speed_limit: speed,
                capacity,
                vehicle_mask: mask,
                base_cost: cost,
                chunk_id,
            });
        }
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

fn gather_node_lanes(manager: &RoadManager, node_id: NodeId) -> (Vec<LaneId>, Vec<LaneId>) {
    let Some(node) = manager.node(node_id) else {
        return (Vec::new(), Vec::new());
    };

    (
        node.incoming_lanes().to_vec(),
        node.outgoing_lanes().to_vec(),
    )
}

fn gather_split_lanes(manager: &RoadManager, lane_id: LaneId) -> (Vec<LaneId>, Vec<LaneId>) {
    let lane = manager.lane(lane_id);

    // At a split point:
    // from → new_node is incoming
    // new_node → to is outgoing
    (
        vec![lane_id], // incoming
        vec![lane_id], // outgoing
    )
}
