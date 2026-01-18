use crate::renderer::world_renderer::{PickedPoint, TerrainRenderer};
use crate::resources::InputState;
use crate::terrain::roads::road_mesh_manager::{CLEARANCE, ChunkId, RoadStyleParams};
use crate::terrain::roads::roads::{
    Lane, LaneGeometry, LaneId, LaneRef, METERS_PER_LANE_POLYLINE_STEP, NodeId, NodeLane,
    NodeLaneId, PolyIdx, RoadCommand, RoadManager, RoadStorage, SegmentId, StructureType, bezier3,
    nearest_lane_to_point, project_point_to_lane_xz, sample_lane_position,
};
use glam::Vec3;
use std::cmp::PartialEq;

const NODE_SNAP_RADIUS: f32 = 8.0;
const LANE_SNAP_RADIUS: f32 = 8.0;
const ENDPOINT_T_EPS: f32 = 0.02;
const MIN_SEGMENT_LENGTH: f32 = 1.0;

// ============================================================================
// Types & Structs
// ============================================================================

#[derive(Debug, Clone)]
pub struct RoadType {
    pub name: &'static str,

    pub lanes_each_direction: (usize, usize),

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
}
impl Default for RoadType {
    fn default() -> Self {
        Self {
            name: "Medium Road",

            lanes_each_direction: (1, 1),

            lane_width: 2.75,
            lane_height: 0.0,
            lane_material_id: 2, // asphalt

            sidewalk_width: 1.75,
            sidewalk_height: 0.15,
            sidewalk_material_id: 0, // concrete

            median_width: 0.3,
            median_height: 0.15,
            median_material_id: 0, // concrete

            speed_limit: 16.7,
            vehicle_mask: 1,
            structure: StructureType::Surface,
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

    pub fn lanes_each_direction(&self) -> (usize, usize) {
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
pub enum PlannedNode {
    Existing(NodeId),
    New { pos: Vec3 },
    Split { lane_id: LaneId, t: f32, pos: Vec3 },
}

impl PlannedNode {
    pub fn position(&self, storage: &RoadStorage) -> Option<Vec3> {
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
pub struct Anchor {
    pub snap: SnapResult,
    pub planned_node: PlannedNode,
}

#[derive(Debug, Clone)]
pub enum EditorState {
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
    allocator: IdAllocator,
}

impl RoadEditor {
    pub fn new() -> Self {
        Self {
            allocator: IdAllocator::new(),
        }
    }

    pub fn update(
        &mut self,
        road_manager: &RoadManager,
        terrain_renderer: &TerrainRenderer,
        road_style_params: &mut RoadStyleParams,
        input: &mut InputState,
        picked_point: &Option<PickedPoint>,
    ) -> Vec<RoadEditorCommand> {
        self.allocator.update(&road_manager.roads);
        let storage = &road_manager.roads;
        let mut output = Vec::new();

        if input.action_pressed_once("Cancel") {
            road_style_params.set_to_idle();
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

        match road_style_params.state().clone() {
            EditorState::Idle => {
                self.handle_idle(
                    road_style_params,
                    storage,
                    &snap,
                    place_pressed,
                    &mut output,
                );
            }
            EditorState::StraightPickEnd { start } => {
                self.handle_straight_pick_end(
                    terrain_renderer,
                    storage,
                    road_style_params,
                    &start,
                    &snap,
                    place_pressed,
                    chunk_id,
                    &mut output,
                );
            }
            EditorState::CurvePickControl { start } => {
                self.handle_curve_pick_control(
                    road_style_params,
                    storage,
                    &start,
                    &snap,
                    place_pressed,
                    &mut output,
                );
            }
            EditorState::CurvePickEnd { start, control } => {
                self.handle_curve_pick_end(
                    terrain_renderer,
                    storage,
                    road_style_params,
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
        road_style_params: &mut RoadStyleParams,
        storage: &RoadStorage,
        snap: &SnapResult,
        place_pressed: bool,
        output: &mut Vec<RoadEditorCommand>,
    ) {
        let node_preview = self.build_node_preview_from_snap(storage, snap);
        output.push(RoadEditorCommand::PreviewNode(node_preview));

        if place_pressed {
            let anchor = self.build_anchor_from_snap(snap);
            match road_style_params.mode() {
                BuildMode::Straight => {
                    road_style_params.set_state(EditorState::StraightPickEnd { start: anchor });
                }
                BuildMode::Curved => {
                    road_style_params.set_state(EditorState::CurvePickControl { start: anchor });
                }
            }
        }
    }

    fn handle_straight_pick_end(
        &mut self,
        terrain_renderer: &TerrainRenderer,
        storage: &RoadStorage,
        road_style_params: &mut RoadStyleParams,
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
            road_style_params.set_state(EditorState::Idle);
            return;
        };

        let end_anchor = self.build_anchor_from_snap(snap);
        let end_pos = snap.world_pos;
        let polyline = make_straight_centerline(terrain_renderer, start_pos, end_pos);
        let estimated_length = (end_pos - start_pos).length();

        let (is_valid, reason) = self.validate_placement(storage, start, &end_anchor);

        let seg_preview = SegmentPreview {
            road_type: road_style_params.road_type().clone(),
            mode: road_style_params.mode(),
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
            lane_count_each_dir: road_style_params.road_type().lanes_each_direction(),
            estimated_length,
        };
        output.push(RoadEditorCommand::PreviewSegment(seg_preview.clone()));

        let node_preview = self.build_node_preview_from_snap(storage, snap);
        output.push(RoadEditorCommand::PreviewNode(node_preview));

        if place_pressed && is_valid {
            let road_cmds = self.commit_road(
                terrain_renderer,
                storage,
                road_style_params,
                start,
                &end_anchor,
                None,
                chunk_id,
                output,
            );
            for cmd in road_cmds {
                output.push(RoadEditorCommand::Road(cmd));
            }
            road_style_params.set_to_idle();
        } else if place_pressed {
            if let Some(err) = seg_preview.reason_invalid {
                output.push(RoadEditorCommand::PreviewError(err));
            }
        }
    }

    fn handle_curve_pick_control(
        &mut self,
        road_style_params: &mut RoadStyleParams,
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
            road_style_params.set_to_idle();
            return;
        };

        let control_pos = snap.world_pos;
        let polyline = vec![start_pos, control_pos];
        let estimated_length = (control_pos - start_pos).length();

        let seg_preview = SegmentPreview {
            road_type: road_style_params.road_type().clone(),
            mode: road_style_params.mode(),
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
            lane_count_each_dir: road_style_params.road_type().lanes_each_direction(),
            estimated_length,
        };
        output.push(RoadEditorCommand::PreviewSegment(seg_preview));

        if place_pressed {
            road_style_params.set_state(EditorState::CurvePickEnd {
                start: start.clone(),
                control: control_pos,
            });
        }
    }

    fn handle_curve_pick_end(
        &mut self,
        terrain_renderer: &TerrainRenderer,
        storage: &RoadStorage,
        road_style_params: &mut RoadStyleParams,
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
            road_style_params.set_to_idle();
            return;
        };

        let end_anchor = self.build_anchor_from_snap(snap);
        let end_pos = snap.world_pos;

        let estimated_length = estimate_bezier_arc_length(start_pos, control, end_pos);
        let segment_count = compute_curve_segment_count(estimated_length);
        let polyline = sample_quadratic_bezier(start_pos, control, end_pos, segment_count);

        let (is_valid, reason) = self.validate_placement(storage, start, &end_anchor);

        let seg_preview = SegmentPreview {
            road_type: road_style_params.road_type().clone(),
            mode: road_style_params.mode(),
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
            lane_count_each_dir: road_style_params.road_type().lanes_each_direction(),
            estimated_length,
        };
        output.push(RoadEditorCommand::PreviewSegment(seg_preview.clone()));

        let node_preview = self.build_node_preview_from_snap(storage, snap);
        output.push(RoadEditorCommand::PreviewNode(node_preview));

        if place_pressed && is_valid {
            let road_cmds = self.commit_road(
                terrain_renderer,
                storage,
                road_style_params,
                start,
                &end_anchor,
                Some(control),
                chunk_id,
                output,
            );
            for cmd in road_cmds {
                output.push(RoadEditorCommand::Road(cmd));
            }
            road_style_params.set_to_idle()
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
        let lane = storage.lane(&lane_id);
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

                for lane_id in node.incoming_lanes() {
                    let lane = storage.lane(lane_id);
                    let dir = lane_direction_at_node(lane, id);
                    in_lanes.push((lane_id.clone(), dir));
                }

                for lane_id in node.outgoing_lanes() {
                    let lane = storage.lane(lane_id);
                    let dir = lane_direction_at_node(lane, id);
                    out_lanes.push((lane_id.clone(), dir));
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
                    let lane = storage.lane(&lane_id);
                    let dir = lane.geometry().points[1] - lane.geometry().points[0];

                    (
                        NodePreviewResult::SplitIntersection,
                        vec![(lane_id.clone(), -dir.normalize())],
                        vec![(lane_id.clone(), dir.normalize())],
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
        let lane = storage.lane(&lane_id);
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
        terrain_renderer: &TerrainRenderer,
        storage: &RoadStorage,
        road_style_params: &RoadStyleParams,
        start: &Anchor,
        end: &Anchor,
        control: Option<Vec3>, // None = straight, Some = curve
        chunk_id: ChunkId,
        output: &mut Vec<RoadEditorCommand>,
    ) -> Vec<RoadCommand> {
        let mut cmds = Vec::new();

        // Resolve anchors
        let Some((start_id, start_pos)) = self.resolve_anchor(
            storage,
            road_style_params,
            start,
            chunk_id,
            &mut cmds,
            output,
        ) else {
            return Vec::new();
        };

        let Some((end_id, end_pos)) =
            self.resolve_anchor(storage, road_style_params, end, chunk_id, &mut cmds, output)
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
            None => make_straight_centerline(terrain_renderer, start_pos, end_pos),
        };

        if centerline.len() < 2 {
            return cmds;
        }

        // Allocate segment (topology only)
        let segment_id = self.allocator.alloc_segment();
        cmds.push(RoadCommand::AddSegment {
            start: start_id,
            end: end_id,
            structure: road_style_params.road_type().structure(),
            chunk_id,
        });

        // Emit lanes from centerline
        self.emit_lanes_from_centerline(
            terrain_renderer,
            &mut cmds,
            road_style_params,
            segment_id,
            start_id,
            end_id,
            &centerline,
            chunk_id,
        );
        push_intersection(
            &mut cmds,
            start_id,
            &start.planned_node,
            road_style_params,
            chunk_id,
        );
        push_intersection(
            &mut cmds,
            end_id,
            &end.planned_node,
            road_style_params,
            chunk_id,
        );

        cmds
    }

    fn resolve_anchor(
        &mut self,
        storage: &RoadStorage,
        road_style_params: &RoadStyleParams,
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
                let result =
                    self.plan_split(road_style_params, storage, *lane_id, *pos, chunk_id)?;
                cmds.extend(result.0);
                Some((result.1, *pos))
            }
        }
    }

    fn plan_split(
        &mut self,
        road_style_params: &RoadStyleParams,
        storage: &RoadStorage,
        lane_id: LaneId,
        split_pos: Vec3,
        chunk_id: ChunkId,
    ) -> Option<(Vec<RoadCommand>, NodeId)> {
        let lane = storage.lane(&lane_id);
        let old_segment_id = lane.segment();
        let old_segment = storage.segment(old_segment_id);

        let a_id = old_segment.start();
        let b_id = old_segment.end();

        let mut cmds = Vec::new();

        cmds.push(RoadCommand::DisableSegment {
            // diables segment and its lanes
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

        for old_lane_id in old_segment.lanes() {
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
                    geometry: geom1, // FIXED: was geom2
                    speed_limit: old_lane.speed_limit(),
                    capacity: old_lane.capacity(),
                    vehicle_mask: old_lane.vehicle_mask(),
                    base_cost: cost1, // FIXED: was cost2
                    chunk_id,
                });

                cmds.push(RoadCommand::AddLane {
                    from: new_node_id,
                    to: a_id,
                    segment: seg1_id,
                    lane_index: old_lane.lane_index(),
                    geometry: geom2, // FIXED: was geom1
                    speed_limit: old_lane.speed_limit(),
                    capacity: old_lane.capacity(),
                    vehicle_mask: old_lane.vehicle_mask(),
                    base_cost: cost2, // FIXED: was cost1
                    chunk_id,
                });
            }
        }

        Some((cmds, new_node_id))
    }

    fn emit_lanes_from_centerline(
        &self,
        terrain_renderer: &TerrainRenderer,
        cmds: &mut Vec<RoadCommand>,
        road_style_params: &RoadStyleParams,
        segment: SegmentId,
        start: NodeId,
        end: NodeId,
        centerline: &[Vec3],
        chunk_id: ChunkId,
    ) {
        let (left_lanes, right_lanes) = road_style_params.road_type().lanes_each_direction();
        let speed = road_style_params.road_type().speed_limit();
        let capacity = road_style_params.road_type().capacity();
        let mask = road_style_params.road_type().vehicle_mask();
        let lane_width = road_style_params.lane_width;

        // Right side lanes (start -> end)
        for i in 0..right_lanes {
            let lane_index = (i as i8) + 1;
            let poly = offset_polyline(terrain_renderer, centerline, lane_index, lane_width);
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
            let mut poly = offset_polyline(terrain_renderer, centerline, lane_index, lane_width);
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

fn make_straight_centerline(
    terrain_renderer: &TerrainRenderer,
    start_pos: Vec3,
    end_pos: Vec3,
) -> Vec<Vec3> {
    let length = (end_pos - start_pos).length();
    let samples = (length / METERS_PER_LANE_POLYLINE_STEP).max(1.0) as usize;

    (0..=samples)
        .map(|i| {
            let t = i as f32 / samples as f32;
            let mut p = start_pos.lerp(end_pos, t);
            let terrain_y = terrain_renderer.get_height_at([p.x, p.z]);
            if p.y < terrain_y {
                p.y = terrain_y;
            }
            p
        })
        .collect()
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
    ((arc_length / METERS_PER_LANE_POLYLINE_STEP).ceil() as usize).clamp(4, 32)
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
pub fn offset_polyline(
    terrain_renderer: &TerrainRenderer,
    center: &[Vec3],
    lane_index: i8,
    lane_width: f32,
) -> Vec<Vec3> {
    let offset = if lane_index < 0 {
        (lane_index as f32 + 0.5) * lane_width
    } else {
        (lane_index as f32 - 0.5) * lane_width
    };
    // println!("Lane Index: {} Lane Width: {} Offset: {}", lane_index, lane_width, offset);
    let mut out = Vec::with_capacity(center.len());

    for i in 0..center.len() {
        let dir = if i + 1 < center.len() {
            center[i + 1] - center[i]
        } else {
            center[i] - center[i - 1]
        };

        let dir_xz = Vec3::new(dir.x, 0.0, dir.z).normalize();
        let right = Vec3::new(-dir_xz.z, 0.0, dir_xz.x);
        let mut final_point = center[i] + right * offset;
        final_point.y = final_point
            .y
            .max(terrain_renderer.get_height_at([final_point.x, final_point.z]))
            + CLEARANCE;
        out.push(final_point);
    }

    out
}
#[derive(Clone, Debug)]
pub struct IntersectionBuildParams {
    /// Degrees: lanes with similar "arm direction" are grouped into the same approach arm.
    pub arm_merge_angle_deg: f32,
    /// Degrees: outgoing arm considered "straight" if within this angle of incoming heading.
    pub straight_angle_deg: f32,
    /// Bezier samples for NodeLane geometry.
    pub turn_samples: usize,
    /// If true: dedicate rightmost lane to right turn and leftmost to left turn (when possible).
    pub dedicate_turn_lanes: bool,
    pub max_turn_angle: f32,    // radians, eg 160° = 2.79 (kills u-turns)
    pub min_turn_radius_m: f32, // eg 6.0
    /// The radius of the "empty intersection area" around the node.
    /// Incoming lanes are trimmed so their end lies on this radius,
    /// outgoing lanes are trimmed so their start lies on this radius.
    pub clearance_radius_m: f32,
}

impl IntersectionBuildParams {
    pub fn from_style(style: &RoadStyleParams) -> Self {
        // A decent default: enough to clear sidewalks + a bit of lane length for the curve start.
        let r = style.sidewalk_width + style.lane_width * 1.8 + style.median_width;

        Self {
            arm_merge_angle_deg: 20.0,
            straight_angle_deg: 25.0,
            turn_samples: 12,
            dedicate_turn_lanes: true,
            max_turn_angle: 3.3,
            min_turn_radius_m: 1.0,
            clearance_radius_m: r.clamp(style.lane_width * 1.5, style.lane_width * 8.0),
        }
    }
}
impl Default for IntersectionBuildParams {
    fn default() -> Self {
        Self {
            arm_merge_angle_deg: 20.0,
            straight_angle_deg: 25.0,
            turn_samples: 12,
            dedicate_turn_lanes: true,
            max_turn_angle: 3.4,
            min_turn_radius_m: 2.0,
            clearance_radius_m: 15.0,
        }
    }
}
#[derive(Clone, Copy, Debug)]
enum Move {
    Right,
    Straight,
    Left,
}
pub fn build_intersection_at_node(
    storage: &mut RoadStorage,
    node_id: NodeId,
    params: &IntersectionBuildParams,
    clear: bool,
) {
    if clear {
        let dynamic_clearance = params.clearance_radius_m;
        carve_intersection_clearance(storage, node_id, dynamic_clearance);
    }
    // Clear node lanes (same as RoadCommand::ClearNodeLanes)
    storage.node_mut(node_id).clear_node_lanes();

    let node_pos = {
        let Some(n) = storage.node(node_id) else {
            return;
        };
        Vec3::new(n.x(), n.y(), n.z())
    };

    // Clone lane lists to avoid borrow issues
    let (incoming_ids, outgoing_ids) = {
        let Some(n) = storage.node(node_id) else {
            return;
        };
        (n.incoming_lanes(), n.outgoing_lanes())
    };
    let up = Vec3::new(0.0, 1.0, 0.0);
    let mut connections: Vec<(&LaneId, &LaneId)> = Vec::new();
    for incoming_id in incoming_ids {
        for outgoing_id in outgoing_ids {
            let incoming_lane = storage.lane(incoming_id);
            let outgoing_lane = storage.lane(outgoing_id);
            if !incoming_lane.is_enabled() || !outgoing_lane.is_enabled() {
                continue;
            }

            connections.push((incoming_id, outgoing_id));
        }
    }
    let mut node_lanes: Vec<NodeLane> = Vec::new();
    for (idx, (incoming_id, outgoing_id)) in connections.iter().enumerate() {
        let incoming_lane = storage.lane(incoming_id);
        let outgoing_lane = storage.lane(outgoing_id);

        let in_poly = incoming_lane.polyline();
        let out_poly = outgoing_lane.polyline();

        let incoming_pos_before = in_poly[in_poly.len() - 2]; // Second-to-Last Vec3 Point of the incoming lane
        let incoming_pos = in_poly[in_poly.len() - 1]; // Last Vec3 Point of the incoming lane
        let incoming_dir = (incoming_pos - incoming_pos_before).normalize(); // Direction from the incoming lane

        let outgoing_pos = out_poly[0]; // Second Vec3 Point of the outgoing lane
        let outgoing_pos_after = out_poly[1]; // First Vec3 Point of the outgoing lane
        let outgoing_dir = (outgoing_pos_after - outgoing_pos).normalize(); // Direction to the outgoing lane

        let angle = incoming_dir.dot(outgoing_dir).clamp(-1.0, 1.0);
        let turn_angle = angle.acos(); // radians, 0 = straight, PI = U-turn

        if turn_angle > params.max_turn_angle {
            println!("hi");
            continue; // u-turn or near u-turn
        }
        let chord = incoming_pos.distance(outgoing_pos);
        let turn_radius = if turn_angle < 0.01 {
            f32::INFINITY
        } else {
            chord / (2.0 * (turn_angle * 0.5).sin())
        };

        if turn_radius < params.min_turn_radius_m {
            continue; // too sharp, no space
        }
        let sharpness = (turn_angle / params.max_turn_angle).clamp(0.0, 1.0);
        let geometry =
            generate_turn_geometry(incoming_pos, incoming_dir, outgoing_pos, outgoing_dir, 12);

        let length_cost = geometry.total_len; // or sum of segments
        let curvature_cost = (params.min_turn_radius_m / turn_radius).powf(1.5);
        let base_cost = length_cost * 0.1 + turn_angle * 2.0 + curvature_cost * 20.0;
        let dynamic_cost = 0.0;
        let node_lane = NodeLane::new(
            idx as NodeLaneId,
            vec![LaneRef::Segment(**incoming_id, 0 as PolyIdx)],
            vec![LaneRef::Segment(
                **outgoing_id,
                (geometry.points.len() - 1) as PolyIdx,
            )],
            geometry,
            base_cost,
            dynamic_cost,
            50.0,
            0,
        );
        node_lanes.push(node_lane);
    }

    storage.node_mut(node_id).add_node_lanes(node_lanes);
}

/// Generates smooth turn geometry using a cubic Bezier curve.
fn generate_turn_geometry(
    start: Vec3,
    start_dir: Vec3,
    end: Vec3,
    end_dir: Vec3,
    samples: usize,
) -> LaneGeometry {
    let dist = start.distance(end);

    // Handle degenerate case where start ≈ end
    if dist < 0.001 {
        return LaneGeometry::from_polyline(vec![start, end]);
    }

    // Control point distance scales with turn length
    let control_len = (dist * 0.4).max(0.5);

    // Cubic bezier control points
    let ctrl1 = start + start_dir * control_len;
    let ctrl2 = end - end_dir * control_len;

    let n = samples.clamp(2, 32);
    let mut points = Vec::with_capacity(n);

    for i in 0..n {
        let t = i as f32 / (n - 1) as f32;
        points.push(bezier3(start, ctrl1, ctrl2, end, t));
    }

    LaneGeometry::from_polyline(points)
}
fn seg_sphere_intersection_t(a: Vec3, b: Vec3, center: Vec3, r: f32) -> Option<f32> {
    let d = b - a;
    let f = a - center;

    let aa = d.dot(d);
    if aa < 1e-8 {
        return None;
    }

    let bb = 2.0 * f.dot(d);
    let cc = f.dot(f) - r * r;

    let disc = bb * bb - 4.0 * aa * cc;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();

    let t1 = (-bb - s) / (2.0 * aa);
    let t2 = (-bb + s) / (2.0 * aa);

    // We want a valid intersection on the segment.
    let in1 = (0.0..=1.0).contains(&t1);
    let in2 = (0.0..=1.0).contains(&t2);

    match (in1, in2) {
        (true, true) => Some(t1.min(t2)),
        (true, false) => Some(t1),
        (false, true) => Some(t2),
        (false, false) => None,
    }
}
fn trim_polyline_end_to_radius(points: &[Vec3], center: Vec3, r: f32) -> Option<Vec<Vec3>> {
    if points.len() < 2 {
        return None;
    }

    let eps = 0.05;
    let end_dist = points[points.len() - 1].distance(center);

    // Already at/near radius or outside => don't shrink further (idempotent).
    if end_dist >= r - eps {
        return None;
    }

    // Walk backwards to find segment crossing radius (outside -> inside).
    for i in (0..points.len() - 1).rev() {
        let a = points[i];
        let b = points[i + 1];
        let da = a.distance(center);
        let db = b.distance(center);

        if da >= r && db <= r {
            if let Some(t) = seg_sphere_intersection_t(a, b, center, r) {
                let p = a + (b - a) * t;
                let mut out = Vec::with_capacity(i + 2);
                out.extend_from_slice(&points[..=i]);
                out.push(p);
                if out.len() >= 2 {
                    return Some(out);
                }
                return None;
            }
        }
    }

    None
}
fn trim_polyline_start_to_radius(points: &[Vec3], center: Vec3, r: f32) -> Option<Vec<Vec3>> {
    if points.len() < 2 {
        return None;
    }

    let eps = 0.05;
    let start_dist = points[0].distance(center);

    // Already at/near radius or outside => don't shrink further (idempotent).
    if start_dist >= r - eps {
        return None;
    }

    // Walk forward to find segment crossing radius (inside -> outside).
    for i in 0..points.len() - 1 {
        let a = points[i];
        let b = points[i + 1];
        let da = a.distance(center);
        let db = b.distance(center);

        if da <= r && db >= r {
            if let Some(t) = seg_sphere_intersection_t(a, b, center, r) {
                let p = a + (b - a) * t;
                let mut out = Vec::with_capacity(points.len() - i);
                out.push(p);
                out.extend_from_slice(&points[i + 1..]);
                if out.len() >= 2 {
                    return Some(out);
                }
                return None;
            }
        }
    }

    None
}
fn carve_intersection_clearance(storage: &mut RoadStorage, node_id: NodeId, r: f32) {
    // Pull everything we need from the node as OWNED values (no references kept).
    let (node_pos, incoming_ids, outgoing_ids) = {
        let Some(n) = storage.node(node_id) else {
            return;
        };

        let pos = Vec3::new(n.x(), n.y(), n.z());

        // IMPORTANT: copy LaneId values out (no Vec<&LaneId>)
        let incoming: Vec<LaneId> = n.incoming_lanes().iter().copied().collect();
        let outgoing: Vec<LaneId> = n.outgoing_lanes().iter().copied().collect();

        (pos, incoming, outgoing)
    };

    // Collect edits first (immutable reads only)
    let mut edits: Vec<(LaneId, LaneGeometry, f32)> = Vec::new(); // (lane_id, new_geom, new_base_cost)

    // Incoming: trim END
    for lane_id in incoming_ids {
        let (maybe_new_geom, maybe_new_cost) = {
            let lane = storage.lane(&lane_id);
            if !lane.is_enabled() {
                (None, None)
            } else {
                let old = lane.geometry();
                let old_len = old.total_len.max(0.001);
                let old_cost = lane.base_cost();

                let new_pts = trim_polyline_end_to_radius(&old.points, node_pos, r);
                if let Some(pts) = new_pts {
                    let new_geom = LaneGeometry::from_polyline(pts);
                    let ratio = (new_geom.total_len / old_len).clamp(0.1, 1.0);
                    (Some(new_geom), Some(old_cost * ratio))
                } else {
                    (None, None)
                }
            }
        };

        if let (Some(new_geom), Some(new_cost)) = (maybe_new_geom, maybe_new_cost) {
            edits.push((lane_id, new_geom, new_cost));
        }
    }

    // Outgoing: trim START
    for lane_id in outgoing_ids {
        let (maybe_new_geom, maybe_new_cost) = {
            let lane = storage.lane(&lane_id);
            if !lane.is_enabled() {
                (None, None)
            } else {
                let old = lane.geometry();
                let old_len = old.total_len.max(0.001);
                let old_cost = lane.base_cost();

                let new_pts = trim_polyline_start_to_radius(&old.points, node_pos, r);
                if let Some(pts) = new_pts {
                    let new_geom = LaneGeometry::from_polyline(pts);
                    let ratio = (new_geom.total_len / old_len).clamp(0.1, 1.0);
                    (Some(new_geom), Some(old_cost * ratio))
                } else {
                    (None, None)
                }
            }
        };

        if let (Some(new_geom), Some(new_cost)) = (maybe_new_geom, maybe_new_cost) {
            edits.push((lane_id, new_geom, new_cost));
        }
    }

    // Apply edits (mutable borrows only)
    for (lane_id, new_geom, new_cost) in edits {
        let lane = storage.lane_mut(lane_id);
        lane.replace_geometry(new_geom);
        lane.replace_base_cost(new_cost);
    }
}
fn push_intersection(
    cmds: &mut Vec<RoadCommand>,
    node_id: NodeId,
    planned: &PlannedNode,
    road_style_params: &RoadStyleParams,
    chunk_id: ChunkId,
) {
    let clear = !matches!(planned, PlannedNode::New { .. });

    cmds.push(RoadCommand::MakeIntersection {
        node_id,
        params: IntersectionBuildParams::from_style(road_style_params),
        chunk_id,
        clear,
    });
}
