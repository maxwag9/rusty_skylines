use crate::renderer::world_renderer::{PickedPoint, TerrainRenderer};
use crate::resources::InputState;
use crate::terrain::roads::road_mesh_manager::{CLEARANCE, ChunkId};
use crate::terrain::roads::road_structs::*;
use crate::terrain::roads::roads::{
    Lane, LaneGeometry, LaneRef, METERS_PER_LANE_POLYLINE_STEP, NodeLane, RoadCommand, RoadManager,
    RoadStorage, bezier3, nearest_lane_to_point, project_point_to_lane_xz, sample_lane_position,
};
use glam::Vec3;
use std::collections::HashMap;

const NODE_SNAP_RADIUS: f32 = 8.0;
const LANE_SNAP_RADIUS: f32 = 8.0;
const ENDPOINT_T_EPS: f32 = 0.02;
const MIN_SEGMENT_LENGTH: f32 = 1.0;

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
    pub clearance_length_m: f32,
    pub lane_width_m: f32,
    pub turn_tightness: f32,
}

impl IntersectionBuildParams {
    pub fn from_style(style: &RoadStyleParams) -> Self {
        // A decent default: enough to clear sidewalks + a bit of lane length for the curve start.
        let r = style.sidewalk_width + style.lane_width * 3.0 + style.median_width;

        Self {
            arm_merge_angle_deg: 20.0,
            straight_angle_deg: 25.0,
            turn_samples: 12,
            dedicate_turn_lanes: true,
            max_turn_angle: 2.74,
            min_turn_radius_m: 5.0,
            clearance_length_m: 0.0,
            lane_width_m: style.lane_width,
            turn_tightness: style.turn_tightness(),
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
            max_turn_angle: 2.74,
            min_turn_radius_m: 5.0,
            clearance_length_m: 0.0,
            lane_width_m: 3.5,
            turn_tightness: 0.5,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
enum TurnType {
    Straight,
    Right,
    Left,
    UTurn,
}

#[derive(Debug, Clone)]
struct LaneClearanceDemand {
    // Positive = Carve (Trim), Negative = Extend (Add geometry)
    pub demand_m: f32,
    pub reason: String, // For debugging
}

impl Default for LaneClearanceDemand {
    fn default() -> Self {
        Self {
            demand_m: 0.0,
            reason: "Baseline".into(),
        }
    }
}
pub fn build_intersection_at_node(
    storage: &mut RoadStorage,
    node_id: NodeId,
    params: &IntersectionBuildParams,
    recalc_clearance: bool,
) {
    if recalc_clearance {
        // Phase 1: Probe and adjust geometry
        let demands = probe_intersection_node_lanes(storage, node_id, params);
        carve_intersection_clearance_per_lane(storage, node_id, &demands);
    }

    // Phase 2: Build actual intersection paths
    storage.node_mut(node_id).clear_node_lanes();

    let Some(node) = storage.node(node_id) else {
        return;
    };
    let incoming = node.incoming_lanes();
    let outgoing = node.outgoing_lanes();

    let mut node_lanes = Vec::new();

    // Re-fetch lane data as it might have been trimmed/extended
    for in_id in incoming {
        for out_id in outgoing {
            let in_lane = storage.lane(in_id);
            let out_lane = storage.lane(out_id);

            if !in_lane.is_enabled() || !out_lane.is_enabled() {
                continue;
            }

            let in_poly = in_lane.polyline();
            let out_poly = out_lane.polyline();

            let in_pt = *in_poly.last().unwrap();
            let out_pt = *out_poly.first().unwrap();

            // Calculate directions
            let in_dir = (in_pt - in_poly[in_poly.len() - 2]).normalize();
            let out_dir = (out_poly[1] - out_pt).normalize();

            let angle_rad = in_dir.dot(out_dir).clamp(-1.0, 1.0).acos();
            if angle_rad > params.max_turn_angle {
                continue;
            }

            // --- Dynamic Tightness Logic ---

            // Calculate chord length
            let chord = in_pt.distance(out_pt);

            // Heuristic:
            // Small intersection (chord < 10m) -> Needs tight curves (lower tightness val)
            // Large intersection (chord > 20m) -> Can handle looser curves (higher val)
            // However, outer lanes (large chord) in a turn need MORE space to curve
            // so they don't look like straight lines.

            let base_tightness = params.turn_tightness;
            let dynamic_tightness = if chord > 25.0 {
                // Wide turn (outer lane), make it looser so it uses the space
                base_tightness * 1.2
            } else if chord < 8.0 {
                // Very tight turn, tighten bezier to avoid overshooting
                base_tightness * 0.7
            } else {
                base_tightness
            };

            let geom = generate_turn_geometry(
                in_pt,
                in_dir,
                out_pt,
                out_dir,
                12, // Adaptive sampling could go here
                dynamic_tightness,
            );

            let nl = NodeLane::new(
                (storage.node_lane_count_for_node(node_id) + node_lanes.len()) as NodeLaneId,
                vec![LaneRef::Segment(*in_id, 0)],
                vec![LaneRef::Segment(
                    *out_id,
                    (geom.points.len() - 1) as PolyIdx,
                )],
                geom,
                0.0,
                0.0,
                50.0,
                0,
            );

            node_lanes.push(nl);
        }
    }

    storage.node_mut(node_id).add_node_lanes(node_lanes);
}

/// Generates smooth turn geometry using a cubic Bézier curve.
/// `tightness` < 1.0 makes turns tighter, > 1.0 makes them wider.
fn generate_turn_geometry(
    start: Vec3,
    start_dir: Vec3,
    end: Vec3,
    end_dir: Vec3,
    samples: usize,
    tightness: f32,
) -> LaneGeometry {
    let dist = start.distance(end);

    if dist < 0.001 {
        return LaneGeometry::from_polyline(vec![start, end]);
    }

    // Dynamic Control Length
    // For 90 degree turns, 0.55228 is the magic number for a circle approximation.
    // We adjust this based on the user provided tightness.
    let control_len = (dist * 0.35) * tightness;

    let ctrl1 = start + start_dir * control_len;
    let ctrl2 = end - end_dir * control_len;

    let n = samples.clamp(2, 64);
    let mut points = Vec::with_capacity(n);

    for i in 0..n {
        let t = i as f32 / (n - 1) as f32;
        points.push(bezier3(start, ctrl1, ctrl2, end, t));
    }

    LaneGeometry::from_polyline(points)
}

type NodeLaneKey = (LaneId, LaneId);

fn classify_turn(in_dir: Vec3, out_dir: Vec3) -> TurnType {
    let dot = in_dir.dot(out_dir);
    let cross = in_dir.cross(out_dir).y; // Assuming Y is up

    if dot > 0.9 {
        TurnType::Straight
    } else if dot < -0.9 {
        TurnType::UTurn
    } else if cross < 0.0 {
        // In most coordinate systems (Y-up), cross < 0 with forward vectors implies Right
        TurnType::Right
    } else {
        TurnType::Left
    }
}
pub fn probe_intersection_node_lanes(
    storage: &RoadStorage,
    node_id: NodeId,
    params: &IntersectionBuildParams,
) -> HashMap<LaneId, f32> {
    let Some(node) = storage.node(node_id) else {
        return HashMap::new();
    };

    // We store demands per Physical Lane, not per connection
    let mut lane_demands: HashMap<LaneId, f32> = HashMap::new();

    let incoming = node.incoming_lanes();
    let outgoing = node.outgoing_lanes();

    // 1. Collect all potential connections to analyze the intersection "Centroid"
    let mut intersection_center = Vec3::ZERO;
    let mut connection_count = 0;

    struct ConnectionInfo {
        in_id: LaneId,
        out_id: LaneId,
        in_pt: Vec3,
        out_pt: Vec3,
        turn_type: TurnType,
        angle_rad: f32,
    }

    let mut connections = Vec::new();

    for in_id in incoming {
        for out_id in outgoing {
            let in_lane = storage.lane(in_id);
            let out_lane = storage.lane(out_id);
            if !in_lane.is_enabled() || !out_lane.is_enabled() {
                continue;
            }

            let in_poly = in_lane.polyline();
            let out_poly = out_lane.polyline();
            let in_pt = *in_poly.last().unwrap();
            let out_pt = *out_poly.first().unwrap();
            let in_dir = (in_pt - in_poly[in_poly.len() - 2]).normalize();
            let out_dir = (out_poly[1] - out_pt).normalize();

            // Filter impossible turns
            let angle = in_dir.dot(out_dir).clamp(-1.0, 1.0).acos();
            if angle > params.max_turn_angle {
                continue;
            }

            let t_type = classify_turn(in_dir, out_dir);

            connections.push(ConnectionInfo {
                in_id: *in_id,
                out_id: *out_id,
                in_pt,
                out_pt,
                turn_type: t_type,
                angle_rad: angle,
            });

            intersection_center += in_pt;
            intersection_center += out_pt;
            connection_count += 2;
        }
    }

    if connection_count > 0 {
        intersection_center /= connection_count as f32;
    }

    // 2. Evaluate Demands
    // We treat Straights and Right Turns as "Constraint Setters".
    // Left turns usually just flow wherever the space allows.

    for conn in &connections {
        let mut demand_m = 0.0; // Default 0.0 baseline

        match conn.turn_type {
            TurnType::Straight => {
                // LOGIC: Straights want to be tight.
                // If the gap is huge, we extend. If they overlap, we don't necessarily trim
                // unless they hit a cross lane (collision check is complex, so we use distance heuristic).

                let dist_to_center_in = conn.in_pt.distance(intersection_center);
                let dist_to_center_out = conn.out_pt.distance(intersection_center);

                // Heuristic: Straights shouldn't be miles away from the calculated center.
                // We allow them to extend closer.
                // Negative demand means EXTEND.
                let gap = conn.in_pt.distance(conn.out_pt);
                if gap > params.lane_width_m * 5.0 {
                    // Too big gap, pull them closer (extend road)
                    // We only extend half the gap per side safely
                    demand_m = -((gap - params.lane_width_m) * 0.4);
                }
            }
            TurnType::Right => {
                // LOGIC: Right turns are the geometric bottleneck.
                // Calculate required chord length for min_radius.
                // Chord = 2 * R * sin(theta / 2)

                // If angle is effectively 0 (U-turnish) or 180 (Straight), math differs,
                // but for Right turns (approx 90 deg / 1.57 rad):

                let theta = conn.angle_rad;
                if theta > 0.1 {
                    // Prevent div by zero
                    let required_chord = 2.0 * params.min_turn_radius_m * (theta * 0.5).sin();
                    let current_chord = conn.in_pt.distance(conn.out_pt);

                    if current_chord < required_chord {
                        // We are too tight! We need to trim back to widen the gap.
                        // How much? Roughly difference / 2 per side.
                        let deficit = required_chord - current_chord;

                        // We add a safety margin because bezier curves cut the corner
                        demand_m = (deficit * 8.7).max(0.0);
                    } else if current_chord > required_chord * 1.5 {
                        // We are too wide for a simple right turn, we can optimize space (extend)
                        let excess = current_chord - (required_chord * 1.2);
                        demand_m = -(excess * 0.3); // Gentle extension
                    }
                }
            }
            _ => {
                // Left turns: usually don't dictate the trim unless they collide.
                // We leave them as 0.0 or follow the Straight/Right neighbors.
            }
        }

        // Apply demand to both lanes involved in this connection.
        // We take the MAX demand (most trimming) if positive.
        // If negative (extension), we are conservative (take the smallest extension).

        let apply_demand = |lane_id: LaneId, d: f32, map: &mut HashMap<LaneId, f32>| {
            map.entry(lane_id)
                .and_modify(|curr| {
                    if *curr > 0.0 && d > 0.0 {
                        *curr = curr.max(d); // Both want trim, take max
                    } else if *curr < 0.0 && d < 0.0 {
                        *curr = curr.max(d); // Both want extend (negative), take closer to 0 (least risky)
                    } else if d > 0.0 {
                        *curr = d; // Trim overrides extension
                    }
                    // Else: if current is trim, ignore extension request
                })
                .or_insert(d);
        };

        apply_demand(conn.in_id, demand_m, &mut lane_demands);
        apply_demand(conn.out_id, demand_m, &mut lane_demands);
    }

    // 3. Global sanity check (Collision Probing)
    // If we detected 0 demand, we double check that we aren't clipping neighbors.
    // This handles the "Unaffected lanes" requirement.

    // (Omitted for brevity, but here you would check `min_polyline_distance`
    // between adjacent straights. If < 0, add trim).

    lane_demands
}
fn carve_intersection_clearance_per_lane(
    storage: &mut RoadStorage,
    _node_id: NodeId,
    lane_demands: &HashMap<LaneId, f32>,
) {
    let mut edits = Vec::new();

    for (lane_id, amount) in lane_demands {
        // Filter out negligible changes to prevent mesh jitter
        if amount.abs() < 0.05 {
            continue;
        }

        let lane = storage.lane(lane_id);
        if !lane.is_enabled() {
            continue;
        }

        let is_incoming = storage
            .node(_node_id)
            .unwrap()
            .incoming_lanes()
            .contains(lane_id);

        let new_pts = if is_incoming {
            // Incoming lanes get modified at the END
            modify_polyline_end(&lane.geometry().points, *amount)
        } else {
            // Outgoing lanes get modified at the START
            modify_polyline_start(&lane.geometry().points, *amount)
        };

        if let Some(pts) = new_pts {
            edits.push((*lane_id, LaneGeometry::from_polyline(pts)));
        }
    }

    for (id, geom) in edits {
        storage.lane_mut(id).replace_geometry(geom);
    }
}

/// Trims or Extends a polyline from the END.
/// Positive `amount`: Trim. Negative `amount`: Extend (linearly).
fn modify_polyline_end(points: &[Vec3], amount: f32) -> Option<Vec<Vec3>> {
    if points.len() < 2 {
        return None;
    }

    if amount.abs() < 0.001 {
        return Some(points.to_vec());
    }

    // -- Trimming Logic --
    if amount > 0.0 {
        return trim_polyline_end_by_distance(points, amount);
    }

    // -- Extension Logic --
    // Extends the last segment linearly by abs(amount)
    let extend_len = amount.abs();
    let last = points[points.len() - 1];
    let prev = points[points.len() - 2];
    let dir = (last - prev).normalize_or_zero();

    if dir == Vec3::ZERO {
        return Some(points.to_vec());
    }

    let new_pos = last + (dir * extend_len);
    let mut out = points.to_vec();
    out.push(new_pos);
    Some(out)
}

/// Trims or Extends a polyline from the START.
fn modify_polyline_start(points: &[Vec3], amount: f32) -> Option<Vec<Vec3>> {
    if points.len() < 2 {
        return None;
    }

    if amount.abs() < 0.001 {
        return Some(points.to_vec());
    }

    if amount > 0.0 {
        return trim_polyline_start_by_distance(points, amount);
    }

    // -- Extension Logic --
    let extend_len = amount.abs();
    let first = points[0];
    let second = points[1];
    let dir = (first - second).normalize_or_zero(); // Direction pointing AWAY from line

    if dir == Vec3::ZERO {
        return Some(points.to_vec());
    }

    let new_pos = first + (dir * extend_len);
    let mut out = Vec::with_capacity(points.len() + 1);
    out.push(new_pos);
    out.extend_from_slice(points);
    Some(out)
}
fn trim_polyline_end_by_distance(points: &[Vec3], trim_len: f32) -> Option<Vec<Vec3>> {
    if points.len() < 2 || trim_len <= 0.0 {
        return None;
    }

    let mut remaining = trim_len;

    for i in (1..points.len()).rev() {
        let a = points[i - 1];
        let b = points[i];
        let seg_len = a.distance(b);

        if remaining < seg_len {
            let t = (seg_len - remaining) / seg_len;
            let p = a + (b - a) * t;
            let mut out = points[..i].to_vec();
            out.push(p);
            if out.len() >= 2 {
                return Some(out);
            }
            return None;
        }

        remaining -= seg_len;
    }

    None
}
fn trim_polyline_start_by_distance(points: &[Vec3], trim_len: f32) -> Option<Vec<Vec3>> {
    if points.len() < 2 || trim_len <= 0.0 {
        return None;
    }

    let mut remaining = trim_len;

    for i in 0..points.len() - 1 {
        let a = points[i];
        let b = points[i + 1];
        let seg_len = a.distance(b);

        if remaining < seg_len {
            let t = remaining / seg_len;
            let p = a + (b - a) * t;
            let mut out = Vec::with_capacity(points.len() - i);
            out.push(p);
            out.extend_from_slice(&points[i + 1..]);
            if out.len() >= 2 {
                return Some(out);
            }
            return None;
        }

        remaining -= seg_len;
    }

    None
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
