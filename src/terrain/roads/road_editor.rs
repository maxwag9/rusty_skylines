use crate::renderer::gizmo::{DEBUG_DRAW_DURATION, Gizmo};
use crate::renderer::world_renderer::{PickedPoint, TerrainRenderer};
use crate::resources::InputState;
use crate::terrain::roads::road_helpers::*;
use crate::terrain::roads::road_mesh_manager::{CLEARANCE, ChunkId};
use crate::terrain::roads::road_structs::*;
use crate::terrain::roads::roads::{
    Lane, LaneGeometry, LaneRef, METERS_PER_LANE_POLYLINE_STEP, NodeLane, RoadCommand, RoadManager,
    RoadStorage, Segment, bezier3, nearest_lane_to_point, project_point_to_lane_xz,
    sample_lane_position,
};
use glam::{Vec2, Vec3, Vec3Swizzles};
use std::collections::{HashMap, HashSet};

const NODE_SNAP_RADIUS: f32 = 8.0;
const LANE_SNAP_RADIUS: f32 = 8.0;
const ENDPOINT_T_EPS: f32 = 0.02;
const MIN_SEGMENT_LENGTH: f32 = 1.0;
const CROSSING_SNAP_TO_NODE_RADIUS: f32 = 20.0;
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

        // Find crossings for preview
        let crossings = self.find_all_crossings(
            storage,
            terrain_renderer,
            start_pos,
            end_pos,
            None,
            start,
            &end_anchor,
        );

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
            crossing_count: crossings.len(),
        };
        output.push(RoadEditorCommand::PreviewSegment(seg_preview.clone()));

        let node_preview = self.build_node_preview_from_snap(storage, snap);
        output.push(RoadEditorCommand::PreviewNode(node_preview));

        // Preview crossing points
        for crossing in crossings {
            output.push(RoadEditorCommand::PreviewCrossing(crossing));
        }

        if place_pressed && is_valid {
            let road_cmds = self.commit_road_with_crossings(
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
            crossing_count: 0,
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

        // Find crossings for preview
        let crossings = self.find_all_crossings(
            storage,
            terrain_renderer,
            start_pos,
            end_pos,
            Some(control),
            start,
            &end_anchor,
        );

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
            crossing_count: crossings.len(),
        };
        output.push(RoadEditorCommand::PreviewSegment(seg_preview.clone()));

        let node_preview = self.build_node_preview_from_snap(storage, snap);
        output.push(RoadEditorCommand::PreviewNode(node_preview));

        // Preview crossing points
        for crossing in crossings {
            output.push(RoadEditorCommand::PreviewCrossing(crossing));
        }

        if place_pressed && is_valid {
            let road_cmds = self.commit_road_with_crossings(
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
    // ==================== CROSSING DETECTION ====================

    /// Find all points where the new road would cross existing infrastructure
    fn find_all_crossings(
        &self,
        storage: &RoadStorage,
        terrain_renderer: &TerrainRenderer,
        start_pos: Vec3,
        end_pos: Vec3,
        control: Option<Vec3>,
        start_anchor: &Anchor,
        end_anchor: &Anchor,
    ) -> Vec<CrossingPoint> {
        let mut crossings = Vec::new();
        let mut crossed_segments: HashSet<SegmentId> = HashSet::new();

        // Build test polyline for intersection testing
        let test_polyline = match control {
            Some(c) => {
                let est_len = estimate_bezier_arc_length(start_pos, c, end_pos);
                let samples = compute_curve_segment_count(est_len).max(20);
                sample_quadratic_bezier(start_pos, c, end_pos, samples)
            }
            None => vec![start_pos, end_pos],
        };

        // Get segments that are excluded (already being split at start/end)
        let excluded_segments = self.get_excluded_segments(storage, start_anchor, end_anchor);

        // Check lane crossings - only one crossing per segment
        for (lane_id, _) in storage.iter_enabled_lanes() {
            let lane = storage.lane(&lane_id);
            let seg_id = lane.segment();

            // Skip if we've already found a crossing on this segment
            if crossed_segments.contains(&seg_id) {
                continue;
            }

            // Skip excluded segments (being split at start/end)
            if excluded_segments.contains(&seg_id) {
                continue;
            }

            if let Some(crossing) = self.find_lane_crossing_point(
                storage,
                terrain_renderer,
                &test_polyline,
                start_pos,
                end_pos,
                control,
                &lane_id,
            ) {
                let lane = storage.lane(&lane_id);
                let seg_id = lane.segment();
                let segment = storage.segment(seg_id);

                let start_node = storage.node(segment.start).unwrap();
                let start_node_pos = Vec3::new(start_node.x(), start_node.y(), start_node.z());

                let end_node = storage.node(segment.end).unwrap();
                let end_node_pos = Vec3::new(end_node.x(), end_node.y(), end_node.z());

                let dist_to_start = (crossing.world_pos - start_node_pos).length();
                let dist_to_end = (crossing.world_pos - end_node_pos).length();

                if dist_to_start < CROSSING_SNAP_TO_NODE_RADIUS
                    || dist_to_end < CROSSING_SNAP_TO_NODE_RADIUS
                {
                    // Snap to the closest endpoint node
                    let (closest_node_id, closest_pos) = if dist_to_start <= dist_to_end {
                        (segment.start, start_node_pos)
                    } else {
                        (segment.end, end_node_pos)
                    };

                    // Project the node position back onto our new road to get an accurate t
                    if let Some((new_t, _proj_dist)) =
                        self.project_point_to_path(start_pos, end_pos, control, closest_pos)
                    {
                        // Only add if the new t is still in valid range (should almost always be true)
                        if new_t > ENDPOINT_T_EPS && new_t < 1.0 - ENDPOINT_T_EPS {
                            crossings.push(CrossingPoint {
                                t: new_t,
                                world_pos: closest_pos,
                                kind: CrossingKind::ExistingNode(closest_node_id),
                            });
                            crossed_segments.insert(seg_id);
                            continue; // skip adding a lane crossing split
                        }
                    }
                }
                // Check if not too close to our road's endpoints
                if crossing.t > ENDPOINT_T_EPS && crossing.t < 1.0 - ENDPOINT_T_EPS {
                    crossed_segments.insert(seg_id);
                    crossings.push(crossing);
                }
            }
        }

        // Check for existing nodes close to our path (not start/end nodes)
        for (node_id, node) in storage.iter_enabled_nodes() {
            // Skip nodes that are our start or end
            if self.is_node_in_anchor(node_id, start_anchor)
                || self.is_node_in_anchor(node_id, end_anchor)
            {
                continue;
            }

            let node_pos = Vec3::new(node.x(), node.y(), node.z());
            if let Some((t, dist)) =
                self.project_point_to_path(start_pos, end_pos, control, node_pos)
            {
                if dist < NODE_SNAP_RADIUS && t > ENDPOINT_T_EPS && t < 1.0 - ENDPOINT_T_EPS {
                    crossings.push(CrossingPoint {
                        t,
                        world_pos: node_pos,
                        kind: CrossingKind::ExistingNode(node_id),
                    });
                }
            }
        }

        // Sort by t (position along our road)
        crossings.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));

        // Deduplicate crossings that are too close together
        self.deduplicate_crossings(crossings)
    }

    fn get_excluded_segments(
        &self,
        storage: &RoadStorage,
        start_anchor: &Anchor,
        end_anchor: &Anchor,
    ) -> HashSet<SegmentId> {
        let mut excluded = HashSet::new();

        if let PlannedNode::Split { lane_id, .. } = &start_anchor.planned_node {
            let lane = storage.lane(lane_id);
            excluded.insert(lane.segment());
        }

        if let PlannedNode::Split { lane_id, .. } = &end_anchor.planned_node {
            let lane = storage.lane(lane_id);
            excluded.insert(lane.segment());
        }

        excluded
    }

    fn is_node_in_anchor(&self, node_id: NodeId, anchor: &Anchor) -> bool {
        match &anchor.planned_node {
            PlannedNode::Existing(id) => *id == node_id,
            _ => false,
        }
    }

    fn find_lane_crossing_point(
        &self,
        storage: &RoadStorage,
        terrain_renderer: &TerrainRenderer,
        test_polyline: &[Vec3],
        start_pos: Vec3,
        end_pos: Vec3,
        control: Option<Vec3>,
        lane_id: &LaneId,
    ) -> Option<CrossingPoint> {
        let lane = storage.lane(lane_id);
        let lane_points = &lane.geometry().points;

        // For each segment in our test polyline
        for i in 0..test_polyline.len() - 1 {
            let p1 = test_polyline[i];
            let p2 = test_polyline[i + 1];

            // Check against each segment of the lane
            for j in 0..lane_points.len() - 1 {
                let q1 = lane_points[j];
                let q2 = lane_points[j + 1];

                if let Some((our_seg_t, lane_seg_t)) =
                    line_segment_intersection_2d(p1.x, p1.z, p2.x, p2.z, q1.x, q1.z, q2.x, q2.z)
                {
                    // Convert segment-local t to global t for our path
                    let polyline_len = test_polyline.len() as f32;
                    let path_t = (i as f32 + our_seg_t) / (polyline_len - 1.0);

                    // Convert to global t for the lane
                    let lane_len = lane_points.len() as f32;
                    let lane_t = (j as f32 + lane_seg_t) / (lane_len - 1.0);

                    // Skip if lane intersection is too close to lane endpoints
                    if lane_t < ENDPOINT_T_EPS || lane_t > 1.0 - ENDPOINT_T_EPS {
                        continue;
                    }

                    // Calculate world position at crossing
                    let world_pos = self.sample_path_at_t(
                        terrain_renderer,
                        start_pos,
                        end_pos,
                        control,
                        path_t,
                    );

                    return Some(CrossingPoint {
                        t: path_t,
                        world_pos,
                        kind: CrossingKind::LaneCrossing {
                            lane_id: lane_id.clone(),
                            lane_t,
                        },
                    });
                }
            }
        }

        None
    }

    fn project_point_to_path(
        &self,
        start: Vec3,
        end: Vec3,
        control: Option<Vec3>,
        point: Vec3,
    ) -> Option<(f32, f32)> {
        // Returns (t, distance)
        match control {
            None => {
                // Straight line projection
                let dx = end.x - start.x;
                let dz = end.z - start.z;
                let len_sq = dx * dx + dz * dz;
                if len_sq < 1e-6 {
                    return None;
                }

                let px = point.x - start.x;
                let pz = point.z - start.z;
                let t = (px * dx + pz * dz) / len_sq;

                if t < 0.0 || t > 1.0 {
                    return None;
                }

                let proj_x = start.x + t * dx;
                let proj_z = start.z + t * dz;
                let dist = ((point.x - proj_x).powi(2) + (point.z - proj_z).powi(2)).sqrt();

                Some((t, dist))
            }
            Some(c) => {
                // Quadratic bezier - sample and find closest point
                let samples = 100;
                let mut best: Option<(f32, f32)> = None;

                for i in 0..=samples {
                    let t = i as f32 / samples as f32;
                    let omt = 1.0 - t;
                    let px = omt * omt * start.x + 2.0 * omt * t * c.x + t * t * end.x;
                    let pz = omt * omt * start.z + 2.0 * omt * t * c.z + t * t * end.z;

                    let dist = ((point.x - px).powi(2) + (point.z - pz).powi(2)).sqrt();

                    if best.is_none() || dist < best.unwrap().1 {
                        best = Some((t, dist));
                    }
                }

                best
            }
        }
    }

    fn sample_path_at_t(
        &self,
        terrain_renderer: &TerrainRenderer,
        start: Vec3,
        end: Vec3,
        control: Option<Vec3>,
        t: f32,
    ) -> Vec3 {
        let (x, z) = match control {
            Some(c) => {
                let omt = 1.0 - t;
                (
                    omt * omt * start.x + 2.0 * omt * t * c.x + t * t * end.x,
                    omt * omt * start.z + 2.0 * omt * t * c.z + t * t * end.z,
                )
            }
            None => (
                start.x + t * (end.x - start.x),
                start.z + t * (end.z - start.z),
            ),
        };
        let y = terrain_renderer.get_height_at([x, z]) + CLEARANCE;
        Vec3::new(x, y, z)
    }

    fn deduplicate_crossings(&self, crossings: Vec<CrossingPoint>) -> Vec<CrossingPoint> {
        const MIN_T_DISTANCE: f32 = 0.02;

        let mut result = Vec::new();

        for crossing in crossings {
            if result.is_empty() {
                result.push(crossing);
                continue;
            }

            let last_t = result.last().unwrap().t;
            if (crossing.t - last_t).abs() > MIN_T_DISTANCE {
                result.push(crossing);
            }
        }

        result
    }

    // ==================== ROAD COMMIT WITH CROSSINGS ====================

    fn commit_road_with_crossings(
        &mut self,
        terrain_renderer: &TerrainRenderer,
        storage: &RoadStorage,
        road_style_params: &RoadStyleParams,
        start: &Anchor,
        end: &Anchor,
        control: Option<Vec3>,
        chunk_id: ChunkId,
        output: &mut Vec<RoadEditorCommand>,
    ) -> Vec<RoadCommand> {
        let mut cmds = Vec::new();

        // Get positions
        let Some(start_pos) = start.planned_node.position(storage) else {
            output.push(RoadEditorCommand::PreviewError(
                PreviewError::MissingNodeData,
            ));
            return Vec::new();
        };
        let end_pos = end.snap.world_pos;

        // Find all crossings along the path
        let crossings = self.find_all_crossings(
            storage,
            terrain_renderer,
            start_pos,
            end_pos,
            control,
            start,
            end,
        );

        // Build waypoint list
        let mut waypoints: Vec<ResolvedWaypoint> = Vec::new();

        // Resolve start anchor
        let Some((start_node_id, start_node_pos)) = self.resolve_anchor(
            storage,
            road_style_params,
            start,
            chunk_id,
            &mut cmds,
            output,
        ) else {
            return Vec::new();
        };

        waypoints.push(ResolvedWaypoint {
            node_id: start_node_id,
            pos: start_node_pos,
            t: 0.0,
        });

        // Process crossings in order
        for crossing in crossings {
            let node_id = match crossing.kind {
                CrossingKind::ExistingNode(id) => id,
                CrossingKind::LaneCrossing { lane_id, .. } => {
                    let Some((split_cmds, new_node_id)) = self.plan_split(
                        road_style_params,
                        storage,
                        lane_id,
                        crossing.world_pos,
                        chunk_id,
                    ) else {
                        continue;
                    };
                    cmds.extend(split_cmds);
                    new_node_id
                }
            };

            waypoints.push(ResolvedWaypoint {
                node_id,
                pos: crossing.world_pos,
                t: crossing.t,
            });
        }

        // Resolve end anchor
        let Some((end_node_id, end_node_pos)) =
            self.resolve_anchor(storage, road_style_params, end, chunk_id, &mut cmds, output)
        else {
            return cmds;
        };

        waypoints.push(ResolvedWaypoint {
            node_id: end_node_id,
            pos: end_node_pos,
            t: 1.0,
        });

        // Remove duplicate consecutive nodes
        waypoints.dedup_by(|a, b| a.node_id == b.node_id);

        // Create segments between consecutive waypoints
        for i in 0..waypoints.len() - 1 {
            let from = &waypoints[i];
            let to = &waypoints[i + 1];

            if from.node_id == to.node_id {
                continue;
            }

            // Calculate centerline for this segment
            let segment_centerline = self.compute_segment_centerline(
                terrain_renderer,
                start_pos,
                end_pos,
                control,
                from.t,
                to.t,
                from.pos,
                to.pos,
            );
            if segment_centerline.len() < 2 {
                continue;
            }

            let segment_id = self.allocator.alloc_segment();
            cmds.push(RoadCommand::AddSegment {
                start: from.node_id,
                end: to.node_id,
                structure: road_style_params.road_type().structure(),
                chunk_id,
            });

            self.emit_lanes_from_centerline(
                terrain_renderer,
                &mut cmds,
                road_style_params,
                segment_id,
                from.node_id,
                to.node_id,
                &segment_centerline,
                chunk_id,
            );
        }

        // Generate intersections for all waypoint nodes
        for waypoint in &waypoints {
            // Use a generic intersection push for intermediate nodes
            push_intersection_for_node(&mut cmds, waypoint.node_id, road_style_params, chunk_id);
        }

        cmds
    }

    fn compute_segment_centerline(
        &self,
        terrain_renderer: &TerrainRenderer,
        original_start: Vec3,
        original_end: Vec3,
        original_control: Option<Vec3>,
        from_t: f32,
        to_t: f32,
        from_pos: Vec3,
        to_pos: Vec3,
    ) -> Vec<Vec3> {
        match original_control {
            None => {
                // Straight segment
                make_straight_centerline(terrain_renderer, from_pos, to_pos)
            }
            Some(c) => {
                // Extract subsection of Bézier curve
                let (sub_start, sub_control, sub_end) =
                    subdivide_quadratic_bezier(original_start, c, original_end, from_t, to_t);

                let est_len = estimate_bezier_arc_length(sub_start, sub_control, sub_end);
                let samples = compute_curve_segment_count(est_len);
                sample_quadratic_bezier(sub_start, sub_control, sub_end, samples)
            }
        }
        // trim_polyline_both_ends(segment_centerline.as_slice(), 3)
    }

    // ==================== EXISTING METHODS (unchanged) ====================

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
                    geometry: geom1,
                    speed_limit: old_lane.speed_limit(),
                    capacity: old_lane.capacity(),
                    vehicle_mask: old_lane.vehicle_mask(),
                    base_cost: cost1,
                    chunk_id,
                });

                cmds.push(RoadCommand::AddLane {
                    from: new_node_id,
                    to: a_id,
                    segment: seg1_id,
                    lane_index: old_lane.lane_index(),
                    geometry: geom2,
                    speed_limit: old_lane.speed_limit(),
                    capacity: old_lane.capacity(),
                    vehicle_mask: old_lane.vehicle_mask(),
                    base_cost: cost2,
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
    pub side_walk_width: f32,
}

impl IntersectionBuildParams {
    pub fn from_style(style: &RoadStyleParams) -> Self {
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
            side_walk_width: style.sidewalk_width,
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
    gizmo: &mut Gizmo,
) {
    if recalc_clearance {
        let Some(node) = storage.node(node_id) else {
            return;
        };
        // if storage.enabled_segment_count_connected_to_node(node_id) == 2 {
        //     let segment_ids = storage.enabled_segments_connected_to_node(node_id);
        //     let mut angle = 180.0f32.to_radians();
        //     for segment_id in segment_ids {
        //         let segment = storage.segment(segment_id);
        //         segment.
        //     }
        //     carve_intersection_clearance(storage, node_id, node.connection_count() as f32 * 0.7);
        // } else {
        initial_carve(storage, node_id, params, gizmo);
        //};
        // let demands = probe_intersection_node_lanes(storage, node_id, params);
        // carve_intersection_clearance_per_lane(storage, node_id, &demands);
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

fn initial_carve(
    storage: &mut RoadStorage,
    node_id: NodeId,
    params: &IntersectionBuildParams,
    gizmo: &mut Gizmo,
) {
    let Some(node) = storage.node(node_id) else {
        return;
    };
    let center = Vec3::from_array(node.position());

    let incoming: HashSet<LaneId> = node.incoming_lanes().iter().copied().collect();

    let segs = segment_ids_sorted_left_to_right(
        storage.enabled_segments_connected_to_node(node_id),
        storage,
        node_id,
    );

    if segs.len() < 2 {
        // Single segment - just do circle carve NO DON'T DO ANYTHING
        // let fallback_radius = params.side_walk_width * 4.0;
        // circle_carve_lanes(storage, node_id, &incoming, center, fallback_radius, gizmo);
        return;
    }

    let mut cut_vertices: Vec<Vec3> = Vec::new();
    let n = segs.len();

    for i in 0..n {
        let left_seg = storage.segment(segs[i]);
        let right_seg = storage.segment(segs[(i + 1) % n]);

        let l_lane = {
            let pts_to_node = left_seg.end() == node_id;
            left_seg
                .lanes()
                .iter()
                .copied()
                .filter(|id| storage.lane(id).is_enabled())
                .max_by_key(|id| {
                    let idx = storage.lane(id).lane_index();
                    if pts_to_node { idx } else { -idx }
                })
        };

        let r_lane = {
            let pts_to_node = right_seg.end() == node_id;
            right_seg
                .lanes()
                .iter()
                .copied()
                .filter(|id| storage.lane(id).is_enabled())
                .min_by_key(|id| {
                    let idx = storage.lane(id).lane_index();
                    if pts_to_node { idx } else { -idx }
                })
        };

        let (Some(l_lane), Some(r_lane)) = (l_lane, r_lane) else {
            println!(
                "initial_carve: segment pair {}/{} missing boundary lanes",
                i,
                (i + 1) % n
            );
            continue;
        };

        let lp_og = storage.lane(&l_lane).polyline();
        let rp_og = storage.lane(&r_lane).polyline();
        let lp = &offset_polyline_f32(lp_og, params.lane_width_m * 2.0 + params.side_walk_width);
        let rp = &offset_polyline_f32(rp_og, params.lane_width_m * 2.0 + params.side_walk_width);
        // Debug: show which lanes we're trying to intersect
        gizmo.render_polyline(lp, [0.8, 0.0, 0.2], 2.0, DEBUG_DRAW_DURATION);
        gizmo.render_polyline(rp, [0.2, 0.0, 0.8], 2.0, DEBUG_DRAW_DURATION);

        let intersection = polyline_intersection_xz_best(lp, rp, center);

        let Some(p2) = intersection else {
            println!(
                "initial_carve: no intersection found for segment pair {}/{} (lanes {:?}/{:?}, {} pts / {} pts)",
                i,
                (i + 1) % n,
                l_lane,
                r_lane,
                lp.len(),
                rp.len()
            );

            // Fallback: use midpoint between lane endpoints as corner approximation
            let (l_node_pt, _) = get_node_side_end(lp, center);
            let (r_node_pt, _) = get_node_side_end(rp, center);
            let fallback = (l_node_pt + r_node_pt) * 0.5;

            let dir = (fallback - center).normalize_or_zero();
            if dir != Vec3::ZERO {
                let shift = 5.2 * params.side_walk_width;
                cut_vertices.push(fallback + dir * shift);
                gizmo.draw_cross(fallback, 0.3, [1.0, 0.5, 0.0], DEBUG_DRAW_DURATION); // orange = fallback
            }
            continue;
        };

        let dir = (p2 - center).normalize_or_zero();
        let shift = 5.2 * params.side_walk_width;
        let v = p2 + dir * shift;

        cut_vertices.push(v);

        gizmo.render_line(
            center.to_array(),
            p2.to_array(),
            [1.0, 1.0, 1.0],
            DEBUG_DRAW_DURATION,
        );
        gizmo.draw_cross(p2, 0.25, [0.0, 1.0, 0.2], DEBUG_DRAW_DURATION); // green = found intersection
    }

    if cut_vertices.len() < 3 {
        println!(
            "initial_carve: cut polygon has {} vertices (<3), falling back to circle",
            cut_vertices.len()
        );
        let fallback_radius = params.side_walk_width * 7.0;
        circle_carve_lanes(storage, node_id, &incoming, center, fallback_radius, gizmo);
        return;
    }

    cut_vertices = enforce_ccw_winding(&cut_vertices);

    let is_simple = is_simple_polygon(&cut_vertices);
    let contains_center = point_in_polygon_xz(center, &cut_vertices);

    if !is_simple || !contains_center {
        println!(
            "initial_carve: polygon invalid (simple={}, contains_center={}), falling back to circle",
            is_simple, contains_center
        );
        let fallback_radius = params.side_walk_width * 6.0;
        circle_carve_lanes(storage, node_id, &incoming, center, fallback_radius, gizmo);
        return;
    }

    cut_vertices = add_edge_midpoints(&cut_vertices);

    let mut cut_dbg = cut_vertices.clone();
    cut_dbg.push(cut_vertices[0]);
    gizmo.render_polyline(&cut_dbg, [0.0, 0.0, 0.0], 10.0, DEBUG_DRAW_DURATION);

    let mut edits: Vec<(LaneId, LaneGeometry)> = Vec::new();

    for seg_id in segs.iter().copied() {
        let seg = storage.segment(seg_id);

        for lane_id in seg.lanes().iter().copied() {
            let lane = storage.lane(&lane_id);
            if !lane.is_enabled() {
                continue;
            }

            let pts = lane.polyline();
            if pts.len() < 2 {
                continue;
            }

            let is_incoming = incoming.contains(&lane_id);

            let result = if is_incoming {
                let rev: Vec<Vec3> = pts.iter().copied().rev().collect();
                closest_hit_distance_from_start_xz(&rev, &cut_vertices)
            } else {
                closest_hit_distance_from_start_xz(pts, &cut_vertices)
            };

            let (amount, hit) = match result {
                Some((a, h)) if a > 0.001 => (a, h),
                _ => continue,
            };

            gizmo.draw_cross(hit, 0.2, [1.0, 0.0, 0.0], DEBUG_DRAW_DURATION);

            let new_pts = if is_incoming {
                modify_polyline_end(pts, amount)
            } else {
                modify_polyline_start(pts, amount)
            };

            let Some(new_pts) = new_pts else {
                continue;
            };

            edits.push((lane_id, LaneGeometry::from_polyline(new_pts)));
        }
    }

    for (lane_id, geom) in edits {
        storage.lane_mut(lane_id).replace_geometry(geom);
    }
}
/// Computes corner vertex where outer edges of two adjacent lanes would meet.
/// All math done in XZ plane, Y is taken from center.
fn compute_corner_from_lane_edges(
    l_poly: &[Vec3],
    r_poly: &[Vec3],
    l_ends_at_node: bool,
    r_ends_at_node: bool,
    margin: f32,
    center: Vec3,
    max_dist: f32,
) -> Vec3 {
    if l_poly.len() < 2 || r_poly.len() < 2 {
        return center;
    }

    // Get node-side position and road direction for each lane
    // Road direction = direction the lane travels (toward or away from node)
    let (l_pos, l_dir) = if l_ends_at_node {
        let p = l_poly[l_poly.len() - 1];
        let prev = l_poly[l_poly.len() - 2];
        (
            Vec2::new(p.x, p.z),
            Vec2::new(p.x - prev.x, p.z - prev.z).normalize_or_zero(),
        )
    } else {
        let p = l_poly[0];
        let next = l_poly[1];
        (
            Vec2::new(p.x, p.z),
            Vec2::new(next.x - p.x, next.z - p.z).normalize_or_zero(),
        )
    };

    let (r_pos, r_dir) = if r_ends_at_node {
        let p = r_poly[r_poly.len() - 1];
        let prev = r_poly[r_poly.len() - 2];
        (
            Vec2::new(p.x, p.z),
            Vec2::new(p.x - prev.x, p.z - prev.z).normalize_or_zero(),
        )
    } else {
        let p = r_poly[0];
        let next = r_poly[1];
        (
            Vec2::new(p.x, p.z),
            Vec2::new(next.x - p.x, next.z - p.z).normalize_or_zero(),
        )
    };

    // Perpendicular offsets to get outer edges:
    // Left lane's outer edge is on its RIGHT (rotate CW: (x,z) -> (z,-x))
    // Right lane's outer edge is on its LEFT (rotate CCW: (x,z) -> (-z,x))
    let l_perp = Vec2::new(l_dir.y, -l_dir.x); // CW 90°
    let r_perp = Vec2::new(-r_dir.y, r_dir.x); // CCW 90°

    // Edge line positions
    let l_edge = l_pos + l_perp * margin;
    let r_edge = r_pos + r_perp * margin;

    // Line-line intersection (infinite lines, not rays!)
    let cross = l_dir.x * r_dir.y - l_dir.y * r_dir.x;

    if cross.abs() < 1e-5 {
        // Nearly parallel - fallback to midpoint pushed outward
        return fallback_corner(l_edge, r_edge, center, margin);
    }

    let diff = r_edge - l_edge;
    let t = (diff.x * r_dir.y - diff.y * r_dir.x) / cross;

    let corner_2d = l_edge + l_dir * t;

    // Sanity check: corner shouldn't be too far from center
    let center_2d = Vec2::new(center.x, center.z);
    let dist = corner_2d.distance(center_2d);

    if dist > max_dist || dist < 0.5 {
        return fallback_corner(l_edge, r_edge, center, margin);
    }

    // Also check that corner is generally "outward" from center
    let to_corner = (corner_2d - center_2d).normalize_or_zero();
    let mid_edge = (l_edge + r_edge) * 0.5;
    let to_mid = (mid_edge - center_2d).normalize_or_zero();

    if to_corner.dot(to_mid) < 0.0 {
        // Corner is on wrong side of center
        return fallback_corner(l_edge, r_edge, center, margin);
    }

    Vec3::new(corner_2d.x, center.y, corner_2d.y)
}

fn fallback_corner(l_edge: Vec2, r_edge: Vec2, center: Vec3, margin: f32) -> Vec3 {
    let mid = (l_edge + r_edge) * 0.5;
    let center_2d = Vec2::new(center.x, center.z);
    let outward = (mid - center_2d).normalize_or_zero();
    let corner = mid + outward * margin;
    Vec3::new(corner.x, center.y, corner.y)
}
fn trim_all_lanes_to_polygon(
    storage: &mut RoadStorage,
    node_id: NodeId,
    incoming: &HashSet<LaneId>,
    segs: &[SegmentId],
    polygon: &[Vec3],
    gizmo: &mut Gizmo,
) {
    let mut edits: Vec<(LaneId, LaneGeometry)> = Vec::new();

    for seg_id in segs.iter().copied() {
        let seg = storage.segment(seg_id);

        for lane_id in seg.lanes().iter().copied() {
            let lane = storage.lane(&lane_id);
            if !lane.is_enabled() {
                continue;
            }

            let pts = lane.polyline();
            if pts.len() < 2 {
                continue;
            }

            let is_incoming = incoming.contains(&lane_id);

            // Find furthest hit from node-side
            let hit_result = if is_incoming {
                let rev: Vec<Vec3> = pts.iter().copied().rev().collect();
                furthest_hit_distance_from_start_xz(&rev, polygon)
            } else {
                furthest_hit_distance_from_start_xz(pts, polygon)
            };

            let Some((amount, hit)) = hit_result else {
                // No intersection - lane might be entirely inside or outside
                // Try a simpler approach: find distance from node-side endpoint to polygon
                continue;
            };

            if amount <= 0.01 {
                continue;
            }

            gizmo.draw_cross(hit, 0.3, [1.0, 0.0, 0.0], DEBUG_DRAW_DURATION);

            let new_pts = if is_incoming {
                modify_polyline_end(pts, amount)
            } else {
                modify_polyline_start(pts, amount)
            };

            if let Some(new_pts) = new_pts {
                if new_pts.len() >= 2 {
                    edits.push((lane_id, LaneGeometry::from_polyline(new_pts)));
                }
            }
        }
    }

    for (lane_id, geom) in edits {
        storage.lane_mut(lane_id).replace_geometry(geom);
    }
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
/// Trims or Extends a polyline from the END.
/// Positive amount: Trim. Negative amount: Extend (linearly).
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
fn polyline_intersection_xz(a: &[Vec3], b: &[Vec3]) -> Option<Vec3> {
    // 1. Check for physical intersection on existing segments (Original Logic)
    for i in 0..a.len() - 1 {
        for j in 0..b.len() - 1 {
            if let Some(p) = segment_intersection_xz(a[i], a[i + 1], b[j], b[j + 1]) {
                return Some(p);
            }
        }
    }

    // 2. No intersection found? Try projecting the ends!
    // We need at least 2 vertices in each polyline to determine a direction.
    if a.len() < 2 || b.len() < 2 {
        return None;
    }

    // Get the last two vertices to determine direction
    let a_prev = a[a.len() - 2];
    let a_last = a[a.len() - 1];

    let b_prev = b[b.len() - 2];
    let b_last = b[b.len() - 1];

    // Calculate the intersection of the two imaginary rays extending from the ends
    get_projected_intersection_xz(a_prev, a_last, b_prev, b_last)
}

/// Helper function to calculate the intersection of two rays on the XZ plane.
/// Ray A starts at `a_last` and goes in direction (a_last - a_prev).
/// Ray B starts at `b_last` and goes in direction (b_last - b_prev).
fn get_projected_intersection_xz(
    a_prev: Vec3,
    a_last: Vec3,
    b_prev: Vec3,
    b_last: Vec3,
) -> Option<Vec3> {
    // Direction vectors
    let da_x = a_last.x - a_prev.x;
    let da_z = a_last.z - a_prev.z;

    let db_x = b_last.x - b_prev.x;
    let db_z = b_last.z - b_prev.z;

    // Determinant (Cross product of directions in 2D)
    let det = da_x * db_z - da_z * db_x;

    // If det is 0, the lines are parallel and will never intersect
    if det.abs() < 1e-6 {
        return None;
    }

    // Vector from Ray A start to Ray B start
    let diff_x = b_last.x - a_last.x;
    let diff_z = b_last.z - a_last.z;

    // Calculate 't' (distance along Ray A) and 'u' (distance along Ray B)
    // Using Cramer's rule for linear systems
    let t = (diff_x * db_z - diff_z * db_x) / det;
    let u = (diff_x * da_z - diff_z * da_x) / det;

    // Check if the intersection is actually "in front" of the polylines.
    // t >= 0 means the intersection is ahead of polyline A.
    // u >= 0 means the intersection is ahead of polyline B.
    // small epsilon to handle pesky floating point imprecision.
    if t >= -1e-4 && u >= -1e-4 {
        return Some(Vec3 {
            x: a_last.x + t * da_x,
            y: a_last.y, // Preserving Y height of polyline A's end
            z: a_last.z + t * da_z,
        });
    }

    // If t or u are negative, the lines diverge away from each other (intersection is behind them)
    None
}

fn segment_intersection_xz(a0: Vec3, a1: Vec3, b0: Vec3, b1: Vec3) -> Option<Vec3> {
    let p = a0.xz();
    let r = (a1 - a0).xz();
    let q = b0.xz();
    let s = (b1 - b0).xz();

    let cross = |u: Vec2, v: Vec2| u.x * v.y - u.y * v.x;
    let rxs = cross(r, s);
    if rxs.abs() < 1e-6 {
        return None;
    }

    let t = cross(q - p, s) / rxs;
    let u = cross(q - p, r) / rxs;

    if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
        Some(a0 + (a1 - a0) * t)
    } else {
        None
    }
}

fn rightmost_lane_id(segment: &Segment, storage: &RoadStorage, node_id: NodeId) -> LaneId {
    let points_to_node = segment.end() == node_id;

    segment
        .lanes()
        .iter()
        .copied()
        .max_by_key(|id| {
            let idx = storage.lane(id).lane_index();
            if points_to_node { idx } else { -idx }
        })
        .unwrap()
}

fn leftmost_lane_id(segment: &Segment, storage: &RoadStorage, node_id: NodeId) -> LaneId {
    let points_to_node = segment.end() == node_id;

    segment
        .lanes()
        .iter()
        .copied()
        .min_by_key(|id| {
            let idx = storage.lane(id).lane_index();
            if points_to_node { idx } else { -idx }
        })
        .unwrap()
}

fn segment_right_hand_width(
    segment: &Segment,
    storage: &RoadStorage,
    params: &IntersectionBuildParams,
) -> f32 {
    let mut right_hand_width: f32 = 0.0;
    for lane_id in segment.lanes().iter() {
        let lane = storage.lane(lane_id);
        if lane.lane_index() > 0 {
            right_hand_width += params.lane_width_m;
        }
    }
    right_hand_width += params.side_walk_width;
    right_hand_width
}
fn segment_left_hand_width(
    segment: &Segment,
    storage: &RoadStorage,
    params: &IntersectionBuildParams,
) -> f32 {
    let mut left_hand_width: f32 = 0.0;
    for lane_id in segment.lanes().iter() {
        let lane = storage.lane(lane_id);
        if lane.lane_index() < 0 {
            left_hand_width += params.lane_width_m;
        }
    }
    left_hand_width += params.side_walk_width;
    left_hand_width
}
fn segment_dir_at_node(segment: &Segment, storage: &RoadStorage, node_id: NodeId) -> Vec3 {
    let lane_id = segment.lanes()[0]; // any lane is fine for direction
    let lane = storage.lane(&lane_id);
    let poly = lane.polyline();

    let node_pos = {
        let n = storage.node(node_id).unwrap();
        Vec3::from_array(n.position())
    };

    let dir = if segment.end() == node_id {
        // segment points INTO node, so direction is reversed
        node_pos - poly[poly.len() - 2]
    } else {
        poly[1] - node_pos
    };

    dir.normalize()
}

fn segment_ids_paired(segment_ids_sorted: Vec<SegmentId>) -> Vec<(SegmentId, SegmentId)> {
    let n = segment_ids_sorted.len();
    if n < 2 {
        return Vec::new();
    }

    let mut pairs = Vec::with_capacity(n);

    for i in 0..n {
        let a = segment_ids_sorted[i];
        let b = segment_ids_sorted[(i + 1) % n];
        pairs.push((a, b));
    }

    pairs
}

fn segment_ids_sorted_left_to_right(
    segment_ids: Vec<SegmentId>,
    storage: &RoadStorage,
    node_id: NodeId,
) -> Vec<SegmentId> {
    let node = storage.node(node_id).expect("node missing");
    let node_pos = Vec3::from_array(node.position());

    let mut with_angles: Vec<(SegmentId, f32)> = segment_ids
        .into_iter()
        .map(|segment_id| {
            let segment = storage.segment(segment_id);
            let points_to_node = segment.end() == node_id;

            // Pick rightmost lane depending on orientation
            let mut chosen_lane_id = None;
            let mut best_index: i8 = if points_to_node { i8::MIN } else { i8::MAX };

            for lane_id in segment.lanes() {
                let lane = storage.lane(lane_id);
                let idx = lane.lane_index();

                if points_to_node {
                    if idx > best_index {
                        best_index = idx;
                        chosen_lane_id = Some(lane_id);
                    }
                } else {
                    if idx < best_index {
                        best_index = idx;
                        chosen_lane_id = Some(lane_id);
                    }
                }
            }

            let lane_id = chosen_lane_id.expect("segment without lanes");
            let lane = storage.lane(lane_id);
            let polyline = lane.polyline();

            // IMPORTANT: direction should be consistent (here: AWAY from node)
            let dir = if points_to_node {
                // node is at end -> step from node to the previous point (away from node)
                polyline[polyline.len() - 2] - node_pos
            } else {
                // node is at start -> step from node to the next point (away from node)
                polyline[1] - node_pos
            };

            let v = dir.xz(); // Vec2 (x, z)
            let mut angle = v.y.atan2(v.x); // [-pi, pi]

            // Normalize to [0, 2pi)
            if angle < 0.0 {
                angle += std::f32::consts::TAU;
            }

            // Optional: rotate so "left" (-x) is near 0 instead of the wrap point.
            // This prevents "left-ish" segments from splitting across the boundary.
            angle = (angle - std::f32::consts::PI + std::f32::consts::TAU) % std::f32::consts::TAU;

            (segment_id, angle)
        })
        .collect();

    with_angles.sort_by(|(id_a, ang_a), (id_b, ang_b)| {
        ang_b.total_cmp(ang_a).then_with(|| id_a.cmp(id_b))
    }); // Descending sort with tiebreaker, ccw left to right!

    with_angles.into_iter().map(|(id, _)| id).collect()
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
    for lane_id in incoming_ids.iter() {
        let (maybe_new_geom, maybe_new_cost) = {
            let lane = storage.lane(lane_id);
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
            edits.push((*lane_id, new_geom, new_cost));
        }
    }

    // Outgoing: trim START
    for lane_id in outgoing_ids.iter() {
        let (maybe_new_geom, maybe_new_cost) = {
            let lane = storage.lane(lane_id);
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
            edits.push((*lane_id, new_geom, new_cost));
        }
    }

    // Apply edits (mutable borrows only)
    for (lane_id, new_geom, new_cost) in edits {
        let lane = storage.lane_mut(lane_id);
        lane.replace_geometry(new_geom);
        lane.replace_base_cost(new_cost);
    }
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

/// Smooth polygon by adding midpoints that curve outward slightly
fn smooth_polygon_corners(vertices: &[Vec3], center: Vec3, bulge_factor: f32) -> Vec<Vec3> {
    let n = vertices.len();
    if n < 3 {
        return vertices.to_vec();
    }

    let mut result = Vec::with_capacity(n * 2);

    for i in 0..n {
        let curr = vertices[i];
        let next = vertices[(i + 1) % n];

        result.push(curr);

        // Midpoint between consecutive vertices
        let mid = (curr + next) * 0.5;

        // Push midpoint outward from center
        let to_mid = (mid - center).normalize_or_zero();
        let edge_len = (next - curr).length();

        // Bulge amount proportional to edge length but capped
        let bulge = (edge_len * bulge_factor).min(3.0);
        let curved_mid = mid + to_mid * bulge;

        result.push(curved_mid);
    }

    result
}
/// Find the intersection of two polylines, preferring the closest valid intersection
/// that is not trivially at the node center.
fn polyline_intersection_xz_best(a: &[Vec3], b: &[Vec3], center: Vec3) -> Option<Vec3> {
    const MIN_VALID_DIST: f32 = 0.05; // Reduced from 2.0 - some tight junctions have close corners
    const MAX_VALID_DIST: f32 = 50.0;

    let mut candidates: Vec<(f32, Vec3)> = Vec::new();

    // Collect all physical intersections
    for i in 0..a.len().saturating_sub(1) {
        for j in 0..b.len().saturating_sub(1) {
            if let Some(p) = segment_intersection_xz(a[i], a[i + 1], b[j], b[j + 1]) {
                let d = p.distance(center);
                if d >= MIN_VALID_DIST && d <= MAX_VALID_DIST {
                    candidates.push((d, p));
                }
            }
        }
    }

    // Pick the closest valid intersection to center
    if let Some((_, p)) = candidates
        .iter()
        .min_by(|(d1, _), (d2, _)| d1.partial_cmp(d2).unwrap())
    {
        return Some(*p);
    }

    // No valid physical intersection - try ray projection
    if a.len() >= 2 && b.len() >= 2 {
        let (a_origin, a_toward) = get_node_side_end(a, center);
        let (b_origin, b_toward) = get_node_side_end(b, center);

        if let Some(p) = project_rays_intersection_xz(a_origin, a_toward, b_origin, b_toward) {
            let d = p.distance(center);
            if d >= MIN_VALID_DIST && d <= MAX_VALID_DIST {
                return Some(p);
            }
        }
    }

    None
}
/// Returns (node-side point, next point going away from node)
/// Uses more points for a stable direction if available
fn get_node_side_end(points: &[Vec3], center: Vec3) -> (Vec3, Vec3) {
    if points.len() < 2 {
        let p = if points.is_empty() {
            Vec3::ZERO
        } else {
            points[0]
        };
        return (p, p);
    }

    let d_start = points[0].distance(center);
    let d_end = points[points.len() - 1].distance(center);

    if d_start <= d_end {
        // Node at start - pick a point further along for stable direction
        let toward_idx = (points.len() / 4).max(1).min(points.len() - 1);
        (points[0], points[toward_idx])
    } else {
        // Node at end - pick a point further back for stable direction
        let toward_idx =
            points.len().saturating_sub(1) - (points.len() / 4).max(1).min(points.len() - 1);
        (points[points.len() - 1], points[toward_idx])
    }
}

/// Project two rays and find their intersection in XZ plane.
/// Ray A: from a_origin in direction (a_toward - a_origin)
/// Ray B: from b_origin in direction (b_toward - b_origin)
fn project_rays_intersection_xz(
    a_origin: Vec3,
    a_toward: Vec3,
    b_origin: Vec3,
    b_toward: Vec3,
) -> Option<Vec3> {
    let da = a_toward - a_origin;
    let db = b_toward - b_origin;

    let da_len = (da.x * da.x + da.z * da.z).sqrt();
    let db_len = (db.x * db.x + db.z * db.z).sqrt();

    if da_len < 1e-6 || db_len < 1e-6 {
        return None;
    }

    // Normalize to unit vectors
    let da_x = da.x / da_len;
    let da_z = da.z / da_len;
    let db_x = db.x / db_len;
    let db_z = db.z / db_len;

    // Determinant (cross product of unit direction vectors)
    let det = da_x * db_z - da_z * db_x;

    if det.abs() < 1e-6 {
        return None; // parallel or nearly parallel
    }

    let diff_x = b_origin.x - a_origin.x;
    let diff_z = b_origin.z - a_origin.z;

    // t and u are arc-length distances since we're using unit vectors
    let t = (diff_x * db_z - diff_z * db_x) / det;
    let u = (diff_x * da_z - diff_z * da_x) / det;

    // Check if intersection is ahead of both ray origins
    // Use small negative epsilon to handle floating point edge cases
    if t >= -1e-4 && u >= -1e-4 {
        // BUG FIX: Don't multiply by da_len!
        // t is already in world units (meters) because we divided by det of unit vectors
        Some(Vec3::new(
            a_origin.x + t * da_x,
            (a_origin.y + b_origin.y) * 0.5,
            a_origin.z + t * da_z,
        ))
    } else {
        None
    }
}

/// Returns (distance_from_start, hit_point) for the *closest* hit when
/// walking from points[0] along the polyline, crossing the polygon boundary.
pub fn closest_hit_distance_from_start_xz(points: &[Vec3], poly: &[Vec3]) -> Option<(f32, Vec3)> {
    if points.len() < 2 || poly.len() < 3 {
        return None;
    }

    let mut acc = 0.0f32;

    for i in 0..points.len() - 1 {
        let a = points[i];
        let b = points[i + 1];
        let seg = b - a;
        let seg_len = seg.length();
        if seg_len < 1e-6 {
            continue;
        }

        let a0 = a.xz();
        let a1 = b.xz();

        let mut best_t_on_seg: Option<f32> = None;

        for j in 0..poly.len() {
            let p0 = poly[j].xz();
            let p1 = poly[(j + 1) % poly.len()].xz();

            if let Some(t) = seg_seg_intersection_t(a0, a1, p0, p1) {
                best_t_on_seg = Some(match best_t_on_seg {
                    None => t,
                    Some(bt) => bt.min(t),
                });
            }
        }

        if let Some(t) = best_t_on_seg {
            let hit = a + seg * t;
            let dist = acc + seg_len * t;
            return Some((dist, hit));
        }

        acc += seg_len;
    }

    None
}

/// Enforce counter-clockwise winding for polygon vertices (in XZ plane)
fn enforce_ccw_winding(vertices: &[Vec3]) -> Vec<Vec3> {
    if vertices.len() < 3 {
        return vertices.to_vec();
    }

    let mut signed_area_2x = 0.0f32;
    let n = vertices.len();
    for i in 0..n {
        let v0 = vertices[i];
        let v1 = vertices[(i + 1) % n];
        signed_area_2x += (v1.x - v0.x) * (v1.z + v0.z);
    }

    if signed_area_2x > 0.0 {
        vertices.iter().copied().rev().collect()
    } else {
        vertices.to_vec()
    }
}

/// Check if polygon is simple (no self-intersections)
fn is_simple_polygon(vertices: &[Vec3]) -> bool {
    let n = vertices.len();
    if n < 3 {
        return false;
    }

    for i in 0..n {
        let a0 = vertices[i].xz();
        let a1 = vertices[(i + 1) % n].xz();

        for j in (i + 2)..n {
            if (i == 0) && (j == n - 1) {
                continue;
            }

            let b0 = vertices[j].xz();
            let b1 = vertices[(j + 1) % n].xz();

            if segments_intersect_proper(a0, a1, b0, b1) {
                return false;
            }
        }
    }

    true
}

fn segments_intersect_proper(a0: Vec2, a1: Vec2, b0: Vec2, b1: Vec2) -> bool {
    let r = a1 - a0;
    let s = b1 - b0;
    let denom = cross2(r, s);

    if denom.abs() < 1e-6 {
        return false;
    }

    let qp = b0 - a0;
    let t = cross2(qp, s) / denom;
    let u = cross2(qp, r) / denom;

    const EPS: f32 = 1e-4;
    t > EPS && t < 1.0 - EPS && u > EPS && u < 1.0 - EPS
}

fn point_in_polygon_xz(point: Vec3, polygon: &[Vec3]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }

    let px = point.x;
    let pz = point.z;

    let mut inside = false;
    let mut j = n - 1;

    for i in 0..n {
        let vi = polygon[i];
        let vj = polygon[j];

        if ((vi.z > pz) != (vj.z > pz)) && (px < (vj.x - vi.x) * (pz - vi.z) / (vj.z - vi.z) + vi.x)
        {
            inside = !inside;
        }

        j = i;
    }

    inside
}

fn add_edge_midpoints(vertices: &[Vec3]) -> Vec<Vec3> {
    let n = vertices.len();
    let mut result = Vec::with_capacity(n * 2);

    for i in 0..n {
        let a = vertices[i];
        let b = vertices[(i + 1) % n];

        result.push(a);
        result.push((a + b) * 0.5);
    }

    result
}

fn find_circle_intersection_distance(
    points: &[Vec3],
    center: Vec3,
    radius: f32,
) -> Option<(f32, Vec3)> {
    if points.len() < 2 {
        return None;
    }

    let mut acc = 0.0f32;

    for i in 0..points.len() - 1 {
        let a = points[i];
        let b = points[i + 1];
        let seg = b - a;
        let seg_len = seg.length();

        if seg_len < 1e-6 {
            continue;
        }

        let d = seg / seg_len;
        let f = a - center;

        let a_coef = d.x * d.x + d.z * d.z;
        let b_coef = 2.0 * (f.x * d.x + f.z * d.z);
        let c_coef = f.x * f.x + f.z * f.z - radius * radius;

        let discriminant = b_coef * b_coef - 4.0 * a_coef * c_coef;

        if discriminant >= 0.0 {
            let sqrt_disc = discriminant.sqrt();
            let t1 = (-b_coef - sqrt_disc) / (2.0 * a_coef);
            let t2 = (-b_coef + sqrt_disc) / (2.0 * a_coef);

            for t in [t1, t2] {
                if t >= 0.0 && t <= seg_len {
                    let hit = a + d * t;
                    return Some((acc + t, hit));
                }
            }
        }

        acc += seg_len;
    }

    None
}

fn circle_carve_lanes(
    storage: &mut RoadStorage,
    node_id: NodeId,
    incoming: &HashSet<LaneId>,
    center: Vec3,
    radius: f32,
    gizmo: &mut Gizmo,
) {
    let segs = storage.enabled_segments_connected_to_node(node_id);
    let mut edits: Vec<(LaneId, LaneGeometry)> = Vec::new();

    for seg_id in segs {
        let seg = storage.segment(seg_id);

        for lane_id in seg.lanes().iter().copied() {
            let lane = storage.lane(&lane_id);
            if !lane.is_enabled() {
                continue;
            }

            let pts = lane.polyline();
            if pts.len() < 2 {
                continue;
            }

            let is_incoming = incoming.contains(&lane_id);

            let (amount, hit) = if is_incoming {
                let rev: Vec<Vec3> = pts.iter().copied().rev().collect();
                find_circle_intersection_distance(&rev, center, radius)
            } else {
                find_circle_intersection_distance(pts, center, radius)
            }
            .unwrap_or((0.0, Vec3::ZERO));

            if amount <= 0.001 || hit == Vec3::ZERO {
                continue;
            }

            gizmo.draw_cross(hit, 0.2, [0.0, 1.0, 0.0], DEBUG_DRAW_DURATION);

            let new_pts = if is_incoming {
                modify_polyline_end(pts, amount)
            } else {
                modify_polyline_start(pts, amount)
            };

            if let Some(new_pts) = new_pts {
                edits.push((lane_id, LaneGeometry::from_polyline(new_pts)));
            }
        }
    }

    for (lane_id, geom) in edits {
        storage.lane_mut(lane_id).replace_geometry(geom);
    }
}
