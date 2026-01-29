use crate::positions::{ChunkSize, WorldPos};
use crate::renderer::gizmo::{DEBUG_DRAW_DURATION, Gizmo};
use crate::renderer::world_renderer::{CursorMode, PickedPoint, TerrainRenderer};
use crate::terrain::roads::road_helpers::*;
use crate::terrain::roads::road_mesh_manager::{CLEARANCE, ChunkId};
use crate::terrain::roads::road_structs::*;
use crate::terrain::roads::roads::{
    Lane, LaneGeometry, LaneRef, METERS_PER_LANE_POLYLINE_STEP, NodeLane, RoadCommand, RoadManager,
    RoadStorage, Segment, bezier3, nearest_lane_to_point, project_point_to_lane_xz,
    sample_lane_position,
};
use crate::ui::input::InputState;
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
        let Some(road_type) = (match terrain_renderer.cursor.mode {
            CursorMode::Roads(r) => Some(r),
            _ => None,
        }) else {
            return Vec::new();
        };
        road_style_params.set_road_type(road_type);
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
                    terrain_renderer.chunk_size,
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
                    terrain_renderer.chunk_size,
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
        chunk_size: ChunkSize,
    ) {
        let node_preview = self.build_node_preview_from_snap(storage, snap, chunk_size);
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
        let polyline = make_straight_centerline(
            terrain_renderer,
            start_pos,
            end_pos,
            road_style_params.road_type().structure,
        );
        let estimated_length = end_pos.length_to(start_pos, terrain_renderer.chunk_size);

        let (is_valid, reason) =
            self.validate_placement(storage, start, &end_anchor, terrain_renderer.chunk_size);

        // Find crossings for preview
        let crossings = self.find_all_crossings(
            storage,
            terrain_renderer,
            road_style_params.road_type().structure(),
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

        let node_preview =
            self.build_node_preview_from_snap(storage, snap, terrain_renderer.chunk_size);
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
        chunk_size: ChunkSize,
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
        let estimated_length = control_pos.length_to(start_pos, chunk_size);

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
        control: WorldPos,
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
        let chunk_size = terrain_renderer.chunk_size;
        let end_anchor = self.build_anchor_from_snap(snap);
        let end_pos = snap.world_pos;

        let estimated_length = estimate_bezier_arc_length(
            terrain_renderer,
            road_style_params.road_type().structure(),
            start_pos,
            control,
            end_pos,
        );
        let segment_count = compute_curve_segment_count(estimated_length);
        let polyline = sample_quadratic_bezier(
            terrain_renderer,
            road_style_params.road_type().structure(),
            start_pos,
            control,
            end_pos,
            segment_count,
        );

        let (is_valid, reason) = self.validate_placement(storage, start, &end_anchor, chunk_size);

        // Find crossings for preview
        let crossings = self.find_all_crossings(
            storage,
            terrain_renderer,
            road_style_params.road_type().structure(),
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

        let node_preview = self.build_node_preview_from_snap(storage, snap, chunk_size);
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
        structure_type: StructureType,
        start_pos: WorldPos,
        end_pos: WorldPos,
        control: Option<WorldPos>,
        start_anchor: &Anchor,
        end_anchor: &Anchor,
    ) -> Vec<CrossingPoint> {
        let chunk_size = terrain_renderer.chunk_size;
        let mut crossings = Vec::new();
        let mut crossed_segments: HashSet<SegmentId> = HashSet::new();

        // Build test polyline for intersection testing
        let test_polyline = match control {
            Some(c) => {
                let est_len = estimate_bezier_arc_length(
                    terrain_renderer,
                    structure_type,
                    start_pos,
                    c,
                    end_pos,
                );
                let samples = compute_curve_segment_count(est_len).max(20);
                sample_quadratic_bezier(
                    terrain_renderer,
                    structure_type,
                    start_pos,
                    c,
                    end_pos,
                    samples,
                )
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
                let start_node_pos = start_node.position();

                let end_node = storage.node(segment.end).unwrap();
                let end_node_pos = end_node.position();

                let dist_to_start = crossing.world_pos.length_to(start_node_pos, chunk_size);
                let dist_to_end = crossing.world_pos.length_to(end_node_pos, chunk_size);

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
                    if let Some((new_t, _proj_dist)) = self.project_point_to_path(
                        start_pos,
                        end_pos,
                        control,
                        closest_pos,
                        chunk_size,
                    ) {
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

            let node_pos = node.position();
            if let Some((t, dist)) =
                self.project_point_to_path(start_pos, end_pos, control, node_pos, chunk_size)
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
        test_polyline: &[WorldPos],
        start_pos: WorldPos,
        end_pos: WorldPos,
        control: Option<WorldPos>,
        lane_id: &LaneId,
    ) -> Option<CrossingPoint> {
        let lane = storage.lane(lane_id);
        let lane_points = &lane.geometry().points;

        // For each segment in our test polyline
        for i in 0..test_polyline.len() - 1 {
            let p1 = test_polyline[i];
            let p2 = test_polyline[i + 1];
            let origin = p1;
            let a1 = Vec2::ZERO;
            let a2 = origin.delta_to(p2, terrain_renderer.chunk_size).xz();
            // Check against each segment of the lane
            for j in 0..lane_points.len() - 1 {
                let q1 = lane_points[j];
                let q2 = lane_points[j + 1];
                let b1 = origin.delta_to(q1, terrain_renderer.chunk_size).xz();
                let b2 = origin.delta_to(q2, terrain_renderer.chunk_size).xz();
                if let Some((our_seg_t, lane_seg_t)) =
                    line_segment_intersection_2d(a1.x, a1.y, a2.x, a2.y, b1.x, b1.y, b2.x, b2.y)
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
        start: WorldPos,
        end: WorldPos,
        control: Option<WorldPos>,
        point: WorldPos,
        chunk_size: ChunkSize,
    ) -> Option<(f32, f32)> {
        // Use start as local origin
        let a = Vec3::ZERO;
        let b = start.delta_to(end, chunk_size);
        let p = start.delta_to(point, chunk_size);
        let c = control.map(|c| start.delta_to(c, chunk_size));

        match c {
            None => {
                // Line projection
                let ab = b;
                let len_sq = ab.x * ab.x + ab.z * ab.z;
                if len_sq < 1e-6 {
                    return None;
                }

                let t = (p.x * ab.x + p.z * ab.z) / len_sq;
                if !(0.0..=1.0).contains(&t) {
                    return None;
                }

                let proj = ab * t;
                let dist = (p - proj).xz().length();

                Some((t, dist))
            }

            Some(c) => {
                // Quadratic Bezier via sampling
                let samples = 100;
                let mut best: Option<(f32, f32)> = None;

                for i in 0..=samples {
                    let t = i as f32 / samples as f32;
                    let omt = 1.0 - t;

                    let pos = c * (2.0 * omt * t) + b * (t * t);

                    let dist = (p - pos).xz().length();

                    if best.map_or(true, |(_, d)| dist < d) {
                        best = Some((t, dist));
                    }
                }

                best
            }
        }
    }

    /// Sample a point along a path (linear or quadratic bezier) at parameter t.
    fn sample_path_at_t(
        &self,
        terrain_renderer: &TerrainRenderer,
        start: WorldPos,
        end: WorldPos,
        control: Option<WorldPos>,
        t: f32,
    ) -> WorldPos {
        let chunk_size = terrain_renderer.chunk_size;
        let mut pos = match control {
            Some(c) => {
                // Quadratic bezier: compute relative to start for precision
                let v_control = c.to_render_pos(start, chunk_size);
                let v_end = end.to_render_pos(start, chunk_size);

                let omt = 1.0 - t;
                // B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
                // P0 is at origin in relative space
                let blend = v_control * (2.0 * omt * t) + v_end * (t * t);
                start.add_vec3(blend, chunk_size)
            }
            None => {
                // Linear interpolation
                start.lerp(end, t, chunk_size)
            }
        };

        // Get height from terrain
        let height = terrain_renderer.get_height_at(pos);
        pos.local.y = height + CLEARANCE;
        pos
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
        control: Option<WorldPos>,
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
            road_style_params.road_type().structure(),
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
            terrain_renderer.chunk_size,
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
                        terrain_renderer.chunk_size,
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
        let Some((end_node_id, end_node_pos)) = self.resolve_anchor(
            storage,
            road_style_params,
            end,
            chunk_id,
            &mut cmds,
            output,
            terrain_renderer.chunk_size,
        ) else {
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
                road_style_params.road_type().structure(),
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
        structure_type: StructureType,
        original_start: WorldPos,
        original_end: WorldPos,
        original_control: Option<WorldPos>,
        from_t: f32,
        to_t: f32,
        from_pos: WorldPos,
        to_pos: WorldPos,
    ) -> Vec<WorldPos> {
        match original_control {
            None => {
                // Straight segment
                make_straight_centerline(terrain_renderer, from_pos, to_pos, structure_type)
            }
            Some(c) => {
                // Extract subsection of Bézier curve
                let (sub_start, sub_control, sub_end) = subdivide_quadratic_bezier(
                    original_start,
                    c,
                    original_end,
                    from_t,
                    to_t,
                    terrain_renderer.chunk_size,
                );

                let est_len = estimate_bezier_arc_length(
                    terrain_renderer,
                    structure_type,
                    sub_start,
                    sub_control,
                    sub_end,
                );
                let samples = compute_curve_segment_count(est_len);
                sample_quadratic_bezier(
                    terrain_renderer,
                    structure_type,
                    sub_start,
                    sub_control,
                    sub_end,
                    samples,
                )
            }
        }
        // trim_polyline_both_ends(segment_centerline.as_slice(), 3)
    }

    // ==================== EXISTING METHODS (unchanged) ====================

    fn find_best_snap(
        &self,
        storage: &RoadStorage,
        terrain_renderer: &TerrainRenderer,
        pos: WorldPos,
    ) -> SnapResult {
        if let Some((node_id, node_pos, dist)) =
            self.find_nearest_node(storage, pos, terrain_renderer.chunk_size)
        {
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

    fn find_nearest_node(
        &self,
        storage: &RoadStorage,
        pos: WorldPos,
        chunk_size: ChunkSize,
    ) -> Option<(NodeId, WorldPos, f32)> {
        let mut best: Option<(NodeId, WorldPos, f32)> = None;

        for (id, node) in storage.iter_enabled_nodes() {
            let node_pos = node.position();
            let dist = pos.distance_to(node_pos, chunk_size);

            if dist < NODE_SNAP_RADIUS {
                if best.is_none() || dist < best.unwrap().2 {
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
        pos: WorldPos,
    ) -> Option<(LaneId, f32, WorldPos, f32)> {
        let lane_id = nearest_lane_to_point(storage, pos, terrain_renderer.chunk_size)?;
        let lane = storage.lane(&lane_id);
        let (t, dist_sq) =
            project_point_to_lane_xz(lane, pos, storage, terrain_renderer.chunk_size)?;
        let dist = dist_sq.sqrt();

        if dist >= LANE_SNAP_RADIUS {
            return None;
        }

        if t < ENDPOINT_T_EPS {
            let node_id = lane.from_node();
            let node = storage.node(node_id)?;
            return Some((lane_id, 0.0, node.position(), dist));
        }

        if t > 1.0 - ENDPOINT_T_EPS {
            let node_id = lane.to_node();
            let node = storage.node(node_id)?;
            return Some((lane_id, 1.0, node.position(), dist));
        }

        let mut p = sample_lane_position(lane, t, storage, terrain_renderer.chunk_size)?;
        p.local.y = terrain_renderer.get_height_at(p) + CLEARANCE;
        Some((lane_id, t, p, dist))
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
        chunk_size: ChunkSize,
    ) -> NodePreview {
        let (result, incoming_lanes, outgoing_lanes) = match snap.kind {
            SnapKind::Node { id } => {
                let mut in_lanes = Vec::new();
                let mut out_lanes = Vec::new();

                let node = storage.node(id).unwrap();

                for lane_id in node.incoming_lanes() {
                    let lane = storage.lane(lane_id);
                    let dir = lane_direction_at_node(lane, id, chunk_size);
                    in_lanes.push((lane_id.clone(), dir));
                }

                for lane_id in node.outgoing_lanes() {
                    let lane = storage.lane(lane_id);
                    let dir = lane_direction_at_node(lane, id, chunk_size);
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
                    let dir = lane.polyline()[0].delta_to(lane.polyline()[1], chunk_size);

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
        let mut p = sample_lane_position(lane, t, storage, terrain_renderer.chunk_size)?;

        let sample_count = 11;
        let mut sample_points = Vec::with_capacity(sample_count);
        for i in 0..sample_count {
            let sample_t = i as f32 / (sample_count - 1) as f32;
            let mut s = sample_lane_position(lane, sample_t, storage, terrain_renderer.chunk_size)?;
            s.local.y = terrain_renderer.get_height_at(s) + CLEARANCE;
            sample_points.push(s);
        }
        p.local.y = terrain_renderer.get_height_at(p) + CLEARANCE;
        Some(LanePreview {
            lane_id,
            projected_t: t,
            projected_point: p,
            sample_points,
        })
    }

    fn validate_placement(
        &self,
        storage: &RoadStorage,
        start: &Anchor,
        end: &Anchor,
        chunk_size: ChunkSize,
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

        let delta = start_pos.delta_to(end_pos, chunk_size);
        let length_xz = delta.xz().length();

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
        chunk_size: ChunkSize,
    ) -> Option<(NodeId, WorldPos)> {
        match &anchor.planned_node {
            PlannedNode::Existing(id) => {
                let node = storage.node(*id)?;
                Some((*id, node.position()))
            }
            PlannedNode::New { pos } => {
                let node_id = self.allocator.alloc_node();
                cmds.push(RoadCommand::AddNode { world_pos: *pos });
                Some((node_id, *pos))
            }
            PlannedNode::Split { lane_id, pos, .. } => {
                let result = self.plan_split(
                    road_style_params,
                    storage,
                    *lane_id,
                    *pos,
                    chunk_id,
                    chunk_size,
                )?;
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
        split_pos: WorldPos,
        chunk_id: ChunkId,
        chunk_size: ChunkSize,
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
            world_pos: split_pos,
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

            let (geom1, geom2) = split_lane_geometry(old_lane.geometry(), split_pos, chunk_size);

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
        centerline: &[WorldPos],
        chunk_id: ChunkId,
    ) {
        let (left_lanes, right_lanes) = road_style_params.road_type().lanes_each_direction();
        let speed = road_style_params.road_type().speed_limit();
        let capacity = road_style_params.road_type().capacity();
        let mask = road_style_params.road_type().vehicle_mask();
        let lane_width = road_style_params.road_type().lane_width;

        // Right side lanes (start -> end)
        for i in 0..right_lanes {
            let lane_index = (i as i8) + 1;
            let poly = offset_polyline(
                terrain_renderer,
                centerline,
                lane_index,
                lane_width,
                road_style_params.road_type().structure,
            );
            let geom = LaneGeometry::from_polyline(poly, terrain_renderer.chunk_size);
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
            let mut poly = offset_polyline(
                terrain_renderer,
                centerline,
                lane_index,
                lane_width,
                road_style_params.road_type().structure,
            );
            poly.reverse();
            let geom = LaneGeometry::from_polyline(poly, terrain_renderer.chunk_size);
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
    start_pos: WorldPos,
    end_pos: WorldPos,
    structure_type: StructureType,
) -> Vec<WorldPos> {
    let length = end_pos.length_to(start_pos, terrain_renderer.chunk_size);
    let samples = (length / METERS_PER_LANE_POLYLINE_STEP).max(1.0) as usize;

    (0..=samples)
        .map(|i| {
            let t = i as f32 / samples as f32;
            let mut p = start_pos.lerp(end_pos, t, terrain_renderer.chunk_size);
            let terrain_y = terrain_renderer.get_height_at(p);
            set_point_height_with_structure_type(terrain_renderer, structure_type, &mut p);
            p
        })
        .collect()
}

// ============================================================================
// Helpers
// ============================================================================

fn split_lane_geometry(
    geom: &LaneGeometry,
    split_pos: WorldPos,
    chunk_size: ChunkSize,
) -> (LaneGeometry, LaneGeometry) {
    let mut best_i = 0;
    let mut best_t = 0.0;
    let mut best_dist = f32::MAX;

    for i in 0..geom.points.len() - 1 {
        let a = geom.points[i];
        let b = geom.points[i + 1];

        // Vector from a -> b
        let ab = a.delta_to(b, chunk_size);

        // Vector from a -> split_pos
        let ap = a.delta_to(split_pos, chunk_size);

        let ab_len2 = ab.length_squared();
        if ab_len2 <= 1e-6 {
            continue;
        }

        let t = (ap.dot(ab) / ab_len2).clamp(0.0, 1.0);

        // p = a + ab * t
        let p = a.add_vec3(ab * t, chunk_size);

        let d = p.distance_to(split_pos, chunk_size);
        if d < best_dist {
            best_dist = d;
            best_i = i;
            best_t = t;
        }
    }

    let a = geom.points[best_i];
    let b = geom.points[best_i + 1];

    let ab = a.delta_to(b, chunk_size);
    let split_point = a.add_vec3(ab * best_t, chunk_size);

    let mut pts1 = Vec::new();
    let mut pts2 = Vec::new();

    pts1.extend_from_slice(&geom.points[..=best_i]);
    pts1.push(split_point);

    pts2.push(split_point);
    pts2.extend_from_slice(&geom.points[best_i + 1..]);

    (
        LaneGeometry::from_polyline(pts1, chunk_size),
        LaneGeometry::from_polyline(pts2, chunk_size),
    )
}

fn lane_direction_at_node(lane: &Lane, node: NodeId, chunk_size: ChunkSize) -> Vec3 {
    let pts = &lane.geometry().points;

    if lane.from_node() == node {
        pts[1].delta_to(pts[0], chunk_size).normalize()
    } else {
        let n = pts.len();
        pts[n - 2].delta_to(pts[n - 1], chunk_size).normalize()
    }
}

/// Sample quadratic Bézier curve with WorldPos control points.
pub fn sample_quadratic_bezier(
    terrain_renderer: &TerrainRenderer,
    structure_type: StructureType,
    p0: WorldPos,
    p1: WorldPos,
    p2: WorldPos,
    segments: usize,
) -> Vec<WorldPos> {
    let segments = segments.max(1);
    let mut points = Vec::with_capacity(segments + 1);

    for i in 0..=segments {
        let t = i as f32 / segments as f32;
        let one_minus_t = 1.0 - t;

        // Quadratic Bézier: B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
        // Compute relative to P0 for precision
        let v1 = p1.to_render_pos(p0, terrain_renderer.chunk_size);
        let v2 = p2.to_render_pos(p0, terrain_renderer.chunk_size);

        let blend = v1 * (2.0 * one_minus_t * t) + v2 * (t * t);
        let mut p = p0.add_vec3(blend, terrain_renderer.chunk_size);

        set_point_height_with_structure_type(terrain_renderer, structure_type, &mut p);
        points.push(p);
    }
    points
}
fn estimate_bezier_arc_length(
    terrain_renderer: &TerrainRenderer,
    structure_type: StructureType,
    p0: WorldPos,
    p1: WorldPos,
    p2: WorldPos,
) -> f32 {
    let samples = sample_quadratic_bezier(terrain_renderer, structure_type, p0, p1, p2, 16);
    polyline_length(&samples, terrain_renderer.chunk_size)
}

fn compute_curve_segment_count(arc_length: f32) -> usize {
    ((arc_length / METERS_PER_LANE_POLYLINE_STEP).ceil() as usize).clamp(4, 32)
}

/// Offset a polyline laterally by lane index.
pub fn offset_polyline(
    terrain_renderer: &TerrainRenderer,
    center: &[WorldPos],
    lane_index: i8,
    lane_width: f32,
    structure_type: StructureType,
) -> Vec<WorldPos> {
    if center.is_empty() {
        return Vec::new();
    }

    let offset = (lane_index as f32 + if lane_index < 0 { 0.5 } else { -0.5 }) * lane_width;

    center
        .iter()
        .enumerate()
        .map(|(i, &pt)| {
            let dir = if i + 1 < center.len() {
                center[i + 1].to_render_pos(pt, terrain_renderer.chunk_size)
            } else {
                pt.to_render_pos(center[i - 1], terrain_renderer.chunk_size)
            };

            let dir_xz = Vec3::new(dir.x, 0.0, dir.z).normalize_or_zero();
            let right = Vec3::new(-dir_xz.z, 0.0, dir_xz.x);
            let mut p = pt.add_vec3(right * offset, terrain_renderer.chunk_size);
            set_point_height_with_structure_type(terrain_renderer, structure_type, &mut p);
            p
        })
        .collect()
}
#[derive(Clone, Debug)]
pub struct IntersectionBuildParams {
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
            straight_angle_deg: 25.0,
            turn_samples: 12,
            dedicate_turn_lanes: true,
            max_turn_angle: 2.74,
            min_turn_radius_m: 5.0,
            clearance_length_m: 0.0,
            lane_width_m: style.road_type().lane_width,
            turn_tightness: style.turn_tightness(),
            side_walk_width: style.road_type().sidewalk_width,
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

pub fn build_intersection_at_node(
    storage: &mut RoadStorage,
    node_id: NodeId,
    params: &IntersectionBuildParams,
    recalc_clearance: bool,
    gizmo: &mut Gizmo,
) {
    let chunk_size = gizmo.chunk_size;
    if recalc_clearance {
        let Some(node) = storage.node(node_id) else {
            return;
        };
        carve_intersection_clearance(
            storage,
            node_id,
            node.connection_count() as f32 * 0.7,
            chunk_size,
        );
        initial_carve(storage, node_id, params, gizmo);
        let demands = probe_intersection_node_lanes(storage, node_id, params, chunk_size);
        carve_intersection_clearance_per_lane(storage, node_id, &demands, chunk_size);
    }

    storage.node_mut(node_id).clear_node_lanes();

    let Some(node) = storage.node(node_id) else {
        return;
    };
    let incoming = node.incoming_lanes();
    let outgoing = node.outgoing_lanes();

    let mut node_lanes = Vec::new();

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

            let in_dir = in_pt
                .to_render_pos(in_poly[in_poly.len() - 2], chunk_size)
                .normalize();
            let out_dir = out_poly[1].to_render_pos(out_pt, chunk_size).normalize();

            let angle_rad = in_dir.dot(out_dir).clamp(-1.0, 1.0).acos();
            if angle_rad > params.max_turn_angle {
                continue;
            }

            let chord = in_pt.distance_to(out_pt, chunk_size);
            let dynamic_tightness = if chord > 25.0 {
                params.turn_tightness * 1.2
            } else if chord < 8.0 {
                params.turn_tightness * 0.7
            } else {
                params.turn_tightness
            };

            let geom = generate_turn_geometry(
                in_pt,
                in_dir,
                out_pt,
                out_dir,
                12,
                dynamic_tightness,
                chunk_size,
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
    let chunk_size = gizmo.chunk_size;
    let center = node.position();

    let incoming: HashSet<LaneId> = node.incoming_lanes().iter().copied().collect();
    let segs = segment_ids_sorted_left_to_right(
        storage.enabled_segments_connected_to_node(node_id),
        storage,
        node_id,
        chunk_size,
    );
    if segs.len() < 2 {
        return;
    }

    let mut cut_vertices: Vec<WorldPos> = Vec::new();
    let n = segs.len();

    for i in 0..n {
        let left_seg = storage.segment(segs[i]);
        let right_seg = storage.segment(segs[(i + 1) % n]);

        let l_lane = find_boundary_lane(storage, left_seg, node_id, true);
        let r_lane = find_boundary_lane(storage, right_seg, node_id, false);

        let (Some(l_lane), Some(r_lane)) = (l_lane, r_lane) else {
            println!(
                "initial_carve: missing boundary lanes for pair {}/{}",
                i,
                (i + 1) % n
            );
            continue;
        };

        let offset = params.lane_width_m * 2.0 + params.side_walk_width;
        let lp = offset_polyline_fixed(&storage.lane(&l_lane).polyline(), offset, chunk_size);
        let rp = offset_polyline_fixed(&storage.lane(&r_lane).polyline(), offset, chunk_size);

        gizmo.polyline(&lp, [0.8, 0.0, 0.2], 2.0, DEBUG_DRAW_DURATION);
        gizmo.polyline(&rp, [0.2, 0.0, 0.8], 2.0, DEBUG_DRAW_DURATION);

        let intersection = polyline_intersection_xz_best(&lp, &rp, center, chunk_size);

        let vertex = match intersection {
            Some(p) => {
                gizmo.cross(p, 0.25, [0.0, 1.0, 0.2], DEBUG_DRAW_DURATION);
                p
            }
            None => {
                let (l_end, _) = get_node_side_end(&lp, center, chunk_size);
                let (r_end, _) = get_node_side_end(&rp, center, chunk_size);
                let fallback = l_end.lerp(r_end, 0.5, chunk_size);
                gizmo.cross(fallback, 0.3, [1.0, 0.5, 0.0], DEBUG_DRAW_DURATION);
                fallback
            }
        };

        let dir = vertex.to_render_pos(center, chunk_size).normalize_or_zero();
        if dir != Vec3::ZERO {
            let shift = 2.2 * params.side_walk_width;
            cut_vertices.push(vertex.add_vec3(dir * shift, chunk_size));
        }
    }

    if cut_vertices.len() < 3 {
        let fallback_radius = params.side_walk_width * 7.0;
        circle_carve_lanes(storage, node_id, &incoming, center, fallback_radius, gizmo);
        return;
    }

    cut_vertices = enforce_ccw_winding(&cut_vertices, center, chunk_size);

    if !is_simple_polygon(&cut_vertices, chunk_size)
        || !point_in_polygon_xz(center, &cut_vertices, chunk_size)
    {
        let fallback_radius = params.side_walk_width * 6.0;
        circle_carve_lanes(storage, node_id, &incoming, center, fallback_radius, gizmo);
        return;
    }

    cut_vertices = smooth_polygon_edges(&cut_vertices, center, 0.5, chunk_size);

    // Debug draw
    let mut cut_dbg = cut_vertices.clone();
    cut_dbg.push(cut_vertices[0]);
    gizmo.polyline(&cut_dbg, [0.0, 0.0, 0.0], 10.0, DEBUG_DRAW_DURATION);

    apply_polygon_carve(storage, &segs, &incoming, &cut_vertices, gizmo, chunk_size);
}
fn find_boundary_lane(
    storage: &RoadStorage,
    segment: &Segment,
    node_id: NodeId,
    is_left: bool,
) -> Option<LaneId> {
    let pts_to_node = segment.end() == node_id;

    segment
        .lanes()
        .iter()
        .copied()
        .filter(|id| storage.lane(id).is_enabled())
        .max_by_key(|id| {
            let idx = storage.lane(id).lane_index();
            let signed = if pts_to_node { idx } else { -idx };
            if is_left { signed } else { -signed }
        })
}
/// Modify polyline by moving start forward by `amount` meters.
pub fn modify_polyline_start(
    points: &[WorldPos],
    amount: f32,
    chunk_size: ChunkSize,
) -> Option<Vec<WorldPos>> {
    if points.len() < 2 {
        return None;
    }

    let lengths = polyline_cumulative_lengths(points, chunk_size);
    let total = *lengths.last().unwrap();

    if amount >= total - 0.01 {
        return None;
    }

    let (new_start, _) = sample_polyline_at(points, &lengths, amount, chunk_size);

    // Find which segment we're in
    let mut i = 1;
    while i < lengths.len() && lengths[i] < amount {
        i += 1;
    }

    let mut out = Vec::with_capacity(points.len() - i + 2);
    out.push(new_start);
    out.extend_from_slice(&points[i..]);

    if out.len() >= 2 { Some(out) } else { None }
}

/// Modify polyline by moving end backward by `amount` meters.
pub fn modify_polyline_end(
    points: &[WorldPos],
    amount: f32,
    chunk_size: ChunkSize,
) -> Option<Vec<WorldPos>> {
    if points.len() < 2 {
        return None;
    }

    let lengths = polyline_cumulative_lengths(points, chunk_size);
    let total = *lengths.last().unwrap();

    if amount >= total - 0.01 {
        return None;
    }

    let target_len = total - amount;
    let (new_end, _) = sample_polyline_at(points, &lengths, target_len, chunk_size);

    // Find which segment we're in
    let mut i = 1;
    while i < lengths.len() && lengths[i] < target_len {
        i += 1;
    }

    let mut out = Vec::with_capacity(i + 1);
    out.extend_from_slice(&points[..i]);
    out.push(new_end);

    if out.len() >= 2 { Some(out) } else { None }
}
/// Extend polyline at start by given amount.
fn extend_polyline_start(
    points: &[WorldPos],
    amount: f32,
    chunk_size: ChunkSize,
) -> Option<Vec<WorldPos>> {
    if points.len() < 2 {
        return None;
    }

    let dir = points[0].to_render_pos(points[1], chunk_size).normalize();
    let new_start = points[0].add_vec3(dir * amount, chunk_size);

    let mut out = Vec::with_capacity(points.len() + 1);
    out.push(new_start);
    out.extend_from_slice(points);
    Some(out)
}

/// Extend polyline at end by given amount.
fn extend_polyline_end(
    points: &[WorldPos],
    amount: f32,
    chunk_size: ChunkSize,
) -> Option<Vec<WorldPos>> {
    if points.len() < 2 {
        return None;
    }

    let n = points.len();
    let dir = points[n - 1]
        .to_render_pos(points[n - 2], chunk_size)
        .normalize();
    let new_end = points[n - 1].add_vec3(dir * amount, chunk_size);

    let mut out = points.to_vec();
    out.push(new_end);
    Some(out)
}

fn segment_ids_sorted_left_to_right(
    segment_ids: Vec<SegmentId>,
    storage: &RoadStorage,
    node_id: NodeId,
    chunk_size: ChunkSize,
) -> Vec<SegmentId> {
    let node = storage.node(node_id).expect("node missing");
    let node_pos = node.position();

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
                polyline[polyline.len() - 2].delta_to(node_pos, chunk_size)
            } else {
                // node is at start -> step from node to the next point (away from node)
                polyline[1].delta_to(node_pos, chunk_size)
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

fn apply_polygon_carve(
    storage: &mut RoadStorage,
    segs: &[SegmentId],
    incoming: &HashSet<LaneId>,
    cut_vertices: &[WorldPos],
    gizmo: &mut Gizmo,
    chunk_size: ChunkSize,
) {
    let mut edits: Vec<(LaneId, LaneGeometry)> = Vec::new();

    for &seg_id in segs {
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
                let rev: Vec<WorldPos> = pts.iter().copied().rev().collect();
                closest_hit_distance_from_start_xz(&rev, cut_vertices, chunk_size)
            } else {
                closest_hit_distance_from_start_xz(&pts, cut_vertices, chunk_size)
            };

            let (amount, hit) = match result {
                Some((a, h)) if a > 0.001 => (a, h),
                _ => continue,
            };

            gizmo.cross(hit, 0.2, [1.0, 0.0, 0.0], DEBUG_DRAW_DURATION);

            let new_pts = if is_incoming {
                modify_polyline_end(&pts, amount, chunk_size)
            } else {
                modify_polyline_start(&pts, amount, chunk_size)
            };

            if let Some(new_pts) = new_pts {
                edits.push((lane_id, LaneGeometry::from_polyline(new_pts, chunk_size)));
            }
        }
    }

    for (lane_id, geom) in edits {
        storage.lane_mut(lane_id).replace_geometry(geom);
    }
}

fn carve_intersection_clearance(
    storage: &mut RoadStorage,
    node_id: NodeId,
    r: f32,
    chunk_size: ChunkSize,
) {
    let (center, incoming_ids, outgoing_ids) = {
        let Some(n) = storage.node(node_id) else {
            return;
        };
        let pos = n.position();
        let incoming: Vec<LaneId> = n.incoming_lanes().iter().copied().collect();
        let outgoing: Vec<LaneId> = n.outgoing_lanes().iter().copied().collect();
        (pos, incoming, outgoing)
    };

    let mut edits: Vec<(LaneId, LaneGeometry, f32)> = Vec::new();

    // Incoming: trim END
    for lane_id in &incoming_ids {
        let lane = storage.lane(lane_id);
        if !lane.is_enabled() {
            continue;
        }

        let old = lane.geometry();
        let old_len = old.total_len.max(0.001);
        let old_cost = lane.base_cost();

        if let Some(pts) = trim_polyline_end_to_radius(&lane.polyline(), center, r, chunk_size) {
            let new_geom = LaneGeometry::from_polyline(pts, chunk_size);
            let ratio = (new_geom.total_len / old_len).clamp(0.1, 1.0);
            edits.push((*lane_id, new_geom, old_cost * ratio));
        }
    }

    // Outgoing: trim START
    for lane_id in &outgoing_ids {
        let lane = storage.lane(lane_id);
        if !lane.is_enabled() {
            continue;
        }

        let old = lane.geometry();
        let old_len = old.total_len.max(0.001);
        let old_cost = lane.base_cost();

        if let Some(pts) = trim_polyline_start_to_radius(&lane.polyline(), center, r, chunk_size) {
            let new_geom = LaneGeometry::from_polyline(pts, chunk_size);
            let ratio = (new_geom.total_len / old_len).clamp(0.1, 1.0);
            edits.push((*lane_id, new_geom, old_cost * ratio));
        }
    }

    for (lane_id, new_geom, new_cost) in edits {
        let lane = storage.lane_mut(lane_id);
        lane.replace_geometry(new_geom);
        lane.replace_base_cost(new_cost);
    }
}
/// Trim polyline START to be at distance r from center.
pub fn trim_polyline_start_to_radius(
    points: &[WorldPos],
    center: WorldPos,
    r: f32,
    chunk_size: ChunkSize,
) -> Option<Vec<WorldPos>> {
    if points.len() < 2 {
        return None;
    }

    let eps = 0.05;
    let start_dist = points[0].distance_to(center, chunk_size);

    if start_dist >= r - eps {
        return None;
    }

    for i in 0..points.len() - 1 {
        let a = points[i];
        let b = points[i + 1];
        let da = a.distance_to(center, chunk_size);
        let db = b.distance_to(center, chunk_size);

        if da <= r && db >= r {
            if let Some(t) = seg_sphere_intersection_t_world(a, b, center, r, chunk_size) {
                let p = a.lerp(b, t, chunk_size);
                let mut out = Vec::with_capacity(points.len() - i);
                out.push(p);
                out.extend_from_slice(&points[i + 1..]);
                return if out.len() >= 2 { Some(out) } else { None };
            }
        }
    }
    None
}

/// Trim polyline END to be at distance r from center.
pub fn trim_polyline_end_to_radius(
    points: &[WorldPos],
    center: WorldPos,
    r: f32,
    chunk_size: ChunkSize,
) -> Option<Vec<WorldPos>> {
    if points.len() < 2 {
        return None;
    }

    let eps = 0.05;
    let end_dist = points[points.len() - 1].distance_to(center, chunk_size);

    if end_dist >= r - eps {
        return None;
    }

    for i in (0..points.len() - 1).rev() {
        let a = points[i];
        let b = points[i + 1];
        let da = a.distance_to(center, chunk_size);
        let db = b.distance_to(center, chunk_size);

        if da >= r && db <= r {
            if let Some(t) = seg_sphere_intersection_t_world(a, b, center, r, chunk_size) {
                let p = a.lerp(b, t, chunk_size);
                let mut out = Vec::with_capacity(i + 2);
                out.extend_from_slice(&points[..=i]);
                out.push(p);
                return if out.len() >= 2 { Some(out) } else { None };
            }
        }
    }
    None
}
/// Generates smooth turn geometry using a cubic Bézier curve.
/// `tightness` < 1.0 makes turns tighter, > 1.0 makes them wider.
/// Generate smooth turn geometry using cubic Bézier.
pub fn generate_turn_geometry(
    start: WorldPos,
    start_dir: Vec3,
    end: WorldPos,
    end_dir: Vec3,
    samples: usize,
    tightness: f32,
    chunk_size: ChunkSize,
) -> LaneGeometry {
    let dist = start.distance_to(end, chunk_size);

    if dist < 0.001 {
        return LaneGeometry::from_polyline(vec![start, end], chunk_size);
    }

    let control_len = (dist * 0.35) * tightness;

    let ctrl1 = start.add_vec3(start_dir * control_len, chunk_size);
    let ctrl2 = end.add_vec3(-end_dir * control_len, chunk_size);

    let n = samples.clamp(2, 64);
    let points: Vec<WorldPos> = (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32;
            bezier3(start, ctrl1, ctrl2, end, t, chunk_size)
        })
        .collect();

    LaneGeometry::from_polyline(points, chunk_size)
}
/// Offset a polyline by a fixed distance (for intersection carving).
pub fn offset_polyline_fixed(
    polyline: &[WorldPos],
    offset: f32,
    chunk_size: ChunkSize,
) -> Vec<WorldPos> {
    if polyline.len() < 2 {
        return polyline.to_vec();
    }

    polyline
        .iter()
        .enumerate()
        .map(|(i, &pt)| {
            let dir = if i + 1 < polyline.len() {
                polyline[i + 1].to_render_pos(pt, chunk_size)
            } else {
                pt.to_render_pos(polyline[i - 1], chunk_size)
            };

            let dir_xz = Vec3::new(dir.x, 0.0, dir.z).normalize_or_zero();
            let right = Vec3::new(-dir_xz.z, 0.0, dir_xz.x);
            pt.add_vec3(right * offset, chunk_size)
        })
        .collect()
}

/// Compute total length of a polyline.
pub fn polyline_length(points: &[WorldPos], chunk_size: ChunkSize) -> f32 {
    points
        .windows(2)
        .map(|w| w[0].distance_to(w[1], chunk_size))
        .sum()
}

/// Compute cumulative lengths along polyline.
pub fn polyline_cumulative_lengths(points: &[WorldPos], chunk_size: ChunkSize) -> Vec<f32> {
    let mut lengths = Vec::with_capacity(points.len());
    lengths.push(0.0);
    for w in points.windows(2) {
        lengths.push(lengths.last().unwrap() + w[0].distance_to(w[1], chunk_size));
    }
    lengths
}

/// Sample position and direction at distance t along polyline.
pub fn sample_polyline_at(
    points: &[WorldPos],
    lengths: &[f32],
    t: f32,
    chunk_size: ChunkSize,
) -> (WorldPos, Vec3) {
    let mut i = 1;
    while i < lengths.len() && lengths[i] < t {
        i += 1;
    }

    let i0 = i - 1;
    let i1 = i.min(points.len() - 1);

    let seg_t = if lengths[i1] > lengths[i0] {
        (t - lengths[i0]) / (lengths[i1] - lengths[i0])
    } else {
        0.0
    };

    let pos = points[i0].lerp(points[i1], seg_t, chunk_size);
    let dir = points[i1]
        .to_render_pos(points[i0], chunk_size)
        .normalize_or_zero();
    (pos, dir)
}
/// Find t where segment a->b crosses sphere of radius r around center.
fn seg_sphere_intersection_t_world(
    a: WorldPos,
    b: WorldPos,
    center: WorldPos,
    r: f32,
    chunk_size: ChunkSize,
) -> Option<f32> {
    let oa = a.to_render_pos(center, chunk_size);
    let ab = b.to_render_pos(a, chunk_size);

    let a_coef = ab.dot(ab);
    let b_coef = 2.0 * oa.dot(ab);
    let c_coef = oa.dot(oa) - r * r;

    let disc = b_coef * b_coef - 4.0 * a_coef * c_coef;
    if disc < 0.0 {
        return None;
    }

    let sqrt_disc = disc.sqrt();
    let t1 = (-b_coef - sqrt_disc) / (2.0 * a_coef);
    let t2 = (-b_coef + sqrt_disc) / (2.0 * a_coef);

    // Return first valid t in [0, 1]
    if (0.0..=1.0).contains(&t1) {
        Some(t1)
    } else if (0.0..=1.0).contains(&t2) {
        Some(t2)
    } else {
        None
    }
}

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
/// Probe intersection to determine lane trimming demands.
pub fn probe_intersection_node_lanes(
    storage: &RoadStorage,
    node_id: NodeId,
    params: &IntersectionBuildParams,
    chunk_size: ChunkSize,
) -> HashMap<LaneId, f32> {
    let Some(node) = storage.node(node_id) else {
        return HashMap::new();
    };

    let mut lane_demands: HashMap<LaneId, f32> = HashMap::new();

    let incoming = node.incoming_lanes();
    let outgoing = node.outgoing_lanes();

    // Collect connection info
    struct ConnectionInfo {
        in_id: LaneId,
        out_id: LaneId,
        in_pt: WorldPos,
        out_pt: WorldPos,
        turn_type: TurnType,
        angle_rad: f32,
    }

    let mut connections = Vec::new();
    let mut center_accum = Vec3::ZERO;
    let mut center_count = 0u32;

    // Use first valid point as reference for center calculation
    let mut reference_pos: Option<WorldPos> = None;

    for in_id in incoming {
        for out_id in outgoing {
            let in_lane = storage.lane(in_id);
            let out_lane = storage.lane(out_id);
            if !in_lane.is_enabled() || !out_lane.is_enabled() {
                continue;
            }

            let in_poly = in_lane.polyline();
            let out_poly = out_lane.polyline();
            if in_poly.len() < 2 || out_poly.len() < 2 {
                continue;
            }

            let in_pt = *in_poly.last().unwrap();
            let out_pt = *out_poly.first().unwrap();

            // Set reference for accumulation
            let ref_pos = *reference_pos.get_or_insert(in_pt);

            let in_dir = in_pt
                .to_render_pos(in_poly[in_poly.len() - 2], chunk_size)
                .normalize();
            let out_dir = out_poly[1].to_render_pos(out_pt, chunk_size).normalize();

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

            // Accumulate center relative to reference
            center_accum += in_pt.to_render_pos(ref_pos, chunk_size);
            center_accum += out_pt.to_render_pos(ref_pos, chunk_size);
            center_count += 2;
        }
    }

    // Compute intersection center
    let intersection_center = if center_count > 0 && reference_pos.is_some() {
        let avg = center_accum / center_count as f32;
        reference_pos.unwrap().add_vec3(avg, chunk_size)
    } else {
        node.position()
    };

    // Evaluate demands
    for conn in &connections {
        let mut demand_m = 0.0f32;

        match conn.turn_type {
            TurnType::Straight => {
                let gap = conn.in_pt.distance_to(conn.out_pt, chunk_size);
                if gap > params.lane_width_m * 5.0 {
                    demand_m = -((gap - params.lane_width_m) * 0.4);
                }
            }
            TurnType::Right => {
                let theta = conn.angle_rad;
                if theta > 0.1 {
                    let required_chord = 2.0 * params.min_turn_radius_m * (theta * 0.5).sin();
                    let current_chord = conn.in_pt.distance_to(conn.out_pt, chunk_size);

                    if current_chord < required_chord {
                        let deficit = required_chord - current_chord;
                        demand_m = (deficit * 8.7).max(0.0);
                    } else if current_chord > required_chord * 1.5 {
                        let excess = current_chord - (required_chord * 1.2);
                        demand_m = -(excess * 0.3);
                    }
                }
            }
            _ => {}
        }

        // Apply demand to both lanes
        apply_lane_demand(conn.in_id, demand_m, &mut lane_demands);
        apply_lane_demand(conn.out_id, demand_m, &mut lane_demands);
    }

    lane_demands
}
#[inline]
fn apply_lane_demand(lane_id: LaneId, demand: f32, map: &mut HashMap<LaneId, f32>) {
    map.entry(lane_id)
        .and_modify(|curr| {
            if *curr > 0.0 && demand > 0.0 {
                *curr = curr.max(demand);
            } else if *curr < 0.0 && demand < 0.0 {
                *curr = curr.max(demand); // Closer to 0 = less extension
            } else if demand > 0.0 {
                *curr = demand; // Trim overrides extension
            }
        })
        .or_insert(demand);
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
    chunk_size: ChunkSize,
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
            modify_polyline_end(&lane.geometry().points, *amount, chunk_size)
        } else {
            // Outgoing lanes get modified at the START
            modify_polyline_start(&lane.geometry().points, *amount, chunk_size)
        };

        if let Some(pts) = new_pts {
            edits.push((*lane_id, LaneGeometry::from_polyline(pts, chunk_size)));
        }
    }

    for (id, geom) in edits {
        storage.lane_mut(id).replace_geometry(geom);
    }
}

/// Smooth polygon by adding midpoints that curve outward slightly
fn smooth_polygon_edges(
    vertices: &[WorldPos],
    center: WorldPos,
    bulge_factor: f32,
    chunk_size: ChunkSize,
) -> Vec<WorldPos> {
    let n = vertices.len();
    if n < 3 {
        return vertices.to_vec();
    }

    let mut result = Vec::with_capacity(n * 2);

    for i in 0..n {
        let a = vertices[i];
        let b = vertices[(i + 1) % n];

        result.push(a);

        // Convert to center-relative Vec3
        let va = center.delta_to(a, chunk_size);
        let vb = center.delta_to(b, chunk_size);

        let mid = (va + vb) * 0.5;
        let edge = vb - va;

        let edge_len = edge.length();
        if edge_len < 1e-6 {
            continue;
        }

        let bulge = (edge_len * bulge_factor).min(3.0);
        let outward = mid.normalize_or_zero();
        let curved_mid = mid + outward * bulge;

        result.push(center.add_vec3(curved_mid, chunk_size));
    }

    result
}

/// Find the best intersection between two polylines near center.
pub fn polyline_intersection_xz_best(
    poly_a: &[WorldPos],
    poly_b: &[WorldPos],
    center: WorldPos,
    chunk_size: ChunkSize,
) -> Option<WorldPos> {
    let mut best: Option<(f32, WorldPos)> = None;

    for i in 0..poly_a.len().saturating_sub(1) {
        for j in 0..poly_b.len().saturating_sub(1) {
            if let Some((t, _)) = segment_intersection_xz(
                poly_a[i],
                poly_a[i + 1],
                poly_b[j],
                poly_b[j + 1],
                chunk_size,
            ) {
                let hit = poly_a[i].lerp(poly_a[i + 1], t, chunk_size);
                let dist = hit.distance_to(center, chunk_size);

                if best.map_or(true, |(d, _)| dist < d) {
                    best = Some((dist, hit));
                }
            }
        }
    }
    best.map(|(_, p)| p)
}
/// Returns (node-side point, next point going away from node)
/// Uses more points for a stable direction if available
fn get_node_side_end(
    points: &[WorldPos],
    center: WorldPos,
    chunk_size: ChunkSize,
) -> (WorldPos, WorldPos) {
    if points.len() < 2 {
        let p = if points.is_empty() {
            WorldPos::zero()
        } else {
            points[0]
        };
        return (p, p);
    }

    let d_start = points[0].distance_to(center, chunk_size);
    let d_end = points[points.len() - 1].distance_to(center, chunk_size);

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
/// Ray A: from a_origin in direction toward a_toward.
/// Ray B: from b_origin in direction toward b_toward.
/// Returns the intersection point as WorldPos.
pub fn project_rays_intersection_xz(
    a_origin: WorldPos,
    a_toward: WorldPos,
    b_origin: WorldPos,
    b_toward: WorldPos,
    chunk_size: ChunkSize,
) -> Option<WorldPos> {
    // Compute directions relative to origins
    let da = a_toward.to_render_pos(a_origin, chunk_size);
    let db = b_toward.to_render_pos(b_origin, chunk_size);

    let da_len_sq = da.x * da.x + da.z * da.z;
    let db_len_sq = db.x * db.x + db.z * db.z;

    if da_len_sq < 1e-12 || db_len_sq < 1e-12 {
        return None;
    }

    let da_len = da_len_sq.sqrt();
    let db_len = db_len_sq.sqrt();

    // Normalize to unit vectors
    let da_x = da.x / da_len;
    let da_z = da.z / da_len;
    let db_x = db.x / db_len;
    let db_z = db.z / db_len;

    // Determinant (cross product of unit direction vectors)
    let det = da_x * db_z - da_z * db_x;

    if det.abs() < 1e-6 {
        return None; // Parallel or nearly parallel
    }

    // Difference from a_origin to b_origin
    let diff = b_origin.to_render_pos(a_origin, chunk_size);

    // t and u are arc-length distances since we're using unit vectors
    let t = (diff.x * db_z - diff.z * db_x) / det;
    let u = (diff.x * da_z - diff.z * da_x) / det;

    // Check if intersection is ahead of both ray origins
    if t >= -1e-4 && u >= -1e-4 {
        // Compute intersection point
        let offset = Vec3::new(t * da_x, 0.0, t * da_z);

        // Average Y from both rays
        let y_a = a_origin.local.y + da.y * (t / da_len);
        let y_b = b_origin.local.y + db.y * (u / db_len);
        let avg_y = (y_a + y_b) * 0.5;

        let mut result = a_origin.add_vec3(offset, chunk_size);
        result.local.y = avg_y;

        Some(result)
    } else {
        None
    }
}

/// Find the closest intersection from polyline start with polygon.
pub fn closest_hit_distance_from_start_xz(
    polyline: &[WorldPos],
    polygon: &[WorldPos],
    chunk_size: ChunkSize,
) -> Option<(f32, WorldPos)> {
    let lengths = polyline_cumulative_lengths(polyline, chunk_size);
    let mut best: Option<(f32, WorldPos)> = None;

    for i in 0..polyline.len() - 1 {
        let seg_a = polyline[i];
        let seg_b = polyline[i + 1];

        for j in 0..polygon.len() {
            let poly_a = polygon[j];
            let poly_b = polygon[(j + 1) % polygon.len()];

            if let Some((t, _)) = segment_intersection_xz(seg_a, seg_b, poly_a, poly_b, chunk_size)
            {
                let dist = lengths[i] + t * seg_a.distance_to(seg_b, chunk_size);
                let hit = seg_a.lerp(seg_b, t, chunk_size);

                if best.map_or(true, |(d, _)| dist < d) {
                    best = Some((dist, hit));
                }
            }
        }
    }
    best
}

/// Enforce CCW winding.
pub fn enforce_ccw_winding(
    verts: &[WorldPos],
    center: WorldPos,
    chunk_size: ChunkSize,
) -> Vec<WorldPos> {
    if is_ccw_polygon(verts, center, chunk_size) {
        verts.to_vec()
    } else {
        verts.iter().copied().rev().collect()
    }
}
/// Check if polygon is CCW wound (XZ plane).
pub fn is_ccw_polygon(verts: &[WorldPos], center: WorldPos, chunk_size: ChunkSize) -> bool {
    let mut sum = 0.0;
    for i in 0..verts.len() {
        let a = verts[i].to_render_pos(center, chunk_size);
        let b = verts[(i + 1) % verts.len()].to_render_pos(center, chunk_size);
        sum += (b.x - a.x) * (b.z + a.z);
    }
    sum < 0.0
}
/// Check if polygon is simple (no self-intersections).
pub fn is_simple_polygon(verts: &[WorldPos], chunk_size: ChunkSize) -> bool {
    let n = verts.len();
    if n < 3 {
        return false;
    }

    for i in 0..n {
        let a1 = verts[i];
        let a2 = verts[(i + 1) % n];

        for j in i + 2..n {
            if j == (i + n - 1) % n {
                continue;
            } // Skip adjacent edges

            let b1 = verts[j];
            let b2 = verts[(j + 1) % n];

            if segments_intersect_xz(a1, a2, b1, b2, chunk_size) {
                return false;
            }
        }
    }
    true
}

/// Check if two segments intersect (XZ plane).
fn segments_intersect_xz(
    a1: WorldPos,
    a2: WorldPos,
    b1: WorldPos,
    b2: WorldPos,
    chunk_size: ChunkSize,
) -> bool {
    let d1 = a2.to_render_pos(a1, chunk_size);
    let d2 = b2.to_render_pos(b1, chunk_size);
    let d12 = b1.to_render_pos(a1, chunk_size);

    let cross = d1.x * d2.z - d1.z * d2.x;
    if cross.abs() < 1e-10 {
        return false;
    }

    let t = (d12.x * d2.z - d12.z * d2.x) / cross;
    let u = (d12.x * d1.z - d12.z * d1.x) / cross;

    (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u)
}

/// Check if point is inside polygon (XZ plane).
pub fn point_in_polygon_xz(pt: WorldPos, verts: &[WorldPos], chunk_size: ChunkSize) -> bool {
    let mut inside = false;
    let n = verts.len();

    for i in 0..n {
        let a = verts[i].to_render_pos(pt, chunk_size);
        let b = verts[(i + 1) % n].to_render_pos(pt, chunk_size);

        if (a.z > 0.0) != (b.z > 0.0) {
            let x_cross = a.x + (0.0 - a.z) / (b.z - a.z) * (b.x - a.x);
            if x_cross > 0.0 {
                inside = !inside;
            }
        }
    }
    inside
}

fn find_circle_intersection_distance(
    points: &[WorldPos],
    center: WorldPos,
    radius: f32,
    chunk_size: ChunkSize,
) -> Option<(f32, WorldPos)> {
    if points.len() < 2 {
        return None;
    }

    let mut acc = 0.0f32;

    for i in 0..points.len() - 1 {
        let a = points[i];
        let b = points[i + 1];

        // segment vector a -> b
        let ab = a.delta_to(b, chunk_size);
        let seg_len = ab.length();
        if seg_len < 1e-6 {
            continue;
        }

        let d = ab / seg_len;

        // vector from center -> a
        let f = center.delta_to(a, chunk_size);

        // quadratic coefficients in XZ plane
        let a_coef = d.x * d.x + d.z * d.z;
        let b_coef = 2.0 * (f.x * d.x + f.z * d.z);
        let c_coef = f.x * f.x + f.z * f.z - radius * radius;

        let discriminant = b_coef * b_coef - 4.0 * a_coef * c_coef;
        if discriminant < 0.0 {
            acc += seg_len;
            continue;
        }

        let sqrt_disc = discriminant.sqrt();
        let inv_denom = 1.0 / (2.0 * a_coef);

        let t1 = (-b_coef - sqrt_disc) * inv_denom;
        let t2 = (-b_coef + sqrt_disc) * inv_denom;

        for t in [t1, t2] {
            if t >= 0.0 && t <= seg_len {
                let hit = a.add_vec3(d * t, chunk_size);
                return Some((acc + t, hit));
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
    center: WorldPos,
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
                let rev: Vec<WorldPos> = pts.iter().copied().rev().collect();
                find_circle_intersection_distance(&rev, center, radius, gizmo.chunk_size)
            } else {
                find_circle_intersection_distance(pts, center, radius, gizmo.chunk_size)
            }
            .unwrap_or((0.0, WorldPos::zero()));

            if amount <= 0.001 || hit == WorldPos::zero() {
                continue;
            }

            gizmo.cross(hit, 0.2, [0.0, 1.0, 0.0], DEBUG_DRAW_DURATION);

            let new_pts = if is_incoming {
                modify_polyline_end(pts, amount, gizmo.chunk_size)
            } else {
                modify_polyline_start(pts, amount, gizmo.chunk_size)
            };

            if let Some(new_pts) = new_pts {
                edits.push((
                    lane_id,
                    LaneGeometry::from_polyline(new_pts, gizmo.chunk_size),
                ));
            }
        }
    }

    for (lane_id, geom) in edits {
        storage.lane_mut(lane_id).replace_geometry(geom);
    }
}
