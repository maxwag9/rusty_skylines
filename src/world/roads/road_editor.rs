use crate::helpers::positions::{ChunkSize, LocalPos, WorldPos};
use crate::renderer::gizmo::Gizmo;
use crate::ui::input::InputState;
use crate::world::roads::intersections::IntersectionBuildParams;
use crate::world::roads::road_helpers::*;
use crate::world::roads::road_mesh_manager::{CLEARANCE, ChunkId};
use crate::world::roads::road_structs::*;
use crate::world::roads::roads::{
    Lane, LaneGeometry, METERS_PER_LANE_POLYLINE_STEP, RoadCommand, RoadManager, RoadStorage,
    nearest_lane_to_point, project_point_to_lane_xz, sample_lane_position,
};
use crate::world::terrain::terrain_subsystem::{CursorMode, TerrainSubsystem};
use glam::{Vec2, Vec3, Vec3Swizzles};
use std::collections::HashSet;

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
    pub(crate) style: RoadStyleParams,
}

impl RoadEditor {
    pub fn new() -> Self {
        Self {
            allocator: IdAllocator::new(),
            style: RoadStyleParams::default(),
        }
    }

    pub fn update(
        &mut self,
        road_manager: &RoadManager,
        terrain_renderer: &TerrainSubsystem,
        input: &mut InputState,
        gizmo: &mut Gizmo,
    ) -> Vec<RoadEditorCommand> {
        let Some(road_type) = (match terrain_renderer.cursor.mode {
            CursorMode::Roads(r) => Some(r),
            _ => None,
        }) else {
            return Vec::new();
        };
        self.style.set_road_type(road_type);
        self.allocator.update(&road_manager.roads);
        let storage = &road_manager.roads;
        let mut output = Vec::new();

        if input.action_pressed_once("Cancel") {
            self.style.set_to_idle();
            output.push(RoadEditorCommand::PreviewClear);
            return output;
        }

        let Some(picked) = &terrain_renderer.last_picked else {
            output.push(RoadEditorCommand::PreviewError(PreviewError::NoPickedPoint));
            output.push(RoadEditorCommand::PreviewClear);
            return output;
        };

        let chunk_id = picked.chunk.id;
        let snap = self.find_best_snap(storage, terrain_renderer, picked.pos, gizmo);

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

        match self.style.state().clone() {
            EditorState::Idle => {
                self.handle_idle(
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
                    &start,
                    &snap,
                    place_pressed,
                    chunk_id,
                    &mut output,
                    gizmo,
                );
            }
            EditorState::CurvePickControl { start } => {
                self.handle_curve_pick_control(
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
                    &start,
                    control,
                    &snap,
                    place_pressed,
                    chunk_id,
                    &mut output,
                    gizmo,
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
        chunk_size: ChunkSize,
    ) {
        let node_preview = self.build_node_preview_from_snap(storage, snap, chunk_size);
        output.push(RoadEditorCommand::PreviewNode(node_preview));

        if place_pressed {
            let anchor = self.build_anchor_from_snap(snap);
            match self.style.mode() {
                BuildMode::Straight => {
                    self.style
                        .set_state(EditorState::StraightPickEnd { start: anchor });
                }
                BuildMode::Curved => {
                    self.style
                        .set_state(EditorState::CurvePickControl { start: anchor });
                }
            }
        }
    }

    fn handle_straight_pick_end(
        &mut self,
        terrain_renderer: &TerrainSubsystem,
        storage: &RoadStorage,
        start: &Anchor,
        snap: &SnapResult,
        place_pressed: bool,
        chunk_id: ChunkId,
        output: &mut Vec<RoadEditorCommand>,
        gizmo: &mut Gizmo,
    ) {
        let Some(start_pos) = start.planned_node.position(storage) else {
            output.push(RoadEditorCommand::PreviewError(
                PreviewError::MissingNodeData,
            ));
            self.style.set_state(EditorState::Idle);
            return;
        };

        let end_anchor = self.build_anchor_from_snap(snap);
        let end_pos = snap.world_pos;
        let polyline = make_straight_centerline(
            terrain_renderer,
            start_pos,
            end_pos,
            self.style.road_type().structure,
        );
        let estimated_length = end_pos.length_to(start_pos, terrain_renderer.chunk_size);

        let (is_valid, reason) =
            self.validate_placement(storage, start, &end_anchor, terrain_renderer.chunk_size);

        // Find crossings for preview
        let crossings = self.find_all_crossings(
            storage,
            terrain_renderer,
            self.style.road_type().structure(),
            start_pos,
            end_pos,
            None,
            start,
            &end_anchor,
            gizmo,
        );
        // println!("{:?}", reason);
        let seg_preview = SegmentPreview {
            road_type: self.style.road_type().clone(),
            mode: self.style.mode(),
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
            lane_count_each_dir: self.style.road_type().lanes_each_direction(),
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
                start,
                &end_anchor,
                None,
                chunk_id,
                output,
                gizmo,
            );
            for cmd in road_cmds {
                output.push(RoadEditorCommand::Road(cmd));
            }
            self.style.set_to_idle();
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
        chunk_size: ChunkSize,
    ) {
        let Some(start_pos) = start.planned_node.position(storage) else {
            output.push(RoadEditorCommand::PreviewError(
                PreviewError::MissingNodeData,
            ));
            self.style.set_to_idle();
            return;
        };

        let control_pos = snap.world_pos;
        let polyline = vec![start_pos, control_pos];
        let estimated_length = control_pos.length_to(start_pos, chunk_size);

        let seg_preview = SegmentPreview {
            road_type: self.style.road_type().clone(),
            mode: self.style.mode(),
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
            lane_count_each_dir: self.style.road_type().lanes_each_direction(),
            estimated_length,
            crossing_count: 0,
        };
        output.push(RoadEditorCommand::PreviewSegment(seg_preview));

        if place_pressed {
            self.style.set_state(EditorState::CurvePickEnd {
                start: start.clone(),
                control: control_pos,
            });
        }
    }

    fn handle_curve_pick_end(
        &mut self,
        terrain_renderer: &TerrainSubsystem,
        storage: &RoadStorage,
        start: &Anchor,
        control: WorldPos,
        snap: &SnapResult,
        place_pressed: bool,
        chunk_id: ChunkId,
        output: &mut Vec<RoadEditorCommand>,
        gizmo: &mut Gizmo,
    ) {
        let Some(start_pos) = start.planned_node.position(storage) else {
            output.push(RoadEditorCommand::PreviewError(
                PreviewError::MissingNodeData,
            ));
            self.style.set_to_idle();
            return;
        };
        let chunk_size = terrain_renderer.chunk_size;
        let end_anchor = self.build_anchor_from_snap(snap);
        let end_pos = snap.world_pos;

        let estimated_length = estimate_bezier_arc_length(
            terrain_renderer,
            self.style.road_type().structure(),
            start_pos,
            control,
            end_pos,
        );
        let segment_count = compute_curve_segment_count(estimated_length);
        let polyline = sample_quadratic_bezier(
            terrain_renderer,
            self.style.road_type().structure(),
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
            self.style.road_type().structure(),
            start_pos,
            end_pos,
            Some(control),
            start,
            &end_anchor,
            gizmo,
        );

        let seg_preview = SegmentPreview {
            road_type: self.style.road_type().clone(),
            mode: self.style.mode(),
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
            lane_count_each_dir: self.style.road_type().lanes_each_direction(),
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
                start,
                &end_anchor,
                Some(control),
                chunk_id,
                output,
                gizmo,
            );
            for cmd in road_cmds {
                output.push(RoadEditorCommand::Road(cmd));
            }
            self.style.set_to_idle()
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
        terrain_renderer: &TerrainSubsystem,
        structure_type: StructureType,
        start_pos: WorldPos,
        end_pos: WorldPos,
        control: Option<WorldPos>,
        start_anchor: &Anchor,
        end_anchor: &Anchor,
        gizmo: &mut Gizmo,
    ) -> Vec<CrossingPoint> {
        let chunk_size = terrain_renderer.chunk_size;
        let mut crossings = Vec::new();
        let mut crossed_segments: HashSet<SegmentId> = HashSet::new();

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

        let excluded_segments = self.get_excluded_segments(storage, start_anchor, end_anchor);

        for (lane_id, _) in storage.iter_enabled_lanes() {
            let lane = storage.lane(&lane_id);
            let seg_id = lane.segment();

            if crossed_segments.contains(&seg_id) {
                continue;
            }

            if excluded_segments.contains(&seg_id) {
                continue;
            }

            if self
                .find_lane_crossing_point(
                    storage,
                    terrain_renderer,
                    &test_polyline,
                    start_pos,
                    end_pos,
                    control,
                    &lane_id,
                )
                .is_none()
            {
                continue;
            }

            if let Some(crossing) = self.find_segment_center_crossing(
                storage,
                terrain_renderer,
                &test_polyline,
                start_pos,
                end_pos,
                control,
                seg_id,
            ) {
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
                    let (closest_node_id, closest_pos) = if dist_to_start <= dist_to_end {
                        (segment.start, start_node_pos)
                    } else {
                        (segment.end, end_node_pos)
                    };

                    if let Some((new_t, _proj_dist)) = self.project_point_to_path(
                        start_pos,
                        end_pos,
                        control,
                        closest_pos,
                        chunk_size,
                    ) {
                        if new_t > ENDPOINT_T_EPS && new_t < 1.0 - ENDPOINT_T_EPS {
                            crossings.push(CrossingPoint {
                                t: new_t,
                                world_pos: closest_pos,
                                kind: CrossingKind::ExistingNode(closest_node_id),
                            });
                            crossed_segments.insert(seg_id);
                            continue;
                        }
                    }
                }

                if crossing.t > ENDPOINT_T_EPS && crossing.t < 1.0 - ENDPOINT_T_EPS {
                    crossed_segments.insert(seg_id);
                    crossings.push(crossing);
                }
            }
        }

        for (node_id, node) in storage.iter_enabled_nodes() {
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

        crossings.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));

        let crossings = self.deduplicate_crossings(crossings);

        // for crossing in crossings.iter() {
        //     gizmo.cross(crossing.world_pos, 1.0, [0.0, 1.0, 1.0], 10.0);
        // }
        crossings
    }

    fn find_segment_center_crossing(
        &self,
        storage: &RoadStorage,
        terrain_renderer: &TerrainSubsystem,
        test_polyline: &[WorldPos],
        start_pos: WorldPos,
        end_pos: WorldPos,
        control: Option<WorldPos>,
        segment_id: SegmentId,
    ) -> Option<CrossingPoint> {
        let segment = storage.segment(segment_id);

        let mut lane_plus_1: Option<(LaneId, &Lane)> = None;
        let mut lane_minus_1: Option<(LaneId, &Lane)> = None;
        let mut closest_to_zero: Option<(LaneId, &Lane)> = None;
        let mut closest_abs_idx = i8::MAX;

        for lane_id in segment.lanes() {
            let lane = storage.lane(lane_id);
            if !lane.is_enabled() {
                continue;
            }

            let idx = lane.lane_index();
            if idx == 1 {
                lane_plus_1 = Some((*lane_id, lane));
            } else if idx == -1 {
                lane_minus_1 = Some((*lane_id, lane));
            }

            if idx.abs() < closest_abs_idx {
                closest_abs_idx = idx.abs();
                closest_to_zero = Some((*lane_id, lane));
            }
        }

        if let (Some((id1, _)), Some((id2, _))) = (&lane_plus_1, &lane_minus_1) {
            let cross1 = self.find_lane_crossing_point(
                storage,
                terrain_renderer,
                test_polyline,
                start_pos,
                end_pos,
                control,
                id1,
            );
            let cross2 = self.find_lane_crossing_point(
                storage,
                terrain_renderer,
                test_polyline,
                start_pos,
                end_pos,
                control,
                id2,
            );

            match (cross1, cross2) {
                (Some(c1), Some(c2)) => {
                    let avg_t = (c1.t + c2.t) * 0.5;
                    let center_pos =
                        self.sample_path_at_t(terrain_renderer, start_pos, end_pos, control, avg_t);

                    let lane_t = match c1.kind {
                        CrossingKind::LaneCrossing { lane_t, .. } => lane_t,
                        _ => 0.5,
                    };

                    Some(CrossingPoint {
                        t: avg_t,
                        world_pos: center_pos,
                        kind: CrossingKind::LaneCrossing {
                            lane_id: *id1,
                            lane_t,
                        },
                    })
                }
                (Some(c), None) | (None, Some(c)) => Some(c),
                (None, None) => None,
            }
        } else if let Some((id, _)) = &closest_to_zero {
            self.find_lane_crossing_point(
                storage,
                terrain_renderer,
                test_polyline,
                start_pos,
                end_pos,
                control,
                id,
            )
        } else {
            None
        }
    }

    /// I FIXED A BUG HERE WHERE A ROAD "INTERSECTED" WITH THE ROAD I WAS BUILDING FROM AT SHALLOW ANGLES,
    /// LEADING TO A FAKE INTERSECTION RUINING THE GEOMETRY! JUST AS INFO, BEWARE!
    fn get_excluded_segments(
        &self,
        storage: &RoadStorage,
        start_anchor: &Anchor,
        end_anchor: &Anchor,
    ) -> HashSet<SegmentId> {
        let mut excluded = HashSet::new();

        for anchor in [start_anchor, end_anchor] {
            match &anchor.planned_node {
                PlannedNode::Split { lane_id, .. } => {
                    excluded.insert(storage.lane(lane_id).segment());
                }
                PlannedNode::Existing(node_id) => {
                    if let Some(node) = storage.node(*node_id) {
                        for lane_id in node.incoming_lanes() {
                            excluded.insert(storage.lane(lane_id).segment());
                        }
                        for lane_id in node.outgoing_lanes() {
                            excluded.insert(storage.lane(lane_id).segment());
                        }
                    }
                }
                PlannedNode::New { .. } => {}
            }
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
        terrain_renderer: &TerrainSubsystem,
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
        terrain_renderer: &TerrainSubsystem,
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
        let height = terrain_renderer.get_height_at(pos, true);
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
        terrain_renderer: &TerrainSubsystem,
        storage: &RoadStorage,
        start: &Anchor,
        end: &Anchor,
        control: Option<WorldPos>,
        chunk_id: ChunkId,
        output: &mut Vec<RoadEditorCommand>,
        gizmo: &mut Gizmo,
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
            self.style.road_type().structure(),
            start_pos,
            end_pos,
            control,
            start,
            end,
            gizmo,
        );
        //println!("Crossing points: {:?}", crossings.len());
        // Build waypoint list
        let mut waypoints: Vec<ResolvedWaypoint> = Vec::new();

        // Resolve start anchor
        let Some((start_node_id, start_node_pos)) = self.resolve_anchor(
            storage,
            start,
            chunk_id,
            &mut cmds,
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
            end,
            chunk_id,
            &mut cmds,
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
                self.style.road_type().structure(),
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
                structure: self.style.road_type().structure(),
                chunk_id,
            });

            self.emit_lanes_from_centerline(
                terrain_renderer,
                &mut cmds,
                segment_id,
                from.node_id,
                to.node_id,
                &segment_centerline,
                chunk_id,
            );
        }

        // Generate intersections for all waypoint nodes
        for waypoint in &waypoints {
            push_intersection_for_node(&mut cmds, waypoint.node_id, &self.style, chunk_id);
        }

        cmds
    }

    fn compute_segment_centerline(
        &self,
        terrain_renderer: &TerrainSubsystem,
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
    }

    fn find_best_snap(
        &self,
        storage: &RoadStorage,
        terrain_renderer: &TerrainSubsystem,
        pos: WorldPos,
        gizmo: &mut Gizmo,
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
            self.find_nearest_lane_snap(storage, terrain_renderer, pos, gizmo)
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
        terrain_renderer: &TerrainSubsystem,
        pos: WorldPos,
        gizmo: &mut Gizmo,
    ) -> Option<(LaneId, f32, WorldPos, f32)> {
        let nearest_lane_id = nearest_lane_to_point(storage, pos, terrain_renderer.chunk_size)?;
        let nearest_lane = storage.lane(&nearest_lane_id);
        let segment_id = nearest_lane.segment();
        let segment = storage.segment(segment_id);

        let (raw_t, _) =
            project_point_to_lane_xz(nearest_lane, pos, storage, terrain_renderer.chunk_size)?;

        let nearest_is_forward = nearest_lane.from_node() == segment.start;
        let segment_t = if nearest_is_forward {
            raw_t
        } else {
            1.0 - raw_t
        };

        let mut lane_plus_1: Option<(LaneId, &Lane)> = None;
        let mut lane_minus_1: Option<(LaneId, &Lane)> = None;
        let mut closest_to_zero: Option<(LaneId, &Lane)> = None;
        let mut closest_abs_idx = i8::MAX;

        for lane_id in segment.lanes() {
            let lane = storage.lane(lane_id);
            if !lane.is_enabled() {
                continue;
            }

            let idx = lane.lane_index();
            if idx == 1 {
                lane_plus_1 = Some((*lane_id, lane));
            } else if idx == -1 {
                lane_minus_1 = Some((*lane_id, lane));
            }

            if idx.abs() < closest_abs_idx {
                closest_abs_idx = idx.abs();
                closest_to_zero = Some((*lane_id, lane));
            }
        }

        let (rep_lane_id, center_pos, rep_t) =
            if let (Some((id1, l1)), Some((_, l2))) = (lane_plus_1, lane_minus_1) {
                let l1_forward = l1.from_node() == segment.start;
                let l2_forward = l2.from_node() == segment.start;

                let t1 = if l1_forward {
                    segment_t
                } else {
                    1.0 - segment_t
                };
                let t2 = if l2_forward {
                    segment_t
                } else {
                    1.0 - segment_t
                };

                let p1 = sample_lane_position(l1, t1, storage, terrain_renderer.chunk_size)?;
                let p2 = sample_lane_position(l2, t2, storage, terrain_renderer.chunk_size)?;

                let center = WorldPos {
                    chunk: p1.chunk,
                    local: LocalPos::new(
                        (p1.local.x + p2.local.x) * 0.5,
                        (p1.local.y + p2.local.y) * 0.5,
                        (p1.local.z + p2.local.z) * 0.5,
                    ),
                };
                (id1, center, t1)
            } else if let Some((id, lane)) = closest_to_zero {
                let is_forward = lane.from_node() == segment.start;
                let lane_t = if is_forward {
                    segment_t
                } else {
                    1.0 - segment_t
                };
                let p = sample_lane_position(lane, lane_t, storage, terrain_renderer.chunk_size)?;
                (id, p, lane_t)
            } else {
                return None;
            };

        let mut final_pos = center_pos;
        final_pos.local.y = terrain_renderer.get_height_at(final_pos, true) + CLEARANCE;

        let dist = pos.distance_to(final_pos, terrain_renderer.chunk_size);

        if dist >= LANE_SNAP_RADIUS {
            return None;
        }

        let rep_lane = storage.lane(&rep_lane_id);

        if rep_t < ENDPOINT_T_EPS {
            let node_id = rep_lane.from_node();
            let node = storage.node(node_id)?;
            return Some((rep_lane_id, 0.0, node.position(), dist));
        }

        if rep_t > 1.0 - ENDPOINT_T_EPS {
            let node_id = rep_lane.to_node();
            let node = storage.node(node_id)?;
            return Some((rep_lane_id, 1.0, node.position(), dist));
        }
        // gizmo.cross(final_pos, 1.0, [0.0, 1.0, 1.0], 10.0);
        Some((rep_lane_id, rep_t, final_pos, dist))
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
        terrain_renderer: &TerrainSubsystem,
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
            s.local.y = terrain_renderer.get_height_at(s, false) + CLEARANCE;
            sample_points.push(s);
        }
        p.local.y = terrain_renderer.get_height_at(p, false) + CLEARANCE;
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
        anchor: &Anchor,
        chunk_id: ChunkId,
        cmds: &mut Vec<RoadCommand>,
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
                let result = self.plan_split(storage, *lane_id, *pos, chunk_id, chunk_size)?;
                cmds.extend(result.0);
                Some((result.1, *pos))
            }
        }
    }

    fn plan_split(
        &mut self,
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
        terrain_renderer: &TerrainSubsystem,
        cmds: &mut Vec<RoadCommand>,
        segment: SegmentId,
        start: NodeId,
        end: NodeId,
        centerline: &[WorldPos],
        chunk_id: ChunkId,
    ) {
        let (left_lanes, right_lanes) = self.style.road_type().lanes_each_direction();
        let speed = self.style.road_type().speed_limit();
        let capacity = self.style.road_type().capacity();
        let mask = self.style.road_type().vehicle_mask();
        let lane_width = self.style.road_type().lane_width;

        // Right side lanes (start -> end)
        for i in 0..right_lanes {
            let lane_index = (i as i8) + 1;
            let poly = offset_polyline(
                terrain_renderer,
                centerline,
                lane_index,
                lane_width,
                self.style.road_type().structure,
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
                self.style.road_type().structure,
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
    terrain_renderer: &TerrainSubsystem,
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
            set_point_height_with_structure_type(terrain_renderer, structure_type, &mut p, false);
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
    if pts.len() < 2 {
        return Vec3::ZERO;
    }

    if lane.from_node() == node {
        // travel direction leaving FROM node
        pts[0].delta_to(pts[1], chunk_size).normalize_or_zero()
    } else {
        // travel direction arriving at TO node (still along lane direction)
        let n = pts.len();
        pts[n - 2]
            .delta_to(pts[n - 1], chunk_size)
            .normalize_or_zero()
    }
}

/// Sample quadratic Bézier curve with WorldPos control points.
pub fn sample_quadratic_bezier(
    terrain_renderer: &TerrainSubsystem,
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

        set_point_height_with_structure_type(terrain_renderer, structure_type, &mut p, true);
        points.push(p);
    }
    points
}
fn estimate_bezier_arc_length(
    terrain_renderer: &TerrainSubsystem,
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
    terrain_renderer: &TerrainSubsystem,
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
            set_point_height_with_structure_type(terrain_renderer, structure_type, &mut p, true);
            p
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

fn _push_intersection(
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
