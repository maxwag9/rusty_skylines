use crate::renderer::world_renderer::PickedPoint;
use crate::resources::InputState;
use crate::terrain::roads::road_mesh_manager::{ChunkId, HorizontalProfile};
use crate::terrain::roads::roads::{
    LaneId, NodeId, RoadCommand, RoadManager, SegmentId, StructureType, VerticalProfile,
    nearest_lane_to_point, project_point_to_lane_xz, sample_lane_position,
};
use glam::Vec3;

#[derive(Debug, Clone, Copy)]
enum SnapType {
    None { pos: Vec3 },
    Node { id: NodeId, pos: Vec3 },
    Lane { lane_id: LaneId, t: f32, pos: Vec3 },
}

impl SnapType {
    fn position(&self) -> Vec3 {
        match self {
            SnapType::None { pos } => *pos,
            SnapType::Node { pos, .. } => *pos,
            SnapType::Lane { pos, .. } => *pos,
        }
    }

    fn target_node_id(&self, predicted_new_node_id: NodeId) -> NodeId {
        match self {
            SnapType::Node { id, .. } => *id,
            _ => predicted_new_node_id,
        }
    }
}

/// Separate road editor that feels very similar to Cities: Skylines road placement.
/// - Click to place points (action "Place Road Node").
/// - Automatic snapping to existing nodes (high priority).
/// - Automatic snapping to existing roads → creates proper intersections by splitting the existing segment.
/// - Builds bidirectional 1-lane-per-direction roads (easily configurable).
/// - Straight segments only (curves are approximated by placing many points).
/// - Outputs pure Commands (no direct mutation of RoadManager).
/// - Cancel with a separate action (e.g., "Cancel" or Esc).
/// - No auto-detection of parallel upgrades (can be added later).
pub struct RoadEditor {
    /// True while we are actively building a road chain.
    is_building: bool,
    /// Nodes that have already been placed in the current chain (guaranteed to exist after commands are applied).
    placed_nodes: Vec<NodeId>,
}

impl RoadEditor {
    pub fn new() -> Self {
        Self {
            is_building: false,
            placed_nodes: Vec::new(),
        }
    }

    /// Main update function. Call every frame while the road tool is active.
    /// Returns the list of Commands to apply this frame.
    pub fn update(
        &mut self,
        manager: &RoadManager,
        input: &mut InputState,
        picked_point: &Option<PickedPoint>,
    ) -> Vec<RoadCommand> {
        let mut commands = Vec::new();

        // Cancel building
        if input.action_pressed_once("Cancel") {
            self.is_building = false;
            self.placed_nodes.clear();
            return commands;
        }

        // Only react to placement action
        let place_pressed = input.action_pressed_once("Place Road Node");
        if !place_pressed {
            return commands;
        }

        let Some(picked) = picked_point else {
            return commands;
        };
        let snap = self.find_best_snap(manager, picked.pos);

        let target_pos = snap.position();

        let chunk_id = picked.chunk.id;

        if !self.is_building {
            // ==================== START A NEW ROAD ====================
            self.is_building = true;

            match snap {
                SnapType::Node { id, .. } => {
                    // Start directly from existing node
                    self.placed_nodes.push(id);
                }
                SnapType::Lane { lane_id, t, .. } => {
                    if t < 0.02 || t > 0.98 {
                        // Too close to endpoint → treat as node snap
                        let node_id = if t < 0.02 {
                            manager.lane(lane_id).from_node()
                        } else {
                            manager.lane(lane_id).to_node()
                        };
                        self.placed_nodes.push(node_id);
                    } else {
                        // Split existing road and start from the new intersection
                        let split_cmds =
                            self.generate_split_commands(manager, lane_id, target_pos, chunk_id);
                        commands.extend(split_cmds);
                        // The new intersection node is always the next node ID (first/only AddNode in the batch)
                        let new_node_id = NodeId::new(manager.nodes.len() as u32);
                        self.placed_nodes.push(new_node_id);
                    }
                }
                SnapType::None { .. } => {
                    // Create completely new starting node
                    let new_id = NodeId::new(manager.nodes.len() as u32);
                    commands.push(RoadCommand::AddNode {
                        x: target_pos.x,
                        y: target_pos.y,
                        z: target_pos.z,
                        chunk_id,
                    });
                    self.placed_nodes.push(new_id);
                }
            }
        } else {
            // ==================== CONTINUE EXISTING ROAD ====================
            let start_id = *self.placed_nodes.last().unwrap();

            // Determine the end node and any prerequisite commands (split or new node creation)
            let (end_id, prepend_commands): (NodeId, Vec<RoadCommand>) = match snap {
                SnapType::Node { id, .. } => (id, vec![]),
                SnapType::Lane { lane_id, t, .. } => {
                    if t < 0.02 || t > 0.98 {
                        let node_id = if t < 0.02 {
                            manager.lane(lane_id).from_node()
                        } else {
                            manager.lane(lane_id).to_node()
                        };
                        (node_id, vec![])
                    } else {
                        let split_cmds =
                            self.generate_split_commands(manager, lane_id, target_pos, chunk_id);
                        // The new intersection node is always the next node ID (first/only AddNode in the batch)
                        let new_id = NodeId::new(manager.nodes.len() as u32);
                        (new_id, split_cmds)
                    }
                }
                SnapType::None { .. } => {
                    let new_node_cmd = RoadCommand::AddNode {
                        x: target_pos.x,
                        y: target_pos.y,
                        z: target_pos.z,
                        chunk_id,
                    };
                    // The new node is always the next node ID (first/only AddNode in the batch)
                    let new_id = NodeId::new(manager.nodes.len() as u32);
                    (new_id, vec![new_node_cmd])
                }
            };

            commands.extend(prepend_commands);

            // ----- Add the new segment itself -----
            // Count how many segments have already been added in this batch (from split replacements)
            let added_segments_so_far = commands
                .iter()
                .filter(|c| matches!(c, RoadCommand::AddSegment { .. }))
                .count() as u32;
            let new_segment_id =
                SegmentId::new(manager.segments.len() as u32 + added_segments_so_far);

            let Some(start_node) = manager.node(start_id) else {
                return commands;
            };
            let start_y = start_node.y();
            let end_y = target_pos.y;

            commands.push(RoadCommand::AddSegment {
                start: start_id,
                end: end_id,
                structure: StructureType::Surface,
                horizontal_profile: HorizontalProfile::Linear,
                vertical_profile: VerticalProfile::Linear { start_y, end_y },
                chunk_id,
            });

            // ----- Add lanes (1 forward + 1 backward) -----
            // Compute 2D length directly from positions

            let dx = target_pos.x - start_node.x();
            let dz = target_pos.z - start_node.z();
            let length_2d = (dx * dx + dz * dz).sqrt();
            let base_cost = length_2d.max(1.0);

            let speed_limit = 13.9; // ~50 km/h
            let capacity = 2000;
            let vehicle_mask = 1; // adjust to your system

            // Forward lane
            commands.push(RoadCommand::AddLane {
                from: start_id,
                to: end_id,
                segment: new_segment_id,
                speed_limit,
                capacity,
                vehicle_mask,
                base_cost,
                chunk_id,
            });

            // Backward lane
            commands.push(RoadCommand::AddLane {
                from: end_id,
                to: start_id,
                segment: new_segment_id,
                speed_limit,
                capacity,
                vehicle_mask,
                base_cost,
                chunk_id,
            });

            self.placed_nodes.push(end_id);
        }

        commands
    }

    /// Prioritizes node snap → lane projection snap → no snap.
    fn find_best_snap(&self, manager: &RoadManager, pos: Vec3) -> SnapType {
        const NODE_SNAP_RADIUS: f32 = 6.0;
        const LANE_SNAP_RADIUS: f32 = 10.0;

        // Node snap (highest priority)
        let mut best_node: Option<(NodeId, f32)> = None;
        for (id, node) in manager.iter_enabled_nodes() {
            let dx = node.x - pos.x;
            let dz = node.z - pos.z;
            let dist = (dx * dx + dz * dz).sqrt();
            if dist < NODE_SNAP_RADIUS {
                if best_node.is_none() || dist < best_node.unwrap().1 {
                    best_node = Some((id, dist));
                }
            }
        }

        if let Some((id, _)) = best_node {
            let Some(node) = manager.node(id) else {
                return SnapType::None { pos };
            };
            return SnapType::Node {
                id,
                pos: Vec3::new(node.x, node.y, node.z),
            };
        }

        // Lane projection snap
        if let Some(lane_id) = nearest_lane_to_point(manager, pos.x, pos.y, pos.z) {
            let lane = manager.lane(lane_id);
            let (t, dist_sq) = project_point_to_lane_xz(lane, pos.x, pos.z, manager);
            if dist_sq < LANE_SNAP_RADIUS * LANE_SNAP_RADIUS {
                let projected_pos = sample_lane_position(lane, t, manager);
                return SnapType::Lane {
                    lane_id,
                    t,
                    pos: Vec3::new(projected_pos.0, projected_pos.1, projected_pos.2),
                };
            }
        }

        SnapType::None { pos }
    }

    /// Generates commands to split an existing segment at the given point.
    fn generate_split_commands(
        &self,
        manager: &RoadManager,
        lane_id: LaneId,
        split_pos: Vec3,
        chunk_id: ChunkId,
    ) -> Vec<RoadCommand> {
        let mut cmds = Vec::new();

        let lane = manager.lane(lane_id);
        let old_segment_id = lane.segment();
        let old_segment = manager.segment(old_segment_id);

        // Disable old segment
        cmds.push(RoadCommand::DisableSegment {
            segment_id: old_segment_id,
            chunk_id,
        });

        let a_id = old_segment.start();
        let b_id = old_segment.end();

        // Create new intersection node
        cmds.push(RoadCommand::AddNode {
            x: split_pos.x,
            y: split_pos.y,
            z: split_pos.z,
            chunk_id,
        });

        // Create two replacement segments
        let base_seg_idx = manager.segments.len() as u32
            + cmds
                .iter()
                .filter(|c| matches!(c, RoadCommand::AddSegment { .. }))
                .count() as u32;
        let seg1_id = SegmentId::new(base_seg_idx);
        let seg2_id = SegmentId::new(base_seg_idx + 1);

        let structure = old_segment.structure();
        // TODO: proper horizontal profile cloning or recreation (assuming straight for now)
        let horizontal = HorizontalProfile::Linear;

        let Some(a_node) = manager.node(a_id) else {
            return Vec::new();
        };
        let Some(b_node) = manager.node(b_id) else {
            return Vec::new();
        };
        let y_a = a_node.y();
        let y_b = b_node.y();

        cmds.push(RoadCommand::AddSegment {
            start: a_id,
            end: NodeId::new(manager.nodes.len() as u32), // new node is added earlier in batch
            structure,
            horizontal_profile: horizontal.clone(),
            vertical_profile: VerticalProfile::Linear {
                start_y: y_a,
                end_y: split_pos.y,
            },
            chunk_id,
        });

        cmds.push(RoadCommand::AddSegment {
            start: NodeId::new(manager.nodes.len() as u32),
            end: b_id,
            structure,
            horizontal_profile: horizontal,
            vertical_profile: VerticalProfile::Linear {
                start_y: split_pos.y,
                end_y: y_b,
            },
            chunk_id,
        });

        // Replicate all old lanes on both new segments
        for &old_lane_id in old_segment.lanes() {
            let old_lane = manager.lane(old_lane_id);
            let (speed, cap, mask, cost) = (
                old_lane.speed_limit(),
                old_lane.capacity(),
                old_lane.vehicle_mask(),
                old_lane.base_cost(),
            );

            if old_lane.from_node() == a_id {
                // Forward direction
                cmds.push(RoadCommand::AddLane {
                    from: a_id,
                    to: NodeId::new(manager.nodes.len() as u32),
                    segment: seg1_id,
                    speed_limit: speed,
                    capacity: cap,
                    vehicle_mask: mask,
                    base_cost: cost,
                    chunk_id,
                });
                cmds.push(RoadCommand::AddLane {
                    from: NodeId::new(manager.nodes.len() as u32),
                    to: b_id,
                    segment: seg2_id,
                    speed_limit: speed,
                    capacity: cap,
                    vehicle_mask: mask,
                    base_cost: cost,
                    chunk_id,
                });
            } else {
                // Backward direction
                cmds.push(RoadCommand::AddLane {
                    from: b_id,
                    to: NodeId::new(manager.nodes.len() as u32),
                    segment: seg2_id,
                    speed_limit: speed,
                    capacity: cap,
                    vehicle_mask: mask,
                    base_cost: cost,
                    chunk_id,
                });
                cmds.push(RoadCommand::AddLane {
                    from: NodeId::new(manager.nodes.len() as u32),
                    to: a_id,
                    segment: seg1_id,
                    speed_limit: speed,
                    capacity: cap,
                    vehicle_mask: mask,
                    base_cost: cost,
                    chunk_id,
                });
            }
        }

        cmds
    }
}
