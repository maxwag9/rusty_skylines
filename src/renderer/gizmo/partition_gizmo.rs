use crate::helpers::hsv::depth_to_color;
use crate::helpers::positions::{ChunkSize, WorldPos};
use crate::renderer::gizmo::gizmo::Gizmo;
use crate::world::cars::car_structs::PartitionId;
use crate::world::cars::partitions::{PartitionManager, PartitionStorage};
use crate::world::roads::road_structs::NodeId;
use crate::world::roads::roads::RoadStorage;
use glam::Vec3;
use std::collections::HashMap;

pub struct PartitionGizmo {
    cached_base_centroids: HashMap<PartitionId, WorldPos>,
    cached_depths: HashMap<PartitionId, u32>,
    cached_child_counts: HashMap<PartitionId, usize>,
    max_depth: u32,
}

impl PartitionGizmo {
    pub fn new() -> Self {
        Self {
            cached_base_centroids: HashMap::new(),
            cached_depths: HashMap::new(),
            cached_child_counts: HashMap::new(),
            max_depth: 0,
        }
    }

    pub fn invalidate_cache(&mut self) {
        self.cached_base_centroids.clear();
        self.cached_depths.clear();
        self.cached_child_counts.clear();
        self.max_depth = 0;
    }

    fn display_position(
        &self,
        id: PartitionId,
        config: &PartitionVisualizationConfig,
        chunk_size: ChunkSize,
    ) -> Option<WorldPos> {
        let base = self.cached_base_centroids.get(&id)?;
        let depth = self.cached_depths.get(&id).copied().unwrap_or(0);
        let height = (self.max_depth.saturating_sub(depth)) as f32 * config.height_per_level;
        Some(base.add_vec3(Vec3::new(0.0, height, 0.0), chunk_size))
    }

    pub fn visualize(
        &mut self,
        gizmo: &mut Gizmo,
        manager: &PartitionManager,
        road_storage: &RoadStorage,
        config: PartitionVisualizationConfig,
    ) {
        self.rebuild_cache_if_needed(manager, road_storage, gizmo.chunk_size);

        if config.show_hierarchy_arrows {
            self.draw_hierarchy_arrows(gizmo, &manager.storage, &config);
        }

        if config.show_node_markers {
            self.draw_node_markers(gizmo, &manager.storage, &config);
        }

        if config.show_root_markers {
            self.draw_root_markers(gizmo, &manager.storage, &config);
        }

        if config.show_partition_ids {
            self.draw_partition_ids(gizmo, &manager.storage, &config);
        }

        if config.show_child_counts {
            self.draw_child_counts(gizmo, &manager.storage, &config);
        }

        if config.show_node_links {
            self.draw_node_links(gizmo, manager, road_storage, &config);
        }

        if config.show_depth_rings {
            self.draw_depth_rings(gizmo, &manager.storage, &config);
        }

        println!(
            "{:#?}",
            manager.check_route_possibility(road_storage, NodeId::new(2), NodeId::new(13))
        );
    }

    fn draw_hierarchy_arrows(
        &self,
        gizmo: &mut Gizmo,
        storage: &PartitionStorage,
        config: &PartitionVisualizationConfig,
    ) {
        let cs = gizmo.chunk_size;

        for (&id, _) in &self.cached_base_centroids {
            let Some(partition) = storage.get(id) else {
                continue;
            };
            let Some(parent_pos) = self.display_position(id, config, cs) else {
                continue;
            };

            let depth = self.cached_depths.get(&id).copied().unwrap_or(0);
            let color = depth_to_color(depth, self.max_depth);

            for &child_id in &partition.children {
                let Some(child_pos) = self.display_position(child_id, config, cs) else {
                    continue;
                };
                gizmo.arrow(
                    parent_pos,
                    child_pos,
                    color,
                    config.dashed_arrows,
                    config.duration,
                );
            }
        }
    }

    fn draw_partition_ids(
        &self,
        gizmo: &mut Gizmo,
        storage: &PartitionStorage,
        config: &PartitionVisualizationConfig,
    ) {
        let cs = gizmo.chunk_size;

        for (&id, _) in &self.cached_base_centroids {
            if storage.get(id).is_none() {
                continue;
            }

            let Some(pos) = self.display_position(id, config, cs) else {
                continue;
            };
            let depth = self.cached_depths.get(&id).copied().unwrap_or(0);
            let color = depth_to_color(depth, self.max_depth);

            let scale = config.id_scale_base / (1.0 + depth as f32 * config.id_scale_depth_factor);
            let label_pos =
                pos.add_vec3(Vec3::new(config.id_offset_x, config.id_offset_y, 0.0), cs);

            gizmo.number(id as isize, label_pos, scale, color, config.duration);
        }
    }

    fn draw_child_counts(
        &self,
        gizmo: &mut Gizmo,
        storage: &PartitionStorage,
        config: &PartitionVisualizationConfig,
    ) {
        let cs = gizmo.chunk_size;

        for (&id, _) in &self.cached_base_centroids {
            let Some(partition) = storage.get(id) else {
                continue;
            };
            if partition.children.is_empty() {
                continue;
            }

            let Some(pos) = self.display_position(id, config, cs) else {
                continue;
            };
            let child_count = partition.children.len();

            let label_pos = pos.add_vec3(
                Vec3::new(-config.id_offset_x * 2.0, config.id_offset_y * 0.5, 0.0),
                cs,
            );

            gizmo.number(
                child_count as isize,
                label_pos,
                config.id_scale_base * 0.6,
                [0.8, 0.8, 0.8],
                config.duration,
            );
        }
    }

    fn draw_node_markers(
        &self,
        gizmo: &mut Gizmo,
        storage: &PartitionStorage,
        config: &PartitionVisualizationConfig,
    ) {
        let cs = gizmo.chunk_size;

        for (&id, _) in &self.cached_base_centroids {
            let Some(partition) = storage.get(id) else {
                continue;
            };
            let Some(pos) = self.display_position(id, config, cs) else {
                continue;
            };

            let depth = self.cached_depths.get(&id).copied().unwrap_or(0);
            let is_leaf = partition.children.is_empty();

            if is_leaf {
                gizmo.circle(
                    pos,
                    config.leaf_marker_radius,
                    config.leaf_color,
                    config.duration,
                );
                gizmo.cross(
                    pos,
                    config.leaf_marker_radius * 0.8,
                    config.leaf_color,
                    config.duration,
                );
            } else {
                let color = depth_to_color(depth, self.max_depth);
                let radius = config.internal_marker_radius / (1.0 + depth as f32 * 0.25);
                gizmo.sphere(pos, radius, color, config.duration);
            }
        }
    }

    fn draw_root_markers(
        &self,
        gizmo: &mut Gizmo,
        storage: &PartitionStorage,
        config: &PartitionVisualizationConfig,
    ) {
        let cs = gizmo.chunk_size;

        for (&id, _) in &self.cached_base_centroids {
            let depth = self.cached_depths.get(&id).copied().unwrap_or(0);
            if depth != 0 {
                continue;
            }
            if storage.get(id).is_none() {
                continue;
            }

            let Some(pos) = self.display_position(id, config, cs) else {
                continue;
            };

            gizmo.cross(
                pos,
                config.root_marker_size,
                config.root_color,
                config.duration,
            );
            gizmo.circle(
                pos,
                config.root_marker_size * 0.7,
                config.root_color,
                config.duration,
            );

            let crown_offset = Vec3::new(0.0, config.root_marker_size * 0.5, 0.0);
            gizmo.circle(
                pos.add_vec3(crown_offset, cs),
                config.root_marker_size * 0.4,
                config.root_color,
                config.duration,
            );
        }
    }

    fn draw_depth_rings(
        &self,
        gizmo: &mut Gizmo,
        storage: &PartitionStorage,
        config: &PartitionVisualizationConfig,
    ) {
        let cs = gizmo.chunk_size;

        for (&id, _) in &self.cached_base_centroids {
            let Some(partition) = storage.get(id) else {
                continue;
            };
            if partition.children.is_empty() {
                continue;
            }

            let Some(pos) = self.display_position(id, config, cs) else {
                continue;
            };
            let depth = self.cached_depths.get(&id).copied().unwrap_or(0);
            let color = depth_to_color(depth, self.max_depth);

            for i in 1..=depth.min(3) {
                let ring_radius = config.internal_marker_radius * (1.0 + i as f32 * 0.3);
                let alpha = 1.0 - (i as f32 * 0.25);
                let ring_color = [color[0] * alpha, color[1] * alpha, color[2] * alpha];
                gizmo.circle(pos, ring_radius, ring_color, config.duration);
            }
        }
    }

    fn draw_node_links(
        &self,
        gizmo: &mut Gizmo,
        manager: &PartitionManager,
        road_storage: &RoadStorage,
        config: &PartitionVisualizationConfig,
    ) {
        let cs = gizmo.chunk_size;

        for (node_id, &partition_id) in &manager.node_to_leaf {
            let Some(node) = road_storage.node(*node_id) else {
                continue;
            };
            let Some(partition_pos) = self.display_position(partition_id, config, cs) else {
                continue;
            };

            let node_pos = node.position();
            gizmo.line(
                node_pos,
                partition_pos,
                config.node_link_color,
                config.duration,
            );
            gizmo.cross(node_pos, 2.0, config.node_link_color, config.duration);
        }
    }

    pub fn highlight_partition_path(
        &self,
        gizmo: &mut Gizmo,
        storage: &PartitionStorage,
        partition_id: PartitionId,
        config: &PartitionVisualizationConfig,
    ) {
        let cs = gizmo.chunk_size;

        let mut path = Vec::new();
        let mut current = Some(partition_id);

        while let Some(id) = current {
            path.push(id);
            current = storage.parent_of(id);
        }

        path.reverse();

        for window in path.windows(2) {
            let (parent_id, child_id) = (window[0], window[1]);
            let Some(p1) = self.display_position(parent_id, config, cs) else {
                continue;
            };
            let Some(p2) = self.display_position(child_id, config, cs) else {
                continue;
            };

            gizmo.arrow(p1, p2, config.highlight_color, false, config.duration);
        }

        for (i, &id) in path.iter().enumerate() {
            let Some(pos) = self.display_position(id, config, cs) else {
                continue;
            };

            let brightness = 1.0 - (i as f32 * 0.12).min(0.6);
            let radius = config.highlight_radius * (1.0 + i as f32 * 0.15);
            let color = [
                config.highlight_color[0] * brightness,
                config.highlight_color[1] * brightness,
                config.highlight_color[2] * brightness,
            ];

            gizmo.sphere(pos, radius, color, config.duration);
            gizmo.number(
                id as isize,
                pos.add_vec3(Vec3::new(radius * 1.5, 0.0, 0.0), cs),
                config.id_scale_base,
                color,
                config.duration,
            );
        }
    }

    pub fn highlight_address(
        &self,
        gizmo: &mut Gizmo,
        address: &[PartitionId],
        config: &PartitionVisualizationConfig,
    ) {
        let cs = gizmo.chunk_size;

        for window in address.windows(2) {
            let (parent, child) = (window[0], window[1]);
            let Some(p1) = self.display_position(parent, config, cs) else {
                continue;
            };
            let Some(p2) = self.display_position(child, config, cs) else {
                continue;
            };

            gizmo.arrow(p1, p2, config.address_color, false, config.duration);
        }

        for (i, &id) in address.iter().enumerate() {
            let Some(pos) = self.display_position(id, config, cs) else {
                continue;
            };
            let radius = config.address_marker_radius * (1.0 - i as f32 * 0.1).max(0.5);

            gizmo.sphere(pos, radius, config.address_color, config.duration);
            gizmo.number(
                (i + 1) as isize,
                pos.add_vec3(Vec3::new(0.0, radius * 2.0, 0.0), cs),
                config.id_scale_base * 0.8,
                config.address_color,
                config.duration,
            );
        }
    }

    pub fn draw_depth_level(
        &self,
        gizmo: &mut Gizmo,
        storage: &PartitionStorage,
        target_depth: u32,
        config: &PartitionVisualizationConfig,
    ) {
        let cs = gizmo.chunk_size;

        for (&id, _) in &self.cached_base_centroids {
            let depth = self.cached_depths.get(&id).copied().unwrap_or(0);
            if depth != target_depth {
                continue;
            }
            if storage.get(id).is_none() {
                continue;
            }

            let Some(pos) = self.display_position(id, config, cs) else {
                continue;
            };
            let color = depth_to_color(depth, self.max_depth);

            gizmo.sphere(pos, config.depth_level_radius, color, config.duration);
            gizmo.number(
                id as isize,
                pos.add_vec3(Vec3::new(0.0, config.id_offset_y, 0.0), cs),
                config.id_scale_base,
                color,
                config.duration,
            );
        }
    }

    pub fn draw_subtree(
        &self,
        gizmo: &mut Gizmo,
        storage: &PartitionStorage,
        root_id: PartitionId,
        config: &PartitionVisualizationConfig,
    ) {
        let cs = gizmo.chunk_size;
        let mut stack = vec![(root_id, 0u32)];

        while let Some((id, local_depth)) = stack.pop() {
            let Some(partition) = storage.get(id) else {
                continue;
            };
            let Some(pos) = self.display_position(id, config, cs) else {
                continue;
            };

            let global_depth = self.cached_depths.get(&id).copied().unwrap_or(0);
            let color = depth_to_color(global_depth, self.max_depth);

            let radius = config.subtree_node_radius / (1.0 + local_depth as f32 * 0.2);
            gizmo.sphere(pos, radius, color, config.duration);
            gizmo.number(
                id as isize,
                pos.add_vec3(Vec3::new(0.0, config.id_offset_y, 0.0), cs),
                config.id_scale_base / (1.0 + local_depth as f32 * 0.15),
                color,
                config.duration,
            );

            for &child_id in &partition.children {
                let Some(child_pos) = self.display_position(child_id, config, cs) else {
                    continue;
                };
                gizmo.arrow(pos, child_pos, color, false, config.duration);
                stack.push((child_id, local_depth + 1));
            }
        }
    }

    pub fn draw_statistics(
        &self,
        gizmo: &mut Gizmo,
        storage: &PartitionStorage,
        anchor: WorldPos,
        config: &PartitionVisualizationConfig,
    ) {
        let cs = gizmo.chunk_size;

        let total_partitions = self.cached_base_centroids.len();
        let leaf_count = self
            .cached_base_centroids
            .keys()
            .filter(|&&id| {
                storage
                    .get(id)
                    .map(|p| p.children.is_empty())
                    .unwrap_or(false)
            })
            .count();
        let internal_count = total_partitions - leaf_count;

        let line_height = config.id_scale_base * 3.0;
        let stats = [
            (total_partitions as isize, [0.9, 0.9, 0.9]),
            (internal_count as isize, [0.7, 0.7, 1.0]),
            (leaf_count as isize, config.leaf_color),
            (self.max_depth as isize, [1.0, 0.8, 0.3]),
        ];

        for (i, (value, color)) in stats.iter().enumerate() {
            let pos = anchor.add_vec3(Vec3::new(0.0, 0.0, i as f32 * line_height), cs);
            gizmo.number(*value, pos, config.id_scale_base, *color, config.duration);
        }
    }

    fn rebuild_cache_if_needed(
        &mut self,
        manager: &PartitionManager,
        road_storage: &RoadStorage,
        chunk_size: ChunkSize,
    ) {
        let partition_count = manager.storage.partition_count();
        if self.cached_base_centroids.len() == partition_count && partition_count > 0 {
            return;
        }

        self.cached_base_centroids.clear();
        self.cached_depths.clear();
        self.cached_child_counts.clear();

        let mut leaf_nodes: HashMap<PartitionId, Vec<WorldPos>> = HashMap::new();

        for (node_id, &partition_id) in &manager.node_to_leaf {
            if let Some(node) = road_storage.node(*node_id) {
                leaf_nodes
                    .entry(partition_id)
                    .or_default()
                    .push(node.position());
            }
        }

        for (&partition_id, positions) in &leaf_nodes {
            if positions.is_empty() {
                continue;
            }
            let centroid = Self::compute_centroid(positions, chunk_size);
            self.cached_base_centroids.insert(partition_id, centroid);
        }

        self.compute_internal_centroids(&manager.storage, chunk_size);
        self.compute_depths(&manager.storage);
        self.compute_child_counts(&manager.storage);

        self.max_depth = self.cached_depths.values().copied().max().unwrap_or(0);
    }

    fn compute_internal_centroids(&mut self, storage: &PartitionStorage, chunk_size: ChunkSize) {
        let mut changed = true;
        while changed {
            changed = false;

            for id in 0..storage.partitions.len() as PartitionId {
                if !storage.is_alive(id) || self.cached_base_centroids.contains_key(&id) {
                    continue;
                }

                let Some(partition) = storage.get(id) else {
                    continue;
                };
                if partition.children.is_empty() {
                    continue;
                }

                let child_positions: Vec<WorldPos> = partition
                    .children
                    .iter()
                    .filter_map(|&c| self.cached_base_centroids.get(&c).copied())
                    .collect();

                if child_positions.len() == partition.children.len() {
                    let centroid = Self::compute_centroid(&child_positions, chunk_size);
                    self.cached_base_centroids.insert(id, centroid);
                    changed = true;
                }
            }
        }
    }

    fn compute_depths(&mut self, storage: &PartitionStorage) {
        for id in 0..storage.partitions.len() as PartitionId {
            if !storage.is_alive(id) {
                continue;
            }
            let depth = storage.ancestors(id).count() as u32;
            self.cached_depths.insert(id, depth);
        }
    }

    fn compute_child_counts(&mut self, storage: &PartitionStorage) {
        for id in 0..storage.partitions.len() as PartitionId {
            if !storage.is_alive(id) {
                continue;
            }
            if let Some(partition) = storage.get(id) {
                self.cached_child_counts
                    .insert(id, partition.children.len());
            }
        }
    }

    fn compute_centroid(positions: &[WorldPos], chunk_size: ChunkSize) -> WorldPos {
        if positions.is_empty() {
            return WorldPos::default();
        }

        let reference = positions[0];
        let mut sum = Vec3::ZERO;

        for pos in positions {
            sum += pos.to_render_pos(reference, chunk_size);
        }

        let avg = sum / positions.len() as f32;
        reference.add_vec3(avg, chunk_size)
    }

    pub fn partition_count(&self) -> usize {
        self.cached_base_centroids.len()
    }

    pub fn get_base_centroid(&self, id: PartitionId) -> Option<WorldPos> {
        self.cached_base_centroids.get(&id).copied()
    }

    pub fn get_display_position(
        &self,
        id: PartitionId,
        config: &PartitionVisualizationConfig,
        chunk_size: ChunkSize,
    ) -> Option<WorldPos> {
        self.display_position(id, config, chunk_size)
    }

    pub fn get_depth(&self, id: PartitionId) -> Option<u32> {
        self.cached_depths.get(&id).copied()
    }

    pub fn get_child_count(&self, id: PartitionId) -> Option<usize> {
        self.cached_child_counts.get(&id).copied()
    }

    pub fn max_depth(&self) -> u32 {
        self.max_depth
    }
}

impl Default for PartitionGizmo {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct PartitionVisualizationConfig {
    pub show_hierarchy_arrows: bool,
    pub show_partition_ids: bool,
    pub show_child_counts: bool,
    pub show_node_markers: bool,
    pub show_node_links: bool,
    pub show_root_markers: bool,
    pub show_depth_rings: bool,
    pub dashed_arrows: bool,
    pub height_per_level: f32,
    pub leaf_marker_radius: f32,
    pub internal_marker_radius: f32,
    pub root_marker_size: f32,
    pub id_scale_base: f32,
    pub id_scale_depth_factor: f32,
    pub id_offset_x: f32,
    pub id_offset_y: f32,
    pub highlight_radius: f32,
    pub highlight_color: [f32; 3],
    pub address_marker_radius: f32,
    pub address_color: [f32; 3],
    pub depth_level_radius: f32,
    pub subtree_node_radius: f32,
    pub leaf_color: [f32; 3],
    pub root_color: [f32; 3],
    pub node_link_color: [f32; 3],
    pub duration: f32,
}

impl Default for PartitionVisualizationConfig {
    fn default() -> Self {
        Self {
            show_hierarchy_arrows: true,
            show_partition_ids: true,
            show_child_counts: false,
            show_node_markers: true,
            show_node_links: false,
            show_root_markers: true,
            show_depth_rings: false,
            dashed_arrows: false,
            height_per_level: 50.0,
            leaf_marker_radius: 4.0,
            internal_marker_radius: 8.0,
            root_marker_size: 15.0,
            id_scale_base: 3.0,
            id_scale_depth_factor: 0.2,
            id_offset_x: 5.0,
            id_offset_y: 10.0,
            highlight_radius: 6.0,
            highlight_color: [1.0, 0.9, 0.1],
            address_marker_radius: 5.0,
            address_color: [1.0, 0.5, 0.0],
            depth_level_radius: 5.0,
            subtree_node_radius: 6.0,
            leaf_color: [0.2, 0.9, 0.3],
            root_color: [1.0, 0.85, 0.0],
            node_link_color: [0.4, 0.4, 0.7],
            duration: 0.0,
        }
    }
}

impl PartitionVisualizationConfig {
    pub fn minimal() -> Self {
        Self {
            show_hierarchy_arrows: true,
            show_partition_ids: false,
            show_child_counts: false,
            show_node_markers: false,
            show_node_links: false,
            show_root_markers: false,
            show_depth_rings: false,
            ..Default::default()
        }
    }

    pub fn detailed() -> Self {
        Self {
            show_hierarchy_arrows: true,
            show_partition_ids: true,
            show_child_counts: true,
            show_node_markers: true,
            show_node_links: true,
            show_root_markers: true,
            show_depth_rings: true,
            ..Default::default()
        }
    }

    pub fn ids_only() -> Self {
        Self {
            show_hierarchy_arrows: false,
            show_partition_ids: true,
            show_child_counts: false,
            show_node_markers: false,
            show_node_links: false,
            show_root_markers: false,
            show_depth_rings: false,
            ..Default::default()
        }
    }

    pub fn hierarchy_only() -> Self {
        Self {
            show_hierarchy_arrows: true,
            show_partition_ids: true,
            show_child_counts: false,
            show_node_markers: true,
            show_node_links: false,
            show_root_markers: true,
            show_depth_rings: false,
            ..Default::default()
        }
    }

    pub fn with_height(mut self, height: f32) -> Self {
        self.height_per_level = height;
        self
    }

    pub fn with_duration(mut self, duration: f32) -> Self {
        self.duration = duration;
        self
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.leaf_marker_radius *= scale;
        self.internal_marker_radius *= scale;
        self.root_marker_size *= scale;
        self.id_scale_base *= scale;
        self.id_offset_x *= scale;
        self.id_offset_y *= scale;
        self.highlight_radius *= scale;
        self.address_marker_radius *= scale;
        self.depth_level_radius *= scale;
        self.subtree_node_radius *= scale;
        self
    }

    pub fn with_dashed_arrows(mut self, dashed: bool) -> Self {
        self.dashed_arrows = dashed;
        self
    }
}
