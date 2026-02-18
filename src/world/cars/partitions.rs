use crate::helpers::positions::ChunkSize;
use crate::world::roads::road_structs::{LaneId, NodeId, SegmentId};
use crate::world::roads::roads::{RoadRegionId, RoadStorage};
use std::collections::HashMap;

pub type PartitionId = u32;
pub type LaneT = f32;

/// Result of checking if a route is possible between two locations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteStatus {
    /// Both locations are in the same road region - routing is possible
    Routable,
    /// Locations are in different disconnected road regions - no path exists
    UnreachableRegion {
        from: RoadRegionId,
        to: RoadRegionId,
    },
    /// One or both locations are invalid or don't exist
    Invalid,
}

#[derive(Debug, Clone)]
pub enum DestinationType {
    Node(NodeId),
    Segment(LaneId, LaneT),
}

/// A hierarchical address within a single road region.
///
/// The address path goes from region root (index 0) down to the leaf partition.
/// Two addresses can only be connected if they share the same region.
#[derive(Debug)]
pub struct HierarchicalAddress {
    pub destination: DestinationType,
    /// The disconnected road region this address belongs to
    pub region: RoadRegionId,
    /// Path from root partition to leaf partition, inclusive.
    /// - `address[0]` is the region's root partition (has no parent)
    /// - `address[len-1]` is the leaf partition containing the destination
    pub address: Vec<PartitionId>,
}

impl HierarchicalAddress {
    /// Returns the root partition (region's top-level partition).
    #[inline]
    pub fn root(&self) -> Option<PartitionId> {
        self.address.first().copied()
    }

    /// Returns the leaf partition (most specific partition containing destination).
    #[inline]
    pub fn leaf(&self) -> Option<PartitionId> {
        self.address.last().copied()
    }

    /// Checks if this address is in the same road region as another.
    /// If false, no route exists between them.
    #[inline]
    pub fn same_region(&self, other: &HierarchicalAddress) -> bool {
        self.region == other.region
    }

    /// Returns the depth in the partition hierarchy (0 = at root).
    #[inline]
    pub fn depth(&self) -> usize {
        self.address.len().saturating_sub(1)
    }
}

pub struct Partition {
    pub parent: Option<PartitionId>,
    pub children: Vec<PartitionId>,
    /// The road region this partition tree belongs to.
    /// All partitions in the same tree share this region ID.
    pub region: RoadRegionId,
}

pub struct PartitionStorage {
    pub(crate) partitions: Vec<Partition>,
    alive: Vec<bool>,
    free_list: Vec<u32>,
}

impl PartitionStorage {
    pub fn new() -> Self {
        Self {
            partitions: Vec::new(),
            alive: Vec::new(),
            free_list: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            partitions: Vec::with_capacity(capacity),
            alive: Vec::with_capacity(capacity),
            free_list: Vec::new(),
        }
    }

    pub fn clear_all(&mut self) {
        self.partitions.clear();
        self.alive.clear();
        self.free_list.clear();
    }

    pub fn add(&mut self, parent: Option<PartitionId>, region: RoadRegionId) -> PartitionId {
        // Validate parent liveness strictly - reject stale IDs
        let validated_parent = parent.filter(|&pid| self.is_alive(pid));

        // If parent exists, inherit its region (sanity check)
        #[cfg(debug_assertions)]
        if let Some(pid) = validated_parent {
            debug_assert_eq!(
                self.partitions[pid as usize].region, region,
                "Child partition must belong to same region as parent"
            );
        }

        let partition = Partition {
            parent: validated_parent,
            children: Vec::new(),
            region,
        };

        let index = if let Some(index) = self.free_list.pop() {
            let idx = index as usize;
            self.alive[idx] = true;
            self.partitions[idx] = partition;
            index
        } else {
            let index = self.partitions.len() as u32;
            self.partitions.push(partition);
            self.alive.push(true);
            index
        };

        // Register with parent
        if let Some(parent_id) = validated_parent {
            self.partitions[parent_id as usize].children.push(index);
        }

        index
    }

    pub fn remove(&mut self, id: PartitionId) {
        let idx = id as usize;

        if idx >= self.alive.len() || !self.alive[idx] {
            return;
        }

        // Remove self from parent's children list
        if let Some(parent_id) = self.partitions[idx].parent {
            let parent_idx = parent_id as usize;
            if self.alive[parent_idx] {
                let children = &mut self.partitions[parent_idx].children;
                if let Some(pos) = children.iter().position(|&c| c == id) {
                    children.swap_remove(pos);
                }
            }
        }

        self.partitions[idx].parent = None;

        // Recursively remove all children
        let mut to_remove: Vec<PartitionId> = std::mem::take(&mut self.partitions[idx].children);

        while let Some(child_id) = to_remove.pop() {
            let child_idx = child_id as usize;

            if child_idx < self.alive.len() && self.alive[child_idx] {
                to_remove.append(&mut self.partitions[child_idx].children);
                self.partitions[child_idx].parent = None;
                self.alive[child_idx] = false;
                self.free_list.push(child_id);
            }
        }

        self.alive[idx] = false;
        self.free_list.push(id);
    }

    #[inline]
    pub fn is_alive(&self, id: PartitionId) -> bool {
        self.alive.get(id as usize).copied().unwrap_or(false)
    }

    #[inline]
    pub fn region_of(&self, id: PartitionId) -> Option<RoadRegionId> {
        if self.is_alive(id) {
            Some(self.partitions[id as usize].region)
        } else {
            None
        }
    }

    pub fn partition_count(&self) -> usize {
        self.partitions.len() - self.free_list.len()
    }

    #[inline]
    pub fn get(&self, id: PartitionId) -> Option<&Partition> {
        let idx = id as usize;
        if self.alive.get(idx).copied().unwrap_or(false) {
            Some(&self.partitions[idx])
        } else {
            None
        }
    }

    #[inline]
    pub fn get_mut(&mut self, id: PartitionId) -> Option<&mut Partition> {
        let idx = id as usize;
        if self.alive.get(idx).copied().unwrap_or(false) {
            Some(&mut self.partitions[idx])
        } else {
            None
        }
    }

    #[inline]
    pub fn parent_of(&self, id: PartitionId) -> Option<PartitionId> {
        self.get(id)?.parent
    }

    #[inline]
    pub fn children_of(&self, id: PartitionId) -> Option<&[PartitionId]> {
        Some(&self.get(id)?.children)
    }

    #[inline]
    pub fn ancestors(&self, id: PartitionId) -> AncestorIter<'_> {
        AncestorIter {
            storage: self,
            current: self.get(id).and_then(|p| p.parent),
        }
    }

    /// Returns true if both partitions are in the same region tree.
    #[inline]
    pub fn same_region(&self, a: PartitionId, b: PartitionId) -> bool {
        match (self.region_of(a), self.region_of(b)) {
            (Some(ra), Some(rb)) => ra == rb,
            _ => false,
        }
    }
}

impl Default for PartitionStorage {
    fn default() -> Self {
        Self::new()
    }
}

pub struct AncestorIter<'a> {
    storage: &'a PartitionStorage,
    current: Option<PartitionId>,
}

impl<'a> Iterator for AncestorIter<'a> {
    type Item = PartitionId;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let id = self.current?;
        self.current = self.storage.get(id).and_then(|p| p.parent);
        Some(id)
    }
}

pub struct PartitionManager {
    pub storage: PartitionStorage,
    pub(crate) node_to_leaf: HashMap<NodeId, PartitionId>,
    /// Maps each active road region to its root partition.
    /// Each root has `parent: None` and represents a completely separate partition tree.
    region_roots: HashMap<RoadRegionId, PartitionId>,
}

impl PartitionManager {
    const MAX_DEPTH: u32 = 6;
    const TARGET_LEAF_SIZE: usize = 8;

    pub fn new() -> Self {
        Self {
            storage: PartitionStorage::with_capacity(32),
            node_to_leaf: HashMap::new(),
            region_roots: HashMap::new(),
        }
    }

    /// Rebuilds all partition trees, creating a separate tree for each disconnected road region.
    pub fn rebuild_all(&mut self, road_storage: &RoadStorage, chunk_size: ChunkSize) {
        self.storage.clear_all();
        self.node_to_leaf.clear();
        self.region_roots.clear();

        // Build a separate partition tree for each active (non-empty) road region
        for (region_id, region) in road_storage.iter_active_regions() {
            let node_positions: Vec<(NodeId, f64, f64)> = region
                .node_indices()
                .iter()
                .filter_map(|&node_idx| {
                    let node_id = NodeId::new(node_idx);
                    let node = road_storage.node(node_id)?;
                    if !node.is_enabled() {
                        return None;
                    }
                    let pos = node.position();
                    let x = pos.chunk.x as f64 * chunk_size as f64 + pos.local.x as f64;
                    let z = pos.chunk.z as f64 * chunk_size as f64 + pos.local.z as f64;
                    Some((node_id, x, z))
                })
                .collect();

            if node_positions.is_empty() {
                continue;
            }

            let (min_x, max_x, min_z, max_z) = Self::compute_bounds(&node_positions);

            // Build tree with no parent (this is a region root)
            let root_partition = self.build_recursive(
                None,
                region_id,
                node_positions,
                min_x - 1.0,
                max_x + 1.0,
                min_z - 1.0,
                max_z + 1.0,
                0,
            );

            self.region_roots.insert(region_id, root_partition);
        }
    }

    fn compute_bounds(nodes: &[(NodeId, f64, f64)]) -> (f64, f64, f64, f64) {
        nodes.iter().fold(
            (f64::MAX, f64::MIN, f64::MAX, f64::MIN),
            |(min_x, max_x, min_z, max_z), &(_, x, z)| {
                (min_x.min(x), max_x.max(x), min_z.min(z), max_z.max(z))
            },
        )
    }

    fn build_recursive(
        &mut self,
        parent: Option<PartitionId>,
        region: RoadRegionId,
        nodes: Vec<(NodeId, f64, f64)>,
        min_x: f64,
        max_x: f64,
        min_z: f64,
        max_z: f64,
        depth: u32,
    ) -> PartitionId {
        let partition_id = self.storage.add(parent, region);

        if depth >= Self::MAX_DEPTH || nodes.len() <= Self::TARGET_LEAF_SIZE {
            for (node_id, _, _) in nodes {
                self.node_to_leaf.insert(node_id, partition_id);
            }
            return partition_id;
        }

        let mid_x = (min_x + max_x) * 0.5;
        let mid_z = (min_z + max_z) * 0.5;

        let mut quadrants: [Vec<(NodeId, f64, f64)>; 4] = Default::default();

        for (id, x, z) in nodes {
            let idx = ((x >= mid_x) as usize) | (((z >= mid_z) as usize) << 1);
            quadrants[idx].push((id, x, z));
        }

        let bounds = [
            (min_x, mid_x, min_z, mid_z),
            (mid_x, max_x, min_z, mid_z),
            (min_x, mid_x, mid_z, max_z),
            (mid_x, max_x, mid_z, max_z),
        ];

        for (i, quad) in quadrants.into_iter().enumerate() {
            if !quad.is_empty() {
                self.build_recursive(
                    Some(partition_id),
                    region,
                    quad,
                    bounds[i].0,
                    bounds[i].1,
                    bounds[i].2,
                    bounds[i].3,
                    depth + 1,
                );
            }
        }

        partition_id
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Region queries
    // ─────────────────────────────────────────────────────────────────────────────

    /// Returns the root partition for a given road region.
    #[inline]
    pub fn region_root(&self, region_id: RoadRegionId) -> Option<PartitionId> {
        self.region_roots.get(&region_id).copied()
    }

    /// Returns the number of separate road regions (disconnected networks).
    #[inline]
    pub fn region_count(&self) -> usize {
        self.region_roots.len()
    }

    /// Returns an iterator over all (region_id, root_partition) pairs.
    pub fn iter_region_roots(&self) -> impl Iterator<Item = (RoadRegionId, PartitionId)> + '_ {
        self.region_roots.iter().map(|(&r, &p)| (r, p))
    }

    /// Checks if a route between two nodes is possible.
    ///
    /// Returns `UnreachableRegion` if the nodes are in different disconnected road networks,
    /// which means there is NO path between them regardless of the road layout.
    pub fn check_route_possibility(
        &self,
        road_storage: &RoadStorage,
        from: NodeId,
        to: NodeId,
    ) -> RouteStatus {
        let from_exists = road_storage.node(from).is_some();
        let to_exists = road_storage.node(to).is_some();

        if !from_exists || !to_exists {
            return RouteStatus::Invalid;
        }

        let from_region = road_storage.region_for_node(from);
        let to_region = road_storage.region_for_node(to);

        if from_region == to_region {
            RouteStatus::Routable
        } else {
            RouteStatus::UnreachableRegion {
                from: from_region,
                to: to_region,
            }
        }
    }

    /// Checks if two addresses can potentially be connected by a route.
    #[inline]
    pub fn addresses_routable(from: &HierarchicalAddress, to: &HierarchicalAddress) -> bool {
        from.region == to.region
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Partition queries
    // ─────────────────────────────────────────────────────────────────────────────

    pub fn get_node_partition(&self, node_id: NodeId) -> Option<PartitionId> {
        self.node_to_leaf.get(&node_id).copied()
    }

    pub fn get_segment_partition(
        &self,
        road_storage: &RoadStorage,
        segment_id: SegmentId,
    ) -> Option<PartitionId> {
        let segment = road_storage.segment(segment_id);
        let start_leaf = *self.node_to_leaf.get(&segment.start)?;
        let end_leaf = *self.node_to_leaf.get(&segment.end)?;

        if start_leaf == end_leaf {
            return Some(start_leaf);
        }

        // Find the lowest common ancestor within the same region tree
        let start_path: Vec<_> = std::iter::once(start_leaf)
            .chain(self.storage.ancestors(start_leaf))
            .collect();

        std::iter::once(end_leaf)
            .chain(self.storage.ancestors(end_leaf))
            .find(|ancestor| start_path.contains(ancestor))
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Address generation
    // ─────────────────────────────────────────────────────────────────────────────

    pub fn get_node_address(
        &self,
        road_storage: &RoadStorage,
        node_id: NodeId,
    ) -> Option<HierarchicalAddress> {
        let leaf = *self.node_to_leaf.get(&node_id)?;
        let region = road_storage.region_for_node(node_id);

        let mut address: Vec<PartitionId> = std::iter::once(leaf)
            .chain(self.storage.ancestors(leaf))
            .collect();
        address.reverse();

        Some(HierarchicalAddress {
            destination: DestinationType::Node(node_id),
            region,
            address,
        })
    }

    pub fn get_segment_address(
        &self,
        road_storage: &RoadStorage,
        lane_id: LaneId,
        t: LaneT,
    ) -> Option<HierarchicalAddress> {
        let lane = road_storage.lane(&lane_id);
        let segment_id = lane.segment();
        let segment = road_storage.segment(segment_id);
        let lca = self.get_segment_partition(road_storage, segment_id)?;

        // Both endpoints are guaranteed to be in the same region
        let region = road_storage.region_for_node(segment.start);

        let mut address: Vec<PartitionId> = std::iter::once(lca)
            .chain(self.storage.ancestors(lca))
            .collect();
        address.reverse();

        Some(HierarchicalAddress {
            destination: DestinationType::Segment(lane_id, t),
            region,
            address,
        })
    }

    /// Finds the lowest common ancestor of two partitions.
    /// Returns None if they're in different regions (impossible to connect).
    pub fn find_common_ancestor(&self, a: PartitionId, b: PartitionId) -> Option<PartitionId> {
        // Quick check: same region?
        if !self.storage.same_region(a, b) {
            return None;
        }

        if a == b {
            return Some(a);
        }

        let a_path: Vec<_> = std::iter::once(a)
            .chain(self.storage.ancestors(a))
            .collect();

        std::iter::once(b)
            .chain(self.storage.ancestors(b))
            .find(|ancestor| a_path.contains(ancestor))
    }
}

impl Default for PartitionManager {
    fn default() -> Self {
        Self::new()
    }
}
