use crate::helpers::positions::{ChunkCoord, WorldPos, chunk_size};
use crate::world::buildings::buildings::{BuildingId, Buildings};
use crate::world::buildings::zoning::DistrictId;
use crate::world::roads::road_structs::{LaneId, NodeId, SegmentId};
use crate::world::roads::roads::{RoadRegionId, RoadStorage};
use serde::{Deserialize, Serialize};
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
    // Node(NodeId),
    // Segment(LaneId, LaneT),
    House(BuildingId),
}

#[derive(Debug)]
pub struct Address {
    pub destination: DestinationType,

    pub partition: PartitionId,

    pub district: DistrictId,
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Partition {
    pub buildings: Vec<BuildingId>,
    // Maybe nodes later idk
}
impl Partition {
    pub fn chunk_coords(&self, buildings: &Buildings) -> Vec<ChunkCoord> {
        self.buildings
            .iter()
            .flat_map(|b| buildings.storage.get(*b))
            .map(|b| b.position.chunk)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }
    pub fn positions(&self, buildings: &Buildings) -> Vec<WorldPos> {
        self.buildings
            .iter()
            .flat_map(|b| buildings.storage.get(*b))
            .map(|b| b.position)
            .collect()
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct PartitionStorage {
    pub partitions: Vec<Partition>,
    alive: Vec<bool>,
    free_list: Vec<u32>,
    chunk_to_partitions: HashMap<ChunkCoord, Vec<PartitionId>>,
}

impl PartitionStorage {
    pub fn new() -> Self {
        Self {
            partitions: Vec::new(),
            alive: Vec::new(),
            free_list: Vec::new(),
            chunk_to_partitions: HashMap::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            partitions: Vec::with_capacity(capacity),
            alive: Vec::with_capacity(capacity),
            free_list: Vec::new(),
            chunk_to_partitions: HashMap::new(),
        }
    }

    pub fn clear_all(&mut self) {
        self.partitions.clear();
        self.alive.clear();
        self.free_list.clear();
        self.chunk_to_partitions.clear();
    }

    pub fn add(&mut self, partition: Partition, buildings: &Buildings) -> PartitionId {
        let chunk_coords = partition.chunk_coords(buildings);

        let index = if let Some(index) = self.free_list.pop() {
            let idx = index as usize;
            self.partitions[idx] = partition;
            self.alive[idx] = true;
            index
        } else {
            let index = self.partitions.len() as u32;
            self.partitions.push(partition);
            self.alive.push(true);
            index
        };
        for chunk_coord in chunk_coords {
            let ids = self.chunk_to_partitions.entry(chunk_coord).or_default();

            if !ids.contains(&index) {
                ids.push(index);
            }
        }
        index
    }

    pub fn remove(&mut self, id: PartitionId) {
        let idx = id as usize;

        if idx >= self.alive.len() || !self.alive[idx] {
            return;
        }

        self.alive[idx] = false;
        self.free_list.push(id);
        for partitions in self.chunk_to_partitions.values_mut() {
            partitions.retain(|&p| p != id);
        }
    }

    #[inline]
    pub fn is_alive(&self, id: PartitionId) -> bool {
        self.alive.get(id as usize).copied().unwrap_or(false)
    }

    #[inline]
    pub fn region_of(&self, id: PartitionId) -> Option<RoadRegionId> {
        unimplemented!("shit")
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
    pub fn buildings_in(&self, id: PartitionId) -> Option<&[BuildingId]> {
        Some(&self.get(id)?.buildings)
    }

    /// Returns true if both partitions are in the same region tree.
    #[inline]
    pub fn same_region(&self, a: PartitionId, b: PartitionId) -> bool {
        match (self.region_of(a), self.region_of(b)) {
            (Some(ra), Some(rb)) => ra == rb,
            _ => false,
        }
    }
    #[inline]
    pub fn get_partitions_in_chunks(&self, chunks: Vec<ChunkCoord>) -> Vec<PartitionId> {
        chunks
            .into_iter()
            .map(|chunk_coord| self.get_partitions_in_chunk(chunk_coord))
            .flatten()
            .collect()
    }
    #[inline]
    pub fn get_partitions_in_chunk(&self, chunk: ChunkCoord) -> Vec<PartitionId> {
        self.chunk_to_partitions
            .get(&chunk)
            .cloned()
            .unwrap_or_default()
    }
}

impl Default for PartitionStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct PartitionManager {
    pub storage: PartitionStorage,
    regions: HashMap<PartitionId, RoadRegionId>,
}

impl PartitionManager {
    pub fn new() -> Self {
        Self {
            storage: PartitionStorage::with_capacity(32),
            regions: HashMap::new(),
        }
    }

    pub fn rebuild_all(&mut self, road_storage: &RoadStorage) {
        self.storage.clear_all();
        self.regions.clear();
    }
    pub fn add_building(&mut self, building_id: BuildingId, building_pos: WorldPos) {
        let chunks = vec![
            building_pos.chunk,
            building_pos.chunk.offset(0, 1),
            building_pos.chunk.offset(1, 0),
            building_pos.chunk.offset(0, -1),
            building_pos.chunk.offset(-1, 0),
        ];

        const MAX_BUILDINGS_PER_PARTITION: usize = 20;

        let partitions = self.storage.get_partitions_in_chunks(chunks);
    }

    /// Returns the number of separate road regions (disconnected networks).
    #[inline]
    pub fn region_count(&self) -> usize {
        self.regions.len()
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
}

impl Default for PartitionManager {
    fn default() -> Self {
        Self::new()
    }
}
