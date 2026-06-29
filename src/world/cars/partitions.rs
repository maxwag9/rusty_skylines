use crate::helpers::positions::{ChunkCoord, WorldPos};
use crate::world::buildings::buildings::{BuildingId, BuildingStorage, Buildings};
use crate::world::buildings::zoning::{DistrictId, ZoningStorage};
use crate::world::roads::road_structs::NodeId;
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

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum DestinationType {
    // Node(NodeId),
    // Segment(LaneId, LaneT),
    Building(BuildingId),
}
impl DestinationType {
    pub fn as_building(&self) -> Option<BuildingId> {
        match self {
            DestinationType::Building(building_id) => Some(*building_id),
        }
    }
}
#[derive(Debug)]
pub struct Address {
    pub destination: DestinationType,
}
impl Address {
    pub fn partition(&self, buildings: &BuildingStorage) -> Option<PartitionId> {
        match self.destination {
            DestinationType::Building(b_id) => buildings.get_partition_of_building(b_id),
        }
    }
    pub fn district(
        &self,
        buildings: &BuildingStorage,
        zoning_storage: &ZoningStorage,
    ) -> Option<DistrictId> {
        match self.destination {
            DestinationType::Building(b_id) => Some(
                zoning_storage
                    .get_lot(buildings.get(b_id)?.lot_id)?
                    .district_id,
            ),
        }
    }
    #[inline]
    pub fn building_id(&self) -> Option<BuildingId> {
        self.destination.as_building()
    }
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
            .map(|b| b.pos.chunk)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }
    pub fn positions(&self, buildings: &Buildings) -> Vec<WorldPos> {
        self.buildings
            .iter()
            .flat_map(|b| buildings.storage.get(*b))
            .map(|b| b.pos)
            .collect()
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct PartitionStorage {
    pub partitions: Vec<Partition>,
    pub alive: Vec<bool>,
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

    pub fn add(buildings: &mut Buildings, partition: Partition) -> PartitionId {
        let chunk_coords = partition.chunk_coords(buildings);
        let storage = &mut buildings.partitions.storage;
        let index = if let Some(index) = storage.free_list.pop() {
            let idx = index as usize;
            storage.partitions[idx] = partition;
            storage.alive[idx] = true;
            index
        } else {
            let index = storage.partitions.len() as u32;
            storage.partitions.push(partition);
            storage.alive.push(true);
            index
        };
        for chunk_coord in chunk_coords {
            let ids = storage.chunk_to_partitions.entry(chunk_coord).or_default();

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
    fn remove_partition_from_chunk_map(
        &mut self,
        partition_id: PartitionId,
        chunks: &[ChunkCoord],
    ) {
        for chunk in chunks.iter().copied() {
            if let Some(ids) = self.chunk_to_partitions.get_mut(&chunk) {
                ids.retain(|&p| p != partition_id);
                if ids.is_empty() {
                    self.chunk_to_partitions.remove(&chunk);
                }
            }
        }
    }

    fn add_partition_to_chunk_map(&mut self, partition_id: PartitionId, chunks: &[ChunkCoord]) {
        for chunk in chunks.iter().copied() {
            let ids = self.chunk_to_partitions.entry(chunk).or_default();
            if !ids.contains(&partition_id) {
                ids.push(partition_id);
            }
        }
    }

    fn reconcile_partition_chunks(
        &mut self,
        partition_id: PartitionId,
        old_chunks: &[ChunkCoord],
        new_chunks: &[ChunkCoord],
    ) {
        let old: std::collections::HashSet<_> = old_chunks.iter().copied().collect();
        let new: std::collections::HashSet<_> = new_chunks.iter().copied().collect();

        for chunk in old.difference(&new).copied() {
            if let Some(ids) = self.chunk_to_partitions.get_mut(&chunk) {
                ids.retain(|&p| p != partition_id);
                if ids.is_empty() {
                    self.chunk_to_partitions.remove(&chunk);
                }
            }
        }

        for chunk in new.difference(&old).copied() {
            let ids = self.chunk_to_partitions.entry(chunk).or_default();
            if !ids.contains(&partition_id) {
                ids.push(partition_id);
            }
        }
    }

    pub fn remove_building_from_partition(
        buildings: &mut Buildings,
        partition_id: PartitionId,
        building_id: BuildingId,
    ) -> bool {
        let Some(old_chunks) = buildings
            .partitions
            .storage
            .get(partition_id)
            .map(|p| p.chunk_coords(buildings))
        else {
            return false;
        };

        let became_empty: bool;

        {
            let Some(partition) = buildings.partitions.storage.get_mut(partition_id) else {
                return false;
            };

            let Some(pos) = partition.buildings.iter().position(|&b| b == building_id) else {
                return false;
            };

            partition.buildings.swap_remove(pos);
            became_empty = partition.buildings.is_empty();
        }

        if became_empty {
            buildings.partitions.storage.remove(partition_id);
        } else {
            let new_chunks = buildings
                .partitions
                .storage
                .get(partition_id)
                .map(|p| p.chunk_coords(buildings))
                .unwrap_or_default();

            buildings.partitions.storage.reconcile_partition_chunks(
                partition_id,
                &old_chunks,
                &new_chunks,
            );
        }

        true
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
        // self.storage.clear_all();
        // self.regions.clear();
    }
    pub fn add_building(
        buildings: &mut Buildings,
        building_id: BuildingId,
        building_pos: WorldPos,
    ) -> PartitionId {
        const MAX_BUILDINGS_PER_PARTITION: usize = 20;
        const MAX_DISTANCE: f64 = 200.0;

        let chunks = building_pos.chunk.get_chunks_plus();
        let candidate_partitions = buildings
            .partitions
            .storage
            .get_partitions_in_chunks(chunks);

        let mut best_partition = None;
        let mut best_distance = f64::MAX;

        for partition_id in candidate_partitions {
            let Some(partition) = buildings.partitions.storage.get(partition_id) else {
                continue;
            };

            if partition.buildings.len() >= MAX_BUILDINGS_PER_PARTITION {
                continue;
            }

            let closest_distance = partition
                .buildings
                .iter()
                .filter_map(|id| buildings.storage.get(*id))
                .map(|building| building.pos.distance_to(building_pos))
                .fold(f64::MAX, |acc, d| acc.min(d));

            if closest_distance < best_distance {
                best_distance = closest_distance;
                best_partition = Some(partition_id);
            }
        }

        if let Some(partition_id) = best_partition {
            if best_distance <= MAX_DISTANCE {
                if let Some(partition) = buildings.partitions.storage.get_mut(partition_id) {
                    partition.buildings.push(building_id);
                    return partition_id;
                }
            }
        }

        PartitionStorage::add(
            buildings,
            Partition {
                buildings: vec![building_id],
            },
        )
    }

    pub fn remove_building(buildings: &mut Buildings, building_id: BuildingId) {
        let Some(partition_id) = buildings.storage.get_partition_of_building(building_id) else {
            return;
        };

        let removed =
            PartitionStorage::remove_building_from_partition(buildings, partition_id, building_id);

        if removed {
            buildings.storage.clear_partition_of_building(building_id);
        }
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
