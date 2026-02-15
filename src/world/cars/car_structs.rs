use crate::helpers::positions::{ChunkCoord, ChunkSize, LocalPos, WorldPos};
use crate::world::cars::car_subsystem::make_random_car;
use crate::world::roads::road_structs::LaneId;
use crate::world::roads::roads::{LaneRef, TurnType};
use glam::{Quat, Vec3};
use rand::rngs::ThreadRng;
use std::collections::HashMap;
use std::slice::{Iter, IterMut};

type SimTime = f64;

pub type CarId = u32;

type VehicleType = u32;
type PartitionId = u32;
#[derive(Debug, PartialEq, Clone)]
pub enum CarChunkDistance {
    Close,
    Medium,
    Far,
}

impl CarChunkDistance {
    #[inline]
    pub fn from_dist2(dist2: u32, chunk_size: ChunkSize) -> Self {
        const BASE_CHUNK_SIZE: f32 = 128.0;
        const CLOSE_CHUNKS: f32 = 10.0;
        const MEDIUM_CHUNKS: f32 = 100.0;

        const CLOSE_DIST2_BASE: f32 = CLOSE_CHUNKS * CLOSE_CHUNKS;
        const MEDIUM_DIST2_BASE: f32 = MEDIUM_CHUNKS * MEDIUM_CHUNKS;

        let cs = chunk_size as f32;
        let scale = BASE_CHUNK_SIZE / cs;
        let close_thresh = CLOSE_DIST2_BASE * scale.powi(2);
        let medium_thresh = MEDIUM_DIST2_BASE * scale.powi(2);

        let d = dist2 as f32;
        if d < close_thresh {
            Self::Close
        } else if d < medium_thresh {
            Self::Medium
        } else {
            Self::Far
        }
    }
    #[inline]
    pub fn from_chunk_positions(
        center_chunk: ChunkCoord,
        other: ChunkCoord,
        chunk_size: ChunkSize,
    ) -> Self {
        let dist2 = center_chunk.dist2(&other);
        CarChunkDistance::from_dist2(dist2, chunk_size)
    }
}
#[derive(Clone, Debug)]
pub struct CarChunk {
    pub distance: CarChunkDistance,
    pub car_ids: Vec<CarId>,
    pub last_update_time: SimTime,
}

impl CarChunk {
    pub fn new(distance: CarChunkDistance, car_ids: Vec<CarId>) -> Self {
        Self {
            distance,
            car_ids,
            last_update_time: 0.0,
        }
    }
    pub fn empty(car_chunk_distance: CarChunkDistance) -> Self {
        Self {
            distance: car_chunk_distance,
            car_ids: Vec::new(),
            last_update_time: 0.0,
        }
    }
}

pub struct CarStorage {
    car_chunk_storage: CarChunkStorage,
    cars: Vec<Option<Car>>,
    free_list: Vec<CarId>,
    // Reverse mapping so I can remove from chunks in O(1) when despawning/moving
    car_locations: HashMap<CarId, ChunkCoord>,
    chunk_size: ChunkSize,
    center_chunk: ChunkCoord,
}

impl CarStorage {
    pub(crate) fn update_target_and_chunk_size(
        &mut self,
        target_chunk: ChunkCoord,
        chunk_size: ChunkSize,
    ) {
        self.center_chunk = target_chunk;
        self.chunk_size = chunk_size;
    }
    pub(crate) fn move_car_between_chunks(
        &mut self,
        from: ChunkCoord,
        to: ChunkCoord,
        car_id: CarId,
    ) {
        self.car_chunk_storage.remove_car(from, car_id);
        let car_chunk_distance =
            CarChunkDistance::from_chunk_positions(self.center_chunk, to, self.chunk_size);
        self.car_chunk_storage
            .add_car(to, car_chunk_distance, car_id);
    }
    pub(crate) fn iter_cars(&mut self) -> Iter<'_, Option<Car>> {
        self.cars.iter()
    }
    pub(crate) fn iter_mut_cars(&mut self) -> IterMut<'_, Option<Car>> {
        self.cars.iter_mut()
    }
}

impl CarStorage {
    pub fn car_chunks(&self) -> &CarChunkStorage {
        &self.car_chunk_storage
    }
    // pub fn car_chunks_mut(&mut self) -> &mut CarChunkStorage {
    //     &mut self.car_chunk_storage
    // }
    // pub fn car_locations(&self) -> &HashMap<CarId, ChunkCoord> {
    //     &self.car_locations
    // }
    // pub fn car_locations_mut(&mut self) -> &mut HashMap<CarId, ChunkCoord> {
    //     &mut self.car_locations
    // }
    pub(crate) fn update_carchunk_distances(
        &mut self,
        target_pos: WorldPos,
        chunk_size: ChunkSize,
    ) {
        let mut to_remove = Vec::new();
        for (coord, carchunk) in self.car_chunk_storage.chunks.iter_mut() {
            if carchunk.car_ids.is_empty() {
                to_remove.push(*coord);
                println!("removed car chunk");
                continue;
            }
            let dist2 = target_pos.chunk.dist2(coord);
            let distance_type = CarChunkDistance::from_dist2(dist2, chunk_size);
            carchunk.distance = distance_type;
        }
        for coord in to_remove {
            self.car_chunk_storage.chunks.remove(&coord);
        }
    }
}

impl CarStorage {
    pub fn new() -> Self {
        let mut cars: Vec<Option<Car>> = Vec::new();
        let mut rng = ThreadRng::default();
        let car = make_random_car(
            WorldPos::new(
                ChunkCoord::new(-69420, -69420),
                LocalPos::new(0.1, 50.5, 20.0),
            ),
            &mut rng,
        );
        cars.push(Some(car)); // reserve index 0 — never use it for a real car, because car 0 doesn't get RTX shadows.
        Self {
            car_chunk_storage: CarChunkStorage::new(),
            cars,
            free_list: Vec::new(),
            car_locations: HashMap::new(),
            chunk_size: 128,
            center_chunk: ChunkCoord::zero(),
        }
    }

    pub fn spawn(
        &mut self,
        chunk_coord: ChunkCoord,
        car_chunk_distance: CarChunkDistance,
        mut car: Car,
    ) -> CarId {
        let car_id = if let Some(reused_id) = self.free_list.pop() {
            // Reuse slot - III know it's None because it's in free_list
            car.id = reused_id;
            self.cars[reused_id as usize] = Some(car);
            reused_id
        } else {
            let new_id = self.cars.len() as u32;
            car.id = new_id;
            self.cars.push(Some(car));
            new_id
        };

        // Add to chunk storage and record location
        self.car_chunk_storage
            .add_car(chunk_coord, car_chunk_distance, car_id);
        self.car_locations.insert(car_id, chunk_coord);

        car_id
    }

    pub fn despawn(&mut self, id: CarId) {
        if id == 0 {
            // index 0 is reserved, ignore attempts to despawn it
            return;
        }
        if self
            .cars
            .get(id as usize)
            .and_then(|opt| opt.as_ref())
            .is_some()
        {
            // Remove from chunk first using reverse lookup
            if let Some(chunk_coord) = self.car_locations.remove(&id) {
                self.car_chunk_storage.remove_car(chunk_coord, id);
            }

            // Actually free the slot
            self.cars[id as usize] = None;
            self.free_list.push(id);
        }
    }

    pub fn car_count(&self) -> usize {
        self.cars.len() - self.free_list.len() - 1
    }

    #[inline]
    pub fn get(&self, id: CarId) -> Option<&Car> {
        self.cars.get(id as usize)?.as_ref()
    }

    #[inline]
    pub fn get_mut(&mut self, id: CarId) -> Option<&mut Car> {
        self.cars.get_mut(id as usize)?.as_mut()
    }
}

pub struct CarChunkStorage {
    pub chunks: HashMap<ChunkCoord, CarChunk>,
}

impl CarChunkStorage {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
        }
    }

    pub fn add_car(&mut self, chunk_coord: ChunkCoord, dist: CarChunkDistance, car_id: CarId) {
        let chunk = if let Some(existing) = self.chunks.get_mut(&chunk_coord) {
            // Enforce distance consistency - panic in debug if caller screws up
            debug_assert_eq!(
                existing.distance, dist,
                "CarChunkDistance mismatch when adding to existing chunk"
            );
            existing
        } else {
            // New chunk
            self.chunks
                .entry(chunk_coord)
                .or_insert_with(|| CarChunk::empty(dist))
        };

        chunk.car_ids.push(car_id);
    }

    pub fn remove_car(&mut self, chunk_coord: ChunkCoord, car_id: CarId) {
        if let Some(chunk) = self.chunks.get_mut(&chunk_coord) {
            chunk.car_ids.retain(|&x| x != car_id);
            if chunk.car_ids.is_empty() {
                self.chunks.remove(&chunk_coord);
            }
        }
    }
    pub fn close_cars(&self) -> Vec<CarId> {
        let mut close_cars: Vec<CarId> = Vec::new();
        for chunk in self.chunks.values() {
            if chunk.distance == CarChunkDistance::Close {
                close_cars.extend(&chunk.car_ids);
            }
        }
        close_cars
    }
}
#[derive(Debug)]
pub struct Car {
    // === Identity ===
    pub id: CarId,
    pub vehicle_type: VehicleType, // bitmask index: car, bus, truck, emergency
    pub color: [f32; 3],

    // === Physical state ===
    pub pos: WorldPos,
    pub quat: Quat,
    pub current_velocity: Vec3, // planar velocity (y = 0)
    pub desired_velocity: Vec3, // Don't use
    pub steering_angle: f32,    // road wheel angle (rad), +left
    pub steering_vel: f32,      // (rad/s)
    pub yaw_rate: f32,          // (rad/s) in SAE sign ( + = yaw left )
    pub length: f32,            // meters
    pub width: f32,
    pub accel: f32, // m/s²
    pub decel: f32, // m/s² (positive value)

    // === Topology state ===
    pub lane: LaneRef, // current lane
    pub lane_s: f32,   // position along lane [0..lane.length]

    // === Short-term intent (IMPORTANT) ===
    pub committed_arm: Option<u8>, // arm index at next_node (None until chosen)
    pub committed_lane: Option<LaneRef>, // chosen outgoing lane

    // === Destination ===
    pub destination_addr: HierarchicalAddress,

    // === Decision memory (very small) ===
    pub last_turn: Option<TurnType>,

    // === Timing ===
    pub spawn_time: SimTime,
    pub last_decision_time: SimTime,

    // === Reporting ===
    pub entered_arm_time: Option<SimTime>, // when entering current arm

    pub driver_profile: DriverProfile,
}

impl Default for Car {
    fn default() -> Car {
        Self {
            id: 0,
            vehicle_type: 0,
            color: [1.0, 0.0, 0.0],
            pos: Default::default(),
            quat: Quat::IDENTITY,
            current_velocity: Vec3::ZERO,
            desired_velocity: Vec3::ZERO, // Don't use

            steering_angle: 0.0,
            steering_vel: 0.0,
            yaw_rate: 0.0,

            length: 4.0,
            width: 2.2,
            accel: 7.0,
            decel: 10.0,
            lane: LaneRef::Segment(LaneId(0), 0),
            lane_s: 0.0,
            committed_arm: None,
            committed_lane: None,
            destination_addr: HierarchicalAddress { address: vec![] },
            last_turn: None,
            spawn_time: 0.0,
            last_decision_time: 0.0,
            entered_arm_time: None,
            driver_profile: DriverProfile::Normal,
        }
    }
}

#[derive(Debug)]
pub enum DriverProfile {
    Normal,           // Normal-ass driver, why not?
    Aggressive,       // Just aggressive, fast and dangerous
    GermanTame,       // Chill, well-educated in german driving school lol
    GermanAggressive, // Likely to honk, drive fast on highways, likelier in cars that resemble Audi or BMW, but SKILLED in comparison to just 'Aggressive'
    Anxious,          // Just got out of driving school or other anxiousness
}
#[derive(Debug)]
pub struct HierarchicalAddress {
    pub address: Vec<PartitionId>, // First index is smallest, most precise, bigger indices are bigger Partitions.
}
