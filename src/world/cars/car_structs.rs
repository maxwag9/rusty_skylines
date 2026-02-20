use crate::helpers::positions::{ChunkCoord, ChunkSize, LocalPos, WorldPos};
use crate::world::cars::car_simulation::CarSplineSegment;
use crate::world::cars::car_subsystem::make_random_car;
use crate::world::cars::partitions::HierarchicalAddress;
use crate::world::roads::road_structs::LaneId;
use crate::world::roads::roads::{LaneRef, TurnType};
use glam::{Quat, Vec3};
use rand::rngs::ThreadRng;
use rayon::iter::IntoParallelRefMutIterator;
use std::collections::HashMap;
use std::slice::{Iter, IterMut};

pub type CarId = u32;

pub type SimTime = f64;

#[derive(Debug)]
pub enum CarBodyType {
    Sedan, // LADA SEDAN BAKLAJAN11!!!1
    SportsCar,
    HyperCar,
    Bus,
    LKW,
    Buggy,
    Pickup,
    Limousine,
    LimousineForReal, // long one
    Jeep,
}
#[derive(Debug)]
pub enum Powertrain {
    Combustion,
    Electric,
    Hydrogen,
    Hybrid,
}
#[derive(Debug)]
pub struct VehicleType {
    pub body_type: CarBodyType,
    pub power_train: Powertrain,
}
impl VehicleType {
    // === Basic Constructors ===

    pub const fn new(body_type: CarBodyType, power_train: Powertrain) -> Self {
        Self {
            body_type,
            power_train,
        }
    }

    // === "Normal" Everyday Vehicles ===

    /// Your average commuter car
    pub const fn normal() -> Self {
        Self::new(CarBodyType::Sedan, Powertrain::Combustion)
    }

    /// LADA SEDAN BAKLAJAN11!!!1
    pub const fn lada() -> Self {
        Self::new(CarBodyType::Sedan, Powertrain::Combustion)
    }

    /// Boring but eco-friendly
    pub const fn eco_commuter() -> Self {
        Self::new(CarBodyType::Sedan, Powertrain::Electric)
    }

    /// The sensible hybrid choice
    pub const fn prius_energy() -> Self {
        Self::new(CarBodyType::Sedan, Powertrain::Hybrid)
    }

    // === Sports & Performance ===

    /// Weekend warrior
    pub const fn sports() -> Self {
        Self::new(CarBodyType::SportsCar, Powertrain::Combustion)
    }

    /// Tesla Roadster vibes
    pub const fn electric_sports() -> Self {
        Self::new(CarBodyType::SportsCar, Powertrain::Electric)
    }

    /// When money is no object
    pub const fn hypercar() -> Self {
        Self::new(CarBodyType::HyperCar, Powertrain::Combustion)
    }

    /// Rimac Nevera energy
    pub const fn electric_hypercar() -> Self {
        Self::new(CarBodyType::HyperCar, Powertrain::Electric)
    }

    /// For the eco-billionaire
    pub const fn hydrogen_hypercar() -> Self {
        Self::new(CarBodyType::HyperCar, Powertrain::Hydrogen)
    }

    // === Work Vehicles ===

    /// American dream
    pub const fn pickup() -> Self {
        Self::new(CarBodyType::Pickup, Powertrain::Combustion)
    }

    /// Ford F-150 Lightning wannabe
    pub const fn electric_pickup() -> Self {
        Self::new(CarBodyType::Pickup, Powertrain::Electric)
    }

    /// Big rig
    pub const fn truck() -> Self {
        Self::new(CarBodyType::LKW, Powertrain::Combustion)
    }

    /// Tesla Semi dreams
    pub const fn electric_truck() -> Self {
        Self::new(CarBodyType::LKW, Powertrain::Electric)
    }

    /// Hydrogen trucking future
    pub const fn hydrogen_truck() -> Self {
        Self::new(CarBodyType::LKW, Powertrain::Hydrogen)
    }

    // === Public Transport ===

    /// School bus yellow
    pub const fn bus() -> Self {
        Self::new(CarBodyType::Bus, Powertrain::Combustion)
    }

    /// City bus, quiet edition
    pub const fn electric_bus() -> Self {
        Self::new(CarBodyType::Bus, Powertrain::Electric)
    }

    /// The future of public transport
    pub const fn hydrogen_bus() -> Self {
        Self::new(CarBodyType::Bus, Powertrain::Hydrogen)
    }

    // === Off-Road ===

    /// Jeep™ or just jeep
    pub const fn jeep() -> Self {
        Self::new(CarBodyType::Jeep, Powertrain::Combustion)
    }

    /// Rivian vibes
    pub const fn electric_jeep() -> Self {
        Self::new(CarBodyType::Jeep, Powertrain::Electric)
    }

    /// Dune basher
    pub const fn buggy() -> Self {
        Self::new(CarBodyType::Buggy, Powertrain::Combustion)
    }

    /// Silent dune basher
    pub const fn electric_buggy() -> Self {
        Self::new(CarBodyType::Buggy, Powertrain::Electric)
    }

    // === Luxury ===

    /// Airport pickup
    pub const fn limo() -> Self {
        Self::new(CarBodyType::Limousine, Powertrain::Combustion)
    }

    /// Silent luxury
    pub const fn electric_limo() -> Self {
        Self::new(CarBodyType::Limousine, Powertrain::Electric)
    }

    /// Prom night special
    pub const fn stretch_limo() -> Self {
        Self::new(CarBodyType::LimousineForReal, Powertrain::Combustion)
    }

    /// When you want to save the planet in style
    pub const fn electric_stretch_limo() -> Self {
        Self::new(CarBodyType::LimousineForReal, Powertrain::Electric)
    }

    // === Meme Tier ===

    /// Cybertruck but make it weird
    pub const fn cybertruck() -> Self {
        Self::new(CarBodyType::Pickup, Powertrain::Electric)
    }

    /// Nuclear winter edition (it's just hydrogen, relax)
    pub const fn apocalypse_ready() -> Self {
        Self::new(CarBodyType::Jeep, Powertrain::Hydrogen)
    }

    /// Maximum overcompensation
    pub const fn midlife_crisis() -> Self {
        Self::new(CarBodyType::SportsCar, Powertrain::Combustion)
    }

    /// Tech bro starter pack
    pub const fn tech_bro() -> Self {
        Self::new(CarBodyType::SportsCar, Powertrain::Electric)
    }

    /// Dad mode activated
    pub const fn dad_car() -> Self {
        Self::new(CarBodyType::Sedan, Powertrain::Hybrid)
    }

    // === Utility Methods ===

    pub const fn is_electric(&self) -> bool {
        matches!(self.power_train, Powertrain::Electric)
    }

    pub const fn is_eco_friendly(&self) -> bool {
        matches!(
            self.power_train,
            Powertrain::Electric | Powertrain::Hydrogen | Powertrain::Hybrid
        )
    }

    pub const fn is_luxury(&self) -> bool {
        matches!(
            self.body_type,
            CarBodyType::Limousine | CarBodyType::LimousineForReal | CarBodyType::HyperCar
        )
    }

    pub const fn is_offroad(&self) -> bool {
        matches!(self.body_type, CarBodyType::Jeep | CarBodyType::Buggy)
    }

    pub const fn is_commercial(&self) -> bool {
        matches!(self.body_type, CarBodyType::Bus | CarBodyType::LKW)
    }
}
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
    pub car_chunk_storage: CarChunkStorage,
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
    /// Returns a parallel mutable iterator over car slots.
    /// Each slot is independent, so this is safe for rayon.
    pub fn par_iter_mut_cars(&mut self) -> rayon::slice::IterMut<'_, Option<Car>> {
        self.cars.par_iter_mut()
    }
}

impl CarStorage {
    pub(crate) fn update_carchunk_distances(
        &mut self,
        target_pos: WorldPos,
        chunk_size: ChunkSize,
    ) {
        let (moved, removed) = self
            .car_chunk_storage
            .update_all_distances(target_pos.chunk, chunk_size);

        if !removed.is_empty() {
            println!("Removed {} empty car chunks", removed.len());
        }
        if moved > 0 {
            println!("Moved {} chunks between distance tiers", moved);
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
    close: HashMap<ChunkCoord, CarChunk>,
    medium: HashMap<ChunkCoord, CarChunk>,
    far: HashMap<ChunkCoord, CarChunk>,
}

impl CarChunkStorage {
    pub fn new() -> Self {
        Self {
            close: HashMap::new(),
            medium: HashMap::new(),
            far: HashMap::new(),
        }
    }

    // ========================================================================
    // Direct tier accessors - O(1) access to distance-specific chunks
    // ========================================================================

    #[inline]
    pub fn close(&self) -> &HashMap<ChunkCoord, CarChunk> {
        &self.close
    }

    #[inline]
    pub fn close_mut(&mut self) -> &mut HashMap<ChunkCoord, CarChunk> {
        &mut self.close
    }

    #[inline]
    pub fn medium(&self) -> &HashMap<ChunkCoord, CarChunk> {
        &self.medium
    }

    #[inline]
    pub fn medium_mut(&mut self) -> &mut HashMap<ChunkCoord, CarChunk> {
        &mut self.medium
    }

    #[inline]
    pub fn far(&self) -> &HashMap<ChunkCoord, CarChunk> {
        &self.far
    }

    #[inline]
    pub fn far_mut(&mut self) -> &mut HashMap<ChunkCoord, CarChunk> {
        &mut self.far
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    #[inline]
    fn map_for_distance(&self, dist: &CarChunkDistance) -> &HashMap<ChunkCoord, CarChunk> {
        match dist {
            CarChunkDistance::Close => &self.close,
            CarChunkDistance::Medium => &self.medium,
            CarChunkDistance::Far => &self.far,
        }
    }

    #[inline]
    fn map_for_distance_mut(
        &mut self,
        dist: &CarChunkDistance,
    ) -> &mut HashMap<ChunkCoord, CarChunk> {
        match dist {
            CarChunkDistance::Close => &mut self.close,
            CarChunkDistance::Medium => &mut self.medium,
            CarChunkDistance::Far => &mut self.far,
        }
    }

    /// Find which distance tier a chunk is in (if it exists)
    #[inline]
    pub fn find_distance(&self, coord: &ChunkCoord) -> Option<CarChunkDistance> {
        if self.close.contains_key(coord) {
            Some(CarChunkDistance::Close)
        } else if self.medium.contains_key(coord) {
            Some(CarChunkDistance::Medium)
        } else if self.far.contains_key(coord) {
            Some(CarChunkDistance::Far)
        } else {
            None
        }
    }

    // ========================================================================
    // Lookup by coord (searches all tiers, still O(1))
    // ========================================================================

    #[inline]
    pub fn get(&self, coord: &ChunkCoord) -> Option<&CarChunk> {
        self.close
            .get(coord)
            .or_else(|| self.medium.get(coord))
            .or_else(|| self.far.get(coord))
    }

    #[inline]
    pub fn get_mut(&mut self, coord: &ChunkCoord) -> Option<&mut CarChunk> {
        if let Some(chunk) = self.close.get_mut(coord) {
            return Some(chunk);
        }
        if let Some(chunk) = self.medium.get_mut(coord) {
            return Some(chunk);
        }
        self.far.get_mut(coord)
    }

    #[inline]
    pub fn contains(&self, coord: &ChunkCoord) -> bool {
        self.close.contains_key(coord)
            || self.medium.contains_key(coord)
            || self.far.contains_key(coord)
    }

    pub fn remove(&mut self, coord: &ChunkCoord) -> Option<CarChunk> {
        self.close
            .remove(coord)
            .or_else(|| self.medium.remove(coord))
            .or_else(|| self.far.remove(coord))
    }

    // ========================================================================
    // Iteration over all chunks
    // ========================================================================

    pub fn iter(&self) -> impl Iterator<Item = (&ChunkCoord, &CarChunk)> {
        self.close
            .iter()
            .chain(self.medium.iter())
            .chain(self.far.iter())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&ChunkCoord, &mut CarChunk)> {
        self.close
            .iter_mut()
            .chain(self.medium.iter_mut())
            .chain(self.far.iter_mut())
    }

    pub fn values(&self) -> impl Iterator<Item = &CarChunk> {
        self.close
            .values()
            .chain(self.medium.values())
            .chain(self.far.values())
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut CarChunk> {
        self.close
            .values_mut()
            .chain(self.medium.values_mut())
            .chain(self.far.values_mut())
    }

    pub fn keys(&self) -> impl Iterator<Item = &ChunkCoord> {
        self.close
            .keys()
            .chain(self.medium.keys())
            .chain(self.far.keys())
    }

    // ========================================================================
    // Fast distance-specific car accessors - TRUE O(1) for close cars!
    // ========================================================================

    /// O(close_chunk_count) - only iterates close chunks
    pub fn close_cars(&self) -> Vec<CarId> {
        self.close
            .values()
            .flat_map(|chunk| chunk.car_ids.iter().copied())
            .collect()
    }

    /// Iterator version - zero allocation
    pub fn close_car_ids(&self) -> impl Iterator<Item = CarId> + '_ {
        self.close
            .values()
            .flat_map(|chunk| chunk.car_ids.iter().copied())
    }

    /// O(medium_chunk_count)
    pub fn medium_cars(&self) -> Vec<CarId> {
        self.medium
            .values()
            .flat_map(|chunk| chunk.car_ids.iter().copied())
            .collect()
    }

    pub fn medium_car_ids(&self) -> impl Iterator<Item = CarId> + '_ {
        self.medium
            .values()
            .flat_map(|chunk| chunk.car_ids.iter().copied())
    }

    /// O(far_chunk_count)
    pub fn far_cars(&self) -> Vec<CarId> {
        self.far
            .values()
            .flat_map(|chunk| chunk.car_ids.iter().copied())
            .collect()
    }

    pub fn far_car_ids(&self) -> impl Iterator<Item = CarId> + '_ {
        self.far
            .values()
            .flat_map(|chunk| chunk.car_ids.iter().copied())
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    pub fn close_chunk_count(&self) -> usize {
        self.close.len()
    }

    pub fn medium_chunk_count(&self) -> usize {
        self.medium.len()
    }

    pub fn far_chunk_count(&self) -> usize {
        self.far.len()
    }

    pub fn total_chunk_count(&self) -> usize {
        self.close.len() + self.medium.len() + self.far.len()
    }

    pub fn close_car_count(&self) -> usize {
        self.close.values().map(|c| c.car_ids.len()).sum()
    }

    pub fn medium_car_count(&self) -> usize {
        self.medium.values().map(|c| c.car_ids.len()).sum()
    }

    pub fn far_car_count(&self) -> usize {
        self.far.values().map(|c| c.car_ids.len()).sum()
    }

    // ========================================================================
    // Mutation operations
    // ========================================================================

    pub fn add_car(&mut self, chunk_coord: ChunkCoord, dist: CarChunkDistance, car_id: CarId) {
        let map = self.map_for_distance_mut(&dist);
        let chunk = map
            .entry(chunk_coord)
            .or_insert_with(|| CarChunk::empty(dist.clone()));

        debug_assert_eq!(
            chunk.distance, dist,
            "CarChunkDistance mismatch when adding to existing chunk"
        );

        chunk.car_ids.push(car_id);
    }

    pub fn remove_car(&mut self, chunk_coord: ChunkCoord, car_id: CarId) {
        // Try each tier - only one can contain the chunk
        let maps: [&mut HashMap<ChunkCoord, CarChunk>; 3] =
            [&mut self.close, &mut self.medium, &mut self.far];

        for map in maps {
            if let Some(chunk) = map.get_mut(&chunk_coord) {
                chunk.car_ids.retain(|&x| x != car_id);
                if chunk.car_ids.is_empty() {
                    map.remove(&chunk_coord);
                }
                return;
            }
        }
    }

    /// Move a chunk to a different distance tier. Returns true if moved.
    pub fn update_chunk_distance(
        &mut self,
        coord: ChunkCoord,
        new_distance: CarChunkDistance,
    ) -> bool {
        // Find current tier
        let current = if self.close.contains_key(&coord) {
            CarChunkDistance::Close
        } else if self.medium.contains_key(&coord) {
            CarChunkDistance::Medium
        } else if self.far.contains_key(&coord) {
            CarChunkDistance::Far
        } else {
            return false;
        };

        // Already in correct tier
        if current == new_distance {
            return false;
        }

        // Remove from old tier
        let mut chunk = match current {
            CarChunkDistance::Close => self.close.remove(&coord),
            CarChunkDistance::Medium => self.medium.remove(&coord),
            CarChunkDistance::Far => self.far.remove(&coord),
        }
        .expect("Chunk must exist, we just checked");

        // Update distance marker and insert into new tier
        chunk.distance = new_distance.clone();
        let new_map = self.map_for_distance_mut(&new_distance);
        new_map.insert(coord, chunk);

        true
    }

    /// Bulk update all chunk distances. Returns number of chunks moved.
    pub fn update_all_distances(
        &mut self,
        center_chunk: ChunkCoord,
        chunk_size: ChunkSize,
    ) -> (usize, Vec<ChunkCoord>) {
        let mut moved = 0;
        let mut to_remove: Vec<ChunkCoord> = Vec::new();

        // Collect all coords and their new distances
        let updates: Vec<(ChunkCoord, CarChunkDistance, bool)> = self
            .iter()
            .map(|(coord, chunk)| {
                let is_empty = chunk.car_ids.is_empty();
                let dist2 = center_chunk.dist2(coord);
                let new_dist = CarChunkDistance::from_dist2(dist2, chunk_size);
                (*coord, new_dist, is_empty)
            })
            .collect();

        // Apply updates
        for (coord, new_dist, is_empty) in updates {
            if is_empty {
                to_remove.push(coord);
            } else if self.update_chunk_distance(coord, new_dist) {
                moved += 1;
            }
        }

        // Remove empty chunks
        for coord in &to_remove {
            self.remove(coord);
        }

        (moved, to_remove)
    }
}
#[derive(Debug)]
pub struct Car {
    // === Identity ===
    pub id: CarId,
    pub vehicle_type: VehicleType,
    pub color: [f32; 3],

    // === Physical state ===
    pub next_splines: Vec<CarSplineSegment>,
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
    pub destination_addr: Option<HierarchicalAddress>,

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
            vehicle_type: VehicleType::normal(),
            color: [1.0, 0.0, 0.0],
            next_splines: vec![],
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
            destination_addr: None,
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
