use crate::positions::WorldPos;
use crate::terrain::roads::roads::{LaneRef, TurnType};

type SimTime = f64;
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CarId {
    index: u32,
    generation: u16,
}
impl CarId {
    #[inline]
    pub fn new(index: u32, generation: u16) -> Self {
        Self { index, generation }
    }
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }
    #[inline]
    pub fn generation(&self) -> u16 {
        self.generation
    }
}

type VehicleType = u32;
type PartitionId = u32;
pub enum CarChunkDistance {
    Close,
    Medium,
    Far,
}

pub struct CarChunk {
    pub distance: CarChunkDistance,
    pub car_ids: Vec<CarId>,
}
pub struct CarSlot {
    car: Option<Car>,
    generation: u16,
}

pub struct CarStorage {
    cars: Vec<CarSlot>,
    free_list: Vec<CarId>,
}
impl CarStorage {
    pub fn new() -> Self {
        Self {
            cars: Vec::new(),
            free_list: Vec::new(),
        }
    }

    pub fn spawn(&mut self, mut car: Car) -> CarId {
        if let Some(index) = self.free_list.pop() {
            let slot = &mut self.cars[index.index() as usize];

            // generation already incremented on despawn
            let generation = slot.generation;
            car.id = CarId::new(index.index(), generation);

            slot.car = Some(car);
            CarId::new(index.index(), generation)
        } else {
            let index = self.cars.len() as u32;
            let generation = 0;

            car.id = CarId::new(index, generation);

            self.cars.push(CarSlot {
                car: Some(car),
                generation,
            });

            CarId::new(index, generation)
        }
    }

    pub fn despawn(&mut self, id: CarId) {
        let index = id.index() as usize;

        if let Some(slot) = self.cars.get_mut(index) {
            // reject stale ids
            if slot.generation != id.generation() {
                return;
            }

            slot.car = None;
            slot.generation = slot.generation.wrapping_add(1);
            self.free_list.push(id);
        }
    }

    #[inline]
    pub fn get(&self, id: CarId) -> Option<&Car> {
        let slot = self.cars.get(id.index() as usize)?;
        if slot.generation == id.generation() {
            slot.car.as_ref()
        } else {
            None
        }
    }

    #[inline]
    pub fn get_mut(&mut self, id: CarId) -> Option<&mut Car> {
        let slot = self.cars.get_mut(id.index() as usize)?;
        if slot.generation == id.generation() {
            slot.car.as_mut()
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct Car {
    // === Identity ===
    pub id: CarId,
    pub vehicle_type: VehicleType, // bitmask index: car, bus, truck, emergency

    // === Physical state ===
    pub pos: WorldPos,
    pub speed: f32,         // m/s
    pub desired_speed: f32, // from lane speed limit + personality
    pub length: f32,        // meters
    pub accel: f32,         // m/s²
    pub decel: f32,         // m/s² (positive value)

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
