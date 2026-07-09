use crate::helpers::positions::ChunkCoord;
use crate::world::buildings::zoning::ParkingSpot;
use serde::{Deserialize, Serialize};
use std::slice::{Iter, IterMut};

pub type ParkingSpotId = u32;
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct ParkingStorage {
    parking_spots: Vec<Option<ParkingSpot>>,
    free_list: Vec<ParkingSpotId>,
    center_chunk: ChunkCoord,
}

impl ParkingStorage {
    pub fn iter(&self) -> Iter<'_, Option<ParkingSpot>> {
        self.parking_spots.iter()
    }
    pub fn iter_mut(&mut self) -> IterMut<'_, Option<ParkingSpot>> {
        self.parking_spots.iter_mut()
    }

    pub fn new() -> Self {
        Self {
            parking_spots: Vec::new(),
            free_list: Vec::new(),
            center_chunk: ChunkCoord::zero(),
        }
    }

    pub fn spawn(&mut self, mut parking_spot: ParkingSpot) -> ParkingSpotId {
        let parking_spot_id = if let Some(reused_id) = self.free_list.pop() {
            // Reuse slot - III know it's None because it's in free_list
            parking_spot.id = reused_id;
            self.parking_spots[reused_id as usize] = Some(parking_spot);
            reused_id
        } else {
            let new_id = self.parking_spots.len() as u32;
            parking_spot.id = new_id;
            self.parking_spots.push(Some(parking_spot));
            new_id
        };

        parking_spot_id
    }

    pub fn despawn<I>(&mut self, id: I)
    where
        I: Into<Option<ParkingSpotId>>,
    {
        let Some(id) = id.into() else {
            return;
        };

        self.free_list.push(id);
        let Some(building) = self.parking_spots[id as usize].take() else {
            return;
        };
    }

    pub fn parking_spot_count(&self) -> usize {
        self.parking_spots.len() - self.free_list.len()
    }

    #[inline]
    pub fn get<I>(&self, id: I) -> Option<&ParkingSpot>
    where
        I: Into<Option<ParkingSpotId>>,
    {
        let id = id.into()?;
        self.parking_spots.get(id as usize)?.as_ref()
    }

    #[inline]
    pub fn get_mut<I>(&mut self, id: I) -> Option<&mut ParkingSpot>
    where
        I: Into<Option<ParkingSpotId>>,
    {
        let id = id.into()?;
        self.parking_spots.get_mut(id as usize)?.as_mut()
    }
}
