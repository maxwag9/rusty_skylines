use crate::world::buildings::zoning::ZoningType;
use crate::world::statisticals::schedule::SchedulePhase;
use rand::{Rng, RngExt};
use revision::revisioned;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
#[revisioned(revision = 1)]
pub struct TransportStats {
    pub commuting_outbound: u32,
    pub commuting_inbound: u32,
    pub shopping_outbound: u32,
    pub shopping_inbound: u32,
    pub transporting: u32,
}
impl TransportStats {
    pub fn new() -> TransportStats {
        TransportStats {
            commuting_outbound: 0,
            commuting_inbound: 0,
            shopping_outbound: 0,
            shopping_inbound: 0,
            transporting: 0,
        }
    }
    pub fn update(&mut self) {}
    #[inline]
    pub fn add_car(&mut self, car_trip_type: CarTripType) { // TODO: This is shit! +1
        // match car_trip_type {
        //     CarTripType::None => {
        //
        //     }
        //     CarTripType::CommutingOutbound => {
        //         self.commuting_outbound += 1;
        //     }
        //     CarTripType::CommutingInbound => {
        //         self.commuting_inbound += 1;
        //     }
        //     CarTripType::ShoppingOutbound => {
        //         self.shopping_outbound += 1;
        //     }
        //     CarTripType::ShoppingInbound => {
        //         self.shopping_inbound += 1;
        //     }
        //     CarTripType::Transporting => {
        //         self.transporting += 1;
        //     }
        // }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CarTripType {
    Commute,
    Shopping,
    Leisure,
    Delivery,
    Service,
}
impl CarTripType {
    pub fn pick_car_trip_type(
        zoning_type: Option<ZoningType>,
        phase: SchedulePhase,
        rng: &mut impl Rng,
    ) -> CarTripType {
        match phase {
            SchedulePhase::CommuteToWork | SchedulePhase::CommuteHome => CarTripType::Commute,

            SchedulePhase::Work => match zoning_type {
                None => CarTripType::Service,
                Some(ZoningType::Industrial) => CarTripType::Delivery,
                Some(ZoningType::Commercial) | Some(ZoningType::Office) => CarTripType::Service,
                Some(ZoningType::Residential) => CarTripType::Shopping,
            },

            SchedulePhase::Lunch => match zoning_type {
                Some(ZoningType::Industrial) => CarTripType::Delivery,
                Some(ZoningType::Commercial) | Some(ZoningType::Office) => {
                    if rng.random_bool(0.7) {
                        CarTripType::Shopping
                    } else {
                        CarTripType::Service
                    }
                }
                Some(ZoningType::Residential) => CarTripType::Shopping,
                None => CarTripType::Leisure,
            },

            SchedulePhase::Evening => match zoning_type {
                Some(ZoningType::Industrial) => CarTripType::Delivery,
                Some(ZoningType::Commercial) | Some(ZoningType::Office) => CarTripType::Service,
                Some(ZoningType::Residential) => {
                    if rng.random_bool(0.65) {
                        CarTripType::Shopping
                    } else {
                        CarTripType::Leisure
                    }
                }
                None => CarTripType::Leisure,
            },

            SchedulePhase::Night => match zoning_type {
                Some(ZoningType::Industrial) | Some(ZoningType::Commercial) => {
                    CarTripType::Delivery
                }
                Some(ZoningType::Office) => CarTripType::Service,
                Some(ZoningType::Residential) | None => CarTripType::Leisure,
            },
        }
    }
}
