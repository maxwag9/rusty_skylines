// signfinding.rs
// NOT pathfinding!!
// Cars approach and intersection, look at the turning options and decide where to drive.
// The cars ask at each intersection: "Which turn would lead me faster to my destination?"
//
// Let's trace this through:
// It's the beginning of the simulation. All intersections are reset and have no breadcrumbs.
// Cars begin to drive, hitting intersections, seeing no breadcrumbs, decide to drive whichever turn faces more to its destination.
// But some cars experiment and take completely random turns, just poking the road network a bit!
// Some cars have reached their desired destination. While driving, they collected info on which intersections they passed and how long the whole trip took.
// These cars give that info to all the turns they passed, for future cars to reference.
// Next time cars want to go to roughly the same destination, they will see that on the left turn on this intersection, a car said that the trip took 5 minutes, so he might take it, or explore other turns.
// Let's say it explores. It ignores the breadcrumbs at this intersection and just, drives... using the right turn.
// Turns out the right turn would make the trip 3 minuts and 40 seconds long!
// The next car will see this faster travel time and MOST cars will take that route, some will explore.
//
// I see an issue with this approach though! Because I have the Spidey Senses, they are tingling! (I haven't watched Spider-Man ever, btw)
// The issue is: The destination of each car is too unique for breadcrumbs!
// If each car gave their exact address as the breadcrumb, then exploration would take a hellish long time and storing every address in every breadcrumb eventually is very wasteful.
//
// I thought of this before, so I added the Partitions! The road network is automatically split into partitions, containing multiple nodes.
// That's what an address is: The walk down the partition tree and eventually the building ID that is the final destination.
// I have some bad news though!
// The partitions are NOT stable... That is SHIT for breadcrumbs!
// Each time you build or remove roads, the ENTIRE partition tree gets replaced! Additionally, if you connect two separate road networks, they both get replaced aswell, ofc!
// I am thinking of a solution... My first thought is to make it not be a tree, but a collection of road nodes.
// The address will then be: DistrictID (kinda stable, depends on the districts, automatic splitting and merging can happen, remap tables might make it work though,
// remapping all breadcrumb addresses. Or NO! Assume each destination is ONLY a building, always, which includes parks and services and everything like that,
// then I can just fetch the new address and only store the building ID and the partition!)
// After DistrictID, it will be PartitionID and then Building ID.
// Signfinding will check if their own destination Building ID matches some breadcrumb's one at the intersection, if it doesn't exist, then it will check the PartitionID the same way.
// If even the PartitionID isn't included in the breadcrumbs, then it will check the DistrictID and drive wherever the DistrictID is in a breadcrumb.
// IF EVEN the DistrictID isn't included, then it will just use its compass and drive in te general direct, following roads, exploring as it goes, and give back the breadcrumbs to the intersections, for others to enjoy! (Bread is the only food cars can eat btw)
//
//
// NOTICE how I didn't mention 🤢 'pathfinding', 'Dijkstra' or 'A*' at all? (I watched Veritasiums' video on Dijkstra, and I must say that Dijkstra guy is fucking awesome and the method is great. BUT not for my citybuilder! I can't do preprocessing every time the Road network gets changed!!)
// That's intentional. This way, hopefully I will avoid the disgusting performance problems of pathfinding, like in Cities Skylines.
// And also get smarter, more realistic traffic!
// ALSO even more goodies and possibilities, like making a car NOT drive head first into a traffic jam if he has the turn options! That would be a path recalculation in pathfinding (Expensive)!
// Also, it makes sense for cars to learn the road network instead of instantly knowing it. The realism here is that some person irl would not know his way around a brand-new city!
// Also, if irl an intersection gets reworked or 1342 houses get bulldozed and a big interstate is built, people do need time to adapt! And I have *near* full control over how long they need to adapt.
// I haven't seen a citybuilder use a 'Signfinding' kind of technique before...
// ALSO: This technique is ESSENTIAL to scale for an infinite world!
// Cities Skylines has a limited map size, so A* can work its globally omniscient wonders reasonably well, with lots of optimizations (respect, Colossal Order!)
//
// TBH: From the beginning, I didn't picture Pathfinding in my head at all when thinking about how cars are supposed to find their way.
// It hasn't crossed my mind to search NEARLY EVERY piece of the road network for the fastest path. That hurts realism and *performance*!
//
// YES I know! I now realize that my signfinding technique is still classified as 'Pathfinding'! But ok then I will just say that my methos is definitely NOT *classical* pathfinding!
//
// So, let me write this file!

// Thinking:
// Car is driving on a segment on a lane poly_idx = 12, lane end is poly_idx = 15. I calculate how long it would take to just drive to the end and use that time to
// wait no this is dumb.
// I calculate how long it takes ok, then if it's under the 2-second tick time, I just run pathfinding on the end part intersection.
// Then I create a SignFindingSection with a lane reference and no arm_identification, because no arm was crossed.
// That first section will hold the section between where the car left off from the last tick, to the end of the lane
// (I confirmed it will definitely cross an intersection within 2 secs, but also... I need to check cars in front!)
// The Section will hold points, with  FUCK THIS IS COMPLICATED!!
// It seems I have to do this in two passes: 1. Signfinding trajectory 2. Physics 3. Signfinding using physics knowledge 4. Physics on the new trajectory. Ok seriously, why is this so complicated?!
// Ok WHY would the physics pose a problem to the trajectory anyway?
// I create the signfinding trajectory with no regards for cars in the way. And no regards to physics.
// Ok so just do all of it in one function instead of spread out over 3 iterators!
// Now the last problem is that with the unified function is that car update order matters!
// So I guess I have to update the front most car first? Front to back?
// Ok but then what about a chunk boundary? I am updating cars only in a given chunk,
// so cars in the front that are behind cars that are past the chunk boundary that isn't in this update tick might not be able to drive!
// Ok so instead of ticking per chunk, tick per node! You will tick an intersection, all cars in the intersection and on its connected segments will be ticked together, even past a chunk boundary!
// Ok for fuck's sake the problem just keeps moving! Now the problem is that cars that are on the other side of the segment are crossing into another intersection, which is the same issue as a chunk boundary!
// Ok so simulate the next intersection too! Are you dumb? Recursive for every intersection after it? Hell nah!
// Then wtf do I do? Scrap the chunking concept completely?
// My idea might be to just let the cars pick which car to simulate next.
// So basically I keep the chunked car ticking, but do this: I tick a car and calculate its trajectory. But in the way of the 'perfect' car-free trajectory is a nasty car!
// So I tick that nasty car on the spot and do a perfect trajectory for it too. If another nasty car is in front, then do that one aswell and recursive.
// Now, this could run in a grid with no solution, so I need to keep track of cars I "did" (but actually didn't, they have just been "ticked" after the initial car).
// And if I hit a ticked car, then don't tick it again.
// Ok so now it has to back propagate. Let's say the cars were perfectly gridlocked in a circle.
// The initial car can't move because of the front car and the car behind the initial car can't move because of the initial car.
// So basically we calculated the perfect trajectory for each car but got nowhere. WTF AM I SUPPOSED TO DO?!
// Ok I think I fucking got it: I was thinking in "trajectory must be exact physical path".
// But that is completely wrong! Signfind shouldn't give a single flying shit about other cars or acceleration physics or time!
// This solution completely fixes every problem I dread.
// Ok so basically, we run ONLY the interpolation with no long tickrates. Fuck the medium and far distance cars for now, they will get separate treatment.
// During interpolation, we make new physics steps on demand and Pre-calculated Signfinding steps are used.
// (just perfect trajectory with no cars, a car will try to follow it as best it can and physics will throttle it because of cars in front or acceleration).
// If we used up the Pre-calculated Signfinding steps, then we just calculate new signfinding steps ON THE SPOT for this car, and it will follow it now.
// This way, we have a perfect path the car wants to take and then give it to physics who will calculate the path as it drives during interpolation.
// Literally inside interpolation, but it's fine because most of the time only interpolation happens with no signfinding and physics.
// Only problem I see is Pre computed physics steps for the interpolation. to-do I guess
//

use crate::data::Settings;
use crate::helpers::positions::{LocalPos, WorldPos};
use crate::resources::Time;
use crate::ui::variables::Variables;
use crate::world::buildings::buildings::Buildings;
use crate::world::buildings::zoning::ZoningStorage;
use crate::world::cars::car_player::sanitize_quat;
use crate::world::cars::car_render::CarChange;
use crate::world::cars::car_simulation::{CarTrajectory, CarTrajectoryPoint};
use crate::world::cars::car_structs::{Car, CarId, CarStorage, SimTime};
use crate::world::roads::road_structs::{LaneId, NodeId, NodeLaneId, SegmentId};
use crate::world::roads::roads::{Arm, LaneRef, Node, RoadStorage};
use glam::{Quat, Vec3, bool};
use rand::distr::weighted::WeightedIndex;
use rand::rngs::ThreadRng;
use rand_distr::Distribution;

#[derive(Debug, Clone, Copy)]
pub enum LaneRefSimple {
    Lane(LaneId),
    NodeLane(NodeId, NodeLaneId),
}
#[derive(Debug, Clone, PartialEq)]
pub struct SFTurnIdentification {
    pub turn_type: SFTurnType,
    pub is_final_turn_to_building: bool,
}
#[derive(Debug, Clone, PartialEq)]
pub enum SFTurnType {
    SegmentLanes {
        segment_id: SegmentId,
        possible_lanes: Vec<LaneId>,
    },
    IntersectionLanes {
        node_id: NodeId,
        possible_paths: Vec<IntersectionPath>,
        to_segment_id: SegmentId,
    },
}
#[derive(Debug, Clone, PartialEq)]
pub struct IntersectionPath(pub(crate) Vec<NodeLaneId>);

#[derive(Debug, Clone)]
pub struct BreadCrumb {
    node_id: NodeId,
    segment_id: SegmentId,
    duration: f32,
}
pub struct SignFindingTrip {
    pub start_time: SimTime,
    pub sections: Vec<BreadCrumb>,
}

/// Returns (perfect_signfinding_trajectory, physical_trajectory)
/// physical is the one that the interpolator will blindly follow. This function should make sure that no cars collide and stops are being done.
pub fn make_new_trajectory(
    time: &Time,
    car: &Car,
    car_storage: &CarStorage,
    road_storage: &RoadStorage,
    buildings: &Buildings,
    zoning: &ZoningStorage,
    sf_options: &SignFindingOptions,
) -> Result<
    (
        Option<CarSignfindingTrajectory>,
        CarTrajectory,
        Option<RoadPathCallback>,
    ),
    MakeNewTrajectoryError,
> {
    let origin = WorldPos::new(car.pos.chunk, LocalPos::zero());
    let now = time.sim_time();

    let mut owned_sf_traj: Option<CarSignfindingTrajectory> = None;
    let sf_traj: &CarSignfindingTrajectory = if let Some(t) = &car.signfinding_trajectory {
        t
    } else {
        let rng = &mut ThreadRng::default();
        let t = make_new_signfinding_traj(car, road_storage, buildings, zoning, rng, sf_options)
            .map_err(MakeNewTrajectoryError::Signfinding)?;
        owned_sf_traj = Some(t);
        owned_sf_traj.as_ref().unwrap()
    };

    let path = build_road_path(car, sf_traj, road_storage, buildings, zoning)
        .map_err(MakeNewTrajectoryError::RoadPath)?;

    // if path.pts.len() < 2 {
    //     return Err(MakeNewTrajectoryError::RoadPath(
    //         RoadPathBuildError::DegeneratePath {
    //             reason: "road path had fewer than 2 points",
    //         },
    //     ));
    // }

    let front = snapshot_front_car(car, car_storage, road_storage);

    let traj = simulate_path_follow(car, &path, front.as_ref(), origin, now, sf_options);

    Ok((owned_sf_traj, traj, path.callback))
}

pub fn make_new_signfinding_traj(
    car: &Car,
    road_storage: &RoadStorage,
    buildings: &Buildings,
    zoning: &ZoningStorage,
    rng: &mut ThreadRng,
    sf_options: &SignFindingOptions,
) -> Result<CarSignfindingTrajectory, SignfindingError> {
    use SignfindingError::*;
    const MAX_TURNS: usize = 6; // Max turns per sf trajectory
    let mut turns: Vec<SFTurnIdentification> = Vec::with_capacity(MAX_TURNS);
    let Some(address) = &car.destination_addr else {
        return Err(NoAddress("Car destination Address is None".to_string()));
    };
    let (building_pos, building_segment_id) =
        if let Some(building) = buildings.storage.get(address.destination.as_building()) {
            if let Some(lot) = zoning.get_lot(building.lot_id) {
                (building.pos, lot.segment_id)
            } else {
                return Err(LotDoesntExist(
                    "Lot of the destination's building doesn't exist".to_string(),
                ));
            }
        } else {
            return Err(BuildingDoesntExist(
                "Building defined in the destination address doesn't exist".to_string(),
            ));
        };
    let mut safety_count = 0;
    while turns.len() < MAX_TURNS {
        safety_count += 1;
        let last_turn = match turns.last() {
            None => &get_last_turn(&turns, car, road_storage, building_pos, building_segment_id)
                .map_err(|e| GetLastTurnError(e))?,
            Some(turn) => turn,
        };
        let mut is_final_turn = false;
        let next_turn_type = match &last_turn.turn_type {
            SFTurnType::SegmentLanes {
                segment_id: previous_segment_id,
                possible_lanes,
            } => {
                // If I just came from a segment, I need to calculate which arm to turn in the next intersection ofc. So Give back an IntersectionLanes()

                let next_node_id = if let Some(lane) = possible_lanes
                    .first()
                    .and_then(|&lane_id| road_storage.lane_safe(lane_id))
                {
                    lane.to_node()
                } else {
                    // Both end points are NOT from the last turn! Segment is unreliable, get new segment from car pos.
                    let lane_ref = if let Some(lane) = car.current_lane {
                        lane
                    } else {
                        match road_storage.lane_ref_closest_to_pos(car.pos) {
                            None => { return Err(NoLaneRefIn3x3Chunks("Tried to get the closest LaneRef for the car in 3x3 chunks, but no active Nodes or Segments exist in this area.".to_string())) }
                            Some(lane_ref) => lane_ref
                        }
                    };
                    match lane_ref {
                        LaneRef::Lane(lane_id, _) => {
                            let Some(lane) = road_storage.lanes.get(lane_id.index()) else {
                                return Err(LaneDoesntExist("Last turn was SegmentLanes, trying to get next NodeId, last turn had NO possible lanes, which is concerning, OR the first lane in possible lanes doesn't exist in the road network. \
                                Then tried getting the current lane of the car instead, matched the laneref and got LaneRef::Lane, where this error occurred, because the lane from the lane_id doesn't exist.".to_string()));
                            };
                            let segment_id = lane.segment();

                            lane.to_node()
                        }
                        LaneRef::NodeLane(node_id, _nl_id, _) => {
                            // We're in a NodeLane (inside an intersection)
                            // Find which outgoing segment leads closest to destination
                            let Some(node) = road_storage.node(node_id) else {
                                return Err(NodeDoesntExist("Last turn was SegmentLanes, trying to get next NodeId, last turn had NO possible lanes, which is concerning, OR the first lane in possible lanes doesn't exist in the road network. \
                                Then tried getting the current lane of the car instead, matched the laneref and got LaneRef::NodeLane, where this error occurred, because the node from the node_id doesn't exist.".to_string()));
                            };

                            node_id
                        }
                    }
                }; // NEXT node
                // Signfinding: Get the next intersection to turn to.

                let Some(node) = road_storage.node(next_node_id) else {
                    return Err(NodeDoesntExist(format!(
                        "Last turn was SegmentLanes, got the next node id {:?} and tried getting it from road storage, but got None. ",
                        next_node_id
                    )));
                }; // NEXT node
                let mut best_arms: Vec<(&Arm, f32)> =
                    node.ranked_arms_for_address(&buildings, zoning, address); // TODO: allow U-turns and Roundabouts
                // if best_arms.len() > 1 {
                //     // If the node has one more arm that is not the previous segment, then I can filter out the previous segment to avoid pathfinding back and forth. Stupid idea, I should just rank better based on the compass in the initial phase like I wanted to!!
                //     best_arms.retain(|(arm, _)| arm.segment() != *previous_segment_id);
                // }
                if let Some(same_arm_idx) = best_arms
                    .iter()
                    .position(|&a| a.0.segment() == *previous_segment_id)
                {
                    best_arms.get_mut(same_arm_idx).map(|arm| arm.1 *= 0.2); // Penalize the score for a U turn
                };
                if best_arms.is_empty() {
                    return Err(NoAvailableArms(format!(
                        "Last turn: SegmentLanes. Ranked all arms of the next node {:?}, also excluding previous segment {:?}, but got no arms.",
                        next_node_id, previous_segment_id
                    )));
                }

                let scores: Vec<f32> = best_arms.iter().map(|(_, s)| *s).collect();

                let dist = make_weighted_index(&scores, sf_options)?;

                let idx = dist.sample(rng);
                let picked_arm = best_arms[idx].0; // NEXT arm

                // Stop if destination reached!!
                let next_segment_id = picked_arm.segment();

                let possible_paths = match find_node_lane_path_any_exit(
                    car.id,
                    node,
                    *previous_segment_id,
                    next_segment_id,
                ) {
                    Ok(p) => p,
                    Err(e) => return Err(IntersectionPathError(e)),
                };

                SFTurnType::IntersectionLanes {
                    node_id: next_node_id,
                    possible_paths,
                    to_segment_id: next_segment_id,
                }
            }
            SFTurnType::IntersectionLanes {
                node_id,
                possible_paths,
                to_segment_id,
            } => {
                // If I just came out of an intersection, I don't need a new arm turn, I only need to give the lanes in front of me, SegmentLanes().
                let possible_lanes = if let Some(node) = road_storage.node(*node_id) {
                    if let Some(arm) = node.arm_for_segment(*to_segment_id) {
                        arm.outgoing_lanes().to_vec()
                    } else {
                        return Err(ArmDoesntExist("Last turn: IntersectionLanes. arm_for_segment() returned None, no arm exists for segment for that node.".to_string()));
                    }
                } else {
                    return Err(NodeDoesntExist(
                        "Last turn: IntersectionLanes. Getting the last node returned None."
                            .to_string(),
                    ));
                };
                is_final_turn = *to_segment_id == building_segment_id; // Fine because this function is only for intersections, not direct-to-building navigation.
                // I ensured that all intersections have been passed and the final turn leads the car to the segment where the building is located on.
                // The finer movement will be done by the physical trajectory outside this function.
                SFTurnType::SegmentLanes {
                    segment_id: *to_segment_id,
                    possible_lanes,
                }
            }
        };

        // NEXT turn use next values bruh
        turns.push(SFTurnIdentification {
            turn_type: next_turn_type,
            is_final_turn_to_building: is_final_turn,
        }); // NEXT!!!
        if is_final_turn {
            break;
        }; // Stop at final turn
        if safety_count > MAX_TURNS * 2 {
            // To prevent freezing the game
            break;
            //return Err()
        }
    }

    Ok(CarSignfindingTrajectory {
        car_id: car.id,
        turns,
        valid: true,
    })
}
// Something in my dream today 09.Juni.2026 where my mom went into my room which didn't look like my room and told me to move the DAS Audio Altea-718A away to make space idk what for! So we both carried it out of my room i don't remember where... I don't remember most of my dream! I don't even have this speaker! (But I want one)
pub fn get_last_turn(
    turns: &Vec<SFTurnIdentification>,
    car: &Car,
    road_storage: &RoadStorage,
    building_pos: WorldPos,
    building_segment_id: SegmentId,
) -> Result<SFTurnIdentification, GetLastTurnError> {
    if let Some(lt) = car.last_turn.clone() {
        Ok(lt)
    } else {
        // Bootstrap: create initial turn from current position
        let lane_ref = if let Some(lane) = car.current_lane {
            lane
        } else {
            match road_storage.lane_ref_closest_to_pos(car.pos) {
                None => return Err(GetLastTurnError::NoLaneRefIn3x3ChunksFound),
                Some(lane_ref) => lane_ref,
            }
        };

        match lane_ref {
            LaneRef::Lane(lane_id, _) => {
                //if let Some(last_lane_id) = car.last_lane_id {lane_id=last_lane_id};
                let Some(lane) = road_storage.lanes.get(lane_id.index()) else {
                    return Err(GetLastTurnError::LaneDoesntExist);
                };
                let segment_id = lane.segment();

                Ok(SFTurnIdentification {
                    turn_type: SFTurnType::SegmentLanes {
                        segment_id,
                        possible_lanes: vec![lane_id],
                    },
                    is_final_turn_to_building: segment_id == building_segment_id,
                })
            }
            LaneRef::NodeLane(node_id, _nl_id, _) => {
                // We're in a NodeLane (inside an intersection)
                // Find which outgoing segment leads closest to destination
                let Some(node) = road_storage.node(node_id) else {
                    return Err(GetLastTurnError::NodeDoesntExist);
                };

                let mut best_segment_id = None;
                let mut best_dist_sq = f32::INFINITY;
                //println!("Arms count: {}", node.arms().len());
                for arm in node.arms() {
                    let seg_id = arm.segment();
                    let Some(segment) = road_storage.segments.get(seg_id.index()) else {
                        continue;
                    };

                    // Find the far endpoint of this segment (opposite from current node)
                    let Some(far_node_id) = segment.other_node(node_id) else {
                        continue;
                    };

                    let Some(far_node) = road_storage.node(far_node_id) else {
                        continue;
                    };
                    let dist_sq = far_node.pos().delta_to(building_pos).length_squared();

                    if dist_sq < best_dist_sq {
                        best_dist_sq = dist_sq;
                        best_segment_id = Some(seg_id);
                    }
                }

                let Some(segment_id) = best_segment_id else {
                    return Err(GetLastTurnError::NoAvailableArms);
                };
                let Some(arm) = node.arm_for_segment(segment_id) else {
                    return Err(GetLastTurnError::ArmDoesntExist);
                };
                let possible_lanes = arm.outgoing_lanes().to_vec();
                //println!("Lanes count: {}", possible_lanes.len());
                Ok(SFTurnIdentification {
                    turn_type: SFTurnType::SegmentLanes {
                        segment_id,
                        possible_lanes,
                    },
                    is_final_turn_to_building: segment_id == building_segment_id,
                })
            }
        }
    }
}
#[derive(Debug, Clone)]
pub struct CarSignfindingTrajectory {
    pub car_id: CarId,
    pub turns: Vec<SFTurnIdentification>,
    pub valid: bool,
}
// #[derive(Debug, Clone)]
// pub struct SignfindingTurn {
//     pub turn_identification: TurnIdentification
// }

#[derive(Debug)]
pub enum SignfindingError {
    NoAddress(String),

    NoLastTurn(String),
    SegmentDoesntExist(String),
    NodeDoesntExist(String),
    NoAvailableArms(String),
    InvalidArmWeights(String),
    BuildingDoesntExist(String),
    LotDoesntExist(String),
    SegmentEndPointsDontMatchLastTurn(String),
    LaneDoesntExist(String),
    ArmDoesntExist(String),
    NoLaneRefIn3x3Chunks(String),
    GetLastTurnError(GetLastTurnError),
    IntersectionPathError(IntersectionPathError),
}
impl From<IntersectionPathError> for SignfindingError {
    fn from(err: IntersectionPathError) -> SignfindingError {
        SignfindingError::IntersectionPathError(err)
    }
}
pub struct SignFindingOptions {
    /// lower = more optimal arms chosen, higher -> more random arms/exploration. This value is quite exponential, beware.
    pub exploration_temperature: f32,
    /// How long a single physical trajectory is in seconds. This controls how quickly the cars react and heavily impacts performance. Higher is better performance.
    pub physical_traj_duration: f32,
    /// Cars don't use physics to follow the road path, just snapping to points.
    pub simple_following: bool,
}
impl SignFindingOptions {
    pub fn new(variables: &Variables, settings: &Settings) -> SignFindingOptions {
        let exploration_temperature = (variables
            .get_f64("sf.exploration_temperature")
            .unwrap_or(0.4) as f32)
            .clamp(0.0001, 10000.0); // Enough range;
        let physical_traj_duration = (variables
            .get_f64("sf.physical_traj_duration")
            .unwrap_or(1.0) as f32)
            .clamp(0.0001, 10000.0);
        SignFindingOptions {
            exploration_temperature,
            physical_traj_duration,
            simple_following: settings.simple_car_following,
        }
    }
}

fn make_weighted_index(
    scores: &[f32],
    sf_options: &SignFindingOptions,
) -> Result<WeightedIndex<f32>, SignfindingError> {
    if scores.is_empty() {
        return Err(SignfindingError::InvalidArmWeights(
            "In make_weighted_index: No scores passed in. That means no arms.".to_string(),
        ));
    }

    let temperature = if sf_options.exploration_temperature.is_finite() {
        sf_options.exploration_temperature
    } else {
        0.4
    };

    let max_score = scores
        .iter()
        .copied()
        .filter(|s| s.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);

    if !max_score.is_finite() {
        return WeightedIndex::new(vec![1.0_f32; scores.len()]).map_err(|_| {
            SignfindingError::InvalidArmWeights(
                "In make_weighted_index: Max score is not finite.".to_string(),
            )
        });
    }

    let weights: Vec<f32> = scores
        .iter()
        .map(|s| {
            let s = if s.is_finite() {
                *s
            } else {
                max_score - 100.0 * temperature
            };
            let w = ((s - max_score) / temperature).exp();
            if w.is_finite() && w > 0.0 { w } else { 0.0 }
        })
        .collect();

    let sum: f32 = weights.iter().sum();
    let final_weights = if sum > 0.0 {
        weights
    } else {
        vec![1.0_f32; scores.len()]
    };

    WeightedIndex::new(&final_weights).map_err(|e| {
        SignfindingError::InvalidArmWeights(format!(
            "In make_weighted_index: Failed to create WeightedIndex: {:?}.",
            e
        ))
    })
}

/// BFS from `from_lane_ref`, exiting onto any enabled lane that belongs to `out_segment_id`.
/// Returns the node-lane path AND the chosen outgoing LaneId (preferring shorter paths + light randomness).
fn find_node_lane_path_any_exit(
    car_id: CarId,
    node: &Node,
    from_segment_id: SegmentId,
    out_segment_id: SegmentId,
) -> Result<Vec<IntersectionPath>, IntersectionPathError> {
    use std::collections::{HashMap, HashSet, VecDeque};

    let node_lanes = node.node_lanes();

    let incoming_lanes = node
        .arm_for_segment(from_segment_id)
        .ok_or(IntersectionPathError::IncomingLaneNotFound)?
        .incoming_lanes();

    let outgoing_lanes = node
        .arm_for_segment(out_segment_id)
        .ok_or(IntersectionPathError::NoExitArm(out_segment_id))?
        .outgoing_lanes();

    if outgoing_lanes.is_empty() {
        return Err(IntersectionPathError::NoOutgoingLanes(out_segment_id));
    }

    // Find all entry NodeLanes
    let mut entry_ids = Vec::new();

    for nl in node_lanes.iter().filter(|nl| nl.is_enabled()) {
        // info!("Car Id: {}, incoming lanes: {:?}", car_id, incoming_lanes);
        // info!("Car Id: {}, merging lanes: {:?}", car_id, nl.merging());
        if nl.merging().iter().any(|merge_ref| {
            matches!(merge_ref, LaneRef::Lane(lane_id, _) if incoming_lanes.contains(lane_id))
        }) {
            entry_ids.push(nl.id());
        }
    }

    if entry_ids.is_empty() {
        return Err(IntersectionPathError::NoPathPossible {
            from: from_segment_id,
            to: out_segment_id,
            checked: node_lanes.len(),
            merged_count: entry_ids.len() as i32,
        });
    }
    //info!("Car Id: {}, Succeeded in seeding path", car_id);
    // Find all exit NodeLanes
    let mut exit_ids = HashSet::new();

    for nl in node_lanes.iter().filter(|nl| nl.is_enabled()) {
        if nl.splitting().iter().any(|split_ref| {
            matches!(split_ref, LaneRef::Lane(lane_id, _) if outgoing_lanes.contains(lane_id))
        }) {
            exit_ids.insert(nl.id());
        }
    }

    let mut found_paths = Vec::new();

    // BFS from every entry NodeLane
    for entry in entry_ids {
        let mut queue = VecDeque::new();

        let mut came_from: HashMap<NodeLaneId, NodeLaneId> = HashMap::new();
        let mut visited: HashSet<NodeLaneId> = HashSet::new();

        queue.push_back(entry);
        visited.insert(entry);

        while let Some(current) = queue.pop_front() {
            // If current NodeLane is also an exit NodeLane
            if exit_ids.contains(&current) {
                let mut path = Vec::new();

                let mut cur = current;

                loop {
                    path.push(cur);

                    if let Some(&prev) = came_from.get(&cur) {
                        cur = prev;
                    } else {
                        break;
                    }
                }

                path.reverse();

                found_paths.push(IntersectionPath(path));

                // Don't stop. There may be more exits reachable.
            }

            let Some(current_nl) = node_lanes
                .iter()
                .find(|nl| nl.id() == current && nl.is_enabled())
            else {
                continue;
            };

            for split_ref in current_nl.splitting() {
                if let LaneRef::NodeLane(_, next_id, _) = split_ref {
                    if visited.insert(*next_id) {
                        came_from.insert(*next_id, current);
                        queue.push_back(*next_id);
                    }
                }
            }
        }
    }

    Ok(found_paths)
}
#[derive(Debug)]
enum IntersectionPathError {
    NoReachableExit(LaneRef, SegmentId),
    NoPathPossible {
        from: SegmentId,
        to: SegmentId,
        checked: usize,
        merged_count: i32,
    },
    NoExitArm(SegmentId),
    InvalidFromLaneRef(LaneRef),
    IncomingLaneNotFound,
    NoEntranceArm(SegmentId),
    NoOutgoingLanes(SegmentId),
}

fn build_road_path(
    car: &Car,
    sf_traj: &CarSignfindingTrajectory,
    road_storage: &RoadStorage,
    buildings: &Buildings,
    zoning_storage: &ZoningStorage,
) -> Result<RoadPath, RoadPathBuildError> {
    let mut pts: Vec<WorldPos> = Vec::new();
    let mut limits: Vec<f32> = Vec::new();
    let mut is_last_pts: Vec<bool> = Vec::new();

    let Some(address) = &car.destination_addr else {
        return Err(RoadPathBuildError::NoAddress);
    };
    let (building_pos, building_segment_id) =
        if let Some(building) = buildings.storage.get(address.destination.as_building()) {
            if let Some(lot) = zoning_storage.get_lot(building.lot_id) {
                (building.pos, lot.segment_id)
            } else {
                return Err(RoadPathBuildError::LotDoesntExist);
            }
        } else {
            return Err(RoadPathBuildError::BuildingDoesntExist);
        };
    let mut current_turn = match get_last_turn(
        &sf_traj.turns,
        car,
        road_storage,
        building_pos,
        building_segment_id,
    ) {
        Ok(ok) => ok,
        Err(e) => return Err(RoadPathBuildError::GetLastTurnError(e)),
    };
    let mut callback = RoadPathCallback {
        last_turn: None,
        last_lane_id: None,
        is_last_turn: false,
        car_changes: vec![],
    };
    let current_lane = match car.current_lane {
        None => {
            println!("Car.current_lane was None");
            match road_storage.lane_ref_closest_to_pos(car.pos) {
                None => return Err(RoadPathBuildError::RoadNetworkDoesntExist),
                Some(lane_ref) => lane_ref,
            }
        }
        Some(lane_ref) => {
            // This is the lane that the car is currently on. IT COULD HAVE BEEN FINISHED! So check.
            let is_finished = match lane_ref {
                LaneRef::Lane(lane_id, poly_idx) => {
                    let Some(lane) = road_storage.lane_safe(lane_id) else {
                        return Err(RoadPathBuildError::LaneDoesntExist {
                            lane_id,
                            context: "",
                        });
                    };
                    let poly_idx = poly_idx as usize;
                    let is_last_point = poly_idx == lane.geometry().points.len().saturating_sub(1);
                    is_last_point
                }
                LaneRef::NodeLane(node_id, nodelane_id, poly_idx) => {
                    let Some(node) = road_storage.node(node_id) else {
                        return Err(RoadPathBuildError::NodeDoesntExist {
                            node_id,
                            context: "",
                        });
                    };
                    let Some(nodelane) = node.node_lane(nodelane_id) else {
                        return Err(RoadPathBuildError::NodeDoesntExist {
                            node_id,
                            context: "",
                        });
                    };
                    let poly_idx = poly_idx as usize;
                    let is_last_point =
                        poly_idx == nodelane.geometry().points.len().saturating_sub(1);
                    is_last_point
                }
            };
            match is_finished {
                // I need to PICK the next lane here.
                true => {
                    // Get new turn first.
                    let next_turn = get_next_turn(&current_turn, &sf_traj.turns);
                    callback.is_last_turn = next_turn.1;

                    let next_turn = match next_turn.0 {
                        None => return Err(RoadPathBuildError::TurnsAreEmpty), // Sf_traj is empty?!
                        Some(t) => t,
                    };
                    current_turn = next_turn.clone(); // Last turn has 100% finished, so the current turn must be this next turn, cuz that's literally the current turn the car is on!
                    // Get the next lane
                    match lane_ref {
                        LaneRef::Lane(lane_id, poly_idx) => {
                            // It's a lane, so next up is an intersection. Get the next nodelane.
                            let Some(lane) = road_storage.lane_safe(lane_id) else {
                                return Err(RoadPathBuildError::LaneDoesntExist {
                                    lane_id,
                                    context: "",
                                });
                            };
                            match &next_turn.turn_type {
                                SFTurnType::SegmentLanes {
                                    segment_id,
                                    possible_lanes,
                                } => {
                                    // Wtf? A segment follows after a segment? Ok whatever. But how? Let's just follow them and see where they are going!
                                    //return Err(RoadPathBuildError::NoPossibleLanes(format!("In build_road_path(), picking next laneref to follow, last lane was a SegmentLane(lane_id: {:?}), this turn is Segment, error: Segment turn after segment turn!", lane_id)));
                                    // TODO: Improve the lane picking based on factors.
                                    let Some(&lane_id) = possible_lanes.last() else {
                                        return Err(RoadPathBuildError::NoPossibleLanes(format!(
                                            "In build_road_path(), picking next laneref to follow, last lane was a SegmentLane(lane_id: {:?}), this turn is Segment, error: No possible lanes at all!",
                                            lane_id
                                        )));
                                    };
                                    LaneRef::Lane(lane_id, 0)
                                }
                                SFTurnType::IntersectionLanes {
                                    node_id,
                                    possible_paths,
                                    to_segment_id,
                                } => {
                                    // Next turn is an intersection, as expected. This is the intersection START because the last laneref was a segment lane.

                                    // TODO: Improve the nodelanes path picking based on factors.
                                    let Some(&nodelane_id) =
                                        possible_paths.first().and_then(|path| path.0.first())
                                    else {
                                        return Err(RoadPathBuildError::NoPossiblePaths(format!(
                                            "In build_road_path(), picking next laneref to follow, last lane was a SegmentLane(lane_id: {:?}), this turn is SegmentTurn, error: No intersection Paths at all!",
                                            lane_id
                                        )));
                                    };
                                    LaneRef::NodeLane(*node_id, nodelane_id, 0)
                                }
                            }
                        }
                        LaneRef::NodeLane(node_id, nodelane_id, poly_idx) => {
                            let Some(node) = road_storage.node(node_id) else {
                                return Err(RoadPathBuildError::NodeDoesntExist {
                                    node_id,
                                    context: "",
                                });
                            };
                            let Some(nodelane) = node.node_lane(nodelane_id) else {
                                return Err(RoadPathBuildError::NodeDoesntExist {
                                    node_id,
                                    context: "",
                                });
                            };

                            let is_nodelane_geom_finished =
                                nodelane.geometry().points.len().saturating_sub(1)
                                    == poly_idx as usize;

                            if is_nodelane_geom_finished {
                                // The car truly reached the end of this nodelane's geometry.
                                // Now we must consolidate paths to pick the next nodelane,
                                // or pick a SegmentLane if exiting the intersection.
                                match &next_turn.turn_type {
                                    SFTurnType::SegmentLanes {
                                        segment_id,
                                        possible_lanes,
                                    } => {
                                        // Next turn is a Segment. This is the Segment START because the last laneref was a nodelane, and it was the LAST nodelane, so it finished the intersection.

                                        // TODO: Improve the lane picking based on factors.
                                        let Some(&lane_id) = possible_lanes.last() else {
                                            return Err(RoadPathBuildError::NoPossibleLanes(
                                                format!(
                                                    "In build_road_path(), picking next laneref to follow, last lane was a NodeLane(node_id: {:?}, nodelane_id: {}), this turn is Segment, error: No possible lanes in segment turn!",
                                                    node_id, nodelane_id
                                                ),
                                            ));
                                        };
                                        LaneRef::Lane(lane_id, 0)
                                    }
                                    SFTurnType::IntersectionLanes {
                                        node_id,
                                        possible_paths,
                                        to_segment_id,
                                    } => {
                                        // The nodelane is completely traversed, find the next one in the intersection path
                                        let prev_nodelane_id = nodelane_id;
                                        // let remaining_paths: Vec<Vec<NodeLaneId>> = possible_paths.iter()
                                        //     .filter_map(|path| {
                                        //         // path.0 is the Vec<NodelaneId>
                                        //         path.0.iter().position(|&nl| nl == prev_nodelane_id).map(|idx| {
                                        //             // take elements after idx (drop including the matched entry)
                                        //             path.0.iter().skip(idx + 1).copied().collect::<Vec<NodeLaneId>>()
                                        //         })
                                        //     })
                                        //     .filter(|v| !v.is_empty()) // discard paths that end at the key (no continuation)
                                        //     .collect();
                                        //
                                        // if remaining_paths.is_empty() {
                                        //     return Err(RoadPathBuildError::NoPossiblePaths(format!("In build_road_path(), picking next laneref to follow, last lane was a NodeLane(node_id: {:?}, nodelane_id: {}), this turn is intersection, error: No remaining Paths after consolidation!", node_id, nodelane_id)));
                                        // }

                                        // TODO: pick a better path; for now choose the first remaining path and its first nodelane
                                        let Some(&next_nodelane_id) =
                                            possible_paths.first().and_then(|path| path.0.first())
                                        else {
                                            return Err(RoadPathBuildError::NoPossiblePaths(
                                                format!(
                                                    "In build_road_path(), picking next laneref to follow, last lane was a NodeLane(node_id: {:?}, nodelane_id: {}), this turn is intersection, error: No remaining Paths after consolidation!",
                                                    node_id, nodelane_id
                                                ),
                                            ));
                                        };
                                        LaneRef::NodeLane(*node_id, next_nodelane_id, 0)
                                    }
                                }
                            } else {
                                // The car hasn't reached the geometric end of the nodelane yet!
                                // It just finished its physical trajectory tick partway through.
                                // Return the exact same nodelane and poly_idx so it continues where it left off.
                                LaneRef::NodeLane(node_id, nodelane_id, poly_idx)
                            }
                        }
                    }
                }
                false => {
                    // Car still driving on previous lane, not finished
                    lane_ref
                }
            }
        }
    };
    // car.lane is the lane that the car is currently on. At the end of a physical trajectory, it will update car.lane with the next lane

    // let next_speed_limit = if let Some(next_turn) = get_next_turn(&current_turn, &sf_traj.turns).0 {
    //     match &next_turn.turn_type {
    //         SFTurnType::SegmentLanes { possible_lanes, .. } => {
    //             possible_lanes.first()
    //                 .and_then(|&id| road_storage.lane_safe(id))
    //                 .map(|l| l.speed_limit())
    //                 .unwrap_or(INTER_SPEED * limits.first().unwrap_or(&14.0))
    //         }
    //         SFTurnType::IntersectionLanes { .. } => {
    //             // Entering an intersection
    //             limits.first().unwrap_or(&14.0) * INTER_SPEED
    //         }
    //     }
    // } else {
    //     0.0 // End of path, come to a full stop
    // };
    let end_speed: f32;
    match current_lane {
        LaneRef::Lane(lane_id, poly_idx) => {
            let Some(lane) = road_storage.lane_safe(lane_id) else {
                return Err(RoadPathBuildError::LaneDoesntExist {
                    lane_id,
                    context: "",
                });
            };
            let poly_idx = poly_idx as usize;
            for (i, &p) in lane.geometry().points.iter().enumerate() {
                if i >= poly_idx {
                    let is_last_point = poly_idx == lane.geometry().points.len().saturating_sub(1);
                    push_wp(
                        &p,
                        lane.speed_limit(),
                        &mut pts,
                        &mut limits,
                        &mut is_last_pts,
                        is_last_point,
                    );
                }
            }
            callback.last_turn = Some(current_turn);
            // At path end the car is entering an intersection → slow to intersection speed
            end_speed = lane.speed_limit() * INTER_SPEED;
        }
        LaneRef::NodeLane(node_id, nodelane_id, poly_idx) => {
            let Some(node) = road_storage.node(node_id) else {
                return Err(RoadPathBuildError::NodeDoesntExist {
                    node_id,
                    context: "",
                });
            };
            let Some(nodelane) = node.node_lane(nodelane_id) else {
                return Err(RoadPathBuildError::NodeDoesntExist {
                    node_id,
                    context: "",
                });
            };
            // TO/DO: When editing an intersection, nodelanes could exist but be totally wrong. 2/10 priority, cuz signfinding runs roughly every 30 seconds anyway.
            let poly_idx = poly_idx as usize;
            let inter_limit = nodelane.speed_limit() * INTER_SPEED; // ← reduced speed for whole nodelane
            for (i, &p) in nodelane.geometry().points.iter().enumerate() {
                if i >= poly_idx {
                    let is_last_point =
                        poly_idx == nodelane.geometry().points.len().saturating_sub(1);
                    push_wp(
                        &p,
                        nodelane.speed_limit(),
                        &mut pts,
                        &mut limits,
                        &mut is_last_pts,
                        is_last_point,
                    );
                }
            }
            callback.last_turn = Some(current_turn);
            // Exit intersection at intersection speed; next segment path handles re-acceleration
            end_speed = inter_limit;
        }
    }
    // Alright so. This gives all points of a whole lane or one whole nodelane. In those points is a boolean marker to either take the last turn of the callback and put it into the car field or not.

    // Determine the speed limit of the NEXT lane (for intersection approach speed)

    // Calculate the arc-length where the current lane ends
    // (The length of the lane geometry minus the starting poly_idx)
    let lane_end_s = match current_lane {
        LaneRef::Lane(lane_id, poly_idx) => road_storage
            .lane_safe(lane_id)
            .map(|l| l.geometry().points.len().saturating_sub(1) - poly_idx as usize)
            .unwrap_or(0) as f64,
        LaneRef::NodeLane(node_id, nodelane_id, poly_idx) => road_storage
            .node(node_id)
            .and_then(|n| n.node_lane(nodelane_id))
            .map(|nl| nl.geometry().points.len().saturating_sub(1) - poly_idx as usize)
            .unwrap_or(0) as f64,
    };

    Ok(RoadPath::build(
        pts,
        limits,
        Some(current_lane),
        Some(callback),
        end_speed,
        lane_end_s as f32,
    ))
}

fn get_next_turn<'a>(
    prev_turn: &SFTurnIdentification,
    turns: &'a Vec<SFTurnIdentification>,
) -> (Option<&'a SFTurnIdentification>, bool) {
    for (idx, turn) in turns.iter().enumerate() {
        if turn == prev_turn {
            let next_turn = match turns.get(idx + 1) {
                None => return (None, true), // Sf_traj refreshes when empty
                Some(t) => t,
            };
            let is_last_turn = turns.get(idx + 2).is_none();
            return (Some(next_turn), is_last_turn);
        }
    }
    (turns.get(0), false)
}

pub struct RoadPathCallback {
    pub last_turn: Option<SFTurnIdentification>,
    pub last_lane_id: Option<LaneId>,
    pub is_last_turn: bool,
    pub car_changes: Vec<CarChange>,
}

/// Push a waypoint with its associated lane, silently dropping near-duplicates.
#[inline]
fn push_wp(
    p: &WorldPos,
    limit: f32,
    pts: &mut Vec<WorldPos>,
    limits: &mut Vec<f32>,
    is_last_pts: &mut Vec<bool>,
    is_last_point: bool,
) {
    if let Some(last) = pts.last() {
        if last.delta_to(*p).length() < DEDUP_M {
            return;
        }
    }
    pts.push(*p);
    limits.push(limit);
    is_last_pts.push(is_last_point);
}

/// Check if two LaneRefs refer to the same lane (ignoring poly idx)
fn lanes_match(a: &LaneRef, b: &LaneRef) -> bool {
    match (a, b) {
        (LaneRef::Lane(id_a, _), LaneRef::Lane(id_b, _)) => id_a == id_b,
        (LaneRef::NodeLane(node_a, id_a, _), LaneRef::NodeLane(node_b, id_b, _)) => {
            node_a == node_b && id_a == id_b
        }
        _ => false,
    }
}
fn snapshot_front_car(car: &Car, cs: &CarStorage, rs: &RoadStorage) -> Option<FrontCarInfo> {
    let fid = cs.cars_ahead_on_segment(car.id, rs).first().copied()?;
    let fc = cs.get(fid)?;
    // Bumper-to-bumper gap
    let gap = (car.pos.delta_to(fc.pos).length() - car.length * 0.5 - fc.length * 0.5).max(0.5);
    Some(FrontCarInfo {
        gap_m: gap,
        speed_mps: fc.current_velocity.length(),
    })
}

fn simulate_path_follow(
    car: &Car,
    path: &RoadPath,
    front: Option<&FrontCarInfo>,
    origin: WorldPos,
    start_t: SimTime,
    sf_options: &SignFindingOptions,
) -> CarTrajectory {
    let duration = sf_options.physical_traj_duration;
    let n_steps = (duration / SIM_DT).ceil() as u32;
    let mut pts = Vec::with_capacity((n_steps / SIM_SAMPLE + 2) as usize);

    let mut pos = car.pos;
    let mut quat = sanitize_quat(car.quat);
    let mut speed = car.current_velocity.length();
    let mut r = car.yaw_rate;
    let mut arc_s = 0.0f32;
    let mut t = start_t;
    let path_total_len = path.total_len();

    pts.push(CarTrajectoryPoint {
        time: t,
        pos: origin.delta_to(pos),
        quat,
        velocity: (quat * Vec3::Z) * speed, // Z is FORWARD, not Y!
        lane_ref: car.current_lane,
    });
    if sf_options.simple_following {
        for step in 1..=n_steps {
            t += SIM_DT as f64;

            // Advance along path at speed limit (or swap for a fixed number to test)
            speed = path.limit_at(arc_s);
            arc_s = (arc_s + (speed * SIM_DT)).min(path_total_len);

            // Snap position + heading directly to path — no steering, no physics
            let (new_pos, fwd) = path.sample(arc_s);
            pos = new_pos;
            if fwd.length_squared() > 1e-6 {
                quat = sanitize_quat(Quat::from_rotation_arc(Vec3::Z, fwd));
            }
            r = 0.0;
            let reached_end = arc_s >= path_total_len;
            if step % SIM_SAMPLE == 0 {
                let lane_ref = path.lane_ref_at(arc_s);
                pts.push(CarTrajectoryPoint {
                    time: t,
                    pos: origin.delta_to(pos),
                    quat,
                    velocity: (quat * Vec3::Z) * speed,
                    lane_ref,
                });
            }
            if reached_end {
                break;
            } // ← trajectory ends here, might be before the budget seconds, that's good!
        }
    } else {
        for step in 1..=n_steps {
            t += SIM_DT as f64;

            let v0 = if arc_s >= path_total_len {
                path.end_speed // maintain end_speed rather than forcing 0, avoids unnecessary stop
            } else {
                let v_limit = path.limit_at(arc_s);
                // Kinematic cap: maximum speed from which we can brake to `end_speed`
                // within the remaining distance using comfort deceleration.
                // Derived from: v_target² = v² - 2·a·d  →  v_max = √(v_target² + 2·a·d)
                let dist_to_end = (path_total_len - arc_s).max(0.0) as f32;
                let v_kinematic =
                    (path.end_speed * path.end_speed + 2.0 * IDM_BRAKE * dist_to_end).sqrt();
                v_limit.min(v_kinematic)
            };

            let a_free = if v0 < 0.01 {
                -IDM_BRAKE
            } else {
                IDM_ACCEL * (1.0 - (speed / v0).powf(IDM_DELTA))
            };

            let a_follow = front.map_or(0.0, |fc| {
                let s_star = (IDM_S0
                    + speed * IDM_THW
                    + speed * (speed - fc.speed_mps) / (2.0 * (IDM_ACCEL * IDM_BRAKE).sqrt()))
                .max(0.0);
                -IDM_ACCEL * (s_star / fc.gap_m.max(0.1)).powi(2)
            });

            let accel = (a_free + a_follow).clamp(-IDM_BRAKE, IDM_ACCEL);

            speed = (speed + accel * SIM_DT).clamp(0.0, MAX_CAR_SPEED);
            arc_s = (arc_s + speed * SIM_DT).min(path_total_len);
            let look_s = (arc_s + LOOKAHEAD_M).min(path_total_len);
            let (look_pt, _) = path.sample(look_s);
            let to_tgt = pos.delta_to(look_pt);

            let desired_r = if to_tgt.length_squared() > 0.04 && speed > 0.2 {
                let fwd = quat * Vec3::Z; // FORWARD is Z, not Y!
                let tgt = to_tgt.normalize();
                // Use Z for forward, cross.y gives turning direction
                let cross_y = fwd.cross(tgt).y;
                2.0 * speed * cross_y / LOOKAHEAD_M
            } else {
                0.0
            };

            const R_TC: f32 = 0.12;
            const MAX_YAW_RATE: f32 = 3.5;
            r += (desired_r - r) * (SIM_DT / R_TC).min(1.0);
            r = r.clamp(-MAX_YAW_RATE, MAX_YAW_RATE);

            if r.abs() > 1e-5 {
                quat = sanitize_quat(Quat::from_axis_angle(Vec3::Y, r * SIM_DT) * quat);
            }

            let vel = (quat * Vec3::Z) * speed; // Z is FORWARD!
            pos = pos.add_vec3(vel * SIM_DT);
            let reached_end = arc_s >= path_total_len;
            // if reached_end {
            //     // Eliminate pure-pursuit drift: land exactly on the lane endpoint
            //     let (end_pos, end_fwd) = path.sample(path_total_len);
            //     pos = end_pos;
            //     if end_fwd.length_squared() > 1e-6 {
            //         quat = sanitize_quat(Quat::from_rotation_arc(Vec3::Z, end_fwd));
            //     }
            // }
            if step % SIM_SAMPLE == 0 || reached_end {
                let lane_ref = path.lane_ref_at(arc_s);
                pts.push(CarTrajectoryPoint {
                    time: t,
                    pos: origin.delta_to(pos),
                    quat,
                    velocity: if reached_end {
                        (quat * Vec3::Z) * speed
                    } else {
                        vel
                    }, // recompute with snapped quat if reached_end
                    lane_ref,
                });
            }
            if reached_end {
                break;
            } // ← trajectory ends here, might be before the budget seconds, that's good!
        }
    }

    let vel_end = (quat * Vec3::Z) * speed; // Z is FORWARD!
    if pts.last().map_or(true, |p| p.time < t - 1e-9) {
        let lane_ref = path.lane_ref_at(arc_s);
        pts.push(CarTrajectoryPoint {
            time: t,
            pos: origin.delta_to(pos),
            quat,
            velocity: vel_end,
            lane_ref,
        });
    }
    let is_last_turn = path.callback.as_ref().is_some_and(|cb| cb.is_last_turn);
    CarTrajectory {
        car_id: car.id,
        origin,
        points: pts,
        end_quat: quat,
        end_yaw_rate: r,
        end_steering_angle: 0.0,
        end_steering_vel: 0.0,
        is_last_turn_of_sf_traj: is_last_turn,
    }
}

pub fn idle_trajectory(
    car: &Car,
    origin: WorldPos,
    start_t: SimTime,
    duration: f32,
) -> CarTrajectory {
    let rel = origin.delta_to(car.pos);
    let lane_ref = car.current_lane;
    CarTrajectory {
        car_id: car.id,
        origin,
        points: vec![
            CarTrajectoryPoint {
                time: start_t,
                pos: rel,
                quat: car.quat,
                velocity: Vec3::ZERO,
                lane_ref,
            },
            CarTrajectoryPoint {
                time: start_t + duration as f64,
                pos: rel,
                quat: car.quat,
                velocity: Vec3::ZERO,
                lane_ref,
            },
        ],
        end_quat: car.quat,
        end_yaw_rate: 0.0,
        end_steering_angle: 0.0,
        end_steering_vel: 0.0,
        is_last_turn_of_sf_traj: false,
    }
}

const IDM_ACCEL: f32 = 2.5; // comfortable acceleration   m/s²
const IDM_BRAKE: f32 = 4.0; // comfortable deceleration   m/s²
const IDM_S0: f32 = 2.5; // minimum stationary gap     m
const IDM_THW: f32 = 1.5; // time headway               s
const IDM_DELTA: f32 = 4.0; // free-road velocity exponent
const LOOKAHEAD_M: f32 = 8.0; // pure-pursuit look-ahead    m
const SIM_DT: f32 = 1.0 / 30.0; // integration step           s
const SIM_SAMPLE: u32 = 2; // record every N steps
const DEDUP_M: f32 = 0.05; // waypoint dedup threshold   m
const INTER_SPEED: f32 = 0.55; // speed fraction inside intersections

const MAX_CAR_SPEED: f32 = 1000.0;
struct FrontCarInfo {
    gap_m: f32,
    speed_mps: f32,
}

struct RoadPath {
    pts: Vec<WorldPos>,
    limits: Vec<f32>, // speed limit (m/s) per waypoint
    arc: Vec<f32>,    // cumulative arc length (m) per waypoint
    lane_ref: Option<LaneRef>,
    end_speed: f32,
    callback: Option<RoadPathCallback>, // To give back to the car. A physical trajectory can only have one turn.
    lane_end_s: f32,
}

impl RoadPath {
    fn build(
        pts: Vec<WorldPos>,
        limits: Vec<f32>,
        lane_ref: Option<LaneRef>,
        callback: Option<RoadPathCallback>,
        end_speed: f32,
        lane_end_s: f32,
    ) -> Self {
        let mut arc = Vec::with_capacity(pts.len());
        let mut cum = 0.0f32;
        arc.push(0.0);
        for i in 1..pts.len() {
            cum += pts[i - 1].delta_to(pts[i]).length();
            arc.push(cum); // CUM HAHAH
        }
        Self {
            pts,
            limits,
            arc,
            lane_ref,
            end_speed,
            callback,
            lane_end_s,
        }
    }

    fn total_len(&self) -> f32 {
        self.arc.last().copied().unwrap_or(0.0)
    }

    /// World position + forward tangent at arc-length `s`.
    fn sample(&self, s: f32) -> (WorldPos, Vec3) {
        let n = self.pts.len();
        match n {
            0 => return (WorldPos::default(), Vec3::Z), // Z is forward!
            1 => return (self.pts[0], Vec3::Z),
            _ => {}
        }
        let s = s.clamp(0.0, self.total_len());
        let hi = self.arc.partition_point(|&a| a < s).min(n - 1);
        let lo = hi.saturating_sub(1);
        if lo == hi {
            return (self.pts[lo], Vec3::Z);
        }
        let span = self.arc[hi] - self.arc[lo];
        let frac = if span < 1e-6 {
            0.0
        } else {
            (s - self.arc[lo]) / span
        };
        let d = self.pts[lo].delta_to(self.pts[hi]);
        (self.pts[lo].add_vec3(d * frac), d.normalize_or_zero())
    }

    /// Speed limit at arc-length `s` — falls back to ~50 km/h.
    fn limit_at(&self, s: f32) -> f32 {
        if self.limits.is_empty() {
            return 14.0;
        }
        let i = self.arc.partition_point(|&a| a <= s).saturating_sub(1);
        self.limits[i.min(self.limits.len() - 1)]
    }

    /// LaneRef at arc-length `s`.
    /// Because a RoadPath only ever follows a single lane/nodelane, this just
    /// advances the poly_idx as the car moves through the waypoints.
    fn lane_ref_at(&self, s: f32) -> Option<LaneRef> {
        let base = self.lane_ref?;
        let n = self.pts.len();
        if n == 0 {
            return Some(base);
        }
        // Index of the waypoint segment we are currently on (same logic as limit_at).
        let i = self
            .arc
            .partition_point(|&a| a <= s)
            .saturating_sub(1)
            .min(n - 1);
        Some(match base {
            LaneRef::Lane(lane_id, start_poly) => LaneRef::Lane(lane_id, start_poly + i as u32),
            LaneRef::NodeLane(node_id, nodelane_id, start_poly) => {
                LaneRef::NodeLane(node_id, nodelane_id, start_poly + i as u32)
            }
        })
    }
}
#[derive(Debug, Clone)]
pub enum NodeLaneSearchError {
    NoExitArm(SegmentId),
    InvalidFromLaneRef(LaneRef),
    IncomingLaneNotFound(LaneRef),
    NoEntranceArm(SegmentId),
    NoOutgoingLanes(SegmentId),
    NoSeedingPossible {
        from: LaneRef,
        to: SegmentId,
        checked: usize,
        merged_count: usize,
    },
    NoReachableExit(LaneRef, SegmentId),
    InternalLaneLookupFailed(LaneId),
}
impl std::fmt::Display for NodeLaneSearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeLaneSearchError::NoExitArm(sid) => write!(f, "No exit arm for segment {:?}", sid),
            NodeLaneSearchError::InvalidFromLaneRef(lr) => {
                write!(f, "Invalid from LaneRef: {:?}", lr)
            }
            NodeLaneSearchError::IncomingLaneNotFound(lr) => {
                write!(f, "Incoming lane not found: {:?}", lr)
            }
            NodeLaneSearchError::NoEntranceArm(sid) => {
                write!(f, "No entrance arm for segment {:?}", sid)
            }
            NodeLaneSearchError::NoOutgoingLanes(sid) => {
                write!(f, "No outgoing lanes on arm for segment {:?}", sid)
            }
            NodeLaneSearchError::NoSeedingPossible {
                from,
                to,
                checked,
                merged_count,
            } => write!(
                f,
                "Could not seed any NodeLane from lane id {:?} toward segment id {:?}, checked: {:?}, merged_count: {:?}",
                from, to, checked, merged_count
            ),
            NodeLaneSearchError::NoReachableExit(lr, sid) => {
                write!(f, "No reachable exit from {:?} to segment {:?}", lr, sid)
            }
            NodeLaneSearchError::InternalLaneLookupFailed(lid) => {
                write!(f, "Failed to look up lane {:?}", lid)
            }
        }
    }
}
#[derive(Debug)]
pub enum MakeNewTrajectoryError {
    Signfinding(SignfindingError),
    RoadPath(RoadPathBuildError),
}

#[derive(Debug)]
pub enum RoadPathBuildError {
    NoCurrentLane {
        car_id: CarId,
        pos: WorldPos,
    },
    ClosestLaneMissing {
        car_id: CarId,
        pos: WorldPos,
    },
    FirstTurnMissing {
        car_id: CarId,
    },
    SegmentMissing {
        segment_id: SegmentId,
        context: &'static str,
    },
    LaneDoesntExist {
        lane_id: LaneId,
        context: &'static str,
    },
    NodeDoesntExist {
        node_id: NodeId,
        context: &'static str,
    },
    StaleNavigationState {
        turn: SFTurnIdentification,
        current_lane_ref: Option<LaneRef>,
    },
    TurnIndexOutOfBounds {
        turn_idx: usize,
        turns_len: usize,
    },
    AppendTurn(AppendTurnError),
    GetLastTurnError(GetLastTurnError),
    DegeneratePath {
        reason: &'static str,
    },
    NoAddress,
    LotDoesntExist,
    BuildingDoesntExist,
    RoadNetworkDoesntExist,
    TurnsAreEmpty,
    NoPossibleLanes(String),
    NoPossiblePaths(String),
}

#[derive(Debug)]
pub enum AppendTurnError {
    NoCurrentLaneRef,
    NodeMissing {
        node_id: NodeId,
    },
    SegmentMissing {
        segment_id: SegmentId,
    },
    NodeLanePathFailed {
        node_id: NodeId,
        next_segment_id: SegmentId,
        reason: String,
    },
    FallbackLaneMissing {
        node_id: NodeId,
        segment_id: SegmentId,
    },
    OutLaneMissing {
        lane_id: LaneId,
    },
}
#[derive(Debug)]
pub enum GetLastTurnError {
    NoLaneRefIn3x3ChunksFound,
    LaneDoesntExist,
    NodeDoesntExist,
    NoAvailableArms,
    ArmDoesntExist,
}
