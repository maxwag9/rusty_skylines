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

use crate::helpers::positions::{LocalPos, WorldPos};
use crate::resources::Time;
use crate::ui::variables::Variables;
use crate::world::buildings::buildings::Buildings;
use crate::world::buildings::zoning::ZoningStorage;
use crate::world::cars::car_player::sanitize_quat;
use crate::world::cars::car_simulation::{CarTrajectory, CarTrajectoryPoint};
use crate::world::cars::car_structs::{Car, CarId, CarStorage, SimTime};
use crate::world::cars::partitions::Address;
use crate::world::roads::road_structs::{LaneId, NodeId, NodeLaneId, SegmentId};
use crate::world::roads::roads::{Arm, Lane, LaneGeometry, LaneRef, Node, RoadStorage, Segment};
use glam::{Quat, Vec3};
use rand::distr::weighted::WeightedIndex;
use rand::rngs::ThreadRng;
use rand_distr::Distribution;

#[derive(Debug, Clone, Copy)]
pub enum LaneRefSimple {
    Lane(LaneId),
    NodeLane(NodeId, NodeLaneId),
}
#[derive(Debug, Clone, Copy)]
pub struct TurnIdentification {
    pub node_id: NodeId,
    pub segment_id: SegmentId, // to get the arm of the intersection // giving the breadcrumb to a specific Arm
    pub is_final_turn: bool,
}
#[derive(Debug, Clone, Copy)]
pub struct BreadCrumb {
    turn_identification: Option<TurnIdentification>, // No turn means no intersection crossed
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
) -> (Option<CarSignfindingTrajectory>, CarTrajectory) {
    let origin = WorldPos::new(car.pos.chunk, LocalPos::zero());
    let now = time.sim_time();
    let mut owned_sf_traj: Option<CarSignfindingTrajectory> = None;
    let sf_traj: &CarSignfindingTrajectory = if let Some(t) = &car.signfinding_trajectory {
        t
    } else {
        let rng = &mut ThreadRng::default();
        let sf_traj_result =
            make_new_signfinding_traj(car, road_storage, buildings, zoning, rng, sf_options);
        match sf_traj_result {
            Ok(t) => {
                owned_sf_traj = Some(t);
                owned_sf_traj.as_ref().unwrap() // Safe because I literally just set it
            }
            Err(e) => {
                println!("Signfinding error for car id {}: {:?}", car.id, e);
                return (
                    None,
                    idle_trajectory(car, origin, now, sf_options.physical_traj_duration),
                );
            }
        }
    };
    // 2 ── Build road-geometry polyline ───────────────────────────────────
    let path = build_road_path(car, sf_traj, road_storage);
    if path.pts.len() < 2 {
        return (
            owned_sf_traj,
            idle_trajectory(car, origin, now, sf_options.physical_traj_duration),
        );
    }

    // 3 ── Snapshot the car directly ahead ────────────────────────────────
    let front = snapshot_front_car(car, car_storage, road_storage);

    // 4 ── Simulate ───────────────────────────────────────────────────────
    let traj = simulate_path_follow(
        car,
        &path,
        front.as_ref(),
        origin,
        now,
        sf_options.physical_traj_duration,
    );

    (owned_sf_traj, traj)
}

/// Perfect trajectory, ignoring the cars in front or any other cars stopping me!
/// IGNORES traffic lights or yielding!!
/// This function just gives the PERFECT no stops path the car will take in the next few intersections at most!
/// Mostly just one intersection. Wait no, I think most of the time a few!
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
    let mut turns: Vec<TurnIdentification> = Vec::with_capacity(MAX_TURNS);
    let Some(address) = &car.destination_addr else {
        return Err(NoAddress);
    };
    let (building_pos, building_segment_id) =
        if let Some(building) = buildings.storage.get(address.destination.as_building()) {
            if let Some(lot) = zoning.get_lot(building.lot_id) {
                (building.pos, lot.segment_id)
            } else {
                return Err(LotDoesntExist);
            }
        } else {
            return Err(BuildingDoesntExist);
        };
    while turns.len() < MAX_TURNS {
        let last_turn = get_last_turn(
            &turns,
            car,
            buildings,
            address,
            road_storage,
            building_pos,
            building_segment_id,
        )?;
        let Some(segment) = road_storage.segments.get(last_turn.segment_id.index()) else {
            return Err(SegmentDoesntExist);
        }; // LAST segment
        let node_id = if segment.start() == last_turn.node_id {
            segment.end()
        } else if segment.end() == last_turn.node_id {
            segment.start()
        } else {
            // Both end points are NOT from the last turn! Segment is unreliable, get new segment from car pos.
            let lane_ref = if let Some(lane) = car.lane {
                lane
            } else {
                road_storage.lane_ref_closest_to_pos(&car.pos)
            };
            match lane_ref {
                LaneRef::Lane(lane_id, _) => {
                    let Some(lane) = road_storage.lanes.get(lane_id.index()) else {
                        return Err(LaneDoesntExist);
                    };
                    let Some(segment) = road_storage.segments.get(lane.segment().index()) else {
                        return Err(SegmentDoesntExist);
                    };
                    let node_id = if segment.start() == last_turn.node_id {
                        segment.end()
                    } else if segment.end() == last_turn.node_id {
                        segment.start()
                    } else {
                        return Err(SegmentEndPointsDontMatchLastTurn);
                    }; // Something is truly fucked. Handled outside.
                    node_id
                }
                LaneRef::NodeLane(node_id, _, _) => node_id,
            }
        }; // NEXT node
        // Do NOT use segment or last_turn past this point!
        let Some(node) = road_storage.node(node_id) else {
            return Err(NodeDoesntExist);
        }; // NEXT node
        let best_arms: Vec<(&Arm, f32)> =
            node.ranked_arms_for_address(&buildings.partitions.storage, zoning, address);
        if best_arms.is_empty() {
            return Err(NoAvailableArms);
        }

        let scores: Vec<f32> = best_arms.iter().map(|(_, s)| *s).collect();

        let dist = make_weighted_index(&scores, sf_options)?;

        let idx = dist.sample(rng);
        let picked_arm = best_arms[idx].0; // NEXT arm

        // Stop if destination reached!!
        let next_segment_id = picked_arm.segment();
        let is_final_turn = next_segment_id == building_segment_id; // Fine because this function is only for intersections, not direct-to-building navigation.
        // I ensured that all intersections have been passed and the final turn leads the car to the segment where the building is located on.
        // The finer movement will be done by the physical trajectory outside this function.

        // NEXT turn use next values bruh
        turns.push(TurnIdentification {
            node_id,
            segment_id: next_segment_id,
            is_final_turn,
        }); // NEXT!!!
        if is_final_turn {
            break;
        }; // Stop at final turn
    }

    Ok(CarSignfindingTrajectory {
        car_id: car.id,
        turns,
    })
}

fn get_last_turn(
    turns: &Vec<TurnIdentification>,
    car: &Car,
    buildings: &Buildings,
    address: &Address,
    road_storage: &RoadStorage,
    building_pos: WorldPos,
    building_segment_id: SegmentId,
) -> Result<TurnIdentification, SignfindingError> {
    if let Some(lt) = turns.last().copied().or(car.last_turn) {
        Ok(lt)
    } else {
        // Bootstrap: create initial turn from current position
        let lane_ref = if let Some(lane) = car.lane {
            lane
        } else {
            road_storage.lane_ref_closest_to_pos(&car.pos)
        };

        match lane_ref {
            LaneRef::Lane(lane_id, _) => {
                let Some(lane) = road_storage.lanes.get(lane_id.index()) else {
                    return Err(SignfindingError::LaneDoesntExist);
                };
                let segment_id = lane.segment();

                // The lane is directional: it goes FROM from_node TO to_node
                // We're on this lane, heading from from_node toward to_node
                // So from_node is BEHIND us, to_node is AHEAD
                let node_id = lane.from_node();

                Ok(TurnIdentification {
                    node_id,    // The node BEHIND us
                    segment_id, // The segment we're currently on
                    is_final_turn: segment_id == building_segment_id,
                })
            }
            LaneRef::NodeLane(node_id, _nl_id, _) => {
                // We're in a NodeLane (inside an intersection)
                // Find which outgoing segment leads closest to destination
                let Some(node) = road_storage.node(node_id) else {
                    return Err(SignfindingError::NodeDoesntExist);
                };

                let mut best_segment_id = None;
                let mut best_dist_sq = f32::INFINITY;

                for arm in node.arms() {
                    let seg_id = arm.segment();
                    let Some(segment) = road_storage.segments.get(seg_id.index()) else {
                        continue;
                    };

                    // Find the far endpoint of this segment (opposite from current node)
                    let far_node_id = if segment.start() == node_id {
                        segment.end()
                    } else if segment.end() == node_id {
                        segment.start()
                    } else {
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
                    return Err(SignfindingError::NoAvailableArms);
                };

                Ok(TurnIdentification {
                    node_id,    // The intersection it're in
                    segment_id, // The segment to take out of this intersection
                    is_final_turn: segment_id == building_segment_id,
                })
            }
        }
    }
}
#[derive(Debug, Clone)]
pub struct CarSignfindingTrajectory {
    pub car_id: CarId,
    pub turns: Vec<TurnIdentification>,
}
// #[derive(Debug, Clone)]
// pub struct SignfindingTurn {
//     pub turn_identification: TurnIdentification
// }

#[derive(Debug)]
pub enum SignfindingError {
    NoAddress,

    NoLastTurn,
    SegmentDoesntExist,
    NodeDoesntExist,
    NoAvailableArms,
    InvalidArmWeights,
    BuildingDoesntExist,
    LotDoesntExist,
    SegmentEndPointsDontMatchLastTurn,
    LaneDoesntExist,
}

pub struct SignFindingOptions {
    /// lower = more optimal arms chosen, higher -> more random arms/exploration. This value is quite exponential, beware.
    pub exploration_temperature: f32,
    /// How long a single physical trajectory is in seconds. This controls how quickly the cars react and heavily impacts performance. Higher is better performance.
    pub physical_traj_duration: f32,
}
impl SignFindingOptions {
    pub fn new(variables: &Variables) -> SignFindingOptions {
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
        }
    }
}

fn make_weighted_index(
    scores: &[f32],
    sf_options: &SignFindingOptions,
) -> Result<WeightedIndex<f32>, SignfindingError> {
    if scores.is_empty() {
        return Err(SignfindingError::InvalidArmWeights);
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
        return WeightedIndex::new(vec![1.0_f32; scores.len()])
            .map_err(|_| SignfindingError::InvalidArmWeights);
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

    WeightedIndex::new(&final_weights).map_err(|_| SignfindingError::InvalidArmWeights)
}

/// Modified version that returns (NodeLaneId, LaneGeometry) pairs
fn find_node_lane_path_with_ids(
    node: &Node,
    from_lane_ref: LaneRef,
    to_lane_id: LaneId,
) -> Vec<(NodeLaneId, &LaneGeometry)> {
    use std::collections::{HashMap, VecDeque};

    let mut queue = VecDeque::new();
    let mut came_from: HashMap<NodeLaneId, Option<NodeLaneId>> = HashMap::new();

    for nl in node.node_lanes() {
        if !nl.is_enabled() {
            continue;
        }
        if nl
            .merging()
            .iter()
            .any(|lr| lanes_match(lr, &from_lane_ref))
        {
            queue.push_back(nl.id());
            came_from.insert(nl.id(), None);
        }
    }

    let mut target_nl_id = None;

    while let Some(current_id) = queue.pop_front() {
        let Some(current_nl) = node.node_lanes().iter().find(|nl| nl.id() == current_id) else {
            continue;
        };

        if current_nl
            .splitting()
            .iter()
            .any(|lr| matches!(lr, LaneRef::Lane(seg_lane_id, _) if *seg_lane_id == to_lane_id))
        {
            target_nl_id = Some(current_id);
            break;
        }

        for split_ref in current_nl.splitting() {
            if let LaneRef::NodeLane(_, next_nl_id, _) = split_ref {
                if !came_from.contains_key(next_nl_id) {
                    came_from.insert(*next_nl_id, Some(current_id));
                    queue.push_back(*next_nl_id);
                }
            }
        }
    }

    let mut path_ids = Vec::new();
    if let Some(mut current_id) = target_nl_id {
        loop {
            path_ids.push(current_id);
            match came_from.get(&current_id) {
                Some(Some(prev_id)) => current_id = *prev_id,
                _ => break,
            }
        }
        path_ids.reverse();
    }

    path_ids
        .iter()
        .filter_map(|&nl_id| {
            node.node_lanes()
                .iter()
                .find(|nl| nl.id() == nl_id)
                .map(|nl| (nl_id, nl.geometry()))
        })
        .collect()
}

fn build_road_path(
    car: &Car,
    sf_traj: &CarSignfindingTrajectory,
    road_storage: &RoadStorage,
) -> RoadPath {
    let mut pts: Vec<WorldPos> = Vec::new();
    let mut limits: Vec<f32> = Vec::new();
    let mut lane_refs: Vec<Option<LaneRef>> = Vec::new();

    println!("=== Building path for car {:?} ===", car.id);
    println!("Car pos: {:?}", car.pos);
    println!("Signfinding trajectory has {} turns", sf_traj.turns.len());

    let cur = resolve_current_lane(car, road_storage);

    // IF no current lane, FIND the closest one!
    let (current_lane_id, current_lane) = if let Some((lid, lane)) = cur {
        (lid, lane)
    } else {
        println!("No explicit lane - finding closest lane...");
        let closest_ref = road_storage.lane_ref_closest_to_pos(&car.pos);
        match closest_ref {
            LaneRef::Lane(lid, _) => {
                let Some(lane) = road_storage.lanes.get(lid.index()) else {
                    println!("  ERROR: Closest lane doesn't exist!");
                    return RoadPath::build(vec![car.pos], vec![14.0], vec![None]);
                };
                println!("  Found lane {:?}", lid);
                (lid, lane)
            }
            LaneRef::NodeLane(node_id, nl_id, _) => {
                println!("  Closest is NodeLane - finding outgoing lane...");
                let Some(node) = road_storage.node(node_id) else {
                    return RoadPath::build(vec![car.pos], vec![14.0], vec![None]);
                };

                if let Some(first_turn) = sf_traj.turns.first() {
                    let Some(seg) = road_storage.segments.get(first_turn.segment_id.index()) else {
                        return RoadPath::build(vec![car.pos], vec![14.0], vec![None]);
                    };
                    if let Some((lid, lane)) = lane_from_node(seg, first_turn.node_id, road_storage)
                    {
                        println!("  Using first turn's lane {:?}", lid);
                        (lid, lane)
                    } else {
                        return RoadPath::build(vec![car.pos], vec![14.0], vec![None]);
                    }
                } else {
                    return RoadPath::build(vec![car.pos], vec![14.0], vec![None]);
                }
            }
        }
    };

    let mut current_lane_ref: Option<LaneRef> = Some(LaneRef::Lane(current_lane_id, 0));

    println!(
        "Starting lane {:?}: from node {:?} to node {:?}",
        current_lane_id,
        current_lane.from_node(),
        current_lane.to_node()
    );
    println!(
        "  Lane has {} geometry points",
        current_lane.geometry().points.len()
    );

    // FIX: Use poly_idx to get the actual arc-length!
    let (_closest_pt, _dist, _, poly_idx) = current_lane.geometry().closest_point_to(&car.pos);
    let proj_s = current_lane
        .geometry()
        .lengths
        .get(poly_idx as usize)
        .copied()
        .unwrap_or(0.0);
    let total_len = current_lane
        .geometry()
        .lengths
        .last()
        .copied()
        .unwrap_or(0.0);

    println!(
        "  Car at poly_idx={}, arc-length s={:.2}, total_length={:.2}",
        poly_idx, proj_s, total_len
    );

    let lane_ref = Some(LaneRef::Lane(current_lane_id, 0));

    // Add ALL points from poly_idx onward (including intermediate points between poly_idx and poly_idx+1)
    // Start from the poly_idx point itself
    for (i, &p) in current_lane.geometry().points.iter().enumerate() {
        if i >= poly_idx as usize {
            push_wp(
                &p,
                current_lane.speed_limit(),
                lane_ref,
                &mut pts,
                &mut limits,
                &mut lane_refs,
            );
        }
    }

    println!(
        "  Added {} waypoints from starting lane (from idx {} onward)",
        current_lane.geometry().points.len() - poly_idx as usize,
        poly_idx
    );

    // Rest of the function - following turns
    for (turn_idx, turn) in sf_traj.turns.iter().enumerate() {
        println!(
            "Turn {}: node {:?}, segment {:?}, final={}",
            turn_idx, turn.node_id, turn.segment_id, turn.is_final_turn
        );

        let Some(node) = road_storage.node(turn.node_id) else {
            println!("  ERROR: Node doesn't exist!");
            break;
        };
        let Some(next_seg) = road_storage.segments.get(turn.segment_id.index()) else {
            println!("  ERROR: Segment doesn't exist!");
            break;
        };

        let Some((out_lid, out_lane)) = lane_from_node(next_seg, turn.node_id, road_storage) else {
            println!("  ERROR: No lane from node!");
            break;
        };

        println!(
            "  Out lane {:?}: from {:?} to {:?}, {} points",
            out_lid,
            out_lane.from_node(),
            out_lane.to_node(),
            out_lane.geometry().points.len()
        );

        let int_limit = out_lane.speed_limit() * INTER_SPEED;

        if let Some(from_ref) = current_lane_ref {
            let node_lane_path = find_node_lane_path_with_ids(node, from_ref, out_lid);

            if !node_lane_path.is_empty() {
                println!("  Found {} NodeLanes", node_lane_path.len());
                for (nl_id, geom) in node_lane_path {
                    let nl_ref = Some(LaneRef::NodeLane(turn.node_id, nl_id, 0));
                    for &p in &geom.points {
                        push_wp(&p, int_limit, nl_ref, &mut pts, &mut limits, &mut lane_refs);
                    }
                }
            } else {
                println!("  No NodeLane path - using fallback");
                if let Some(&first) = out_lane.geometry().points.first() {
                    push_wp(
                        &first,
                        int_limit,
                        None,
                        &mut pts,
                        &mut limits,
                        &mut lane_refs,
                    );
                }
            }
        }

        let out_lane_ref = Some(LaneRef::Lane(out_lid, 0));
        for &p in &out_lane.geometry().points {
            push_wp(
                &p,
                out_lane.speed_limit(),
                out_lane_ref,
                &mut pts,
                &mut limits,
                &mut lane_refs,
            );
        }

        current_lane_ref = out_lane_ref;
        if turn.is_final_turn {
            break;
        }
    }

    println!("Final path: {} waypoints, {:.2}m total", pts.len(), {
        let mut sum = 0.0;
        for i in 1..pts.len() {
            sum += pts[i - 1].delta_to(pts[i]).length();
        }
        sum
    });
    println!("===================\n");

    RoadPath::build(pts, limits, lane_refs)
}

/// Push a waypoint with its associated lane, silently dropping near-duplicates.
#[inline]
fn push_wp(
    p: &WorldPos,
    limit: f32,
    lane_ref: Option<LaneRef>,
    pts: &mut Vec<WorldPos>,
    limits: &mut Vec<f32>,
    lane_refs: &mut Vec<Option<LaneRef>>,
) {
    if let Some(last) = pts.last() {
        if last.delta_to(*p).length() < DEDUP_M {
            return;
        }
    }
    pts.push(*p);
    limits.push(limit);
    lane_refs.push(lane_ref);
}

/// Resolve the lane the car is currently on.
fn resolve_current_lane<'a>(car: &Car, rs: &'a RoadStorage) -> Option<(LaneId, &'a Lane)> {
    // Prefer explicit lane assignment.
    if let Some(LaneRef::Lane(lid, _)) = car.lane {
        if let Some(lane) = rs.lanes.get(lid.index()) {
            return Some((lid, lane));
        }
    }
    // Mid-intersection or unknown: derive from last_turn.
    let lt = car.last_turn?;
    let seg = rs.segments.get(lt.segment_id.index())?;
    lane_from_node(seg, lt.node_id, rs)
}

/// First enabled lane in `seg` whose `from` node is `from_node`.
/// (If `Lane::from` is private in your codebase, replace with `lane.from()`.)
fn lane_from_node<'a>(
    seg: &Segment,
    from_node: NodeId,
    rs: &'a RoadStorage,
) -> Option<(LaneId, &'a Lane)> {
    seg.lanes.iter().copied().find_map(|lid| {
        let lane = rs.lanes.get(lid.index())?;
        (lane.from_node() == from_node && lane.is_enabled()).then_some((lid, lane))
    })
}

/// Find the path through the NodeLane graph from incoming lane to outgoing lane.
/// Returns the sequence of NodeLane geometries to traverse through the intersection.
fn find_node_lane_path(
    node: &Node,
    from_lane_ref: LaneRef,
    to_lane_id: LaneId,
) -> Vec<&LaneGeometry> {
    use std::collections::{HashMap, VecDeque};

    // BFS through the NodeLane graph
    let mut queue = VecDeque::new();
    let mut came_from: HashMap<NodeLaneId, Option<NodeLaneId>> = HashMap::new();

    // Find entry NodeLanes: ones that merge from our incoming lane
    for nl in node.node_lanes() {
        if !nl.is_enabled() {
            continue;
        }
        if nl
            .merging()
            .iter()
            .any(|lr| lanes_match(lr, &from_lane_ref))
        {
            queue.push_back(nl.id());
            came_from.insert(nl.id(), None);
        }
    }

    let mut target_nl_id = None;

    // BFS to find a NodeLane that splits into our target segment lane
    while let Some(current_id) = queue.pop_front() {
        let Some(current_nl) = node.node_lanes().iter().find(|nl| nl.id() == current_id) else {
            continue;
        };

        // Check if this NodeLane splits into our target
        if current_nl
            .splitting()
            .iter()
            .any(|lr| matches!(lr, LaneRef::Lane(seg_lane_id, _) if *seg_lane_id == to_lane_id))
        {
            target_nl_id = Some(current_id);
            break;
        }

        // Explore NodeLanes that this one splits into
        for split_ref in current_nl.splitting() {
            if let LaneRef::NodeLane(_, next_nl_id, _) = split_ref {
                if !came_from.contains_key(next_nl_id) {
                    came_from.insert(*next_nl_id, Some(current_id));
                    queue.push_back(*next_nl_id);
                }
            }
        }
    }

    // Reconstruct path by backtracking
    let mut path_ids = Vec::new();
    if let Some(mut current_id) = target_nl_id {
        loop {
            path_ids.push(current_id);
            match came_from.get(&current_id) {
                Some(Some(prev_id)) => current_id = *prev_id,
                _ => break,
            }
        }
        path_ids.reverse();
    }

    // Convert NodeLane IDs to their geometries
    path_ids
        .iter()
        .filter_map(|&nl_id| {
            node.node_lanes()
                .iter()
                .find(|nl| nl.id() == nl_id)
                .map(|nl| nl.geometry())
        })
        .collect()
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
    duration: f32,
) -> CarTrajectory {
    let n_steps = (duration / SIM_DT).ceil() as u32;
    let mut pts = Vec::with_capacity((n_steps / SIM_SAMPLE + 2) as usize);

    let mut pos = car.pos;
    let mut quat = sanitize_quat(car.quat);
    let mut speed = car.current_velocity.length();
    let mut r = car.yaw_rate;
    let mut arc_s = 0.0f64;
    let mut t = start_t;

    pts.push(CarTrajectoryPoint {
        time: t,
        pos: origin.delta_to(pos),
        quat,
        velocity: (quat * Vec3::Z) * speed, // Z is FORWARD, not Y!
        lane_ref: car.lane,
    });

    for step in 1..=n_steps {
        t += SIM_DT as f64;

        let v0 = if arc_s >= path.total_len() {
            0.0
        } else {
            path.limit_at(arc_s)
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
        arc_s += (speed * SIM_DT) as f64;

        let look_s = (arc_s + LOOKAHEAD_M).min(path.total_len());
        let (look_pt, _) = path.sample(look_s);
        let to_tgt = pos.delta_to(look_pt);

        let desired_r = if to_tgt.length_squared() > 0.04 && speed > 0.2 {
            let fwd = quat * Vec3::Z; // FORWARD is Z, not Y!
            let tgt = to_tgt.normalize();
            // Use Z for forward, cross.y gives turning direction
            let cross_y = fwd.cross(tgt).y;
            2.0 * speed * cross_y / LOOKAHEAD_M as f32
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

        if step % SIM_SAMPLE == 0 {
            let lane_ref = path.lane_ref_at(arc_s);
            pts.push(CarTrajectoryPoint {
                time: t,
                pos: origin.delta_to(pos),
                quat,
                velocity: vel,
                lane_ref,
            });
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

    CarTrajectory {
        car_id: car.id,
        origin,
        points: pts,
        end_quat: quat,
        end_yaw_rate: r,
        end_steering_angle: 0.0,
        end_steering_vel: 0.0,
    }
}

fn idle_trajectory(car: &Car, origin: WorldPos, start_t: f64, duration: f32) -> CarTrajectory {
    let rel = origin.delta_to(car.pos);
    let lane_ref = car.lane;
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
    }
}

const IDM_ACCEL: f32 = 2.5; // comfortable acceleration   m/s²
const IDM_BRAKE: f32 = 4.0; // comfortable deceleration   m/s²
const IDM_S0: f32 = 2.5; // minimum stationary gap     m
const IDM_THW: f32 = 1.5; // time headway               s
const IDM_DELTA: f32 = 4.0; // free-road velocity exponent
const LOOKAHEAD_M: f64 = 8.0; // pure-pursuit look-ahead    m
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
    limits: Vec<f32>,                // speed limit (m/s) per waypoint
    lane_refs: Vec<Option<LaneRef>>, // lane ref per waypoint
    arc: Vec<f64>,                   // cumulative arc length (m) per waypoint
}

impl RoadPath {
    fn build(pts: Vec<WorldPos>, limits: Vec<f32>, lane_refs: Vec<Option<LaneRef>>) -> Self {
        let mut arc = Vec::with_capacity(pts.len());
        let mut cum = 0.0f64;
        arc.push(0.0);
        for i in 1..pts.len() {
            cum += pts[i - 1].delta_to(pts[i]).length() as f64;
            arc.push(cum); // CUM HAHAH
        }
        Self {
            pts,
            limits,
            lane_refs,
            arc,
        }
    }

    fn total_len(&self) -> f64 {
        self.arc.last().copied().unwrap_or(0.0)
    }

    /// World position + forward tangent at arc-length `s`.
    fn sample(&self, s: f64) -> (WorldPos, Vec3) {
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
        let span = (self.arc[hi] - self.arc[lo]) as f32;
        let frac = if span < 1e-6 {
            0.0
        } else {
            ((s - self.arc[lo]) as f32) / span
        };
        let d = self.pts[lo].delta_to(self.pts[hi]);
        (self.pts[lo].add_vec3(d * frac), d.normalize_or_zero())
    }

    /// Speed limit at arc-length `s` — falls back to ~50 km/h.
    fn limit_at(&self, s: f64) -> f32 {
        if self.limits.is_empty() {
            return 14.0;
        }
        let i = self.arc.partition_point(|&a| a <= s).saturating_sub(1);
        self.limits[i.min(self.limits.len() - 1)]
    }

    fn lane_ref_at(&self, s: f64) -> Option<LaneRef> {
        if self.lane_refs.is_empty() {
            return None;
        }
        let i = self.arc.partition_point(|&a| a <= s).saturating_sub(1);
        self.lane_refs[i.min(self.lane_refs.len() - 1)]
    }
}
