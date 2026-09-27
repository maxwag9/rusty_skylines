// Utilities are water and sewage pipes, electricity conduits, internet conduits and more.
// Utilities are pipes.
// Ok so the struct needs to store: nothing
//
// The road network can store it all, it has nodes and segments.
// What the fuck am I doing here? I created a module directory called 'utilities' and added a file called 'utilities.rs' and I am writing in it right now and it's useless.
//
// Maybe I can put a struct here that takes the road network and manages the pipe crap. And this file could solve the pathfinding issue, to find out whether pipes are connected.
// But no, I already Road Regions... Looks like I did all the work already. The Road Region stuff is extremely buggy though. Well not that extreme...
// It can detect when I am obviously not connecting a new road to an existing road network, but it can't do anything when I remove a road or change its capacity.
//
// So in my opinion, I will just do all of that here, explicitly, with some pathfinding finding out what the road regions are, and most importantly, how the load distribution is.
//
// Since Utilities and props are the ones feeding and sapping electricity, I don't need to add power poles explicitely here.
// But I do need to make it possible to let props use utilities and to make power lines smhw. Super Mario Hunts Wario
// A prop can be connected to a road network if it is 5m or closer to its outermost point. This requires tracking on the roads' side.
// I place a 'connectable' prop and it registers itself into the connectable prop chunk storage, which I just store in the normal prop chunks.
// Since the outside of a road can be multiple lanes from center, I must test against basically all points of a road and find the closest.
// But maybe I can speed it up somehow, by adding a helper function to get the left and right sides of the road as a slice of points.
// So this recalculation happens when a Road is moved, edited in any way, added or whatever, it always checks the chunks near it.
// I hate this approach, there is too much to track, too complicated.
//
//DnB Jungle makes me think.
//
// I need to make it attached.
// When I try to place a connectable prop, I need to restrict it to roads or other connecting spots. So basically I hover near a road and it snaps, showing that it connects to that road.
// When I want to place is solitarily, it will let me place it, but will complain about not being connected. The only way to resolve that is to connect it to a road or a Utility that has a connector prop on-site or just some other connector prop.
// A power transformer will have a connector prop and that connector prop will be bound to the Utility, so the conversion function inside the Utility will change the electricity or water or whatever. Nah idk about that part.
//
// Each utility will be defined in a yaml file in the utilities folder and fully moddable, because I will use that mod.
// …
// And a utility is defined by its name 'Electricity', its unit String 'kWh',
// Automatic Utilities define it however I see fit, ploppable Utilities define their usage of certain utilities using the name 'Electricity' and the amount of consumption per hour at night vs at day and the production of the unit at night vs day.
// Also, there should be a secondary value, so that I can set '500' and unit 'V' and I can set it as conditional or something... Then the road network it is connected to must have the same 500 and V values in its secondary.
// And distribution can calculate using that. distribution/conversion Utilities can of course convert the voltage and stuff... I could also add multiple rails per road, so larger roads have more rails and therefore higher capacities.
// I can then define how much one rail can handle. A simple 2-way one lane each road can just have like 8 lanes, and conditionally upgrade it if you feel like it. More lanes equals more rails, like 4 rails per lane.
// The problem I see is, that if a modder adds a fifth utility, then all one way roads won't be able to support all 5 utilities, which is very annoying. So I will make sure that each lane will have n_of_utilities rails.
// This means that I have to keep the rail count in runtime, and not store the rails in the roads. Which I wouldn't do anyway, because for now there is no need to.
// Players defining rails manually shouldn't be possible, that kind of micromanagement is not what my citybuilder is for, at least for now. They can just add more rails using the road upgrade.
// So those rails are just in-simulation! I can't wait to implement this, but I still have some annoying conundrums to solve. What the hell is a conundrum??!?
//
// 1. I have to think more and define the structs in this file (The structs that will be deserialized from yaml later)
// 2. Think more and solve the Road Region crap. Or just get rid of it? It is kinda useless anyway. If a car needs to check if a route is possible at all, then it must consult this UtilitiesNetwork...
// But the utilities network also has non-road connections! OR does it? Maybe I will handle the non-road connections separately somehow?? Goddamnit. The non-road connection is just a consumer for all that it cares!
// It consumes and then transfers it over to another place where it is handled somehow, for example a power pole bypassing small roads to transfer electricity to another part of the city.
// But the problem again is that that transfering thing must EQUALIZE with the roads, or else the pole will cary 2 lanes of electricity and put it into a 1 lane road and overcrowd its rails maybe.
// Why do I like the rails approach?! I think it is because it sounds so discreet and as if the road is actually carrying something underneath. Also it makes for an easy way to limit and visualize utilities... I will keep it tbh.
// Ok my take is: Let the UtilitiesNetwork pathfind its own network around and while doing that, just mark which road networks are separate, keep my road network id stuff.
// 3. …

// I place a road and the gizmo draws the 4 utility rails. There are 4 because there are 4 utility types in total. Each lane has as many utility rails as there are utilities.
// The simulation will prioritize having all utility types using the utility rails rather than one type hogging 2 of them.
// The road knows what the utility rails are used for right now, it owns a Vec of UtilityRail and each rail has a utility String and the last usage percentage.
// The Gizmo can draw each rail by fetching the Utility struct and taking the defined color there.
// When the UtilityNetwork struct wants to find out whether the network is connected, it runs some light pathfinding to update the network with new info.
// It updates: Road Regions, capacity of each road (bottleneck search), it checks whether all the conditionals are fine too.
// It updates when: Any road gets any change, Any new Utility spawns, and more I don't know yet.
// Each rail has a f32 capacity. It is in whatever value the Utility struct sets it to be. Each rail has a f32 usage. The unit of those f32s is not linear, it is the unit that the Utility struct set. So the modder or me.
// I am terrified of pathfinding the network.
// I do not want to make a utility graph, because the roads can just contain the utility rails and the UtilityNetwork can just read the roads as if it is its own graph... And then the non-road stuff is added.
// So for the pathfinding, it is actually simple because the weights are just the capacity of the rail and then walking all the rails. And if some outside thing connects to the rail, it will walk that too.
// Maybe I just scrap pathfinding? Maybe I just run signfinding lol. Idk man.
// When I try to place a connecting prop, it can snap to roads and it will when close enough.
// Then the prop just exists there, connected to the UtilitiesNetwork by putting its PropInstanceId into a Hashmap<PropInstanceId, SegmentId> and another one Hashmap<SegmentId, PropInstanceId> when it was placed.
// If anything happens with the Segment, I will get the PropInstanceId and try to move it with the road or delete it with the road or shift it when upgrading a road... A solitary connectable propInstance doesn't need all that, it is independent.
// Nah I hate this, I hate 2 Hashmaps. But maybe it's fine, actually, and also I can just use one single function for any of these road changes!

// pub utility: UtilityId,!! UtilityId is u8.

use revision::revisioned;
use serde::Deserialize;

pub type UtilityId = u8;

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Utility {
    pub name: String,
    pub primary_unit: String,
    pub secondary_unit: String,
    pub amount_per_rail: f32,
    pub rail_color: [f32; 4],
    pub second_rail_color: [f32; 4],
    pub automatic_buildings_usage: UtilityUsage,
}

#[revisioned(revision = 1)]
#[derive(Clone, Copy, Debug, Default)]
pub struct UtilityRail {
    pub utility_id: UtilityId,
    pub capacity: f32, // For example kWh for total electricity.
    pub usage: f32,
    pub secondary_value: f32, // For example Voltage, for compatibility of rails and connections!
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[revisioned(revision = 1)]
pub struct UtilityUsage {
    #[serde(default)]
    pub consumption: f32,
    #[serde(default)]
    pub production: f32,
}
impl std::hash::Hash for UtilityUsage {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.consumption.to_bits().hash(state);
        self.production.to_bits().hash(state);
    }
}
