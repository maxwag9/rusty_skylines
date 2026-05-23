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
// NOTICE how I didn't mention 🤢 'pathfinding', 'Dijkstra' or 'A*' at all?
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
// So, let me write this file! (Tomorrow)
