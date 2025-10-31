use crate::resources::Resources;
use crate::world::World;

pub fn simulation_system(_world: &mut World, resources: &mut Resources) {
    let dt = resources.time.delta;
    resources.simulation.process_events(&mut resources.events);
    resources.simulation.update(dt);
}
