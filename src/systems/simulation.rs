use crate::resources::Resources;
use crate::world::World;

pub fn simulation_system(_world: &mut World, resources: &mut Resources) {
    resources.time.update_sim();
    let dt = resources.time.sim_dt;
    resources.simulation.update(dt);
}
