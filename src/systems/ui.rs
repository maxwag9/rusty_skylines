use crate::resources::Resources;
use crate::world::World;

pub fn ui_system(_world: &mut World, resources: &mut Resources) {
    let dt = resources.time.target_frametime;
    resources.input.now = resources.time.total_time;
    resources
        .ui_loader
        .handle_touches(dt, &mut resources.input, &resources.time);
}
