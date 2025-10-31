use crate::resources::Resources;
use crate::world::World;

pub fn ui_system(_world: &mut World, resources: &mut Resources) {
    let dt = resources.time.delta;
    resources.ui_loader.handle_touches(&resources.mouse, dt);
}
