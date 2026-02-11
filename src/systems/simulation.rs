use crate::resources::Resources;
use crate::world::World;

pub fn simulation_system(world: &mut World, resources: &mut Resources) {
    resources.time.update_sim();
    let Some(camera_bundle) = world.camera_and_controller_mut(world.main_camera()) else {
        return;
    };
    resources.simulation.update(
        &resources.renderer.core.terrain_renderer,
        &mut resources.renderer.core.car_subsystem,
        &resources.settings,
        &resources.time,
        &mut resources.input,
        camera_bundle,
    );
}
