use crate::resources::Resources;

pub fn simulation_system(resources: &mut Resources) {
    let world = &mut resources.world_core;
    let renderer = &mut resources.render_core;
    let Some(camera_bundle) = world
        .world_state
        .camera_and_controller_mut(world.world_state.main_camera())
    else {
        return;
    };
    world.simulation.update(
        &mut world.terrain_subsystem,
        &mut world.road_subsystem,
        &mut world.car_subsystem,
        &resources.settings,
        &world.time,
        &mut world.input,
        &mut resources.ui_loader.variables,
        camera_bundle,
        &renderer.device,
        &renderer.queue,
        &renderer.config,
        &mut renderer.gizmo,
    );
}
