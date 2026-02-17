use crate::resources::Resources;

pub fn render_system(resources: &mut Resources) {
    let world = &mut resources.world_core;
    let renderer = &mut resources.render_core;
    let Some(camera) = world.world_state.camera(world.world_state.main_camera()) else {
        return;
    };
    renderer.render(
        &resources.surface,
        camera,
        &mut resources.ui_loader,
        &world.time,
        &mut world.input,
        &resources.settings,
        &mut world.terrain_subsystem,
        &world.road_subsystem,
        &world.car_subsystem,
        &world.world_state.astronomy,
    );
}
