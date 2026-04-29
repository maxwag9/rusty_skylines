use crate::resources::Resources;
use crate::world::cars::car_render::interpolate_cars;
use winit::event_loop::ActiveEventLoop;

pub fn run_ticked(resources: &mut Resources) {
    let world = &mut resources.world;
    let renderer = &mut resources.render_core;
    let camera = &world.world_state.camera;
    let (time, terrain, roads, cars, input) = (
        &mut world.time,
        &mut world.terrain,
        &mut world.roads,
        &mut world.cars,
        &mut world.input,
    );
    let (settings, gizmo, device, queue) = (
        &mut resources.settings,
        &mut renderer.gizmo,
        &renderer.device,
        &renderer.queue,
    );
    let aspect = renderer.config.width as f32 / renderer.config.height as f32;

    if world.simulation.running {
        terrain.update(device, queue, camera, aspect, settings, input, time);
    }

    roads.update(terrain, cars, input, time, gizmo);
}
pub fn run_sim(resources: &mut Resources) {
    let world = &mut resources.world;
    world.simulation.update(
        &mut world.world_state,
        &mut world.cars,
        &world.roads,
        &world.terrain,
        &mut world.input,
        &world.time,
        &mut world.buildings,
        &mut resources.render_core,
        &mut resources.ui,
        &resources.settings,
    );
}

pub fn run_ui(resources: &mut Resources, event_loop: &dyn ActiveEventLoop) {
    let input = &mut resources.world.input;
    let time = &resources.world.time;
    let dt = time.target_frametime;
    input.now = time.total_time;
    resources.ui.handle_touches(
        dt,
        input,
        time,
        &mut resources.world.terrain,
        &mut resources.render_core.props,
        &mut resources.world.buildings,
        resources.window.surface_size(),
        &mut resources.world.roads,
        &mut resources.command_queues,
        &mut resources.settings,
        &mut resources.world.world_state.camera,
        &mut resources.world.world_state.cam_controller,
        event_loop,
        &mut resources.world.world_state.game_state,
    );
}

pub fn run_interpolation(resources: &mut Resources) {
    interpolate_cars(&mut resources.world, &resources.settings);
}

pub fn run_render(resources: &mut Resources) {
    let aspect =
        resources.render_core.config.width as f32 / resources.render_core.config.height as f32;
    resources
        .world
        .world_state
        .camera
        .compute_matrices(aspect, &resources.settings);

    resources.render_core.render(
        &resources.surface,
        &mut resources.world,
        &mut resources.ui,
        &resources.settings,
    );

    resources.world.world_state.camera.end_frame();
}
