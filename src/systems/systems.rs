use crate::resources::Resources;
use crate::world::cars::car_render::interpolate_cars;

pub fn run_ticked(resources: &mut Resources) {
    let world = &mut resources.world_core;
    let renderer = &mut resources.render_core;
    let camera = &world.world_state.camera;
    let (time, terrain, roads, cars, input) = (
        &mut world.time,
        &mut world.terrain,
        &mut world.road_subsystem,
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
    let world = &mut resources.world_core;
    let renderer = &mut resources.render_core;
    let camera = &mut world.world_state.camera;
    let cam_controller = &mut world.world_state.cam_controller;
    world.simulation.update(
        &mut world.terrain,
        &mut world.road_subsystem,
        &mut world.cars,
        &resources.settings,
        &world.time,
        &mut world.input,
        &mut resources.ui_loader.variables,
        camera,
        cam_controller,
        &renderer.device,
        &renderer.queue,
        &renderer.config,
        &mut renderer.gizmo,
    );
}

pub fn run_ui(resources: &mut Resources) {
    let input = &mut resources.world_core.input;
    let time = &resources.world_core.time;
    let dt = time.target_frametime;
    input.now = time.total_time;
    resources.ui_loader.handle_touches(
        dt,
        input,
        time,
        &mut resources.world_core.terrain,
        resources.window.inner_size(),
        &mut resources.world_core.road_subsystem.road_editor.style,
        &mut resources.command_queues,
        &mut resources.settings,
    );
}

pub fn run_interpolation(resources: &mut Resources) {
    interpolate_cars(&mut resources.world_core, &resources.settings);
}

pub fn run_render(resources: &mut Resources) {
    let aspect =
        resources.render_core.config.width as f32 / resources.render_core.config.height as f32;
    resources
        .world_core
        .world_state
        .camera
        .compute_matrices(aspect, &resources.settings);

    resources.render_core.render(
        &resources.surface,
        &mut resources.world_core,
        &mut resources.ui_loader,
        &resources.settings,
    );

    resources.world_core.world_state.camera.end_frame();
}
