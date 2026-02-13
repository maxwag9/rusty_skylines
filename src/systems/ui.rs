use crate::resources::Resources;

pub fn ui_system(resources: &mut Resources) {
    let input = &mut resources.world_core.input;
    let time = &resources.world_core.time;
    let dt = time.target_frametime;
    input.now = time.total_time;
    resources.ui_loader.handle_touches(
        dt,
        input,
        time,
        &mut resources.world_core.terrain_subsystem,
        resources.window.inner_size(),
        &mut resources.world_core.road_subsystem.road_editor.style,
        &mut resources.command_queues,
    );
}
