use crate::commands::Command;
use crate::resources::Resources;
use crate::ui::variables::Variables;
use crate::world::terrain::terrain_subsystem::Cursor;

pub fn cursor_system(cursor: &mut Cursor, event: &Command, variables: &mut Variables) {
    match event {
        Command::SetCursorMode(mode) => {
            cursor.mode = mode.clone();
            variables.set_string("cursor_mode", format!("{:?}", mode));
        }
        _ => {}
    }
}
pub fn run_commands(resources: &mut Resources) {
    let world = &mut resources.world;
    world.events.flip();

    for event in world.events.drain() {
        // order is explicit and intentional
        resources
            .simulation
            .process_simulation_state_commands(&event);
        cursor_system(
            &mut world.terrain.cursor,
            &event,
            &mut resources.ui.variables,
        );
        // later:
        // audio_event_system(...)
        // ui_event_system(...)
    }
}
