use crate::commands::Command;
use crate::resources::Resources;
use crate::world::terrain::terrain_subsystem::Cursor;

pub fn cursor_system(cursor: &mut Cursor, event: &Command) {
    match event {
        Command::SetCursorMode(mode) => {
            cursor.mode = *mode;
        }
        _ => {}
    }
}
pub fn run_commands(resources: &mut Resources) {
    let world = &mut resources.world_core;
    world.events.flip();

    for event in world.events.drain() {
        // order is explicit and intentional
        world.simulation.process_simulation_state_commands(&event);
        cursor_system(&mut world.terrain_subsystem.cursor, &event);
        // later:
        // audio_event_system(...)
        // ui_event_system(...)
    }
}
