use crate::commands::Command;
use crate::world::terrain::terrain_subsystem::Cursor;

pub fn cursor_system(cursor: &mut Cursor, event: &Command) {
    match event {
        Command::SetCursorMode(mode) => {
            cursor.mode = *mode;
        }
        _ => {}
    }
}
