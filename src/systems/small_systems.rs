use crate::events::Event;
use crate::world::terrain_subsystem::Cursor;

pub fn cursor_system(cursor: &mut Cursor, event: &Event) {
    match event {
        Event::SetCursorMode(mode) => {
            cursor.mode = *mode;
        }
        _ => {}
    }
}
