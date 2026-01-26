use crate::events::Event;
use crate::renderer::world_renderer::Cursor;

pub fn cursor_system(cursor: &mut Cursor, event: &Event) {
    match event {
        Event::SetCursorMode(mode) => {
            cursor.mode = *mode;
        }
        _ => {}
    }
}
