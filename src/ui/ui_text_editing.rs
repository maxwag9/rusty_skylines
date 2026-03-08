use crate::ui::input::{Input, Mouse};
use crate::ui::menu::Menu;
use crate::ui::selections::SelectionManager;
use crate::ui::ui_edit_manager::{TextEditCommand, UiEditManager};
use crate::ui::ui_touch_manager::{EditorTouchExtension, ElementRef};
use crate::ui::vertex::{ElementKind, LayerDirty, UiButtonText, UiElement};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy)]
pub(crate) struct MouseSnapshot {
    pub mx: f32,
    pub my: f32,
    pub pressed: bool,
    pub just_pressed: bool,
    pub scroll: f32,
}

impl MouseSnapshot {
    pub fn from_mouse(mouse: &Mouse) -> Self {
        Self {
            mx: mouse.pos.x,
            my: mouse.pos.y,
            pressed: mouse.buttons.left.pressed,
            just_pressed: mouse.buttons.left.just_pressed,
            scroll: mouse.scroll_delta.y,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct HitResult {
    pub element: ElementRef,
    pub layer_order: u32,
    pub actions: Vec<String>,
    pub(crate) element_order: usize,
}

// ============================================================================
// TEXT EDITING
// ============================================================================

/// Handle text editing with undo support
pub fn handle_text_editing(
    selection: &mut SelectionManager,
    editor: &mut EditorTouchExtension,
    menus: &mut HashMap<String, Menu>,
    undo_manager: &mut UiEditManager,
    input: &mut Input,
    mouse_snapshot: MouseSnapshot,
) {
    let Some(sel) = &selection.primary else {
        return;
    };
    let sel_menu = sel.menu.clone();
    let sel_layer = sel.layer.clone();
    let sel_element_id = sel.id.clone();

    let Some((menu_name, menu)) = menus.iter_mut().find(|(n, m)| **n == sel_menu && m.active)
    else {
        return;
    };

    let Some(layer) = menu.layers.iter_mut().find(|l| l.name == sel_layer) else {
        return;
    };

    let Some(text) = layer
        .elements
        .iter_mut()
        .filter_map(UiElement::as_text_mut)
        .find(|t| t.id == sel_element_id)
    else {
        return;
    };

    let before_text = text.text.clone();
    let before_template = text.template.clone();
    let before_caret = text.caret;
    process_text_editing_input(editor, input, mouse_snapshot, text, &mut layer.dirty);

    if text.text != before_text || text.template != before_template {
        undo_manager.push_command(
            TextEditCommand {
                affected_element: ElementRef {
                    menu: menu_name.clone(),
                    layer: layer.name.clone(),
                    id: sel_element_id.clone(),
                    kind: ElementKind::Text,
                },
                before_text,
                after_text: text.text.clone(),
                before_template,
                after_template: text.template.clone(),
                before_caret,
                after_caret: text.caret,
            },
            false,
        );
    }
}

pub(crate) fn process_text_editing_input(
    editor: &mut EditorTouchExtension,
    input: &mut Input,
    mouse_snapshot: MouseSnapshot,
    text: &mut UiButtonText,
    dirty: &mut LayerDirty,
) {
    if handle_mouse_caret_selection(editor, mouse_snapshot, text) {
        return;
    }

    if handle_clipboard_commands(input, text, dirty) {
        return;
    }

    if handle_backspace(input, text, dirty) {
        return;
    }

    if handle_character_input(input, text, dirty) {
        return;
    }

    handle_arrow_navigation(input, text, dirty);
}

fn handle_mouse_caret_selection(
    editor: &mut EditorTouchExtension,
    mouse_snapshot: MouseSnapshot,
    text: &mut UiButtonText,
) -> bool {
    let mx = mouse_snapshot.mx;
    let my = mouse_snapshot.my;

    let x0 = text.x;
    let y0 = text.y;
    let x1 = x0 + text.natural_width;
    let y1 = y0 + text.natural_height;

    if mouse_snapshot.just_pressed && mx >= x0 && mx <= x1 && my >= y0 && my <= y1 {
        let new_caret = pick_caret(text, mx);
        text.caret = new_caret;
        text.sel_start = new_caret;
        text.sel_end = new_caret;
        text.has_selection = false;
        editor.dragging_text_selection = true;
        return true;
    }

    if editor.dragging_text_selection && mouse_snapshot.pressed {
        let new_pos = pick_caret(text, mx);
        text.sel_end = new_pos;
        text.has_selection = text.sel_end != text.sel_start;
        text.caret = new_pos;
        return true;
    }

    if editor.dragging_text_selection && !mouse_snapshot.pressed {
        editor.dragging_text_selection = false;
    }

    false
}

fn handle_clipboard_commands(
    input: &mut Input,
    text: &mut UiButtonText,
    dirty: &mut LayerDirty,
) -> bool {
    if !input.ctrl {
        return false;
    }

    enum Cmd {
        Copy,
        Cut,
        Paste,
    }

    let cmd = if input.action_repeat("Paste text") {
        Cmd::Paste
    } else if input.action_pressed_once("Copy text") {
        Cmd::Copy
    } else if input.action_pressed_once("Cut text") {
        Cmd::Cut
    } else {
        return false;
    };

    let clipboard = &mut input.clipboard;

    let is_template_mode = !text.input_box;

    let (l, r) = text.selection_range();
    let active = if is_template_mode {
        &mut text.template
    } else {
        &mut text.text
    };

    match cmd {
        Cmd::Copy => {
            if !text.has_selection {
                return false;
            }

            let Some(slice) = active.get(l..r) else {
                return false;
            };

            clipboard.set_text(slice.to_string()).is_ok()
        }

        Cmd::Cut => {
            if !text.has_selection {
                return false;
            }

            let Some(slice) = active.get(l..r) else {
                return false;
            };

            if clipboard.set_text(slice.to_string()).is_err() {
                return false;
            }

            active.replace_range(l..r, "");
            text.caret = l;
            text.clear_selection();

            if is_template_mode {
                text.text = text.template.clone();
            }

            dirty.mark_texts();
            true
        }

        Cmd::Paste => {
            let Ok(clip) = clipboard.get_text() else {
                println!("Failed to get text from clipboard");
                return false;
            };

            if clip.is_empty() {
                println!("Clipboard is empty '{}'", clip);
                return false;
            }

            if text.has_selection {
                if active.get(l..r).is_none() {
                    println!("Failed to get range from text");
                    return false;
                }

                active.replace_range(l..r, &clip);
                text.caret = l + clip.len();
                text.clear_selection();
            } else {
                if text.caret > active.len() {
                    println!("Failed to get range from non selected text");
                    return false;
                }

                active.insert_str(text.caret, &clip);
                text.caret += clip.len();
            }

            if is_template_mode {
                text.text = text.template.clone();
            }

            dirty.mark_texts();
            true
        }
    }
}

fn handle_backspace(input: &mut Input, text: &mut UiButtonText, dirty: &mut LayerDirty) -> bool {
    if !input.action_repeat("Backspace") {
        return false;
    }

    let is_template_mode = !text.input_box;

    if text.has_selection {
        let (l, r) = text.selection_range();
        if is_template_mode {
            text.template.replace_range(l..r, "");
            text.text = text.template.clone();
        } else {
            text.text.replace_range(l..r, "");
        }
        text.caret = l;
        text.clear_selection();
        dirty.mark_texts();
        return true;
    }

    if text.caret > 0 {
        if is_template_mode {
            text.template.remove(text.caret - 1);
            text.text = text.template.clone();
        } else {
            text.text.remove(text.caret - 1);
        }
        text.caret -= 1;
        dirty.mark_texts();
    }

    true
}

fn handle_character_input(
    input: &mut Input,
    text: &mut UiButtonText,
    dirty: &mut LayerDirty,
) -> bool {
    if !input.repeat("char_repeat", !input.text_chars.is_empty()) {
        return false;
    }

    let is_template_mode = !text.input_box; // or override mode!!

    if text.has_selection {
        delete_selection(text, is_template_mode);
    }

    insert_characters(text, &input.text_chars, is_template_mode);
    dirty.mark_texts();

    true
}

fn delete_selection(text: &mut UiButtonText, is_template_mode: bool) {
    let (l, r) = text.selection_range();

    if is_template_mode {
        let bl = caret_to_byte(&text.template, l);
        let br = caret_to_byte(&text.template, r);
        text.template.replace_range(bl..br, "");
        text.text = text.template.clone();
    } else {
        let bl = caret_to_byte(&text.text, l);
        let br = caret_to_byte(&text.text, r);
        text.text.replace_range(bl..br, "");
    }

    text.caret = l;
    text.clear_selection();
}

fn insert_characters(text: &mut UiButtonText, chars: &HashSet<String>, is_template_mode: bool) {
    for s in chars {
        if s.is_empty() {
            continue;
        }

        if is_template_mode {
            let bi = caret_to_byte(&text.template, text.caret);
            text.template.insert_str(bi, s);
            text.text = text.template.clone();
        } else {
            let bi = caret_to_byte(&text.text, text.caret);
            text.text.insert_str(bi, s);
        }

        text.caret += s.chars().count();
    }
}

fn handle_arrow_navigation(input: &mut Input, text: &mut UiButtonText, dirty: &mut LayerDirty) {
    if text.has_selection {
        handle_selection_collapse(input, text, dirty);
        return;
    }

    if input.action_repeat("Move Cursor Left") && text.caret > 0 {
        text.caret -= 1;
        dirty.mark_texts();
    }

    if input.action_repeat("Move Cursor Right") && text.caret < text.template.len() {
        text.caret += 1;
        dirty.mark_texts();
    }
}

fn handle_selection_collapse(input: &mut Input, text: &mut UiButtonText, dirty: &mut LayerDirty) {
    let (l, r) = text.selection_range();

    if input.action_pressed_once("Move Cursor Left") {
        text.caret = l;
        text.clear_selection();
        dirty.mark_texts();
    }

    if input.action_pressed_once("Move Cursor Right") {
        text.caret = r;
        text.clear_selection();
        dirty.mark_texts();
    }
}

fn pick_caret(text: &UiButtonText, mx: f32) -> usize {
    for (i, (_, gx1)) in text.glyph_bounds.iter().enumerate() {
        if mx < *gx1 {
            return i;
        }
    }
    text.glyph_bounds.len()
}

fn caret_to_byte(text: &str, caret: usize) -> usize {
    text.char_indices()
        .nth(caret)
        .map(|(i, _)| i)
        .unwrap_or_else(|| text.len())
}
