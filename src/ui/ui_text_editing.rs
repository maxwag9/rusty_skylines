use crate::ui::input::{Input, MouseState};
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
    pub fn from_mouse(mouse: &MouseState) -> Self {
        Self {
            mx: mouse.pos.x,
            my: mouse.pos.y,
            pressed: mouse.left_pressed,
            just_pressed: mouse.left_just_pressed,
            scroll: mouse.scroll_delta.y,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum HitElement {
    Circle(usize),
    Handle(usize),
    Polygon(usize),
    Text(usize),
}

#[derive(Clone, Debug)]
pub(crate) struct HitResult {
    pub menu_name: String,
    pub layer_name: String,
    pub element: HitElement,
    pub layer_order: u32,
    pub action: Option<String>,
    pub(crate) element_order: usize,
}

impl HitResult {
    pub fn matches(&self, menu_name: &str, layer_name: &str, element: HitElement) -> bool {
        self.menu_name == menu_name && self.layer_name == layer_name && self.element == element
    }
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
        undo_manager.push_command(TextEditCommand {
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
        });
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

    if handle_ctrl_commands(editor, input, text, dirty) {
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

fn handle_ctrl_commands(
    editor: &mut EditorTouchExtension,
    input: &mut Input,
    text: &mut UiButtonText,
    dirty: &mut LayerDirty,
) -> bool {
    if !input.ctrl {
        return false;
    }

    if handle_paste(editor, input, text, dirty) {
        return true;
    }

    if handle_copy(editor, input, text) {
        return true;
    }

    if handle_cut(editor, input, text, dirty) {
        return true;
    }

    true
}

fn handle_paste(
    editor: &mut EditorTouchExtension,
    input: &mut Input,
    text: &mut UiButtonText,
    dirty: &mut LayerDirty,
) -> bool {
    if !input.action_repeat("Paste text") {
        return false;
    }

    let is_template_mode = !text.input_box;

    if text.has_selection {
        let (l, r) = text.selection_range();
        if is_template_mode {
            text.template.replace_range(l..r, &editor.clipboard);
        } else {
            text.text.replace_range(l..r, &editor.clipboard);
        }
        text.caret = l + editor.clipboard.len();
        text.clear_selection();
    } else {
        for c in editor.clipboard.clone().chars() {
            if is_template_mode {
                text.template.insert(text.caret, c);
            } else {
                text.text.insert(text.caret, c);
            }
            text.caret += 1;
        }
    }

    text.text = text.template.clone();
    dirty.mark_texts();

    true
}

fn handle_copy(editor: &mut EditorTouchExtension, input: &mut Input, text: &UiButtonText) -> bool {
    if !input.action_pressed_once("Copy text") {
        return false;
    }

    let (l, r) = text.selection_range();
    let is_template_mode = !text.input_box;

    editor.clipboard = if is_template_mode {
        text.template.get(l..r).unwrap_or("").to_string()
    } else {
        text.text.get(l..r).unwrap_or("").to_string()
    };

    true
}

fn handle_cut(
    editor: &mut EditorTouchExtension,
    input: &mut Input,
    text: &mut UiButtonText,
    dirty: &mut LayerDirty,
) -> bool {
    if !input.action_pressed_once("Cut text") {
        return false;
    }

    let (l, r) = text.selection_range();
    let is_template_mode = !text.input_box;

    if is_template_mode {
        editor.clipboard = text.template.get(l..r).unwrap_or("").to_string();
        text.template.replace_range(l..r, "");
        text.text = text.template.clone();
    } else {
        editor.clipboard = text.text.get(l..r).unwrap_or("").to_string();
        text.text.replace_range(l..r, "");
    }

    text.caret = l;
    text.clear_selection();
    dirty.mark_texts();

    true
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
