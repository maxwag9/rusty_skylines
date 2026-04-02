use crate::renderer::ui_text_rendering::{Anchor, anchor_to};
use crate::ui::input::{Input, Mouse};
use crate::ui::menu::Menu;
use crate::ui::selections::SelectionManager;
use crate::ui::ui_edit_manager::{TextEditCommand, UiEditManager};
use crate::ui::ui_touch_manager::{EditorTouchExtension, ElementRef};
use crate::ui::vertex::{ElementKind, LayerDirty, UiButtonText, UiElement};
use std::collections::HashMap;
use std::ops::Range;
use winit::keyboard::NamedKey;

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
    edit_manager: &mut UiEditManager,
    input: &mut Input,
    mouse_snapshot: MouseSnapshot,
) {
    for sel in &selection.selected {
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
            edit_manager.push_command(TextEditCommand {
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
}

pub fn process_text_editing_input(
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
    t: &mut UiButtonText,
) -> bool {
    let mx = mouse_snapshot.mx;
    let my = mouse_snapshot.my;
    let pos = anchor_to(
        t.anchor.unwrap_or(Anchor::Center),
        [t.x, t.y],
        t.width,
        t.height,
    );
    let x0 = pos[0];
    let y0 = pos[1];
    let x1 = x0 + t.width;
    let y1 = y0 + t.height;

    if mouse_snapshot.just_pressed && mx >= x0 && mx <= x1 && my >= y0 && my <= y1 {
        let new_caret = pick_caret(t, mx, my);
        t.caret = new_caret;
        t.sel_start = new_caret;
        t.sel_end = new_caret;
        t.has_selection = false;
        editor.dragging_text_selection = true;
        return true;
    }

    if editor.dragging_text_selection && mouse_snapshot.pressed {
        let new_pos = pick_caret(t, mx, my);
        t.sel_end = new_pos;
        t.has_selection = t.sel_end != t.sel_start;
        t.caret = new_pos;
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
            let clip = match clipboard.get_text() {
                Ok(c) => c,
                Err(e) => {
                    println!("Failed to get text from clipboard: {:#?}", e);
                    return false;
                }
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
        let byte_start = logical_to_byte(&text.char_spans, l);
        let byte_end = logical_to_byte(&text.char_spans, r);

        if is_template_mode {
            text.template.replace_range(byte_start..byte_end, "");
            text.text = text.template.clone();
        } else {
            text.text.replace_range(byte_start..byte_end, "");
        }
        text.caret = l;
        text.clear_selection();
        dirty.mark_texts();
        return true;
    }

    if text.caret > 0 && text.caret <= text.char_spans.len() {
        let span = text.char_spans[text.caret - 1].clone();
        if is_template_mode {
            text.template.replace_range(span.clone(), "");
            text.text = text.template.clone();
        } else {
            text.text.replace_range(span, "");
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
    if !input.repeat("char_repeat", !input.text_chars.is_empty())
        && !input.named_just_pressed(NamedKey::Enter)
        && !input.named_just_pressed(NamedKey::Tab)
    {
        return false;
    }

    let is_template_mode = !text.input_box; // or override mode!!

    if text.has_selection {
        delete_selection(text, is_template_mode);
    }

    insert_characters(text, input, is_template_mode);
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

fn insert_characters(text: &mut UiButtonText, input: &mut Input, is_template_mode: bool) {
    if input.named_just_pressed(NamedKey::Enter) {
        if is_template_mode {
            let bi = caret_to_byte(&text.template, text.caret);
            text.template.insert_str(bi, "\n");
            text.text = text.template.clone();
        } else {
            let bi = caret_to_byte(&text.text, text.caret);
            text.text.insert_str(bi, "\n");
        }
        text.caret += 1;
    }
    if input.named_just_pressed(NamedKey::Tab) {
        let tab = "    "; // 4 Spaces
        if is_template_mode {
            let bi = caret_to_byte(&text.template, text.caret);
            text.template.insert_str(bi, tab);
            text.text = text.template.clone();
        } else {
            let bi = caret_to_byte(&text.text, text.caret);
            text.text.insert_str(bi, tab);
        }
        text.caret += tab.len();
    }
    for s in input.text_chars.iter() {
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

    if input.action_repeat("Move Cursor Right") && text.caret < text.glyph_bounds.len() {
        text.caret += 1;
        dirty.mark_texts();
    }

    if input.action_repeat("Move Cursor Up") {
        if let Some(new_caret) = navigate_vertical(text, true) {
            text.caret = new_caret;
            dirty.mark_texts();
        }
    }

    if input.action_repeat("Move Cursor Down") {
        if let Some(new_caret) = navigate_vertical(text, false) {
            text.caret = new_caret;
            dirty.mark_texts();
        }
    }
}

fn navigate_vertical(text: &UiButtonText, up: bool) -> Option<usize> {
    if text.glyph_bounds.is_empty() {
        return None;
    }

    let (caret_x, caret_y) = get_caret_position(text);

    // Collect unique line Y values
    let mut line_ys: Vec<f32> = Vec::new();
    for rect in &text.glyph_bounds {
        let y = rect.min.y;
        if !line_ys.iter().any(|&ly| (ly - y).abs() < 0.5) {
            line_ys.push(y);
        }
    }
    line_ys.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Find current line index
    let current_line = line_ys.iter().position(|&y| (y - caret_y).abs() < 0.5)?;

    // Find target line index
    let target_line = if up {
        current_line.checked_sub(1)?
    } else {
        let next = current_line + 1;
        if next >= line_ys.len() {
            return None;
        }
        next
    };

    let target_y = line_ys[target_line];

    // Find the best caret position on target line at similar X
    find_caret_at_x_on_line(text, caret_x, target_y)
}

fn get_caret_position(t: &UiButtonText) -> (f32, f32) {
    if t.glyph_bounds.is_empty() {
        let pos = anchor_to(
            t.anchor.unwrap_or(Anchor::Center),
            [t.x, t.y],
            t.width,
            t.height,
        );
        return pos.into();
    }

    if t.caret == 0 {
        let rect = &t.glyph_bounds[0];
        return (rect.min.x, rect.min.y);
    }

    if t.caret >= t.glyph_bounds.len() {
        let rect = t.glyph_bounds.last().unwrap();
        return (rect.max.x, rect.min.y);
    }

    let rect = &t.glyph_bounds[t.caret];
    (rect.min.x, rect.min.y)
}

fn find_caret_at_x_on_line(text: &UiButtonText, target_x: f32, line_y: f32) -> Option<usize> {
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;
    let mut found = false;

    for (i, rect) in text.glyph_bounds.iter().enumerate() {
        if (rect.min.y - line_y).abs() > 0.5 {
            continue;
        }
        found = true;

        // Check left edge (caret before this glyph)
        let dist_left = (rect.min.x - target_x).abs();
        if dist_left < best_dist {
            best_dist = dist_left;
            best_idx = i;
        }

        // Check right edge (caret after this glyph)
        let dist_right = (rect.max.x - target_x).abs();
        if dist_right < best_dist {
            best_dist = dist_right;
            best_idx = i + 1;
        }
    }

    if found {
        Some(best_idx.min(text.glyph_bounds.len()))
    } else {
        None
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

fn pick_caret(text: &UiButtonText, mx: f32, my: f32) -> usize {
    if text.glyph_bounds.is_empty() {
        return 0;
    }

    let mut best_idx = 0;
    let mut best_dist = f32::MAX;

    for (i, rect) in text.glyph_bounds.iter().enumerate() {
        let line_top = rect.min.y;
        let line_bottom = rect.max.y;

        if my >= line_top && my < line_bottom {
            let glyph_center_x = (rect.min.x + rect.max.x) * 0.5;
            if mx < glyph_center_x {
                return i;
            }
            best_idx = i + 1;
        }

        let center_y = (rect.min.y + rect.max.y) * 0.5;
        let dist = (my - center_y).abs();
        if dist < best_dist {
            best_dist = dist;
            let glyph_center_x = (rect.min.x + rect.max.x) * 0.5;
            if mx < glyph_center_x {
                best_idx = i;
            } else {
                best_idx = i + 1;
            }
        }
    }

    best_idx.min(text.glyph_bounds.len())
}

fn caret_to_byte(text: &str, caret: usize) -> usize {
    text.char_indices()
        .nth(caret)
        .map(|(i, _)| i)
        .unwrap_or_else(|| text.len())
}

fn logical_to_byte(char_spans: &[Range<usize>], logical: usize) -> usize {
    if logical == 0 || char_spans.is_empty() {
        0
    } else if logical >= char_spans.len() {
        char_spans.last().map(|s| s.end).unwrap_or(0)
    } else {
        char_spans[logical].start
    }
}
