use crate::renderer::ui_text_rendering::{Anchor, anchor_to};
use crate::ui::input::{Input, Mouse};
use crate::ui::menu::Menu;
use crate::ui::selections::SelectionManager;
use crate::ui::ui_edit_manager::{TextEditCommand, UiEditManager};
use crate::ui::ui_touch_manager::{EditorTouchExtension, ElementRef};
use crate::ui::vertex::{ElementKind, LayerDirty, UiButtonText, UiElement};
use std::collections::HashMap;
use std::ops::Range;
use unicode_segmentation::UnicodeSegmentation;
use winit::keyboard::NamedKey;

#[derive(Clone, Copy)]
pub struct MouseSnapshot {
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

// TEXT EDITING

/// Handle text editing with undo support
pub fn handle_text_editing(
    selection: &mut SelectionManager,
    editor: &mut EditorTouchExtension,
    menus: &mut HashMap<String, Menu>,
    edit_manager: &mut UiEditManager,
    input: &mut Input,
    mouse_snapshot: MouseSnapshot
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
    dirty: &mut LayerDirty
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

fn handle_backspace(
    input: &mut Input,
    text: &mut UiButtonText,
    dirty: &mut LayerDirty,
) -> bool {
    if !input.action_repeat("Backspace") {
        return false;
    }

    let is_template_mode = !text.input_box;

    if text.has_selection {
        let (l, r) = text.selection_range();

        let byte_start = caret_to_byte(&text.text, l);
        let byte_end = caret_to_byte(&text.text, r);

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

    if text.caret > 0 {
        let byte_start = caret_to_byte(&text.text, text.caret - 1);
        let byte_end = caret_to_byte(&text.text, text.caret);

        if is_template_mode {
            text.template.replace_range(byte_start..byte_end, "");
            text.text = text.template.clone();
        } else {
            text.text.replace_range(byte_start..byte_end, "");
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
    let has_text = input
        .text_input
        .iter()
        .any(|s| s.chars().any(|c| !c.is_control()));

    let enter = input.named_just_pressed(NamedKey::Enter);
    let tab = input.named_just_pressed(NamedKey::Tab);

    if !has_text && !enter && !tab {
        return false;
    }

    // if !input.repeat("char_repeat", has_text)
    //     && !enter
    //     && !tab
    // {
    //     return false;
    // }

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

fn insert_characters(
    text: &mut UiButtonText,
    input: &mut Input,
    is_template_mode: bool,
) {
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

        text.caret += tab.graphemes(true).count();
    }

    for s in &input.text_input {
        let filtered: String = s
            .chars()
            .filter(|c| !c.is_control())
            .collect();

        if filtered.is_empty() {
            continue;
        }

        if is_template_mode {
            let bi = caret_to_byte(&text.template, text.caret);
            text.template.insert_str(bi, &filtered);
            text.text = text.template.clone();
        } else {
            let bi = caret_to_byte(&text.text, text.caret);
            text.text.insert_str(bi, &filtered);
        }

        text.caret += filtered.graphemes(true).count();
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

    let grapheme_count = text.text.graphemes(true).count();
    if input.action_repeat("Move Cursor Right") && text.caret < grapheme_count {
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

pub fn text_top_left(text: &UiButtonText) -> (f32, f32, f32, f32) {
    let (width, height) = text
        .cache
        .as_ref()
        .map(|c| (c.width.max(1.0), c.height.max(1.0)))
        .unwrap_or((text.width.max(1.0), text.height.max(1.0)));

    let top_left = anchor_to(text.anchor.unwrap_or_default(), [text.x, text.y], width, height);
    (top_left[0], top_left[1], width, height)
}

pub fn line_start_grapheme_index(text: &str, line_i: usize) -> usize {
    text.split('\n')
        .take(line_i)
        .map(|line| line.graphemes(true).count() + 1)
        .sum()
}

fn navigate_vertical(text: &UiButtonText, up: bool) -> Option<usize> {
    let (caret_x, caret_y) = get_caret_position(text);
    let (_, _, _, _) = text_top_left(text);

    let mut lines: Vec<(usize, f32, f32)> = Vec::new(); // (line_i, absolute_top, line_height)

    let (text_left, text_top, _, _) = text_top_left(text);
    for run in text.buffer.layout_runs() {
        lines.push((run.line_i, text_top + run.line_top, run.line_height));
    }

    if lines.is_empty() {
        return None;
    }

    let current_idx = lines
        .iter()
        .enumerate()
        .find(|(_, (_, top, height))| caret_y >= *top && caret_y <= *top + *height)
        .map(|(i, _)| i)
        .or_else(|| {
            lines
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    let a_center = a.1.1 + a.1.2 * 0.5;
                    let b_center = b.1.1 + b.1.2 * 0.5;
                    (caret_y - a_center)
                        .abs()
                        .partial_cmp(&(caret_y - b_center).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
        })?;

    let target_idx = if up {
        current_idx.checked_sub(1)?
    } else {
        let next = current_idx + 1;
        if next >= lines.len() {
            return None;
        }
        next
    };

    let target_line_top = lines[target_idx].1;
    find_caret_at_x_on_line(text, caret_x, target_line_top)
}

pub fn get_caret_position(t: &UiButtonText) -> (f32, f32) {
    let (text_left, text_top, _, _) = text_top_left(t);
    let caret = t.caret;

    let mut last_pos = (text_left, text_top);

    for run in t.buffer.layout_runs() {
        let line_start = line_start_grapheme_index(&t.text, run.line_i);
        let line_grapheme_count = run.text.graphemes(true).count();
        let line_end = line_start + line_grapheme_count;
        let line_top = text_top + run.line_top;

        let mut caret_x = run.line_w;

        if caret <= line_end {
            let caret_in_line = caret.saturating_sub(line_start).min(line_grapheme_count);

            if run.glyphs.is_empty() {
                caret_x = 0.0;
            } else {
                for glyph in run.glyphs {
                    let cluster = &run.text[glyph.start..glyph.end];
                    let cluster_start = run.text[..glyph.start].graphemes(true).count();
                    let cluster_len = cluster.graphemes(true).count();
                    let cluster_end = cluster_start + cluster_len;

                    if caret_in_line <= cluster_start {
                        caret_x = glyph.x;
                        break;
                    }

                    if caret_in_line < cluster_end {
                        let offset = caret_in_line - cluster_start;
                        caret_x = glyph.x
                            + glyph.w * (offset as f32 / cluster_len.max(1) as f32);
                        break;
                    }

                    caret_x = glyph.x + glyph.w;
                }
            }

            return (text_left + caret_x, line_top);
        }

        last_pos = (text_left + run.line_w, line_top);
    }

    last_pos
}

fn find_caret_at_x_on_line(
    text: &UiButtonText,
    target_x: f32,
    line_y: f32,
) -> Option<usize> {
    let (text_left, text_top, _, _) = text_top_left(text);
    let epsilon = 0.5f32;

    for run in text.buffer.layout_runs() {
        let abs_top = text_top + run.line_top;

        if (abs_top - line_y).abs() > epsilon {
            continue;
        }

        let line_start = line_start_grapheme_index(&text.text, run.line_i);
        let line_len = run.text.graphemes(true).count();

        if run.glyphs.is_empty() {
            return Some(line_start);
        }

        let mut best_idx = line_start;
        let mut best_dist = f32::MAX;

        for glyph in run.glyphs {
            let cluster = &run.text[glyph.start..glyph.end];
            let graphemes: Vec<_> = cluster.grapheme_indices(true).collect();

            if graphemes.is_empty() {
                continue;
            }

            let cluster_start = run.text[..glyph.start].graphemes(true).count();
            let cluster_len = graphemes.len();

            for i in 0..=cluster_len {
                let x = text_left
                    + glyph.x
                    + glyph.w * (i as f32 / cluster_len as f32);

                let dist = (target_x - x).abs();

                if dist < best_dist {
                    best_dist = dist;
                    best_idx = line_start + cluster_start + i;
                }
            }
        }

        return Some(best_idx.min(line_start + line_len));
    }

    None
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
    let (text_left, text_top, _, _) = text_top_left(text);

    let mut best_line_top: Option<f32> = None;
    let mut best_dist = f32::MAX;

    for run in text.buffer.layout_runs() {
        let line_top = text_top + run.line_top;
        let line_bottom = line_top + run.line_height;

        if my >= line_top && my <= line_bottom {
            best_line_top = Some(line_top);
            break;
        }

        let center_y = line_top + run.line_height * 0.5;
        let dist = (my - center_y).abs();
        if dist < best_dist {
            best_dist = dist;
            best_line_top = Some(line_top);
        }
    }

    let Some(line_top) = best_line_top else {
        return 0;
    };

    find_caret_at_x_on_line(text, mx, line_top).unwrap_or(0)
}

fn caret_to_byte(s: &str, caret: usize) -> usize {
    s.grapheme_indices(true)
        .nth(caret)
        .map(|(index, _)| index)
        .unwrap_or(s.len())
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