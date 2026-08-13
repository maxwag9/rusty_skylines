use crate::renderer::ui::UiRenderer;
use crate::resources::Time;
use crate::ui::vertex::{UiButtonText, UiVertexText};
use serde::{Deserialize, Serialize};
use unicode_segmentation::UnicodeSegmentation;
use crate::ui::ui_text_editing::{get_caret_position, line_start_grapheme_index, text_top_left};

#[derive(Deserialize, Serialize, Clone, Copy, Debug, Default)]
pub enum Anchor {
    TopLeft,
    #[default]
    Center,
    CenterLeft,
}

/// Returned should be used as topleft
pub fn anchor_to(anchor: Anchor, pos: [f32; 2], w: f32, h: f32) -> [f32; 2] {
    match anchor {
        Anchor::TopLeft => pos,
        Anchor::Center => [pos[0] - w * 0.5, pos[1] - h * 0.5],
        Anchor::CenterLeft => [pos[0], pos[1] - h * 0.5],
    }
}

fn push_quad(
    text_vertices: &mut Vec<UiVertexText>,
    xa: f32,
    ya: f32,
    xb: f32,
    yb: f32,
    col: [f32; 4],
    depth: f32
) {
    text_vertices.extend_from_slice(&[
        UiVertexText {
            pos: [xa, ya],
            color: col,
            depth,
            _pad0: [0.0; 3]
        },
        UiVertexText {
            pos: [xb, ya],
            color: col,
            depth,
            _pad0: [0.0; 3]
        },
        UiVertexText {
            pos: [xb, yb],
            color: col,
            depth,
            _pad0: [0.0; 3]
        },
        UiVertexText {
            pos: [xa, ya],
            color: col,
            depth,
            _pad0: [0.0; 3]
        },
        UiVertexText {
            pos: [xb, yb],
            color: col,
            depth,
            _pad0: [0.0; 3]
        },
        UiVertexText {
            pos: [xa, yb],
            color: col,
            depth,
            _pad0: [0.0; 3]
        },
    ]);
}

pub fn render_selection(
    t: &UiButtonText,
    text_vertices: &mut Vec<UiVertexText>,
    depth: f32,
) {
    if !t.has_selection {
        return;
    }

    let (sel_start, sel_end) = t.selection_range();

    if sel_start == sel_end {
        return;
    }

    let (text_left, text_top, _, _) = text_top_left(t);

    let color = [0.3, 0.5, 1.0, 0.35];

    for run in t.buffer.layout_runs() {
        let line_start = line_start_grapheme_index(&t.text, run.line_i);
        let line_len = run.text.graphemes(true).count();
        let line_end = line_start + line_len;

        let start = sel_start.max(line_start);
        let end = sel_end.min(line_end);

        if start >= end {
            continue;
        }

        let start_local = start - line_start;
        let end_local = end - line_start;

        let mut x0 = 0.0;
        let mut x1 = run.line_w;

        if !run.glyphs.is_empty() {
            for glyph in run.glyphs {
                let cluster = &run.text[glyph.start..glyph.end];
                let cluster_start = run.text[..glyph.start].graphemes(true).count();
                let cluster_len = cluster.graphemes(true).count();
                let cluster_end = cluster_start + cluster_len;

                if start_local >= cluster_start && start_local <= cluster_end {
                    if start_local == cluster_start {
                        x0 = glyph.x;
                    } else if start_local == cluster_end {
                        x0 = glyph.x + glyph.w;
                    } else {
                        let offset = start_local - cluster_start;
                        x0 = glyph.x
                            + glyph.w * (offset as f32 / cluster_len.max(1) as f32);
                    }

                    break;
                }
            }

            for glyph in run.glyphs {
                let cluster = &run.text[glyph.start..glyph.end];
                let cluster_start = run.text[..glyph.start].graphemes(true).count();
                let cluster_len = cluster.graphemes(true).count();
                let cluster_end = cluster_start + cluster_len;

                if end_local >= cluster_start && end_local <= cluster_end {
                    if end_local == cluster_start {
                        x1 = glyph.x;
                    } else if end_local == cluster_end {
                        x1 = glyph.x + glyph.w;
                    } else {
                        let offset = end_local - cluster_start;
                        x1 = glyph.x
                            + glyph.w * (offset as f32 / cluster_len.max(1) as f32);
                    }

                    break;
                }
            }
        }

        let y0 = text_top + run.line_top;
        let y1 = y0 + run.line_height;

        push_quad(
            text_vertices,
            text_left + x0,
            y0,
            text_left + x1,
            y1,
            color,
            depth,
        );
    }
}
pub fn render_editor_outline(
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    text_vertices: &mut Vec<UiVertexText>,
    pad: f32,
    being_hovered: bool,
    depth: f32
) {
    let x0 = min_x - pad;
    let y0 = min_y - pad;
    let x1 = max_x + pad;
    let y1 = max_y + pad;

    let base_alpha = if being_hovered { 0.30 } else { 0.01 };
    let col = [0.9, 0.9, 1.0, base_alpha];
    let t = 1.5;

    push_quad(text_vertices, x0, y0, x1, y0 + t, col, depth);
    push_quad(text_vertices, x0, y1 - t, x1, y1, col, depth);
    push_quad(text_vertices, x0, y0, x0 + t, y1, col, depth);
    push_quad(text_vertices, x1 - t, y0, x1, y1, col, depth);
}

pub fn render_corner_brackets(
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    text_vertices: &mut Vec<UiVertexText>,
    being_hovered: bool,
    depth: f32
) {
    let base_len = 6.0;
    let base_pad = 4.0;
    let thick = 2.0;
    let hover_factor = if being_hovered { 1.6 } else { 1.0 };

    let br = base_len * hover_factor;
    let pad = base_pad * hover_factor;

    let x0 = min_x - pad;
    let y0 = min_y - pad;
    let x1 = max_x + pad;
    let y1 = max_y + pad;

    let col = [1.0, 0.85, 0.2, 1.0];

    push_quad(text_vertices, x0, y0, x0 + br, y0 + thick, col, depth);
    push_quad(text_vertices, x0, y0, x0 + thick, y0 + br, col, depth);

    push_quad(text_vertices, x1 - br, y0, x1, y0 + thick, col, depth);
    push_quad(text_vertices, x1 - thick, y0, x1, y0 + br, col, depth);

    push_quad(text_vertices, x0, y1 - thick, x0 + br, y1, col, depth);
    push_quad(text_vertices, x0, y1 - br, x0 + thick, y1, col, depth);

    push_quad(text_vertices, x1 - br, y1 - thick, x1, y1, col, depth);
    push_quad(text_vertices, x1 - thick, y1 - br, x1, y1, col, depth);
}
pub fn render_editor_caret(
    _ui_renderer: &UiRenderer,
    t: &UiButtonText,
    text_vertices: &mut Vec<UiVertexText>,
    time_system: &Time,
    depth: f32,
) {
    let caret_width = 2.0;

    let (x, y) = get_caret_position(t);

    let (_, text_top, _, _) = text_top_left(t);

    let mut height = t.pt;

    for run in t.buffer.layout_runs() {
        let run_top = text_top + run.line_top;

        if (run_top - y).abs() < 0.5 {
            height = run.line_height;
            break;
        }
    }


    let blink_t = time_system.total_time * 3.0;
    let caret_alpha = (0.5 + 0.5 * blink_t.cos()).clamp(0.0, 1.0) as f32;

    push_quad(
        text_vertices,
        x,
        y,
        x + caret_width,
        y + height,
        [1.0, 1.0, 1.0, caret_alpha],
        depth,
    );
}
