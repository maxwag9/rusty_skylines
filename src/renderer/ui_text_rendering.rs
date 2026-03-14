use crate::renderer::ui::UiRenderer;
use crate::resources::Time;
use crate::ui::vertex::{UiButtonText, UiVertexText};
use serde::{Deserialize, Serialize};
use wgpu_text::glyph_brush::ab_glyph::Font;

#[derive(Deserialize, Serialize, Clone, Copy, Debug)]
pub enum Anchor {
    TopLeft,
    Center,
    CenterLeft,
}

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
) {
    text_vertices.extend_from_slice(&[
        UiVertexText {
            pos: [xa, ya],
            color: col,
        },
        UiVertexText {
            pos: [xb, ya],
            color: col,
        },
        UiVertexText {
            pos: [xb, yb],
            color: col,
        },
        UiVertexText {
            pos: [xa, ya],
            color: col,
        },
        UiVertexText {
            pos: [xb, yb],
            color: col,
        },
        UiVertexText {
            pos: [xa, yb],
            color: col,
        },
    ]);
}

pub fn render_selection(t: &UiButtonText, text_vertices: &mut Vec<UiVertexText>) {
    if !t.has_selection || t.glyph_bounds.is_empty() {
        return;
    }

    let (l, r) = if t.sel_start < t.sel_end {
        (t.sel_start, t.sel_end)
    } else {
        (t.sel_end, t.sel_start)
    };

    if l >= r {
        return;
    }

    let col = [0.3, 0.5, 1.0, 0.35];

    let mut current_line_start = l;
    let mut i = l;

    while i < r && i < t.glyph_bounds.len() {
        let current_y = t.glyph_bounds[i].min.y;

        let mut line_end = i;
        while line_end < r
            && line_end < t.glyph_bounds.len()
            && t.glyph_bounds[line_end].min.y == current_y
        {
            line_end += 1;
        }

        let x_start = t.glyph_bounds[current_line_start].min.x;
        let x_end = if line_end > 0 && line_end - 1 < t.glyph_bounds.len() {
            t.glyph_bounds[line_end - 1].max.x
        } else {
            x_start
        };

        let y0 = t.glyph_bounds[current_line_start].min.y;
        let y1 = t.glyph_bounds[current_line_start].max.y;

        push_quad(text_vertices, x_start, y0, x_end, y1, col);

        current_line_start = line_end;
        i = line_end;
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
) {
    let x0 = min_x - pad;
    let y0 = min_y - pad;
    let x1 = max_x + pad;
    let y1 = max_y + pad;

    let base_alpha = if being_hovered { 0.30 } else { 0.01 };
    let col = [0.9, 0.9, 1.0, base_alpha];
    let t = 1.5;

    push_quad(text_vertices, x0, y0, x1, y0 + t, col);
    push_quad(text_vertices, x0, y1 - t, x1, y1, col);
    push_quad(text_vertices, x0, y0, x0 + t, y1, col);
    push_quad(text_vertices, x1 - t, y0, x1, y1, col);
}

pub fn render_corner_brackets(
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    text_vertices: &mut Vec<UiVertexText>,
    being_hovered: bool,
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

    push_quad(text_vertices, x0, y0, x0 + br, y0 + thick, col);
    push_quad(text_vertices, x0, y0, x0 + thick, y0 + br, col);

    push_quad(text_vertices, x1 - br, y0, x1, y0 + thick, col);
    push_quad(text_vertices, x1 - thick, y0, x1, y0 + br, col);

    push_quad(text_vertices, x0, y1 - thick, x0 + br, y1, col);
    push_quad(text_vertices, x0, y1 - br, x0 + thick, y1, col);

    push_quad(text_vertices, x1 - br, y1 - thick, x1, y1, col);
    push_quad(text_vertices, x1 - thick, y1 - br, x1, y1, col);
}
pub fn render_editor_caret(
    ui_renderer: &UiRenderer,
    t: &UiButtonText,
    text_vertices: &mut Vec<UiVertexText>,
    time_system: &Time,
) {
    let caret_width = 2.0;
    let Some(font) = ui_renderer.brush.fonts().first() else {
        return;
    };
    let Some(px) = font.pt_to_px_scale(t.pt) else {
        return;
    };
    let (x, y, height) = if t.glyph_bounds.is_empty() {
        let pos = anchor_to(
            t.anchor.unwrap_or(Anchor::Center),
            [t.x, t.y],
            t.width,
            t.height,
        );
        (pos[0], pos[1], px.y)
    } else if t.caret < t.glyph_bounds.len() {
        let rect = &t.glyph_bounds[t.caret];
        (rect.min.x, rect.min.y, rect.max.y - rect.min.y)
    } else {
        let rect = &t.glyph_bounds[t.glyph_bounds.len() - 1];
        (rect.max.x, rect.min.y, rect.max.y - rect.min.y)
    };

    let x0 = x;
    let y0 = y;
    let x1 = x + caret_width;
    let y1 = y + height;

    let blink_t = time_system.total_time * 3.0;
    let blink_alpha = (0.5 + 0.5 * blink_t.cos()) as f32;
    let caret_alpha = blink_alpha.clamp(0.0, 1.0);

    let caret_color = [1.0, 1.0, 1.0, caret_alpha];
    push_quad(text_vertices, x0, y0, x1, y1, caret_color);

    // let debug_glyph_color = [0.8, 0.0, 1.0, 0.5];
    // for rect in t.glyph_bounds.iter() {
    //     push_quad(text_vertices, rect.min.x, rect.min.y, rect.max.x-1.0, rect.max.y-1.0, debug_glyph_color);
    // }
}

pub struct TextLayout {
    pub lines: Vec<LineLayout>,
}

pub struct LineLayout {
    pub glyph_x: Vec<f32>,
    pub y: f32,
    pub width: f32,
}

pub fn layout_text(text: &str, px: f32, advance: impl Fn(char) -> f32) -> TextLayout {
    let mut lines = Vec::new();
    let mut current = LineLayout {
        glyph_x: vec![0.0],
        y: 0.0,
        width: 0.0,
    };

    let mut x = 0.0;
    let mut y = 0.0;
    let line_height = px * 1.2;

    for c in text.chars() {
        if c == '\n' {
            current.width = x;
            lines.push(current);

            y += line_height;
            x = 0.0;

            current = LineLayout {
                glyph_x: vec![0.0],
                y,
                width: 0.0,
            };

            continue;
        }

        x += advance(c);
        current.glyph_x.push(x);
    }

    current.width = x;
    lines.push(current);

    TextLayout { lines }
}
