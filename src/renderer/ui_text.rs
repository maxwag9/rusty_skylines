use crate::renderer::ui::{GlyphUv, PAD, TextParams};
use crate::renderer::ui_pipelines::UiPipelines;
use crate::resources::TimeSystem;
use crate::ui::ui_editor::UiVertexText;
use crate::ui::vertex::UiButtonText;
use fontdue::Font;
use rect_packer::DensePacker;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use wgpu::*;

#[derive(Deserialize, Serialize, Clone, Copy, Debug)]
pub(crate) enum Anchor {
    TopLeft,
    Center,
    CenterLeft,
}

pub fn anchor_to_top_left(
    anchor: Anchor,
    given_pos: [f32; 2],
    natural_w: f32,
    natural_h: f32,
) -> [f32; 2] {
    match anchor {
        Anchor::TopLeft => given_pos,
        Anchor::Center => [
            given_pos[0] - natural_w * 0.5,
            given_pos[1] - natural_h * 0.5,
        ],
        Anchor::CenterLeft => [given_pos[0], given_pos[1] - natural_h * 0.5],
    }
}
#[derive(Clone)]
pub struct FontMetrics {
    pub ascent: f32,
    pub descent: f32,
    pub line_height: f32,
}

pub struct TextAtlas {
    pub tex: Texture, // R8Unorm
    pub view: TextureView,
    pub sampler: Sampler,
    pub size: (u32, u32),
    pub glyphs: HashMap<(char, u16), GlyphUv>, // (char, px_size) -> uv+metrics
    pub packed_sizes: HashSet<u16>,            // already packed font sizes
    pub font: Font,
    pub packer: DensePacker,
    pub cpu_atlas: Vec<u8>,
    pub metrics: HashMap<u16, FontMetrics>,
}
impl TextAtlas {
    pub fn new(device: &Device, size: (u32, u32)) -> Self {
        let tex = device.create_texture(&TextureDescriptor {
            label: Some("Text Atlas"),
            size: Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let font_ttf: &[u8] = include_bytes!("../../data/ui_data/ttf/JetBrainsMono-Regular.ttf");
        let font = Font::from_bytes(font_ttf, fontdue::FontSettings::default())
            .unwrap_or_else(|e| panic!("Failed to load font: {e}"));
        let packer = DensePacker::new(size.0 as i32, size.1 as i32);
        let cpu_atlas = vec![0u8; (size.0 * size.1) as usize];

        Self {
            tex,
            view,
            sampler,
            size,
            glyphs: HashMap::new(),
            packed_sizes: HashSet::new(),
            font,
            packer,
            cpu_atlas,
            metrics: Default::default(),
        }
    }

    pub fn ensure_px_size(
        &mut self,
        device: &Device,
        queue: &Queue,
        px: u16,
    ) -> Result<AtlasResult, Box<dyn std::error::Error>> {
        if self.packed_sizes.contains(&px) {
            return Ok(AtlasResult::Unchanged);
        }

        let charset: Vec<char> = (32u8..127).map(|c| c as char).collect();

        let mut target_w = self.size.0;
        let mut target_h = self.size.1;
        let mut grew = false;

        loop {
            let mut try_packer = DensePacker::new(target_w as i32, target_h as i32);
            let mut try_cpu_atlas = vec![0u8; (target_w * target_h) as usize];
            let mut try_glyphs = self.glyphs.clone();
            let mut try_metrics = self.metrics.clone();

            let mut sizes: Vec<u16> = self.packed_sizes.iter().cloned().collect();
            sizes.push(px);
            sizes.sort_unstable();
            sizes.dedup();

            let mut failed = false;

            for &size in &sizes {
                let vertical = self.font.vertical_line_metrics(size as f32);

                let metrics = if let Some(v) = vertical {
                    FontMetrics {
                        ascent: v.ascent,
                        descent: v.descent.abs(),
                        line_height: (v.ascent - v.descent + v.line_gap).ceil(),
                    }
                } else {
                    // --- glyph-based fallback ---
                    let mut min_y = f32::MAX;
                    let mut max_y = f32::MIN;

                    for &ch in &charset {
                        let (m, _) = self.font.rasterize(ch, size as f32);
                        if m.width == 0 || m.height == 0 {
                            continue;
                        }

                        let top = m.ymin as f32;
                        let bottom = (m.ymin + m.height as i32) as f32;

                        min_y = min_y.min(top);
                        max_y = max_y.max(bottom);
                    }

                    if min_y == f32::MAX {
                        // absolute last resort, should never happen
                        FontMetrics {
                            ascent: size as f32 * 0.8,
                            descent: size as f32 * 0.2,
                            line_height: size as f32,
                        }
                    } else {
                        let ascent = -min_y;
                        let descent = max_y.max(0.0);

                        FontMetrics {
                            ascent,
                            descent,
                            line_height: (ascent + descent).ceil(),
                        }
                    }
                };
                let ascent = metrics.ascent;
                try_metrics.insert(size, metrics);

                for &ch in &charset {
                    let (m, bitmap) = self.font.rasterize(ch, size as f32);

                    if m.width == 0 || m.height == 0 {
                        try_glyphs.insert(
                            (ch, size),
                            GlyphUv {
                                u0: 0.0,
                                v0: 0.0,
                                u1: 0.0,
                                v1: 0.0,
                                advance: m.advance_width,
                                width: 0.0,
                                height: 0.0,
                                bearing_x: m.xmin as f32,
                                bearing_y: 0.0,
                            },
                        );
                        continue;
                    }

                    let rect = match try_packer.pack(
                        m.width as i32 + 2 * PAD,
                        m.height as i32 + 2 * PAD,
                        false,
                    ) {
                        Some(r) => r,
                        None => {
                            failed = true;
                            break;
                        }
                    };

                    let gx = rect.x + PAD;
                    let gy = rect.y + PAD;

                    for row in 0..m.height {
                        let src = row * m.width;
                        let dst = (gy as usize + row as usize) * target_w as usize + gx as usize;
                        try_cpu_atlas[dst..dst + m.width as usize]
                            .copy_from_slice(&bitmap[src..src + m.width]);
                    }

                    try_glyphs.insert(
                        (ch, size),
                        GlyphUv {
                            u0: gx as f32 / target_w as f32,
                            v0: gy as f32 / target_h as f32,
                            u1: (gx + m.width as i32) as f32 / target_w as f32,
                            v1: (gy + m.height as i32) as f32 / target_h as f32,

                            advance: m.advance_width,
                            width: m.width as f32,
                            height: m.height as f32,

                            bearing_x: m.xmin as f32,
                            bearing_y: (m.ymin + m.height as i32) as f32,
                        },
                    );
                }

                if failed {
                    break;
                }
            }

            if failed {
                target_w *= 2;
                target_h *= 2;
                grew = true;
                continue;
            }

            self.glyphs = try_glyphs;
            self.metrics = try_metrics;
            self.packed_sizes.insert(px);

            if grew {
                self.size = (target_w, target_h);
                self.tex = device.create_texture(&TextureDescriptor {
                    label: Some("Text Atlas"),
                    size: Extent3d {
                        width: target_w,
                        height: target_h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::R8Unorm,
                    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                self.view = self.tex.create_view(&TextureViewDescriptor::default());
            }

            let padded_bpr = ((target_w as usize + 255) / 256) * 256;
            let mut padded = vec![0u8; padded_bpr * target_h as usize];

            for y in 0..target_h as usize {
                padded[y * padded_bpr..y * padded_bpr + target_w as usize].copy_from_slice(
                    &try_cpu_atlas[y * target_w as usize..(y + 1) * target_w as usize],
                );
            }
            #[cfg(debug_assertions)]
            {
                let non_zero_padded = padded.iter().filter(|&&b| b != 0).count();
                println!(
                    "DEBUG: padded non-zero bytes = {} padded_bpr = {}",
                    non_zero_padded, padded_bpr
                );
            }

            queue.write_texture(
                TexelCopyTextureInfo {
                    texture: &self.tex,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                &padded,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr as u32),
                    rows_per_image: Some(target_h),
                },
                Extent3d {
                    width: target_w,
                    height: target_h,
                    depth_or_array_layers: 1,
                },
            );

            self.cpu_atlas = try_cpu_atlas;
            self.packer = try_packer;

            return Ok(if grew {
                AtlasResult::Recreated
            } else {
                AtlasResult::Unchanged
            });
        }
    }
}

pub enum AtlasResult {
    Unchanged,
    Recreated,
}
fn push_quad(
    text_vertices: &mut Vec<UiVertexText>,
    xa: f32,
    ya: f32,
    xb: f32,
    yb: f32,
    uv: [f32; 2],
    col: [f32; 4],
) {
    text_vertices.extend_from_slice(&[
        UiVertexText {
            pos: [xa, ya],
            uv,
            color: col,
        },
        UiVertexText {
            pos: [xb, ya],
            uv,
            color: col,
        },
        UiVertexText {
            pos: [xb, yb],
            uv,
            color: col,
        },
        UiVertexText {
            pos: [xa, ya],
            uv,
            color: col,
        },
        UiVertexText {
            pos: [xb, yb],
            uv,
            color: col,
        },
        UiVertexText {
            pos: [xa, yb],
            uv,
            color: col,
        },
    ]);
}

pub fn render_selection(
    orig: &mut UiButtonText,
    min_y: f32,
    max_y: f32,
    text_vertices: &mut Vec<UiVertexText>,
) {
    if orig.has_selection {
        let (l, r) = if orig.sel_start < orig.sel_end {
            (orig.sel_start, orig.sel_end)
        } else {
            (orig.sel_end, orig.sel_start)
        };

        if l < r && l < orig.glyph_bounds.len() {
            let x_start = orig.glyph_bounds[l].0;
            let x_end = if r == 0 {
                x_start
            } else if r - 1 < orig.glyph_bounds.len() {
                orig.glyph_bounds[r - 1].1
            } else {
                orig.glyph_bounds.last().unwrap().1
            };

            let y0 = min_y - 2.0;
            let y1 = max_y + 2.0;

            let col = [0.3, 0.5, 1.0, 0.35];
            let uv = [-1.0, -1.0];

            text_vertices.extend_from_slice(&[
                UiVertexText {
                    pos: [x_start, y0],
                    uv,
                    color: col,
                },
                UiVertexText {
                    pos: [x_end, y0],
                    uv,
                    color: col,
                },
                UiVertexText {
                    pos: [x_end, y1],
                    uv,
                    color: col,
                },
                UiVertexText {
                    pos: [x_start, y0],
                    uv,
                    color: col,
                },
                UiVertexText {
                    pos: [x_end, y1],
                    uv,
                    color: col,
                },
                UiVertexText {
                    pos: [x_start, y1],
                    uv,
                    color: col,
                },
            ]);
        }
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
    let uv = [-1.0, -1.0];
    let t = 1.5;

    push_quad(text_vertices, x0, y0, x1, y0 + t, uv, col);
    push_quad(text_vertices, x0, y1 - t, x1, y1, uv, col);
    push_quad(text_vertices, x0, y0, x0 + t, y1, uv, col);
    push_quad(text_vertices, x1 - t, y0, x1, y1, uv, col);
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
    let uv = [-1.0, -1.0];

    push_quad(text_vertices, x0, y0, x0 + br, y0 + thick, uv, col);
    push_quad(text_vertices, x0, y0, x0 + thick, y0 + br, uv, col);

    push_quad(text_vertices, x1 - br, y0, x1, y0 + thick, uv, col);
    push_quad(text_vertices, x1 - thick, y0, x1, y0 + br, uv, col);

    push_quad(text_vertices, x0, y1 - thick, x0 + br, y1, uv, col);
    push_quad(text_vertices, x0, y1 - br, x0 + thick, y1, uv, col);

    push_quad(text_vertices, x1 - br, y1 - thick, x1, y1, uv, col);
    push_quad(text_vertices, x1 - thick, y1 - br, x1, y1, uv, col);
}
pub fn render_editor_caret(
    tp: &TextParams,
    caret_x: f32,
    text_vertices: &mut Vec<UiVertexText>,
    metrics: &FontMetrics,
    time_system: &TimeSystem,
) {
    let caret_width = 2.0;
    let caret_offset_y = 4.0;

    let x0 = caret_x;
    let y0 = tp.pos[1] + caret_offset_y;
    let x1 = caret_x + caret_width;
    // use per-size line height
    let y1 = tp.pos[1] + metrics.line_height + caret_offset_y;

    let t = time_system.total_time * 3.0;
    let blink_alpha = (0.5 + 0.5 * t.cos()) as f32;
    let caret_alpha = blink_alpha.clamp(0.0, 1.0);

    let caret_color = [1.0, 1.0, 1.0, caret_alpha];
    let uv = [-1.0, -1.0];

    text_vertices.extend_from_slice(&[
        UiVertexText {
            pos: [x0, y0],
            uv,
            color: caret_color,
        },
        UiVertexText {
            pos: [x1, y0],
            uv,
            color: caret_color,
        },
        UiVertexText {
            pos: [x1, y1],
            uv,
            color: caret_color,
        },
        UiVertexText {
            pos: [x0, y0],
            uv,
            color: caret_color,
        },
        UiVertexText {
            pos: [x1, y1],
            uv,
            color: caret_color,
        },
        UiVertexText {
            pos: [x0, y1],
            uv,
            color: caret_color,
        },
    ]);
}

// Placed outside the impl or as a private associated function
pub fn glyphs_to_vertices(
    pipelines: &UiPipelines,
    text_vertices: &mut Vec<UiVertexText>,
    tp: &mut TextParams,
    char_i: &mut usize,
    caret_index: usize,
    baseline_y: f32,
    pen_x: &mut f32,
    caret_x: &mut f32,
    original_text: &mut Option<&mut UiButtonText>,
    min_x: &mut f32,
    min_y: &mut f32,
    max_x: &mut f32,
    max_y: &mut f32,
) {
    // CRITICAL: We take the text out of the struct temporarily to avoid
    // "cannot borrow `*tp` as mutable because it is also borrowed as immutable"
    let text_content = std::mem::take(&mut tp.text);

    for ch in text_content.chars() {
        if let Some(g) = pipelines.text_atlas.glyphs.get(&(ch, tp.px)) {
            // ---- Handle zero-sized glyphs (spaces, control chars) ----
            if g.width == 0.0 || g.height == 0.0 {
                // If caret is exactly here
                if *char_i == caret_index {
                    *caret_x = *pen_x;
                }

                let x0 = (*pen_x + g.bearing_x).round();
                let x1 = x0 + g.width;

                tp.glyph_bounds.push((x0, x1));

                if let Some(orig) = original_text {
                    orig.glyph_bounds.push((x0, x1));
                }

                *pen_x += g.advance;
                *char_i += 1;
                continue; // Use continue, not return!
            }

            // ---- Handle Visible Glyphs ----
            let x0 = (*pen_x + g.bearing_x).round();
            let y0 = (baseline_y - g.bearing_y).round();
            let x1 = x0 + g.width;
            let y1 = y0 + g.height;

            // Caret check
            if *char_i == caret_index {
                *caret_x = x0;
            }

            // Bounds
            if let Some(orig) = original_text {
                orig.glyph_bounds.push((x0, x1));
            }
            tp.glyph_bounds.push((x0, x1));

            // Overall box
            *min_x = min_x.min(x0);
            *min_y = min_y.min(y0);
            *max_x = max_x.max(x1);
            *max_y = max_y.max(y1);

            // Sanity check
            if g.u0 == g.u1 || g.v0 == g.v1 || !(0.0..=1.0).contains(&g.u0) {
                *pen_x += g.advance;
                *char_i += 1;
                continue; // Use continue!
            }

            // Vertices
            text_vertices.extend_from_slice(&[
                UiVertexText {
                    pos: [x0, y0],
                    uv: [g.u0, g.v0],
                    color: tp.color,
                },
                UiVertexText {
                    pos: [x1, y0],
                    uv: [g.u1, g.v0],
                    color: tp.color,
                },
                UiVertexText {
                    pos: [x1, y1],
                    uv: [g.u1, g.v1],
                    color: tp.color,
                },
                UiVertexText {
                    pos: [x0, y0],
                    uv: [g.u0, g.v0],
                    color: tp.color,
                },
                UiVertexText {
                    pos: [x1, y1],
                    uv: [g.u1, g.v1],
                    color: tp.color,
                },
                UiVertexText {
                    pos: [x0, y1],
                    uv: [g.u0, g.v1],
                    color: tp.color,
                },
            ]);

            *pen_x += g.advance;
        }
        *char_i += 1;
    }

    // Put the text back into the struct
    tp.text = text_content;
}
