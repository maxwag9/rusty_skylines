use crate::ui::ui_runtime::UiRuntimes;
use phf::phf_map;
use rand::RngExt;
use sluggrs_skylines::cosmic_text::{Attrs, Font, Style, UnderlineStyle, Weight};
use sluggrs_skylines::{Buffer, Color, FontSystem};
use std::collections::HashMap;
use std::ops::Range;
use unicode_segmentation::UnicodeSegmentation;

static COLOR_PALETTE: phf::Map<&'static str, [u8; 4]> = phf_map! {
    "BLACK" => [0, 0, 0, 255],
    "DARK_BLUE" => [0, 0, 170, 255],
    "DARK_GREEN" => [0, 170, 0, 255],
    "DARK_AQUA" => [0, 170, 170, 255],
    "DARK_RED" => [170, 0, 0, 255],
    "DARK_PURPLE" => [170, 0, 170, 255],
    "GOLD" => [255, 170, 0, 255],
    "GRAY" => [170, 170, 170, 255],
    "DARK_GRAY" => [85, 85, 85, 255],
    "BLUE" => [85, 85, 255, 255],
    "GREEN" => [85, 255, 85, 255],
    "AQUA" => [85, 255, 255, 255],
    "RED" => [255, 85, 85, 255],
    "LIGHT_PURPLE" => [255, 85, 255, 255],
    "YELLOW" => [255, 255, 85, 255],
    "WHITE" => [255, 255, 255, 255],

    "ORANGE" => [255, 165, 0, 255],
    "PINK" => [255, 105, 180, 255],
    "LIME" => [50, 205, 50, 255],
    "TURQUOISE" => [64, 224, 208, 255],
    "CYAN" => [0, 255, 255, 255],
    "MAGENTA" => [255, 0, 255, 255],
    "PURPLE" => [128, 0, 128, 255],
    "VIOLET" => [138, 43, 226, 255],
    "INDIGO" => [75, 0, 130, 255],
    "NAVY" => [0, 0, 128, 255],
    "TEAL" => [0, 128, 128, 255],
    "OLIVE" => [128, 128, 0, 255],
    "MAROON" => [128, 0, 0, 255],
    "CORAL" => [255, 127, 80, 255],
    "SALMON" => [250, 128, 114, 255],
    "BEIGE" => [245, 245, 220, 255],
    "BROWN" => [139, 69, 19, 255],
    "CHOCOLATE" => [210, 105, 30, 255],
    "CRIMSON" => [220, 20, 60, 255],
    "KHAKI" => [240, 230, 140, 255],
    "MINT" => [152, 255, 152, 255],
    "SKY_BLUE" => [135, 206, 235, 255],
    "ROYAL_BLUE" => [65, 105, 225, 255],
    "FOREST_GREEN" => [34, 139, 34, 255],
    "SEA_GREEN" => [46, 139, 87, 255],
    "SLATE_GRAY" => [112, 128, 144, 255],
    "SILVER" => [192, 192, 192, 255],
    "CHARCOAL" => [54, 69, 79, 255],
    "TRANSPARENT" => [255, 255, 255, 0],
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FormatStyle {
    pub color: Option<[u8; 4]>,
    pub bold: bool,
    pub italic: bool,
    pub strikethrough: bool,
    pub underline: bool,
    pub obfuscated: bool,
}

#[derive(Clone, Debug)]
pub struct FormatSpan {
    pub range: Range<usize>,
    pub style: FormatStyle,
}

#[derive(Clone, Debug)]
pub struct FormattedText {
    pub text: String,
    pub spans: Vec<FormatSpan>,
}

#[derive(Clone, Copy, Debug)]
enum FormatCommand {
    SetColor([u8; 4]),
    ResetColor,
    ResetAll,
    SetBold,
    ResetBold,
    SetItalic,
    ResetItalic,
    SetStrikethrough,
    ResetStrikethrough,
    SetUnderline,
    ResetUnderline,
    SetObfuscated,
    ResetObfuscated,
}

fn parse_hex_color(value: &str) -> Option<[u8; 4]> {
    let (rgb, alpha) = match value.split_once('/') {
        Some((rgb, alpha)) => {
            if alpha.len() != 2
                || !alpha.is_ascii()
                || !alpha.bytes().all(|b| b.is_ascii_hexdigit())
            {
                return None;
            }

            (rgb, u8::from_str_radix(alpha, 16).ok()?)
        }
        None => (value, 255),
    };

    if rgb.len() != 6 || !rgb.is_ascii() || !rgb.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }

    Some([
        u8::from_str_radix(&rgb[0..2], 16).ok()?,
        u8::from_str_radix(&rgb[2..4], 16).ok()?,
        u8::from_str_radix(&rgb[4..6], 16).ok()?,
        alpha,
    ])
}

fn parse_single_command(token: &str) -> Option<FormatCommand> {
    let token = token.trim();

    match token {
        "r" => return Some(FormatCommand::ResetAll),
        "rc" | "r#" => return Some(FormatCommand::ResetColor),
        "b" => return Some(FormatCommand::SetBold),
        "rb" => return Some(FormatCommand::ResetBold),
        "i" => return Some(FormatCommand::SetItalic),
        "ri" => return Some(FormatCommand::ResetItalic),
        "s" => return Some(FormatCommand::SetStrikethrough),
        "rs" => return Some(FormatCommand::ResetStrikethrough),
        "u" => return Some(FormatCommand::SetUnderline),
        "ru" => return Some(FormatCommand::ResetUnderline),
        "o" => return Some(FormatCommand::SetObfuscated),
        "ro" => return Some(FormatCommand::ResetObfuscated),
        _ => {}
    }

    if let Some(name) = token.strip_prefix('c') {
        if name.is_empty() {
            return None;
        }

        let name = name.to_ascii_uppercase();
        return COLOR_PALETTE
            .get(name.as_str())
            .copied()
            .map(FormatCommand::SetColor);
    }

    if let Some(hex) = token.strip_prefix('#') {
        return parse_hex_color(hex).map(FormatCommand::SetColor);
    }

    None
}

fn parse_commands(token: &str) -> Option<Vec<FormatCommand>> {
    let token = token.trim();

    if let Some(inner) = token
        .strip_prefix('[')
        .and_then(|inner| inner.strip_suffix(']'))
    {
        if inner.trim().is_empty() {
            return None;
        }

        inner
            .split(',')
            .map(parse_single_command)
            .collect::<Option<Vec<_>>>()
    } else {
        Some(vec![parse_single_command(token)?])
    }
}

fn apply_command(style: &mut FormatStyle, command: FormatCommand) {
    match command {
        FormatCommand::SetColor(color) => style.color = Some(color),
        FormatCommand::ResetColor => style.color = None,
        FormatCommand::ResetAll => *style = FormatStyle::default(),

        FormatCommand::SetBold => style.bold = true,
        FormatCommand::ResetBold => style.bold = false,

        FormatCommand::SetItalic => style.italic = true,
        FormatCommand::ResetItalic => style.italic = false,

        FormatCommand::SetStrikethrough => style.strikethrough = true,
        FormatCommand::ResetStrikethrough => style.strikethrough = false,

        FormatCommand::SetUnderline => style.underline = true,
        FormatCommand::ResetUnderline => style.underline = false,

        FormatCommand::SetObfuscated => style.obfuscated = true,
        FormatCommand::ResetObfuscated => style.obfuscated = false,
    }
}

fn obfuscate_char(ch: char, rng: &mut impl RngExt) -> char {
    const UPPER: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const LOWER: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    const DIGITS: &[u8] = b"0123456789";
    const PUNCTUATION: &[u8] = b"!\"#$%&'()*+,-./:;<=>?@[]^_`{|}~";

    let pool = if ch.is_ascii_uppercase() {
        UPPER
    } else if ch.is_ascii_lowercase() {
        LOWER
    } else if ch.is_ascii_digit() {
        DIGITS
    } else if ch.is_ascii_punctuation() {
        PUNCTUATION
    } else {
        return ch;
    };

    char::from(pool[rng.random_range(0..pool.len())])
}

fn push_span(spans: &mut Vec<FormatSpan>, start: usize, end: usize, style: FormatStyle) {
    if start < end {
        spans.push(FormatSpan {
            range: start..end,
            style,
        });
    }
}

pub fn format_text(input: &str) -> FormattedText {
    let mut text = String::with_capacity(input.len());
    let mut spans = Vec::new();
    let mut style = FormatStyle::default();
    let mut span_start = 0;
    let mut i = 0;
    //let rng = rand::rng();

    while i < input.len() {
        let remaining = &input[i..];

        if remaining.starts_with("§§") {
            text.push_str("§");
            i += "§§".len();
            continue;
        }

        if remaining.starts_with('§') {
            let token_start = i + '§'.len_utf8();

            if let Some(relative_end) = input[token_start..].find('§') {
                let token_end = token_start + relative_end;
                let token = &input[token_start..token_end];
                let end = token_end + '§'.len_utf8();

                if let Some(commands) = parse_commands(token) {
                    push_span(&mut spans, span_start, text.len(), style);

                    for command in commands {
                        apply_command(&mut style, command);
                    }

                    span_start = text.len();
                    i = end;
                    continue;
                }

                text.push_str(&input[i..end]);
                i = end;
                continue;
            }

            text.push_str("§");
            i += '§'.len_utf8();
            continue;
        }

        let ch = remaining.chars().next().unwrap();
        text.push_str(&ch.to_string());
        i += ch.len_utf8();
    }

    push_span(&mut spans, span_start, text.len(), style);

    FormattedText { text, spans }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn style_at<'a>(formatted: &'a FormattedText, needle: &str) -> &'a FormatStyle {
        let start = formatted.text.find(needle).unwrap();

        &formatted
            .spans
            .iter()
            .find(|span| span.range.start <= start && start < span.range.end)
            .unwrap()
            .style
    }

    #[test]
    fn colors_and_styles_remain_independent() {
        let formatted = format_text("§b§Bold §cRED§red§rc§ still bold §rb§plain");

        assert_eq!(formatted.text, "Bold red still bold plain");
        assert!(style_at(&formatted, "Bold").bold);

        assert_eq!(style_at(&formatted, "red").color, Some([255, 85, 85, 255]));
        assert!(style_at(&formatted, "red").bold);

        assert!(style_at(&formatted, "still").bold);
        assert_eq!(style_at(&formatted, "still").color, None);

        assert!(!style_at(&formatted, "plain").bold);
    }

    #[test]
    fn hexadecimal_alpha_works() {
        let formatted = format_text("§#AAFF00/F0§Hello");

        assert_eq!(
            style_at(&formatted, "Hello").color,
            Some([170, 255, 0, 240])
        );
    }

    #[test]
    fn arrays_apply_multiple_commands() {
        let formatted = format_text("§[r, u]§under§ru§normal");

        assert_eq!(formatted.text, "undernormal");
        assert!(style_at(&formatted, "under").underline);
        assert!(!style_at(&formatted, "normal").underline);
    }

    #[test]
    fn escaped_and_unknown_codes_remain_visible() {
        let escaped = format_text("Literal §§ and §b§bold");
        assert_eq!(escaped.text, "Literal § and bold");

        let unknown = format_text("§cNOT_A_COLOR§");
        assert_eq!(unknown.text, "§cNOT_A_COLOR§");
    }

    #[test]
    fn color_resets_are_aliases() {
        let a = format_text("§#FF0000§A§rc§B");
        let b = format_text("§#FF0000§A§r#§B");

        assert_eq!(a.text, "AB");
        assert_eq!(b.text, "AB");
        assert_eq!(style_at(&a, "B").color, None);
        assert_eq!(style_at(&b, "B").color, None);
    }
}

#[derive(Debug, Default)]
pub struct ObfuscationFontCache {
    pub candidates_by_width: HashMap<u32, Vec<char>>,
    pub width_by_char: HashMap<char, u32>,
}

fn obfuscation_width_key(advance: f32) -> Option<u32> {
    if !advance.is_finite() || advance <= 0.0 {
        return None;
    }

    Some(advance.round() as u32)
}

pub fn build_obfuscation_font_cache(font: &Font) -> ObfuscationFontCache {
    let mut cache = ObfuscationFontCache::default();

    let swash_font = font.as_swash();
    let charmap = swash_font.charmap();
    let font_metrics = swash_font.metrics(&[]);
    let units_per_em = f32::from(font_metrics.units_per_em.max(1));
    let glyph_metrics = swash_font.glyph_metrics(&[]).scale(units_per_em);

    let codepoints: Vec<u32> = if font.unicode_codepoints().is_empty() {
        (0x21..=0x7E).collect()
    } else {
        font.unicode_codepoints().to_vec()
    };

    for codepoint in codepoints {
        let Some(ch) = char::from_u32(codepoint) else {
            continue;
        };

        if ch.is_control() || ch.is_whitespace() {
            continue;
        }

        let glyph_id = charmap.map(ch);

        if glyph_id == 0 {
            continue;
        }

        let advance = glyph_metrics.advance_width(glyph_id);

        let Some(width) = obfuscation_width_key(advance) else {
            continue;
        };

        cache.width_by_char.insert(ch, width);
        cache.candidates_by_width.entry(width).or_default().push(ch);
    }

    cache
}

fn is_obfuscated_at(formatted: &FormattedText, byte_index: usize) -> bool {
    formatted
        .spans
        .iter()
        .find(|span| span.range.contains(&byte_index))
        .map_or(false, |span| span.style.obfuscated)
}

pub fn replace_formatted_characters(
    formatted: &FormattedText,
    replacements: &HashMap<usize, char>,
) -> FormattedText {
    let mut text = String::with_capacity(formatted.text.len());
    let mut spans = Vec::with_capacity(formatted.spans.len());

    for span in &formatted.spans {
        let start = text.len();
        let original = &formatted.text[span.range.clone()];

        for (offset, ch) in original.char_indices() {
            text.push(
                replacements
                    .get(&(span.range.start + offset))
                    .copied()
                    .unwrap_or(ch),
            );
        }

        let end = text.len();

        if start < end {
            spans.push(FormatSpan {
                range: start..end,
                style: span.style,
            });
        }
    }

    FormattedText { text, spans }
}

pub fn build_rich_spans<'a>(
    formatted: &'a FormattedText,
    attrs: &Attrs<'a>,
) -> Vec<(&'a str, Attrs<'a>)> {
    let mut rich_spans = Vec::with_capacity(formatted.spans.len());

    for span in &formatted.spans {
        append_emoji_font_spans(
            &formatted.text[span.range.clone()],
            span.style,
            attrs,
            &mut rich_spans,
        );
    }

    rich_spans
}

pub fn collect_obfuscated_replacements(
    font_system: &mut FontSystem,
    buffer: &Buffer,
    formatted: &FormattedText,
    runtime: &mut UiRuntimes,
) -> HashMap<usize, char> {
    let mut line_starts = Vec::new();
    let mut line_start = 0;

    for line in formatted.text.split('\n') {
        line_starts.push(line_start);
        line_start += line.len() + 1;
    }

    let mut replacements = HashMap::new();
    let mut rng = rand::rng();

    for run in buffer.layout_runs() {
        let Some(&line_start) = line_starts.get(run.line_i) else {
            continue;
        };

        for glyph in run.glyphs {
            let Some(cluster) = run.text.get(glyph.start..glyph.end) else {
                continue;
            };

            let global_start = line_start + glyph.start;

            if !cluster
                .char_indices()
                .any(|(offset, _)| is_obfuscated_at(formatted, global_start + offset))
            {
                continue;
            }

            let key = (glyph.font_id, glyph.font_weight);

            if !runtime.obfuscation_fonts.contains_key(&key) {
                let cache = font_system
                    .get_font(glyph.font_id, glyph.font_weight)
                    .map(|font| build_obfuscation_font_cache(&font))
                    .unwrap_or_default();

                runtime.obfuscation_fonts.insert(key, cache);
            }

            let Some(cache) = runtime.obfuscation_fonts.get(&key) else {
                continue;
            };

            for (offset, ch) in cluster.char_indices() {
                let global_index = global_start + offset;

                if !is_obfuscated_at(formatted, global_index) {
                    continue;
                }

                let Some(width) = cache.width_by_char.get(&ch).copied() else {
                    continue;
                };

                let Some(candidates) = cache.candidates_by_width.get(&width) else {
                    continue;
                };

                if candidates.is_empty() {
                    continue;
                }

                let index = rng.random_range(0..candidates.len());
                let mut replacement = candidates[index];

                if replacement == ch && candidates.len() > 1 {
                    replacement = candidates[(index + 1) % candidates.len()];
                }

                if replacement != ch {
                    replacements.insert(global_index, replacement);
                }
            }
        }
    }

    replacements
}

pub fn append_emoji_font_spans<'a>(
    text: &'a str,
    style: FormatStyle,
    base_attrs: &Attrs<'a>,
    output: &mut Vec<(&'a str, Attrs<'a>)>,
) -> bool {
    if text.is_empty() {
        return false;
    }

    let mut run_start = 0;
    let mut run_is_emoji = None;
    let mut has_emoji = false;

    for (byte_idx, grapheme) in text.grapheme_indices(true) {
        let is_emoji = emojis::get(grapheme).is_some();
        has_emoji |= is_emoji;

        if let Some(previous_is_emoji) = run_is_emoji {
            if previous_is_emoji != is_emoji {
                let run_attrs = attrs_for_format(base_attrs, style);

                output.push((&text[run_start..byte_idx], run_attrs));
                run_start = byte_idx;
            }
        }

        run_is_emoji = Some(is_emoji);
    }

    if let Some(is_emoji) = run_is_emoji {
        let run_attrs = attrs_for_format(base_attrs, style);

        output.push((&text[run_start..], run_attrs));
    }

    has_emoji
}
fn attrs_for_format<'a>(base: &Attrs<'a>, format: FormatStyle) -> Attrs<'a> {
    let mut attrs = base.clone();
    let color = format.color.map(|[r, g, b, a]| Color::rgba(r, g, b, a));
    attrs.color_opt = color.or(base.color_opt);
    attrs.weight = if format.bold {
        Weight::BOLD
    } else {
        base.weight
    };
    attrs.style = if format.italic {
        Style::Italic
    } else {
        base.style
    };
    attrs.text_decoration.underline = if format.underline {
        UnderlineStyle::Single
    } else {
        base.text_decoration.underline
    };
    attrs.text_decoration.strikethrough =
        format.strikethrough || base.text_decoration.strikethrough;
    if let Some(color) = color {
        attrs.text_decoration.underline_color_opt = Some(color);
        attrs.text_decoration.strikethrough_color_opt = Some(color);
    } else {
        attrs.text_decoration.underline_color_opt = base.text_decoration.underline_color_opt;
        attrs.text_decoration.strikethrough_color_opt =
            base.text_decoration.strikethrough_color_opt;
    }
    attrs
}
