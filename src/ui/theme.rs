use egui::{Color32, Stroke};

/// Palette and layout constants shared across UI components.
#[derive(Debug, Clone)]
pub struct UiTheme {
    pub panel_fill: Color32,
    pub panel_stroke: Stroke,
    pub accent: Color32,
    pub muted: Color32,
    pub rounding: f32,
}

impl Default for UiTheme {
    fn default() -> Self {
        Self {
            panel_fill: Color32::from_rgba_unmultiplied(24, 28, 38, 200),
            panel_stroke: Stroke::new(1.0, Color32::from_rgba_unmultiplied(120, 130, 155, 180)),
            accent: Color32::from_rgb(0x6a, 0xe4, 0xff),
            muted: Color32::from_gray(180),
            rounding: 10.0,
        }
    }
}
