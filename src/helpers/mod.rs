use sluggrs_skylines::Color;

pub mod hsv;
pub mod implementations;
pub mod modpack;
pub mod mouse_ray;
pub mod paths;
pub mod positions;

fn f32_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

pub fn stupid_color_from_rgba(color: [f32; 4]) -> Color {
    Color::rgba(
        f32_to_u8(color[0]),
        f32_to_u8(color[1]),
        f32_to_u8(color[2]),
        f32_to_u8(color[3]),
    )
}
pub fn rgba_from_stupid_color(color: Color) -> [f32; 4] {
    color.as_rgba().map(|v| v as f32 / 255.0)
}
