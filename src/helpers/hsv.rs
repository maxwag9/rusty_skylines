#[derive(Copy, Clone)]
pub struct HSV {
    pub(crate) h: f32, // 0..1
    pub(crate) s: f32, // 0..1
    pub(crate) v: f32, // 0..1
}

pub fn rgb_to_hsv([r, g, b, _a]: [f32; 4]) -> HSV {
    let max = r.max(g.max(b));
    let min = r.min(g.min(b));
    let d = max - min;

    let h = if d == 0.0 {
        0.0
    } else if max == r {
        ((g - b) / d).rem_euclid(6.0) / 6.0
    } else if max == g {
        ((b - r) / d + 2.0) / 6.0
    } else {
        ((r - g) / d + 4.0) / 6.0
    };

    let s = if max == 0.0 { 0.0 } else { d / max };

    HSV { h, s, v: max }
}

pub fn hsv_to_rgb(hsv: HSV) -> [f32; 3] {
    let h = hsv.h * 6.0;
    let i = h.floor();
    let f = h - i;

    let p = hsv.v * (1.0 - hsv.s);
    let q = hsv.v * (1.0 - hsv.s * f);
    let t = hsv.v * (1.0 - hsv.s * (1.0 - f));

    match i as i32 {
        0 => [hsv.v, t, p],
        1 => [q, hsv.v, p],
        2 => [p, hsv.v, t],
        3 => [p, q, hsv.v],
        4 => [t, p, hsv.v],
        _ => [hsv.v, p, q],
    }
}

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub fn lerp_hsv(a: HSV, b: HSV, t: f32) -> HSV {
    let mut dh = b.h - a.h;
    if dh.abs() > 0.5 {
        dh -= dh.signum();
    }

    HSV {
        h: (a.h + dh * t).rem_euclid(1.0),
        s: lerp(a.s, b.s, t),
        v: lerp(a.v, b.v, t),
    }
}

/// Convert depth to rainbow color for BVH visualization
pub fn depth_to_color(depth: u32, max_depth: u32) -> [f32; 3] {
    let t = if max_depth > 0 {
        (depth as f32 / max_depth as f32).min(1.0)
    } else {
        0.0
    };

    // Rainbow: red -> yellow -> green -> cyan -> blue
    let hue = t * 0.7;
    hsv_to_rgb(HSV {
        h: hue,
        s: 0.9,
        v: 1.0,
    })
}
