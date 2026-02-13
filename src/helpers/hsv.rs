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
    let c = hsv.v * hsv.s;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let m = hsv.v - c;

    let (r, g, b) = match h as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    [r + m, g + m, b + m]
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
