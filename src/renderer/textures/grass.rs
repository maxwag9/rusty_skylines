#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GrassParams {
    pub grass_color: [f32; 4], // rgba
    pub blade_density: f32,
    pub blade_height: f32,
    pub wind_phase: f32,
    pub time: f32,
    pub noise_scale: f32,
    pub _pad: [f32; 3], // padding to 32 bytes
}
pub fn generate_noise(size: usize) -> Vec<f32> {
    let side = (size as f32).sqrt() as usize;
    assert!(side * side == size);

    let mut noise = vec![0.0f32; size];

    // simple hash-based pseudo random
    fn hash(x: i32, y: i32) -> f32 {
        let mut n = x + y * 57;
        n = (n << 13) ^ n;
        let nn = (n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff;
        nn as f32 / 2147483647.0
    }

    // smooth interpolation
    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + t * (b - a)
    }

    fn smoothstep(t: f32) -> f32 {
        t * t * (3.0 - 2.0 * t)
    }

    let scale = 8.0; // controls grass clumping size

    for y in 0..side {
        for x in 0..side {
            let fx = x as f32 / scale;
            let fy = y as f32 / scale;

            let x0 = fx.floor() as i32;
            let y0 = fy.floor() as i32;
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            let sx = smoothstep(fx - x0 as f32);
            let sy = smoothstep(fy - y0 as f32);

            let n0 = lerp(hash(x0, y0), hash(x1, y0), sx);
            let n1 = lerp(hash(x0, y1), hash(x1, y1), sx);
            let value = lerp(n0, n1, sy);

            noise[y * side + x] = value;
        }
    }

    noise
}
