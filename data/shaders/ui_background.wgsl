struct ScreenUniform {
    size: vec2<f32>,
    time: f32,
    enable_dither: u32,
    mouse: vec2<f32>,
};
@group(1) @binding(0) var<uniform> screen: ScreenUniform;
@group(1) @binding(1) var<uniform> background: Background;

struct Background {
    primary_color: vec4<f32>,
    secondary_color: vec4<f32>,
    block_size: f32,

    warp_strength: f32,
    warp_radius: f32,
    time_scale: f32,
    wave_strength: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 2);
    let y = f32(i32(vertex_index >> 1u) * 2);
    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

// Attempt to produce a filtered square wave.
// For a square wave with period 1, the integral over [x - w/2, x + w/2]
// gives the fraction of the pixel that's in the "on" phase.
// This returns a value in [0, 1] representing the filtered square wave.
fn filtered_square(x: f32, w: f32) -> f32 {
    // Clamp filter width to avoid division issues; if the pixel covers
    // more than one full period, just return 0.5 (perfect blend).
    let fw = max(abs(w), 1e-6);
    if (fw >= 1.0) {
        return 0.5;
    }
    // Map x into [0,1) period
    // The square wave is 0 for fract(x) < 0.5, 1 for fract(x) >= 0.5
    // The box-filtered integral:
    //   integral of step(0.5, fract(t)) from x-w/2 to x+w/2, divided by w
    //
    // A simpler robust approach: use the triangle wave as the integral of the square wave.
    // The integral of a square wave (period 1) is a triangle wave.
    // filtered_square = (tri(x + w/2) - tri(x - w/2)) / w
    // where tri(x) = abs(2 * fract(x) - 1) mapped appropriately.
    //
    // Actually, the cleanest formulation:
    // For a square wave s(x) = step(0.5, fract(x)), the antiderivative is
    //   S(x) = x - max(fract(x) - 0.5, 0.0)  ... but that's tricky with fract.
    //
    // Use the standard IQ approach:
    // filtered checker = 0.5 - 0.5 * tri_filtered
    // where tri_filtered uses the cos approach.

    // Simplest correct method: use the triangle-wave filtering trick.
    // A square wave can be filtered by computing the average of the triangle wave integral.
    // tri(x) = 1 - 2*abs(fract(x) - 0.5)  (maps to [-1,1], period 1, peaks at integers)
    // The box filter of the square wave over width w equals:
    //   (tri_integral(x+w/2) - tri_integral(x-w/2)) / w
    // But it's easier to just compute it as:

    let xp = x + 0.5 * fw;
    let xm = x - 0.5 * fw;
    // Antiderivative of square wave s(t) = step(0.5, fract(t)):
    // S(t) = floor(t)*0.5 + max(fract(t) - 0.5, 0.0)
    let sp = floor(xp) * 0.5 + max(fract(xp) - 0.5, 0.0);
    let sm = floor(xm) * 0.5 + max(fract(xm) - 0.5, 0.0);
    return (sp - sm) / fw;
}

// 2D filtered checkerboard. Returns the blend factor: 0 = primary, 1 = secondary.
fn filtered_checker(uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> f32 {
    // The pixel footprint width in each axis (conservative estimate)
    let w = abs(ddx) + abs(ddy);

    // Filter each axis independently.
    // The checker pattern is: (floor(x) + floor(y)) % 2
    // Which equals: square_wave(x) XOR square_wave(y)
    // The filtered version is: sx + sy - 2*sx*sy  (XOR in probability)
    let sx = filtered_square(uv.x, w.x);
    let sy = filtered_square(uv.y, w.y);
    return sx + sy - 2.0 * sx * sy;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    // Work in pixel space so block_size means pixels
    let uv = in.position.xy;

    // Mouse in pixel space
    let mouse_px = screen.mouse;

    // Normalize distance so warp_radius works like it did in 0–1 space
    let screen_scale = min(screen.size.x, screen.size.y);
    let dist = distance(uv, mouse_px) / screen_scale;

    // Smooth radial falloff
    let warp_factor = exp(-dist * background.warp_radius) * background.warp_strength * screen_scale;

    // Direction from mouse
    let raw_dir = uv - mouse_px;
    let dir = select(normalize(raw_dir), vec2<f32>(0.0, 0.0), length(raw_dir) < 1e-8);

    // Apply radial warp (now in pixel units)
    var warped_uv = uv + dir * warp_factor;

    // Add animated sine wobble (scaled to pixel space)
    warped_uv += sin((warped_uv.yx / screen_scale + screen.time * background.time_scale) * 6.0)
                 * background.wave_strength * screen_scale;

    // Divide by block_size so each cell is block_size pixels wide
    let scaled = warped_uv / background.block_size;

    // Compute screen-space derivatives of the scaled UV
    let ddx_scaled = dpdx(scaled);
    let ddy_scaled = dpdy(scaled);

    // Analytically filtered checkerboard
    let checker_blend = filtered_checker(scaled, ddx_scaled, ddy_scaled);

    // Mix between primary and secondary based on filtered result
    let final_col = mix(background.primary_color, background.secondary_color, checker_blend);

    return final_col;
}
