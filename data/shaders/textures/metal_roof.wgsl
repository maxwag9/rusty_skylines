// metal_roof.wgsl
struct Params {
    color_primary: vec4<f32>,   // base metal color (painted)
    color_secondary: vec4<f32>, // dirt / rust tint
    seed: u32,
    scale: f32,
    roughness: f32,
    octaves: f32,
    persistence: f32,
    lacunarity: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: Params;

// Hash
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + vec3<f32>(33.33));
    return fract((p3.x + p3.y) * p3.z);
}

// Noise
fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// FBM
fn fbm(p: vec2<f32>) -> f32 {
    var value: f32 = 0.0;
    var amp: f32 = 0.5;
    var pp = p;

    let octs = max(1, i32(params.octaves));
    for (var i: i32 = 0; i < octs; i = i + 1) {
        value += amp * value_noise(pp);
        pp *= params.lacunarity;
        pp += vec2<f32>(100.0, 50.0);
        amp *= params.persistence;
    }

    return value;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(output);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }

    let uv = vec2<f32>(gid.xy) / vec2<f32>(size.xy);

    let seed_offset = f32(params.seed);
    let coord = uv * params.scale + seed_offset * vec2<f32>(73.81, 41.37);

    var col = params.color_primary.rgb;

    // --- Large panel segmentation (roof sheets) ---
    let panel_width = 0.12;
    let panel_id = floor(uv.x / panel_width);
    let panel_uv = fract(uv.x / panel_width);

    // subtle brightness variation per panel
    let panel_rand = hash21(vec2<f32>(panel_id, seed_offset));
    col *= 0.9 + panel_rand * 0.2;

    // seams between panels
    let seam = smoothstep(0.0, 0.02, panel_uv) * (1.0 - smoothstep(0.98, 1.0, panel_uv));
    col *= 1.0 - seam * 0.25;

    // --- Slight waviness (cheap metal bending) ---
    let wave = fbm(coord * 2.0);
    col *= 0.95 + wave * 0.1;

    // --- Brushed / rolled directionality (vertical) ---
    let brush = fbm(vec2<f32>(coord.x * 0.8, coord.y * 12.0));
    col += (brush - 0.5) * (0.04 + 0.05 * params.roughness);

    // --- Dirt streaks (rain runoff) ---
    let streak_base = fbm(vec2<f32>(coord.x * 3.0, coord.y * 0.4));
    let streaks = smoothstep(0.7, 1.0, streak_base) * (0.08 + 0.1 * params.roughness);
    col = mix(col, params.color_secondary.rgb, streaks);

    // --- Patchy oxidation / fading ---
    let oxid = fbm(coord * 1.5 + vec2<f32>(12.0, 77.0));
    col = mix(col, params.color_secondary.rgb * 0.8, (oxid - 0.6) * 0.2 * params.roughness);

    // --- Fine surface noise (not too strong) ---
    let fine = fbm(coord * 20.0);
    col += (fine - 0.5) * 0.02;

    col = clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(output, gid.xy, vec4<f32>(col, 1.0));
}