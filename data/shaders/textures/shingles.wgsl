// shingles.wgsl
struct Params {
    color_primary: vec4<f32>,   // main shingle color
    color_secondary: vec4<f32>, // darker/lighter variation + dirt
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

    // --- Shingle layout ---
    let row_height = 0.08;
    let shingle_width = 0.12;

    let row = floor(uv.y / row_height);
    let row_uv = fract(uv.y / row_height);

    // offset every second row
    let offset = select(0.0, 0.5, i32(row) % 2 == 1);
    let col_id = floor((uv.x + offset * shingle_width) / shingle_width);
    let col_uv = fract((uv.x + offset * shingle_width) / shingle_width);

    let id = vec2<f32>(col_id, row);

    // per-shingle variation
    let rnd = hash21(id + seed_offset);
    col *= 0.85 + rnd * 0.3;

    // --- Horizontal separation between rows ---
    let row_edge = smoothstep(0.0, 0.04, row_uv);
    col *= 1.0 - (1.0 - row_edge) * 0.35;

    // --- Vertical cuts (tabs) ---
    let tab_edge = smoothstep(0.0, 0.03, col_uv) * (1.0 - smoothstep(0.97, 1.0, col_uv));
    col *= 1.0 - tab_edge * 0.25;

    // --- Bottom shadow (gives thickness illusion) ---
    let shadow = smoothstep(0.6, 1.0, row_uv);
    col *= 1.0 - shadow * 0.25;

    // --- Subtle asphalt grain (very toned down) ---
    let grain = fbm(coord * 8.0);
    col += (grain - 0.5) * (0.04 + 0.05 * params.roughness);

    // --- Color mottling ---
    let mottling = fbm(coord * 2.5 + vec2<f32>(19.0, 7.0));
    col = mix(col, params.color_secondary.rgb, (mottling - 0.5) * 0.2);

    // --- Dirt streaks (rain) ---
    let streak = fbm(vec2<f32>(coord.x * 2.0, coord.y * 0.4));
    let streak_mask = smoothstep(0.7, 1.0, streak) * (0.05 + 0.08 * params.roughness);
    col = mix(col, params.color_secondary.rgb * 0.8, streak_mask);

    // --- Slight random wear ---
    let wear = fbm(coord * 1.2 + vec2<f32>(8.0, 33.0));
    col *= 0.95 + (wear - 0.5) * 0.1;

    col = clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(output, gid.xy, vec4<f32>(col, 1.0));
}