// paint.wgsl
struct Params {
    color_primary: vec4<f32>,   // main paint color
    color_secondary: vec4<f32>, // undercoat / plaster / dirt tint
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

// High-quality 2D hashes
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + vec3<f32>(33.33, 33.33, 33.33));
    return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + vec3<f32>(33.33, 33.33, 33.33));
    return fract((p3.xx + p3.yz) * p3.zy);
}

// Smooth value noise
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

// Fractional Brownian Motion
fn fbm(p: vec2<f32>) -> f32 {
    var value: f32 = 0.0;
    var amplitude: f32 = 0.5;
    var pp = p;

    let octs = max(1, i32(params.octaves));
    for (var i: i32 = 0; i < octs; i = i + 1) {
        value += amplitude * value_noise(pp);
        pp *= params.lacunarity;
        pp += vec2<f32>(100.0, 50.0);
        amplitude *= params.persistence;
    }

    return value;
}

// Voronoi-like cell info for chips / patched areas
struct CellInfo {
    dist: f32,
    rnd: f32,
};

fn get_closest_cell(p: vec2<f32>) -> CellInfo {
    let i = floor(p);
    let f = fract(p);

    var info: CellInfo;
    info.dist = 8.0;
    info.rnd = 0.0;

    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let offset = vec2<f32>(f32(dx), f32(dy));
            let cell = i + offset;
            let pnt = hash22(cell);
            let pos = offset + pnt;
            let dist = length(pos - f);
            let rnd = hash21(cell + 123.456);

            if (dist < info.dist) {
                info.dist = dist;
                info.rnd = rnd;
            }
        }
    }

    return info;
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

    var base = params.color_primary.rgb;
    var col = base;

    // Wall-scale unevenness from plaster / old paint
    let low = fbm(coord * 1.4);
    let mid = fbm(coord * 4.2 + vec2<f32>(19.7, 8.3));
    let fine = fbm(coord * 13.0 + vec2<f32>(3.1, 91.4));

    // Slight brightness and saturation variation
    let tone = (low - 0.5) * (0.10 + params.roughness * 0.08);
    col *= 1.0 + tone;
    col = mix(col, params.color_secondary.rgb, (mid - 0.5) * 0.05 * params.roughness);

    // Soft roller texture, mostly directional
    let roller = sin(coord.y * 45.0 + low * 6.0 + seed_offset * 0.01);
    col += roller * (0.015 + 0.02 * params.roughness);

    // Vertical streaks from rain and aging
    let streak_seed = fbm(vec2<f32>(coord.x * 2.2, coord.y * 0.35));
    let streaks = smoothstep(0.72, 1.0, streak_seed) * (0.03 + 0.04 * params.roughness);
    col -= vec3<f32>(streaks);

    // Small paint specks and plaster grain
    let grain = fine - 0.5;
    col += vec3<f32>(grain * (0.025 + 0.035 * params.roughness));

    // Slightly rougher patches / undercoat showing through
    var info = get_closest_cell(coord * 10.0);
    var patchy = 0.16 + info.rnd * 0.18 + params.roughness * 0.10;
    var mask = 1.0 - smoothstep(patchy * 0.8, patchy, info.dist);
    var undercoat = mix(params.color_secondary.rgb, base * 0.85, 0.35 + info.rnd * 0.25);
    col = mix(col, undercoat, mask * 0.18);

    // Fine chipped spots
    info = get_closest_cell(coord * 34.0 + vec2<f32>(17.0, 61.0));
    patchy = 0.045 + info.rnd * 0.03 + params.roughness * 0.03;
    mask = 1.0 - smoothstep(patchy * 0.75, patchy, info.dist);
    col = mix(col, params.color_secondary.rgb * 0.95, mask * 0.22);

    // Extra subtle mottling so it does not look flat
    let mottling = fbm(coord * 0.7 + vec2<f32>(8.0, 14.0));
    col += (mottling - 0.5) * 0.035;

    col = clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(output, gid.xy, vec4<f32>(col, 1.0));
}