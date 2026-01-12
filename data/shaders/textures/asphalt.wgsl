// shaders/asphalt.wgsl
struct Params {
    seed: u32,
    scale: f32,
    roughness: f32,
    _padding: u32,
    color_primary: vec4<f32>,
    color_secondary: vec4<f32>,
}

@group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: Params;

// High-quality 2D hashes (adapted from Inigo Quilez)
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

// Simple smooth value noise
fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let a = hash21(i);
    let b = hash21(i + vec2(1.0, 0.0));
    let c = hash21(i + vec2(0.0, 1.0));
    let d = hash21(i + vec2(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Fractional Brownian Motion (5 octaves)
fn fbm(p: vec2<f32>) -> f32 {
    var value: f32 = 0.0;
    var amplitude: f32 = 0.5;
    var pp = p;
    for (var i: i32 = 0; i < 5; i = i + 1) {
        value += amplitude * value_noise(pp);
        pp *= 2.02; // slight rotation to reduce axis alignment
        pp += vec2(100.0, 50.0);
        amplitude *= 0.5;
    }
    return value;
}

// Return distance to closest feature point + random value for that cell
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

    // Seeded offset for pattern variation
    let seed_offset = f32(params.seed);
    let coord = uv * params.scale + seed_offset * vec2(73.81, 41.37);

    var col = params.color_primary.rgb;

    // Subtle overall tone variation
    let tone = fbm(coord * 1.8);
    col *= mix(0.88, 1.15, tone);

    // Large aggregates
    var info = get_closest_cell(coord * 6.0);
    var radius = 0.28 + info.rnd * 0.22 + params.roughness * 0.25;
    var mask = 1.0 - smoothstep(radius * 0.75, radius, info.dist);
    var stone_col = params.color_secondary.rgb * (0.75 + info.rnd * 0.5);
    col = mix(col, stone_col, mask);

    // Medium aggregates (offset to avoid perfect alignment)
    info = get_closest_cell(coord * 18.0 + vec2(27.3, 61.9));
    radius = 0.14 + info.rnd * 0.12 + params.roughness * 0.15;
    mask = 1.0 - smoothstep(radius * 0.7, radius, info.dist);
    stone_col = params.color_secondary.rgb * (0.85 + info.rnd * 0.4);
    col = mix(col, stone_col, mask);

    // Fine aggregates
    info = get_closest_cell(coord * 55.0 + vec2(82.4, 19.6));
    radius = 0.065 + info.rnd * 0.05 + params.roughness * 0.08;
    mask = 1.0 - smoothstep(radius * 0.65, radius, info.dist);
    stone_col = params.color_secondary.rgb * (0.95 + info.rnd * 0.35);
    col = mix(col, stone_col, mask * 0.85);

    // Tiny high-frequency detail for extra realism
    let fine = fbm(coord * 12.0);
    col += (fine - 0.5) * 0.03 * params.roughness;

    col = clamp(col, vec3(0.0), vec3(1.0));

    textureStore(output, gid.xy, vec4<f32>(col, 1.0));
}