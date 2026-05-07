// driveway_bricks.wgsl
struct Params {
    color_primary: vec4<f32>,   // stone color
    color_secondary: vec4<f32>, // black joints / gaps
    seed: u32,
    scale: f32,       // interpreted as stone count across the texture
    roughness: f32,   // more irregular edges and wear
    octaves: f32,     // used for subtle detail layers
    persistence: f32,
    lacunarity: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: Params;

const TAU: f32 = 6.28318530718;

// Hash
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + vec3<f32>(33.33));
    return fract((p3.x + p3.y) * p3.z);
}

fn wrap_i(v: i32, m: i32) -> i32 {
    return ((v % m) + m) % m;
}

// Tileable value noise over a finite integer period
fn tile_value_noise(p: vec2<f32>, period: i32, seed: f32) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let x0 = wrap_i(i.x, period);
    let y0 = wrap_i(i.y, period);
    let x1 = wrap_i(i.x + 1, period);
    let y1 = wrap_i(i.y + 1, period);

    let a = hash21(vec2<f32>(f32(x0), f32(y0)) + seed);
    let b = hash21(vec2<f32>(f32(x1), f32(y0)) + seed);
    let c = hash21(vec2<f32>(f32(x0), f32(y1)) + seed);
    let d = hash21(vec2<f32>(f32(x1), f32(y1)) + seed);

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Tileable FBM
fn tile_fbm(p: vec2<f32>, period: i32, octaves: i32, persistence: f32, lacunarity: f32, seed: f32) -> f32 {
    var value: f32 = 0.0;
    var amp: f32 = 0.5;
    var pp = p;
    var per = period;

    let o = max(1, octaves);
    for (var i: i32 = 0; i < o; i = i + 1) {
        value += amp * tile_value_noise(pp, per, seed + f32(i) * 17.0);

        let l = max(2.0, round(lacunarity));
        pp *= l;
        per = max(1, per * i32(l));

        amp *= persistence;
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

    // Force integer repetition so the result tiles cleanly.
    let stone_count = max(2, i32(floor(params.scale)));
    let p = uv * f32(stone_count);

    // Brick/paver layout in repeating cells
    let row = i32(floor(p.y));
    let row_id = wrap_i(row, stone_count);

    let stagger = select(0.0, 0.5, (row_id % 2) == 1);
    let xw = p.x + stagger;
    let col = i32(floor(xw));
    let col_id = wrap_i(col, stone_count);

    let local = vec2<f32>(fract(xw), fract(p.y));
    let id = vec2<f32>(f32(col_id), f32(row_id));

    // Per-stone randomness
    let rnd1 = hash21(id + seed_offset * vec2<f32>(17.0, 29.0));
    let rnd2 = hash21(id + seed_offset * vec2<f32>(41.0, 11.0) + vec2<f32>(9.0, 3.0));

    let lp = clamp(local, vec2<f32>(0.0), vec2<f32>(1.0));

    // Stone mask: dark joints between stones, primary color inside stones.
    let edge_dist = min(min(lp.x, 1.0 - lp.x), min(lp.y, 1.0 - lp.y));

    let joint_base = 0.01;
    let joint_var = 0.03 * rnd1 + 0.02 * params.roughness;
    let joint_width = joint_base + joint_var;

    let feather = 0.03 + 0.02 * params.roughness;
    var stone_mask = smoothstep(joint_width, joint_width + feather, edge_dist);

    stone_mask = smoothstep(joint_width, joint_width + feather, edge_dist);

    // Base colors
    let mortar = params.color_secondary.rgb;
    var stone = params.color_primary.rgb;

    // Per-stone tone variation
    let tone = 0.82 + rnd1 * 0.28;
    stone *= tone;

    // Subtle stone mottling
    let mottling = tile_fbm(
        p * 2.5 + vec2<f32>(19.0, 7.0),
        stone_count,
        max(1, i32(params.octaves)),
        clamp(params.persistence, 0.1, 0.95),
        max(2.0, params.lacunarity),
        seed_offset + 13.0
    );
    stone = mix(stone, stone * (0.85 + 0.25 * mottling), 0.35);

    // Slight roughness in the face of the stones
    let grain = tile_fbm(
        p * 8.0 + vec2<f32>(5.0, 21.0),
        stone_count,
        2,
        0.5,
        2.0,
        seed_offset + 77.0
    );
    stone += (grain - 0.5) * (0.04 + 0.05 * params.roughness);

    // Faint wear and dirt near joints
    let joint_darkening = (1.0 - stone_mask) * (0.08 + 0.12 * params.roughness);
    stone *= 1.0 - joint_darkening;

    // Final blend
    var color = mix(mortar, stone, stone_mask);

    // Darken the very edges a bit more to enhance the joint look
    let edge_shadow = 1.0 - smoothstep(0.0, 0.03, edge_dist);
    color = mix(color, mortar, edge_shadow * 0.65);

    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(output, gid.xy, vec4<f32>(color, 1.0));
}