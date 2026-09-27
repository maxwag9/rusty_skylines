// wood_slab.wgsl
struct Params {
    color_primary: vec4<f32>,    // light wood tone
    color_secondary: vec4<f32>,  // darker stain / shadow tone
    seed: u32,
    scale: f32,                  // overall grain frequency
    roughness: f32,              // 0..1
    octaves: f32,                // used as a detail multiplier
    persistence: f32,            // detail strength
    lacunarity: f32,             // detail frequency multiplier
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

fn hash22(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        hash21(p + vec2<f32>(17.1, 31.7)),
        hash21(p + vec2<f32>(91.3, 47.9))
    );
}

fn periodic_delta(a: f32, b: f32) -> f32 {
    let d = abs(a - b);
    return min(d, 1.0 - d);
}

fn periodic_dist(a: vec2<f32>, b: vec2<f32>) -> f32 {
    let dx = periodic_delta(a.x, b.x);
    let dy = periodic_delta(a.y, b.y);
    return sqrt(dx * dx + dy * dy);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(output);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }

    let uv = vec2<f32>(gid.xy) / vec2<f32>(size.xy);
    let seedf = f32(params.seed);

    // Base wood palette
    var col = mix(params.color_primary.rgb, params.color_secondary.rgb, 0.18);

    // Keep everything tileable by only using periodic functions of uv
    // Long grain running mostly horizontally
    let grain_freq = max(1.0, params.scale * 18.0);
    let wobble_1 = sin((uv.x * TAU * 2.0 + seedf * 0.013) + uv.y * TAU * 0.35) * 0.020;
    let wobble_2 = sin((uv.x * TAU * 5.0 + seedf * 0.021) + uv.y * TAU * 0.90) * 0.010;
    let warp = wobble_1 + wobble_2;

    let grain_phase = (uv.y + warp) * TAU * grain_freq;
    let grain_wave = sin(grain_phase);
    let grain_ridge = abs(grain_wave);

    // Main wood grain
    let dark_grain = smoothstep(0.55, 1.0, grain_ridge);
    let light_grain = smoothstep(0.00, 0.35, grain_ridge);

    col = mix(col, params.color_secondary.rgb * 0.90, dark_grain * (0.28 + 0.18 * params.roughness));
    col = mix(col, params.color_primary.rgb * 1.05, light_grain * 0.10);

    // Gentle banding so it reads like a slab
    let band = 0.5 + 0.5 * sin((uv.y * TAU * (3.0 + params.scale * 1.5)) + seedf * 0.007);
    col += (band - 0.5) * (0.05 + 0.03 * params.roughness);

    // Two tileable knots, wrapped around the UV domain
    let knot0 = hash22(vec2<f32>(seedf + 11.7, seedf + 29.1));
    let knot1 = hash22(vec2<f32>(seedf + 53.4, seedf + 77.8));

    let d0 = periodic_dist(uv, knot0);
    let d1 = periodic_dist(uv, knot1);

    let knot_core_0 = 1.0 - smoothstep(0.00, 0.055, d0);
    let knot_ring_0 = smoothstep(0.035, 0.13, abs(sin(d0 * 70.0 - seedf * 0.01)));

    let knot_core_1 = 1.0 - smoothstep(0.00, 0.045, d1);
    let knot_ring_1 = smoothstep(0.030, 0.11, abs(sin(d1 * 85.0 + seedf * 0.008)));

    let knots = clamp(knot_core_0 * 0.9 + knot_ring_0 * 0.35 + knot_core_1 * 0.8 + knot_ring_1 * 0.30, 0.0, 1.0);

    col = mix(col, params.color_secondary.rgb * 0.72, knots * (0.45 + 0.25 * params.roughness));

    // Fine porous detail, still periodic
    let micro_1 = sin((uv.x * TAU * 41.0 + uv.y * TAU * 19.0) + seedf * 0.031);
    let micro_2 = sin((uv.x * TAU * 73.0 - uv.y * TAU * 27.0) + seedf * 0.019);
    let micro = (micro_1 * 0.5 + micro_2 * 0.5) * 0.5 + 0.5;
    col += (micro - 0.5) * (0.02 + 0.02 * params.roughness);

    // Slight edge-to-center color variance, but periodic
    let edge_warp = 0.5 + 0.5 * sin(uv.x * TAU * 2.0 + seedf * 0.004) * sin(uv.y * TAU * 2.0 + seedf * 0.006);
    col *= 0.96 + edge_warp * 0.06;

    // Optional roughness breakup
    let r = clamp(params.roughness, 0.0, 1.0);
    col += vec3<f32>(micro - 0.5) * (0.02 * r);

    col = clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(output, gid.xy, vec4<f32>(col, 1.0));
}