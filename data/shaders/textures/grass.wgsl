// grass.wgsl
struct Params {
    color_primary:   vec4<f32>,
    color_secondary: vec4<f32>,
    seed:        u32,
    scale:       f32,
    roughness:   f32,
    octaves:     f32,
    persistence: f32,
    lacunarity:  f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: Params;

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Tiling hash: wraps integer lattice coords at `period` before hashing.
// This is what actually makes the texture seamless.
fn hash2(ip: vec2<f32>, period: f32) -> f32 {
    let w = ip - floor(ip / period) * period;
    return fract(sin(dot(w, vec2(127.1, 311.7))) * 43758.5453);
}

fn vnoise(p: vec2<f32>, period: f32) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = vec2<f32>(fade(f.x), fade(f.y));
    let a = hash2(i,                  period);
    let b = hash2(i + vec2(1.0, 0.0), period);
    let c = hash2(i + vec2(0.0, 1.0), period);
    let d = hash2(i + vec2(1.0, 1.0), period);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Tiling FBM — period scales with frequency so every octave stays seamless.
// Requires lacunarity = 2.0 (integer) to keep periods as integers.
fn fbm(p: vec2<f32>, base_period: f32) -> f32 {
    var acc  = 0.0;
    var amp  = 0.5;
    var norm = 0.0;
    var freq = 1.0;
    let oct  = i32(clamp(params.octaves, 1.0, 16.0));

    for (var i: i32 = 0; i < oct; i = i + 1) {
        acc  += amp * vnoise(p * freq, base_period * freq);
        norm += amp;
        freq *= params.lacunarity;
        amp  *= params.persistence;
    }
    return acc / norm;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(output);
    if (gid.x >= size.x || gid.y >= size.y) { return; }

    let uv = vec2<f32>(f32(gid.x), f32(gid.y)) / vec2<f32>(f32(size.x), f32(size.y));

    // Round to integer so all octave periods are integers → guaranteed seamless
    let T = round(params.scale);

    // Integer seed offset: adding integers to integer-period noise preserves tiling
    let sx = f32(params.seed % 97u);
    let sy = f32((params.seed / 97u) % 97u);
    let p  = uv * T + vec2<f32>(sx, sy);

    // Tileable domain warp: warp noise uses period = T/2.
    // p*0.5 spans exactly [0, T/2] as uv spans [0,1] → exactly one period → seamless.
    let half = T * 0.5;
    let wx = vnoise(p * 0.5 + vec2<f32>(17.0,  0.0), half);
    let wy = vnoise(p * 0.5 + vec2<f32>( 0.0, 23.0), half);
    let wp = p + (vec2<f32>(wx, wy) - 0.5) * 2.2;

    // Three clearly separated frequency bands
    let macro_clump = fbm(wp, T);                         // large clump silhouettes
    let mid_bundle  = vnoise(wp * 5.0,  T * 5.0);        // mid-size blade bundles
    let fine_tips   = vnoise(wp * 16.0, T * 16.0);       // fine individual tips

    // Sharpen clump contrast so tufts read as distinct shapes
    let clump_s = pow(clamp(macro_clump, 0.0, 1.0), 2.0);

    // Dryness bleeds into secondary color in sparse areas only
    let dryness = smoothstep(0.45, 0.9, 1.0 - clump_s) * params.roughness;

    // Shadow deepens the cores of dense tufts — gives volume
    let shadow = smoothstep(0.58, 1.0, clump_s) * 0.42;

    // Tip highlight: only the very top of fine_tips catches light
    let highlight = smoothstep(0.72, 1.0, fine_tips) * 0.14;

    // Color pipeline — each stage has a clear job, no muddying overlap
    var color = mix(params.color_primary.rgb, params.color_secondary.rgb, dryness);
    color += color * highlight;
    color *= 1.0 - shadow;
    // Final micro-contrast boost from blended density
    let density = clump_s * 0.55 + mid_bundle * 0.3 + fine_tips * 0.15;
    color *= 0.78 + density * 0.22;

    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    textureStore(output, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
}