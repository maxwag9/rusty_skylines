struct Params {
    color_primary: vec4<f32>,
    color_secondary: vec4<f32>,
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

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn hash2(ip: vec2<f32>, period: f32) -> f32 {
    let w = ip - floor(ip / period) * period;
    return fract(sin(dot(w, vec2(127.1, 311.7))) * 43758.5453);
}

fn vnoise(p: vec2<f32>, period: f32) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = vec2<f32>(fade(f.x), fade(f.y));
    let a = hash2(i, period);
    let b = hash2(i + vec2(1.0, 0.0), period);
    let c = hash2(i + vec2(0.0, 1.0), period);
    let d = hash2(i + vec2(1.0, 1.0), period);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p: vec2<f32>, base_period: f32) -> f32 {
    var acc = 0.0;
    var amp = 0.5;
    var norm = 0.0;
    var freq = 1.0;
    let oct = i32(clamp(params.octaves, 1.0, 16.0));

    for (var i: i32 = 0; i < oct; i = i + 1) {
        acc += amp * vnoise(p * freq, base_period * freq);
        norm += amp;
        freq *= params.lacunarity;
        amp *= params.persistence;
    }
    return acc / norm;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(output);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }

    let uv = vec2<f32>(f32(gid.x), f32(gid.y)) / vec2<f32>(f32(size.x), f32(size.y));
    let T = round(params.scale);
    let sx = f32(params.seed % 97u);
    let sy = f32((params.seed / 97u) % 97u);
    let p = uv * T + vec2<f32>(sx, sy);

    let half = T * 0.5;
    let wx = vnoise(p * 0.5 + vec2<f32>(17.0, 0.0), half);
    let wy = vnoise(p * 0.5 + vec2<f32>(0.0, 23.0), half);
    let wp = p + (vec2<f32>(wx, wy) - 0.5) * (1.2 + params.roughness * 2.2);

    let macro_stone = fbm(wp, T);
    let micro_grain = vnoise(wp * 12.0, T * 12.0);
    let crack_mask = vnoise(wp * 3.0, T * 3.0);

    let stone = pow(clamp(macro_stone, 0.0, 1.0), 1.6);
    let grain = smoothstep(0.35, 0.9, micro_grain);
    let cracks = smoothstep(0.55, 0.92, 1.0 - crack_mask) * 0.28;
    let shade = clamp(stone * 0.62 + grain * 0.22 + cracks, 0.0, 1.0);

    var color = mix(params.color_primary.rgb, params.color_secondary.rgb, shade);
    color *= 0.82 + stone * 0.18;
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));

    textureStore(output, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
}