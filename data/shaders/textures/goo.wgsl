// goo.wgsl
struct Params {
    seed: u32,
    scale: f32,
    roughness: f32,
    _padding: u32,
    color_primary: vec4<f32>,   // Black tar base
    color_secondary: vec4<f32>, // Slight sheen color
}

@group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: Params;

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
               mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x), u.y);
}

fn fbm(p: vec2<f32>) -> f32 {
    return value_noise(p) * 0.5 + value_noise(p * 2.0) * 0.3 + value_noise(p * 4.0) * 0.2;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(output);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }

    let uv = vec2<f32>(f32(gid.x), f32(gid.y)) / f32(min(size.x, size.y));
    let seed_offset = vec2<f32>(f32(params.seed) * 3.14, f32(params.seed) * 2.71);
    let p = (uv + seed_offset) * params.scale;

    // Subtle surface undulation
    let surface = fbm(p * 2.0);

    // Fake glossy highlight (brighter in some spots)
    let sheen = pow(surface, 3.0) * 0.2 * (1.0 - params.roughness);

    // Very subtle texture
    let fine = hash21(p * 30.0) * 0.03 * params.roughness;

    // Dark base with slight variation
    let color = params.color_primary.rgb * (0.95 + surface * 0.1)
              + params.color_secondary.rgb * sheen
              + fine;

    textureStore(output, vec2<i32>(gid.xy), vec4(color, 1.0));
}