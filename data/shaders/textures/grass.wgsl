struct Params {
    seed: u32,
    scale: f32,       // Controls tuft size: Lower = larger grass clumps, Higher = smaller, tighter tufts
    roughness: f32,   // Now controls dryness: 0 = fully healthy green, 1 = mostly yellow/dry grass
    _padding: u32,
    color_primary: vec4<f32>,   // Healthy grass base color (use a deep natural green)
    color_secondary: vec4<f32>, // Dry/yellow grass color for sparse patches
}

@group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: Params;

// Simple 2d->1d hash function for consistent noise
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Smooth value noise with cubic interpolation for natural gradients
fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f); // Cubic curve for smooth transitions between noise cells
    return mix(mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
               mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x), u.y);
}

// 4-octave FBM for layered, natural grass clumping detail
fn fbm(p: vec2<f32>) -> f32 {
    var result: f32 = 0.0;
    var amplitude: f32 = 0.5;
    var frequency: f32 = 1.0;

    for (var i = 0; i < 4; i++) {
        result += amplitude * value_noise(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }

    return result;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(output);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }

    // Normalize UVs and apply seed offset for random texture placement
    let uv = vec2<f32>(f32(gid.x), f32(gid.y)) / f32(min(size.x, size.y));
    let seed_offset = vec2<f32>(f32(params.seed) * 3.14159, f32(params.seed) * 2.71828);
    let p = (uv + seed_offset) * params.scale;

    // Base grass clump noise - pow sharpens transitions between dense tufts and sparse areas
    let clump_noise = pow(fbm(p), 2.2);

    // High-frequency noise to simulate subtle individual grass blade detail
    let blade_detail = (value_noise(p * 60.0) - 0.5) * 0.04;

    // Apply dry grass color only to the most sparse areas
    let dryness = smoothstep(0.65, 0.95, clump_noise) * params.roughness;

    // Subtle shadowing in dense clumps to add depth and make tufts look more volumetric
    let clump_shadow = smoothstep(0.1, 0.35, clump_noise) * 0.18;

    // Very subtle specular sheen to simulate light reflecting off grass blades
    let sheen = pow(clump_noise, 8.0) * 0.08 * (1.0 - params.roughness);

    // Final color mixing pipeline
    let base_color = mix(params.color_primary.rgb, params.color_secondary.rgb, dryness);
    let final_color = base_color * (1.0 - clump_shadow) + blade_detail + sheen;

    textureStore(output, vec2<i32>(gid.xy), vec4(final_color, 1.0));
}