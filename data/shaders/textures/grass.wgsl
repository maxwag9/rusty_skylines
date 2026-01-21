struct Params {
    color_primary: vec4<f32>,   // Healthy grass color
    color_secondary: vec4<f32>, // Dry grass
    seed: u32,
    scale: f32,                 // Tuft density: lower = big clumps, higher = fine grass
    roughness: f32,             // Dryness bias: 0 = lush, 1 = dead
    moisture: f32,              // Counteracts roughness, adds saturation and sheen
    shadow_strength: f32,       // How dark dense clumps get
    sheen_strength: f32,        // Blade light reflection amount
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: Params;

// High-quality hash
fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

// Quintic fade for ultra-smooth interpolation
fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Value noise with quintic interpolation
fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let ux = fade(f.x);
    let uy = fade(f.y);

    let a = hash21(i);
    let b = hash21(i + vec2(1.0, 0.0));
    let c = hash21(i + vec2(0.0, 1.0));
    let d = hash21(i + vec2(1.0, 1.0));

    return mix(mix(a, b, ux), mix(c, d, ux), uy);
}

// 8-octave FBM with domain rotation for isotropy
fn fbm(p_in: vec2<f32>) -> f32 {
    let ROT = mat2x2<f32>(vec2(0.8, 0.6), vec2(-0.6, 0.8));
    var acc: f32 = 0.0;
    var amp: f32 = 0.5;
    var p = p_in;

    for (var i: i32 = 0; i < 8; i = i + 1) {
        acc += amp * value_noise(p);
        p = ROT * p * 2.0;
        amp *= 0.5;
    }
    return acc;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(output);
    if (gid.x >= size.x || gid.y >= size.y) { return; }

    let uv = vec2<f32>(f32(gid.x), f32(gid.y)) / f32(min(size.x, size.y));
    let seed_offset = vec2<f32>(f32(params.seed) * 3.14159, f32(params.seed) * 2.71828);
    let p = (uv + seed_offset) * params.scale;

    // Domain warping for organic clump shapes
    let warp_strength = 2.5;
    let warp_freq = 2.0;
    let wp = p * warp_freq;
    let wx = value_noise(wp);
    let wy = value_noise(wp + vec2(13.37, 9.42));
    let warp = (vec2(wx, wy) - 0.5) * warp_strength;
    let warped_p = p + warp;

    // Base clump noise
    let clump_raw = fbm(warped_p);
    let clump_noise = pow(clump_raw, 2.4); // Softer contrast than 2.2

    // High-frequency blade detail
    let blade_detail = (value_noise(warped_p * 80.0) - 0.5) * 0.1;

    // Dryness in sparser areas
    let dryness = smoothstep(0.5, 0.9, clump_noise) * params.roughness * (1.0 - params.moisture);

    // Shadowing in dense clumps for volume
    let clump_shadow = smoothstep(0.3, 0.7, clump_noise) * params.shadow_strength;

    // Subtle specular sheen on blades
    let sheen = pow(clump_noise, 7.0) * params.sheen_strength * params.moisture;

    // Subtle color variation (greener in moist spots)
    let color_var = (value_noise(warped_p * 40.0) - 0.5) * 0.08;
    let tint = vec3(color_var * -0.1, color_var * 0.2, color_var * -0.05);

    // Final color pipeline
    var base_color = mix(params.color_primary.rgb, params.color_secondary.rgb, dryness) + tint;
    var lit_color = base_color * (1.0 - clump_shadow);
    var final_color = lit_color + vec3(blade_detail + sheen);

    final_color = clamp(final_color, vec3(0.0), vec3(1.0));

    textureStore(output, vec2<i32>(gid.xy), vec4<f32>(final_color, 1.0));
}
