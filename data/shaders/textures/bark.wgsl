// bark.wgsl - Seamless tileable tree bark texture for trees
struct Params {
    color_primary: vec4<f32>,
    color_secondary: vec4<f32>,
    seed: u32,
    scale: f32,
    roughness: f32,
    moisture: f32,
    shadow_strength: f32,
    sheen_strength: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: Params;

// Seamless hash - wraps at period boundaries
fn hash21_tiled(p: vec2<f32>, period: vec2<f32>) -> f32 {
    let wrapped = ((p % period) + period) % period;
    var p3 = fract(vec3(wrapped.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash22_tiled(p: vec2<f32>, period: vec2<f32>) -> vec2<f32> {
    let wrapped = ((p % period) + period) % period;
    var p3 = fract(vec3(wrapped.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

// Non-tiled hash for final pixel noise
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Seamless value noise
fn value_noise_tiled(p: vec2<f32>, period: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let i00 = i;
    let i10 = i + vec2(1.0, 0.0);
    let i01 = i + vec2(0.0, 1.0);
    let i11 = i + vec2(1.0, 1.0);

    return mix(
        mix(hash21_tiled(i00, period), hash21_tiled(i10, period), u.x),
        mix(hash21_tiled(i01, period), hash21_tiled(i11, period), u.x),
        u.y
    );
}

// Seamless gradient noise for smoother results
fn gradient_noise_tiled(p: vec2<f32>, period: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    let g00 = hash22_tiled(i, period) * 2.0 - 1.0;
    let g10 = hash22_tiled(i + vec2(1.0, 0.0), period) * 2.0 - 1.0;
    let g01 = hash22_tiled(i + vec2(0.0, 1.0), period) * 2.0 - 1.0;
    let g11 = hash22_tiled(i + vec2(1.0, 1.0), period) * 2.0 - 1.0;

    let n00 = dot(g00, f);
    let n10 = dot(g10, f - vec2(1.0, 0.0));
    let n01 = dot(g01, f - vec2(0.0, 1.0));
    let n11 = dot(g11, f - vec2(1.0, 1.0));

    return mix(mix(n00, n10, u.x), mix(n01, n11, u.x), u.y) * 0.5 + 0.5;
}

// Seamless FBM with multiple octaves
fn fbm_tiled(p: vec2<f32>, period: vec2<f32>, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var freq = 1.0;
    var per = period;

    for (var i = 0; i < octaves; i++) {
        value += amplitude * gradient_noise_tiled(p * freq, per);
        amplitude *= 0.5;
        freq *= 2.0;
        per *= 2.0;
    }
    return value;
}

// Seamless Worley/cellular noise for crack patterns
fn worley_tiled(p: vec2<f32>, period: vec2<f32>) -> vec2<f32> {
    let i = floor(p);
    let f = fract(p);

    var min_dist1 = 10.0;
    var min_dist2 = 10.0;

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let neighbor = vec2(f32(x), f32(y));
            let cell = i + neighbor;
            let pnt = hash22_tiled(cell, period);
            let diff = neighbor + pnt - f;
            let dist = length(diff);

            if (dist < min_dist1) {
                min_dist2 = min_dist1;
                min_dist1 = dist;
            } else if (dist < min_dist2) {
                min_dist2 = dist;
            }
        }
    }

    return vec2(min_dist1, min_dist2);
}

// Ridged noise for sharp bark ridges
fn ridged_noise_tiled(p: vec2<f32>, period: vec2<f32>) -> f32 {
    let n = gradient_noise_tiled(p, period);
    return 1.0 - abs(n * 2.0 - 1.0);
}

// Vertical bark fissure pattern
fn bark_fissures(uv: vec2<f32>, period: vec2<f32>, seed: f32) -> f32 {
    // Stretch vertically for bark-like appearance
    let bark_uv = vec2(uv.x * 3.0, uv.y * 0.8);
    let bark_period = vec2(period.x * 3.0, period.y * 0.8);

    // Primary vertical fissures
    let worley1 = worley_tiled(bark_uv * 2.5 + seed, bark_period * 2.5);
    let edge1 = worley1.y - worley1.x;
    let fissure1 = smoothstep(0.0, 0.2, edge1);

    // Secondary finer cracks
    let worley2 = worley_tiled(bark_uv * 6.0 + seed * 2.3, bark_period * 6.0);
    let edge2 = worley2.y - worley2.x;
    let fissure2 = smoothstep(0.0, 0.15, edge2);

    // Combine - deep fissures with surface cracks
    return fissure1 * 0.65 + fissure2 * 0.35;
}

// Vertical ridge pattern
fn bark_ridges(uv: vec2<f32>, period: vec2<f32>, seed: f32) -> f32 {
    // Warp for organic irregularity
    let warp_strength = 0.12;
    let warp = fbm_tiled(uv * 3.0 + seed, period * 3.0, 3) * warp_strength;
    let warped_uv = vec2(uv.x + warp, uv.y);

    // Vertical ridged pattern
    let ridge_freq = 6.0;
    let ridge1 = ridged_noise_tiled(
        vec2(warped_uv.x * ridge_freq, warped_uv.y * 1.5) + seed,
        vec2(period.x * ridge_freq, period.y * 1.5)
    );

    // Add variation with different frequency
    let ridge2 = ridged_noise_tiled(
        vec2(warped_uv.x * ridge_freq * 2.3, warped_uv.y * 2.0) + seed * 1.7,
        vec2(period.x * ridge_freq * 2.3, period.y * 2.0)
    );

    return ridge1 * 0.7 + ridge2 * 0.3;
}

// Horizontal crack pattern (breaks up vertical lines)
fn horizontal_cracks(uv: vec2<f32>, period: vec2<f32>, seed: f32) -> f32 {
    // Stretch horizontally
    let crack_uv = vec2(uv.x * 0.6, uv.y * 4.0);
    let crack_period = vec2(period.x * 0.6, period.y * 4.0);

    let worley = worley_tiled(crack_uv * 2.0 + seed * 3.7, crack_period * 2.0);
    let crack = smoothstep(0.0, 0.08, worley.y - worley.x);

    return crack;
}

// Surface detail/roughness texture
fn surface_detail(uv: vec2<f32>, period: vec2<f32>, seed: f32) -> f32 {
    // Multi-scale detail
    let detail1 = fbm_tiled(uv * 8.0 + seed, period * 8.0, 4);
    let detail2 = value_noise_tiled(uv * 24.0 + seed * 2.0, period * 24.0);
    let detail3 = value_noise_tiled(uv * 48.0 + seed * 3.0, period * 48.0);

    return detail1 * 0.5 + detail2 * 0.3 + detail3 * 0.2;
}

// Height/depth map for normal-like shading
fn bark_height(uv: vec2<f32>, period: vec2<f32>, seed: f32) -> f32 {
    let fissures = bark_fissures(uv, period, seed);
    let ridges = bark_ridges(uv, period, seed);
    let h_cracks = horizontal_cracks(uv, period, seed);

    // Combine: ridges are high, fissures are low
    var height = ridges * fissures * h_cracks;

    // Add surface variation
    height += surface_detail(uv, period, seed) * 0.15;

    return height;
}

// Calculate pseudo-normal from height for lighting
fn calc_normal(uv: vec2<f32>, period: vec2<f32>, seed: f32, texel_size: vec2<f32>) -> vec3<f32> {
    let eps = texel_size * 2.0;

    let h_center = bark_height(uv, period, seed);
    let h_right = bark_height(uv + vec2(eps.x, 0.0), period, seed);
    let h_up = bark_height(uv + vec2(0.0, eps.y), period, seed);

    let dx = (h_right - h_center) / eps.x;
    let dy = (h_up - h_center) / eps.y;

    return normalize(vec3(-dx * 0.5, -dy * 0.5, 1.0));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(output);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }

    let uv = vec2<f32>(f32(gid.x), f32(gid.y)) / vec2<f32>(f32(size.x), f32(size.y));
    let texel_size = 1.0 / vec2<f32>(f32(size.x), f32(size.y));
    let seed_f = f32(params.seed) * 0.1237;

    // Scale determines tiling frequency
    let base_scale = params.scale * 4.0;
    let scaled_uv = uv * base_scale;
    let period = vec2(base_scale);

    // Get bark patterns
    let fissures = bark_fissures(scaled_uv, period, seed_f);
    let ridges = bark_ridges(scaled_uv, period, seed_f);
    let h_cracks = horizontal_cracks(scaled_uv, period, seed_f);
    let detail = surface_detail(scaled_uv, period, seed_f);

    // Combined depth: 0 = deep fissure, 1 = ridge surface
    let depth = fissures * h_cracks;
    let surface_height = ridges * depth;

    // Base bark colors
    let bark_deep = vec3(0.08, 0.05, 0.03);      // Deep fissure color
    let bark_dark = vec3(0.18, 0.12, 0.07);      // Shadow areas
    let bark_mid = vec3(0.32, 0.22, 0.14);       // Mid-tone
    let bark_light = vec3(0.45, 0.33, 0.22);     // Highlights

    // Tint with user colors
    let user_tint = mix(params.color_primary.rgb, params.color_secondary.rgb, 0.5);
    let tint_strength = 0.25;
    let tinted_dark = mix(bark_dark, bark_dark * user_tint * 2.5, tint_strength);
    let tinted_mid = mix(bark_mid, bark_mid * user_tint * 2.0, tint_strength);
    let tinted_light = mix(bark_light, bark_light * user_tint * 1.8, tint_strength);

    // Large scale color variation
    let large_var = fbm_tiled(scaled_uv * 0.7 + seed_f * 5.0, period * 0.7, 3);

    // Build base color from depth
    var color = bark_deep;
    color = mix(color, tinted_dark, smoothstep(0.0, 0.3, depth));
    color = mix(color, tinted_mid, smoothstep(0.3, 0.6, depth));
    color = mix(color, tinted_light, smoothstep(0.6, 0.95, surface_height));

    // Apply large scale variation
    color = mix(color * 0.85, color * 1.1, large_var);

    // Ridge highlighting
    let ridge_highlight = pow(ridges, 2.0) * depth;
    color = mix(color, tinted_light * 1.15, ridge_highlight * 0.3);

    // Fissure darkening with shadow strength
    let fissure_shadow = (1.0 - fissures) * params.shadow_strength;
    color = mix(color, bark_deep, fissure_shadow * 0.7);

    // Horizontal crack darkening
    let crack_shadow = (1.0 - h_cracks) * params.shadow_strength * 0.6;
    color = mix(color, bark_deep * 1.2, crack_shadow);

    // Surface detail variation
    color *= 0.9 + detail * 0.2 * params.roughness;

    // Calculate lighting from pseudo-normal
    let normal = calc_normal(scaled_uv, period, seed_f, texel_size * base_scale);
    let light_dir = normalize(vec3(0.3, 0.5, 1.0));
    let ndotl = max(dot(normal, light_dir), 0.0);

    // Apply lighting
    let ambient = 0.4;
    let diffuse = ndotl * 0.6;
    color *= ambient + diffuse;

    // Specular/sheen on ridges
    let view_dir = vec3(0.0, 0.0, 1.0);
    let half_vec = normalize(light_dir + view_dir);
    let spec = pow(max(dot(normal, half_vec), 0.0), 20.0);
    let sheen = spec * surface_height * params.sheen_strength * (1.0 - params.roughness) * 0.3;
    color += sheen;

    // Moss/lichen in crevices based on moisture
    if (params.moisture > 0.2) {
        let moss_color = vec3(0.15, 0.25, 0.1);
        let lichen_color = vec3(0.35, 0.38, 0.3);

        // Moss grows in sheltered areas (fissures)
        let moss_mask = (1.0 - depth) * fbm_tiled(scaled_uv * 5.0 + seed_f * 7.0, period * 5.0, 3);
        let moss_amount = moss_mask * (params.moisture - 0.2) * 1.2;
        color = mix(color, moss_color, clamp(moss_amount, 0.0, 0.4));

        // Lichen patches on surfaces
        let lichen_noise = fbm_tiled(scaled_uv * 3.0 + seed_f * 11.0, period * 3.0, 4);
        let lichen_mask = smoothstep(0.55, 0.7, lichen_noise) * depth * params.moisture;
        color = mix(color, lichen_color, lichen_mask * 0.35);
    }

    // Fine grain texture
    let fine_noise = value_noise_tiled(scaled_uv * 64.0 + seed_f, period * 64.0);
    color += (fine_noise - 0.5) * 0.04 * params.roughness;

    // Micro detail (per-pixel noise, doesn't need to tile as it's imperceptible)
    let micro = hash21(uv * vec2<f32>(f32(size.x), f32(size.y)) + seed_f * 100.0);
    color += (micro - 0.5) * 0.015 * params.roughness;

    // Final color adjustment
    color = clamp(color, vec3(0.0), vec3(1.0));

    textureStore(output, vec2<i32>(gid.xy), vec4(color, 1.0));
}