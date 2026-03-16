// leaves.wgsl - Scattered leaf card texture with many leaves
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

fn rotate2d(p: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2(p.x * c - p.y * s, p.x * s + p.y * c);
}

// Returns vec3(alpha, vein_intensity, tip_t)
fn leaf_shape(p: vec2<f32>, seed: f32) -> vec3<f32> {
    let t = (p.y + 1.0) * 0.5;

    // Leaf width - pointed oval shape
    let base_width = sin(clamp(t, 0.0, 1.0) * 3.14159) * 0.35;
    let edge_noise = value_noise(vec2(t * 12.0, seed)) * 0.04 * base_width;
    let serration = sin(t * 20.0 + seed * 4.0) * 0.01 * base_width;
    let width = base_width + edge_noise + serration;

    let alpha = smoothstep(width + 0.02, width - 0.015, abs(p.x));
    let v_mask = smoothstep(-0.92, -0.8, p.y) * smoothstep(0.92, 0.8, p.y);

    // Midrib
    let midrib = exp(-abs(p.x) * 50.0);

    // Secondary veins
    var secondary = 0.0;
    for (var i = 0; i < 5; i++) {
        let fi = f32(i);
        let vein_y = -0.5 + fi * 0.25;
        let dy = p.y - vein_y;
        if (dy > 0.0 && dy < 0.18) {
            let expected_x = dy * 0.55;
            let dist = abs(abs(p.x) - expected_x);
            secondary = max(secondary, exp(-dist * 60.0) * smoothstep(0.18, 0.01, dy));
        }
    }

    return vec3(alpha * v_mask, midrib + secondary * 0.5, t);
}

fn sample_leaf(uv: vec2<f32>, leaf_pos: vec2<f32>, leaf_rot: f32, leaf_scale: f32, leaf_seed: f32) -> vec4<f32> {
    var local = uv - leaf_pos;
    local = rotate2d(local, leaf_rot);
    local = local / leaf_scale;

    if (abs(local.x) > 0.5 || abs(local.y) > 1.05) {
        return vec4(0.0);
    }

    let shape = leaf_shape(local, leaf_seed);
    let alpha = shape.x;

    if (alpha < 0.01) {
        return vec4(0.0);
    }

    let vein = shape.y;
    let t = shape.z;

    // Color variation per leaf
    let color_var = hash21(vec2(leaf_seed, leaf_seed * 1.7));
    var color = mix(params.color_primary.rgb, params.color_secondary.rgb, color_var * 0.5 + fbm(local * 4.0) * 0.15);

    // Vein darkening
    color = mix(color, color * 0.45, vein * 0.3);

    // Center translucency
    color += (1.0 - abs(local.x) * 2.5) * params.moisture * 0.06;

    // Tip/edge aging
    let age = hash21(vec2(leaf_seed * 2.0, 0.0));
    color = mix(color, vec3(0.5, 0.45, 0.25), smoothstep(0.65, 0.95, t) * age * 0.2);

    // Highlight
    let highlight = exp(-length(local - vec2(-0.1, 0.15)) * 3.0) * params.sheen_strength * 0.1;
    color += highlight * (1.0 - params.roughness);

    // Fine texture
    color += (hash21(uv * 250.0 + leaf_seed) - 0.5) * 0.025 * params.roughness;

    return vec4(clamp(color, vec3(0.0), vec3(1.0)), alpha);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(output);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }

    let uv = vec2<f32>(f32(gid.x), f32(gid.y)) / vec2<f32>(f32(size.x), f32(size.y));
    let seed_f = f32(params.seed);

    // Grid density controlled by scale
    let grid_size = 10.0 * params.scale;
    let cell_size = 1.0 / grid_size;

    var final_color = vec3(0.0);
    var final_alpha = 0.0;

    // Multiple depth layers
    let num_layers = 4;
    let leaves_per_cell = 3;

    for (var layer = 0; layer < num_layers; layer++) {
        let layer_seed = seed_f + f32(layer) * 137.0;
        let cell = floor(uv * grid_size);

        // Check 3x3 neighborhood of cells
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let check_cell = cell + vec2<f32>(f32(dx), f32(dy));

                for (var leaf_idx = 0; leaf_idx < leaves_per_cell; leaf_idx++) {
                    let idx_f = f32(leaf_idx);
                    let cell_key = check_cell + vec2(layer_seed, idx_f * 31.0);

                    // Skip some cells randomly for variation
                    if (hash21(cell_key * 0.7) > 0.75) {
                        continue;
                    }

                    let rand1 = hash22(cell_key * 1.1);
                    let rand2 = hash22(cell_key * 2.3);
                    let rand3 = hash21(cell_key * 3.7);

                    // Leaf position with jitter beyond cell
                    let leaf_pos = (check_cell + 0.5 + (rand1 - 0.5) * 1.4) * cell_size;
                    let leaf_rot = rand2.x * 6.28318;
                    let leaf_scale = (0.04 + rand2.y * 0.04) / params.scale;
                    let leaf_seed = rand3 * 100.0;

                    let leaf_result = sample_leaf(uv, leaf_pos, leaf_rot, leaf_scale, leaf_seed);

                    if (leaf_result.a > 0.01) {
                        // Self-shadowing from leaves above
                        let shadow = final_alpha * params.shadow_strength * 0.2;
                        let shaded_color = leaf_result.rgb * (1.0 - shadow);

                        // Alpha composite
                        final_color = mix(final_color, shaded_color, leaf_result.a);
                        final_alpha = final_alpha + leaf_result.a * (1.0 - final_alpha);
                    }
                }
            }
        }
    }

    textureStore(output, vec2<i32>(gid.xy), vec4(final_color, final_alpha));
}