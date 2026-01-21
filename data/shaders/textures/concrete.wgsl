// concrete.wgsl
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

    let a = hash21(i);
    let b = hash21(i + vec2(1.0, 0.0));
    let c = hash21(i + vec2(0.0, 1.0));
    let d = hash21(i + vec2(1.0, 1.0));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p: vec2<f32>, octaves: i32) -> f32 {
    var value: f32 = 0.0;
    var amplitude: f32 = 0.5;
    var pp = p;
    for (var i: i32 = 0; i < octaves; i = i + 1) {
        value += amplitude * value_noise(pp);
        pp *= 2.03;
        pp += vec2(100.0, 50.0);
        amplitude *= 0.5;
    }
    return value;
}

// Voronoi for aggregate stones
fn voronoi(p: vec2<f32>) -> vec3<f32> {
    let n = floor(p);
    let f = fract(p);

    var min_dist: f32 = 8.0;
    var cell_id: f32 = 0.0;

    for (var j: i32 = -1; j <= 1; j++) {
        for (var i: i32 = -1; i <= 1; i++) {
            let g = vec2<f32>(f32(i), f32(j));
            let o = hash22(n + g);
            let r = g + o - f;
            let d = dot(r, r);

            if (d < min_dist) {
                min_dist = d;
                cell_id = hash21(n + g);
            }
        }
    }

    return vec3(sqrt(min_dist), cell_id, 0.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(output);
    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }

    let uv = vec2<f32>(f32(gid.x), f32(gid.y)) / f32(min(size.x, size.y));
    let seed_offset = vec2<f32>(f32(params.seed) * 7.234, f32(params.seed) * 3.891);
    let p = (uv + seed_offset) * params.scale;

    // Fine sand/cement grain
    let fine_grain = fbm(p * 20.0, 4) * 0.15;

    // Medium texture variation
    let medium_tex = fbm(p * 5.0, 3) * 0.25;

    // Large scale weathering/staining
    let weathering = fbm(p * 1.5, 2) * 0.2;

    // Aggregate stones (visible gravel)
    let agg = voronoi(p * 6.0);
    let stone_mask = smoothstep(0.08, 0.15, agg.x);
    let stone_color_var = agg.y * 0.15 - 0.075; // per-stone color variation

    // Tiny pores/air bubbles
    let pores = voronoi(p * 25.0);
    let pore_dark = smoothstep(0.02, 0.05, pores.x);

    // Combine base texture
    let base_variation = fine_grain + medium_tex + weathering;

    // Base concrete color with variation
    var color = mix(params.color_primary.rgb, params.color_secondary.rgb,
                    base_variation * params.roughness);

    // Add aggregate color variation (some stones lighter, some darker)
    color = mix(color + stone_color_var, color, stone_mask);

    // Darken pores slightly
    color *= 0.92 + pore_dark * 0.08;

    // Add very fine sparkle (ite/sand glitter)
    let sparkle = pow(hash21(p * 50.0), 8.0) * 0.08;
    color += sparkle;

    textureStore(output, vec2<i32>(gid.xy), vec4(clamp(color, vec3(0.0), vec3(1.0)), 1.0));
}