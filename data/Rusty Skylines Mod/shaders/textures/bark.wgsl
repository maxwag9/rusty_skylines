
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

struct BarkSample {
    height: f32,
    plate: f32,
    edge: f32,
    bevel: f32,
    cross_cracks: f32,
    grain: f32,
    coarse: f32,
}

@group(0) @binding(0) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: Params;

fn hash22_tiled(p: vec2<f32>, period: vec2<f32>) -> vec2<f32> {
    let wrapped = ((p % period) + period) % period;
    var p3 = fract(wrapped.xyx * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += vec3<f32>(dot(p3, p3.yzx + vec3<f32>(33.33)));
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn value_noise_tiled(p: vec2<f32>, period: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let a = hash22_tiled(i, period).x;
    let b = hash22_tiled(i + vec2<f32>(1.0, 0.0), period).x;
    let c = hash22_tiled(i + vec2<f32>(0.0, 1.0), period).x;
    let d = hash22_tiled(i + vec2<f32>(1.0, 1.0), period).x;

    return mix(
        mix(a, b, u.x),
        mix(c, d, u.x),
        u.y
    );
}

fn gradient_noise_tiled(p: vec2<f32>, period: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    let g00 = hash22_tiled(i, period) * 2.0 - 1.0;
    let g10 = hash22_tiled(i + vec2<f32>(1.0, 0.0), period) * 2.0 - 1.0;
    let g01 = hash22_tiled(i + vec2<f32>(0.0, 1.0), period) * 2.0 - 1.0;
    let g11 = hash22_tiled(i + vec2<f32>(1.0, 1.0), period) * 2.0 - 1.0;

    let n00 = dot(g00, f);
    let n10 = dot(g10, f - vec2<f32>(1.0, 0.0));
    let n01 = dot(g01, f - vec2<f32>(0.0, 1.0));
    let n11 = dot(g11, f - vec2<f32>(1.0, 1.0));

    return mix(
        mix(n00, n10, u.x),
        mix(n01, n11, u.x),
        u.y
    ) * 0.5 + 0.5;
}

fn fbm_tiled(p: vec2<f32>, period: vec2<f32>, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var amplitude_sum = 0.0;
    var octave_period = period;
    var frequency = 1.0;

    for (var i = 0; i < octaves; i = i + 1) {
        value += gradient_noise_tiled(p * frequency, octave_period) * amplitude;
        amplitude_sum += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
        octave_period *= 2.0;
    }

    return value / max(amplitude_sum, 0.0001);
}

fn cellular_tiled(
    p: vec2<f32>,
    period: vec2<f32>,
    seed: f32
) -> vec2<f32> {
    let i = floor(p);
    let f = fract(p);

    var min_dist1 = 10.0;
    var min_dist2 = 10.0;

    let seed_offset = vec2<f32>(seed, seed * 1.371);

    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let neighbor = vec2<f32>(f32(x), f32(y));
            let cell = i + neighbor;

            let random = hash22_tiled(cell + seed_offset, period);
            let feature = neighbor + vec2<f32>(0.18) + random * 0.64;

            let distance = length(feature - f);

            if (distance < min_dist1) {
                min_dist2 = min_dist1;
                min_dist1 = distance;
            } else if (distance < min_dist2) {
                min_dist2 = distance;
            }
        }
    }

    return vec2<f32>(min_dist1, min_dist2);
}


fn warp_bark_uv(
    p: vec2<f32>,
    period: vec2<f32>,
    seed: f32
) -> vec2<f32> {
    let wx = fbm_tiled(
        p * 2.0 + vec2<f32>(seed, seed * 0.73),
        period * 2.0,
        3
    );

    let wy = fbm_tiled(
        p * 2.0 + vec2<f32>(seed * 1.91 + 17.0, seed * 2.37 + 41.0),
        period * 2.0,
        3
    );

    return p + (vec2<f32>(wx, wy) - vec2<f32>(0.5))
        * vec2<f32>(0.07, 0.14);
}



fn bark_sample(
    p: vec2<f32>,
    period: vec2<f32>,
    seed: f32
) -> BarkSample {
    let domain = warp_bark_uv(p, period, seed);

    let plate_uv = domain * vec2<f32>(12.0, 1.0);
    let plate_period = period * vec2<f32>(12.0, 1.0);

    let cells = cellular_tiled(
        plate_uv,
        plate_period,
        seed
    );

    let edge = max(cells.y - cells.x, 0.0);

    let plate = smoothstep(0.045, 0.22, edge);

    let bevel = smoothstep(0.025, 0.11, edge)
        * (1.0 - smoothstep(0.16, 0.32, edge));

    let cross_field = fbm_tiled(
        domain * vec2<f32>(1.0, 3.0) + seed * 1.73,
        period * vec2<f32>(1.0, 3.0),
        3
    );

    let cross_line = 1.0 - smoothstep(
        0.018,
        0.055,
        abs(cross_field - 0.5)
    );

    let cross_breakup = fbm_tiled(
        domain * vec2<f32>(12.0, 2.0) + seed * 3.17,
        period * vec2<f32>(12.0, 2.0),
        2
    );

    let broken_segments = smoothstep(0.38, 0.62, cross_breakup);
    let cross_cracks = cross_line * broken_segments * plate;

    let grain = fbm_tiled(
        domain * vec2<f32>(32.0, 3.0) + seed,
        period * vec2<f32>(32.0, 3.0),
        3
    );

    let coarse = fbm_tiled(
        domain * vec2<f32>(2.0, 2.0) + seed * 4.13,
        period * vec2<f32>(2.0, 2.0),
        3
    );

    let height = plate * 0.78
        + (grain - 0.5) * 0.045
        - cross_cracks * 0.20
        + (coarse - 0.5) * 0.06;

    return BarkSample(
        height,
        plate,
        edge,
        bevel,
        cross_cracks,
        grain,
        coarse
    );
}


fn calc_normal(
    p: vec2<f32>,
    period: vec2<f32>,
    seed: f32,
    texel_size: vec2<f32>
) -> vec3<f32> {
    let eps = texel_size * 1.5;

    let center = bark_sample(p, period, seed);
    let right = bark_sample(p + vec2<f32>(eps.x, 0.0), period, seed);
    let up = bark_sample(p + vec2<f32>(0.0, eps.y), period, seed);

    let dx = (right.height - center.height) / max(eps.x, 0.00001);
    let dy = (up.height - center.height) / max(eps.y, 0.00001);

    return normalize(vec3<f32>(
        -dx * 0.016,
        -dy * 0.016,
        1.0
    ));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(output);

    if (gid.x >= size.x || gid.y >= size.y) {
        return;
    }

    let size_f = vec2<f32>(size);
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / size_f;
    let texel_size = 1.0 / size_f;

    let seed_f = f32(params.seed % 4096u) * 0.1237;

    let base_scale = max(floor(abs(params.scale) * 4.0 + 0.5), 1.0);
    let scaled_uv = uv * base_scale;
    let period = vec2<f32>(base_scale);

    let sample = bark_sample(scaled_uv, period, seed_f);

    let roughness = clamp(params.roughness, 0.0, 1.0);
    let shadow_strength = clamp(params.shadow_strength, 0.0, 1.5);
    let moisture = clamp(params.moisture, 0.0, 1.0);

    let bark_deep = vec3<f32>(0.035, 0.030, 0.025);
    let bark_groove = vec3<f32>(0.105, 0.087, 0.068);
    let bark_mid = vec3<f32>(0.275, 0.255, 0.215);
    let bark_light = vec3<f32>(0.395, 0.370, 0.310);

    var color = mix(bark_groove, bark_mid, sample.plate);

    color = mix(
        color,
        bark_light,
        sample.bevel * 0.18
    );

    let fissure = 1.0 - smoothstep(0.015, 0.22, sample.edge);

    color = mix(
        color,
        bark_deep,
        fissure * shadow_strength * 0.85
    );

    color = mix(
        color,
        bark_deep * 1.2,
        sample.cross_cracks * shadow_strength * 0.45
    );

    let large_variation = mix(0.84, 1.12, sample.coarse);
    color *= large_variation;

    let grain_variation = (sample.grain - 0.5) * 0.20 * roughness;
    color *= 1.0 + grain_variation;

    let user_tint = mix(
        params.color_primary.rgb,
        params.color_secondary.rgb,
        0.5
    );

    let tint_multiplier = mix(
        vec3<f32>(1.0),
        clamp(user_tint * 2.0, vec3<f32>(0.65), vec3<f32>(1.25)),
        0.12
    );

    color *= tint_multiplier;

    if (moisture > 0.05) {
        let moss_noise = fbm_tiled(
            scaled_uv * vec2<f32>(8.0, 6.0) + seed_f * 7.0,
            period * vec2<f32>(8.0, 6.0),
            3
        );

        let moss_mask = smoothstep(0.62, 0.78, moss_noise)
            * fissure
            * moisture;

        color = mix(
            color,
            vec3<f32>(0.105, 0.145, 0.075),
            moss_mask * 0.35
        );

        let lichen_noise = fbm_tiled(
            scaled_uv * vec2<f32>(3.0, 4.0) + seed_f * 11.0,
            period * vec2<f32>(3.0, 4.0),
            4
        );

        let lichen_mask = smoothstep(0.66, 0.82, lichen_noise)
            * sample.plate
            * moisture;

        color = mix(
            color,
            vec3<f32>(0.42, 0.43, 0.36),
            lichen_mask * 0.22
        );
    }

    let normal = calc_normal(
        scaled_uv,
        period,
        seed_f,
        texel_size * base_scale
    );

    let light_dir = normalize(vec3<f32>(-0.35, 0.45, 0.82));
    let ndotl = max(dot(normal, light_dir), 0.0);

    color *= 0.72 + 0.28 * ndotl;

    let view_dir = vec3<f32>(0.0, 0.0, 1.0);
    let half_vec = normalize(light_dir + view_dir);

    let spec = pow(max(dot(normal, half_vec), 0.0), 32.0);

    let sheen = spec
        * sample.plate
        * clamp(params.sheen_strength, 0.0, 1.0)
        * (1.0 - roughness)
        * 0.08;

    color += vec3<f32>(sheen);

    let fine_noise = value_noise_tiled(
        scaled_uv * 64.0 + seed_f,
        period * 64.0
    );

    color += (fine_noise - 0.5) * 0.022 * roughness;

    let micro = hash22_tiled(
        vec2<f32>(gid.xy),
        size_f
    ).x;

    color += (micro - 0.5) * 0.012 * roughness;

    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));

    textureStore(output, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
}
