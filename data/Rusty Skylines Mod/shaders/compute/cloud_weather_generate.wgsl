struct Params {
    size: u32,
    seed: u32,
    frequency: u32,
    padding: u32,
};

@group(1) @binding(0)
var output_weather: texture_storage_2d<rgba8unorm, write>;

@group(2) @binding(0)
var<uniform> params: Params;

fn hash_u32(value: u32) -> u32 {
    var state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn positive_mod(value: i32, divisor: i32) -> i32 {
    let r = value % divisor;
    return select(r, r + divisor, r < 0);
}

fn random_cell2(cell: vec2<i32>, period: u32, seed: u32) -> f32 {
    let p = i32(period);
    let x = u32(positive_mod(cell.x, p));
    let y = u32(positive_mod(cell.y, p));
    let h = hash_u32(x * 374761393u ^ y * 668265263u ^ seed * 3266489917u);
    return f32(h >> 8u) / 16777215.0;
}

fn value_noise2(p: vec2<f32>, frequency: u32, seed: u32) -> f32 {
    let q = p * f32(frequency);
    let base = vec2<i32>(floor(q));
    let f = fract(q);
    let t = f * f * (vec2<f32>(3.0) - 2.0 * f);
    let a = random_cell2(base, frequency, seed);
    let b = random_cell2(base + vec2<i32>(1, 0), frequency, seed);
    let c = random_cell2(base + vec2<i32>(0, 1), frequency, seed);
    let d = random_cell2(base + vec2<i32>(1, 1), frequency, seed);
    return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
}

fn fbm2(p: vec2<f32>, frequency: u32, seed: u32) -> f32 {
    let n0 = value_noise2(p, frequency, seed);
    let n1 = value_noise2(p, frequency * 2u, seed + 11u);
    let n2 = value_noise2(p, frequency * 4u, seed + 23u);
    let n3 = value_noise2(p, frequency * 8u, seed + 37u);
    return n0 * 0.5 + n1 * 0.25 + n2 * 0.15 + n3 * 0.1;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output_weather);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let p = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / f32(params.size);

    let warp_x = value_noise2(p, 4u, params.seed + 101u) - 0.5;
    let warp_y = value_noise2(p, 4u, params.seed + 211u) - 0.5;
    let warped = p + vec2<f32>(warp_x, warp_y) * 0.10;

    let broad = fbm2(warped, params.frequency, params.seed);
    let detail = fbm2(warped, params.frequency * 2u, params.seed + 307u);
    let coverage = clamp(broad * 0.75 + detail * 0.25, 0.0, 1.0);

    let cloud_type = value_noise2(p, 3u, params.seed + 401u);
    let humidity = fbm2(p, 4u, params.seed + 503u);

    // An independent high-cloud weather field. Cirrus coverage must not depend
    // on selecting the rare cirrus interval in the main cloud-type channel.
    let cirrus_warp = vec2<f32>(
        value_noise2(p, 3u, params.seed + 601u) - 0.5,
        value_noise2(p, 3u, params.seed + 607u) - 0.5
    ) * 0.12;
    let cirrus_domain = p + cirrus_warp;
    let cirrus_broad = fbm2(cirrus_domain, 2u, params.seed + 613u);
    let cirrus_detail = fbm2(cirrus_domain, 4u, params.seed + 617u);
    let cirrus_coverage = clamp(cirrus_broad * 0.78 + cirrus_detail * 0.22, 0.0, 1.0);

    textureStore(
        output_weather,
        vec2<i32>(gid.xy),
        vec4<f32>(coverage, cloud_type, humidity, cirrus_coverage)
    );
}
