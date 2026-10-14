struct Params {
    size: u32,
    seed: u32,
    frequency: u32,
    padding: u32,
};

@group(1) @binding(0)
var output_volume: texture_storage_3d<rgba8unorm, write>;

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

fn random_cell(cell: vec3<i32>, period: u32, seed: u32) -> f32 {
    let p = i32(period);
    let x = u32(positive_mod(cell.x, p));
    let y = u32(positive_mod(cell.y, p));
    let z = u32(positive_mod(cell.z, p));

    let h = hash_u32(
        x * 374761393u ^
        y * 668265263u ^
        z * 2246822519u ^
        seed * 3266489917u
    );

    return f32(h >> 8u) / 16777215.0;
}
fn random_feature3(cell: vec3<i32>, period: u32, seed: u32) -> vec3<f32> {
    let p = i32(period);
    let x = u32(positive_mod(cell.x, p));
    let y = u32(positive_mod(cell.y, p));
    let z = u32(positive_mod(cell.z, p));

    let h = hash_u32(
        x * 374761393u ^
        y * 668265263u ^
        z * 2246822519u ^
        seed * 3266489917u
    );

    return vec3<f32>(
        f32(h & 1023u),
        f32((h >> 10u) & 1023u),
        f32((h >> 20u) & 1023u)
    ) / 1024.0;
}
fn value_noise3(p: vec3<f32>, frequency: u32, seed: u32) -> f32 {
    let q = p * f32(frequency);
    let base = vec3<i32>(floor(q));
    let f = fract(q);
    let t = f * f * (vec3<f32>(3.0) - 2.0 * f);

    let x00 = mix(
        random_cell(base, frequency, seed),
        random_cell(base + vec3<i32>(1, 0, 0), frequency, seed),
        t.x
    );
    let x10 = mix(
        random_cell(base + vec3<i32>(0, 1, 0), frequency, seed),
        random_cell(base + vec3<i32>(1, 1, 0), frequency, seed),
        t.x
    );
    let x01 = mix(
        random_cell(base + vec3<i32>(0, 0, 1), frequency, seed),
        random_cell(base + vec3<i32>(1, 0, 1), frequency, seed),
        t.x
    );
    let x11 = mix(
        random_cell(base + vec3<i32>(0, 1, 1), frequency, seed),
        random_cell(base + vec3<i32>(1, 1, 1), frequency, seed),
        t.x
    );

    return mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
}

fn worley3(p: vec3<f32>, frequency: u32, seed: u32) -> f32 {
    let q = p * f32(frequency);
    let base = vec3<i32>(floor(q));
    let f = fract(q);
    var nearest = 1000.0;

    for (var z = -1; z <= 1; z = z + 1) {
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let offset = vec3<i32>(x, y, z);
                let cell = base + offset;

                let jitter = random_feature3(cell, frequency, seed);

                let delta = vec3<f32>(offset) + jitter - f;
                nearest = min(nearest, dot(delta, delta));
            }
        }
    }

    return clamp(sqrt(nearest) / 1.7320508, 0.0, 1.0);
}

fn fbm3(p: vec3<f32>, frequency: u32, seed: u32) -> f32 {
    let n0 = value_noise3(p, frequency, seed);
    let n1 = value_noise3(p, frequency * 2u, seed + 11u);
    let n2 = value_noise3(p, frequency * 4u, seed + 23u);
    let n3 = value_noise3(p, frequency * 8u, seed + 37u);

    return n0 * 0.5 + n1 * 0.25 + n2 * 0.15 + n3 * 0.1;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output_volume);

    if (any(gid >= dims)) {
        return;
    }

    let p = (vec3<f32>(gid) + vec3<f32>(0.5)) / f32(params.size);
    var result: vec4<f32>;

    if (params.frequency <= 4u) {
        result = vec4<f32>(
            fbm3(p, params.frequency, params.seed),
            worley3(p, params.frequency, params.seed + 101u),
            worley3(p, params.frequency * 2u, params.seed + 211u),
            worley3(p, params.frequency * 4u, params.seed + 307u)
        );
    } else {
        result = vec4<f32>(
            worley3(p, params.frequency, params.seed + 401u),
            worley3(p, params.frequency * 2u, params.seed + 503u),
            worley3(p, params.frequency * 4u, params.seed + 601u),
            fbm3(p, params.frequency, params.seed + 701u)
        );
    }

    textureStore(output_volume, vec3<i32>(gid), result);
}