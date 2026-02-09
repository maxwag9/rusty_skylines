// gtao_upsample_apply.wgsl
struct CameraUniforms {
    view:                mat4x4<f32>,
    inv_view:            mat4x4<f32>,
    proj:                mat4x4<f32>,
    inv_proj:            mat4x4<f32>,
    view_proj:           mat4x4<f32>,
    inv_view_proj:       mat4x4<f32>,
    lighting_view_proj:  array<mat4x4<f32>, 4>,
    cascade_splits:      vec4<f32>,
    sun_direction:       vec3<f32>,
    time:                f32,
    camera_local:        vec3<f32>,
    chunk_size:          f32,
    camera_chunk:        vec2<i32>,
    _pad_cam:            vec2<i32>,
    moon_direction:      vec3<f32>,
    orbit_radius:        f32,
    reversed_depth_z:    u32,
    shadows_enabled:     u32,
    near_far_depth:      vec2<f32>,
};

struct UpsampleApplyParams {
    full_size:        vec2<f32>,
    half_size:        vec2<f32>,
    inv_full_size:    vec2<f32>,
    inv_half_size:    vec2<f32>,
    depth_threshold:  f32,
    normal_threshold: f32,
    use_normal_check: u32,
    power:            f32,
    apply_intensity:  f32,
    min_ao:           f32,
    debug_mode:       u32,
    _padding:         u32,
};

@group(0) @binding(0) var trilinear_sampler: sampler;
@group(0) @binding(1) var ao_half:      texture_2d<f32>;
@group(0) @binding(2) var depth_half:   texture_2d<f32>;
@group(0) @binding(3) var normals_half: texture_2d<f32>;
#ifdef MSAA
@group(0) @binding(4) var depth_full_raw: texture_depth_multisampled_2d;
#else
@group(0) @binding(4) var depth_full_raw: texture_depth_2d;
#endif
@group(0) @binding(5) var normals_full: texture_2d<f32>;

@group(1) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(1) var<uniform> params: UpsampleApplyParams;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 2);
    let y = f32(i32(vertex_index >> 1u) * 2);
    out.uv = vec2<f32>(x, 1.0 - y);
    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

fn decode_normal(encoded: vec3<f32>) -> vec3<f32> {
    return normalize(encoded * 2.0 - 1.0);
}

fn linearize_depth(d: f32, z_near: f32, z_far: f32, reversed: bool) -> f32 {
    if (reversed) {
        if (d <= 0.000001) { return z_far; }
        return z_near / d;
    } else {
        return (z_near * z_far) / (z_far - d * (z_far - z_near));
    }
}

fn load_full_depth_linear(coords: vec2<i32>) -> f32 {
    let z_near = camera.near_far_depth.x;
    let z_far  = camera.near_far_depth.y;
    let is_rev = camera.reversed_depth_z != 0u;
    let raw = textureLoad(depth_full_raw, coords, 0);
    return linearize_depth(raw, z_near, z_far, is_rev);
}

fn compute_depth_weight(full_d: f32, half_d: f32, threshold: f32) -> f32 {
    let avg = (full_d + half_d) * 0.5;
    let rel = abs(full_d - half_d) / max(avg, 0.001);
    return exp(-rel * rel / (2.0 * threshold * threshold));
}

fn compute_normal_weight(full_n: vec3<f32>, half_n: vec3<f32>, threshold: f32) -> f32 {
    let n_dot = max(0.0, dot(full_n, half_n));
    let diff  = max(0.0, (threshold - n_dot) / threshold);
    return exp(-diff * diff * 8.0);
}

fn clamp_coords(c: vec2<i32>, dims: vec2<i32>) -> vec2<i32> {
    return clamp(c, vec2<i32>(0), dims - 1);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let full_coords = vec2<i32>(floor(in.position.xy));
    let full_dims   = vec2<i32>(params.full_size);

    let full_depth  = load_full_depth_linear(full_coords);
    let full_normal = decode_normal(textureLoad(normals_full, full_coords, 0).rgb);

    if (full_depth > 10000.0) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }

    let full_uv  = (vec2<f32>(full_coords) + 0.5) * params.inv_full_size;
    let half_pos = full_uv * params.half_size - 0.5;

    let half_base = vec2<i32>(floor(half_pos));
    let frac      = half_pos - vec2<f32>(half_base);
    let half_dims = vec2<i32>(textureDimensions(ao_half));

    let p00 = clamp_coords(half_base,                   half_dims);
    let p10 = clamp_coords(half_base + vec2<i32>(1, 0), half_dims);
    let p01 = clamp_coords(half_base + vec2<i32>(0, 1), half_dims);
    let p11 = clamp_coords(half_base + vec2<i32>(1, 1), half_dims);

    let ao00 = textureLoad(ao_half, p00, 0).r;
    let ao10 = textureLoad(ao_half, p10, 0).r;
    let ao01 = textureLoad(ao_half, p01, 0).r;
    let ao11 = textureLoad(ao_half, p11, 0).r;

    let d00 = textureLoad(depth_half, p00, 0).r;
    let d10 = textureLoad(depth_half, p10, 0).r;
    let d01 = textureLoad(depth_half, p01, 0).r;
    let d11 = textureLoad(depth_half, p11, 0).r;

    let dw00 = compute_depth_weight(full_depth, d00, params.depth_threshold);
    let dw10 = compute_depth_weight(full_depth, d10, params.depth_threshold);
    let dw01 = compute_depth_weight(full_depth, d01, params.depth_threshold);
    let dw11 = compute_depth_weight(full_depth, d11, params.depth_threshold);

    var nw00: f32 = 1.0; var nw10: f32 = 1.0;
    var nw01: f32 = 1.0; var nw11: f32 = 1.0;

    if (params.use_normal_check != 0u) {
        let n00 = decode_normal(textureLoad(normals_half, p00, 0).rgb);
        let n10 = decode_normal(textureLoad(normals_half, p10, 0).rgb);
        let n01 = decode_normal(textureLoad(normals_half, p01, 0).rgb);
        let n11 = decode_normal(textureLoad(normals_half, p11, 0).rgb);

        nw00 = compute_normal_weight(full_normal, n00, params.normal_threshold);
        nw10 = compute_normal_weight(full_normal, n10, params.normal_threshold);
        nw01 = compute_normal_weight(full_normal, n01, params.normal_threshold);
        nw11 = compute_normal_weight(full_normal, n11, params.normal_threshold);
    }

    let bw00 = (1.0 - frac.x) * (1.0 - frac.y);
    let bw10 = frac.x          * (1.0 - frac.y);
    let bw01 = (1.0 - frac.x) * frac.y;
    let bw11 = frac.x          * frac.y;

    let w00 = bw00 * dw00 * nw00;
    let w10 = bw10 * dw10 * nw10;
    let w01 = bw01 * dw01 * nw01;
    let w11 = bw11 * dw11 * nw11;

    let total_w = w00 + w10 + w01 + w11;

    var raw_ao: f32;
    if (total_w > 0.0001) {
        let weighted_ao = (ao00 * w00 + ao10 * w10 + ao01 * w01 + ao11 * w11) / total_w;
        let confidence  = min(total_w / (bw00 + bw10 + bw01 + bw11), 1.0);
        raw_ao = mix(1.0, weighted_ao, confidence);
    } else {
        raw_ao = 1.0;
    }

    let powered_ao = pow(raw_ao, params.power);
    let final_ao   = max(mix(1.0, powered_ao, params.apply_intensity), params.min_ao);

    switch (params.debug_mode) {
        case 1u: {
            return vec4<f32>(final_ao, final_ao, final_ao, 1.0);
        }
        case 2u: {
            let v = vec3<f32>(final_ao, final_ao * final_ao, final_ao * final_ao);
            return vec4<f32>(v, 1.0);
        }
        case 3u: {
            return vec4<f32>(raw_ao, raw_ao, raw_ao, 1.0);
        }
        default: {
            return vec4<f32>(final_ao, final_ao, final_ao, 1.0);
        }
    }
}