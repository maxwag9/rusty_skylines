#include "../includes/uniforms.wgsl"
// ============================================================================
// GTAO Generate + Temporal Accumulate
// ============================================================================
// Computes half-res AO then reprojects and blends with previous frame's AO.
// Successive frames use different blue-noise offsets (via frame_index),
// and the exponential moving average cancels the noise over time.
// ============================================================================

const PI: f32 = 3.14159265359;
const HALF_PI: f32 = 1.57079632679;

const NUM_DIRECTIONS: u32 = 4u;
const NUM_STEPS: u32 = 4u;
const BASE_DIRS_4: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>( 1.0,  0.0),
    vec2<f32>( 0.70710678, 0.70710678),
    vec2<f32>( 0.0,  1.0),
    vec2<f32>(-0.70710678, 0.70710678),
);
const AO_FADE_START_FAR_RATIO: f32 = 0.05;
const AO_FADE_END_FAR_RATIO:   f32 = 0.10;

struct AOParams {
    radius_world:    f32,
    intensity:       f32,
    bias:            f32,
    frame_index:     u32,
    screen_size:     vec2<f32>,
    inv_screen_size: vec2<f32>,
    temporal_blend:  f32,       // 0.05 = keep 95% history, 1.0 = no history
    _pad0:           u32,
    _pad1:           u32,
    _pad2:           u32,
    prev_view_proj:  mat4x4<f32>,
};

@group(2) @binding(0) var<uniform> camera:    Uniforms;
@group(2) @binding(1) var<uniform> ao_params: AOParams;

@group(0) @binding(0) var linear_depth_half: texture_2d<f32>;
@group(0) @binding(1) var normals_half:      texture_2d<f32>;
@group(0) @binding(2) var blue_noise_tex:    texture_2d<f32>;
@group(0) @binding(3) var ao_history:        texture_2d<f32>; // prev frame

@group(1) @binding(0) var ao_output: texture_storage_2d<r32float, write>;

// Helpers
fn decode_normal(encoded: vec3<f32>) -> vec3<f32> {
    return normalize(encoded * 2.0 - 1.0);
}

fn reconstruct_view_position(uv: vec2<f32>, linear_z: f32) -> vec3<f32> {
    // uv (0,0) top-left -> NDC (-1,+1) top-left
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);

    // Account for asymmetric/jittered projection offsets (if any)
    let off = vec2<f32>(camera.proj[2][0], camera.proj[2][1]);

    let inv_px = 1.0 / camera.proj[0][0];
    let inv_py = 1.0 / camera.proj[1][1];

    let x = (ndc.x - off.x) * linear_z * inv_px;
    let y = (ndc.y - off.y) * linear_z * inv_py;

    return vec3<f32>(x, y, -linear_z);
}

fn sample_blue_noise(pixel_coords: vec2<i32>, frame: u32) -> vec4<f32> {
    let sz_u: vec2<u32> = textureDimensions(blue_noise_tex);
    let sz   = vec2<i32>(i32(sz_u.x), i32(sz_u.y));

    let off = vec2<i32>(
        i32((frame * 7u)  % u32(sz.x)),
        i32((frame * 11u) % u32(sz.y))
    );

    return textureLoad(blue_noise_tex, (pixel_coords + off) % sz, 0);
}

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

fn dims_i32_tex2d(t: texture_2d<f32>) -> vec2<i32> {
    let d: vec2<u32> = textureDimensions(t);
    return vec2<i32>(i32(d.x), i32(d.y));
}

fn safe_normalize2(v: vec2<f32>) -> vec2<f32> {
    let lsq = dot(v, v);
    if (lsq <= 1e-8) { return vec2<f32>(1.0, 0.0); }
    return v * inverseSqrt(lsq);
}

fn get_rotation(noise: vec4<f32>) -> vec2<f32> {
    return safe_normalize2(vec2<f32>(noise.g * 2.0 - 1.0, noise.b * 2.0 - 1.0));
}

fn rotate_dir(d: vec2<f32>, cs: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(d.x * cs.x - d.y * cs.y,
                     d.x * cs.y + d.y * cs.x);
}

fn compute_falloff(dist_sq: f32, radius_sq: f32) -> f32 {
    return saturate(1.0 - dist_sq / radius_sq);
}


fn compute_horizon_occlusion(
    center_px:   vec2<f32>,   // pixel center in pixel units (e.g. pixel + 0.5)
    view_pos:    vec3<f32>,
    view_normal: vec3<f32>,
    screen_dir:  vec2<f32>,
    step_px:     f32,
    radius_sq:   f32,
    far_z:       f32,
    dims_f:      vec2<f32>,   // ao_params.screen_size
) -> f32 {
    var max_sin: f32 = 0.0;
    var occ:     f32 = 0.0;

    let step_vec_px = screen_dir * step_px;
    let inv_size    = ao_params.inv_screen_size;
    let bias        = ao_params.bias;

    for (var step = 1u; step <= NUM_STEPS; step++) {
        let spx = center_px + step_vec_px * f32(step);

        // bounds in pixel space (cheap)
        if (spx.x < 0.0 || spx.y < 0.0 || spx.x >= dims_f.x || spx.y >= dims_f.y) {
            continue;
        }

        // truncation == floor for positive values
        let sc = vec2<i32>(spx);

        let sd = textureLoad(linear_depth_half, sc, 0).r;
        if (sd >= far_z) { continue; }

        // reconstruct using the sampled pixel center
        let s_uv = (vec2<f32>(sc) + 0.5) * inv_size;
        let sp   = reconstruct_view_position(s_uv, sd);

        let hv  = sp - view_pos;
        let dsq = dot(hv, hv);
        if (dsq > radius_sq) { continue; }

        let inv_len = inverseSqrt(dsq);
        let hs      = dot(hv, view_normal) * inv_len;

        let fo = compute_falloff(dsq, radius_sq);
        let bh = hs - bias;

        if (bh > max_sin) {
            occ    += (bh - max_sin) * fo;
            max_sin = bh;
        }
    }

    return occ;
}

// ----------------------------------------------------------------------------
// Bilinear history sample
// ----------------------------------------------------------------------------
fn sample_history_bilinear(uv: vec2<f32>, dims: vec2<i32>) -> f32 {
    let texel = uv * vec2<f32>(dims) - 0.5;
    let base  = vec2<i32>(floor(texel));
    let f     = texel - vec2<f32>(base);

    let p00 = clamp(base,                       vec2<i32>(0), dims - 1);
    let p10 = clamp(base + vec2<i32>(1, 0),     vec2<i32>(0), dims - 1);
    let p01 = clamp(base + vec2<i32>(0, 1),     vec2<i32>(0), dims - 1);
    let p11 = clamp(base + vec2<i32>(1, 1),     vec2<i32>(0), dims - 1);

    let h00 = textureLoad(ao_history, p00, 0).r;
    let h10 = textureLoad(ao_history, p10, 0).r;
    let h01 = textureLoad(ao_history, p01, 0).r;
    let h11 = textureLoad(ao_history, p11, 0).r;

    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

// Main
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Use uniform-provided size (guaranteed correct)
    let dims_f = ao_params.screen_size;
    let dims_u = vec2<u32>(u32(dims_f.x), u32(dims_f.y));
    let dims_i = vec2<i32>(i32(dims_u.x), i32(dims_u.y));

    if (gid.x >= dims_u.x || gid.y >= dims_u.y) { return; }

    let pixel = vec2<i32>(i32(gid.x), i32(gid.y));
    let far_plane = camera.near_far_depth.y;

    // already have:
    let far_z = far_plane * 0.99;

    let linear_z = textureLoad(linear_depth_half, pixel, 0).r;

    // Sky
    if (linear_z >= far_z) {
        textureStore(ao_output, pixel, vec4<f32>(1.0, 0.0, 0.0, 1.0));
        return;
    }
    let fade_start = far_plane * AO_FADE_START_FAR_RATIO;
    let fade_end   = far_plane * AO_FADE_END_FAR_RATIO;

    // 1 near, 0 far
    let dist_fade = saturate((fade_end - linear_z) / max(fade_end - fade_start, 1e-6));

    if (dist_fade <= 0.0) {
        textureStore(ao_output, pixel, vec4<f32>(1.0, 0.0, 0.0, 1.0));
        return;
    }
    let inv_size  = ao_params.inv_screen_size;
    let center_px = vec2<f32>(pixel) + 0.5;
    let uv        = center_px * inv_size;

    let view_pos = reconstruct_view_position(uv, linear_z);

    let wn        = decode_normal(textureLoad(normals_half, pixel, 0).rgb);
    let view_norm = (camera.view * vec4<f32>(wn, 0.0)).xyz;

    let proj_scale   = camera.proj[1][1] * dims_f.y * 0.5;
    let radius_px_raw = (ao_params.radius_world * proj_scale) / max(linear_z, 1e-3);

    // If the AO footprint is subpixel, any result is basically noise → just disable AO.
    if (radius_px_raw < 1.0) {
        textureStore(ao_output, pixel, vec4<f32>(1.0, 0.0, 0.0, 1.0));
        return;
    }

    // Fade factor: 0 at 1px, 1 at 3px (tweak 3.0 if you want)
    let radius_fade = saturate((radius_px_raw - 1.0) / (3.0 - 1.0));

    // Now clamp only the MAX, not the MIN
    let radius_px = min(radius_px_raw, 128.0);
    let step_px   = radius_px / f32(NUM_STEPS);
    let radius_sq = ao_params.radius_world * ao_params.radius_world;

    let noise    = sample_blue_noise(pixel, ao_params.frame_index);
    let rotation = get_rotation(noise);

    var total_occ: f32 = 0.0;

    // No trig: use precomputed base dirs
    for (var d = 0u; d < NUM_DIRECTIONS; d++) {
        let base_dir = BASE_DIRS_4[i32(d)];
        let rd       = rotate_dir(base_dir, rotation);

        total_occ += compute_horizon_occlusion(center_px, view_pos, view_norm,  rd, step_px, radius_sq, far_z, dims_f);
        total_occ += compute_horizon_occlusion(center_px, view_pos, view_norm, -rd, step_px, radius_sq, far_z, dims_f);
    }

    total_occ /= f32(NUM_DIRECTIONS * 2u);
    let ao_unfaded = saturate(1.0 - total_occ * ao_params.intensity);
    let raw_ao_dist_faded = mix(1.0, ao_unfaded, dist_fade);

    // Fade to 1.0 when radius is tiny (kills the far speckle without needing “close” far fades)
    let raw_ao = mix(1.0, raw_ao_dist_faded, radius_fade);
    // Fast path: no temporal accumulation requested
    if (ao_params.temporal_blend >= 0.999) {
        textureStore(ao_output, pixel, vec4<f32>(raw_ao, 0.0, 0.0, 1.0));
        return;
    }

    // Temporal accumulation (unchanged logic)
    var final_ao = raw_ao;

    let world_pos = camera.inv_view * vec4<f32>(view_pos, 1.0);
    let prev_clip = ao_params.prev_view_proj * world_pos;

    if (prev_clip.w > 0.001) {
        let prev_ndc = prev_clip.xy / prev_clip.w;
        let prev_uv  = vec2<f32>(prev_ndc.x * 0.5 + 0.5,
                                 1.0 - (prev_ndc.y * 0.5 + 0.5));

        if (prev_uv.x >= 0.0 && prev_uv.x <= 1.0 &&
            prev_uv.y >= 0.0 && prev_uv.y <= 1.0) {

            let history_ao = sample_history_bilinear(prev_uv, dims_i);

            let neigh_offsets = array<vec2<i32>, 4>(
                vec2<i32>(-1,  0), vec2<i32>( 1, 0),
                vec2<i32>( 0, -1), vec2<i32>( 0, 1)
            );

            var depth_variance: f32 = 0.0;
            for (var i = 0; i < 4; i++) {
                let nc = clamp(pixel + neigh_offsets[i], vec2<i32>(0), dims_i - 1);
                let nd = textureLoad(linear_depth_half, nc, 0).r;
                let rel_diff = abs(nd - linear_z) / max(linear_z, 0.001);
                depth_variance = max(depth_variance, rel_diff);
            }

            let edge_rejection = smoothstep(0.05, 0.15, depth_variance);
            let motion_uv = length(prev_uv - uv);
            let motion_px = motion_uv * dims_f.y; // approx pixels using height
            let motion_reject = smoothstep(0.5, 2.0, motion_px); // reject if moved > ~1px

            var alpha = clamp(ao_params.temporal_blend + edge_rejection,
                              ao_params.temporal_blend, 1.0);
            alpha = max(alpha, motion_reject);
            final_ao = mix(history_ao, raw_ao, alpha);
        }
    }

    textureStore(ao_output, pixel, vec4<f32>(final_ao, 0.0, 0.0, 1.0));
}
