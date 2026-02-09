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

// ----------------------------------------------------------------------------
// Uniforms
// ----------------------------------------------------------------------------
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

@group(2) @binding(0) var<uniform> camera:    CameraUniforms;
@group(2) @binding(1) var<uniform> ao_params: AOParams;

// ----------------------------------------------------------------------------
// I/O
// ----------------------------------------------------------------------------
@group(0) @binding(0) var linear_depth_half: texture_2d<f32>;
@group(0) @binding(1) var normals_half:      texture_2d<f32>;
@group(0) @binding(2) var blue_noise_tex:    texture_2d<f32>;
@group(0) @binding(3) var ao_history:        texture_2d<f32>; // prev frame

@group(1) @binding(0) var ao_output: texture_storage_2d<r32float, write>;

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------
fn decode_normal(encoded: vec3<f32>) -> vec3<f32> {
    return normalize(encoded * 2.0 - 1.0);
}

fn reconstruct_view_position(uv: vec2<f32>, linear_z: f32) -> vec3<f32> {
    let clip     = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let view_ray = camera.inv_proj * vec4<f32>(clip, 1.0, 1.0);
    let ray_dir  = view_ray.xyz / view_ray.w;
    return ray_dir * (linear_z / -ray_dir.z);
}

fn sample_blue_noise(pixel_coords: vec2<i32>, frame: u32) -> vec4<f32> {
    let sz  = vec2<i32>(textureDimensions(blue_noise_tex));
    let off = vec2<i32>(i32((frame * 7u) % u32(sz.x)),
                        i32((frame * 11u) % u32(sz.y)));
    return textureLoad(blue_noise_tex, (pixel_coords + off) % sz, 0);
}

fn get_rotation(noise: vec4<f32>) -> vec2<f32> {
    return normalize(vec2<f32>(noise.g * 2.0 - 1.0, noise.b * 2.0 - 1.0));
}

fn rotate_dir(d: vec2<f32>, cs: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(d.x * cs.x - d.y * cs.y,
                     d.x * cs.y + d.y * cs.x);
}

fn compute_falloff(dist_sq: f32, radius_sq: f32) -> f32 {
    return saturate(1.0 - dist_sq / radius_sq);
}

// ----------------------------------------------------------------------------
// Horizon occlusion (unchanged)
// ----------------------------------------------------------------------------
fn compute_horizon_occlusion(
    view_pos:    vec3<f32>,
    view_normal: vec3<f32>,
    screen_dir:  vec2<f32>,
    step_px:     f32,
    uv:          vec2<f32>,
    radius_sq:   f32,
    dims:        vec2<i32>,
) -> f32 {
    var max_sin: f32  = 0.0;
    var occ:     f32  = 0.0;

    for (var step = 1u; step <= NUM_STEPS; step++) {
        let s_uv = uv + screen_dir * (f32(step) * step_px) * ao_params.inv_screen_size;
        if (s_uv.x < 0.0 || s_uv.x > 1.0 || s_uv.y < 0.0 || s_uv.y > 1.0) { continue; }

        let sc = vec2<i32>(s_uv * ao_params.screen_size);
        let sd = textureLoad(linear_depth_half, sc, 0).r;
        if (sd >= camera.near_far_depth.y * 0.99) { continue; }

        let sp  = reconstruct_view_position(s_uv, sd);
        let hv  = sp - view_pos;
        let dsq = dot(hv, hv);
        if (dsq > radius_sq) { continue; }

        let hs = dot(hv, view_normal) / sqrt(dsq);
        let fo = compute_falloff(dsq, radius_sq);
        let bh = hs - ao_params.bias;

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

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = vec2<i32>(gid.xy);
    let dims  = vec2<i32>(textureDimensions(ao_output));

    if (pixel.x >= dims.x || pixel.y >= dims.y) { return; }

    let linear_z = textureLoad(linear_depth_half, pixel, 0).r;

    // Sky → no occlusion, still write so history converges to 1.0
    if (linear_z >= camera.near_far_depth.y * 0.99) {
        textureStore(ao_output, pixel, vec4<f32>(1.0, 0.0, 0.0, 1.0));
        return;
    }

    let uv       = (vec2<f32>(pixel) + 0.5) * ao_params.inv_screen_size;
    let view_pos = reconstruct_view_position(uv, linear_z);

    let wn         = decode_normal(textureLoad(normals_half, pixel, 0).rgb);
    let view_norm  = normalize((camera.view * vec4<f32>(wn, 0.0)).xyz);

    let proj_scale  = camera.proj[1][1] * ao_params.screen_size.y * 0.5;
    let radius_px   = clamp((ao_params.radius_world * proj_scale) / linear_z, 2.0, 128.0);
    let step_px     = radius_px / f32(NUM_STEPS);
    let radius_sq   = ao_params.radius_world * ao_params.radius_world;

    let noise    = sample_blue_noise(pixel, ao_params.frame_index);
    let rotation = get_rotation(noise);

    var total_occ: f32 = 0.0;
    let angle_step = PI / f32(NUM_DIRECTIONS);

    for (var d = 0u; d < NUM_DIRECTIONS; d++) {
        let base_dir = vec2<f32>(cos(f32(d) * angle_step),
                                 sin(f32(d) * angle_step));
        let rd       = rotate_dir(base_dir, rotation);

        total_occ += compute_horizon_occlusion(view_pos, view_norm,  rd, step_px, uv, radius_sq, dims);
        total_occ += compute_horizon_occlusion(view_pos, view_norm, -rd, step_px, uv, radius_sq, dims);
    }

    total_occ /= f32(NUM_DIRECTIONS * 2u);
    let raw_ao = saturate(1.0 - total_occ * ao_params.intensity);

    // =====================================================================
    // Temporal Accumulation
    // =====================================================================
    var final_ao = raw_ao;

    let world_pos = camera.inv_view * vec4<f32>(view_pos, 1.0);
    let prev_clip = ao_params.prev_view_proj * world_pos;

    // Only reproject if the point was in front of the previous camera
    if (prev_clip.w > 0.001) {
        let prev_ndc = prev_clip.xy / prev_clip.w;
        let prev_uv  = vec2<f32>(prev_ndc.x * 0.5 + 0.5,
                                  1.0 - (prev_ndc.y * 0.5 + 0.5));

        if (prev_uv.x >= 0.0 && prev_uv.x <= 1.0 &&
            prev_uv.y >= 0.0 && prev_uv.y <= 1.0) {

            let history_ao = sample_history_bilinear(prev_uv, dims);

            // Neighbourhood clamp — prevent ghosting by bounding
            // the history to the local neighbourhood of current-frame AO.
            // We sample the 3×3 current-frame neighbourhood (cheap because
            // we already have depth loaded in cache / L1).
            var local_min = raw_ao;
            var local_max = raw_ao;

            // We don't have the neighbours' AO yet (they run in parallel),
            // but we CAN use depth discontinuity as a rejection signal:
            // large depth change ≈ likely disocclusion.
            let neigh_offsets = array<vec2<i32>, 4>(
                vec2<i32>(-1,  0), vec2<i32>(1, 0),
                vec2<i32>( 0, -1), vec2<i32>(0, 1),
            );

            var depth_variance: f32 = 0.0;
            for (var i = 0; i < 4; i++) {
                let nc = clamp(pixel + neigh_offsets[i], vec2<i32>(0), dims - 1);
                let nd = textureLoad(linear_depth_half, nc, 0).r;
                let rel_diff = abs(nd - linear_z) / max(linear_z, 0.001);
                depth_variance = max(depth_variance, rel_diff);
            }

            // At depth edges or disocclusions, trust history less
            let edge_rejection = smoothstep(0.05, 0.15, depth_variance);
            let alpha = clamp(ao_params.temporal_blend + edge_rejection, ao_params.temporal_blend, 1.0);

            final_ao = mix(history_ao, raw_ao, alpha);
        }
    }

    textureStore(ao_output, pixel, vec4<f32>(final_ao, 0.0, 0.0, 1.0));
}