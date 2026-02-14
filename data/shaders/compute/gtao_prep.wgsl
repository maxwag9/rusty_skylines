#include "../includes/uniforms.wgsl"
// ----------------------------------------------------------------------------
// GTAO Prep — Single-pass resolve + linearize + downsample (depth & normals)
// ----------------------------------------------------------------------------
// Inputs:  full-res depth (MSAA or single-sample) + full-res normals
// Outputs: half-res linearized depth (min/closest) + half-res averaged normals
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Group 0: Input Textures (full resolution)
// ----------------------------------------------------------------------------
#ifdef MSAA
@group(0) @binding(0) var input_depth:   texture_depth_multisampled_2d;
#else
@group(0) @binding(0) var input_depth:   texture_depth_2d;
#endif
@group(0) @binding(1) var input_normals: texture_2d<f32>;

// ----------------------------------------------------------------------------
// Group 1: Output Storage (half resolution)
// ----------------------------------------------------------------------------
@group(1) @binding(0) var output_linear_depth_full: texture_storage_2d<r32float, write>;
@group(1) @binding(1) var output_linear_depth_half: texture_storage_2d<r32float, write>;
@group(1) @binding(2) var output_normals_half:      texture_storage_2d<rgba8unorm, write>;


// ----------------------------------------------------------------------------
// Group 2: Uniforms
// ----------------------------------------------------------------------------
@group(2) @binding(0) var<uniform> global_uniforms: Uniforms;

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------
fn linearize_depth(d: f32, z_near: f32, z_far: f32, reversed: bool) -> f32 {
    if (reversed) {
        if (d <= 0.000001) {
            return z_far;
        }
        return z_near / d;
    } else {
        return (z_near * z_far) / (z_far - d * (z_far - z_near));
    }
}

fn resolve_depth_at(coords: vec2<i32>, is_reversed: bool) -> f32 {
#ifdef MSAA
    let sample_count = textureNumSamples(input_depth);
    var best = select(1.0, 0.0, is_reversed);
    for (var i = 0u; i < sample_count; i++) {
        let s = textureLoad(input_depth, coords, i);
        if (is_reversed) {
            best = max(best, s);
        } else {
            best = min(best, s);
        }
    }
    return best;
#else
    return textureLoad(input_depth, coords, 0);
#endif
}

fn decode_normal(encoded: vec3<f32>) -> vec3<f32> {
    return encoded * 2.0 - 1.0;
}

fn encode_normal(n: vec3<f32>) -> vec3<f32> {
    return n * 0.5 + 0.5;
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_coords = vec2<i32>(gid.xy);
    let out_dims   = vec2<i32>(textureDimensions(output_linear_depth_half));

    if (out_coords.x >= out_dims.x || out_coords.y >= out_dims.y) {
        return;
    }

    let z_near      = global_uniforms.near_far_depth.x;
    let z_far       = global_uniforms.near_far_depth.y;
    let is_reversed = global_uniforms.reversed_depth_z != 0u;

    // ---- Full-res 2×2 block coordinates (clamped) -------------------------
    let in_base  = out_coords * 2;
    let depth_dims  = vec2<i32>(textureDimensions(input_depth));
    let normal_dims = vec2<i32>(textureDimensions(input_normals));

    let dc00 = min(in_base,                       depth_dims - 1);
    let dc10 = min(in_base + vec2<i32>(1, 0),     depth_dims - 1);
    let dc01 = min(in_base + vec2<i32>(0, 1),     depth_dims - 1);
    let dc11 = min(in_base + vec2<i32>(1, 1),     depth_dims - 1);

    // ---- Resolve + Linearize depth (per texel) ----------------------------
    let ld00 = linearize_depth(resolve_depth_at(dc00, is_reversed), z_near, z_far, is_reversed);
    let ld10 = linearize_depth(resolve_depth_at(dc10, is_reversed), z_near, z_far, is_reversed);
    let ld01 = linearize_depth(resolve_depth_at(dc01, is_reversed), z_near, z_far, is_reversed);
    let ld11 = linearize_depth(resolve_depth_at(dc11, is_reversed), z_near, z_far, is_reversed);

    // write full-res linear depth for those 4 pixels
    textureStore(output_linear_depth_full, dc00, vec4<f32>(ld00, 0.0, 0.0, 1.0));
    textureStore(output_linear_depth_full, dc10, vec4<f32>(ld10, 0.0, 0.0, 1.0));
    textureStore(output_linear_depth_full, dc01, vec4<f32>(ld01, 0.0, 0.0, 1.0));
    textureStore(output_linear_depth_full, dc11, vec4<f32>(ld11, 0.0, 0.0, 1.0));

    let min_depth = min(min(ld00, ld10), min(ld01, ld11));
    textureStore(output_linear_depth_half, out_coords, vec4<f32>(min_depth, 0.0, 0.0, 1.0));

    // ---- Downsample normals (average + renormalize) -----------------------
    let nc00 = min(in_base,                       normal_dims - 1);
    let nc10 = min(in_base + vec2<i32>(1, 0),     normal_dims - 1);
    let nc01 = min(in_base + vec2<i32>(0, 1),     normal_dims - 1);
    let nc11 = min(in_base + vec2<i32>(1, 1),     normal_dims - 1);

    let n00 = decode_normal(textureLoad(input_normals, nc00, 0).rgb);
    let n10 = decode_normal(textureLoad(input_normals, nc10, 0).rgb);
    let n01 = decode_normal(textureLoad(input_normals, nc01, 0).rgb);
    let n11 = decode_normal(textureLoad(input_normals, nc11, 0).rgb);

    var avg = (n00 + n10 + n01 + n11) * 0.25;
    let len = length(avg);
    if (len > 0.0001) {
        avg = avg / len;
    } else {
        avg = vec3<f32>(0.0, 1.0, 0.0);
    }

    textureStore(output_normals_half, out_coords, vec4<f32>(encode_normal(avg), 1.0));
}