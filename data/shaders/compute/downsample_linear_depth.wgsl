// ----------------------------------------------------------------------------
// Downsample Linear Depth to Half Resolution
// ----------------------------------------------------------------------------
// Takes already-linearized depth and downsamples using min (closest depth)
// to preserve foreground edges for SSAO.
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Group 0: Input Texture (Full Resolution Linear Depth)
// ----------------------------------------------------------------------------
@group(0) @binding(0) var input_linear_depth: texture_2d<f32>;

// ----------------------------------------------------------------------------
// Group 1: Output Storage (Half Resolution Linear Depth)
// ----------------------------------------------------------------------------
@group(1) @binding(0) var output_linear_depth_half: texture_storage_2d<r32float, write>;

// ----------------------------------------------------------------------------
// Main Compute
// ----------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_coords = vec2<i32>(gid.xy);
    let out_dims = vec2<i32>(textureDimensions(output_linear_depth_half));

    // Bounds check against output dimensions
    if (out_coords.x >= out_dims.x || out_coords.y >= out_dims.y) {
        return;
    }

    // Map output coordinates to input coordinates (2x2 block)
    let in_coords = out_coords * 2;
    let in_dims = vec2<i32>(textureDimensions(input_linear_depth));

    // Sample 2x2 block from full resolution texture
    // Clamp to valid range to handle edge cases
    let c00 = min(in_coords + vec2<i32>(0, 0), in_dims - 1);
    let c10 = min(in_coords + vec2<i32>(1, 0), in_dims - 1);
    let c01 = min(in_coords + vec2<i32>(0, 1), in_dims - 1);
    let c11 = min(in_coords + vec2<i32>(1, 1), in_dims - 1);

    let d00 = textureLoad(input_linear_depth, c00, 0).r;
    let d10 = textureLoad(input_linear_depth, c10, 0).r;
    let d01 = textureLoad(input_linear_depth, c01, 0).r;
    let d11 = textureLoad(input_linear_depth, c11, 0).r;

    // Use minimum (closest) depth - preserves foreground for proper SSAO occlusion
    // This prevents halo artifacts around object edges
    let min_depth = min(min(d00, d10), min(d01, d11));

    // Store result
    textureStore(output_linear_depth_half, out_coords, vec4<f32>(min_depth, 0.0, 0.0, 1.0));
}