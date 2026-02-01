// ----------------------------------------------------------------------------
// Group 2: Uniforms
// ----------------------------------------------------------------------------
struct Uniforms {
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    lighting_view_proj: array<mat4x4<f32>, 4>,
    cascade_splits: vec4<f32>,
    sun_direction: vec3<f32>,
    time: f32,
    camera_local: vec3<f32>,
    chunk_size: f32,
    camera_chunk: vec2<i32>,
    _pad_cam: vec2<i32>,
    moon_direction: vec3<f32>,
    orbit_radius: f32,
    reversed_depth_z: u32,     // 1 = Reversed (1.0 near, 0.0 far), 0 = Standard
    shadows_enabled: u32,
    near_far_depth: vec2<f32>, // .x = near, .y = far
};

@group(2) @binding(0) var<uniform> global_uniforms: Uniforms;

// ----------------------------------------------------------------------------
// Group 0: Input Texture (Resolved Depth from previous pass)
// ----------------------------------------------------------------------------
// Since this is the result of the previous resolve pass, it is no longer Multisampled.
// It is read as a standard float texture.
@group(0) @binding(0) var input_resolved_depth: texture_2d<f32>;

// ----------------------------------------------------------------------------
// Group 1: Output Storage (Linear Depth)
// ----------------------------------------------------------------------------
@group(1) @binding(0) var output_linear_depth: texture_storage_2d<r32float, write>;

// ----------------------------------------------------------------------------
// Helper: Linearize Depth Math
// ----------------------------------------------------------------------------
fn linearize_depth(d: f32, z_near: f32, z_far: f32, reversed: bool) -> f32 {
    if (reversed) {
        // Infinite Reverse Z (1.0 = Near, 0.0 = Infinity)
        // In this projection, z_view = z_near / depth_buffer_value
        
        // Prevent division by zero if depth is 0 (Infinity)
        if (d <= 0.000001) {
            return z_far; // Return Far (or a large number) representing skybox
        }
        return z_near / d;
    } else {
        // Standard Z (0.0 = Near, 1.0 = Far)
        // Formula: z_view = (z_near * z_far) / (z_far - depth * (z_far - z_near))
        return (z_near * z_far) / (z_far - d * (z_far - z_near));
    }
}

// ----------------------------------------------------------------------------
// Main Compute
// ----------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coords = vec2<i32>(gid.xy);
    let dims = vec2<i32>(textureDimensions(output_linear_depth));

    // Bounds check
    if (coords.x >= dims.x || coords.y >= dims.y) {
        return;
    }

    // 1. Load the raw non-linear depth from the previous resolve pass
    // mipLevel 0
    let raw_depth = textureLoad(input_resolved_depth, coords, 0).r;

    // 2. Prepare parameters
    let z_near = global_uniforms.near_far_depth.x;
    let z_far = global_uniforms.near_far_depth.y;
    let is_reversed = global_uniforms.reversed_depth_z != 0u;

    // 3. Calculate Linear Depth (View Space Z)
    let linear_z = linearize_depth(raw_depth, z_near, z_far, is_reversed);

    // 4. Store result
    textureStore(output_linear_depth, coords, vec4<f32>(linear_z, 0.0, 0.0, 1.0));
}