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
    near_far_depth: vec2<f32>,
};

@group(2) @binding(0) var<uniform> global_uniforms: Uniforms;

// ----------------------------------------------------------------------------
// Group 0: Input Textures
// ----------------------------------------------------------------------------
@group(0) @binding(0) var msaa_depth: texture_depth_2d_multisampled;

// ----------------------------------------------------------------------------
// Group 1: Output Storage
// ----------------------------------------------------------------------------
@group(1) @binding(0) var resolved_depth: texture_storage_2d<r32float, write>;

// ----------------------------------------------------------------------------
// Compute Logic
// ----------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coords = vec2<i32>(gid.xy);
    let dims = vec2<i32>(textureDimensions(resolved_depth));

    // 1. Boundary Check
    if (coords.x >= dims.x || coords.y >= dims.y) {
        return;
    }

    // 2. Determine Depth Logic
    // If Reversed Z (1.0 = Near, 0.0 = Far): We want the MAX value (closest to 1.0)
    // If Standard Z (0.0 = Near, 1.0 = Far): We want the MIN value (closest to 0.0)
    let is_reversed = global_uniforms.reversed_depth_z != 0u;

    // 3. Initialize "furthest" possible depth
    // Rev: Start at 0.0. Std: Start at 1.0.
    var best_depth = select(1.0, 0.0, is_reversed);

    let samples = textureNumSamples(msaa_depth);

    // 4. Resolve Loop
    for (var i = 0u; i < samples; i++) {
        let sample_depth = textureLoad(msaa_depth, coords, i);

        if (is_reversed) {
            // Reverse Z: Larger values are closer to camera
            best_depth = max(best_depth, sample_depth);
        } else {
            // Standard Z: Smaller values are closer to camera
            best_depth = min(best_depth, sample_depth);
        }
    }

    // 5. Store Result
    // textureStore for r32float usually requires a vec4, even if only red is used
    textureStore(resolved_depth, coords, vec4<f32>(best_depth, 0.0, 0.0, 1.0));
}