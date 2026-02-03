// ============================================================================
// Ground Truth Ambient Occlusion (GTAO) - Half Resolution
// ============================================================================
// Based on:
// - "Practical Realtime Strategies for Accurate Indirect Occlusion" (Jimenez et al.)
// - Horizon-based approach with screen-space sampling
// ============================================================================

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;
const HALF_PI: f32 = 1.57079632679;

// Sampling configuration
const NUM_DIRECTIONS: u32 = 4u;  // 4-6 directions
const NUM_STEPS: u32 = 4u;       // 4-6 steps per direction

// ----------------------------------------------------------------------------
// Uniforms
// ----------------------------------------------------------------------------
struct CameraUniforms {
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
    reversed_depth_z: u32,
    shadows_enabled: u32,
    near_far_depth: vec2<f32>,
};

struct AOParams {
    radius_world: f32,
    intensity: f32,
    bias: f32,
    frame_index: u32,
    screen_size: vec2<f32>,
    inv_screen_size: vec2<f32>,
};

@group(2) @binding(0) var<uniform> camera: CameraUniforms;
@group(2) @binding(1) var<uniform> ao_params: AOParams;

// ----------------------------------------------------------------------------
// Input Textures
// ----------------------------------------------------------------------------
@group(0) @binding(0) var linear_depth_half: texture_2d<f32>;
@group(0) @binding(1) var normals_half: texture_2d<f32>;
@group(0) @binding(2) var blue_noise_tex: texture_2d<f32>;

// ----------------------------------------------------------------------------
// Output
// ----------------------------------------------------------------------------
@group(1) @binding(0) var ao_raw_half: texture_storage_2d<r32float, write>;

// ----------------------------------------------------------------------------
// Helper: Decode normal from [0,1] to [-1,1]
// ----------------------------------------------------------------------------
fn decode_normal(encoded: vec3<f32>) -> vec3<f32> {
    return normalize(encoded * 2.0 - 1.0);
}

// ----------------------------------------------------------------------------
// Helper: Reconstruct view-space position from UV and linear depth
// ----------------------------------------------------------------------------
fn reconstruct_view_position(uv: vec2<f32>, linear_z: f32) -> vec3<f32> {
    // Convert UV to clip space
    let clip = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);

    // Use inverse projection to get the view ray direction
    // For a point at z=1 in clip space
    let view_ray = camera.inv_proj * vec4<f32>(clip, 1.0, 1.0);
    let ray_dir = view_ray.xyz / view_ray.w;

    // Scale by linear depth to get view position
    // The ray direction's z component tells us how far we go per unit depth
    return ray_dir * (linear_z / -ray_dir.z);
}

// ----------------------------------------------------------------------------
// Helper: Get UV from view position
// ----------------------------------------------------------------------------
fn view_to_uv(view_pos: vec3<f32>) -> vec2<f32> {
    let clip = camera.proj * vec4<f32>(view_pos, 1.0);
    let ndc = clip.xy / clip.w;
    return vec2<f32>(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
}

// ----------------------------------------------------------------------------
// Helper: Sample blue noise with tiling
// ----------------------------------------------------------------------------
fn sample_blue_noise(pixel_coords: vec2<i32>, frame: u32) -> vec4<f32> {
    let noise_size = vec2<i32>(textureDimensions(blue_noise_tex));
    // Tile the noise and add temporal offset
    let offset = vec2<i32>(
        i32((frame * 7u) % u32(noise_size.x)),
        i32((frame * 11u) % u32(noise_size.y))
    );
    let sample_coords = (pixel_coords + offset) % noise_size;
    return textureLoad(blue_noise_tex, sample_coords, 0);
}

// ----------------------------------------------------------------------------
// Helper: Decode rotation from blue noise
// ----------------------------------------------------------------------------
fn get_rotation_from_noise(noise: vec4<f32>) -> vec2<f32> {
    // G and B channels contain encoded cos/sin
    let cos_a = noise.g * 2.0 - 1.0;
    let sin_a = noise.b * 2.0 - 1.0;
    return normalize(vec2<f32>(cos_a, sin_a));
}

// ----------------------------------------------------------------------------
// Helper: Rotate a 2D direction
// ----------------------------------------------------------------------------
fn rotate_direction(dir: vec2<f32>, cos_sin: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        dir.x * cos_sin.x - dir.y * cos_sin.y,
        dir.x * cos_sin.y + dir.y * cos_sin.x
    );
}

// ----------------------------------------------------------------------------
// Helper: Compute falloff based on distance
// ----------------------------------------------------------------------------
fn compute_falloff(dist_sq: f32, radius_sq: f32) -> f32 {
    // Smooth falloff: 1 - (dist/radius)^2
    return saturate(1.0 - dist_sq / radius_sq);
}

// ----------------------------------------------------------------------------
// Core: Compute horizon angle for one direction
// ----------------------------------------------------------------------------
fn compute_horizon_occlusion(
    view_pos: vec3<f32>,
    view_normal: vec3<f32>,
    screen_dir: vec2<f32>,
    step_size_px: f32,
    uv: vec2<f32>,
    radius_sq: f32
) -> f32 {
    // Initialize horizon as the surface tangent plane (no occlusion)
    // sin(horizon_angle) starts at 0 (flat horizon)
    var max_horizon_sin: f32 = 0.0;
    var occlusion: f32 = 0.0;

    // Step along the direction in screen space
    for (var step: u32 = 1u; step <= NUM_STEPS; step++) {
        // Calculate sample position in UV space
        let step_offset = screen_dir * (f32(step) * step_size_px);
        let sample_uv = uv + step_offset * ao_params.inv_screen_size;

        // Bounds check
        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 ||
            sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            continue;
        }

        // Sample depth at this location
        let sample_coords = vec2<i32>(sample_uv * ao_params.screen_size);
        let sample_depth = textureLoad(linear_depth_half, sample_coords, 0).r;

        // Skip sky/far pixels
        if (sample_depth >= camera.near_far_depth.y * 0.99) {
            continue;
        }

        // Reconstruct sample position in view space
        let sample_pos = reconstruct_view_position(sample_uv, sample_depth);

        // Vector from current pixel to sample
        let horizon_vec = sample_pos - view_pos;
        let dist_sq = dot(horizon_vec, horizon_vec);

        // Skip samples outside AO radius
        if (dist_sq > radius_sq) {
            continue;
        }

        // Compute the horizon angle's sine
        // sin(angle) = (sample.z - current.z) / distance
        // In view space, -Z is forward, so we use the difference
        let dist = sqrt(dist_sq);
        let horizon_sin = dot(horizon_vec, view_normal) / dist;

        // Apply distance falloff
        let falloff = compute_falloff(dist_sq, radius_sq);

        // Update maximum horizon if this sample occludes more
        // With bias applied to the comparison
        let biased_horizon = horizon_sin - ao_params.bias;

        if (biased_horizon > max_horizon_sin) {
            // Accumulate occlusion weighted by how much the horizon increased
            let delta = biased_horizon - max_horizon_sin;
            occlusion += delta * falloff;
            max_horizon_sin = biased_horizon;
        }
    }

    return occlusion;
}

// ----------------------------------------------------------------------------
// Main Compute Entry Point
// ----------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel_coords = vec2<i32>(gid.xy);
    let dims = vec2<i32>(textureDimensions(ao_raw_half));

    // Bounds check
    if (pixel_coords.x >= dims.x || pixel_coords.y >= dims.y) {
        return;
    }

    // Load linear depth
    let linear_z = textureLoad(linear_depth_half, pixel_coords, 0).r;

    // Skip sky pixels (at or beyond far plane)
    if (linear_z >= camera.near_far_depth.y * 0.99) {
        textureStore(ao_raw_half, pixel_coords, vec4<f32>(1.0, 0.0, 0.0, 1.0));
        return;
    }

    // Compute UV coordinates
    let uv = (vec2<f32>(pixel_coords) + 0.5) * ao_params.inv_screen_size;

    // Reconstruct view-space position
    let view_pos = reconstruct_view_position(uv, linear_z);

    // Load and decode normal (in view space)
    let encoded_normal = textureLoad(normals_half, pixel_coords, 0).rgb;
    let world_normal = decode_normal(encoded_normal);
    // Transform normal to view space
    let view_normal = normalize((camera.view * vec4<f32>(world_normal, 0.0)).xyz);

    // Calculate AO radius in pixels
    // proj[1][1] is the focal length for Y axis (cot(fov/2))
    let proj_scale = camera.proj[1][1] * ao_params.screen_size.y * 0.5;
    let radius_px = (ao_params.radius_world * proj_scale) / linear_z;

    // Clamp radius to reasonable bounds
    let clamped_radius_px = clamp(radius_px, 2.0, 128.0);
    let step_size_px = clamped_radius_px / f32(NUM_STEPS);

    // Squared world-space radius for falloff
    let radius_sq = ao_params.radius_world * ao_params.radius_world;

    // Sample blue noise for this pixel
    let noise = sample_blue_noise(pixel_coords, ao_params.frame_index);
    let rotation = get_rotation_from_noise(noise);

    // Jitter starting step with noise.r
    let jitter = noise.r;

    // Accumulate occlusion from all directions
    var total_occlusion: f32 = 0.0;
    let angle_step = PI / f32(NUM_DIRECTIONS); // Sample half-circle (other half is symmetric)

    for (var dir_idx: u32 = 0u; dir_idx < NUM_DIRECTIONS; dir_idx++) {
        // Base direction angle (evenly spaced around half circle)
        let base_angle = f32(dir_idx) * angle_step;

        // Create direction vector and rotate by noise
        let base_dir = vec2<f32>(cos(base_angle), sin(base_angle));
        let rotated_dir = rotate_direction(base_dir, rotation);

        // Sample in both positive and negative directions (full sphere coverage)
        // Positive direction
        let occ_pos = compute_horizon_occlusion(
            view_pos, view_normal, rotated_dir, step_size_px, uv, radius_sq
        );

        // Negative direction
        let occ_neg = compute_horizon_occlusion(
            view_pos, view_normal, -rotated_dir, step_size_px, uv, radius_sq
        );

        total_occlusion += occ_pos + occ_neg;
    }

    // Normalize by number of direction pairs
    total_occlusion /= f32(NUM_DIRECTIONS * 2u);

    // Apply intensity and clamp
    let ao = saturate(1.0 - total_occlusion * ao_params.intensity);

    // Store raw AO (no blur, no power curve)
    textureStore(ao_raw_half, pixel_coords, vec4<f32>(ao, 0.0, 0.0, 1.0));
}