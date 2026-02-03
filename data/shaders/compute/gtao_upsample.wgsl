// ============================================================================
// Edge-Aware Bilateral Upsampling for GTAO
// ============================================================================
// Upsamples half-resolution AO to full resolution while preserving edges.
// Uses depth and normal similarity to avoid halos at discontinuities.
// ============================================================================

// ----------------------------------------------------------------------------
// Uniforms
// ----------------------------------------------------------------------------
struct UpsampleParams {
    full_size: vec2<f32>,        // Full resolution dimensions
    half_size: vec2<f32>,        // Half resolution dimensions
    inv_full_size: vec2<f32>,    // 1.0 / full_size
    inv_half_size: vec2<f32>,    // 1.0 / half_size
    depth_threshold: f32,        // Relative depth difference threshold (e.g., 0.05)
    normal_threshold: f32,       // Normal dot product threshold (e.g., 0.9)
    use_normal_check: u32,       // 0 = depth only, 1 = depth + normal
    _padding: u32,
};

@group(2) @binding(0) var<uniform> params: UpsampleParams;

// ----------------------------------------------------------------------------
// Input Textures
// ----------------------------------------------------------------------------
@group(0) @binding(0) var ao_half: texture_2d<f32>;           // Half-res blurred AO
@group(0) @binding(1) var depth_half: texture_2d<f32>;        // Half-res linear depth
@group(0) @binding(2) var normals_half: texture_2d<f32>;      // Half-res normals
@group(0) @binding(3) var depth_full: texture_2d<f32>;        // Full-res linear depth
@group(0) @binding(4) var normals_full: texture_2d<f32>;      // Full-res normals

// Sampler for bilinear filtering (used for reference, we do manual sampling)
@group(0) @binding(5) var linear_sampler: sampler;

// ----------------------------------------------------------------------------
// Output
// ----------------------------------------------------------------------------
@group(1) @binding(0) var ao_full: texture_storage_2d<r32float, write>;

// ----------------------------------------------------------------------------
// Helper: Decode normal from [0,1] to [-1,1]
// ----------------------------------------------------------------------------
fn decode_normal(encoded: vec3<f32>) -> vec3<f32> {
    return normalize(encoded * 2.0 - 1.0);
}

// ----------------------------------------------------------------------------
// Helper: Compute depth similarity weight
// ----------------------------------------------------------------------------
fn compute_depth_weight(full_depth: f32, half_depth: f32, threshold: f32) -> f32 {
    // Use relative depth difference for scale-invariance
    let avg_depth = (full_depth + half_depth) * 0.5;
    let rel_diff = abs(full_depth - half_depth) / max(avg_depth, 0.001);

    // Smooth falloff using exponential
    // threshold controls the sensitivity
    let weight = exp(-rel_diff * rel_diff / (2.0 * threshold * threshold));

    return weight;
}

// ----------------------------------------------------------------------------
// Helper: Compute normal similarity weight
// ----------------------------------------------------------------------------
fn compute_normal_weight(full_normal: vec3<f32>, half_normal: vec3<f32>, threshold: f32) -> f32 {
    let n_dot = max(0.0, dot(full_normal, half_normal));
    let diff = max(0.0, (threshold - n_dot) / threshold);
    return exp(-diff * diff * 8.0);
}

// ----------------------------------------------------------------------------
// Helper: Safe texture coordinate clamping
// ----------------------------------------------------------------------------
fn clamp_coords(coords: vec2<i32>, dims: vec2<i32>) -> vec2<i32> {
    return clamp(coords, vec2<i32>(0), dims - 1);
}

// ----------------------------------------------------------------------------
// Main: Joint Bilateral Upsampling
// ----------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full_coords = vec2<i32>(gid.xy);
    let full_dims = vec2<i32>(textureDimensions(ao_full));

    // Bounds check
    if (full_coords.x >= full_dims.x || full_coords.y >= full_dims.y) {
        return;
    }

    // Load full-resolution reference data
    let full_depth = textureLoad(depth_full, full_coords, 0).r;
    let full_normal = decode_normal(textureLoad(normals_full, full_coords, 0).rgb);

    // Early out for sky pixels
    if (full_depth > 10000.0) {
        textureStore(ao_full, full_coords, vec4<f32>(1.0, 0.0, 0.0, 1.0));
        return;
    }

    // Calculate corresponding half-res position (with subpixel precision)
    // Full-res pixel center -> half-res UV
    let full_uv = (vec2<f32>(full_coords) + 0.5) * params.inv_full_size;
    let half_pos = full_uv * params.half_size - 0.5; // Position in half-res pixel coordinates

    // Get the 4 nearest half-res pixels for bilinear interpolation
    let half_base = vec2<i32>(floor(half_pos));
    let frac = half_pos - vec2<f32>(half_base); // Fractional part for interpolation

    let half_dims = vec2<i32>(textureDimensions(ao_half));

    // Sample positions (clamped to valid range)
    let p00 = clamp_coords(half_base + vec2<i32>(0, 0), half_dims);
    let p10 = clamp_coords(half_base + vec2<i32>(1, 0), half_dims);
    let p01 = clamp_coords(half_base + vec2<i32>(0, 1), half_dims);
    let p11 = clamp_coords(half_base + vec2<i32>(1, 1), half_dims);

    // Load AO values from half-res
    let ao00 = textureLoad(ao_half, p00, 0).r;
    let ao10 = textureLoad(ao_half, p10, 0).r;
    let ao01 = textureLoad(ao_half, p01, 0).r;
    let ao11 = textureLoad(ao_half, p11, 0).r;

    // Load depth values from half-res
    let depth00 = textureLoad(depth_half, p00, 0).r;
    let depth10 = textureLoad(depth_half, p10, 0).r;
    let depth01 = textureLoad(depth_half, p01, 0).r;
    let depth11 = textureLoad(depth_half, p11, 0).r;

    // Compute depth-based weights
    let dw00 = compute_depth_weight(full_depth, depth00, params.depth_threshold);
    let dw10 = compute_depth_weight(full_depth, depth10, params.depth_threshold);
    let dw01 = compute_depth_weight(full_depth, depth01, params.depth_threshold);
    let dw11 = compute_depth_weight(full_depth, depth11, params.depth_threshold);

    // Optional: Compute normal-based weights
    var nw00: f32 = 1.0;
    var nw10: f32 = 1.0;
    var nw01: f32 = 1.0;
    var nw11: f32 = 1.0;

    if (params.use_normal_check != 0u) {
        let normal00 = decode_normal(textureLoad(normals_half, p00, 0).rgb);
        let normal10 = decode_normal(textureLoad(normals_half, p10, 0).rgb);
        let normal01 = decode_normal(textureLoad(normals_half, p01, 0).rgb);
        let normal11 = decode_normal(textureLoad(normals_half, p11, 0).rgb);

        nw00 = compute_normal_weight(full_normal, normal00, params.normal_threshold);
        nw10 = compute_normal_weight(full_normal, normal10, params.normal_threshold);
        nw01 = compute_normal_weight(full_normal, normal01, params.normal_threshold);
        nw11 = compute_normal_weight(full_normal, normal11, params.normal_threshold);
    }

    // Bilinear interpolation weights
    let bw00 = (1.0 - frac.x) * (1.0 - frac.y);
    let bw10 = frac.x * (1.0 - frac.y);
    let bw01 = (1.0 - frac.x) * frac.y;
    let bw11 = frac.x * frac.y;

    // Combined weights: bilinear * depth * normal
    let w00 = bw00 * dw00 * nw00;
    let w10 = bw10 * dw10 * nw10;
    let w01 = bw01 * dw01 * nw01;
    let w11 = bw11 * dw11 * nw11;

    let total_weight = w00 + w10 + w01 + w11;

    // Compute weighted AO
    var final_ao: f32;

    if (total_weight > 0.0001) {
        // Normal case: weighted average
        let weighted_ao = (ao00 * w00 + ao10 * w10 + ao01 * w01 + ao11 * w11) / total_weight;

        // Bias toward 1.0 (no occlusion) when weights are low
        // This prevents dark halos at edges
        let confidence = min(total_weight / (bw00 + bw10 + bw01 + bw11), 1.0);
        final_ao = mix(1.0, weighted_ao, confidence);
    } else {
        // Edge case: all samples rejected, use no occlusion
        final_ao = 1.0;
    }

    // Store result
    textureStore(ao_full, full_coords, vec4<f32>(final_ao, 0.0, 0.0, 1.0));
}