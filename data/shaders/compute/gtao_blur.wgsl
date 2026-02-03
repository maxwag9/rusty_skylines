// ============================================================================
// Edge-Aware Bilateral Blur - Optimized with Shared Memory
// ============================================================================
// Uses workgroup shared memory to reduce texture loads
// ============================================================================

const WORKGROUP_SIZE: i32 = 8;
const MAX_KERNEL_RADIUS: i32 = 6;
const TILE_SIZE: i32 = WORKGROUP_SIZE + 2 * MAX_KERNEL_RADIUS; // 8 + 12 = 20
const TILE_AREA: i32 = TILE_SIZE * TILE_SIZE;                 // 400

// Shared memory for the tile (with halo for blur radius)
var<workgroup> shared_ao: array<f32, TILE_AREA>;
var<workgroup> shared_depth: array<f32, TILE_AREA>;
var<workgroup> shared_normal: array<vec3<f32>, TILE_AREA>;

const GAUSSIAN_WEIGHTS: array<f32, 7> = array<f32, 7>(
    0.1964825501511404,
    0.2969069646728344,
    0.09447039785044732,
    0.010381362401148057,
    0.0003951963710896622,
    0.000005231848807099,
    0.0000000239279,
);

struct BlurParams {
    direction: vec2<i32>,
    texel_size: vec2<f32>,
    depth_sigma: f32,
    normal_sigma: f32,
    kernel_radius: i32,
    _padding: i32,
};

@group(2) @binding(0) var<uniform> blur_params: BlurParams;

@group(0) @binding(0) var ao_input: texture_2d<f32>;
@group(0) @binding(1) var linear_depth: texture_2d<f32>;
@group(0) @binding(2) var normals: texture_2d<f32>;

@group(1) @binding(0) var ao_output: texture_storage_2d<r32float, write>;

fn decode_normal(encoded: vec3<f32>) -> vec3<f32> {
    return normalize(encoded * 2.0 - 1.0);
}

fn shared_index(local_x: i32, local_y: i32) -> i32 {
    let x = local_x + MAX_KERNEL_RADIUS;
    let y = local_y + MAX_KERNEL_RADIUS;
    return y * TILE_SIZE + x;
}

fn get_gaussian_weight(offset: i32) -> f32 {
    let abs_offset = abs(offset);
    if (abs_offset > MAX_KERNEL_RADIUS) { return 0.0; }
    return GAUSSIAN_WEIGHTS[abs_offset];
}

fn compute_depth_weight(center_depth: f32, sample_depth: f32, sigma: f32) -> f32 {
    let avg_depth = (center_depth + sample_depth) * 0.5;
    let depth_diff = abs(center_depth - sample_depth) / max(avg_depth, 0.001);
    return exp(-depth_diff * depth_diff / (2.0 * sigma * sigma));
}

fn compute_normal_weight(center_normal: vec3<f32>, sample_normal: vec3<f32>, sigma: f32) -> f32 {
    let n_dot = max(0.0, dot(center_normal, sample_normal));
    let angle_diff = 1.0 - n_dot;
    return exp(-angle_diff * angle_diff / (2.0 * sigma * sigma));
}

@compute @workgroup_size(8, 8)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let dims = vec2<i32>(textureDimensions(ao_output));
    let local_id = vec2<i32>(lid.xy);
    let global_id = vec2<i32>(gid.xy);
    let group_start = vec2<i32>(wid.xy) * WORKGROUP_SIZE;

    // Load data into shared memory (including halo region)
    // Each thread loads multiple pixels to fill the halo
    let radius = min(blur_params.kernel_radius, MAX_KERNEL_RADIUS);

    for (var dy: i32 = local_id.y - radius; dy <= local_id.y + radius; dy += WORKGROUP_SIZE) {
        for (var dx: i32 = local_id.x - radius; dx <= local_id.x + radius; dx += WORKGROUP_SIZE) {
            let sample_pos = group_start + vec2<i32>(dx, dy);
            let clamped_pos = clamp(sample_pos, vec2<i32>(0), dims - 1);

            let tile_x = dx + MAX_KERNEL_RADIUS;
            let tile_y = dy + MAX_KERNEL_RADIUS;

            if (tile_x >= 0 && tile_x < TILE_SIZE &&
                tile_y >= 0 && tile_y < TILE_SIZE) {

                let idx = tile_y * TILE_SIZE + tile_x;

                shared_ao[idx]     = textureLoad(ao_input, clamped_pos, 0).r;
                shared_depth[idx]  = textureLoad(linear_depth, clamped_pos, 0).r;
                shared_normal[idx] = decode_normal(
                    textureLoad(normals, clamped_pos, 0).rgb
                );
            }
        }
    }

    workgroupBarrier();

    // Bounds check for output
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    // Get center data from shared memory
    let center_idx = shared_index(local_id.x, local_id.y);
    let center_ao = shared_ao[center_idx];
    let center_depth = shared_depth[center_idx];
    let center_normal = shared_normal[center_idx];

    // Skip sky
    if (center_depth > 10000.0) {
        textureStore(ao_output, global_id, vec4<f32>(center_ao, 0.0, 0.0, 1.0));
        return;
    }

    let blur_dir = blur_params.direction;
    let actual_radius = min(blur_params.kernel_radius, MAX_KERNEL_RADIUS);

    var weighted_sum: f32 = 0.0;
    var weight_sum: f32 = 0.0;

    for (var offset: i32 = -actual_radius; offset <= actual_radius; offset++) {
        let sample_local = local_id + blur_dir * offset;
        let sample_idx = shared_index(sample_local.x, sample_local.y);

        if (sample_idx >= 0 && sample_idx < 400) {
            let sample_ao = shared_ao[sample_idx];
            let sample_depth = shared_depth[sample_idx];
            let sample_normal = shared_normal[sample_idx];

            let spatial_weight = get_gaussian_weight(offset);
            let depth_weight = compute_depth_weight(center_depth, sample_depth, blur_params.depth_sigma);
            let normal_weight = compute_normal_weight(center_normal, sample_normal, blur_params.normal_sigma);

            let bilateral_weight = spatial_weight * depth_weight * normal_weight;

            weighted_sum += sample_ao * bilateral_weight;
            weight_sum += bilateral_weight;
        }
    }

    var blurred_ao: f32;
    if (weight_sum > 0.0001) {
        blurred_ao = weighted_sum / weight_sum;
    } else {
        blurred_ao = center_ao;
    }

    textureStore(ao_output, global_id, vec4<f32>(blurred_ao, 0.0, 0.0, 1.0));
}