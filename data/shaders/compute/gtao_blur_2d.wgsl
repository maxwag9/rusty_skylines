// ============================================================================
// Edge-Aware 2D Bilateral Blur (replaces separate H + V passes)
// ============================================================================

const WORKGROUP_SIZE: i32 = 8;
const MAX_KERNEL_RADIUS: i32 = 6;
const TILE_SIZE: i32 = WORKGROUP_SIZE + 2 * MAX_KERNEL_RADIUS; // 20
const TILE_AREA: i32 = TILE_SIZE * TILE_SIZE;                  // 400
const WG_THREADS: i32 = WORKGROUP_SIZE * WORKGROUP_SIZE;       // 64

var<workgroup> shared_ao:     array<f32, 400>;
var<workgroup> shared_depth:  array<f32, 400>;
var<workgroup> shared_normal: array<vec3<f32>, 400>;

const GAUSSIAN_WEIGHTS: array<f32, 7> = array<f32, 7>(
    0.1964825501511404,
    0.2969069646728344,
    0.09447039785044732,
    0.010381362401148057,
    0.0003951963710896622,
    0.000005231848807099,
    0.0000000239279,
);

// ----------------------------------------------------------------------------
// Uniforms
// ----------------------------------------------------------------------------
struct BlurParams {
    depth_sigma:   f32,
    normal_sigma:  f32,
    kernel_radius: i32,
    _padding:      i32,
};

@group(2) @binding(0) var<uniform> blur_params: BlurParams;

// ----------------------------------------------------------------------------
// I/O
// ----------------------------------------------------------------------------
@group(0) @binding(0) var ao_input:     texture_2d<f32>;
@group(0) @binding(1) var linear_depth: texture_2d<f32>;
@group(0) @binding(2) var normals:      texture_2d<f32>;

@group(1) @binding(0) var ao_output: texture_storage_2d<r32float, write>;

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------
fn decode_normal(encoded: vec3<f32>) -> vec3<f32> {
    return normalize(encoded * 2.0 - 1.0);
}

fn get_gaussian_weight(offset: i32) -> f32 {
    let a = abs(offset);
    if (a > MAX_KERNEL_RADIUS) { return 0.0; }
    return GAUSSIAN_WEIGHTS[a];
}

fn compute_depth_weight(center: f32, sample_d: f32, sigma: f32) -> f32 {
    let avg   = (center + sample_d) * 0.5;
    let rel   = abs(center - sample_d) / max(avg, 0.001);
    return exp(-rel * rel / (2.0 * sigma * sigma));
}

fn compute_normal_weight(center: vec3<f32>, sample_n: vec3<f32>, sigma: f32) -> f32 {
    let diff = 1.0 - max(0.0, dot(center, sample_n));
    return exp(-diff * diff / (2.0 * sigma * sigma));
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
    @builtin(workgroup_id)         wid: vec3<u32>,
) {
    let dims         = vec2<i32>(textureDimensions(ao_output));
    let group_origin = vec2<i32>(wid.xy) * WORKGROUP_SIZE;
    let local_linear = i32(lid.y) * WORKGROUP_SIZE + i32(lid.x);

    // ---- Cooperative tile load (20×20 = 400 entries, 64 threads) ----------
    for (var i = local_linear; i < TILE_AREA; i += WG_THREADS) {
        let tile_x = i % TILE_SIZE;
        let tile_y = i / TILE_SIZE;
        let gpos   = group_origin + vec2<i32>(tile_x, tile_y) - MAX_KERNEL_RADIUS;
        let cpos   = clamp(gpos, vec2<i32>(0), dims - 1);

        shared_ao[i]     = textureLoad(ao_input, cpos, 0).r;
        shared_depth[i]  = textureLoad(linear_depth, cpos, 0).r;
        shared_normal[i] = decode_normal(textureLoad(normals, cpos, 0).rgb);
    }

    workgroupBarrier();

    let global_id = vec2<i32>(gid.xy);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    // ---- Center sample from shared memory ---------------------------------
    let center_tile = vec2<i32>(lid.xy) + MAX_KERNEL_RADIUS;
    let center_idx  = center_tile.y * TILE_SIZE + center_tile.x;

    let center_ao     = shared_ao[center_idx];
    let center_depth  = shared_depth[center_idx];
    let center_normal = shared_normal[center_idx];

    // Sky early-out
    if (center_depth > 10000.0) {
        textureStore(ao_output, global_id, vec4<f32>(center_ao, 0.0, 0.0, 1.0));
        return;
    }

    // ---- 2D bilateral blur ------------------------------------------------
    let radius = min(blur_params.kernel_radius, MAX_KERNEL_RADIUS);

    var weighted_sum: f32 = 0.0;
    var weight_sum:   f32 = 0.0;

    for (var dy = -radius; dy <= radius; dy++) {
        let gy = get_gaussian_weight(dy);
        for (var dx = -radius; dx <= radius; dx++) {
            let s_tile = center_tile + vec2<i32>(dx, dy);
            let idx    = s_tile.y * TILE_SIZE + s_tile.x;

            let spatial_w = get_gaussian_weight(dx) * gy;
            let depth_w   = compute_depth_weight(center_depth, shared_depth[idx], blur_params.depth_sigma);
            let normal_w  = compute_normal_weight(center_normal, shared_normal[idx], blur_params.normal_sigma);

            let w = spatial_w * depth_w * normal_w;
            weighted_sum += shared_ao[idx] * w;
            weight_sum   += w;
        }
    }

    var blurred: f32;
    if (weight_sum > 0.0001) {
        blurred = weighted_sum / weight_sum;
    } else {
        blurred = center_ao;
    }

    textureStore(ao_output, global_id, vec4<f32>(blurred, 0.0, 0.0, 1.0));
}