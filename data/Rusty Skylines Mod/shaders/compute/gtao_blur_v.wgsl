const WORKGROUP_SIZE: i32 = 8;
const KERNEL_RADIUS: i32 = 4;
const TILE_SIZE: i32 = WORKGROUP_SIZE + 2 * KERNEL_RADIUS;
const TILE_AREA: i32 = TILE_SIZE * TILE_SIZE;
const WG_THREADS: i32 = WORKGROUP_SIZE * WORKGROUP_SIZE;
var<workgroup> shared_ao: array<f32, 256>;
var<workgroup> shared_depth: array<f32, 256>;
var<workgroup> shared_normal: array<vec3<f32>, 256>;
const GAUSSIAN_WEIGHTS: array<f32, 5> = array<f32, 5>(
    0.2270270270,
    0.1945945946,
    0.1216216216,
    0.0540540541,
    0.0162162162
);
struct BlurParams {
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
    return encoded * 2.0 - 1.0;
}
fn get_gaussian_weight(offset: i32) -> f32 {
    let a = abs(offset);
    if (a > KERNEL_RADIUS) { return 0.0; }
    return GAUSSIAN_WEIGHTS[a];
}
fn compute_depth_weight(center: f32, sample_d: f32, inv_sigma: f32) -> f32 {
    let avg = (center + sample_d) * 0.5;
    let rel = abs(center - sample_d) / max(avg, 0.001);
    return saturate(1.0 - rel * inv_sigma);
}
fn compute_normal_weight(center: vec3<f32>, sample_n: vec3<f32>, inv_sigma: f32) -> f32 {
    let diff = 1.0 - max(0.0, dot(center, sample_n));
    return saturate(1.0 - diff * inv_sigma);
}
@compute @workgroup_size(8, 8)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let dims = vec2<i32>(textureDimensions(ao_output));
    let group_origin = vec2<i32>(wid.xy) * WORKGROUP_SIZE;
    let local_linear = i32(lid.y) * WORKGROUP_SIZE + i32(lid.x);
    for (var i = local_linear; i < TILE_AREA; i += WG_THREADS) {
        let tile_x = i % TILE_SIZE;
        let tile_y = i / TILE_SIZE;
        let gpos = group_origin + vec2<i32>(tile_x, tile_y) - KERNEL_RADIUS;
        let cpos = clamp(gpos, vec2<i32>(0), dims - 1);
        shared_ao[i] = textureLoad(ao_input, cpos, 0).r;
        shared_depth[i] = textureLoad(linear_depth, cpos, 0).r;
        shared_normal[i] = decode_normal(textureLoad(normals, cpos, 0).rgb);
    }
    workgroupBarrier();
    let global_id = vec2<i32>(gid.xy);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }
    let center_tile = vec2<i32>(lid.xy) + KERNEL_RADIUS;
    let center_idx = center_tile.y * TILE_SIZE + center_tile.x;
    let center_ao = shared_ao[center_idx];
    let center_depth = shared_depth[center_idx];
    let center_normal = shared_normal[center_idx];
    if (center_depth > 10000.0) {
        textureStore(ao_output, global_id, vec4<f32>(center_ao, 0.0, 0.0, 1.0));
        return;
    }
    let inv_depth_sigma = 1.0 / max(blur_params.depth_sigma, 0.0001);
    let inv_normal_sigma = 1.0 / max(blur_params.normal_sigma, 0.0001);
    var weighted_sum: f32 = 0.0;
    var weight_sum: f32 = 0.0;
    for (var dy = -KERNEL_RADIUS; dy <= KERNEL_RADIUS; dy++) {
        let s_tile = center_tile + vec2<i32>(0, dy);
        let idx = s_tile.y * TILE_SIZE + s_tile.x;
        let spatial_w = get_gaussian_weight(dy);
        let depth_w = compute_depth_weight(center_depth, shared_depth[idx], inv_depth_sigma);
        let normal_w = compute_normal_weight(center_normal, shared_normal[idx], inv_normal_sigma);
        let w = spatial_w * depth_w * normal_w;
        weighted_sum += shared_ao[idx] * w;
        weight_sum += w;
    }
    var blurred: f32;
    if (weight_sum > 0.0001) {
        blurred = weighted_sum / weight_sum;
    } else {
        blurred = center_ao;
    }
    textureStore(ao_output, global_id, vec4<f32>(blurred, 0.0, 0.0, 1.0));
}