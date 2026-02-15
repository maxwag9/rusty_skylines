#include "../includes/uniforms.wgsl"
// ============================================================
// Sun Shadow Ray Tracing - Optimized for City Builder
// Single sample, binary output, exit-on-first-hit
// ============================================================

const INF_F32 = 1e30;

struct RTVertex {
    position: vec3<f32>,
    _pad: f32,
};

struct BvhNode {
    aabb_min: vec3<f32>,
    _pad0: f32,
    aabb_max: vec3<f32>,
    _pad1: f32,
    left_or_tri_offset: u32,
    meta_info: u32,                  // bit 0: is_leaf
    right_or_tri_count: u32,
    _pad2: u32,
};

struct TlasInstance {
    transform: mat4x4<f32>,
    inv_transform: mat4x4<f32>,

    blas_root: u32,
    instance_id: u32,           // For self-shadow skip
    _pad0: vec2<u32>,

    // World-space AABB - CRITICAL for early rejection
    aabb_min: vec3<f32>,
    _pad1: f32,
    aabb_max: vec3<f32>,
    _pad2: f32,
};

struct SunShadowParams {
    max_distance: f32,          // e.g. 200.0
    normal_bias: f32,           // e.g. 0.05
    origin_bias: f32,           // e.g. 0.02
    ndotl_fade: f32,            // e.g. 0.10  (smooth ramp around N·L=0)

    soft_start: f32,            // e.g. 5.0   (close blockers = hard/dark)
    soft_end: f32,              // e.g. 80.0  (far blockers = softer/lighter)
    far_hit_visibility: f32,    // e.g. 0.85  (visibility when blocked very far)
    _pad0: f32,
};

// G-buffer inputs (half res)
@group(0) @binding(0) var depth_tex: texture_2d<f32>; //Half res
@group(0) @binding(1) var normal_tex: texture_2d<f32>; //Half-res
@group(0) @binding(2) var instance_id_tex: texture_2d<u32>; //Full-res

// Output
@group(1) @binding(0) var rt_out: texture_storage_2d<r8unorm, write>; //Half-res

// Acceleration structure
@group(2) @binding(0) var<storage, read> blas_vertices: array<RTVertex>;
@group(2) @binding(1) var<storage, read> blas_indices: array<u32>;
@group(2) @binding(2) var<storage, read> blas_bvh: array<BvhNode>;
@group(2) @binding(3) var<storage, read> tlas_instances: array<TlasInstance>;
@group(2) @binding(4) var<storage, read> tlas_bvh: array<BvhNode>;
@group(2) @binding(5) var<uniform> uniforms: Uniforms;
@group(2) @binding(6) var<uniform> params: SunShadowParams;


const TLAS_STACK_SIZE = 22u;
const BLAS_STACK_SIZE = 16u;
const TRI_EPSILON = 0.00001;

struct StackDebug {
    tlas_overflow: bool,
    blas_overflow: bool,
};
// PUSH helper
fn tlas_push(
    stack: ptr<function, array<u32, TLAS_STACK_SIZE>>,
    ptr_idx: ptr<function, i32>,
    v: u32,
    dbg: ptr<function, StackDebug>
) {
    let next = (*ptr_idx) + 1;
    if (next >= i32(TLAS_STACK_SIZE)) {
        (*dbg).tlas_overflow = true;
        return;
    }
    *ptr_idx = next;
    (*stack)[*ptr_idx] = v;
}

fn blas_push(
    stack: ptr<function, array<u32, BLAS_STACK_SIZE>>,
    ptr_idx: ptr<function, i32>,
    v: u32,
    dbg: ptr<function, StackDebug>
) {
    let next = (*ptr_idx) + 1;
    if (next >= i32(BLAS_STACK_SIZE)) {
        (*dbg).blas_overflow = true;
        return;
    }
    *ptr_idx = next;
    (*stack)[*ptr_idx] = v;
}


// ============================================================
// Ray structure - precompute inv_dir once
// ============================================================

struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
    inv_dir: vec3<f32>,
    max_t: f32,
};

fn safe_inv_dir(d: vec3<f32>) -> vec3<f32> {
    let eps = 1e-8;

    // Choose a sign even when component is 0: (dir >= 0) => +1 else -1
    let s = select(vec3<f32>(-1.0), vec3<f32>(1.0), d >= vec3<f32>(0.0));

    // Clamp magnitude away from 0, preserve sign
    let dd = s * max(abs(d), vec3<f32>(eps));

    return 1.0 / dd;
}

fn make_ray(origin: vec3<f32>, dir_in: vec3<f32>, max_t: f32) -> Ray {
    let dir = normalize(dir_in);
    return Ray(origin, dir, safe_inv_dir(dir), max_t);
}

// ============================================================
// AABB intersection - returns (tmin, tmax), miss if tmin > tmax
// ============================================================

fn aabb_hit(r: Ray, bmin: vec3<f32>, bmax: vec3<f32>) -> bool {
    let t0 = (bmin - r.origin) * r.inv_dir;
    let t1 = (bmax - r.origin) * r.inv_dir;

    let tmin = min(t0, t1);
    let tmax = max(t0, t1);

    let enter = max(max(tmin.x, tmin.y), tmin.z);
    let exit = min(min(tmax.x, tmax.y), tmax.z);

    // Valid hit: exit >= enter, exit > 0, enter < max_t
    return exit >= enter && exit > 0.0 && enter < r.max_t;
}

// Returns entry distance for traversal ordering
fn aabb_entry_t(r: Ray, bmin: vec3<f32>, bmax: vec3<f32>) -> f32 {
    let t0 = (bmin - r.origin) * r.inv_dir;
    let t1 = (bmax - r.origin) * r.inv_dir;
    let tmin = min(t0, t1);
    return max(max(tmin.x, tmin.y), tmin.z);
}

// ============================================================
// Möller-Trumbore - any hit (no t output needed)
// ============================================================

fn triangle_hit_t(r: Ray, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> f32 {
    let e1 = v1 - v0;
    let e2 = v2 - v0;

    let h = cross(r.dir, e2);
    let a = dot(e1, h);

    if (abs(a) < TRI_EPSILON) { return INF_F32; }

    let f = 1.0 / a;
    let s = r.origin - v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) { return INF_F32; }

    let q = cross(s, e1);
    let v = f * dot(r.dir, q);
    if (v < 0.0 || (u + v) > 1.0) { return INF_F32; }

    let t = f * dot(e2, q);
    if (t > TRI_EPSILON && t < r.max_t) { return t; }

    return INF_F32;
}

// ============================================================
// BLAS traversal - object space, exit on first hit
// ============================================================


fn traverse_blas_first_t(ray: Ray, root: u32, dbg: ptr<function, StackDebug>) -> f32 {
    var stack: array<u32, BLAS_STACK_SIZE>;
    var pter: i32 = 0;
    stack[0] = root;

    loop {
        if (pter < 0) { break; }

        let idx = stack[pter];
        pter -= 1;

        let node = blas_bvh[idx];

        if (!aabb_hit(ray, node.aabb_min, node.aabb_max)) {
            continue;
        }

        let is_leaf = (node.meta_info & 1u) != 0u;

        if (is_leaf) {
            let tri_start = node.left_or_tri_offset;
            let tri_count = node.right_or_tri_count;

            var best_t = INF_F32;

            // Min within leaf (cheap) so triangle order doesn't randomize softness too much.
            for (var i = 0u; i < tri_count; i++) {
                let base = (tri_start + i) * 3u;

                let v0 = blas_vertices[blas_indices[base + 0u]].position;
                let v1 = blas_vertices[blas_indices[base + 1u]].position;
                let v2 = blas_vertices[blas_indices[base + 2u]].position;

                let t = triangle_hit_t(ray, v0, v1, v2);
                best_t = min(best_t, t);
            }

            if (best_t < INF_F32) {
                return best_t; // EXIT: "first hit" behavior preserved
            }
        } else {
            let left = node.left_or_tri_offset;
            let right = node.right_or_tri_count;

            let left_t = aabb_entry_t(ray, blas_bvh[left].aabb_min, blas_bvh[left].aabb_max);
            let right_t = aabb_entry_t(ray, blas_bvh[right].aabb_min, blas_bvh[right].aabb_max);

            if (left_t < right_t) {
                blas_push(&stack, &pter, right, dbg);
                blas_push(&stack, &pter, left, dbg);
            } else {
                blas_push(&stack, &pter, left, dbg);
                blas_push(&stack, &pter, right, dbg);
            }
        }
    }

    return INF_F32;
}

// ============================================================
// Transform ray to object space
// ============================================================

struct LocalRayXform {
    ray: Ray,
    local_t_to_world: f32, // world_t = local_t * local_t_to_world
};

fn transform_ray_to_local_xform(r: Ray, inv_xform: mat4x4<f32>) -> LocalRayXform {
    let local_origin = (inv_xform * vec4(r.origin, 1.0)).xyz;

    let local_dir_raw = (inv_xform * vec4(r.dir, 0.0)).xyz;

    // Avoid divide-by-zero for pathological transforms
    let scale = max(1e-6, length(local_dir_raw));
    let local_dir = local_dir_raw / scale;

    // Convert world max_t into local param units (t_local = t_world * scale)
    let local_max_t = r.max_t * scale;

    return LocalRayXform(make_ray(local_origin, local_dir, local_max_t), 1.0 / scale);
}

fn is_sky_from_linear_depth(linear_depth: f32) -> bool {
    // GTAO-prep writes sky as z_far
    let z_far = uniforms.near_far_depth.y;
    return linear_depth >= (z_far - 0.01); // small epsilon in meters
}

// ============================================================
// World position reconstruction
// ============================================================

fn uv_from_pixel(src_dims: vec2<u32>, p_src: vec2<i32>) -> vec2<f32> {
    var uv = (vec2<f32>(vec2<u32>(p_src)) + 0.5) / vec2<f32>(src_dims);
    uv.y = 1.0 - uv.y; // texture Y-down -> NDC Y-up
    return uv;
}

fn view_ray_dir_from_uv(uv: vec2<f32>) -> vec3<f32> {
    let ndc_xy = uv * 2.0 - 1.0;

    // For reversed-Z, near plane is at z_ndc = 1
    // For standard, use z_ndc = 0 or any value < 1
    var z_clip = 1.0;
    if (uniforms.reversed_depth_z == 0u) {
        z_clip = 0.5; // arbitrary valid depth for standard
    }

    let clip = vec4<f32>(ndc_xy, z_clip, 1.0);
    let view_h = uniforms.inv_proj * clip;
    let view_pt = view_h.xyz / view_h.w;

    // Direction from camera (at origin in view space) to this point
    return normalize(view_pt);
}

// linear_depth is meters, expected to be -view_pos.z (>0 in front of camera)
fn reconstruct_render_pos(uv: vec2<f32>, linear_depth: f32) -> vec3<f32> {
    let dir_vs = view_ray_dir_from_uv(uv);
    let t = linear_depth / max(1e-6, -dir_vs.z);
    let view_pos = dir_vs * t;

    // view -> render/world axes (view has rotation only; translation is handled by render-space)
    return (uniforms.inv_view * vec4<f32>(view_pos, 1.0)).xyz;
}

fn camera_world_offset() -> vec3<f32> {
    // camera_chunk is (x, z)
    let cx = f32(uniforms.camera_chunk.x) * uniforms.chunk_size;
    let cz = f32(uniforms.camera_chunk.y) * uniforms.chunk_size;
    return uniforms.camera_local + vec3<f32>(cx, 0.0, cz);
}
// ============================================================
// Main entry point
// ============================================================

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(rt_out);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let pixel = vec2<i32>(gid.xy);

    let depth = textureLoad(depth_tex, pixel, 0).x;
    if (is_sky_from_linear_depth(depth)) {
        textureStore(rt_out, pixel, vec4(1.0, 0.0, 0.0, 0.0));
        return;
    }

    let normal_raw = textureLoad(normal_tex, pixel, 0).xyz;
    let N = normalize(normal_raw * 2.0 - 1.0);

    // --------------------------------------------------------
    // Gradual N·L "early-out" (soft instead of binary)
    // --------------------------------------------------------
    let NdotL = dot(N, uniforms.sun_direction);

    // 0 when NdotL <= -fade, 1 when NdotL >= +fade
    let normal_term = smoothstep(-params.ndotl_fade, params.ndotl_fade, NdotL);

    // Still keep a real early out when it's fully backfacing
    if (normal_term <= 0.0) {
        textureStore(rt_out, pixel, vec4(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let full_res_pixel = pixel * 2;
    let self_id = textureLoad(instance_id_tex, full_res_pixel, 0).x;

    let depth_dims = textureDimensions(depth_tex);
    let uv = uv_from_pixel(depth_dims, pixel);
    let world_pos = reconstruct_render_pos(uv, depth);

    let bias_normal = N * params.normal_bias;
    let bias_sun = uniforms.sun_direction * params.origin_bias;
    let ray_origin = world_pos + bias_normal + bias_sun;

    let ray = make_ray(ray_origin, uniforms.sun_direction, params.max_distance);

    // --------------------------------------------------------
    // TLAS traversal (capture first-hit distance)
    // --------------------------------------------------------
    var tlas_stack: array<u32, TLAS_STACK_SIZE>;
    var tlas_ptr: i32 = 0;
    tlas_stack[0] = 0u;

    var hit = false;
    var hit_t_world = INF_F32;
    var dbg = StackDebug(false, false);

    loop {
        if (tlas_ptr < 0 || hit) { break; }

        let node_idx = tlas_stack[tlas_ptr];
        tlas_ptr -= 1;

        let node = tlas_bvh[node_idx];
        if (!aabb_hit(ray, node.aabb_min, node.aabb_max)) { continue; }

        let is_leaf = (node.meta_info & 1u) != 0u;

        if (is_leaf) {
            let first_inst = node.left_or_tri_offset;
            let inst_count = node.right_or_tri_count;

            for (var i = 0u; i < inst_count; i++) {
                if (hit) { break; }

                let inst = tlas_instances[first_inst + i];

                if (inst.instance_id == self_id) { continue; }
                if (!aabb_hit(ray, inst.aabb_min, inst.aabb_max)) { continue; }

                let lx = transform_ray_to_local_xform(ray, inst.inv_transform);

                let t_local = traverse_blas_first_t(lx.ray, inst.blas_root, &dbg);

                if (t_local < INF_F32) {
                    hit = true;
                    hit_t_world = t_local * lx.local_t_to_world;
                }
            }
        } else {
            let left = node.left_or_tri_offset;
            let right = node.right_or_tri_count;

            let left_t = aabb_entry_t(ray, tlas_bvh[left].aabb_min, tlas_bvh[left].aabb_max);
            let right_t = aabb_entry_t(ray, tlas_bvh[right].aabb_min, tlas_bvh[right].aabb_max);

            if (left_t < right_t) {
                tlas_push(&tlas_stack, &tlas_ptr, right, &dbg);
                tlas_push(&tlas_stack, &tlas_ptr, left, &dbg);
            } else {
                tlas_push(&tlas_stack, &tlas_ptr, left, &dbg);
                tlas_push(&tlas_stack, &tlas_ptr, right, &dbg);
            }
        }
    }

    // --------------------------------------------------------
    // Softening with travel distance (single-ray approximation)
    //  - near blocker => dark (visibility ~ 0)
    //  - far blocker  => lighter (visibility -> far_hit_visibility)
    // --------------------------------------------------------
    var visibility = 1.0;

    if (hit) {
        // let s = smoothstep(params.soft_start, params.soft_end, hit_t_world);
        visibility = 0.0;
    }

    visibility = saturate(visibility * normal_term);
    let debug_frame = (uniforms.frame_index & 1u) == 0u;

    if (debug_frame && (dbg.tlas_overflow || dbg.blas_overflow)) {
        // In r8unorm, flash between 0 and 1 — the ONLY values
        // that will be visually obvious in the final composite
        textureStore(rt_out, pixel, vec4(0.0, 0.0, 0.0, 0.0)); // full shadow
        return;
    }
    if (!debug_frame && (dbg.tlas_overflow || dbg.blas_overflow)) {
        textureStore(rt_out, pixel, vec4(1.0, 0.0, 0.0, 0.0)); // full light
        return;
    }

    textureStore(rt_out, pixel, vec4(visibility, 0.0, 0.0, 0.0));
}