// ============================================================
// Ray Tracing Compute Shader (Bring-up)
// ============================================================

struct RTVertex {
    position : vec3<f32>,
    _pad     : f32,
};

struct BvhNode {
    aabb_min : vec3<f32>,
    _pad0    : f32,
    aabb_max : vec3<f32>,
    _pad1    : f32,
    child_or_tri_offset : u32,
    meta_info                : u32,
    tri_count_or_right  : u32,
    _pad2               : u32,
};

struct TlasInstance {
    transform : mat4x4<f32>,
    inv_transform : mat4x4<f32>,
    blas_root : u32,
    _pad0 : vec3<u32>,
};

// ------------------------------------------------------------
// Bindings (group 2)
// ------------------------------------------------------------

// binding 0: BLAS vertices
@group(2) @binding(0)
var<storage, read> blas_vertices : array<RTVertex>;

// binding 1: BLAS indices
@group(2) @binding(1)
var<storage, read> blas_indices : array<u32>;

// binding 2: BLAS BVH nodes
@group(2) @binding(2)
var<storage, read> blas_bvh : array<BvhNode>;

// binding 3: TLAS instances
@group(2) @binding(3)
var<storage, read> tlas_instances : array<TlasInstance>;

// binding 4: TLAS BVH nodes
@group(2) @binding(4)
var<storage, read> tlas_bvh : array<BvhNode>;

// ------------------------------------------------------------
// Ray + helpers
// ------------------------------------------------------------

struct Ray {
    origin : vec3<f32>,
    dir    : vec3<f32>,
};

fn intersect_aabb(ray : Ray, minv : vec3<f32>, maxv : vec3<f32>) -> bool {
    let inv_dir = 1.0 / ray.dir;

    let t0 = (minv - ray.origin) * inv_dir;
    let t1 = (maxv - ray.origin) * inv_dir;

    let tmin = max(min(t0.x, t1.x), max(min(t0.y, t1.y), min(t0.z, t1.z)));
    let tmax = min(max(t0.x, t1.x), min(max(t0.y, t1.y), max(t0.z, t1.z)));

    return tmax >= max(tmin, 0.0);
}

// ------------------------------------------------------------
// Compute entry
// ------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) gid : vec3<u32>
) {
    // Screen-space ray (temporary)
    let ray = Ray(
        vec3<f32>(0.0, 1.0, 5.0),
        normalize(vec3<f32>(
            f32(gid.x) * 0.001 - 1.0,
            0.0,
            -1.0
        ))
    );

    var stack : array<u32, 64>;
    var stack_ptr : i32 = 0;

    stack[0] = 0u;

    loop {
        if (stack_ptr < 0) {
            break;
        }

        let node_idx = stack[stack_ptr];
        stack_ptr -= 1;

        let node = tlas_bvh[node_idx];

        if (!intersect_aabb(ray, node.aabb_min, node.aabb_max)) {
            continue;
        }

        if ((node.meta_info & 1u) == 1u) {
            // Leaf: instance range
            let first = node.child_or_tri_offset;
            let count = node.tri_count_or_right;

            for (var i = 0u; i < count; i = i + 1u) {
                let inst = tlas_instances[first + i];

                // TODO: transform ray and traverse BLAS
                _ = inst.blas_root;
            }
        } else {
            // Inner node
            let left = node.child_or_tri_offset;
            let right = node.tri_count_or_right;

            stack_ptr += 1;
            stack[stack_ptr] = right;

            stack_ptr += 1;
            stack[stack_ptr] = left;
        }
    }
}
