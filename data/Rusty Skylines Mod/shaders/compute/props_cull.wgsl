struct PropInstance {
    chunk_x: i32,
    chunk_z: i32,
    local_x: f32,
    local_y: f32,
    local_z: f32,
    scale: f32,
    rotation_sin: f32,
    rotation_cos: f32,
    seed: f32,
    wind_strength: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    color_a: f32,
}

struct DrawArgs {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct CullParams {
    camera_chunk: vec2<i32>,
    padding0: vec2<u32>,
    camera_local: vec4<f32>,
    chunk_size: f32,
    bounds_radius: f32,
    instance_count: u32,
    capacity: u32,
    lod_distances_squared: vec4<f32>,
    frustum_planes: array<vec4<f32>, 6>,
}

@group(2) @binding(0)
var<storage, read> source_instances: array<PropInstance>;

@group(2) @binding(1)
var<storage, read_write> visible_lod0: array<PropInstance>;

@group(2) @binding(2)
var<storage, read_write> visible_lod1: array<PropInstance>;

@group(2) @binding(3)
var<storage, read_write> visible_lod2: array<PropInstance>;

@group(2) @binding(4)
var<storage, read_write> visible_lod3: array<PropInstance>;

@group(2) @binding(5)
var<storage, read_write> draw_args: array<DrawArgs, 4>;

@group(2) @binding(6)
var<uniform> params: CullParams;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;

    if (id >= params.instance_count) {
        return;
    }

    let instance = source_instances[id];

    let chunk_delta = vec2<i32>(
        instance.chunk_x,
        instance.chunk_z
    ) - params.camera_chunk;

    let world_position = vec3<f32>(
        f32(chunk_delta.x) * params.chunk_size
            + instance.local_x - params.camera_local.x,
        instance.local_y - params.camera_local.y,
        f32(chunk_delta.y) * params.chunk_size
            + instance.local_z - params.camera_local.z
    );

    let radius = params.bounds_radius * abs(instance.scale);

    for (var i = 0u; i < 6u; i++) {
        let plane = params.frustum_planes[i];

        if (dot(plane.xyz, world_position) + plane.w < -radius) {
            return;
        }
    }

    let distance_squared = dot(world_position, world_position);

    var lod = 0u;

    if (distance_squared >= params.lod_distances_squared.x) {
        lod = 1u;
    }

    if (distance_squared >= params.lod_distances_squared.y) {
        lod = 2u;
    }

    if (distance_squared >= params.lod_distances_squared.z) {
        lod = 3u;
    }

    let slot = atomicAdd(&draw_args[lod].instance_count, 1u);

    if (slot >= params.capacity) {
        return;
    }

    switch (lod) {
        case 0u: {
            visible_lod0[slot] = instance;
        }
        case 1u: {
            visible_lod1[slot] = instance;
        }
        case 2u: {
            visible_lod2[slot] = instance;
        }
        default: {
            visible_lod3[slot] = instance;
        }
    }
}