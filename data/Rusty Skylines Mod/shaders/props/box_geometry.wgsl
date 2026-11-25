struct PropVertexOut {
    position_x: f32,
    position_y: f32,
    position_z: f32,
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    color_a: f32,
    uv_x: f32,
    uv_y: f32,
    texture_id: u32,
}

struct GeometryParams {
    vertex_count: u32,
    index_count: u32,
    lod: u32,
    seed: u32,
    data: array<vec4<f32>, 4>,
}

const CUBE_INDICES: array<u32, 36> = array<u32, 36>(
    0u, 2u, 1u, 0u, 3u, 2u,
    4u, 5u, 6u, 4u, 6u, 7u,
    0u, 4u, 7u, 0u, 7u, 3u,
    1u, 2u, 6u, 1u, 6u, 5u,
    0u, 1u, 5u, 0u, 5u, 4u,
    3u, 7u, 6u, 3u, 6u, 2u
);

@group(2) @binding(0)
var<storage, read_write> vertices: array<PropVertexOut>;

@group(2) @binding(1)
var<storage, read_write> indices: array<u32>;

@group(2) @binding(2)
var<uniform> params: GeometryParams;

fn cube_corner(id: u32, h: vec3<f32>) -> vec3<f32> {
    switch id {
        case 0u: { return vec3<f32>(-h.x, -h.y, -h.z); }
        case 1u: { return vec3<f32>( h.x, -h.y, -h.z); }
        case 2u: { return vec3<f32>( h.x,  h.y, -h.z); }
        case 3u: { return vec3<f32>(-h.x,  h.y, -h.z); }
        case 4u: { return vec3<f32>(-h.x, -h.y,  h.z); }
        case 5u: { return vec3<f32>( h.x, -h.y,  h.z); }
        case 6u: { return vec3<f32>( h.x,  h.y,  h.z); }
        default: { return vec3<f32>(-h.x, h.y, h.z); }
    }
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < params.index_count) {
        indices[gid.x] = gid.x;
    }

    if (gid.x >= params.vertex_count) {
        return;
    }

    let h = max(params.data[0].xyz, vec3<f32>(0.001));

    let tri = gid.x / 3u;
    let corner = gid.x % 3u;
    let base = tri * 3u;

    let a = cube_corner(CUBE_INDICES[base], h);
    let b = cube_corner(CUBE_INDICES[base + 1u], h);
    let c = cube_corner(CUBE_INDICES[base + 2u], h);

    let normal = normalize(cross(b - a, c - a));

    var position = a;

    if (corner == 1u) {
        position = b;
    } else if (corner == 2u) {
        position = c;
    }

    let uv = position.xz / (h.xz * 2.0) + vec2<f32>(0.5);

    vertices[gid.x] = PropVertexOut(
        position.x,
        position.y,
        position.z,
        normal.x,
        normal.y,
        normal.z,
        params.data[1].x,
        params.data[1].y,
        params.data[1].z,
        1.0,
        uv.x,
        uv.y,
        0u
    );
}