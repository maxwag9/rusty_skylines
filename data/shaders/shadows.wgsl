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
    reversed_depth_z: u32,
    shadows_enabled: u32,
    _pad_2: vec2<u32>,     // padding to 16 bytes
};
struct ShadowUniform {
    light_view_proj: mat4x4<f32>,
};

@group(1) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(1) var<uniform> shadow_uniforms: ShadowUniform;

struct VertexInput {
    @location(0) chunk_xz: vec2<i32>,
    @location(1) local_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    let dc = in.chunk_xz - uniforms.camera_chunk;
    let world_x = f32(dc.x) * uniforms.chunk_size + (in.local_pos.x - uniforms.camera_local.x);
    let world_y = in.local_pos.y - uniforms.camera_local.y;
    let world_z = f32(dc.y) * uniforms.chunk_size + (in.local_pos.z - uniforms.camera_local.z);

    let world_pos = vec3<f32>(world_x, world_y, world_z);
    return shadow_uniforms.light_view_proj * vec4<f32>(world_pos, 1.0);
}