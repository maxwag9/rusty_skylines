#include "includes/uniforms.wgsl"

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