#include "includes/shadow.wgsl"
#include "includes/uniforms.wgsl"
#include "includes/simple_lighting.wgsl"

struct VertexInput {
    @location(0) chunk_xz: vec2<i32>,
    @location(1) position : vec3<f32>,
    @location(2) normal   : vec3<f32>,
    @location(3) uv       : vec2<f32>,
    @location(4) color   : vec4<f32>,
    @location(5) material_id : u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) material_id: u32,
    @location(3) world_normal: vec3<f32>,
    @location(4) world_pos: vec3<f32>,
    @location(5) curr_clip: vec4<f32>,
    @location(6) prev_clip: vec4<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let dc: vec2<i32> = input.chunk_xz - uniforms.camera_chunk;

    let rx = f32(dc.x) * uniforms.chunk_size + (input.position.x - uniforms.camera_local.x);
    let ry = input.position.y - uniforms.camera_local.y;
    let rz = f32(dc.y) * uniforms.chunk_size + (input.position.z - uniforms.camera_local.z);

    let render_pos = vec3<f32>(rx, ry, rz);

    let clip = uniforms.view_proj * vec4<f32>(render_pos, 1.0);
    out.clip_position = clip;
    out.curr_clip = clip;

    // Previous frame
    let prev_dc = input.chunk_xz - uniforms.prev_camera_chunk;
    let prev_rx = f32(prev_dc.x) * uniforms.chunk_size + (input.position.x - uniforms.prev_camera_local.x);
    let prev_ry = input.position.y - uniforms.prev_camera_local.y;
    let prev_rz = f32(prev_dc.y) * uniforms.chunk_size + (input.position.z - uniforms.prev_camera_local.z);

    let prev_render_pos = vec3<f32>(prev_rx, prev_ry, prev_rz);
    out.prev_clip = uniforms.prev_view_proj * vec4<f32>(prev_render_pos, 1.0);

    out.uv = input.uv;
    out.color = input.color;
    out.material_id = input.material_id;
    out.world_normal = input.normal;
    out.world_pos = render_pos;

    return out;
}

@group(0) @binding(0) var building_sampler : sampler;
@group(0) @binding(1) var textures : texture_2d_array<f32>;
@group(0) @binding(2) var s_shadow: sampler_comparison;
@group(0) @binding(3) var t_shadow: texture_depth_2d_array;

@group(1) @binding(0) var<uniform> uniforms : Uniforms;

struct FragmentOut {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) motion: vec2<f32>,
};

@fragment
fn fs_main(input: VertexOutput) -> FragmentOut {
    var out: FragmentOut;

    let tex = textureSample(
        textures,
        building_sampler,
        input.uv,
        input.material_id
    );

    let N = normalize(input.world_normal);
    let albedo = tex.rgb * input.color.rgb;

    let lighting = calculate_world_lighting(
        input.world_pos,
        N,
        0.0,
        uniforms.solar_eclipse_coverage
    );

    let rgb = albedo * lighting;

    out.color = vec4<f32>(rgb, input.color.a);
    out.normal = vec4<f32>(N * 0.5 + 0.5, 1.0);

    let curr_ndc = input.curr_clip.xy / input.curr_clip.w;
    let prev_ndc = input.prev_clip.xy / input.prev_clip.w;
    let curr_uv = curr_ndc * vec2(0.5, -0.5) + 0.5;
    let prev_uv = prev_ndc * vec2(0.5, -0.5) + 0.5;

    out.motion = curr_uv - prev_uv;

    return out;
}
