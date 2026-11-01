#include "includes/shadow.wgsl"
#include "includes/uniforms.wgsl"
#include "includes/simple_lighting.wgsl"

struct RoadAppearance {
    tint: vec4<f32>,
}

struct VertexInput {
    @location(0) chunk_xz: vec2<i32>,
    @location(1) position : vec3<f32>,
    @location(2) normal   : vec3<f32>,
    @location(3) uv       : vec2<f32>,
    @location(4) material_id : u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) material_id: u32,
    @location(2) world_normal: vec3<f32>,
    @location(3) world_pos: vec3<f32>,
    @location(4) curr_clip: vec4<f32>,
    @location(5) prev_clip: vec4<f32>,
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
    out.material_id = input.material_id;
    out.world_normal = input.normal;
    out.world_pos = render_pos;

    return out;
}

// ---- Bind group 0: road materials ----
@group(0) @binding(0) var road_sampler : sampler;
@group(0) @binding(2) var tex0 : texture_2d<f32>; // concrete
@group(0) @binding(3) var tex1 : texture_2d<f32>; // goo (filler between new and old asphalt)
@group(0) @binding(4) var tex2 : texture_2d<f32>; // asphalt (new, black)
@group(0) @binding(5) var tex3 : texture_2d<f32>; // asphalt (brighter black, "new" but worn)
@group(0) @binding(6) var tex4 : texture_2d<f32>; // asphalt (kinda orange old "new age" asphalt I see in germany)
@group(0) @binding(7) var tex5 : texture_2d<f32>; // asphalt (old, gray, rough asphalt)
// Keep these comments!
@group(0) @binding(8) var s_shadow: sampler_comparison;
@group(0) @binding(9) var t_shadow: texture_depth_2d_array;

@group(1) @binding(0) var<uniform> uniforms : Uniforms;
@group(1) @binding(1) var<uniform> road_appearance: RoadAppearance;

struct FragmentOut {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) motion: vec2<f32>,
};

@fragment
fn fs_main(input: VertexOutput) -> FragmentOut {
    var out: FragmentOut;

    let uv = input.uv;
    var tex: vec4<f32>;

    if (input.material_id == 0u) {
        tex = textureSample(tex0, road_sampler, uv);
    } else if (input.material_id == 1u) {
        tex = textureSample(tex1, road_sampler, uv);
    } else if (input.material_id == 2u) {
        tex = textureSample(tex2, road_sampler, uv);
    } else if (input.material_id == 3u) {
        tex = textureSample(tex3, road_sampler, uv);
    } else if (input.material_id == 4u) {
        tex = textureSample(tex4, road_sampler, uv);
    } else if (input.material_id == 5u) {
        tex = textureSample(tex5, road_sampler, uv);
    } else {
        tex = vec4<f32>(1.0, 0.0, 1.0, 1.0);
    }

    let N = normalize(input.world_normal);
    let albedo = tex.rgb * road_appearance.tint.rgb;

    let lighting = calculate_world_lighting(
        input.world_pos,
        N,
        0.0,
        uniforms.solar_eclipse_coverage
    );

    let rgb = albedo * lighting;
    let a = tex.a * road_appearance.tint.a;

    out.color = vec4<f32>(rgb, a);
    out.normal = vec4<f32>(N * 0.5 + 0.5, 1.0);

    let curr_ndc = input.curr_clip.xy / input.curr_clip.w;
    let prev_ndc = input.prev_clip.xy / input.prev_clip.w;
    let curr_uv = curr_ndc * vec2(0.5, -0.5) + 0.5;
    let prev_uv = prev_ndc * vec2(0.5, -0.5) + 0.5;

    out.motion = curr_uv - prev_uv;

    return out;
}

