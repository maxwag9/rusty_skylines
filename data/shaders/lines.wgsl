#include "includes/uniforms.wgsl"

@group(1) @binding(0) var<uniform> uniforms: Uniforms;

struct PickUniform {
    pos: vec3<f32>,
    radius: f32,
    enabled: u32,
    color: vec3<f32>,
}

//@group(2) @binding(0)
//var<uniform> pick: PickUniform;
struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) curr_clip: vec4<f32>,
    @location(2) prev_clip: vec4<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>, @location(1) color: vec3<f32>) -> VSOut {
    var out: VSOut;
    let clip = uniforms.view_proj * vec4<f32>(pos, 1.0);
    out.pos = clip;
    out.color = color;
    out.curr_clip = clip;
    // Debug lines are in render-space (camera-relative), same position both frames
    // If camera moved, prev_view_proj handles it
    out.prev_clip = uniforms.prev_view_proj * vec4<f32>(pos, 1.0);
    return out;
}


struct FragOut {
    @location(0) color: vec4<f32>,
    @location(2) motion: vec2<f32>,
};

@fragment
fn fs_main(in: VSOut) -> FragOut {
    var out: FragOut;
    out.color = vec4<f32>(in.color, 1.0);

    let curr_ndc = in.curr_clip.xy / in.curr_clip.w;
    let prev_ndc = in.prev_clip.xy / in.prev_clip.w;
    let curr_uv = curr_ndc * vec2(0.5, -0.5) + 0.5;
    let prev_uv = prev_ndc * vec2(0.5, -0.5) + 0.5;
    out.motion = curr_uv - prev_uv;

    return out;
}
