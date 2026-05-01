#include "includes/uniforms.wgsl"

@group(1) @binding(0) var<uniform> uniforms: Uniforms;

struct PickUniform {
    pos: vec3<f32>,
    radius: f32,
    enabled: u32,
    color: vec3<f32>,
}
@group(0) @binding(0)
var atlas_sampler: sampler;
@group(0) @binding(2)
var atlas_texture: texture_2d<f32>;

//@group(2) @binding(0)
//var<uniform> pick: PickUniform;
struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) curr_clip: vec4<f32>,
    @location(3) prev_clip: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>
) -> VSOut {

    var out: VSOut;

    let clip = uniforms.view_proj * vec4<f32>(pos, 1.0);

    out.pos = clip;
    out.uv = uv;
    out.color = color;

    out.curr_clip = clip;
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

    let tex_color = textureSample(atlas_texture, atlas_sampler, in.uv);

    let alpha = tex_color.r * in.color.a;
    out.color = vec4<f32>(in.color.rgb, alpha);

    let curr_ndc = in.curr_clip.xy / in.curr_clip.w;
    let prev_ndc = in.prev_clip.xy / in.prev_clip.w;

    let curr_uv = curr_ndc * vec2<f32>(0.5, -0.5) + 0.5;
    let prev_uv = prev_ndc * vec2<f32>(0.5, -0.5) + 0.5;

    out.motion = curr_uv - prev_uv;

    return out;
}
