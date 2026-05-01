#include "includes/uniforms.wgsl"

struct ShadowUniform {
    light_view_proj: mat4x4<f32>,
    cascade_idx: u32
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) texture_id: u32
};

struct InstanceInput {
    @location(5)  model_col0: vec4<f32>,
    @location(6)  model_col1: vec4<f32>,
    @location(7)  model_col2: vec4<f32>,
    @location(8)  model_col3: vec4<f32>,
    @location(9)  prev_model_col0: vec4<f32>,
    @location(10) prev_model_col1: vec4<f32>,
    @location(11) prev_model_col2: vec4<f32>,
    @location(12) prev_model_col3: vec4<f32>,
    @location(13) color: vec4<f32>,
    @location(14) misc: vec4<f32>, // x: seed, y: wind_strength, z: variant, w: unused
};
struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) texture_id: u32,
};
@group(0) @binding(0) var texture_sampler: sampler;
@group(0) @binding(2) var tex1: texture_2d<f32>;
@group(0) @binding(3) var tex2: texture_2d<f32>;
@group(0) @binding(4) var tex3: texture_2d<f32>;
@group(0) @binding(5) var tex4: texture_2d<f32>;
@group(1) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(1) var<uniform> shadow_uniforms: ShadowUniform;

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VSOut {
    var out: VSOut;

    let model = mat4x4<f32>(
        instance.model_col0,
        instance.model_col1,
        instance.model_col2,
        instance.model_col3
    );

    let seed = instance.misc.x;
    let wind_strength = instance.misc.y;
    let time = uniforms.time;

    var local_pos = vertex.position;

    let height_factor = max(local_pos.y, 0.0);
    let wind_offset = sin(time * 2.0 + seed * 6.283) * wind_strength * height_factor * 0.1;
    local_pos.x += wind_offset;
    local_pos.z += wind_offset * 0.5;

    let world_pos = model * vec4<f32>(local_pos, 1.0);

    out.position = shadow_uniforms.light_view_proj * world_pos;
    out.uv = vertex.uv;
    out.texture_id = vertex.texture_id;

    return out;
}

@fragment
fn fs_main(in: VSOut) {
    var alpha: f32;

    switch (in.texture_id) {
        case 1u: {
            alpha = textureSample(tex1, texture_sampler, in.uv).a;
        }
        case 2u: {
            alpha = textureSample(tex2, texture_sampler, in.uv).a;
        }
        case 3u: {
            alpha = textureSample(tex3, texture_sampler, in.uv).a;
        }
        case 4u: {
            alpha = textureSample(tex4, texture_sampler, in.uv).a;
        }
        default: {
            alpha = 1.0;
        }
    }

    let t = f32(shadow_uniforms.cascade_idx) / 3.0;
    let cutoff = mix(0.75, 0.0, t);

    if (alpha < cutoff) {
        discard;
    }
}