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
    @location(5) chunk_xz: vec2<i32>,
    @location(6) local_pos: vec3<f32>,
    @location(7) scale: f32,

    @location(8) rotation_sin: f32,
    @location(9) rotation_cos: f32,
    @location(10) seed: f32,
    @location(11) wind_strength: f32,

    @location(12) color: vec4<f32>,
};


struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) texture_id: u32,
};


@group(0) @binding(0) var texture_sampler: sampler;
@group(0) @binding(2) var tex1: texture_2d<f32>;
@group(0) @binding(3) var tex2: texture_2d<f32>;
@group(0) @binding(4) var tex3: texture_2d<f32>;
@group(0) @binding(5) var tex4: texture_2d<f32>;

@group(1) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(1) var<uniform> shadow_uniforms: ShadowUniform;


fn get_instance_position(instance: InstanceInput) -> vec3<f32> {
    let dc = instance.chunk_xz - uniforms.camera_chunk;

    return vec3<f32>(
        f32(dc.x) * uniforms.chunk_size +
            (instance.local_pos.x - uniforms.camera_local.x),

        instance.local_pos.y - uniforms.camera_local.y,

        f32(dc.y) * uniforms.chunk_size +
            (instance.local_pos.z - uniforms.camera_local.z)
    );
}


fn transform_vertex(pos: vec3<f32>, instance: InstanceInput) -> vec3<f32> {
    let c = instance.rotation_cos;
    let s = instance.rotation_sin;

    let rotated = vec3<f32>(
        pos.x * c - pos.z * s,
        pos.y,
        pos.x * s + pos.z * c
    );

    return get_instance_position(instance) + rotated * instance.scale;
}


@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> VSOut {
    var out: VSOut;

    var local_pos = vertex.position;

    let wind_offset =
        sin(uniforms.time * 2.0 + instance.seed * 6.283)
        * instance.wind_strength
        * max(local_pos.y, 0.0)
        * 0.1;

    local_pos.x += wind_offset;
    local_pos.z += wind_offset * 0.5;


    let world_pos = transform_vertex(local_pos, instance);

    out.position =
        shadow_uniforms.light_view_proj *
        vec4<f32>(world_pos, 1.0);

    out.uv = vertex.uv;
    out.texture_id = vertex.texture_id;

    return out;
}


@fragment
fn fs_main(in: VSOut) {

    var alpha = 1.0;

    switch(in.texture_id) {
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
        default: {}
    }

    let t = f32(shadow_uniforms.cascade_idx) / 3.0;
    let cutoff = mix(0.75, 0.0, t);

    if (alpha < cutoff) {
        discard;
    }
}