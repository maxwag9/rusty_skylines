#include "includes/uniforms.wgsl"

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct InstanceInput {
    @location(4) model_col0: vec4<f32>,
    @location(5) model_col1: vec4<f32>,
    @location(6) model_col2: vec4<f32>,
    @location(7) model_col3: vec4<f32>,
};

struct VSOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) instance_id: u32,
};

@group(1) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(
    v: VertexInput,
    i: InstanceInput,
    @builtin(instance_index) instance_index: u32
) -> VSOut {
    var out: VSOut;

    let model = mat4x4<f32>(
        i.model_col0,
        i.model_col1,
        i.model_col2,
        i.model_col3,
    );

    let world = model * vec4<f32>(v.position, 1.0);

    out.clip_position = uniforms.view_proj * world;
    out.instance_id = instance_index;

    return out;
}

struct FSOut {
    @location(0) instance_id: u32,
};

@fragment
fn fs_main(in: VSOut) -> FSOut {
    var out: FSOut;
    out.instance_id = in.instance_id;
    return out;
}