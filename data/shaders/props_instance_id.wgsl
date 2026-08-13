#include "includes/uniforms.wgsl"

struct VertexInput {
    @location(0) position: vec3<f32>
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
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) instance_id: u32
};

@group(1) @binding(0) var<uniform> uniforms: Uniforms;


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
fn vs_main(
    v: VertexInput,
    i: InstanceInput,
    @builtin(instance_index) instance_index: u32
) -> VSOut {
    var out: VSOut;

    let world = transform_vertex(v.position, i);

    out.clip_position = uniforms.view_proj * vec4<f32>(world, 1.0);
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