#include "includes/uniforms.wgsl"

@group(0) @binding(0) var texture_sampler: sampler;
@group(0) @binding(1) var leaves: texture_2d<f32>;
@group(1) @binding(0) var<uniform> uniforms: Uniforms;


struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) uv: vec2<f32>,
};

struct InstanceInput {
    @location(4)  model_col0: vec4<f32>,
    @location(5)  model_col1: vec4<f32>,
    @location(6)  model_col2: vec4<f32>,
    @location(7)  model_col3: vec4<f32>,
    @location(8)  prev_model_col0: vec4<f32>,
    @location(9)  prev_model_col1: vec4<f32>,
    @location(10) prev_model_col2: vec4<f32>,
    @location(11) prev_model_col3: vec4<f32>,
    @location(12) color: vec4<f32>,
    @location(13) misc: vec4<f32>, // x: seed, y: wind_strength, z: variant, w: unused
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) instance_color: vec4<f32>,
    @location(4) @interpolate(flat) instance_id: u32,
    @location(5) curr_pos_cs: vec4<f32>,
    @location(6) prev_pos_cs: vec4<f32>,
    @location(7) misc: vec4<f32>,
    @location(8) vertex_color: vec4<f32>,
};

struct FragmentOut {
    @location(0) color: vec4<f32>,     // color target
    @location(1) normal: vec4<f32>,    // normal target
    @location(2) instance_id: u32,     // R32Uint instance id target for RAY TRACING
    @location(3) motion: vec2<f32>,
};


@vertex
fn vs_main(
    vertex: VertexInput,
    instance: InstanceInput,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    var out: VertexOutput;

    let model = mat4x4<f32>(
        instance.model_col0,
        instance.model_col1,
        instance.model_col2,
        instance.model_col3,
    );
    let prev_model = mat4x4<f32>(
        instance.prev_model_col0,
        instance.prev_model_col1,
        instance.prev_model_col2,
        instance.prev_model_col3,
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
    let prev_world_pos = prev_model * vec4<f32>(vertex.position, 1.0);

    out.clip_position = uniforms.view_proj * world_pos;
    out.uv = vertex.uv;
    out.world_normal = normalize((model * vec4<f32>(vertex.normal, 0.0)).xyz);
    out.world_pos = world_pos.xyz;
    out.instance_color = instance.color;
    out.vertex_color = vertex.color;
    out.instance_id = instance_index;
    out.curr_pos_cs = out.clip_position;
    out.prev_pos_cs = uniforms.prev_view_proj * prev_world_pos;
    out.misc = instance.misc;

    return out;
}


@fragment
fn fs_main(in: VertexOutput) -> FragmentOut {
    var out: FragmentOut;

    let base_color = in.vertex_color * in.instance_color;

    let variant = in.misc.z;
    let seed = in.misc.x;
    let color_variation = vec3<f32>(
        1.0 + sin(variant * 1.1 + seed) * 0.1,
        1.0 + sin(variant * 2.3 + seed) * 0.1,
        1.0 + sin(variant * 3.7 + seed) * 0.05
    );
    let varied_color = base_color.xyz * color_variation;

    let light_dir = normalize(vec3<f32>(0.4, 0.8, 0.3));
    let n_dot_l = max(dot(in.world_normal, light_dir), 0.0);

    let ambient = 0.35;
    let diffuse = n_dot_l * 0.65;
    let lit_color = varied_color * (ambient + diffuse);

    out.color = vec4<f32>(lit_color, base_color.w);

    out.normal = vec4<f32>(in.world_normal * 0.5 + 0.5, 1.0);

    let curr_ndc = in.curr_pos_cs.xy / in.curr_pos_cs.w;
    let prev_ndc = in.prev_pos_cs.xy / in.prev_pos_cs.w;
    let velocity = (curr_ndc - prev_ndc) * 0.5;
    out.motion = velocity;

    out.instance_id = in.instance_id;

    return out;
}