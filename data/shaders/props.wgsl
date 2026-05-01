#include "includes/shadow.wgsl"
#include "includes/uniforms.wgsl"

@group(0) @binding(0) var texture_sampler: sampler;
@group(0) @binding(2) var tex1: texture_2d<f32>;
@group(0) @binding(3) var tex2: texture_2d<f32>;
@group(0) @binding(4) var tex3: texture_2d<f32>;
@group(0) @binding(5) var tex4: texture_2d<f32>;
@group(0) @binding(6) var s_shadow: sampler_comparison;
@group(0) @binding(7) var t_shadow: texture_depth_2d_array;

@group(1) @binding(0) var<uniform> uniforms: Uniforms;


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

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) instance_color: vec4<f32>,
    @location(4) texture_id: u32,
    @location(5) @interpolate(flat) instance_id: u32,
    @location(6) curr_pos_cs: vec4<f32>,
    @location(7) prev_pos_cs: vec4<f32>,
    @location(8) misc: vec4<f32>,
    @location(9) vertex_color: vec4<f32>,
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
    out.texture_id = vertex.texture_id;
    out.instance_id = instance_index;
    out.curr_pos_cs = out.clip_position;
    out.prev_pos_cs = uniforms.prev_view_proj * prev_world_pos;
    out.misc = instance.misc;

    return out;
}


@fragment
fn fs_main(in: VertexOutput) -> FragmentOut {
    var out: FragmentOut;

    // Sample texture based on texture_id, multiply by instance color
    var tex_color: vec4<f32>;

    switch (in.texture_id) {
        case 1u: {
            tex_color = textureSample(tex1, texture_sampler, in.uv);
        }
        case 2u: {
            tex_color = textureSample(tex2, texture_sampler, in.uv);
        }
        case 3u: {
            tex_color = textureSample(tex3, texture_sampler, in.uv);
        }
        case 4u: {
            tex_color = textureSample(tex4, texture_sampler, in.uv);
        }
        default: {
            // texture_id 0: flat color, no texture
            tex_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
        }
    }

    // Combine texture with vertex and instance colors
    let base_color = tex_color * in.vertex_color * in.instance_color;

    // Alpha test for leaf cards and other transparent textures
    if (base_color.a < 0.8) {
        discard;
    }

    let variant = in.misc.z;
    let seed = in.misc.x;
    let color_variation = vec3<f32>(
        1.0 + sin(variant * 1.1 + seed) * 0.1,
        1.0 + sin(variant * 2.3 + seed) * 0.1,
        1.0 + sin(variant * 3.7 + seed) * 0.05
    );
    let varied_color = base_color.rgb * color_variation;

    let N = normalize(in.world_normal);
    let L = normalize(uniforms.sun_direction);
    let n_dot_l = max(dot(N, L), 0.0);
    let shadow = fetch_shadow(in.world_pos, N, L);
    let horizon_fade = smoothstep(0.0, 0.1, saturate(L.y));
    let ambient = 0.2*horizon_fade;
    let diffuse = n_dot_l * 0.5;
    let light_factor = diffuse * shadow * horizon_fade + ambient;
    let lit_color = varied_color * light_factor;
    out.color = vec4<f32>(lit_color, base_color.a);

    out.normal = vec4<f32>(in.world_normal * 0.5 + 0.5, 1.0);

    let curr_ndc = in.curr_pos_cs.xy / in.curr_pos_cs.w;
    let prev_ndc = in.prev_pos_cs.xy / in.prev_pos_cs.w;
    let velocity = (curr_ndc - prev_ndc) * 0.5;
    out.motion = velocity;

    out.instance_id = in.instance_id;

    return out;
}