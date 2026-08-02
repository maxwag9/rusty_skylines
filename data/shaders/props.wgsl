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
    @location(5) chunk_xz: vec2<i32>,
    @location(6) local_pos: vec3<f32>,
    @location(7) scale: f32,

    @location(8) rotation: f32,
    @location(9) seed: f32,
    @location(10) wind_strength: f32,

    @location(11) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) render_pos: vec3<f32>,
    @location(3) instance_color: vec4<f32>,
    @location(4) @interpolate(flat) texture_id: u32,
    @location(5) prev_pos_cs: vec4<f32>,
    @location(6) seed: f32,
    @location(7) vertex_color: vec4<f32>
};

struct FragmentOut {
    @location(0) color: vec4<f32>,     // color target
    @location(1) normal: vec4<f32>,    // normal target
    @location(2) motion: vec2<f32>
};


@vertex
fn vs_main(
    vertex: VertexInput,
    instance: InstanceInput
) -> VertexOutput {
    var out: VertexOutput;

    let dc: vec2<i32> = instance.chunk_xz - uniforms.camera_chunk;

    let rx = f32(dc.x) * uniforms.chunk_size + (instance.local_pos.x - uniforms.camera_local.x);
    let ry = instance.local_pos.y - uniforms.camera_local.y;
    let rz = f32(dc.y) * uniforms.chunk_size + (instance.local_pos.z - uniforms.camera_local.z);

    let instance_pos = vec3<f32>(rx, ry, rz);

    let seed = instance.seed;
    let wind_strength = instance.wind_strength;
    let time = uniforms.time;

    var local_pos = vertex.position;

    let height_factor = max(local_pos.y, 0.0);
    let wind_offset = sin(time * 2.0 + seed * 6.283) * wind_strength * height_factor * 0.1;
    local_pos.x += wind_offset;
    local_pos.z += wind_offset * 0.5;

    let c = cos(instance.rotation);
    let s = sin(instance.rotation);

    let rotated = vec3<f32>(
        local_pos.x * c - local_pos.z * s,
        local_pos.y,
        local_pos.x * s + local_pos.z * c
    );

    let transformed = rotated * instance.scale;

    let render_pos = instance_pos + transformed;

    let normal_rotated = vec3<f32>(
        vertex.normal.x * c - vertex.normal.z * s,
        vertex.normal.y,
        vertex.normal.x * s + vertex.normal.z * c
    );

    out.clip_position = uniforms.view_proj * vec4<f32>(render_pos, 1.0);
    out.uv = vertex.uv;
    out.world_normal = normalize(normal_rotated);
    out.render_pos = render_pos;
    out.instance_color = instance.color;
    out.vertex_color = vertex.color;
    out.texture_id = vertex.texture_id;
    out.prev_pos_cs = out.clip_position;
    out.seed = instance.seed;
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
    if (base_color.a < 0.25) {
        discard;
    }

    let seed = in.seed;
    let color_variation = vec3<f32>(
        1.0 + sin(1.1 + seed) * 0.1,
        1.0 + sin(2.3 + seed) * 0.1,
        1.0 + sin(3.7 + seed) * 0.05
    );
    let varied_color = base_color.rgb * color_variation;

    let N = normalize(in.world_normal);
    let L = normalize(uniforms.sun_direction);
    let n_dot_l = max(dot(N, L), 0.0);
    let shadow = fetch_shadow(in.render_pos, N, L);
    let horizon_fade = smoothstep(0.0, 0.1, saturate(L.y));
    let ambient = 0.2*horizon_fade;
    let diffuse = n_dot_l * 0.5;
    let light_factor = diffuse * shadow * horizon_fade + ambient;
    let lit_color = varied_color * light_factor;
    out.color = vec4<f32>(lit_color, base_color.a);

    out.normal = vec4<f32>(in.world_normal * 0.5 + 0.5, 1.0);

//    let curr_ndc = in.curr_pos_cs.xy / in.curr_pos_cs.w;
//    let prev_ndc = in.prev_pos_cs.xy / in.prev_pos_cs.w;
//    let velocity = (curr_ndc - prev_ndc) * 0.5;
//    out.motion = velocity;
    out.motion = vec2<f32>(0.0, 0.0);

    return out;
}