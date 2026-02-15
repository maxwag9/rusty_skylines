#include "includes/shadow.wgsl"
#include "includes/uniforms.wgsl"

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) uv: vec2<f32>,
};

struct InstanceInput {
    @location(4) model_col0: vec4<f32>,
    @location(5) model_col1: vec4<f32>,
    @location(6) model_col2: vec4<f32>,
    @location(7) model_col3: vec4<f32>,
    @location(8) prev_model_col0: vec4<f32>,
    @location(9) prev_model_col1: vec4<f32>,
    @location(10) prev_model_col2: vec4<f32>,
    @location(11) prev_model_col3: vec4<f32>,
    @location(12) color: vec4<f32>, // rgb + pad
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) @interpolate(flat) instance_color: vec3<f32>,

    @location(4) @interpolate(flat) instance_id: u32,
    @location(5) prev_pos_cs: vec4<f32>,  // previous clip-space
    @location(6) curr_pos_cs: vec4<f32>,
};


@group(0) @binding(0) var car_sampler: sampler;
@group(0) @binding(1) var tex0: texture_2d<f32>;
@group(0) @binding(2) var s_shadow: sampler_comparison;
@group(0) @binding(3) var t_shadow: texture_depth_2d_array;

@group(1) @binding(0) var<uniform> uniforms: Uniforms;

fn fresnel_schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(1.0 - cos_theta, 5.0);
}

fn d_ggx(NdotH: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * denom * denom);
}

fn g_schlick_ggx(NdotX: f32, k: f32) -> f32 {
    return NdotX / (NdotX * (1.0 - k) + k);
}

fn g_smith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return g_schlick_ggx(NdotV, k) * g_schlick_ggx(NdotL, k);
}

struct FragmentOut {
    @location(0) color: vec4<f32>,     // color target
    @location(1) normal: vec4<f32>,    // normal target
    @location(2) instance_id: u32,     // R32Uint instance id target for RAY TRACING
    @location(3) motion: vec2<f32>,
};

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput, @builtin(instance_index) instance_index: u32) -> VertexOutput {
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

    let world_pos = model * vec4<f32>(vertex.position, 1.0);
    let prev_world_pos = prev_model * vec4<f32>(vertex.position, 1.0);

    out.clip_position = uniforms.view_proj * world_pos;
    out.uv = vertex.uv;
    out.world_normal = normalize((model * vec4<f32>(vertex.normal, 0.0)).xyz);
    out.world_pos = world_pos.xyz;
    out.instance_color = instance.color.rgb;

    // pass instance id (instanced draws only)
    out.instance_id = instance_index;
    out.curr_pos_cs = out.clip_position;
    out.prev_pos_cs = uniforms.prev_view_proj * prev_world_pos;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOut {
    var out: FragmentOut;

    // Sample procedural shiny_metal texture and tint with per-car color
    let tex = textureSample(tex0, car_sampler, input.uv);
    let albedo = tex.rgb * input.instance_color;

    let N = normalize(input.world_normal);
    let V = normalize(-input.world_pos);
    let L = normalize(uniforms.sun_direction);

    let NdotL = saturate(dot(N, L));
    let NdotV = saturate(dot(N, V));

    let elevation = saturate(dot(L, vec3<f32>(0.0, 1.0, 0.0)));
    let sun_color = mix(vec3<f32>(1.0, 0.55, 0.25), vec3<f32>(1.0, 1.0, 1.0), pow(elevation, 0.35));
    let sun_intensity = mix(0.25, 1.0, pow(elevation, 0.25));

    let roughness = 0.15; // Shiny car paint (0.4 in material key, egaaaaal)
    let alpha = roughness * roughness;
    let F0 = vec3<f32>(0.07);

    let H = normalize(V + L);
    let NdotH = saturate(dot(N, H));
    let VdotH = saturate(dot(V, H));

    let D = d_ggx(NdotH, alpha);
    let G = g_smith(NdotV, NdotL, roughness);
    let F = fresnel_schlick(VdotH, F0);

    let spec = (D * G) * F / max(4.0 * NdotV * NdotL, 1e-4);
    let kd = (vec3<f32>(1.0) - F);
    let diff = kd * albedo / 3.14159265;

    let shadow = fetch_shadow(input.world_pos, N, L);

    let direct = (diff + spec) * (NdotL * sun_intensity * shadow) * sun_color;

    let hemi = saturate(N.y * 0.5 + 0.5);
    let sky_ambient = vec3<f32>(0.22, 0.28, 0.35);
    let ground_ambient = vec3<f32>(0.12, 0.10, 0.08);
    let ambient_light = mix(ground_ambient, sky_ambient, hemi);
    let ambient = ambient_light * albedo;

    let rgb = direct + ambient;

    out.color = vec4<f32>(rgb, 1.0);
    out.normal = vec4<f32>(N * 0.5 + 0.5, 1.0);
    out.instance_id = input.instance_id;

    if (input.clip_position.w <= 0.0 || input.prev_pos_cs.w <= 0.0) {
        out.motion = vec2<f32>(0.0);
        return out;
    }
    let curr_ndc = input.curr_pos_cs.xy / input.curr_pos_cs.w;
    let prev_ndc = input.prev_pos_cs.xy / input.prev_pos_cs.w;
    let curr_uv = curr_ndc * vec2(0.5, -0.5) + 0.5;
    let prev_uv = prev_ndc * vec2(0.5, -0.5) + 0.5;
    out.motion = curr_uv - prev_uv;
    return out;
}