struct Uniforms {
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    lighting_view_proj: array<mat4x4<f32>, 4>,
    cascade_splits: vec4<f32>,
    sun_direction: vec3<f32>,
    time: f32,
    camera_local: vec3<f32>,
    chunk_size: f32,
    camera_chunk: vec2<i32>,
    _pad_cam: vec2<i32>,
    moon_direction: vec3<f32>,
    orbit_radius: f32,
    reversed_depth_z: u32,
    shadows_enabled: u32,
    _pad_2: vec2<u32>,     // padding to 16 bytes
};

struct RoadAppearance {
    tint: vec4<f32>,
}

struct VertexInput {
    @location(0) chunk_xz: vec2<i32>,
    @location(1) position : vec3<f32>,
    @location(2) normal   : vec3<f32>,
    @location(3) uv       : vec2<f32>,
    @location(4) material_id : u32,
};

struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>,
    @location(0) uv : vec2<f32>,
    @location(1) @interpolate(flat) material_id : u32,
    @location(2) world_normal : vec3<f32>,
    @location(3) world_pos : vec3<f32>,
};

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
    var out : VertexOutput;

    let lp = input.position;
    // ----- render-space position (WorldPos - EyeWorldPos) -----
    let dc: vec2<i32> = input.chunk_xz - uniforms.camera_chunk;

    let rx = f32(dc.x) * uniforms.chunk_size + (input.position.x - uniforms.camera_local.x);
    let ry = input.position.y - uniforms.camera_local.y;
    let rz = f32(dc.y) * uniforms.chunk_size + (input.position.z - uniforms.camera_local.z);

    let render_pos = vec3<f32>(rx, ry, rz);
    out.clip_position = uniforms.view_proj * vec4<f32>(render_pos, 1.0);

    out.uv = input.uv;
    out.material_id = input.material_id;
    out.world_normal = input.normal;
    out.world_pos = render_pos;

    return out;
}

// ---- Bind group 0: road materials ----
@group(0) @binding(0) var road_sampler : sampler;
@group(0) @binding(1) var tex0 : texture_2d<f32>; // concrete
@group(0) @binding(2) var tex1 : texture_2d<f32>; // goo (filler between new and old asphalt)
@group(0) @binding(3) var tex2 : texture_2d<f32>; // asphalt (new, black)
@group(0) @binding(4) var tex3 : texture_2d<f32>; // asphalt (brighter black, "new" but worn)
@group(0) @binding(5) var tex4 : texture_2d<f32>; // asphalt (kinda orange old "new age" asphalt I see in germany)
@group(0) @binding(6) var tex5 : texture_2d<f32>; // asphalt (old, gray, rough asphalt)
// Keep these comments!
@group(0) @binding(7) var s_shadow: sampler_comparison;
@group(0) @binding(8) var t_shadow: texture_depth_2d_array;

@group(1) @binding(0) var<uniform> uniforms : Uniforms;
@group(1) @binding(1) var<uniform> road_appearance: RoadAppearance;

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

fn fresnel_schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(1.0 - cos_theta, 5.0);
}

// GGX / Trowbridge-Reitz NDF
fn d_ggx(NdotH: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * denom * denom);
}

// Schlick-GGX geometry term (Smith)
fn g_schlick_ggx(NdotX: f32, k: f32) -> f32 {
    return NdotX / (NdotX * (1.0 - k) + k);
}

fn g_smith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return g_schlick_ggx(NdotV, k) * g_schlick_ggx(NdotL, k);
}
struct FragmentOut {
    @location(0) color : vec4<f32>,
    @location(1) normal : vec4<f32>
};
@fragment
fn fs_main(input : VertexOutput) -> FragmentOut {
    var out: FragmentOut;
    // --- sample material ---
    let uv = input.uv;
    var tex : vec4<f32>;
    if (input.material_id == 0u) {
        tex = textureSample(tex0, road_sampler, uv);
    } else if (input.material_id == 1u) {
        tex = textureSample(tex1, road_sampler, uv);
    } else if (input.material_id == 2u) {
        tex = textureSample(tex2, road_sampler, uv);
    } else if (input.material_id == 3u) {
        tex = textureSample(tex3, road_sampler, uv);
    } else if (input.material_id == 4u) {
        tex = textureSample(tex4, road_sampler, uv);
    } else if (input.material_id == 5u) {
        tex = textureSample(tex5, road_sampler, uv);
    } else {
        tex = vec4<f32>(1.0, 0.0, 1.0, 1.0);
    }

    let albedo = tex.rgb;

    // --- basis vectors ---
    let N = normalize(input.world_normal);
    let V = normalize(-input.world_pos);
    let L = normalize(uniforms.sun_direction);

    let NdotL = saturate(dot(N, L));
    let NdotV = saturate(dot(N, V));

    let elevation = saturate(dot(L, vec3<f32>(0.0, 1.0, 0.0)));
    let sun_color = mix(vec3<f32>(1.0, 0.55, 0.25), vec3<f32>(1.0, 1.0, 1.0), pow(elevation, 0.35));
    let sun_intensity = mix(0.25, 1.0, pow(elevation, 0.25));

    let roughness = 0.85;
    let alpha = roughness * roughness;
    let F0 = vec3<f32>(0.04);

    let H = normalize(V + L);
    let NdotH = saturate(dot(N, H));
    let VdotH = saturate(dot(V, H));

    let D = d_ggx(NdotH, alpha);
    let G = g_smith(NdotV, NdotL, roughness);
    let F = fresnel_schlick(VdotH, F0);

    let spec = (D * G) * F / max(4.0 * NdotV * NdotL, 1e-4);
    let kd = (vec3<f32>(1.0) - F);
    let diff = kd * albedo / 3.14159265;

    // --- SHADOW ---
    let shadow = fetch_shadow(input.world_pos, N, L);

    let direct = (diff + spec) * (NdotL * sun_intensity * shadow) * sun_color;

    let hemi = saturate(N.y * 0.5 + 0.5);
    let sky_ambient = vec3<f32>(0.22, 0.28, 0.35);
    let ground_ambient = vec3<f32>(0.12, 0.10, 0.08);
    let ambient_light = mix(ground_ambient, sky_ambient, hemi);

    let ambient = ambient_light * albedo;

    let rgb = (direct + ambient) * road_appearance.tint.xyz;
    let a   = road_appearance.tint.w;
    out.color = vec4<f32>(rgb, a);
    out.normal = vec4<f32>(N * 0.5 + 0.5, 1.0);
    return out;
}

// -----------------------------------------------------------------------------
// Shadow constants
// -----------------------------------------------------------------------------
const SHADOW_FADE_UV: f32 = 0.05;
const PCF_RADIUS: f32 = 1.5;
const BASE_BIAS: f32 = 0.000002;
const SLOPE_BIAS: f32 = 0.0005;
const CASCADE_BLEND_RATIO: f32 = 0.20;

fn shadow_texel_size() -> vec2<f32> {
    let du: vec2<u32> = textureDimensions(t_shadow, 0);
    // Correct WGSL conversion (vec2<u32> -> vec2<f32>)
    return vec2<f32>(1.0 / f32(du.x), 1.0 / f32(du.y));
}

fn select_cascade(view_depth: f32) -> u32 {
    // Branch-light cascade selection (assumes splits are increasing)
    let s = uniforms.cascade_splits;

    var c: u32 = 0u;
    c += select(0u, 1u, view_depth >= s.x);
    c += select(0u, 1u, view_depth >= s.y);
    c += select(0u, 1u, view_depth >= s.z);

    return min(c, 3u);
}

fn project_to_shadow(cascade: u32, world_pos: vec3<f32>) -> vec3<f32> {
    let p = uniforms.lighting_view_proj[cascade] * vec4<f32>(world_pos, 1.0);
    let inv_w = 1.0 / p.w;
    let ndc = p.xyz * inv_w;

    // NDC [-1,1] -> UV [0,1] (Y flipped)
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return vec3<f32>(uv, ndc.z);
}

// Fade based on UV distance to edge, but *reduced by the PCF footprint* so that
// when your kernel would sample outside, you’re already fading out.
fn cascade_edge_fade_uv(uv: vec2<f32>, pcf_step: vec2<f32>) -> f32 {
    // Edge distance per axis, minus max axis offset we will sample (pcf_step).
    let edge_x = min(uv.x, 1.0 - uv.x) - pcf_step.x;
    let edge_y = min(uv.y, 1.0 - uv.y) - pcf_step.y;
    let edge = min(edge_x, edge_y);
    return saturate(edge / SHADOW_FADE_UV);
}

fn shadow_pcf_3x3(cascade: u32, uv: vec2<f32>, depth_ref: f32, pcf_step: vec2<f32>) -> f32 {
    // Unrolled 3x3 taps: avoids loop overhead + int->float conversions.
    let layer = i32(cascade);
    var sum: f32 = 0.0;

    sum += textureSampleCompare(t_shadow, s_shadow, uv + vec2<f32>(-pcf_step.x, -pcf_step.y), layer, depth_ref);
    sum += textureSampleCompare(t_shadow, s_shadow, uv + vec2<f32>( 0.0,        -pcf_step.y), layer, depth_ref);
    sum += textureSampleCompare(t_shadow, s_shadow, uv + vec2<f32>( pcf_step.x, -pcf_step.y), layer, depth_ref);

    sum += textureSampleCompare(t_shadow, s_shadow, uv + vec2<f32>(-pcf_step.x,  0.0),        layer, depth_ref);
    sum += textureSampleCompare(t_shadow, s_shadow, uv,                                     layer, depth_ref);
    sum += textureSampleCompare(t_shadow, s_shadow, uv + vec2<f32>( pcf_step.x,  0.0),        layer, depth_ref);

    sum += textureSampleCompare(t_shadow, s_shadow, uv + vec2<f32>(-pcf_step.x,  pcf_step.y), layer, depth_ref);
    sum += textureSampleCompare(t_shadow, s_shadow, uv + vec2<f32>( 0.0,         pcf_step.y), layer, depth_ref);
    sum += textureSampleCompare(t_shadow, s_shadow, uv + vec2<f32>( pcf_step.x,  pcf_step.y), layer, depth_ref);

    return sum * (1.0 / 9.0);
}

fn shadow_for_cascade(
    cascade: u32,
    world_pos: vec3<f32>,
    N: vec3<f32>,
    L: vec3<f32>,
    pcf_step: vec2<f32>,
) -> f32 {
    let proj = project_to_shadow(cascade, world_pos);
    let uv = proj.xy;
    let z  = proj.z;

    // Quick reject outside shadow clip
    let uv_ok = all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0));
    if (!uv_ok || z < 0.0 || z > 1.0) {
        return 1.0;
    }

    // Fade out near edges (computed BEFORE PCF so we can early-out)
    let edge_fade = cascade_edge_fade_uv(uv, pcf_step);
    if (edge_fade <= 0.0) {
        return 1.0;
    }

    // Bias
    let ndotl = saturate(dot(N, L));
    let bias = BASE_BIAS + SLOPE_BIAS * (1.0 - ndotl);

    // NOTE: requires sampler compare op to match reversed_depth_z convention.
    let depth_ref = clamp(
        select(z - bias, z + bias, uniforms.reversed_depth_z != 0u),
        0.0, 1.0
    );

    let vis = shadow_pcf_3x3(cascade, uv, depth_ref, pcf_step);

    // edge_fade: 1 inside, 0 at boundary => fade to fully lit
    return mix(1.0, vis, edge_fade);
}

fn fetch_shadow(world_pos: vec3<f32>, N: vec3<f32>, L: vec3<f32>) -> f32 {
    if (uniforms.shadows_enabled == 0u) {
        return 1.0;
    }

    // Compute once; reused by both cascades when blending
    let texel = shadow_texel_size();
    let pcf_step = texel * PCF_RADIUS;

    let vpos = uniforms.view * vec4<f32>(world_pos, 1.0);
    let view_depth = max(0.0, -vpos.z);

    let cascade = select_cascade(view_depth);
    let vis0 = shadow_for_cascade(cascade, world_pos, N, L, pcf_step);

    if (cascade >= 3u) {
        return vis0;
    }

    // Far split for this cascade (x/y/z for 0/1/2)
    let current_far = uniforms.cascade_splits[i32(cascade)];
    let blend_start = current_far * (1.0 - CASCADE_BLEND_RATIO);

    if (view_depth <= blend_start) {
        return vis0;
    }

    let vis1 = shadow_for_cascade(cascade + 1u, world_pos, N, L, pcf_step);

    let denom = max(current_far - blend_start, 1e-6);
    let t = saturate((view_depth - blend_start) / denom);

    return mix(vis0, vis1, t);
}