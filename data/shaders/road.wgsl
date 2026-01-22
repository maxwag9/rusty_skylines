struct Uniforms {
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    lighting_view_proj: array<mat4x4<f32>, 4>,
    cascade_splits: vec4<f32>,     // end distance of each cascade in view-space units

    sun_direction: vec3<f32>,
    time: f32,

    camera_pos: vec3<f32>,
    orbit_radius: f32,

    moon_direction: vec3<f32>,
    shadow_cascade_index: u32,     // used only during shadow rendering
};

struct RoadAppearance {
    tint: vec4<f32>,
}

struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) normal   : vec3<f32>,
    @location(2) uv       : vec2<f32>,
    @location(3) material_id : u32
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

    let wp = input.position;
    out.clip_position = uniforms.view_proj * vec4<f32>(wp, 1.0);

    out.uv = input.uv;
    out.material_id = input.material_id;
    out.world_normal = input.normal;
    out.world_pos = wp;

    return out;
}

// ---- Bind group 0: road materials ----
@group(0) @binding(0) var tex0 : texture_2d<f32>; // concrete
@group(0) @binding(1) var tex1 : texture_2d<f32>; // goo (filler between new and old asphalt)
@group(0) @binding(2) var tex2 : texture_2d<f32>; // asphalt (new, black)
@group(0) @binding(3) var tex3 : texture_2d<f32>; // asphalt (brighter black, "new" but worn)
@group(0) @binding(4) var tex4 : texture_2d<f32>; // asphalt (kinda orange old "new age" asphalt I see in germany)
@group(0) @binding(5) var tex5 : texture_2d<f32>; // asphalt (old, gray, rough asphalt)
// Keep these comments!
@group(0) @binding(6) var road_sampler : sampler;
@group(0) @binding(7) var t_shadow: texture_depth_2d_array;
@group(0) @binding(8) var s_shadow: sampler_comparison;
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

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
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
    let V = normalize(uniforms.camera_pos - input.world_pos);
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
    return vec4<f32>(rgb, a);
}

fn select_cascade(view_depth: f32) -> u32 {
    if (view_depth < uniforms.cascade_splits.x) { return 0u; }
    if (view_depth < uniforms.cascade_splits.y) { return 1u; }
    if (view_depth < uniforms.cascade_splits.z) { return 2u; }
    return 3u;
}

struct CascadeBlend {
    i0: u32,
    i1: u32,
    t: f32, // 0 -> use i0, 1 -> use i1
};

fn select_cascade_with_blend(view_depth: f32) -> CascadeBlend {
    let i0 = select_cascade(view_depth);

    // Fade size: choose something stable-ish in view units.
    // Start with 5% of the current cascade length, clamped.
    var start: f32 = 0.0;
    var end: f32 = uniforms.cascade_splits.x;

    if (i0 == 0u) {
        start = 0.0;
        end = uniforms.cascade_splits.x;
    } else if (i0 == 1u) {
        start = uniforms.cascade_splits.x;
        end = uniforms.cascade_splits.y;
    } else if (i0 == 2u) {
        start = uniforms.cascade_splits.y;
        end = uniforms.cascade_splits.z;
    } else {
        start = uniforms.cascade_splits.z;
        end = uniforms.cascade_splits.w;
    }

    let len = max(end - start, 1.0);
    let fade = clamp(len * 0.05, 5.0, 50.0); // tweak later

    // Blend only near the END of cascade i0 into i1
    let i1 = min(i0 + 1u, 3u);
    var t = 0.0;
    if (i0 < 3u) {
        t = saturate((view_depth - (end - fade)) / fade);
    } else {
        t = 0.0;
    };

    return CascadeBlend(i0, i1, t);
}

fn project_to_shadow(cascade: u32, world_pos: vec3<f32>) -> vec3<f32> {
    let p = uniforms.lighting_view_proj[cascade] * vec4<f32>(world_pos, 1.0);
    let ndc = p.xyz / p.w;

    // NDC -> UV (flip Y to match texture space)
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return vec3<f32>(uv, ndc.z);
}

// 3x3 PCF
fn shadow_pcf_3x3(cascade: u32, uv: vec2<f32>, depth: f32) -> f32 {
    let dims = vec2<f32>(textureDimensions(t_shadow, 0));
    let texel = 1.0 / dims;

    var sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            sum += textureSampleCompare(t_shadow, s_shadow, uv + offset, i32(cascade), depth);
        }
    }
    return sum / 9.0;
}

fn shadow_for_cascade(cascade: u32, world_pos: vec3<f32>, N: vec3<f32>, L: vec3<f32>) -> f32 {
    let proj = project_to_shadow(cascade, world_pos);
    let uv = proj.xy;
    let z  = proj.z;

    // Outside shadow frustum => lit
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || z < 0.0 || z > 1.0) {
        return 1.0;
    }

    // Bias (still basic; good enough to start)
    let ndotl = saturate(dot(N, L));
    let bias = max(0.00002, 0.00025 * (1.0 - ndotl));

    // PCF
    return shadow_pcf_3x3(cascade, uv, z - bias);
}

fn fetch_shadow(world_pos: vec3<f32>, N: vec3<f32>, L: vec3<f32>) -> f32 {
    // view-space depth (RH camera looking down -Z)
    let vpos = uniforms.view * vec4<f32>(world_pos, 1.0);
    let view_depth = -vpos.z;

    let cb = select_cascade_with_blend(view_depth);

    let s0 = shadow_for_cascade(cb.i0, world_pos, N, L);
    if (cb.i0 == cb.i1) {
        return s0;
    }
    let s1 = shadow_for_cascade(cb.i1, world_pos, N, L);

    return mix(s0, s1, cb.t);
}