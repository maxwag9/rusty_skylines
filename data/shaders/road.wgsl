struct Uniforms {
    view : mat4x4<f32>,
    inv_view : mat4x4<f32>,
    proj : mat4x4<f32>,
    inv_proj : mat4x4<f32>,
    view_proj : mat4x4<f32>,
    inv_view_proj : mat4x4<f32>,

    sun_direction : vec3<f32>,
    time : f32,

    camera_pos : vec3<f32>,
    orbit_radius : f32,

    moon_direction : vec3<f32>,
    _pad0 : f32
};

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
@group(1) @binding(0) var<uniform> uniforms : Uniforms;

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

    // If your uniforms.sun_direction is already the direction *towards the sun*, use it directly.
    // If it is "sun rays direction" (from sun to world), then use -uniforms.sun_direction instead.
    let L = normalize(uniforms.sun_direction);

    let NdotL = saturate(dot(N, L));
    let NdotV = saturate(dot(N, V));

    // --- "sun angle realism": change sun color/intensity with elevation ---
    // elevation=1 at noon, 0 at horizon
    let elevation = saturate(dot(L, vec3<f32>(0.0, 1.0, 0.0)));

    // warmer at low sun:
    let sun_color = mix(vec3<f32>(1.0, 0.55, 0.25), vec3<f32>(1.0, 1.0, 1.0), pow(elevation, 0.35));

    // dim near horizon (simple atmosphere-ish):
    let sun_intensity = mix(0.25, 1.0, pow(elevation, 0.25));

    // --- material params (tweak per material_id if you want) ---
    let roughness = 0.85;                // roads are quite rough
    let alpha = roughness * roughness;
    let F0 = vec3<f32>(0.04);            // dielectric baseline

    // --- specular (GGX) ---
    let H = normalize(V + L);
    let NdotH = saturate(dot(N, H));
    let VdotH = saturate(dot(V, H));

    let D = d_ggx(NdotH, alpha);
    let G = g_smith(NdotV, NdotL, roughness);
    let F = fresnel_schlick(VdotH, F0);

    let spec = (D * G) * F / max(4.0 * NdotV * NdotL, 1e-4);

    // --- diffuse (energy conserving-ish) ---
    let kd = (vec3<f32>(1.0) - F); // (ignore metalness for now)
    let diff = kd * albedo / 3.14159265;

    let direct = (diff + spec) * (NdotL * sun_intensity) * sun_color;

    // --- hemisphere ambient: sky vs ground depends on normal.y ---
    let hemi = saturate(N.y * 0.5 + 0.5);
    let sky_ambient = vec3<f32>(0.22, 0.28, 0.35);
    let ground_ambient = vec3<f32>(0.12, 0.10, 0.08);
    let ambient_light = mix(ground_ambient, sky_ambient, hemi);

    let ambient = ambient_light * albedo;

    let rgb = direct + ambient;

    return vec4<f32>(rgb, tex.a);
}