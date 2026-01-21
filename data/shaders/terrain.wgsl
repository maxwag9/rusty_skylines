struct Uniforms {
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    lighting_view_proj: mat4x4<f32>,
    sun_direction: vec3<f32>,
    time: f32,

    camera_pos: vec3<f32>,
    orbit_radius: f32,

    moon_direction: vec3<f32>,
    _pad0: f32,
};

struct PickUniform {
    pos: vec3<f32>,
    radius: f32,
    underwater: u32,
    color: vec3<f32>,
}
@group(0) @binding(0) var grass_tex: texture_2d<f32>;
@group(0) @binding(1) var material_sampler: sampler;
@group(0) @binding(2) var t_shadow: texture_depth_2d; // <--- The depth map
@group(0) @binding(3) var s_shadow: sampler_comparison;
@group(1) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(1)
var<uniform> pick: PickUniform;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;

    let world_pos = in.position;

    out.position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_normal = normalize(in.normal);
    out.color = in.color;
    out.world_pos = world_pos;

    return out;
}

// Cheap hash for procedural patterns
fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let l = normalize(uniforms.sun_direction);
    let up = vec3<f32>(0.0, 1.0, 0.0);

    // Hemisphere ambient
    let hemi = clamp(dot(n, up) * 0.5 + 0.5, 0.0, 1.0);
    let sky_ambient = vec3<f32>(0.01, 0.01, 0.01);
    let ground_ambient = vec3<f32>(0.02, 0.02, 0.02);
    let ambient = mix(ground_ambient, sky_ambient, hemi);

    // Diffuse lighting
    let ndotl = dot(n, l);
    let diffuse = max(ndotl, 0.0);

    // --- SHADOW ---
    let shadow = fetch_shadow(in.world_pos, n, l);

    // Combine lighting with shadow
    var base_color = in.color * (ambient + diffuse * shadow);

    // ============ GRASS EFFECT ============
    let greenness = in.color.g - max(in.color.r, in.color.b);
    let up_facing = saturate(dot(n, up));
    let grass_amount = saturate(greenness * 2.5) * up_facing * up_facing;

    let p_fine = floor(in.world_pos.xz * 600.0);
    let p_coarse = floor(in.world_pos.xz * 200.0);
    let h_fine = hash2(p_fine);
    let h_coarse = hash2(p_coarse * 0.4 + 31.7);
    let grass_pattern = h_fine * 0.55 + h_coarse * 1.85;
    let shade = mix(0.58, 1.28, grass_pattern);

    base_color.g *= 1.0 + grass_amount * 0.25;

    let grass_uv_scale: f32 = 0.025;
    let grass_uv = in.world_pos.xz * grass_uv_scale;
    let grass_sample = textureSample(grass_tex, material_sampler, grass_uv).rgb;
    let grass_tint = grass_sample * (0.85 + 0.5 * grass_pattern);

    base_color = mix(base_color, grass_tint, grass_amount);

    var final_color = base_color;

    // Highlight pick
    if (pick.radius > 0.0) {
        let d = distance(in.world_pos, pick.pos);
        if (d < pick.radius) {
            let t = 1.0 - smoothstep(0.0, pick.radius, d);
            final_color += pick.color * t;
        }
    }

    return vec4<f32>(final_color, 1.0);
}

fn fetch_shadow(world_pos: vec3<f32>, N: vec3<f32>, L: vec3<f32>) -> f32 {
    let pos_from_light = uniforms.lighting_view_proj * vec4<f32>(world_pos, 1.0);

    // Perspective divide (w=1 for ortho, but good practice)
    let ndc = pos_from_light.xyz / pos_from_light.w;

    // Clip space to UV space
    let shadow_coords = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);

    // Clamp to valid range (points outside shadow map should be lit)
    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 ||
        shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
        ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0; // Lit (outside shadow frustum)
    }

    // Slope-scaled bias: more bias when surface is nearly parallel to light
    let bias = max(0.005 * (1.0 - dot(N, L)), 0.001);

    return textureSampleCompare(t_shadow, s_shadow, shadow_coords, ndc.z - bias);
}