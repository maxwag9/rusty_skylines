#include "includes/shadow.wgsl"
#include "includes/uniforms.wgsl"

struct PickUniform {
    pos: vec3<f32>,
    radius: f32,
    underwater: u32,
    color: vec3<f32>,
}
@group(0) @binding(0) var material_sampler: sampler;
@group(0) @binding(1) var grass_tex: texture_2d<f32>;
@group(0) @binding(2) var grass_tex2: texture_2d<f32>;

@group(0) @binding(3) var s_shadow: sampler_comparison;
@group(0) @binding(4) var t_shadow: texture_depth_2d_array;

@group(1) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(1) var<uniform> pick: PickUniform;

struct VertexIn {
    @location(0) chunk_xz: vec2<i32>,
    @location(1) position: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) color: vec4<f32>,
    @location(4) quad_uv: vec2<f32>,  // NEW: 0-1 within each quad
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) quad_uv: vec2<f32>,  // For edge detection
    @location(4) chunk_xz: vec2<f32>, // To fix swimming UVs
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;

    // Camera-relative render position (keeps precision stable)
    let dc: vec2<i32> = in.chunk_xz - uniforms.camera_chunk;

    let rx = f32(dc.x) * uniforms.chunk_size + (in.position.x - uniforms.camera_local.x);
    let ry = in.position.y - uniforms.camera_local.y;
    let rz = f32(dc.y) * uniforms.chunk_size + (in.position.z - uniforms.camera_local.z);

    let render_pos = vec3<f32>(rx, ry, rz);

    out.position = uniforms.view_proj * vec4<f32>(render_pos, 1.0);
    out.world_pos = render_pos;
    out.quad_uv = in.quad_uv;

    out.world_normal = normalize(in.normal);
    out.color = in.color.rgb;

    out.chunk_xz = vec2<f32>(
        f32(in.chunk_xz.x) * uniforms.chunk_size + in.position.x,
        f32(in.chunk_xz.y) * uniforms.chunk_size + in.position.z
    );

    return out;
}

fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn quintic(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let ux = quintic(f.x);
    let uy = quintic(f.y);

    let a = hash2(i);
    let b = hash2(i + vec2(1.0, 0.0));
    let c = hash2(i + vec2(0.0, 1.0));
    let d = hash2(i + vec2(1.0, 1.0));

    return mix(mix(a, b, ux), mix(c, d, ux), uy);
}

fn fbm(p_in: vec2<f32>) -> f32 {
    let ROT = mat2x2<f32>(vec2(0.8, 0.6), vec2(-0.6, 0.8));
    var acc: f32 = 0.0;
    var amp: f32 = 0.5;
    var p = p_in;

    for (var i: i32 = 0; i < 4; i = i + 1) {
        acc += amp * value_noise(p);
        p = ROT * p * 2.0;
        amp *= 0.5;
    }
    return acc;
}

// Edge rendering constants
const EDGE_WIDTH: f32 = 0.0025;       // Width as fraction of quad (0.02 = 2% of quad width)
const EDGE_COLOR: vec3<f32> = vec3<f32>(0.1, 0.1, 0.1);  // Black edges
const EDGE_ENABLED: bool = false;
const SHOW_DIAGONAL: bool = true;   // Show triangle diagonal too

struct FragmentOut {
    @location(0) color : vec4<f32>,
    @location(1) normal : vec4<f32>
};

@fragment
fn fs_main(in: VertexOut) -> FragmentOut {
    var out: FragmentOut;
    let n = normalize(in.world_normal);
    let up = vec3<f32>(0.0, 1.0, 0.0);

    let sun_l  = normalize(uniforms.sun_direction);
    let moon_l = normalize(uniforms.moon_direction);

    let sun_elev   = dot(sun_l, up);
    let sun_amount = smoothstep(-0.06, 0.10, sun_elev);
    let night_amount = 1.0 - sun_amount;

    let moon_elev    = dot(moon_l, up);
    let moon_visible = smoothstep(-0.08, 0.08, moon_elev);

    let hemi = saturate(dot(n, up) * 0.5 + 0.5);

    let sky_ambient_day    = vec3<f32>(0.10, 0.12, 0.15);
    let ground_ambient_day = vec3<f32>(0.05, 0.05, 0.05);

    let sky_ambient_night    = vec3<f32>(0.01, 0.01, 0.01);
    let ground_ambient_night = vec3<f32>(0.02, 0.02, 0.02);

    let sky_ambient    = mix(sky_ambient_night,    sky_ambient_day,    sun_amount);
    let ground_ambient = mix(ground_ambient_night, ground_ambient_day, sun_amount);

    let moon_ambient_color = vec3<f32>(0.02, 0.03, 0.06);
    let moon_ambient = moon_ambient_color * (0.6 * night_amount * moon_visible);

    let ambient = mix(ground_ambient, sky_ambient, hemi) + moon_ambient * hemi;

    let sun_ndotl = dot(n, sun_l);
    let sun_diffuse = max(sun_ndotl, 0.0);
    let sun_wrapped = saturate(sun_ndotl + 0.4) / 1.4;

    let greenness = in.color.g - max(in.color.r, in.color.b);
    let up_facing = saturate(dot(n, up));
    let grass_amount = saturate(greenness * 2.5) * up_facing * up_facing;

    let sun_direct = mix(sun_diffuse, sun_wrapped, grass_amount) * sun_amount;

    var sun_shadow: f32 = 1.0;
    if (uniforms.csm_enabled != 0u && sun_amount > 0.0 && sun_direct > 0.0001) {
        sun_shadow = fetch_shadow(in.world_pos, n, sun_l);
    }

    let moon_ndotl = dot(n, moon_l);
    let moon_diffuse = max(moon_ndotl, 0.0);
    let moon_wrapped = saturate(moon_ndotl + 0.15) / 1.15;

    let moon_intensity: f32 = 0.22;
    let moon_color = vec3<f32>(0.35, 0.42, 0.60);

    let moon_direct =
        mix(moon_diffuse, moon_wrapped, grass_amount) *
        (moon_visible * night_amount * moon_intensity);

    // -------------------------------------------------------------------------
    // ============ GRASS TEXTURING (FIXED: NO SWIMMING UVs) ============
    // -------------------------------------------------------------------------
    // in.chunk_xz now contains stable world-space XZ (computed in the vertex shader)
    let stable_world_xz = in.chunk_xz;

    let grass_uv_scale1: f32 = 0.025;
    let grass_uv_scale2: f32 = 0.011;

    let grass_uv1 = fract(stable_world_xz * grass_uv_scale1);

    let rot_angle: f32 = 0.615;
    let ca = cos(rot_angle);
    let sa = sin(rot_angle);
    let rot = mat2x2<f32>(ca, -sa, sa, ca);

    let uv2_base = stable_world_xz * grass_uv_scale2;
    let grass_uv2 = fract(rot * uv2_base + vec2<f32>(17.3, 9.1));

    let grass_a = textureSample(grass_tex, material_sampler, grass_uv1).rgb;
    let grass_b = textureSample(grass_tex2, material_sampler, grass_uv2).rgb;

    let mix_scale: f32 = 0.4;
    let mix_offset = vec2<f32>(42.0, 87.0);
    let mix_p = stable_world_xz * mix_scale + mix_offset;
    let mix_noise = fbm(mix_p);

    let grass_color = mix(grass_a, grass_b, mix_noise);
    let albedo = mix(in.color, grass_color, grass_amount);

    // -------------------------------------------------------------------------
    // Final lighting
    // -------------------------------------------------------------------------
    let sun_color = vec3<f32>(1.0, 0.98, 0.92);

    var final_color =
        albedo * ambient +
        albedo * (sun_color * (sun_direct * sun_shadow)) +
        albedo * (moon_color * moon_direct);

    if (pick.radius > 0.0) {
        let d = distance(in.world_pos, pick.pos);
        if (d < pick.radius) {
            let t = 1.0 - smoothstep(0.0, pick.radius, d);
            final_color += pick.color * t;
        }
    }

    if (EDGE_ENABLED) {
        let uv = in.quad_uv;

        let d_left   = uv.x;
        let d_right  = 1.0 - uv.x;
        let d_bottom = uv.y;
        let d_top    = 1.0 - uv.y;

        var edge_dist = min(min(d_left, d_right), min(d_bottom, d_top));

        if (SHOW_DIAGONAL) {
            let diag_dist = abs(uv.x + uv.y - 1.0) * 0.7071067811865476;
            edge_dist = min(edge_dist, diag_dist);
        }

        let edge_w = fwidth(edge_dist);
        let edge_threshold = max(edge_w * 1.5, EDGE_WIDTH);
        let edge_factor = smoothstep(0.0, edge_threshold, edge_dist);

        final_color = mix(EDGE_COLOR, final_color, edge_factor);
    }

    out.color = vec4<f32>(final_color, 1.0);
    out.normal = vec4<f32>(n * 0.5 + 0.5, 1.0);
    return out;
}
