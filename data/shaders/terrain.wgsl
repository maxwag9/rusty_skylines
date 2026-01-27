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
};

struct PickUniform {
    pos: vec3<f32>,
    radius: f32,
    underwater: u32,
    color: vec3<f32>,
}

@group(0) @binding(0) var grass_tex: texture_2d<f32>;
@group(0) @binding(1) var grass_tex2: texture_2d<f32>;
@group(0) @binding(2) var material_sampler: sampler;
@group(0) @binding(3) var t_shadow: texture_depth_2d_array;
@group(0) @binding(4) var s_shadow: sampler_comparison;

@group(1) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(1) var<uniform> pick: PickUniform;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) chunk_xz: vec2<i32>,
    @location(4) quad_uv: vec2<f32>,  // NEW: 0-1 within each quad
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) quad_uv: vec2<f32>,  // For edge detection
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;

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
const EDGE_WIDTH: f32 = 0.005;       // Width as fraction of quad (0.02 = 2% of quad width)
const EDGE_COLOR: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);  // Black edges
const EDGE_ENABLED: bool = false;
const SHOW_DIAGONAL: bool = false;   // Show triangle diagonal too
struct FragmentOut {
    @location(0) color : vec4<f32>,
    @location(1) normal : vec4<f32>
};
@fragment
fn fs_main(in: VertexOut) -> FragmentOut {
    var out: FragmentOut;
    let n = normalize(in.world_normal);
    let up = vec3<f32>(0.0, 1.0, 0.0);

    // Your convention: direction from origin -> sun/moon in the sky.
    // For Lambert (dot(n, l)) we want direction from surface -> light.
    // With a directional light at infinity, these are the same unit vector.
    let sun_l  = normalize(uniforms.sun_direction);
    let moon_l = normalize(uniforms.moon_direction);

    // Fade the SUN out when it goes below the horizon to stop wrapped lighting
    // from "glowing" at night.
    let sun_elev   = dot(sun_l, up);                    // [-1..1]
    let sun_amount = smoothstep(-0.06, 0.10, sun_elev); // twilight band
    let night_amount = 1.0 - sun_amount;

    // Moon visibility (also fades if the moon is below the horizon)
    let moon_elev    = dot(moon_l, up);
    let moon_visible = smoothstep(-0.08, 0.08, moon_elev);

    // -------------------------------------------------------------------------
    // Hemisphere ambient (blend day/night)
    // -------------------------------------------------------------------------
    let hemi = saturate(dot(n, up) * 0.5 + 0.5);

    // Day ambient
    let sky_ambient_day    = vec3<f32>(0.10, 0.12, 0.15);
    let ground_ambient_day = vec3<f32>(0.05, 0.05, 0.05);

    // Night ambient (your original-ish values)
    let sky_ambient_night    = vec3<f32>(0.01, 0.01, 0.01);
    let ground_ambient_night = vec3<f32>(0.02, 0.02, 0.02);

    let sky_ambient    = mix(sky_ambient_night,    sky_ambient_day,    sun_amount);
    let ground_ambient = mix(ground_ambient_night, ground_ambient_day, sun_amount);

    // Slight moon-tinted ambient at night (constant, no new uniforms)
    let moon_ambient_color = vec3<f32>(0.02, 0.03, 0.06);
    let moon_ambient = moon_ambient_color * (0.6 * night_amount * moon_visible);

    let ambient = mix(ground_ambient, sky_ambient, hemi) + moon_ambient * hemi;

    // -------------------------------------------------------------------------
    // Sun diffuse + wrapped (grass), then SHADOW
    // -------------------------------------------------------------------------
    let sun_ndotl = dot(n, sun_l);
    let sun_diffuse = max(sun_ndotl, 0.0);

    // Wrapped lighting (allows some light around terminator for grass)
    // The important part: multiply by sun_amount so it doesn't light at night.
    let sun_wrapped = saturate(sun_ndotl + 0.4) / 1.4;

    let sun_shadow = fetch_shadow(in.world_pos, n, sun_l);

    // -------------------------------------------------------------------------
    // Grass mask
    // -------------------------------------------------------------------------
    let greenness = in.color.g - max(in.color.r, in.color.b);
    let up_facing = saturate(dot(n, up));
    let grass_amount = saturate(greenness * 2.5) * up_facing * up_facing;

    // Sun direct term (wrap only for grass) + fade by sun_amount to prevent night glow
    let sun_direct = mix(sun_diffuse, sun_wrapped, grass_amount) * sun_amount;

    // -------------------------------------------------------------------------
    // Moon lighting (constant color/intensity, no extra uniforms)
    // -------------------------------------------------------------------------
    let moon_ndotl = dot(n, moon_l);
    let moon_diffuse = max(moon_ndotl, 0.0);

    // Smaller wrap for moon (optional, helps grass a bit but not too much)
    let moon_wrapped = saturate(moon_ndotl + 0.15) / 1.15;

    let moon_intensity: f32 = 0.22;
    let moon_color = vec3<f32>(0.35, 0.42, 0.60);

    let moon_direct =
        mix(moon_diffuse, moon_wrapped, grass_amount) *
        (moon_visible * night_amount * moon_intensity);

    // -------------------------------------------------------------------------
    // ============ GRASS TEXTURING ============
    // -------------------------------------------------------------------------
    let grass_uv_scale1: f32 = 0.025;
    let grass_uv_scale2: f32 = 0.011;

    let grass_uv1 = in.world_pos.xz * grass_uv_scale1;

    let rot_angle: f32 = 0.615;
    let ca = cos(rot_angle);
    let sa = sin(rot_angle);
    let rot = mat2x2<f32>(ca, -sa, sa, ca);
    let grass_uv2 = rot * (in.world_pos.xz * grass_uv_scale2) + vec2<f32>(17.3, 9.1);

    let grass_a = textureSample(grass_tex, material_sampler, grass_uv1).rgb;
    let grass_b = textureSample(grass_tex2, material_sampler, grass_uv2).rgb;

    let mix_scale: f32 = 0.4;
    let mix_offset = vec2<f32>(42.0, 87.0);
    let mix_p = in.world_pos.xz * mix_scale + mix_offset;
    let mix_noise = fbm(mix_p);

    let grass_color = mix(grass_a, grass_b, mix_noise);
    let albedo = mix(in.color, grass_color, grass_amount);

    // -------------------------------------------------------------------------
    // Final lighting
    // -------------------------------------------------------------------------
    // Constant sun color (no extra uniforms)
    let sun_color = vec3<f32>(1.0, 0.98, 0.92);

    var final_color =
        albedo * ambient +
        albedo * (sun_color * (sun_direct * sun_shadow)) +
        albedo * (moon_color * moon_direct);

    // Pick highlight
    if (pick.radius > 0.0) {
        let d = distance(in.world_pos, pick.pos);
        if (d < pick.radius) {
            let t = 1.0 - smoothstep(0.0, pick.radius, d);
            final_color += pick.color * t;
        }
    }

    // ============ MESH EDGE VISUALIZATION ============
    if (EDGE_ENABLED) {
        let uv = in.quad_uv;

        let d_left   = uv.x;
        let d_right  = 1.0 - uv.x;
        let d_bottom = uv.y;
        let d_top    = 1.0 - uv.y;

        var edge_dist = min(min(d_left, d_right), min(d_bottom, d_top));

        if (SHOW_DIAGONAL) {
            let diag_dist = abs(uv.x + uv.y - 1.0) * 0.7071067811865476; // 1/sqrt(2)
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

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

fn select_cascade(view_depth: f32) -> u32 {
    if (view_depth < uniforms.cascade_splits.x) { return 0u; }
    if (view_depth < uniforms.cascade_splits.y) { return 1u; }
    if (view_depth < uniforms.cascade_splits.z) { return 2u; }
    return 3u;
}

fn project_to_shadow(cascade: u32, world_pos: vec3<f32>) -> vec3<f32> {
    let p = uniforms.lighting_view_proj[cascade] * vec4<f32>(world_pos, 1.0);
    let ndc = p.xyz / p.w;
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return vec3<f32>(uv, ndc.z);
}

fn shadow_pcf(cascade: u32, uv: vec2<f32>, depth: f32) -> f32 {
    let dims = vec2<f32>(textureDimensions(t_shadow, 0));
    let texel = 1.0 / dims;

    var sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel * PCF_RADIUS;
            sum += textureSampleCompare(
                t_shadow,
                s_shadow,
                uv + offset,
                i32(cascade),
                depth
            );
        }
    }
    return sum / 9.0;
}

fn shadow_for_cascade(
    cascade: u32,
    world_pos: vec3<f32>,
    N: vec3<f32>,
    L: vec3<f32>
) -> f32 {
    let proj = project_to_shadow(cascade, world_pos);
    let uv = proj.xy;
    let z  = proj.z;

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || z < 0.0 || z > 1.0) {
        return 1.0;
    }
    // If shadow map is cleared/unused, NO SHADOW1!
    if (z >= 0.9999) {
        return 1.0;
    }

    let ndotl = saturate(dot(N, L));
    let bias = BASE_BIAS + SLOPE_BIAS * (1.0 - ndotl);

    return shadow_pcf(cascade, uv, z - bias);
}

fn cascade_edge_fade(cascade: u32, world_pos: vec3<f32>) -> f32 {
    let proj = project_to_shadow(cascade, world_pos);
    let uv = proj.xy;

    let edge = min(
        min(uv.x, 1.0 - uv.x),
        min(uv.y, 1.0 - uv.y)
    );

    return saturate(edge / SHADOW_FADE_UV);
}

fn fetch_shadow(world_pos: vec3<f32>, N: vec3<f32>, L: vec3<f32>) -> f32 {
    let vpos = uniforms.view * vec4<f32>(world_pos, 1.0);
    let view_depth = -vpos.z;

    let c0 = select_cascade(view_depth);
    let s0 = shadow_for_cascade(c0, world_pos, N, L);

    if (c0 == 3u) {
        return s0;
    }

    let fade = cascade_edge_fade(c0, world_pos);
    if (fade >= 0.999) {
        return s0;
    }

    let c1 = c0 + 1u;
    let s1 = shadow_for_cascade(c1, world_pos, N, L);

    return mix(min(s0, s1), s0, fade);
}

const SHADOW_FADE_UV: f32 = 0.05;
const PCF_RADIUS: f32 = 1.5;
const BASE_BIAS: f32 = 0.000002;
const SLOPE_BIAS: f32 = 0.0005;