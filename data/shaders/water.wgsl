#include "includes/uniforms.wgsl"

struct WaterUniform {
    sea_level: f32,
    color: vec4<f32>,
    wave_tiling: f32,
    wave_strength: f32,
};

struct SkyUniform {
    star_rotation: mat4x4<f32>,
    exposure: f32,
    moon_phase: f32,
    sun_size: f32,
    sun_intensity: f32,
    moon_size: f32,
    moon_intensity: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(1) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(1) var<uniform> water: WaterUniform;
@group(1) @binding(2) var<uniform> sky: SkyUniform;

fn hash2(p: vec2<i32>) -> f32 {
    let pf = vec2<f32>(f32(p.x), f32(p.y));
    return fract(sin(dot(pf, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    // quintic smoothstep
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

    let i00 = vec2<i32>(i);
    let i10 = i00 + vec2<i32>(1, 0);
    let i01 = i00 + vec2<i32>(0, 1);
    let i11 = i00 + vec2<i32>(1, 1);

    let v00 = hash2(i00);
    let v10 = hash2(i10);
    let v01 = hash2(i01);
    let v11 = hash2(i11);

    let vx0 = mix(v00, v10, u.x);
    let vx1 = mix(v01, v11, u.x);
    return mix(vx0, vx1, u.y);
}

fn fbm(p: vec2<f32>, t: f32, speed: f32) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var pp = p;

    v += value_noise(pp + vec2<f32>( 0.07 * t * speed,  0.11 * t * speed)) * a;
    a *= 0.5;
    pp = mat2x2<f32>(0.8, -0.6, 0.6, 0.8) * pp * 1.9;

    v += value_noise(pp + vec2<f32>(-0.13 * t * speed,  0.05 * t * speed)) * a;
    a *= 0.5;
    pp = mat2x2<f32>(0.3, -0.95, 0.95, 0.3) * pp * 2.1;

    v += value_noise(pp + vec2<f32>( 0.19 * t * speed, -0.09 * t * speed)) * a;
    a *= 0.5;
    pp = mat2x2<f32>(0.99, -0.14, 0.14, 0.99) * pp * 2.5;

    v += value_noise(pp + vec2<f32>(-0.03 * t * speed, -0.17 * t * speed)) * a;

    return v * 2.0 - 1.0;
}

fn layered_wave_normal(
    p: vec2<f32>,
    tiling: f32,
    strength: f32,
    t: f32,
    eps_scale: f32,
) -> vec3<f32> {
    let wp = p * tiling;

    let eps_large = 1.2 * eps_scale;
    let eps_small = 0.25 * eps_scale;

    // large layer
    let hL  = fbm(wp * 0.35, t * 0.35, strength * 0.7);
    let hLx = fbm((wp + vec2<f32>(eps_large, 0.0)) * 0.35, t * 0.35, strength * 0.7);
    let hLz = fbm((wp + vec2<f32>(0.0, eps_large)) * 0.35, t * 0.35, strength * 0.7);

    // small layer
    let hS  = fbm(wp * 3.0, t * 1.8, strength * 0.4);
    let hSx = fbm((wp + vec2<f32>(eps_small, 0.0)) * 3.0, t * 1.8, strength * 0.4);
    let hSz = fbm((wp + vec2<f32>(0.0, eps_small)) * 3.0, t * 1.8, strength * 0.4);

    let dhdx = (hLx - hL) * 0.8 + (hSx - hS) * 1.6;
    let dhdz = (hLz - hL) * 0.8 + (hSz - hS) * 1.6;

    let base_n = normalize(vec3<f32>(-dhdx * strength, 1.0, -dhdz * strength));

    let chop = 1.5;
    let ny = max(base_n.y, 0.2);
    return normalize(vec3<f32>(base_n.x * chop, ny, base_n.z * chop));
}

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) curr_clip: vec4<f32>,
    @location(2) prev_clip: vec4<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> VSOut {
    var out: VSOut;

    let world_xz = vec2<f32>(pos.x, pos.z) + uniforms.camera_local.xz;
    let world_y = water.sea_level;
    out.world = vec3<f32>(world_xz.x, world_y, world_xz.y);

    let view_pos = vec3<f32>(
        pos.x,
        water.sea_level - uniforms.camera_local.y,
        pos.z
    );

    let clip = uniforms.view_proj * vec4<f32>(view_pos, 1.0);
    out.pos = clip;
    out.curr_clip = clip;

    // Previous frame: same world point, but relative to previous camera
    var prev_view_pos = vec3<f32>(
        out.world.x - uniforms.prev_camera_local.x,
        water.sea_level - uniforms.prev_camera_local.y,
        out.world.z - uniforms.prev_camera_local.z
    );
    let chunk_delta = vec2<f32>(
        f32(uniforms.camera_chunk.x - uniforms.prev_camera_chunk.x) * uniforms.chunk_size,
        f32(uniforms.camera_chunk.y - uniforms.prev_camera_chunk.y) * uniforms.chunk_size
    );
    prev_view_pos = vec3<f32>(
        prev_view_pos.x + chunk_delta.x,
        prev_view_pos.y,
        prev_view_pos.z + chunk_delta.y
    );
    out.prev_clip = uniforms.prev_view_proj * vec4<f32>(prev_view_pos, 1.0);

    return out;
}

struct FragOut {
    @location(0) color: vec4<f32>,
    @location(2) motion: vec2<f32>,
};


@fragment
fn fs_main(in: VSOut) -> FragOut {
    var out: FragOut;

    let sun_dir = normalize(uniforms.sun_direction);
    let moon_dir = normalize(uniforms.moon_direction);

    let sun_elev = sun_dir.y;
    let moon_elev = moon_dir.y;

    let wave_xz = in.world.xz + vec2<f32>(uniforms.camera_chunk) * uniforms.chunk_size;

    let sun_vis = smoothstep(-0.05, 0.02, sun_elev);
    let moon_vis = smoothstep(-0.05, 0.02, moon_elev);

    let sun_height = clamp(sun_elev * 0.5 + 0.5, 0.0, 1.0);

    let water_to_cam_xz = uniforms.camera_local.xz - in.world.xz;
    let water_to_cam_y = uniforms.camera_local.y - water.sea_level;
    let to_cam = normalize(vec3<f32>(water_to_cam_xz.x, water_to_cam_y, water_to_cam_xz.y));
    let dist = length(vec3<f32>(water_to_cam_xz.x, water_to_cam_y, water_to_cam_xz.y));

    let t = uniforms.time * 300.0;

    let tiling = water.wave_tiling;
    let strength = water.wave_strength;

    // LOD
    let x = clamp((dist - 200.0) / 1800.0, 0.0, 1.0);
    let detail_lod = x * x * x;

    let hf = clamp(dist / 1200.0, 0.0, 1.0);
    let tile_lod = mix(tiling, tiling * 0.09, hf);
    let eps_scale = mix(1.0, 0.25, hf);

    let n_near = layered_wave_normal(wave_xz, tiling, strength, t, 1.0);
    let n_far = layered_wave_normal(wave_xz, tile_lod, strength * 0.6, t, eps_scale);
    let n = normalize(mix(n_near, n_far, detail_lod));

    let n_dot_l = max(dot(n, sun_dir), 0.0);
    let n_dot_m = max(dot(n, moon_dir), 0.0);

    let view_n = max(dot(n, to_cam), 0.0);
    let f0 = 0.02;
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - view_n, 5.0);

    // Night factor
    let night = 1.0 - smoothstep(-0.08, 0.10, sun_elev);
    let moon_day_suppress = pow(night, 2.5);

    // Moon phase visibility
    let phase = abs(sky.moon_phase * 2.0 - 1.0);
    let moon_phase_vis = clamp(1.0 - phase, 0.0, 1.0);


    let sun_reflect_dir = reflect(-sun_dir, n);
    let sun_reflect_align = max(dot(sun_reflect_dir, to_cam), 0.0);

    let sun_spec_power_base = mix(20.0, 80.0, sun_height);
    let sun_spec_power = sun_spec_power_base / sky.sun_size;

    // Core tight reflection
    let sun_spec_core = pow(sun_reflect_align, sun_spec_power);

    // Wider glow (lower power = broader)
    let sun_spec_glow = pow(sun_reflect_align, sun_spec_power * 0.15) * 0.25;

    // Glitter for close water
    let glitter_fade = clamp(dist / 600.0, 0.0, 1.0);
    let glitter_mask = fbm(wave_xz * tile_lod * 4.0, t * 1.7, strength * 0.5) * 0.5 + 0.5;
    let sun_glitter = pow(sun_reflect_align, sun_spec_power * 0.3) * glitter_mask * (1.0 - glitter_fade) * 0.3;

    let sun_spec_total = (sun_spec_core + sun_spec_glow + sun_glitter * (1.0 - detail_lod)) * sun_vis;
    let sun_spec_color = vec3<f32>(1.0, 0.98, 0.92);

    let moon_reflect_dir = reflect(-moon_dir, n);
    let moon_reflect_align = max(dot(moon_reflect_dir, to_cam), 0.0);

    let moon_spec_power = 30.0 / sky.moon_size;

    let moon_spec_core = pow(moon_reflect_align, moon_spec_power);
    let moon_spec_glow = pow(moon_reflect_align, moon_spec_power * 0.2) * 0.15;

    let moon_spec_total = (moon_spec_core + moon_spec_glow) * moon_vis * moon_phase_vis * moon_day_suppress;
    let moon_spec_color = vec3<f32>(0.8, 0.85, 0.95);

    let view_down = clamp(n.y, 0.0, 1.0);
    let horizon_factor = smoothstep(0.3, 0.0, view_down);

    // Day sky
    let sky_zenith = vec3<f32>(0.4, 0.6, 0.9);
    let sky_horizon = vec3<f32>(0.7, 0.8, 0.95);
    let day_sky = mix(sky_zenith, sky_horizon, horizon_factor);

    // Night sky
    let night_zenith = vec3<f32>(0.005, 0.01, 0.025);
    let night_horizon = vec3<f32>(0.01, 0.02, 0.04);
    let night_sky = mix(night_zenith, night_horizon, horizon_factor);

    // Blend and apply exposure
    let env_sky = mix(day_sky, night_sky, night) * sky.exposure;
    let sky_fresnel_reflection = env_sky * fresnel;

    let base_water = water.color.rgb;
    let depth_noise = fbm(wave_xz * tile_lod * 0.6, t * 0.2, strength * 0.5) * 0.5 + 0.5;
    let view_tint = smoothstep(0.2, 1.0, view_down);

    let deep_color = base_water * vec3<f32>(0.85, 0.9, 0.95);
    let shallow_color = base_water * vec3<f32>(1.0, 1.0, 1.0);

    let body_mix = clamp(depth_noise * 0.7 + view_tint * 0.3, 0.0, 1.0);
    let body_color = mix(shallow_color, deep_color, body_mix);

    let sun_diffuse = 0.15 + 0.85 * n_dot_l;
    let moon_diffuse = 0.02 + 0.98 * n_dot_m;
    let moon_diffuse_lit = moon_diffuse * sky.moon_intensity * 0.1 * moon_day_suppress * moon_vis;
    let total_diffuse = mix(sun_diffuse * sun_vis, moon_diffuse_lit, night);

    let slope = length(n.xz);
    let foam_noise = fbm(wave_xz * tile_lod * 1.3, t * 0.6, strength * 1.3);
    let foam_edges = smoothstep(0.7, 1.0, slope + foam_noise * 0.4);
    let foam_dist_fade = 1.0 - clamp(dist / 2500.0, 0.0, 1.0);
    let foam_amount = foam_edges * foam_dist_fade * (1.0 - detail_lod);
    let foam_color = vec3<f32>(0.95, 0.97, 1.0);


    // Base: body color with diffuse lighting
    var color = body_color * total_diffuse;

    // Atmospheric extinction at night
    let night_ext = exp(-dist * 0.001 * night);

    // Add foam
    color += foam_color * foam_amount * night_ext;

    // Ambient
    let ambient_day = vec3<f32>(0.9, 0.95, 1.0);
    let ambient_night = vec3<f32>(0.05, 0.07, 0.15);
    let ambient_strength = mix(0.03, 0.01, night) * sky.exposure;
    color += body_color * ambient_strength * mix(ambient_day, ambient_night, night);

    // Night overall darkening
    color *= mix(1.0, 0.4, night);

    // Distance fog / horizon blend
    let far = smoothstep(600.0, 4500.0, dist);
    let horizon_view = smoothstep(0.15, 0.75, 1.0 - view_down);
    let far_color = env_sky * 0.5;
    let far_blend = far * mix(0.3, 0.8, night) * (0.6 + 0.4 * horizon_view);
    color = mix(color, far_color, far_blend);

    // Sky fresnel reflection (fades with distance)
    let sky_refl_strength = mix(0.5, 0.15, smoothstep(0.0, 2000.0, dist));
    color += sky_fresnel_reflection * sky_refl_strength;

    let sun_refl_intensity = sun_spec_total * sky.sun_intensity;
    let sun_dist_fade = mix(1.0, 0.5, smoothstep(0.0, 3000.0, dist));
    color += sun_spec_color * sun_refl_intensity * sun_dist_fade * 3.0;

    let moon_refl_intensity = moon_spec_total * sky.moon_intensity * sky.exposure;
    let moon_dist_fade = mix(1.0, 0.4, smoothstep(0.0, 2000.0, dist));
    color += moon_spec_color * moon_refl_intensity * moon_dist_fade * 5.0;

    let alpha = clamp(water.color.a + foam_amount * 0.15, 0.0, 1.0);
    out.color = vec4<f32>(color, alpha);

    // Motion vector
    let curr_ndc = in.curr_clip.xy / in.curr_clip.w;
    let prev_ndc = in.prev_clip.xy / in.prev_clip.w;
    let curr_uv = curr_ndc * vec2<f32>(0.5, -0.5) + 0.5;
    let prev_uv = prev_ndc * vec2<f32>(0.5, -0.5) + 0.5;
    out.motion = curr_uv - prev_uv;

    return out;
}
