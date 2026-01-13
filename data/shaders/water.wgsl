struct WaterUniform {
    sea_level: f32,
    color: vec4<f32>,
    wave_tiling: f32,
    wave_strength: f32,
};

struct Uniforms {
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,

    sun_direction: vec3<f32>,
    time: f32,

    camera_pos: vec3<f32>,
    orbit_radius: f32,

    moon_direction: vec3<f32>,
    _pad0: f32,
};

struct SkyUniform {
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

fn get_orbit_target() -> vec3<f32> {
    // Camera forward in world space (-Z of inv_view)
    let forward = normalize(-uniforms.inv_view[2].xyz);

    return uniforms.camera_pos + forward * uniforms.orbit_radius;
}

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
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> VSOut {
    var wp = vec3<f32>(pos.x, water.sea_level, pos.z);

    var out: VSOut;
    let target_pos = get_orbit_target();
    wp.x += target_pos.x;
    wp.z += target_pos.z;
    out.world = wp;
    out.pos = uniforms.view_proj * vec4<f32>(wp, 1.0);
    return out;
}


@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let sun_dir = normalize(uniforms.sun_direction);
    let moon_dir = normalize(uniforms.moon_direction);

    let sun_elev = sun_dir.y;
    let moon_elev = moon_dir.y;

    let sun_vis = smoothstep(-0.05, 0.02, sun_elev);
    let moon_vis = smoothstep(-0.05, 0.02, moon_elev);

    // 0 = deep night, 0.5 = horizon, 1 = midday
    let sun_height = clamp(sun_elev * 0.5 + 0.5, 0.0, 1.0);

    let cam_pos = uniforms.camera_pos;
    let to_cam = normalize(cam_pos - in.world);
    let dist = length(cam_pos - in.world);

    let t = uniforms.time * 300.0;

    var tiling = water.wave_tiling;
    if (tiling <= 0.0) {
        tiling = 0.0008;
    }

    var strength = water.wave_strength;
    if (strength <= 0.0) {
        strength = 0.8;
    }

    // LOD
    let x = clamp((dist - 200.0) / 1800.0, 0.0, 1.0);
    let detail_lod = x * x * x;

    let hf = clamp(dist / 1200.0, 0.0, 1.0);
    let tile_lod = mix(tiling, tiling * 0.03, hf);
    let eps_scale = mix(1.0, 0.25, hf);

    let n_near = layered_wave_normal(in.world.xz, tiling, strength, t, 1.0);
    let n_far = layered_wave_normal(in.world.xz, tile_lod, strength * 0.6, t, eps_scale);
    let n = normalize(mix(n_near, n_far, detail_lod));

    let n_dot_l = max(dot(n, sun_dir), 0.0);
    let n_dot_m = max(dot(n, moon_dir), 0.0);

    let view_n = max(dot(n, to_cam), 0.0);
    let f0 = 0.02;
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - view_n, 5.0);

    // Specular sparkle shaping
    let sun_size = max(sky.sun_size, 0.001);
    let k = smoothstep(0.0, 1.0, sun_height);
    let blend = k * k;

    let glossy_exp = mix(0.5 / sun_size, 4.0 / sun_size, blend);
    let sparkle_exp = mix(1.0 / sun_size, 8.0 / sun_size, blend);

    let half_vec = normalize(sun_dir + to_cam);
    let glitter_fade = clamp(dist / 600.0, 0.0, 1.0);
    let glitter_mask =
        fbm(in.world.xz * tile_lod * 4.0, t * 1.7, strength * 0.5) * 0.5 + 0.5;

    let glossy =
        pow(max(dot(n, half_vec), 0.0), glossy_exp) *
        sun_vis *
        sky.sun_intensity;

    let sun_ref = reflect(-sun_dir, n);
    let sparkle_core =
        pow(max(dot(sun_ref, to_cam), 0.0), sparkle_exp) *
        (1.0 - glitter_fade) *
        sun_vis *
        sky.sun_intensity;

    let sun_spec =
        (glossy * 0.7 + sparkle_core * glitter_mask * 0.6 * (1.0 - detail_lod)) *
        sun_vis;

    let moon_spec =
        pow(max(dot(reflect(-moon_dir, n), to_cam), 0.0), 60.0 / max(sky.moon_size, 0.001)) *
        moon_vis *
        sky.moon_intensity *
        sky.exposure;

    // Body color
    let base_water = water.color.rgb;
    let depth_noise =
        fbm(in.world.xz * tile_lod * 0.6, t * 0.2, strength * 0.5) * 0.5 + 0.5;

    let view_down = clamp(n.y, 0.0, 1.0);
    let view_tint = smoothstep(0.2, 1.0, view_down);

    let deep_color = base_water * vec3<f32>(0.95, 0.90, 0.85);
    let shallow_color = base_water * vec3<f32>(1.05, 1.00, 0.95);

    let body_mix = clamp(depth_noise * 0.7 + view_tint * 0.3, 0.0, 1.0);
    let body_color = mix(shallow_color, deep_color, body_mix);

    // Environment
    let sky_color = vec3<f32>(0.55, 0.70, 0.95);
    let horizon_color = vec3<f32>(0.65, 0.80, 0.98);

    let horizon_factor = smoothstep(0.3, 0.0, view_down);
    let sky_reflection = mix(sky_color, horizon_color, horizon_factor);

    let night = 1.0 - smoothstep(-0.08, 0.10, sun_elev);
    let moon_day_suppress = pow(night, 2.5);

    let phase = abs(sky.moon_phase * 2.0 - 1.0);
    let moon_phase_vis = clamp(1.0 - phase, 0.0, 1.0);
    let moon_visibility = moon_vis * moon_phase_vis;

    let moon_color = vec3<f32>(0.75, 0.78, 0.82);
    let moon_glow = moon_color * moon_visibility * sky.moon_intensity * sky.exposure * moon_day_suppress;

    let night_zenith = vec3<f32>(0.01, 0.02, 0.05);
    let night_horizon = vec3<f32>(0.02, 0.04, 0.08);

    let night_grad = mix(night_zenith, night_horizon, horizon_factor);
    let night_sky = night_grad * sky.exposure;
    let day_sky = sky_reflection * sky.exposure;

    let env = mix(day_sky, night_sky, night) + moon_glow;
    let reflection = env * fresnel;

    // Diffuse lighting
    let sun_diffuse = 0.15 + 0.85 * n_dot_l;
    let moon_diffuse = (0.02 + 0.98 * n_dot_m) * moon_vis;
    let moon_diffuse_lit = moon_diffuse * sky.moon_intensity * 0.15 * moon_day_suppress;
    let total_diffuse = mix(sun_diffuse * sun_vis, moon_diffuse_lit, night);

    var color = body_color * total_diffuse;

    // Distance and horizon shaping
    let far = smoothstep(600.0, 4500.0, dist);
    let horizon_view = smoothstep(0.15, 0.75, 1.0 - view_down);

    // Atmospheric extinction at night
    let night_ext = exp(-dist * 0.0012 * night);

    // Foam
    let slope = length(n.xz);
    let foam_noise = fbm(in.world.xz * tile_lod * 1.3, t * 0.6, strength * 1.3);
    let foam_edges = smoothstep(0.7, 1.0, slope + foam_noise * 0.4);
    let foam_dist_fade = 1.0 - clamp(dist / 2500.0, 0.0, 1.0);
    let foam = foam_edges * foam_dist_fade * (1.0 - detail_lod);

    let foam_color = vec3<f32>(0.90, 0.97, 1.0) * foam;

    // Specular + foam
    color += foam_color * night_ext;
    color += sun_spec * vec3<f32>(1.0, 0.98, 0.9) * sky.sun_intensity;
    color += moon_spec * moon_color * moon_day_suppress;

    // Ambient
    let ambient_col_day = vec3<f32>(1.0, 1.0, 1.0);
    let ambient_col_night = vec3<f32>(0.06, 0.075, 0.2);

    let ambient = mix(0.02, 0.003, night) * sky.exposure;
    color += body_color * ambient * mix(ambient_col_day, ambient_col_night, night);

    // Night diffuse suppression
    color *= mix(1.0, 0.35, night);

    // Far water -> sky convergence
    let far_target_dark = min(env, vec3<f32>(0.0003, 0.0004, 0.0006));
    let far_strength = mix(0.25, 0.95, night);

    color = mix(
        color,
        far_target_dark,
        far * far_strength * (0.6 + 0.4 * horizon_view)
    );

    // Reflections (kept readable at night)
    let reflection_fade = smoothstep(0.0, 1800.0, dist);
    let reflection_darkening_night = mix(1.0, 0.00, reflection_fade);
    let reflection_darkening_day = mix(1.0, 0.70, reflection_fade);
    let reflection_darkening = mix(reflection_darkening_day, reflection_darkening_night, night);

    let reflection_final = reflection * reflection_darkening;

    let refl_ext = mix(1.0, night_ext, 0.25);
    let night_reflect_boost = mix(0.0, 1.6, moon_day_suppress);

    color += reflection_final * 0.9 * refl_ext * night_reflect_boost;

    // Alpha
    let alpha = clamp(water.color.a + foam * 0.2, 0.0, 1.0);
    return vec4<f32>(color, alpha);
}