struct WaterUniform {
    sea_level: f32,
    color: vec4<f32>,
    wave_tiling: f32,
    wave_strength: f32,
};

struct Uniforms {
    view_proj: mat4x4<f32>,
    sun_direction: vec3<f32>,
    time: f32,
    _pad1: vec4<f32>,
    camera_pos: vec3<f32>,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<uniform> water: WaterUniform;

fn hash2(p: vec2<i32>) -> f32 {
    let p2 = vec2<f32>(f32(p.x), f32(p.y));
    let h = dot(p2, vec2<f32>(127.1, 311.7));
    let s = sin(h) * 43758.5453;
    return fract(s);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

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
    let v = mix(vx0, vx1, u.y);

    return v;
}

fn fbm(p: vec2<f32>, t: f32, speed: f32) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var pp = p;

    let s = speed;

    v += value_noise(pp + vec2<f32>(0.07 * t * s, 0.11 * t * s)) * a;
    a *= 0.5;
    pp = mat2x2<f32>(0.8, -0.6, 0.6, 0.8) * pp * 1.9;

    v += value_noise(pp + vec2<f32>(-0.13 * t * s, 0.05 * t * s)) * a;
    a *= 0.5;
    pp = mat2x2<f32>(0.3, -0.95, 0.95, 0.3) * pp * 2.1;

    v += value_noise(pp + vec2<f32>(0.19 * t * s, -0.09 * t * s)) * a;
    a *= 0.5;
    pp = mat2x2<f32>(0.99, -0.14, 0.14, 0.99) * pp * 2.5;

    v += value_noise(pp + vec2<f32>(-0.03 * t * s, -0.17 * t * s)) * a;

    return v * 2.0 - 1.0;
}

fn layered_wave_normal(p: vec2<f32>, tiling: f32, strength: f32, t: f32) -> vec3<f32> {
    let wp = p * tiling;

    let eps_large = 1.2;
    let eps_small = 0.25;

    let hL  = fbm(wp * 0.35, t * 0.35, strength * 0.7);
    let hLx = fbm((wp + vec2<f32>(eps_large, 0.0)) * 0.35, t * 0.35, strength * 0.7);
    let hLz = fbm((wp + vec2<f32>(0.0, eps_large)) * 0.35, t * 0.35, strength * 0.7);

    let hS  = fbm(wp * 3.0, t * 1.8, strength * 0.4);
    let hSx = fbm((wp + vec2<f32>(eps_small, 0.0)) * 3.0, t * 1.8, strength * 0.4);
    let hSz = fbm((wp + vec2<f32>(0.0, eps_small)) * 3.0, t * 1.8, strength * 0.4);

    let dhdx = (hLx - hL) * 0.8 + (hSx - hS) * 1.6;
    let dhdz = (hLz - hL) * 0.8 + (hSz - hS) * 1.6;

    let base_n = normalize(vec3<f32>(-dhdx * strength, 1.0, -dhdz * strength));

    let chop = 1.5;
    let nxz = base_n.xz * chop;
    let ny = max(base_n.y, 0.2);

    return normalize(vec3<f32>(nxz.x, ny, nxz.y));
}

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world: vec3<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> VSOut {
    let wp = vec3<f32>(pos.x, water.sea_level, pos.z);

    var out: VSOut;
    out.world = wp;
    out.pos = uniforms.view_proj * vec4<f32>(wp, 1.0);
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let sun_dir = normalize(uniforms.sun_direction);
    let cam_pos = uniforms.camera_pos;
    let to_cam = normalize(cam_pos - in.world);
    let t = uniforms.time * 300.0;

    var tiling = water.wave_tiling;
    if (tiling <= 0.0) {
        tiling = 0.0008;
    }

    var strength = water.wave_strength;
    if (strength <= 0.0) {
        strength = 0.8;
    }

    let dist = length(cam_pos - in.world);
    let x = clamp((dist - 200.0) / 1800.0, 0.0, 1.0);
    let detail_lod = x * x * x;



    let n_near = layered_wave_normal(in.world.xz, tiling, strength, t);
    let n_far  = layered_wave_normal(in.world.xz, tiling * 0.3, strength * 0.6, t);
    let n = normalize(mix(n_near, n_far, detail_lod));

    let up = vec3<f32>(0.0, 1.0, 0.0);

    let n_dot_l = max(dot(n, sun_dir), 0.0);
    let view_n = max(dot(n, to_cam), 0.0);

    let f0 = 0.02;
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - view_n, 5.0);

    let half_vec = normalize(sun_dir + to_cam);
    let glossy = pow(max(dot(n, half_vec), 0.0), 100.0);

    let sun_ref = reflect(-sun_dir, n);
    let sparkle_core = pow(max(dot(sun_ref, to_cam), 0.0), 250.0);
    let glitter_mask = fbm(in.world.xz * tiling * 4.0, t * 1.7, strength * 0.5) * 0.5 + 0.5;
    let spec = glossy * 0.7 + sparkle_core * glitter_mask * 0.6 * (1.0 - detail_lod);

    let base_water = water.color.rgb;

    let depth_noise = fbm(in.world.xz * tiling * 0.6, t * 0.2, strength * 0.5) * 0.5 + 0.5;
    let view_down = clamp(dot(n, up), 0.0, 1.0);
    let view_tint = smoothstep(0.2, 1.0, view_down);

    let deep_color = base_water * vec3<f32>(0.95, 0.90, 0.85);
    let shallow_color = base_water * vec3<f32>(1.05, 1.00, 0.95);

    let body_mix = clamp(depth_noise * 0.7 + view_tint * 0.3, 0.0, 1.0);
    let body_color = mix(shallow_color, deep_color, body_mix);

    let sky_color = vec3<f32>(0.55, 0.70, 0.95);
    let horizon_color = vec3<f32>(0.65, 0.80, 0.98);
    let horizon_factor = smoothstep(0.3, 0.0, view_down);
    let sky_reflection = mix(sky_color, horizon_color, horizon_factor);
    let reflection = sky_reflection * fresnel;

    let slope = length(n.xz);
    let foam_noise = fbm(in.world.xz * tiling * 1.3, t * 0.6, strength * 1.3);
    let foam_edges = smoothstep(0.7, 1.0, slope + foam_noise * 0.4);
    let foam_dist_fade = 1.0 - clamp(dist / 2500.0, 0.0, 1.0);
    let foam = foam_edges * foam_dist_fade * (1.0 - detail_lod);

    let foam_color = vec3<f32>(0.90, 0.97, 1.0) * foam;

    let diffuse = 0.15 + 0.85 * n_dot_l;

    let dist_fade = clamp(dist / 4000.0, 0.0, 1.0);
    let far_tint = vec3<f32>(0.25, 0.35, 0.55);

    var color = body_color * diffuse;
    color = mix(color, far_tint, dist_fade * 0.35);
    color += reflection * 0.9;
    color += foam_color;
    color += spec * vec3<f32>(1.0, 0.98, 0.9) * 1.3;

    let alpha_base = water.color.a;
    let alpha = clamp(alpha_base + foam * 0.2, 0.0, 1.0);

    return vec4<f32>(color, alpha);
}
