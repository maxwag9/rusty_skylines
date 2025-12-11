// ----------------------------------------
// SKY UNIFORMS
// ----------------------------------------
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

@group(0) @binding(0)
var<uniform> u: Uniforms;

struct SkyUniform {
    day_time: f32,
    day_length: f32,

    exposure: f32,
    _pad0: f32,

    sun_size: f32,
    sun_intensity: f32,

    moon_size: f32,
    moon_intensity: f32,

    moon_phase: f32,
    _pad1: f32,
};

@group(1) @binding(0)
var<uniform> sky: SkyUniform;

// ----------------------------------------
// VERTEX
// ----------------------------------------
// fullscreen triangle
struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) dir: vec3<f32>,
    @location(1) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
    var clip: vec4<f32>;

    if vid == 0u {
        clip = vec4<f32>(-1.0, -1.0, 1.0, 1.0);
    } else if vid == 1u {
        clip = vec4<f32>( 3.0, -1.0, 1.0, 1.0);
    } else {
        clip = vec4<f32>(-1.0,  3.0, 1.0, 1.0);
    }

    // world direction reconstruction
    let inv_vp = u.inv_view_proj;
    let world_pos = inv_vp * clip;
    let dir = normalize(world_pos.xyz / world_pos.w - u.camera_pos);

    let ndc = clip.xy / clip.w;

    return VSOut(clip, dir, ndc);
}



// ----------------------------------------
// HELPERS
// ----------------------------------------
fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = clamp((x - a) / (b - a), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn hash1(x: f32) -> f32 {
    let n = fract(x * 0.1031);
    let n2 = n * n;
    return fract(n2 * (n + 33.33));
}


fn hash2(x: f32) -> vec2<f32> {
    let h = hash1(x);
    return vec2<f32>(h, hash1(h + 31.7));
}

fn hash12(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(hash12(p), hash12(p + vec2<f32>(5.2, 1.3)));
}

fn crater_shape(N: vec3<f32>, idx: i32) -> f32 {
    // deterministic seed
    let s = f32(idx);

    // random position on sphere
    let h = hash2(s);
    let yaw   = h.x * 6.2831853;
    let pitch = (h.y * 2.0 - 1.0);

    let cp = sqrt(max(1.0 - pitch * pitch, 0.0));
    let crater_pos = normalize(vec3<f32>(
        cp * cos(yaw),
        pitch,
        cp * sin(yaw)
    ));

    // angular distance (dot on unit sphere)
    let d = dot(N, crater_pos);

    // crater radius
    let rad = mix(0.985, 0.998, hash1(s + 13.7));

    if d > rad {
        let t = (d - rad) / (1.0 - rad);
        return pow(1.0 - t, 3.0);
    }

    return 0.0;
}

// ----------------------------------------
// ATMOSPHERE & STARS (ANALYTIC)
// ----------------------------------------

fn phase_rayleigh(cos_theta: f32) -> f32 {
    let c2 = cos_theta * cos_theta;
    return 0.75 * (1.0 + c2);
}

fn phase_mie(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);
    return (1.0 - g2) / max(denom, 1e-3);
}

fn hash3(p: vec3<f32>) -> f32 {
    let q = fract(p * 0.3183099 + vec3<f32>(0.1, 0.2, 0.3));
    return fract(q.x * q.y * q.z * 23.133);
}


fn air_mass(mu: f32) -> f32 {
    let m = 1.0 / max(mu * 0.9 + 0.15, 0.06);
    return clamp(m, 0.0, 10.0);
}

fn compute_atmosphere(
    view_dir_in: vec3<f32>,
    sun_dir: vec3<f32>,
    moon_dir: vec3<f32>,
    ndc: vec2<f32>
) -> vec3<f32> {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let view_dir = normalize(view_dir_in);

    let mu = clamp(dot(view_dir, up), -1.0, 1.0);

    let view_up = mu * 0.5 + 0.5;
    let view_h  = 1.0 - view_up;

    let sun_alt  = clamp(dot(sun_dir, up), -1.0, 1.0);
    let moon_alt = clamp(dot(moon_dir, up), -1.0, 1.0);

    let day_weight   = smoothstep(-0.90, 0.45, sun_alt);
    let night_factor = 1.0 - smoothstep(-0.45, 0.10, sun_alt);

    let day_zenith  = vec3<f32>(0.06, 0.32, 0.80);
    let day_horizon = vec3<f32>(0.45, 0.70, 0.98);
    var day_col     = mix(day_horizon, day_zenith, pow(view_up, 0.55));

    let tw_raw = clamp(1.0 - abs(sun_alt) / 0.9, 0.0, 1.0);
    let tw = pow(tw_raw, 1.8);

    let sun_dir_h = normalize(vec3<f32>(sun_dir.x, 0.0, sun_dir.z));
    let view_dir_h = normalize(vec3<f32>(view_dir.x, 0.0, view_dir.z));
    let horiz_align = max(dot(view_dir_h, sun_dir_h), 0.0);
    let tw_geom = pow(horiz_align, 3.0) * pow(view_h, 1.2);

    let sunset_low  = vec3<f32>(1.30, 0.50, 0.18);
    let sunset_mid  = vec3<f32>(0.80, 0.30, 0.32);
    let sunset_high = vec3<f32>(0.45, 0.30, 0.70);

    let tint = mix(mix(sunset_low, sunset_mid, view_up), sunset_high, pow(view_up, 2.0));
    let twilight_factor = clamp(tw * tw_geom, 0.0, 1.0);
    day_col = mix(day_col, tint, twilight_factor);

    var night_col = vec3<f32>(0.0);

    let moon_phase = clamp(sky.moon_phase, 0.0, 1.0);
    let moon_lum = sky.moon_intensity * moon_phase * clamp(moon_alt + 0.1, 0.0, 1.0);

    let zenith_factor = pow(max(mu, 0.0), 3.0);
    night_col += vec3<f32>(0.015, 0.02, 0.03) * moon_lum * zenith_factor * 0.25;
    let horizon_lift = vec3<f32>(0.01, 0.015, 0.02) * pow(1.0 - max(mu, 0.0), 2.0);
    night_col += horizon_lift;
    let airglow = vec3<f32>(0.008, 0.012, 0.006) * pow(max(mu, 0.0), 4.0);
    night_col += airglow;
    let np = ndc * 50.0; // very soft
    let n = hash12(np);
    let night_noise = (n - 0.5) * 0.01;
    night_col += night_noise;

    var sky_col = mix(night_col, day_col, day_weight);

    let hemi_fade = smoothstep(-0.60, 0.10, mu);
    sky_col *= hemi_fade;

    let p = ndc * 0.5 + vec2<f32>(0.5, 0.5);
    let dither_amp = 1.0 / 255.0;
    sky_col += (n - 0.5) * dither_amp;

    return max(sky_col, vec3<f32>(0.0));
}

// ----------------------------------------
// FRAGMENT
// ----------------------------------------
@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    var col = vec3<f32>(0.0);

    let sun_dir = normalize(u.sun_direction);
    let sun_pos_world = sun_dir * 1000000.0;
    let sun_clip = u.view_proj * vec4<f32>(sun_pos_world, 1.0);

    var sun_ndc = vec2<f32>(9999.0);
    let sun_visible = sun_clip.w > 0.0;

    var sun_color_final = vec3<f32>(1.0, 1.0, 1.0);

    let alt = clamp(dot(sun_dir, up), -1.0, 1.0);
    let h = clamp((alt + 0.05) / 1.05, 0.0, 1.0);

    let sunrise_color = vec3<f32>(1.35, 0.55, 0.22);
    let midday_color  = vec3<f32>(1.0, 0.97, 0.90);
    let solar_color   = mix(sunrise_color, midday_color, h);

    let air_mass = 1.0 / max(alt * 0.9 + 0.1, 0.02);
    let extinction = exp(-vec3<f32>(0.15, 0.10, 0.05) * (air_mass - 1.0));
    let extinction_strength = 0.4;
    sun_color_final = mix(solar_color, solar_color * extinction, extinction_strength);

    if sun_visible {
        sun_ndc = sun_clip.xy / sun_clip.w;

        let d = distance(input.ndc, sun_ndc);
        let radius = sky.sun_size;
        let intensity = sky.sun_intensity;

        let core_r = radius * 0.75;

        if d < core_r {
            let x = 1.0 - d / core_r;
            let limb = 0.55 + 0.45 * pow(x, 0.35);
            col += sun_color_final * limb * intensity;
        } else if d < radius {
            let t = 1.0 - (d - core_r) / (radius - core_r);
            let glow = pow(t, 3.0);
            col += sun_color_final * glow * intensity;
        }

        {
            let halo_radius = radius * 16.0;
            let halo_d = distance(input.ndc, sun_ndc);

            if (halo_d < halo_radius) {
                let t = 1.0 - halo_d / halo_radius;

                let tight = pow(t, 2.0);
                let soft  = pow(t, 4.0);

                let halo = tight * 0.9 + soft * 0.25;

                let sun_alt = clamp(dot(sun_dir, vec3<f32>(0.0, 1.0, 0.0)), -1.0, 1.0);
                let sun_lum = sky.sun_intensity * clamp(sun_alt + 0.05, 0.0, 1.0);

                let corona_low  = vec3<f32>(1.2, 0.7, 0.35);
                let corona_high = vec3<f32>(0.7, 0.85, 1.1);
                let halo_color  = mix(corona_low, corona_high, h);

                col += halo_color * halo * sun_lum;
            }
        }

        if d < core_r {
            let uv = (input.ndc - sun_ndc) / core_r;
            let n = fract(sin(dot(uv, vec2<f32>(12.9898, 78.233))) * 43758.5453);
            let spots = pow(n, 16.0) * 0.18;
            col *= 1.0 - spots;
        }
    }

    let moon_dir = normalize(u.moon_direction);
    let moon_dist = 10000.0;
    let moon_world_pos = moon_dir * moon_dist;
    let moon_clip = u.view_proj * vec4<f32>(moon_world_pos, 1.0);

    if moon_clip.w > 0.0 {
        let moon_ndc = moon_clip.xy / moon_clip.w;
        let m_radius = sky.moon_size;

        let rel = (input.ndc - moon_ndc) / m_radius;
        let r2  = dot(rel, rel);

        if r2 <= 1.0 {
            let z = sqrt(max(1.0 - r2, 0.0));
            let N_local = normalize(vec3<f32>(rel.x, rel.y, z));

            let forward_m = normalize(-moon_dir);
            var up_ref_m = vec3<f32>(0.0, 1.0, 0.0);
            if abs(dot(up_ref_m, forward_m)) > 0.95 {
                up_ref_m = vec3<f32>(1.0, 0.0, 0.0);
            }
            let right_m = normalize(cross(up_ref_m, forward_m));
            let up_m    = normalize(cross(forward_m, right_m));

            let N_face =
                N_local.x * right_m +
                N_local.y * up_m +
                N_local.z * forward_m;

            let r = clamp(sqrt(r2), 0.0, 1.0);

            let view_ray = normalize(-input.dir);
            let forward  = view_ray;

            var ref_up = vec3<f32>(0.0, 1.0, 0.0);
            if abs(dot(ref_up, forward)) > 0.95 {
                ref_up = vec3<f32>(1.0, 0.0, 0.0);
            }

            let right    = normalize(cross(ref_up, forward));
            let up_local = normalize(cross(forward, right));

            let N_world =
                N_local.x * right +
                N_local.y * up_local +
                N_local.z * forward;

            let N = normalize(N_world);

            let L = normalize(u.sun_direction);
            let V = normalize(-view_ray);

            let raw_light = dot(N, L);

            let phase = smoothstep(-0.2, 0.2, raw_light);
            let limb = mix(1.0, 0.50, r);

            let crescent_intensity = clamp(1.0 - phase, 0.0, 1.0);
            let crescent_glow = pow(crescent_intensity, 3.5) * 0.0005;
            let crescent_color = vec3<f32>(1.2, 1.05, 0.90) * crescent_glow;

            let maria_pattern =
                0.5 + 0.5 * sin(N_world.x * 18.0 + N_world.y * 7.0 + N_world.z * 4.0);

            let maria = mix(0.92, 0.89, maria_pattern);

            var bowl_acc = 0.0;
            var rim_acc  = 0.0;

            let CRATERS = 32;

            for (var i = 0; i < CRATERS; i = i + 1) {
                let cr = crater_shape(N_face, i);

                bowl_acc = max(bowl_acc, cr);

                let rim = pow(cr, 12.0);
                rim_acc = max(rim_acc, rim);
            }

            let bowl   = bowl_acc * 0.25;
            let rim    = rim_acc  * 0.10;

            var albedo_mod = maria;

            albedo_mod = mix(albedo_mod, albedo_mod * 0.65, bowl);
            albedo_mod = mix(albedo_mod, albedo_mod * 1.25, rim);

            let crater_soft = smoothstep(0.0, 1.0, bowl + rim * 0.5);
            albedo_mod = mix(albedo_mod, albedo_mod * 0.92, crater_soft * 0.2);

            let crater_centers = array<vec3<f32>, 6>(
                normalize(vec3<f32>( 0.20,  0.10, 0.98)),
                normalize(vec3<f32>(-0.32,  0.15, 0.94)),
                normalize(vec3<f32>( 0.05, -0.35, 0.93)),
                normalize(vec3<f32>( 0.40, -0.10, 0.91)),
                normalize(vec3<f32>(-0.28, -0.20, 0.92)),
                normalize(vec3<f32>( 0.00,  0.00, 1.00))
            );

            let crater_radius = array<f32, 6>(
                0.22,
                0.18,
                0.25,
                0.15,
                0.13,
                0.10
            );

            var crater_factor = 1.0;

            for (var i = 0u; i < 6u; i = i + 1u) {
                let center = crater_centers[i];
                let radius = crater_radius[i];

                let d2 = distance(N_world, center);

                let crater = smoothstep(radius, 0.0, d2);

                crater_factor -= crater * 0.08;
            }

            let rough = mix(1.0, 0.78, pow(crater_factor, 8.0));

            let base_sun = vec3<f32>(1.05, 1.0, 0.97);
            let illum = base_sun * sky.moon_intensity;

            let light = phase * limb * rough;

            let albedo = vec3<f32>(albedo_mod, albedo_mod, albedo_mod);

            var moon_col = albedo * illum * light;

            let dark_fill = 0.0005 * (1.0 - phase);
            moon_col += vec3<f32>(0.12, 0.13, 0.17) * dark_fill;

            moon_col += crescent_color;

            let edge = smoothstep(1.0, 0.40, r2);
            moon_col *= edge;

            col += moon_col;
        }

        {
            let m_radius = sky.moon_size;

            let halo_radius = m_radius * 4.0;

            let halo_d = distance(input.ndc, moon_ndc);

            if (halo_d < halo_radius) {
                let t = 1.0 - halo_d / halo_radius;

                let tight = pow(t, 2.0);
                let soft  = pow(t, 4.0);

                let halo = tight * 0.9 + soft * 0.25;

                let moon_alt = clamp(dot(moon_dir, vec3<f32>(0.0,1.0,0.0)), -1.0, 1.0);
                let moon_lum = sky.moon_intensity * clamp(moon_alt + 0.1, 0.0, 1.0) * 0.1;

                let halo_color = vec3<f32>(0.45, 0.47, 0.55);

                col += halo_color * halo * moon_lum;
            }
        }
    }

    let view_dir = normalize(input.dir);
    let atmosphere = compute_atmosphere(view_dir, sun_dir, moon_dir, input.ndc);
    col += atmosphere;

    let sun_alt_for_alpha = clamp(dot(sun_dir, up), -1.0, 1.0);
    let night_factor_alpha = 1.0 - smoothstep(-0.45, 0.10, sun_alt_for_alpha);
    let hemi_fade_alpha = smoothstep(-0.60, 0.10, dot(view_dir, up));
    let sky_alpha = clamp(1.0 - night_factor_alpha, 0.0, 1.0) * hemi_fade_alpha;


    let exposure = max(sky.exposure, 0.00001);
    let mapped = vec3<f32>(1.0) - exp(-col * exposure);

    return vec4<f32>(mapped, sky_alpha);
}
