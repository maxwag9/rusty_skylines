#include "includes/uniforms.wgsl"
// sky.wgsl - Optimized

@group(1) @binding(0) var<uniform> u: Uniforms;

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

@group(1) @binding(1) var<uniform> sky: SkyUniform;

struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) dir: vec3<f32>,
    @location(1) ndc: vec2<f32>,
    @location(2) prev_clip: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
    let clip_xy = vec2<f32>(
        f32((vid << 1u) & 2u) * 2.0 - 1.0,
        f32(vid & 2u) * 2.0 - 1.0
    );
    let clip = vec4<f32>(clip_xy, 1.0, 1.0);
    let view_h = u.inv_proj * clip;
    let view_dir = view_h.xyz / view_h.w;
    let world_dir = (u.inv_view * vec4<f32>(view_dir, 0.0)).xyz;

    var out: VSOut;
    out.clip = clip;
    out.dir = world_dir;
    out.ndc = clip_xy;

    // Reproject: same world direction through previous view_proj
    // w=0 means direction only (infinitely far)
    out.prev_clip = u.prev_view_proj * vec4<f32>(world_dir, 0.0);

    return out;
}

struct FragOut {
    @location(0) color: vec4<f32>,
    @location(2) motion: vec2<f32>,
};

@fragment
fn fs_main(input: VSOut) -> FragOut {
    var out: FragOut;
    let view_dir = normalize(input.dir);
    let sun_dir = u.sun_direction;
    let moon_dir = u.moon_direction;
    let sun_alt = sun_dir.y;
    let ndc = input.ndc;

    var col = vec3<f32>(0.0);

    // Pre-compute sun color
    let h = saturate((sun_alt + 0.05) * 0.952381);
    let solar_base = mix(vec3<f32>(1.35, 0.55, 0.62), vec3<f32>(1.0, 0.97, 0.90), h);
    let air_mass = 1.0 / max(sun_alt * 0.9 + 0.1, 0.02);
    let extinction = exp(vec3<f32>(-0.15, -0.10, -0.05) * (air_mass - 1.0));
    let sun_color = solar_base * mix(vec3<f32>(1.0), extinction, 0.2);

    // Sun disc & halo
    let sun_clip = u.view_proj * vec4<f32>(sun_dir, 0.0);
    if sun_clip.w > 0.0 {
        let sun_ndc = sun_clip.xy / sun_clip.w;
        let d = distance(ndc, sun_ndc);
        let radius = sky.sun_size;
        let intensity = sky.sun_intensity;
        let core_r = radius * 0.85;
        let halo_r = radius * 24.0;

        if d < halo_r {
            // Halo
            let ht = 1.0 - d / halo_r;
            let ht2 = ht * ht;
            let ht4 = ht2 * ht2;
            let halo_col = mix(vec3<f32>(1.2, 0.7, 0.35), vec3<f32>(0.7, 0.85, 1.1), h);
            col += halo_col * (ht4 * ht2 * 0.9 + ht4 * ht4 * 0.35) * intensity * 2.8 * saturate(sun_alt + 0.05);

            if d < radius {
                if d < core_r {
                    // Core with limb darkening
                    let x = 1.0 - d / core_r;
                    col += sun_color * (0.65 + 0.45 * pow(x, 0.35)) * intensity * 2.5;
                    // Surface texture
                    let uv = (ndc - sun_ndc) / core_r;
                    let n = fract(sin(dot(uv, vec2<f32>(12.9898, 78.233))) * 43758.5453);
                    let n2 = n * n; let n4 = n2 * n2; let n8 = n4 * n4; let n16 = n8 * n8;
                    col *= 1.0 - n16 * 0.18;
                } else {
                    // Corona
                    let t = 1.0 - (d - core_r) / (radius - core_r);
                    let t2 = t * t;
                    col += sun_color * t2 * t2 * t2 * intensity;
                }
            }
        }
    }

    // Moon
    let moon_view = normalize((u.view * vec4<f32>(moon_dir, 0.0)).xyz);
    let moon_clip = u.proj * vec4<f32>(moon_view, 1.0);

    if moon_clip.w > 0.0 {
        let moon_ndc = moon_clip.xy / moon_clip.w;
        let m_rad = sky.moon_size;
        let rel = (ndc - moon_ndc) / m_rad;
        let r2 = dot(rel, rel);
        let halo_r = m_rad * 4.0;

        if r2 <= 1.0 {
            // Moon surface
            let z = sqrt(1.0 - r2);
            let N_local = vec3<f32>(rel.x, rel.y, z);

            // View basis for normal
            let view_ray = -view_dir;
            let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(view_ray.y) <= 0.95);
            let right = normalize(cross(up, view_ray));
            let up_loc = cross(view_ray, right);
            let N = normalize(N_local.x * right + N_local.y * up_loc + N_local.z * view_ray);

            let phase = smoothstep(-0.2, 0.2, dot(N, sun_dir));
            let limb = mix(1.0, 0.5, sqrt(r2));

            // Simple maria pattern
            let maria = 0.5 + 0.5 * sin(N.x * 18.0 + N.y * 7.0 + N.z * 4.0);
            let albedo = mix(0.92, 0.89, maria);

            var moon_col = vec3<f32>(albedo) * vec3<f32>(1.05, 1.0, 0.97) * sky.moon_intensity * phase * limb;

            // Crescent glow & earthshine
            let ci = saturate(1.0 - phase);
            moon_col += vec3<f32>(1.2, 1.05, 0.90) * ci * ci * sqrt(ci) * 0.0005;
            moon_col += vec3<f32>(0.12, 0.13, 0.17) * 0.0005 * (1.0 - phase);
            moon_col *= smoothstep(1.0, 0.4, r2);

            col += moon_col;
        } else {
            // Moon halo (only when not on disc)
            let m_d = sqrt(r2) * m_rad;
            if m_d < halo_r {
                let t = 1.0 - m_d / halo_r;
                let t2 = t * t;
                col += vec3<f32>(0.45, 0.47, 0.55) * (t2 * 0.9 + t2 * t2 * 0.25) * sky.moon_intensity * saturate(moon_dir.y + 0.1) * 0.1;
            }
        }
    }

    // Atmosphere (inlined)
    let mu = view_dir.y;
    let view_up = mu * 0.5 + 0.5;
    let view_h = 1.0 - view_up;
    let day_w = smoothstep(-0.9, 0.45, sun_alt);

    // Day gradient
    var day_col = mix(vec3<f32>(0.45, 0.70, 0.98), vec3<f32>(0.06, 0.32, 0.80), pow(view_up, 0.55));

    // Twilight
    let tw = pow(saturate(1.0 - abs(sun_alt) * 1.111), 1.8);
    let sun_h = normalize(vec3<f32>(sun_dir.x, 0.0, sun_dir.z));
    let view_dir_h = normalize(vec3<f32>(view_dir.x, 0.0, view_dir.z));
    let ha = max(dot(view_dir_h, sun_h), 0.0);
    let tw_g = ha * ha * ha * pow(view_h, 1.2);
    let tint = mix(mix(vec3<f32>(1.3, 0.5, 0.18), vec3<f32>(0.8, 0.3, 0.32), view_up),
                   vec3<f32>(0.45, 0.3, 0.7), view_up * view_up);
    day_col = mix(day_col, tint, saturate(tw * tw_g));

    // Night
    let moon_lum = sky.moon_intensity * saturate(sky.moon_phase) * saturate(moon_dir.y + 0.1);
    let mu_p = max(mu, 0.0);
    let mu_p3 = mu_p * mu_p * mu_p;
    let inv_mu = 1.0 - mu_p;
    let night_col = vec3<f32>(0.015, 0.02, 0.03) * moon_lum * mu_p3 * 0.25
                  + vec3<f32>(0.01, 0.015, 0.02) * inv_mu * inv_mu
                  + vec3<f32>(0.008, 0.012, 0.006) * mu_p3 * mu_p;

    var sky_col = mix(night_col, day_col, day_w) * smoothstep(-0.6, 0.1, mu);

    // Dither
    let dither = fract(sin(dot(ndc * 50.0, vec2<f32>(127.1, 311.7))) * 43758.5453);
    sky_col += (dither - 0.5) * 0.00392;

    col += max(sky_col, vec3<f32>(0.0));

    // Tone map
    let mapped = 1.0 - exp(-col * max(sky.exposure, 0.00001));

    // Alpha
    let lum = dot(mapped, vec3<f32>(0.2126, 0.7152, 0.0722));
    let blue = saturate((mapped.b - max(mapped.r, mapped.g)) * 4.0);

    out.color = vec4<f32>(mapped, saturate(lum + blue * (1.0 - lum)));

    // Motion vector
    let curr_ndc = input.ndc;
    let prev_ndc = input.prev_clip.xy / input.prev_clip.w;
    let curr_uv = curr_ndc * vec2(0.5, -0.5) + 0.5;
    let prev_uv = prev_ndc * vec2(0.5, -0.5) + 0.5;
    out.motion = curr_uv - prev_uv;

    return out;
}
