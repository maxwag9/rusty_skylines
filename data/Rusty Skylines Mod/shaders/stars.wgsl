#include "includes/uniforms.wgsl"
#include "includes/sky_uniforms.wgsl"

const STAR_COUNT: u32 = 116812u;
const STAR_MAX_MAGNITUDE: f32 = 9.5;
const STAR_INTENSITY_SCALE: f32 = 10.0;

@group(1) @binding(0)
var<uniform> u: Uniforms;

@group(1) @binding(1)
var<uniform> sky: SkyUniform;

fn ra_dec_to_dir(ra: f32, dec: f32) -> vec3<f32> {
    return vec3<f32>(
        cos(dec) * cos(ra),
        sin(dec),
        cos(dec) * sin(ra)
    );
}

fn exposure_to_star_visibility(exposure: f32) -> f32 {
    return smoothstep(0.02, 0.85, max(exposure, 0.0));
}

fn mag_to_intensity(m: f32) -> f32 {
    return pow(2.512, -m);
}

fn srgb_channel_to_linear(c: f32) -> f32 {
    if (c <= 0.04045) {
        return c / 12.92;
    }

    return pow((c + 0.055) / 1.055, 2.4);
}

fn blackbody_srgb(temperature_k: f32) -> vec3<f32> {
    let t = clamp(temperature_k, 3000.0, 40000.0) / 100.0;
    var rgb: vec3<f32>;

    if (t <= 66.0) {
        rgb.r = 255.0;
        rgb.g = 99.4708025861 * log(t) - 161.1195681661;
        rgb.b = 138.5177312231 * log(t - 10.0) - 305.0447927307;
    } else {
        rgb.r = 329.698727446 * pow(t - 60.0, -0.1332047592);
        rgb.g = 288.1221695283 * pow(t - 60.0, -0.0755148492);
        rgb.b = 255.0;
    }

    return clamp(rgb / 255.0, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn bv_to_rgb(bv: f32) -> vec3<f32> {
    let x = clamp(bv, -0.4, 2.0);

    let temperature_k = clamp(
        4600.0 * (
            1.0 / (0.92 * x + 1.7) +
            1.0 / (0.92 * x + 0.62)
        ),
        3000.0,
        40000.0
    );

    let srgb = blackbody_srgb(temperature_k);

    return vec3<f32>(
        srgb_channel_to_linear(srgb.r),
        srgb_channel_to_linear(srgb.g),
        srgb_channel_to_linear(srgb.b)
    );
}

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) mag: f32,
    @location(1) star_color: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) curr_clip: vec4<f32>,
    @location(4) prev_clip: vec4<f32>,
    @location(5) visibility: f32,
    @location(6) halo_radius_px: f32,
    @location(7) core_radius_px: f32,
};

fn culled_star(mag: f32) -> VSOut {
    var out: VSOut;

    out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.mag = mag;
    out.star_color = vec3<f32>(0.0);
    out.uv = vec2<f32>(0.0);
    out.curr_clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.prev_clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.visibility = 0.0;
    out.halo_radius_px = 0.0;
    out.core_radius_px = 0.0;

    return out;
}

@vertex
fn vs_main(
    @location(0) ra: f32,
    @location(1) dec: f32,
    @location(2) mag: f32,
    @location(3) bv: f32,
    @builtin(vertex_index) vid: u32,
) -> VSOut {
    let visibility = exposure_to_star_visibility(sky.exposure);

    if (mag > STAR_MAX_MAGNITUDE || visibility <= 0.0005) {
        return culled_star(mag);
    }

    let celestial_dir = ra_dec_to_dir(ra, dec);

    let horizontal_dir = (
        sky.star_rotation * vec4<f32>(celestial_dir, 0.0)
    ).xyz;

    let view_dir = (
        u.view * vec4<f32>(horizontal_dir, 0.0)
    ).xyz;

    let center = normalize(view_dir) * 1000.0;

    let corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0)
    );

    let corner = corners[vid & 3u];

    let magnitude_size = max(mag, 0.0);

    let base_core_radius_px = clamp(
        1.20 - magnitude_size * 0.078,
        0.42,
        1.20
    );

    let base_halo_radius_px = clamp(
        2.80 - magnitude_size * 0.16,
        1.25,
        2.80
    );

    let size_scale = pow(visibility, 0.72);

    let core_radius_px = max(
        base_core_radius_px * size_scale,
        0.08
    );

    let halo_radius_px = max(
        base_halo_radius_px * size_scale,
        0.20
    );

    let clip_center = u.proj * vec4<f32>(center, 1.0);

    var clip: vec4<f32>;

    let viewport_width = u.screen_size.x;
    let viewport_height = u.screen_size.y;

    if (viewport_width > 1.0 && viewport_height > 1.0) {
        let pixel_to_ndc = vec2<f32>(
            2.0 / viewport_width,
            2.0 / viewport_height
        );

        let clip_offset =
            corner *
            halo_radius_px *
            pixel_to_ndc *
            clip_center.w;

        clip = clip_center + vec4<f32>(
            clip_offset.x,
            clip_offset.y,
            0.0,
            0.0
        );
    } else {
        let right = vec3<f32>(1.0, 0.0, 0.0);
        let up = vec3<f32>(0.0, 1.0, 0.0);

        let pos_view =
            center +
            right * corner.x * halo_radius_px +
            up * corner.y * halo_radius_px;

        clip = u.proj * vec4<f32>(pos_view, 1.0);
    }

    let prev_horizontal_dir = (
        sky.star_rotation * vec4<f32>(celestial_dir, 0.0)
    ).xyz;

    let prev_clip = u.prev_view_proj * vec4<f32>(
        prev_horizontal_dir * 1000.0,
        0.0
    );

    var out: VSOut;

    out.pos = clip;
    out.mag = mag;
    out.star_color = bv_to_rgb(bv);
    out.uv = corner * 0.5 + 0.5;
    out.curr_clip = clip;
    out.prev_clip = prev_clip;
    out.visibility = visibility;
    out.halo_radius_px = halo_radius_px;
    out.core_radius_px = core_radius_px;

    return out;
}

struct FragOut {
    @location(0) color: vec4<f32>,
    @location(2) motion: vec2<f32>,
};

@fragment
fn fs_main(
    @location(0) mag: f32,
    @location(1) star_color: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) curr_clip: vec4<f32>,
    @location(4) prev_clip: vec4<f32>,
    @location(5) visibility: f32,
    @location(6) halo_radius_px: f32,
    @location(7) core_radius_px: f32,
) -> FragOut {
    var out: FragOut;

    if (mag > STAR_MAX_MAGNITUDE || visibility <= 0.0005) {
        out.color = vec4<f32>(0.0);
        out.motion = vec2<f32>(0.0);
        return out;
    }

    let curr_ndc = curr_clip.xy / curr_clip.w;
    let prev_ndc = prev_clip.xy / prev_clip.w;

    let curr_uv = curr_ndc * vec2<f32>(0.5, -0.5) + 0.5;
    let prev_uv = prev_ndc * vec2<f32>(0.5, -0.5) + 0.5;

    out.motion = curr_uv - prev_uv;

    let local = (uv - vec2<f32>(0.5, 0.5)) * 2.0;
    let pixel_offset = local * halo_radius_px;
    let distance_squared = dot(pixel_offset, pixel_offset);

    let core_sigma = max(core_radius_px * 0.50, 0.12);
    let halo_sigma = max(halo_radius_px * 0.52, 0.22);

    let core_profile = exp(
        -distance_squared /
        (2.0 * core_sigma * core_sigma)
    );

    let halo_profile = exp(
        -distance_squared /
        (2.0 * halo_sigma * halo_sigma)
    );

    let flux = mag_to_intensity(mag);
    let base_intensity = flux * STAR_INTENSITY_SCALE;

    let halo_strength = mix(
        0.035,
        0.16,
        clamp((5.5 - mag) / 6.5, 0.0, 1.0)
    );

    let halo_visibility = pow(visibility, 1.65);

    let core_light = core_profile * visibility;

    let halo_light =
        halo_profile *
        halo_strength *
        halo_visibility;

    let intensity = base_intensity * (core_light + halo_light);

    let alpha = clamp(
        core_light + halo_light,
        0.0,
        1.0
    );

    out.color = vec4<f32>(
        star_color * intensity,
        alpha
    );
//    out.color = vec4<f32>(
//        0.0
//    );
    return out;
}