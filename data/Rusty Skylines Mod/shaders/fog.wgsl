#include "includes/uniforms.wgsl"

struct FogUniforms {
    fog_density: f32,
    fog_height: f32,
    fog_height_falloff: f32,
    fog_start: f32,

    fog_end: f32,
    fog_sky_factor: f32,
    _pad0: f32,
    _pad1: f32,

    fog_color: vec3<f32>,
    _pad2: f32,
};

struct CloudUniforms {
    screen_size: vec2<f32>,
    time: f32,
    coverage: f32,
    density: f32,
    detail_strength: f32,
    wind: vec2<f32>,
    camera_height: f32,
    enabled: u32,
    frame_index: u32,
    history_valid: u32,
    terrain_map_origin_relative: vec2<f32>,
    padding: vec2<u32>,
};

@group(0) @binding(0) var trilinear_sampler: sampler;
@group(0) @binding(2) var depth_half: texture_2d<f32>;

#ifdef MSAA
@group(0) @binding(3) var depth_full_raw: texture_depth_multisampled_2d;
#else
@group(0) @binding(3) var depth_full_raw: texture_depth_2d;
#endif

@group(0) @binding(4) var terrain_height_map: texture_2d<f32>;

@group(1) @binding(0) var<uniform> camera: Uniforms;
@group(1) @binding(1) var<uniform> fog: FogUniforms;
@group(1) @binding(2) var<uniform> cloud: CloudUniforms;

const TERRAIN_MAP_WORLD_SIZE: f32 = 32768.0;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let x = f32(i32(vertex_index & 1u) * 2);
    let y = f32(i32(vertex_index >> 1u) * 2);

    var out: VertexOutput;
    out.uv = vec2<f32>(x, 1.0 - y);
    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

fn saturate_f(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

fn camera_forward() -> vec3<f32> {
    return normalize(
        (camera.inv_view * vec4<f32>(0.0, 0.0, -1.0, 0.0)).xyz
    );
}

fn load_depth_raw(coords: vec2<i32>) -> f32 {
    return textureLoad(depth_full_raw, coords, 0);
}

fn depth_is_sky(raw_depth: f32) -> bool {
    if camera.reversed_depth_z != 0u {
        return raw_depth <= 0.000001;
    }
    return raw_depth >= 0.999999;
}

fn linearize_depth(raw_depth: f32) -> f32 {
    let z_near = camera.near_far_depth.x;
    let z_far = camera.near_far_depth.y;

    if camera.reversed_depth_z != 0u {
        return min(z_far, z_near / max(raw_depth, 0.0000001));
    }

    return (z_near * z_far) / max(
        z_far - raw_depth * (z_far - z_near),
        0.0000001
    );
}

fn get_ray_dir(pixel_xy: vec2<f32>, screen_size: vec2<f32>) -> vec3<f32> {
    let uv = pixel_xy / screen_size;
    let ndc_xy = vec2<f32>(
        uv.x * 2.0 - 1.0,
        (1.0 - uv.y) * 2.0 - 1.0
    );

    let view_h = camera.inv_proj * vec4<f32>(ndc_xy, 0.5, 1.0);
    let view_dir = normalize(view_h.xyz / view_h.w);

    return normalize(
        (camera.inv_view * vec4<f32>(view_dir, 0.0)).xyz
    );
}

fn reconstruct_camera_relative_position(
    ray_dir: vec3<f32>,
    linear_view_depth: f32
) -> vec3<f32> {
    let forward_cos = max(dot(ray_dir, camera_forward()), 0.0001);
    return ray_dir * (linear_view_depth / forward_cos);
}

fn sample_terrain_height(point_relative_to_camera_xz: vec2<f32>) -> vec2<f32> {
    if cloud.padding.y == 0u {
        return vec2<f32>(0.0, 0.0);
    }

    let uv = (
        point_relative_to_camera_xz
        + cloud.terrain_map_origin_relative
        + vec2<f32>(TERRAIN_MAP_WORLD_SIZE * 0.5)
    ) / TERRAIN_MAP_WORLD_SIZE;

    let half_texel = 0.5 / vec2<f32>(textureDimensions(terrain_height_map, 0));
    let sample_uv = clamp(uv, half_texel, vec2<f32>(1.0) - half_texel);
    let height = textureSampleLevel(terrain_height_map, trilinear_sampler, sample_uv, 0.0).r;

    let edge_distance = min(
        min(uv.x, 1.0 - uv.x),
        min(uv.y, 1.0 - uv.y)
    );
    let edge_fade = smoothstep(-0.04, 0.0, edge_distance);

    return vec2<f32>(height, edge_fade);
}

fn height_fog_factor(world_y: f32, terrain_sample: vec2<f32>) -> f32 {
    // Outside the valid terrain-map area, fall back to the original absolute fog height.
    let terrain_base = mix(
        fog.fog_height,
        terrain_sample.x + fog.fog_height,
        saturate_f(terrain_sample.y)
    );
    let height_above_base = max(world_y - terrain_base, 0.0);
    return exp(-height_above_base * max(fog.fog_height_falloff, 0.0));
}

fn hash2(p: vec2<i32>) -> f32 {
    let p_f = vec2<f32>(p);
    return fract(sin(dot(p_f, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn value_noise2(p: vec2<f32>) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let smooth_f = f * f * (vec2<f32>(3.0) - 2.0 * f);

    let a = hash2(i);
    let b = hash2(i + vec2<i32>(1, 0));
    let c = hash2(i + vec2<i32>(0, 1));
    let d = hash2(i + vec2<i32>(1, 1));

    return mix(mix(a, b, smooth_f.x), mix(c, d, smooth_f.x), smooth_f.y);
}

fn fog_noise(point_relative_to_camera_xz: vec2<f32>) -> f32 {
    let map_relative_xz = point_relative_to_camera_xz + cloud.terrain_map_origin_relative;
    let phase = cloud.time * 0.015;
    let wind_phase = cloud.time * 0.009;
    let warp = vec2<f32>(
        sin(map_relative_xz.y * 0.0007 + phase + cloud.wind.x * 0.001),
        cos(map_relative_xz.x * 0.0008 - wind_phase + cloud.wind.y * 0.001)
    ) * 38.0;
    let p = map_relative_xz + warp;

    let broad = value_noise2(p * 0.0011 + vec2<f32>(17.3, 41.7));
    let detail = value_noise2(p * 0.0038 + vec2<f32>(83.1, 9.2));
    return saturate_f(broad * 0.76 + detail * 0.24);
}

fn distance_fog_factor(view_distance: f32) -> f32 {
    let start_distance = max(fog.fog_start, 0.0);
    let end_distance = max(fog.fog_end, start_distance + 0.001);
    return smoothstep(start_distance, end_distance, view_distance);
}

fn scattering_color(ray_dir: vec3<f32>) -> vec3<f32> {
    let sun_dir = normalize(camera.sun_direction);
    let sun_altitude = clamp(sun_dir.y, -1.0, 1.0);
    let daylight = smoothstep(-0.12, 0.16, sun_altitude);
    let twilight = exp(-abs(sun_altitude) * 8.0) * (1.0 - daylight * 0.65);

    let horizon = pow(saturate_f(1.0 - abs(ray_dir.y)), 1.6);
    let night_tint = vec3<f32>(0.47, 0.56, 0.77);
    let day_tint = vec3<f32>(0.76, 0.88, 1.0);
    let dusk_tint = vec3<f32>(1.0, 0.57, 0.36);

    var atmosphere_tint = mix(night_tint, day_tint, daylight);
    atmosphere_tint = mix(atmosphere_tint, dusk_tint, twilight * 0.72);

    let horizon_tint = mix(
        vec3<f32>(0.72, 0.82, 0.96),
        vec3<f32>(1.0, 0.48, 0.27),
        saturate_f(twilight)
    );

    let base = fog.fog_color * atmosphere_tint;
    let haze = mix(base, horizon_tint, horizon * (0.18 + twilight * 0.25));

    let sun_phase = pow(saturate_f(dot(ray_dir, sun_dir)), 12.0);
    let sun_tint = mix(
        vec3<f32>(1.0, 0.50, 0.29),
        vec3<f32>(1.0, 0.88, 0.70),
        daylight
    );

    return max(
        haze + sun_tint * sun_phase * (0.035 + 0.16 * daylight + 0.09 * twilight),
        vec3<f32>(0.0)
    );
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let full_coords = vec2<i32>(floor(in.position.xy));
    let screen_size = vec2<f32>(textureDimensions(depth_full_raw));
    let pixel_xy = vec2<f32>(full_coords) + vec2<f32>(0.5);
    let raw_depth = load_depth_raw(full_coords);
    let ray_dir = get_ray_dir(pixel_xy, screen_size);

    var fog_amount = 0.0;
    var fog_colour = scattering_color(ray_dir);

    if depth_is_sky(raw_depth) {
        let sky_factor = max(fog.fog_sky_factor, 0.0);

        // Only rays directed below the horizon receive sky fog.
        // Fade in over a few degrees to avoid a hard seam at the horizon.
        let below_horizon = smoothstep(0.0, 0.035, -ray_dir.y);
        let horizon_band = pow(
            saturate_f(1.0 - abs(ray_dir.y)),
            1.45
        );

        // Keep the haze concentrated near the horizon and below it.
        fog_amount = saturate_f(
            sky_factor * horizon_band * below_horizon * 0.88
        );
    } else {
        let linear_view_depth = linearize_depth(raw_depth);
        let relative_end = reconstruct_camera_relative_position(ray_dir, linear_view_depth);
        let relative_mid = relative_end * 0.5;

        let terrain_camera = sample_terrain_height(vec2<f32>(0.0));
        let terrain_mid = sample_terrain_height(relative_mid.xz);
        let terrain_end = sample_terrain_height(relative_end.xz);

        let camera_height_density = height_fog_factor(
            cloud.camera_height,
            terrain_camera
        );
        let mid_height_density = height_fog_factor(
            cloud.camera_height + relative_mid.y,
            terrain_mid
        );
        let end_height_density = height_fog_factor(
            cloud.camera_height + relative_end.y,
            terrain_end
        );

        // Simpson-style weighting approximates density through the camera-to-surface path.
        let path_height_density = (
            camera_height_density
            + 4.0 * mid_height_density
            + end_height_density
        ) / 6.0;

        let distance_factor = distance_fog_factor(length(relative_end));
        let noise = fog_noise(relative_mid.xz);
        let noise_factor = mix(0.80, 1.22, noise);
        let optical_depth = max(fog.fog_density, 0.0)
            * distance_factor
            * path_height_density
            * noise_factor;

        fog_amount = 1.0 - exp(-optical_depth);
        fog_amount = saturate_f(fog_amount);
    }

    return vec4<f32>(fog_colour * fog_amount, 1.0 - fog_amount);
}
