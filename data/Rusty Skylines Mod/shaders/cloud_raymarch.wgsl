#include "includes/uniforms.wgsl"
#include "includes/cloud_geometry.wgsl"

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

@group(0) @binding(0)
var cloud_sampler: sampler;

@group(0) @binding(2)
var weather_map: texture_2d<f32>;

@group(0) @binding(3)
var shape_noise: texture_3d<f32>;

@group(0) @binding(4)
var detail_noise: texture_3d<f32>;

@group(0) @binding(5)
var blue_noise_texture: texture_2d<f32>;

@group(0) @binding(6)
var terrain_height_map: texture_2d<f32>;

@group(1) @binding(0)
var<uniform> camera: Uniforms;

@group(1) @binding(1)
var<uniform> cloud: CloudUniforms;


const MAX_PRIMARY_STEPS: u32 = 640u;

const MIN_PRIMARY_STEP_SIZE: f32 = 128.0;
const PRIMARY_TARGET_STEP_SIZE: f32 = 160.0;
const MAX_PRIMARY_STEP_SIZE: f32 = 256.0;
const PRIMARY_REFINE_STEP_THRESHOLD: f32 = 192.0;
const CLOUD_SUN_TRANSMITTANCE_STEPS: u32 = 16u;
const CLOUD_MAX_SUN_PATH: f32 = 20000.0;
const CLOUD_SINGLE_SCATTERING_ALBEDO: f32 = 0.999;
const WEATHER_PERIOD_CHUNKS: i32 = 512;
const SHAPE_PERIOD_CHUNKS: i32 = 512;
const DETAIL_PERIOD_CHUNKS: i32 = 16;
const TERRAIN_MAP_WORLD_SIZE: f32 = 32768.0;
const MIN_WORLD_CLOUD_HEIGHT: f32 = 0.0;
const MAX_WORLD_CLOUD_HEIGHT: f32 = 16000.0;
const MAX_CLOUD_DISTANCE: f32 = 60000.0;
const CLOUD_EXTINCTION: f32 = 0.010;
const CLOUD_SUN_INTENSITY: f32 = 0.5;
const MAIN_SHAPE_XZ_SCALE: f32 = 4.0;
const CIRRUS_SHAPE_XZ_SCALE: f32 = 4.0;
const PI: f32 = 3.14159265358979323846;
const TAU: f32 = 6.28318530717958647692;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct CloudFragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) depth: vec4<f32>,
};

struct CloudLayerProfile {
    base_agl: f32,
    top_agl: f32,
    density_scale: f32,
    erosion_strength: f32,
    shape_power: f32,
    sheet_bias: f32,
    vertical_frequency: f32,
};

struct CloudDensitySample {
    density: f32,
    height_fraction: f32,
    detail_uv: vec3<f32>,
    erosion_strength: f32,
    storm_density: f32,
    storm_height_fraction: f32,
    storm_detail_uv: vec3<f32>,
    storm_erosion_strength: f32,
    cirrus_density: f32,
    cirrus_height_fraction: f32,
    cirrus_detail_uv: vec3<f32>,
    cirrus_erosion_strength: f32,
};

struct CloudLightResult {
    optical_depth: f32,
};

struct CloudResult {
    radiance: vec3<f32>,
    transmittance: f32,
    representative_distance: f32,
    debug_value: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let x = f32(i32(vertex_index & 1u) * 2);
    let y = f32(i32(vertex_index >> 1u) * 2);

    var output: VertexOutput;
    output.uv = vec2<f32>(x, 1.0 - y);
    output.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

fn saturate(value: f32) -> f32 {
    return clamp(value, 0.0, 1.0);
}

fn remap_saturate(value: f32, old_min: f32, old_max: f32) -> f32 {
    return saturate((value - old_min) / max(old_max - old_min, 0.0001));
}

fn cloud_wind_offset_for(period_world: f32, wind: vec2<f32>) -> vec2<f32> {
    let safe_period = max(period_world, 1.0);
    let period_x = safe_period / max(abs(wind.x), 0.001);
    let period_z = safe_period / max(abs(wind.y), 0.001);
    let time_x = cloud.time - floor(cloud.time / period_x) * period_x;
    let time_z = cloud.time - floor(cloud.time / period_z) * period_z;
    return vec2<f32>(wind.x * time_x, wind.y * time_z);
}

fn cloud_wind_offset(period_world: f32) -> vec2<f32> {
    return cloud_wind_offset_for(period_world, cloud.wind);
}

fn cloud_world_position_period(
    point_relative_to_camera: vec3<f32>,
    wind_offset: vec2<f32>,
    period_chunks: i32
) -> vec3<f32> {
    let wrapped_chunk = vec2<f32>(
        f32(positive_mod_i32(camera.camera_chunk.x, period_chunks)),
        f32(positive_mod_i32(camera.camera_chunk.y, period_chunks))
    );
    let world_xz = wrapped_chunk * camera.chunk_size
        + camera.camera_local.xz
        + point_relative_to_camera.xz
        + wind_offset;
    return vec3<f32>(world_xz.x, cloud.camera_height + point_relative_to_camera.y, world_xz.y);
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
    let height = textureSampleLevel(terrain_height_map, cloud_sampler, sample_uv, 0.0).r;

    let edge_distance = min(
        min(uv.x, 1.0 - uv.x),
        min(uv.y, 1.0 - uv.y)
    );
    let edge_fade = smoothstep(-0.04, 0.0, edge_distance);
    return vec2<f32>(height, edge_fade);
}

fn wrapped_cloud_phase(rate: f32) -> f32 {
    let safe_rate = max(abs(rate), 0.000001);
    let cycle = TAU / safe_rate;
    let wrapped_time = cloud.time - floor(cloud.time / cycle) * cycle;
    return wrapped_time * rate;
}

fn animate_weather_uv(uv: vec2<f32>) -> vec2<f32> {
    let p = uv * TAU;
    let slow_phase = wrapped_cloud_phase(0.000020);
    let fast_phase = wrapped_cloud_phase(0.000070);

    let broad_warp = vec2<f32>(
        sin(p.y * 2.0 + p.x * 0.7 + slow_phase),
        cos(p.x * 2.3 - p.y * 0.6 - slow_phase * 0.83)
    );
    let curl_warp = vec2<f32>(
        sin((p.x + p.y) * 4.1 + fast_phase),
        cos((p.y - p.x) * 3.7 - fast_phase * 1.17)
    );
    return fract(uv + broad_warp * 0.014 + curl_warp * 0.005);
}

fn animate_volume_uv(uv: vec3<f32>, phase_scale: f32, amplitude: f32) -> vec3<f32> {
    let p = uv * TAU;
    let phase = wrapped_cloud_phase(phase_scale);

    let warp_a = vec3<f32>(
        sin(p.y * 1.7 + p.z * 2.3 + phase),
        cos(p.z * 1.9 - p.x * 1.5 - phase * 0.79),
        sin(p.x * 1.6 + p.y * 2.1 + phase * 0.63)
    );
    let warp_b = vec3<f32>(
        cos((p.y + p.z) * 4.4 - phase * 1.31),
        sin((p.x - p.z) * 3.8 + phase * 1.07),
        cos((p.x + p.y) * 4.7 - phase * 0.91)
    );

    return fract(uv + warp_a * amplitude + warp_b * (amplitude * 0.32));
}

fn empty_cloud_density_sample() -> CloudDensitySample {
    return CloudDensitySample(
        0.0, 0.0, vec3<f32>(0.0), 0.0,
        0.0, 0.0, vec3<f32>(0.0), 0.0,
        0.0, 0.0, vec3<f32>(0.0), 0.0
    );
}

fn cloud_profile(index: u32) -> CloudLayerProfile {
    switch (index) {
        case 0u: {
            return CloudLayerProfile(120.0, 750.0, 0.66, 0.24, 0.86, 0.82, 42.0);
        }
        case 1u: {
            return CloudLayerProfile(320.0, 1900.0, 0.78, 0.34, 0.88, 0.42, 30.0);
        }
        case 2u: {
            return CloudLayerProfile(650.0, 2800.0, 0.90, 0.42, 0.78, 0.03, 24.0);
        }
        case 3u: {
            return CloudLayerProfile(2200.0, 4300.0, 0.62, 0.48, 0.92, 0.18, 22.0);
        }
        case 4u: {
            return CloudLayerProfile(3000.0, 6500.0, 0.46, 0.24, 0.72, 0.88, 14.0);
        }
        default: {
            return CloudLayerProfile(1400.0, 7600.0, 0.76, 0.26, 0.68, 0.92, 10.0);
        }
    }
}

fn cloud_type_profile_coordinate(cloud_type: f32) -> f32 {
    let t = saturate(cloud_type);
    if t < 0.15 {
        return smoothstep(0.0, 0.15, t) * 0.12;
    }
    if t < 0.48 {
        return mix(0.12, 1.0, smoothstep(0.15, 0.48, t));
    }
    if t < 0.73 {
        return mix(1.0, 2.0, smoothstep(0.48, 0.73, t));
    }
    if t < 0.82 {
        return mix(2.0, 3.0, smoothstep(0.73, 0.82, t));
    }
    if t < 0.91 {
        return mix(3.0, 4.0, smoothstep(0.82, 0.91, t));
    }
    return mix(4.0, 5.0, smoothstep(0.91, 0.99, t));
}

fn cloud_vertical_profile(index: u32, h: f32) -> f32 {
    switch (index) {
        case 0u: {
            return smoothstep(0.0, 0.025, h) * (1.0 - smoothstep(0.82, 1.0, h));
        }
        case 1u: {
            return smoothstep(0.025, 0.10, h) * (1.0 - smoothstep(0.76, 0.98, h));
        }
        case 2u: {
            return smoothstep(0.0, 0.15, h) * (1.0 - smoothstep(0.66, 0.98, h));
        }
        case 3u: {
            return smoothstep(0.05, 0.20, h) * (1.0 - smoothstep(0.72, 0.98, h));
        }
        case 4u: {
            return smoothstep(0.0, 0.025, h) * (1.0 - smoothstep(0.92, 1.0, h));
        }
        default: {
            return smoothstep(0.0, 0.045, h) * (1.0 - smoothstep(0.94, 1.0, h));
        }
    }
}

fn evaluate_main_profile(
    index: u32,
    world_height: f32,
    terrain_height: f32,
    terrain_lift: f32,
    shape_density: f32,
    humidity: f32,
    weather_coverage: f32
) -> vec3<f32> {
    let profile = cloud_profile(index);
    let base_height = terrain_height + terrain_lift + profile.base_agl;
    let top_height = terrain_height + profile.top_agl + terrain_lift;
    let thickness = max(top_height - base_height, 1.0);
    let h = (world_height - base_height) / thickness;

    if h <= 0.0 || h >= 1.0 {
        return vec3<f32>(0.0, saturate(h), profile.erosion_strength);
    }

    let vertical = cloud_vertical_profile(index, h);
    let round_shape = pow(saturate(shape_density), profile.shape_power);
    let sheet_shape = pow(saturate(shape_density), max(profile.shape_power * 0.52, 0.35));
    let type_shape = mix(round_shape, sheet_shape, profile.sheet_bias);
    let humidity_factor = mix(0.72, 1.14, saturate(humidity));
    let density = type_shape * vertical * profile.density_scale
        * humidity_factor * weather_coverage;

    return vec3<f32>(saturate(density), saturate(h), profile.erosion_strength);
}

fn perlin_worley_shape(shape: vec4<f32>) -> f32 {
    let low_frequency_worley = shape.g * 0.625
        + shape.b * 0.25
        + shape.a * 0.125;
    let perlin_signed = shape.r * 2.0 - 1.0;
    let worley_lower_bound = -(1.0 - low_frequency_worley);
    let remapped = saturate(
        (perlin_signed - worley_lower_bound)
            / max(1.0 - worley_lower_bound, 0.001)
    );
    return mix(remapped, saturate(1.0 - low_frequency_worley), 0.28);
}

fn cloud_density_cheap(
    point_relative_to_camera: vec3<f32>,
    sample_step: f32,
    terrain_enabled: bool
) -> CloudDensitySample {
    let world_height = cloud.camera_height + point_relative_to_camera.y;
    if world_height < MIN_WORLD_CLOUD_HEIGHT || world_height > MAX_WORLD_CLOUD_HEIGHT {
        return empty_cloud_density_sample();
    }

    let weather_period = camera.chunk_size * f32(WEATHER_PERIOD_CHUNKS);
    let weather_uv_base = stable_cloud_weather_uv(
        point_relative_to_camera.xz + cloud_wind_offset(weather_period),
        camera.camera_chunk,
        camera.camera_local.xz,
        camera.chunk_size,
        WEATHER_PERIOD_CHUNKS
    );
    let weather_uv = animate_weather_uv(weather_uv_base);

    let weather_texel_world = weather_period / f32(textureDimensions(weather_map, 0).x);
    let weather_lod = clamp(
        log2(max(sample_step / weather_texel_world, 1.0)),
        0.0,
        2.0
    );
    let weather = textureSampleLevel(weather_map, cloud_sampler, weather_uv, weather_lod);

    let cloud_type = saturate(weather.g);
    let humidity = saturate(weather.b);
    let local_coverage = saturate(cloud.coverage * mix(0.78, 1.16, humidity));
    let coverage_threshold = 1.0 - local_coverage;
    let weather_coverage = smoothstep(
        coverage_threshold - 0.085,
        coverage_threshold + 0.085,
        weather.r
    );

    var ground_height = 0.0;
    var terrain_lift = 0.0;
    var terrain_influence = 0.0;
    if terrain_enabled {
        let terrain = sample_terrain_height(point_relative_to_camera.xz);
        ground_height = terrain.x;
        terrain_influence = terrain.y;
        let full_terrain_lift = clamp(max(terrain.x - 180.0, 0.0) * 0.24, 0.0, 1000.0);
        terrain_lift = full_terrain_lift * terrain.y;
    }
    let agl = world_height - ground_height;

    let shape_period_world = camera.chunk_size * f32(SHAPE_PERIOD_CHUNKS);
    let shape_world = cloud_world_position_period(
        point_relative_to_camera,
        cloud_wind_offset(shape_period_world),
        SHAPE_PERIOD_CHUNKS
    );

    let profile_position = cloud_type_profile_coordinate(cloud_type);
    let profile_a = min(u32(floor(profile_position)), 5u);
    let profile_b = min(profile_a + 1u, 5u);
    let profile_blend = smoothstep(0.0, 1.0, fract(profile_position));
    let profile_a_data = cloud_profile(profile_a);
    let profile_b_data = cloud_profile(profile_b);
    let vertical_frequency = mix(
        profile_a_data.vertical_frequency,
        profile_b_data.vertical_frequency,
        profile_blend
    );

    let minimum_main_agl = min(profile_a_data.base_agl, profile_b_data.base_agl) + terrain_lift;
    let maximum_main_agl = max(profile_a_data.top_agl, profile_b_data.top_agl) + terrain_lift;
    let main_possible = weather_coverage > 0.001
        && agl >= minimum_main_agl - 20.0
        && agl <= maximum_main_agl + 20.0;

    var main_density = 0.0;
    var main_height_fraction = 0.0;
    var main_erosion = mix(profile_a_data.erosion_strength, profile_b_data.erosion_strength, profile_blend);

    if main_possible {
        let shape_uv_base = vec3<f32>(
            shape_world.x / shape_period_world * MAIN_SHAPE_XZ_SCALE,
            world_height / shape_period_world * vertical_frequency,
            shape_world.z / shape_period_world * MAIN_SHAPE_XZ_SCALE
        );
        let shape_uv = animate_volume_uv(shape_uv_base, 0.000070, 0.020);
        let shape_texel_world = shape_period_world
            / f32(textureDimensions(shape_noise, 0).x)
            / MAIN_SHAPE_XZ_SCALE;
        let shape_lod = clamp(
            log2(max(sample_step / shape_texel_world, 1.0)),
            0.0,
            4.0
        );
        let shape = textureSampleLevel(shape_noise, cloud_sampler, shape_uv, shape_lod);
        let shape_field = perlin_worley_shape(shape);
        let shape_coverage = clamp(0.42 + cloud.coverage * 0.58, 0.0, 1.0);
        let blended_base_agl = mix(profile_a_data.base_agl, profile_b_data.base_agl, profile_blend);
        let blended_top_agl = mix(profile_a_data.top_agl, profile_b_data.top_agl, profile_blend);
        let profile_height = saturate(
            (agl - blended_base_agl - terrain_lift)
                / max(blended_top_agl - blended_base_agl, 1.0)
        );
        let cumuliform_strength = smoothstep(0.85, 1.55, profile_position)
            * (1.0 - smoothstep(2.35, 2.90, profile_position));
        let stratiform_strength = mix(
            profile_a_data.sheet_bias,
            profile_b_data.sheet_bias,
            profile_blend
        );
        let shape_threshold = 1.0 - shape_coverage
            + cumuliform_strength * (0.42 - profile_height) * 0.13
            + stratiform_strength * (profile_height - 0.5) * 0.035;
        let covered_shape = saturate(
            (shape_field - shape_threshold) / max(shape_coverage, 0.001)
        );

        let main_a = evaluate_main_profile(
            profile_a, world_height, ground_height, terrain_lift,
            covered_shape, humidity, weather_coverage
        );
        let main_b = evaluate_main_profile(
            profile_b, world_height, ground_height, terrain_lift,
            covered_shape, humidity, weather_coverage
        );
        let orographic_influence = smoothstep(300.0, 2400.0, ground_height)
            * smoothstep(0.18, 0.62, humidity)
            * terrain_influence;
        let orographic_density_boost = 1.0 + orographic_influence * 0.22;
        main_density = saturate(mix(main_a.x, main_b.x, profile_blend) * orographic_density_boost);
        main_height_fraction = mix(main_a.y, main_b.y, profile_blend);
        main_erosion = mix(main_a.z, main_b.z, profile_blend);
    }

    let detail_period_world = camera.chunk_size * f32(DETAIL_PERIOD_CHUNKS);
    let detail_uv_main_base = vec3<f32>(
        shape_world.x / detail_period_world,
        world_height / detail_period_world,
        shape_world.z / detail_period_world
    );
    let detail_uv_main = animate_volume_uv(detail_uv_main_base, 0.00025, 0.022);

    var storm_density = 0.0;
    var storm_height_fraction = 0.0;
    var storm_detail_uv = vec3<f32>(0.0);
    var storm_erosion = 0.0;

    let storm_probability = smoothstep(0.88, 0.985, cloud_type)
        * smoothstep(0.58, 0.82, humidity)
        * smoothstep(0.48, 0.76, weather.r)
        * smoothstep(0.12, 0.42, cloud.coverage);
    let storm_base_agl = 650.0 + humidity * 850.0 + terrain_lift;
    let storm_top_agl = min(11200.0 + humidity * 1700.0 + terrain_lift, 14000.0);
    let storm_h = (agl - storm_base_agl) / max(storm_top_agl - storm_base_agl, 1.0);

    if storm_probability > 0.001 && storm_h > 0.0 && storm_h < 1.0 {
        let storm_uv_base = vec3<f32>(
            shape_world.x / shape_period_world * 2.0,
            world_height / shape_period_world * 8.0,
            shape_world.z / shape_period_world * 2.0
        );
        let storm_uv = animate_volume_uv(storm_uv_base, 0.000060, 0.018);
        let storm_shape_lod = clamp(
            log2(max(sample_step / (shape_period_world / 128.0 / 2.0), 1.0)),
            0.0,
            4.0
        );
        let storm_shape = textureSampleLevel(shape_noise, cloud_sampler, storm_uv, storm_shape_lod);
        let storm_field = perlin_worley_shape(storm_shape);
        let storm_covered_shape = smoothstep(0.40, 0.74, storm_field);
        let storm_vertical = smoothstep(0.0, 0.055, storm_h)
            * (1.0 - smoothstep(0.91, 1.0, storm_h));
        let anvil = smoothstep(0.68, 0.91, storm_h);
        let storm_roundness = mix(
            pow(storm_covered_shape, 0.82),
            pow(storm_covered_shape, 0.48),
            anvil * 0.72
        );
        storm_density = saturate(storm_roundness * storm_vertical * storm_probability * 1.05);
        storm_height_fraction = saturate(storm_h);
        storm_detail_uv = animate_volume_uv(
            vec3<f32>(shape_world.x / detail_period_world, world_height / detail_period_world, shape_world.z / detail_period_world),
            0.00022,
            0.020
        );
        storm_erosion = 0.67;
    }

    var cirrus_density = 0.0;
    var cirrus_height_fraction = 0.0;
    var cirrus_detail_uv = vec3<f32>(0.0);
    var cirrus_erosion = 0.0;

    let cirrus_threshold = mix(0.78, 0.36, cloud.coverage);
    let cirrus_coverage = smoothstep(
        cirrus_threshold - 0.14,
        cirrus_threshold + 0.14,
        weather.a
    );
    let cirrus_base_agl = 6800.0 + humidity * 1500.0;
    let cirrus_top_agl = min(cirrus_base_agl + 1800.0 + weather.g * 1700.0, 13200.0);
    let cirrus_h = (agl - cirrus_base_agl) / max(cirrus_top_agl - cirrus_base_agl, 1.0);

    if cirrus_coverage > 0.001 && cirrus_h > 0.0 && cirrus_h < 1.0 {
        let cirrus_period_world = camera.chunk_size * f32(WEATHER_PERIOD_CHUNKS);
        let high_wind = vec2<f32>(
            cloud.wind.x * 1.65 + cloud.wind.y * 0.25,
            cloud.wind.y * 1.65 - cloud.wind.x * 0.25
        );
        let cirrus_world = cloud_world_position_period(
            point_relative_to_camera,
            cloud_wind_offset_for(cirrus_period_world, high_wind),
            WEATHER_PERIOD_CHUNKS
        );
        let cirrus_uv_base = vec3<f32>(
            cirrus_world.x / cirrus_period_world * CIRRUS_SHAPE_XZ_SCALE,
            world_height / 2100.0,
            cirrus_world.z / cirrus_period_world * CIRRUS_SHAPE_XZ_SCALE
        );
        let cirrus_uv = animate_volume_uv(cirrus_uv_base, 0.000080, 0.028);
        let cirrus_texel_world = cirrus_period_world
            / f32(textureDimensions(shape_noise, 0).x)
            / CIRRUS_SHAPE_XZ_SCALE;
        let cirrus_lod = clamp(
            log2(max(sample_step / cirrus_texel_world, 1.0)),
            0.0,
            4.0
        );
        let cirrus_shape = textureSampleLevel(shape_noise, cloud_sampler, cirrus_uv, cirrus_lod);
        let cirrus_field = perlin_worley_shape(cirrus_shape);
        let wispy_shape = smoothstep(0.43, 0.73, cirrus_field);
        let thin_veil = smoothstep(0.38, 0.67, cirrus_field);
        let cirrus_style = smoothstep(0.48, 0.88, cloud_type);
        let cirrus_form = mix(wispy_shape, thin_veil, cirrus_style * 0.28);
        let cirrus_vertical = smoothstep(0.0, 0.12, cirrus_h)
            * (1.0 - smoothstep(0.87, 1.0, cirrus_h));
        let humidity_factor = mix(0.55, 1.0, humidity);
        cirrus_density = saturate(
            cirrus_form * cirrus_vertical * cirrus_coverage * humidity_factor * 0.34
        );
        cirrus_height_fraction = saturate(cirrus_h);
        let cirrus_detail_period_xz = cirrus_period_world / CIRRUS_SHAPE_XZ_SCALE;
        cirrus_detail_uv = animate_volume_uv(
            vec3<f32>(
                cirrus_world.x / cirrus_detail_period_xz,
                world_height / 900.0,
                cirrus_world.z / cirrus_detail_period_xz
            ),
            0.00032,
            0.034
        );
        cirrus_erosion = 0.86;
    }

    return CloudDensitySample(
        saturate(main_density),
        main_height_fraction,
        detail_uv_main,
        main_erosion,
        saturate(storm_density),
        storm_height_fraction,
        storm_detail_uv,
        storm_erosion,
        saturate(cirrus_density),
        cirrus_height_fraction,
        cirrus_detail_uv,
        cirrus_erosion
    );
}

fn total_cheap_density(sample: CloudDensitySample) -> f32 {
    return saturate(sample.density + sample.storm_density * 1.15 + sample.cirrus_density * 0.22);
}

fn erode_density_component(
    density: f32,
    detail_uv: vec3<f32>,
    erosion_strength: f32,
    height_fraction: f32,
    sample_step: f32,
    erosion_scale: f32
) -> f32 {
    if density <= 0.001 {
        return 0.0;
    }

    let detail_period_world = camera.chunk_size * f32(DETAIL_PERIOD_CHUNKS);
    let detail_texel_world = detail_period_world / f32(textureDimensions(detail_noise, 0).x);
    let detail_lod = clamp(
        log2(max(sample_step / detail_texel_world, 1.0)),
        0.0,
        3.0
    );
    let detail = textureSampleLevel(detail_noise, cloud_sampler, detail_uv, detail_lod);

    // For the detail generator, R/G/B are Worley frequencies 1/2/4;
    // A is Perlin fBm and is not a Worley distance channel.
    let detail_worley = detail.r * 0.625 + detail.g * 0.25 + detail.b * 0.125;
    let boundary_erosion = mix(
        detail_worley,
        1.0 - detail_worley,
        smoothstep(0.08, 0.62, height_fraction)
    );
    let fine_variation = (detail.a - 0.5) * 0.065;
    let erosion = max(
        boundary_erosion * erosion_strength * clamp(cloud.detail_strength, 0.0, 1.0) * erosion_scale
            + fine_variation * clamp(cloud.detail_strength, 0.0, 1.0),
        0.0
    );
    return smoothstep(0.005, 0.17, density - erosion);
}

fn cloud_density_full_components(sample: CloudDensitySample, sample_step: f32) -> vec3<f32> {
    let main_density = erode_density_component(
        sample.density,
        sample.detail_uv,
        sample.erosion_strength,
        sample.height_fraction,
        sample_step,
        0.24
    );
    let storm_density = erode_density_component(
        sample.storm_density,
        sample.storm_detail_uv,
        sample.storm_erosion_strength,
        sample.storm_height_fraction,
        sample_step,
        0.20
    );
    let cirrus_density = erode_density_component(
        sample.cirrus_density,
        sample.cirrus_detail_uv,
        sample.cirrus_erosion_strength,
        sample.cirrus_height_fraction,
        sample_step,
        0.13
    );
    return vec3<f32>(main_density, storm_density, cirrus_density);
}

fn cloud_height_fraction(sample: CloudDensitySample) -> f32 {
    if sample.cirrus_density >= sample.density && sample.cirrus_density >= sample.storm_density {
        return sample.cirrus_height_fraction;
    }
    if sample.storm_density > sample.density {
        return sample.storm_height_fraction;
    }
    return sample.height_fraction;
}

fn henyey_greenstein(cos_theta: f32, eccentricity: f32) -> f32 {
    let g = clamp(eccentricity, -0.95, 0.95);
    let g2 = g * g;
    let denominator = max(
        1.0 + g2 - 2.0 * g * clamp(cos_theta, -1.0, 1.0),
        0.001
    );
    return (1.0 - g2) / (4.0 * PI * pow(denominator, 1.5));
}


fn cloud_phase_function(cos_theta: f32, phase_scale: f32) -> f32 {
    return henyey_greenstein(cos_theta, 0.65 * phase_scale) * 0.92
        + henyey_greenstein(cos_theta, 0.89 * phase_scale) * 0.08;
}


fn calculate_cloud_sun_energy(
    sun_optical_depth: f32,
    view_transmittance: vec3<f32>,
    view_step_transmittance: vec3<f32>,
    cos_theta: f32,
    sun_color: vec3<f32>,
    sunlight: f32
) -> vec3<f32> {
    let extinction_scales = vec3<f32>(1.0, 0.5, 0.25);
    let energy_scales = vec3<f32>(1.0, 0.5, 0.25);

    let sun_transmittance = exp(-sun_optical_depth * extinction_scales);

    let single_scattering_phase = vec3<f32>(
        cloud_phase_function(cos_theta, 1.0),
        cloud_phase_function(cos_theta, 0.5),
        cloud_phase_function(cos_theta, 0.25)
    );

    let multiple_scattering_phase = vec3<f32>(
        8.0 / (4.0 * PI)
    );

    let effective_phase = single_scattering_phase
        + multiple_scattering_phase;

    let step_alpha = vec3<f32>(1.0) - view_step_transmittance;

    let octave_source = view_transmittance
        * step_alpha
        * sun_transmittance
        * effective_phase
        * energy_scales;

    let source = octave_source.x
        + octave_source.y
        + octave_source.z;

    return sun_color
        * source
        * sunlight
        * CLOUD_SUN_INTENSITY
        * CLOUD_SINGLE_SCATTERING_ALBEDO;
}



fn cloud_sun_light(
    point: vec3<f32>,
    sun_direction: vec3<f32>
) -> CloudLightResult {
    if sun_direction.y <= 0.001 {
        return CloudLightResult(80.0);
    }

    let world_height = cloud.camera_height + point.y;
    let distance_to_top = max(
        (MAX_WORLD_CLOUD_HEIGHT - world_height) / max(sun_direction.y, 0.025),
        0.0
    );
    let light_path_length = min(distance_to_top, CLOUD_MAX_SUN_PATH);

    if light_path_length <= 1.0 {
        return CloudLightResult(0.0);
    }

    let segment_length = light_path_length / f32(CLOUD_SUN_TRANSMITTANCE_STEPS);
    var optical_depth = 0.0;

    for (var i = 0u; i < CLOUD_SUN_TRANSMITTANCE_STEPS; i = i + 1u) {
        let distance_along_light = (f32(i) + 0.5) * segment_length;
        let light_point = point + sun_direction * distance_along_light;
        let coarse_sample = cloud_density_cheap(light_point, segment_length, true);

        if total_cheap_density(coarse_sample) > 0.001 {
            let components = cloud_density_full_components(coarse_sample, segment_length);
            let extinction_density = components.x
                + components.y * 1.15
                + components.z * 0.22;

            optical_depth += extinction_density
                * clamp(cloud.density, 0.0, 4.0)
                * segment_length
                * CLOUD_EXTINCTION;
        }

        if optical_depth >= 32.0 {
            return CloudLightResult(32.0);
        }
    }

    return CloudLightResult(max(optical_depth, 0.0));
}



fn reconstruct_view_direction(uv: vec2<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let view_h = camera.inv_proj * vec4<f32>(ndc, 1.0, 1.0);
    return normalize(view_h.xyz / view_h.w);
}

fn reconstruct_world_direction(view_direction: vec3<f32>) -> vec3<f32> {
    return normalize((camera.inv_view * vec4<f32>(view_direction, 0.0)).xyz);
}




fn march_clouds(ray_direction: vec3<f32>) -> CloudResult {
    let empty_debug = select(
        vec3<f32>(0.0),
        vec3<f32>(1.0),
        cloud.padding.x == 4u
    );
    let empty_result = CloudResult(vec3<f32>(0.0), 1.0, 0.0, empty_debug);

    if cloud.enabled == 0u || cloud.coverage <= 0.001 {
        return empty_result;
    }

    let cloud_bottom = MIN_WORLD_CLOUD_HEIGHT - cloud.camera_height;
    let cloud_top = MAX_WORLD_CLOUD_HEIGHT - cloud.camera_height;

    let interval = intersect_cloud_slab(
        vec3<f32>(0.0),
        ray_direction,
        cloud_bottom,
        cloud_top,
        MAX_CLOUD_DISTANCE
    );

    if !interval.hit {
        return empty_result;
    }

    let ray_length = interval.end_distance - interval.entry_distance;

    if ray_length <= 0.0 {
        return empty_result;
    }

    let target_step_size = clamp(ray_length / 448.0, 24.0, 128.0);
    let medium_step_size = target_step_size * 1.5;
    let max_step_size = min(target_step_size * 1.75, 256.0);
    let refine_step_threshold = target_step_size * 1.5;

    let sun_direction = normalize(camera.sun_direction);
    let sun_altitude = sun_direction.y;
    let sunlight = smoothstep(0.0, 0.18, sun_altitude);
    let solar_blend = smoothstep(-0.05, 0.45, sun_altitude);

    let sunset_color = vec3<f32>(1.20, 0.72, 0.57);
    let daylight_color = vec3<f32>(1.0, 0.97, 0.91);
    let sun_color = mix(sunset_color, daylight_color, solar_blend);

    let daylight = smoothstep(-0.08, 0.20, sun_altitude);
    let twilight = exp(-abs(sun_altitude) * 8.0) * (1.0 - daylight);

    let day_top = vec3<f32>(0.24, 0.29, 0.38);
    let day_bottom = vec3<f32>(0.065, 0.080, 0.105);
    let dusk_top = vec3<f32>(0.10, 0.065, 0.12);
    let dusk_bottom = vec3<f32>(0.025, 0.020, 0.038);
    let night_top = vec3<f32>(0.014, 0.020, 0.036);
    let night_bottom = vec3<f32>(0.004, 0.006, 0.012);

    var ambient_top = mix(night_top, day_top, daylight);
    var ambient_bottom = mix(night_bottom, day_bottom, daylight);

    ambient_top = mix(ambient_top, dusk_top, twilight * 0.70);
    ambient_bottom = mix(ambient_bottom, dusk_bottom, twilight * 0.70);

    let cos_theta = clamp(dot(ray_direction, sun_direction), -1.0, 1.0);

    var transmittance = 1.0;
    var multi_scattering_transmittance = vec3<f32>(1.0);
    var radiance = vec3<f32>(0.0);
    var direct_debug = vec3<f32>(0.0);
    var ambient_debug = vec3<f32>(0.0);
    var max_density = 0.0;
    var representative_distance = 0.0;

    var distance_cursor = interval.entry_distance;
    var empty_step_run = 0u;
    var inside_cloud = false;
    var refine_next_step = false;

    for (var i = 0u; i < MAX_PRIMARY_STEPS; i = i + 1u) {
        if distance_cursor >= interval.end_distance {
            break;
        }

        var nominal_step_size = target_step_size;

        if refine_next_step || inside_cloud {
            nominal_step_size = target_step_size;
        } else if empty_step_run >= 3u {
            nominal_step_size = max_step_size;
        } else if empty_step_run >= 1u {
            nominal_step_size = medium_step_size;
        }

        let segment_length = min(
            nominal_step_size,
            interval.end_distance - distance_cursor
        );

        let distance_along_ray = distance_cursor + segment_length * 0.5;
        let point = ray_direction * distance_along_ray;

        let density_sample = cloud_density_cheap(
            point,
            segment_length,
            true
        );
        let coarse_density = total_cheap_density(density_sample);

        refine_next_step = false;

        if coarse_density > 0.001
            && segment_length > refine_step_threshold {
            refine_next_step = true;
            empty_step_run = 0u;
            continue;
        }

        let component_density = cloud_density_full_components(
            density_sample,
            segment_length
        );

        let main_density = component_density.x;
        let storm_density = component_density.y;
        let cirrus_density = component_density.z;

        let effective_density = main_density
            + storm_density * 1.15
            + cirrus_density * 0.22;

        max_density = max(max_density, saturate(effective_density));

        if effective_density > 0.001 {
            empty_step_run = 0u;
            inside_cloud = true;

            let optical_depth = effective_density
                * clamp(cloud.density, 0.0, 4.0)
                * segment_length
                * CLOUD_EXTINCTION;

            let sample_transmittance = exp(-optical_depth);
            let sample_alpha = 1.0 - sample_transmittance;
            let contribution = transmittance * sample_alpha;

            let extinction_scales = vec3<f32>(1.0, 0.5, 0.25);
            let octave_step_transmittance = exp(
                -optical_depth * extinction_scales
            );

            var sun_energy = vec3<f32>(0.0);

            if sunlight > 0.001 {
                let light = cloud_sun_light(point, sun_direction);

                sun_energy = calculate_cloud_sun_energy(
                    light.optical_depth,
                    multi_scattering_transmittance,
                    octave_step_transmittance,
                    cos_theta,
                    sun_color,
                    sunlight
                );
            }

            let ambient_height = cloud_height_fraction(density_sample);
            let ambient_color = mix(
                ambient_bottom,
                ambient_top,
                ambient_height
            );

            let ambient_light = ambient_color
                * (0.65 + 0.35 * sqrt(max(effective_density, 0.0)));

            radiance += sun_energy + contribution * ambient_light;
            direct_debug += sun_energy;
            ambient_debug += contribution * ambient_light;

            multi_scattering_transmittance *= octave_step_transmittance;

            let next_transmittance = transmittance * sample_transmittance;

            if representative_distance <= 0.0
                && next_transmittance < 0.95 {
                representative_distance = distance_along_ray;
            }

            transmittance = next_transmittance;

            if transmittance < 0.01
                && max(
                    max(
                        multi_scattering_transmittance.x,
                        multi_scattering_transmittance.y
                    ),
                    multi_scattering_transmittance.z
                ) < 0.01 {
                break;
            }
        } else {
            inside_cloud = false;
            empty_step_run = min(empty_step_run + 1u, 8u);
        }

        distance_cursor += segment_length;
    }

    var debug_value = radiance;

    if cloud.padding.x == 1u {
        debug_value = vec3<f32>(max_density);
    } else if cloud.padding.x == 2u {
        debug_value = direct_debug;
    } else if cloud.padding.x == 3u {
        debug_value = ambient_debug;
    } else if cloud.padding.x == 4u {
        debug_value = vec3<f32>(transmittance);
    }

    return CloudResult(
        radiance,
        transmittance,
        representative_distance,
        debug_value
    );
}





@fragment
fn fs_main(input: VertexOutput) -> CloudFragmentOutput {
    let view_direction = reconstruct_view_direction(input.uv);
    let ray_direction = reconstruct_world_direction(view_direction);

    let result = march_clouds(ray_direction);

    var output: CloudFragmentOutput;

    if cloud.padding.x == 0u {
        output.color = vec4<f32>(
            result.radiance,
            result.transmittance
        );
    } else {
        output.color = vec4<f32>(result.debug_value, 0.0);
    }

    output.depth = vec4<f32>(
        result.representative_distance,
        result.transmittance,
        0.0,
        1.0
    );

    return output;
}



fn primary_ray_jitter(
    pixel: vec2<u32>,
    noise_dimensions: vec2<u32>
) -> f32 {
    let frame_offset = vec2<u32>(
        cloud.frame_index * 17u,
        cloud.frame_index * 29u
    );

    let noise_coords = vec2<i32>(
        (pixel + frame_offset) % noise_dimensions
    );

    let noise = textureLoad(
        blue_noise_texture,
        noise_coords,
        0
    ).r;

    return mix(0.1, 0.9, noise);
}
