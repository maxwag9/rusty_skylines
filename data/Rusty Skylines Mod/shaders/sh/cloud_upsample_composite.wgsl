#include "includes/uniforms.wgsl"

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
var cloud_image: texture_2d<f32>;

@group(0) @binding(3)
var cloud_depth_image: texture_2d<f32>;

@group(0) @binding(4)
var linear_depth_full: texture_2d<f32>;

@group(1) @binding(0)
var<uniform> camera: Uniforms;

@group(1) @binding(1)
var<uniform> cloud: CloudUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
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

fn reconstruct_view_direction(uv: vec2<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let view_h = camera.inv_proj * vec4<f32>(ndc, 1.0, 1.0);
    return normalize(view_h.xyz / view_h.w);
}

fn cloud_view_depth(uv: vec2<f32>, ray_distance: f32) -> f32 {
    if ray_distance <= 0.001 {
        return 1.0e30;
    }
    let view_direction = reconstruct_view_direction(uv);
    return ray_distance * max(-view_direction.z, 0.0);
}

fn is_far_scene_depth(depth: f32) -> bool {
    return depth >= 1.0e19;
}

fn scene_depth_weight(center_depth: f32, sample_depth: f32) -> f32 {
    let center_is_far = is_far_scene_depth(center_depth);
    let sample_is_far = is_far_scene_depth(sample_depth);
    if center_is_far && sample_is_far {
        return 1.0;
    }
    if center_is_far != sample_is_far {
        return 0.08;
    }
    let depth_scale = max(6.0, min(center_depth, sample_depth) * 0.0025);
    return exp(-abs(sample_depth - center_depth) / depth_scale);
}

fn cloud_visibility_weight(
    uv: vec2<f32>,
    ray_distance: f32,
    transmittance: f32,
    scene_depth: f32,
    pixel_footprint_fraction: f32
) -> f32 {
    let opacity = clamp(1.0 - transmittance, 0.0, 1.0);
    let confidence = smoothstep(0.015, 0.20, opacity);
    if confidence <= 0.001 || ray_distance <= 0.001 || is_far_scene_depth(scene_depth) {
        return 1.0;
    }

    let cloud_depth = cloud_view_depth(uv, ray_distance);
    let tolerance = max(8.0, cloud_depth * pixel_footprint_fraction);
    let signed_separation = scene_depth - cloud_depth;
    let visibility = smoothstep(-tolerance, tolerance, signed_separation);
    return mix(1.0, visibility, confidence);
}

fn scene_depth_for_cloud_pixel(
    cloud_pixel: vec2<i32>,
    cloud_dimensions: vec2<i32>,
    scene_dimensions: vec2<i32>
) -> f32 {
    let cloud_uv = (vec2<f32>(cloud_pixel) + vec2<f32>(0.5)) / vec2<f32>(cloud_dimensions);
    let scene_pixel = clamp(
        vec2<i32>(cloud_uv * vec2<f32>(scene_dimensions)),
        vec2<i32>(0),
        scene_dimensions - vec2<i32>(1)
    );
    return textureLoad(linear_depth_full, scene_pixel, 0).r;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    if cloud.enabled == 0u {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    if cloud.padding.x != 0u {
        return textureSampleLevel(cloud_image, cloud_sampler, input.uv, 0.0);
    }

    let cloud_dimensions = vec2<i32>(textureDimensions(cloud_image, 0));
    let scene_dimensions = vec2<i32>(textureDimensions(linear_depth_full, 0));
    let scene_pixel = clamp(vec2<i32>(input.position.xy), vec2<i32>(0), scene_dimensions - vec2<i32>(1));
    let target_scene_depth = textureLoad(linear_depth_full, scene_pixel, 0).r;

    let cloud_position = input.uv * vec2<f32>(cloud_dimensions) - vec2<f32>(0.5);
    let base = vec2<i32>(floor(cloud_position));
    let downsample_scale = max(
        f32(scene_dimensions.x) / f32(cloud_dimensions.x),
        f32(scene_dimensions.y) / f32(cloud_dimensions.y)
    );
    var filter_radius: i32 = 1;
    if downsample_scale >= 3.0 {
        filter_radius = 2;
    }
    let spatial_sigma = max(0.75, downsample_scale * 0.42);
    let pixel_footprint_fraction = max(0.0015, 2.0 * downsample_scale / f32(scene_dimensions.x));

    var filtered = vec4<f32>(0.0);
    var total_weight = 0.0;

    for (var oy: i32 = -2; oy <= 2; oy = oy + 1) {
        for (var ox: i32 = -2; ox <= 2; ox = ox + 1) {
            if abs(ox) > filter_radius || abs(oy) > filter_radius {
                continue;
            }

            let tap = base + vec2<i32>(ox, oy);
            if any(tap < vec2<i32>(0)) || any(tap >= cloud_dimensions) {
                continue;
            }

            let cloud_value = textureLoad(cloud_image, tap, 0);
            let ray_distance = textureLoad(cloud_depth_image, tap, 0).r;
            let tap_uv = (vec2<f32>(tap) + vec2<f32>(0.5)) / vec2<f32>(cloud_dimensions);
            let tap_scene_depth = scene_depth_for_cloud_pixel(tap, cloud_dimensions, scene_dimensions);

            let delta = vec2<f32>(tap) - cloud_position;
            let spatial_weight = exp(-dot(delta, delta) / (2.0 * spatial_sigma * spatial_sigma));
            let bilateral = scene_depth_weight(target_scene_depth, tap_scene_depth);
            let bilateral_weight = mix(0.12, 1.0, bilateral);
            let visibility_weight = cloud_visibility_weight(
                tap_uv,
                ray_distance,
                cloud_value.a,
                target_scene_depth,
                pixel_footprint_fraction
            );
            let weight = spatial_weight * bilateral_weight * visibility_weight;

            filtered += cloud_value * weight;
            total_weight += weight;

        }
    }

    if total_weight > 0.00001 {
        return filtered / total_weight;
    }

    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
