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

fn depth_weight(
    cloud_pixel: vec2<i32>,
    center_depth: f32,
    cloud_dimensions: vec2<i32>,
    scene_dimensions: vec2<i32>
) -> f32 {
    let cloud_uv = (vec2<f32>(cloud_pixel) + vec2<f32>(0.5)) / vec2<f32>(cloud_dimensions);
    let scene_pixel = clamp(
        vec2<i32>(cloud_uv * vec2<f32>(scene_dimensions)),
        vec2<i32>(0),
        scene_dimensions - vec2<i32>(1)
    );
    let sample_depth = textureLoad(linear_depth_full, scene_pixel, 0).r;
    let depth_scale = max(6.0, min(center_depth, sample_depth) * 0.02);
    return exp(-abs(sample_depth - center_depth) / depth_scale);
}

fn is_cloud_sample_visible(uv: vec2<f32>, ray_distance: f32, scene_depth: f32) -> bool {
    if ray_distance <= 0.001 {
        return true;
    }
    let cloud_depth = cloud_view_depth(uv, ray_distance);
    let tolerance = max(5.0, cloud_depth * 0.001);
    return scene_depth + tolerance >= cloud_depth;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    if cloud.enabled == 0u {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let cloud_dimensions = vec2<i32>(textureDimensions(cloud_image, 0u));
    let scene_dimensions = vec2<i32>(textureDimensions(linear_depth_full, 0u));
    let scene_pixel = clamp(
        vec2<i32>(input.position.xy),
        vec2<i32>(0),
        scene_dimensions - vec2<i32>(1)
    );
    let scene_depth = textureLoad(linear_depth_full, scene_pixel, 0).r;

    let cloud_position = input.uv * vec2<f32>(cloud_dimensions) - vec2<f32>(0.5);
    let base = vec2<i32>(floor(cloud_position));
    let fraction = fract(cloud_position);
    let cloud_max = cloud_dimensions - vec2<i32>(1);

    let p00 = clamp(base, vec2<i32>(0), cloud_max);
    let p10 = clamp(base + vec2<i32>(1, 0), vec2<i32>(0), cloud_max);
    let p01 = clamp(base + vec2<i32>(0, 1), vec2<i32>(0), cloud_max);
    let p11 = clamp(base + vec2<i32>(1, 1), vec2<i32>(0), cloud_max);

    let c00 = textureLoad(cloud_image, p00, 0);
    let c10 = textureLoad(cloud_image, p10, 0);
    let c01 = textureLoad(cloud_image, p01, 0);
    let c11 = textureLoad(cloud_image, p11, 0);

    let d00 = textureLoad(cloud_depth_image, p00, 0).r;
    let d10 = textureLoad(cloud_depth_image, p10, 0).r;
    let d01 = textureLoad(cloud_depth_image, p01, 0).r;
    let d11 = textureLoad(cloud_depth_image, p11, 0).r;

    let uv00 = (vec2<f32>(p00) + vec2<f32>(0.5)) / vec2<f32>(cloud_dimensions);
    let uv10 = (vec2<f32>(p10) + vec2<f32>(0.5)) / vec2<f32>(cloud_dimensions);
    let uv01 = (vec2<f32>(p01) + vec2<f32>(0.5)) / vec2<f32>(cloud_dimensions);
    let uv11 = (vec2<f32>(p11) + vec2<f32>(0.5)) / vec2<f32>(cloud_dimensions);

    let visible00 = is_cloud_sample_visible(uv00, d00, scene_depth);
    let visible10 = is_cloud_sample_visible(uv10, d10, scene_depth);
    let visible01 = is_cloud_sample_visible(uv01, d01, scene_depth);
    let visible11 = is_cloud_sample_visible(uv11, d11, scene_depth);

    let w00 = (1.0 - fraction.x) * (1.0 - fraction.y)
        * depth_weight(p00, scene_depth, cloud_dimensions, scene_dimensions)
        * select(0.0, 1.0, visible00);
    let w10 = fraction.x * (1.0 - fraction.y)
        * depth_weight(p10, scene_depth, cloud_dimensions, scene_dimensions)
        * select(0.0, 1.0, visible10);
    let w01 = (1.0 - fraction.x) * fraction.y
        * depth_weight(p01, scene_depth, cloud_dimensions, scene_dimensions)
        * select(0.0, 1.0, visible01);
    let w11 = fraction.x * fraction.y
        * depth_weight(p11, scene_depth, cloud_dimensions, scene_dimensions)
        * select(0.0, 1.0, visible11);

    let total_weight = w00 + w10 + w01 + w11;
    if total_weight > 0.00001 {
        return (c00 * w00 + c10 * w10 + c01 * w01 + c11 * w11) / total_weight;
    }

    let nearest = clamp(vec2<i32>(round(cloud_position)), vec2<i32>(0), cloud_max);
    let nearest_uv = (vec2<f32>(nearest) + vec2<f32>(0.5)) / vec2<f32>(cloud_dimensions);
    let nearest_depth = textureLoad(cloud_depth_image, nearest, 0).r;
    if is_cloud_sample_visible(nearest_uv, nearest_depth, scene_depth) {
        return textureLoad(cloud_image, nearest, 0);
    }

    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
