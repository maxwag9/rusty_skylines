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

struct CloudTemporalUniforms {
    previous_time: f32,
    history_valid: u32,
    padding: vec2<u32>,
};

@group(0) @binding(0)
var cloud_sampler: sampler;

@group(0) @binding(2)
var current_trace: texture_2d<f32>;

@group(0) @binding(3)
var previous_history: texture_2d<f32>;

@group(0) @binding(4)
var current_trace_depth: texture_2d<f32>;

@group(1) @binding(0)
var<uniform> camera: Uniforms;

@group(1) @binding(1)
var<uniform> cloud: CloudUniforms;

@group(1) @binding(2)
var<uniform> temporal: CloudTemporalUniforms;

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

fn reconstruct_world_direction(view_direction: vec3<f32>) -> vec3<f32> {
    return normalize((camera.inv_view * vec4<f32>(view_direction, 0.0)).xyz);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions = vec2<i32>(textureDimensions(current_trace, 0));
    let pixel = clamp(
        vec2<i32>(input.position.xy),
        vec2<i32>(0),
        dimensions - vec2<i32>(1)
    );

    let current_value = textureLoad(current_trace, pixel, 0);
    let current_distance = textureLoad(current_trace_depth, pixel, 0).r;

    if cloud.enabled == 0u
        || cloud.padding.x != 0u
        || temporal.history_valid == 0u
        || current_distance <= 0.001 {
        return current_value;
    }

    let view_direction = reconstruct_view_direction(input.uv);
    let world_direction = reconstruct_world_direction(view_direction);
    let current_point = camera.camera_local + world_direction * current_distance;

    let wind_delta = cloud.wind * (cloud.time - temporal.previous_time);
    let previous_point = current_point + vec3<f32>(
        wind_delta.x,
        0.0,
        wind_delta.y
    );

    let chunk_delta = camera.camera_chunk - camera.prev_camera_chunk;

    let previous_render_position = vec3<f32>(
        f32(chunk_delta.x) * camera.chunk_size
            + previous_point.x - camera.prev_camera_local.x,
        previous_point.y - camera.prev_camera_local.y,
        f32(chunk_delta.y) * camera.chunk_size
            + previous_point.z - camera.prev_camera_local.z
    );

    let previous_clip = camera.prev_view_proj
        * vec4<f32>(previous_render_position, 1.0);

    if previous_clip.w <= 0.0 {
        return current_value;
    }

    let previous_ndc = previous_clip.xy / previous_clip.w;

    let previous_uv = vec2<f32>(
        previous_ndc.x * 0.5 + 0.5,
        0.5 - previous_ndc.y * 0.5
    );

    if any(previous_uv < vec2<f32>(0.0))
        || any(previous_uv > vec2<f32>(1.0)) {
        return current_value;
    }

    let history_value = textureSampleLevel(
        previous_history,
        cloud_sampler,
        previous_uv,
        0.0
    );

    var neighbourhood_min = vec4<f32>(3.402823e+38);
    var neighbourhood_max = vec4<f32>(-3.402823e+38);

    for (var oy = -1; oy <= 1; oy = oy + 1) {
        for (var ox = -1; ox <= 1; ox = ox + 1) {
            let sample_pixel = clamp(
                pixel + vec2<i32>(ox, oy),
                vec2<i32>(0),
                dimensions - vec2<i32>(1)
            );

            let sample_value = textureLoad(current_trace, sample_pixel, 0);

            neighbourhood_min = min(neighbourhood_min, sample_value);
            neighbourhood_max = max(neighbourhood_max, sample_value);
        }
    }

    let opacity_mismatch = abs(history_value.a - current_value.a);

    let radiance_scale = max(
        max(length(current_value.rgb), length(history_value.rgb)),
        0.15
    );

    let radiance_mismatch = length(
        history_value.rgb - current_value.rgb
    ) / radiance_scale;

    if opacity_mismatch > 0.72 || radiance_mismatch > 1.75 {
        return current_value;
    }

    let neighbourhood_span = neighbourhood_max - neighbourhood_min;

    let rgb_padding = max(
        neighbourhood_span.rgb * 0.25,
        vec3<f32>(0.06)
    );

    let opacity_padding = max(
        neighbourhood_span.a * 0.25,
        0.05
    );

    let history_min = vec4<f32>(
        max(neighbourhood_min.rgb - rgb_padding, vec3<f32>(0.0)),
        max(neighbourhood_min.a - opacity_padding, 0.0)
    );

    let history_max = vec4<f32>(
        neighbourhood_max.rgb + rgb_padding,
        min(neighbourhood_max.a + opacity_padding, 1.0)
    );

    let clamped_history = clamp(
        history_value,
        history_min,
        history_max
    );

    let opacity_confidence = 1.0 - smoothstep(
        0.35,
        0.70,
        opacity_mismatch
    );

    let radiance_confidence = 1.0 - smoothstep(
        0.75,
        1.50,
        radiance_mismatch
    );

    let history_confidence = min(
        opacity_confidence,
        radiance_confidence
    );

    let history_weight = 0.92 * history_confidence;

    return mix(current_value, clamped_history, history_weight);
}
