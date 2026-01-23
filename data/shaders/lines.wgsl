struct Uniforms {
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    lighting_view_proj: array<mat4x4<f32>, 4>,
    cascade_splits: vec4<f32>,     // end distance of each cascade in view-space units

    sun_direction: vec3<f32>,
    time: f32,

    camera_pos: vec3<f32>,
    orbit_radius: f32,

    moon_direction: vec3<f32>,
    shadow_cascade_index: u32,     // used only during shadow rendering
};
@group(1) @binding(0) var<uniform> uniforms: Uniforms;

struct PickUniform {
    pos: vec3<f32>,
    radius: f32,
    enabled: u32,
    color: vec3<f32>,
}

//@group(2) @binding(0)
//var<uniform> pick: PickUniform;
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) color: vec3<f32>};

@vertex
fn vs_main(@location(0) pos: vec3<f32>, @location(1) color: vec3<f32>) -> VSOut {
    var out: VSOut;
    out.pos = uniforms.view_proj * vec4<f32>(pos, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
