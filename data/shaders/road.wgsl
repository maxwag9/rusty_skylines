struct Uniforms {
    view : mat4x4<f32>,
    inv_view : mat4x4<f32>,
    proj : mat4x4<f32>,
    inv_proj : mat4x4<f32>,
    view_proj : mat4x4<f32>,
    inv_view_proj : mat4x4<f32>,

    sun_direction : vec3<f32>,
    time : f32,

    camera_pos : vec3<f32>,
    orbit_radius : f32,

    moon_direction : vec3<f32>,
    _pad0 : f32
};

struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) normal   : vec3<f32>,
    @location(2) uv       : vec2<f32>,
    @location(3) material_id : u32
};

struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>,
    @location(0) uv : vec2<f32>,
    @location(1) material_id : u32
};

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
    var out : VertexOutput;

    let world_pos = vec4<f32>(input.position, 1.0);
    out.clip_position = uniforms.view_proj * world_pos;

    out.uv = input.uv;
    out.material_id = input.material_id;

    return out;
}


// ---- Bind group 0: road materials ----
@group(0) @binding(0) var tex0 : texture_2d<f32>; // asphalt
@group(0) @binding(1) var tex1 : texture_2d<f32>; // concrete
@group(0) @binding(2) var tex2 : texture_2d<f32>; // goo
@group(0) @binding(3) var road_sampler : sampler;
@group(1) @binding(0) var<uniform> uniforms : Uniforms;

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;

    var color : vec4<f32>;
    if (input.material_id == 0u) {
        color = textureSample(tex0, road_sampler, uv);
    } else if (input.material_id == 1u) {
        color = textureSample(tex1, road_sampler, uv);
    } else if (input.material_id == 2u) {
        color = textureSample(tex2, road_sampler, uv);
    } else {
        color = vec4<f32>(1.0, 0.0, 1.0, 1.0);
    }

    return color;
}
