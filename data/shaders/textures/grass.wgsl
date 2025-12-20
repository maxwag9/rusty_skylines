@group(0) @binding(0)
var output_tex: texture_storage_2d<rgba8unorm, write>;

struct GrassParams {
    grass_color : vec4<f32>,
    blade_density : f32,
    blade_height : f32,
    wind_phase : f32,
    time : f32,
    noise_scale : f32,
};

@group(0) @binding(1)
var<uniform> params : GrassParams;

@group(0) @binding(2)
var<storage, read> noise : array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output_tex);

    if (id.x >= size.x || id.y >= size.y) {
        return;
    }

    let uv = vec2<f32>(id.xy) / vec2<f32>(size);
    let color = vec4<f32>(uv, 0.0, 1.0);

    textureStore(output_tex, vec2<i32>(id.xy), color);
}
