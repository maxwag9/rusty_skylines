@group(0) @binding(0)
var input_weather: texture_2d<f32>;

@group(1) @binding(0)
var output_weather: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dims = textureDimensions(output_weather);

    if (gid.x >= out_dims.x || gid.y >= out_dims.y) {
        return;
    }

    let in_dims = textureDimensions(input_weather);
    let base = vec2<i32>(gid.xy * 2u);
    var sum = vec4<f32>(0.0);

    for (var y = 0; y < 2; y = y + 1) {
        for (var x = 0; x < 2; x = x + 1) {
            let coord = min(
                base + vec2<i32>(x, y),
                vec2<i32>(in_dims) - vec2<i32>(1)
            );

            sum += textureLoad(input_weather, coord, 0);
        }
    }

    textureStore(
        output_weather,
        vec2<i32>(gid.xy),
        sum * 0.25
    );
}