@group(0) @binding(0)
var input_volume: texture_3d<f32>;

@group(1) @binding(0)
var output_volume: texture_storage_3d<rgba8unorm, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dims = textureDimensions(output_volume);

    if (any(gid >= out_dims)) {
        return;
    }

    let in_dims = textureDimensions(input_volume);
    let base = vec3<i32>(gid * 2u);
    var sum = vec4<f32>(0.0);

    for (var z = 0; z < 2; z = z + 1) {
        for (var y = 0; y < 2; y = y + 1) {
            for (var x = 0; x < 2; x = x + 1) {
                let coord = min(
                    base + vec3<i32>(x, y, z),
                    vec3<i32>(in_dims) - vec3<i32>(1)
                );

                sum += textureLoad(input_volume, coord, 0);
            }
        }
    }

    textureStore(
        output_volume,
        vec3<i32>(gid),
        sum * 0.125
    );
}