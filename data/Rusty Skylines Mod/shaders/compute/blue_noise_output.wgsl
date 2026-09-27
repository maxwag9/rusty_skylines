struct Params {
    size: u32,
    seed: u32,
    sigma: f32,
    kernel_size: i32,
};

@group(1) @binding(0)
var output_texture : texture_storage_2d<rgba8unorm, write>;

@group(2) @binding(0)
var<storage, read_write> ranks : array<f32>;

@group(2) @binding(1)
var<uniform> params : Params;

fn pcg(n: u32) -> u32 {
    var state = n * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let n = params.size * params.size;
    if (idx >= n) { return; }

    let x = idx % params.size;
    let y = idx / params.size;
    let v = ranks[idx] / f32(n - 1u);
    let h2 = pcg(idx ^ params.seed ^ 0xDEADBEEFu);
    let a = f32(h2) / 4294967295.0 * 6.28318530718;

    textureStore(
        output_texture,
        vec2<u32>(x, y),
        vec4<f32>(v, cos(a) * 0.5 + 0.5, sin(a) * 0.5 + 0.5, 1.0)
    );
}