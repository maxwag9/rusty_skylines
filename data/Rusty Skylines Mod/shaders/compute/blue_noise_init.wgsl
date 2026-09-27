struct Params {
    size: u32,
    seed: u32,
    sigma: f32,
    kernel_size: i32,
};
@group(2) @binding(0)
var<storage, read_write> binary : array<u32>;

@group(2) @binding(1)
var<storage, read_write> ranks : array<f32>;

@group(2) @binding(2)
var<storage, read_write> energy : array<f32>;

@group(2) @binding(3)
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

    let h = pcg(idx ^ params.seed);
    binary[idx] = select(0u, 1u, f32(h) / 4294967295.0 < 0.1);
    ranks[idx] = 0.0;
    energy[idx] = 0.0;
}