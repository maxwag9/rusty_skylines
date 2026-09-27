struct Params {
    size: u32,
    seed: u32,
    sigma: f32,
    kernel_size: i32,
};

@group(1) @binding(0)
var output_texture : texture_storage_2d<rgba8unorm, write>;

@group(2) @binding(0)
var<storage, read_write> binary : array<u32>;

@group(2) @binding(1)
var<storage, read_write> ranks : array<f32>;

@group(2) @binding(2)
var<storage, read_write> energy : array<f32>;

@group(2) @binding(3)
var<storage, read_write> extremum : array<atomic<u32>>; // [value, index]

@group(2) @binding(4)
var<uniform> params : Params;

@group(2) @binding(5)
var<storage, read_write> state : array<u32>; // [remaining, rank]

fn compute_energy(idx: u32) {
    let size = params.size;
    let x = idx % size;
    let y = idx / size;

    let sigma2 = params.sigma * params.sigma;
    let ks = params.kernel_size;

    var e: f32 = 0.0;

    for (var dy: i32 = -ks; dy <= ks; dy++) {
        for (var dx: i32 = -ks; dx <= ks; dx++) {
            let nx = u32((i32(x) + dx + i32(size)) % i32(size));
            let ny = u32((i32(y) + dy + i32(size)) % i32(size));
            let nidx = ny * size + nx;
            if (binary[nidx] == 1u) {
                let d2 = f32(dx * dx + dy * dy);
                e += exp(-d2 / (2.0 * sigma2));
            }
        }
    }

    energy[idx] = e;

}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let n = params.size * params.size;
    if (idx >= n) { return; }

    // Reset extremum (only thread 0)
    if (idx == 0u) {
        atomicStore(&extremum[0], 0u);
        atomicStore(&extremum[1], 0xFFFFFFFFu);
    }
    storageBarrier();

    // Compute energy
    compute_energy(idx);
    storageBarrier();

    // Find max energy point
    let e = energy[idx];
    let bits = bitcast<u32>(e);
    if (binary[idx] == 1u) {
        atomicMax(&extremum[0], bits);
    }
    storageBarrier();

    // Find index with that energy
    let target_e = bitcast<f32>(atomicLoad(&extremum[0]));
    if (binary[idx] == 1u && abs(e - target_e) < 0.0001) {
        atomicMin(&extremum[1], idx);
    }
    storageBarrier();

    // Kill the selected point
    let kill = atomicLoad(&extremum[1]);
    if (idx == kill) {
        binary[idx] = 0u;
        ranks[idx] = f32(state[1]);
    }

    // Update state (only thread 0)
    if (idx == 0u) {
        state[0] -= 1u;
        state[1] += 1u;
    }
}