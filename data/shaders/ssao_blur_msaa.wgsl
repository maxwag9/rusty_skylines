struct Uniforms {
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    lighting_view_proj: array<mat4x4<f32>, 4>,
    cascade_splits: vec4<f32>,
    sun_direction: vec3<f32>, time: f32,
    camera_pos: vec3<f32>, orbit_radius: f32,
    moon_direction: vec3<f32>, shadow_cascade_index: u32,
};
const KERNEL_SIZE: u32 = 32u; // match CPU

struct SsaoUniforms {
    kernel: array<vec4<f32>, KERNEL_SIZE>, // xyz = sample dir (hemisphere), w unused
    params0: vec4<f32>, // (radius, bias, intensity, power)
    params1: vec4<u32>, // (reversed_z, noise_tile_px, unused, unused)
};
@group(0) @binding(0) var ao_tex: texture_2d<f32>;
@group(0) @binding(1) var depth_tex: texture_depth_multisampled_2d;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(1) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(1) var<uniform> ssao: SsaoUniforms;

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }
fn decode_normal(n: vec3<f32>) -> vec3<f32> { return normalize(n * 2.0 - 1.0); }

fn ndc_xy_from_uv(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
}
fn depth_to_ndc_z(d: f32) -> f32 { return d; }

fn is_background_depth(d: f32) -> bool {
    if (ssao.params1.x != 0u) { return d <= 1e-6; }
    return d >= 0.999999;
}

fn load_depth_resolved(pix: vec2<i32>) -> f32 {
    let ns: u32 = textureNumSamples(depth_tex);
    var best: f32 = textureLoad(depth_tex, pix, 0);
    if (ns <= 1u) { return best; }

    for (var s: u32 = 1u; s < ns; s = s + 1u) {
        let d = textureLoad(depth_tex, pix, i32(s));
        if (ssao.params1.x != 0u) {
            best = max(best, d);
        } else {
            best = min(best, d);
        }
    }
    return best;
}

fn reconstruct_view_z(uv: vec2<f32>, depth: f32) -> f32 {
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let ndc_z  = depth_to_ndc_z(depth);
    let view_h = uniforms.inv_proj * vec4<f32>(ndc_xy, ndc_z, 1.0);
    let view = view_h.xyz / view_h.w;
    return view.z;
}

struct VsOut { @builtin(position) pos: vec4<f32>, };

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var o: VsOut;
    o.pos = vec4<f32>(positions[vi], 0.0, 1.0);
    return o;
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dims_i = vec2<i32>(textureDimensions(ao_tex));
    let dims = vec2<f32>(dims_i);
    let p0 = vec2<i32>(frag_pos.xy);

    let d0 = load_depth_resolved(p0);
    if (is_background_depth(d0)) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
    let uv0 = (frag_pos.xy + vec2<f32>(0.5)) / dims;
    let z0 = reconstruct_view_z(uv0, d0);

    let n0w = decode_normal(textureLoad(normal_tex, p0, 0).xyz);
    let n0 = normalize((uniforms.view * vec4<f32>(n0w, 0.0)).xyz);

    // blur params
    let radius: i32 = 2;              //
    let sigma_spatial: f32 = 3.0;     // pixels
    let sigma_depth: f32 = 2.0;       // view-space units (tune 1..6 depending on scale)
    let normal_power: f32 = 1.5;      // much lower than 4.0 (try 1.0..2.5)

    var sum: f32 = 0.0;
    var wsum: f32 = 0.0;

    for (var y: i32 = -radius; y <= radius; y = y + 1) {
        for (var x: i32 = -radius; x <= radius; x = x + 1) {
            let p = clamp(p0 + vec2<i32>(x, y), vec2<i32>(0), dims_i - vec2<i32>(1));

            let ao = textureLoad(ao_tex, p, 0).r;

            let d = load_depth_resolved(p);
            if (is_background_depth(d)) { continue; }

            let uv = (vec2<f32>(p) + vec2<f32>(0.5)) / dims;
            let z = reconstruct_view_z(uv, d);

            let nw = decode_normal(textureLoad(normal_tex, p, 0).xyz);
            let n = normalize((uniforms.view * vec4<f32>(nw, 0.0)).xyz);

            // weights
            let r2 = f32(x*x + y*y);
            let w_spatial = exp(-r2 / (2.0 * sigma_spatial * sigma_spatial));
            // depth gaussian in view-space *units*; use linear depth magnitude (-z) for RH
            let dz = abs((-z) - (-z0));
            let w_depth = exp(-(dz * dz) / (2.0 * sigma_depth * sigma_depth));

            let w_normal = pow(saturate(dot(n0, n)), normal_power);

            let w = w_spatial * w_depth * w_normal;

            sum = sum + ao * w;
            wsum = wsum + w;
        }
    }

    let out_ao = sum / max(1e-5, wsum);
    return vec4<f32>(out_ao, out_ao, out_ao, 1.0);
}