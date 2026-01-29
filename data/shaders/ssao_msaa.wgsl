// shaders/ssao_msaa.wgsl
//
// Expects bindings:
//   @group(0) @binding(0) depth_tex  : texture_depth_multisampled_2d
//   @group(0) @binding(1) normal_tex : texture_2d<f32>
//   @group(1) @binding(0) Uniforms   : camera uniforms
//
// Output is AO in rgb (1 = unoccluded). Use multiply blending in the render pipeline:
//   outColor = dstColor * srcColor

struct Uniforms {
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    lighting_view_proj: array<mat4x4<f32>, 4>,
    cascade_splits: vec4<f32>,

    sun_direction: vec3<f32>,
    time: f32,

    camera_pos: vec3<f32>,
    orbit_radius: f32,

    moon_direction: vec3<f32>,
    shadow_cascade_index: u32,
};

@group(0) @binding(0) var depth_tex: texture_depth_multisampled_2d;
@group(0) @binding(1) var normal_tex: texture_2d<f32>;

@group(1) @binding(0) var<uniform> uniforms: Uniforms;

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

fn decode_normal(n: vec3<f32>) -> vec3<f32> {
    return normalize(n * 2.0 - 1.0);
}

// Hash / noise: stable per-pixel pseudo-random
fn hash12(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn rand2(p: vec2<f32>) -> vec2<f32> {
    let r1 = hash12(p);
    let r2 = hash12(p + vec2<f32>(17.0, 59.0));
    return vec2<f32>(r1, r2);
}

fn ndc_xy_from_uv(uv: vec2<f32>) -> vec2<f32> {
    // NOTE: match your fog shader convention (flip Y)
    return vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
}

fn reconstruct_view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc_xy = ndc_xy_from_uv(uv);
    let ndc = vec4<f32>(ndc_xy, depth, 1.0);
    let view_h = uniforms.inv_proj * ndc;
    return view_h.xyz / view_h.w;
}

fn project_view_to_uv(view_pos: vec3<f32>) -> vec2<f32> {
    let clip = uniforms.proj * vec4<f32>(view_pos, 1.0);
    let ndc = clip.xyz / max(1e-6, clip.w);

    // convert NDC -> UV, and flip Y to match screen UV space
    let uv_x = ndc.x * 0.5 + 0.5;
    let uv_y = 1.0 - (ndc.y * 0.5 + 0.5);
    return vec2<f32>(uv_x, uv_y);
}

// A small SSAO kernel (hemisphere)
const KERNEL_SIZE: u32 = 16u;
const KERNEL: array<vec3<f32>, 16> = array<vec3<f32>, 16>(
    vec3<f32>( 0.5381,  0.1856, 0.4319),
    vec3<f32>( 0.1379,  0.2486, 0.4430),
    vec3<f32>( 0.3371,  0.5679, 0.0057),
    vec3<f32>(-0.6999, -0.0451, 0.0019),
    vec3<f32>( 0.0689, -0.1598, 0.8547),
    vec3<f32>( 0.0560,  0.0069, 0.1843),
    vec3<f32>(-0.0146,  0.1402, 0.0762),
    vec3<f32>( 0.0100, -0.1924, 0.0344),
    vec3<f32>(-0.3577, -0.5301, 0.4358),
    vec3<f32>(-0.3169,  0.1063, 0.0158),
    vec3<f32>( 0.0103, -0.5869, 0.0046),
    vec3<f32>(-0.0897, -0.4940, 0.3287),
    vec3<f32>( 0.7119, -0.0154, 0.0918),
    vec3<f32>(-0.0533,  0.0596, 0.5411),
    vec3<f32>( 0.0352, -0.0631, 0.5460),
    vec3<f32>(-0.4776,  0.2847, 0.0271)
);
fn vogel_disk(i: f32, n: f32, phi: f32) -> vec2<f32> {
    // Golden angle spiral
    let golden: f32 = 2.3999632;
    let r = sqrt((i + 0.5) / n);
    let theta = (i + phi) * golden;
    return vec2<f32>(cos(theta), sin(theta)) * r;
}
struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );

    var out: VsOut;
    out.pos = vec4<f32>(positions[vi], 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dims_i = vec2<i32>(textureDimensions(depth_tex));
    let dims = vec2<f32>(dims_i);

    let pixel = vec2<i32>(frag_pos.xy);
    if (pixel.x < 0 || pixel.y < 0 || pixel.x >= dims_i.x || pixel.y >= dims_i.y) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }

    let depth = textureLoad(depth_tex, pixel, 0);

    // Sky / far plane: no occlusion
    if (depth >= 0.999999) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }

    let uv = (frag_pos.xy + vec2<f32>(0.5, 0.5)) / dims;

    // View-space position
    let view_pos = reconstruct_view_pos(uv, depth);

    // Normal fetch
    let n_raw = textureLoad(normal_tex, pixel, 0).xyz;
    var n_world_or_view = decode_normal(n_raw);

    // If your normal buffer is WORLD-space (common), convert to VIEW-space:
    // (If it's already view-space, comment this out.)
    let n_view = normalize((uniforms.view * vec4<f32>(n_world_or_view, 0.0)).xyz);

    // SSAO params (hardcoded to keep it "one shader")
    let radius: f32 = 0.75;     // view-space radius
    let bias: f32 = 0.025;      // reduces self-occlusion
    let intensity: f32 = 1.15;  // overall strength
    let power: f32 = 1.75;      // contrast

    // Per-pixel random vector to rotate the kernel
    let r = rand2(frag_pos.xy);
    let angle = r.x * 6.2831853;
    let rand_vec = normalize(vec3<f32>(cos(angle), sin(angle), r.y * 2.0 - 1.0));

    // Build TBN around the normal in view space
    let tangent = normalize(rand_vec - n_view * dot(rand_vec, n_view));
    let bitangent = cross(n_view, tangent);
    let tbn = mat3x3<f32>(tangent, bitangent, n_view);

    var occ: f32 = 0.0;
    let N: u32 = 32u;           // try 24/32
    let phi: f32 = angle;       // from tiled noise above

    for (var i: u32 = 0u; i < N; i = i + 1u) {
        let fi = f32(i);
        let disk = vogel_disk(fi, f32(N), phi);

        // hemisphere-ish: push along normal so samples are in front of the surface
        let z = sqrt(max(0.0, 1.0 - dot(disk, disk)));
        let hemi = vec3<f32>(disk.x, disk.y, z);

        let sample_dir = tbn * hemi;
        let sample_pos = view_pos + sample_dir * radius;

        // project sample position to screen
        let sample_uv = project_view_to_uv(sample_pos);

        // skip out-of-screen samples
        if (sample_uv.x <= 0.0 || sample_uv.x >= 1.0 || sample_uv.y <= 0.0 || sample_uv.y >= 1.0) {
            continue;
        }

        let sp = vec2<i32>(sample_uv * dims);
        let sp_clamped = clamp(sp, vec2<i32>(0, 0), dims_i - vec2<i32>(1, 1));

        let sample_depth = textureLoad(depth_tex, sp_clamped, 0);
        if (sample_depth >= 0.999999) {
            continue;
        }

        // reconstruct view-space depth at that sample pixel
        let sample_view = reconstruct_view_pos(sample_uv, sample_depth);

        // If depth buffer says geometry is closer than our sample point => occluded
        let occluded = select(0.0, 1.0, sample_view.z > (sample_pos.z + bias));

        // range attenuation
        let dz = abs(view_pos.z - sample_view.z);
        let range = smoothstep(0.0, 1.0, radius / max(1e-4, dz));

        occ = occ + occluded * range;
    }

    let ao_raw = 1.0 - (occ / f32(KERNEL_SIZE)) * intensity;
    let ao = pow(saturate(ao_raw), power);

    // We output AO in RGB; pipeline uses multiply blending to apply it to existing HDR.
    return vec4<f32>(ao, ao, ao, 1.0);
}