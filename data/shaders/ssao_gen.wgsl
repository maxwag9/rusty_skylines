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
@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(1) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(1) var<uniform> ssao: SsaoUniforms;

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

fn decode_normal(n: vec3<f32>) -> vec3<f32> {
    // if your normal RT stores [-1..1] already, change to normalize(n)
    return normalize(n * 2.0 - 1.0);
}

fn depth_to_ndc_z(d: f32) -> f32 {
    // WebGPU/D3D/Vulkan style: depth is already NDC.z in [0..1]
    return d;
}

fn is_background_depth(d: f32) -> bool {
    // If you clear depth to 1.0 in normal mode, and 0.0 in reversed-z mode:
    if (ssao.params1.x != 0u) { // reversed_z
        return d <= 1e-5;
    }
    return d >= 0.999999;
}

// Resolve normal depth
fn load_depth_resolved(pix: vec2<i32>) -> f32 {
    let best: f32 = textureLoad(depth_tex, pix, 0);

    return best;
}

fn ndc_xy_from_uv(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
}

fn reconstruct_view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc_xy = ndc_xy_from_uv(uv);
    let ndc_z  = depth_to_ndc_z(depth);
    let view_h = uniforms.inv_proj * vec4<f32>(ndc_xy, ndc_z, 1.0);
    return view_h.xyz / view_h.w;
}

fn project_view_to_uv(view_pos: vec3<f32>) -> vec2<f32> {
    let clip = uniforms.proj * vec4<f32>(view_pos, 1.0);
    let ndc = clip.xyz / max(1e-6, clip.w);
    return vec2<f32>(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
}

fn hash12(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}
fn rand2(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(hash12(p), hash12(p + vec2<f32>(17.0, 59.0)));
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
    let dims_i = vec2<i32>(textureDimensions(depth_tex));
    let dims = vec2<f32>(dims_i);
    let pix = vec2<i32>(frag_pos.xy);

    let depth = load_depth_resolved(pix);
    var valid_depth = depth;
    if (is_background_depth(depth)) {
        // clamp depth to just below 1.0 (or above 0.0 for reversed-Z)
        if (ssao.params1.x != 0u) {
            valid_depth = 1e-6; // reversed Z
        } else {
            valid_depth = 1.0 - 1e-6; // normal Z
        }
    }

    let uv = (frag_pos.xy + vec2<f32>(0.5)) / dims;
    let view_pos = reconstruct_view_pos(uv, valid_depth);


    let n_raw = textureLoad(normal_tex, pix, 0).xyz;
    let n_world_or_view = decode_normal(n_raw);

    // if normals are world-space (common), convert to view-space:
    let n_view = normalize((uniforms.view * vec4<f32>(n_world_or_view, 0.0)).xyz);

    // params
    let radius    = ssao.params0.x;
    let bias      = ssao.params0.y;
    let intensity = ssao.params0.z;
    let power     = ssao.params0.w;

    let tile_px: f32 = f32(max(1u, ssao.params1.y)); // e.g. 4 or 8
    let r = rand2(floor(frag_pos.xy / tile_px));
    let angle = r.x * 6.2831853;

    let rand_vec = normalize(vec3<f32>(
        cos(angle),
        sin(angle),
        0.0
    ));

    // build stable TBN
    let tangent = normalize(rand_vec - n_view * dot(rand_vec, n_view));
    let bitangent = cross(n_view, tangent);
    let tbn = mat3x3<f32>(tangent, bitangent, n_view);

    var occ: f32 = 0.0;

    for (var i: u32 = 0u; i < KERNEL_SIZE; i = i + 1u) {
        // kernel is already hemisphere-distributed by CPU generator (ALREADY NORMALIZED DON'T NORMALIZE!!)
        let k = ssao.kernel[i].xyz;
        let sample_dir = tbn * k;

        // bias along normal to reduce acne
        let sample_pos = view_pos + n_view * bias + sample_dir * radius;

        let suv = project_view_to_uv(sample_pos);
        if (suv.x <= 0.0 || suv.x >= 1.0 || suv.y <= 0.0 || suv.y >= 1.0) {
            continue;
        }

        let sp = clamp(vec2<i32>(suv * dims), vec2<i32>(0), dims_i - vec2<i32>(1));
        let sd = load_depth_resolved(sp);
        if (is_background_depth(sd)) { continue; }

        let sview = reconstruct_view_pos(suv, sd);

        // With RH view where forward is -Z: closer = larger z (less negative).
        //let occluded = select(0.0, 1.0, sview.z > sample_pos.z);
        let occluded = select(0.0, 1.0, sview.z >= sample_pos.z + bias); // non-reversed z, or not?

        let dist = length(sview - view_pos);
        let range = smoothstep(0.0, 1.0, radius / max(1e-4, dist));

        occ = occ + occluded * range;
    }

    let ao_raw = 1.0 - (occ / f32(KERNEL_SIZE)) * intensity;
    let ao = pow(clamp(ao_raw, 0.0, 1.0), power);

    return vec4<f32>(ao, ao, ao, 1.0);
}