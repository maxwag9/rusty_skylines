// shaders/fog_msaa.wgsl

struct Uniforms {
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    lighting_view_proj: array<mat4x4<f32>, 4>,
    cascade_splits: vec4<f32>,     // end distance of each cascade in view-space units

    sun_direction: vec3<f32>,
    time: f32,

    camera_pos: vec3<f32>,
    orbit_radius: f32,

    moon_direction: vec3<f32>,
    shadow_cascade_index: u32,     // used only during shadow rendering
};

struct FogUniforms {
    screen_size: vec2<f32>,
    proj_params: vec2<f32>,      // (near, far) if you want; not strictly required here

    fog_density: f32,
    fog_height: f32,             // world-space height where fog is strongest below
    cam_height: f32,             // camera height (world-space)
    _pad0: f32,

    fog_color: vec3<f32>,
    _pad1: f32,

    fog_sky_factor: f32,         // applied when depth==1 (sky)
    fog_height_falloff: f32,     // how quickly fog increases as you go lower
    fog_start: f32,              // distance start
    fog_end: f32,                // distance end
};

struct PickUniform {
    pos: vec3<f32>,
    radius: f32,
    underwater: u32,
    color: vec3<f32>,
    // implicit padding in WGSL
};
@group(0) @binding(0) var depth_tex: texture_depth_multisampled_2d;

@group(1) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(1) var<uniform> fog: FogUniforms;
@group(1) @binding(2) var<uniform> pick: PickUniform;

fn saturate(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// ----------------------------
// Reconstruct world position
// ----------------------------
fn reconstruct_world(pixel_xy: vec2<f32>, depth: f32) -> vec3<f32> {
    let uv = pixel_xy / fog.screen_size;
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let ndc = vec4<f32>(ndc_xy, depth, 1.0);

    let world_h = uniforms.inv_view_proj * ndc;
    return world_h.xyz / world_h.w;
}

// ----------------------------
// View-space distance (linear)
// ----------------------------
fn view_distance(world_pos: vec3<f32>) -> f32 {
    let view_pos = uniforms.view * vec4<f32>(world_pos, 1.0);
    return abs(view_pos.z);
}


// ----------------------------
// Distance fog (guaranteed 0..1)
// ----------------------------
fn distance_fog_factor(view_dist: f32) -> f32 {
    return saturate(
        (view_dist - fog.fog_start) /
        max(0.001, fog.fog_end - fog.fog_start)
    );
}

// ----------------------------
// Atmospheric scattering-lite
// ----------------------------
fn scattering_color(ray_dir: vec3<f32>) -> vec3<f32> {
    // Horizon boost
    let horizon = saturate(1.0 - abs(ray_dir.y));

    // Sun forward scatter
    let sun_dot = saturate(dot(ray_dir, normalize(uniforms.sun_direction)));
    let sun_scatter = pow(sun_dot, 8.0);

    let sky_tint = fog.fog_color;
    let sun_tint = vec3<f32>(1.0, 0.95, 0.85);

    return sky_tint
        + horizon * 0.25 * sky_tint
        + sun_scatter * 0.35 * sun_tint;
}

// ----------------------------
// Fog color with pick tint
// ----------------------------
fn fog_color_tinted(world_pos: vec3<f32>, base_col: vec3<f32>) -> vec3<f32> {
    let r = max(pick.radius, 0.0001);
    let d = distance(world_pos, pick.pos);

    let t = saturate(1.0 - d / r);
    let smoothed = t * t * (3.0 - 2.0 * t);

    return mix(base_col, pick.color, smoothed);
}

// ----------------------------
// Vertex
// ----------------------------
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

// ----------------------------
// Fragment
// ----------------------------
@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(frag_pos.xy);
    let depth = textureLoad(depth_tex, pixel, 0);

    let uv = frag_pos.xy / fog.screen_size;
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);

    // View ray for sky + scattering
    let far_h = uniforms.inv_view_proj * vec4<f32>(ndc_xy, 1.0, 1.0);
    let far_pos = far_h.xyz / far_h.w;
    let ray_dir = normalize(far_pos - uniforms.camera_pos);

    let is_sky = depth >= 0.999999;

    var fog_amt: f32;
    var fog_col: vec3<f32>;

    if (is_sky) {
        // SKY FOG
        let horizon = saturate(1.0 - abs(ray_dir.y));
        fog_amt = horizon * fog.fog_sky_factor;
        fog_col = scattering_color(ray_dir);
    } else {
        // GEOMETRY FOG
        let world_pos = reconstruct_world(frag_pos.xy, depth);
        let view_dist = view_distance(world_pos);
        fog_amt = distance_fog_factor(view_dist) * fog.fog_density;

        fog_col = fog_color_tinted(world_pos, scattering_color(ray_dir));
    }


    return vec4<f32>(fog_col, fog_amt);
}
