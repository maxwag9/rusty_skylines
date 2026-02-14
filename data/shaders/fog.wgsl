// fog.wgsl — Render pass fog (fullscreen triangle, blends into HDR)
#include "includes/uniforms.wgsl"

struct FogUniforms {
    fog_density:        f32,
    fog_height:         f32,
    fog_height_falloff: f32,
    fog_start:          f32,

    fog_end:            f32,
    fog_sky_factor:     f32,
    _pad0:              f32,
    _pad1:              f32,

    fog_color:          vec3<f32>,
    _pad2:              f32,
};

// ── Bindings ──

@group(0) @binding(0) var trilinear_sampler: sampler;
@group(0) @binding(1) var depth_half: texture_2d<f32>;   // linear depth at half-res
#ifdef MSAA
@group(0) @binding(2) var depth_full_raw: texture_depth_multisampled_2d;
#else
@group(0) @binding(2) var depth_full_raw: texture_depth_2d;
#endif

@group(1) @binding(0) var<uniform> camera: Uniforms;
@group(1) @binding(1) var<uniform> fog:    FogUniforms;

// ── Vertex ──

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 2);
    let y = f32(i32(vertex_index >> 1u) * 2);
    out.uv = vec2<f32>(x, 1.0 - y);
    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

// ── Helpers ──

fn saturate_f(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

fn linearize_depth(d: f32, z_near: f32, z_far: f32, reversed: bool) -> f32 {
    if reversed {
        if d <= 0.000001 { return z_far; }
        return z_near / d;
    } else {
        return (z_near * z_far) / (z_far - d * (z_far - z_near));
    }
}

fn load_full_depth_linear(coords: vec2<i32>) -> f32 {
    let z_near = camera.near_far_depth.x;
    let z_far  = camera.near_far_depth.y;
    let is_rev = camera.reversed_depth_z != 0u;
    let raw = textureLoad(depth_full_raw, coords, 0);
    return linearize_depth(raw, z_near, z_far, is_rev);
}

fn distance_fog_factor(view_dist: f32) -> f32 {
    return saturate_f(
        (view_dist - fog.fog_start) /
        max(0.001, fog.fog_end - fog.fog_start)
    );
}

// Height-based density attenuation
fn height_fog_factor(world_y: f32) -> f32 {
    let relative = max(0.0, world_y - fog.fog_height);
    return exp(-relative * fog.fog_height_falloff);
}

fn get_ray_dir(pixel_xy: vec2<f32>, screen_size: vec2<f32>) -> vec3<f32> {
    let uv = pixel_xy / screen_size;
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let far_h = camera.inv_view_proj * vec4<f32>(ndc_xy, 1.0, 1.0);
    let far_pos = far_h.xyz / far_h.w;
    return normalize(far_pos - camera.camera_local);
}

fn reconstruct_world_pos(pixel_xy: vec2<f32>, screen_size: vec2<f32>, linear_depth: f32) -> vec3<f32> {
    let ray = get_ray_dir(pixel_xy, screen_size);
    return camera.camera_local + ray * linear_depth;
}

// Atmospheric scattering approximation
fn scattering_color(ray_dir: vec3<f32>) -> vec3<f32> {
    let horizon = saturate_f(1.0 - abs(ray_dir.y));

    // Mie-like sun forward scatter
    let sun_dot = saturate_f(dot(ray_dir, normalize(camera.sun_direction)));
    let sun_scatter = pow(sun_dot, 8.0);

    let sun_tint = vec3<f32>(1.0, 0.95, 0.85);

    return fog.fog_color
        + horizon * 0.25 * fog.fog_color
        + sun_scatter * 0.35 * sun_tint;
}

// ── Fragment ──

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let full_coords = vec2<i32>(floor(in.position.xy));
    let screen_size = vec2<f32>(textureDimensions(depth_full_raw));

    let z_near   = camera.near_far_depth.x;
    let z_far    = camera.near_far_depth.y;
    let full_depth = load_full_depth_linear(full_coords);

    let pixel_xy = vec2<f32>(full_coords) + 0.5;
    let ray_dir  = get_ray_dir(pixel_xy, screen_size);

    let is_sky = full_depth >= z_far * 0.999;

    var fog_amt: f32;
    var fog_col: vec3<f32>;

    if is_sky {
        // Sky: horizon-based atmospheric haze
        let horizon = saturate_f(1.0 - abs(ray_dir.y));
        fog_amt = horizon * fog.fog_sky_factor;
        fog_col = scattering_color(ray_dir);
    } else {
        // Geometry: distance + height fog
        let world_pos = reconstruct_world_pos(pixel_xy, screen_size, full_depth);
        let dist_factor   = distance_fog_factor(full_depth);
        let height_factor = height_fog_factor(world_pos.y);

        fog_amt = dist_factor * height_factor * fog.fog_density;
        fog_col = scattering_color(ray_dir);
    }

    fog_amt = saturate_f(fog_amt);

    // Output fog color with fog amount as alpha.
    // The blend state will multiply the existing HDR color by (1 - fog_amt)
    // and add (fog_col * fog_amt) on top.
    return vec4<f32>(fog_col * fog_amt, 1.0 - fog_amt);
}