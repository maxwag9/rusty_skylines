// shaders/fog.wgsl

struct Uniforms {
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,

    sun_direction: vec3<f32>,
    time: f32,

    camera_pos: vec3<f32>,
    orbit_radius: f32,

    moon_direction: vec3<f32>,
    _pad0: f32,
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

@group(0) @binding(0) var depth_tex: texture_depth_2d;

@group(1) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(1) var<uniform> fog: FogUniforms;
@group(1) @binding(2) var<uniform> pick: PickUniform;

fn saturate(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

fn reconstruct_world(pixel_xy: vec2<f32>, depth: f32) -> vec3<f32> {
    // pixel_xy is in pixels (origin top-left)
    let uv = pixel_xy / fog.screen_size; // 0..1, origin top-left

    // NDC x,y in [-1,1], y flipped for typical clip space
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);

    // For wgpu/Vulkan-style depth range, NDC z is 0..1
    let ndc = vec4<f32>(ndc_xy, depth, 1.0);

    let world_h = uniforms.inv_view_proj * ndc;
    return world_h.xyz / world_h.w;
}

fn fog_amount(world_pos: vec3<f32>, view_dist: f32) -> f32 {
    let dist01 = saturate((view_dist - fog.fog_start) / max(0.0001, (fog.fog_end - fog.fog_start)));

    // More fog at LOWER heights:
    // height_term grows as world_pos.y goes below fog.fog_height
    let height_delta = (fog.fog_height - world_pos.y);
    let height_term = exp(height_delta * fog.fog_height_falloff);

    // Combine into a smooth-ish exponential fog
    let x = fog.fog_density * dist01 * height_term;
    return saturate(1.0 - exp(-x));
}

fn fog_color_tinted(world_pos: vec3<f32>) -> vec3<f32> {
    let r = max(pick.radius, 0.0001);
    let d = distance(world_pos, pick.pos);

    // 1.0 at center, 0.0 outside radius
    let tint = saturate(1.0 - d / r);

    // Slightly softer edge
    let tint_smooth = tint * tint * (3.0 - 2.0 * tint);

    return mix(fog.fog_color, pick.color, tint_smooth);
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // Fullscreen triangle
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
    let pixel_i = vec2<i32>(frag_pos.xy);
    let depth = textureLoad(depth_tex, pixel_i, 0);

    // If depth is 1.0, treat as sky/no-geometry. Apply a subtle sky fog.
    if (depth >= 0.999999) {
        // Approximate more fog near horizon using view ray "verticalness"
        let uv = frag_pos.xy / fog.screen_size;
        let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
        let far_h = uniforms.inv_view_proj * vec4<f32>(ndc_xy, 1.0, 1.0);
        let far_pos = far_h.xyz / far_h.w;
        let ray_dir = normalize(far_pos - uniforms.camera_pos);

        let horizon = saturate(1.0 - abs(ray_dir.y));
        let a = saturate(fog.fog_sky_factor * horizon);

        return vec4<f32>(fog.fog_color, a);
    }

    let world_pos = reconstruct_world(frag_pos.xy, depth);
    let view_dist = distance(uniforms.camera_pos, world_pos);

    let a = fog_amount(world_pos, view_dist);
    let col = fog_color_tinted(world_pos);

    return vec4<f32>(col, a);
}