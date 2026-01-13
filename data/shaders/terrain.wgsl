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
    proj_params: vec2<f32>,

    fog_density: f32,
    fog_height: f32,
    cam_height: f32,
    _pad0: f32,

    fog_color: vec3<f32>,
    _pad1: f32,

    fog_sky_factor: f32,
    fog_height_falloff: f32,
    fog_start: f32,
    fog_end: f32,
};

struct PickUniform {
    pos: vec3<f32>,
    radius: f32,
    underwater: u32,
    color: vec3<f32>,
}
@group(0) @binding(0) var texture_sampler: sampler;
@group(1) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(1)
var<uniform> fog: FogUniforms;

@group(1) @binding(2)
var<uniform> pick: PickUniform;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;

    let world_pos = in.position;

    out.position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_normal = normalize(in.normal);
    out.color = in.color;
    out.world_pos = world_pos;

    return out;
}

// Cheap hash for procedural patterns
fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let l = normalize(uniforms.sun_direction);

    let up = vec3<f32>(0.0, 1.0, 0.0);
    let hemi = clamp(dot(n, up) * 0.5 + 0.5, 0.0, 1.0);
    let sky_ambient = vec3<f32>(0.01, 0.01, 0.01);
    let ground_ambient = vec3<f32>(0.02, 0.02, 0.02);
    let ambient = mix(ground_ambient, sky_ambient, hemi);

    let ndotl = dot(n, l);
    let diffuse = max(ndotl, 0.0) * 0.60;

    var base_color = in.color * (ambient + diffuse);

    // ============ GRASS EFFECT ============
    // Detect grassy areas: green-dominant + upward-facing
    let greenness = in.color.g - max(in.color.r, in.color.b);
    let up_facing = saturate(dot(n, up));
    let grass_amount = saturate(greenness * 2.5) * up_facing * up_facing;

    // Multi-scale procedural grass pattern (2 octaves)
    let p_fine = floor(in.world_pos.xz * 600.0);
    let p_coarse = floor(in.world_pos.xz * 200.0);
    let h_fine = hash2(p_fine);
    let h_coarse = hash2(p_coarse * 0.4 + 31.7);
    let grass_pattern = h_fine * 0.55 + h_coarse * 1.85;

    // Simulate light/shadow variation between grass blades
    let shade = mix(0.78, 1.18, grass_pattern);
    base_color *= mix(1.0, shade, grass_amount);

    // Subtle green boost for lush grass look
    base_color.g *= 1.0 + grass_amount * 0.12;
    // ======================================

    let dist = distance(in.world_pos, uniforms.camera_pos);

    let f = fog_factor(dist, in.world_pos.y);

    // slight desaturation improves realism
    let gray = dot(base_color, vec3<f32>(0.3, 0.59, 0.11));
    let desat = mix(base_color, vec3<f32>(gray), f * 0.05);

    let final_color = mix(desat, fog.fog_color, f);

    var color = final_color;

    if (pick.radius > 0.0) {
        let d = distance(in.world_pos, pick.pos);

        if (d < pick.radius) {
            let t = 1.0 - smoothstep(0.0, pick.radius, d);
            color = color + pick.color * t;
        }
    }

    return vec4<f32>(color, 1.0);
}

fn height_factor_at(pos_y: f32) -> f32 {
    let dy = pos_y - fog.fog_height;
    return exp(-dy * fog.fog_height_falloff);
}

fn fog_factor(dist: f32, pos_y: f32) -> f32 {
    let h = height_factor_at(pos_y);
    return 1.0 - exp(-dist * fog.fog_density * h);
}