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

struct PickUniform {
    pos: vec3<f32>,
    radius: f32,
    underwater: u32,
    color: vec3<f32>,
}
@group(0) @binding(0) var grass_tex: texture_2d<f32>;
@group(0) @binding(1) var material_sampler: sampler;
@group(1) @binding(0)
var<uniform> uniforms: Uniforms;

@group(1) @binding(1)
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

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

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
    let shade = mix(0.58, 1.28, grass_pattern);
    //base_color *= mix(1.0, shade, grass_amount);

    // Subtle green boost for lush grass look
    base_color.g *= 1.0 + grass_amount * 0.25;

    // ---- Sample grass texture (group 0 binding 0) ----
    // derive a tiled UV from world position; tweak scale to taste
    let grass_uv_scale: f32 = 0.025; // world units -> texture space
    let grass_uv = in.world_pos.xz * grass_uv_scale;
    let grass_sample = textureSample(grass_tex, material_sampler, grass_uv).rgb;

    // Blend between vertex color/procedural base and the grass texture using grass_amount.
    // Slightly tint sampled texture by procedural shade for variety.
    let grass_tint = grass_sample * (0.85 + 0.5 * grass_pattern);
    base_color = mix(base_color, grass_tint, grass_amount);

    var final_color = base_color;

    if (pick.radius > 0.0) {
        let d = distance(in.world_pos, pick.pos);

        if (d < pick.radius) {
            let t = 1.0 - smoothstep(0.0, pick.radius, d);
            final_color = final_color + pick.color * t;
        }
    }

    return vec4<f32>(final_color, 1.0);
}