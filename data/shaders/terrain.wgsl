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

struct PickUniform {
    pos: vec3<f32>,
    radius: f32,
    underwater: u32,
    color: vec3<f32>,
}

@group(0) @binding(0) var grass_tex: texture_2d<f32>;
@group(0) @binding(1) var grass_tex2: texture_2d<f32>;
@group(0) @binding(2) var material_sampler: sampler;
@group(0) @binding(3) var t_shadow: texture_depth_2d_array;
@group(0) @binding(4) var s_shadow: sampler_comparison;

@group(1) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(1) var<uniform> pick: PickUniform;

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

// Cheap high-quality hash
fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// Quintic smoothstep for buttery noise
fn quintic(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Value noise 0-1
fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let ux = quintic(f.x);
    let uy = quintic(f.y);

    let a = hash2(i);
    let b = hash2(i + vec2(1.0, 0.0));
    let c = hash2(i + vec2(0.0, 1.0));
    let d = hash2(i + vec2(1.0, 1.0));

    return mix(mix(a, b, ux), mix(c, d, ux), uy);
}

// 4-octave rotated FBM for organic broad patterns
fn fbm(p_in: vec2<f32>) -> f32 {
    let ROT = mat2x2<f32>(vec2(0.8, 0.6), vec2(-0.6, 0.8));
    var acc: f32 = 0.0;
    var amp: f32 = 0.5;
    var p = p_in;

    for (var i: i32 = 0; i < 4; i = i + 1) {
        acc += amp * value_noise(p);
        p = ROT * p * 2.0;
        amp *= 0.5;
    }
    return acc;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    let l = normalize(uniforms.sun_direction);
    let up = vec3<f32>(0.0, 1.0, 0.0);

    // Hemisphere ambient
    let hemi = saturate(dot(n, up) * 0.5 + 0.5);
    let sky_ambient = vec3<f32>(0.01, 0.01, 0.01);
    let ground_ambient = vec3<f32>(0.02, 0.02, 0.02);
    let ambient = mix(ground_ambient, sky_ambient, hemi);

    // Basic diffuse + wrapped for grass
    let ndotl = dot(n, l);
    let diffuse = max(ndotl, 0.0);
    let wrapped = saturate(ndotl + 0.4) / 1.4; // Soft subsurface feel

    // Shadow
    let shadow = fetch_shadow(in.world_pos, n, l);

    // Grass mask
    let greenness = in.color.g - max(in.color.r, in.color.b);
    let up_facing = saturate(dot(n, up));
    let grass_amount = saturate(greenness * 2.5) * up_facing * up_facing;

    // Direct light with grass wrap
    let direct_light = mix(diffuse, wrapped, grass_amount);

    // ============ GRASS TEXTURING ============
    let grass_uv_scale1: f32 = 0.025;
    let grass_uv_scale2: f32 = 0.011;

    let grass_uv1 = in.world_pos.xz * grass_uv_scale1;

    // Rotate second sample to break alignment
    let rot_angle: f32 = 0.615; // ~35 degrees, tweak if you want
    let ca = cos(rot_angle);
    let sa = sin(rot_angle);
    let rot = mat2x2<f32>(ca, -sa, sa, ca);
    let grass_uv2 = rot * (in.world_pos.xz * grass_uv_scale2) + vec2<f32>(17.3, 9.1);

    let grass_a = textureSample(grass_tex, material_sampler, grass_uv1).rgb;
    let grass_b = textureSample(grass_tex2, material_sampler, grass_uv2).rgb;

    // Organic broad mixing — lower mix_scale = broader patches
    let mix_scale: f32 = 0.4; // Try 0.2-0.8 range
    let mix_offset = vec2<f32>(42.0, 87.0);
    let mix_p = in.world_pos.xz * mix_scale + mix_offset;
    let mix_noise = fbm(mix_p);
    let tex_mix = mix_noise;

    let grass_color = mix(grass_a, grass_b, tex_mix);

    // Apply procedural grass detail where it belongs
    let albedo = mix(in.color, grass_color, grass_amount);

    // Final lighting
    var final_color = albedo * (ambient + direct_light * shadow);

    // Pick highlight
    if (pick.radius > 0.0) {
        let d = distance(in.world_pos, pick.pos);
        if (d < pick.radius) {
            let t = 1.0 - smoothstep(0.0, pick.radius, d);
            final_color += pick.color * t;
        }
    }

    return vec4<f32>(final_color, 1.0);
}

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

fn select_cascade(view_depth: f32) -> u32 {
    if (view_depth < uniforms.cascade_splits.x) { return 0u; }
    if (view_depth < uniforms.cascade_splits.y) { return 1u; }
    if (view_depth < uniforms.cascade_splits.z) { return 2u; }
    return 3u;
}

struct CascadeBlend {
    i0: u32,
    i1: u32,
    t: f32, // 0 -> use i0, 1 -> use i1
};

fn select_cascade_with_blend(view_depth: f32) -> CascadeBlend {
    let i0 = select_cascade(view_depth);

    // Fade size: choose something stable-ish in view units.
    // Start with 5% of the current cascade length, clamped.
    var start: f32 = 0.0;
    var end: f32 = uniforms.cascade_splits.x;

    if (i0 == 0u) {
        start = 0.0;
        end = uniforms.cascade_splits.x;
    } else if (i0 == 1u) {
        start = uniforms.cascade_splits.x;
        end = uniforms.cascade_splits.y;
    } else if (i0 == 2u) {
        start = uniforms.cascade_splits.y;
        end = uniforms.cascade_splits.z;
    } else {
        start = uniforms.cascade_splits.z;
        end = uniforms.cascade_splits.w;
    }

    let len = max(end - start, 1.0);
    let fade = clamp(len * 0.05, 5.0, 50.0); // tweak later

    // Blend only near the END of cascade i0 into i1
    let i1 = min(i0 + 1u, 3u);
    var t = 0.0;
    if (i0 < 3u) {
        t = saturate((view_depth - (end - fade)) / fade);
    } else {
        t = 0.0;
    };

    return CascadeBlend(i0, i1, t);
}

fn project_to_shadow(cascade: u32, world_pos: vec3<f32>) -> vec3<f32> {
    let p = uniforms.lighting_view_proj[cascade] * vec4<f32>(world_pos, 1.0);
    let ndc = p.xyz / p.w;

    // NDC -> UV (flip Y to match texture space)
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return vec3<f32>(uv, ndc.z);
}

// 3x3 PCF
fn shadow_pcf_3x3(cascade: u32, uv: vec2<f32>, depth: f32) -> f32 {
    let dims = vec2<f32>(textureDimensions(t_shadow, 0));
    let texel = 1.0 / dims;

    var sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            sum += textureSampleCompare(t_shadow, s_shadow, uv + offset, i32(cascade), depth);
        }
    }
    return sum / 9.0;
}

fn shadow_for_cascade(cascade: u32, world_pos: vec3<f32>, N: vec3<f32>, L: vec3<f32>) -> f32 {
    let proj = project_to_shadow(cascade, world_pos);
    let uv = proj.xy;
    let z  = proj.z;

    // Outside shadow frustum => lit
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || z < 0.0 || z > 1.0) {
        return 1.0;
    }

    // Bias (still basic; good enough to start)
    let ndotl = saturate(dot(N, L));
    let bias = max(0.00002, 0.00025 * (1.0 - ndotl));

    // PCF
    return shadow_pcf_3x3(cascade, uv, z - bias);
}

fn fetch_shadow(world_pos: vec3<f32>, N: vec3<f32>, L: vec3<f32>) -> f32 {
    // view-space depth (RH camera looking down -Z)
    let vpos = uniforms.view * vec4<f32>(world_pos, 1.0);
    let view_depth = -vpos.z;

    let cb = select_cascade_with_blend(view_depth);

    let s0 = shadow_for_cascade(cb.i0, world_pos, N, L);
    if (cb.i0 == cb.i1) {
        return s0;
    }
    let s1 = shadow_for_cascade(cb.i1, world_pos, N, L);

    return mix(s0, s1, cb.t);
}