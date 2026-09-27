// shadow.wgsl
const CASCADE_BLEND_RATIO: f32 = 0.25;

// Fade width in UV, but we'll ALSO scale it with the kernel radius so it stays sane.
const SHADOW_FADE_UV_BASE: f32 = 0.01;

// Bias tuning
const BASE_BIAS: f32  = 0.00005;
const SLOPE_BIAS: f32 = 0.00125;

// Receiver-plane depth bias multiplier (huge for killing "thin acne lines")
const RECEIVER_PLANE_BIAS_MUL: f32 = 2.5;

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

fn shadow_texel_size() -> vec2<f32> {
    let du: vec2<u32> = textureDimensions(t_shadow, 0);
    return vec2<f32>(1.0 / f32(du.x), 1.0 / f32(du.y));
}

// PCF radius varies by cascade: tighter for far cascades to prevent smudgy shadows
fn get_pcf_radius(cascade: u32) -> f32 {
    if (cascade == 0u) { return 1.25; }
    if (cascade == 1u) { return 1.25; }
    if (cascade == 2u) { return 0.85; }
    return 0.65; // cascade 3
}

fn select_cascade(view_depth: f32) -> u32 {
    let s = uniforms.cascade_splits;

    var c: u32 = 0u;
    c += select(0u, 1u, view_depth >= s.x);
    c += select(0u, 1u, view_depth >= s.y);
    c += select(0u, 1u, view_depth >= s.z);
    return min(c, 3u);
}

fn project_to_shadow(cascade: u32, world_pos: vec3<f32>) -> vec3<f32> {
    let p = uniforms.lighting_view_proj[cascade] * vec4<f32>(world_pos, 1.0);
    let ndc = p.xyz / p.w;

    // NDC [-1,1] -> UV [0,1] (Y flipped)
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return vec3<f32>(uv, ndc.z);
}

// Stable hash (world-space) for kernel rotation
fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn rot2(a: f32) -> mat2x2<f32> {
    let c = cos(a);
    let s = sin(a);
    return mat2x2<f32>(c, -s, s, c);
}

fn cascade_edge_fade_uv(uv: vec2<f32>, kernel_radius_uv: vec2<f32>) -> f32 {
    let edge_x = min(uv.x, 1.0 - uv.x) - kernel_radius_uv.x;
    let edge_y = min(uv.y, 1.0 - uv.y) - kernel_radius_uv.y;
    let edge = min(edge_x, edge_y);

    let fade_w = SHADOW_FADE_UV_BASE + 4.0 * max(kernel_radius_uv.x, kernel_radius_uv.y);
    return smoothstep(0.0, fade_w, edge);
}

// ============================================================================
// PCF SAMPLING FUNCTIONS - Different tap counts per cascade
// ============================================================================

// 12-tap Poisson PCF for cascades 0 and 1
fn shadow_pcf_poisson12(
    cascade: u32,
    uv: vec2<f32>,
    depth_ref: f32,
    kernel_radius_uv: vec2<f32>,
    rot: mat2x2<f32>,
) -> f32 {
    let layer = i32(cascade);

    let p0  = vec2<f32>(-0.326, -0.406);
    let p1  = vec2<f32>(-0.840, -0.074);
    let p2  = vec2<f32>(-0.696,  0.457);
    let p3  = vec2<f32>(-0.203,  0.621);
    let p4  = vec2<f32>( 0.962, -0.195);
    let p5  = vec2<f32>( 0.473, -0.480);
    let p6  = vec2<f32>( 0.519,  0.767);
    let p7  = vec2<f32>( 0.185, -0.893);
    let p8  = vec2<f32>( 0.507,  0.064);
    let p9  = vec2<f32>( 0.896,  0.412);
    let p10 = vec2<f32>(-0.322, -0.933);
    let p11 = vec2<f32>(-0.792, -0.598);

    var sum: f32 = 0.0;
    var wsum: f32 = 0.0;

    {
        let o = (rot * p0) * kernel_radius_uv;
        let w = 1.0 - 0.35 * length(p0);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p1) * kernel_radius_uv;
        let w = 1.0 - 0.35 * length(p1);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p2) * kernel_radius_uv;
        let w = 1.0 - 0.35 * length(p2);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p3) * kernel_radius_uv;
        let w = 1.0 - 0.35 * length(p3);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p4) * kernel_radius_uv;
        let w = 1.0 - 0.35 * length(p4);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p5) * kernel_radius_uv;
        let w = 1.0 - 0.35 * length(p5);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p6) * kernel_radius_uv;
        let w = 1.0 - 0.35 * length(p6);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p7) * kernel_radius_uv;
        let w = 1.0 - 0.35 * length(p7);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p8) * kernel_radius_uv;
        let w = 1.0 - 0.35 * length(p8);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p9) * kernel_radius_uv;
        let w = 1.0 - 0.35 * length(p9);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p10) * kernel_radius_uv;
        let w = 1.0 - 0.35 * length(p10);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p11) * kernel_radius_uv;
        let w = 1.0 - 0.35 * length(p11);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }

    return sum / max(wsum, 1e-6);
}

// Stochastic 8-tap PCF for cascade 2 - sharper, less blur
fn shadow_pcf_stochastic8(
    cascade: u32,
    uv: vec2<f32>,
    depth_ref: f32,
    kernel_radius_uv: vec2<f32>,
    rot: mat2x2<f32>,
) -> f32 {
    let layer = i32(cascade);

    // Well-distributed 8-point subset
    let p0 = vec2<f32>(-0.326, -0.406);
    let p1 = vec2<f32>(-0.840, -0.074);
    let p2 = vec2<f32>(-0.696,  0.457);
    let p3 = vec2<f32>( 0.962, -0.195);
    let p4 = vec2<f32>( 0.519,  0.767);
    let p5 = vec2<f32>( 0.185, -0.893);
    let p6 = vec2<f32>( 0.507,  0.064);
    let p7 = vec2<f32>(-0.792, -0.598);

    var sum: f32 = 0.0;
    var wsum: f32 = 0.0;

    {
        let o = (rot * p0) * kernel_radius_uv;
        let w = 1.0 - 0.3 * length(p0);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p1) * kernel_radius_uv;
        let w = 1.0 - 0.3 * length(p1);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p2) * kernel_radius_uv;
        let w = 1.0 - 0.3 * length(p2);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p3) * kernel_radius_uv;
        let w = 1.0 - 0.3 * length(p3);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p4) * kernel_radius_uv;
        let w = 1.0 - 0.3 * length(p4);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p5) * kernel_radius_uv;
        let w = 1.0 - 0.3 * length(p5);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p6) * kernel_radius_uv;
        let w = 1.0 - 0.3 * length(p6);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p7) * kernel_radius_uv;
        let w = 1.0 - 0.3 * length(p7);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }

    return sum / max(wsum, 1e-6);
}

// Stochastic 6-tap PCF for cascade 3 - sharpest, minimal blur for distant shadows
fn shadow_pcf_stochastic6(
    cascade: u32,
    uv: vec2<f32>,
    depth_ref: f32,
    kernel_radius_uv: vec2<f32>,
    rot: mat2x2<f32>,
) -> f32 {
    let layer = i32(cascade);

    // Hexagonal-ish distribution for 6 taps
    let p0 = vec2<f32>(-0.500,  0.000);
    let p1 = vec2<f32>( 0.500,  0.000);
    let p2 = vec2<f32>(-0.250, -0.433);
    let p3 = vec2<f32>( 0.250,  0.433);
    let p4 = vec2<f32>(-0.250,  0.433);
    let p5 = vec2<f32>( 0.250, -0.433);

    var sum: f32 = 0.0;
    var wsum: f32 = 0.0;

    {
        let o = (rot * p0) * kernel_radius_uv;
        let w = 1.0 - 0.25 * length(p0);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p1) * kernel_radius_uv;
        let w = 1.0 - 0.25 * length(p1);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p2) * kernel_radius_uv;
        let w = 1.0 - 0.25 * length(p2);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p3) * kernel_radius_uv;
        let w = 1.0 - 0.25 * length(p3);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p4) * kernel_radius_uv;
        let w = 1.0 - 0.25 * length(p4);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }
    {
        let o = (rot * p5) * kernel_radius_uv;
        let w = 1.0 - 0.25 * length(p5);
        sum += w * textureSampleCompare(t_shadow, s_shadow, uv + o, layer, depth_ref);
        wsum += w;
    }

    return sum / max(wsum, 1e-6);
}

// Dispatcher: select PCF method based on cascade
fn shadow_pcf(
    cascade: u32,
    uv: vec2<f32>,
    depth_ref: f32,
    kernel_radius_uv: vec2<f32>,
    rot: mat2x2<f32>,
) -> f32 {
    if (cascade <= 1u) {
        return shadow_pcf_poisson12(cascade, uv, depth_ref, kernel_radius_uv, rot);
    } else if (cascade == 2u) {
        return shadow_pcf_stochastic8(cascade, uv, depth_ref, kernel_radius_uv, rot);
    } else {
        return shadow_pcf_stochastic6(cascade, uv, depth_ref, kernel_radius_uv, rot);
    }
}

// ============================================================================

struct ShadowEval {
    vis: f32,
    edge: f32,
    valid: bool,
};

fn shadow_eval_cascade(
    cascade: u32,
    world_pos: vec3<f32>,
    N: vec3<f32>,
    L: vec3<f32>,
) -> ShadowEval {
    let proj = project_to_shadow(cascade, world_pos);
    let uv = proj.xy;
    let z  = proj.z;

    // kernel radius varies by cascade (tighter for far = sharper shadows)
    let texel = shadow_texel_size();
    let radius_texels = get_pcf_radius(cascade);
    let kernel_radius_uv = texel * radius_texels;

    // validity
    let uv_ok = all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0));
    let z_ok  = (z >= 0.0) && (z <= 1.0);

    let edge = select(0.0, cascade_edge_fade_uv(uv, kernel_radius_uv), (uv_ok && z_ok));

    if (!(uv_ok && z_ok)) {
        return ShadowEval(1.0, 0.0, false);
    }

    // Bias
    let ndotl = saturate(dot(N, L));
    var bias = BASE_BIAS + SLOPE_BIAS * (1.0 - ndotl);

    let dzdx = abs(dpdx(z));
    let dzdy = abs(dpdy(z));
    bias = max(bias, (dzdx + dzdy) * RECEIVER_PLANE_BIAS_MUL);

    let depth_ref = clamp(
        select(z - bias, z + bias, uniforms.reversed_depth_z != 0u),
        0.0, 1.0
    );

    // Rotate kernel in a stable way (world-space hash)
    let a = hash12(world_pos.xz + vec2<f32>(13.1 * f32(cascade), 7.7)) * 6.28318530718;
    let r = rot2(a);

    let vis = shadow_pcf(cascade, uv, depth_ref, kernel_radius_uv, r);
    return ShadowEval(vis, edge, true);
}

fn fetch_shadow(world_pos: vec3<f32>, N: vec3<f32>, L: vec3<f32>) -> f32 {
    if (uniforms.csm_enabled == 0u) {
        return 1.0;
    }

    let vpos = uniforms.view * vec4<f32>(world_pos, 1.0);
    let view_depth = max(0.0, -vpos.z);

    let c = select_cascade(view_depth);
    let s0 = shadow_eval_cascade(c, world_pos, N, L);

    // Last cascade: fade to lit at the border
    if (c >= 3u) {
        return mix(1.0, s0.vis, s0.edge);
    }

    // Depth-based cascade blending region
    let current_far = uniforms.cascade_splits[i32(c)];
    let blend_start = current_far * (1.0 - CASCADE_BLEND_RATIO);

    let t_depth = smoothstep(blend_start, current_far, view_depth);
    let t_edge = 1.0 - s0.edge;
    let t = max(t_depth, t_edge);

    if (t <= 0.0) {
        return s0.vis;
    }

    let s1 = shadow_eval_cascade(c + 1u, world_pos, N, L);
    return mix(s0.vis, s1.vis, t);
}