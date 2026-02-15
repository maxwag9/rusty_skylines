// shadow.wgsl
const CASCADE_BLEND_RATIO: f32 = 0.25;

// Fade width in UV, but we’ll ALSO scale it with the kernel radius so it stays sane.
const SHADOW_FADE_UV_BASE: f32 = 0.01;

// PCF radius in texels (we’ll scale up slightly in farther cascades)
const PCF_RADIUS_NEAR: f32 = 1.25;
const PCF_RADIUS_FAR:  f32 = 2.75;

// Bias tuning (start here; these are intentionally larger than your 2e-6)
const BASE_BIAS: f32  = 0.00005;
const SLOPE_BIAS: f32 = 0.00125;

// Receiver-plane depth bias multiplier (huge for killing “thin acne lines”)
const RECEIVER_PLANE_BIAS_MUL: f32 = 2.5;

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

fn shadow_texel_size() -> vec2<f32> {
    let du: vec2<u32> = textureDimensions(t_shadow, 0);
    return vec2<f32>(1.0 / f32(du.x), 1.0 / f32(du.y));
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
    // Dave Hoskins-ish hash
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
    // How close are we to the texture border, accounting for our max sampling radius?
    let edge_x = min(uv.x, 1.0 - uv.x) - kernel_radius_uv.x;
    let edge_y = min(uv.y, 1.0 - uv.y) - kernel_radius_uv.y;
    let edge = min(edge_x, edge_y);

    // Fade width grows with kernel (prevents “hard” edge shimmer as kernel changes)
    let fade_w = SHADOW_FADE_UV_BASE + 4.0 * max(kernel_radius_uv.x, kernel_radius_uv.y);

    // Smooth (NOT linear) makes transitions much less noticeable
    return smoothstep(0.0, fade_w, edge);
}

// Rotated Poisson PCF (12 taps). With a LINEAR compare sampler, each tap is already “softer”.
fn shadow_pcf_poisson12(
    cascade: u32,
    uv: vec2<f32>,
    depth_ref: f32,
    kernel_radius_uv: vec2<f32>,
    rot: mat2x2<f32>,
) -> f32 {
    let layer = i32(cascade);

    // Precomputed Poisson-ish points (roughly unit disk)
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

    // Light “tent-ish” weighting: center-ish taps contribute slightly more
    // (reduces noise without reintroducing grid artifacts)
    var sum: f32 = 0.0;
    var wsum: f32 = 0.0;

    // Inline macro-ish helper
    // (WGSL has no macros; just repeat)
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

struct ShadowEval {
    vis: f32,
    edge: f32,   // 1 inside, 0 near/outside border
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

    // kernel radius grows with cascade (hides far aliasing / reduces “sparkle”)
    let texel = shadow_texel_size();
    let t = f32(cascade) / 3.0;
    let radius_texels = mix(PCF_RADIUS_NEAR, PCF_RADIUS_FAR, t);
    let kernel_radius_uv = texel * radius_texels;

    // validity
    let uv_ok = all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0));
    let z_ok  = (z >= 0.0) && (z <= 1.0);

    // Edge fade (computed even if invalid; invalid => edge=0)
    let edge = select(0.0, cascade_edge_fade_uv(uv, kernel_radius_uv), (uv_ok && z_ok));

    if (!(uv_ok && z_ok)) {
        return ShadowEval(1.0, 0.0, false);
    }

    // Bias:
    // 1) slope term (your original idea)
    let ndotl = saturate(dot(N, L));
    var bias = BASE_BIAS + SLOPE_BIAS * (1.0 - ndotl);

    // 2) receiver-plane bias (kills “fine acne lines” especially on roads)
    // This is in *shadow depth units* because it operates on projected z.
    let dzdx = abs(dpdx(z));
    let dzdy = abs(dpdy(z));
    bias = max(bias, (dzdx + dzdy) * RECEIVER_PLANE_BIAS_MUL);

    // Apply bias with reversed-z awareness
    let depth_ref = clamp(
        select(z - bias, z + bias, uniforms.reversed_depth_z != 0u),
        0.0, 1.0
    );

    // Rotate kernel in a stable way (world-space hash)
    let a = hash12(world_pos.xz + vec2<f32>(13.1 * f32(cascade), 7.7)) * 6.28318530718;
    let r = rot2(a);

    let vis = shadow_pcf_poisson12(cascade, uv, depth_ref, kernel_radius_uv, r);
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

    // Last cascade: fade to lit at the border (no next cascade to blend to)
    if (c >= 3u) {
        return mix(1.0, s0.vis, s0.edge);
    }

    // Depth-based cascade blending region
    let current_far = uniforms.cascade_splits[i32(c)];
    let blend_start = current_far * (1.0 - CASCADE_BLEND_RATIO);

    let t_depth = smoothstep(blend_start, current_far, view_depth);

    // Edge-based blending: as we approach the UV border, prefer the next cascade
    // instead of fading to fully lit (THIS fixes “shimmering corners” behavior).
    let t_edge = 1.0 - s0.edge;

    let t = max(t_depth, t_edge);

    // Evaluate next cascade only if needed (minor perf win)
    if (t <= 0.0) {
        return s0.vis;
    }

    let s1 = shadow_eval_cascade(c + 1u, world_pos, N, L);
    return mix(s0.vis, s1.vis, t);
}