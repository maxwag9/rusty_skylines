// === ui_circle_outline.wgsl ===

struct ScreenUniform {
    size: vec2<f32>,
    time: f32,
    enable_dither: u32,
};

@group(0) @binding(0)
var<uniform> screen: ScreenUniform;

struct CircleOutlineParams {
    center_radius_border: vec4<f32>, // cx, cy, radius, thickness
    dash_color: vec4<f32>,
    dash_misc:  vec4<f32>,         // (dash_len, dash_spacing, dash_roundness, speed)
    sub_dash_color: vec4<f32>,
    sub_dash_misc:  vec4<f32>,         // (sub_dash_len, sub_dash_spacing, sub_dash_roundness, sub_speed)
    misc: vec4<f32>
};

@group(1) @binding(0)
var<storage, read> circles: array<CircleOutlineParams>;

// ==== VERTEX ====
struct VertexInput {
    @location(0) pos: vec2<f32>,
    @location(1) color: vec4<f32>,
    @builtin(instance_index) instance: u32,
};

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) circle_index: u32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let c = circles[in.instance];
    let crb = c.center_radius_border;

    let center = crb.xy;
    let radius = crb.z + crb.w * crb.z * 0.01; // slightly expand for outline thickness
    let world = center + in.pos * radius;

    let x = (world.x / screen.size.x) * 2.0 - 1.0;
    let y = 1.0 - (world.y / screen.size.y) * 2.0;

    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.local_pos = world;
    out.circle_index = in.instance;
    return out;
}



const PI: f32 = 3.14159265;

fn repeat(x: f32, p: f32) -> f32 {
    return x - p * floor(x / p);
}

// Rounded-rectangle SDF; sharp corners when r == 0.
fn rr_sd(s: f32, v: f32, L: f32, W: f32, r: f32) -> f32 {
    if (r <= 0.0001) {
        return max(abs(s) - L, abs(v) - W);
    }
    let q = vec2<f32>(abs(s), abs(v)) - vec2<f32>(L - r, W - r);
    let outside = max(q, vec2<f32>(0.0));
    let d_out = length(outside) - r;
    let d_in  = min(max(q.x, q.y), 0.0);
    return d_out + d_in;
}

// === unified sub-dash version ===

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = circles[in.circle_index];
    let crb = c.center_radius_border;
    let center = crb.xy;
    let radius = crb.z;
    let hundredth_radius = radius * 0.015;
    let thickness = crb.w * hundredth_radius;

    // --- main dash pattern setup ---
    let dash_len   = max(0.001, c.dash_misc.x);
    let dash_space = max(0.001, c.dash_misc.y);
    let dash_round = clamp(c.dash_misc.z, 0.0, 1.0);
    let speed      = c.dash_misc.w;

    let to_center = in.local_pos - center;
    let dist = length(to_center);
    let v_ring = dist - radius;

    let angle   = atan2(to_center.y, to_center.x);
    let arc_px  = (angle + PI) * radius;
    let circ_px = 2.0 * PI * radius;

    var dash_px   = dash_len * thickness * hundredth_radius;
    var space_px  = dash_space * thickness * hundredth_radius;
    var period_px = dash_px + space_px;

    // fit integer number of dashes around circumference
    let n = max(1.0, floor(circ_px / period_px + 0.5));
    let period_adj = circ_px / n;
    dash_px   *= period_adj / period_px;
    space_px   = period_adj - dash_px;
    period_px  = period_adj;

    let scroll_px = speed * screen.time * thickness;
    let t  = repeat(arc_px + scroll_px + 0.5 * dash_px, period_px);
    let s0 = t - 0.5 * dash_px;

    let L = 0.5 * dash_px;
    let W = 0.5 * thickness;
    let rcap = dash_round * W;

    // main capsule SDF and mask
    let sd0 = rr_sd(s0,             v_ring, L, W, rcap);
    let sd1 = rr_sd(s0 - period_px, v_ring, L, W, rcap);
    let sd_capsule = min(sd0, sd1);

    let aa_main = max(0.75, fwidth(sd_capsule) * 0.75);
    let main_mask = 1.0 - smoothstep(0.0, aa_main, sd_capsule);

    // === sub-dashes: continuous track crossing inner/outer ===
    var sub_mask = 0.0;
    if (main_mask > 0.0) {
        // user parameters
        let sub_len   = max(0.001, c.sub_dash_misc.x);
        let sub_space = max(0.001, c.sub_dash_misc.y);
        let sub_round = clamp(c.sub_dash_misc.z, 0.0, 1.0);
        let sub_speed = c.sub_dash_misc.w;

        // physical dash/space sizes in pixels
        var sub_dash_px  = sub_len * thickness;
        var sub_space_px = sub_space * thickness;
        var sub_period   = sub_dash_px + sub_space_px;

        // true capsule perimeter in pixels
        let P = 2.0 * PI * W + 4.0 * L;

        // fit integer number of periods exactly around capsule
        let n_sub = max(1.0, floor(P / sub_period + 0.5));
        let sub_period_adj = P / n_sub;
        sub_dash_px  *= sub_period_adj / sub_period;
        sub_space_px  = sub_period_adj - sub_dash_px;
        sub_period    = sub_period_adj;

        // continuous coordinate around capsule
        var u = 0.0;
        if (s0 <= -L) {
            let theta = atan2(v_ring, s0 + L);
            u = (PI * 0.5 - theta) * W;
        } else if (s0 >= L) {
            let theta = atan2(v_ring, s0 - L);
            u = PI * W + 2.0 * L + (theta + PI * 0.5) * W;
        } else if (v_ring < 0.0) {
            u = PI * W + (s0 + L);
        } else {
            u = 2.0 * PI * W + 2.0 * L + (L - s0);
        }

        // scrolling and centering
        let u_scrolled = repeat(u + sub_speed * screen.time * thickness + 0.5 * sub_dash_px, sub_period);
        let su = u_scrolled - 0.5 * sub_dash_px;

        // geometry for sub-dashes
        let sub_L  = 0.5 * sub_dash_px;
        let sub_W  = 0.2 * W;
        let sub_rc = sub_round * sub_W;

        // signed boundary distance: flips sign across caps so track crosses over
        let vb = sd_capsule * sign(v_ring);

        // signed-distance field for one sub-dash (two repeats)
        let sds0 = rr_sd(su,              vb, sub_L, sub_W, sub_rc);
        let sds1 = rr_sd(su - sub_period, vb, sub_L, sub_W, sub_rc);
        let sds  = min(sds0, sds1);

        // analytic AA
        let aa_sub = max(0.75, fwidth(sds) * 0.75);
        sub_mask = 1.0 - smoothstep(0.0, aa_sub, sds);
    }


    // --- composite ---
    let base_rgb = c.dash_color.rgb;
    let sub_rgb  = c.sub_dash_color.rgb;

    let m_main = clamp(main_mask, 0.0, 1.0);
    let m_sub  = clamp(sub_mask, 0.0, 1.0);

    let rgb   = mix(base_rgb, sub_rgb, m_sub);
    let alpha = mix(c.dash_color.a * m_main, c.sub_dash_color.a * m_sub, m_sub);

    return vec4<f32>(rgb, alpha);
}
