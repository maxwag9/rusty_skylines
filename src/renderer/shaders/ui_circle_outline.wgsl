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
    sub_dash_misc:  vec4<f32>         // (sub_dash_len, sub_dash_spacing, sub_dash_roundness, sub_speed)
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
    let radius = crb.z + crb.w; // slightly expand for outline thickness
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

fn rr_sd(s: f32, v: f32, L: f32, W: f32, r: f32) -> f32 {
    let q = vec2<f32>(abs(s), abs(v)) - vec2<f32>(L - r, W - r);
    let outside = max(q, vec2<f32>(0.0));
    let d_out = length(outside) - r;
    let d_in  = min(max(q.x, q.y), 0.0);
    return d_out + d_in;
}

// Proper continuous CCW perimeter coordinate around capsule
fn capsule_perimeter_u(s: f32, v: f32, L: f32, W: f32) -> f32 {
    let P = 2.0 * PI * W + 4.0 * L;
    if (s <= -L) {
        // left cap, CCW from top(+pi/2) to bottom(-pi/2)
        let theta = atan2(v, (s + L));
        return repeat((PI * 0.5 - theta) * W, P);
    } else if (s >= L) {
        // right cap, CCW from bottom(-pi/2) to top(+pi/2)
        let theta = atan2(v, (s - L));
        return repeat(PI * W + 2.0 * L + (theta + PI * 0.5) * W, P);
    } else if (v < 0.0) {
        // bottom straight, left→right
        return repeat(PI * W + (s + L), P);
    } else {
        // top straight, right→left
        return repeat(2.0 * PI * W + 2.0 * L + (L - s), P);
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = circles[in.circle_index];
    let crb = c.center_radius_border;
    let center = crb.xy;
    let radius = crb.z;
    let thickness = crb.w;

    // --- main ring pattern ---
    let dash_len   = max(0.001, c.dash_misc.x);
    let dash_space = max(0.001, c.dash_misc.y);
    let dash_round = clamp(c.dash_misc.z, 0.0, 1.0);
    let speed      = c.dash_misc.w;

    let to_center = in.local_pos - center;
    let dist = length(to_center);
    let v_ring = dist - radius;

    let angle = atan2(to_center.y, to_center.x);
    let arc_px  = (angle + PI) * radius;
    let circ_px = 2.0 * PI * radius;

    var dash_px   = dash_len * thickness;
    var space_px  = dash_space * thickness;
    var period_px = dash_px + space_px;

    let n = max(1.0, floor(circ_px / period_px + 0.5));
    let period_adj = circ_px / n;
    dash_px   = dash_px   * (period_adj / period_px);
    space_px  = period_adj - dash_px;
    period_px = period_adj;

    let scroll_px = speed * screen.time * thickness;
    let t  = repeat(arc_px + scroll_px + 0.5 * dash_px, period_px);
    let s0 = t - 0.5 * dash_px;

    let L = 0.5 * dash_px;
    let W = 0.5 * thickness;
    let rcap = dash_round * W;

    // primary capsule distance
    let sd0 = rr_sd(s0,             v_ring, L, W, rcap);
    let sd1 = rr_sd(s0 - period_px, v_ring, L, W, rcap);
    let sd_capsule = min(sd0, sd1);
    let main_mask = 1.0 - smoothstep(0.0, 1.0, sd_capsule);

    // --- sub-dashes following capsule perimeter ---
    var sub_mask = 0.0;
    if (main_mask > 0.0) {
        let P = 2.0 * PI * W + 4.0 * L;

        // Continuous arclength around capsule
        let u = capsule_perimeter_u(s0, v_ring, L, W);

        // Compute true signed normal distance scaled by curvature correction
        //  → flatten the band visually on the caps
        var nrm = sd_capsule;
        let cap_weight = smoothstep(L - W, L, abs(s0));
        nrm *= mix(1.0, 0.6, cap_weight); // 0.6 reduces stretching at semicircles

        let band_half = 0.2 * W; // visual width of sub track

        let sub_len   = max(0.001, c.sub_dash_misc.x);
        let sub_space = max(0.001, c.sub_dash_misc.y);
        let sub_round = clamp(c.sub_dash_misc.z, 0.0, 1.0);
        let sub_speed = c.sub_dash_misc.w;

        var sub_dash_px  = sub_len * thickness;
        var sub_space_px = sub_space * thickness;
        var sub_period   = sub_dash_px + sub_space_px;

        let n_sub = max(1.0, floor(P / sub_period + 0.5));
        let sub_period_adj = P / n_sub;
        sub_dash_px  = sub_dash_px  * (sub_period_adj / sub_period);
        sub_space_px = sub_period_adj - sub_dash_px;
        sub_period   = sub_period_adj;

        let u_scrolled = repeat(u + sub_speed * screen.time * thickness + 0.5 * sub_dash_px, sub_period);
        let su = u_scrolled - 0.5 * sub_dash_px;

        let sub_L = 0.5 * sub_dash_px;
        let sub_W = band_half;
        let sub_rcap = sub_round * sub_W;

        let sds0 = rr_sd(su,             nrm, sub_L, sub_W, sub_rcap);
        let sds1 = rr_sd(su - sub_period, nrm, sub_L, sub_W, sub_rcap);
        let sds  = min(sds0, sds1);

        sub_mask = 1.0 - smoothstep(0.0, 1.0, sds);
    }

    // compose: base dash always visible, sub-dashes on top
    let base_rgb = c.dash_color.rgb;
    let sub_rgb  = c.sub_dash_color.rgb;
    let rgb   = base_rgb * main_mask + sub_rgb * sub_mask;
    let alpha = c.dash_color.a * max(main_mask, sub_mask);

    return vec4<f32>(rgb, alpha);
}