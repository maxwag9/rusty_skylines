// === ui_shape_outline.wgsl ===
// Unified outline shader: mode=0 => circle (dashed), mode=1 => polygon (solid outline)

struct ScreenUniform {
    size: vec2<f32>,
    time: f32,
    enable_dither: u32
};

@group(0) @binding(0)
var<uniform> screen: ScreenUniform;

// Shape params: works for both circle and polygon
struct ShapeParams {
    mode: f32,      // 0.0 = circle, 1.0 = polygon
    vertex_offset: u32,   // polygon vertices start index (ignored for circle)
    vertex_count: u32,    // polygon vertex count (ignored for circle)
    _pad0: u32,

    shape_data: vec4<f32>,     // (cx, cy, radius, thickness_factor)
    dash_color: vec4<f32>,
    dash_misc:  vec4<f32>,     // (dash_len, dash_spacing, dash_roundness, speed)
    sub_dash_color: vec4<f32>,
    sub_dash_misc: vec4<f32>,  // (sub_dash_len, sub_dash_spacing, sub_roundness, sub_speed)
    misc: vec4<f32>           // unchanged, for whatever you already use it for
};

@group(1) @binding(0)
var<storage, read> shapes: array<ShapeParams>;

// All polygon vertices for all instances packed in one buffer.
// Coordinates are in screen-space pixels.
struct UiVertexPoly {
    pos: vec2<f32>,
    data: vec2<f32>, // [roundness_px, polygon_index]
    color: vec4<f32>,
    misc: vec4<f32>, // active, touched_time, is_touched, hash
};

@group(1) @binding(1)
var<storage, read> poly_vertices: array<UiVertexPoly>;

// ==== VERTEX ====
struct VertexInput {
    @location(0) pos: vec2<f32>,   // unit quad vertices (e.g. in [-1, 1])
    @location(1) color: vec4<f32>, // unused here, but kept for compatibility
    @builtin(instance_index) instance: u32,
};

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) local_pos: vec2<f32>, // world/screen-space position in pixels
    @location(1) shape_index: u32,
    @location(2) center: vec2<f32>,
};
const MAX_POLY_VERTS: u32 = 64u;
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let s = shapes[in.instance];
    let sd = s.shape_data;

    let center = sd.xy;
    let base_radius = sd.z;
    let thickness_factor = sd.w;

    // Slightly expand for outline thickness, same logic as before.
    let radius = base_radius + thickness_factor * base_radius;

    // 'in.pos' is assumed to be a unit quad in some range (e.g. [-1, 1]).
    // We scale by 'radius' around 'center' for both circle and polygon.
    let world = center + in.pos * radius;

    let x = (world.x / screen.size.x) * 2.0 - 1.0;
    let y = 1.0 - (world.y / screen.size.y) * 2.0;

    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.local_pos = world;
    out.shape_index = in.instance;
    out.center = center;


    return out;
}

// ==== HELPERS ====

const PI: f32 = 3.14159265;
fn repeat(x: f32, period: f32) -> f32 {
    return x - floor(x / period) * period;
}

// Rounded-rectangle / capsule SDF; sharp corners when r == 0.
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

// ======================================================================
// Fragment
// ======================================================================

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let s = shapes[in.shape_index];
    let mode = s.mode;

    // common parameters (used by both circle and polygon)
    let sd      = s.shape_data;
    let radius  = sd.z;
    let thickness_factor = sd.w;

    let thickness = thickness_factor * radius;
    let W = thickness;

    // Dash parameters - now interpreted as fractions of circumference/perimeter
    let dash_len     = max(0.0001, s.dash_misc.x);   // fraction of circumference
    let dash_space   = max(0.0001, s.dash_misc.y);   // fraction of circumference
    let dash_round   = clamp(s.dash_misc.z, 0.0, 1.0);
    let speed        = s.dash_misc.w;                 // revolutions per second

    // Sub-dash parameters - fractions of capsule perimeter
    let sub_len      = max(0.0001, s.sub_dash_misc.x);
    let sub_space    = max(0.0001, s.sub_dash_misc.y);
    let sub_round    = clamp(s.sub_dash_misc.z, 0.0, 1.0);
    let sub_speed    = s.sub_dash_misc.w;             // loops around capsule per second

    // =========================
    // MODE 0: circle outline
    // =========================
    if (mode < 0.5) {
        let center = s.shape_data.xy;
        let p = in.local_pos - center;
        let dist = length(p);

        let v_ring = dist - radius;

        // angular coordinate -> arc length
        var ang = atan2(p.y, p.x);
        if (ang < 0.0) {
            ang = ang + 2.0 * PI;
        }
        let arc_px = ang * radius;
        let circ_px = 2.0 * PI * radius;

        // Main dash pattern - NORMALIZED TO CIRCUMFERENCE
        // dash_len and dash_space are now fractions of the full circle
        var dash_px   = dash_len   * circ_px;
        var space_px  = dash_space * circ_px;
        var period_px = dash_px + space_px;

        // Snap to integer number of periods around the circle
        let n = max(1.0, floor(circ_px / period_px + 0.5));
        let period_adj = circ_px / n;
        dash_px  = dash_px  * (period_adj / period_px);
        space_px = period_adj - dash_px;
        period_px = period_adj;

        // Scroll speed normalized to circumference (revolutions per second)
        let scroll_px = speed * screen.time * circ_px;
        let tcoord = repeat(arc_px + scroll_px + 0.5 * dash_px, period_px);
        let s0 = tcoord - 0.5 * dash_px;

        let L = 0.5 * dash_px;
        let rc = dash_round * W;

        let sd0 = rr_sd(s0,              v_ring, L, W, rc);
        let sd1 = rr_sd(s0 - period_px,  v_ring, L, W, rc);
        let sd_capsule = min(sd0, sd1);

        let aa_main = max(0.75, fwidth(sd_capsule) * 0.75);
        let main_mask = 1.0 - smoothstep(0.0, aa_main, sd_capsule);

        // Sub-dashes along capsule perimeter - NORMALIZED TO CAPSULE PERIMETER
        var sub_mask = 0.0;
        if (main_mask > 0.0) {
            let P_capsule = 2.0 * PI * W + 4.0 * L;

            // sub_len and sub_space are fractions of the capsule perimeter
            var sub_dash_px  = sub_len   * P_capsule;
            var sub_space_px = sub_space * P_capsule;
            var sub_period   = sub_dash_px + sub_space_px;

            // Snap to integer number of sub-dashes around the capsule
            let n_sub = max(1.0, floor(P_capsule / sub_period + 0.5));
            let sub_period_adj = P_capsule / n_sub;
            sub_dash_px  = sub_dash_px  * (sub_period_adj / sub_period);
            sub_space_px = sub_period_adj - sub_dash_px;
            sub_period   = sub_period_adj;

            // Compute position along capsule perimeter
            var u = 0.0;
            if (s0 <= -L) {
                let theta = atan2(v_ring, s0 + L);
                u = (0.5 * PI - theta) * W;
            } else if (s0 >= L) {
                let theta = atan2(v_ring, s0 - L);
                u = PI * W + 2.0 * L + (theta + 0.5 * PI) * W;
            } else if (v_ring < 0.0) {
                u = PI * W + (s0 + L);
            } else {
                u = 2.0 * PI * W + 2.0 * L + (L - s0);
            }

            // Sub-dash scroll normalized to capsule perimeter (loops per second)
            let u_scrolled = repeat(u + sub_speed * screen.time * P_capsule + 0.5 * sub_dash_px,
                                     sub_period);
            let su = u_scrolled - 0.5 * sub_dash_px;

            let sub_L  = 0.5 * sub_dash_px;
            let sub_W  = 0.2 * W;
            let sub_rc = sub_round * sub_W;

            let vb = sd_capsule * sign(v_ring);

            let sds0 = rr_sd(su,             vb, sub_L, sub_W, sub_rc);
            let sds1 = rr_sd(su - sub_period, vb, sub_L, sub_W, sub_rc);
            let sds  = min(sds0, sds1);

            let aa_sub = max(0.75, fwidth(sds) * 0.75);
            sub_mask = 1.0 - smoothstep(0.0, aa_sub, sds);
        }

        let base_rgb = s.dash_color.rgb;
        let sub_rgb  = s.sub_dash_color.rgb;
        let m_main   = clamp(main_mask, 0.0, 1.0);
        let m_sub    = clamp(sub_mask,  0.0, 1.0);

        let rgb   = mix(base_rgb, sub_rgb, m_sub);
        let alpha = mix(s.dash_color.a * m_main, s.sub_dash_color.a * m_sub, m_sub);

        return vec4<f32>(rgb, alpha);
    }

    // =========================
    // MODE 1: polygon outline
    // =========================
    let ofs      = s.vertex_offset;
    let cnt_raw  = s.vertex_count;
    let cnt      = min(cnt_raw, MAX_POLY_VERTS);

    if (cnt < 2u) {
        return vec4<f32>(0.0);
    }

    // centroid
    var centrid = vec2<f32>(0.0);
    for (var i: u32 = 0u; i < cnt; i = i + 1u) {
        centrid = centrid + poly_vertices[ofs + i].pos;
    }
    centrid = centrid / f32(cnt);

    // copy + sort vertices by angle around centroid
    var vtx : array<vec2<f32>, MAX_POLY_VERTS>;
    var ang : array<f32,       MAX_POLY_VERTS>;
    var ord : array<u32,       MAX_POLY_VERTS>;

    for (var i: u32 = 0u; i < cnt; i = i + 1u) {
        let p = poly_vertices[ofs + i].pos;
        vtx[i] = p;
        ang[i] = atan2(p.y - centrid.y, p.x - centrid.x);
        ord[i] = i;
    }

    // insertion sort ord[] by ang[]
    for (var i: u32 = 1u; i < cnt; i = i + 1u) {
        var j = i;
        loop {
            if (j == 0u) { break; }
            let a0 = ang[ord[j - 1u]];
            let a1 = ang[ord[j]];
            if (a0 <= a1) { break; }
            let tmp = ord[j];
            ord[j] = ord[j - 1u];
            ord[j - 1u] = tmp;
            j = j - 1u;
        }
    }

    let p = in.local_pos;

    // 1) edge-based geometry: perimeter, best_u, best_v
    var P        = 0.0;
    var best_u   = 0.0;
    var best_v   = 0.0;
    var best_ad  = 1e9;
    var acc_len  = 0.0;

    // also accumulate min distance for polygon SDF magnitude
    var min_d2 = 1e20;

    for (var e: u32 = 0u; e < cnt; e = e + 1u) {
        let i0 = ord[e];
        let i1 = ord[(e + 1u) % cnt];
        let a  = vtx[i0];
        let b  = vtx[i1];
        let ab = b - a;
        let seg_len = length(ab);

        if (seg_len > 0.0) {
            P = P + seg_len;

            let dir = ab / seg_len;
            let ap  = p - a;
            let t_line = dot(ap, dir);
            let t = clamp(t_line, 0.0, seg_len);
            let proj = a + dir * t;

            let n = vec2<f32>(-dir.y, dir.x);
            let d_signed = dot(p - proj, n);
            let d_abs    = abs(d_signed);

            if (d_abs < best_ad) {
                best_ad = d_abs;
                best_v  = d_signed;
                best_u  = acc_len + t;
            }

            let diff = p - proj;
            let d2 = dot(diff, diff);
            if (d2 < min_d2) {
                min_d2 = d2;
            }
        }

        acc_len = acc_len + seg_len;
    }

    if (P <= 0.0) {
        return vec4<f32>(0.0);
    }

    // 2) polygon SDF sign using ray-cast; magnitude from min_d2 above
    var inside = false;
    for (var e: u32 = 0u; e < cnt; e = e + 1u) {
        let i0 = ord[e];
        let i1 = ord[(e + 1u) % cnt];
        let a = vtx[i0];
        let b = vtx[i1];

        let cond1 = (a.y > p.y);
        let cond2 = (b.y > p.y);
        if (cond1 != cond2) {
            let t = (p.y - a.y) / (b.y - a.y);
            let x_int = a.x + t * (b.x - a.x);
            if (p.x < x_int) {
                inside = !inside;
            }
        }
    }

    let dist_poly = sqrt(min_d2);
    var sd_poly = dist_poly;
    if (inside) {
        sd_poly = -dist_poly;
    }

    // 3) ring coordinates:
    //    - v_ring from edge SDF (gives nice straight segments)
    //    - arc_px / circ_px identical to circle logic
    let v_ring = best_v;
    let arc_px = best_u;
    let circ_px = P;  // perimeter for polygon

    // --------- main dash pattern - NORMALIZED TO PERIMETER ----------
    var dash_px   = dash_len   * circ_px;
    var space_px  = dash_space * circ_px;
    var period_px = dash_px + space_px;

    let n = max(1.0, floor(circ_px / period_px + 0.5));
    let period_adj = circ_px / n;
    dash_px  = dash_px  * (period_adj / period_px);
    space_px = period_adj - dash_px;
    period_px = period_adj;

    // Scroll speed normalized to perimeter
    let scroll_px = speed * screen.time * circ_px;
    let tcoord = repeat(arc_px + scroll_px + 0.5 * dash_px, period_px);
    let s0 = tcoord - 0.5 * dash_px;

    let L = 0.5 * dash_px;
    let rc = dash_round * W;

    let sd0 = rr_sd(s0,              v_ring, L, W, rc);
    let sd1 = rr_sd(s0 - period_px,  v_ring, L, W, rc);
    let sd_capsule = min(sd0, sd1);

    let aa_main = max(0.75, fwidth(sd_capsule) * 0.75);
    var main_mask = 1.0 - smoothstep(0.0, aa_main, sd_capsule);

    // 4) clip using polygon band: removes corner "extensions"
    let band = abs(sd_poly) - W;
    let aa_band = max(0.75, fwidth(sd_poly) * 1.5);
    let poly_mask = 1.0 - smoothstep(0.0, aa_band, band);

    main_mask = main_mask * poly_mask;

    // --------- sub-dashes - NORMALIZED TO CAPSULE PERIMETER ----------
    var sub_mask = 0.0;
    if (main_mask > 0.0) {
        let P_capsule = 2.0 * PI * W + 4.0 * L;

        var sub_dash_px  = sub_len   * P_capsule;
        var sub_space_px = sub_space * P_capsule;
        var sub_period   = sub_dash_px + sub_space_px;

        let n_sub = max(1.0, floor(P_capsule / sub_period + 0.5));
        let sub_period_adj = P_capsule / n_sub;
        sub_dash_px  = sub_dash_px  * (sub_period_adj / sub_period);
        sub_space_px = sub_period_adj - sub_dash_px;
        sub_period   = sub_period_adj;

        var u = 0.0;
        if (s0 <= -L) {
            let theta = atan2(v_ring, s0 + L);
            u = (0.5 * PI - theta) * W;
        } else if (s0 >= L) {
            let theta = atan2(v_ring, s0 - L);
            u = PI * W + 2.0 * L + (theta + 0.5 * PI) * W;
        } else if (v_ring < 0.0) {
            u = PI * W + (s0 + L);
        } else {
            u = 2.0 * PI * W + 2.0 * L + (L - s0);
        }

        // Sub-dash scroll normalized to capsule perimeter
        let u_scrolled = repeat(u + sub_speed * screen.time * P_capsule + 0.5 * sub_dash_px,
                                 sub_period);
        let su = u_scrolled - 0.5 * sub_dash_px;

        let sub_L  = 0.5 * sub_dash_px;
        let sub_W  = 0.2 * W;
        let sub_rc = sub_round * sub_W;

        let vb = sd_capsule * sign(v_ring);

        let sds0 = rr_sd(su,             vb, sub_L, sub_W, sub_rc);
        let sds1 = rr_sd(su - sub_period, vb, sub_L, sub_W, sub_rc);
        let sds  = min(sds0, sds1);

        let aa_sub = max(0.75, fwidth(sds) * 0.75);
        sub_mask = 1.0 - smoothstep(0.0, aa_sub, sds);

        // also respect polygon band for sub-dashes
        sub_mask = sub_mask * poly_mask;
    }

    let base_rgb = s.dash_color.rgb;
    let sub_rgb  = s.sub_dash_color.rgb;
    let m_main   = clamp(main_mask, 0.0, 1.0);
    let m_sub    = clamp(sub_mask,  0.0, 1.0);

    let rgb   = mix(base_rgb, sub_rgb, m_sub);
    let alpha = mix(s.dash_color.a * m_main, s.sub_dash_color.a * m_sub, m_sub);

    return vec4<f32>(rgb, alpha);
}