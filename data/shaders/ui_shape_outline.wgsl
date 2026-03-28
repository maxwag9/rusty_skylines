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
    misc: vec4<f32>
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

    var world: vec2<f32>;

    if (s.mode > 0.5) {
        // Polygon mode: compute bounding box from polygon vertices
        let ofs = s.vertex_offset;
        let cnt = min(s.vertex_count, MAX_POLY_VERTS);

        // Initialize bounds
        var min_pos = vec2<f32>(100000.0);
        var max_pos = vec2<f32>(-1e20);

        // Find bounding box of all polygon vertices
        for (var i: u32 = 0u; i < cnt; i = i + 1u) {
            let p = poly_vertices[ofs + i].pos;
            min_pos = min(min_pos, p);
            max_pos = max(max_pos, p);
        }

        let thickness = thickness_factor;
        min_pos = min_pos;// - vec2<f32>(thickness + 1.0); // +1 for AA margin
        max_pos = max_pos;// + vec2<f32>(thickness + 1.0);

        // Map unit quad [-1, 1] to bounding box
        let t = (in.pos + 1.0) * 0.5; // convert to [0, 1]
        world = mix(min_pos, max_pos, t);
    } else {
        // Circle mode: expand quad around center
        let radius = base_radius + thickness_factor * base_radius;
        world = center + in.pos * radius;
    }

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
    discard;
    return vec4<f32>(1.0, 0.0, 0.0, 0.00);
    let ofs = s.vertex_offset;
    let cnt = min(s.vertex_count, MAX_POLY_VERTS);

    if (cnt < 2u) {
        return vec4<f32>(0.0);
    }

    // centroid
    var centro = vec2<f32>(0.0);
    for (var i: u32 = 0u; i < cnt; i = i + 1u) {
        centro = centro + poly_vertices[ofs + i].pos;
    }
    centro = centro / f32(cnt);

    // copy + sort
    var vtx : array<vec2<f32>, MAX_POLY_VERTS>;
    var ang : array<f32, MAX_POLY_VERTS>;
    var ord : array<u32, MAX_POLY_VERTS>;

    for (var i: u32 = 0u; i < cnt; i = i + 1u) {
        let p = poly_vertices[ofs + i].pos;
        vtx[i] = p;
        ang[i] = atan2(p.y - centro.y, p.x - centro.x);
        ord[i] = i;
    }

    // insertion sort
    for (var i: u32 = 1u; i < cnt; i = i + 1u) {
        var j = i;
        loop {
            if (j == 0u) { break; }
            if (ang[ord[j - 1u]] <= ang[ord[j]]) { break; }
            let tmp = ord[j];
            ord[j] = ord[j - 1u];
            ord[j - 1u] = tmp;
            j = j - 1u;
        }
    }

    let p = in.local_pos;

    // distance to edges
    var min_d2 = 1e20;

    for (var e: u32 = 0u; e < cnt; e = e + 1u) {
        let a = vtx[ord[e]];
        let b = vtx[ord[(e + 1u) % cnt]];
        let ab = b - a;

        let t = clamp(dot(p - a, ab) / dot(ab, ab), 0.0, 1.0);
        let proj = a + t * ab;

        let diff = p - proj;
        let d2 = dot(diff, diff);

        if (d2 < min_d2) {
            min_d2 = d2;
        }
    }

    // inside test
    var inside = false;
    for (var e: u32 = 0u; e < cnt; e = e + 1u) {
        let a = vtx[ord[e]];
        let b = vtx[ord[(e + 1u) % cnt]];

        if ((a.y > p.y) != (b.y > p.y)) {
            let t = (p.y - a.y) / (b.y - a.y);
            let x_int = a.x + t * (b.x - a.x);
            if (p.x < x_int) {
                inside = !inside;
            }
        }
    }

    var sdd = sqrt(min_d2);
    if (inside) {
        sdd = -sdd;
    }

    let d = abs(sdd) - thickness;

    let aa = max(0.75, fwidth(sdd));
    let alpha = 1.0 - smoothstep(0.0, aa, d);

    return vec4<f32>(1.0, 0.0, 0.0, alpha);
}
