// === ui_handle_sdf.wgsl ===

struct ScreenUniform {
    size: vec2<f32>,
    time: f32,
    enable_dither: u32,
    mouse: vec2<f32>,
};
@group(0) @binding(0)
var<uniform> screen: ScreenUniform;

struct HandleParams {
    // cx, cy, radius, mode (1=circle-follow, 0=fixed)
    center_iscircle_border: vec4<f32>,
    handle_color: vec4<f32>,
    // (len_ratio, width_ratio, roundness, unused)
    handle_misc: vec4<f32>,
    sub_handle_color: vec4<f32>,
    // (len_ratio_rel_to_main, width_ratio_rel_to_main, roundness, unused)
    sub_handle_misc: vec4<f32>,
    // misc (x,y,z,w free if you later want last_angle, custom deadzone, etc.)
    misc: vec4<f32>,
};
@group(1) @binding(0)
var<storage, read> handles: array<HandleParams>;

struct VertexInput {
    @location(0) pos: vec2<f32>,
    @builtin(instance_index) instance: u32,
};
struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) handle_index: u32,
};

// Vertex: expand quad to cover circle + handle (handle len is ratio of radius)
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let h = handles[in.instance];
    let cr = h.center_iscircle_border;
    let radius = cr.z;

    // handle length is ratio of radius
    let len_ratio = h.handle_misc.x;
    let len_px = radius * len_ratio;

    // generous extent
    let extent = radius + len_px * 2.0;
    let center = cr.xy;
    let world  = center + in.pos * extent;

    let x = (world.x / screen.size.x) * 2.0 - 1.0;
    let y = 1.0 - (world.y / screen.size.y) * 2.0;

    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.local_pos = world;
    out.handle_index = in.instance;
    return out;
}

const PI = 3.14159265;

// Rounded-rectangle SDF (kept for possible future use)
fn rr_sd(p: vec2<f32>, half: vec2<f32>, r: f32) -> f32 {
    if (r <= 0.0001) {
        let d = abs(p) - half;
        return max(d.x, d.y);
    }
    let q = abs(p) - (half - vec2<f32>(r));
    let outside = max(q, vec2<f32>(0.0));
    let d_out = length(outside) - r;
    let d_in = min(max(q.x, q.y), 0.0);
    return d_out + d_in;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let h  = handles[in.handle_index];
    let cr = h.center_iscircle_border;
    let circle_center = cr.xy;
    let radius = cr.z;
    let mode   = cr.w;

    // --- Handle orientation with dead-zone to avoid spin near center ---
    // dead_zone scales with radius (e.g., 8% of radius)
    let to_mouse = screen.mouse - circle_center;
    let d_mouse  = length(to_mouse);
    let dead_zone = max(radius * 0.08, 1.0); // never below 1 px

    var dir = vec2<f32>(1.0, 0.0); // default fixed direction (east)
    if (mode > 0.5) {
        if (d_mouse > dead_zone) {
            dir = to_mouse / d_mouse;
        }
        // else keep default dir; no spin jitter
    }
    let base_angle = atan2(dir.y, dir.x);

    // --- Proportional geometry (ratios of radius) ---
    let len_ratio   = h.handle_misc.x;             // fraction of radius along arc (converted to linear)
    let width_ratio = h.handle_misc.y;             // fraction of radius for radial thickness
    let round       = clamp(h.handle_misc.z, 0.0, 1.0);

    let len_linear  = radius * len_ratio;          // linear arc length at this radius
    let half_w      = 0.5 * radius * width_ratio;  // radial half-thickness

    // fragment polar coords
    let rel = in.local_pos - circle_center;
    let r   = length(rel);
    let ang = atan2(rel.y, rel.x);

    // angle delta in [-PI, PI]
    var da = ang - base_angle;
    if (da >  PI) { da -= 2.0 * PI; }
    if (da < -PI) { da += 2.0 * PI; }

    // strip distances
    let tangential = abs(da) * radius - 0.5 * len_linear;
    let radial     = abs(r - radius)    - half_w;

    // rounded caps
    var dist = max(tangential, radial);
    if (round > 0.0001) {
        let rcap = round * half_w;
        let q = vec2<f32>(tangential, radial);
        let outside = max(q, vec2<f32>(0.0));
        dist = length(outside) - rcap;
        dist = dist + min(max(q.x, q.y), 0.0);
    }

    // AA
    let aa = max(0.75, fwidth(dist) * 0.75);
    let main_mask = 1.0 - smoothstep(0.0, aa, dist);

    // --- Sub-handle: scale exactly like main (ratios relative to main) ---
    let sub_len_ratio_rel   = h.sub_handle_misc.x; // multiply main length
    let sub_width_ratio_rel = h.sub_handle_misc.y; // multiply main width
    let sub_round           = clamp(h.sub_handle_misc.z, 0.0, 1.0);

    let sub_len_linear = len_linear * sub_len_ratio_rel;
    let sub_half_w     = half_w     * sub_width_ratio_rel;

    let tangential_s = abs(da) * radius - 0.5 * sub_len_linear;
    let radial_s     = abs(r - radius)    - sub_half_w;

    var dist_sub = max(tangential_s, radial_s);
    if (sub_round > 0.0001) {
        let rcap = sub_round * sub_half_w;
        let q = vec2<f32>(tangential_s, radial_s);
        let outside = max(q, vec2<f32>(0.0));
        dist_sub = length(outside) - rcap;
        dist_sub = dist_sub + min(max(q.x, q.y), 0.0);
    }

    let aa_sub = max(0.75, fwidth(dist_sub) * 0.75);
    let sub_mask = 1.0 - smoothstep(0.0, aa_sub, dist_sub);

    // --- Distance-based fade (farther mouse -> more fade) ---
    // Choose fade range relative to radius
    let fade_start = radius * 1.2;  // start fading when mouse roughly near the circle
    let fade_end   = radius * 12.0;  // fully faded when quite far
    let fade = 1.0 - smoothstep(fade_start, fade_end, d_mouse);

    // Compose
    let base_rgb = h.handle_color.rgb;
    let sub_rgb  = h.sub_handle_color.rgb;

    // Layer color by sub-mask
    let rgb = mix(base_rgb, sub_rgb, sub_mask);

    // Alpha with masks and fade
    let alpha_main = h.handle_color.a * main_mask;
    let alpha_sub  = h.sub_handle_color.a * sub_mask;
    let alpha = (alpha_main + alpha_sub) * fade;

    return vec4<f32>(rgb, alpha);
}
