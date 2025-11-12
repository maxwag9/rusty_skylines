// === ui_handle_sdf.wgsl ===

struct ScreenUniform {
    size: vec2<f32>,
    time: f32,
    enable_dither: u32,
    mouse: vec2<f32>,
};
@group(0) @binding(0)
var<uniform> screen: ScreenUniform;

// Semantics (ALL ratios are absolute to the circle, not relative to main):
// - handle_misc.x = length_ratio (0..1) of full circumference (arc coverage)
// - handle_misc.y = width_ratio  (0..1) of radius (radial thickness)
// - handle_misc.z = roundness    (0..1) cap roundness, automatically clamped to not grow size
// - sub_handle_misc mirrors the same semantics for the sub handle.
struct HandleParams {
    // cx, cy, radius, mode (1=circle-follow, 0=fixed-east)
    center_iscircle_border: vec4<f32>,
    handle_color: vec4<f32>,
    handle_misc: vec4<f32>,       // (len_ratio_circumf, width_ratio_radius, roundness, _)
    sub_handle_color: vec4<f32>,
    sub_handle_misc: vec4<f32>,   // (len_ratio_circumf, width_ratio_radius, roundness, _)
    misc: vec4<f32>,
};
@group(1) @binding(0)
var<storage, read> handles: array<HandleParams>;

struct VertexInput {
    @location(0) pos: vec2<f32>,          // fullscreen unit quad [-1..1]
    @builtin(instance_index) instance: u32,
};
struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) local_pos: vec2<f32>,    // world-space (pixel) position
    @location(1) handle_index: u32,
};

const PI: f32 = 3.141592653589793;

// Compute a conservative extent so the quad covers both main and sub handle fully
fn compute_extent(radius: f32, main_len_ratio: f32, main_w_ratio: f32, sub_len_ratio: f32, sub_w_ratio: f32) -> f32 {
    // tangential half span ~ 0.5 * angle_extent * radius
    let ang_main = clamp(main_len_ratio, 0.0, 1.0) * (2.0 * PI);
    let ang_sub  = clamp(sub_len_ratio , 0.0, 1.0) * (2.0 * PI);
    let tan_half_main = 0.5 * ang_main * radius;
    let tan_half_sub  = 0.5 * ang_sub  * radius;

    // radial half thickness
    let half_w_main = 0.5 * radius * clamp(main_w_ratio, 0.0, 1.0);
    let half_w_sub  = 0.5 * radius * clamp(sub_w_ratio , 0.0, 1.0);

    // circle center to outermost pixel
    // add a little guard factor
    let span_tan = max(tan_half_main, tan_half_sub);
    let span_rad = max(half_w_main, half_w_sub);
    return radius + span_rad + span_tan + 4.0;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let h = handles[in.instance];
    let cr = h.center_iscircle_border;
    let radius = cr.z;

    let extent = compute_extent(
        radius,
        h.handle_misc.x, h.handle_misc.y,
        h.sub_handle_misc.x, h.sub_handle_misc.y
    );

    let center = cr.xy;
    // expand a quad around the center by 'extent'
    let world = center + in.pos * extent;

    // NDC
    let x = (world.x / screen.size.x) * 2.0 - 1.0;
    let y = 1.0 - (world.y / screen.size.y) * 2.0;

    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.local_pos = world; // pass world pixel coords to FS
    out.handle_index = in.instance;
    return out;
}

// Signed distance of an angular strip centered at base_angle with angular half-extent a_half
// and radial half-thickness half_w, at ring radius R. This is a rounded-rect in (tangential, radial) space.
fn sd_handle_strip(
    p: vec2<f32>,
    center: vec2<f32>,
    R: f32,
    base_angle: f32,
    a_half: f32,      // angular half-extent in radians
    half_w: f32,      // radial half thickness in pixels
    roundness: f32    // 0..1, caps radius clamped to not exceed extents
) -> f32 {
    let rel = p - center;
    let r   = length(rel);
    var ang = atan2(rel.y, rel.x);

    // wrap angle to [-PI, PI] relative to base_angle
    var da = ang - base_angle;
    if (da >  PI) { da -= 2.0 * PI; }
    if (da < -PI) { da += 2.0 * PI; }

    // map to rectangular distance space: (tangential, radial)
    // tangential in pixels by multiplying angle delta by R
    let t = abs(da) * R - a_half * R;
    let s = abs(r - R)  - half_w;

    // Rounded caps: cap radius limited by available half-sizes so geometry doesn't "grow"
    // Cap cannot exceed both half extents simultaneously; pick min of them for stability
    let cap_max = max(1e-4, min(a_half * R, half_w));
    let rcap = clamp(roundness, 0.0, 1.0) * cap_max;

    // Standard rounded-rect SDF
    let q = vec2<f32>(t, s);
    let outside = max(q, vec2<f32>(0.0));
    let d_out = length(outside) - rcap;
    let d_in  = min(max(q.x, q.y), 0.0);
    return d_out + d_in;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let h  = handles[in.handle_index];
    let cr = h.center_iscircle_border;
    let C  = cr.xy;
    let R  = max(cr.z, 0.0001);
    let mode = cr.w;

    // Mouse vector
    let to_mouse = screen.mouse - C;
    let d_mouse  = length(to_mouse);
    let dead_zone = max(R * 0.08, 1.0);

    var dir = vec2<f32>(1.0, 0.0);
    if (mode > 0.5 && d_mouse > dead_zone) {
        dir = to_mouse / d_mouse;
    }
    let base_angle = atan2(dir.y, dir.x);

    // === MAIN handle ratios (absolute to circle) ===
    let len_ratio   = clamp(h.handle_misc.x, 0.0, 1.0); // fraction of circumference
    let width_ratio = clamp(h.handle_misc.y, 0.0, 1.0); // fraction of radius
    let round_main  = clamp(h.handle_misc.z, 0.0, 1.0);

    let angle_extent = len_ratio * (2.0 * PI);
    let a_half       = 0.5 * angle_extent;
    let half_w       = 0.5 * R * width_ratio;

    // === SUB handle ratios ===
    let sub_len_ratio   = clamp(h.sub_handle_misc.x, 0.0, 1.0);
    let sub_width_ratio = clamp(h.sub_handle_misc.y, 0.0, 1.0);
    let round_sub       = clamp(h.sub_handle_misc.z, 0.0, 1.0);

    let sub_angle_extent = sub_len_ratio * (2.0 * PI);
    let sub_a_half       = 0.5 * sub_angle_extent;
    let sub_half_w       = 0.5 * R * sub_width_ratio;

    // Signed distances
    let d_main = sd_handle_strip(in.local_pos, C, R, base_angle, a_half, half_w, round_main);
    let d_sub  = sd_handle_strip(in.local_pos, C, R, base_angle, sub_a_half, sub_half_w, round_sub);

    // AA
    let aa_main = max(0.75, fwidth(d_main));
    let aa_sub  = max(0.75, fwidth(d_sub));

    let mask_main = 1.0 - smoothstep(0.0, aa_main, d_main);
    let mask_sub  = 1.0 - smoothstep(0.0, aa_sub,  d_sub);

    // === Accurate hover detection ===
    // Compute polar angle of mouse and see if it's within the arc range
    let mouse_ang = atan2(to_mouse.y, to_mouse.x);
    var da = mouse_ang - base_angle;
    if (da > PI)  { da -= 2.0 * PI; }
    if (da < -PI) { da += 2.0 * PI; }

    // Mouse hover test: must be near the circle radius *and* inside angular half-span
    let radial_diff = abs(d_mouse - R);
    let ang_diff = abs(da) * R;
    let hover_radial_range = half_w * 1.4;        // generous radial hover zone
    let hover_tangential_range = a_half * R * 1.1; // within angular arc

    let hover_inside = step(abs(da), a_half * 1.1) * step(radial_diff, hover_radial_range);
    // Smooth hover fade near edges
    let hover_fade_r = 1.0 - smoothstep(hover_radial_range * 0.7, hover_radial_range, radial_diff);
    let hover_fade_a = 1.0 - smoothstep(a_half * 0.9, a_half * 1.1, abs(da));
    let hover = clamp(hover_inside * hover_fade_r * hover_fade_a, 0.0, 1.0);

    // === Hold feedback ===
    let is_down = clamp(h.misc.z, 0.0, 1.0);

    // Combined feedback intensities
    let hover_feedback = hover * (1.0 - is_down);
    let press_feedback = is_down;

    // === Visual response scaling ===
    let shrink_factor = mix(1.0, 0.9, hover_feedback + 0.15 * press_feedback);
    let color_brighten = -0.15 * hover_feedback;
    let color_darken = 0.4 * press_feedback;

    // Shrink visually — re-evaluate SDF with scaled thickness
    let d_main_s = sd_handle_strip(in.local_pos, C, R, base_angle, a_half, half_w * shrink_factor, round_main);
    let d_sub_s  = sd_handle_strip(in.local_pos, C, R, base_angle, sub_a_half, sub_half_w * shrink_factor, round_sub);

    let mask_main_s = 1.0 - smoothstep(0.0, aa_main, d_main_s);
    let mask_sub_s  = 1.0 - smoothstep(0.0, aa_sub,  d_sub_s);

    // Distance-based fade
    let fade_start = R * 1.2;
    let fade_end   = R * 12.0;
    let fade = 1.0 - smoothstep(fade_start, fade_end, d_mouse);

    // === Combine main & sub (sub replaces main)
    let main_masked = mask_main_s * (1.0 - mask_sub_s);

    var rgb = mix(h.handle_color.rgb, h.sub_handle_color.rgb, mask_sub_s);

    // Apply interaction colors
    rgb = mix(rgb, vec3<f32>(1.0), color_brighten); // hover: brighten
    rgb = mix(rgb, rgb * 0.6, color_darken);        // press: darken
    let luma = dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
    rgb = mix(vec3<f32>(luma), rgb, 1.25);
    let alpha = (h.handle_color.a * main_masked + h.sub_handle_color.a * mask_sub_s) * fade;

    return vec4<f32>(rgb, alpha);
}
