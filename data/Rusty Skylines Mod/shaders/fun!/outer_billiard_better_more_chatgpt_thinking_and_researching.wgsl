struct ScreenUniform {
    size: vec2<f32>,
    time: f32,
    enable_dither: u32,
    mouse: vec2<f32>,
}

struct RectGpu {
    center: vec2<f32>,
    half_size: vec2<f32>,
    color: vec4<f32>,
    border_color: vec4<f32>,
    roundness: f32,
    border_thickness: f32,
    rotation: f32,
    fade: f32,
    glow_color: vec4<f32>,
    glow_misc: vec4<f32>,
    misc: vec4<f32>,
    blur: f32,
    depth: f32,
    _pad0: vec2<f32>,
}

@group(1) @binding(0)
var<uniform> screen: ScreenUniform;

@group(2) @binding(0)
var<storage, read> rects: array<RectGpu>;

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
const MAX_VERTICES: u32 = 10u;
const ORBIT_STEPS: u32 = 8u;

struct VertexInput {
    @location(0) pos: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) mouse_local: vec2<f32>,
    @location(3) rect_half_size: vec2<f32>,
    @location(4) roundness: f32,
    @location(5) border_thickness: f32,
    @location(6) color: vec4<f32>,
    @location(7) border_color: vec4<f32>,
}

fn cross2(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

fn hash11(x: f32) -> f32 {
    return fract(sin(x * 127.131) * 43758.5453123);
}

fn value_noise(x: f32) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(hash11(i), hash11(i + 1.0), u);
}

fn smooth_noise(x: f32) -> f32 {
    var result = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;

    for (var i = 0u; i < 4u; i++) {
        result += value_noise(x * frequency) * amplitude;
        frequency *= 2.0;
        amplitude *= 0.5;
    }

    return result;
}

fn palette(t: f32) -> vec3<f32> {
    let a = vec3<f32>(0.12, 0.08, 0.22);
    let b = vec3<f32>(0.30, 0.22, 0.34);
    let c = vec3<f32>(0.85, 0.65, 0.90);
    let d = vec3<f32>(0.18, 0.04, 0.32);
    return a + b * cos(TAU * (c * t + d));
}

fn raw_vertex(index: u32, count: f32, time: f32) -> vec2<f32> {
    let fi = f32(index);

    let base_angle = TAU * fi / max(count, 3.0);

    let angle_noise = smooth_noise(
        time * 0.075 + fi * 5.173
    );

    let angle_wobble =
        (angle_noise - 0.5) * 0.12 +
        sin(time * 0.11 + fi * 1.731) * 0.025;

    let angle = base_angle + angle_wobble;

    let radius_noise = smooth_noise(
        time * 0.062 + fi * 8.417
    );

    let radius =
        0.52 +
        (radius_noise - 0.5) * 0.075 +
        sin(time * 0.083 + fi * 2.317) * 0.018;

    return vec2<f32>(
        cos(angle) * radius,
        sin(angle) * radius
    );
}

fn animated_vertex(index: u32, count: f32, time: f32) -> vec2<f32> {
    let nominal = raw_vertex(index, count, time);

    if (index == 0u) {
        return nominal;
    }

    let previous = raw_vertex(index - 1u, count, time);

    let fi = f32(index);

    let activation = smoothstep(
        fi,
        fi + 1.0,
        count
    );

    return mix(previous, nominal, activation);
}

fn segment_distance(
    p: vec2<f32>,
    a: vec2<f32>,
    b: vec2<f32>
) -> f32 {
    let ab = b - a;
    let denom = max(dot(ab, ab), 0.000001);
    let h = clamp(dot(p - a, ab) / denom, 0.0, 1.0);
    return length(p - (a + ab * h));
}

fn polygon_sdf(
    p: vec2<f32>,
    verts: array<vec2<f32>, MAX_VERTICES>
) -> f32 {
    var min_distance = 1000.0;
    var inside = true;

    for (var i = 0u; i < MAX_VERTICES; i++) {
        let next_i = (i + 1u) % MAX_VERTICES;

        let a = verts[i];
        let b = verts[next_i];
        let edge = b - a;
        let edge_len = length(edge);

        if (edge_len > 0.00005) {
            let d = segment_distance(p, a, b);
            min_distance = min(min_distance, d);

            let side = cross2(edge, p - a);
            inside = inside && side >= 0.0;
        }
    }

    if (inside) {
        return -min_distance;
    }

    return min_distance;
}

fn tangent_vertex(
    point: vec2<f32>,
    verts: array<vec2<f32>, MAX_VERTICES>
) -> vec2<f32> {
    let direction = normalize(point);

    var best = verts[0];
    var best_side = -1000000.0;
    var found = false;

    for (var i = 0u; i < MAX_VERTICES; i++) {
        var prev_i: u32;
        if (i == 0u) {
            prev_i = MAX_VERTICES - 1u;
        } else {
            prev_i = i - 1u;
        }

        let next_i = (i + 1u) % MAX_VERTICES;

        let prev = verts[prev_i];
        let current = verts[i];
        let next = verts[next_i];

        let incoming = current - prev;
        let outgoing = next - current;

        let incoming_len = length(incoming);
        let outgoing_len = length(outgoing);

        if (incoming_len > 0.00005 && outgoing_len > 0.00005) {
            let side_in =
                cross2(incoming, point - prev);

            let side_out =
                cross2(outgoing, point - current);

            let tangent =
                side_in * side_out < 0.0;

            if (tangent) {
                let side =
                    cross2(direction, current - point);

                if (side > best_side) {
                    best_side = side;
                    best = current;
                    found = true;
                }
            }
        }
    }

    if (found) {
        return best;
    }

    var fallback = verts[0];
    var fallback_score = -1000000.0;

    for (var i = 0u; i < MAX_VERTICES; i++) {
        let score =
            cross2(direction, verts[i] - point);

        if (score > fallback_score) {
            fallback_score = score;
            fallback = verts[i];
        }
    }

    return fallback;
}

fn outer_billiard_step(
    point: vec2<f32>,
    verts: array<vec2<f32>, MAX_VERTICES>
) -> vec2<f32> {
    let tangent = tangent_vertex(point, verts);
    return tangent * 2.0 - point;
}

@vertex
fn vs_main(
    input: VertexInput,
    @builtin(instance_index) instance: u32
) -> VertexOutput {
    let rect = rects[instance];

    let c = cos(rect.rotation);
    let s = sin(rect.rotation);

    let rot = mat2x2<f32>(
        c, -s,
        s, c
    );

    let inverse_rot = mat2x2<f32>(
        c, s,
        -s, c
    );

    let local = input.pos * rect.half_size;
    let world = rect.center + rot * local;

    let mouse_local =
        inverse_rot * (screen.mouse - rect.center);

    let ndc =
        (world / screen.size) * 2.0 - 1.0;

    var out: VertexOutput;

    out.position = vec4<f32>(
        ndc.x,
        -ndc.y,
        rect.depth,
        1.0
    );

    out.local_pos = local;
    out.uv = input.pos * 0.5 + 0.5;
    out.mouse_local = mouse_local;
    out.rect_half_size = rect.half_size;
    out.roundness = rect.roundness;
    out.border_thickness = rect.border_thickness;
    out.color = rect.color;
    out.border_color = rect.border_color;

    return out;
}

@fragment
fn fs_main(
    input: VertexOutput
) -> @location(0) vec4<f32> {
    let scale = min(
        input.rect_half_size.x,
        input.rect_half_size.y
    );

    let p = input.local_pos / max(scale, 0.001);

    let t = screen.time;

    let count_noise =
        value_noise(t * 0.060 + 17.0);

    let vertex_count =
        3.0 + count_noise * 7.0;

    var verts: array<vec2<f32>, MAX_VERTICES>;

    for (var i = 0u; i < MAX_VERTICES; i++) {
        verts[i] =
            animated_vertex(
                i,
                vertex_count,
                t
            );
    }

    let d = polygon_sdf(p, verts);

    let aa =
        max(fwidth(d), 0.0025);

    let outer_alpha =
        1.0 - smoothstep(
            -aa,
            aa,
            d
        );

    let border_width =
        max(
            0.008,
            input.border_thickness / max(scale, 0.001)
        );

    let inside_distance =
        max(-d, 0.0);

    let border_mask =
        1.0 - smoothstep(
            border_width,
            border_width + aa * 2.0,
            inside_distance
        );

    let centered = input.uv - 0.5;

    let aspect =
        input.rect_half_size.x /
        max(input.rect_half_size.y, 0.001);

    let aspect_uv =
        vec2<f32>(
            centered.x * aspect,
            centered.y
        );

    let mouse_uv =
        vec2<f32>(
            input.mouse_local.x / scale,
            input.mouse_local.y / scale
        );

    let mouse_distance =
        length(mouse_uv);

    let mouse_dir =
        normalize(
            select(
                vec2<f32>(1.0, 0.0),
                mouse_uv,
                mouse_distance > 0.01
            )
        );

    let animated_angle =
        t * 0.085 +
        value_noise(t * 0.021) * TAU;

    let default_dir =
        vec2<f32>(
            cos(animated_angle),
            sin(animated_angle)
        );

    let start_direction =
        normalize(
            mix(
                default_dir,
                mouse_dir,
                smoothstep(
                    0.0,
                    1.0,
                    min(mouse_distance, 1.0)
                ) * 0.28
            )
        );

    let start_radius =
        1.30 +
        value_noise(t * 0.031 + 23.0) * 0.22;

    var ball =
        start_direction * start_radius;

    var orbit_distance = 1000.0;
    var current_point = ball;

    for (var i = 0u; i < ORBIT_STEPS; i++) {
        let next_point =
            outer_billiard_step(
                current_point,
                verts
            );

        let segment_d =
            segment_distance(
                p,
                current_point,
                next_point
            );

        orbit_distance =
            min(
                orbit_distance,
                segment_d
            );

        current_point = next_point;

        if (length(current_point) > 3.2) {
            current_point =
                normalize(current_point) * 3.2;
        }
    }

    let orbit_width =
        0.012 +
        0.004 *
        sin(t * 0.7);

    let orbit_mask =
        1.0 -
        smoothstep(
            orbit_width,
            orbit_width + 0.012,
            orbit_distance
        );

    let ball_distance =
        length(
            p - current_point
        );

    let ball_mask =
        1.0 -
        smoothstep(
            0.018,
            0.065,
            ball_distance
        );

    let radial =
        length(aspect_uv);

    let angle =
        atan2(aspect_uv.y, aspect_uv.x);

    let shape_field =
        0.5 +
        0.5 * sin(
            angle * 4.0 +
            t * 0.3 +
            radial * 7.0
        );

    var fill =
        palette(
            input.uv.x * 0.35 +
            input.uv.y * 0.25 +
            shape_field * 0.18 +
            t * 0.018
        );

    let center_glow =
        1.0 -
        smoothstep(
            0.0,
            1.15,
            length(p)
        );

    fill +=
        vec3<f32>(
            0.08,
            0.02,
            0.13
        ) *
        center_glow;

    let edge_shimmer =
        sin(
            angle * 9.0 -
            radial * 13.0 +
            t * 0.8
        ) *
        0.025;

    fill += vec3<f32>(
        edge_shimmer,
        edge_shimmer * 0.7,
        edge_shimmer * 1.3
    );

    let billiard_color =
        vec3<f32>(
            0.35,
            0.75,
            1.0
        );

    let orbit_glow =
        orbit_mask *
        (1.0 - smoothstep(
            0.0,
            2.3,
            radial
        ));

    fill =
        mix(
            fill,
            billiard_color,
            orbit_glow * 0.72
        );

    fill +=
        billiard_color *
        ball_mask *
        0.75;

    fill =
        mix(
            fill,
            input.border_color.rgb,
            border_mask * 0.9
        );

    let outer_glow =
        exp(
            -max(d, 0.0) * 18.0
        ) *
        0.16;

    fill +=
        input.border_color.rgb *
        outer_glow;

    let alpha =
        max(
            outer_alpha,
            orbit_mask * 0.5
        );

    return vec4<f32>(
        fill * alpha,
        alpha
    );
}