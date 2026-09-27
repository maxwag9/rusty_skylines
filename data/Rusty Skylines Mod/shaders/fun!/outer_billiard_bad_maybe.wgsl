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

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

const BALL_RADIUS: f32 = 0.018;
const MAX_STEPS: u32 = 50u;

fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + r;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

fn palette(t: f32) -> vec3<f32> {
    let a = vec3<f32>(0.25, 0.12, 0.35);
    let b = vec3<f32>(0.35, 0.25, 0.30);
    let c = vec3<f32>(1.0, 0.8, 0.7);
    let d = vec3<f32>(0.15, 0.05, 0.35);

    return a + b * cos(TAU * (c * t + d));
}

const POLYGON_RADIUS: f32 = 0.30;
const SHAPE_DURATION: f32 = 2.75;
const MAX_VERTICES: u32 = 10u;

fn hash11(n: f32) -> f32 {
    return fract(sin(n * 127.1) * 43758.5453);
}

fn shape_count(segment: f32) -> u32 {
    return 3u + u32(floor(hash11(segment * 19.37) * 8.0));
}

fn current_vertex_count() -> u32 {
    let segment = floor(screen.time / SHAPE_DURATION);
    return shape_count(segment);
}

fn pentagon_vertex(index: u32) -> vec2<f32> {
    let segment = floor(screen.time / SHAPE_DURATION);
    let local_time = fract(screen.time / SHAPE_DURATION);

    let count = shape_count(segment);
    let next_count = shape_count(segment + 1.0);

    let i = f32(index);

    let angle_a =
        PI * 0.5 +
        TAU * i / f32(count);

    let angle_b =
        PI * 0.5 +
        TAU * i / f32(next_count);

    let radius_a =
        POLYGON_RADIUS *
        (
            0.82 +
            hash11(segment * 31.71 + i * 17.23) * 0.36
        );

    let radius_b =
        POLYGON_RADIUS *
        (
            0.82 +
            hash11((segment + 1.0) * 31.71 + i * 17.23) * 0.36
        );

    let morph =
        smoothstep(0.72, 1.0, local_time);

    let angle =
        mix(angle_a, angle_b, morph);

    let radius =
        mix(radius_a, radius_b, morph);

    return vec2<f32>(
        cos(angle),
        sin(angle)
    ) * radius;
}

fn polygon_sdf(p: vec2<f32>) -> f32 {
    let count = current_vertex_count();

    var result = -1000000.0;

    for (var i: u32 = 0u; i < MAX_VERTICES; i = i + 1u) {
        if i >= count {
            break;
        }

        let next = (i + 1u) % count;

        let a = pentagon_vertex(i);
        let b = pentagon_vertex(next);

        let edge = normalize(b - a);
        let normal = vec2<f32>(edge.y, -edge.x);

        let d = dot(p - a, normal);

        result = max(result, d);
    }

    return result;
}

fn closest_pentagon_vertex(
    p: vec2<f32>
) -> vec2<f32> {
    let count = current_vertex_count();

    var closest = pentagon_vertex(0u);
    var closest_dist = 1000000.0;

    for (var i: u32 = 0u; i < MAX_VERTICES; i = i + 1u) {
        if i >= count {
            break;
        }

        let vertex = pentagon_vertex(i);
        let delta = p - vertex;
        let dist = dot(delta, delta);

        if dist < closest_dist {
            closest_dist = dist;
            closest = vertex;
        }
    }

    return closest;
}

fn closest_pentagon_vertex_index(
    p: vec2<f32>
) -> u32 {
    let count = current_vertex_count();

    var closest = 0u;
    var closest_dist = 1000000.0;

    for (var i: u32 = 0u; i < MAX_VERTICES; i = i + 1u) {
        if i >= count {
            break;
        }

        let vertex = pentagon_vertex(i);
        let delta = p - vertex;
        let dist = dot(delta, delta);

        if dist < closest_dist {
            closest_dist = dist;
            closest = i;
        }
    }

    return closest;
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
    let mouse_local = inverse_rot * (screen.mouse - rect.center);

    let ndc = (world / screen.size) * 2.0 - 1.0;

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
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let max_round = min(
        input.rect_half_size.x,
        input.rect_half_size.y
    );

    let radius = input.roundness * max_round;

    let d = sd_rounded_box(
        input.local_pos,
        input.rect_half_size,
        radius
    );

    let aa = max(fwidth(d), 0.75);
    let outer_alpha = 1.0 - smoothstep(-aa, aa, d);

    let inner_half = max(
        input.rect_half_size - input.border_thickness,
        vec2<f32>(0.0)
    );

    let inner_radius = max(
        radius - input.border_thickness,
        0.0
    );

    let d_inner = sd_rounded_box(
        input.local_pos,
        inner_half,
        inner_radius
    );

    let inner_alpha =
        1.0 - smoothstep(-aa, aa, d_inner);

    let border_mask =
        outer_alpha - inner_alpha;

    let uv = input.uv;

    let centered = uv - 0.5;

    let aspect =
        input.rect_half_size.x /
        max(input.rect_half_size.y, 0.001);

    let scale =
        min(
            input.rect_half_size.x,
            input.rect_half_size.y
        );

    var p = vec2<f32>(
        centered.x * 2.0 * input.rect_half_size.x / scale,
        -centered.y * 2.0 * input.rect_half_size.y / scale
    );

    let poly_d = polygon_sdf(p);

    let inside_pentagon =
        1.0 - smoothstep(
            -0.005,
            0.005,
            poly_d
        );

    let ball_clearance =
        max(poly_d - BALL_RADIUS, 0.0);

    var accumulated = 0.0;
    var phase = 0.0;
    var path_position = p;

    for (var i: u32 = 0u; i < MAX_STEPS; i = i + 1u) {
        let vertex_index =
            closest_pentagon_vertex_index(path_position);

        let vertex =
            pentagon_vertex(vertex_index);

        let to_vertex =
            vertex - path_position;

        let distance_to_vertex =
            length(to_vertex);

        if distance_to_vertex <= BALL_RADIUS + 0.0001 {
            break;
        }

        let base =
            to_vertex / distance_to_vertex;

        let perpendicular =
            vec2<f32>(
                -base.y,
                base.x
            );

        let tangent_ratio =
            clamp(
                BALL_RADIUS / distance_to_vertex,
                0.0,
                0.9999
            );

        let tangent_cos =
            sqrt(
                max(
                    1.0 -
                    tangent_ratio *
                    tangent_ratio,
                    0.0
                )
            );

        let direction_a =
            base * tangent_cos +
            perpendicular * tangent_ratio;

        let direction_b =
            base * tangent_cos -
            perpendicular * tangent_ratio;

        let tangent_length =
            sqrt(
                max(
                    distance_to_vertex *
                    distance_to_vertex -
                    BALL_RADIUS *
                    BALL_RADIUS,
                    0.0
                )
            );

        let contact_a =
            path_position +
            direction_a * tangent_length;

        let contact_b =
            path_position +
            direction_b * tangent_length;

        let radial =
            normalize(vertex);

        let score_a =
            dot(
                contact_a - vertex,
                radial
            );

        let score_b =
            dot(
                contact_b - vertex,
                radial
            );

        let use_b =
            score_b > score_a;

        let direction =
            select(
                direction_a,
                direction_b,
                use_b
            );

        let contact =
            path_position +
            direction * tangent_length;

        path_position =
            path_position +
            direction *
            (tangent_length * 2.0);

        let shortness =
            1.0 /
            (1.0 + tangent_length * tangent_length * 3.0);

        accumulated += shortness;

        phase +=
            f32(vertex_index + 1u) *
            0.31 +
            tangent_length * 1.7;

        let contact_glow =
            exp(
                -abs(
                    length(p - contact)
                ) * 18.0
            );

        accumulated +=
            contact_glow *
            0.025;
    }

    let travel =
        length(path_position - p);

    let angle =
        atan2(
            path_position.y,
            path_position.x
        );

    let animated =
        accumulated * 0.19 +
        travel * 0.33 +
        angle * 0.22 +
        phase * 0.035 +
        screen.time * 0.035;

    var fill =
        palette(animated);

    let turbulence =
        sin(
            travel * 15.0 -
            phase * 2.0 +
            screen.time * 0.7
        );

    fill *=
        0.65 +
        0.20 * turbulence;

    let trajectory_energy =
        1.0 -
        exp(
            -accumulated *
            0.18
        );

    fill +=
        vec3<f32>(
            0.10,
            0.04,
            0.18
        ) *
        trajectory_energy;

    let final_phase =
        sin(
            path_position.x * 11.0 +
            path_position.y * 17.0 +
            phase +
            screen.time * 0.4
        );

    fill +=
        vec3<f32>(
            0.03,
            0.02,
            0.05
        ) *
        final_phase;

    let pentagon_edge =
        1.0 -
        smoothstep(
            0.0,
            0.035,
            abs(poly_d)
        );

    let vertex_glow =
        exp(
            -max(poly_d, 0.0) * 14.0
        );

    let pentagon_color =
        mix(
            vec3<f32>(
                0.015,
                0.008,
                0.025
            ),
            vec3<f32>(
                0.20,
                0.05,
                0.30
            ),
            pentagon_edge
        );

    let glow_color =
        vec3<f32>(
            0.35,
            0.12,
            0.65
        ) *
        vertex_glow *
        0.15;

    fill += glow_color;

    fill = mix(
        fill,
        pentagon_color,
        inside_pentagon
    );

    fill = mix(
        fill,
        input.border_color.rgb,
        border_mask
    );

    let clear_fade =
        smoothstep(
            0.0,
            0.08,
            ball_clearance
        );

    fill *=
        0.82 +
        0.18 * clear_fade;

    let alpha =
        outer_alpha;

    return vec4<f32>(
        fill * alpha,
        alpha
    );
}