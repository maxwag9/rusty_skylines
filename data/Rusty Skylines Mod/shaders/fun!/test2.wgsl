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

fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + r;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

fn palette(t: f32) -> vec3<f32> {
    let tau = 6.28318530718;
    let h = fract(t);

    let r = 0.5 + 0.5 * cos(tau * (h + 0.00));
    let g = 0.5 + 0.5 * cos(tau * (h + 0.18));
    let b = 0.5 + 0.5 * cos(tau * (h + 0.36));

    let color = vec3<f32>(r, g, b);

    return pow(
        color,
        vec3<f32>(0.72)
    );
}

fn hash21(p: vec2<f32>) -> f32 {
    let h = dot(
        p,
        vec2<f32>(127.1, 311.7)
    );

    return fract(
        sin(h) * 43758.5453123
    );
}

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));

    let u = f * f * f * (
        f * (f * 6.0 - 15.0) + 10.0
    );

    return mix(
        mix(a, b, u.x),
        mix(c, d, u.x),
        u.y
    );
}

fn fbm(p: vec2<f32>) -> f32 {
    var q = p;
    var value = 0.0;
    var amplitude = 0.5;

    let rot = mat2x2<f32>(
        0.7648422, -0.6442177,
        0.6442177,  0.7648422
    );

    for (var i = 0; i < 6; i++) {
        value += noise(q) * amplitude;

        q = rot * q * 2.03 +
            vec2<f32>(31.7, 47.3);

        amplitude *= 0.48;
    }

    return value;
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

    let outer_alpha =
        1.0 - smoothstep(-aa, aa, d);

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
    let t = screen.time;

    let centered = uv - 0.5;

    let aspect =
        input.rect_half_size.x /
        max(input.rect_half_size.y, 0.001);

    var p = vec2<f32>(
        centered.x * aspect,
        centered.y
    );

    let mouse_local_uv = vec2<f32>(
        input.mouse_local.x /
            input.rect_half_size.x * 0.5 + 0.5,
        input.mouse_local.y /
            input.rect_half_size.y * 0.5 + 0.5
    );

    let mouse_delta = uv - mouse_local_uv;

    let mouse_p = vec2<f32>(
        mouse_delta.x * aspect,
        mouse_delta.y
    );

    let mouse_dist = length(mouse_p);

    let mouse_glow =
        exp(-mouse_dist * mouse_dist * 9.0);

    let mouse_ring =
        exp(-abs(mouse_dist - 0.24) * 35.0);

    let flow_1 = fbm(
        p * 1.65 +
        vec2<f32>(
            t * 0.045,
            -t * 0.032
        )
    );

    let flow_2 = fbm(
        p * 3.4 -
        vec2<f32>(
            t * 0.025,
            t * 0.05
        ) +
        flow_1
    );

    let warp = vec2<f32>(
        flow_1 - 0.5,
        flow_2 - 0.5
    );

    p += warp * 0.34;

    let angle = atan2(p.y, p.x);
    let radius_p = length(p);

    let spiral =
        angle * 0.42 +
        radius_p * 2.7 -
        t * 0.09;

    let wave_a = sin(
        spiral * 5.5 +
        flow_2 * 4.0
    );

    let wave_b = sin(
        spiral * 11.0 -
        flow_1 * 7.0 +
        t * 0.18
    );

    let interference =
        wave_a * 0.5 +
        wave_b * 0.25;

    let depth =
        smoothstep(
            1.25,
            0.0,
            radius_p
        );

    let hue =
        flow_1 * 0.32 +
        flow_2 * 0.18 +
        spiral * 0.045 +
        interference * 0.045 +
        t * 0.008 +
        mouse_dist * 0.04;

    var fill = palette(hue);

    let luminous_ribbons =
        pow(
            max(
                0.0,
                0.5 + 0.5 * interference
            ),
            3.0
        );

    fill += palette(
        hue + 0.12 + luminous_ribbons * 0.08
    ) * luminous_ribbons * 0.38;

    let center_light =
        exp(-radius_p * radius_p * 2.4);

    fill += vec3<f32>(
        0.10,
        0.15,
        0.24
    ) * center_light;

    fill += palette(
        hue + 0.22
    ) * mouse_glow * 0.26;

    fill += vec3<f32>(
        0.35,
        0.55,
        1.0
    ) * mouse_ring * 0.16;

    let vignette =
        smoothstep(
            1.15,
            0.25,
            radius_p
        );

    fill *=
        0.48 +
        depth * 0.42 +
        vignette * 0.28;

    let shimmer =
        pow(
            max(
                0.0,
                sin(
                    angle * 7.0 -
                    radius_p * 13.0 +
                    t * 0.55
                )
            ),
            12.0
        );

    fill += vec3<f32>(
        0.7,
        0.85,
        1.0
    ) * shimmer * 0.07;

    let edge_distance = min(
        min(uv.x, 1.0 - uv.x),
        min(uv.y, 1.0 - uv.y)
    );

    let edge =
        1.0 - smoothstep(
            0.0,
            0.09,
            edge_distance
        );

    fill += vec3<f32>(
        0.08,
        0.12,
        0.24
    ) * edge;

    fill = mix(
        fill,
        input.border_color.rgb,
        border_mask
    );

    let alpha = outer_alpha;

    return vec4<f32>(
        fill * alpha,
        alpha
    );
}