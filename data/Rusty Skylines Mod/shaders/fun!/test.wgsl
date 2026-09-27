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
    let a = vec3<f32>(0.25, 0.12, 0.35);
    let b = vec3<f32>(0.35, 0.25, 0.30);
    let c = vec3<f32>(1.0, 0.8, 0.7);
    let d = vec3<f32>(0.15, 0.05, 0.35);

    return a + b * cos(6.28318 * (c * t + d));
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
    let max_round = min(input.rect_half_size.x, input.rect_half_size.y);
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

    let inner_alpha = 1.0 - smoothstep(-aa, aa, d_inner);
    let border_mask = outer_alpha - inner_alpha;

    let uv = input.uv;
    let t = screen.time;

    let centered = uv - 0.5;
    let aspect = input.rect_half_size.x / max(input.rect_half_size.y, 0.001);
    let aspect_uv = vec2<f32>(centered.x * aspect, centered.y);

    let mouse_local_uv = vec2<f32>(
        input.mouse_local.x / input.rect_half_size.x * 0.5 + 0.5,
        input.mouse_local.y / input.rect_half_size.y * 0.5 + 0.5
    );

    let mouse_offset = uv - mouse_local_uv;
    let mouse_distance = length(
        vec2<f32>(
            mouse_offset.x * aspect,
            mouse_offset.y
        )
    );

    let soft_mouse = 1.0 - smoothstep(0.0, 0.65, mouse_distance);

    let subtle_wave_1 = sin(
        aspect_uv.x * 5.0 +
        aspect_uv.y * 7.0 +
        t * 0.7
    );

    let subtle_wave_2 = sin(
        aspect_uv.x * 9.0 -
        aspect_uv.y * 4.0 -
        t * 0.45
    );

    let field = subtle_wave_1 * 0.04 + subtle_wave_2 * 0.03;

    let mouse_wave = sin(
        mouse_distance * 24.0 -
        t * 2.0
    ) * soft_mouse * 0.065;

    let mouse_light = soft_mouse * 0.08;

    let animated = uv.x * 0.65 +
        uv.y * 0.35 +
        t * 0.025 +
        field +
        mouse_wave;

    var fill = palette(animated);

    fill *= 0.72 + mouse_light;

    let diagonal = sin(
        (uv.x + uv.y) * 14.0 +
        t * 0.35
    ) * 0.025;

    fill += vec3<f32>(
        diagonal,
        diagonal * 0.6,
        diagonal * 1.2
    );

    let edge_distance = min(
        min(uv.x, 1.0 - uv.x),
        min(uv.y, 1.0 - uv.y)
    );

    let edge = 1.0 - smoothstep(
        0.0,
        0.06,
        edge_distance
    );

    fill += vec3<f32>(
        0.12,
        0.03,
        0.18
    ) * edge;

    let border_color = input.border_color.rgb;
    fill = mix(
        fill,
        border_color,
        border_mask
    );

    let alpha = outer_alpha;

    return vec4<f32>(
        fill * alpha,
        alpha
    );
}