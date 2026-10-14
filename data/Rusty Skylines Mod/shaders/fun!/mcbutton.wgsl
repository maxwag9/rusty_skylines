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
    @location(8) depth: f32,
}

fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + r;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

fn hash2(p: vec2<f32>) -> f32 {
    let h = dot(
        p,
        vec2<f32>(127.1, 311.7)
    );

    return fract(sin(h) * 43758.5453123);
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
    out.depth = rect.depth;

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
    let alpha = 1.0 - smoothstep(-aa, aa, d);

    let p = input.local_pos;
    let h = input.rect_half_size;

    let edge_x = min(
        p.x + h.x,
        h.x - p.x
    );

    let edge_y = min(
        p.y + h.y,
        h.y - p.y
    );

    let edge = min(edge_x, edge_y);

    let outer_edge = 1.0 - smoothstep(
        0.0,
        1.0,
        edge
    );

    let bevel_size = clamp(
        input.border_thickness,
        1.0,
        3.0
    );

    let edge_seed = input.depth * 10000.0;

    let edge_cell_x = floor(p.y / 4.0);
    let edge_cell_y = floor(p.x / 4.0);

    let edge_noise_x = fract(
        sin(
            edge_cell_x * 127.1 +
            edge_seed
        ) * 43758.5453
    );

    let edge_noise_y = fract(
        sin(
            edge_cell_y * 311.7 +
            edge_seed * 1.731
        ) * 43758.5453
    );

    let warp_x = (edge_noise_x - 0.5) * 15.0;
    let warp_y = (edge_noise_y - 0.5) * 10.0;

    let left = 1.0 - smoothstep(
        0.0,
        bevel_size,
        p.x + h.x + warp_x
    );

    let right = 1.0 - smoothstep(
        0.0,
        bevel_size,
        h.x - p.x - warp_x
    );

    let top = 1.0 - smoothstep(
        0.0,
        bevel_size,
        p.y + h.y + warp_y
    );

    let bottom = 1.0 - smoothstep(
        0.0,
        bevel_size,
        h.y - p.y - warp_y
    );

    let light_bevel = max(left, top);
    let dark_bevel = max(right, bottom);

    let mouse_distance = length(p - input.mouse_local);

    let pixelate = 1.0 - smoothstep(
        0.0,
        200.0,
        mouse_distance
    );

    let fine_cell = floor(p / 2.0);
    let coarse_cell = floor(p / 1.0);

    let depth_seed = input.depth * 10000.0;

    let fine_noise = fract(
        sin(
            dot(
                fine_cell,
                vec2<f32>(127.1, 311.7)
            ) +
            depth_seed
        ) * 43758.5453
    );

    let coarse_noise = fract(
        sin(
            dot(
                coarse_cell,
                vec2<f32>(269.5, 183.3)
            ) +
            depth_seed * 1.731
        ) * 43758.5453
    );

    let noise = mix(
        fine_noise,
        coarse_noise,
        pixelate
    );

    let texture_noise = (noise - 0.5) * 0.14;

    let mouse_darkness = pixelate * 0.20;

    let base = 0.25818 + texture_noise - mouse_darkness;

    var fill = vec3<f32>(
        base,
        base,
        base
    );

    fill = mix(
        fill,
        vec3<f32>(0.56471),
        light_bevel * 0.72
    );

    fill = mix(
        fill,
        vec3<f32>(0.09084),
        dark_bevel * 0.90
    );

    let inner_edge = 1.0 - smoothstep(
        0.0,
        0.65,
        edge
    );

    fill = mix(
        fill,
        vec3<f32>(0.03820),
        inner_edge
    );

    let highlight_band = max(
        top,
        left
    );

    fill += vec3<f32>(0.035) * highlight_band;



    let hover = 1.0 - smoothstep(
        0.0,
        0.65,
        mouse_distance
    );

    fill += vec3<f32>(0.018) * hover;

    fill = clamp(
        fill,
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );

    return vec4<f32>(
        fill * alpha,
        alpha
    );
}