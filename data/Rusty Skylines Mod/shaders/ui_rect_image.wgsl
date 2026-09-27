struct ScreenUniform {
    size: vec2<f32>,
    time: f32,
    enable_dither: u32,
    mouse: vec2<f32>,
};

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
};

@group(0) @binding(0)
var texture_sampler: sampler;

@group(0) @binding(2)
var image_texture: texture_2d<f32>;

@group(1) @binding(0)
var<uniform> screen: ScreenUniform;

@group(2) @binding(0)
var<storage, read> rects: array<RectGpu>;

struct VertexInput {
    @location(0) pos: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) rect_half_size: vec2<f32>,
    @location(3) roundness: f32,
};

fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + r;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@vertex
fn vs_main(
    in: VertexInput,
    @builtin(instance_index) instance: u32
) -> VertexOutput {
    let rect = rects[instance];

    let half_size = rect.half_size;

    let c = cos(rect.rotation);
    let s = sin(rect.rotation);
    let rot = mat2x2<f32>(
        c, -s,
        s, c
    );

    let local = in.pos * half_size;
    let rotated = rot * local;
    let world_pos = rect.center + rotated;

    let ndc = (world_pos / screen.size) * 2.0 - 1.0;

    var out: VertexOutput;
    out.clip_position = vec4<f32>(
        ndc.x,
        -ndc.y,
        rect.depth,
        1.0
    );
    out.local_pos = local;
    out.rect_half_size = half_size;
    out.roundness = rect.roundness;

    out.uv = vec2<f32>(
        local.x / half_size.x * 0.5 + 0.5,
        local.y / half_size.y * 0.5 + 0.5
    );

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let max_round = min(in.rect_half_size.x, in.rect_half_size.y);
    let roundness = in.roundness * max_round;

    let d = sd_rounded_box(
        in.local_pos,
        in.rect_half_size,
        roundness
    );

    let aa = max(fwidth(d), 0.75);
    let alpha = 1.0 - smoothstep(-aa, aa, d);

    let dimensions = textureDimensions(image_texture, 0);
    let texture_size = vec2<f32>(dimensions);

    let uv = clamp(in.uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let coord_f = uv * texture_size;
    let coord = min(
        vec2<u32>(coord_f),
        dimensions - vec2<u32>(1u)
    );

    let sampled = textureLoad(image_texture, coord, 0);

    let final_alpha = sampled.a * alpha;

    return vec4<f32>(
        sampled.rgb * final_alpha,
        final_alpha
    );
}