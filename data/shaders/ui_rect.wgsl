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
    misc: vec4<f32>,  // active, touched_time, is_down, hash
};

@group(0) @binding(0) var<uniform> screen: ScreenUniform;
@group(1) @binding(0) var<storage, read> rects: array<RectGpu>;

struct VertexInput {
    @location(0) pos: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local_pos: vec2<f32>,  // [-1, 1] within rect
    @location(1) color: vec4<f32>,
    @location(2) rect_half_size: vec2<f32>,
    @location(3) roundness: f32,
    @location(4) border_thickness: f32,
    @location(5) fade: f32,
    @location(6) misc: vec4<f32>,
};

// Signed distance function for rounded rectangle
fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + r;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@vertex
fn vs_main(in: VertexInput, @builtin(instance_index) instance: u32) -> VertexOutput {
    let rect = rects[instance];

    let padding = 2.0;
    let half_size = rect.half_size + padding;

    let c = cos(rect.rotation);
    let s = sin(rect.rotation);
    let rot = mat2x2<f32>(c, -s, s, c);

    let local = in.pos * half_size;
    let rotated = rot * local;

    let world_pos = rect.center + rotated;

    let ndc = (world_pos / screen.size) * 2.0 - 1.0;

    var out: VertexOutput;
    out.clip_position = vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
    out.local_pos = local;
    out.color = rect.color;
    out.rect_half_size = rect.half_size;
    out.roundness = rect.roundness;
    out.border_thickness = rect.border_thickness;
    out.fade = rect.fade;
    out.misc = rect.misc;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let half_size = in.rect_half_size;

    let max_round = min(half_size.x, half_size.y);
    let roundness = in.roundness * max_round;  // Multiply instead of min()

    let d = sd_rounded_box(in.local_pos, half_size, roundness);

    let aa = fwidth(d) * 1.5;
    var alpha = 1.0 - smoothstep(-aa, aa, d);

    var color = in.color;

    if in.border_thickness > 0.0 {
        let inner_half = half_size - in.border_thickness;
        let inner_round = max(roundness - in.border_thickness, 0.0);
        let d_inner = sd_rounded_box(in.local_pos, inner_half, inner_round);
        let inner_alpha = smoothstep(-aa, aa, d_inner);

        let border_mask = alpha * (1.0 - inner_alpha);
        let fill_mask = alpha * inner_alpha;

        color = mix(rects[0].border_color, color, fill_mask / max(fill_mask + border_mask, 0.0001));
        alpha = fill_mask + border_mask;
    }

    alpha *= mix(1.0, in.misc.x, in.fade);

    let touched_time = in.misc.y;
    let is_down = in.misc.z;
    if is_down > 0.5 {
        color = mix(color, vec4<f32>(1.0, 1.0, 1.0, color.a), 0.2);
    }

    return vec4<f32>(color.rgb, color.a * alpha);
}
