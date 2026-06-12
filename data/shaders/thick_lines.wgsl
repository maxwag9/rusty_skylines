#include "includes/uniforms.wgsl"

@group(1) @binding(0) var<uniform> uniforms: Uniforms;

struct VSIn {
    @location(0) start: vec3<f32>,
    @location(1) end: vec3<f32>,
    @location(2) end_sign: f32,
    @location(3) side_sign: f32,
    @location(4) width_px: f32,
    @location(5) color: vec4<f32>,
};

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) start_px: vec2<f32>,
    @location(2) end_px: vec2<f32>,
    @location(3) half_width_px: f32,
    @location(4) curr_clip: vec4<f32>,
    @location(5) prev_clip: vec4<f32>
};

fn clip_to_px(clip: vec4<f32>) -> vec2<f32> {
    let ndc = clip.xy / clip.w;
    return (ndc * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5)) * vec2<f32>(2560.0, 1440.0);
}

fn offset_clip(
    clip: vec4<f32>,
    dir: vec2<f32>,
    normal: vec2<f32>,
    end_sign: f32,
    side_sign: f32,
    half_width_px: f32,
) -> vec4<f32> {
    var out = clip;

    // Expand in NDC by half width in pixels.
    let ndc_offset = (dir * end_sign + normal * side_sign)
        * (half_width_px * 2.0 / vec2<f32>(2560.0, 1440.0));

    let offset = ndc_offset * clip.w;

    out.x += offset.x;
    out.y += offset.y;
    return out;
}

@vertex
fn vs_main(input: VSIn) -> VSOut {
    let start_clip = uniforms.view_proj * vec4<f32>(input.start, 1.0);
    let end_clip = uniforms.view_proj * vec4<f32>(input.end, 1.0);

    let start_ndc = start_clip.xy / start_clip.w;
    let end_ndc = end_clip.xy / end_clip.w;

    let dir = normalize(end_ndc - start_ndc);
    let normal = vec2<f32>(-dir.y, dir.x);

    let half_width_px = input.width_px * 0.5;

    let base_clip = select(start_clip, end_clip, input.end_sign > 0.0);
    let pos_clip = offset_clip(
        base_clip,
        dir,
        normal,
        input.end_sign,
        input.side_sign,
        half_width_px,
    );

    var out: VSOut;
    out.pos = pos_clip;
    out.color = input.color;
    out.start_px = clip_to_px(start_clip);
    out.end_px = clip_to_px(end_clip);
    out.half_width_px = half_width_px;
    out.curr_clip = pos_clip;

    let prev_start_clip = uniforms.prev_view_proj * vec4<f32>(input.start, 1.0);
    let prev_end_clip = uniforms.prev_view_proj * vec4<f32>(input.end, 1.0);

    let prev_start_ndc = prev_start_clip.xy / prev_start_clip.w;
    let prev_end_ndc = prev_end_clip.xy / prev_end_clip.w;

    let prev_dir = normalize(prev_end_ndc - prev_start_ndc);
    let prev_normal = vec2<f32>(-prev_dir.y, prev_dir.x);

    let prev_base_clip = select(prev_start_clip, prev_end_clip, input.end_sign > 0.0);
    out.prev_clip = offset_clip(
        prev_base_clip,
        prev_dir,
        prev_normal,
        input.end_sign,
        input.side_sign,
        half_width_px,
    );

    return out;
}

struct FragOut {
    @location(0) color: vec4<f32>,
    @location(2) motion: vec2<f32>,
};

fn dist_to_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let ap = p - a;
    let t = clamp(dot(ap, ab) / max(dot(ab, ab), 0.000001), 0.0, 1.0);
    let closest = a + t * ab;
    return distance(p, closest);
}

@fragment
fn fs_main(in: VSOut) -> FragOut {
    var out: FragOut;

    let d = dist_to_segment(in.pos.xy, in.start_px, in.end_px);
    let aa = max(fwidth(d), 1.0);
    let alpha = 1.0 - smoothstep(in.half_width_px - aa, in.half_width_px + aa, d);

    out.color = vec4<f32>(in.color.rgb, in.color.a * alpha);

    let curr_ndc = in.curr_clip.xy / in.curr_clip.w;
    let prev_ndc = in.prev_clip.xy / in.prev_clip.w;
    let curr_uv = curr_ndc * vec2<f32>(0.5, -0.5) + 0.5;
    let prev_uv = prev_ndc * vec2<f32>(0.5, -0.5) + 0.5;
    out.motion = curr_uv - prev_uv;

    return out;
}