#include "includes/uniforms.wgsl"

@group(1) @binding(0) var<uniform> uniforms: Uniforms;

struct VSIn {
    @location(0) start: vec3<f32>,
    @location(1) end: vec3<f32>,
    @location(2) end_sign: f32,
    @location(3) side_sign: f32,
    @location(4) thickness: f32,
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
    return (ndc * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5)) * uniforms.screen_size;
}

fn billboard_right(dir: vec3<f32>, to_eye: vec3<f32>) -> vec3<f32> {
    var right = cross(dir, to_eye);
    if (dot(right, right) < 1e-10) {
        right = cross(dir, vec3<f32>(0.0, 1.0, 0.0));
    }
    if (dot(right, right) < 1e-10) {
        right = cross(dir, vec3<f32>(1.0, 0.0, 0.0));
    }
    return normalize(right);
}

@vertex
fn vs_main(input: VSIn) -> VSOut {
    let seg = input.end - input.start;
    var dir = seg;
    if (dot(dir, dir) < 1e-10) {
        dir = vec3<f32>(1.0, 0.0, 0.0);
    } else {
        dir = normalize(dir);
    }

    let mid = (input.start + input.end) * 0.5;
    var to_eye = -mid;
    if (dot(to_eye, to_eye) < 1e-10) {
        to_eye = vec3<f32>(0.0, 0.0, 1.0);
    } else {
        to_eye = normalize(to_eye);
    }

    let right = billboard_right(dir, to_eye);
    let half_thickness = input.thickness * 0.5;

    let base = select(input.start, input.end, input.end_sign > 0.0);
    let world_pos = base
        + right * (input.side_sign * half_thickness)
        + dir * (input.end_sign * input.thickness);

    let pos_clip = uniforms.view_proj * vec4<f32>(world_pos, 1.0);

    var out: VSOut;
    out.pos = pos_clip;
    out.color = input.color;

    let start_clip = uniforms.view_proj * vec4<f32>(input.start, 1.0);
    let end_clip = uniforms.view_proj * vec4<f32>(input.end, 1.0);
    out.start_px = clip_to_px(start_clip);
    out.end_px = clip_to_px(end_clip);

    let base_clip = select(start_clip, end_clip, input.end_sign > 0.0);
    let edge_clip = uniforms.view_proj * vec4<f32>(base + right * half_thickness, 1.0);
    out.half_width_px = distance(clip_to_px(edge_clip), clip_to_px(base_clip));

    out.curr_clip = pos_clip;
    out.prev_clip = uniforms.prev_view_proj * vec4<f32>(world_pos, 1.0);

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