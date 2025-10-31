// text.wgsl
struct VertexInput {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> screen: vec4<f32>; // [width, height, time, enable_dither]

// === Vertex Shader ===
@vertex
fn vs_main(v: VertexInput) -> VSOut {
    var out: VSOut;

    // convert pixel coordinates (0..width, 0..height) → NDC (-1..1)
    let ndc_x = (v.pos.x / screen.x) * 2.0 - 1.0;
    // flip Y so 0 is top
    let ndc_y = 1.0 - (v.pos.y / screen.y) * 2.0;

    out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = v.uv;
    out.color = v.color;
    return out;
}

// === Fragment Shader ===
@group(1) @binding(0) var font_tex: texture_2d<f32>;
@group(1) @binding(1) var font_sampler: sampler;

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let alpha = textureSample(font_tex, font_sampler, in.uv).r;
    return vec4<f32>(in.color.rgb, alpha * in.color.a);
}
