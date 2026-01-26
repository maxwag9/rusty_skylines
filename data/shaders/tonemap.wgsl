struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VSOut {
    var positions = array<vec2<f32>, 3>(
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0),
    );

    var uvs = array<vec2<f32>, 3>(
        vec2(0.0, 0.0),
        vec2(2.0, 0.0),
        vec2(0.0, -2.0),
    );

    var out: VSOut;
    out.pos = vec4(positions[i], 0.0, 1.0);
    out.uv = uvs[i];
    return out;
}
@group(0) @binding(0)
var hdr_tex: texture_2d<f32>;

@group(0) @binding(1)
var hdr_sampler: sampler;

fn tonemap_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 3.21;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3(0.0), vec3(1.0));
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let hdr = textureSample(hdr_tex, hdr_sampler, in.uv).rgb;
    let ldr = tonemap_aces(hdr);
    return vec4(ldr, 1.0);
}
