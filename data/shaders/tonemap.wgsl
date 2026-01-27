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
        vec2(0.0, 2.0),
    );

    var out: VSOut;
    out.pos = vec4(positions[i], 0.0, 1.0);
    out.uv = uvs[i];
    out.uv.y = 1.0 - out.uv.y;
    return out;
}
@group(0) @binding(0)
var hdr_tex: texture_2d<f32>;

@group(0) @binding(1)
var hdr_sampler: sampler;

fn tonemap_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.55;
    let b = 0.02;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3(0.0), vec3(1.0));
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let uv = in.uv;

    let hdr = textureSample(hdr_tex, hdr_sampler, uv).rgb;

    let dims = vec2<f32>(textureDimensions(hdr_tex, 0));
    let aspect = dims.x / dims.y;

    let v = vignette(uv, 0.60, 0.75, 0.35, aspect);

    // Apply vignette in HDR (exposure domain)
    let hdr_v = hdr * v;

    let color = tonemap_aces(hdr_v);
    return vec4(color, 1.0);
}

fn vignette(uv: vec2<f32>, strength: f32, radius: f32, softness: f32, aspect: f32) -> f32 {
    // center in [-0.5, +0.5] around the screen center
    var p = uv - vec2(0.5, 0.5);
    p.x *= aspect;

    let dist = length(p);
    let t = smoothstep(radius, radius + softness, dist);

    return 1.0 - t * strength;
}
