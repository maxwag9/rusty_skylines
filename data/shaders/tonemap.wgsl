// tonemap.wgsl
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
@group(0) @binding(0) var hdr_sampler: sampler;
@group(0) @binding(2) var hdr_tex: texture_2d<f32>;
@group(0) @binding(3) var ui_tex: texture_2d<f32>;

struct ToneMappingUniforms {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32
}
@group(1) @binding(0)
var<uniform> tm_uniforms: ToneMappingUniforms;

fn tonemap_aces(a: f32, b: f32, c: f32, d: f32, e: f32, x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3(0.0), vec3(1.0));
}
struct ColorGrade {
    lift: vec4<f32>,
    gamma: vec4<f32>,
    gain: vec4<f32>,
}

@group(1) @binding(1)
var<uniform> color_grade: ColorGrade;

fn apply_color_grade(color: vec3<f32>) -> vec3<f32> {
    var c = color;

    // Lift shadows
    c += color_grade.lift.xyz;

    // Gamma controls midtones
    c = pow(
        max(c, vec3(0.0)),
        vec3(1.0) / color_grade.gamma.xyz
    );

    // Gain controls highlights
    c *= color_grade.gain.xyz;

    return c;
}
struct FSOut {
    @location(0) surface: vec4<f32>,
    @location(1) screenshot: vec4<f32>,
};
@fragment
fn fs_main(in: VSOut) -> FSOut {
    var out: FSOut;
    let uv = in.uv;

    let hdr = textureSample(hdr_tex, hdr_sampler, uv).rgb;

    #ifdef TONEMAP_UI
        // UI participates in tonemapping
        let ui = textureSample(ui_tex, hdr_sampler, uv);
        let processed_input = mix(hdr, ui.rgb, ui.a);
    #else
        // Keep UI out of HDR post processing
        let processed_input = hdr;
    #endif

    let dims = vec2<f32>(textureDimensions(hdr_tex, 0));
    let aspect = dims.x / dims.y;

    let v = vignette(uv, 0.70, 0.65, 0.40, aspect);

    // HDR post processing
    let hdr_v = processed_input * v;
    let graded = apply_color_grade(hdr_v);

    let color = tonemap_aces(
        tm_uniforms.a,
        tm_uniforms.b,
        tm_uniforms.c,
        tm_uniforms.d,
        tm_uniforms.e,
        graded
    );

    #ifdef TONEMAP_UI
        let final_color = color;
    #else
        let ui = textureSample(ui_tex, hdr_sampler, uv);
        let final_color = mix(color, ui.rgb, ui.a);
    #endif

    out.surface = vec4(final_color, 1.0);
    out.screenshot = vec4(final_color, 1.0);

    return out;
}

fn vignette(uv: vec2<f32>, strength: f32, radius: f32, softness: f32, aspect: f32) -> f32 {
    let p = (uv - vec2(0.5, 0.5)) * vec2(aspect, 1.0);

    let max_dist = length(vec2(aspect, 1.0) * 0.5);
    let dist = length(p) / max_dist;

    let t = smoothstep(radius, radius + softness, dist);

    return 1.0 - t * strength;
}
