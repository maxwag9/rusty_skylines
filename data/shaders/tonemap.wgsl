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

// Post processing uniforms
//
// tone_abcd = ACES parameters a,b,c,d
// tone_e.x  = ACES parameter e
// tone_e.y  = exposure
// tone_e.z  = brightness
// tone_e.w  = contrast
//
// lift      = RGB lift
// gamma     = RGB gamma
// gain      = RGB gain
//
// color_adj.x = saturation
// color_adj.y = vignette strength
// color_adj.z = vignette radius
// color_adj.w = vignette softness
//
// vignette.x = center X
// vignette.y = center Y

struct PostProcessUniforms {
    tone_abcd: vec4<f32>,
    tone_e: vec4<f32>,

    lift: vec4<f32>,
    gamma: vec4<f32>,
    gain: vec4<f32>,

    color_adj: vec4<f32>,
    vignette: vec4<f32>,
};

@group(1) @binding(0)
var<uniform> post: PostProcessUniforms;

fn tonemap_aces(
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    x: vec3<f32>,
) -> vec3<f32> {
    return clamp(
        (x * (a * x + b)) /
        (x * (c * x + d) + e),
        vec3(0.0),
        vec3(1.0)
    );
}

fn apply_color_grade(color: vec3<f32>) -> vec3<f32> {
    var c = color;

    // Lift shadows
    c += post.lift.xyz;

    // Gamma controls midtones
    c = pow(
        max(c, vec3(0.0)),
        vec3(1.0) / max(post.gamma.xyz, vec3(0.0001))
    );

    // Gain controls highlights
    c *= post.gain.xyz;

    return c;
}

fn apply_saturation(color: vec3<f32>, saturation: f32) -> vec3<f32> {
    let luminance = dot(
        color,
        vec3(0.2126, 0.7152, 0.0722)
    );

    return mix(
        vec3(luminance),
        color,
        saturation
    );
}
fn vignette(uv: vec2<f32>, aspect: f32) -> f32 {
    let center = vec2(
        post.vignette.x,
        post.vignette.y
    );

    let p = (uv - center) * vec2(aspect, 1.0);

    let max_dist = length(vec2(aspect, 1.0) * 0.5);
    let dist = length(p) / max_dist;

    let t = smoothstep(
        post.color_adj.z,
        post.color_adj.z + post.color_adj.w,
        dist
    );

    return 1.0 - t * post.color_adj.y;
}

struct FSOut {
    @location(0) surface: vec4<f32>,
    @location(1) screenshot: vec4<f32>,
};


@fragment
fn fs_main(in: VSOut) -> FSOut {
    var out: FSOut;

    let uv = in.uv;

    let hdr = textureSample(
        hdr_tex,
        hdr_sampler,
        uv
    ).rgb;

    #ifdef TONEMAP_UI
        // UI gets HDR post processing
        let ui = textureSample(
            ui_tex,
            hdr_sampler,
            uv
        );

        let processed_input = mix(
            hdr,
            ui.rgb,
            ui.a
        );
    #else
        // UI doesn't get HDR post processing
        let processed_input = hdr;
    #endif

    let exposed = processed_input * pow(
        2.0,
        post.tone_e.y
    );

    let dims = vec2<f32>(
        textureDimensions(hdr_tex, 0)
    );

    let aspect = dims.x / dims.y;

    let v = vignette(
        uv,
        aspect
    );

    let vignetted = exposed * v;

    let graded = apply_color_grade(
        vignetted
    );

    let brightness = graded + vec3(
        post.tone_e.z
    );

    let contrasted =
        (brightness - vec3(0.5)) *
        post.tone_e.w +
        vec3(0.5);

    let adjusted = apply_saturation(
        contrasted,
        post.color_adj.x
    );

    let color = tonemap_aces(
        post.tone_abcd.x,
        post.tone_abcd.y,
        post.tone_abcd.z,
        post.tone_abcd.w,
        post.tone_e.x,
        adjusted
    );


    #ifdef TONEMAP_UI
        let final_color = color;
    #else
        let ui = textureSample(
            ui_tex,
            hdr_sampler,
            uv
        );

        let final_color = mix(
            color,
            ui.rgb,
            ui.a
        );
    #endif


    out.surface = vec4(final_color, 1.0);
    out.screenshot = vec4(final_color, 1.0);

    return out;
}

@group(0) @binding(0)
var hdr_sampler: sampler;

@group(0) @binding(2)
var hdr_tex: texture_2d<f32>;

@group(0) @binding(3)
var ui_tex: texture_2d<f32>;
