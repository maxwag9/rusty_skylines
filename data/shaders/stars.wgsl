#include "includes/uniforms.wgsl"
const STAR_COUNT: u32 = 116812u;

@group(1) @binding(0)
var<uniform> u: Uniforms;

fn ra_dec_to_dir(ra: f32, dec: f32) -> vec3<f32> {
    return vec3(
        cos(dec) * cos(ra),
        sin(dec),
        cos(dec) * sin(ra)
    );
}

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) mag: f32,
    @location(1) bv: f32,
    @location(2) uv: vec2<f32>,
    @location(3) curr_clip: vec4<f32>,
    @location(4) prev_clip: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) ra: f32,
    @location(1) dec: f32,
    @location(2) mag: f32,
    @location(3) bv: f32,
    @builtin(vertex_index) vid: u32,
) -> VSOut {

    let dir = ra_dec_to_dir(ra, dec);

    let view_dir = (u.view * vec4<f32>(dir, 0.0)).xyz;
    let center = normalize(view_dir) * 1000.0;

    if (mag > 9.5) {
        var out: VSOut;
        out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.mag = mag;
        out.bv = bv;
        out.uv = vec2<f32>(0.0);
        out.curr_clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.prev_clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }

    let corners = array<vec2<f32>, 4>(
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2( 1.0,  1.0)
    );

    let corner = corners[vid & 3u];
    let size = clamp(10.0 - mag * 0.8, 0.4, 10.0);

    let right = vec3<f32>(1.0, 0.0, 0.0);
    let up    = vec3<f32>(0.0, 1.0, 0.0);

    let pos_view = center + right * corner.x * size + up * corner.y * size;

    let clip = u.proj * vec4<f32>(pos_view, 1.0);

    // Previous frame: reproject the star direction through previous view
    // dir is world-space, so transform by previous view, same billboard offset
    let prev_view_dir = (u.prev_view_proj * vec4<f32>(dir, 0.0));

    var out: VSOut;
    out.pos = clip;
    out.mag = mag;
    out.bv = bv;
    out.uv = corner * 0.5 + 0.5;
    out.curr_clip = clip;

    // For stars we reproject the center direction only (billboard is screen-aligned)
    // Use the same world direction through prev_view_proj with w=0
    out.prev_clip = u.prev_view_proj * vec4<f32>(dir * 1000.0, 0.0);

    return out;
}



fn mag_to_intensity(m: f32) -> f32 {
    return pow(2.512, -m);
}

fn bv_to_rgb(bv: f32) -> vec3<f32> {
    let x = clamp(bv, -0.4, 2.0);
    return clamp(vec3(
        1.0 - 0.5 * x,
        0.82 - 0.3 * x,
        1.0 - 1.2 * (x - 0.4)
    ), vec3(0.0), vec3(1.0));
}
struct FragOut {
    @location(0) color: vec4<f32>,
    @location(2) motion: vec2<f32>,
};

@fragment
fn fs_main(
    @location(0) mag: f32,
    @location(1) bv: f32,
    @location(2) uv: vec2<f32>,
    @location(3) curr_clip: vec4<f32>,
    @location(4) prev_clip: vec4<f32>,
) -> FragOut {
    var out: FragOut;

    if (mag > 9.5) {
        out.color = vec4(0.0);
        out.motion = vec2(0.0);
        return out;
    }

    // Motion vector
    let curr_ndc = curr_clip.xy / curr_clip.w;
    let prev_ndc = prev_clip.xy / prev_clip.w;
    let curr_uv = curr_ndc * vec2(0.5, -0.5) + 0.5;
    let prev_uv = prev_ndc * vec2(0.5, -0.5) + 0.5;
    out.motion = curr_uv - prev_uv;

    if (mag > 5.0) {
        let dx = uv.x - 0.5;
        let dy = uv.y - 0.5;
        let r2 = dx*dx + dy*dy;
        let g = exp(-r2 * 100.0);
        let color = bv_to_rgb(bv);
        let base = mag_to_intensity(mag) * 10.0;
        out.color = vec4(color * base * g, g);
        return out;
    }

    let color = bv_to_rgb(bv);
    let base = mag_to_intensity(mag) * 10.0;

    let dx = uv.x - 0.5;
    let dy = uv.y - 0.5;
    let r2 = dx*dx + dy*dy;

    let g = exp(-r2 * 300.0);
    let intensity = base * g;

    out.color = vec4(color * intensity, g);
    return out;
}
