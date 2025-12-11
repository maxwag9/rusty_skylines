const STAR_COUNT: u32 = 116812u;

struct Uniforms {
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    sun_direction: vec3<f32>,
    time: f32,
    camera_pos: vec3<f32>,
    orbit_radius: f32,
    moon_direction: vec3<f32>,
    _pad0: f32,
};

@group(0) @binding(0)
var<uniform> u: Uniforms;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) mag: f32,
    @location(1) bv: f32,
    @location(2) uv: vec2<f32>,
};

fn ra_dec_to_dir(ra: f32, dec: f32) -> vec3<f32> {
    return vec3(
        cos(dec) * cos(ra),
        sin(dec),
        cos(dec) * sin(ra)
    );
}

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

    // Cull extremely dim stars at the vertex stage (saves pixels)
    if (mag > 9.5) {
        var out: VSOut;
        out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0); // send offscreen
        out.mag = mag;
        out.bv = bv;
        out.uv = vec2<f32>(0.0);
        return out;
    }

    // 4 billboard corners
    let corners = array<vec2<f32>, 4>(
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2( 1.0,  1.0)
    );

    let corner = corners[vid & 3u];

    // Better size curve: dim stars shrink a LOT
    let size = clamp(10.0 - mag * 0.8, 0.4, 10.0);

    // view-space billboard (cheap)
    let right = vec3<f32>(1.0, 0.0, 0.0);
    let up    = vec3<f32>(0.0, 1.0, 0.0);

    let pos_view =
        center +
        right * corner.x * size +
        up    * corner.y * size;

    var out: VSOut;
    out.pos = u.proj * vec4<f32>(pos_view, 1.0);
    out.mag = mag;
    out.bv = bv;
    out.uv = corner * 0.5 + 0.5;

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
@fragment
fn fs_main(
    @location(0) mag: f32,
    @location(1) bv: f32,
    @location(2) uv: vec2<f32>
) -> @location(0) vec4<f32> {

    // dim stars: early-out (major perf win)
    if (mag > 9.5) {
        return vec4(0.0);
    }
    if (mag > 5.0) {
        // ultra-fast cheap version
        let dx = uv.x - 0.5;
        let dy = uv.y - 0.5;
        let r2 = dx*dx + dy*dy;
        let g = exp(-r2 * 100.0);
        let color = bv_to_rgb(bv);
        let base = mag_to_intensity(mag) * 10.0;
        return vec4(color * base * g, g);
    }

    let color = bv_to_rgb(bv);
    let base = mag_to_intensity(mag) * 10.0;

    // radial distance squared (avoid sqrt)
    let dx = uv.x - 0.5;
    let dy = uv.y - 0.5;
    let r2 = dx*dx + dy*dy;

    // your ideal sigma = 0.12
    // compute (r/sigma) without sqrt: (sqrt(r2)/s)^1 == r2^(0.5)/s
    // but we can approximate by scaling r2 directly:
    let k = 1.0 / (0.12 * 0.12);

    // Exponential falloff:
    //   exp(-(r/sigma))^2 == exp(-2*(r/sigma))
    // Using r2 to reduce sqrt cost
    let g = exp(-r2 * 300.0);



    let intensity = base * g;

    return vec4(color * intensity, g);
}
