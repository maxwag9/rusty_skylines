struct ScreenUniform {
    size: vec2<f32>,
    time: f32,
    enable_dither: u32,
    mouse: vec2<f32>,
}

struct RectGpu {
    center: vec2<f32>,
    half_size: vec2<f32>,
    color: vec4<f32>,
    border_color: vec4<f32>,
    roundness: f32,
    border_thickness: f32,
    rotation: f32,
    fade: f32,
    glow_color: vec4<f32>,
    glow_misc: vec4<f32>,
    misc: vec4<f32>,
    blur: f32,
    depth: f32,
    _pad0: vec2<f32>,
}

@group(1) @binding(0) var<uniform> screen: ScreenUniform;
@group(2) @binding(0) var<storage, read> rects: array<RectGpu>;

struct VertexInput {
    @location(0) pos: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) mouse_local: vec2<f32>,
    @location(3) rect_half_size: vec2<f32>,
    @location(4) roundness: f32,
    @location(5) border_thickness: f32,
    @location(6) color: vec4<f32>,
    @location(7) border_color: vec4<f32>,
}

fn sd_poly(p: vec2<f32>, r: f32, n: f32) -> f32 {
    let an = 6.28318530718 / n;
    var a = atan2(p.x, p.y) + an * 0.5;
    a = a - floor(a / an) * an;
    let d = r * cos(an * 0.5) / max(cos(a - an * 0.5), 0.001);
    return length(p) - d;
}

fn outer_billiard_step(x: vec2<f32>, R: f32, n: f32) -> vec2<f32> {
    let an = 6.28318530718 / n;
    let base = floor(atan2(x.y, x.x) / an);
    var best = x;
    var best_score = 1e30;

    for (var i = -1; i <= 2; i = i + 1) {
        let k = base + f32(i);
        let edge_ang = k * an;
        let nrm = vec2<f32>(cos(edge_ang), sin(edge_ang));
        let pt = nrm * R;
        let img = 2.0 * pt - x;
        let cross = (img.x - x.x) * (-x.y) - (img.y - x.y) * (-x.x);
        let score = abs(cross) + length(pt - x) * 0.01;
        if (cross > 0.0 && score < best_score) {
            best_score = score;
            best = img;
        }
    }
    return best;
}

// Much better color mapping
fn oklch_to_rgb(l: f32, c: f32, h: f32) -> vec3<f32> {
    let a = c * cos(h);
    let b = c * sin(h);

    let l_ = l + 0.3963377774 * a + 0.2158037573 * b;
    let m_ = l - 0.1055613458 * a - 0.0638541728 * b;
    let s_ = l - 0.0894841775 * a - 1.2914855480 * b;

    let l3 = l_ * l_ * l_;
    let m3 = m_ * m_ * m_;
    let s3 = s_ * s_ * s_;

    return vec3<f32>(
        4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3,
       -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3,
       -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3
    );
}

fn period_color(per: f32, fingerprint: f32, t: f32) -> vec3<f32> {
    let tau = 6.28318530718;
    let golden = 0.61803398875;

    let hue = fract(
        per * golden +
        fingerprint * 0.173 +
        t * 0.003
    ) * tau;

    let lightness = 0.68;
    var chroma = 0.16;

    var rgb = oklch_to_rgb(lightness, chroma, hue);

    for (var i = 0; i < 8; i++) {
        let out_of_gamut =
            max(0.0, -min(rgb.x, min(rgb.y, rgb.z))) +
            max(0.0, max(rgb.x, max(rgb.y, rgb.z)) - 1.0);

        if (out_of_gamut <= 0.00001) {
            break;
        }

        chroma *= 0.82;
        rgb = oklch_to_rgb(lightness, chroma, hue);
    }

    return clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
}
@vertex
fn vs_main(
    input: VertexInput,
    @builtin(instance_index) instance: u32
) -> VertexOutput {
    let rect = rects[instance];
    let c = cos(rect.rotation);
    let s = sin(rect.rotation);
    let rot = mat2x2<f32>(c, -s, s, c);
    let inverse_rot = mat2x2<f32>(c, s, -s, c);

    let local = input.pos * rect.half_size;
    let world = rect.center + rot * local;
    let mouse_local = inverse_rot * (screen.mouse - rect.center);

    let ndc = (world / screen.size) * 2.0 - 1.0;

    var out: VertexOutput;
    out.position = vec4<f32>(ndc.x, -ndc.y, rect.depth, 1.0);
    out.local_pos = local;
    out.uv = input.pos * 0.5 + 0.5;
    out.mouse_local = mouse_local;
    out.rect_half_size = rect.half_size;
    out.roundness = rect.roundness;
    out.border_thickness = rect.border_thickness;
    out.color = rect.color;
    out.border_color = rect.border_color;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let t = screen.time;

    // continuous side count
    let n = 3.5 + 4.0 * (0.5 + 0.5 * sin(t * 0.17));

    // MUCH SMALLER central polygon
    let R = min(input.rect_half_size.x, input.rect_half_size.y) * 0.28;

    let mouse_force = input.mouse_local * 0.0006;
    let table_center = mouse_force;

    var p = input.local_pos - table_center;

    let d_table = sd_poly(p, R, n);
    let aa = max(fwidth(d_table), 0.7);

    var color = vec3<f32>(0.015, 0.008, 0.03); // deep void

    if (d_table > 0.0) {
        var z = p;
        var escaped = false;
        var period = 0u;
        let max_iter = 72u;
        let escape_r2 = 36.0 * R * R;

        let z0 = z;
        var fingerprint = 0.0;

        for (var i = 0u; i < max_iter; i = i + 1u) {
            z = outer_billiard_step(z, R, n);

            if (length(z) > 1e4) {
                escaped = true;
                break;
            }

            let dist0 = length(z - z0);
            if (i > 2u && dist0 < 0.012 * R) {
                period = i + 1u;
                break;
            }

            fingerprint += sin(z.x * 0.08 + z.y * 0.11) * 0.018;

            if (dot(z, z) > escape_r2) {
                escaped = true;
                period = i + 1u;
                break;
            }
        }

        if (escaped) {
            // nuclear unbounded orbits
            let esc = f32(period) / f32(max_iter);
            color = mix(
                vec3<f32>(1.0, 0.25, 0.05),
                vec3<f32>(1.0, 0.85, 0.4),
                esc * esc
            );
            color += fingerprint * vec3<f32>(0.6, 0.3, 0.1);
        } else if (period > 0u) {
            // beautiful period bands
            color = period_color(f32(period), fingerprint, t);
        } else {
            // dense chaotic sea – rich and weird
            color = vec3<f32>(0.04, 0.08, 0.22)
                  + fingerprint * vec3<f32>(0.55, 0.25, 0.7)
                  + vec3<f32>(0.0, 0.05, 0.1) * sin(fingerprint * 12.0 + t);
        }

        // gentle falloff far away
        let far = smoothstep(R * 2.5, R * 12.0, length(p));
        color *= 1.0 - far * 0.65;
    } else {
        // inside – almost pure black
        color = vec3<f32>(0.008, 0.004, 0.015);
    }

    // sharp neon rim
    let rim = exp(-abs(d_table) * 28.0) * 1.8;
    color += vec3<f32>(0.7, 0.25, 1.0) * rim;

    return vec4<f32>(color, 1.0);
}