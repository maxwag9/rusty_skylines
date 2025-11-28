const PI: f32 = 3.14159265;

struct ScreenUniform {
    size: vec2<f32>,
    time: f32,
    enable_dither: u32,
    mouse: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> screen: ScreenUniform;

struct CircleParams {
    center_radius_border: vec4<f32>, // cx, cy, radius, border
    fill_color: vec4<f32>,
    border_color: vec4<f32>,
    glow_color: vec4<f32>,
    glow_misc: vec4<f32>,         // (glow_size, ...)
    misc: vec4<f32>, // THIS MOTHERFUCKER11111!!!11 AMERICA NUMBER 1

    fade: f32,
    style: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(1) @binding(0)
var<storage, read> circles: array<CircleParams>;

struct VertexInput {
    @location(0) pos: vec2<f32>,
    @location(1) color: vec4<f32>,
    @builtin(instance_index) instance: u32,
};

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) circle_index: u32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let params = circles[in.instance];
    let crb = params.center_radius_border;

    let center = crb.xy;
    let radius = crb.z;

    // only cover the visible circle area (no halo)
    let world = center + in.pos * radius;

    let x = (world.x / screen.size.x) * 2.0 - 1.0;
    let y = 1.0 - (world.y / screen.size.y) * 2.0;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.local_pos = world;
    out.circle_index = in.instance;
    return out;
}

fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    let h = hsv.x * 6.0;
    let s = hsv.y;
    let v = hsv.z;

    let i = floor(h);
    let f = h - i;

    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    if i == 0.0 { return vec3<f32>(v, t, p); }
    if i == 1.0 { return vec3<f32>(q, v, p); }
    if i == 2.0 { return vec3<f32>(p, v, t); }
    if i == 3.0 { return vec3<f32>(p, q, v); }
    if i == 4.0 { return vec3<f32>(t, p, v); }
    return vec3<f32>(v, p, q);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = circles[in.circle_index];
    let crb = p.center_radius_border;
    let center = crb.xy;
    let radius = crb.z;
    let border = crb.w;

    let dist = distance(in.local_pos, center);

    // ---- Border with outward AA ----
    let border_outer = smoothstep(radius + 1.0, radius, dist);
    let border_inner = smoothstep(radius - border, radius - border - 1.0, dist);
    let border_mask = border_outer - border_inner;

    // Base color (fill or border)
    var col = mix(p.fill_color, p.border_color, border_mask);
    col.a *= border_outer;

    // ---- Hue Circle Style ----
    if p.style == 1u && dist < radius {
        let uv = (in.local_pos - center) / radius;

        // Compute angle and hue
        let angle = atan2(uv.y, uv.x);
        let h = (angle / (2.0 * PI)) + 0.5;

        let rgb = hsv_to_rgb(vec3<f32>(h, 1.0, 1.0));

        // Replace fill area with hue (border stays)
        col = mix(vec4<f32>(rgb, 1.0), p.border_color, border_mask);
        col.a *= border_outer;
    }

    // ---- Fade Effect (Independent of Style!) ----
    if p.fade > 0.9 {
        let d = distance(in.local_pos, screen.mouse);
        let fade = clamp(1.0 - d / 300.0, 0.0, 1.0);
        col.a *= fade * fade;
    }

    return col;
}



