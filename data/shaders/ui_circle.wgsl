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
    // center.x, center.y, radius, outer_border_thickness_percentage
    center_radius_border: vec4<f32>,
    fill_color: vec4<f32>,
    inside_border_color: vec4<f32>,
    border_color: vec4<f32>,
    glow_color: vec4<f32>,
    glow_misc: vec4<f32>,
    misc: vec4<f32>,

    fade: f32,
    style: u32,
    inside_border_thickness_percentage: f32,
    _pad0: u32,
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

    // Only cover the visible circle area (no halo)
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

    // --- CHANGES HERE ---
    // Convert normalized percentage (0.0 - 1.0) to absolute pixels
    let border_thick = crb.w * radius;
    let inside_thick = p.inside_border_thickness_percentage * radius;

    let dist = distance(in.local_pos, center);

    // 1. Start with the Base Fill Color
    var col = p.fill_color;

    // ---- Blender style hue wheel ----
    // This effectively replaces the Fill Color
    if p.style == 1u {
        let uv = (in.local_pos - center) / radius;

        // Hue from angle
        let angle = atan2(uv.y, uv.x);
        let angle_shifted = angle + (PI / 2.0);
        let angle_wrapped = atan2(sin(angle_shifted), cos(angle_shifted));
        let h = (angle_wrapped / (2.0 * PI)) + 0.5;

        // Saturation
        let s_linear = clamp(length(uv), 0.0, 1.0);
        let s = pow(s_linear, 0.47);
        let v = 1.0;

        let rgb = hsv_to_rgb(vec3<f32>(h, s, v));
        col = vec4<f32>(rgb, 1.0);
    }

    // 2. Mix Inside Border (Layered on top of Fill)
    // The boundary between Fill and Inside Border
    if inside_thick > 0.0 {
        // Calculate the inner edge pixel position based on the calculated thicknesses
        let edge_fill_inner = radius - border_thick - inside_thick;

        // 0.0 = Fill, 1.0 = Inside Border (and everything outwards)
        let aa_inner = smoothstep(edge_fill_inner - 0.5, edge_fill_inner + 0.5, dist);

        col = mix(col, p.inside_border_color, aa_inner);
    }

    // 3. Mix Outer Border (Layered on top of Inside Border + Fill)
    // The boundary between Inside Border and Outer Border
    if border_thick > 0.0 {
        let edge_inner_outer = radius - border_thick;

        // 0.0 = Previous Layers, 1.0 = Outer Border (and everything outwards)
        let aa_outer = smoothstep(edge_inner_outer - 0.5, edge_inner_outer + 0.5, dist);

        col = mix(col, p.border_color, aa_outer);
    }

    // 4. Apply Shape Alpha (The actual outer edge of the circle)
    // Everything outside 'radius' fades to transparent
    let alpha_shape = smoothstep(radius + 0.5, radius - 0.5, dist);
    col.a *= alpha_shape;

    // ---- Fade logic ----
    if p.fade > 0.9 {
        let d = distance(in.local_pos, screen.mouse);
        let fade = clamp(1.0 - d / 300.0, 0.0, 1.0);
        col.a *= fade * fade;
    }
    if p.misc.z > 0.5 && p.style != 1u {
        col.a *= 0.9;
    }

    return col;
}
