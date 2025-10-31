struct ScreenUniform {
    size: vec2<f32>,
    time: f32,
    enable_dither: u32,
};

@group(0) @binding(0)
var<uniform> screen: ScreenUniform;

struct CircleParams {
    center_radius_border: vec4<f32>, // cx, cy, radius, border
    fill_color:   vec4<f32>,
    border_color: vec4<f32>,
    glow_color:   vec4<f32>,
    glow_misc:    vec4<f32>,         // (glow_size, ...)
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = circles[in.circle_index];
    let crb = p.center_radius_border;
    let center = crb.xy;
    let radius = crb.z;
    let border = crb.w;

    let dist = distance(in.local_pos, center);

    // Smooth edge
    let outer_edge = smoothstep(radius, radius - 1.0, dist);
    let inner_edge = smoothstep(radius - border, radius - border - 1.0, dist);
    let border_mask = outer_edge - inner_edge;

    // Fill + border
    var col = mix(p.fill_color, p.border_color, border_mask);
    col.a *= outer_edge;

    return col;
}
