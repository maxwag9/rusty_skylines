@group(0) @binding(0)
var<uniform> screen: vec2<f32>;

struct CircleParams {
    // (cx, cy, radius, border)
    center_radius_border: vec4<f32>,
    // straight-alpha colors
    fill_color:   vec4<f32>,
    border_color: vec4<f32>,
    glow_color:   vec4<f32>,
    // (glow_size, pad, pad, pad)
    glow_misc:    vec4<f32>,
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
    @location(0) color: vec4<f32>,
    @location(1) local_pos: vec2<f32>,
    @location(2) circle_index: u32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let params = circles[in.instance];
    let crb = params.center_radius_border;

    let center = crb.xy;
    let radius = crb.z;

    // in.pos is [-1..1] quad vertex
    let world = center + in.pos * radius;

    let x = (world.x / screen.x) * 2.0 - 1.0;
    let y = 1.0 - (world.y / screen.y) * 2.0;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.local_pos = world;
    out.circle_index = in.instance;
    out.color = in.color;

    return out;
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let params = circles[in.circle_index];
    let crb = params.center_radius_border;

    let center = crb.xy;
    let radius = crb.z;
    let border = crb.w;

    let dist = distance(in.local_pos, center);

    // glow
    let glow_size = params.glow_misc.x;
    let glow_alpha = smoothstep(radius + glow_size, radius, dist);
    var glow_col = params.glow_color;
    glow_col.a *= glow_alpha;

    // edges
    let outer_edge = smoothstep(radius, radius - 1.0, dist);
    let inner_edge = smoothstep(radius - border, radius - border - 1.0, dist);
    let border_mask = outer_edge - inner_edge;

    // base color
    var col = mix(params.fill_color, params.border_color, border_mask);
    col.a *= outer_edge;

    // additive glow (expand swizzle)
    col = vec4<f32>(
        col.r + glow_col.r * glow_col.a,
        col.g + glow_col.g * glow_col.a,
        col.b + glow_col.b * glow_col.a,
        col.a
    );

    return col;

}
