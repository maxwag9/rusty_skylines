struct VertexInput {
    @location(0) pos: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) local_pos: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> screen: vec2<f32>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let x = (in.pos.x / screen.x) * 2.0 - 1.0;
    let y = 1.0 - (in.pos.y / screen.y) * 2.0;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.color = in.color;
    out.local_pos = in.pos;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var col = in.color;

    // Smooth circle logic
    let center = vec2<f32>(80.0, screen.y - 80.0);
    let radius = 40.0;
    let dist = distance(in.local_pos, center);
    let edge = smoothstep(radius + 1.0, radius - 1.0, dist);
    col.a *= edge;

    return col;
}
