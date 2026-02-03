// fullscreen_copy.wgsl
@group(0) @binding(0) var src_sampler: sampler;
@group(0) @binding(1) var src_tex: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var verts = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var uv_coords = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(0.0, 2.0),
    );

    var out: VertexOutput;
    out.position = vec4<f32>(verts[idx], 0.0, 1.0);
    out.uv = uv_coords[idx];
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    // Sample the single-sample Fogged HDR texture
    let col = textureSample(src_tex, src_sampler, uv);
    return col; // Into the msaa hdr view!
}
