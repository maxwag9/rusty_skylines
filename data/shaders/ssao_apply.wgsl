@group(0) @binding(0) var ao_tex: texture_2d<f32>;

struct VsOut { @builtin(position) pos: vec4<f32>, };

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var o: VsOut;
    o.pos = vec4<f32>(positions[vi], 0.0, 1.0);
    return o;
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(frag_pos.xy);
    let ao = textureLoad(ao_tex, pix, 0).r; // 0..1
    return vec4<f32>(ao, ao, ao, 1.0);     // multiply blended into HDR
}