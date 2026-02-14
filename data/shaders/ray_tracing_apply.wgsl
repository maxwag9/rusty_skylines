// Apply pre-upsampled shadow visibility to HDR via multiply blend
// Blend state: src=Zero, dst=Src → result = hdr * shadow_factor

@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var rt_visibility_full : texture_2d<f32>;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    // Fullscreen triangle
    var p = vec2<f32>(-1.0, -1.0);
    if (vi == 1u) { p = vec2<f32>(3.0, -1.0); }
    if (vi == 2u) { p = vec2<f32>(-1.0, 3.0); }
    return VSOut(vec4<f32>(p, 0.0, 1.0));
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(rt_visibility_full));
    let uv = in.pos.xy / dims;

    let vis = textureSample(rt_visibility_full, samp, uv).r;

    // Ambient floor - shadows never fully black
    let shadow_factor = vis * 0.7 + 0.3;

    return vec4<f32>(shadow_factor, shadow_factor, shadow_factor, 1.0);
}
