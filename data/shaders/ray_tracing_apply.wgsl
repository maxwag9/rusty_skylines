#include "includes/uniforms.wgsl"
// Apply pre-upsampled shadow visibility to HDR via multiply blend
// Blend state: src=Zero, dst=Src → result = hdr * shadow_factor

@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var rt_visibility_full : texture_2d<f32>;
@group(1) @binding(0) var<uniform> uniforms: Uniforms;

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
    let base_shadow = vis * 0.9 + 0.1;

    let elev = max(uniforms.sun_direction.y, 0.0);
    let shadow_strength = clamp(1.0 - exp2(-elev * 20.0), 0.0, 1.0);

    let twilight = 1.0 - smoothstep(0.02, 0.15, elev);
    let tint = mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 0.85, 0.65), twilight);

    let shadow_rgb = mix(vec3<f32>(1.0, 1.0, 1.0), tint * base_shadow, shadow_strength);

    return vec4<f32>(shadow_rgb, 1.0);
}