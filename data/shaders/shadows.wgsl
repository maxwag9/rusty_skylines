struct ShadowUniform {
    light_view_proj: mat4x4<f32>,
};
@group(1) @binding(0)
var<uniform> uniforms: ShadowUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> @builtin(position) vec4<f32> {
    let m = uniforms.light_view_proj;
    return m * vec4<f32>(in.position, 1.0);
}