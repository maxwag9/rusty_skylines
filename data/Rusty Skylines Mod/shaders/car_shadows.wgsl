
struct ShadowUniform {
    light_view_proj: mat4x4<f32>,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) uv: vec2<f32>,
};

struct InstanceInput {
    @location(4) model_col0: vec4<f32>,
    @location(5) model_col1: vec4<f32>,
    @location(6) model_col2: vec4<f32>,
    @location(7) model_col3: vec4<f32>,
    @location(8) color: vec4<f32>, // rgb + pad (unused in shadow pass)
};

@group(1) @binding(0) var<uniform> shadow_uniforms: ShadowUniform;

@vertex
fn vs_main(vertex: VertexInput, instance: InstanceInput) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(
        instance.model_col0,
        instance.model_col1,
        instance.model_col2,
        instance.model_col3
    );

    let world_pos = model * vec4<f32>(vertex.position, 1.0);

    return shadow_uniforms.light_view_proj * world_pos;
}