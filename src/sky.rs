use crate::renderer::pipelines::Pipelines;
use wgpu::RenderPass;

pub struct SkyRenderer;

impl SkyRenderer {
    pub(crate) fn new() -> Self {
        // nothing lol
        SkyRenderer
    }
}

impl SkyRenderer {
    pub fn render(&self, pass: &mut RenderPass, pipelines: &Pipelines) {
        // just render
        pass.set_pipeline(&pipelines.sky_pipeline);
        pass.set_bind_group(0, &pipelines.uniform_bind_group, &[]);
        pass.set_bind_group(1, &pipelines.sky_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyUniform {
    pub day_time: f32,
    pub day_length: f32,

    pub exposure: f32,
    pub _pad0: f32,

    pub sun_size: f32,
    pub sun_intensity: f32,

    pub moon_size: f32,
    pub moon_intensity: f32,

    pub moon_phase: f32,
    pub _pad1: f32,
}
