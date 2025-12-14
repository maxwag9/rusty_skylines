use crate::renderer::pipelines::Pipelines;
use wgpu::RenderPass;

const STAR_COUNT: u32 = 116812;

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
        pass.set_pipeline(&pipelines.stars_pipeline);
        pass.set_bind_group(0, &pipelines.uniform_bind_group, &[]);
        pass.set_vertex_buffer(0, pipelines.stars_vertex_buffer.slice(..));
        pass.draw(0..4, 0..STAR_COUNT); // 4 verts, STAR_COUNT instances

        pass.set_pipeline(&pipelines.sky_pipeline);
        pass.set_bind_group(0, &pipelines.uniform_bind_group, &[]);
        pass.set_bind_group(1, &pipelines.sky_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyUniform {
    pub exposure: f32,
    pub moon_phase: f32,

    pub sun_size: f32,
    pub sun_intensity: f32,

    pub moon_size: f32,
    pub moon_intensity: f32,

    pub _pad1: f32,
    pub _pad2: f32,
}
