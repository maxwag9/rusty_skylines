use crate::components::camera::Camera;
use crate::renderer::astronomy::AstronomyState;
use crate::renderer::pipelines::{FogUniforms, Pipelines, make_new_uniforms};
use crate::renderer::shadows::compute_light_matrix;
use crate::terrain::sky::SkyUniform;
use crate::terrain::water::WaterUniform;
use glam::Mat4;
use wgpu::Queue;

pub struct UniformUpdater<'a> {
    queue: &'a Queue,
    pipelines: &'a Pipelines,
}

impl<'a> UniformUpdater<'a> {
    pub fn new(queue: &'a Queue, pipelines: &'a Pipelines) -> Self {
        Self { queue, pipelines }
    }

    pub fn update_camera_uniforms(
        &self,
        view: Mat4,
        proj: Mat4,
        view_proj: Mat4,
        astronomy: &AstronomyState,
        camera: &Camera,
        total_time: f32,
        aspect: f32,
    ) {
        // let light_matrix = compute_light_matrix_fit_to_camera(camera.position(), camera.target, camera.fov.to_radians(), aspect,
        //                                                       camera.near,250.0, astronomy.sun_dir, 2048.0, true, true);
        let light_matrix = compute_light_matrix(camera.target, astronomy.sun_dir);
        let new_uniforms = make_new_uniforms(
            view,
            proj,
            view_proj,
            astronomy.sun_dir,
            astronomy.moon_dir,
            camera.position(),
            camera.orbit_radius,
            total_time,
            light_matrix,
        );
        self.queue.write_buffer(
            &self.pipelines.uniforms.buffer,
            0,
            bytemuck::bytes_of(&new_uniforms),
        );
    }

    pub fn update_fog_uniforms(&self, config: &wgpu::SurfaceConfiguration, camera: &Camera) {
        let fog_uniforms = FogUniforms {
            screen_size: [config.width as f32, config.height as f32],
            proj_params: [camera.near, camera.far],
            fog_density: 0.15,
            fog_height: 0.0,
            cam_height: camera.position().y,
            _pad0: 0.0,
            fog_color: [0.55, 0.55, 0.6],
            _pad1: 0.0,
            fog_sky_factor: 0.41,
            fog_height_falloff: 0.0,
            fog_start: 1000.0,
            fog_end: 5000.0,
        };

        self.queue.write_buffer(
            &self.pipelines.fog_uniforms.buffer,
            0,
            bytemuck::bytes_of(&fog_uniforms),
        );
    }

    pub fn update_sky_uniforms(&self, moon_phase: f32) {
        let sky_uniform = SkyUniform {
            exposure: 1.0,
            moon_phase,
            sun_size: 0.0465,
            sun_intensity: 1.0,
            moon_size: 0.03,
            moon_intensity: 1.0,
            _pad1: 1.0,
            _pad2: 0.0,
        };

        self.queue.write_buffer(
            &self.pipelines.sky_uniforms.buffer,
            0,
            bytemuck::bytes_of(&sky_uniform),
        );
    }

    pub fn update_water_uniforms(&self) {
        let wu = WaterUniform {
            sea_level: 0.0,
            _pad0: [0.0; 3],
            color: [0.05, 0.25, 0.35, 0.95],
            wave_tiling: 0.5,
            wave_strength: 0.1,
            _pad1: [0.0; 2],
        };

        self.queue.write_buffer(
            &self.pipelines.water_uniforms.buffer,
            0,
            bytemuck::bytes_of(&wu),
        );
    }
}
