use crate::components::camera::Camera;
use crate::data::Settings;
use crate::renderer::astronomy::AstronomyState;
use crate::renderer::pipelines::{
    FogUniforms, Pipelines, ToneMappingState, ToneMappingUniforms, make_new_uniforms_csm,
};
use crate::renderer::pipelines_outsource::{SsaoUniforms, make_ssao_kernel};
use crate::renderer::shadows::{CSM_CASCADES, compute_csm_matrices};
use crate::resources::Uniforms;
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
        total_time: f64,
        aspect: f32,
        settings: &Settings,
    ) -> (Uniforms, [Mat4; CSM_CASCADES], [f32; 4]) {
        // Build 4 cascade matrices + splits (defaults baked in: shadow distance, lambda, padding).
        let (light_mats, splits) = compute_csm_matrices(
            view,
            camera.fov.to_radians(),
            aspect,
            camera.near,
            camera.far,
            astronomy.sun_dir,
            /*shadow_map_size:*/ self.pipelines.cascaded_shadow_map.size,
            /*stabilize:*/ true,
            settings.reversed_depth_z,
        );

        // This is the uniforms used for *normal* rendering (shadow_cascade_index unused there).
        let new_uniforms = make_new_uniforms_csm(
            view,
            proj,
            view_proj,
            astronomy.sun_dir,
            astronomy.moon_dir,
            total_time,
            light_mats,
            splits,
            camera,
            settings,
        );

        self.queue.write_buffer(
            &self.pipelines.uniforms.buffer,
            0,
            bytemuck::bytes_of(&new_uniforms),
        );
        (new_uniforms, light_mats, splits)
    }

    pub fn update_fog_uniforms(&self, config: &wgpu::SurfaceConfiguration, camera: &Camera) {
        let fog_uniforms = FogUniforms {
            screen_size: [config.width as f32, config.height as f32],
            proj_params: [camera.near, camera.far],
            fog_density: 1.0,
            fog_height: 200.0,
            cam_height: camera.target.local.y,
            _pad0: 0.0,
            fog_color: [0.55, 0.55, 0.7],
            _pad1: 0.0,
            fog_sky_factor: 0.05,
            fog_height_falloff: 0.0,
            fog_start: camera.far * 0.70,
            fog_end: camera.far * 1.05,
        };

        self.queue.write_buffer(
            &self.pipelines.fog_uniforms.buffer,
            0,
            bytemuck::bytes_of(&fog_uniforms),
        );
    }
    pub fn update_tonemapping_uniforms(&self, tonemapping_state: &ToneMappingState) {
        let tonemapping_uniforms = ToneMappingUniforms::from_state(tonemapping_state);

        self.queue.write_buffer(
            &self.pipelines.tonemapping_uniforms.buffer,
            0,
            bytemuck::bytes_of(&tonemapping_uniforms),
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
    pub fn update_ssao_uniforms(&self, settings: &Settings) {
        let radius = 5.0f32;
        let bias = 0.20f32;
        let intensity = 1.2f32;
        let power = 1.22f32;
        let reversed_z = settings.reversed_depth_z as u32;
        let noise_tile_px = 8u32;
        let params = SsaoUniforms {
            kernel: make_ssao_kernel(69420, 2.0),
            params0: [radius, bias, intensity, power],
            params1: [reversed_z, noise_tile_px, 0, 0],
        };

        self.queue.write_buffer(
            &self.pipelines.ssao_uniforms.buffer,
            0,
            bytemuck::bytes_of(&params),
        );
    }
}
