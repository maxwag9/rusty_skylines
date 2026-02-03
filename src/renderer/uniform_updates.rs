use crate::components::camera::Camera;
use crate::data::Settings;
use crate::renderer::astronomy::AstronomyState;
use crate::renderer::gtao::gtao::{GtaoApplyParams, GtaoParams, GtaoUpsampleParams};
use crate::renderer::pipelines::{
    FogUniforms, Pipelines, ToneMappingState, ToneMappingUniforms, make_new_uniforms_csm,
};
use crate::renderer::shadows::compute_csm_matrices;
use crate::resources::TimeSystem;
use crate::terrain::sky::SkyUniform;
use crate::terrain::water::WaterUniform;
use glam::Mat4;
use wgpu::Queue;

pub struct UniformUpdater<'a> {
    queue: &'a Queue,
    pipelines: &'a mut Pipelines,
}

impl<'a> UniformUpdater<'a> {
    pub fn new(queue: &'a Queue, pipelines: &'a mut Pipelines) -> Self {
        Self { queue, pipelines }
    }

    pub fn update_camera_uniforms(
        &mut self,
        view: Mat4,
        proj: Mat4,
        view_proj: Mat4,
        astronomy: &AstronomyState,
        camera: &Camera,
        total_time: f64,
        aspect: f32,
        settings: &Settings,
    ) {
        // Build 4 cascade matrices + splits (defaults baked in: shadow distance, lambda, padding).
        let (light_mats, splits) = compute_csm_matrices(
            view,
            camera.fov.to_radians(),
            aspect,
            camera.near,
            camera.far,
            astronomy.sun_dir,
            /*shadow_map_size:*/ self.pipelines.resources.csm_shadows.size,
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
            &self.pipelines.buffers.camera,
            0,
            bytemuck::bytes_of(&new_uniforms),
        );
        self.pipelines.resources.csm_shadows.light_mats = light_mats;
        self.pipelines.resources.csm_shadows.splits = splits;
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
            &self.pipelines.buffers.fog,
            0,
            bytemuck::bytes_of(&fog_uniforms),
        );
    }
    pub fn update_tonemapping_uniforms(&self, tonemapping_state: &ToneMappingState) {
        let tonemapping_uniforms = ToneMappingUniforms::from_state(tonemapping_state);

        self.queue.write_buffer(
            &self.pipelines.buffers.tonemapping,
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
            &self.pipelines.buffers.sky,
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

        self.queue
            .write_buffer(&self.pipelines.buffers.water, 0, bytemuck::bytes_of(&wu));
    }
    pub fn update_ssao_uniforms(&self, time: &TimeSystem, settings: &Settings) {
        let screen_size = [
            self.pipelines.post_fx.linear_depth_half.texture().width() as f32,
            self.pipelines.post_fx.linear_depth_half.texture().height() as f32,
        ];
        let inv_screen_size = [1.0 / screen_size[0], 1.0 / screen_size[1]];
        let gtao_params = GtaoParams {
            radius_world: 1.0,
            intensity: 1.5,
            bias: 0.02,
            frame_index: time.frame_count,
            screen_size,
            inv_screen_size,
        };
        let full_width = self.pipelines.post_fx.linear_depth_full.texture().width();
        let full_height = self.pipelines.post_fx.linear_depth_full.texture().height();
        let half_width = self.pipelines.post_fx.linear_depth_half.texture().width();
        let half_height = self.pipelines.post_fx.linear_depth_half.texture().height();
        let upsample_params = GtaoUpsampleParams {
            full_size: [full_width as f32, full_height as f32],
            half_size: [half_width as f32, half_height as f32],
            inv_full_size: [1.0 / full_width as f32, 1.0 / full_height as f32],
            inv_half_size: [1.0 / half_width as f32, 1.0 / half_height as f32],
            depth_threshold: 0.1,  // Adjust based on scene scale
            normal_threshold: 0.9, // Cosine of angle (approx 25 degrees)
            use_normal_check: 1,
            _padding: 0,
        };

        // 2. WRITE TO GPU (This was missing)
        self.queue.write_buffer(
            &self.pipelines.buffers.gtao_upsample,
            0,
            bytemuck::bytes_of(&upsample_params),
        );
        self.queue.write_buffer(
            &self.pipelines.buffers.gtao,
            0,
            bytemuck::bytes_of(&gtao_params),
        );
        // Update apply params
        let apply_params = GtaoApplyParams {
            power: 1.5,     // e.g., 1.5
            intensity: 1.0, // e.g., 1.0
            min_ao: 0.1,    // e.g., 0.1
            debug_mode: 0,
        };

        self.queue.write_buffer(
            &self.pipelines.buffers.gtao_apply,
            0,
            bytemuck::bytes_of(&apply_params),
        );
    }
}
