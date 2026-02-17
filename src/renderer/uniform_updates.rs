use crate::data::Settings;
use crate::renderer::gtao::gtao::GtaoParams;
use crate::renderer::pipelines::{
    FogUniforms, Pipelines, ToneMappingState, ToneMappingUniforms, make_new_camera_uniforms,
};
use crate::renderer::shadows::compute_csm_matrices;
use crate::resources::TimeSystem;
use crate::world::astronomy::AstronomyState;
use crate::world::camera::Camera;
use crate::world::terrain::sky::SkyUniform;
use crate::world::terrain::terrain_subsystem::TerrainSubsystem;
use crate::world::terrain::water::WaterUniform;
use glam::Mat4;
use wgpu::{Queue, SurfaceConfiguration};

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
        terrain_renderer: &TerrainSubsystem,
        astronomy: &AstronomyState,
        camera: &Camera,
        time_system: &TimeSystem,
        aspect: f32,
        settings: &Settings,
        config: &SurfaceConfiguration,
    ) {
        // Build 4 cascade matrices + splits (defaults baked in: shadow distance, lambda, padding).
        let (light_mats, splits, texels) = compute_csm_matrices(
            terrain_renderer,
            camera,
            aspect,
            astronomy.sun_dir,
            self.pipelines.resources.csm_shadows.size,
            true,
            settings.reversed_depth_z,
        );
        self.pipelines.resources.csm_shadows.texels = texels;
        // This is the uniforms used for *normal* rendering (shadow_cascade_index unused there).
        let new_uniforms = make_new_camera_uniforms(
            astronomy.sun_dir,
            astronomy.moon_dir,
            time_system,
            light_mats,
            splits,
            camera,
            settings,
            config,
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
            fog_density: 1.0,
            fog_height: 200.0,
            _pad0: 0.0,
            fog_color: [0.55, 0.55, 0.7],
            _pad1: 0.0,
            fog_sky_factor: 0.05,
            fog_height_falloff: 0.0,
            fog_start: camera.far * 0.70,
            fog_end: camera.far * 1.05,
            _pad2: 0.0,
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
    pub fn update_sky_uniforms(&self, astronomy: &AstronomyState) {
        let moon_phase: f32 = astronomy.moon_phase;
        let sky_uniform = SkyUniform {
            star_rotation: astronomy.star_rotation.to_cols_array_2d(),
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
            _pad0: [1.0; 3],
            color: [0.05, 0.25, 0.35, 0.95],
            wave_tiling: 0.5,
            wave_strength: 0.1,
            _pad1: [1.0; 2],
        };

        self.queue
            .write_buffer(&self.pipelines.buffers.water, 0, bytemuck::bytes_of(&wu));
    }
    pub fn update_ssao_uniforms(
        &self,
        time: &TimeSystem,
        settings: &Settings,
        prev_view_proj: Mat4,
    ) {
        let half_w = self.pipelines.post_fx.linear_depth_half.texture().width();
        let half_h = self.pipelines.post_fx.linear_depth_half.texture().height();
        let hw = half_w as f32;
        let hh = half_h as f32;

        let temporal_blend = if time.frame_count == 0 { 1.0_f32 } else { 0.1 };

        let ao_params = GtaoParams {
            radius_world: 2.0,
            intensity: 1.6,
            bias: 0.02,
            frame_index: time.frame_count as u32,
            screen_size: [hw, hh],
            inv_screen_size: [1.0 / hw, 1.0 / hh],
            temporal_blend,
            _pad: [0; 3],
            prev_view_proj: prev_view_proj.to_cols_array_2d(),
        };
        self.queue.write_buffer(
            &self.pipelines.buffers.gtao,
            0,
            bytemuck::bytes_of(&ao_params),
        );
    }
}
