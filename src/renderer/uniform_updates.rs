use crate::renderer::astronomy::AstronomyState;
use crate::renderer::pipelines::{FogUniforms, Pipelines, make_new_uniforms};
use crate::terrain::sky::SkyUniform;
use crate::terrain::water::WaterUniform;
use crate::ui::vertex::LineVtx;

pub struct UniformUpdater<'a> {
    queue: &'a wgpu::Queue,
    pipelines: &'a Pipelines,
}

impl<'a> UniformUpdater<'a> {
    pub fn new(queue: &'a wgpu::Queue, pipelines: &'a Pipelines) -> Self {
        Self { queue, pipelines }
    }

    pub fn update_camera_uniforms(
        &self,
        view: glam::Mat4,
        proj: glam::Mat4,
        view_proj: glam::Mat4,
        astronomy: &AstronomyState,
        cam_pos: glam::Vec3,
        orbit_radius: f32,
        total_time: f32,
    ) {
        let new_uniforms = make_new_uniforms(
            view,
            proj,
            view_proj,
            astronomy.sun_dir,
            astronomy.moon_dir,
            cam_pos,
            orbit_radius,
            total_time,
        );
        self.queue.write_buffer(
            &self.pipelines.uniforms.buffer,
            0,
            bytemuck::bytes_of(&new_uniforms),
        );
    }

    pub fn update_fog_uniforms(
        &self,
        config: &wgpu::SurfaceConfiguration,
        view: glam::Mat4,
        cam_height: f32,
    ) {
        let proj_params = [view.col(2).z, view.col(3).z];

        let fog_uniforms = FogUniforms {
            screen_size: [config.width as f32, config.height as f32],
            proj_params,
            fog_density: 0.0000,
            fog_height: 0.0,
            cam_height,
            _pad0: 0.0,
            fog_color: [0.55, 0.55, 0.6],
            _pad1: 0.0,
            fog_sky_factor: 0.4,
            fog_height_falloff: 0.12,
            fog_start: 1000.0,
            fog_end: 10000.0,
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

    pub fn update_gizmo_vertices(&self, target: glam::Vec3, orbit_radius: f32) {
        let t = target;
        let s = orbit_radius * 0.2;

        let axes = [
            LineVtx {
                pos: [t.x, t.y, t.z],
                color: [1.0, 0.2, 0.2],
            },
            LineVtx {
                pos: [t.x + s, t.y, t.z],
                color: [1.0, 0.2, 0.2],
            },
            LineVtx {
                pos: [t.x, t.y, t.z],
                color: [0.2, 1.0, 0.2],
            },
            LineVtx {
                pos: [t.x, t.y + s, t.z],
                color: [0.2, 1.0, 0.2],
            },
            LineVtx {
                pos: [t.x, t.y, t.z],
                color: [0.2, 0.6, 1.0],
            },
            LineVtx {
                pos: [t.x, t.y, t.z + s],
                color: [0.2, 0.6, 1.0],
            },
        ];

        self.queue.write_buffer(
            &self.pipelines.gizmo_mesh_buffers.vertex,
            0,
            bytemuck::cast_slice(&axes),
        );
    }
}
