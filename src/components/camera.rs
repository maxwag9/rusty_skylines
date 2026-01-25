use crate::positions::*;
use crate::renderer::world_renderer::TerrainRenderer;
use glam::{Mat4, Vec3};

#[derive(Debug, Clone)]
pub struct Camera {
    pub target: WorldPos,
    pub orbit_radius: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub near: f32,
    pub far: f32,
    pub fov: f32,
    pub(crate) chunk_size: ChunkSize,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            target: WorldPos::zero(),
            orbit_radius: 50.0,
            yaw: -230f32.to_radians(),
            pitch: 52f32.to_radians(),
            near: 2.5,
            far: 10_000.0,
            fov: 80.0,
            chunk_size: 256,
        }
    }
    #[inline]
    pub fn orbit_offset(&self) -> Vec3 {
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();

        Vec3::new(
            self.orbit_radius * cp * cy,
            self.orbit_radius * sp,
            self.orbit_radius * cp * sy,
        )
    }

    pub fn matrices(&self, aspect: f32) -> (Mat4, Mat4, Mat4) {
        let eye = Vec3::ZERO;
        let target = -self.orbit_offset();

        let view = Mat4::look_at_rh(eye, target, Vec3::Y);
        let proj = Mat4::perspective_rh(self.fov.to_radians(), aspect, self.near, self.far);

        (view, proj, proj * view)
    }
    #[inline]
    pub fn eye_world(&self) -> WorldPos {
        self.target
            .add_render_offset(self.orbit_offset(), self.chunk_size)
    }

    #[inline]
    pub fn world_to_render(&self, pos: WorldPos) -> Vec3 {
        let eye = self.eye_world();
        pos.to_render_pos(eye, self.chunk_size) // subtract eye, not target
    }

    #[inline]
    pub fn render_to_world(&self, render_pos: Vec3) -> WorldPos {
        self.eye_world()
            .add_render_offset(render_pos, self.chunk_size)
    }
}

#[derive(Debug, Clone)]
pub struct CameraController {
    pub velocity: Vec3,
    pub zoom_velocity: f32,
    pub target_yaw: f32,
    pub target_pitch: f32,
    pub orbit_smoothness: f32,
    pub yaw_velocity: f32,
    pub pitch_velocity: f32,
    pub orbit_damping_release: f32,
    pub zoom_damping: f32,
    pub base_fov: f32,
    pub follow_vel: Vec3, // spring state for target follow
    pub fov_vel: f32,     // optional (nice) spring state for fov
}

impl CameraController {
    pub fn new(camera: &Camera) -> Self {
        Self {
            velocity: Vec3::ZERO,
            zoom_velocity: 0.0,
            target_yaw: camera.yaw,
            target_pitch: camera.pitch,
            orbit_smoothness: 0.25,
            yaw_velocity: 0.0,
            pitch_velocity: 0.0,
            orbit_damping_release: 2.0,
            zoom_damping: 12.0,
            base_fov: camera.fov,
            follow_vel: Default::default(),
            fov_vel: 0.0,
        }
    }
}
pub fn ground_camera_target(
    camera: &mut Camera,
    camera_controller: &mut CameraController,
    terrain: &TerrainRenderer,
    min_clearance: f32,
) {
    let ground_y = terrain.get_height_at(camera.target);
    let penetration = (ground_y + min_clearance) - camera.target.local.y;

    if penetration > 0.0 {
        camera.target.local.y += penetration;
        camera_controller.velocity.y = camera_controller.velocity.y.max(0.0);
    }
}
pub fn resolve_pitch_by_search(
    camera: &mut Camera,
    camera_controller: &mut CameraController,
    world_renderer: &TerrainRenderer,
) {
    let target = camera.target;
    let orbit_radius = camera.orbit_radius;

    let samples = 8;
    let mut max_terrain_y = f32::MIN;

    let offset = camera.orbit_offset(); // Vec3 in meters

    for i in 0..=samples {
        let t = i as f32 / samples as f32;
        let sample_pos: WorldPos = target.add_vec3(offset * t, world_renderer.chunk_size); // uses WorldPos + Vec3

        let terrain_y = world_renderer.get_height_at(sample_pos);
        max_terrain_y = max_terrain_y.max(terrain_y);
    }

    let min_clearance = 1.0;
    let desired_y = max_terrain_y + min_clearance;

    let dy = desired_y - target.local.y;
    let horizontal_dist = orbit_radius;

    let new_pitch = (dy / horizontal_dist)
        .asin()
        .clamp(-85.0_f32.to_radians(), 85.0_f32.to_radians());

    if new_pitch > camera.pitch || new_pitch > camera_controller.target_pitch {
        camera_controller.target_pitch = new_pitch;
        camera.pitch = new_pitch;
        camera_controller.pitch_velocity = 0.0;
    }
}

// Stricter version: checks multiple points along orbit for terrain collision
fn is_clear_strict(camera: &Camera, terrain: &TerrainRenderer, pitch: f32) -> bool {
    let mut tmp = camera.clone();
    tmp.pitch = pitch;

    let orbit_samples = 16;
    let offset = tmp.orbit_offset();

    for i in 0..=orbit_samples {
        let t = i as f32 / orbit_samples as f32;
        let pos: WorldPos = tmp.target.add_vec3(offset * t, terrain.chunk_size);

        let ground_y = terrain.get_height_at(pos);
        if pos.local.y < ground_y + 2.0 {
            return false;
        }
    }
    true
}
