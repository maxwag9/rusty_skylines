use crate::renderer::world_renderer::TerrainRenderer;
use crate::terrain::terrain::TerrainGenerator;
use glam::Vec3;

#[derive(Debug, Clone)]
pub struct Camera {
    pub target: Vec3,
    pub orbit_radius: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            target: Vec3::new(0.0, 10.0, 0.0),
            orbit_radius: 50.0,
            yaw: -230f32.to_radians(),
            pitch: 52f32.to_radians(),
        }
    }

    pub fn position(&self) -> Vec3 {
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        let offset = Vec3::new(
            self.orbit_radius * cp * cy,
            self.orbit_radius * sp,
            self.orbit_radius * cp * sy,
        );
        self.target + offset
    }

    pub fn matrices(&self, aspect: f32) -> (glam::Mat4, glam::Mat4, glam::Mat4) {
        let eye = self.position();

        let view = glam::Mat4::look_at_rh(eye, self.target, Vec3::Y);

        let proj = glam::Mat4::perspective_rh(80f32.to_radians(), aspect, 0.5, 10_000.0);

        let view_proj = proj * view;

        (view, proj, view_proj)
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
            orbit_damping_release: 4.0,
            zoom_damping: 12.0,
        }
    }
}
pub fn ground_camera_target(
    camera: &mut Camera,
    camera_controller: &mut CameraController,
    terrain: &TerrainGenerator,
    min_clearance: f32,
) {
    let x = camera.target.x;
    let z = camera.target.z;

    let ground_y = terrain.height(x, z);

    let penetration = (ground_y + min_clearance) - camera.target.y;

    if penetration > 0.0 {
        camera.target.y += penetration;
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

    let samples = 1; // number of points along the orbit line
    let mut max_terrain_y = f32::MIN;

    for i in 0..=samples {
        let t = i as f32 / samples as f32;
        let cam_pos = camera.position() * t + target * (1.0 - t); // interpolate along line
        let terrain_y = world_renderer.terrain_gen.height(cam_pos.x, cam_pos.z);
        max_terrain_y = max_terrain_y.max(terrain_y);
    }

    let min_clearance = 1.0; // small extra buffer
    let desired_y = max_terrain_y + min_clearance;

    let dy = desired_y - target.y;
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
fn is_clear_strict(camera: &Camera, world: &TerrainRenderer, pitch: f32) -> bool {
    let mut tmp = camera.clone();
    tmp.pitch = pitch;

    let orbit_samples = 16;
    for i in 0..=orbit_samples {
        let t = i as f32 / orbit_samples as f32;
        let pos = tmp.position() * t + tmp.target * (1.0 - t);
        let ground_y = world.terrain_gen.height(pos.x, pos.z);
        if pos.y < ground_y + 2.0 {
            // a small clearance buffer
            return false;
        }
    }
    true
}
