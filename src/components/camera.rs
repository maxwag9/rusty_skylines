use crate::mouse_ray::*;
use crate::renderer::world_renderer::WorldRenderer;
use crate::terrain::TerrainGenerator;
use glam::Vec3;

#[derive(Debug, Clone)]
pub struct Camera {
    pub target: Vec3,
    pub orbit_radius: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub pitch_resolved: Option<f32>,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            target: Vec3::new(0.0, 800.0, 0.0),
            orbit_radius: 1000.0,
            yaw: -45f32.to_radians(),
            pitch: 20f32.to_radians(),
            pitch_resolved: None,
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

        let proj = glam::Mat4::perspective_rh(60f32.to_radians(), aspect, 5.0, 100_000.0);

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
pub fn ground_camera_target(camera: &mut Camera, terrain: &TerrainGenerator, min_clearance: f32) {
    let x = camera.target.x;
    let z = camera.target.z;

    let ground_y = terrain.height(x, z);

    if camera.target.y < ground_y + min_clearance {
        camera.target.y = ground_y + min_clearance;
    }
}
pub fn resolve_pitch_by_search(camera: &mut Camera, terrain: &WorldRenderer) {
    let pitch_user = camera.pitch;
    let pitch_max = 85.0_f32.to_radians();

    let baseline = 4.0_f32.to_radians();
    let release_margin = 1.5_f32.to_radians(); // hysteresis

    // If we already have a _resolved pitch, try to release it
    if camera.pitch_resolved.is_some() {
        // Only release if user pitch is clearly safe
        if is_clear(camera, terrain, pitch_user - release_margin) {
            camera.pitch_resolved = None;
            return;
        }
    }

    // Try user pitch directly
    if is_clear(camera, terrain, pitch_user - release_margin) {
        return;
    }

    // Binary search upward
    let mut lo = pitch_user;
    let mut hi = pitch_max;

    if !is_clear(camera, terrain, hi) {
        return;
    }

    for _ in 0..12 {
        let mid = 0.5 * (lo + hi);
        if is_clear(camera, terrain, mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let solved = hi + baseline;

    camera.pitch = solved;
    camera.pitch_resolved = Some(solved);
}

fn is_clear(camera: &Camera, world: &WorldRenderer, pitch: f32) -> bool {
    let mut tmp = camera.clone();
    tmp.pitch = pitch;

    let origin = tmp.position();
    let target = tmp.target;

    let v = target - origin;
    let dist = v.length();
    if dist < 1e-4 {
        return true;
    }

    let ray = Ray {
        origin,
        dir: v / dist,
    };

    let cs = world.chunk_size as f32;
    let mut t = 0.0;

    while t < dist {
        let p = ray.origin + ray.dir * t;
        let cx = (p.x / cs).floor() as i32;
        let cz = (p.z / cs).floor() as i32;

        let Some(chunk) = world.chunks.get(&(cx, cz)) else {
            t += cs;
            continue;
        };

        let grid = chunk.height_grid.as_ref();

        if let Some((t_hit, _)) = raycast_chunk_heightgrid(ray, grid, t, dist) {
            return false;
        }

        t += cs;
    }

    true
}
