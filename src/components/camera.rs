use glam::{Mat4, Vec3};

#[derive(Debug, Clone)]
pub struct Camera {
    pub target: Vec3,
    pub radius: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            target: Vec3::ZERO,
            radius: 5.0,
            yaw: -45f32.to_radians(),
            pitch: 20f32.to_radians(),
        }
    }

    pub fn position(&self) -> Vec3 {
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        let offset = Vec3::new(
            self.radius * cp * cy,
            self.radius * sp,
            self.radius * cp * sy,
        );
        self.target + offset
    }

    pub fn view_proj(&self, aspect: f32) -> [[f32; 4]; 4] {
        let eye = self.position();
        let view = Mat4::look_at_rh(eye, self.target, Vec3::Y);
        let proj = Mat4::perspective_rh_gl(45f32.to_radians(), aspect, 0.1, 1_000.0);
        (proj * view).to_cols_array_2d()
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
