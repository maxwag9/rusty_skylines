use glam::{Mat4, Vec3};

pub(crate) struct Camera {
    pub(crate) target: Vec3,
    pub(crate) radius: f32,
    pub(crate) yaw: f32,   // radians, rotates around +Y
    pub(crate) pitch: f32, // radians, elevation, clamp to (-PI/2, PI/2)
}

impl Camera {
    pub(crate) fn new() -> Self {
        Self {
            target: Vec3::ZERO,
            radius: 5.0,
            yaw: -45f32.to_radians(),
            pitch: 20f32.to_radians(),
        }
    }

    pub(crate) fn position(&self) -> Vec3 {
        // Spherical coordinates, RH system
        // x = r cos(pitch) cos(yaw)
        // y = r sin(pitch)
        // z = r cos(pitch) sin(yaw)
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

    pub(crate) fn view_proj(&self, aspect: f32) -> [[f32; 4]; 4] {
        let eye = self.position();
        let view = Mat4::look_at_rh(eye, self.target, Vec3::Y);
        let proj = Mat4::perspective_rh_gl(45f32.to_radians(), aspect, 0.1, 1_000.0);
        (proj * view).to_cols_array_2d()
    }
}
