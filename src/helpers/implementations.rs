use glam::Vec3;
use revision::revisioned;

#[revisioned(revision = 1)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SerializableVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
impl SerializableVec3 {
    pub fn as_vec3(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}
impl From<glam::Vec3> for SerializableVec3 {
    fn from(v: glam::Vec3) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

impl From<SerializableVec3> for Vec3 {
    fn from(v: SerializableVec3) -> Self {
        glam::Vec3::new(v.x, v.y, v.z)
    }
}
