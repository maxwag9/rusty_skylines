use glam::Vec3;

pub trait Falloff {
    fn weight(d2: f32, r2: f32) -> f32;
}

pub struct SmoothFalloff;
impl Falloff for SmoothFalloff {
    fn weight(d2: f32, r2: f32) -> f32 {
        let t = 1.0 - (d2 / r2);
        if t <= 0.0 {
            return 0.0;
        }
        t * t * (3.0 - 2.0 * t)
    }
}

pub trait BrushOp {
    fn apply(y: &mut f32, strength: f32, weight: f32);
}

pub struct Raise;
impl BrushOp for Raise {
    fn apply(y: &mut f32, strength: f32, weight: f32) {
        *y += strength * weight;
    }
}
pub fn affected_chunks(center: Vec3, radius: f32, chunk_size: f32) -> (i32, i32, i32, i32) {
    let min_x = ((center.x - radius) / chunk_size).floor() as i32;
    let max_x = ((center.x + radius) / chunk_size).floor() as i32;
    let min_z = ((center.z - radius) / chunk_size).floor() as i32;
    let max_z = ((center.z + radius) / chunk_size).floor() as i32;
    (min_x, max_x, min_z, max_z)
}
