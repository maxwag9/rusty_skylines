use glam::Vec3;
use serde::{Deserialize, Serialize};

pub type LodStep = u16;
pub type ChunkSize = u16;
pub const CHUNK_MIN_Y: f32 = -512.0;
pub const CHUNK_MAX_Y: f32 = 4096.0;
#[derive(Serialize, Deserialize, Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct ChunkCoord {
    pub(crate) x: i32,
    pub(crate) z: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct LocalPos {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) z: f32,
}
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct WorldPos {
    pub(crate) chunk: ChunkCoord,
    pub(crate) local: LocalPos,
}

impl ChunkCoord {
    #[inline]
    pub fn new(x: i32, z: i32) -> Self {
        Self { x, z }
    }

    #[inline]
    pub fn zero() -> Self {
        Self { x: 0, z: 0 }
    }

    #[inline]
    pub fn offset(self, dx: i32, dz: i32) -> Self {
        Self {
            x: self.x + dx,
            z: self.z + dz,
        }
    }
    #[inline]
    pub fn dist2(&self, other: &ChunkCoord) -> u32 {
        let dx = self.x as i64 - other.x as i64;
        let dz = self.z as i64 - other.z as i64;
        (dx * dx + dz * dz) as u32
    }
}
impl LocalPos {
    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    #[inline]
    pub fn as_vec3(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}
impl WorldPos {
    #[inline]
    pub fn new(chunk: ChunkCoord, local: LocalPos) -> Self {
        Self { chunk, local }
    }

    #[inline]
    pub fn zero() -> Self {
        Self {
            chunk: ChunkCoord::zero(),
            local: LocalPos::zero(),
        }
    }

    #[inline]
    pub fn from_world_f32(fpos: Vec3, chunk_size: f32) -> Self {
        let cx = (fpos.x / chunk_size).floor() as i32;
        let cz = (fpos.z / chunk_size).floor() as i32;

        Self {
            chunk: ChunkCoord { x: cx, z: cz },
            local: LocalPos {
                x: fpos.x - cx as f32 * chunk_size,
                y: fpos.y,
                z: fpos.z - cz as f32 * chunk_size,
            },
        }
    }

    #[inline]
    pub fn to_render_pos(&self, cam: WorldPos, chunk_size: ChunkSize) -> Vec3 {
        let cs = chunk_size as f64;

        // 1. Calculate delta in INTEGERS first (lossless)
        let dcx = self.chunk.x as i64 - cam.chunk.x as i64;
        let dcz = self.chunk.z as i64 - cam.chunk.z as i64;

        // 2. Convert to f64 for the multiply (safe up to 9 quadrillion chunks)
        let dx_chunk = dcx as f64 * cs;
        let dz_chunk = dcz as f64 * cs;

        // 3. Add local offsets in f64
        let dx = dx_chunk + (self.local.x as f64 - cam.local.x as f64);
        let dy = self.local.y as f64 - cam.local.y as f64;
        let dz = dz_chunk + (self.local.z as f64 - cam.local.z as f64);

        // 4. Finally cast to f32.
        // Since this result is relative to the camera, it is small and fits in f32.
        Vec3::new(dx as f32, dy as f32, dz as f32)
    }
    pub fn normalize(mut self, chunk_size: ChunkSize) -> Self {
        // push local.x into [0, chunk_size)
        let dx = (self.local.x / chunk_size as f32).floor() as i32;
        self.chunk.x += dx;
        self.local.x -= dx as f32 * chunk_size as f32;

        let dz = (self.local.z / chunk_size as f32).floor() as i32;
        self.chunk.z += dz;
        self.local.z -= dz as f32 * chunk_size as f32;

        // handle negative edge cases (if local becomes -eps due to float error)
        if self.local.x < 0.0 {
            self.chunk.x -= 1;
            self.local.x += chunk_size as f32;
        }
        if self.local.z < 0.0 {
            self.chunk.z -= 1;
            self.local.z += chunk_size as f32;
        }

        self
    }

    pub fn add_render_offset(self, v: Vec3, chunk_size: ChunkSize) -> Self {
        WorldPos {
            chunk: self.chunk,
            local: LocalPos {
                x: self.local.x + v.x,
                y: self.local.y + v.y,
                z: self.local.z + v.z,
            },
        }
        .normalize(chunk_size)
    }

    #[inline]
    fn add_assign_vec3(&mut self, rhs: Vec3, chunk_size: ChunkSize) {
        *self = self.add_vec3(rhs, chunk_size);
    }

    #[inline]
    pub(crate) fn sub_vec3(self, rhs: Vec3, chunk_size: ChunkSize) -> WorldPos {
        self.add_vec3(-rhs, chunk_size)
    }
    /// WorldPos - WorldPos = delta meters (Vec3)
    #[inline]
    pub(crate) fn delta_to(self, rhs: Self, chunk_size: ChunkSize) -> Vec3 {
        let cs = chunk_size as f32;
        let dcx = self.chunk.x as i64 - rhs.chunk.x as i64;
        let dcz = self.chunk.z as i64 - rhs.chunk.z as i64;

        Vec3::new(
            dcx as f32 * cs + (self.local.x - rhs.local.x),
            self.local.y - rhs.local.y,
            dcz as f32 * cs + (self.local.z - rhs.local.z),
        )
    }

    #[inline]
    fn sub_assign_vec3(&mut self, rhs: Vec3, chunk_size: ChunkSize) {
        *self = self.add_vec3(rhs, chunk_size);
    }
    /// WorldPos + Vec3 = moved WorldPos (meters)
    #[inline]
    pub fn add_vec3(self, rhs: Vec3, chunk_size: ChunkSize) -> WorldPos {
        WorldPos {
            chunk: self.chunk,
            local: LocalPos {
                x: self.local.x + rhs.x,
                y: self.local.y + rhs.y,
                z: self.local.z + rhs.z,
            },
        }
        .normalize(chunk_size)
    }

    /// WorldPos + WorldPos = WorldPos
    /// Treats `rhs` as a displacement from world origin and adds it to `self`.
    /// Uses integer arithmetic for chunk components to maintain precision.
    #[inline]
    pub fn add_world_pos(self, rhs: WorldPos, chunk_size: ChunkSize) -> WorldPos {
        WorldPos {
            chunk: ChunkCoord {
                x: self.chunk.x + rhs.chunk.x,
                z: self.chunk.z + rhs.chunk.z,
            },
            local: LocalPos {
                x: self.local.x + rhs.local.x,
                y: self.local.y + rhs.local.y,
                z: self.local.z + rhs.local.z,
            },
        }
        .normalize(chunk_size)
    }

    /// WorldPos - WorldPos = WorldPos (as displacement)
    /// Returns the displacement from `rhs` to `self` as a WorldPos.
    #[inline]
    pub fn sub_world_pos(self, rhs: WorldPos, chunk_size: ChunkSize) -> WorldPos {
        WorldPos {
            chunk: ChunkCoord {
                x: self.chunk.x - rhs.chunk.x,
                z: self.chunk.z - rhs.chunk.z,
            },
            local: LocalPos {
                x: self.local.x - rhs.local.x,
                y: self.local.y - rhs.local.y,
                z: self.local.z - rhs.local.z,
            },
        }
        .normalize(chunk_size)
    }

    /// Scale the position (as displacement from origin) by a scalar.
    #[inline]
    pub fn scale(self, factor: f32, chunk_size: ChunkSize) -> WorldPos {
        let cs = chunk_size as f64;

        // Convert to world coordinates in f64
        let world_x = self.chunk.x as f64 * cs + self.local.x as f64;
        let world_z = self.chunk.z as f64 * cs + self.local.z as f64;

        // Scale
        let scaled_x = world_x * factor as f64;
        let scaled_y = self.local.y as f64 * factor as f64;
        let scaled_z = world_z * factor as f64;

        // Convert back
        let chunk_x = (scaled_x / cs).floor() as i32;
        let chunk_z = (scaled_z / cs).floor() as i32;

        WorldPos {
            chunk: ChunkCoord::new(chunk_x, chunk_z),
            local: LocalPos::new(
                (scaled_x - chunk_x as f64 * cs) as f32,
                scaled_y as f32,
                (scaled_z - chunk_z as f64 * cs) as f32,
            ),
        }
    }

    /// Lerp between two WorldPos with maximum precision.
    #[inline]
    pub fn lerp(self, other: WorldPos, t: f32, chunk_size: ChunkSize) -> WorldPos {
        let cs = chunk_size as f64;
        let t64 = t as f64;

        // Convert both to f64 world coords
        let self_x = self.chunk.x as f64 * cs + self.local.x as f64;
        let self_y = self.local.y as f64;
        let self_z = self.chunk.z as f64 * cs + self.local.z as f64;

        let other_x = other.chunk.x as f64 * cs + other.local.x as f64;
        let other_y = other.local.y as f64;
        let other_z = other.chunk.z as f64 * cs + other.local.z as f64;

        // Lerp in f64
        let result_x = self_x + (other_x - self_x) * t64;
        let result_y = self_y + (other_y - self_y) * t64;
        let result_z = self_z + (other_z - self_z) * t64;

        // Convert back
        let chunk_x = (result_x / cs).floor() as i32;
        let chunk_z = (result_z / cs).floor() as i32;

        WorldPos {
            chunk: ChunkCoord::new(chunk_x, chunk_z),
            local: LocalPos::new(
                (result_x - chunk_x as f64 * cs) as f32,
                result_y as f32,
                (result_z - chunk_z as f64 * cs) as f32,
            ),
        }
    }

    /// Distance between two WorldPos in meters.
    #[inline]
    pub fn distance_to(self, rhs: WorldPos, chunk_size: ChunkSize) -> f32 {
        self.to_render_pos(rhs, chunk_size).length()
    }
    /// Distance between two WorldPos in meters.
    #[inline]
    pub fn length_to(self, rhs: WorldPos, chunk_size: ChunkSize) -> f32 {
        self.distance_to(rhs, chunk_size)
    }

    /// Distance squared (cheaper, for comparisons).
    #[inline]
    pub fn distance_squared(self, rhs: WorldPos, chunk_size: ChunkSize) -> f32 {
        self.to_render_pos(rhs, chunk_size).length_squared()
    }

    /// Normalize as direction from origin (for displacement vectors).
    #[inline]
    pub fn normalize_direction(self, chunk_size: ChunkSize) -> Vec3 {
        let origin = WorldPos::zero();
        self.to_render_pos(origin, chunk_size).normalize_or_zero()
    }

    /// Dot product treating both as displacement vectors.
    #[inline]
    pub fn dot(self, rhs: WorldPos, chunk_size: ChunkSize) -> f32 {
        let origin = WorldPos::zero();
        let a = self.to_render_pos(origin, chunk_size);
        let b = rhs.to_render_pos(origin, chunk_size);
        a.dot(b)
    }
}
impl Default for WorldPos {
    fn default() -> Self {
        WorldPos::zero()
    }
}
