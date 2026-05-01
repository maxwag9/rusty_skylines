use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU16, Ordering};

pub type LodStep = u16;
pub type ChunkSize = u16;
static CHUNK_SIZE: AtomicU16 = AtomicU16::new(128);

pub fn chunk_size() -> ChunkSize {
    CHUNK_SIZE.load(Ordering::Relaxed)
}

pub fn set_chunk_size(cs: ChunkSize) {
    CHUNK_SIZE.store(cs, Ordering::Relaxed);
}
pub const CHUNK_MIN_Y: f32 = -512.0;
pub const CHUNK_MAX_Y: f32 = 4096.0;
#[derive(Serialize, Deserialize, Debug, Clone, Copy, Hash, Eq, PartialEq, Default)]
pub struct ChunkCoord {
    pub x: i32,
    pub z: i32,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub struct LocalPos {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Hash)]
pub struct WorldPos {
    pub chunk: ChunkCoord,
    pub local: LocalPos,
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
    pub fn dist2(&self, other: &ChunkCoord) -> u64 {
        let dx = self.x as i64 - other.x as i64;
        let dz = self.z as i64 - other.z as i64;
        (dx * dx + dz * dz) as u64
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
    pub fn from_world_f32(fpos: Vec3) -> Self {
        let chunk_size = chunk_size() as f32;
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
    pub fn to_render_pos(&self, cam: WorldPos) -> Vec3 {
        let cs = chunk_size() as f64;

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
    pub fn normalize(mut self) -> Self {
        let cs = chunk_size() as f32;
        // push local.x into [0, chunk_size)
        let dx = (self.local.x / cs).floor() as i32;
        self.chunk.x += dx;
        self.local.x -= dx as f32 * cs;

        let dz = (self.local.z / cs).floor() as i32;
        self.chunk.z += dz;
        self.local.z -= dz as f32 * cs;

        // handle negative edge cases (if local becomes -eps due to float error)
        if self.local.x < 0.0 {
            self.chunk.x -= 1;
            self.local.x += cs;
        }
        if self.local.z < 0.0 {
            self.chunk.z -= 1;
            self.local.z += cs;
        }

        self
    }

    pub fn add_render_offset(self, v: Vec3) -> Self {
        self.add_vec3(v)
    }

    #[inline]
    fn add_assign_vec3(&mut self, rhs: Vec3) {
        *self = self.add_vec3(rhs);
    }

    #[inline]
    pub fn sub_vec3(self, rhs: Vec3) -> WorldPos {
        self.add_vec3(-rhs)
    }

    /// Vector (delta meters) from `self` to `rhs`  (rhs - self)
    #[inline]
    pub fn delta_to(self, rhs: Self) -> Vec3 {
        let cs = chunk_size() as f32;

        let dcx = rhs.chunk.x as i64 - self.chunk.x as i64;
        let dcz = rhs.chunk.z as i64 - self.chunk.z as i64;

        Vec3::new(
            dcx as f32 * cs + (rhs.local.x - self.local.x),
            rhs.local.y - self.local.y,
            dcz as f32 * cs + (rhs.local.z - self.local.z),
        )
    }
    /// This is the SAME as delta_to(), wraps it, made for clarity!
    #[inline]
    pub fn direction_to(self, rhs: Self) -> Vec3 {
        self.delta_to(rhs)
    }
    #[inline]
    fn sub_assign_vec3(&mut self, rhs: Vec3) {
        *self = self.add_vec3(rhs);
    }
    /// WorldPos + Vec3 = moved WorldPos (meters)
    #[inline]
    pub fn add_vec3(self, rhs: Vec3) -> WorldPos {
        WorldPos {
            chunk: self.chunk,
            local: LocalPos {
                x: self.local.x + rhs.x,
                y: self.local.y + rhs.y,
                z: self.local.z + rhs.z,
            },
        }
        .normalize()
    }

    /// WorldPos + WorldPos = WorldPos
    /// Treats `rhs` as a displacement from world origin and adds it to `self`.
    /// Uses integer arithmetic for chunk components to maintain precision.
    #[inline]
    pub fn add_world_pos(self, rhs: WorldPos) -> WorldPos {
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
        .normalize()
    }

    /// WorldPos - WorldPos = WorldPos (as displacement)
    /// Returns the displacement from `rhs` to `self` as a WorldPos.
    #[inline]
    pub fn sub_world_pos(self, rhs: WorldPos) -> WorldPos {
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
        .normalize()
    }

    /// Scale the position (as displacement from origin) by a scalar.
    #[inline]
    pub fn scale(self, factor: f32) -> WorldPos {
        let cs = chunk_size() as f64;

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
    pub fn lerp(self, other: WorldPos, t: f64) -> WorldPos {
        let cs = chunk_size() as f64;
        let t64 = t;

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

    /// Distance between two WorldPos in meters (computed in WorldPos space).
    #[inline]
    pub fn distance_to(self, rhs: WorldPos) -> f64 {
        let cs = chunk_size() as f64;

        // lossless chunk delta
        let dcx = rhs.chunk.x as i64 - self.chunk.x as i64;
        let dcz = rhs.chunk.z as i64 - self.chunk.z as i64;

        // full-precision meter delta
        let dx = dcx as f64 * cs + (rhs.local.x as f64 - self.local.x as f64);
        let dy = rhs.local.y as f64 - self.local.y as f64;
        let dz = dcz as f64 * cs + (rhs.local.z as f64 - self.local.z as f64);

        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Same as distance_to
    #[inline]
    pub fn length_to(self, rhs: WorldPos) -> f64 {
        self.distance_to(rhs)
    }

    /// Often useful to avoid sqrt.
    #[inline]
    pub fn distance_squared(self, rhs: WorldPos) -> f64 {
        let cs = chunk_size() as f64;

        let dcx = rhs.chunk.x as i64 - self.chunk.x as i64;
        let dcz = rhs.chunk.z as i64 - self.chunk.z as i64;

        let dx = dcx as f64 * cs + (rhs.local.x as f64 - self.local.x as f64);
        let dy = rhs.local.y as f64 - self.local.y as f64;
        let dz = dcz as f64 * cs + (rhs.local.z as f64 - self.local.z as f64);

        dx * dx + dy * dy + dz * dz
    }

    /// Normalize as direction from origin (for displacement vectors).
    #[inline]
    pub fn normalize_direction(self) -> Vec3 {
        let origin = WorldPos::zero();
        self.to_render_pos(origin).normalize_or_zero()
    }

    /// Dot product treating both as displacement vectors.
    #[inline]
    pub fn dot(self, rhs: WorldPos) -> f32 {
        let origin = WorldPos::zero();
        let a = self.to_render_pos(origin);
        let b = rhs.to_render_pos(origin);
        a.dot(b)
    }

    #[inline]
    pub fn quadratic_bezier_xz(p0: WorldPos, p1: WorldPos, p2: WorldPos, t: f32) -> WorldPos {
        let cs = chunk_size() as f64;
        let t64 = t as f64;
        let omt = 1.0 - t64;

        let c0 = omt * omt;
        let c1 = 2.0 * omt * t64;
        let c2 = t64 * t64;

        let p0_x = p0.chunk.x as f64 * cs + p0.local.x as f64;
        let p0_z = p0.chunk.z as f64 * cs + p0.local.z as f64;

        let p1_x = p1.chunk.x as f64 * cs + p1.local.x as f64;
        let p1_z = p1.chunk.z as f64 * cs + p1.local.z as f64;

        let p2_x = p2.chunk.x as f64 * cs + p2.local.x as f64;
        let p2_z = p2.chunk.z as f64 * cs + p2.local.z as f64;

        let result_x = c0 * p0_x + c1 * p1_x + c2 * p2_x;
        let result_z = c0 * p0_z + c1 * p1_z + c2 * p2_z;

        let chunk_x = (result_x / cs).floor() as i32;
        let chunk_z = (result_z / cs).floor() as i32;

        WorldPos {
            chunk: ChunkCoord::new(chunk_x, chunk_z),
            local: LocalPos::new(
                (result_x - chunk_x as f64 * cs) as f32,
                0.0,
                (result_z - chunk_z as f64 * cs) as f32,
            ),
        }
    }

    #[inline]
    pub fn cubic_bezier_xz(
        p0: WorldPos,
        p1: WorldPos,
        p2: WorldPos,
        p3: WorldPos,
        t: f32,
    ) -> WorldPos {
        let cs = chunk_size() as f64;
        let t64 = t as f64;
        let omt = 1.0 - t64;

        let c0 = omt * omt * omt;
        let c1 = 3.0 * omt * omt * t64;
        let c2 = 3.0 * omt * t64 * t64;
        let c3 = t64 * t64 * t64;

        let p0_x = p0.chunk.x as f64 * cs + p0.local.x as f64;
        let p0_z = p0.chunk.z as f64 * cs + p0.local.z as f64;

        let p1_x = p1.chunk.x as f64 * cs + p1.local.x as f64;
        let p1_z = p1.chunk.z as f64 * cs + p1.local.z as f64;

        let p2_x = p2.chunk.x as f64 * cs + p2.local.x as f64;
        let p2_z = p2.chunk.z as f64 * cs + p2.local.z as f64;

        let p3_x = p3.chunk.x as f64 * cs + p3.local.x as f64;
        let p3_z = p3.chunk.z as f64 * cs + p3.local.z as f64;

        let result_x = c0 * p0_x + c1 * p1_x + c2 * p2_x + c3 * p3_x;
        let result_z = c0 * p0_z + c1 * p1_z + c2 * p2_z + c3 * p3_z;

        let chunk_x = (result_x / cs).floor() as i32;
        let chunk_z = (result_z / cs).floor() as i32;

        WorldPos {
            chunk: ChunkCoord::new(chunk_x, chunk_z),
            local: LocalPos::new(
                (result_x - chunk_x as f64 * cs) as f32,
                p0.local.y,
                (result_z - chunk_z as f64 * cs) as f32,
            ),
        }
    }

    /// Signed XZ offset from `self` to `other`, in world units.
    /// (chunk delta × chunk_size) + local delta
    #[inline]
    pub fn dx(self, other: WorldPos) -> f64 {
        let cs = chunk_size() as f64;
        (other.chunk.x - self.chunk.x) as f64 * cs + (other.local.x as f64 - self.local.x as f64)
    }
    /// Signed XZ offset from `self` to `other`, in world units.
    /// (chunk delta × chunk_size) + local delta
    #[inline]
    pub fn dz(self, other: WorldPos) -> f64 {
        let cs = chunk_size() as f64;
        (other.chunk.z - self.chunk.z) as f64 * cs + (other.local.z as f64 - self.local.z as f64)
    }
    pub fn area(points: &[WorldPos]) -> f64 {
        let n = points.len();
        if n < 3 {
            return 0.0;
        }

        let origin = points[0];
        let mut sum = 0.0;

        for i in 1..n - 1 {
            let a = points[i];
            let b = points[i + 1];

            let ax = origin.dx(a) as f64;
            let az = origin.dz(a) as f64;

            let bx = origin.dx(b) as f64;
            let bz = origin.dz(b) as f64;

            sum += ax * bz - az * bx;
        }

        sum.abs() * 0.5
    }
    /// Returns the area-weighted centroid of the polygon (true geometric center, may lie outside for concave shapes).
    pub fn centroid(points: &[WorldPos]) -> WorldPos {
        let n = points.len();
        if n == 0 {
            return WorldPos::zero();
        }
        if n == 1 {
            return points[0];
        }
        if n == 2 {
            return points[0].lerp(points[1], 0.5);
        }

        let origin = points[0];

        let mut area_acc = 0.0f64;
        let mut cx_acc = 0.0f64;
        let mut cz_acc = 0.0f64;

        for i in 1..n - 1 {
            let a = points[i];
            let b = points[i + 1];

            // Work in origin-relative space (stable, no precision loss)
            let ax = origin.dx(a) as f64;
            let az = origin.dz(a) as f64;

            let bx = origin.dx(b) as f64;
            let bz = origin.dz(b) as f64;

            // Triangle area (signed *2)
            let cross = ax * bz - az * bx;

            // Triangle centroid = (0 + A + B) / 3
            let tri_cx = (ax + bx) / 3.0;
            let tri_cz = (az + bz) / 3.0;

            cx_acc += tri_cx * cross;
            cz_acc += tri_cz * cross;
            area_acc += cross;
        }

        if area_acc.abs() < 1e-6 {
            // fallback: average (degenerate polygon) Because you are an insane, degenerate piece of filth, and you deserve to die!!
            let mut sum = Vec3::ZERO;
            for p in points {
                sum += origin.direction_to(*p);
            }
            let avg = sum / points.len() as f32;
            return origin.add_vec3(avg);
        }

        let inv = 1.0 / area_acc;

        let cx = cx_acc * inv;
        let cz = cz_acc * inv;

        origin.add_vec3(Vec3::new(cx as f32, 0.0, cz as f32))
    }

    /// Returns the average of all input points (fast barycenter, biased by vertex distribution).
    /// Centroid of a WorldPos slice, fully WorldPos-native (dx/dz offsets from points[0]).
    pub fn barycenter(points: &[WorldPos]) -> WorldPos {
        if points.is_empty() {
            return WorldPos::zero();
        }
        let origin = points[0];
        let n = points.len() as f32;
        let mut sum = Vec3::ZERO;
        for p in points {
            sum.x += origin.dx(*p) as f32;
            sum.y += p.local.y - origin.local.y;
            sum.z += origin.dz(*p) as f32;
        }
        origin.add_vec3(sum / n)
    }

    /// Andrew's monotone-chain convex hull over XZ, working in
    /// full-precision f64 via WorldPos::dx / dz.
    /// Computes the outermost points I guess
    pub fn convex_hull(points: &[WorldPos]) -> Vec<WorldPos> {
        if points.len() < 3 {
            return points.to_vec();
        }

        // Tag each point with its index so we can recover WorldPos after sorting.
        let origin = points[0];
        let mut tagged: Vec<(f64, f64, usize)> = points
            .iter()
            .enumerate()
            .map(|(i, p)| (origin.dx(*p), origin.dz(*p), i))
            .collect();

        // Lexicographic sort: primary X, secondary Z.
        tagged.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap()
                .then(a.1.partial_cmp(&b.1).unwrap())
        });

        // 2-D cross product of vectors O→A and O→B.
        let cross = |o: &(f64, f64, usize), a: &(f64, f64, usize), b: &(f64, f64, usize)| -> f64 {
            (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
        };

        let mut lower: Vec<usize> = Vec::new();
        for i in 0..tagged.len() {
            while lower.len() >= 2
                && cross(
                    &tagged[lower[lower.len() - 2]],
                    &tagged[lower[lower.len() - 1]],
                    &tagged[i],
                ) <= 0.0
            {
                lower.pop();
            }
            lower.push(i);
        }

        let mut upper: Vec<usize> = Vec::new();
        for i in (0..tagged.len()).rev() {
            while upper.len() >= 2
                && cross(
                    &tagged[upper[upper.len() - 2]],
                    &tagged[upper[upper.len() - 1]],
                    &tagged[i],
                ) <= 0.0
            {
                upper.pop();
            }
            upper.push(i);
        }

        // Last point of each half duplicates the first point of the other.
        lower.pop();
        upper.pop();
        lower.extend(upper);

        lower.iter().map(|&i| points[tagged[i].2]).collect()
    }
}
impl Default for WorldPos {
    fn default() -> Self {
        WorldPos::zero()
    }
}
impl fmt::Display for ChunkCoord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Chunk({}, {})", self.x, self.z)
    }
}

impl fmt::Display for LocalPos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Local({:.2}, {:.2}, {:.2})", self.x, self.y, self.z)
    }
}

impl fmt::Display for WorldPos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}", self.chunk, self.local)
    }
}
impl Hash for LocalPos {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.to_bits().hash(state);
        self.y.to_bits().hash(state);
        self.z.to_bits().hash(state);
    }
}
