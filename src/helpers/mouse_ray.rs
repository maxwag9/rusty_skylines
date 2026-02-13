use crate::helpers::positions::{ChunkCoord, ChunkSize, LocalPos, WorldPos};
use crate::terrain::chunk_builder::ChunkHeightGrid;
use glam::{Mat4, Vec2, Vec3, Vec4};
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PickUniform {
    pub pos: [f32; 3],
    pub radius: f32, // offset 16

    pub underwater: u32, // offset 20
    pub _pad0: [u32; 3], // pad to 32

    pub color: [f32; 3], // offset 32
    pub _pad1: f32,      // pad to 48
}
/// A ray represented in world space using WorldPos for maximum precision.
/// The origin uses chunk-relative coordinates, preserving precision at any distance.
#[derive(Clone, Copy, Debug)]
pub struct WorldRay {
    pub origin: WorldPos,
    pub dir: Vec3, // Normalized direction (f32 is fine for unit vectors)
}
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3, // normalized
}
impl WorldRay {
    /// Create a WorldRay from camera position and direction.
    #[inline]
    pub fn new(origin: WorldPos, dir: Vec3) -> Self {
        Self { origin, dir }
    }

    /// Create a WorldRay from a camera-relative render ray.
    /// `render_ray` has origin relative to camera (typically Vec3::ZERO for mouse picking).
    #[inline]
    pub fn from_render_ray(render_ray: &Ray, camera: &WorldPos, chunk_size: ChunkSize) -> Self {
        let origin = camera.add_render_offset(render_ray.origin, chunk_size);
        Self {
            origin,
            dir: render_ray.dir,
        }
    }

    /// Compute t value to reach an X chunk boundary.
    /// `boundary_chunk_x` is the chunk index where the boundary lies
    /// (boundary is at x = boundary_chunk_x * chunk_size).
    #[inline]
    pub(crate) fn t_to_x_boundary(&self, boundary_chunk_x: i32, chunk_size: ChunkSize) -> f32 {
        if self.dir.x.abs() < 1e-12 {
            return f32::INFINITY;
        }

        let cs_f64 = chunk_size as f64;

        // Distance = (boundary_chunk - origin_chunk) * chunk_size - origin_local
        // This keeps the large values in integer math as long as possible
        let chunk_diff = (boundary_chunk_x - self.origin.chunk.x) as f64;
        let distance = chunk_diff * cs_f64 - self.origin.local.x as f64;

        (distance / self.dir.x as f64) as f32
    }

    /// Compute t value to reach a Z chunk boundary.
    #[inline]
    pub(crate) fn t_to_z_boundary(&self, boundary_chunk_z: i32, chunk_size: ChunkSize) -> f32 {
        if self.dir.z.abs() < 1e-12 {
            return f32::INFINITY;
        }

        let cs_f64 = chunk_size as f64;
        let chunk_diff = (boundary_chunk_z - self.origin.chunk.z) as f64;
        let distance = chunk_diff * cs_f64 - self.origin.local.z as f64;

        (distance / self.dir.z as f64) as f32
    }

    /// Convert to a local Ray for intersection with a specific chunk's heightgrid.
    /// Uses f64 intermediate math to preserve precision.
    #[inline]
    pub fn to_local_ray(&self, target_chunk: ChunkCoord, chunk_size: ChunkSize) -> Ray {
        let cs_f64 = chunk_size as f64;

        // Compute origin relative to target chunk using f64
        // local = (origin_chunk - target_chunk) * chunk_size + origin_local
        let chunk_diff_x = (self.origin.chunk.x - target_chunk.x) as f64;
        let chunk_diff_z = (self.origin.chunk.z - target_chunk.z) as f64;

        let local_x = chunk_diff_x * cs_f64 + self.origin.local.x as f64;
        let local_y = self.origin.local.y as f64;
        let local_z = chunk_diff_z * cs_f64 + self.origin.local.z as f64;

        Ray {
            origin: Vec3::new(local_x as f32, local_y as f32, local_z as f32),
            dir: self.dir,
        }
    }

    /// Get position along ray at parameter t with f64 precision,
    /// useful for very long rays.
    pub fn at_precise(&self, t: f64, chunk_size: ChunkSize) -> WorldPos {
        let cs_f64 = chunk_size as f64;

        // Compute world position in f64
        let world_x = self.origin.chunk.x as f64 * cs_f64
            + self.origin.local.x as f64
            + self.dir.x as f64 * t;
        let world_y = self.origin.local.y as f64 + self.dir.y as f64 * t;
        let world_z = self.origin.chunk.z as f64 * cs_f64
            + self.origin.local.z as f64
            + self.dir.z as f64 * t;

        // Convert back to WorldPos
        let chunk_x = (world_x / cs_f64).floor() as i32;
        let chunk_z = (world_z / cs_f64).floor() as i32;

        let local_x = (world_x - chunk_x as f64 * cs_f64) as f32;
        let local_z = (world_z - chunk_z as f64 * cs_f64) as f32;

        WorldPos::new(
            ChunkCoord::new(chunk_x, chunk_z),
            LocalPos::new(local_x, world_y as f32, local_z),
        )
    }

    /// Create WorldRay directly from mouse position and camera.
    pub fn from_mouse(
        mouse_px: Vec2,
        screen_width: f32,
        screen_height: f32,
        view: Mat4,
        proj: Mat4,
        camera: WorldPos,
        chunk_size: ChunkSize,
    ) -> Self {
        let ndc_x = (mouse_px.x / screen_width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (mouse_px.y / screen_height) * 2.0;

        let inv_view_proj = (proj * view).inverse();

        let p_near4 = inv_view_proj * Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
        let p_far4 = inv_view_proj * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

        let p_near = p_near4.truncate() / p_near4.w;
        let p_far = p_far4.truncate() / p_far4.w;

        let dir = (p_far - p_near).normalize();

        // p_near is the offset from camera in render space
        let origin = camera.add_render_offset(p_near, chunk_size);

        Self { origin, dir }
    }

    /// Compute t to reach a chunk X boundary.
    /// Boundary is at `boundary_chunk_x * chunk_size` in world X.
    /// Uses integer chunk arithmetic for precision.
    #[inline]
    pub fn t_to_chunk_x_boundary(&self, boundary_chunk_x: i32, chunk_size: ChunkSize) -> f32 {
        if self.dir.x.abs() < 1e-12 {
            return f32::INFINITY;
        }
        // Distance = (boundary_chunk - origin.chunk) * cs - origin.local.x
        // chunk_diff is exact integer, multiplication by cs is precise for reasonable cs
        let chunk_diff = boundary_chunk_x - self.origin.chunk.x;
        let cs = chunk_size as f32;
        let distance = chunk_diff as f32 * cs - self.origin.local.x;
        distance / self.dir.x
    }

    /// Compute t to reach a chunk Z boundary.
    #[inline]
    pub fn t_to_chunk_z_boundary(&self, boundary_chunk_z: i32, chunk_size: ChunkSize) -> f32 {
        if self.dir.z.abs() < 1e-12 {
            return f32::INFINITY;
        }
        let chunk_diff = boundary_chunk_z - self.origin.chunk.z;
        let cs = chunk_size as f32;
        let distance = chunk_diff as f32 * cs - self.origin.local.z;
        distance / self.dir.z
    }

    /// Get point along ray at parameter t as WorldPos.
    #[inline]
    pub fn at(&self, t: f32, chunk_size: ChunkSize) -> WorldPos {
        self.origin.add_render_offset(self.dir * t, chunk_size)
    }

    /// Compute a WorldPos relative to ray origin as Vec3.
    /// The result is small and precise when `point` is near `origin`.
    #[inline]
    pub fn relative_position(&self, point: &WorldPos, chunk_size: ChunkSize) -> Vec3 {
        point.to_render_pos(self.origin, chunk_size)
    }

    /// Compute a grid vertex position relative to ray origin.
    /// `grid_chunk` is the chunk the grid belongs to.
    /// `local_x`, `local_y`, `local_z` are the vertex's local coordinates within that chunk.
    #[inline]
    pub fn grid_vertex_relative(
        &self,
        grid_chunk: ChunkCoord,
        local_x: f32,
        local_y: f32,
        local_z: f32,
        chunk_size: ChunkSize,
    ) -> Vec3 {
        let cs = chunk_size as f32;
        // Compute using chunk differences (small integers) for precision
        let chunk_diff_x = grid_chunk.x - self.origin.chunk.x;
        let chunk_diff_z = grid_chunk.z - self.origin.chunk.z;

        Vec3::new(
            chunk_diff_x as f32 * cs + local_x - self.origin.local.x,
            local_y - self.origin.local.y,
            chunk_diff_z as f32 * cs + local_z - self.origin.local.z,
        )
    }
}

/// Raycast against a chunk's heightgrid using WorldRay.
/// All calculations use chunk-relative arithmetic for precision.
/// Returns (t, hit_position) where hit_position is a WorldPos.
pub fn raycast_chunk_heightgrid(
    ray: &WorldRay,
    grid: &ChunkHeightGrid,
    t_min: f32,
    t_max: f32,
) -> Option<(f32, WorldPos)> {
    let chunk_size = grid.chunk_size;
    let cs = chunk_size as f32;
    let cell = grid.cell_f32();
    let eps = 1e-6 * cell;

    let size_x = grid.extent_x();
    let size_z = grid.extent_z();

    // Compute ray origin relative to grid chunk using chunk differences
    let chunk_diff_x = ray.origin.chunk.x - grid.chunk_coord.x;
    let chunk_diff_z = ray.origin.chunk.z - grid.chunk_coord.z;
    let ox = chunk_diff_x as f32 * cs + ray.origin.local.x;
    let oz = chunk_diff_z as f32 * cs + ray.origin.local.z;

    // XZ slab test
    let mut t0 = t_min.max(0.0);
    let mut t1 = t_max;

    for (o, d, size) in [(ox, ray.dir.x, size_x), (oz, ray.dir.z, size_z)] {
        if d.abs() < 1e-10 {
            if o < 0.0 || o > size {
                return None;
            }
        } else {
            let inv = 1.0 / d;
            let mut a = -o * inv;
            let mut b = (size - o) * inv;
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            t0 = t0.max(a);
            t1 = t1.min(b);
            if t0 > t1 {
                return None;
            }
        }
    }

    // DDA start
    let mut t = t0 + eps;
    let p_x = ox + ray.dir.x * t;
    let p_z = oz + ray.dir.z * t;

    let mut ix = (p_x / cell).floor() as i32;
    let mut iz = (p_z / cell).floor() as i32;
    ix = ix.clamp(0, grid.nx as i32 - 2);
    iz = iz.clamp(0, grid.nz as i32 - 2);

    let step_x = if ray.dir.x >= 0.0 { 1i32 } else { -1i32 };
    let step_z = if ray.dir.z >= 0.0 { 1i32 } else { -1i32 };

    // Compute t to next cell boundaries using local coordinates
    let next_local_x = (ix + if step_x > 0 { 1 } else { 0 }) as f32 * cell;
    let next_local_z = (iz + if step_z > 0 { 1 } else { 0 }) as f32 * cell;

    let mut t_max_x = if ray.dir.x.abs() < 1e-10 {
        f32::INFINITY
    } else {
        (next_local_x - ox) / ray.dir.x
    };

    let mut t_max_z = if ray.dir.z.abs() < 1e-10 {
        f32::INFINITY
    } else {
        (next_local_z - oz) / ray.dir.z
    };

    let t_delta_x = if ray.dir.x.abs() < 1e-10 {
        f32::INFINITY
    } else {
        cell / ray.dir.x.abs()
    };

    let t_delta_z = if ray.dir.z.abs() < 1e-10 {
        f32::INFINITY
    } else {
        cell / ray.dir.z.abs()
    };

    let mut prev_t = t;

    // DDA loop
    while t <= t1 {
        if ix < 0 || iz < 0 || ix >= grid.nx as i32 - 1 || iz >= grid.nz as i32 - 1 {
            break;
        }

        let seg_end = t_max_x.min(t_max_z).min(t1);

        if let Some(hit_t) = ray_hit_cell(ray, grid, ix as usize, iz as usize, prev_t, seg_end) {
            // Compute hit position as WorldPos using ray.at()
            let hit_pos = ray.at(hit_t, chunk_size);

            // Refine Y using bilinear height sampling
            let hit_y = height_bilinear_world(grid, hit_pos);
            let refined_pos = WorldPos::new(
                hit_pos.chunk,
                LocalPos::new(hit_pos.local.x, hit_y, hit_pos.local.z),
            );

            return Some((hit_t, refined_pos));
        }

        prev_t = seg_end;

        dda_advance(
            &mut ix,
            &mut iz,
            step_x,
            step_z,
            &mut t,
            &mut t_max_x,
            &mut t_max_z,
            t_delta_x,
            t_delta_z,
        );
    }

    None
}

/// Test ray against a heightgrid cell, computing vertices relative to ray origin.
#[inline]
fn ray_hit_cell(
    ray: &WorldRay,
    grid: &ChunkHeightGrid,
    ix: usize,
    iz: usize,
    t_min: f32,
    t_max: f32,
) -> Option<f32> {
    let cell = grid.cell_f32();
    let chunk_size = grid.chunk_size;

    // Cell corner local positions within the grid's chunk
    let x0 = ix as f32 * cell;
    let z0 = iz as f32 * cell;
    let x1 = x0 + cell;
    let z1 = z0 + cell;

    let i = ix * grid.nz + iz;

    let h00 = grid.heights[i];
    let h10 = grid.heights[i + grid.nz];
    let h01 = grid.heights[i + 1];
    let h11 = grid.heights[i + grid.nz + 1];

    // Compute vertices RELATIVE to ray origin using WorldPos arithmetic
    // This keeps all values small and precise regardless of world position
    let v00 = ray.grid_vertex_relative(grid.chunk_coord, x0, h00, z0, chunk_size);
    let v10 = ray.grid_vertex_relative(grid.chunk_coord, x1, h10, z0, chunk_size);
    let v01 = ray.grid_vertex_relative(grid.chunk_coord, x0, h01, z1, chunk_size);
    let v11 = ray.grid_vertex_relative(grid.chunk_coord, x1, h11, z1, chunk_size);

    // Ray origin is at Vec3::ZERO in this relative coordinate system
    let mut best: Option<f32> = None;

    for (a, b, c) in [(v00, v10, v01), (v01, v10, v11)] {
        if let Some(t) = ray_tri_origin_zero(ray.dir, a, b, c) {
            if t >= t_min && t <= t_max {
                best = Some(best.map_or(t, |bt| bt.min(t)));
            }
        }
    }

    best
}

#[inline]
fn ray_tri_origin_zero(rd: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let ab = b - a;
    let ac = c - a;
    let p = rd.cross(ac);
    let det = ab.dot(p);

    if det.abs() < 1e-10 {
        return None;
    }

    let inv = 1.0 / det;
    let tvec = -a; // ro - a where ro = Vec3::ZERO
    let u = tvec.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let q = tvec.cross(ab);
    let v = rd.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = ac.dot(q) * inv;
    if t >= 0.0 { Some(t) } else { None }
}

/// Advance the DDA grid traversal.
#[inline]
pub fn dda_advance(
    cell_a: &mut i32,
    cell_b: &mut i32,
    step_a: i32,
    step_b: i32,
    t: &mut f32,
    t_max_a: &mut f32,
    t_max_b: &mut f32,
    t_delta_a: f32,
    t_delta_b: f32,
) {
    const TIE_EPS: f32 = 1e-7;

    let tie = (*t_max_a - *t_max_b).abs() < TIE_EPS;

    if tie {
        *cell_a += step_a;
        *cell_b += step_b;
        *t = *t_max_a;
        *t_max_a += t_delta_a;
        *t_max_b += t_delta_b;
    } else if *t_max_a < *t_max_b {
        *cell_a += step_a;
        *t = *t_max_a;
        *t_max_a += t_delta_a;
    } else {
        *cell_b += step_b;
        *t = *t_max_b;
        *t_max_b += t_delta_b;
    }
}

/// Bilinear height interpolation using LOCAL coordinates within the chunk.
/// `local_x` and `local_z` should be in range [0, chunk_size].
#[inline(always)]
pub fn height_bilinear(grid: &ChunkHeightGrid, local_x: f32, local_z: f32) -> f32 {
    let cell = grid.cell_f32();
    let eps = 1e-6 * cell;

    let max_x = grid.extent_x();
    let max_z = grid.extent_z();

    // Clamp to valid range
    let x = local_x.clamp(eps, max_x - eps);
    let z = local_z.clamp(eps, max_z - eps);

    let fx = x / cell;
    let fz = z / cell;

    let mut ix = fx.floor() as i32;
    let mut iz = fz.floor() as i32;

    // Clamp cell indices
    ix = ix.clamp(0, grid.nx as i32 - 2);
    iz = iz.clamp(0, grid.nz as i32 - 2);

    let ix = ix as usize;
    let iz = iz as usize;

    let tx = fx - ix as f32;
    let tz = fz - iz as f32;

    let i = ix * grid.nz + iz;

    let h00 = grid.heights[i];
    let h10 = grid.heights[i + grid.nz];
    let h01 = grid.heights[i + 1];
    let h11 = grid.heights[i + grid.nz + 1];

    let hx0 = h00 + (h10 - h00) * tx;
    let hx1 = h01 + (h11 - h01) * tx;
    hx0 + (hx1 - hx0) * tz
}

/// Sample height at a WorldPos by converting to local coords for the grid's chunk.
/// Returns None if the position is outside this chunk.
#[inline]
pub fn height_bilinear_world(grid: &ChunkHeightGrid, pos: WorldPos) -> f32 {
    height_bilinear(grid, pos.local.x, pos.local.z)
}

/// Create a Ray in camera-relative render space from mouse pixels.
/// The ray origin is at Vec3::ZERO (camera position in render space).
pub fn ray_from_mouse_pixels(
    mouse_px: glam::Vec2,
    width: f32,
    height: f32,
    view: glam::Mat4,
    proj: glam::Mat4,
) -> Ray {
    // Pixel -> NDC
    let ndc_x = (mouse_px.x / width) * 2.0 - 1.0;
    let ndc_y = 1.0 - (mouse_px.y / height) * 2.0;

    let inv_view_proj = (proj * view).inverse();

    // Unproject near and far points
    let p_near4 = inv_view_proj * glam::Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
    let p_far4 = inv_view_proj * glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

    let p_near = p_near4.truncate() / p_near4.w;
    let p_far = p_far4.truncate() / p_far4.w;

    let dir = (p_far - p_near).normalize();

    // Origin at camera (render space origin)
    Ray {
        origin: Vec3::ZERO,
        dir,
    }
}
