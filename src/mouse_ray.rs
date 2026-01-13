use crate::terrain::chunk_builder::ChunkHeightGrid;
use glam::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3, // normalized
}

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

#[inline(always)]
pub fn height_bilinear(grid: &ChunkHeightGrid, x: f32, z: f32) -> f32 {
    let eps = 1e-6 * grid.cell;

    let min_x = grid.base_x;
    let max_x = grid.base_x + (grid.nx as f32 - 1.0) * grid.cell;
    let min_z = grid.base_z;
    let max_z = grid.base_z + (grid.nz as f32 - 1.0) * grid.cell;

    let x = x.clamp(min_x + eps, max_x - eps);
    let z = z.clamp(min_z + eps, max_z - eps);

    let fx = (x - grid.base_x) / grid.cell;
    let fz = (z - grid.base_z) / grid.cell;

    let mut ix = fx.floor() as i32;
    let mut iz = fz.floor() as i32;

    // CRITICAL: clamp cell indices, not floats
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

#[inline(always)]
fn ray_tri(ro: Vec3, rd: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let ab = b - a;
    let ac = c - a;
    let p = rd.cross(ac);
    let det = ab.dot(p);

    if det.abs() < 1e-8 {
        return None;
    }

    let inv = 1.0 / det;
    let tvec = ro - a;
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

pub fn raycast_chunk_heightgrid(
    ray: Ray,
    grid: &ChunkHeightGrid,
    t_min: f32,
    t_max: f32,
) -> Option<(f32, Vec3)> {
    let cell = grid.cell;
    let eps = 1e-6 * cell;

    // ---------- XZ slab test ----------
    let ox = ray.origin.x - grid.base_x;
    let oz = ray.origin.z - grid.base_z;

    let size_x = (grid.nx - 1) as f32 * cell;
    let size_z = (grid.nz - 1) as f32 * cell;

    let mut t0 = t_min.max(0.0);
    let mut t1 = t_max;

    for (o, d, size) in [(ox, ray.dir.x, size_x), (oz, ray.dir.z, size_z)] {
        if d.abs() < 1e-8 {
            if o < 0.0 || o > size {
                return None;
            }
        } else {
            let inv = 1.0 / d;
            let mut a = (-o) * inv;
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

    // ---------- DDA start (nudged) ----------
    let mut t = t0 + eps;
    let p = ray.origin + ray.dir * t;

    let mut ix = ((p.x - grid.base_x) / cell).floor() as i32;
    let mut iz = ((p.z - grid.base_z) / cell).floor() as i32;

    ix = ix.clamp(0, grid.nx as i32 - 2);
    iz = iz.clamp(0, grid.nz as i32 - 2);

    let step_x = ray.dir.x.signum() as i32;
    let step_z = ray.dir.z.signum() as i32;

    let next_x = grid.base_x + (ix + if step_x > 0 { 1 } else { 0 }) as f32 * cell;
    let next_z = grid.base_z + (iz + if step_z > 0 { 1 } else { 0 }) as f32 * cell;

    let mut t_max_x = if ray.dir.x.abs() < 1e-8 {
        f32::INFINITY
    } else {
        (next_x - ray.origin.x) / ray.dir.x
    };

    let mut t_max_z = if ray.dir.z.abs() < 1e-8 {
        f32::INFINITY
    } else {
        (next_z - ray.origin.z) / ray.dir.z
    };

    let t_delta_x = if ray.dir.x.abs() < 1e-8 {
        f32::INFINITY
    } else {
        cell / ray.dir.x.abs()
    };

    let t_delta_z = if ray.dir.z.abs() < 1e-8 {
        f32::INFINITY
    } else {
        cell / ray.dir.z.abs()
    };

    let mut prev_t = t;

    // ---------- DDA loop ----------
    while t <= t1 {
        if ix < 0 || iz < 0 || ix >= grid.nx as i32 - 1 || iz >= grid.nz as i32 - 1 {
            break;
        }

        let seg_end = t_max_x.min(t_max_z).min(t1);

        // exact cell intersection
        if let Some(hit_t) = ray_hit_cell(&ray, grid, ix, iz, prev_t, seg_end) {
            let p = ray.origin + ray.dir * hit_t;
            let h = height_bilinear(grid, p.x, p.z);
            return Some((hit_t, Vec3::new(p.x, h, p.z)));
        }

        prev_t = seg_end;

        // instead of the manual tie block, call:
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

#[inline(always)]
fn ray_hit_cell(
    ray: &Ray,
    grid: &ChunkHeightGrid,
    ix: i32,
    iz: i32,
    t_min: f32,
    t_max: f32,
) -> Option<f32> {
    let ix = ix as usize;
    let iz = iz as usize;

    let x0 = grid.base_x + ix as f32 * grid.cell;
    let z0 = grid.base_z + iz as f32 * grid.cell;
    let x1 = x0 + grid.cell;
    let z1 = z0 + grid.cell;

    let i = ix * grid.nz + iz;

    let h00 = grid.heights[i];
    let h10 = grid.heights[i + grid.nz];
    let h01 = grid.heights[i + 1];
    let h11 = grid.heights[i + grid.nz + 1];

    let v00 = Vec3::new(x0, h00, z0);
    let v10 = Vec3::new(x1, h10, z0);
    let v01 = Vec3::new(x0, h01, z1);
    let v11 = Vec3::new(x1, h11, z1);

    let mut best = None;

    for (a, b, c) in [(v00, v10, v01), (v01, v10, v11)] {
        if let Some(t) = ray_tri(ray.origin, ray.dir, a, b, c) {
            if t >= t_min && t <= t_max {
                best = Some(best.map_or(t, |bt: f32| bt.min(t)));
            }
        }
    }

    best
}

pub fn ray_from_mouse_pixels(
    mouse_px: glam::Vec2,
    width: f32,
    height: f32,
    view: glam::Mat4,
    proj: glam::Mat4,
) -> Ray {
    // pixel -> NDC
    let x = (mouse_px.x / width) * 2.0 - 1.0;
    let y = 1.0 - (mouse_px.y / height) * 2.0;

    let inv_view = view.inverse();
    let inv_view_proj = (proj * view).inverse();

    let p_far4 = inv_view_proj * glam::Vec4::new(x, y, 1.0, 1.0);
    let p_far = p_far4.truncate() / p_far4.w;

    let cam_pos = inv_view.transform_point3(Vec3::ZERO);
    let dir = (p_far - cam_pos).normalize();

    Ray {
        origin: cam_pos,
        dir,
    }
}

/// Advance the DDA grid traversal.
///
/// `cell_a` / `cell_b` are the two integer coordinates to advance (e.g. ix, iz or cx, cz).
/// `step_a` / `step_b` are their step directions (signum results).
/// `t` is the current t along the ray (updated).
/// `t_max_a` / `t_max_b` are the next boundary t values for the corresponding axes
/// (pass the X values first, Z second to preserve existing behavior).
/// `t_delta_a` / `t_delta_b` are the respective delta times per cell.
/// Behavior mirrors the original code: when the two t_max values are within 1e-7 they are considered tied.
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
