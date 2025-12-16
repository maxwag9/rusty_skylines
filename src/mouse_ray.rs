use crate::chunk_builder::ChunkHeightGrid;
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

    pub enabled: u32,    // offset 20
    pub _pad0: [u32; 3], // pad to 32

    pub color: [f32; 3], // offset 32
    pub _pad1: f32,      // pad to 48
}

pub fn raycast_chunk_heightgrid(
    ray: Ray,
    grid: &ChunkHeightGrid,
    t_min: f32,
    t_max: f32,
) -> Option<(f32, Vec3)> {
    #[inline]
    fn clamp_i(v: i32, lo: i32, hi: i32) -> i32 {
        v.max(lo).min(hi)
    }

    #[inline]
    fn height_bilinear(grid: &ChunkHeightGrid, x: f32, z: f32) -> f32 {
        let fx = (x - grid.base_x) / grid.cell;
        let fz = (z - grid.base_z) / grid.cell;

        let mut ix0 = fx.floor() as i32;
        let mut iz0 = fz.floor() as i32;

        ix0 = clamp_i(ix0, 0, grid.nx as i32 - 2);
        iz0 = clamp_i(iz0, 0, grid.nz as i32 - 2);

        let tx = (fx - ix0 as f32).clamp(0.0, 1.0);
        let tz = (fz - iz0 as f32).clamp(0.0, 1.0);

        let i = ix0 as usize * grid.nz + iz0 as usize;

        let h00 = grid.heights[i];
        let h10 = grid.heights[i + grid.nz];
        let h01 = grid.heights[i + 1];
        let h11 = grid.heights[i + grid.nz + 1];

        let hx0 = h00 + (h10 - h00) * tx;
        let hx1 = h01 + (h11 - h01) * tx;
        hx0 + (hx1 - hx0) * tz
    }

    #[inline]
    fn cell_hmin_hmax(grid: &ChunkHeightGrid, ix: i32, iz: i32) -> (f32, f32) {
        let ix = ix as usize;
        let iz = iz as usize;
        let i = ix * grid.nz + iz;

        let h00 = grid.heights[i];
        let h10 = grid.heights[i + grid.nz];
        let h01 = grid.heights[i + 1];
        let h11 = grid.heights[i + grid.nz + 1];

        let mut hmin = h00.min(h10).min(h01).min(h11);
        let mut hmax = h00.max(h10).max(h01).max(h11);

        let sx = (h10 - h00).abs().max((h11 - h01).abs());
        let sz = (h01 - h00).abs().max((h11 - h10).abs());
        let inflate = 0.5 * (sx + sz);

        hmin -= inflate;
        hmax += inflate;

        (hmin, hmax)
    }

    // ---------- slab in chunk-local XZ ----------
    let ox = ray.origin.x - grid.base_x;
    let oz = ray.origin.z - grid.base_z;

    let mut t0 = t_min.max(0.0);
    let mut t1 = t_max;

    let size_x = (grid.nx - 1) as f32 * grid.cell;
    let size_z = (grid.nz - 1) as f32 * grid.cell;

    // X slab
    if ray.dir.x.abs() < 1e-8 {
        if ox < 0.0 || ox > size_x {
            return None;
        }
    } else {
        let inv = 1.0 / ray.dir.x;
        let mut a = (-ox) * inv;
        let mut b = (size_x - ox) * inv;
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        t0 = t0.max(a);
        t1 = t1.min(b);
        if t0 > t1 {
            return None;
        }
    }

    // Z slab
    if ray.dir.z.abs() < 1e-8 {
        if oz < 0.0 || oz > size_z {
            return None;
        }
    } else {
        let inv = 1.0 / ray.dir.z;
        let mut a = (-oz) * inv;
        let mut b = (size_z - oz) * inv;
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        t0 = t0.max(a);
        t1 = t1.min(b);
        if t0 > t1 {
            return None;
        }
    }

    // ---------- DDA setup ----------
    let cell = grid.cell;
    let mut t = t0;

    let p0 = ray.origin + ray.dir * t;
    let mut ix = ((p0.x - grid.base_x) / cell).floor() as i32;
    let mut iz = ((p0.z - grid.base_z) / cell).floor() as i32;

    let step_x = if ray.dir.x >= 0.0 { 1 } else { -1 };
    let step_z = if ray.dir.z >= 0.0 { 1 } else { -1 };

    let next_x = if step_x > 0 {
        (ix + 1) as f32 * cell + grid.base_x
    } else {
        ix as f32 * cell + grid.base_x
    };
    let next_z = if step_z > 0 {
        (iz + 1) as f32 * cell + grid.base_z
    } else {
        iz as f32 * cell + grid.base_z
    };

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

    // ---------- helper: signed distance to surface ----------
    #[inline]
    fn signed_d(ray: &Ray, grid: &ChunkHeightGrid, t: f32) -> f32 {
        let p = ray.origin + ray.dir * t;
        let h = height_bilinear(grid, p.x, p.z);
        p.y - h
    }

    // If we start below, we can treat it as an immediate hit at t0 (optional).
    // Comment out if you only want front-face hits.
    if signed_d(&ray, grid, t0) <= 0.0 {
        let p = ray.origin + ray.dir * t0;
        let h = height_bilinear(grid, p.x, p.z);
        return Some((t0, Vec3::new(p.x, h, p.z)));
    }

    // ---------- robust per-cell segment test ----------
    const BRACKET_SAMPLES: usize = 8; // more = less flicker, small cost
    const REFINE_ITERS: usize = 16; // bisection iters

    let mut prev_t = t;

    while t <= t1 + 1e-6 {
        if ix < 0 || iz < 0 || ix as usize >= grid.nx - 1 || iz as usize >= grid.nz - 1 {
            break;
        }

        // This step's end time is the next cell boundary (or t1)
        let seg_end = t_max_x.min(t_max_z).min(t1);

        // Conservative "could hit" check using corner min/max in this cell
        let (hmin, hmax) = cell_hmin_hmax(grid, ix, iz);

        let y0 = ray.origin.y + ray.dir.y * prev_t;
        let y1 = ray.origin.y + ray.dir.y * seg_end;
        let (ymin, ymax) = if y0 < y1 { (y0, y1) } else { (y1, y0) };

        if ymax >= hmin && ymin <= hmax {
            // We might hit inside [prev_t, seg_end]. Actively bracket a sign change.
            let mut a = prev_t;
            let mut fa = signed_d(&ray, grid, a);

            // sample forward to find a point with opposite sign
            let mut b = seg_end;
            let mut fb = signed_d(&ray, grid, b);

            let mut bracket_found = (fa > 0.0 && fb < 0.0);

            if !bracket_found {
                // try a few interior samples to find a bracket
                let mut last_t = a;
                let mut last_f = fa;

                for s in 1..=BRACKET_SAMPLES {
                    let tt =
                        prev_t + (seg_end - prev_t) * (s as f32 / (BRACKET_SAMPLES as f32 + 1.0));
                    let ff = signed_d(&ray, grid, tt);

                    if last_f > 0.0 && ff < 0.0 {
                        a = last_t;
                        fa = last_f;
                        b = tt;
                        fb = ff;
                        bracket_found = true;
                        break;
                    }

                    last_t = tt;
                    last_f = ff;
                }

                // also allow the end point to complete a bracket
                if !bracket_found && last_f > 0.0 && fb < 0.0 {
                    a = last_t;
                    fa = last_f;
                    b = seg_end;
                    fb = fb;
                    bracket_found = true;
                }
            }

            if bracket_found {
                // bisection refine
                for _ in 0..REFINE_ITERS {
                    let m = 0.5 * (a + b);
                    let fm = signed_d(&ray, grid, m);
                    if fm > 0.0 {
                        a = m;
                        fa = fm;
                    } else {
                        b = m;
                        fb = fm;
                    }
                }

                let hit_t = 0.5 * (a + b);
                let hit_p = ray.origin + ray.dir * hit_t;
                let hit_h = height_bilinear(grid, hit_p.x, hit_p.z);
                return Some((hit_t, Vec3::new(hit_p.x, hit_h, hit_p.z)));
            }
        }

        // advance DDA to next cell
        prev_t = seg_end;

        if t_max_x < t_max_z {
            ix += step_x;
            t = t_max_x;
            t_max_x += t_delta_x;
        } else {
            iz += step_z;
            t = t_max_z;
            t_max_z += t_delta_z;
        }

        // keep t in sync with segment start
        if t < prev_t {
            t = prev_t;
        }
    }

    None
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

    let cam_pos = inv_view.transform_point3(glam::Vec3::ZERO);
    let dir = (p_far - cam_pos).normalize();

    Ray {
        origin: cam_pos,
        dir,
    }
}
