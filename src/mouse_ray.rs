#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: glam::Vec3,
    pub dir: glam::Vec3, // normalized
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

    // unproject far point
    let p_far4 = inv_view_proj * glam::Vec4::new(x, y, 1.0, 1.0);
    let p_far = p_far4.truncate() / p_far4.w;

    // camera position
    let cam_pos = inv_view.transform_point3(glam::Vec3::ZERO);

    let dir = (p_far - cam_pos).normalize();

    Ray {
        origin: cam_pos,
        dir,
    }
}

pub fn raycast_heightfield(
    ray: Ray,
    height_at: impl Fn(f32, f32) -> f32,
    cell_size: u32,
    t_min: f32,
    t_max: f32,
) -> Option<(f32, glam::Vec3)> {
    if ray.dir.y.abs() < 1e-5 {
        return None;
    }

    let dir_xz = glam::Vec2::new(ray.dir.x, ray.dir.z);
    let len_xz = dir_xz.length();
    if len_xz < 1e-5 {
        return None;
    }

    let step_t = cell_size as f32 / len_xz;

    let mut t = t_min.max(0.0);
    let mut p = ray.origin + ray.dir * t;
    let mut f_prev = p.y - height_at(p.x, p.z);

    while t <= t_max {
        t += step_t;
        p = ray.origin + ray.dir * t;
        let f = p.y - height_at(p.x, p.z);

        if f <= 0.0 && f_prev > 0.0 {
            // binary refinement
            let mut a = t - step_t;
            let mut b = t;

            for _ in 0..20 {
                let m = 0.5 * (a + b);
                let pm = ray.origin + ray.dir * m;
                let fm = pm.y - height_at(pm.x, pm.z);
                if fm > 0.0 {
                    a = m;
                } else {
                    b = m;
                }
            }

            let t_hit = 0.5 * (a + b);
            let ph = ray.origin + ray.dir * t_hit;
            let y = height_at(ph.x, ph.z);

            return Some((t_hit, glam::Vec3::new(ph.x, y, ph.z)));
        }

        f_prev = f;
    }

    None
}

#[inline]
pub(crate) fn distance2_point_to_ray(p: glam::Vec3, ray: Ray) -> f32 {
    let v = p - ray.origin;
    let t = v.dot(ray.dir).max(0.0);
    let proj = ray.origin + ray.dir * t;
    p.distance_squared(proj)
}
