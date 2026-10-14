use crate::helpers::positions::{WorldPos, chunk_size};
use glam::{Vec2, Vec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudRayInterval {
    pub entry_distance: f32,
    pub exit_distance: f32,
}

pub fn intersect_horizontal_cloud_slab(
    ray_origin: Vec3,
    ray_direction: Vec3,
    slab_bottom: f32,
    slab_top: f32,
    max_distance: f32,
) -> Option<CloudRayInterval> {
    if !ray_origin.is_finite()
        || !ray_direction.is_finite()
        || !slab_bottom.is_finite()
        || !slab_top.is_finite()
        || !max_distance.is_finite()
        || slab_top <= slab_bottom
        || max_distance <= 0.0
    {
        return None;
    }

    let length_squared = ray_direction.length_squared();

    if !length_squared.is_finite() || length_squared <= 1.0e-12 {
        return None;
    }

    let direction = ray_direction / length_squared.sqrt();

    if direction.y.abs() < 1.0e-5 {
        if ray_origin.y < slab_bottom || ray_origin.y > slab_top {
            return None;
        }

        return Some(CloudRayInterval {
            entry_distance: 0.0,
            exit_distance: max_distance,
        });
    }

    let bottom_distance = (slab_bottom - ray_origin.y) / direction.y;
    let top_distance = (slab_top - ray_origin.y) / direction.y;

    let entry_distance = bottom_distance.min(top_distance).max(0.0);
    let exit_distance = bottom_distance.max(top_distance).min(max_distance);

    if exit_distance <= entry_distance {
        return None;
    }

    Some(CloudRayInterval {
        entry_distance,
        exit_distance,
    })
}

pub fn stable_cloud_weather_uv(
    camera_world: WorldPos,
    point_relative_to_camera_xz: Vec2,
    period_chunks: i32,
) -> Vec2 {
    assert!(period_chunks > 0);

    let chunk_size = chunk_size() as f32;
    assert!(chunk_size > 0.0);

    let period_world = chunk_size * period_chunks as f32;

    let wrapped_chunk_x = camera_world.chunk.x.rem_euclid(period_chunks) as f32;
    let wrapped_chunk_z = camera_world.chunk.z.rem_euclid(period_chunks) as f32;

    let anchor_x = wrapped_chunk_x * chunk_size + camera_world.local.x;
    let anchor_z = wrapped_chunk_z * chunk_size + camera_world.local.z;

    let x = (anchor_x + point_relative_to_camera_xz.x).rem_euclid(period_world) / period_world;

    let z = (anchor_z + point_relative_to_camera_xz.y).rem_euclid(period_world) / period_world;

    Vec2::new(x, z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::positions::{ChunkCoord, LocalPos};

    #[test]
    fn upward_ray_intersects_cloud_layer() {
        let hit =
            intersect_horizontal_cloud_slab(Vec3::ZERO, Vec3::Y, 100.0, 200.0, 1000.0).unwrap();

        assert!((hit.entry_distance - 100.0).abs() < 1.0e-5);
        assert!((hit.exit_distance - 200.0).abs() < 1.0e-5);
    }

    #[test]
    fn downward_ray_intersects_cloud_layer() {
        let hit = intersect_horizontal_cloud_slab(
            Vec3::new(0.0, 300.0, 0.0),
            Vec3::NEG_Y,
            100.0,
            200.0,
            1000.0,
        )
        .unwrap();

        assert!((hit.entry_distance - 100.0).abs() < 1.0e-5);
        assert!((hit.exit_distance - 200.0).abs() < 1.0e-5);
    }

    #[test]
    fn ray_inside_layer_starts_at_zero() {
        let hit = intersect_horizontal_cloud_slab(
            Vec3::new(0.0, 150.0, 0.0),
            Vec3::Y,
            100.0,
            200.0,
            1000.0,
        )
        .unwrap();

        assert_eq!(hit.entry_distance, 0.0);
        assert!((hit.exit_distance - 50.0).abs() < 1.0e-5);
    }

    #[test]
    fn horizontal_ray_inside_layer_uses_distance_limit() {
        let hit = intersect_horizontal_cloud_slab(
            Vec3::new(0.0, 150.0, 0.0),
            Vec3::X,
            100.0,
            200.0,
            800.0,
        )
        .unwrap();

        assert_eq!(hit.entry_distance, 0.0);
        assert_eq!(hit.exit_distance, 800.0);
    }

    #[test]
    fn horizontal_ray_outside_layer_misses() {
        assert!(
            intersect_horizontal_cloud_slab(Vec3::ZERO, Vec3::X, 100.0, 200.0, 1000.0,).is_none()
        );
    }

    #[test]
    fn ray_interval_is_clipped_to_maximum_distance() {
        let hit =
            intersect_horizontal_cloud_slab(Vec3::ZERO, Vec3::Y, 100.0, 200.0, 150.0).unwrap();

        assert_eq!(hit.entry_distance, 100.0);
        assert_eq!(hit.exit_distance, 150.0);
    }

    #[test]
    fn ray_point_behind_origin_is_rejected() {
        assert!(
            intersect_horizontal_cloud_slab(
                Vec3::new(0.0, 300.0, 0.0),
                Vec3::Y,
                100.0,
                200.0,
                1000.0,
            )
            .is_none()
        );
    }

    #[test]
    fn weather_coordinates_remain_stable_across_chunk_boundary() {
        let cs = chunk_size() as f32;

        let camera_a = WorldPos::new(ChunkCoord::new(7, -3), LocalPos::new(cs - 0.25, 10.0, 12.0));

        let camera_b = WorldPos::new(ChunkCoord::new(8, -3), LocalPos::new(0.25, 10.0, 12.0));

        let uv_a = stable_cloud_weather_uv(camera_a, Vec2::new(3.0, 20.0), 64);

        let uv_b = stable_cloud_weather_uv(camera_b, Vec2::new(2.5, 20.0), 64);

        assert!((uv_a - uv_b).abs().max_element() < 1.0e-5);
    }

    #[test]
    fn negative_chunks_use_positive_weather_coordinates() {
        let position = WorldPos::new(ChunkCoord::new(-1, -65), LocalPos::new(20.0, 0.0, 30.0));

        let uv = stable_cloud_weather_uv(position, Vec2::ZERO, 64);

        assert!(uv.x >= 0.0 && uv.x < 1.0);
        assert!(uv.y >= 0.0 && uv.y < 1.0);
    }
}
