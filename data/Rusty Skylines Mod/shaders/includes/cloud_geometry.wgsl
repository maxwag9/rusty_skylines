
struct CloudRayInterval {
    hit: bool,
    entry_distance: f32,
    end_distance: f32,
};

fn intersect_cloud_slab(
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    slab_bottom: f32,
    slab_top: f32,
    max_distance: f32,
) -> CloudRayInterval {
    if slab_top <= slab_bottom || max_distance <= 0.0 {
        return CloudRayInterval(false, 0.0, 0.0);
    }

    let length_squared = dot(ray_direction, ray_direction);

    if length_squared <= 1.0e-12 {
        return CloudRayInterval(false, 0.0, 0.0);
    }

    let direction = normalize(ray_direction);

    if abs(direction.y) < 1.0e-5 {
        if ray_origin.y < slab_bottom || ray_origin.y > slab_top {
            return CloudRayInterval(false, 0.0, 0.0);
        }

        return CloudRayInterval(true, 0.0, max_distance);
    }

    let bottom_distance =
        (slab_bottom - ray_origin.y) / direction.y;

    let top_distance =
        (slab_top - ray_origin.y) / direction.y;

    let entry_distance =
        max(min(bottom_distance, top_distance), 0.0);

    let end_distance =
        min(max(bottom_distance, top_distance), max_distance);

    if end_distance <= entry_distance {
        return CloudRayInterval(false, 0.0, 0.0);
    }

    return CloudRayInterval(true, entry_distance, end_distance);
}

fn positive_mod_i32(value: i32, divisor: i32) -> i32 {
    let remainder = value % divisor;
    return select(remainder, remainder + divisor, remainder < 0);
}

fn stable_cloud_weather_uv(
    point_relative_to_camera_xz: vec2<f32>,
    camera_chunk: vec2<i32>,
    camera_local_xz: vec2<f32>,
    chunk_size: f32,
    period_chunks: i32,
) -> vec2<f32> {
    let safe_period_chunks = max(period_chunks, 1);

    let wrapped_chunk = vec2<f32>(
        f32(positive_mod_i32(camera_chunk.x, safe_period_chunks)),
        f32(positive_mod_i32(camera_chunk.y, safe_period_chunks))
    );

    let anchor_xz = wrapped_chunk * chunk_size + camera_local_xz;
    let period_world = chunk_size * f32(safe_period_chunks);

    return fract(
        (anchor_xz + point_relative_to_camera_xz) / period_world
    );
}
