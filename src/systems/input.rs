use crate::resources::Resources;
use crate::world::World;
use glam::Vec3;

pub fn camera_input_system(world: &mut World, resources: &mut Resources) {
    let dt = resources.time.delta;
    if dt <= 0.0 {
        return;
    }

    let entity = world.main_camera();
    let Some((camera, controller)) = world.camera_and_controller_mut(entity) else {
        return;
    };

    let eye = camera.position();
    let mut fwd3d = camera.target - eye;
    if fwd3d.length_squared() > 0.0 {
        fwd3d = fwd3d.normalize();
    }

    let mut forward = Vec3::new(fwd3d.x, 0.0, fwd3d.z);
    if forward.length_squared() > 0.0 {
        forward = forward.normalize();
    }

    let right = forward.cross(Vec3::Y).normalize();
    let up = Vec3::Y;

    let mut wish = Vec3::ZERO;
    if resources.input.pressed("w") {
        wish += forward;
    }
    if resources.input.pressed("s") {
        wish -= forward;
    }
    if resources.input.pressed("a") {
        wish -= right;
    }
    if resources.input.pressed("d") {
        wish += right;
    }
    if resources.input.pressed("q") {
        wish += up;
    }
    if resources.input.pressed("e") {
        wish -= up;
    }

    let base_speed = 8.0;
    let mut speed = base_speed;

    match (resources.input.shift_pressed, resources.input.ctrl_pressed) {
        (true, false) => speed *= 3.0,
        (false, true) => speed *= 0.4,
        (true, true) => speed *= 0.1,
        _ => {}
    }

    let decay_rate = 6.0;
    let dist = camera.radius;
    let speed_factor = (dist / 10.0).clamp(0.1, 10.0);

    if wish.length_squared() > 0.0 {
        wish = wish.normalize();
        controller.velocity = wish * speed * speed_factor;
    } else {
        let k = (1.0 - decay_rate * dt).max(0.0);
        controller.velocity *= k;
        if controller.velocity.length_squared() < 1e-5 {
            controller.velocity = Vec3::ZERO;
        }
    }

    if controller.zoom_velocity.abs() > 0.0001 {
        camera.radius += controller.zoom_velocity * dt;
        controller.zoom_velocity *= (1.0 - controller.zoom_damping * dt).max(0.0);
        camera.radius = camera.radius.clamp(1.0, 10_000.0);
    } else {
        controller.zoom_velocity = 0.0;
    }

    let t = 1.0 - (-controller.orbit_smoothness * 60.0 * dt).exp();
    camera.yaw += (controller.target_yaw - camera.yaw) * t;
    camera.pitch += (controller.target_pitch - camera.pitch) * t;

    if !resources.mouse.middle_pressed {
        controller.target_yaw += controller.yaw_velocity;
        controller.target_pitch += controller.pitch_velocity;
        controller.yaw_velocity *= (1.0 - controller.orbit_damping_release * dt).max(0.0);
        controller.pitch_velocity *= (1.0 - controller.orbit_damping_release * dt).max(0.0);
    }

    camera.pitch = camera
        .pitch
        .clamp(10.0f32.to_radians(), 89.0f32.to_radians());
    camera.target += controller.velocity * dt;
}
