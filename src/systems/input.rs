use crate::components::camera::{ground_camera_target, resolve_pitch_by_search};
use crate::resources::Resources;
use crate::world::World;
use glam::Vec3;

pub fn camera_input_system(world: &mut World, resources: &mut Resources) {
    let dt = resources.time.sim_dt;
    if dt <= 0.0 {
        return;
    }

    let entity = world.main_camera();
    let Some(camera_bundle) = world.camera_and_controller_mut(entity) else {
        return;
    };
    let camera = &mut camera_bundle.camera;
    let cam_ctrl = &mut camera_bundle.controller;
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
    if !resources.settings.editor_mode {
        if resources.input.action_down("Fly Camera Forward") {
            wish += forward;
        }
        if resources.input.action_down("Fly Camera Backward") {
            wish -= forward;
        }
        if resources.input.action_down("Fly Camera Left") {
            wish -= right;
        }
        if resources.input.action_down("Fly Camera Right") {
            wish += right;
        }
        if resources.input.action_down("Fly Camera Up") {
            wish += up;
        }
        if resources.input.action_down("Fly Camera Down") {
            wish -= up;
        }
    }

    let base_speed = 8.0;
    let mut speed = base_speed;

    match (resources.input.shift, resources.input.ctrl) {
        (true, false) => speed *= 8.0,
        (false, true) => speed *= 0.4,
        (true, true) => speed *= 0.1,
        _ => {}
    }

    let decay_rate = 6.0;
    let dist = camera.orbit_radius;
    let speed_factor = (dist / 10.0).clamp(50.0, 100.0);

    if wish.length_squared() > 0.0 {
        wish = wish.normalize();
        let target_vel = wish * speed * speed_factor;
        cam_ctrl.velocity = cam_ctrl.velocity.lerp(target_vel, 1.0 - (-10.0 * dt).exp());
    } else {
        let k = (1.0 - decay_rate * dt).max(0.0);
        cam_ctrl.velocity *= k;
        if cam_ctrl.velocity.length_squared() < 1e-5 {
            cam_ctrl.velocity = Vec3::ZERO;
        }
    }

    if cam_ctrl.zoom_velocity.abs() > 0.0001 {
        camera.orbit_radius += cam_ctrl.zoom_velocity * dt * 1.5;
        cam_ctrl.zoom_velocity *= (1.0 - cam_ctrl.zoom_damping * dt).max(0.0);
        camera.orbit_radius = camera.orbit_radius.clamp(15.0, 10_000.0);
    } else {
        cam_ctrl.zoom_velocity = 0.0;
    }

    // DAMPING (exponential, smooth)
    let dv = (-cam_ctrl.orbit_damping_release * dt).exp();
    cam_ctrl.yaw_velocity *= dv;
    cam_ctrl.pitch_velocity *= dv;

    if !resources.input.action_down("Orbit") {
        cam_ctrl.target_yaw += cam_ctrl.yaw_velocity;
        cam_ctrl.target_pitch += cam_ctrl.pitch_velocity;
    }
    camera.target += cam_ctrl.velocity * dt;
    ground_camera_target(
        camera,
        cam_ctrl,
        &resources.renderer.core.world.terrain_gen,
        1.0,
    );
    resolve_pitch_by_search(camera, cam_ctrl, &resources.renderer.core.world);
    // SMOOTH target → camera
    let t = 1.0 - (-cam_ctrl.orbit_smoothness * 60.0 * dt).exp();

    camera.yaw += (cam_ctrl.target_yaw - camera.yaw) * t;
    camera.pitch += (cam_ctrl.target_pitch - camera.pitch) * t;

    // Micro-deadzone to avoid infinite interpolation
    if (cam_ctrl.target_yaw - camera.yaw).abs() < 0.0001 {
        camera.yaw = cam_ctrl.target_yaw;
    }
    if (cam_ctrl.target_pitch - camera.pitch).abs() < 0.0001 {
        camera.pitch = cam_ctrl.target_pitch;
    }

    camera.pitch = camera
        .pitch
        .clamp(-50.0f32.to_radians(), 89.0f32.to_radians());
}
