use crate::resources::Resources;
use crate::ui::helper::calc_move_speed;
use crate::world::camera::{ground_camera_target, resolve_pitch_by_search};
use glam::Vec3;

pub fn camera_input_system(resources: &mut Resources) {
    let world = &mut resources.world_core;
    let dt = world.time.target_sim_dt;
    if dt <= 0.0 {
        return;
    }
    let terrain_subsystem = &world.terrain_subsystem;
    let Some(camera_bundle) = world
        .world_state
        .camera_and_controller_mut(world.world_state.main_camera())
    else {
        return;
    };
    let camera = &mut camera_bundle.camera;
    let cam_ctrl = &mut camera_bundle.controller;
    let input = &mut world.input;
    let time = &world.time;
    let chunk_size = resources.settings.chunk_size;
    camera.chunk_size = chunk_size;
    let eye = camera.eye_world();
    let mut fwd3d = camera.target.delta_to(eye, chunk_size);
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
    if !resources.settings.editor_mode && !resources.settings.drive_car {
        if input.gameplay_down("Fly Camera Forward") {
            wish += forward;
        }
        if input.gameplay_down("Fly Camera Backward") {
            wish -= forward;
        }
        if input.gameplay_down("Fly Camera Left") {
            wish -= right;
        }
        if input.gameplay_down("Fly Camera Right") {
            wish += right;
        }
        if input.gameplay_down("Fly Camera Up") {
            wish += up;
        }
        if input.gameplay_down("Fly Camera Down") {
            wish -= up;
        }
    }

    let speed = calc_move_speed(input);

    let decay_rate = 3.0;
    let dist = camera.orbit_radius;
    let speed_factor = (dist / 10.0).max(0.1);

    if wish.length_squared() > 0.0 {
        wish = wish.normalize();
        let baseline = 64.0;
        let chunk_size_f = chunk_size as f32;
        let target_vel = wish * speed * speed_factor * (baseline / chunk_size_f);
        cam_ctrl.velocity = cam_ctrl.velocity.lerp(target_vel, 1.0 - (-15.0 * dt).exp());
    } else {
        let k = (1.0 - decay_rate * dt).max(0.0);
        cam_ctrl.velocity *= k;
        if cam_ctrl.velocity.length_squared() < 1e-5 {
            cam_ctrl.velocity = Vec3::ZERO;
        }
    }

    if cam_ctrl.zoom_velocity.abs() > 0.00001 {
        let r = camera.orbit_radius;

        // Target planes
        // let min_near = 0.0001;
        // let near_scale = 0.004; // 0.2% of radius
        // let far_scale = 4000.0;
        //
        // let target_near = (r * near_scale).max(min_near);
        // let target_far = (r * far_scale).max(target_near * 10.0);
        //
        // // Smooth to avoid popping
        // let smooth = 1.0 - (-dt * 12.0).exp();
        // camera.near += (target_near - camera.near) * smooth;
        // camera.far += (target_far - camera.far) * smooth;

        // Adaptive zoom step
        let base_step = 0.005; // meters, allows crawling near 1 m
        let scale_step = r * 0.40; // exponential feel at distance
        let zoom_step = (base_step + scale_step) * cam_ctrl.zoom_velocity * dt;

        camera.orbit_radius = (r + zoom_step).clamp(camera.near * 2.0, 1000.0);

        //println!("{} {}", camera.near, camera.far);
        // Radius-aware damping
        let damping = cam_ctrl.zoom_damping * (1.0 + (r / 500.0).sqrt());
        cam_ctrl.zoom_velocity *= (1.0 - damping * dt).max(0.0);
    } else {
        cam_ctrl.zoom_velocity = 0.0;
    }

    // DAMPING (exponential, smooth)
    let dv = (-cam_ctrl.orbit_damping_release * dt).exp();
    cam_ctrl.yaw_velocity *= dv;
    cam_ctrl.pitch_velocity *= dv;

    if !input.action_down("Orbit") {
        cam_ctrl.target_yaw += cam_ctrl.yaw_velocity;
        cam_ctrl.target_pitch += cam_ctrl.pitch_velocity;
    }
    camera.target = camera.target.add_vec3(cam_ctrl.velocity * dt, chunk_size);
    ground_camera_target(camera, cam_ctrl, terrain_subsystem, 0.1);
    resolve_pitch_by_search(camera, cam_ctrl, terrain_subsystem);
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

    clamp_pitch(&mut camera.pitch);
    clamp_pitch(&mut cam_ctrl.target_pitch);
}
fn clamp_pitch(p: &mut f32) {
    *p = p.clamp(-60.0f32.to_radians(), 89.0f32.to_radians());
}
