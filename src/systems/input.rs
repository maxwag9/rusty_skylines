use crate::helpers::positions::chunk_size;
use crate::resources::Resources;
use crate::ui::helper::calc_move_speed;
use crate::world::camera::{CameraMode, ground_camera_target, resolve_pitch_by_search};
use glam::Vec3;

pub fn run_inputs(resources: &mut Resources) {
    let world = &mut resources.world;

    let dt = world.time.target_sim_dt;
    if dt <= 0.0 {
        return;
    }
    let terrain_subsystem = &world.terrain;

    let camera = &mut world.world_state.camera;
    let cam_ctrl = &mut world.world_state.cam_controller;
    let input = &mut world.input;

    // Mode toggle — edge-triggered so holding the key doesn't retrigger.
    if input.action_repeat("Toggle Camera Mode") {
        let next = match camera.mode {
            CameraMode::Orbit => CameraMode::FirstPerson,
            CameraMode::FirstPerson => CameraMode::Orbit,
        };
        let duration = resources
            .ui
            .variables
            .get_f64("camera_mode_transition_time")
            .unwrap_or(0.35) as f32;
        cam_ctrl.switch_mode(camera, next, duration);
    }
    cam_ctrl.update_mode_transition(camera);

    let chunk_size = chunk_size();
    let noclip = resources.settings.noclip;

    let fwd3d = camera.forward();

    let mut forward = Vec3::new(fwd3d.x, 0.0, fwd3d.z);
    if forward.length_squared() > 0.0 {
        forward = forward.normalize();
    }

    let right = forward.cross(Vec3::Y).normalize();
    let up = Vec3::Y;

    let zooming = input.action_down("Zoom");

    if zooming {
        cam_ctrl.zoom(
            camera,
            resources.ui.variables.get_f64("zoom_speed").unwrap_or(1.0) as f32,
        );
        resources.ui.variables.set_bool("zoomed", true);
    } else {
        if cam_ctrl.zoom_time_start.is_some() {
            cam_ctrl.zoom_deactivate(camera);
        }
        cam_ctrl.zoom_end(
            camera,
            resources
                .ui
                .variables
                .get_f64("unzoom_speed")
                .unwrap_or(1.0) as f32,
        );
        resources.ui.variables.set_bool("zoomed", false);
    }

    let mut wish = Vec3::ZERO;
    if !resources.settings.editor_mode
        && !resources.settings.drive_car
        && !resources.ui.touch_manager.hovered().is_some()
    {
        if input.action_down("Fly Camera Forward") {
            wish += forward;
        }
        if input.action_down("Fly Camera Backward") {
            wish -= forward;
        }
        if input.action_down("Fly Camera Left") {
            wish -= right;
        }
        if input.action_down("Fly Camera Right") {
            wish += right;
        }

        // if camera.mode == CameraMode::Orbit {
        if input.action_down("Fly Camera Up") {
            wish += up;
        }
        if input.action_down("Fly Camera Down") {
            wish -= up;
        }
        //}

        let orbit_speed = 1.5 * world.time.render_dt;
        if cam_ctrl.target_yaw.is_nan() {
            camera.yaw = 0.0;
            cam_ctrl.target_yaw = 0.0;
            eprintln!("Yaw was NaN!");
        }
        if cam_ctrl.target_pitch.is_nan() {
            camera.pitch = 45.0;
            cam_ctrl.target_pitch = 45.0;
            eprintln!("Pitch was NaN!");
        }

        if input.action_down("Orbit Left") {
            cam_ctrl.target_yaw += orbit_speed;
        }
        if input.action_down("Orbit Right") {
            cam_ctrl.target_yaw -= orbit_speed;
        }
        if input.action_down("Orbit Up") {
            cam_ctrl.target_pitch += orbit_speed;
        }
        if input.action_down("Orbit Down") {
            cam_ctrl.target_pitch -= orbit_speed;
        }
    }

    let speed = calc_move_speed(input);
    let decay_rate = 3.0;

    // orbit_radius only means something while orbiting; in first person it's
    // a stale leftover value, so movement speed there isn't scaled by it.
    let speed_factor = match camera.mode {
        CameraMode::Orbit => (camera.orbit_radius / 10.0).max(0.1),
        CameraMode::FirstPerson => (camera.orbit_radius / 10.0).max(0.1),
    };

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

    // Dolly zoom (scroll wheel pushing orbit_radius) only makes sense while
    // orbiting; there's no pivot distance to change in first person.
    if cam_ctrl.zoom_velocity.abs() > 0.00001 {
        let r = camera.orbit_radius;
        let base_step = 0.005;
        let scale_step = r * 0.40;
        let zoom_step = (base_step + scale_step) * cam_ctrl.zoom_velocity * dt;

        camera.orbit_radius = (r + zoom_step).clamp(camera.near * 2.0, 1000.0);

        let damping = cam_ctrl.zoom_damping * (1.0 + (r / 500.0).sqrt());
        cam_ctrl.zoom_velocity *= (1.0 - damping * dt).max(0.0);
    } else {
        cam_ctrl.zoom_velocity = 0.0;
    }

    let dv = (-cam_ctrl.orbit_damping_release * dt).exp();
    cam_ctrl.yaw_velocity *= dv;
    cam_ctrl.pitch_velocity *= dv;

    if !input.action_down("Orbit") {
        cam_ctrl.target_yaw += cam_ctrl.yaw_velocity;
        cam_ctrl.target_pitch += cam_ctrl.pitch_velocity;
    }
    if !resources.settings.drive_car {
        camera.target = camera.target.add_vec3(cam_ctrl.velocity * dt);
        if !noclip {
            // First person needs eye-height clearance above ground, not the
            // small pivot clearance used while orbiting.
            let clearance = if camera.mode == CameraMode::FirstPerson {
                1.7
            } else {
                0.1
            };
            ground_camera_target(camera, cam_ctrl, terrain_subsystem, clearance);
        }
    }
    // Pitch-by-terrain-search sweeps along the orbit arm to keep the pivot
    // camera from clipping through terrain; there's no arm to sweep in first
    // person (eye == target), so it's Orbit-only.
    if !noclip && camera.mode == CameraMode::Orbit {
        resolve_pitch_by_search(camera, cam_ctrl, terrain_subsystem);
    }

    let t = 1.0 - (-cam_ctrl.orbit_smoothness * 60.0 * dt).exp();

    camera.yaw += (cam_ctrl.target_yaw - camera.yaw) * t;
    camera.pitch += (cam_ctrl.target_pitch - camera.pitch) * t;

    if (cam_ctrl.target_yaw - camera.yaw).abs() < 0.0001 {
        camera.yaw = cam_ctrl.target_yaw;
    }
    if (cam_ctrl.target_pitch - camera.pitch).abs() < 0.0001 {
        camera.pitch = cam_ctrl.target_pitch;
    }

    // Near-plane sizing by orbit distance only applies while orbiting; a
    // mode-switch transition drives `camera.near` itself while it's active.
    if camera.mode == CameraMode::Orbit && !cam_ctrl.is_transitioning() {
        if camera.orbit_radius < 17.0 {
            camera.near = 0.1;
        } else {
            camera.near = 8.0;
        }
    }

    clamp_pitch(&mut camera.pitch);
    clamp_pitch(&mut cam_ctrl.target_pitch);
}

fn clamp_pitch(p: &mut f32) {
    *p = p.clamp(-60.0f32.to_radians(), 89.99f32.to_radians());
}
