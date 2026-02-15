use crate::data::Settings;
use crate::ui::input::InputState;
use crate::world::camera::{Camera, CameraController};
use crate::world::cars::car_structs::Car;
use crate::world::cars::car_subsystem::CarSubsystem;
use crate::world::terrain::terrain_subsystem::TerrainSubsystem;
use glam::{Quat, Vec3};
use rayon::iter::ParallelIterator;

#[derive(Clone, Copy, Debug)]
pub struct GroundFit {
    pub normal: Vec3,
    pub traction: f32,
    pub engine_mult: f32,
}

#[derive(Clone, Copy, Debug)]
struct DriveTuning {
    // Smoothing (Hz-ish)
    height_conform_hz: f32,
    orient_conform_hz: f32,
    grip_hz: f32,
    drift_grip_hz: f32,

    // Steering
    turn_rate_low_speed: f32,
    turn_rate_high_speed: f32,
    stop_speed_epsilon: f32,

    // Longitudinal
    reverse_max: f32,
    idle_cruise_frac: f32,
    handbrake_decel_mul: f32,

    // Safety caps
    absolute_speed_cap: f32,

    // Ground response shaping
    min_traction: f32,
    min_engine_mult: f32,
}

const DRIVE: DriveTuning = DriveTuning {
    height_conform_hz: 22.0,
    orient_conform_hz: 16.0,
    grip_hz: 50.0,
    drift_grip_hz: 8.0,

    turn_rate_low_speed: 3.0,
    turn_rate_high_speed: 0.6,
    stop_speed_epsilon: 0.05,

    reverse_max: 6.0,
    idle_cruise_frac: 0.95,
    handbrake_decel_mul: 4.0,

    absolute_speed_cap: 200.0,

    min_traction: 0.20,
    min_engine_mult: 0.15,
};

#[inline]
fn exp_alpha(hz: f32, dt: f32) -> f32 {
    // alpha = 1 - e^(-hz*dt), stable for small dt, clamps to [0,1] for sane inputs
    if hz > 0.0 && dt > 0.0 {
        (1.0 - (-hz * dt).exp()).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

#[inline]
fn safe_unit(v: Vec3, fallback: Vec3) -> Vec3 {
    if v.is_finite() && v.length_squared() > 1e-12 {
        v.normalize()
    } else {
        fallback
    }
}

#[inline]
fn sanitize_quat(q: Quat) -> Quat {
    if q.is_finite() {
        q.normalize()
    } else {
        Quat::IDENTITY
    }
}

#[inline]
fn yaw_from_quat(q: Quat) -> f32 {
    // World yaw from the car's forward (Z) direction projected onto XZ plane.
    let fwd = q * Vec3::Z;
    fwd.x.atan2(fwd.z)
}

#[inline]
fn planar_forward_from_yaw(yaw: f32) -> Vec3 {
    Vec3::new(yaw.sin(), 0.0, yaw.cos())
}

#[inline]
fn planar_right_from_forward(planar_fwd: Vec3) -> Vec3 {
    Vec3::new(planar_fwd.z, 0.0, -planar_fwd.x)
}

/// Build an orientation that preserves *world yaw* while conforming the local-up to `ground_normal`.
fn quat_from_yaw_and_up(yaw: f32, ground_normal: Vec3) -> Quat {
    let up = safe_unit(ground_normal, Vec3::Y);

    // Start with yaw around world Y.
    let yaw_q = Quat::from_rotation_y(yaw);

    // Then tilt so that (yaw_q * Y) becomes the ground normal.
    let current_up = safe_unit(yaw_q * Vec3::Y, Vec3::Y);
    let tilt = if current_up.dot(up) < -0.9999 {
        // 180° case: choose any stable orthogonal axis
        let mut axis = current_up.cross(Vec3::X);
        if axis.length_squared() < 1e-8 {
            axis = current_up.cross(Vec3::Z);
        }
        Quat::from_axis_angle(safe_unit(axis, Vec3::X), core::f32::consts::PI)
    } else {
        Quat::from_rotation_arc(current_up, up)
    };

    sanitize_quat(tilt * yaw_q)
}

fn sample_terrain_plane(car: &Car, yaw: f32, terrain: &TerrainSubsystem) -> (f32, Vec3) {
    let chunk_size = terrain.chunk_size;

    let planar_fwd = planar_forward_from_yaw(yaw);
    let planar_right = planar_right_from_forward(planar_fwd);

    let half_len = 0.5 * car.length.max(0.001);
    let half_wid = 0.5 * car.width.max(0.001);

    // Four corners in world space (via chunk-aware add).
    let fl = car
        .pos
        .add_vec3(planar_fwd * half_len - planar_right * half_wid, chunk_size);
    let fr = car
        .pos
        .add_vec3(planar_fwd * half_len + planar_right * half_wid, chunk_size);
    let rl = car
        .pos
        .add_vec3(-planar_fwd * half_len - planar_right * half_wid, chunk_size);
    let rr = car
        .pos
        .add_vec3(-planar_fwd * half_len + planar_right * half_wid, chunk_size);

    // Heights
    let h_fl = terrain.get_height_at(fl.into());
    let h_fr = terrain.get_height_at(fr.into());
    let h_rl = terrain.get_height_at(rl.into());
    let h_rr = terrain.get_height_at(rr.into());

    let avg_y = 0.25 * (h_fl + h_fr + h_rl + h_rr);

    // Convert corner positions into car-local delta vectors, centered on avg_y.
    let d_fl = fl.delta_to(car.pos, chunk_size);
    let d_fr = fr.delta_to(car.pos, chunk_size);
    let d_rl = rl.delta_to(car.pos, chunk_size);
    let d_rr = rr.delta_to(car.pos, chunk_size);

    let p_fl = Vec3::new(d_fl.x, h_fl - avg_y, d_fl.z);
    let p_fr = Vec3::new(d_fr.x, h_fr - avg_y, d_fr.z);
    let p_rl = Vec3::new(d_rl.x, h_rl - avg_y, d_rl.z);
    let p_rr = Vec3::new(d_rr.x, h_rr - avg_y, d_rr.z);

    // Two-triangle normal estimate (averaged).
    let n1 = (p_fr - p_fl).cross(p_rl - p_fl);
    let n2 = (p_rr - p_fr).cross(p_fl - p_fr);
    let mut n = n1 + n2;

    n = safe_unit(n, Vec3::Y);
    if n.y < 0.0 {
        n = -n;
    }

    (avg_y, n)
}

pub fn conform_car_to_terrain(car: &mut Car, terrain: &TerrainSubsystem, dt: f32) -> GroundFit {
    car.quat = sanitize_quat(car.quat);

    let yaw = yaw_from_quat(car.quat);
    let (target_y, normal) = sample_terrain_plane(car, yaw, terrain);

    // Height conform (local chunk y only).
    let h_alpha = exp_alpha(DRIVE.height_conform_hz, dt);
    car.pos.local.y += (target_y - car.pos.local.y) * h_alpha;

    // Orientation conform (preserve yaw, match up to normal).
    let target_rot = quat_from_yaw_and_up(yaw, normal);
    let r_alpha = exp_alpha(DRIVE.orient_conform_hz, dt);
    car.quat = sanitize_quat(car.quat.slerp(target_rot, r_alpha));

    // Ground response scalars.
    let traction = normal.dot(Vec3::Y).clamp(DRIVE.min_traction, 1.0);

    // “Uphill” based on car forward pitch after conform.
    let uphill = (car.quat * Vec3::Z).y.max(0.0);
    let engine_mult = ((1.0 - 1.8 * uphill) * traction).clamp(DRIVE.min_engine_mult, 1.0);

    GroundFit {
        normal,
        traction,
        engine_mult,
    }
}

#[inline]
fn axis(right: bool, left: bool) -> f32 {
    (right as i8 - left as i8) as f32
}

pub fn drive_car(
    car_subsystem: &mut CarSubsystem,
    terrain: &TerrainSubsystem,
    settings: &Settings,
    input: &mut InputState,
    _cam_ctrl: &mut CameraController,
    camera: &mut Camera,
    dt: f32,
) {
    drive_ai_cars(car_subsystem, terrain, dt);
    if !settings.drive_car || dt <= 0.0 {
        return;
    }

    const G: f32 = 9.81;

    const MASS: f32 = 1620.0;
    const IZ: f32 = 2850.0;
    const WHEELBASE: f32 = 2.851;
    const CG_HEIGHT: f32 = 0.54;
    const FRONT_WEIGHT_BIAS: f32 = 0.52;
    const ENGINE_POWER: f32 = 275_000.0;
    const DRIVETRAIN_EFFICIENCY: f32 = 0.85;
    const WHEEL_POWER: f32 = ENGINE_POWER * DRIVETRAIN_EFFICIENCY;

    const BRAKE_DECEL_MAX: f32 = 10.5;
    const BRAKE_FRONT_BIAS: f32 = 0.68;

    const MU_PEAK: f32 = 1.08;
    const MU_SLIDE: f32 = 0.95;

    const C_ALPHA_F: f32 = 115_000.0;
    const C_ALPHA_R: f32 = 128_000.0;

    const SLIP_ANGLE_PEAK: f32 = 0.14;

    const FRONTAL_AREA: f32 = 2.22;
    const CD: f32 = 0.28;
    const AIR_DENSITY: f32 = 1.225;

    const AERO_DRAG_COEFF: f32 = 0.5 * AIR_DENSITY * CD * FRONTAL_AREA;

    const CL: f32 = 0.12;
    const AERO_LIFT_COEFF: f32 = 0.5 * AIR_DENSITY * CL * FRONTAL_AREA;

    const ROLL_RESISTANCE_COEFF: f32 = 0.010;

    const HANDBRAKE_REAR_GRIP_MULT: f32 = 0.30;
    const HANDBRAKE_DECEL_MAX: f32 = 5.5;

    const STEER_RATE_MAX: f32 = 6.5;
    const STEER_SPRING: f32 = 75.0;
    const STEER_DAMPING: f32 = 18.0;

    const STEER_ANGLE_PARKING: f32 = 0.60;
    const STEER_ANGLE_HIGHWAY: f32 = 0.09;
    const STEER_ANGLE_FADE_SPEED: f32 = 35.0;

    const SUBSTEP_DT: f32 = 1.0 / 120.0;
    const MIN_SPEED_FOR_SLIP: f32 = 0.4;

    let chunk_size = terrain.chunk_size;

    let throttle = if input.gameplay_down("Fly Camera Forward") {
        1.0_f32
    } else {
        0.0
    };
    let brake = if input.gameplay_down("Fly Camera Backward") {
        1.0_f32
    } else {
        0.0
    };
    let steer_left = input.gameplay_down("Fly Camera Left");
    let steer_right = input.gameplay_down("Fly Camera Right");
    let handbrake_engaged = input.gameplay_down("Handbrake");

    // Steering input: -1 = full left, +1 = full right
    let steer_input: f32 =
        (if steer_left { 1.0 } else { 0.0 }) - (if steer_right { 1.0 } else { 0.0 });

    let (car_id, previous_chunk, new_chunk) = {
        let Some(car) = car_subsystem.get_player_car() else {
            return;
        };

        let wheelbase = car.length.max(WHEELBASE);
        let a = wheelbase * (1.0 - FRONT_WEIGHT_BIAS);
        let b = wheelbase * FRONT_WEIGHT_BIAS;

        let inv_quat = car.quat.conjugate();
        let local_vel = inv_quat * car.current_velocity;

        let mut u = local_vel.z;
        let mut v = -local_vel.x;
        let mut r = car.yaw_rate;

        let forward_speed = u.max(0.0);

        let speed_ratio = (forward_speed / STEER_ANGLE_FADE_SPEED).clamp(0.0, 1.0);
        let lock_blend = 1.0 - speed_ratio.powf(1.5);
        let max_steer_angle =
            STEER_ANGLE_HIGHWAY + (STEER_ANGLE_PARKING - STEER_ANGLE_HIGHWAY) * lock_blend;

        let delta_command = -steer_input * max_steer_angle;

        let mut ax_command = 0.0;

        if throttle > 0.01 {
            let traction_limited = car.accel;
            let power_limited = if forward_speed > 2.0 {
                WHEEL_POWER / (MASS * forward_speed)
            } else {
                traction_limited
            };

            ax_command = throttle * traction_limited.min(power_limited);
        }

        if brake > 0.01 {
            ax_command -= brake * BRAKE_DECEL_MAX;
        }

        if handbrake_engaged && u.abs() > 0.5 {
            ax_command -= HANDBRAKE_DECEL_MAX * u.signum();
        }

        let fz_static_f = MASS * G * (b / wheelbase);
        let fz_static_r = MASS * G * (a / wheelbase);
        let previous_chunk = car.pos.chunk;

        let mut time_remaining = dt;
        while time_remaining > 0.0 {
            let h = time_remaining.min(SUBSTEP_DT);
            time_remaining -= h;

            let delta = car.steering_angle;
            let delta_vel = car.steering_vel;

            let delta_accel = STEER_SPRING * (delta_command - delta) - STEER_DAMPING * delta_vel;

            let mut new_delta_vel = delta_vel + delta_accel * h;
            new_delta_vel = new_delta_vel.clamp(-STEER_RATE_MAX, STEER_RATE_MAX);

            let mut new_delta = delta + new_delta_vel * h;
            new_delta = new_delta.clamp(-max_steer_angle, max_steer_angle);

            car.steering_angle = new_delta;
            car.steering_vel = new_delta_vel;

            let weight_transfer_x = (CG_HEIGHT / wheelbase) * MASS * ax_command;

            let aero_lift = AERO_LIFT_COEFF * forward_speed * forward_speed;

            let fz_front = (fz_static_f - weight_transfer_x - aero_lift * 0.5).max(500.0);
            let fz_rear = (fz_static_r + weight_transfer_x - aero_lift * 0.5).max(500.0);

            let u_safe = if u.abs() < MIN_SPEED_FOR_SLIP {
                MIN_SPEED_FOR_SLIP * u.signum().max(1.0)
            } else {
                u
            };

            let alpha_front = (v + a * r).atan2(u_safe) - car.steering_angle;
            let alpha_rear = (v - b * r).atan2(u_safe);

            fn tire_lateral_force(alpha: f32, c_alpha: f32, fz: f32, mu: f32) -> f32 {
                let f_max = mu * fz;
                let f_linear = -c_alpha * alpha;

                let slip_ratio = f_linear.abs() / f_max;

                let saturation = if slip_ratio > 0.01 {
                    slip_ratio.tanh() / slip_ratio
                } else {
                    1.0
                };

                let peak_mult = 1.02;

                f_linear * saturation * peak_mult
            }

            let mu_front = MU_PEAK;
            let mu_rear = if handbrake_engaged {
                MU_SLIDE * HANDBRAKE_REAR_GRIP_MULT
            } else {
                MU_PEAK
            };

            let fy_front = tire_lateral_force(alpha_front, C_ALPHA_F, fz_front, mu_front);
            let fy_rear = tire_lateral_force(alpha_rear, C_ALPHA_R, fz_rear, mu_rear);

            let f_rolling = -ROLL_RESISTANCE_COEFF * (fz_front + fz_rear) * u.signum();

            let f_aero = -AERO_DRAG_COEFF * u * u.abs();

            let ax_resist = (f_rolling + f_aero) / MASS;

            let cos_delta = car.steering_angle.cos();
            let sin_delta = car.steering_angle.sin();

            let u_dot = ax_command + ax_resist - (fy_front * sin_delta) / MASS + r * v;

            let v_dot = (fy_front * cos_delta + fy_rear) / MASS - r * u;

            let r_dot = (a * fy_front * cos_delta - b * fy_rear) / IZ;

            u += u_dot * h;
            v += v_dot * h;
            r += r_dot * h;

            if u.abs() < 0.8 && throttle < 0.01 && brake < 0.01 {
                let decay = 0.92_f32.powf(h * 60.0);
                u *= decay;
                v *= decay;
                r *= decay;

                if u.abs() < 0.05 {
                    u = 0.0;
                }
                if v.abs() < 0.02 {
                    v = 0.0;
                }
                if r.abs() < 0.01 {
                    r = 0.0;
                }
            }

            const MAX_SPEED: f32 = 90.0;
            const MAX_YAW_RATE: f32 = 4.0;

            u = u.clamp(-MAX_SPEED, MAX_SPEED);
            v = v.clamp(-MAX_SPEED * 0.5, MAX_SPEED * 0.5);
            r = r.clamp(-MAX_YAW_RATE, MAX_YAW_RATE);

            // Integrate yaw per substep
            let yaw_delta_world = -r * h;
            if yaw_delta_world.abs() > 0.0 {
                let rotation = Quat::from_axis_angle(Vec3::Y, yaw_delta_world);
                car.quat = (rotation * car.quat).normalize();
            }

            // Reconstruct world velocity and integrate position per substep
            let local_velocity = Vec3::new(-v, 0.0, u);
            car.current_velocity = car.quat * local_velocity;
            car.pos = car.pos.add_vec3(car.current_velocity * h, chunk_size);
        }

        car.yaw_rate = r;

        car.pos.local.y = terrain.get_height_at(car.pos);

        camera.target = car.pos;

        let speed_kmh = car.current_velocity.length() * 3.6;
        let steer_deg = car.steering_angle.to_degrees();
        let body_slip = if u.abs() > 1.0 {
            (v / u).atan().to_degrees()
        } else {
            0.0
        };
        let lat_g = (v * r) / G;

        println!(
            "{:6.1} km/h │ δ={:+5.1}° │ β={:+5.1}° │ ṙ={:+5.2} │ Lat≈{:+4.2}g",
            speed_kmh, steer_deg, body_slip, r, lat_g
        );

        (car.id, previous_chunk, car.pos.chunk)
    };

    if previous_chunk != new_chunk {
        car_subsystem
            .car_storage_mut()
            .move_car_between_chunks(previous_chunk, new_chunk, car_id);
    }
}

pub fn drive_ai_cars(car_subsystem: &mut CarSubsystem, terrain: &TerrainSubsystem, dt: f32) {
    if dt <= 0.0 {
        return;
    }

    const G: f32 = 9.81;
    const MASS: f32 = 1500.0;
    const IZ: f32 = 2500.0;
    const MU: f32 = 1.05;
    const C_AF: f32 = 90000.0;
    const C_AR: f32 = 100000.0;
    const ROLL_DRAG: f32 = 0.02;
    const AERO_DRAG: f32 = 0.0007;
    const STEER_RATE_MAX: f32 = 6.0;
    const STEER_KP: f32 = 70.0;
    const STEER_KD: f32 = 14.0;
    const STEER_LOCK_LOW: f32 = 0.62;
    const STEER_LOCK_HIGH: f32 = 0.14;
    const STEER_LOCK_FADE_MPS: f32 = 28.0;
    const MAX_STEP: f32 = 1.0 / 120.0;
    const MIN_U: f32 = 0.5;

    let chunk_size = terrain.chunk_size;
    let player_car_id = car_subsystem.player_car_id();
    let cs = chunk_size as f64;

    let car_storage = car_subsystem.car_storage_mut();

    // Parallel physics computation over all cars, collecting chunk moves
    let moves: Vec<_> = car_storage
        .par_iter_mut_cars()
        .filter_map(|car| {
            let car = car.as_mut()?;
            if car.id == player_car_id {
                return None;
            }

            let previous_chunk = car.pos.chunk;

            // === AI Inputs: smooth procedural wandering ===
            let global_x = car.pos.chunk.x as f64 * cs + car.pos.local.x as f64;
            let global_z = car.pos.chunk.z as f64 * cs + car.pos.local.z as f64;

            let wander_phase = global_x * 0.012 + global_z * 0.008 + (car.id as f64 * 7.3);
            let wander = wander_phase.sin() as f32;

            let steer_input = wander * 0.55;
            let throttle_input = 0.85 + wander * 0.15;
            let brake_input = if wander.abs() > 0.8 { 0.3 } else { 0.0 };

            // === Convert world velocity to LOCAL space via quaternion ===
            let inv_q = car.quat.conjugate();
            let local_v = inv_q * car.current_velocity;

            let mut u = local_v.z;
            let mut v = -local_v.x;
            let mut r = car.yaw_rate;

            let wheelbase = car.length.max(0.1);
            let a = 0.5 * wheelbase;
            let b = wheelbase - a;

            let fz_f = MASS * G * (b / wheelbase);
            let fz_r = MASS * G * (a / wheelbase);

            // === Speed-sensitive steering lock ===
            let speed = u.abs();
            let t = (speed / STEER_LOCK_FADE_MPS).clamp(0.0, 1.0);
            let steer_lock =
                STEER_LOCK_HIGH + (STEER_LOCK_LOW - STEER_LOCK_HIGH) * (1.0 - t).powf(1.7);

            let delta_cmd = (steer_input * steer_lock).clamp(-steer_lock, steer_lock);

            // === Longitudinal accel command ===
            let ax_cmd = throttle_input * car.accel - brake_input * car.accel * 3.0;

            // === Substep integration ===
            let mut time_left = dt;
            while time_left > 0.0 {
                let h = time_left.min(MAX_STEP);
                time_left -= h;

                let delta = car.steering_angle;
                let mut delta_dot = car.steering_vel;

                delta_dot += (STEER_KP * (delta_cmd - delta) - STEER_KD * delta_dot) * h;
                delta_dot = delta_dot.clamp(-STEER_RATE_MAX, STEER_RATE_MAX);

                let mut delta_new = delta + delta_dot * h;
                delta_new = delta_new.clamp(-steer_lock, steer_lock);

                car.steering_angle = delta_new;
                car.steering_vel = delta_dot;

                let u_safe = if u.abs() < MIN_U {
                    MIN_U * u.signum().max(1.0)
                } else {
                    u
                };

                let alpha_f = (v + a * r).atan2(u_safe) - car.steering_angle;
                let alpha_r = (v - b * r).atan2(u_safe);

                let fy_f_lin = -C_AF * alpha_f;
                let fy_r_lin = -C_AR * alpha_r;

                let fy_f_max = MU * fz_f;
                let fy_r_max = MU * fz_r;

                let fy_f = fy_f_lin.clamp(-fy_f_max, fy_f_max);
                let fy_r = fy_r_lin.clamp(-fy_r_max, fy_r_max);

                let ax_drag = -ROLL_DRAG * u - AERO_DRAG * u * u.abs();

                let u_dot = (ax_cmd + ax_drag) + r * v;
                let v_dot = (fy_f + fy_r) / MASS - r * u;
                let r_dot = (a * fy_f - b * fy_r) / IZ;

                u += u_dot * h;
                v += v_dot * h;
                r += r_dot * h;
            }

            car.yaw_rate = r;

            let yaw_delta_world = -r * dt;
            if yaw_delta_world.abs() > 0.0 {
                let yaw_q = Quat::from_axis_angle(Vec3::Y, yaw_delta_world);
                car.quat = (yaw_q * car.quat).normalize();
            }

            let new_local_v = Vec3::new(-v, 0.0, u);
            car.current_velocity = car.quat * new_local_v;

            car.pos = car.pos.add_vec3(car.current_velocity * dt, chunk_size);
            car.pos.local.y = terrain.get_height_at(car.pos);

            if previous_chunk != car.pos.chunk {
                Some((previous_chunk, car.pos.chunk, car.id))
            } else {
                None
            }
        })
        .collect();

    // Apply chunk moves sequentially (requires &mut access to spatial index)
    for (prev, new, id) in moves {
        car_storage.move_car_between_chunks(prev, new, id);
    }
}
