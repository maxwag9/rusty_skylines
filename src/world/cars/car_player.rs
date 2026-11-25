use crate::data::Settings;
use rusty_skylines_utils::positions::{chunk_size, WorldPos};
use crate::ui::input::Input;
use crate::world::camera::camera::{Camera, CameraController};
use crate::world::cars::car_structs::{Car, CarBodyType, CarMode, Powertrain, SimTime};
use crate::world::cars::car_subsystem::Cars;
use crate::world::terrain::terrain_subsystem::Terrain;
use glam::{Quat, Vec3};
use rayon::iter::ParallelIterator;
use crate::world::cars::car_simulation::{CarTrajectory, CarTrajectoryPoint};

#[derive(Clone, Copy, Debug)]
pub struct GroundFit {
    pub normal: Vec3,
    pub traction: f32,
    pub engine_mult: f32,
}

#[derive(Clone, Copy, Debug)]
struct TerrainConformTuning {
    height_conform_hz: f32,
    orient_conform_hz: f32,
    min_traction: f32,
    min_engine_mult: f32,
}

const TERRAIN: TerrainConformTuning = TerrainConformTuning {
    height_conform_hz: 220.0,
    orient_conform_hz: 16.0,
    min_traction: 0.20,
    min_engine_mult: 0.15,
};

#[inline]
fn exp_alpha(hz: f32, dt: f32) -> f32 {
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
pub fn sanitize_quat(q: Quat) -> Quat {
    if q.is_finite() {
        q.normalize()
    } else {
        Quat::IDENTITY
    }
}

#[inline]
fn yaw_from_quat(q: Quat) -> f32 {
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

fn quat_from_yaw_and_up(yaw: f32, ground_normal: Vec3) -> Quat {
    let up = safe_unit(ground_normal, Vec3::Y);
    let yaw_q = Quat::from_rotation_y(yaw);
    let current_up = safe_unit(yaw_q * Vec3::Y, Vec3::Y);

    let tilt = if current_up.dot(up) < -0.9999 {
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

fn sample_terrain_plane(car: &Car, yaw: f32, terrain: &Terrain) -> (f32, Vec3) {
    let planar_fwd = planar_forward_from_yaw(yaw);
    let planar_right = planar_right_from_forward(planar_fwd);

    let half_len = 0.5 * car.length.max(0.01);
    let half_wid = 0.5 * car.width.max(0.01);

    let fl = car
        .pos
        .add_vec3(planar_fwd * half_len - planar_right * half_wid);
    let fr = car
        .pos
        .add_vec3(planar_fwd * half_len + planar_right * half_wid);
    let rl = car
        .pos
        .add_vec3(-planar_fwd * half_len - planar_right * half_wid);
    let rr = car
        .pos
        .add_vec3(-planar_fwd * half_len + planar_right * half_wid);

    let h_fl = terrain.get_height_at(fl.into(), true);
    let h_fr = terrain.get_height_at(fr.into(), true);
    let h_rl = terrain.get_height_at(rl.into(), true);
    let h_rr = terrain.get_height_at(rr.into(), true);

    let avg_y = 0.25 * (h_fl + h_fr + h_rl + h_rr);

    let d_fl = car.pos.direction_to(fl);
    let d_fr = car.pos.direction_to(fr);
    let d_rl = car.pos.direction_to(rl);
    let d_rr = car.pos.direction_to(rr);

    let p_fl = Vec3::new(d_fl.x, h_fl - avg_y, d_fl.z);
    let p_fr = Vec3::new(d_fr.x, h_fr - avg_y, d_fr.z);
    let p_rl = Vec3::new(d_rl.x, h_rl - avg_y, d_rl.z);
    let p_rr = Vec3::new(d_rr.x, h_rr - avg_y, d_rr.z);

    let n1 = (p_fr - p_fl).cross(p_rl - p_fl);
    let n2 = (p_rr - p_fr).cross(p_rl - p_rr);
    let mut n = n1 + n2;

    n = safe_unit(n, Vec3::Y);
    if n.y < 0.0 {
        n = -n;
    }

    (avg_y, n)
}

pub fn conform_car_to_terrain(car: &mut Car, terrain: &Terrain, dt: f32) -> GroundFit {
    car.quat = sanitize_quat(car.quat);

    let yaw = yaw_from_quat(car.quat);
    let (target_y, normal) = sample_terrain_plane(car, yaw, terrain);
    let diff = target_y - car.pos.local.y;

    //let h_alpha = exp_alpha(TERRAIN.height_conform_hz, dt);
    car.pos.local.y += diff;

    let target_rot = quat_from_yaw_and_up(yaw, normal);
    let r_alpha = exp_alpha(TERRAIN.orient_conform_hz, dt);
    car.quat = sanitize_quat(car.quat.slerp(target_rot, r_alpha));

    let traction = normal.dot(Vec3::Y).clamp(TERRAIN.min_traction, 1.0);
    let uphill = (car.quat * Vec3::Z).y.max(0.0);
    let engine_mult = ((1.0 - 1.8 * uphill) * traction).clamp(TERRAIN.min_engine_mult, 1.0);

    GroundFit {
        normal,
        traction,
        engine_mult,
    }
}

pub fn drive_car(
    cars: &mut Cars,
    terrain: &Terrain,
    settings: &Settings,
    input: &mut Input,
    _cam_ctrl: &mut CameraController,
    camera: &mut Camera,
    dt: f32,
) {
    // drive_ai_cars(car_subsystem, terrain, dt);
    // if !settings.drive_car || dt <= 0.0 {
    //     return;
    // }

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
    const MU_PEAK: f32 = 1.08;
    const MU_SLIDE: f32 = 0.95;
    const C_ALPHA_F: f32 = 115_000.0;
    const C_ALPHA_R: f32 = 128_000.0;
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
    const MAX_SPEED: f32 = 90.0;
    const MAX_YAW_RATE: f32 = 4.0;

    let throttle = if input.action_down("Fly Camera Forward") {
        1.0_f32
    } else {
        0.0
    };
    let brake = if input.action_down("Fly Camera Backward") {
        1.0_f32
    } else {
        0.0
    };
    let steer_left = input.action_down("Fly Camera Left");
    let steer_right = input.action_down("Fly Camera Right");
    let handbrake_engaged = input.action_down("Handbrake");
    let steer_input: f32 =
        (if steer_left { 1.0 } else { 0.0 }) - (if steer_right { 1.0 } else { 0.0 });

    let (car_id, previous_chunk, new_chunk) = {
        let Some(car) = cars.get_player_car() else {
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
                f_linear * saturation * 1.02
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

            u = u.clamp(-MAX_SPEED, MAX_SPEED);
            v = v.clamp(-MAX_SPEED * 0.5, MAX_SPEED * 0.5);
            r = r.clamp(-MAX_YAW_RATE, MAX_YAW_RATE);

            let yaw_delta_world = -r * h;
            if yaw_delta_world.abs() > 1e-9 {
                let rotation = Quat::from_axis_angle(Vec3::Y, yaw_delta_world);
                car.quat = sanitize_quat(rotation * car.quat);
            }

            let local_velocity = Vec3::new(-v, 0.0, u);
            car.current_velocity = car.quat * local_velocity;
            car.pos = car.pos.add_vec3(car.current_velocity * h);
        }

        car.yaw_rate = r;

        conform_car_to_terrain(car, terrain, dt);

        camera.target = car.pos;

        let speed_kmh = car.current_velocity.length() * 3.6;
        let steer_deg = car.steering_angle.to_degrees();
        let body_slip = if u.abs() > 1.0 {
            (v / u).atan().to_degrees()
        } else {
            0.0
        };
        let lat_g = (v * r) / G;

        // println!(
        //     "{:6.1} km/h │ δ={:+5.1}° │ β={:+5.1}° │ ṙ={:+5.2} │ Lat≈{:+4.2}g",
        //     speed_kmh, steer_deg, body_slip, r, lat_g
        // );

        (car.id, previous_chunk, car.pos.chunk)
    };

    if previous_chunk != new_chunk {
        cars
            .car_storage_mut()
            .move_car_between_chunks(previous_chunk, new_chunk, car_id);
    }
}

pub fn drive_ai_cars(cars: &mut Cars, terrain: &Terrain, dt: f32) {
    if dt <= 0.0 {
        return;
    }

    const G: f32 = 9.81;
    const MASS: f32 = 1500.0;
    const IZ: f32 = 2500.0;
    const MU: f32 = 1.05;
    const C_AF: f32 = 90_000.0;
    const C_AR: f32 = 100_000.0;
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
    const MAX_SPEED: f32 = 50.0;
    const MAX_YAW_RATE: f32 = 3.5;

    //let player_car = cars.player_car();

    let car_storage = cars.car_storage_mut();
    let cs = chunk_size() as f64;
    let moves: Vec<_> = car_storage
        .par_iter_mut_cars()
        .filter_map(|car| {
            let car = car.as_mut()?;
            if car.mode.is_player() {
                return None;
            }

            let previous_chunk = car.pos.chunk;

            let global_x = car.pos.chunk.x as f64 * cs + car.pos.local.x as f64;
            let global_z = car.pos.chunk.z as f64 * cs + car.pos.local.z as f64;
            let wander_phase = global_x * 0.012 + global_z * 0.008 + (car.id as f64 * 7.3);
            let wander = wander_phase.sin() as f32;

            let steer_input = wander * 0.55;
            let throttle_input = 0.85 + wander * 0.15;
            let brake_input = if wander.abs() > 0.8 { 0.3 } else { 0.0 };

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

            let speed = u.abs();
            let t = (speed / STEER_LOCK_FADE_MPS).clamp(0.0, 1.0);
            let steer_lock =
                STEER_LOCK_HIGH + (STEER_LOCK_LOW - STEER_LOCK_HIGH) * (1.0 - t).powf(1.7);
            let delta_cmd = (steer_input * steer_lock).clamp(-steer_lock, steer_lock);

            let ax_cmd = throttle_input * car.accel - brake_input * car.accel * 3.0;

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

                let u_dot = ax_cmd + ax_drag + r * v;
                let v_dot = (fy_f + fy_r) / MASS - r * u;
                let r_dot = (a * fy_f - b * fy_r) / IZ;

                u += u_dot * h;
                v += v_dot * h;
                r += r_dot * h;

                u = u.clamp(-MAX_SPEED, MAX_SPEED);
                v = v.clamp(-MAX_SPEED * 0.4, MAX_SPEED * 0.4);
                r = r.clamp(-MAX_YAW_RATE, MAX_YAW_RATE);

                let yaw_delta = -r * h;
                if yaw_delta.abs() > 1e-9 {
                    let yaw_q = Quat::from_axis_angle(Vec3::Y, yaw_delta);
                    car.quat = sanitize_quat(yaw_q * car.quat);
                }

                let local_velocity = Vec3::new(-v, 0.0, u);
                car.current_velocity = car.quat * local_velocity;
                car.pos = car.pos.add_vec3(car.current_velocity * h);
            }

            car.yaw_rate = r;

            conform_car_to_terrain(car, terrain, dt);

            if previous_chunk != car.pos.chunk {
                Some((previous_chunk, car.pos.chunk, car.id))
            } else {
                None
            }
        })
        .collect();

    for (prev, new, id) in moves {
        car_storage.move_car_between_chunks(prev, new, id);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PlayerControls {
    pub throttle: f32,
    pub brake: f32,
    pub steer: f32,
    pub handbrake: bool,
}

impl PlayerControls {
    pub fn read(input: &mut Input) -> Self {
        let left = input.action_down("Fly Camera Left");
        let right = input.action_down("Fly Camera Right");

        Self {
            throttle: if input.action_down("Fly Camera Forward") { 1.0 } else { 0.0 },
            brake: if input.action_down("Fly Camera Backward") { 1.0 } else { 0.0 },
            steer: (if left { 1.0 } else { 0.0 }) - (if right { 1.0 } else { 0.0 }),
            handbrake: input.action_down("Handbrake"),
        }
    }
}
#[derive(Clone, Copy)]
struct VehicleDynamics {
    mass: f32,
    iz: f32,
    wheelbase: f32,
    a: f32,
    b: f32,
    cg_height: f32,
    wheel_power: f32,
    max_accel: f32,
    max_brake: f32,
    mu_peak: f32,
    mu_slide: f32,
    c_alpha_f: f32,
    c_alpha_r: f32,
    aero_drag: f32,
    aero_lift: f32,
    rolling_resistance: f32,
    handbrake_grip: f32,
    handbrake_decel: f32,
    steer_rate: f32,
    steer_spring: f32,
    steer_damping: f32,
    steer_low: f32,
    steer_high: f32,
    steer_fade: f32,
    max_speed: f32,
    max_yaw_rate: f32,
}

impl VehicleDynamics {
    fn from_car(car: &Car) -> Self {
        let (
            base_mass,
            ref_length,
            ref_width,
            wheelbase_ratio,
            cg_height,
            front_bias,
            engine_power,
            max_accel,
            max_brake,
            mu_peak,
            mu_slide,
            body_height,
            cd,
            cl,
            steer_low,
            steer_high,
            steer_fade,
            max_speed,
            rolling_resistance,
            heavy,
        ) = match &car.vehicle_type.body_type {
            CarBodyType::Sedan => (
                1620.0, 4.5, 1.85, 0.63, 0.54, 0.52,
                275_000.0, 5.5, 10.5, 1.08, 0.95,
                1.45, 0.28, 0.12, 0.60, 0.09, 35.0, 65.0, 0.010, false,
            ),
            CarBodyType::SportsCar => (
                1450.0, 4.3, 1.9, 0.60, 0.48, 0.48,
                320_000.0, 6.5, 11.5, 1.12, 1.0,
                1.25, 0.31, 0.05, 0.62, 0.10, 38.0, 85.0, 0.010, false,
            ),
            CarBodyType::HyperCar => (
                1650.0, 4.6, 2.0, 0.60, 0.45, 0.46,
                650_000.0, 8.5, 12.5, 1.20, 1.05,
                1.22, 0.30, -0.12, 0.62, 0.10, 40.0, 100.0, 0.010, false,
            ),
            CarBodyType::Bus => (
                12_000.0, 12.0, 2.5, 0.54, 1.0, 0.55,
                230_000.0, 2.3, 7.0, 0.85, 0.78,
                3.0, 0.65, 0.0, 0.45, 0.06, 20.0, 35.0, 0.012, true,
            ),
            CarBodyType::LKW => (
                16_000.0, 9.0, 2.5, 0.55, 1.15, 0.48,
                330_000.0, 2.5, 7.5, 0.88, 0.80,
                3.2, 0.70, 0.0, 0.46, 0.06, 22.0, 38.0, 0.012, true,
            ),
            CarBodyType::Buggy => (
                900.0, 3.5, 1.6, 0.62, 0.50, 0.45,
                110_000.0, 6.0, 9.5, 1.12, 1.0,
                1.35, 0.35, 0.05, 0.62, 0.11, 30.0, 55.0, 0.012, false,
            ),
            CarBodyType::Pickup => (
                2400.0, 5.5, 2.0, 0.60, 0.78, 0.55,
                260_000.0, 4.2, 9.5, 0.98, 0.88,
                1.9, 0.42, 0.08, 0.56, 0.08, 30.0, 60.0, 0.012, false,
            ),
            CarBodyType::Limousine => (
                2200.0, 5.5, 1.9, 0.62, 0.60, 0.54,
                220_000.0, 4.2, 9.5, 1.0, 0.90,
                1.55, 0.30, 0.08, 0.58, 0.08, 34.0, 62.0, 0.010, false,
            ),
            CarBodyType::LimousineForReal => (
                3400.0, 8.0, 1.9, 0.62, 0.68, 0.53,
                300_000.0, 3.5, 8.5, 0.95, 0.85,
                1.65, 0.31, 0.08, 0.52, 0.07, 30.0, 55.0, 0.011, false,
            ),
            CarBodyType::Jeep => (
                2300.0, 4.7, 2.0, 0.60, 0.82, 0.53,
                190_000.0, 3.8, 9.5, 0.98, 0.90,
                1.9, 0.42, 0.08, 0.56, 0.08, 28.0, 55.0, 0.012, false,
            ),
        };

        let (power_mult, efficiency) = match &car.vehicle_type.power_train {
            Powertrain::Combustion => (1.0, 0.85),
            Powertrain::Electric => (1.08, 0.93),
            Powertrain::Hydrogen => (0.95, 0.88),
            Powertrain::Hybrid => (1.05, 0.88),
        };

        let length = car.length.max(0.1);
        let width = car.width.max(0.1);

        let dimension_scale =
            ((length * width) / (ref_length * ref_width)).clamp(0.55, 2.5);

        let mass = base_mass * dimension_scale;
        let wheelbase = (length * wheelbase_ratio).clamp(1.5, 8.0);
        let a = wheelbase * (1.0 - front_bias);
        let b = wheelbase * front_bias;

        let iz = (mass * (length * length + width * width) / 12.0).max(100.0);

        let stiffness_scale = (mass / 1620.0).powf(0.9);

        Self {
            mass,
            iz,
            wheelbase,
            a,
            b,
            cg_height,
            wheel_power: engine_power * power_mult * efficiency,
            max_accel,
            max_brake,
            mu_peak,
            mu_slide,
            c_alpha_f: 115_000.0 * stiffness_scale,
            c_alpha_r: 128_000.0 * stiffness_scale,
            aero_drag: 0.5 * 1.225 * cd * width * body_height,
            aero_lift: 0.5 * 1.225 * cl * width * body_height,
            rolling_resistance,
            handbrake_grip: if heavy { 0.55 } else { 0.30 },
            handbrake_decel: if heavy { 3.0 } else { 5.5 },
            steer_rate: if heavy { 3.5 } else { 6.5 },
            steer_spring: if heavy { 55.0 } else { 75.0 },
            steer_damping: if heavy { 16.0 } else { 18.0 },
            steer_low,
            steer_high,
            steer_fade,
            max_speed,
            max_yaw_rate: if heavy { 1.7 } else { 4.0 },
        }
    }
}
pub fn simulate_player_trajectory(
    car: &Car,
    controls: PlayerControls,
    origin: WorldPos,
    start_time: SimTime,
    duration: f32,
) -> CarTrajectory {
    const G: f32 = 9.81;
    const SUBSTEP_DT: f32 = 1.0 / 120.0;
    const SAMPLE_INTERVAL: u32 = 2;

    let d = VehicleDynamics::from_car(car);

    let mut pos = car.pos;
    let mut quat = sanitize_quat(car.quat);

    let local_velocity = quat.conjugate() * car.current_velocity;

    let mut u = local_velocity.z;
    let mut v = -local_velocity.x;
    let mut r = car.yaw_rate;

    let mut steering_angle = car.steering_angle;
    let mut steering_vel = car.steering_vel;

    let throttle = controls.throttle.clamp(0.0, 1.0);
    let brake = controls.brake.clamp(0.0, 1.0);
    let steer = controls.steer.clamp(-1.0, 1.0);

    let fz_static_f = d.mass * G * (d.b / d.wheelbase);
    let fz_static_r = d.mass * G * (d.a / d.wheelbase);

    let mut pts = Vec::with_capacity(
        (duration / SUBSTEP_DT / SAMPLE_INTERVAL as f32) as usize + 2,
    );

    let mut t = start_time;

    pts.push(CarTrajectoryPoint {
        time: t,
        pos: origin.delta_to(pos),
        quat,
        velocity: car.current_velocity,
        lane_ref: car.current_lane,
    });

    let mut time_remaining = duration;
    let mut step = 0u32;

    while time_remaining > 0.000001 {
        let h = time_remaining.min(SUBSTEP_DT);
        time_remaining -= h;
        step += 1;

        t += h as f64;

        let forward_speed = u.max(0.0);
        let absolute_speed = u.abs();

        let speed_ratio = (absolute_speed / d.steer_fade).clamp(0.0, 1.0);
        let lock_blend = 1.0 - speed_ratio.powf(1.5);

        let max_steer_angle =
            d.steer_high + (d.steer_low - d.steer_high) * lock_blend;

        let delta_command = steer * max_steer_angle;

        let delta_accel =
            d.steer_spring * (delta_command - steering_angle)
                - d.steer_damping * steering_vel;

        steering_vel =
            (steering_vel + delta_accel * h).clamp(-d.steer_rate, d.steer_rate);

        steering_angle =
            (steering_angle + steering_vel * h)
                .clamp(-max_steer_angle, max_steer_angle);

        let power_limited = if forward_speed > 2.0 {
            d.wheel_power / (d.mass * forward_speed)
        } else {
            d.max_accel
        };

        let travel_sign = if u.abs() > 0.1 { u.signum() } else { 1.0 };

        let mut ax_command =
            throttle * d.max_accel.min(power_limited);

        ax_command -= brake * d.max_brake * travel_sign;

        if controls.handbrake && u.abs() > 0.5 {
            ax_command -= d.handbrake_decel * travel_sign;
        }

        let weight_transfer =
            (d.cg_height / d.wheelbase) * d.mass * ax_command;

        let aero_speed = u * u;
        let aero_lift = d.aero_lift * aero_speed;

        let minimum_load = d.mass * G * 0.05;

        let fz_front =
            (fz_static_f - weight_transfer - aero_lift * 0.5)
                .max(minimum_load);

        let fz_rear =
            (fz_static_r + weight_transfer - aero_lift * 0.5)
                .max(minimum_load);

        let u_safe = if u.abs() < 0.4 {
            if u < 0.0 { -0.4 } else { 0.4 }
        } else {
            u
        };

        let alpha_front =
            (v + d.a * r).atan2(u_safe) - steering_angle;

        let alpha_rear =
            (v - d.b * r).atan2(u_safe);

        let mu_rear = if controls.handbrake {
            d.mu_slide * d.handbrake_grip
        } else {
            d.mu_peak
        };

        let fy_front =
            tire_lateral_force(alpha_front, d.c_alpha_f, fz_front, d.mu_peak);

        let fy_rear =
            tire_lateral_force(alpha_rear, d.c_alpha_r, fz_rear, mu_rear);

        let f_rolling =
            -d.rolling_resistance * (fz_front + fz_rear) * u.signum();

        let f_aero = -d.aero_drag * u * u.abs();

        let ax_resist = (f_rolling + f_aero) / d.mass;

        let cos_delta = steering_angle.cos();
        let sin_delta = steering_angle.sin();

        let u_dot =
            ax_command + ax_resist
                - (fy_front * sin_delta) / d.mass
                + r * v;

        let v_dot =
            (fy_front * cos_delta + fy_rear) / d.mass
                - r * u;

        let r_dot =
            (d.a * fy_front * cos_delta - d.b * fy_rear) / d.iz;

        u += u_dot * h;
        v += v_dot * h;
        r += r_dot * h;

        if u.abs() < 0.8
            && throttle < 0.01
            && brake < 0.01
            && !controls.handbrake
        {
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

        u = u.clamp(-d.max_speed, d.max_speed);
        v = v.clamp(-d.max_speed * 0.5, d.max_speed * 0.5);
        r = r.clamp(-d.max_yaw_rate, d.max_yaw_rate);

        let yaw_delta = -r * h;

        if yaw_delta.abs() > 1e-9 {
            quat = sanitize_quat(
                Quat::from_axis_angle(Vec3::Y, yaw_delta) * quat,
            );
        }

        let velocity = quat * Vec3::new(-v, 0.0, u);

        pos = pos.add_vec3(velocity * h);

        if step % SAMPLE_INTERVAL == 0 || time_remaining <= 0.000001 {
            pts.push(CarTrajectoryPoint {
                time: t,
                pos: origin.delta_to(pos),
                quat,
                velocity,
                lane_ref: car.current_lane,
            });
        }
    }

    CarTrajectory {
        car_id: car.id,
        origin,
        points: pts,
        end_quat: quat,
        end_yaw_rate: r,
        end_steering_angle: steering_angle,
        end_steering_vel: steering_vel,
        is_last_turn_of_sf_traj: false,
    }
}

fn tire_lateral_force(
    alpha: f32,
    c_alpha: f32,
    fz: f32,
    mu: f32,
) -> f32 {
    let f_max = (mu * fz).max(1.0);
    let f_linear = -c_alpha * alpha;
    let slip_ratio = f_linear.abs() / f_max;

    let saturation = if slip_ratio > 0.01 {
        slip_ratio.tanh() / slip_ratio
    } else {
        1.0
    };

    f_linear * saturation
}

