use crate::cars::car_structs::{Car, CarChunkDistance};
use crate::cars::car_subsystem::CarSubsystem;
use crate::components::camera::{Camera, CameraController};
use crate::data::Settings;
use crate::renderer::world_renderer::TerrainRenderer;
use crate::terrain::roads::road_mesh_manager::CLEARANCE;
use crate::ui::input::InputState;
use glam::{Quat, Vec3};

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

fn sample_terrain_plane(car: &Car, yaw: f32, terrain: &TerrainRenderer) -> (f32, Vec3) {
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

pub fn conform_car_to_terrain(car: &mut Car, terrain: &TerrainRenderer, dt: f32) -> GroundFit {
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
    terrain: &TerrainRenderer,
    settings: &Settings,
    input: &mut InputState,
    _cam_ctrl: &mut CameraController,
    camera: &mut Camera,
    dt: f32,
) {
    if !settings.drive_car || dt <= 0.0 {
        return;
    }
    drive_ai_cars(car_subsystem, terrain, dt);
    // ---------- tunables / physical-ish constants ----------
    const G: f32 = 9.81;

    // Mass & yaw inertia (typical midsize car)
    const MASS: f32 = 1500.0; // kg
    const IZ: f32 = 2500.0; // kg*m^2 (yaw moment of inertia)

    // Tire model (linear w/ saturation)
    const MU: f32 = 1.05; // peak friction
    const C_AF: f32 = 90000.0; // N/rad (front axle cornering stiffness)
    const C_AR: f32 = 100000.0; // N/rad (rear axle cornering stiffness)

    // Longitudinal resistances (simple but better than pure linear)
    const ROLL_DRAG: f32 = 0.1; // 1/s  (rough rolling resistance as accel)
    const AERO_DRAG: f32 = 0.0007; // 1/m  (accel = k*u*|u|)

    // Braking / handbrake behavior
    const HANDBRAKE_REAR_MU_MULT: f32 = 0.55; // reduce rear lateral grip when handbrake
    const HANDBRAKE_BRAKE_A: f32 = 8.0; // extra braking m/s^2 when engaged

    // Steering system (rack dynamics)
    const STEER_RATE_MAX: f32 = 6.0; // rad/s max rack speed
    const STEER_KP: f32 = 70.0; // stiffness toward commanded angle
    const STEER_KD: f32 = 14.0; // damping of steering velocity

    // Speed-sensitive steering lock (road wheel angle, not steering wheel)
    const STEER_LOCK_LOW: f32 = 0.62; // ~35 deg at parking
    const STEER_LOCK_HIGH: f32 = 0.14; // ~8 deg at highway
    const STEER_LOCK_FADE_MPS: f32 = 28.0; // ~100 km/h

    // numerical stability
    const MAX_STEP: f32 = 1.0 / 120.0; // substep for stable tire dynamics

    const MIN_U: f32 = 0.5; // prevents slip-angle singularities

    let chunk_size = terrain.chunk_size;

    // inputs
    let forward = input.gameplay_down("Fly Camera Forward");
    let backward = input.gameplay_down("Fly Camera Backward");
    let left = input.gameplay_down("Fly Camera Left");
    let right = input.gameplay_down("Fly Camera Right");
    let handbrake = input.gameplay_down("Handbrake");

    let (car_id, previous_chunk, new_chunk) = {
        let Some(car) = car_subsystem.car_storage_mut().get_mut(0) else {
            return;
        };

        // Convert world velocity -> vehicle local, then map to SAE-like signs for the model:
        // Your local: x=right, z=forward
        // SAE model used here: u=forward(+), v=left(+), r=yaw left(+)
        let inv_q = car.quat.conjugate();
        let local_v = inv_q * car.current_velocity;

        let mut u = local_v.z; // forward speed (m/s)
        let mut v = -local_v.x; // left-positive lateral speed (m/s)
        let mut r = car.yaw_rate; // yaw rate left-positive (rad/s)

        // wheelbase + CG split (if you have real a/b use those)
        let wheelbase = car.length.max(0.1);
        let a = 0.5 * wheelbase; // CG -> front axle
        let b = wheelbase - a; // CG -> rear axle

        // static normal loads
        let fz_f = MASS * G * (b / wheelbase);
        let fz_r = MASS * G * (a / wheelbase);

        // ---------- steering command (speed-sensitive lock) ----------
        let steer_input = (if right { 1.0 } else { 0.0 }) - (if left { 1.0 } else { 0.0 });

        let speed = u.abs();
        let t = (speed / STEER_LOCK_FADE_MPS).clamp(0.0, 1.0);
        let steer_lock = STEER_LOCK_HIGH + (STEER_LOCK_LOW - STEER_LOCK_HIGH) * (1.0 - t).powf(1.7);

        // +delta = steer left (SAE sign)
        let delta_cmd = (steer_input * steer_lock).clamp(-steer_lock, steer_lock);

        // ---------- longitudinal accel command ----------
        // Treat car.accel/decel as m/s^2 (as your code effectively does).
        let mut ax_cmd = if forward {
            car.accel
        } else if backward {
            -car.accel
        } else {
            0.0
        };

        // handbrake adds braking accel opposing motion
        if handbrake && u.abs() > 0.2 {
            ax_cmd += -HANDBRAKE_BRAKE_A * u.signum();
        }

        // ---------- integrate with substeps ----------
        let mut time_left = dt;
        while time_left > 0.0 {
            let h = time_left.min(MAX_STEP);
            time_left -= h;

            // steering rack dynamics (2nd order)
            // delta_ddot ~ kp*(delta_cmd-delta) - kd*delta_dot
            let delta = car.steering_angle;
            let mut delta_dot = car.steering_vel;

            delta_dot += (STEER_KP * (delta_cmd - delta) - STEER_KD * delta_dot) * h;
            delta_dot = delta_dot.clamp(-STEER_RATE_MAX, STEER_RATE_MAX);

            let mut delta_new = delta + delta_dot * h;
            delta_new = delta_new.clamp(-steer_lock, steer_lock);

            car.steering_angle = delta_new;
            car.steering_vel = delta_dot;

            // safe forward speed for slip calculations
            let u_safe = if u.abs() < MIN_U {
                MIN_U * u.signum().max(1.0)
            } else {
                u
            };

            // slip angles (SAE signs)
            let alpha_f = (v + a * r).atan2(u_safe) - car.steering_angle;
            let alpha_r = (v - b * r).atan2(u_safe);

            // lateral tire forces (linear + saturation)
            let mut mu_r = MU;
            if handbrake {
                mu_r *= HANDBRAKE_REAR_MU_MULT; // handbrake breaks rear grip -> easier oversteer
            }

            let fy_f_lin = -C_AF * alpha_f;
            let fy_r_lin = -C_AR * alpha_r;

            let fy_f_max = MU * fz_f;
            let fy_r_max = mu_r * fz_r;

            let fy_f = fy_f_lin.clamp(-fy_f_max, fy_f_max);
            let fy_r = fy_r_lin.clamp(-fy_r_max, fy_r_max);

            // longitudinal drag (as acceleration)
            let ax_drag = -ROLL_DRAG * u - AERO_DRAG * u * u.abs();

            // bicycle model dynamics
            // u_dot = ax + r*v
            // v_dot = (Fy_f + Fy_r)/m - r*u
            // r_dot = (a*Fy_f - b*Fy_r)/Iz
            let u_dot = (ax_cmd + ax_drag) + r * v;
            let v_dot = (fy_f + fy_r) / MASS - r * u;
            let r_dot = (a * fy_f - b * fy_r) / IZ;

            u += u_dot * h;
            v += v_dot * h;
            r += r_dot * h;
        }

        // store yaw-rate (SAE sign)
        car.yaw_rate = r;

        // integrate orientation from yaw rate.
        // Your world yaw convention is opposite SAE; SAE(+left) -> world yaw delta is negative.
        let yaw_delta_world = -r * dt;
        if yaw_delta_world.abs() > 0.0 {
            let yaw_q = Quat::from_axis_angle(Vec3::Y, yaw_delta_world);
            car.quat = (yaw_q * car.quat).normalize();
        }

        // reconstruct world velocity from (u,v) back to your local coords:
        // SAE v is left+, your local x is right+ => local_x = -v
        let new_local_v = Vec3::new(-v, 0.0, u);
        car.current_velocity = car.quat * new_local_v;

        // integrate position
        let prev_chunk = car.pos.chunk;
        car.pos = car.pos.add_vec3(car.current_velocity * dt, chunk_size);
        car.pos.local.y = terrain.get_height_at(car.pos) + CLEARANCE;

        camera.target = car.pos;

        println!(
            "{:.1} km/h  steer={:.2}deg  u={:.2} v={:.2} r={:.3}",
            car.current_velocity.length() * 3.6,
            car.steering_angle.to_degrees(),
            u,
            v,
            r
        );

        (car.id, prev_chunk, car.pos.chunk)
    };

    if previous_chunk != new_chunk {
        car_subsystem.car_storage_mut().move_car_between_chunks(
            previous_chunk,
            new_chunk,
            CarChunkDistance::Close,
            car_id,
        );
    }
}
pub fn drive_ai_cars(car_subsystem: &mut CarSubsystem, terrain: &TerrainRenderer, dt: f32) {
    if dt <= 0.0 {
        return;
    }

    // ---------- Same constants as player ----------
    const G: f32 = 9.81;
    const MASS: f32 = 1500.0;
    const IZ: f32 = 2500.0;
    const MU: f32 = 1.05;
    const C_AF: f32 = 90000.0;
    const C_AR: f32 = 100000.0;
    const ROLL_DRAG: f32 = 0.1;
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
    let cs = chunk_size as f64;

    for car in car_subsystem.car_storage_mut().iter_mut_cars() {
        let Some(car) = car else {
            continue;
        };
        if car.id == 0 {
            continue;
        }

        // === AI Inputs: smooth procedural wandering ===
        let global_x = car.pos.chunk.x as f64 * cs + car.pos.local.x as f64;
        let global_z = car.pos.chunk.z as f64 * cs + car.pos.local.z as f64;

        let wander_phase = global_x * 0.012 + global_z * 0.008 + (car.id as f64 * 7.3);
        let wander = wander_phase.sin() as f32;

        let steer_input = wander * 0.55;
        let throttle_input = 0.85 + wander * 0.15;
        let brake_input = if wander.abs() > 0.8 { 0.3 } else { 0.0 };

        // === FIX #1: Convert world velocity to LOCAL space via quaternion ===
        let inv_q = car.quat.conjugate();
        let local_v = inv_q * car.current_velocity;

        let mut u = local_v.z; // forward speed (m/s)
        let mut v = -local_v.x; // lateral speed, SAE left-positive (m/s)
        let mut r = car.yaw_rate;

        let wheelbase = car.length.max(0.1);
        let a = 0.5 * wheelbase;
        let b = wheelbase - a;

        let fz_f = MASS * G * (b / wheelbase);
        let fz_r = MASS * G * (a / wheelbase);

        // === Speed-sensitive steering lock (same as player) ===
        let speed = u.abs();
        let t = (speed / STEER_LOCK_FADE_MPS).clamp(0.0, 1.0);
        let steer_lock = STEER_LOCK_HIGH + (STEER_LOCK_LOW - STEER_LOCK_HIGH) * (1.0 - t).powf(1.7);

        let delta_cmd = (steer_input * steer_lock).clamp(-steer_lock, steer_lock);

        // === Longitudinal accel command ===
        let ax_cmd = throttle_input * car.accel - brake_input * car.accel * 3.0;

        // === FIX #2: Substep integration for stability ===
        let mut time_left = dt;
        while time_left > 0.0 {
            let h = time_left.min(MAX_STEP);
            time_left -= h;

            // === FIX #3: Steering rack dynamics (2nd order, same as player) ===
            let delta = car.steering_angle;
            let mut delta_dot = car.steering_vel;

            delta_dot += (STEER_KP * (delta_cmd - delta) - STEER_KD * delta_dot) * h;
            delta_dot = delta_dot.clamp(-STEER_RATE_MAX, STEER_RATE_MAX);

            let mut delta_new = delta + delta_dot * h;
            delta_new = delta_new.clamp(-steer_lock, steer_lock);

            car.steering_angle = delta_new;
            car.steering_vel = delta_dot;

            // Safe forward speed for slip calculations
            let u_safe = if u.abs() < MIN_U {
                MIN_U * u.signum().max(1.0)
            } else {
                u
            };

            // === FIX #4: Correct slip angle formula with atan2 ===
            let alpha_f = (v + a * r).atan2(u_safe) - car.steering_angle;
            let alpha_r = (v - b * r).atan2(u_safe);

            // Lateral tire forces (linear + saturation)
            let fy_f_lin = -C_AF * alpha_f;
            let fy_r_lin = -C_AR * alpha_r;

            let fy_f_max = MU * fz_f;
            let fy_r_max = MU * fz_r;

            let fy_f = fy_f_lin.clamp(-fy_f_max, fy_f_max);
            let fy_r = fy_r_lin.clamp(-fy_r_max, fy_r_max);

            // Drag
            let ax_drag = -ROLL_DRAG * u - AERO_DRAG * u * u.abs();

            // Bicycle model dynamics
            let u_dot = (ax_cmd + ax_drag) + r * v;
            let v_dot = (fy_f + fy_r) / MASS - r * u;
            let r_dot = (a * fy_f - b * fy_r) / IZ;

            u += u_dot * h;
            v += v_dot * h;
            r += r_dot * h;
        }

        // Store yaw rate
        car.yaw_rate = r;

        // === FIX #5: Proper orientation integration ===
        let yaw_delta_world = -r * dt;
        if yaw_delta_world.abs() > 0.0 {
            let yaw_q = Quat::from_axis_angle(Vec3::Y, yaw_delta_world);
            car.quat = (yaw_q * car.quat).normalize();
        }

        // === Reconstruct world velocity from local (u,v) ===
        let new_local_v = Vec3::new(-v, 0.0, u);
        car.current_velocity = car.quat * new_local_v;

        // Position integration + terrain snap
        car.pos = car.pos.add_vec3(car.current_velocity * dt, chunk_size);
        car.pos.local.y = terrain.get_height_at(car.pos) + CLEARANCE;
    }
}
