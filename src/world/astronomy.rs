#![allow(dead_code)]

use core::default::Default;
use glam::{Mat3, Mat4, Vec3};
use std::f64::consts::TAU;

pub const MUNICH_LATITUDE: f64 = 48.1351;
pub const MUNICH_LONGITUDE: f64 = 11.5820;
pub const J2000: f64 = 2451545.0;
pub const GAME_EPOCH_JD: f64 = 2461041.5; // 2026-01-01 00:00 UTC

pub struct Astronomy {
    pub star_rotation: Mat4,
    pub sun_dir: Vec3,
    pub sun_size_ndc: f64,
    pub moon_dir: Vec3,
    pub moon_size_ndc: f64,
    pub moon_phase: f64,
    pub sun_declination: f64,
    pub day_phase: f64,
    pub current_year: f64,
}

impl Default for Astronomy {
    fn default() -> Self {
        Self {
            star_rotation: Default::default(),
            sun_dir: Default::default(),
            sun_size_ndc: 0.0465,
            moon_dir: Default::default(),
            moon_size_ndc: 0.045,
            moon_phase: 0.0,
            sun_declination: 0.0,
            day_phase: 0.0,
            current_year: 0.0,
        }
    }
}
pub struct TimeScales {
    pub day_length: f64,
    pub total_days: f64,
    pub jd: f64,
    pub day_phase: f64,
    pub day_angle: f64,
    pub year_phase: f64,
    pub year_angle: f64,
    pub base_year: f64,
    pub current_year: f64,
}

impl TimeScales {
    pub fn from_game_time(total_game_time: f64, always_day: bool) -> Self {
        let day_length: f64 = 960.0;
        let total_days = total_game_time / day_length as f64;
        let jd = GAME_EPOCH_JD + total_days;

        let base_day_phase = (total_days as f64) % 1.0;

        let day_phase = if always_day {
            let t = (total_game_time as f64 / day_length) * TAU;
            let ping_pong = t.sin().abs();
            0.25 + 0.25 * ping_pong
        } else {
            base_day_phase
        };

        let day_angle = day_phase * TAU;

        let year_phase = ((total_days / 365.2425) % 1.0) as f64;
        let year_angle = year_phase * TAU;

        let base_year = 2026.0;
        let current_year = base_year + (total_days / 365.2425) as f64;

        Self {
            day_length,
            total_days,
            jd,
            day_phase,
            day_angle,
            year_phase,
            year_angle,
            base_year,
            current_year,
        }
    }
}

pub struct ObserverParams {
    pub latitude: f64,
    pub lat_rotation: glam::Quat,
    pub earth_rotation: glam::Quat,
    pub obliquity: f64,
    pub obliquity_rotation: glam::Quat,
}

impl ObserverParams {
    pub fn from_jd(jd: f64) -> Self {
        let latitude = MUNICH_LATITUDE.to_radians() as f64;
        let lat_rotation = glam::Quat::from_rotation_x(latitude as f32);

        let lst = (compute_gmst(jd) + MUNICH_LONGITUDE.to_radians()).rem_euclid(TAU);

        let earth_rotation = glam::Quat::from_rotation_y(lst as f32);

        let obliquity = 23.439_f64.to_radians();
        let obliquity_rotation = glam::Quat::from_rotation_x(-obliquity as f32);

        Self {
            latitude,
            lat_rotation,
            earth_rotation,
            obliquity,
            obliquity_rotation,
        }
    }
}
pub fn equatorial_to_local(eq: Vec3, jd: f64) -> Vec3 {
    let lat = MUNICH_LATITUDE.to_radians() as f64;
    let lst = (compute_gmst(jd) + MUNICH_LONGITUDE.to_radians()) as f64;

    // derive RA/Dec from equatorial unit vector
    let ra = eq.z.atan2(eq.x) as f64;
    let dec = eq.y.asin() as f64;

    let h = lst - ra; // hour angle

    let sin_alt = dec.sin() * lat.sin() + dec.cos() * lat.cos() * h.cos();
    let alt = sin_alt.asin();

    let y = alt.sin(); // up

    let x = -dec.cos() * h.sin(); // east-west
    let z = dec.sin() * lat.cos() - dec.cos() * lat.sin() * h.cos(); // north-forward

    Vec3::new(x as f32, y as f32, z as f32).normalize()
}
pub fn compute_gmst(jd: f64) -> f64 {
    let t = (jd - J2000) / 36525.0;
    let gmst_0h = 24110.54841 + 8640184.812866 * t + 0.093104 * t * t - 6.2e-6 * t * t * t;
    let ut1_frac = (jd + 0.5).fract();
    let gmst_seconds = gmst_0h + 86400.0 * 1.00273790935 * ut1_frac;
    ((gmst_seconds / 86400.0) * TAU).rem_euclid(TAU)
}

/// Returns the Earth–Sun distance in astronomical units (AU) for a given Julian Day.
///
/// This uses a Kepler-based approximation (good enough to reproduce the annual
/// apparent-size variation of the Sun).
pub fn compute_sun_distance_au(jd: f64) -> f64 {
    let n = jd - J2000;

    let mean_anomaly_rad = (357.52911 + 0.98560028 * n).to_radians();

    let e = 0.016708634_f64;
    let a_au = 1.000001018_f64;

    let e_anom = mean_anomaly_rad + e * mean_anomaly_rad.sin() * (1.0 + e * mean_anomaly_rad.cos());

    a_au * (1.0 - e * e_anom.cos())
}

/// Returns the apparent angular radius of the Sun in radians for a given Julian Day.
pub fn compute_sun_angular_radius_rad(jd: f64) -> f64 {
    let d_au = compute_sun_distance_au(jd);

    let au_km = 149_597_870.7_f64;
    let sun_radius_km = 695_700.0_f64;

    let d_km = d_au * au_km;
    (sun_radius_km / d_km).atan()
}

pub fn compute_moon_equatorial_direction_and_distance(jd: f64) -> (Vec3, f64) {
    let d = jd - J2000;

    let n = (125.1228 - 0.0529538083 * d).to_radians();
    let i = 5.145_f64.to_radians();
    let w = (318.0634 + 0.1643573223 * d).to_radians();

    let a = 60.2666_f64;
    let e = 0.054900_f64;
    let m = (115.3654 + 13.0649929509 * d).to_radians();

    let e_anom = m + e * m.sin() * (1.0 + e * m.cos());
    let xv = a * (e_anom.cos() - e);
    let yv = a * ((1.0 - e * e).sqrt() * e_anom.sin());
    let v = yv.atan2(xv);
    let r = (xv * xv + yv * yv).sqrt();

    let xh = r * (n.cos() * (v + w).cos() - n.sin() * (v + w).sin() * i.cos());
    let yh = r * (n.sin() * (v + w).cos() + n.cos() * (v + w).sin() * i.cos());
    let zh = r * ((v + w).sin() * i.sin());

    let moon_ecl = Vec3::new(xh as f32, yh as f32, zh as f32);
    let moon_eq =
        (glam::Quat::from_rotation_x(23.439_f64.to_radians() as f32) * moon_ecl).normalize(); // +ε

    (moon_eq, r)
}

pub fn compute_moon_direction(observer: &ObserverParams, jd: f64) -> Vec3 {
    let (moon_eq, _) = compute_moon_equatorial_direction_and_distance(jd);
    equatorial_to_local(moon_eq, jd)
}

pub fn compute_moon_topocentric_distance_earth_radii(
    geocentric_distance_earth_radii: f64,
    altitude_rad: f64,
) -> f64 {
    let r = geocentric_distance_earth_radii;
    (r * r + 1.0 - 2.0 * r * altitude_rad.sin()).sqrt()
}

pub fn compute_moon_angular_radius_rad(observer: &ObserverParams, jd: f64) -> f64 {
    let (moon_eq, r_geo) = compute_moon_equatorial_direction_and_distance(jd);
    let moon_local = equatorial_to_local(moon_eq, jd);

    let altitude_rad = moon_local.y.clamp(-1.0, 1.0).asin() as f64;
    let r_topo = compute_moon_topocentric_distance_earth_radii(r_geo, altitude_rad);

    let earth_radius_km = 6378.137_f64;
    let moon_radius_km = 1737.4_f64;

    let d_km = r_topo * earth_radius_km;
    (moon_radius_km / d_km).atan()
}

pub fn compute_moon_phase(jd: f64) -> f64 {
    let sun_eq = compute_sun_equatorial_direction(jd);
    let (moon_eq, _) = compute_moon_equatorial_direction_and_distance(jd);

    let illumination = (1.0 - sun_eq.dot(moon_eq).clamp(-1.0, 1.0) as f64) * 0.5;
    illumination
}

/// Converts a true angular radius (radians) into the NDC radius your shader expects.
/// This assumes a conventional perspective projection where `proj.y_axis.y = 1/tan(fov_y/2)`.
pub fn angular_radius_rad_to_ndc_radius(proj: Mat4, angular_radius_rad: f64) -> f64 {
    angular_radius_rad.tan() * proj.y_axis.y as f64
}
pub fn compute_star_rotation(jd: f64) -> Mat4 {
    let gmst = compute_gmst(jd);
    let lst = (gmst + MUNICH_LONGITUDE.to_radians()).rem_euclid(TAU);

    let sidereal_rot = Mat3::from_rotation_y(-lst as f32);
    let lat = MUNICH_LATITUDE.to_radians();
    let latitude_rot = Mat3::from_rotation_x((lat - std::f64::consts::FRAC_PI_2) as f32);

    Mat4::from_mat3(latitude_rot * sidereal_rot)
}

pub fn compute_sun_equatorial_direction(jd: f64) -> Vec3 {
    let n = jd - J2000;

    let l_deg = (280.460 + 0.9856474 * n).rem_euclid(360.0);
    let g_deg = (357.528 + 0.9856003 * n).rem_euclid(360.0);

    let g = g_deg.to_radians();
    let lambda_deg = l_deg + 1.915 * g.sin() + 0.020 * (2.0 * g).sin();
    let lambda = lambda_deg.to_radians();

    let epsilon = (23.439 - 0.0000004 * n).to_radians();

    let x = lambda.cos();
    let y = epsilon.cos() * lambda.sin();
    let z = epsilon.sin() * lambda.sin();

    Vec3::new(x as f32, y as f32, z as f32).normalize()
}

pub fn compute_sun_direction(observer: &ObserverParams, jd: f64) -> (Vec3, f64) {
    let sun_eq = compute_sun_equatorial_direction(jd);
    let sun_declination = sun_eq.y.asin().to_degrees() as f64;
    let sun_dir = equatorial_to_local(sun_eq, jd);
    (sun_dir, sun_declination)
}

/// Computes `AstronomyState` plus Sun/Moon disc radii in the exact units expected by your shader.
///
/// The returned `sun_size_ndc` and `moon_size_ndc` should be written into `SkyUniform.sun_size`
/// and `SkyUniform.moon_size`.
pub fn compute_astronomy(time_scales: &TimeScales, proj: Mat4) -> Astronomy {
    let jd = time_scales.jd;
    let observer = ObserverParams::from_jd(jd);

    let (sun_dir, sun_declination) = compute_sun_direction(&observer, jd);
    let moon_dir = compute_moon_direction(&observer, jd);
    let moon_phase = compute_moon_phase(jd);

    let star_rotation = compute_star_rotation(jd);

    let sun_ang_rad = compute_sun_angular_radius_rad(jd);
    let moon_ang_rad = compute_moon_angular_radius_rad(&observer, jd);

    let sun_size_ndc = angular_radius_rad_to_ndc_radius(proj, sun_ang_rad) * 2.5;
    let moon_size_ndc = angular_radius_rad_to_ndc_radius(proj, moon_ang_rad) * 2.5;
    Astronomy {
        star_rotation,
        sun_dir,
        sun_size_ndc,
        moon_dir,
        moon_size_ndc,
        moon_phase,
        sun_declination,
        day_phase: time_scales.day_phase,
        current_year: time_scales.current_year,
    }
}
