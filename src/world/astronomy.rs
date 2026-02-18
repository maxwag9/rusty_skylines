#![allow(dead_code)]

use core::default::Default;
use glam::{Mat3, Mat4, Vec3};
use std::f32::consts::TAU;

pub const MUNICH_LATITUDE: f64 = 48.1351;
pub const MUNICH_LONGITUDE: f64 = 11.5820;
pub const J2000: f64 = 2451545.0;

pub struct AstronomyState {
    pub star_rotation: Mat4,
    pub sun_dir: Vec3,
    pub sun_size_ndc: f32,
    pub moon_dir: Vec3,
    pub moon_size_ndc: f32,
    pub moon_phase: f32,
    pub sun_declination: f32,
    pub day_phase: f32,
    pub current_year: f32,
}

impl Default for AstronomyState {
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
    pub day_length: f32,
    pub total_days: f32,
    pub day_phase: f32,
    pub day_angle: f32,
    pub year_phase: f32,
    pub year_angle: f32,
    pub base_year: f32,
    pub current_year: f32,
}

impl TimeScales {
    pub fn from_game_time(total_game_time: f64, always_day: bool) -> Self {
        let day_length: f32 = 960.0;
        let total_days: f32 = (total_game_time / day_length as f64) as f32;

        let base_day_phase = total_days % 1.0;

        let day_phase = if always_day {
            // ping-pong between morning (0.25) and noon (0.5)
            let t = (total_game_time as f32 / day_length) * TAU;
            let ping_pong = t.sin().abs(); // 0 → 1 → 0
            0.25 + 0.25 * ping_pong
        } else {
            base_day_phase
        };

        let day_angle = day_phase * TAU;

        let year_phase = (total_days / 365.0) % 1.0;
        let year_angle = year_phase * TAU;

        let base_year = 2026.0;
        let current_year = base_year + total_days / 365.0;

        Self {
            day_length,
            total_days,
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
    pub latitude: f32,
    pub lat_rotation: glam::Quat,
    pub earth_rotation: glam::Quat,
    pub obliquity: f32,
    pub obliquity_rotation: glam::Quat,
}

impl ObserverParams {
    pub fn new(day_angle: f32) -> Self {
        let latitude = MUNICH_LATITUDE.to_radians() as f32;
        let lat_rotation = glam::Quat::from_rotation_x(latitude);

        let sidereal_factor = 1.0027379_f32;
        let sidereal_angle = day_angle * sidereal_factor;
        let earth_rotation = glam::Quat::from_rotation_y(sidereal_angle);

        let obliquity = 23.439_f32.to_radians();
        let obliquity_rotation = glam::Quat::from_rotation_x(-obliquity);

        Self {
            latitude,
            lat_rotation,
            earth_rotation,
            obliquity,
            obliquity_rotation,
        }
    }
}

pub fn compute_gmst(jd: f64) -> f64 {
    let t = (jd - J2000) / 36525.0;
    let gmst_0h = 24110.54841 + 8640184.812866 * t + 0.093104 * t * t - 6.2e-6 * t * t * t;
    let ut1_frac = (jd + 0.5).fract();
    let gmst_seconds = gmst_0h + 86400.0 * 1.00273790935 * ut1_frac;
    (gmst_seconds / 86400.0) * core::f64::consts::TAU
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
pub fn compute_sun_angular_radius_rad(jd: f64) -> f32 {
    let d_au = compute_sun_distance_au(jd);

    let au_km = 149_597_870.7_f64;
    let sun_radius_km = 695_700.0_f64;

    let d_km = d_au * au_km;
    (sun_radius_km / d_km).atan() as f32
}

/// Returns a normalized geocentric Moon direction in your local sky coordinates,
/// plus the geocentric distance in Earth radii.
pub fn compute_moon_geocentric_direction_and_distance(
    observer: &ObserverParams,
    total_days: f32,
) -> (Vec3, f32) {
    let jd = J2000 + total_days as f64;
    let t = (jd - J2000) as f32 / 36525.0;

    let n = (125.122 - 0.0529538083 * t).to_radians();
    let i = 5.145_f32.to_radians();
    let w = (318.063 + 0.1643573223 * t).to_radians();

    let a = 60.2666_f32;
    let e = 0.054900_f32;
    let m = (115.3654 + 13.06499295 * total_days).to_radians();

    let e_anom = m + e * m.sin() * (1.0 + e * m.cos());
    let xv = a * (e_anom.cos() - e);
    let yv = a * ((1.0 - e * e).sqrt() * e_anom.sin());
    let v = yv.atan2(xv);
    let r = (xv * xv + yv * yv).sqrt();

    let xh = r * (n.cos() * (v + w).cos() - n.sin() * (v + w).sin() * i.cos());
    let zh = r * ((v + w).sin() * i.sin());
    let yh = r * (n.sin() * (v + w).cos() + n.cos() * (v + w).sin() * i.cos());

    let moon_ecl_pos = Vec3::new(xh, zh, yh);
    let moon_eq_pos = observer.obliquity_rotation * moon_ecl_pos;
    let moon_local_pos = observer.lat_rotation * observer.earth_rotation * moon_eq_pos;

    (moon_local_pos.normalize(), r)
}

/// Converts geocentric Moon distance (Earth radii) into topocentric distance (Earth radii),
/// using the Moon altitude in radians as seen by the observer.
///
/// This captures the small Munich-dependent effect from being on Earth's surface
/// (largest when the Moon is high in the sky).
pub fn compute_moon_topocentric_distance_earth_radii(
    geocentric_distance_earth_radii: f32,
    altitude_rad: f32,
) -> f32 {
    let r = geocentric_distance_earth_radii;
    (r * r + 1.0 - 2.0 * r * altitude_rad.sin()).sqrt()
}

/// Returns the apparent angular radius of the Moon in radians as seen by the observer in Munich
/// (via `ObserverParams`), including the small topocentric distance change.
pub fn compute_moon_angular_radius_rad(observer: &ObserverParams, total_days: f32) -> f32 {
    let (moon_dir_geo, r_geo) =
        compute_moon_geocentric_direction_and_distance(observer, total_days);

    let altitude_rad = moon_dir_geo.y.clamp(-1.0, 1.0).asin();
    let r_topo = compute_moon_topocentric_distance_earth_radii(r_geo, altitude_rad);

    let earth_radius_km = 6378.137_f32;
    let moon_radius_km = 1737.4_f32;

    let d_km = r_topo * earth_radius_km;
    (moon_radius_km / d_km).atan()
}

/// Converts a true angular radius (radians) into the NDC radius your shader expects.
/// This assumes a conventional perspective projection where `proj.y_axis.y = 1/tan(fov_y/2)`.
pub fn angular_radius_rad_to_ndc_radius(proj: Mat4, angular_radius_rad: f32) -> f32 {
    angular_radius_rad.tan() * proj.y_axis.y
}
pub fn compute_star_rotation(jd: f64) -> Mat4 {
    let gmst = compute_gmst(jd);
    let lst = gmst + MUNICH_LONGITUDE.to_radians();

    let sidereal_rot = Mat3::from_rotation_y(-lst as f32);
    let lat = MUNICH_LATITUDE.to_radians() as f32;
    let latitude_rot = Mat3::from_rotation_x(lat - std::f32::consts::FRAC_PI_2);

    glam::Mat4::from_mat3(latitude_rot * sidereal_rot)
}

pub fn compute_sun_direction(observer: &ObserverParams, year_phase: f32) -> (glam::Vec3, f32) {
    let sun_ecliptic_lon = year_phase * TAU;
    let sun_ecl = glam::Vec3::new(sun_ecliptic_lon.cos(), 0.0, sun_ecliptic_lon.sin());
    let sun_eq = (observer.obliquity_rotation * sun_ecl).normalize();
    let sun_declination = sun_eq.y.asin().to_degrees();
    let sun_dir = (observer.lat_rotation * observer.earth_rotation * sun_eq).normalize();

    (sun_dir, sun_declination)
}

pub fn compute_moon_direction(observer: &ObserverParams, total_days: f32) -> glam::Vec3 {
    let jd = 2451545.0 + total_days;
    let t = (jd - 2451545.0) / 36525.0;

    let n = (125.122 - 0.0529538083 * t).to_radians();
    let i = 5.145_f32.to_radians();
    let w = (318.063 + 0.1643573223 * t).to_radians();

    let a = 60.2666_f32;
    let e = 0.054900_f32;
    let m = (115.3654 + 13.06499295 * total_days).to_radians();

    let e_anom = m + e * m.sin() * (1.0 + e * m.cos());
    let xv = a * (e_anom.cos() - e);
    let yv = a * ((1.0 - e * e).sqrt() * e_anom.sin());
    let v = yv.atan2(xv);
    let r = (xv * xv + yv * yv).sqrt();

    let xh = r * (n.cos() * (v + w).cos() - n.sin() * (v + w).sin() * i.cos());
    let zh = r * ((v + w).sin() * i.sin());
    let yh = r * (n.sin() * (v + w).cos() + n.cos() * (v + w).sin() * i.cos());

    let moon_ecl = glam::Vec3::new(xh, zh, yh).normalize();
    let moon_eq = (observer.obliquity_rotation * moon_ecl).normalize();

    (observer.lat_rotation * observer.earth_rotation * moon_eq).normalize()
}

pub fn compute_moon_phase(observer: &ObserverParams, year_phase: f32, total_days: f32) -> f32 {
    let sun_ecliptic_lon = year_phase * TAU;
    let sun_ecl = glam::Vec3::new(sun_ecliptic_lon.cos(), 0.0, sun_ecliptic_lon.sin());
    let sun_eq = (observer.obliquity_rotation * sun_ecl).normalize();

    // Recompute moon_eq for phase calculation
    let jd = 2451545.0 + total_days;
    let t = (jd - 2451545.0) / 36525.0;
    let n = (125.122 - 0.0529538083 * t).to_radians();
    let i = 5.145_f32.to_radians();
    let w = (318.063 + 0.1643573223 * t).to_radians();
    let a = 60.2666_f32;
    let e = 0.054900_f32;
    let m = (115.3654 + 13.06499295 * total_days).to_radians();
    let e_anom = m + e * m.sin() * (1.0 + e * m.cos());
    let xv = a * (e_anom.cos() - e);
    let yv = a * ((1.0 - e * e).sqrt() * e_anom.sin());
    let v = yv.atan2(xv);
    let r = (xv * xv + yv * yv).sqrt();
    let xh = r * (n.cos() * (v + w).cos() - n.sin() * (v + w).sin() * i.cos());
    let zh = r * ((v + w).sin() * i.sin());
    let yh = r * (n.sin() * (v + w).cos() + n.cos() * (v + w).sin() * i.cos());
    let moon_ecl = glam::Vec3::new(xh, zh, yh).normalize();
    let moon_eq = (observer.obliquity_rotation * moon_ecl).normalize();

    let phase_angle = sun_eq.dot(moon_eq).clamp(-1.0, 1.0).acos();
    (1.0 - phase_angle.cos()) * 0.5
}

/// Computes `AstronomyState` plus Sun/Moon disc radii in the exact units expected by your shader.
///
/// The returned `sun_size_ndc` and `moon_size_ndc` should be written into `SkyUniform.sun_size`
/// and `SkyUniform.moon_size`.
pub fn compute_astronomy(time_scales: &TimeScales, proj: Mat4) -> AstronomyState {
    let base_jd_2026_01_01 = J2000 + 9131.5;
    let jd = base_jd_2026_01_01 + time_scales.total_days as f64;
    let days_since_j2000 = (jd - J2000) as f32;

    let observer = ObserverParams::new(time_scales.day_angle);

    let (sun_dir, sun_declination) = compute_sun_direction(&observer, time_scales.year_phase);
    let moon_dir = compute_moon_direction(&observer, days_since_j2000);
    let moon_phase = compute_moon_phase(&observer, time_scales.year_phase, days_since_j2000);

    let star_rotation = compute_star_rotation(jd);

    let sun_ang_rad = compute_sun_angular_radius_rad(jd);
    let moon_ang_rad = compute_moon_angular_radius_rad(&observer, days_since_j2000);

    let sun_size_ndc = angular_radius_rad_to_ndc_radius(proj, sun_ang_rad) * 2.5;
    let moon_size_ndc = angular_radius_rad_to_ndc_radius(proj, moon_ang_rad) * 2.5;

    let astronomy = AstronomyState {
        star_rotation,
        sun_dir,
        sun_size_ndc,
        moon_dir,
        moon_size_ndc,
        moon_phase,
        sun_declination,
        day_phase: time_scales.day_phase,
        current_year: time_scales.current_year,
    };

    astronomy
}
