#![allow(dead_code)]

use crate::resources::DAYS_PER_YEAR;
use crate::world::satellites::{CELESTIAL_BODY_COUNT, CelestialBody, compute_celestial_bodies};
use core::default::Default;
use glam::{Mat3, Mat4, Vec3};
use std::f64::consts::TAU;

const MUNICH_LATITUDE: f64 = 48.1351;
const MUNICH_LONGITUDE: f64 = 11.5820;
const J2000: f64 = 2451545.0;
const GAME_EPOCH_JD: f64 = 2461397.0; // Somewhere in the middle of Winter, NOT: / 2026-04-01 00:00 UTC // 2026-04-01 00:00 UTC April April!
const SKY_OFFSET_HOURS: f64 = -2.0;

const SKY_DISC_SCALE: f64 = 2.5;
const ECLIPSE_COVERAGE_DISC_SCALE: f64 = SKY_DISC_SCALE;

pub struct Astronomy {
    pub star_rotation: Mat4,
    pub sun_dir: Vec3,
    pub sun_size_ndc: f64,
    pub moon_dir: Vec3,
    pub moon_size_ndc: f64,
    pub moon_phase: f64,
    pub sun_declination: f64,
    pub solar_eclipse_coverage: f32,
    pub day_phase: f64,
    pub current_year: f64,
    pub celestial_bodies: [CelestialBody; CELESTIAL_BODY_COUNT],
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
            solar_eclipse_coverage: 0.0,
            day_phase: 0.0,
            current_year: 0.0,
            celestial_bodies: [CelestialBody::default(); CELESTIAL_BODY_COUNT],
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
    pub fn from_game_time(total_game_time: f64, day_length: f64, always_day: bool) -> Self {
        const TROPICAL_YEAR_DAYS: f64 = 365.24219;

        let total_days = total_game_time / day_length;
        let normal_day_phase = total_days.rem_euclid(1.0);

        let day_phase = if always_day {
            0.5 + 0.05 * (total_days * TAU).sin()
        } else {
            normal_day_phase
        };

        let jd = GAME_EPOCH_JD + total_days * TROPICAL_YEAR_DAYS / DAYS_PER_YEAR;

        let day_angle = day_phase * TAU;

        let year_phase = (total_days / DAYS_PER_YEAR).rem_euclid(1.0);
        let year_angle = year_phase * TAU;

        let base_year = 2026.0;
        let current_year = base_year + total_days / DAYS_PER_YEAR;

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

        let obliquity = mean_obliquity_rad(jd);
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
    let lat = MUNICH_LATITUDE.to_radians();
    let lst = (compute_gmst(jd) + MUNICH_LONGITUDE.to_radians()).rem_euclid(TAU);

    let ra = (eq.y as f64).atan2(eq.x as f64);
    let dec = (eq.z as f64).asin();

    let h = lst - ra;

    let sin_alt = dec.sin() * lat.sin() + dec.cos() * lat.cos() * h.cos();
    let alt = sin_alt.asin();

    let x = -dec.cos() * h.sin(); // east
    let y = alt.sin(); // up
    let z = dec.sin() * lat.cos() - dec.cos() * lat.sin() * h.cos(); // north

    Vec3::new(x as f32, y as f32, z as f32).normalize()
}

pub fn compute_gmst(jd: f64) -> f64 {
    let jd_0h = (jd - 0.5).floor() + 0.5;
    let t = (jd_0h - J2000) / 36525.0;

    let gmst_0h = 24110.54841 + 8640184.812866 * t + 0.093104 * t * t - 6.2e-6 * t * t * t;

    let ut1_fraction = jd - jd_0h;
    let gmst_seconds = gmst_0h + 86400.0 * 1.00273790935 * ut1_fraction;

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

const MOON_LR_TERMS: &[([i8; 4], f64, f64)] = &[
    ([0, 0, 1, 0], 6288774.0, -20905355.0),
    ([2, 0, -1, 0], 1274027.0, -3699111.0),
    ([2, 0, 0, 0], 658314.0, -2955968.0),
    ([0, 0, 2, 0], 213618.0, -569925.0),
    ([0, 1, 0, 0], -185116.0, 48888.0),
    ([0, 0, 0, 2], -114332.0, -3149.0),
    ([2, 0, -2, 0], 58793.0, 246158.0),
    ([2, -1, -1, 0], 57066.0, -152138.0),
    ([2, 0, 1, 0], 53322.0, -170733.0),
    ([2, -1, 0, 0], 45758.0, -204586.0),
    ([0, 1, -1, 0], -40923.0, -129620.0),
    ([1, 0, 0, 0], -34720.0, 108743.0),
    ([0, 1, 1, 0], -30383.0, 104755.0),
    ([2, 0, 0, -2], 15327.0, 10321.0),
    ([0, 0, 1, 2], -12528.0, 0.0),
    ([0, 0, 1, -2], 10980.0, 79661.0),
    ([4, 0, -1, 0], 10675.0, -34782.0),
    ([0, 0, 3, 0], 10034.0, -23210.0),
    ([4, 0, -2, 0], 8548.0, -21636.0),
    ([2, 1, -1, 0], -7888.0, 24208.0),
    ([2, 1, 0, 0], -6766.0, 30824.0),
    ([1, 0, -1, 0], -5163.0, -8379.0),
    ([1, 1, 0, 0], 4987.0, -16675.0),
    ([2, -1, 1, 0], 4036.0, -12831.0),
    ([2, 0, 2, 0], 3994.0, -10445.0),
    ([4, 0, 0, 0], 3861.0, -11650.0),
    ([2, 0, -3, 0], 3665.0, 14403.0),
    ([0, 1, -2, 0], -2689.0, -7003.0),
    ([2, 0, -1, 2], -2602.0, 0.0),
    ([2, -1, -2, 0], 2390.0, 10056.0),
    ([1, 0, 1, 0], -2348.0, 6322.0),
    ([2, -2, 0, 0], 2236.0, -9884.0),
    ([0, 1, 2, 0], -2120.0, 5751.0),
    ([0, 2, 0, 0], -2069.0, 0.0),
    ([2, -2, -1, 0], 2048.0, -4950.0),
    ([2, 0, 1, -2], -1773.0, 4130.0),
    ([2, 0, 0, 2], -1595.0, 0.0),
    ([4, -1, -1, 0], 1215.0, -3958.0),
    ([0, 0, 2, 2], -1110.0, 0.0),
    ([3, 0, -1, 0], -892.0, 3258.0),
    ([2, 1, 1, 0], -810.0, 2616.0),
    ([4, -1, -2, 0], 759.0, -1897.0),
    ([0, 2, -1, 0], -713.0, -2117.0),
    ([2, 2, -1, 0], -700.0, 2354.0),
    ([2, 1, -2, 0], 691.0, 0.0),
    ([2, -1, 0, -2], 596.0, 0.0),
    ([4, 0, 1, 0], 549.0, -1423.0),
    ([0, 0, 4, 0], 537.0, -1117.0),
    ([4, -1, 0, 0], 520.0, -1571.0),
    ([1, 0, -2, 0], -487.0, -1739.0),
    ([2, 1, 0, -2], -399.0, 0.0),
    ([0, 0, 2, -2], -381.0, -4421.0),
    ([1, 1, 1, 0], 351.0, 0.0),
    ([3, 0, -2, 0], -340.0, 0.0),
    ([4, 0, -3, 0], 330.0, 0.0),
    ([2, -1, 2, 0], 327.0, 0.0),
    ([0, 2, 1, 0], -323.0, 1165.0),
    ([1, 1, -1, 0], 299.0, 0.0),
    ([2, 0, 3, 0], 294.0, 0.0),
    ([2, 0, -1, -2], 0.0, 8752.0),
];

const MOON_B_TERMS: &[([i8; 4], f64)] = &[
    ([0, 0, 0, 1], 5128122.0),
    ([0, 0, 1, 1], 280602.0),
    ([0, 0, 1, -1], 277693.0),
    ([2, 0, 0, -1], 173237.0),
    ([2, 0, -1, 1], 55413.0),
    ([2, 0, -1, -1], 46271.0),
    ([2, 0, 0, 1], 32573.0),
    ([0, 0, 2, 1], 17198.0),
    ([2, 0, 1, -1], 9266.0),
    ([0, 0, 2, -1], 8822.0),
    ([2, -1, 0, -1], 8216.0),
    ([2, 0, -2, -1], 4324.0),
    ([2, 0, 1, 1], 4200.0),
    ([2, 1, 0, -1], -3359.0),
    ([2, -1, -1, 1], 2463.0),
    ([2, -1, 0, 1], 2211.0),
    ([2, -1, -1, -1], 2065.0),
    ([0, 1, -1, -1], -1870.0),
    ([4, 0, -1, -1], 1828.0),
    ([0, 1, 0, 1], -1794.0),
    ([0, 0, 0, 3], -1749.0),
    ([0, 1, -1, 1], -1565.0),
    ([1, 0, 0, 1], -1491.0),
    ([0, 1, 1, 1], -1475.0),
    ([0, 1, 1, -1], -1410.0),
    ([0, 1, 0, -1], -1344.0),
    ([1, 0, 0, -1], -1335.0),
    ([0, 0, 3, 1], 1107.0),
    ([4, 0, 0, -1], 1021.0),
    ([4, 0, -1, 1], 833.0),
    ([0, 0, 1, -3], 777.0),
    ([4, 0, -2, 1], 671.0),
    ([2, 0, 0, -3], 607.0),
    ([2, 0, 2, -1], 596.0),
    ([2, -1, 1, -1], 491.0),
    ([2, 0, -2, 1], -451.0),
    ([0, 0, 3, -1], 439.0),
    ([2, 0, 2, 1], 422.0),
    ([2, 0, -3, -1], 421.0),
    ([2, 1, -1, 1], -366.0),
    ([2, 1, 0, 1], -351.0),
    ([4, 0, 0, 1], 331.0),
    ([2, -1, 1, 1], 315.0),
    ([2, -2, 0, -1], 302.0),
    ([0, 0, 1, 3], -283.0),
    ([2, 1, 1, -1], -229.0),
    ([1, 1, 0, -1], 223.0),
    ([1, 1, 0, 1], 223.0),
    ([0, 1, -2, -1], -220.0),
    ([2, 1, -1, -1], -220.0),
    ([1, 0, 1, 1], -185.0),
    ([2, -1, -2, -1], 181.0),
    ([0, 1, 2, 1], -177.0),
    ([4, 0, -2, -1], 176.0),
    ([4, -1, -1, -1], 166.0),
    ([1, 0, 1, -1], -164.0),
    ([4, 0, 1, -1], 132.0),
    ([1, 0, -1, -1], -119.0),
    ([4, -1, 0, -1], 115.0),
    ([2, -2, 0, 1], 107.0),
];

fn mean_obliquity_rad(jd: f64) -> f64 {
    let t = (jd - J2000) / 36525.0;

    (23.439291111 - 0.013004167 * t - 0.000000164 * t * t + 0.000000504 * t * t * t).to_radians()
}

pub fn compute_moon_equatorial_direction_and_distance(jd: f64) -> (Vec3, f64) {
    let t = (jd - J2000) / 36525.0;

    let l_prime_deg = (218.3164477
        + (481267.88123421 + (-0.0015786 + (1.0 / 538841.0 - t / 65194000.0) * t) * t) * t)
        .rem_euclid(360.0);

    let d_deg = (297.8501921
        + (445267.1114034 + (-0.0018819 + (1.0 / 545868.0 - t / 113065000.0) * t) * t) * t)
        .rem_euclid(360.0);

    let m_deg =
        (357.5291092 + (35999.0502909 + (-0.0001536 + t / 24490000.0) * t) * t).rem_euclid(360.0);

    let m_prime_deg = (134.9633964
        + (477198.8675055 + (0.0087414 + (1.0 / 69699.9 - t / 14712000.0) * t) * t) * t)
        .rem_euclid(360.0);

    let f_deg = (93.2720950
        + (483202.0175233 + (-0.0036539 + (-1.0 / 3526000.0 + t / 863310000.0) * t) * t) * t)
        .rem_euclid(360.0);

    let l_prime = l_prime_deg.to_radians();
    let d = d_deg.to_radians();
    let m = m_deg.to_radians();
    let m_prime = m_prime_deg.to_radians();
    let f = f_deg.to_radians();

    let a1 = (119.75 + 131.849 * t).rem_euclid(360.0).to_radians();
    let a2 = (53.09 + 479264.290 * t).rem_euclid(360.0).to_radians();
    let a3 = (313.45 + 481266.484 * t).rem_euclid(360.0).to_radians();

    let e = 1.0 - 0.002516 * t - 0.0000074 * t * t;
    let e2 = e * e;

    let arguments = [d, m, m_prime, f];

    let mut sigma_l = 0.0;
    let mut sigma_r = 0.0;

    for &(multipliers, longitude_coefficient, distance_coefficient) in MOON_LR_TERMS {
        let argument = multipliers[0] as f64 * arguments[0]
            + multipliers[1] as f64 * arguments[1]
            + multipliers[2] as f64 * arguments[2]
            + multipliers[3] as f64 * arguments[3];

        let eccentricity_factor = match multipliers[1].abs() {
            1 => e,
            2 => e2,
            _ => 1.0,
        };

        let (sin_argument, cos_argument) = argument.sin_cos();

        sigma_l += longitude_coefficient * eccentricity_factor * sin_argument;
        sigma_r += distance_coefficient * eccentricity_factor * cos_argument;
    }

    sigma_l += 3958.0 * a1.sin() + 1962.0 * (l_prime - f).sin() + 318.0 * a2.sin();

    let mut sigma_b = 0.0;

    for &(multipliers, coefficient) in MOON_B_TERMS {
        let argument = multipliers[0] as f64 * arguments[0]
            + multipliers[1] as f64 * arguments[1]
            + multipliers[2] as f64 * arguments[2]
            + multipliers[3] as f64 * arguments[3];

        let eccentricity_factor = match multipliers[1].abs() {
            1 => e,
            2 => e2,
            _ => 1.0,
        };

        sigma_b += coefficient * eccentricity_factor * argument.sin();
    }

    sigma_b += -2235.0 * l_prime.sin()
        + 382.0 * a3.sin()
        + 175.0 * (a1 - f).sin()
        + 175.0 * (a1 + f).sin()
        + 127.0 * (l_prime - m_prime).sin()
        - 115.0 * (l_prime + m_prime).sin();

    let longitude = (l_prime_deg + sigma_l / 1_000_000.0)
        .rem_euclid(360.0)
        .to_radians();

    let latitude = (sigma_b / 1_000_000.0).to_radians();
    let distance_km = 385000.56 + sigma_r / 1000.0;

    let cos_latitude = latitude.cos();

    let moon_ecliptic = Vec3::new(
        (cos_latitude * longitude.cos()) as f32,
        (cos_latitude * longitude.sin()) as f32,
        latitude.sin() as f32,
    );

    let moon_equatorial =
        (glam::Quat::from_rotation_x(mean_obliquity_rad(jd) as f32) * moon_ecliptic).normalize();

    (moon_equatorial, distance_km / 6378.137)
}

pub fn compute_moon_direction(observer: &ObserverParams, jd: f64) -> Vec3 {
    let (moon_eq, distance_earth_radii) = compute_moon_equatorial_direction_and_distance(jd);

    let moon_geocentric = equatorial_to_local(moon_eq, jd);

    let observer_position = Vec3::new(
        0.0,
        observer.latitude.cos() as f32,
        observer.latitude.sin() as f32,
    );

    (moon_geocentric * distance_earth_radii as f32 - observer_position).normalize()
}
pub fn equatorial_to_local_at_lst(eq: Vec3, lst: f64) -> Vec3 {
    let lat = MUNICH_LATITUDE.to_radians();

    let ra = (eq.y as f64).atan2(eq.x as f64);
    let dec = (eq.z as f64).clamp(-1.0, 1.0).asin();
    let h = lst - ra;

    let sin_alt = dec.sin() * lat.sin() + dec.cos() * lat.cos() * h.cos();

    let alt = sin_alt.asin();

    let x = -dec.cos() * h.sin();
    let y = alt.sin();
    let z = dec.sin() * lat.cos() - dec.cos() * lat.sin() * h.cos();

    Vec3::new(x as f32, y as f32, z as f32).normalize()
}
pub fn compute_moon_topocentric_distance_earth_radii(
    geocentric_distance_earth_radii: f64,
    altitude_rad: f64,
) -> f64 {
    let r = geocentric_distance_earth_radii;
    (r * r + 1.0 - 2.0 * r * altitude_rad.sin()).sqrt()
}
pub fn compute_moon_angular_radius_at_lst(jd: f64, lst: f64) -> f64 {
    let (moon_eq, r_geo) = compute_moon_equatorial_direction_and_distance(jd);
    let moon_local = equatorial_to_local_at_lst(moon_eq, lst);

    let altitude_rad = (moon_local.y as f64).clamp(-1.0, 1.0).asin();
    let r_topo = compute_moon_topocentric_distance_earth_radii(r_geo, altitude_rad);

    let earth_radius_km = 6378.137_f64;
    let moon_radius_km = 1737.4_f64;

    (moon_radius_km / (r_topo * earth_radius_km)).atan()
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
pub fn compute_moon_direction_at_lst(jd: f64, lst: f64) -> Vec3 {
    let (moon_eq, _) = compute_moon_equatorial_direction_and_distance(jd);
    equatorial_to_local_at_lst(moon_eq, lst)
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
pub fn compute_star_rotation_at_lst(lst: f64) -> Mat4 {
    let sidereal_rot = Mat3::from_rotation_y(-lst as f32);
    let lat = MUNICH_LATITUDE.to_radians();
    let latitude_rot = Mat3::from_rotation_x((lat - std::f64::consts::FRAC_PI_2) as f32);

    Mat4::from_mat3(latitude_rot * sidereal_rot)
}

pub fn compute_star_rotation(jd: f64) -> Mat4 {
    let lst = (compute_gmst(jd) + MUNICH_LONGITUDE.to_radians()).rem_euclid(TAU);

    compute_star_rotation_at_lst(lst)
}

pub fn compute_sun_equatorial_direction(jd: f64) -> Vec3 {
    let n = jd - J2000;

    let l_deg = (280.460 + 0.9856474 * n).rem_euclid(360.0);
    let g_deg = (357.528 + 0.9856003 * n).rem_euclid(360.0);

    let g = g_deg.to_radians();
    let lambda_deg = l_deg + 1.915 * g.sin() + 0.020 * (2.0 * g).sin();
    let lambda = lambda_deg.to_radians();

    let epsilon = mean_obliquity_rad(jd);

    let x = lambda.cos();
    let y = epsilon.cos() * lambda.sin();
    let z = epsilon.sin() * lambda.sin();

    Vec3::new(x as f32, y as f32, z as f32).normalize()
}

pub fn compute_sun_direction(observer: &ObserverParams, jd: f64) -> (Vec3, f64) {
    let sun_eq = compute_sun_equatorial_direction(jd);
    let sun_declination = sun_eq.z.asin().to_degrees() as f64;
    let sun_dir = equatorial_to_local(sun_eq, jd);
    (sun_dir, sun_declination)
}

pub fn compute_solar_eclipse_coverage(
    sun_dir: Vec3,
    moon_dir: Vec3,
    sun_radius_rad: f64,
    moon_radius_rad: f64,
) -> f32 {
    let sun_radius = (sun_radius_rad.max(0.0).tan() * ECLIPSE_COVERAGE_DISC_SCALE).atan();
    let moon_radius = (moon_radius_rad.max(0.0).tan() * ECLIPSE_COVERAGE_DISC_SCALE).atan();

    if sun_radius <= 0.0 || moon_radius <= 0.0 {
        return 0.0;
    }

    let sun_altitude = (sun_dir.y as f64).clamp(-1.0, 1.0).asin();
    let moon_altitude = (moon_dir.y as f64).clamp(-1.0, 1.0).asin();

    if sun_altitude + sun_radius <= 0.0 || moon_altitude + moon_radius <= 0.0 {
        return 0.0;
    }

    let sx = sun_dir.x as f64;
    let sy = sun_dir.y as f64;
    let sz = sun_dir.z as f64;

    let mx = moon_dir.x as f64;
    let my = moon_dir.y as f64;
    let mz = moon_dir.z as f64;

    let sun_len = (sx * sx + sy * sy + sz * sz).sqrt();
    let moon_len = (mx * mx + my * my + mz * mz).sqrt();

    if sun_len <= 1e-12 || moon_len <= 1e-12 {
        return 0.0;
    }

    let sx = sx / sun_len;
    let sy = sy / sun_len;
    let sz = sz / sun_len;

    let mx = mx / moon_len;
    let my = my / moon_len;
    let mz = mz / moon_len;

    let dot = (sx * mx + sy * my + sz * mz).clamp(-1.0, 1.0);

    let cx = sy * mz - sz * my;
    let cy = sz * mx - sx * mz;
    let cz = sx * my - sy * mx;

    let cross_len = (cx * cx + cy * cy + cz * cz).sqrt();
    let separation = cross_len.atan2(dot);

    if separation >= sun_radius + moon_radius {
        return 0.0;
    }

    if separation <= (sun_radius - moon_radius).abs() {
        return if moon_radius >= sun_radius {
            1.0
        } else {
            ((moon_radius / sun_radius).powi(2)).clamp(0.0, 1.0) as f32
        };
    }

    let d2 = separation * separation;
    let sun_r2 = sun_radius * sun_radius;
    let moon_r2 = moon_radius * moon_radius;

    let sun_arg = ((d2 + sun_r2 - moon_r2) / (2.0 * separation * sun_radius)).clamp(-1.0, 1.0);

    let moon_arg = ((d2 + moon_r2 - sun_r2) / (2.0 * separation * moon_radius)).clamp(-1.0, 1.0);

    let radicand = ((-separation + sun_radius + moon_radius)
        * (separation + sun_radius - moon_radius)
        * (separation - sun_radius + moon_radius)
        * (separation + sun_radius + moon_radius))
        .max(0.0);

    let overlap_area = sun_r2 * sun_arg.acos() + moon_r2 * moon_arg.acos() - 0.5 * radicand.sqrt();

    (overlap_area / (std::f64::consts::PI * sun_r2)).clamp(0.0, 1.0) as f32
}

/// Computes `AstronomyState` plus Sun/Moon disc radii in the exact units expected by your shader.
///
/// The returned `sun_size_ndc` and `moon_size_ndc` should be written into `SkyUniform.sun_size`
/// and `SkyUniform.moon_size`.
pub fn compute_astronomy(time_scales: &TimeScales, proj: Mat4) -> Astronomy {
    let jd = time_scales.jd;

    let epoch_midnight_jd = (GAME_EPOCH_JD - 0.5).floor() + 0.5;

    let rotation_jd = epoch_midnight_jd
        + time_scales.total_days.floor()
        + time_scales.day_phase
        + SKY_OFFSET_HOURS / 24.0;

    let sun_eq = compute_sun_equatorial_direction(jd);
    let clock_sun_eq = compute_sun_equatorial_direction(rotation_jd);

    let sun_ra = (sun_eq.y as f64).atan2(sun_eq.x as f64);
    let clock_sun_ra = (clock_sun_eq.y as f64).atan2(clock_sun_eq.x as f64);

    let lst = (compute_gmst(rotation_jd) + MUNICH_LONGITUDE.to_radians() + sun_ra - clock_sun_ra)
        .rem_euclid(TAU);

    let sun_declination = (sun_eq.z as f64).asin().to_degrees();
    let sun_dir = equatorial_to_local_at_lst(sun_eq, lst);
    let moon_dir = compute_moon_direction_at_lst(jd, lst);

    let moon_phase = compute_moon_phase(jd);
    let star_rotation = compute_star_rotation_at_lst(lst);

    let sun_ang_rad = compute_sun_angular_radius_rad(jd);
    let moon_ang_rad = compute_moon_angular_radius_at_lst(jd, lst);

    let solar_eclipse_coverage =
        compute_solar_eclipse_coverage(sun_dir, moon_dir, sun_ang_rad, moon_ang_rad);

    let sun_size_ndc = angular_radius_rad_to_ndc_radius(proj, sun_ang_rad) * SKY_DISC_SCALE;

    let moon_size_ndc = angular_radius_rad_to_ndc_radius(proj, moon_ang_rad) * SKY_DISC_SCALE;
    let celestial_bodies = compute_celestial_bodies(jd, lst, proj);
    Astronomy {
        star_rotation,
        sun_dir,
        sun_size_ndc,
        moon_dir,
        moon_size_ndc,
        moon_phase,
        sun_declination,
        solar_eclipse_coverage,
        day_phase: time_scales.day_phase,
        current_year: time_scales.current_year,
        celestial_bodies,
    }
}
