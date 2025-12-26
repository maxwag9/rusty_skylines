use std::f32::consts::TAU;

pub struct AstronomyState {
    pub sun_dir: glam::Vec3,
    pub moon_dir: glam::Vec3,
    pub moon_phase: f32,
    pub sun_declination: f32,
    pub day_phase: f32,
    pub current_year: f32,
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
    pub fn from_game_time(total_game_time: f64) -> Self {
        let day_length: f32 = 960.0;
        let total_days: f32 = (total_game_time / day_length as f64) as f32;
        let day_phase = total_days % 1.0;
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
        let latitude = 48.0_f32.to_radians();
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

pub fn compute_astronomy(time_scales: &TimeScales) -> AstronomyState {
    let observer = ObserverParams::new(time_scales.day_angle);
    let (sun_dir, sun_declination) = compute_sun_direction(&observer, time_scales.year_phase);
    let moon_dir = compute_moon_direction(&observer, time_scales.total_days);
    let moon_phase = compute_moon_phase(&observer, time_scales.year_phase, time_scales.total_days);

    AstronomyState {
        sun_dir,
        moon_dir,
        moon_phase,
        sun_declination,
        day_phase: time_scales.day_phase,
        current_year: time_scales.current_year,
    }
}
