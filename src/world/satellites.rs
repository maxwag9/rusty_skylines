use crate::world::astronomy::{angular_radius_rad_to_ndc_radius, equatorial_to_local_at_lst};
use glam::{DVec3, Mat4, Vec3};
use std::f64::consts::TAU;

pub const PLANET_COUNT: usize = 8;
pub const CELESTIAL_BODY_COUNT: usize = 20;

pub const BODY_MARS: usize = 0;
pub const BODY_PLUTO: usize = 1;
pub const BODY_NEPTUNE: usize = 2;
pub const BODY_URANUS: usize = 3;
pub const BODY_JUPITER: usize = 4;
pub const BODY_VENUS: usize = 5;
pub const BODY_MERCURY: usize = 6;
pub const BODY_SATURN: usize = 7;

pub const BODY_IO: usize = 8;
pub const BODY_EUROPA: usize = 9;
pub const BODY_GANYMEDE: usize = 10;
pub const BODY_CALLISTO: usize = 11;
pub const BODY_TITAN: usize = 12;
pub const BODY_RHEA: usize = 13;
pub const BODY_TETHYS: usize = 14;
pub const BODY_DIONE: usize = 15;
pub const BODY_IAPETUS: usize = 16;
pub const BODY_ENCELADUS: usize = 17;
pub const BODY_TRITON: usize = 18;
pub const BODY_CHARON: usize = 19;

const AU_KM: f64 = 149_597_870.7;
const J2000_JD: f64 = 2451545.0;
const OBLIQUITY_J2000_RAD: f64 = 0.40909280422232897;

#[derive(Clone, Copy, Default)]
pub struct CelestialBody {
    pub direction: Vec3,
    pub angular_radius_ndc: f64,
    pub distance_km: f64,
    pub illumination: f64,
    pub radius_km: f64,
    pub parent_index: i32,
    pub parent_relative_position_ecliptic_km: Vec3,
    pub parent_distance_km: f64,
}

#[derive(Clone, Copy)]
struct PlanetElements {
    a: f64,
    da: f64,
    e: f64,
    de: f64,
    i: f64,
    di: f64,
    l: f64,
    dl: f64,
    p: f64,
    dp: f64,
    n: f64,
    dn: f64,
    b: f64,
    c: f64,
    s: f64,
    f: f64,
}

const MERCURY_ELEMENTS: PlanetElements = PlanetElements {
    a: 0.38709843,
    da: 0.0,
    e: 0.20563661,
    de: 0.00002123,
    i: 7.00559432,
    di: -0.00590158,
    l: 252.25166724,
    dl: 149472.67486623,
    p: 77.45771895,
    dp: 0.15940013,
    n: 48.33961819,
    dn: -0.12214182,
    b: 0.0,
    c: 0.0,
    s: 0.0,
    f: 0.0,
};

const VENUS_ELEMENTS: PlanetElements = PlanetElements {
    a: 0.72332102,
    da: -0.00000026,
    e: 0.00676399,
    de: -0.00005107,
    i: 3.39777545,
    di: 0.00043494,
    l: 181.97970850,
    dl: 58517.81560260,
    p: 131.76755713,
    dp: 0.05679648,
    n: 76.67261496,
    dn: -0.27274174,
    b: 0.0,
    c: 0.0,
    s: 0.0,
    f: 0.0,
};

const EARTH_ELEMENTS: PlanetElements = PlanetElements {
    a: 1.00000018,
    da: -0.00000003,
    e: 0.01673163,
    de: -0.00003661,
    i: -0.00054346,
    di: -0.01337178,
    l: 100.46691572,
    dl: 35999.37306329,
    p: 102.93005885,
    dp: 0.31795260,
    n: -5.11260389,
    dn: -0.24123856,
    b: 0.0,
    c: 0.0,
    s: 0.0,
    f: 0.0,
};

const MARS_ELEMENTS: PlanetElements = PlanetElements {
    a: 1.52371243,
    da: 0.00000097,
    e: 0.09336511,
    de: 0.00009149,
    i: 1.85181869,
    di: -0.00724757,
    l: -4.56813164,
    dl: 19140.29934243,
    p: -23.91744784,
    dp: 0.45223625,
    n: 49.71320984,
    dn: -0.26852431,
    b: 0.0,
    c: 0.0,
    s: 0.0,
    f: 0.0,
};

const JUPITER_ELEMENTS: PlanetElements = PlanetElements {
    a: 5.20248019,
    da: -0.00002864,
    e: 0.04853590,
    de: 0.00018026,
    i: 1.29861416,
    di: -0.00322699,
    l: 34.33479152,
    dl: 3034.90371757,
    p: 14.27495244,
    dp: 0.18199196,
    n: 100.29282654,
    dn: 0.13024619,
    b: -0.00012452,
    c: 0.06064060,
    s: -0.35635438,
    f: 38.35125000,
};

const SATURN_ELEMENTS: PlanetElements = PlanetElements {
    a: 9.54149883,
    da: -0.00003065,
    e: 0.05550825,
    de: -0.00032044,
    i: 2.49424102,
    di: 0.00451969,
    l: 50.07571329,
    dl: 1222.11494724,
    p: 92.86136063,
    dp: 0.54179478,
    n: 113.63998702,
    dn: -0.25015002,
    b: 0.00025899,
    c: -0.13434469,
    s: 0.87320147,
    f: 38.35125000,
};

const URANUS_ELEMENTS: PlanetElements = PlanetElements {
    a: 19.18797948,
    da: -0.00020455,
    e: 0.04685740,
    de: -0.00001550,
    i: 0.77298127,
    di: -0.00180155,
    l: 314.20276625,
    dl: 428.49512595,
    p: 172.43404441,
    dp: 0.09266985,
    n: 73.96250215,
    dn: 0.05739699,
    b: 0.00058331,
    c: -0.97731848,
    s: 0.17689245,
    f: 7.67025000,
};

const NEPTUNE_ELEMENTS: PlanetElements = PlanetElements {
    a: 30.06952752,
    da: 0.00006447,
    e: 0.00895439,
    de: 0.00000818,
    i: 1.77005520,
    di: 0.00022400,
    l: 304.22289287,
    dl: 218.46515314,
    p: 46.68158724,
    dp: 0.01009938,
    n: 131.78635853,
    dn: -0.00606302,
    b: -0.00041348,
    c: 0.68346318,
    s: -0.10162547,
    f: 7.67025000,
};

const PLUTO_ELEMENTS: PlanetElements = PlanetElements {
    a: 39.48168677,
    da: -0.00076912,
    e: 0.24880766,
    de: 0.00006465,
    i: 17.14175,
    di: 0.00000501,
    l: 238.92903833,
    dl: 145.20780515,
    p: 224.06891629,
    dp: -0.04062942,
    n: 110.30393684,
    dn: -0.01183482,
    b: 0.0,
    c: 0.0,
    s: 0.0,
    f: 0.0,
};

const PLANET_ELEMENTS: [PlanetElements; PLANET_COUNT] = [
    MARS_ELEMENTS,
    PLUTO_ELEMENTS,
    NEPTUNE_ELEMENTS,
    URANUS_ELEMENTS,
    JUPITER_ELEMENTS,
    VENUS_ELEMENTS,
    MERCURY_ELEMENTS,
    SATURN_ELEMENTS,
];

const PLANET_RADII_KM: [f64; PLANET_COUNT] = [
    3389.5, 1188.3, 24622.0, 25362.0, 69911.0, 6051.8, 2439.7, 58232.0,
];

#[derive(Clone, Copy)]
struct SatelliteOrbit {
    parent_index: usize,
    a_km: f64,
    e: f64,
    omega_deg: f64,
    mean_anomaly_deg: f64,
    inclination_deg: f64,
    node_deg: f64,
    period_days: f64,
    apsidal_period_years: f64,
    nodal_period_years: f64,
    radius_km: f64,
}

const SATELLITE_ORBITS: [SatelliteOrbit; 12] = [
    SatelliteOrbit {
        parent_index: BODY_JUPITER,
        a_km: 421800.0,
        e: 0.004,
        omega_deg: 49.1,
        mean_anomaly_deg: 330.9,
        inclination_deg: 0.0,
        node_deg: 0.0,
        period_days: 1.762732,
        apsidal_period_years: 1.333,
        nodal_period_years: 0.0,
        radius_km: 1821.6,
    },
    SatelliteOrbit {
        parent_index: BODY_JUPITER,
        a_km: 671100.0,
        e: 0.009,
        omega_deg: 45.0,
        mean_anomaly_deg: 345.4,
        inclination_deg: 0.5,
        node_deg: 184.0,
        period_days: 3.525463,
        apsidal_period_years: 1.394,
        nodal_period_years: 30.202,
        radius_km: 1560.8,
    },
    SatelliteOrbit {
        parent_index: BODY_JUPITER,
        a_km: 1070400.0,
        e: 0.001,
        omega_deg: 198.3,
        mean_anomaly_deg: 324.8,
        inclination_deg: 0.2,
        node_deg: 58.5,
        period_days: 7.155588,
        apsidal_period_years: 68.301,
        nodal_period_years: 137.812,
        radius_km: 2634.1,
    },
    SatelliteOrbit {
        parent_index: BODY_JUPITER,
        a_km: 1882700.0,
        e: 0.007,
        omega_deg: 43.8,
        mean_anomaly_deg: 87.4,
        inclination_deg: 0.3,
        node_deg: 309.1,
        period_days: 16.690440,
        apsidal_period_years: 277.921,
        nodal_period_years: 577.264,
        radius_km: 2410.3,
    },
    SatelliteOrbit {
        parent_index: BODY_SATURN,
        a_km: 1221900.0,
        e: 0.029,
        omega_deg: 78.3,
        mean_anomaly_deg: 11.7,
        inclination_deg: 0.3,
        node_deg: 78.6,
        period_days: 15.945448,
        apsidal_period_years: 346.680,
        nodal_period_years: 687.370,
        radius_km: 2574.76,
    },
    SatelliteOrbit {
        parent_index: BODY_SATURN,
        a_km: 527200.0,
        e: 0.001,
        omega_deg: 44.3,
        mean_anomaly_deg: 31.5,
        inclination_deg: 0.3,
        node_deg: 133.7,
        period_days: 4.517503,
        apsidal_period_years: 33.939,
        nodal_period_years: 35.775,
        radius_km: 763.8,
    },
    SatelliteOrbit {
        parent_index: BODY_SATURN,
        a_km: 295000.0,
        e: 0.001,
        omega_deg: 335.3,
        mean_anomaly_deg: 0.0,
        inclination_deg: 1.1,
        node_deg: 273.0,
        period_days: 1.887802,
        apsidal_period_years: 0.005,
        nodal_period_years: 4.982,
        radius_km: 531.1,
    },
    SatelliteOrbit {
        parent_index: BODY_SATURN,
        a_km: 377700.0,
        e: 0.002,
        omega_deg: 116.0,
        mean_anomaly_deg: 212.0,
        inclination_deg: 0.0,
        node_deg: 0.0,
        period_days: 2.736916,
        apsidal_period_years: 11.698,
        nodal_period_years: 0.0,
        radius_km: 561.4,
    },
    SatelliteOrbit {
        parent_index: BODY_SATURN,
        a_km: 3561700.0,
        e: 0.028,
        omega_deg: 254.5,
        mean_anomaly_deg: 74.8,
        inclination_deg: 7.6,
        node_deg: 86.5,
        period_days: 79.331002,
        apsidal_period_years: 1662.900,
        nodal_period_years: 3130.302,
        radius_km: 734.5,
    },
    SatelliteOrbit {
        parent_index: BODY_SATURN,
        a_km: 238400.0,
        e: 0.005,
        omega_deg: 119.5,
        mean_anomaly_deg: 57.0,
        inclination_deg: 0.0,
        node_deg: 0.0,
        period_days: 1.370218,
        apsidal_period_years: 2.916,
        nodal_period_years: 0.0,
        radius_km: 252.1,
    },
    SatelliteOrbit {
        parent_index: BODY_NEPTUNE,
        a_km: 354800.0,
        e: 0.0,
        omega_deg: 0.0,
        mean_anomaly_deg: 63.0,
        inclination_deg: 157.3,
        node_deg: 177.7,
        period_days: 5.8792,
        apsidal_period_years: 675.2,
        nodal_period_years: 299.4,
        radius_km: 1353.4,
    },
    SatelliteOrbit {
        parent_index: BODY_PLUTO,
        a_km: 19600.0,
        e: 0.0,
        omega_deg: 0.0,
        mean_anomaly_deg: 304.1,
        inclination_deg: 0.0,
        node_deg: 0.0,
        period_days: 6.387222,
        apsidal_period_years: 0.0,
        nodal_period_years: 0.0,
        radius_km: 606.0,
    },
];

fn solve_kepler(mean_anomaly: f64, eccentricity: f64) -> f64 {
    let m = mean_anomaly.rem_euclid(TAU);
    let mut eccentric_anomaly = m;

    for _ in 0..12 {
        let denominator = 1.0 - eccentricity * eccentric_anomaly.cos();
        let delta = (eccentric_anomaly - eccentricity * eccentric_anomaly.sin() - m) / denominator;

        eccentric_anomaly -= delta;

        if delta.abs() < 1e-13 {
            break;
        }
    }

    eccentric_anomaly
}

fn planet_heliocentric_position_au(jd: f64, elements: PlanetElements) -> DVec3 {
    let t = (jd - J2000_JD) / 36525.0;

    let a = elements.a + elements.da * t;
    let e = (elements.e + elements.de * t).clamp(0.0, 0.99);
    let inclination = (elements.i + elements.di * t).to_radians();
    let mean_longitude = elements.l + elements.dl * t;
    let longitude_perihelion = elements.p + elements.dp * t;
    let longitude_node = elements.n + elements.dn * t;

    let ft = (elements.f * t).to_radians();

    let mean_anomaly = (mean_longitude - longitude_perihelion
        + elements.b * t * t
        + elements.c * ft.cos()
        + elements.s * ft.sin())
    .to_radians()
    .rem_euclid(TAU);

    let eccentric_anomaly = solve_kepler(mean_anomaly, e);
    let x_orbit = a * (eccentric_anomaly.cos() - e);
    let y_orbit = a * (1.0 - e * e).sqrt() * eccentric_anomaly.sin();

    let omega = (longitude_perihelion - longitude_node).to_radians();
    let node = longitude_node.to_radians();

    let cw = omega.cos();
    let sw = omega.sin();
    let cn = node.cos();
    let sn = node.sin();
    let ci = inclination.cos();
    let si = inclination.sin();

    DVec3::new(
        (cw * cn - sw * sn * ci) * x_orbit + (-sw * cn - cw * sn * ci) * y_orbit,
        (cw * sn + sw * cn * ci) * x_orbit + (-sw * sn + cw * cn * ci) * y_orbit,
        sw * si * x_orbit + cw * si * y_orbit,
    )
}

fn ecliptic_to_equatorial_dvec(v: DVec3) -> DVec3 {
    let c = OBLIQUITY_J2000_RAD.cos();
    let s = OBLIQUITY_J2000_RAD.sin();

    DVec3::new(v.x, c * v.y - s * v.z, s * v.y + c * v.z)
}

fn equatorial_to_ecliptic_dvec(v: DVec3) -> DVec3 {
    let c = OBLIQUITY_J2000_RAD.cos();
    let s = OBLIQUITY_J2000_RAD.sin();

    DVec3::new(v.x, c * v.y + s * v.z, -s * v.y + c * v.z)
}

fn ecliptic_to_local_direction(v: DVec3, lst: f64) -> Vec3 {
    let eq = ecliptic_to_equatorial_dvec(v).normalize();

    equatorial_to_local_at_lst(Vec3::new(eq.x as f32, eq.y as f32, eq.z as f32), lst)
}

fn illuminated_fraction(body_heliocentric_km: DVec3, observer_to_body_km: DVec3) -> f64 {
    let body_to_sun = -body_heliocentric_km;
    let body_to_observer = -observer_to_body_km;

    ((1.0 + body_to_sun.normalize().dot(body_to_observer.normalize())) * 0.5).clamp(0.0, 1.0)
}

fn planet_pole_equatorial(parent_index: usize) -> (f64, f64) {
    match parent_index {
        BODY_JUPITER => (268.056595, 64.495303),
        BODY_PLUTO => (132.993, -6.163),
        BODY_NEPTUNE => (299.36, 43.46),
        BODY_SATURN => (40.589, 83.537),
        _ => (0.0, 90.0),
    }
}

fn satellite_position_ecliptic_km(jd: f64, orbit: SatelliteOrbit) -> DVec3 {
    let days = jd - J2000_JD;

    let mean_anomaly = orbit.mean_anomaly_deg.to_radians() + TAU * days / orbit.period_days;

    let omega = orbit.omega_deg.to_radians()
        + if orbit.apsidal_period_years > 0.0 {
            TAU * days / (orbit.apsidal_period_years * 365.25)
        } else {
            0.0
        };

    let node = orbit.node_deg.to_radians()
        - if orbit.nodal_period_years > 0.0 {
            TAU * days / (orbit.nodal_period_years * 365.25)
        } else {
            0.0
        };

    let inclination = orbit.inclination_deg.to_radians();
    let eccentric_anomaly = solve_kepler(mean_anomaly.rem_euclid(TAU), orbit.e);

    let x_orbit = orbit.a_km * (eccentric_anomaly.cos() - orbit.e);
    let y_orbit = orbit.a_km * (1.0 - orbit.e * orbit.e).sqrt() * eccentric_anomaly.sin();

    let cw = omega.cos();
    let sw = omega.sin();
    let cn = node.cos();
    let sn = node.sin();
    let ci = inclination.cos();
    let si = inclination.sin();

    let x = (cw * cn - sw * sn * ci) * x_orbit + (-sw * cn - cw * sn * ci) * y_orbit;

    let y = (cw * sn + sw * cn * ci) * x_orbit + (-sw * sn + cw * cn * ci) * y_orbit;

    let z = sw * si * x_orbit + cw * si * y_orbit;

    let (pole_ra_deg, pole_dec_deg) = planet_pole_equatorial(orbit.parent_index);

    let ra = pole_ra_deg.to_radians();
    let dec = pole_dec_deg.to_radians();

    let pole = DVec3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin());

    let frame_x = DVec3::new(-pole.y, pole.x, 0.0).normalize();
    let frame_y = pole.cross(frame_x).normalize();

    let relative_equatorial_km = frame_x * x + frame_y * y + pole * z;

    equatorial_to_ecliptic_dvec(relative_equatorial_km)
}

pub fn compute_celestial_bodies(
    jd: f64,
    lst: f64,
    proj: Mat4,
) -> [CelestialBody; CELESTIAL_BODY_COUNT] {
    let earth_heliocentric_km = planet_heliocentric_position_au(jd, EARTH_ELEMENTS) * AU_KM;

    let planet_heliocentric_km: [DVec3; PLANET_COUNT] =
        std::array::from_fn(|i| planet_heliocentric_position_au(jd, PLANET_ELEMENTS[i]) * AU_KM);

    let mut bodies = [CelestialBody::default(); CELESTIAL_BODY_COUNT];

    for index in 0..PLANET_COUNT {
        let body_heliocentric_km = planet_heliocentric_km[index];
        let observer_to_body_km = body_heliocentric_km - earth_heliocentric_km;

        let distance_km = observer_to_body_km.length();
        let angular_radius_rad = (PLANET_RADII_KM[index] / distance_km).atan();

        bodies[index] = CelestialBody {
            direction: ecliptic_to_local_direction(observer_to_body_km, lst),
            angular_radius_ndc: angular_radius_rad_to_ndc_radius(proj, angular_radius_rad),
            distance_km,
            illumination: illuminated_fraction(body_heliocentric_km, observer_to_body_km),
            radius_km: PLANET_RADII_KM[index],
            parent_index: -1,
            parent_relative_position_ecliptic_km: Vec3::ZERO,
            parent_distance_km: 0.0,
        };
    }

    for (moon_offset, orbit) in SATELLITE_ORBITS.iter().enumerate() {
        let index = PLANET_COUNT + moon_offset;

        let parent_heliocentric_km = planet_heliocentric_km[orbit.parent_index];

        let parent_relative_position_ecliptic_km = satellite_position_ecliptic_km(jd, *orbit);

        let body_heliocentric_km = parent_heliocentric_km + parent_relative_position_ecliptic_km;

        let observer_to_body_km = body_heliocentric_km - earth_heliocentric_km;

        let distance_km = observer_to_body_km.length();
        let angular_radius_rad = (orbit.radius_km / distance_km).atan();

        bodies[index] = CelestialBody {
            direction: ecliptic_to_local_direction(observer_to_body_km, lst),
            angular_radius_ndc: angular_radius_rad_to_ndc_radius(proj, angular_radius_rad),
            distance_km,
            illumination: illuminated_fraction(body_heliocentric_km, observer_to_body_km),
            radius_km: orbit.radius_km,
            parent_index: orbit.parent_index as i32,
            parent_relative_position_ecliptic_km: Vec3::new(
                parent_relative_position_ecliptic_km.x as f32,
                parent_relative_position_ecliptic_km.y as f32,
                parent_relative_position_ecliptic_km.z as f32,
            ),
            parent_distance_km: parent_relative_position_ecliptic_km.length(),
        };
    }

    bodies
}
