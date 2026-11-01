#include "includes/uniforms.wgsl"
#include "includes/sky_uniforms.wgsl"

@group(1) @binding(0) var<uniform> u: Uniforms;

@group(1) @binding(1) var<uniform> sky: SkyUniform;

struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) dir: vec3<f32>,
    @location(1) ndc: vec2<f32>,
    @location(2) prev_clip: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
    let clip_xy = vec2<f32>(
        f32((vid << 1u) & 2u) * 2.0 - 1.0,
        f32(vid & 2u) * 2.0 - 1.0
    );
    let ray_clip = vec4<f32>(clip_xy, 1.0, 1.0);
    let view_h = u.inv_proj * ray_clip;
    let view_dir = view_h.xyz / view_h.w;
    let world_dir = (u.inv_view * vec4<f32>(view_dir, 0.0)).xyz;

    let sky_depth = select(
        1.0,
        0.0,
        u.reversed_depth_z != 0u
    );

    var out: VSOut;
    out.clip = vec4<f32>(clip_xy, sky_depth, 1.0);
    out.dir = world_dir;
    out.ndc = clip_xy;
    out.prev_clip = u.prev_view_proj * vec4<f32>(world_dir, 0.0);

    return out;
}

struct FragOut {
    @location(0) color: vec4<f32>,
    @location(2) motion: vec2<f32>,
};

fn circle_overlap_fraction(sun_r: f32, moon_r: f32, d: f32) -> f32 {
    if (sun_r <= 0.000001 || moon_r <= 0.000001) {
        return 0.0;
    }

    if (d >= sun_r + moon_r) {
        return 0.0;
    }

    if (d <= abs(sun_r - moon_r)) {
        return clamp((moon_r * moon_r) / (sun_r * sun_r), 0.0, 1.0);
    }

    let d2 = d * d;
    let sun_r2 = sun_r * sun_r;
    let moon_r2 = moon_r * moon_r;

    let sun_arg = clamp(
        (d2 + sun_r2 - moon_r2) / (2.0 * d * sun_r),
        -1.0,
        1.0
    );

    let moon_arg = clamp(
        (d2 + moon_r2 - sun_r2) / (2.0 * d * moon_r),
        -1.0,
        1.0
    );

    let radicand = max(
        (-d + sun_r + moon_r)
        * (d + sun_r - moon_r)
        * (d - sun_r + moon_r)
        * (d + sun_r + moon_r),
        0.0
    );

    let overlap_area =
        sun_r2 * acos(sun_arg)
        + moon_r2 * acos(moon_arg)
        - 0.5 * sqrt(radicand);

    return clamp(
        overlap_area / (3.141592653589793 * sun_r2),
        0.0,
        1.0
    );
}

fn solar_spot(
    uv: vec2<f32>,
    center: vec2<f32>,
    radii: vec2<f32>
) -> f32 {
    let p = (uv - center) / radii;
    let d = length(p);

    let penumbra = 1.0 - smoothstep(0.48, 1.0, d);
    let umbra = 1.0 - smoothstep(0.18, 0.46, d);

    return saturate(penumbra * 0.58 + umbra * 0.42);
}

fn solar_corona(
    uv: vec2<f32>,
    intensity: f32,
    eclipse_coverage: f32
) -> vec3<f32> {
    let r = length(uv);

    if (r < 0.88 || r >= 24.0) {
        return vec3<f32>(0.0);
    }

    let angle = atan2(uv.y, uv.x);
    let radial_distance = max(r - 1.0, 0.0);

    let ray_a = pow(
        max(cos(angle * 4.0 + 0.27), 0.0),
        10.0
    );

    let ray_b = pow(
        max(cos(angle * 7.0 - 0.81), 0.0),
        18.0
    );

    let ray_c = pow(
        max(cos(angle * 11.0 + 1.43), 0.0),
        30.0
    );

    let ray_d = pow(
        max(cos(angle * 3.0 - 1.17), 0.0),
        8.0
    );

    let streamers =
        0.20
        + ray_a * 0.60
        + ray_b * 0.42
        + ray_c * 0.28
        + ray_d * 0.18;

    let inner_corona =
        0.48
        * exp(-radial_distance * 1.65)
        * smoothstep(0.84, 1.03, r);

    let outer_corona =
        0.72
        * pow(max(r, 1.0), -1.45)
        * exp(-radial_distance * 0.08)
        * streamers;

    let angular_structure =
        0.86
        + 0.14 * cos(angle * 2.0 + r * 0.48);

    let corona_visibility = mix(
        0.075,
        1.0,
        smoothstep(0.025, 0.82, saturate(eclipse_coverage))
    );

    let corona_energy =
        (inner_corona + outer_corona)
        * angular_structure
        * corona_visibility
        * max(intensity, 0.0)
        * 1.8;

    let color_transition = smoothstep(1.0, 8.0, r);

    let inner_color = vec3<f32>(1.35, 1.02, 0.76);
    let outer_color = vec3<f32>(0.64, 0.78, 1.05);

    let corona_color = mix(
        inner_color,
        outer_color,
        color_transition
    );

    let streamer_brightness = 0.88 + streamers * 0.48;

    return corona_color * corona_energy * streamer_brightness;
}

// =============================================================================
//  PLANETS, MOONS, RINGS
//  Body indices: 0 Mars 1 Pluto 2 Neptune 3 Uranus 4 Jupiter 5 Venus
//  6 Mercury 7 Saturn | 8 Io 9 Europa 10 Ganymede 11 Callisto 12 Titan
//  13 Rhea 14 Tethys 15 Dione 16 Iapetus 17 Enceladus 18 Triton 19 Charon
//  sky._pad2 may carry (jd - 2451545.0) to spin the planets (0 = frozen).
// =============================================================================
const PL_PI: f32 = 3.14159265358979;
const PL_TAU: f32 = 6.28318530717959;
const PL_OBLIQ_COS: f32 = 0.9174820620691818;
const PL_OBLIQ_SIN: f32 = 0.3977771559319137;
const PL_SUN_ANG_RADIUS_1AU: f32 = 0.00465047;
const PL_PLANET_COUNT: u32 = 8u;
const PL_BODY_COUNT: u32 = 20u;
const PL_NONE: u32 = 255u;

const PB_MARS: u32 = 0u;
const PB_PLUTO: u32 = 1u;
const PB_NEPTUNE: u32 = 2u;
const PB_URANUS: u32 = 3u;
const PB_JUPITER: u32 = 4u;
const PB_VENUS: u32 = 5u;
const PB_MERCURY: u32 = 6u;
const PB_SATURN: u32 = 7u;
const PB_CHARON: u32 = 19u;

var<private> PL_SUN_AU: array<f32, 8> = array<f32, 8>(1.524, 39.48, 30.07, 19.19, 5.203, 0.723, 0.387, 9.537);
var<private> PL_POLE_RA: array<f32, 8> = array<f32, 8>(317.681, 132.993, 299.36, 257.311, 268.0566, 272.76, 281.0103, 40.589);
var<private> PL_POLE_DEC: array<f32, 8> = array<f32, 8>(52.8865, -6.163, 43.46, -15.175, 64.4953, 67.16, 61.4155, 83.537);
var<private> PL_SPIN_W0: array<f32, 8> = array<f32, 8>(176.63, 302.695, 249.978, 203.81, 284.95, 160.20, 329.5469, 38.90);
var<private> PL_SPIN_RATE: array<f32, 8> = array<f32, 8>(350.89198, 56.3625, 541.13978, -501.16009, 870.53664, -1.4813688, 6.1385025, 810.7939);
var<private> PL_GEOM_ALBEDO: array<f32, 20> = array<f32, 20>(
    0.17, 0.52, 0.41, 0.49, 0.54, 0.69, 0.14, 0.47,
    0.63, 0.67, 0.43, 0.19, 0.22, 0.95, 1.10, 1.00, 0.25, 1.30, 0.76, 0.35
);
var<private> PL_TINT: array<vec3<f32>, 8> = array<vec3<f32>, 8>(
    vec3<f32>(1.0, 0.62, 0.42), vec3<f32>(1.0, 0.88, 0.75),
    vec3<f32>(0.55, 0.70, 1.0), vec3<f32>(0.75, 0.95, 1.0),
    vec3<f32>(1.0, 0.90, 0.78), vec3<f32>(1.0, 0.97, 0.88),
    vec3<f32>(0.98, 0.93, 0.88), vec3<f32>(1.0, 0.90, 0.70)
);

struct PlOut {
    glow: vec3<f32>,
    rgb: vec3<f32>,
    alpha: f32,
};

struct PlSurf {
    alb: vec3<f32>,
    craters: f32,
    mare: f32,
    wrap: f32,
    limb: f32,
    haze: vec3<f32>,
};

struct PlRing {
    rgb: vec3<f32>,
    alpha: f32,
    depth: f32,
};

// ---------- uniform accessors ----------
fn pl_dir(i: u32) -> vec3<f32> { return normalize(sky.celestial_bodies[i].direction_size.xyz); }
fn pl_distance(i: u32) -> f32 { return sky.celestial_bodies[i].properties.x; }
fn pl_illum(i: u32) -> f32 { return sky.celestial_bodies[i].properties.y; }
fn pl_radius_km(i: u32) -> f32 { return sky.celestial_bodies[i].properties.z; }
fn pl_parent(i: u32) -> i32 { return i32(round(sky.celestial_bodies[i].properties.w)); }
fn pl_system(i: u32) -> u32 {
    let p = pl_parent(i);
    if p < 0 { return i; }
    return u32(p);
}

// Position relative to the system's planet, in the local sky frame (km).
fn pl_rel(i: u32) -> vec3<f32> {
    let parent = pl_parent(i);
    if parent < 0 { return vec3<f32>(0.0); }
    let pidx = u32(parent);
    let di = pl_dir(i);
    let dp = pl_dir(pidx);
    let dist_i = pl_distance(i);
    let dist_p = pl_distance(pidx);
    let rel = dp * (dist_i - dist_p) + (di - dp) * dist_i;
    let want = sky.celestial_bodies[i].parent_relative_position.w;
    return rel * (want / max(length(rel), 1.0));
}

// Unit vector from the system toward the Sun, rebuilt from the phase angle.
fn pl_system_sun(p: u32) -> vec3<f32> {
    let d = pl_dir(p);
    let sd = normalize(u.sun_direction);
    let cos_a = clamp(2.0 * pl_illum(p) - 1.0, -1.0, 1.0);
    let sin_a = sqrt(max(1.0 - cos_a * cos_a, 0.0));
    var t = sd - d * dot(sd, d);
    let tl = length(t);
    if tl > 1.0e-5 {
        t = t / tl;
    } else {
        t = normalize(cross(d, vec3<f32>(0.0, 1.0, 0.0)) + vec3<f32>(1.0e-4, 0.0, 0.0));
    }
    return normalize(-d * cos_a + t * sin_a);
}

fn pl_irradiance(p: u32) -> f32 {
    return 3.2 * pow(PL_SUN_AU[p], -0.55) * max(sky.sun_intensity, 0.0);
}

// Naked-eye flux from magnitude physics: albedo * phase * (R/D)^2 / r^2
fn pl_point_intensity(i: u32) -> f32 {
    let p = pl_system(i);
    let cos_a = clamp(2.0 * pl_illum(i) - 1.0, -1.0, 1.0);
    let alpha = acos(cos_a);
    let phase = max((sin(alpha) + (PL_PI - alpha) * cos_a) / PL_PI, 0.0);
    let ratio = pl_radius_km(i) / max(pl_distance(i), 1.0);
    let r_au = PL_SUN_AU[p];
    var flux = PL_GEOM_ALBEDO[i] * phase * ratio * ratio / (r_au * r_au);
    if i == PB_SATURN { flux = flux * 2.3; }
    return min(4.5e10 * flux * max(sky.sun_intensity, 0.0), 40.0);
}

fn pl_tangent_frame(d: vec3<f32>) -> mat3x3<f32> {
    let back = -d;
    let cr = u.inv_view[0].xyz;
    let cu = u.inv_view[1].xyz;
    let right = normalize(cr - back * dot(cr, back));
    var upv = cu - back * dot(cu, back);
    upv = normalize(upv - right * dot(upv, right));
    return mat3x3<f32>(right, upv, back);
}

// ---------- orientation ----------
// The CPU gives moon offsets in ecliptic coords, but directions in the local
// sky frame. Rebuilding the ecliptic->local rotation from the moon vectors
// (TRIAD on the best conditioned pair) lets us place every pole in the sky.
fn pl_ecl_to_local() -> mat3x3<f32> {
    var ve: array<vec3<f32>, 12>;
    var vl: array<vec3<f32>, 12>;
    var vn: array<f32, 12>;

    for (var k = 0u; k < 12u; k = k + 1u) {
        let idx = PL_PLANET_COUNT + k;
        let pe = sky.celestial_bodies[idx].parent_relative_position;
        let len = max(pe.w, 1.0);
        ve[k] = pe.xyz / len;
        vl[k] = pl_rel(idx) / len;
        vn[k] = len;
    }

    var bi = 0u;
    var bj = 1u;
    var best_score = -1.0;
    for (var a = 0u; a < 12u; a = a + 1u) {
        for (var b = a + 1u; b < 12u; b = b + 1u) {
            let score = length(cross(ve[a], ve[b])) * min(vn[a], vn[b]);
            if score > best_score {
                best_score = score;
                bi = a;
                bj = b;
            }
        }
    }

    let cr = cross(ve[bi], ve[bj]);
    var bk = 0u;
    var best_triple = -1.0;
    for (var k = 0u; k < 12u; k = k + 1u) {
        let tp = abs(dot(cr, ve[k]));
        if tp > best_triple {
            best_triple = tp;
            bk = k;
        }
    }

    let ae = ve[bi];
    let be = ve[bj];
    let ce = ve[bk];
    let al = vl[bi];
    let bl = vl[bj];
    let cl = vl[bk];

    let t1e = ae;
    let t2e = normalize(cross(ae, be));
    let t3e = cross(t1e, t2e);
    let t1l = al;
    let t2l_raw = normalize(cross(al, bl));
    let t3l = cross(t1l, t2l_raw);
    let handed = select(-1.0, 1.0, dot(cross(al, bl), cl) * dot(cross(ae, be), ce) >= 0.0);

    let tl = mat3x3<f32>(t1l, t2l_raw * handed, t3l);
    let te = mat3x3<f32>(t1e, t2e, t3e);
    return tl * transpose(te);
}

fn pl_planet_pole_ecl(p: u32) -> vec3<f32> {
    let ra = radians(PL_POLE_RA[p]);
    let dec = radians(PL_POLE_DEC[p]);
    let eq = vec3<f32>(cos(dec) * cos(ra), cos(dec) * sin(ra), sin(dec));
    return vec3<f32>(
        eq.x,
        PL_OBLIQ_COS * eq.y + PL_OBLIQ_SIN * eq.z,
        -PL_OBLIQ_SIN * eq.y + PL_OBLIQ_COS * eq.z
    );
}

fn pl_spin_angle(p: u32) -> f32 {
    let rev = sky._pad2 * PL_SPIN_RATE[p] / 360.0;
    return (rev - floor(rev)) * PL_TAU + radians(PL_SPIN_W0[p]);
}

// Columns = body-fixed x, y, z axes in the local sky frame.
// Moons are tidally locked (x toward parent); Pluto faces Charon.
fn pl_body_basis(i: u32, rot: mat3x3<f32>) -> mat3x3<f32> {
    let p = pl_system(i);
    let pole = normalize(rot * pl_planet_pole_ecl(p));
    var xb = vec3<f32>(1.0, 0.0, 0.0);
    if i == PB_PLUTO {
        xb = pl_rel(PB_CHARON);
    } else if i >= PL_PLANET_COUNT {
        xb = -pl_rel(i);
    } else {
        let x0 = normalize(cross(rot[2], pole));
        let y0 = cross(pole, x0);
        let w = pl_spin_angle(i);
        xb = x0 * cos(w) + y0 * sin(w);
    }
    xb = normalize(xb - pole * dot(xb, pole));
    let yb = cross(pole, xb);
    return mat3x3<f32>(xb, yb, pole);
}

fn pl_seed_rotate(v: vec3<f32>, i: u32) -> vec3<f32> {
    let fi = f32(i);
    let axis = normalize(vec3<f32>(sin(fi * 1.71 + 0.4), cos(fi * 2.13 + 1.1), sin(fi * 0.77 + 2.3) + 1.3));
    let ang = fi * 1.93 + 0.7;
    let c = cos(ang);
    let sn = sin(ang);
    return v * c + cross(axis, v) * sn + axis * dot(axis, v) * (1.0 - c);
}

// ---------- shadows (sun disc vs occluder disc, exact overlap) ----------
fn pl_occlusion(x: vec3<f32>, s: vec3<f32>, c: vec3<f32>, r: f32, sun_r: f32) -> f32 {
    let w = c - x;
    let tc = dot(w, s);
    if tc <= 0.0 { return 0.0; }
    let wl = length(w);
    let m = length(w - s * tc);
    let ang_o = asin(clamp(r / max(wl, 1.0), 0.0, 1.0));
    let theta = atan2(m, tc);
    return circle_overlap_fraction(sun_r, ang_o, theta);
}

fn pl_system_shadow(sys: u32, self_idx: u32, x: vec3<f32>, s: vec3<f32>, sun_r: f32) -> f32 {
    var vis = 1.0;
    for (var j = 0u; j < PL_BODY_COUNT; j = j + 1u) {
        if j == self_idx { continue; }
        let par = pl_parent(j);
        var member = j == sys;
        if par >= 0 && u32(par) == sys { member = true; }
        if !member { continue; }
        vis = vis * (1.0 - pl_occlusion(x, s, pl_rel(j), pl_radius_km(j), sun_r));
    }
    return vis;
}

// ---------- Saturn's rings ----------
fn pl_hash11(x: f32) -> f32 { return fract(sin(x * 127.1 + 311.7) * 43758.5453); }
fn pl_noise11(x: f32) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let w = f * f * (3.0 - 2.0 * f);
    return mix(pl_hash11(i), pl_hash11(i + 1.0), w);
}

fn pl_ring_structure(r: f32, pixel_km: f32) -> f32 {
    var acc = 0.0;
    var lam = 2600.0;
    var amp = 1.0;
    var wsum = 0.0;
    for (var k = 0u; k < 5u; k = k + 1u) {
        let vis = saturate((lam / max(pixel_km, 1.0) - 1.0) * 0.5);
        acc += (pl_noise11(r / lam + f32(k) * 17.3) - 0.5) * amp * vis;
        wsum += amp;
        lam *= 0.38;
        amp *= 0.72;
    }
    return acc / wsum * 2.0;
}

fn pl_ring_box(r: f32, a: f32, b: f32, w: f32) -> f32 {
    return smoothstep(a - w, a + w, r) * (1.0 - smoothstep(b - w, b + w, r));
}

fn pl_ring_gap(r: f32, c: f32, w: f32, depth: f32, pixel_km: f32) -> f32 {
    let we = max(w, pixel_km * 0.8);
    let t = (r - c) / we;
    return 1.0 - depth * (w / we) * exp(-t * t);
}

// Optical depth vs radius (km): D, C, B, Cassini, A (Encke/Keeler), F
fn pl_ring_tau(r: f32, pixel_km: f32) -> f32 {
    if r < 66000.0 || r > 141500.0 { return 0.0; }
    let st = pl_ring_structure(r, pixel_km);
    var tau = 0.012 * pl_ring_box(r, 66900.0, 74500.0, 400.0);

    let c_ring = pl_ring_box(r, 74658.0, 91975.0, 250.0);
    tau += c_ring * max(0.075 + 0.075 * smoothstep(76000.0, 90000.0, r) + 0.07 * st, 0.01);

    let b_ring = pl_ring_box(r, 91975.0, 117507.0, 200.0);
    let b_base = mix(0.45, 1.55, smoothstep(92000.0, 99000.0, r));
    tau += b_ring * max(b_base * (1.0 + 0.55 * st) + 0.25 * smoothstep(110000.0, 117000.0, r), 0.2);

    tau += pl_ring_box(r, 117507.0, 122050.0, 150.0) * max(0.07 + 0.05 * st, 0.01);

    let a_ring = pl_ring_box(r, 122050.0, 136775.0, 120.0);
    var a_tau = max(0.62 + 0.28 * smoothstep(122000.0, 130000.0, r) + 0.30 * st, 0.08);
    a_tau = a_tau * pl_ring_gap(r, 133589.0, 160.0, 0.95, pixel_km);
    a_tau = a_tau * pl_ring_gap(r, 136505.0, 25.0, 0.8, pixel_km);
    tau += a_ring * a_tau;

    let fw = max(110.0, pixel_km);
    let ft = (r - 140180.0) / fw;
    tau += 0.18 * (110.0 / fw) * exp(-ft * ft) * (0.6 + 0.8 * pl_noise11(r / 60.0));
    return tau;
}

fn pl_ring_albedo(r: f32, st: f32) -> vec3<f32> {
    var c = vec3<f32>(0.34, 0.32, 0.30);
    c = mix(c, vec3<f32>(0.86, 0.72, 0.54), smoothstep(91000.0, 93500.0, r));
    c = mix(c, vec3<f32>(0.40, 0.37, 0.34), smoothstep(117000.0, 118800.0, r));
    c = mix(c, vec3<f32>(0.74, 0.67, 0.56), smoothstep(121500.0, 123500.0, r));
    c = mix(c, vec3<f32>(0.65, 0.62, 0.58), smoothstep(137000.0, 139500.0, r));
    return c * (1.0 + 0.30 * st);
}

// Sunlight transmitted through the ring plane to point x (planet-relative km).
fn pl_ring_transmittance(x: vec3<f32>, s: vec3<f32>, ring_n: vec3<f32>, pixel_km: f32) -> f32 {
    let sn = dot(s, ring_n);
    if abs(sn) < 0.002 { return 1.0; }
    let t = -dot(x, ring_n) / sn;
    if t <= 0.0 { return 1.0; }
    let y = x + s * t;
    let tau = pl_ring_tau(length(y), pixel_km);
    return exp(-tau / abs(sn));
}

fn pl_ring_layer(q: vec2<f32>, px_rel: f32, rot: mat3x3<f32>) -> PlRing {
    var out: PlRing;
    out.rgb = vec3<f32>(0.0);
    out.alpha = 0.0;
    out.depth = 0.0;

    let sat = PB_SATURN;
    let d = pl_dir(sat);
    let frame = pl_tangent_frame(d);
    let right = frame[0];
    let upv = frame[1];
    let ring_n = normalize(rot * pl_planet_pole_ecl(sat));
    let dn = dot(d, ring_n);
    if abs(dn) < 0.003 { return out; }

    let rk = pl_radius_km(sat);
    let px_km = q.x * rk;
    let py_km = q.y * rk;
    let t = -(px_km * dot(right, ring_n) + py_km * dot(upv, ring_n)) / dn;
    let rr = sqrt(px_km * px_km + py_km * py_km + t * t);
    if rr < 66000.0 || rr > 141500.0 { return out; }

    let pixel_km = px_rel * rk / max(abs(dn), 0.12);
    let tau = pl_ring_tau(rr, pixel_km);
    if tau < 0.002 { return out; }

    let st = pl_ring_structure(rr, pixel_km);
    let mu_v = abs(dn);
    let alpha = 1.0 - exp(-tau / max(mu_v, 0.03));

    let s = pl_system_sun(sat);
    let sn = dot(s, ring_n);
    let mu_s = abs(sn);
    let same_side = (-dn) * sn > 0.0;

    let x_rel = px_km * right + py_km * upv + t * d;
    let sun_r = PL_SUN_ANG_RADIUS_1AU / PL_SUN_AU[sat];
    let vis = pl_system_shadow(sat, PL_NONE, x_rel, s, sun_r);

    var light = 0.0;
    if same_side {
        light = mix(0.45, 1.0, sqrt(saturate(mu_s * 1.6)));
    } else {
        light = exp(-tau / max(mu_s, 0.03)) * 0.85;
    }

    out.rgb = pl_ring_albedo(rr, st) * (light * vis + 0.012) * pl_irradiance(sat);
    out.alpha = alpha;
    out.depth = t;
    return out;
}

// ---------- procedural surfaces ----------
fn pl_fbm(p: vec3<f32>, cpp: f32) -> f32 {
    var q = p;
    var sum = 0.0;
    var amp = 0.5;
    var norm = 0.0;
    var f = cpp;
    for (var k = 0u; k < 5u; k = k + 1u) {
        let fade = 1.0 - smoothstep(0.22, 0.55, f);
        sum += mix(0.5, value_noise3(q), fade) * amp;
        norm += amp;
        q = q * 2.03 + vec3<f32>(13.1, 7.7, 17.3);
        amp *= 0.5;
        f *= 2.03;
    }
    return saturate((sum / norm - 0.5) * 2.0 + 0.5);
}

fn pl_n(nb: vec3<f32>, scale: f32, off: vec3<f32>, px: f32) -> f32 {
    return pl_fbm(nb * scale + off, scale * px);
}

// anisotropic noise: stretched along longitude for cloud bands
fn pl_na(nb: vec3<f32>, sxy: f32, sz: f32, off: vec3<f32>, px: f32) -> f32 {
    return pl_fbm(vec3<f32>(nb.xy * sxy, nb.z * sz) + off, max(sxy, sz) * px);
}

fn pl_bump(x: f32, c: f32, w: f32) -> f32 {
    let t = (x - c) / w;
    return exp(-t * t);
}

fn pl_wrap_pi(a: f32) -> f32 { return a - PL_TAU * floor((a + PL_PI) / PL_TAU); }

fn pl_lonlat(nb: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(atan2(nb.y, nb.x), asin(clamp(nb.z, -1.0, 1.0)));
}

fn pl_ellipse(lon: f32, lat: f32, lon0: f32, lat0: f32, wlon: f32, wlat: f32) -> vec2<f32> {
    return vec2<f32>(pl_wrap_pi(lon - lon0) * cos(lat0) / wlon, (lat - lat0) / wlat);
}

fn pl_blob(e: vec2<f32>, soft: f32) -> f32 {
    return 1.0 - smoothstep(soft, 1.0, length(e));
}

fn pl_surf_base(col: vec3<f32>) -> PlSurf {
    var s: PlSurf;
    s.alb = col;
    s.craters = 0.0;
    s.mare = 0.0;
    s.wrap = 0.0;
    s.limb = 0.0;
    s.haze = vec3<f32>(0.0);
    return s;
}

fn pl_mars(nb: vec3<f32>, px: f32) -> PlSurf {
    let ll = pl_lonlat(nb);
    let lat_deg = degrees(ll.y);
    let big = pl_n(nb, 2.0, vec3<f32>(11.0, 3.7, 5.9), px);
    let mid = pl_n(nb, 5.5, vec3<f32>(2.0, 9.0, 4.0), px);
    let fine = pl_n(nb, 15.0, vec3<f32>(6.0, 1.0, 8.0), px);
    let lat_w = 1.0 - smoothstep(12.0, 55.0, abs(lat_deg + 10.0));
    let dark = smoothstep(0.52, 0.70, big + (mid - 0.5) * 0.35) * lat_w;
    var col = mix(vec3<f32>(0.50, 0.25, 0.12), vec3<f32>(0.62, 0.33, 0.16), smoothstep(0.3, 0.7, mid));
    col = mix(col, vec3<f32>(0.13, 0.085, 0.065), dark * 0.85);
    let hellas = pl_ellipse(ll.x, ll.y, radians(70.0), radians(-42.0), radians(14.0), radians(10.0));
    col = mix(col, vec3<f32>(0.72, 0.46, 0.30), pl_blob(hellas, 0.45) * 0.65);
    let edge_n = 83.0 + (mid - 0.5) * 6.0;
    let edge_s = -77.0 - (mid - 0.5) * 6.0;
    let cap = max(
        smoothstep(edge_n - 1.2, edge_n + 1.2, lat_deg),
        1.0 - smoothstep(edge_s - 1.2, edge_s + 1.2, lat_deg)
    );
    col = mix(col, vec3<f32>(0.92, 0.93, 0.95), cap);
    col *= 0.90 + 0.20 * fine;
    var s = pl_surf_base(col);
    s.wrap = 0.04;
    s.haze = vec3<f32>(0.78, 0.56, 0.46) * 0.2;
    return s;
}

fn pl_venus(nb: vec3<f32>, px: f32) -> PlSurf {
    let warp = pl_n(nb, 2.0, vec3<f32>(3.3, 1.2, 7.7), px);
    let v = pl_fbm(vec3<f32>(nb.xy * 3.0, nb.z * 5.0) + (warp - 0.5) * 2.4, 5.0 * px);
    let m = pl_fbm(vec3<f32>(nb.xy * 2.0, nb.z * 6.0) + (warp - 0.5) * 3.0 + 4.0, 6.0 * px);
    let ll = pl_lonlat(nb);
    let bands = 0.5 + 0.5 * sin(ll.y * 10.0 + 6.0 * (warp - 0.5));
    var col = mix(vec3<f32>(0.64, 0.58, 0.42), vec3<f32>(0.78, 0.72, 0.55), smoothstep(0.35, 0.7, v));
    col *= 1.0 - 0.16 * smoothstep(0.55, 0.78, m) * (0.5 + 0.5 * bands);
    var s = pl_surf_base(col);
    s.wrap = 0.45;
    s.limb = -0.18;
    s.haze = vec3<f32>(1.0, 0.85, 0.55) * 0.35;
    return s;
}

fn pl_mercury(nb: vec3<f32>, px: f32) -> PlSurf {
    let a = pl_n(nb, 2.0, vec3<f32>(5.0, 1.0, 9.0), px);
    let b = pl_n(nb, 5.0, vec3<f32>(2.0, 8.0, 3.0), px);
    let tint = pl_n(nb, 1.6, vec3<f32>(9.0, 4.0, 2.0), px);
    let plains = smoothstep(0.52, 0.68, a);
    var col = mix(vec3<f32>(0.150, 0.138, 0.126), vec3<f32>(0.095, 0.100, 0.112), tint);
    col = mix(col, vec3<f32>(0.128, 0.118, 0.108), plains * 0.8);
    col *= 0.90 + 0.20 * b;
    var s = pl_surf_base(col);
    s.craters = 0.55;
    s.mare = plains * 0.85;
    return s;
}

fn pl_jupiter(nb: vec3<f32>, px: f32) -> PlSurf {
    let ll = pl_lonlat(nb);
    let flow = pl_na(nb, 1.3, 13.0, vec3<f32>(1.7, 4.1, 9.3), px);
    let turb = pl_na(nb, 3.2, 30.0, vec3<f32>(5.3, 1.1, 7.7), px);
    let fine = pl_na(nb, 7.0, 64.0, vec3<f32>(2.2, 8.8, 3.3), px);

    let grs = pl_ellipse(ll.x, ll.y, radians(40.0), radians(-22.0), radians(11.5), radians(6.5));
    let grs_r = length(grs);
    let lat_deg = degrees(ll.y) + (flow - 0.5) * 3.4 + (turb - 0.5) * 1.1;

    var belt = 0.0;
    belt += 1.00 * pl_bump(lat_deg, 14.5, 3.5);
    belt += 0.95 * pl_bump(lat_deg, -15.5, 4.0);
    belt += 0.50 * pl_bump(lat_deg, 24.5, 2.0);
    belt += 0.40 * pl_bump(lat_deg, -31.0, 2.0);
    belt += 0.60 * pl_bump(lat_deg, 41.0, 3.0);
    belt += 0.60 * pl_bump(lat_deg, -43.5, 3.2);
    belt += 0.30 * pl_bump(lat_deg, 33.0, 1.6);
    belt = saturate(belt);
    let polar = smoothstep(50.0, 68.0, abs(lat_deg));

    var col = mix(vec3<f32>(0.90, 0.82, 0.68), vec3<f32>(0.93, 0.84, 0.62), 0.55 * pl_bump(lat_deg, 0.0, 5.5));
    let neb_w = pl_bump(lat_deg, 14.5, 4.0);
    col = mix(col, mix(vec3<f32>(0.46, 0.27, 0.16), vec3<f32>(0.50, 0.25, 0.14), neb_w), belt);
    col = mix(col, vec3<f32>(0.46, 0.42, 0.42) * (0.8 + 0.4 * turb), polar);

    let edge = 4.0 * belt * (1.0 - belt);
    col *= 0.86 + 0.28 * turb + 0.24 * edge * (fine - 0.5);
    let fest = smoothstep(0.58, 0.80, pl_na(nb, 6.0, 40.0, vec3<f32>(9.0, 3.0, 5.0), px));
    col = mix(col, vec3<f32>(0.95, 0.90, 0.80), fest * edge * 0.55);

    let oval_a = pl_ellipse(ll.x, ll.y, radians(150.0), radians(-33.0), radians(3.0), radians(2.2));
    let oval_b = pl_ellipse(ll.x, ll.y, radians(190.0), radians(-33.0), radians(3.2), radians(2.4));
    col = mix(col, vec3<f32>(0.96, 0.93, 0.86), pl_blob(oval_a, 0.5) * 0.85);
    col = mix(col, vec3<f32>(0.72, 0.38, 0.27), pl_blob(oval_b, 0.5) * 0.85);

    if grs_r < 1.8 {
        let ang = atan2(grs.y, grs.x);
        let sw = grs_r * 2.0;
        let swirl = pl_fbm(vec3<f32>(sw * cos(ang + grs_r * 2.5), sw * sin(ang + grs_r * 2.5), 3.1), 2.0 * px);
        let spot = 1.0 - smoothstep(0.62, 1.0, grs_r);
        let collar = smoothstep(0.9, 1.15, grs_r) * (1.0 - smoothstep(1.3, 1.75, grs_r));
        var spot_col = mix(vec3<f32>(0.60, 0.25, 0.14), vec3<f32>(0.78, 0.40, 0.26), swirl);
        spot_col = mix(spot_col, vec3<f32>(0.85, 0.55, 0.42), smoothstep(0.55, 0.95, grs_r) * 0.5);
        col = mix(col, vec3<f32>(0.92, 0.88, 0.80), collar * 0.55);
        col = mix(col, spot_col, spot);
    }

    var s = pl_surf_base(col);
    s.wrap = 0.10;
    s.limb = 0.50;
    s.haze = vec3<f32>(0.70, 0.62, 0.55) * 0.12;
    return s;
}

fn pl_saturn(nb: vec3<f32>, px: f32) -> PlSurf {
    let ll = pl_lonlat(nb);
    let flow = pl_na(nb, 1.2, 11.0, vec3<f32>(4.0, 2.0, 7.0), px);
    let turb = pl_na(nb, 3.0, 28.0, vec3<f32>(8.0, 3.0, 1.0), px);
    let lat_deg = degrees(ll.y) + (flow - 0.5) * 2.4;
    let bands = 0.5 + 0.5 * sin(radians(lat_deg) * 26.0 + 3.0 * (turb - 0.5));
    let wide = 0.5 + 0.5 * sin(radians(lat_deg) * 9.0 + 1.0);
    var col = mix(vec3<f32>(0.70, 0.57, 0.38), vec3<f32>(0.88, 0.77, 0.55), saturate(0.35 * bands + 0.45 * wide + 0.2));
    col = mix(col, vec3<f32>(0.93, 0.84, 0.60), 0.55 * pl_bump(lat_deg, 2.0, 9.0));
    col = mix(col, vec3<f32>(0.64, 0.50, 0.34), 0.35 * pl_bump(lat_deg, -13.0, 4.0));

    let hex_edge = 76.0 + 1.3 * cos(6.0 * ll.x);
    col = mix(col, vec3<f32>(0.50, 0.55, 0.54), smoothstep(hex_edge - 1.2, hex_edge + 1.2, lat_deg) * 0.8);
    col = mix(col, vec3<f32>(0.95, 0.88, 0.68), pl_bump(lat_deg, hex_edge - 1.8, 0.9) * 0.35);
    col = mix(col, vec3<f32>(0.55, 0.52, 0.42), smoothstep(62.0, 78.0, -lat_deg) * 0.6);
    col *= 0.92 + 0.16 * turb;

    var s = pl_surf_base(col);
    s.wrap = 0.10;
    s.limb = 0.38;
    s.haze = vec3<f32>(0.9, 0.78, 0.55) * 0.07;
    return s;
}

fn pl_uranus(nb: vec3<f32>, px: f32) -> PlSurf {
    let ll = pl_lonlat(nb);
    let turb = pl_na(nb, 2.0, 16.0, vec3<f32>(2.0, 5.0, 3.0), px);
    var col = vec3<f32>(0.58, 0.80, 0.84);
    col = mix(col, vec3<f32>(0.72, 0.90, 0.90), nb.z * nb.z * 0.5);
    col *= 0.96 + 0.05 * sin(ll.y * 14.0 + 2.0 * (turb - 0.5)) + 0.04 * (turb - 0.5);
    var s = pl_surf_base(col);
    s.wrap = 0.08;
    s.limb = 0.22;
    s.haze = vec3<f32>(0.55, 0.9, 0.95) * 0.35;
    return s;
}

fn pl_neptune(nb: vec3<f32>, px: f32) -> PlSurf {
    let ll = pl_lonlat(nb);
    let flow = pl_na(nb, 1.2, 12.0, vec3<f32>(3.0, 8.0, 1.0), px);
    let lat_deg = degrees(ll.y) + (flow - 0.5) * 3.0;
    let bands = 0.5 + 0.5 * sin(radians(lat_deg) * 11.0 + 2.0 * (flow - 0.5));
    var col = mix(vec3<f32>(0.10, 0.21, 0.68), vec3<f32>(0.19, 0.36, 0.86), bands * 0.8 + 0.1);
    col *= 0.88 + 0.24 * pl_na(nb, 2.5, 22.0, vec3<f32>(5.0, 2.0, 9.0), px);
    let gds = pl_ellipse(ll.x, ll.y, radians(120.0), radians(-20.0), radians(15.0), radians(8.0));
    col = mix(col, vec3<f32>(0.04, 0.08, 0.38), pl_blob(gds, 0.5) * 0.85);
    let comp = pl_ellipse(ll.x, ll.y, radians(150.0), radians(-28.0), radians(10.0), radians(3.0));
    col = mix(col, vec3<f32>(0.85, 0.92, 1.0), pl_blob(comp, 0.4) * 0.8);
    let cir = smoothstep(0.62, 0.82, pl_na(nb, 1.6, 20.0, vec3<f32>(1.0, 6.0, 4.0), px));
    let cir_lat = pl_bump(lat_deg, -42.0, 7.0) + 0.6 * pl_bump(lat_deg, -27.0, 5.0) + 0.5 * pl_bump(lat_deg, 22.0, 6.0);
    col = mix(col, vec3<f32>(0.88, 0.94, 1.0), saturate(cir * cir_lat) * 0.65);
    var s = pl_surf_base(col);
    s.wrap = 0.06;
    s.limb = 0.30;
    s.haze = vec3<f32>(0.3, 0.5, 1.0) * 0.45;
    return s;
}

fn pl_pluto(nb: vec3<f32>, px: f32) -> PlSurf {
    let ll = pl_lonlat(nb);
    var ld = degrees(ll.x);
    if ld < 0.0 { ld = ld + 360.0; }
    let a = pl_n(nb, 3.2, vec3<f32>(1.0, 5.0, 9.0), px);
    let b = pl_n(nb, 9.0, vec3<f32>(7.0, 2.0, 3.0), px);
    var col = mix(vec3<f32>(0.36, 0.25, 0.18), vec3<f32>(0.58, 0.44, 0.32), a);
    col *= 0.88 + 0.24 * b;
    let cth = smoothstep(30.0, 60.0, ld) * (1.0 - smoothstep(130.0, 150.0, ld));
    col = mix(col, vec3<f32>(0.12, 0.05, 0.035), cth * (1.0 - smoothstep(0.1, 0.3, abs(nb.z))) * 0.9);
    let sp = pl_ellipse(ll.x, ll.y, radians(178.0), radians(18.0), radians(20.0), radians(21.0));
    let lb = pl_ellipse(ll.x, ll.y, radians(150.0), radians(32.0), radians(13.0), radians(12.0));
    let heart = max(pl_blob(sp, 0.7), pl_blob(lb, 0.7));
    col = mix(col, vec3<f32>(0.86, 0.80, 0.70), heart * (0.85 + 0.15 * b));
    col = mix(col, vec3<f32>(0.25, 0.12, 0.08), smoothstep(0.7, 0.92, nb.z) * 0.6);
    var s = pl_surf_base(col);
    s.wrap = 0.02;
    s.haze = vec3<f32>(0.3, 0.4, 0.6) * 0.04;
    return s;
}

fn pl_io(nb: vec3<f32>, px: f32) -> PlSurf {
    let a = pl_n(nb, 3.0, vec3<f32>(1.3, 6.1, 2.2), px);
    let b = pl_n(nb, 9.0, vec3<f32>(4.4, 0.7, 8.8), px);
    let c = pl_n(nb, 18.0, vec3<f32>(9.1, 3.3, 5.5), px);
    let fr = pl_n(nb, 5.5, vec3<f32>(7.0, 1.0, 3.0), px);
    var col = mix(vec3<f32>(0.78, 0.66, 0.26), vec3<f32>(0.93, 0.87, 0.58), smoothstep(0.30, 0.72, a));
    col = mix(col, vec3<f32>(0.86, 0.46, 0.14), smoothstep(0.55, 0.75, b) * 0.75);
    col = mix(col, vec3<f32>(0.95, 0.94, 0.88), smoothstep(0.62, 0.80, fr) * 0.7);
    col = mix(col, vec3<f32>(0.05, 0.04, 0.04), smoothstep(0.74, 0.82, c) * 0.9);
    let pele = normalize(vec3<f32>(cos(-0.33) * cos(4.45), cos(-0.33) * sin(4.45), sin(-0.33)));
    let ang = acos(clamp(dot(nb, pele), -1.0, 1.0));
    let rt = (ang - 0.10) / 0.035;
    col = mix(col, vec3<f32>(0.74, 0.16, 0.07), exp(-rt * rt) * 0.9);
    col = mix(col, vec3<f32>(0.03, 0.02, 0.02), 1.0 - smoothstep(0.015, 0.05, ang));
    col = mix(col, vec3<f32>(0.62, 0.30, 0.10), smoothstep(0.78, 0.95, abs(nb.z)) * 0.6);
    return pl_surf_base(col);
}

fn pl_europa(nb: vec3<f32>, px: f32) -> PlSurf {
    let w = vec3<f32>(
        pl_n(nb, 2.0, vec3<f32>(1.0, 2.0, 3.0), px),
        pl_n(nb, 2.0, vec3<f32>(4.0, 5.0, 6.0), px),
        pl_n(nb, 2.0, vec3<f32>(7.0, 8.0, 9.0), px)
    ) - vec3<f32>(0.5);
    let p = nb + w * 0.35;
    let fade = 1.0 - smoothstep(0.03, 0.12, px);
    let l1 = 1.0 - smoothstep(0.0, 0.030, abs(pl_n(p, 3.2, vec3<f32>(1.0), px) - 0.5));
    let l2 = 1.0 - smoothstep(0.0, 0.040, abs(pl_n(p, 6.5, vec3<f32>(5.0), px) - 0.5));
    let l3 = 1.0 - smoothstep(0.0, 0.045, abs(pl_n(p, 12.0, vec3<f32>(9.0), px) - 0.5));
    let lines = saturate(l1 + 0.8 * l2 + 0.5 * l3) * fade;
    let chaos = smoothstep(0.56, 0.74, pl_n(p, 4.5, vec3<f32>(2.0, 6.0, 1.0), px));
    var col = mix(vec3<f32>(0.84, 0.82, 0.76), vec3<f32>(0.66, 0.58, 0.48), chaos * 0.8);
    col = mix(col, vec3<f32>(0.50, 0.30, 0.20), lines * 0.75);
    return pl_surf_base(col);
}

fn pl_ganymede(nb: vec3<f32>, px: f32) -> PlSurf {
    let reg = smoothstep(0.50, 0.64, pl_n(nb, 2.2, vec3<f32>(3.0, 1.0, 6.0), px));
    let n2 = pl_n(nb, 6.0, vec3<f32>(8.0, 2.0, 5.0), px);
    var s = pl_surf_base(mix(vec3<f32>(0.46, 0.44, 0.40), vec3<f32>(0.21, 0.19, 0.16), reg) * (0.9 + 0.2 * n2));
    s.craters = 1.3;
    return s;
}

fn pl_callisto(nb: vec3<f32>, px: f32) -> PlSurf {
    let a = pl_n(nb, 3.0, vec3<f32>(2.0, 5.0, 1.0), px);
    var s = pl_surf_base(vec3<f32>(0.19, 0.165, 0.14) * (0.8 + 0.4 * a));
    s.craters = 1.7;
    return s;
}

fn pl_titan(nb: vec3<f32>, px: f32) -> PlSurf {
    let n = pl_n(nb, 2.0, vec3<f32>(4.0, 1.0, 8.0), px);
    var col = mix(vec3<f32>(0.74, 0.50, 0.22), vec3<f32>(0.58, 0.34, 0.13), smoothstep(-0.3, 0.6, nb.z));
    col *= 0.94 + 0.12 * n;
    var s = pl_surf_base(col);
    s.wrap = 0.55;
    s.limb = -0.2;
    s.haze = vec3<f32>(1.0, 0.60, 0.22) * 0.5;
    return s;
}

fn pl_icy(nb: vec3<f32>, px: f32, tint: vec3<f32>, contrast: f32, seed: f32, gain: f32) -> PlSurf {
    let a = pl_n(nb, 2.6, vec3<f32>(seed, seed * 1.7, seed * 0.6), px);
    let b = pl_n(nb, 7.0, vec3<f32>(seed * 2.3, 4.0, seed), px);
    var s = pl_surf_base(tint * (1.0 + contrast * ((a - 0.5) * 0.9 + (b - 0.5) * 0.35)));
    s.craters = gain;
    return s;
}

fn pl_iapetus(nb: vec3<f32>, px: f32) -> PlSurf {
    let n = pl_n(nb, 4.0, vec3<f32>(6.0, 2.0, 9.0), px);
    let n2 = pl_n(nb, 9.0, vec3<f32>(1.0, 7.0, 3.0), px);
    let dark = (1.0 - smoothstep(-0.25, 0.10, nb.y + 0.22 * (n - 0.5))) * (1.0 - smoothstep(0.55, 0.85, abs(nb.z)));
    var s = pl_surf_base(mix(vec3<f32>(0.52, 0.51, 0.49), vec3<f32>(0.055, 0.040, 0.032), dark) * (0.85 + 0.3 * n2));
    s.craters = 1.0;
    return s;
}

fn pl_enceladus(nb: vec3<f32>, px: f32) -> PlSurf {
    let n = pl_n(nb, 4.0, vec3<f32>(3.0, 6.0, 1.0), px);
    let lon = atan2(nb.y, nb.x);
    let t1 = (nb.z + 0.88 + 0.03 * sin(lon * 3.0 + 0.7)) / 0.012;
    let t2 = (nb.z + 0.92 + 0.025 * sin(lon * 4.0 + 2.1)) / 0.012;
    let t3 = (nb.z + 0.84 + 0.03 * sin(lon * 5.0 + 4.0)) / 0.012;
    let tiger = saturate(exp(-t1 * t1) + exp(-t2 * t2) + exp(-t3 * t3));
    var col = vec3<f32>(0.95, 0.97, 1.0) * (0.9 + 0.12 * n);
    col = mix(col, vec3<f32>(0.55, 0.78, 0.98), tiger * 0.8);
    var s = pl_surf_base(col);
    s.craters = 0.7;
    return s;
}

fn pl_triton(nb: vec3<f32>, px: f32) -> PlSurf {
    let n = pl_n(nb, 3.0, vec3<f32>(2.0, 4.0, 6.0), px);
    let cant = pl_n(nb, 22.0, vec3<f32>(1.0, 9.0, 3.0), px);
    let south = 1.0 - smoothstep(-0.65, -0.15, nb.z);
    var col = mix(vec3<f32>(0.62, 0.55, 0.50), vec3<f32>(0.88, 0.78, 0.72), n) * (0.85 + 0.3 * cant);
    col = mix(col, vec3<f32>(0.97, 0.86, 0.80), south * 0.8);
    let streak = smoothstep(0.62, 0.82, pl_fbm(vec3<f32>(nb.x * 3.0, nb.y * 10.0, nb.z * 3.0) + 5.0, 10.0 * px));
    col *= 1.0 - 0.35 * streak * south;
    return pl_surf_base(col);
}

fn pl_charon(nb: vec3<f32>, px: f32) -> PlSurf {
    let n = pl_n(nb, 3.0, vec3<f32>(7.0, 3.0, 2.0), px);
    var col = vec3<f32>(0.38, 0.37, 0.38) * (0.8 + 0.4 * n);
    col = mix(col, vec3<f32>(0.34, 0.14, 0.09), smoothstep(0.55, 0.85, nb.z + 0.2 * (n - 0.5)) * 0.85);
    var s = pl_surf_base(col);
    s.craters = 0.9;
    s.mare = smoothstep(0.0, 0.4, -nb.z) * 0.7;
    return s;
}

fn pl_surface(i: u32, nb: vec3<f32>, px: f32) -> PlSurf {
    var s: PlSurf;
    switch i {
        case 0u: { s = pl_mars(nb, px); }
        case 1u: { s = pl_pluto(nb, px); }
        case 2u: { s = pl_neptune(nb, px); }
        case 3u: { s = pl_uranus(nb, px); }
        case 4u: { s = pl_jupiter(nb, px); }
        case 5u: { s = pl_venus(nb, px); }
        case 6u: { s = pl_mercury(nb, px); }
        case 7u: { s = pl_saturn(nb, px); }
        case 8u: { s = pl_io(nb, px); }
        case 9u: { s = pl_europa(nb, px); }
        case 10u: { s = pl_ganymede(nb, px); }
        case 11u: { s = pl_callisto(nb, px); }
        case 12u: { s = pl_titan(nb, px); }
        case 13u: { s = pl_icy(nb, px, vec3<f32>(0.52, 0.52, 0.51), 0.5, 3.1, 1.0); }
        case 14u: { s = pl_icy(nb, px, vec3<f32>(0.80, 0.80, 0.79), 0.25, 5.7, 1.0); }
        case 15u: { s = pl_icy(nb, px, vec3<f32>(0.62, 0.62, 0.61), 0.4, 8.3, 1.0); }
        case 16u: { s = pl_iapetus(nb, px); }
        case 17u: { s = pl_enceladus(nb, px); }
        case 18u: { s = pl_triton(nb, px); }
        default: { s = pl_charon(nb, px); }
    }
    return s;
}

fn pl_shade_body(i: u32, q: vec2<f32>, r2: f32, px_rel: f32, rot: mat3x3<f32>) -> vec3<f32> {
    let p = pl_system(i);
    let d = pl_dir(i);
    let frame = pl_tangent_frame(d);
    let z = sqrt(max(1.0 - r2, 0.0));
    let n = normalize(frame * vec3<f32>(q.x, q.y, z));
    let s = pl_system_sun(p);

    let bt = transpose(pl_body_basis(i, rot));
    let nb = bt * n;
    let sb = bt * s;
    let px = px_rel / max(z, 0.25);

    let surf = pl_surface(i, nb, px);
    let irr = pl_irradiance(p);
    let sun_r = PL_SUN_ANG_RADIUS_1AU / PL_SUN_AU[p];
    let ndl = dot(n, s);

    var albedo = surf.alb;
    var diffuse = 0.0;

    if surf.craters > 0.0 {
        let nr = pl_seed_rotate(nb, i);
        let sr = pl_seed_rotate(sb, i);
        let cf = crater_field(nr, px, surf.mare);
        albedo = albedo * (1.0 + cf.bright * surf.craters - cf.floor_dark * 0.12);
        let ln = crater_surface_normal(nr, cf.grad);
        let sdot = dot(nr, sr);
        var shadow = 1.0;
        if sdot > -0.12 {
            shadow = lunar_self_shadow(nr, sr, cf.height, px, surf.mare);
        }
        let horizon = -sqrt(2.0 * max(cf.height, 0.0));
        let lit = smoothstep(horizon - 0.006, horizon + 0.006, sdot);
        diffuse = pow(max(dot(ln, sr), 0.0), 0.72) * shadow * lit;
    } else {
        diffuse = saturate((ndl + surf.wrap) / (1.0 + surf.wrap));
    }

    // eclipses: planet <-> moons, moons <-> moons, rings -> everything
    let x_rel = pl_rel(i) + n * pl_radius_km(i);
    var vis = pl_system_shadow(p, i, x_rel, s, sun_r);
    if p == PB_SATURN {
        let ring_n = normalize(rot * pl_planet_pole_ecl(PB_SATURN));
        let blur_km = max(px_rel * pl_radius_km(i), 150.0);
        vis = vis * pl_ring_transmittance(x_rel, s, ring_n, blur_km);
    }

    var rad = albedo * diffuse * vis * irr;
    rad = rad * mix(1.0 - surf.limb, 1.0, pow(z, 0.6));
    rad += surf.haze * pow(1.0 - z, 3.0) * saturate(ndl + 0.25) * vis * irr;

    // planetshine on moons
    if i >= PL_PLANET_COUNT {
        let rel = pl_rel(i);
        let dist = max(length(rel), 1.0);
        let rhat = rel / dist;
        let ratio = pl_radius_km(p) / dist;
        let frac = 0.5 * (1.0 + dot(rhat, s));
        let e = PL_GEOM_ALBEDO[p] * 0.45 * frac * ratio * ratio;
        rad += albedo * PL_TINT[p] * e * saturate(dot(n, -rhat)) * irr;
    }
    return rad;
}

fn pl_halo(i: u32, q: vec2<f32>, r2: f32, size_px: f32, cam_right: vec3<f32>, cam_up: vec3<f32>) -> vec3<f32> {
    var hc = vec3<f32>(0.0);
    switch i {
        case 0u: { hc = vec3<f32>(0.80, 0.60, 0.50) * 0.10; }
        case 2u: { hc = vec3<f32>(0.30, 0.50, 1.00) * 0.45; }
        case 3u: { hc = vec3<f32>(0.55, 0.90, 0.95) * 0.35; }
        case 4u: { hc = vec3<f32>(0.75, 0.60, 0.50) * 0.08; }
        case 5u: { hc = vec3<f32>(1.00, 0.82, 0.50) * 0.90; }
        case 7u: { hc = vec3<f32>(0.90, 0.80, 0.55) * 0.08; }
        case 12u: { hc = vec3<f32>(1.00, 0.55, 0.18) * 0.55; }
        default: { }
    }
    if hc.x + hc.y + hc.z <= 0.0 { return vec3<f32>(0.0); }

    let p = pl_system(i);
    let s = pl_system_sun(p);
    let sun_scr = vec2<f32>(dot(s, cam_right), dot(s, cam_up));
    let sl = length(sun_scr);
    let rr = sqrt(r2);
    var side = 0.5;
    if sl > 1.0e-3 {
        side = smoothstep(-0.35, 0.85, dot(q / rr, sun_scr / sl));
    }
    let gate = smoothstep(4.0, 14.0, size_px);
    let fall = exp(-(rr - 1.0) * 16.0);
    return hc * fall * side * (0.3 + 0.7 * saturate(sl * 1.6)) * pl_irradiance(p) * gate * 0.25;
}

fn pl_render(ndc: vec2<f32>, pix: f32, day_w: f32) -> PlOut {
    var res: PlOut;
    res.glow = vec3<f32>(0.0);
    res.rgb = vec3<f32>(0.0);
    res.alpha = 0.0;

    let cam_right = u.inv_view[0].xyz;
    let cam_up = u.inv_view[1].xyz;
    let night_vis = mix(1.0, 0.03, day_w);

    var best = PL_NONE;
    var best_q = vec2<f32>(0.0);
    var best_r2 = 0.0;
    var best_cov = 0.0;
    var best_size = 1.0;
    var best_base = 1.0e30;
    var best_local = 0.0;
    var sat_q = vec2<f32>(1000.0);
    var sat_size = 0.0;

    for (var i = 0u; i < PL_BODY_COUNT; i = i + 1u) {
        let ds = sky.celestial_bodies[i].direction_size;
        let rad = ds.w*200.0;
        if rad <= 0.0 { continue; }
        let clip = u.view_proj * vec4<f32>(ds.xyz, 0.0);
        if clip.w <= 0.0 { continue; }
        let dn = ndc - clip.xy / clip.w;
        let size_px = rad / pix;

        // star-like point for unresolved bodies (magnitude based)
        let glow_w = 1.0 - smoothstep(1.0, 6.0, size_px);
        if glow_w > 0.001 {
            let sigma = 0.85 * pix;
            let x = length(dn) / sigma;
            if x < 60.0 {
                let psf = exp(-0.5 * x * x) + 0.04 / pow(1.0 + x * x / 9.0, 1.5);
                let fade = 1.0 - smoothstep(30.0, 60.0, x);
                var tint = vec3<f32>(1.0);
                if i < PL_PLANET_COUNT { tint = PL_TINT[i]; }
                res.glow += tint * pl_point_intensity(i) * glow_w * night_vis * psf * fade;
            }
        }

        if size_px < 0.35 { continue; }
        let q = dn / rad;
        let r2 = dot(q, q);

        if i == PB_SATURN {
            sat_q = q;
            sat_size = size_px;
        }
        if size_px > 4.0 && r2 > 1.0 && r2 < 2.25 {
            res.glow += pl_halo(i, q, r2, size_px, cam_right, cam_up);
        }

        let ew = min(0.75 / size_px, 0.9);
        let lim = 1.0 + ew;
        if r2 < lim * lim {
            let cov = (1.0 - smoothstep(1.0 - ew, 1.0 + ew, sqrt(r2))) * smoothstep(0.35, 1.5, size_px);
            if cov > 0.002 {
                let p = pl_system(i);
                let zf = sqrt(max(1.0 - r2, 0.0));
                let base = pl_distance(p);
                let local = dot(pl_rel(i), pl_dir(p)) - zf * pl_radius_km(i);
                if best == PL_NONE || ((base - best_base) + (local - best_local) < 0.0) {
                    best = i;
                    best_q = q;
                    best_r2 = r2;
                    best_cov = cov;
                    best_size = size_px;
                    best_base = base;
                    best_local = local;
                }
            }
        }
    }

    let ring_possible = sat_size >= 0.35 && dot(sat_q, sat_q) < 6.3;
    if best == PL_NONE && !ring_possible { return res; }

    let rot = pl_ecl_to_local();
    var acc_rgb = vec3<f32>(0.0);
    var acc_a = 0.0;

    if best != PL_NONE {
        let body_col = pl_shade_body(best, best_q, min(best_r2, 1.0), 1.0 / best_size, rot);
        acc_rgb = body_col * best_cov;
        acc_a = best_cov;
    }

    if ring_possible {
        let ring = pl_ring_layer(sat_q, 1.0 / sat_size, rot);
        let ra = ring.alpha * smoothstep(0.35, 1.5, sat_size);
        if ra > 0.0 {
            var in_front = true;
            if best != PL_NONE {
                in_front = (pl_distance(PB_SATURN) - best_base) + (ring.depth - best_local) < 0.0;
            }
            if in_front {
                acc_rgb = ring.rgb * ra + acc_rgb * (1.0 - ra);
                acc_a = ra + acc_a * (1.0 - ra);
            } else {
                acc_rgb = acc_rgb + ring.rgb * ra * (1.0 - acc_a);
                acc_a = acc_a + ra * (1.0 - acc_a);
            }
        }
    }

    res.rgb = acc_rgb;
    res.alpha = acc_a;
    return res;
}

@fragment
fn fs_main(input: VSOut) -> FragOut {
    var out: FragOut;

    let view_dir = normalize(input.dir);
    let sun_dir = normalize(u.sun_direction);
    let moon_dir = normalize(u.moon_direction);
    let sun_alt = sun_dir.y;
    let sun_elevation_rad = asin(clamp(sun_alt, -1.0, 1.0));
    let sun_elevation_degrees = sun_elevation_rad * 57.2957795;
    let sun_above_degrees = max(sun_elevation_degrees, 0.0);
    let sun_below_degrees = max(-sun_elevation_degrees, 0.0);

    let twilight_visibility =
        (1.0 - smoothstep(0.0, 18.0, sun_above_degrees))
        * exp(-sun_below_degrees / 6.5);
    let ndc = input.ndc;

    // pixel size in NDC (must be sampled in uniform control flow)
    let pixel_ndc = max(max(fwidth(ndc.x), fwidth(ndc.y)), 0.000001);
    let day_w = smoothstep(-0.22, 0.06, sun_alt);

    var col = vec3<f32>(0.0);

    let sun_clip = u.view_proj * vec4<f32>(sun_dir, 0.0);
    let moon_clip = u.view_proj * vec4<f32>(moon_dir, 0.0);

    let sun_visible = sun_clip.w > 0.0;
    let moon_visible = moon_clip.w > 0.0;

    var sun_ndc = vec2<f32>(0.0);
    var moon_ndc = vec2<f32>(0.0);

    if sun_visible {
        sun_ndc = sun_clip.xy / sun_clip.w;
    }

    if moon_visible {
        moon_ndc = moon_clip.xy / moon_clip.w;
    }

    let eclipse_coverage = saturate(u.solar_eclipse_coverage);
    let eclipse_sky_factor = 1.0 - 0.55 * pow(eclipse_coverage, 0.8);

    var moon_rel = vec2<f32>(0.0);
    var moon_r2 = 1000000.0;
    var moon_coverage = 0.0;
    var moon_surface_col = vec3<f32>(0.0);

    if moon_visible && sky.moon_size > 0.000001 {
        moon_rel = (ndc - moon_ndc) / sky.moon_size;
        moon_r2 = dot(moon_rel, moon_rel);

        let moon_distance = sqrt(max(moon_r2, 0.0));
        let edge_width = max(fwidth(moon_distance), 0.00001);

        moon_coverage = 1.0 - smoothstep(
            1.0 - edge_width,
            1.0 + edge_width,
            moon_distance
        );

        if moon_r2 <= 1.0 {
            let z = sqrt(max(1.0 - moon_r2, 0.0));
            let n_local = vec3<f32>(moon_rel.x, moon_rel.y, z);

            let view_ray = -view_dir;
            let up = select(
                vec3<f32>(1.0, 0.0, 0.0),
                vec3<f32>(0.0, 1.0, 0.0),
                abs(view_ray.y) <= 0.95
            );

            let right = normalize(cross(up, view_ray));
            let up_local = cross(view_ray, right);

            let n = normalize(
                n_local.x * right
                + n_local.y * up_local
                + n_local.z * view_ray
            );

            let solar_dot = dot(n, sun_dir);

            let phase = smoothstep(-0.008, 0.008, solar_dot);
            let dark_side = 1.0 - phase;
            let limb = mix(1.0, 0.68, sqrt(moon_r2));

            let maria_field = fbm3(
                n * 4.5 + vec3<f32>(3.4, 8.2, -2.7)
            );

            let maria = smoothstep(0.43, 0.62, maria_field);

            let terrain = fbm3(n * 22.0);
            let fine_detail = fbm3(n * 42.0);

            let px = edge_width / max(z, 0.25);
            let craters = crater_field(n, px, maria);

            let highlands =
                0.77
                + (terrain - 0.5) * 0.16
                + (fine_detail - 0.5) * 0.06;

            let mare =
                0.55
                + (terrain - 0.5) * 0.12
                + (fine_detail - 0.5) * 0.04;

            var albedo = mix(highlands, mare, maria);

            albedo += craters.bright * CRATER_BRIGHT_GAIN;
            albedo -= craters.floor_dark * 0.05;
            albedo = clamp(albedo, 0.28, 1.0);

            let surface_tint = mix(
                vec3<f32>(0.96, 0.98, 1.0),
                vec3<f32>(1.0, 0.95, 0.88),
                maria * 0.35
            );

            let lunar_normal = crater_surface_normal(n, craters.grad);

            var shadow_visibility = 1.0;

            if solar_dot > -0.12 {
                shadow_visibility = lunar_self_shadow(
                    n,
                    sun_dir,
                    craters.height,
                    px,
                    maria
                );
            }

            let horizon = -sqrt(2.0 * max(craters.height, 0.0));

            let lit = smoothstep(
                horizon - 0.006,
                horizon + 0.006,
                solar_dot
            );

            let diffuse = pow(
                max(dot(lunar_normal, sun_dir), 0.0),
                0.72
            );

            let direct_light = diffuse * shadow_visibility * lit;

            var moon_col = vec3<f32>(albedo)
                * surface_tint
                * sky.moon_intensity
                * direct_light;

            let earthshine = vec3<f32>(0.055, 0.062, 0.078)
                * dark_side
                * (0.55 + albedo * 0.5);

            moon_col += earthshine;
            moon_surface_col = moon_col * limb;
        }

        let halo_distance = sqrt(max(moon_r2, 0.0));

        if halo_distance < 4.0 {
            let t = 1.0 - halo_distance / 4.0;
            let t2 = t * t;

            col += vec3<f32>(0.45, 0.47, 0.55)
                * (t2 * 0.9 + t2 * t2 * 0.25)
                * sky.moon_intensity
                * saturate(moon_dir.y + 0.1)
                * 0.1;
        }
    }

    let h = smoothstep(0.0, 0.28, max(sun_alt, 0.0));

    let solar_base = mix(
        vec3<f32>(1.18, 0.50, 0.23),
        vec3<f32>(1.0, 0.97, 0.90),
        h
    );

    let air_mass_elevation_degrees = max(sun_elevation_degrees, 0.0);

    let air_mass = 1.0 / max(
        max(sun_alt, 0.0)
        + 0.50572 * pow(
            air_mass_elevation_degrees + 6.07995,
            -1.6364
        ),
        0.026
    );

    let extinction = exp(
        -vec3<f32>(0.025, 0.055, 0.11) * (air_mass - 1.0)
    );

    let sun_color = solar_base
        * mix(vec3<f32>(1.0), extinction, 0.32);

    var sun_halo_col = vec3<f32>(0.0);
    var sun_corona_col = vec3<f32>(0.0);
    if sun_visible {
        let radius = max(sky.sun_size, 0.000001);
        let sun_delta = ndc - sun_ndc;
        let d = length(sun_delta);
        let sun_uv = sun_delta / radius;

        let intensity = max(sky.sun_intensity, 0.0);
        let core_r = radius * 0.85;
        let halo_r = radius * 24.0;

        sun_corona_col = solar_corona(
            sun_uv,
            intensity,
            eclipse_coverage
        );

        if d < halo_r {
            let ht = 1.0 - d / halo_r;
            let ht2 = ht * ht;
            let ht4 = ht2 * ht2;

            let halo_col = mix(
                vec3<f32>(1.25, 0.42, 0.14),
                vec3<f32>(1.05, 0.88, 0.72),
                h
            );

            sun_halo_col += halo_col
                * (ht4 * ht2 * 0.9 + ht4 * ht4 * 0.35)
                * intensity
                * 2.8
                * exp(-sun_below_degrees / 6.5);

            if d < radius {
                if d < core_r {
                    let x = 1.0 - d / core_r;
                    let uv = sun_uv;

                    let radial = dot(uv, uv);
                    let solar_mu = sqrt(max(1.0 - radial, 0.0));

                    let granulation = fbm3(
                        vec3<f32>(uv * 18.0, 2.73)
                    );

                    let fine_granulation = fbm3(
                        vec3<f32>(uv * 43.0, 8.17)
                    );

                    let micro_granulation = fbm3(
                        vec3<f32>(uv * 79.0, 14.31)
                    );

                    let spot_a = solar_spot(
                        uv,
                        vec2<f32>(-0.32, 0.19),
                        vec2<f32>(0.095, 0.065)
                    );

                    let spot_b = solar_spot(
                        uv,
                        vec2<f32>(-0.22, 0.24),
                        vec2<f32>(0.043, 0.032)
                    );

                    let spot_c = solar_spot(
                        uv,
                        vec2<f32>(0.31, -0.24),
                        vec2<f32>(0.072, 0.052)
                    );

                    let spot_d = solar_spot(
                        uv,
                        vec2<f32>(0.40, -0.19),
                        vec2<f32>(0.035, 0.026)
                    );

                    let spot_e = solar_spot(
                        uv,
                        vec2<f32>(0.48, 0.20),
                        vec2<f32>(0.033, 0.045)
                    );

                    let spots = max(
                        max(spot_a, spot_b),
                        max(max(spot_c, spot_d), spot_e)
                    );

                    let faculae = smoothstep(
                        0.55,
                        0.76,
                        fine_granulation
                    ) * (1.0 - spots);

                    let granule_structure =
                        (granulation - 0.5) * 0.34
                        + (fine_granulation - 0.5) * 0.13
                        + (micro_granulation - 0.5) * 0.045;

                    let solar_texture = max(
                        0.1,
                        1.0
                        + granule_structure * 1.2
                        + faculae * 0.095
                        - spots * 0.72
                    );

                    let limb_darkening =
                        0.68 + 0.32 * pow(solar_mu, 0.65);

                    let photosphere_color = mix(
                        vec3<f32>(1.0, 0.78, 0.52),
                        vec3<f32>(1.0, 0.94, 0.79),
                        h
                    );

                    let core_profile =
                        0.65 + 0.45 * pow(x, 0.35);

                    col += sun_color
                        * photosphere_color
                        * limb_darkening
                        * solar_texture
                        * core_profile
                        * intensity
                        * 2.5;

                } else {
                    let t = 1.0 - (d - core_r) / (radius - core_r);
                    let t2 = t * t;

                    col += sun_color
                        * t2 * t2 * t2
                        * intensity;
                }
            }
        }
    }

    let mu = view_dir.y;
    let view_up = mu * 0.5 + 0.5;

    let day_col = mix(
        vec3<f32>(0.45, 0.70, 0.98),
        vec3<f32>(0.06, 0.32, 0.80),
        pow(view_up, 0.55)
    );

    let moon_lum = sky.moon_intensity
        * saturate(sky.moon_phase)
        * saturate(moon_dir.y + 0.1);

    let mu_p = max(mu, 0.0);
    let mu_p3 = mu_p * mu_p * mu_p;
    let inv_mu = 1.0 - mu_p;

    let night_col =
        vec3<f32>(0.015, 0.02, 0.03) * moon_lum * mu_p3 * 0.25
        + vec3<f32>(0.01, 0.015, 0.02) * inv_mu * inv_mu
        + vec3<f32>(0.008, 0.012, 0.006) * mu_p3 * mu_p;

    let horizon_mask = smoothstep(-0.6, 0.1, mu);

    var sky_col = mix(night_col, day_col, day_w)
        * horizon_mask;

    let sun_h_raw = vec3<f32>(sun_dir.x, 0.0, sun_dir.z);
    let sun_h = sun_h_raw / max(length(sun_h_raw), 0.0001);

    let view_h_raw = vec3<f32>(view_dir.x, 0.0, view_dir.z);
    let view_dir_h = view_h_raw / max(length(view_h_raw), 0.0001);

    let toward_sun = saturate(dot(view_dir_h, sun_h));
    let away_from_sun = saturate(dot(view_dir_h, -sun_h));

    let horizon_band =
        exp(-pow(max(mu, 0.0) / 0.28, 2.0))
        * horizon_mask;

    let sunward_haze =
        horizon_band
        * twilight_visibility
        * pow(toward_sun, 2.0)
        * 0.72;

    sky_col = mix(
        sky_col,
        vec3<f32>(1.20, 0.30, 0.07),
        sunward_haze
    );

    let antisolar_haze =
        horizon_band
        * twilight_visibility
        * pow(away_from_sun, 2.0)
        * 0.34;

    sky_col = mix(
        sky_col,
        vec3<f32>(0.34, 0.11, 0.25),
        antisolar_haze
    );

    let upper_twilight_band =
        exp(-pow((mu - 0.20) / 0.23, 2.0))
        * horizon_mask
        * twilight_visibility;

    sky_col = mix(
        sky_col,
        vec3<f32>(0.17, 0.09, 0.27),
        upper_twilight_band * 0.25
    );

    sky_col *= eclipse_sky_factor;

    let dither = fract(
        sin(dot(ndc * 50.0, vec2<f32>(127.1, 311.7)))
        * 43758.5453
    );

    sky_col += (dither - 0.5) * 0.00392;

    // planets, moons and rings: composited over the sun (transits) and
    // under Earth's Moon (occultations), then washed by the sky like the rest
    let planets = pl_render(ndc, pixel_ndc, day_w);

    col += sun_halo_col * eclipse_coverage;
    col += sun_corona_col * eclipse_coverage;
    col += planets.glow;
    col = col * (1.0 - planets.alpha) + planets.rgb;
    col = mix(col, moon_surface_col, moon_coverage);
    col += max(sky_col, vec3<f32>(0.0));
    col += sun_halo_col * (1.0 - eclipse_coverage);

    let mapped = 1.0 - exp(-col * max(sky.exposure, 0.00001));

    let lum = dot(mapped, vec3<f32>(0.2126, 0.7152, 0.0722));
    let blue = saturate((mapped.b - max(mapped.r, mapped.g)) * 4.0);

    let base_alpha = saturate(lum + blue * (1.0 - lum));
    let body_alpha = 1.0 - (1.0 - moon_coverage) * (1.0 - planets.alpha);
    let final_alpha = mix(base_alpha, 1.0, body_alpha);

    out.color = vec4<f32>(mapped, final_alpha);

    let curr_ndc = input.ndc;
    let prev_ndc = input.prev_clip.xy / input.prev_clip.w;

    let curr_uv = curr_ndc * vec2<f32>(0.5, -0.5) + 0.5;
    let prev_uv = prev_ndc * vec2<f32>(0.5, -0.5) + 0.5;

    out.motion = curr_uv - prev_uv;

    return out;
}

fn hash3(p: vec3<f32>) -> f32 {
    return fract(
        sin(dot(p, vec3<f32>(127.1, 311.7, 74.7)))
        * 43758.5453
    );
}

fn value_noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (vec3<f32>(3.0) - 2.0 * f);

    let x00 = mix(
        hash3(i + vec3<f32>(0.0, 0.0, 0.0)),
        hash3(i + vec3<f32>(1.0, 0.0, 0.0)),
        u.x
    );

    let x10 = mix(
        hash3(i + vec3<f32>(0.0, 1.0, 0.0)),
        hash3(i + vec3<f32>(1.0, 1.0, 0.0)),
        u.x
    );

    let x01 = mix(
        hash3(i + vec3<f32>(0.0, 0.0, 1.0)),
        hash3(i + vec3<f32>(1.0, 0.0, 1.0)),
        u.x
    );

    let x11 = mix(
        hash3(i + vec3<f32>(0.0, 1.0, 1.0)),
        hash3(i + vec3<f32>(1.0, 1.0, 1.0)),
        u.x
    );

    return mix(
        mix(x00, x10, u.y),
        mix(x01, x11, u.y),
        u.z
    );
}

fn fbm3(p: vec3<f32>) -> f32 {
    var q = p;
    var sum = 0.0;
    var amplitude = 0.5;
    var normalization = 0.0;

    for (var i = 0u; i < 4u; i = i + 1u) {
        sum += value_noise3(q) * amplitude;
        normalization += amplitude;

        q = q * 2.03 + vec3<f32>(13.1, 7.7, 17.3);
        amplitude *= 0.5;
    }

    return sum / normalization;
}

const LUNAR_PI: f32 = 3.14159265358979;
const LUNAR_HALF_PI: f32 = 1.5707963267949;

const CRATER_OCTAVES: u32 = 7u;
const CRATER_BASE_CELLS: f32 = 8.0;
const CRATER_SLOPE: f32 = 2.15;
const CRATER_DENSITY: f32 = 0.40;
const CRATER_RHO_MIN: f32 = 0.12;
const CRATER_RHO_MAX: f32 = 0.50;
const CRATER_REACH: f32 = 1.6;
const CRATER_RELIEF: f32 = 1.5;
const CRATER_BRIGHT_GAIN: f32 = 0.20;
const CRATER_MAX_RELIEF: f32 = 0.009;

const SHADOW_STEPS: u32 = 12u;
const SUN_ANGULAR_DIAMETER: f32 = 0.0093;

struct CraterField {
    height: f32,
    grad: vec3<f32>,
    bright: f32,
    floor_dark: f32,
};

fn crater_pcg(v_in: vec3<u32>) -> vec3<f32> {
    var v = v_in * 1664525u + vec3<u32>(1013904223u);

    v.x = v.x + v.y * v.z;
    v.y = v.y + v.z * v.x;
    v.z = v.z + v.x * v.y;

    v = v ^ (v >> vec3<u32>(16u));

    v.x = v.x + v.y * v.z;
    v.y = v.y + v.z * v.x;
    v.z = v.z + v.x * v.y;

    return vec3<f32>(v >> vec3<u32>(8u)) * (1.0 / 16777216.0);
}

fn cube_face_coords(n: vec3<f32>, k: i32) -> vec3<f32> {
    if (k == 0) {
        return vec3<f32>(n.y, n.z, n.x);
    }

    if (k == 1) {
        return vec3<f32>(n.z, n.x, n.y);
    }

    return n;
}

fn cube_face_dir(u: f32, v: f32, w: f32, k: i32) -> vec3<f32> {
    if (k == 0) {
        return vec3<f32>(w, u, v);
    }

    if (k == 1) {
        return vec3<f32>(v, w, u);
    }

    return vec3<f32>(u, v, w);
}

fn crater_face_angles(
    n: vec3<f32>,
    face_angles: ptr<function, array<vec2<f32>, 3>>,
    face_signs: ptr<function, array<f32, 3>>,
    face_valid: ptr<function, array<u32, 3>>
) {
    for (var k = 0; k < 3; k = k + 1) {
        let index = u32(k);
        let fc = cube_face_coords(n, k);

        (*face_valid)[index] = 0u;
        (*face_angles)[index] = vec2<f32>(0.0);
        (*face_signs)[index] = 1.0;

        if (abs(fc.z) < 0.3) {
            continue;
        }

        let aw = abs(fc.z);

        (*face_angles)[index] = vec2<f32>(
            atan2(fc.x, aw),
            atan2(fc.y, aw)
        );

        (*face_signs)[index] = select(
            -1.0,
            1.0,
            fc.z >= 0.0
        );

        (*face_valid)[index] = 1u;
    }
}

fn crater_field(
    n: vec3<f32>,
    px_in: f32,
    mare: f32
) -> CraterField {
    let px = max(px_in, 0.00001);
    let first_cell_w = LUNAR_HALF_PI / CRATER_BASE_CELLS;

    var result: CraterField;
    result.height = 0.0;
    result.grad = vec3<f32>(0.0);
    result.bright = 0.0;
    result.floor_dark = 0.0;

    if (2.0 * CRATER_RHO_MAX * first_cell_w < 0.9 * px) {
        return result;
    }

    var bowl_h = 0.0;
    var bowl_g = vec3<f32>(0.0);
    var rim_h = 0.0;
    var rim_g = vec3<f32>(0.0);
    var keep = 1.0;
    var keep_g = vec3<f32>(0.0);
    var bright = 0.0;
    var floor_dark = 0.0;

    let rho_lo = pow(CRATER_RHO_MIN, -CRATER_SLOPE);
    let rho_hi = pow(CRATER_RHO_MAX, -CRATER_SLOPE);

    var face_angles: array<vec2<f32>, 3>;
    var face_signs: array<f32, 3>;
    var face_valid: array<u32, 3>;

    crater_face_angles(
        n,
        &face_angles,
        &face_signs,
        &face_valid
    );

    for (var oct = 0u; oct < CRATER_OCTAVES; oct = oct + 1u) {
        let cells = CRATER_BASE_CELLS * exp2(f32(oct));
        let cell_w = LUNAR_HALF_PI / cells;

        if (2.0 * CRATER_RHO_MAX * cell_w < 0.9 * px) {
            break;
        }

        let occ = clamp(
            CRATER_DENSITY * exp2(f32(oct) * (CRATER_SLOPE - 2.0)),
            0.0,
            1.0
        );

        for (var k = 0; k < 3; k = k + 1) {
            let face_index = u32(k);

            if (face_valid[face_index] == 0u) {
                continue;
            }

            let ang = face_angles[face_index];
            let s = face_signs[face_index];

            let g = (ang * (2.0 / LUNAR_PI) + 0.5) * cells;
            let base = vec2<i32>(floor(g));

            let face = face_index * 2u + select(0u, 1u, s < 0.0);

            for (var dy = -1; dy <= 1; dy = dy + 1) {
                for (var dx = -1; dx <= 1; dx = dx + 1) {
                    let cx = base.x + dx;
                    let cy = base.y + dy;

                    if (
                        cx < 0
                        || cy < 0
                        || f32(cx) >= cells
                        || f32(cy) >= cells
                    ) {
                        continue;
                    }

                    let key = vec3<u32>(
                        u32(cx),
                        u32(cy),
                        face * 16u + oct
                    );

                    let h = crater_pcg(key);

                    if (h.z > occ) {
                        continue;
                    }

                    let jitter = mix(
                        vec2<f32>(0.5),
                        h.xy,
                        0.7
                    );

                    let gc = vec2<f32>(f32(cx), f32(cy)) + jitter;

                    let th = (gc / cells * 2.0 - 1.0)
                        * (LUNAR_PI * 0.25);

                    let c = normalize(
                        cube_face_dir(
                            tan(th.x),
                            tan(th.y),
                            s,
                            k
                        )
                    );

                    let h2 = crater_pcg(
                        key + vec3<u32>(
                            0x9E3779B9u,
                            0x7F4A7C15u,
                            0x85EBCA6Bu
                        )
                    );

                    let rho = pow(
                        mix(rho_lo, rho_hi, h2.x),
                        -1.0 / CRATER_SLOPE
                    );

                    let r_w = rho * cell_w;
                    let dvec = n - c;
                    let d2 = dot(dvec, dvec);
                    let x2 = d2 / (r_w * r_w);

                    if (x2 >= CRATER_REACH * CRATER_REACH) {
                        continue;
                    }

                    let dd = 2.0 * r_w;

                    let aa = smoothstep(
                        0.8,
                        2.5,
                        dd / px
                    );

                    if (aa <= 0.0) {
                        continue;
                    }

                    let dist = sqrt(x2);
                    let dir = dvec / max(sqrt(d2), 0.000001);

                    let age = h2.y;
                    let fresh = pow(1.0 - age, 4.0);
                    let old = smoothstep(0.15, 0.40, age);

                    let cplx = smoothstep(0.009, 0.03, dd);

                    let ratio = 0.2
                        * pow(max(dd / 0.009, 1.0), -0.6)
                        * mix(0.85, 1.15, h2.z);

                    let survive = mix(
                        1.0,
                        smoothstep(0.015, 0.07, dd),
                        mare * old
                    );

                    let vis = aa * survive;

                    let degr = saturate(
                        age * pow(0.03 / dd, 0.7) * 1.2
                    );

                    let dep = ratio
                        * mix(1.0, 0.18, degr)
                        * CRATER_RELIEF
                        * vis;

                    let rim = 0.22 * dep;

                    let tc = saturate((dist - 0.85) / 0.15);

                    let cover = vis
                        * (1.0 - tc * tc * (3.0 - 2.0 * tc));

                    let dcover = -vis
                        * 6.0
                        * tc
                        * (1.0 - tc)
                        / (0.15 * r_w);

                    keep_g = keep_g * (1.0 - cover)
                        - keep * dcover * dir;

                    keep = keep * (1.0 - cover);

                    if (dist < 1.0) {
                        let xf = 0.5 * cplx;
                        let t = saturate((dist - xf) / (1.0 - xf));
                        let pw = mix(2.0, 1.5, cplx);
                        let soften = 0.8 * degr;

                        let w_sharp = pow(t, pw);
                        let w_soft = t * t * (3.0 - 2.0 * t);

                        var s_h = (rim - dep)
                            + dep * mix(w_sharp, w_soft, soften);

                        var s_d = 0.0;

                        if (dist > xf) {
                            let dw_sharp = pw * pow(t, pw - 1.0);
                            let dw_soft = 6.0 * t * (1.0 - t);

                            s_d = dep
                                * mix(dw_sharp, dw_soft, soften)
                                / (1.0 - xf);
                        }

                        let pk = 0.35
                            * dep
                            * smoothstep(0.45, 0.9, cplx);

                        let q = dist / 0.12;
                        let pe = pk * exp(-q * q);

                        s_h += pe;
                        s_d -= pe * 2.0 * q / 0.12;

                        bowl_h += dd * s_h;
                        bowl_g += 2.0 * s_d * dir;

                        let fd = (1.0 - fresh)
                            * cplx
                            * vis
                            * (1.0 - smoothstep(0.55, 0.95, dist));

                        floor_dark = max(floor_dark, fd);
                        bright = max(bright, fresh * aa);
                    } else {
                        let tm = saturate(
                            (dist - 1.0) / (CRATER_REACH - 1.0)
                        );

                        let m = 1.0 - tm * tm * (3.0 - 2.0 * tm);

                        let dm = -6.0
                            * tm
                            * (1.0 - tm)
                            / (CRATER_REACH - 1.0);

                        let xi = 1.0 / dist;
                        let x3 = xi * xi * xi;

                        let s_h = rim * x3 * m;
                        let s_d = rim
                            * (-3.0 * x3 * xi * m + x3 * dm);

                        rim_h += dd * s_h;
                        rim_g += 2.0 * s_d * dir;

                        bright = max(bright, fresh * aa * m);
                    }
                }
            }
        }
    }

    result.height = bowl_h + rim_h * keep;
    result.grad = bowl_g + rim_g * keep + rim_h * keep_g;
    result.bright = bright;
    result.floor_dark = floor_dark;

    return result;
}

fn crater_shadow_height(
    n: vec3<f32>,
    px_in: f32,
    mare: f32
) -> f32 {
    let px = max(px_in, 0.00001);
    let first_cell_w = LUNAR_HALF_PI / CRATER_BASE_CELLS;

    if (2.0 * CRATER_RHO_MAX * first_cell_w < 0.9 * px) {
        return 0.0;
    }

    var bowl_h = 0.0;
    var rim_h = 0.0;
    var keep = 1.0;

    let rho_lo = pow(CRATER_RHO_MIN, -CRATER_SLOPE);
    let rho_hi = pow(CRATER_RHO_MAX, -CRATER_SLOPE);

    var face_angles: array<vec2<f32>, 3>;
    var face_signs: array<f32, 3>;
    var face_valid: array<u32, 3>;

    crater_face_angles(
        n,
        &face_angles,
        &face_signs,
        &face_valid
    );

    for (var oct = 0u; oct < CRATER_OCTAVES; oct = oct + 1u) {
        let cells = CRATER_BASE_CELLS * exp2(f32(oct));
        let cell_w = LUNAR_HALF_PI / cells;

        if (2.0 * CRATER_RHO_MAX * cell_w < 0.9 * px) {
            break;
        }

        let occ = clamp(
            CRATER_DENSITY * exp2(f32(oct) * (CRATER_SLOPE - 2.0)),
            0.0,
            1.0
        );

        for (var k = 0; k < 3; k = k + 1) {
            let face_index = u32(k);

            if (face_valid[face_index] == 0u) {
                continue;
            }

            let ang = face_angles[face_index];
            let s = face_signs[face_index];

            let g = (ang * (2.0 / LUNAR_PI) + 0.5) * cells;
            let base = vec2<i32>(floor(g));

            let face = face_index * 2u + select(0u, 1u, s < 0.0);

            for (var dy = -1; dy <= 1; dy = dy + 1) {
                for (var dx = -1; dx <= 1; dx = dx + 1) {
                    let cx = base.x + dx;
                    let cy = base.y + dy;

                    if (
                        cx < 0
                        || cy < 0
                        || f32(cx) >= cells
                        || f32(cy) >= cells
                    ) {
                        continue;
                    }

                    let key = vec3<u32>(
                        u32(cx),
                        u32(cy),
                        face * 16u + oct
                    );

                    let h = crater_pcg(key);

                    if (h.z > occ) {
                        continue;
                    }

                    let jitter = mix(
                        vec2<f32>(0.5),
                        h.xy,
                        0.7
                    );

                    let gc = vec2<f32>(f32(cx), f32(cy)) + jitter;

                    let th = (gc / cells * 2.0 - 1.0)
                        * (LUNAR_PI * 0.25);

                    let c = normalize(
                        cube_face_dir(
                            tan(th.x),
                            tan(th.y),
                            s,
                            k
                        )
                    );

                    let h2 = crater_pcg(
                        key + vec3<u32>(
                            0x9E3779B9u,
                            0x7F4A7C15u,
                            0x85EBCA6Bu
                        )
                    );

                    let rho = pow(
                        mix(rho_lo, rho_hi, h2.x),
                        -1.0 / CRATER_SLOPE
                    );

                    let r_w = rho * cell_w;
                    let dvec = n - c;
                    let d2 = dot(dvec, dvec);
                    let x2 = d2 / (r_w * r_w);

                    if (x2 >= CRATER_REACH * CRATER_REACH) {
                        continue;
                    }

                    let dd = 2.0 * r_w;

                    let aa = smoothstep(
                        0.8,
                        2.5,
                        dd / px
                    );

                    if (aa <= 0.0) {
                        continue;
                    }

                    let dist = sqrt(x2);
                    let age = h2.y;
                    let old = smoothstep(0.15, 0.40, age);
                    let cplx = smoothstep(0.009, 0.03, dd);

                    let ratio = 0.2
                        * pow(max(dd / 0.009, 1.0), -0.6)
                        * mix(0.85, 1.15, h2.z);

                    let survive = mix(
                        1.0,
                        smoothstep(0.015, 0.07, dd),
                        mare * old
                    );

                    let vis = aa * survive;

                    let degr = saturate(
                        age * pow(0.03 / dd, 0.7) * 1.2
                    );

                    let dep = ratio
                        * mix(1.0, 0.18, degr)
                        * CRATER_RELIEF
                        * vis;

                    let rim = 0.22 * dep;

                    let tc = saturate((dist - 0.85) / 0.15);

                    let cover = vis
                        * (1.0 - tc * tc * (3.0 - 2.0 * tc));

                    keep *= 1.0 - cover;

                    if (dist < 1.0) {
                        let xf = 0.5 * cplx;
                        let t = saturate((dist - xf) / (1.0 - xf));
                        let pw = mix(2.0, 1.5, cplx);
                        let soften = 0.8 * degr;

                        let w_sharp = pow(t, pw);
                        let w_soft = t * t * (3.0 - 2.0 * t);

                        var s_h = (rim - dep)
                            + dep * mix(w_sharp, w_soft, soften);

                        let pk = 0.35
                            * dep
                            * smoothstep(0.45, 0.9, cplx);

                        let q = dist / 0.12;
                        let pe = pk * exp(-q * q);

                        s_h += pe;
                        bowl_h += dd * s_h;
                    } else {
                        let tm = saturate(
                            (dist - 1.0) / (CRATER_REACH - 1.0)
                        );

                        let m = 1.0 - tm * tm * (3.0 - 2.0 * tm);

                        let xi = 1.0 / dist;
                        let x3 = xi * xi * xi;

                        let s_h = rim * x3 * m;

                        rim_h += dd * s_h;
                    }
                }
            }
        }
    }

    return bowl_h + rim_h * keep;
}

fn crater_surface_normal(n: vec3<f32>, grad: vec3<f32>) -> vec3<f32> {
    let g_t = grad - n * dot(grad, n);
    return normalize(n - g_t);
}

fn lunar_self_shadow(
    n: vec3<f32>,
    sun_dir: vec3<f32>,
    height_here: f32,
    px: f32,
    mare: f32
) -> f32 {
    let solar_dot = dot(n, sun_dir);
    let sun_tangent = sun_dir - n * solar_dot;
    let tangent_length = length(sun_tangent);

    if (tangent_length < 0.0001) {
        return 1.0;
    }

    let tangent_dir = sun_tangent / tangent_length;
    let light_slope = solar_dot / tangent_length;

    let t_start = max(px * 1.5, 0.0003);

    let t_max = clamp(
        CRATER_MAX_RELIEF / max(light_slope, 0.02),
        t_start * 4.0,
        0.10
    );

    let ratio = pow(
        t_max / t_start,
        1.0 / f32(SHADOW_STEPS)
    );

    var visibility = 1.0;
    var t = t_start;

    for (var i = 0u; i < SHADOW_STEPS; i = i + 1u) {
        let sample_dir = normalize(n + tangent_dir * t);
        let shadow_px = max(px, 0.65 * t);

        let sample_h = crater_shadow_height(
            sample_dir,
            shadow_px,
            mare
        );

        let ray_h = height_here + t * light_slope + 0.5 * t * t;

        let soft = t * SUN_ANGULAR_DIAMETER
            + 0.5 * px * clamp(light_slope, 0.05, 1.0);

        visibility = min(
            visibility,
            smoothstep(-soft, soft, ray_h - sample_h)
        );

        if (visibility < 0.01) {
            break;
        }

        t *= ratio;
    }

    return visibility;
}
