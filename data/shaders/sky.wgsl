// ------------------------------------------------------------
// Bind groups
// ------------------------------------------------------------

struct Uniforms {
    view_proj: mat4x4<f32>,
    sun_direction: vec3<f32>,
    time: f32,
    camera_pos: vec3<f32>,
    orbit_radius: f32,
    moon_direction: vec3<f32>,
    _pad0: f32,
};

@group(0) @binding(0)
var<uniform> u: Uniforms;

struct SkyUniform {
    day_time: f32,
    day_length: f32,
    sun_size: f32,
    sun_intensity: f32,

    exposure: f32,
    moon_size: f32,
    moon_intensity: f32,
    moon_phase: f32,
};

@group(1) @binding(0)
var<uniform> sky: SkyUniform;


// ------------------------------------------------------------
// Vertex output
// ------------------------------------------------------------
struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) screen_uv: vec2<f32>,
};


// ------------------------------------------------------------
// Vertex shader: fullscreen triangle
// ------------------------------------------------------------
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VSOut {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );

    var out: VSOut;
    out.clip = vec4<f32>(positions[idx], 0.0, 1.0);
    out.screen_uv = positions[idx] * 0.5 + vec2<f32>(0.5);
    return out;
}


// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------
fn saturate(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

fn hash21(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    let s = sin(h) * 43758.5453;
    return fract(s);
}

// Project a *direction* to screen UV using an orbiting camera.
// Assumes the camera orbits a center that lies on the ray from
// world origin through camera_pos at distance orbit_radius.
fn project_direction_to_uv(dir: vec3<f32>) -> vec2<f32> {
    let len2 = dot(dir, dir);
    if len2 < 1e-5 {
        return vec2<f32>(9999.0, 9999.0);
    }

    let n = dir / sqrt(len2);

    // Reconstruct approximate orbit center from camera_pos and orbit_radius.
    // If the camera really orbits around the world origin, then
    // length(camera_pos) ≈ orbit_radius and this gives center ≈ (0,0,0).
    let cam_off = u.camera_pos;
    let cam_len2 = dot(cam_off, cam_off);
    var orbit_center = vec3<f32>(0.0, 0.0, 0.0);

    if cam_len2 > 1e-5 && u.orbit_radius > 1e-4 {
        let cam_len = sqrt(cam_len2);
        let cam_dir = cam_off / cam_len;
        orbit_center = u.camera_pos - cam_dir * u.orbit_radius;
    }

    // Place sun / moon on a big sphere around the orbit center so
    // their apparent position doesn't wobble when you orbit.
    let sky_radius = max(u.orbit_radius * 4.0, 1000.0);
    let world_pos = orbit_center + n * sky_radius;

    let clip = u.view_proj * vec4<f32>(world_pos, 1.0);

    if clip.w <= 0.0 {
        return vec2<f32>(9999.0, 9999.0);
    }

    let ndc = clip.xyz / clip.w;          // [-1, 1]
    return ndc.xy * 0.5 + vec2<f32>(0.5); // [0, 1]
}

fn compute_day_phase() -> f32 {
    let len = max(sky.day_length, 0.001);
    let phase = sky.day_time / len;
    return fract(phase);
}

// returns (day, twilight, night)
fn compute_day_weights(day_phase: f32) -> vec3<f32> {
    let noon_factor = 1.0 - saturate(abs(day_phase - 0.5) * 2.0);
    let day_weight = saturate(noon_factor * 1.35);

    let sunrise = exp(-pow((day_phase - 0.25) * 16.0, 2.0));
    let sunset  = exp(-pow((day_phase - 0.75) * 16.0, 2.0));
    let twilight_weight = saturate((sunrise + sunset) * 1.6);

    let night_weight = saturate(1.0 - day_weight - twilight_weight);

    return vec3<f32>(day_weight, twilight_weight, night_weight);
}


// ------------------------------------------------------------
// Sky color
// ------------------------------------------------------------
fn compute_sky_color(uv: vec2<f32>, day_phase: f32, weights: vec3<f32>) -> vec3<f32> {
    let day_w = weights.x;
    let twi_w = weights.y;
    let night_w = weights.z;

    let y = saturate(uv.y);
    let t_vert = saturate((y - 0.05) / 0.95);

    let zenith_day       = vec3<f32>(0.08, 0.45, 0.85);
    let horizon_day      = vec3<f32>(0.62, 0.78, 0.93);
    let zenith_twilight  = vec3<f32>(0.08, 0.18, 0.40);
    let horizon_twilight = vec3<f32>(1.10, 0.45, 0.15);

    let zenith_night     = vec3<f32>(0.02, 0.04, 0.10);
    let horizon_night    = vec3<f32>(0.03, 0.06, 0.12);

    let col_day      = mix(horizon_day,      zenith_day,      t_vert);
    let col_twilight = mix(horizon_twilight, zenith_twilight, t_vert);
    let col_night    = mix(horizon_night,    zenith_night,    t_vert);

    var sky_col = col_day * day_w +
                  col_twilight * twi_w +
                  col_night * night_w;

    let horizon_band = saturate(1.0 - y * 4.0);
    let warm_boost = vec3<f32>(1.2, 0.6, 0.3) * horizon_band * twi_w * 0.6;
    sky_col += warm_boost;

    let zenith_boost = saturate((y - 0.4) * 1.8);
    sky_col += vec3<f32>(0.02, 0.03, 0.05) * zenith_boost * day_w;

    return sky_col;
}


// ------------------------------------------------------------
// Sun
// ------------------------------------------------------------
fn compute_sun(day_phase: f32, uv: vec2<f32>, sun_uv: vec2<f32>, weights: vec3<f32>) -> vec3<f32> {
    let day_w = weights.x;
    let twi_w = weights.y;

    let d = distance(uv, sun_uv);

    let disk = 1.0 - smoothstep(sky.sun_size * 0.8, sky.sun_size * 1.2, d);
    let inner_glow = exp(-d * 220.0);
    let outer_glow = exp(-pow(d * 40.0, 2.0));

    let noon_factor = 1.0 - saturate(abs(day_phase - 0.5) * 2.0);
    let visibility = saturate(day_w + twi_w * 1.2);

    let sun_white = vec3<f32>(1.0, 0.98, 0.95);
    let sun_warm  = vec3<f32>(1.30, 0.75, 0.35);
    let base_col = mix(sun_warm, sun_white, saturate(noon_factor * 1.5));

    let intensity = sky.sun_intensity * visibility;

    let sun = base_col * intensity * (
        disk * 40.0 +
        inner_glow * 4.0 +
        outer_glow * 1.2
    );

    return sun;
}

// ------------------------------------------------------------
// Moon
// ------------------------------------------------------------
fn compute_moon(uv: vec2<f32>, moon_uv: vec2<f32>, weights: vec3<f32>) -> vec3<f32> {
    let day_w = weights.x;
    let twi_w = weights.y;
    let night_w = weights.z;

    if day_w > 0.9 {
        return vec3<f32>(0.0);
    }

    let d = distance(uv, moon_uv);

    let disk = 1.0 - smoothstep(sky.moon_size * 0.9, sky.moon_size * 1.1, d);

    let phase = saturate(sky.moon_phase);

    let base_col = vec3<f32>(0.8, 0.82, 0.9) * phase;

    let glow = exp(-d * 60.0);

    let visibility = saturate(twi_w * 0.6 + night_w * 1.0);

    let moon = base_col * (disk * 4.0 + glow * 0.7) * sky.moon_intensity * visibility;

    return moon;
}


// ------------------------------------------------------------
// Stars
// ------------------------------------------------------------
fn compute_stars(uv: vec2<f32>, weights: vec3<f32>, sun_uv: vec2<f32>, moon_uv: vec2<f32>) -> vec3<f32> {
    let night_w = weights.z;
    if night_w <= 0.001 {
        return vec3<f32>(0.0);
    }

    let p = floor(uv * 900.0);
    let rnd = hash21(p);
    let mask = step(0.9985, rnd);

    let brightness = (rnd - 0.9985) * 300.0;

    var star = vec3<f32>(0.9, 0.95, 1.0) * brightness * mask * night_w;

    let height = saturate((uv.y - 0.1) * 2.5);
    star *= height;

    let ds = distance(uv, sun_uv);
    let dm = distance(uv, moon_uv);

    let sun_block = saturate(1.0 - exp(-ds * 20.0));
    let moon_block = saturate(1.0 - exp(-dm * 12.0));

    star *= sun_block * moon_block;

    return star;
}


// ------------------------------------------------------------
// Tone mapping
// ------------------------------------------------------------
fn tone_map(color: vec3<f32>) -> vec3<f32> {
    let mapped = vec3<f32>(
        1.0 - exp(-color.x * sky.exposure),
        1.0 - exp(-color.y * sky.exposure),
        1.0 - exp(-color.z * sky.exposure),
    );
    return mapped;
}


// ------------------------------------------------------------
// Fragment shader
// ------------------------------------------------------------
@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let day_phase = compute_day_phase();
    let weights = compute_day_weights(day_phase);

    let sun_uv = project_direction_to_uv(u.sun_direction);
    let moon_uv = project_direction_to_uv(u.moon_direction);

    let base_sky = compute_sky_color(in.screen_uv, day_phase, weights);
    let sun = compute_sun(day_phase, in.screen_uv, sun_uv, weights);
    let moon = compute_moon(in.screen_uv, moon_uv, weights);
    let stars = compute_stars(in.screen_uv, weights, sun_uv, moon_uv);

    let hdr = base_sky + sun + moon + stars;
    let ldr = tone_map(hdr);

    return vec4<f32>(ldr, 1.0);
}
