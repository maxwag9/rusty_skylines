// ----------------------------------------
// SKY UNIFORMS
// ----------------------------------------
struct Uniforms {
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,

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

    exposure: f32,
    _pad0: f32,

    sun_size: f32,
    sun_intensity: f32,

    moon_size: f32,
    moon_intensity: f32,

    moon_phase: f32,
    _pad1: f32,
};

@group(1) @binding(0)
var<uniform> sky: SkyUniform;


// ----------------------------------------
// VERTEX
// ----------------------------------------
// fullscreen triangle
struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) dir: vec3<f32>,
    @location(1) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
    var clip: vec4<f32>;

    if vid == 0u {
        clip = vec4<f32>(-1.0, -1.0, 1.0, 1.0);
    } else if vid == 1u {
        clip = vec4<f32>( 3.0, -1.0, 1.0, 1.0);
    } else {
        clip = vec4<f32>(-1.0,  3.0, 1.0, 1.0);
    }

    // world direction reconstruction
    let inv_vp = u.inv_view_proj;
    let world_pos = inv_vp * clip;
    let dir = normalize(world_pos.xyz / world_pos.w - u.camera_pos);

    let ndc = clip.xy / clip.w;

    return VSOut(clip, dir, ndc);
}



// ----------------------------------------
// HELPERS
// ----------------------------------------
fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = clamp((x - a) / (b - a), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn hash3(p: vec3<f32>) -> f32 {
    let dotp = dot(p, vec3<f32>(127.1, 311.7, 74.7));
    return fract(sin(dotp) * 43758.5453);
}

// ----------------------------------------
// FRAGMENT
// ----------------------------------------
@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    var col = vec3<f32>(0.0);

    // =========================================================
    // SUN  (disc + corona + simple atmospheric extinction)
    // =========================================================
    let sun_dir = normalize(u.sun_direction);
    let sun_pos_world = sun_dir * 1000000.0;
    let sun_clip = u.view_proj * vec4<f32>(sun_pos_world, 1.0);

    var sun_ndc = vec2<f32>(9999.0);
    let sun_visible = sun_clip.w > 0.0;


    var sun_color_final = vec3<f32>(1.0, 1.0, 1.0);

    let alt = clamp(dot(sun_dir, up), -1.0, 1.0);
    let h = clamp((alt + 0.05) / 1.05, 0.0, 1.0);

    let sunrise_color = vec3<f32>(1.35, 0.55, 0.22);
    let midday_color  = vec3<f32>(1.0, 0.97, 0.90);
    let solar_color   = mix(sunrise_color, midday_color, h);

    let air_mass = 1.0 / max(alt * 0.9 + 0.1, 0.02);
    let extinction = exp(-vec3<f32>(0.45, 0.35, 0.20) * (air_mass - 1.0));
    sun_color_final = solar_color * extinction;

    if sun_visible {
        sun_ndc = sun_clip.xy / sun_clip.w;

        let d = distance(input.ndc, sun_ndc);
        let radius = sky.sun_size;
        let intensity = sky.sun_intensity;



        let core_r = radius * 0.75;

        if d < core_r {
            let x = 1.0 - d / core_r;
            let limb = 0.55 + 0.45 * pow(x, 0.35);
            col += sun_color_final * limb * intensity;
        } else if d < radius {
            let t = 1.0 - (d - core_r) / (radius - core_r);
            let glow = pow(t, 3.0);
            col += sun_color_final * glow * intensity * 0.6;
        }

        let corona_r = radius * 4.0;
        if d < corona_r {
            let t = 1.0 - (d - radius) / (corona_r - radius);
            let halo = pow(max(t, 0.0), 5.0) * intensity * 0.15;

            let corona_low  = vec3<f32>(1.2, 0.7, 0.35);
            let corona_high = vec3<f32>(0.7, 0.85, 1.1);
            let corona_color = mix(corona_low, corona_high, h);

            col += corona_color * halo;
        }

        if d < core_r {
            let uv = (input.ndc - sun_ndc) / core_r;
            let n = fract(sin(dot(uv, vec2<f32>(12.9898, 78.233))) * 43758.5453);
            let spots = pow(n, 16.0) * 0.18;
            col *= 1.0 - spots;
        }
    }

    let moon_dir = normalize(u.moon_direction);
    let moon_dist = 10000.0;
    let moon_world_pos = moon_dir * moon_dist;
    let moon_clip = u.view_proj * vec4<f32>(moon_world_pos, 1.0);

    if moon_clip.w > 0.0 {
        let moon_ndc = moon_clip.xy / moon_clip.w;
        let m_radius = sky.moon_size;

        let rel = (input.ndc - moon_ndc) / m_radius;
        let r2 = dot(rel, rel);

        if r2 <= 1.0 {
            // unit sphere in "moon local space" (screen disc)
            let z = sqrt(max(1.0 - r2, 0.0));
            let N_local = normalize(vec3<f32>(rel.x, rel.y, z));

            // ==========================================
            // Camera facing billboard, using YOUR fix:
            // view_ray points from camera to sky
            // ==========================================
            let view_ray = normalize(-input.dir);   // your fix, keep this
            let forward  = view_ray;               // outward from camera

            var ref_up = vec3<f32>(0.0, 1.0, 0.0);
            if abs(dot(ref_up, forward)) > 0.95 {
                ref_up = vec3<f32>(1.0, 0.0, 0.0);
            }

            // right-handed basis
            let right    = normalize(cross(ref_up, forward));
            let up_local = normalize(cross(forward, right));

            // local sphere normal to world space
            let N_world =
                N_local.x * right +
                N_local.y * up_local +
                N_local.z * forward;

            // ==========================================
            // Lighting: Hapke-style lunar reflectance
            // ==========================================
            let L = normalize(u.sun_direction);    // sun direction in world
            let V = normalize(-view_ray);          // from surface to camera
            let N = normalize(N_world);

            let cosNL = dot(N, L);
            let mu0   = max(cosNL, 0.0);
            let mu    = max(dot(N, V), 0.0);

            // physically neutral lunar grey, not using sun_color_final
            let albedo = vec3<f32>(0.85, 0.85, 0.90);

            // "space" sunlight, mildly warm, scaled by moon intensity
            let base_sun = vec3<f32>(1.05, 1.0, 0.97);
            let light_color = base_sun * sky.moon_intensity;

            // optional: very slight atmospheric extinction based on moon altitude
            let alt_moon = clamp(dot(moon_dir, vec3<f32>(0.0, 1.0, 0.0)), -1.0, 1.0);
            let air_mass_moon = 1.0 / max(alt_moon * 0.9 + 0.1, 0.02);
            let extinction_moon = exp(-vec3<f32>(0.08, 0.06, 0.04) * (air_mass_moon - 1.0));

            let illum_color = light_color * extinction_moon;

            var moon_col = vec3<f32>(0.0);

            if mu0 > 0.0 {
                // Hapke parameters tuned for a nice but still realistic look
                let w  = 0.85;
                let g  = -0.4;
                let B0 = 1.2;
                let h  = 0.05;

                let cos_phase = clamp(dot(L, V), -1.0, 1.0);
                let phase = acos(cos_phase);

                // opposition surge near full moon
                let B = B0 / (1.0 + (1.0 / h) * tan(phase * 0.5));

                // Henyey-Greenstein phase function
                let P = (1.0 - g * g) /
                        pow(1.0 + g * g - 2.0 * g * cos_phase, 1.5);

                // multiple scattering approximation
                let Hmu0 = 1.0 / (1.0 + 2.0 * mu0);
                let Hmu  = 1.0 / (1.0 + 2.0 * mu);

                var hapke = w * 0.25 * ((1.0 + B) * P + Hmu0 * Hmu);

                // soften contrast a bit so the sphere feels smoother
                hapke = pow(hapke, 1.2);

                // soft geometric terminator instead of hard mu0 cut
                let terminator = smoothstep(-0.4, 0.4, cosNL);

                hapke *= terminator;


                moon_col = albedo * illum_color * hapke;
            }

            // dark side: subtle earthshine-ish fill so it is not dead black
            let dark_fill = 0.02;
            moon_col += albedo * dark_fill;

            // clean circular edge
            let edge = smoothstep(1.0, 0.90, r2);
            moon_col *= edge;

            col += moon_col;
        }
    }




    return vec4<f32>(col, 1.0);
}
