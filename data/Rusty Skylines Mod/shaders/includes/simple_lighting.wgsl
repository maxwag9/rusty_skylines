
fn calculate_world_lighting(
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    wrap_amount: f32,
) -> vec3<f32> {
    let n = normalize(world_normal);
    let sun_l = normalize(uniforms.sun_direction);
    let moon_l = normalize(uniforms.moon_direction);

    let sun_elev = sun_l.y;
    let sun_amount = smoothstep(-0.025, 0.035, sun_elev);
    let sun_direct_visibility = smoothstep(0.0, 0.025, sun_elev);
    let night_amount = 1.0 - sun_direct_visibility;

    let moon_visible = smoothstep(-0.08, 0.08, moon_l.y);
    let moon_factor = moon_visible * night_amount;

    let hemi = saturate(n.y * 0.5 + 0.5);

    let sky_ambient = mix(
        vec3<f32>(0.02, 0.02, 0.02),
        vec3<f32>(0.10, 0.12, 0.15),
        sun_amount,
    );

    let ground_ambient = mix(
        vec3<f32>(0.03, 0.03, 0.03),
        vec3<f32>(0.05, 0.05, 0.05),
        sun_amount,
    );

    let ambient = mix(ground_ambient, sky_ambient, hemi)
        + vec3<f32>(0.02, 0.03, 0.06) * (0.6 * moon_factor * hemi);

    let wrapped = saturate(wrap_amount);

    let sun_ndotl = dot(n, sun_l);
    let sun_diffuse = max(sun_ndotl, 0.0);
    let sun_wrapped = saturate(sun_ndotl + 0.4) / 1.4;

    let sun_direct = mix(sun_diffuse, sun_wrapped, wrapped)
        * sun_direct_visibility;

    var sun_shadow = 1.0;

    if uniforms.csm_enabled != 0u && sun_direct > 0.0001 {
        sun_shadow = fetch_shadow(world_pos, n, sun_l);
    }

    let moon_ndotl = dot(n, moon_l);
    let moon_diffuse = max(moon_ndotl, 0.0);
    let moon_wrapped = saturate(moon_ndotl + 0.15) / 1.15;

    let moon_direct = mix(moon_diffuse, moon_wrapped, wrapped)
        * (moon_factor * 0.06);

    return ambient
        + vec3<f32>(1.0, 0.98, 0.92) * (sun_direct * sun_shadow)
        + vec3<f32>(0.35, 0.42, 0.60) * moon_direct;
}
