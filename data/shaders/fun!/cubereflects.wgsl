struct ScreenUniform {
    size: vec2<f32>,
    time: f32,
    enable_dither: u32,
    mouse: vec2<f32>
}

struct RectGpu {
    center: vec2<f32>,
    half_size: vec2<f32>,
    color: vec4<f32>,
    border_color: vec4<f32>,
    roundness: f32,
    border_thickness: f32,
    rotation: f32,
    fade: f32,
    glow_color: vec4<f32>,
    glow_misc: vec4<f32>,
    misc: vec4<f32>,
    blur: f32,
    depth: f32,
    _pad0: vec2<f32>
}

@group(1) @binding(0)
var<uniform> screen: ScreenUniform;

@group(2) @binding(0)
var<storage, read> rects: array<RectGpu>;

struct VertexInput {
    @location(0) pos: vec2<f32>
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) mouse_local: vec2<f32>,
    @location(3) rect_half_size: vec2<f32>,
    @location(4) roundness: f32,
    @location(5) border_thickness: f32,
    @location(6) color: vec4<f32>,
    @location(7) border_color: vec4<f32>
}

struct Hit {
    hit: bool,
    distance: f32,
    position: vec3<f32>
}

struct CubeInfo {
    distance: f32,
    center: vec3<f32>,
    local: vec3<f32>
}

fn sd_rounded_box_2d(
    p: vec2<f32>,
    b: vec2<f32>,
    r: f32
) -> f32 {
    let q = abs(p) - b + r;

    return length(max(q, vec2<f32>(0.0))) +
        min(max(q.x, q.y), 0.0) -
        r;
}

fn rot_x(
    p: vec3<f32>,
    a: f32
) -> vec3<f32> {
    let c = cos(a);
    let s = sin(a);

    return vec3<f32>(
        p.x,
        p.y * c - p.z * s,
        p.y * s + p.z * c
    );
}

fn rot_y(
    p: vec3<f32>,
    a: f32
) -> vec3<f32> {
    let c = cos(a);
    let s = sin(a);

    return vec3<f32>(
        p.x * c + p.z * s,
        p.y,
        -p.x * s + p.z * c
    );
}

fn rot_z(
    p: vec3<f32>,
    a: f32
) -> vec3<f32> {
    let c = cos(a);
    let s = sin(a);

    return vec3<f32>(
        p.x * c - p.y * s,
        p.x * s + p.y * c,
        p.z
    );
}

fn cube_center(t: f32) -> vec3<f32> {
    let angle = t * 0.55;

    return vec3<f32>(
        cos(angle) * 1.65,
        0.95 + sin(t * 1.1) * 0.18,
        sin(angle) * 1.65
    );
}

fn cube_rotation(
    t: f32
) -> vec3<f32> {
    return vec3<f32>(
        t * 0.73,
        t * 0.57,
        t * 0.31
    );
}

fn sd_box(
    p: vec3<f32>,
    b: vec3<f32>
) -> f32 {
    let q = abs(p) - b;

    return length(max(q, vec3<f32>(0.0))) +
        min(
            max(q.x, max(q.y, q.z)),
            0.0
        );
}

fn scene_cube(
    p: vec3<f32>,
    t: f32
) -> f32 {
    let center = cube_center(t);
    let angles = cube_rotation(t);

    var q = p - center;

    q = rot_z(q, -angles.z);
    q = rot_y(q, -angles.y);
    q = rot_x(q, -angles.x);

    return sd_box(
        q,
        vec3<f32>(0.72, 0.72, 0.72)
    );
}

fn scene_distance(
    p: vec3<f32>,
    t: f32
) -> f32 {
    let cube = scene_cube(
        p,
        t
    );

    let floor = p.y + 0.02;

    return min(
        cube,
        floor
    );
}

fn get_cube_info(
    p: vec3<f32>,
    t: f32
) -> CubeInfo {
    let center = cube_center(t);
    let angles = cube_rotation(t);

    var q = p - center;

    q = rot_z(q, -angles.z);
    q = rot_y(q, -angles.y);
    q = rot_x(q, -angles.x);

    return CubeInfo(
        sd_box(
            q,
            vec3<f32>(0.72, 0.72, 0.72)
        ),
        center,
        q
    );
}

fn calc_normal(
    p: vec3<f32>,
    t: f32
) -> vec3<f32> {
    let e = 0.001;

    let dx = scene_distance(
        p + vec3<f32>(e, 0.0, 0.0),
        t
    ) - scene_distance(
        p - vec3<f32>(e, 0.0, 0.0),
        t
    );

    let dy = scene_distance(
        p + vec3<f32>(0.0, e, 0.0),
        t
    ) - scene_distance(
        p - vec3<f32>(0.0, e, 0.0),
        t
    );

    let dz = scene_distance(
        p + vec3<f32>(0.0, 0.0, e),
        t
    ) - scene_distance(
        p - vec3<f32>(0.0, 0.0, e),
        t
    );

    return normalize(
        vec3<f32>(
            dx,
            dy,
            dz
        )
    );
}

fn raymarch(
    ro: vec3<f32>,
    rd: vec3<f32>,
    t: f32
) -> Hit {
    var distance = 0.0;

    for (var i = 0; i < 140; i++) {
        let p = ro + rd * distance;

        let d = scene_distance(
            p,
            t
        );

        if (d < 0.0006) {
            return Hit(
                true,
                distance,
                p
            );
        }

        distance += max(
            d * 0.82,
            0.001
        );

        if (distance > 18.0) {
            break;
        }
    }

    return Hit(
        false,
        distance,
        vec3<f32>(0.0)
    );
}

fn shadow_ray(
    origin: vec3<f32>,
    direction: vec3<f32>,
    max_distance: f32,
    t: f32
) -> f32 {
    var distance = 0.025;
    var result = 1.0;

    for (var i = 0; i < 80; i++) {
        let p = origin + direction * distance;

        let d = scene_distance(
            p,
            t
        );

        if (d < 0.0007) {
            return 0.0;
        }

        result = min(
            result,
            18.0 * d / max(distance, 0.001)
        );

        distance += clamp(
            d,
            0.012,
            0.14
        );

        if (distance >= max_distance) {
            break;
        }
    }

    return clamp(
        result,
        0.0,
        1.0
    );
}

fn cube_edge_mask(
    local: vec3<f32>
) -> f32 {
    let d = abs(abs(local) - vec3<f32>(0.72));

    let nearest_two =
        min(
            d.x + min(d.y, d.z),
            d.y + min(d.x, d.z)
        );

    return 1.0 -
        smoothstep(
            0.025,
            0.10,
            nearest_two
        );
}

fn aces(
    x: vec3<f32>
) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;

    return clamp(
        (x * (a * x + b)) /
        (x * (c * x + d) + e),
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );
}

fn light(
    position: vec3<f32>,
    color: vec3<f32>,
    intensity: f32,
    point: vec3<f32>
) -> vec3<f32> {
    let delta = position - point;
    let distance_sq = max(
        dot(delta, delta),
        0.03
    );

    return color *
        intensity /
        distance_sq;
}

fn glossy_reflection(
    rd: vec3<f32>,
    normal: vec3<f32>,
    roughness: f32,
    t: f32
) -> vec3<f32> {
    let reflected =
        reflect(
            rd,
            normal
        );

    let hit = raymarch(
        normal * 0.008,
        reflected,
        t
    );

    if (hit.hit) {
        let p = hit.position;
        let n = calc_normal(
            p,
            t
        );

        let facing =
            max(
                dot(
                    n,
                    -reflected
                ),
                0.0
            );

        let cube_info =
            get_cube_info(
                p,
                t
            );

        let local =
            cube_info.local;

        let checker_a =
            0.5 +
            0.5 *
            sin(
                local.x * 8.0 +
                local.z * 6.0
            );

        let reflection_color =
            mix(
                vec3<f32>(
                    0.015,
                    0.025,
                    0.035
                ),
                vec3<f32>(
                    0.15,
                    0.18,
                    0.20
                ),
                checker_a
            );

        return reflection_color *
            facing *
            (1.0 - roughness);
    }

    let horizon =
        0.5 +
        0.5 *
        reflected.y;

    return mix(
        vec3<f32>(
            0.008,
            0.012,
            0.020
        ),
        vec3<f32>(
            0.08,
            0.11,
            0.14
        ),
        horizon
    ) * (1.0 - roughness);
}

@vertex
fn vs_main(
    input: VertexInput,
    @builtin(instance_index) instance: u32
) -> VertexOutput {
    let rect = rects[instance];

    let c = cos(rect.rotation);
    let s = sin(rect.rotation);

    let rot = mat2x2<f32>(
        c, -s,
        s, c
    );

    let inverse_rot = mat2x2<f32>(
        c, s,
        -s, c
    );

    let local =
        input.pos *
        rect.half_size;

    let world =
        rect.center +
        rot * local;

    let mouse_local =
        inverse_rot *
        (
            screen.mouse -
            rect.center
        );

    let ndc =
        (
            world /
            screen.size
        ) * 2.0 - 1.0;

    var out: VertexOutput;

    out.position = vec4<f32>(
        ndc.x,
        -ndc.y,
        rect.depth,
        1.0
    );

    out.local_pos =
        local;

    out.uv =
        input.pos * 0.5 +
        0.5;

    out.mouse_local =
        mouse_local;

    out.rect_half_size =
        rect.half_size;

    out.roundness =
        rect.roundness;

    out.border_thickness =
        rect.border_thickness;

    out.color =
        rect.color;

    out.border_color =
        rect.border_color;

    return out;
}

@fragment
fn fs_main(
    input: VertexOutput
) -> @location(0) vec4<f32> {
    let max_round =
        min(
            input.rect_half_size.x,
            input.rect_half_size.y
        );

    let radius =
        input.roundness *
        max_round;

    let d_rect =
        sd_rounded_box_2d(
            input.local_pos,
            input.rect_half_size,
            radius
        );

    let aa = max(
        fwidth(d_rect),
        0.75
    );

    let outer_alpha =
        1.0 -
        smoothstep(
            -aa,
            aa,
            d_rect
        );

    let inner_half = max(
        input.rect_half_size -
        input.border_thickness,
        vec2<f32>(0.0)
    );

    let inner_radius =
        max(
            radius -
            input.border_thickness,
            0.0
        );

    let d_inner =
        sd_rounded_box_2d(
            input.local_pos,
            inner_half,
            inner_radius
        );

    let inner_alpha =
        1.0 -
        smoothstep(
            -aa,
            aa,
            d_inner
        );

    let border_mask =
        outer_alpha -
        inner_alpha;

    let centered =
        input.uv - 0.5;

    let aspect =
        input.rect_half_size.x /
        max(
            input.rect_half_size.y,
            0.001
        );

    let screen_pos =
        vec2<f32>(
            centered.x * aspect,
            centered.y
        );

    let t =
        screen.time;

    let camera =
        vec3<f32>(
            0.0,
            2.25,
            -7.0
        );

    let camera_target =
        vec3<f32>(
            0.0,
            0.75,
            0.0
        );

    let forward =
        normalize(
            camera_target -
            camera
        );

    let right =
        normalize(
            cross(
                forward,
                vec3<f32>(
                    0.0,
                    1.0,
                    0.0
                )
            )
        );

    let up =
        cross(
            right,
            forward
        );

    let ray_target =
        camera +
        forward * 4.0 +
        right *
        (
            screen_pos.x *
            1.815
        ) +
        up *
        (
            -screen_pos.y *
            1.815
        );

    let ray_direction =
        normalize(
            ray_target -
            camera
        );

    let primary =
        raymarch(
            camera,
            ray_direction,
            t
        );

    var result =
        vec3<f32>(
            0.003,
            0.005,
            0.010
        );

    let floor_grid_x =
        abs(
            fract(
                screen_pos.x * 7.0
            ) - 0.5
        );

    let floor_grid_y =
        abs(
            fract(
                screen_pos.y * 7.0
            ) - 0.5
        );

    let grid =
        1.0 -
        smoothstep(
            0.46,
            0.50,
            min(
                floor_grid_x,
                floor_grid_y
            )
        );

    result +=
        vec3<f32>(
            0.008,
            0.010,
            0.015
        ) *
        grid;

    if (primary.hit) {
        let position =
            primary.position;

        let normal =
            calc_normal(
                position,
                t
            );

        let view_dir =
            normalize(
                camera -
                position
            );

        let cube_info =
            get_cube_info(
                position,
                t
            );

        let is_cube =
            cube_info.distance <
            0.002;

        if (is_cube) {
            let red_light =
                vec3<f32>(
                    1.0,
                    0.0,
                    0.0
                );

            let cyan_light =
                vec3<f32>(
                    0.0,
                    1.0,
                    1.0
                );

            let green_light =
                vec3<f32>(
                    0.0,
                    1.0,
                    0.0
                );

            let red_delta =
                red_light -
                position;

            let cyan_delta =
                cyan_light -
                position;

            let green_delta =
                green_light -
                position;

            let red_distance =
                length(red_delta);

            let cyan_distance =
                length(cyan_delta);

            let green_distance =
                length(green_delta);

            let red_direction =
                red_delta /
                max(
                    red_distance,
                    0.001
                );

            let cyan_direction =
                cyan_delta /
                max(
                    cyan_distance,
                    0.001
                );

            let green_direction =
                green_delta /
                max(
                    green_distance,
                    0.001
                );

            let red_shadow =
                shadow_ray(
                    position +
                    normal * 0.012,
                    red_direction,
                    red_distance,
                    t
                );

            let cyan_shadow =
                shadow_ray(
                    position +
                    normal * 0.012,
                    cyan_direction,
                    cyan_distance,
                    t
                );

            let green_shadow =
                shadow_ray(
                    position +
                    normal * 0.012,
                    green_direction,
                    green_distance,
                    t
                );

            let red_diffuse =
                max(
                    dot(
                        normal,
                        red_direction
                    ),
                    0.0
                ) *
                red_shadow *
                3.0 /
                (
                    red_distance *
                    red_distance
                );

            let cyan_diffuse =
                max(
                    dot(
                        normal,
                        cyan_direction
                    ),
                    0.0
                ) *
                cyan_shadow *
                3.0 /
                (
                    cyan_distance *
                    cyan_distance
                );

            let green_diffuse =
                max(
                    dot(
                        normal,
                        green_direction
                    ),
                    0.0
                ) *
                green_shadow *
                3.0 /
                (
                    green_distance *
                    green_distance
                );

            let red_half =
                normalize(
                    red_direction +
                    view_dir
                );

            let cyan_half =
                normalize(
                    cyan_direction +
                    view_dir
                );

            let green_half =
                normalize(
                    green_direction +
                    view_dir
                );

            let red_spec =
                pow(
                    max(
                        dot(
                            normal,
                            red_half
                        ),
                        0.0
                    ),
                    72.0
                ) *
                red_shadow *
                4.0;

            let cyan_spec =
                pow(
                    max(
                        dot(
                            normal,
                            cyan_half
                        ),
                        0.0
                    ),
                    72.0
                ) *
                cyan_shadow *
                4.0;

            let green_spec =
                pow(
                    max(
                        dot(
                            normal,
                            green_half
                        ),
                        0.0
                    ),
                    72.0
                ) *
                green_shadow *
                4.0;

            let red =
                vec3<f32>(
                    1.0,
                    0.0,
                    0.0
                );

            let cyan =
                vec3<f32>(
                    0.0,
                    1.0,
                    1.0
                );

            let green =
                vec3<f32>(
                    0.0,
                    1.0,
                    0.0
                );

            let edge =
                cube_edge_mask(
                    cube_info.local
                );

            let base =
                vec3<f32>(
                    0.13,
                    0.15,
                    0.17
                );

            result +=
                base * 0.16;

            result +=
                red *
                red_diffuse;

            result +=
                cyan *
                cyan_diffuse;

            result +=
                green *
                green_diffuse;

            result +=
                red *
                red_spec;

            result +=
                cyan *
                cyan_spec;

            result +=
                green *
                green_spec;

            let reflection =
                glossy_reflection(
                    ray_direction,
                    normal,
                    0.08,
                    t
                );

            let fresnel =
                pow(
                    1.0 -
                    max(
                        dot(
                            normal,
                            view_dir
                        ),
                        0.0
                    ),
                    5.0
                );

            result +=
                reflection *
                (
                    0.25 +
                    fresnel *
                    0.75
                );

            result +=
                mix(
                    red,
                    cyan,
                    0.5 +
                    0.5 *
                    sin(
                        cube_info.local.y *
                        5.0 +
                        t * 1.4
                    )
                ) *
                edge *
                0.12;

            let face_gradient =
                0.5 +
                0.5 *
                cube_info.local.y /
                0.72;

            result +=
                vec3<f32>(
                    0.03,
                    0.04,
                    0.05
                ) *
                face_gradient;
        } else {
            let plane_normal =
                vec3<f32>(
                    0.0,
                    1.0,
                    0.0
                );

            let red_light =
                vec3<f32>(
                    1.0,
                    0.0,
                    0.0
                );

            let cyan_light =
                vec3<f32>(
                    0.0,
                    1.0,
                    1.0
                );

            let green_light =
                vec3<f32>(
                    0.0,
                    1.0,
                    0.0
                );

            let red_delta =
                red_light -
                position;

            let cyan_delta =
                cyan_light -
                position;

            let green_delta =
                green_light -
                position;

            let red_dist =
                length(red_delta);

            let cyan_dist =
                length(cyan_delta);

            let green_dist =
                length(green_delta);

            let red_dir =
                red_delta /
                max(
                    red_dist,
                    0.001
                );

            let cyan_dir =
                cyan_delta /
                max(
                    cyan_dist,
                    0.001
                );

            let green_dir =
                green_delta /
                max(
                    green_dist,
                    0.001
                );

            let red_intensity =
                max(
                    dot(
                        plane_normal,
                        red_dir
                    ),
                    0.0
                ) *
                2.5 /
                max(
                    red_dist *
                    red_dist,
                    0.1
                );

            let cyan_intensity =
                max(
                    dot(
                        plane_normal,
                        cyan_dir
                    ),
                    0.0
                ) *
                2.5 /
                max(
                    cyan_dist *
                    cyan_dist,
                    0.1
                );

            let green_intensity =
                max(
                    dot(
                        plane_normal,
                        green_dir
                    ),
                    0.0
                ) *
                2.5 /
                max(
                    green_dist *
                    green_dist,
                    0.1
                );

            result +=
                vec3<f32>(
                    0.012,
                    0.014,
                    0.018
                );

            result +=
                vec3<f32>(
                    1.0,
                    0.01,
                    0.015
                ) *
                red_intensity;

            result +=
                vec3<f32>(
                    0.0,
                    0.75,
                    1.0
                ) *
                cyan_intensity;

            result +=
                vec3<f32>(
                    0.05,
                    1.0,
                    0.15
                ) *
                green_intensity;

            let reflection =
                reflect(
                    ray_direction,
                    plane_normal
                );

            let reflected_hit =
                raymarch(
                    position +
                    plane_normal *
                    0.01,
                    reflection,
                    t
                );

            if (reflected_hit.hit) {
                result +=
                    vec3<f32>(
                        0.10,
                        0.12,
                        0.14
                    ) *
                    0.22;
            }
        }
    } else {
        let horizon =
            0.5 +
            0.5 *
            ray_direction.y;

        result =
            mix(
                vec3<f32>(
                    0.003,
                    0.005,
                    0.010
                ),
                vec3<f32>(
                    0.02,
                    0.035,
                    0.06
                ),
                horizon
            );

        let sun_1 =
            pow(
                max(
                    dot(
                        ray_direction,
                        normalize(
                            vec3<f32>(
                                -0.5,
                                0.4,
                                -0.8
                            )
                        )
                    ),
                    0.0
                ),
                64.0
            );

        let sun_2 =
            pow(
                max(
                    dot(
                        ray_direction,
                        normalize(
                            vec3<f32>(
                                0.8,
                                0.15,
                                -0.4
                            )
                        )
                    ),
                    0.0
                ),
                96.0
            );

        result +=
            vec3<f32>(
                1.0,
                0.02,
                0.01
            ) *
            sun_1 *
            0.08;

        result +=
            vec3<f32>(
                0.01,
                0.65,
                1.0
            ) *
            sun_2 *
            0.08;
    }

    let vignette =
        1.0 -
        smoothstep(
            0.45,
            0.95,
            length(
                centered
            )
        );

    result *=
        0.72 +
        vignette *
        0.28;

    result +=
        vec3<f32>(
            1.0,
            0.015,
            0.01
        ) *
        exp(
            -length(
                screen_pos -
                vec2<f32>(
                    0.35,
                    -0.15
                )
            ) * 5.0
        ) *
        0.018;

    result +=
        vec3<f32>(
            0.0,
            0.55,
            1.0
        ) *
        exp(
            -length(
                screen_pos +
                vec2<f32>(
                    0.40,
                    0.10
                )
            ) * 4.5
        ) *
        0.018;

    result =
        aces(
            result
        );

    result =
        pow(
            result,
            vec3<f32>(
                0.92
            )
        );

    result = mix(
        result,
        input.border_color.rgb,
        border_mask
    );

    return vec4<f32>(
        result *
        outer_alpha,
        outer_alpha
    );
}