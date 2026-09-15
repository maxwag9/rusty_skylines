struct ScreenUniform {
    size: vec2<f32>,
    time: f32,
    enable_dither: u32,
    mouse: vec2<f32>,
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

struct RayHit {
    hit: bool,
    distance: f32,
    position: vec3<f32>
}

fn sd_rounded_box_2d(
    p: vec2<f32>,
    b: vec2<f32>,
    r: f32
) -> f32 {
    let q = abs(p) - b + r;

    return length(
        max(q, vec2<f32>(0.0))
    ) +
    min(
        max(q.x, q.y),
        0.0
    ) - r;
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

fn rotate_object(
    p: vec3<f32>,
    t: f32
) -> vec3<f32> {
    var q = p;

    q = rot_y(q, t * 0.48);
    q = rot_x(q, t * 0.31);
    q = rot_z(q, t * 0.17);

    return q;
}

fn morph_amount(
    t: f32
) -> f32 {
    let phase = fract(t * 0.24);

    return 0.5 -
        0.5 * cos(
            phase * 6.28318530718
        );
}

fn sd_box(
    p: vec3<f32>,
    b: vec3<f32>
) -> f32 {
    let q = abs(p) - b;

    return length(
        max(q, vec3<f32>(0.0))
    ) +
    min(
        max(
            q.x,
            max(q.y, q.z)
        ),
        0.0
    );
}

fn sd_sphere(
    p: vec3<f32>,
    radius: f32
) -> f32 {
    return length(p) - radius;
}

fn sd_morph(
    p: vec3<f32>,
    morph: f32
) -> f32 {
    let cube = sd_box(
        p,
        vec3<f32>(
            0.68,
            0.68,
            0.68
        )
    );

    let sphere = sd_sphere(
        p,
        0.82
    );

    return mix(
        cube,
        sphere,
        morph
    );
}

fn scene(
    p: vec3<f32>,
    morph: f32,
    t: f32
) -> f32 {
    let q = rotate_object(
        p,
        t
    );

    return sd_morph(
        q,
        morph
    );
}

fn calc_normal(
    p: vec3<f32>,
    morph: f32,
    t: f32
) -> vec3<f32> {
    let e = 0.0012;

    let dx = scene(
        p + vec3<f32>(e, 0.0, 0.0),
        morph,
        t
    ) - scene(
        p - vec3<f32>(e, 0.0, 0.0),
        morph,
        t
    );

    let dy = scene(
        p + vec3<f32>(0.0, e, 0.0),
        morph,
        t
    ) - scene(
        p - vec3<f32>(0.0, e, 0.0),
        morph,
        t
    );

    let dz = scene(
        p + vec3<f32>(0.0, 0.0, e),
        morph,
        t
    ) - scene(
        p - vec3<f32>(0.0, 0.0, e),
        morph,
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
    morph: f32,
    t: f32
) -> RayHit {
    var distance = 0.0;

    for (var i = 0; i < 112; i++) {
        let p = ro + rd * distance;

        let d = scene(
            p,
            morph,
            t
        );

        if (d < 0.0005) {
            return RayHit(
                true,
                distance,
                p
            );
        }

        distance += max(
            d * 0.82,
            0.001
        );

        if (distance > 10.0) {
            break;
        }
    }

    return RayHit(
        false,
        distance,
        vec3<f32>(0.0)
    );
}

fn soft_shadow(
    ro: vec3<f32>,
    rd: vec3<f32>,
    morph: f32,
    t: f32
) -> f32 {
    var result = 1.0;
    var distance = 0.03;

    for (var i = 0; i < 28; i++) {
        let p = ro + rd * distance;

        let h = scene(
            p,
            morph,
            t
        );

        result = min(
            result,
            14.0 * h / max(distance, 0.001)
        );

        distance += clamp(
            h,
            0.015,
            0.16
        );

        if (
            h < 0.001 ||
            distance > 6.0
        ) {
            break;
        }
    }

    return clamp(
        result,
        0.0,
        1.0
    );
}

fn ambient_occlusion(
    p: vec3<f32>,
    n: vec3<f32>,
    morph: f32,
    t: f32
) -> f32 {
    var result = 0.0;
    var weight = 1.0;

    for (var i = 0; i < 5; i++) {
        let distance =
            0.035 +
            f32(i) * 0.055;

        let sample = scene(
            p + n * distance,
            morph,
            t
        );

        result += (
            distance - sample
        ) * weight;

        weight *= 0.58;
    }

    return clamp(
        1.0 - result * 1.8,
        0.0,
        1.0
    );
}

fn palette(
    t: f32
) -> vec3<f32> {
    let a = vec3<f32>(
        0.48,
        0.26,
        0.20
    );

    let b = vec3<f32>(
        0.42,
        0.34,
        0.32
    );

    let c = vec3<f32>(
        1.0,
        1.0,
        1.0
    );

    let d = vec3<f32>(
        0.00,
        0.18,
        0.37
    );

    return a +
        b * cos(
            6.2831853 *
            (c * t + d)
        );
}

fn saturate(
    x: f32
) -> f32 {
    return clamp(
        x,
        0.0,
        1.0
    );
}

fn light_contribution(
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    light_color: vec3<f32>,
    intensity: f32,
    shadow: f32,
    roughness: f32
) -> vec3<f32> {
    let diffuse = max(
        dot(
            normal,
            light_dir
        ),
        0.0
    );

    let half_dir = normalize(
        light_dir +
        view_dir
    );

    let specular_power = mix(
        80.0,
        24.0,
        roughness
    );

    let specular =
        pow(
            max(
                dot(
                    normal,
                    half_dir
                ),
                0.0
            ),
            specular_power
        );

    return light_color * (
        diffuse * 0.72 +
        specular * 0.38
    ) * intensity * shadow;
}

@vertex
fn vs_main(
    input: VertexInput,
    @builtin(instance_index) instance: u32
) -> VertexOutput {
    let rect = rects[instance];

    let c = cos(
        rect.rotation
    );

    let s = sin(
        rect.rotation
    );

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
    let max_round = min(
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

    let t = screen.time;

    let morph =
        morph_amount(t);

    let camera =
        vec3<f32>(
            0.0,
            0.0,
            -4.8
        );

    let ray_target =
        vec3<f32>(
            screen_pos.x * 1.815,
            -screen_pos.y * 1.815,
            0.0
        );

    let ray_origin =
        camera;

    let ray_direction =
        normalize(
            ray_target -
            ray_origin
        );

    let hit = raymarch(
        ray_origin,
        ray_direction,
        morph,
        t
    );

    var background =
        vec3<f32>(
            0.008,
            0.010,
            0.018
        );

    let background_wave =
        sin(
            screen_pos.x * 3.4 +
            screen_pos.y * 5.7 +
            t * 0.4
        );

    background +=
        vec3<f32>(
            0.012,
            0.020,
            0.035
        ) *
        (
            0.5 +
            background_wave *
            0.5
        );

    let background_ripple =
        exp(
            -length(screen_pos) *
            2.8
        );

    background +=
        vec3<f32>(
            0.018,
            0.008,
            0.028
        ) *
        background_ripple;

    var result =
        background;

    if (hit.hit) {
        let position =
            hit.position;

        let normal =
            calc_normal(
                position,
                morph,
                t
            );

        let view_dir =
            normalize(
                camera -
                position
            );

        let object_local =
            rotate_object(
                position,
                -t
            );

        let spherical =
            0.5 +
            0.5 *
            sin(
                object_local.y * 5.0 +
                t * 0.8
            );

        let angular =
            0.5 +
            0.5 *
            sin(
                atan2(
                    object_local.z,
                    object_local.x
                ) * 6.0 -
                t * 0.55
            );

        let base_mix =
            spherical * 0.45 +
            angular * 0.35 +
            morph * 0.2;

        var base_color =
            palette(
                base_mix +
                t * 0.035
            );

        let cube_emphasis =
            1.0 -
            smoothstep(
                0.55,
                0.92,
                morph
            );

        let sphere_emphasis =
            smoothstep(
                0.08,
                0.72,
                morph
            );

        base_color =
            mix(
                base_color,
                vec3<f32>(
                    0.12,
                    0.62,
                    0.92
                ),
                sphere_emphasis * 0.34
            );

        base_color =
            mix(
                base_color,
                vec3<f32>(
                    0.92,
                    0.30,
                    0.10
                ),
                cube_emphasis * 0.18
            );

        let light_a_pos =
            vec3<f32>(
                -3.2,
                -2.8,
                -3.6
            );

        let light_b_pos =
            vec3<f32>(
                3.5,
                -1.0,
                -1.5
            );

        let light_c_pos =
            vec3<f32>(
                0.8,
                3.5,
                1.2
            );

        let light_a_dir =
            normalize(
                light_a_pos -
                position
            );

        let light_b_dir =
            normalize(
                light_b_pos -
                position
            );

        let light_c_dir =
            normalize(
                light_c_pos -
                position
            );

        let shadow_a =
            soft_shadow(
                position +
                normal * 0.008,
                light_a_dir,
                morph,
                t
            );

        let shadow_b =
            soft_shadow(
                position +
                normal * 0.008,
                light_b_dir,
                morph,
                t
            );

        let shadow_c =
            soft_shadow(
                position +
                normal * 0.008,
                light_c_dir,
                morph,
                t
            );

        let light_a =
            light_contribution(
                normal,
                view_dir,
                light_a_dir,
                vec3<f32>(
                    1.0,
                    0.22,
                    0.07
                ),
                1.35,
                shadow_a,
                0.34
            );

        let light_b =
            light_contribution(
                normal,
                view_dir,
                light_b_dir,
                vec3<f32>(
                    0.04,
                    0.55,
                    1.0
                ),
                1.25,
                shadow_b,
                0.30
            );

        let light_c =
            light_contribution(
                normal,
                view_dir,
                light_c_dir,
                vec3<f32>(
                    0.22,
                    1.0,
                    0.42
                ),
                1.05,
                shadow_c,
                0.40
            );

        let ao =
            ambient_occlusion(
                position,
                normal,
                morph,
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
                3.5
            );

        let rim_color =
            mix(
                vec3<f32>(
                    1.0,
                    0.16,
                    0.08
                ),
                vec3<f32>(
                    0.12,
                    0.55,
                    1.0
                ),
                morph
            );

        result =
            base_color *
            0.24 *
            ao;

        result +=
            base_color *
            (
                light_a +
                light_b +
                light_c
            ) *
            0.9;

        result +=
            vec3<f32>(
                0.08,
                0.10,
                0.12
            ) *
            ao;

        result +=
            rim_color *
            fresnel *
            0.48;

        let surface_ripple =
            sin(
                length(
                    object_local
                ) * 15.0 -
                t * 2.0
            );

        result +=
            base_color *
            surface_ripple *
            0.018;

        let highlight =
            pow(
                max(
                    dot(
                        normal,
                        normalize(
                            vec3<f32>(
                                -0.3,
                                -0.4,
                                -1.0
                            )
                        )
                    ),
                    0.0
                ),
                8.0
            );

        result +=
            vec3<f32>(
                1.0,
                0.85,
                0.65
            ) *
            highlight *
            0.22;
    } else {
        let glow_d =
            scene(
                ray_origin +
                ray_direction *
                2.8,
                morph,
                t
            );

        let glow =
            exp(
                -abs(glow_d) * 7.5
            );

        let glow_color =
            mix(
                vec3<f32>(
                    1.0,
                    0.18,
                    0.05
                ),
                vec3<f32>(
                    0.05,
                    0.60,
                    1.0
                ),
                morph
            );

        result +=
            glow_color *
            glow *
            0.15;
    }

    let center_glow =
        exp(
            -length(
                screen_pos
            ) * 2.2
        );

    result +=
        vec3<f32>(
            0.018,
            0.025,
            0.045
        ) *
        center_glow;

    let edge_distance =
        min(
            min(
                input.uv.x,
                1.0 - input.uv.x
            ),
            min(
                input.uv.y,
                1.0 - input.uv.y
            )
        );

    let edge =
        1.0 -
        smoothstep(
            0.0,
            0.055,
            edge_distance
        );

    result +=
        vec3<f32>(
            0.055,
            0.08,
            0.12
        ) *
        edge;

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