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
    _pad0: vec2<f32>,
}

@group(1) @binding(0)
var<uniform> screen: ScreenUniform;

@group(2) @binding(0)
var<storage, read> rects: array<RectGpu>;

struct VertexInput {
    @location(0) pos: vec2<f32>,
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
    @location(7) border_color: vec4<f32>,
}

fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + r;

    return length(max(q, vec2<f32>(0.0))) +
        min(max(q.x, q.y), 0.0) -
        r;
}

fn hash21(p: vec2<f32>) -> f32 {
    let h = dot(
        p,
        vec2<f32>(127.1, 311.7)
    );

    return fract(
        sin(h) * 43758.5453123
    );
}

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));

    let u = f * f * f * (
        f * (f * 6.0 - 15.0) + 10.0
    );

    return mix(
        mix(a, b, u.x),
        mix(c, d, u.x),
        u.y
    );
}

fn fbm(p: vec2<f32>) -> f32 {
    var q = p;
    var value = 0.0;
    var amplitude = 0.5;

    let rot = mat2x2<f32>(
        0.7648422, -0.6442177,
        0.6442177,  0.7648422
    );

    for (var i = 0; i < 5; i++) {
        value += noise(q) * amplitude;

        q = rot * q * 2.03 +
            vec2<f32>(17.1, 31.7);

        amplitude *= 0.48;
    }

    return value;
}

fn billow_noise(p: vec2<f32>) -> f32 {
    let n = fbm(p);

    return 1.0 - abs(
        n * 2.0 - 1.0
    );
}

fn cloud_density(
    p: vec2<f32>,
    depth: f32,
    height: f32,
    t: f32
) -> f32 {
    let wind = vec2<f32>(
        t * 0.020,
        -t * 0.006
    );

    var q = p;

    q += vec2<f32>(
        depth * 0.18,
        depth * 0.10
    );

    q += wind;

    let warp_a = fbm(
        q * 0.72 +
        vec2<f32>(13.7, 7.1)
    );

    let warp_b = fbm(
        q * 0.90 -
        vec2<f32>(5.2, 19.4)
    );

    q += vec2<f32>(
        warp_a - 0.5,
        warp_b - 0.5
    ) * 0.34;

    let macron = fbm(
        q * 0.72
    );

    let billows = billow_noise(
        q * 1.45 +
        vec2<f32>(41.3, 17.8)
    );

    let structure = mix(
        macron,
        billows,
        0.43
    );

    let coverage = smoothstep(
        0.29,
        0.66,
        structure
    );

    let detail = fbm(
        q * 5.4 +
        vec2<f32>(
            t * 0.045,
            -t * 0.018
        )
    );

    let fine = noise(
        q * 12.0 -
        vec2<f32>(
            t * 0.05,
            t * 0.025
        )
    );

    let erosion = mix(
        detail,
        fine,
        0.32
    );

    let eroded = coverage -
        erosion * 0.27;

    let base_density = smoothstep(
        0.08,
        0.58,
        eroded
    );

    let lower_transition = smoothstep(
        0.035,
        0.22,
        height
    );

    let upper_transition =
        1.0 - smoothstep(
            0.68,
            0.98,
            height
        );

    let vertical_shape =
        lower_transition *
        upper_transition;

    let lower_wisps =
        smoothstep(
            0.03,
            0.22,
            height
        );

    let density =
        base_density *
        vertical_shape *
        (0.72 + 0.28 * lower_wisps);

    return density;
}

fn phase_hg(mu: f32, g: f32) -> f32 {
    let gg = g * g;

    return (
        1.0 - gg
    ) / pow(
        1.0 + gg - 2.0 * g * mu,
        1.5
    );
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

    let local = input.pos * rect.half_size;
    let world = rect.center + rot * local;
    let mouse_local =
        inverse_rot *
        (screen.mouse - rect.center);

    let ndc =
        (world / screen.size) * 2.0 - 1.0;

    var out: VertexOutput;

    out.position = vec4<f32>(
        ndc.x,
        -ndc.y,
        rect.depth,
        1.0
    );

    out.local_pos = local;
    out.uv = input.pos * 0.5 + 0.5;
    out.mouse_local = mouse_local;
    out.rect_half_size = rect.half_size;
    out.roundness = rect.roundness;
    out.border_thickness =
        rect.border_thickness;
    out.color = rect.color;
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
        input.roundness * max_round;

    let d = sd_rounded_box(
        input.local_pos,
        input.rect_half_size,
        radius
    );

    let aa = max(
        fwidth(d),
        0.75
    );

    let outer_alpha =
        1.0 - smoothstep(
            -aa,
            aa,
            d
        );

    let inner_half = max(
        input.rect_half_size -
        input.border_thickness,
        vec2<f32>(0.0)
    );

    let inner_radius = max(
        radius -
        input.border_thickness,
        0.0
    );

    let d_inner = sd_rounded_box(
        input.local_pos,
        inner_half,
        inner_radius
    );

    let inner_alpha =
        1.0 - smoothstep(
            -aa,
            aa,
            d_inner
        );

    let border_mask =
        outer_alpha -
        inner_alpha;

    let uv = input.uv;
    let t = screen.time;

    let centered =
        uv - 0.5;

    let aspect =
        input.rect_half_size.x /
        max(
            input.rect_half_size.y,
            0.001
        );

    let p = vec2<f32>(
        centered.x * aspect,
        centered.y
    );

    let height =
        clamp(
            uv.y + 0.18,
            0.0,
            1.0
        );

    let sun_dir = normalize(
        vec3<f32>(
            -0.52,
            0.43,
            0.76
        )
    );

    let view_dir = vec3<f32>(
        0.0,
        0.0,
        1.0
    );

    let phase_forward =
        phase_hg(
            dot(
                view_dir,
                sun_dir
            ),
            0.48
        );

    let phase_back =
        phase_hg(
            dot(
                view_dir,
                -sun_dir
            ),
            -0.18
        );

    let forward_scatter =
        phase_forward * 0.32;

    let back_scatter =
        phase_back * 0.10;

    let sky_horizon =
        vec3<f32>(
            0.68,
            0.80,
            0.96
        );

    let sky_zenith =
        vec3<f32>(
            0.045,
            0.11,
            0.25
        );

    let sky_factor =
        pow(
            clamp(
                uv.y,
                0.0,
                1.0
            ),
            0.72
        );

    var sky = mix(
        sky_horizon,
        sky_zenith,
        sky_factor
    );

    let sun_uv = vec2<f32>(
        0.72,
        0.24
    );

    let sun_delta =
        uv - sun_uv;

    let sun_distance =
        length(
            vec2<f32>(
                sun_delta.x * aspect,
                sun_delta.y
            )
        );

    let sun_glow =
        exp(
            -sun_distance *
            sun_distance *
            11.0
        );

    sky += vec3<f32>(
        1.0,
        0.76,
        0.43
    ) * sun_glow * 0.20;

    let steps = 12;

    var transmittance = 1.0;
    var accumulated =
        vec3<f32>(0.0);

    for (var i = 0; i < steps; i++) {
        let fi = f32(i);

        let depth =
            (fi + 0.5) /
            f32(steps);

        let depth_curve =
            depth +
            sin(
                depth * 8.0 +
                t * 0.12
            ) * 0.018;

        let depth_offset = (depth_curve - 0.5) * 0.08;

        let sample_p =
            p +
            vec2<f32>(
                depth_offset,
                depth_offset * 0.72
            );

        let density =
            cloud_density(
                sample_p,
                depth_curve,
                height,
                t
            );

        if (density > 0.001) {
            let light_sample_pos =
                sample_p +
                sun_dir.xy * 0.16;

            let light_depth =
                depth_curve +
                sun_dir.z * 0.14;

            let light_density =
                cloud_density(
                    light_sample_pos,
                    light_depth,
                    clamp(
                        height +
                        sun_dir.y * 0.08,
                        0.0,
                        1.0
                    ),
                    t
                );

            let direct_transmission =
                exp(
                    -light_density *
                    4.2
                );

            let powder =
                1.0 -
                exp(
                    -density * 5.0
                );

            let deep_shadow =
                exp(
                    -density * 3.2
                );

            let multiple_scatter =
                0.22 +
                powder * 0.34 +
                direct_transmission * 0.28 +
                deep_shadow * 0.16;

            let direct_light =
                direct_transmission *
                (
                    0.52 +
                    forward_scatter
                );

            let silver =
                powder *
                back_scatter *
                1.8;

            let ambient =
                multiple_scatter *
                0.65;

            let sun_color =
                vec3<f32>(
                    1.0,
                    0.79,
                    0.60
                );

            let ambient_color =
                vec3<f32>(
                    0.55,
                    0.68,
                    0.92
                );

            let sample_color =
                ambient_color *
                ambient +
                sun_color *
                direct_light +
                vec3<f32>(
                    1.0,
                    0.86,
                    0.67
                ) * silver;

            let bottom_fill =
                (1.0 - height) *
                0.13;

            let lit_color =
                sample_color +
                vec3<f32>(
                    0.32,
                    0.38,
                    0.48
                ) * bottom_fill;

            let optical_depth =
                density *
                0.72;

            let sample_alpha =
                1.0 -
                exp(
                    -optical_depth
                );

            accumulated +=
                transmittance *
                lit_color *
                sample_alpha;

            transmittance *=
                1.0 -
                sample_alpha;

            if (transmittance < 0.01) {
                break;
            }
        }
    }

    let cloud_alpha =
        1.0 -
        transmittance;

    let cloud_color =
        accumulated /
        max(
            cloud_alpha,
            0.001
        );

    var fill =
        mix(
            sky,
            cloud_color,
            cloud_alpha
        );

    let cloud_shadow =
        smoothstep(
            0.15,
            0.72,
            cloud_alpha
        );

    fill *=
        0.90 +
        cloud_shadow * 0.10;

    let mouse_local_uv = vec2<f32>(
        input.mouse_local.x /
            input.rect_half_size.x *
            0.5 +
            0.5,

        input.mouse_local.y /
            input.rect_half_size.y *
            0.5 +
            0.5
    );

    let mouse_delta =
        uv -
        mouse_local_uv;

    let mouse_distance =
        length(
            vec2<f32>(
                mouse_delta.x * aspect,
                mouse_delta.y
            )
        );

    let mouse_light =
        exp(
            -mouse_distance *
            mouse_distance *
            8.0
        );

    fill += vec3<f32>(
        0.10,
        0.14,
        0.20
    ) * mouse_light * 0.35;

    let edge_distance =
        min(
            min(
                uv.x,
                1.0 - uv.x
            ),
            min(
                uv.y,
                1.0 - uv.y
            )
        );

    let edge =
        1.0 -
        smoothstep(
            0.0,
            0.07,
            edge_distance
        );

    fill += vec3<f32>(
        0.025,
        0.035,
        0.06
    ) * edge;

    fill = mix(
        fill,
        input.border_color.rgb,
        border_mask
    );

    return vec4<f32>(
        fill * outer_alpha,
        outer_alpha
    );
}