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

fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + r;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

fn palette(t: f32) -> vec3<f32> {
    let a = vec3<f32>(0.18, 0.06, 0.28);
    let b = vec3<f32>(0.35, 0.20, 0.35);
    let c = vec3<f32>(1.0, 0.8, 0.65);
    let d = vec3<f32>(0.10, 0.22, 0.35);

    return a + b * cos(6.28318 * (c * t + d));
}

fn rot4_xy(p: vec4<f32>, a: f32) -> vec4<f32> {
    let c = cos(a);
    let s = sin(a);

    return vec4<f32>(
        p.x * c - p.y * s,
        p.x * s + p.y * c,
        p.z,
        p.w
    );
}

fn rot4_xz(p: vec4<f32>, a: f32) -> vec4<f32> {
    let c = cos(a);
    let s = sin(a);

    return vec4<f32>(
        p.x * c - p.z * s,
        p.y,
        p.x * s + p.z * c,
        p.w
    );
}

fn rot4_xw(p: vec4<f32>, a: f32) -> vec4<f32> {
    let c = cos(a);
    let s = sin(a);

    return vec4<f32>(
        p.x * c - p.w * s,
        p.y,
        p.z,
        p.x * s + p.w * c
    );
}

fn rot4_yw(p: vec4<f32>, a: f32) -> vec4<f32> {
    let c = cos(a);
    let s = sin(a);

    return vec4<f32>(
        p.x,
        p.y * c - p.w * s,
        p.z,
        p.y * s + p.w * c
    );
}

fn rot4_zw(p: vec4<f32>, a: f32) -> vec4<f32> {
    let c = cos(a);
    let s = sin(a);

    return vec4<f32>(
        p.x,
        p.y,
        p.z * c - p.w * s,
        p.z * s + p.w * c
    );
}

fn rotate_hypercube(p: vec4<f32>, t: f32) -> vec4<f32> {
    var q = p;

    q = rot4_xw(q, t * 0.47);
    q = rot4_yw(q, t * 0.31);
    q = rot4_zw(q, t * 0.22);

    q = rot4_xy(q, t * 0.19);
    q = rot4_xz(q, t * 0.27);

    return q;
}

fn project_4d(p: vec4<f32>) -> vec3<f32> {
    let w_camera = 3.2;
    let w_scale = w_camera / max(w_camera - p.w, 0.1);

    return vec3<f32>(
        p.x * w_scale,
        p.y * w_scale,
        p.z * w_scale
    );
}

fn rotate_3d(p: vec3<f32>, t: f32) -> vec3<f32> {
    let cy = cos(t * 0.18);
    let sy = sin(t * 0.18);

    let xz = vec2<f32>(
        p.x * cy - p.z * sy,
        p.x * sy + p.z * cy
    );

    let cx = cos(t * 0.11);
    let sx = sin(t * 0.11);

    return vec3<f32>(
        xz.x,
        p.y * cx - xz.y * sx,
        p.y * sx + xz.y * cx
    );
}

fn project_3d(p: vec3<f32>) -> vec2<f32> {
    let camera = 6.0;
    let depth = camera + p.z;
    let perspective = 2.8 / max(depth, 0.1);

    return p.xy * perspective;
}

fn hypercube_vertex(index: u32) -> vec4<f32> {
    let x = select(-1.0, 1.0, (index & 1u) != 0u);
    let y = select(-1.0, 1.0, (index & 2u) != 0u);
    let z = select(-1.0, 1.0, (index & 4u) != 0u);
    let w = select(-1.0, 1.0, (index & 8u) != 0u);

    return vec4<f32>(x, y, z, w);
}

fn edge_distance(
    p: vec2<f32>,
    a: vec2<f32>,
    b: vec2<f32>
) -> f32 {
    let ba = b - a;
    let pa = p - a;

    let h = clamp(
        dot(pa, ba) / max(dot(ba, ba), 0.00001),
        0.0,
        1.0
    );

    return length(pa - ba * h);
}

fn edge_depth(
    a: vec3<f32>,
    b: vec3<f32>
) -> f32 {
    return max(a.z, b.z);
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
    let mouse_local = inverse_rot * (screen.mouse - rect.center);

    let ndc = (world / screen.size) * 2.0 - 1.0;

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
    out.border_thickness = rect.border_thickness;
    out.color = rect.color;
    out.border_color = rect.border_color;

    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let max_round = min(
        input.rect_half_size.x,
        input.rect_half_size.y
    );

    let radius = input.roundness * max_round;

    let d = sd_rounded_box(
        input.local_pos,
        input.rect_half_size,
        radius
    );

    let aa = max(fwidth(d), 0.75);

    let outer_alpha =
        1.0 - smoothstep(-aa, aa, d);

    let inner_half = max(
        input.rect_half_size - input.border_thickness,
        vec2<f32>(0.0)
    );

    let inner_radius = max(
        radius - input.border_thickness,
        0.0
    );

    let d_inner = sd_rounded_box(
        input.local_pos,
        inner_half,
        inner_radius
    );

    let inner_alpha =
        1.0 - smoothstep(-aa, aa, d_inner);

    let border_mask =
        outer_alpha - inner_alpha;

    let smallest = min(
        input.rect_half_size.x,
        input.rect_half_size.y
    );

    let p = input.local_pos / smallest;

    var vertices_2d: array<vec2<f32>, 16>;
    var vertices_3d: array<vec3<f32>, 16>;

    for (var i = 0u; i < 16u; i++) {
        let h = hypercube_vertex(i);

        let rotated = rotate_hypercube(
            h,
            screen.time
        );

        let p3 = project_4d(rotated);
        let r3 = rotate_3d(
            p3,
            screen.time
        );

        vertices_3d[i] = r3;
        vertices_2d[i] = project_3d(r3);
    }

    var nearest = 1000.0;
    var nearest_depth = 0.0;

    for (var i = 0u; i < 16u; i++) {
        for (var axis = 0u; axis < 4u; axis++) {
            let j = i ^ (1u << axis);

            if (j > i) {
                let a = vertices_2d[i];
                let b = vertices_2d[j];

                let dist = edge_distance(
                    p,
                    a,
                    b
                );

                if (dist < nearest) {
                    nearest = dist;
                    nearest_depth = edge_depth(
                        vertices_3d[i],
                        vertices_3d[j]
                    );
                }
            }
        }
    }

    let line_width = 0.012;
    let core = 1.0 - smoothstep(
        line_width * 0.35,
        line_width,
        nearest
    );

    let glow_width = 0.07;
    let glow = 1.0 - smoothstep(
        glow_width * 0.35,
        glow_width,
        nearest
    );

    let depth_light = clamp(
        0.55 + nearest_depth * 0.18,
        0.35,
        1.25
    );

    let t = screen.time;

    let base = palette(
        p.x * 0.8 +
        p.y * 0.45 +
        t * 0.025
    );

    var result = base * 0.12;

    result += vec3<f32>(
        0.20,
        0.07,
        0.32
    ) * glow * depth_light;

    result += vec3<f32>(
        0.75,
        0.35,
        1.0
    ) * core * depth_light;

    let pulse = 0.85 +
        0.15 * sin(t * 2.0);

    result *= pulse;

    let center_glow =
        1.0 - smoothstep(
            0.0,
            1.4,
            length(p)
        );

    result += vec3<f32>(
        0.08,
        0.025,
        0.12
    ) * center_glow;

    let edge_distance = min(
        min(
            input.uv.x,
            1.0 - input.uv.x
        ),
        min(
            input.uv.y,
            1.0 - input.uv.y
        )
    );

    let edge = 1.0 - smoothstep(
        0.0,
        0.06,
        edge_distance
    );

    result += vec3<f32>(
        0.10,
        0.025,
        0.16
    ) * edge;

    result = mix(
        result,
        input.border_color.rgb,
        border_mask
    );

    return vec4<f32>(
        result * outer_alpha,
        outer_alpha
    );
}