struct ScreenUniform {
    size: vec2<f32>,
    time: f32,
    enable_dither: u32,
    mouse: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> screen: ScreenUniform;

const DITHER_MATRIX : array<array<f32, 4>, 4> = array(
    array( 0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0 ),
    array(12.0/16.0,  4.0/16.0, 14.0/16.0,  6.0/16.0 ),
    array( 3.0/16.0, 11.0/16.0,  1.0/16.0,  9.0/16.0 ),
    array(15.0/16.0,  7.0/16.0, 13.0/16.0,  5.0/16.0 )
);

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
    // (glow_size, glow_speed, glow_intensity, pad)
    glow_misc: vec4<f32>,
    misc: vec4<f32>, // active, touched_time, is_down, hash
};

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn rot2(a: f32) -> mat2x2<f32> {
    let c = cos(a);
    let s = sin(a);
    return mat2x2<f32>(c, -s, s, c);
}

// Signed distance to a rounded rectangle in local space.
// p is in rect-local coordinates, centered at 0.
fn sd_round_rect(p: vec2<f32>, half_size: vec2<f32>, r: f32) -> f32 {
    let rr = clamp(r, 0.0, min(half_size.x, half_size.y));
    let q = abs(p) - (half_size - vec2<f32>(rr, rr));
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - rr;
}

@group(1) @binding(0)
var<storage, read> rects: array<RectGpu>;

struct VertexInput {
    @location(0) pos: vec2<f32>,   // quad corners, usually [-1, 1]
    @location(1) color: vec4<f32>,
    @builtin(instance_index) instance: u32,
};

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) world_pos: vec2<f32>,
    @location(1) local_pos: vec2<f32>,
    @location(2) rect_index: u32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let p = rects[in.instance];
    let glow_size = p.glow_misc.x;
    if (glow_size < 1.0) {
        out.pos = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.world_pos = vec2<f32>(0.0);
        out.local_pos = vec2<f32>(0.0);
        out.rect_index = in.instance;
        return out;
    }
    let local = in.pos * (p.half_size + vec2<f32>(glow_size, glow_size));
    let rotated = rot2(p.rotation) * local;
    let world = p.center + rotated;

    let x = (world.x / screen.size.x) * 2.0 - 1.0;
    let y = 1.0 - (world.y / screen.size.y) * 2.0;

    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.world_pos = world;
    out.local_pos = local;
    out.rect_index = in.instance;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = rects[in.rect_index];

    let glow_size = p.glow_misc.x;
    let glow_speed = p.glow_misc.y;
    let glow_intensity = p.glow_misc.z;
    let held_time = p.misc.y;
    let is_down = p.misc.z;
    let id_hash = p.misc.w;

    let local_rot = rot2(-p.rotation) * (in.world_pos - p.center);

    let max_round = min(p.half_size.x, p.half_size.y);
    let roundness = clamp(p.roundness, 0.0, 1.0) * max_round;

    let dist = sd_round_rect(local_rot, p.half_size, roundness);

    let edge_dist = max(dist, 0.0);
    let radial_mask = step(0.0, dist) * (1.0 - smoothstep(0.0, glow_size, edge_dist));

    let pulse = 0.5 + 0.5 * sin(screen.time * glow_speed + id_hash * 314.15);
    let base_glow = mix(0.15, 1.0, pulse) * glow_intensity;
    let flash = exp(-held_time * 4.0) * is_down;

    let glow_strength = radial_mask * (base_glow + flash);

    var col = vec4<f32>(p.glow_color.rgb * glow_strength, p.glow_color.a * glow_strength);

    if (screen.enable_dither != 0u) {
        let px = i32(floor(in.world_pos.x)) & 3;
        let py = i32(floor(in.world_pos.y)) & 3;
        let dither = DITHER_MATRIX[py][px] - 0.5;
        col = vec4<f32>(col.rgb + vec3<f32>(dither * (3.0 / 255.0) * glow_strength), col.a);
    }

    let n = hash(in.world_pos.xy + vec2<f32>(screen.time, screen.time * 37.0)) - 0.5;
    let noise_scale = radial_mask * radial_mask;
    let offset = n * (3.0 / 255.0) * noise_scale;
    col = vec4<f32>(col.rgb + vec3<f32>(offset), col.a);

    if (col.a < 0.01) {
        col = vec4<f32>(0.0);
    }

    return col;
}