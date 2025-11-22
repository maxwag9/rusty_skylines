
struct ScreenUniform {
    size: vec2<f32>,
    time: f32,
    enable_dither: u32,
    mouse: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> screen: ScreenUniform;


// 4x4 Bayer matrix, normalized to [0,1)!!
const DITHER_MATRIX : array<array<f32, 4>, 4> = array(
    array( 0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0 ),
    array(12.0/16.0,  4.0/16.0, 14.0/16.0,  6.0/16.0 ),
    array( 3.0/16.0, 11.0/16.0,  1.0/16.0,  9.0/16.0 ),
    array(15.0/16.0,  7.0/16.0, 13.0/16.0,  5.0/16.0 )
);


struct CircleParams {
    // (cx, cy, radius, border)
    center_radius_border: vec4<f32>,
    fade: vec4<f32>,
    // colors + alpha :)
    fill_color:   vec4<f32>,
    border_color: vec4<f32>,
    glow_color:   vec4<f32>,
    // (glow_size, glow_speed, glow_intensity, pad)
    glow_misc:    vec4<f32>,
    misc:         vec4<f32>,
};

fn hash(p: vec2<f32>) -> f32 {
    // deterministic pseudo-random hash
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}


@group(1) @binding(0)
var<storage, read> circles: array<CircleParams>;


struct VertexInput {
    @location(0) pos: vec2<f32>,
    @location(1) color: vec4<f32>,
    @builtin(instance_index) instance: u32,
};

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) local_pos: vec2<f32>,
    @location(2) circle_index: u32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let params = circles[in.instance];
    let glow_size = params.glow_misc.x;
    let crb = params.center_radius_border;

    let center = crb.xy;
    let radius = crb.z;

    // quad covers full circle + halo
    let world = center + in.pos * (radius + glow_size);




    let x = (world.x / screen.size.x) * 2.0 - 1.0;
    let y = 1.0 - (world.y / screen.size.y) * 2.0;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.local_pos = world;
    out.circle_index = in.instance;
    out.color = vec4<f32>(1.0, 0.0, 0.0, 0.2);


    return out;

}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = circles[in.circle_index];

    let center      = p.center_radius_border.xy;
    let radius      = p.center_radius_border.z;
    let glow_size   = p.glow_misc.x;
    let glow_speed  = p.glow_misc.y;
    let glow_intensity = p.glow_misc.z;
    let held_time   = p.misc.y;
    let is_down     = p.misc.z;
    let id_hash     = p.misc.w;

    // radial distance from center
    let dist = distance(in.local_pos, center);

    // ---------------------------------------------------------------
    // Base pulsing glow
    // ---------------------------------------------------------------
    let pulse = 0.5 + 0.5 * sin(screen.time * glow_speed + id_hash * 314.15);
    let base_glow = mix(0.15, 1.0, pulse) * glow_intensity;

    // ---------------------------------------------------------------
    // Press animation (flash on click)
    // ---------------------------------------------------------------
    let flash = exp(-held_time * 4.0) * is_down;

    // radial mask from inner edge of glow to outer falloff
    // 1.0 at radius, 0.0 at radius + glow_size
    let radial_mask = 1.0 - smoothstep(radius, radius + glow_size, dist);

    // combine pulsing and press effects
    let glow_strength = radial_mask * (base_glow + flash);

    // ---------------------------------------------------------------
    // Apply color + alpha
    // ---------------------------------------------------------------
    var col = vec4<f32>(p.glow_color.rgb * glow_strength, glow_strength);

    // ---------------------------------------------------------------
    // Bayer dither (optional)
    // use pixel-ish coordinates, not radius based
    // ---------------------------------------------------------------
    let px = i32(floor(in.local_pos.x)) & 3;
    let py = i32(floor(in.local_pos.y)) & 3;
    let dither = DITHER_MATRIX[py][px] - 0.5;

    // ---------------------------------------------------------------
    // Temporal noise scaled by radial glow
    // ---------------------------------------------------------------
    // raw noise in [-0.5, 0.5]
    let n = hash(in.local_pos.xy + vec2<f32>(screen.time, screen.time * 37.0)) - 0.5;

    // make noise strongest near the bright part and fade out toward outer edge
    // square the mask for a steeper falloff at the edge
    let noise_scale = (radial_mask * radial_mask);

    // final offset is tiny and radially attenuated
    let offset = n * (3.0 / 255.0) * noise_scale;

    col = vec4<f32>(col.rgb + vec3<f32>(offset), col.a);
    col.r = 1.0;
    return col;
}

