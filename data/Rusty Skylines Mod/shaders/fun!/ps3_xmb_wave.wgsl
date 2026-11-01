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
    @location(8) world_pos: vec2<f32>,
}

const PI: f32 = 3.14159265358979;
const WAVE_FREQ_0: f32 = 6.2831;
const WAVE_FREQ_1: f32 = 12.566;

const VIEW_DIR_RAW: vec3<f32> = vec3<f32>(3.12367, -0.0166247, -0.0852542);
const LIGHT_0_RAW: vec3<f32> = vec3<f32>(0.484729, 0.382283, -0.97268);
const LIGHT_1_RAW: vec3<f32> = vec3<f32>(2.45394, 0.923896, 0.215928);

const FRESNEL_OFFSET_R: f32 = -0.370096;
const FRESNEL_OFFSET_G: f32 = -0.439489;
const FRESNEL_OFFSET_B: f32 = -0.555144;

const AMBIENT: f32 = 0.107345;
const SPEC_POWER_0: f32 = 44.8563;
const SPEC_POWER_1: f32 = 74.1114;
const FRESNEL_AMOUNT: f32 = 0.667;

const RIBBON_TINT: vec3<f32> = vec3<f32>(0.651153, 0.758947, 1.0);
const UPPER_WHITE: vec3<f32> = vec3<f32>(1.05025, 1.05025, 1.05025);
const LOWER_WHITE: vec3<f32> = vec3<f32>(0.898039, 0.898039, 0.898039);
const BRIGHTNESS: f32 = 0.698039;
const EXPOSURE: f32 = 1.1574;
const FINAL_TINT: vec3<f32> = vec3<f32>(0.651153, 0.758947, 1.0);

const RIBBON_COUNT: i32 = 6;
const REFERENCE_HALF_WIDTH: f32 = 1080.0;
const RIBBON_SPAN: f32 = 2.4;
const RIBBON_THICKNESS: f32 = 0.06;
const RIBBON_BASE_Y: f32 = -0.2;
const RIBBON_ALPHA_SCALE: f32 = 0.8;
const RIBBON_PHASE_RATE: f32 = 0.5;
const RIBBON_PHASE_STRIDE: f32 = 0.42;
const RIBBON_WAVE_A: f32 = 0.1;
const RIBBON_WAVE_B: f32 = 0.04;
const RIBBON_NORMAL_GAIN: f32 = 0.15;
const RIBBON_NORMAL_FREQ: f32 = 12.56637;

var<private> RIBBON_X: array<f32, 6> = array<f32, 6>(1405.0, 1205.0, 1005.0, 805.0, 605.0, 405.0);

fn sd_rounded_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + r;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

fn ramp_alpha(u: f32) -> f32 {
    return sin(u * PI);
}

fn ramp_normal_xy(u: f32) -> vec2<f32> {
    return vec2<f32>(0.5, 0.5) + vec2<f32>(cos(u * RIBBON_NORMAL_FREQ), sin(u * RIBBON_NORMAL_FREQ)) * RIBBON_NORMAL_GAIN;
}

fn gradient_base(v: f32) -> vec3<f32> {
    let vc = clamp(v, 0.0, 1.0);
    let t = 1.0 - abs(vc - 0.5) * 2.0;
    return vec3<f32>(0.7 + t * 0.3, 0.8 + t * 0.2, 1.0);
}

fn fresnel_lut(x: f32) -> vec3<f32> {
    let c = clamp(x, 0.0, 1.0);
    let p = pow(1.0 - c, 5.0);
    return vec3<f32>(p, p * 0.95, p * 0.9);
}

fn ribbon_center_y(u: f32, phase: f32) -> f32 {
    return RIBBON_BASE_Y + sin(u * WAVE_FREQ_0 + phase) * RIBBON_WAVE_A + sin(u * WAVE_FREQ_1 + phase * 1.7) * RIBBON_WAVE_B;
}

fn ribbon_shade(u: f32, v: f32, view_dir: vec3<f32>, half0: vec3<f32>, half1: vec3<f32>) -> vec3<f32> {
    let nrm2 = (ramp_normal_xy(u) - vec2<f32>(0.5, 0.5)) * 2.0;
    let nz = sqrt(max(0.0, 1.0 - dot(nrm2, nrm2)));
    let normal = vec3<f32>(nrm2, nz);

    let ndoth0 = max(dot(normal, half0), 0.0);
    let ndoth1 = max(dot(normal, half1), 0.0);
    let spec0 = pow(ndoth0, SPEC_POWER_0);
    let spec1 = pow(ndoth1, SPEC_POWER_1);

    let fresnel_t = clamp(1.0 - dot(normal, view_dir), 0.0, 1.0);
    let fres = vec3<f32>(
        fresnel_lut(fresnel_t + FRESNEL_OFFSET_R * 0.01).r,
        fresnel_lut(fresnel_t + FRESNEL_OFFSET_G * 0.01).g,
        fresnel_lut(fresnel_t + FRESNEL_OFFSET_B * 0.01).b
    );

    let base = gradient_base(v);

    var color = LOWER_WHITE * AMBIENT;
    color = color + UPPER_WHITE * spec0 * 0.6;
    color = color + UPPER_WHITE * spec1 * 0.4;
    color = color + fres * FRESNEL_AMOUNT;
    color = color * RIBBON_TINT * base;
    color = vec3<f32>(1.0) - exp(-color * EXPOSURE);
    color = color * FINAL_TINT * BRIGHTNESS;

    return color;
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
    out.world_pos = world;

    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let max_round = min(input.rect_half_size.x, input.rect_half_size.y);
    let radius = input.roundness * max_round;

    let d = sd_rounded_box(
        input.local_pos,
        input.rect_half_size,
        radius
    );

    let aa = max(fwidth(d), 0.75);
    let outer_alpha = 1.0 - smoothstep(-aa, aa, d);

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

    let inner_alpha = 1.0 - smoothstep(-aa, aa, d_inner);
    let border_mask = outer_alpha - inner_alpha;

    let x_ndc = (input.world_pos.x / screen.size.x) * 2.0 - 1.0;
    let y_ndc = 1.0 - (input.world_pos.y / screen.size.y) * 2.0;

    let t = screen.time;
    let v_per_pixel = (2.0 / max(screen.size.y, 1.0)) / RIBBON_THICKNESS;

    let view_dir = normalize(VIEW_DIR_RAW);
    let light0 = normalize(LIGHT_0_RAW);
    let light1 = normalize(LIGHT_1_RAW);
    let half0 = normalize(light0 + view_dir);
    let half1 = normalize(light1 + view_dir);

    var wave = vec3<f32>(0.0);
    var wave_alpha = 0.0;

    for (var i: i32 = 0; i < RIBBON_COUNT; i = i + 1) {
        let base_x = (RIBBON_X[i] - REFERENCE_HALF_WIDTH) / REFERENCE_HALF_WIDTH;
        let u = 0.5 + (x_ndc - base_x) / RIBBON_SPAN;

        if (u <= 0.0 || u >= 1.0) {
            continue;
        }

        let phase = t * RIBBON_PHASE_RATE + f32(i) * RIBBON_PHASE_STRIDE;
        let center_y = ribbon_center_y(u, phase);
        let v = (y_ndc - center_y) / RIBBON_THICKNESS + 0.5;

        let edge = smoothstep(0.0, v_per_pixel, v) * smoothstep(0.0, v_per_pixel, 1.0 - v);

        if (edge <= 0.0) {
            continue;
        }

        let ramp_a = ramp_alpha(u);

        if (ramp_a <= 0.001) {
            continue;
        }

        let shaded = ribbon_shade(u, v, view_dir, half0, half1);
        let a = ramp_a * RIBBON_ALPHA_SCALE * edge;

        wave = wave + shaded * a;
        wave_alpha = wave_alpha + a;
    }

    let wave_masked = wave * outer_alpha;
    let backdrop = input.color.rgb * input.color.a * outer_alpha;

    var color = backdrop + wave_masked;
    var alpha = max(input.color.a * outer_alpha, clamp(wave_alpha, 0.0, 1.0) * outer_alpha);

    color = mix(color, input.border_color.rgb, border_mask);
    alpha = max(alpha, border_mask * input.border_color.a);

    return vec4<f32>(color, clamp(alpha, 0.0, 1.0));
}