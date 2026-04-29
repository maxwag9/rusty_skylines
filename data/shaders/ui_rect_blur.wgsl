struct ScreenUniform {
    size: vec2<f32>,
    time: f32,
    enable_dither: u32,
    mouse: vec2<f32>,
};

@group(0) @binding(0) var hdr_sampler: sampler;
@group(0) @binding(1) var hdr_tex: texture_2d<f32>;
@group(1) @binding(0) var<uniform> screen: ScreenUniform;
@group(2) @binding(0) var<storage, read> rects: array<RectGpu>;

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
    glow_misc: vec4<f32>,// (glow_size, glow_speed, glow_intensity, pad)
    misc: vec4<f32>, // active, touched_time, is_down, hash
    blur: f32,
    //_pad0: vec3<f32>
};

// ── helpers ──────────────────────────────────────────────────────────────────

fn rot2(a: f32) -> mat2x2<f32> {
    let c = cos(a); let s = sin(a);
    return mat2x2<f32>(c, -s, s, c);
}

fn sd_round_rect(p: vec2<f32>, half_size: vec2<f32>, r: f32) -> f32 {
    let rr = clamp(r, 0.0, min(half_size.x, half_size.y));
    let q  = abs(p) - (half_size - vec2<f32>(rr));
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - rr;
}

// Golden-angle spiral disc blur.
// Samples are uniformly distributed over the disc so every pixel
// gets the same expected number of neighbours regardless of angle.
const GOLDEN_ANGLE: f32  = 2.39996322972;   // radians  (π(3 − √5))
const NUM_SAMPLES:  i32  = 32;


fn disc_blur(center_uv: vec2<f32>, radius_px: f32) -> vec4<f32> {
    let px    = 1.0 / screen.size;           // one texel in UV space
    var col   = vec4<f32>(0.0);
    var total = 0.0;

    for (var i: i32 = 0; i < NUM_SAMPLES; i++) {
        // sqrt gives uniform area coverage across the disc
        let t      = (f32(i) + 0.5) / f32(NUM_SAMPLES);
        let r      = sqrt(t) * radius_px;
        let angle  = f32(i) * GOLDEN_ANGLE;
        let offset = vec2<f32>(cos(angle), sin(angle)) * r * px;

        // Gaussian weight so the centre contributes more than the rim
        let w  = exp(-3.0 * t);              // t ∈ [0,1] → w ∈ [1, e⁻³]
        col   += textureSampleLevel(hdr_tex, hdr_sampler, center_uv + offset, 0.0) * w;
        total += w;
    }

    // Always include the exact centre sample
    col   += textureSampleLevel(hdr_tex, hdr_sampler, center_uv, 0.0);
    total += 1.0;

    return col / total;
}

// ── vertex ───────────────────────────────────────────────────────────────────

struct VertexInput {
    @location(0) pos:   vec2<f32>,
    @location(1) color: vec4<f32>,
    @builtin(instance_index) instance: u32,
};

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) world_pos:  vec2<f32>,
    @location(1) rect_index: u32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let p = rects[in.instance];

    // 1.5 px fringe so the AA gradient never gets clipped by the quad edge
    let local   = in.pos * (p.half_size + vec2<f32>(1.5));
    let rotated = rot2(p.rotation) * local;
    let world   = p.center + rotated;

    out.pos        = vec4<f32>((world.x / screen.size.x) * 2.0 - 1.0,
                                1.0 - (world.y / screen.size.y) * 2.0,
                                0.0, 1.0);
    out.world_pos  = world;
    out.rect_index = in.instance;
    return out;
}

// ── fragment ─────────────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = rects[in.rect_index];

    // SDF in rect-local (rotated) space
    let local_pos = rot2(-p.rotation) * (in.world_pos - p.center);
    let max_round = min(p.half_size.x, p.half_size.y);
    let roundness = clamp(p.roundness, 0.0, 1.0) * max_round;
    let dist      = sd_round_rect(local_pos, p.half_size, roundness);

    // Outer AA: 1 px smooth transition at the rounded boundary
    let edge_alpha = 1.0 - smoothstep(-1.0, 0.0, dist);
    if (edge_alpha < 0.001) { discard; }

    // Inner vignette: fade field softens the interior towards the edges
    // (dist is negative inside, so –dist is the distance from the rim inward)
    let inner_alpha = select(
        1.0,
        smoothstep(0.0, max(p.fade, 0.001), -dist),
        p.fade > 0.0
    );

    // Blur the HDR content beneath the rect
    let uv      = in.world_pos / screen.size;
    var blurred = disc_blur(uv, max(p.blur, 0.0));

    // Tint: lerp between the raw blur and the rect's colour (alpha-weighted)
    let tint_a = p.color.a;
    blurred    = vec4<f32>(mix(blurred.rgb, p.color.rgb, tint_a), blurred.a);

    let final_alpha = edge_alpha * inner_alpha;
    return vec4<f32>(blurred.rgb, final_alpha);
}