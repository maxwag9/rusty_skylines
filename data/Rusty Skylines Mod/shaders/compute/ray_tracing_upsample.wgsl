#include "../includes/uniforms.wgsl"

@group(0) @binding(0) var rt_history          : texture_2d<f32>;   // previous frame TAA result (full res)
@group(0) @binding(1) var rt_visibility_half  : texture_2d<f32>;   // current raw half-res RT
@group(0) @binding(2) var depth_full_tex      : texture_2d<f32>;   // linear depth full res
@group(0) @binding(3) var motion_tex          : texture_2d<f32>;   // motion vectors (Rg16Float)

@group(1) @binding(0) var rt_out              : texture_storage_2d<r8unorm, write>;  // TAA output

const WG: u32 = 8u;
const SKY_DEPTH: f32 = 10000.0;
const BASE_SIGMA: f32 = 1e-3;
const TAA_BLEND: f32 = 0.05;       // weight for current frame (lower = more temporal smoothing)
const TAA_BLEND_CLAMP: f32 = 0.2;  // weight when history is clamped (trust current more)

// ── Workgroup shared memory ──
// Half-res tile (8×8)
var<workgroup> s_half_vis : array<f32, 64>;
var<workgroup> s_half_dg  : array<f32, 64>;

// Full-res upsampled tile (12×12 = 8×8 + 2-pixel halo)
var<workgroup> s_full_up  : array<f32, 144>;
var<workgroup> s_full_d   : array<f32, 144>;

fn lorentz(diff: f32, sigma: f32) -> f32 {
    let t = diff / sigma;
    return 1.0 / (1.0 + t * t);
}

fn w4(i: i32) -> f32 {
    return select(
        select(3.0, 1.0, (i == 0) || (i == 3)),
        0.0,
        (i < 0) || (i > 3)
    );
}

// ── Neighborhood clamp for TAA ──
// Computes min/max of a 3×3 neighborhood in the upsampled tile
// to clamp the history sample and prevent ghosting.
fn neighborhood_minmax(cxy12: vec2<i32>) -> vec2<f32> {
    var lo = 1.0f;
    var hi = 0.0f;
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let sx = clamp(cxy12.x + dx, 0, 11);
            let sy = clamp(cxy12.y + dy, 0, 11);
            let v = s_full_up[u32(sx + sy * 12)];
            lo = min(lo, v);
            hi = max(hi, v);
        }
    }
    return vec2<f32>(lo, hi);
}

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let full_dims = textureDimensions(depth_full_tex);
    let half_dims = textureDimensions(rt_visibility_half);
    let full_max = vec2<i32>(full_dims) - vec2<i32>(1);
    let half_max = vec2<i32>(half_dims) - vec2<i32>(1);

    // ════════════════════════════════════════════
    // Stage 0: Load 8×8 half-res tile into shared
    // ════════════════════════════════════════════
    let half_tile_origin = vec2<i32>(wid.xy) * 4 - vec2<i32>(2);
    let lxy8 = vec2<i32>(i32(li) & 7, i32(li) >> 3);
    let hxy = clamp(half_tile_origin + lxy8, vec2<i32>(0), half_max);

    s_half_vis[li] = textureLoad(rt_visibility_half, hxy, 0).x;

    let guide_xy = clamp(hxy * 2 + vec2<i32>(1), vec2<i32>(0), full_max);
    s_half_dg[li] = textureLoad(depth_full_tex, guide_xy, 0).x;

    workgroupBarrier();

    // ════════════════════════════════════════════════════
    // Stage A: Build 12×12 full-res upsampled tile (depth-matched)
    // ════════════════════════════════════════════════════
    let full_tile_origin = vec2<i32>(wid.xy) * 8 - vec2<i32>(2);

    for (var t: u32 = li; t < 144u; t += 64u) {
        let lxy12 = vec2<i32>(i32(t % 12u), i32(t / 12u));
        let fxy = clamp(full_tile_origin + lxy12, vec2<i32>(0), full_max);

        let d0 = textureLoad(depth_full_tex, fxy, 0).x;
        s_full_d[t] = d0;

        if (d0 > SKY_DEPTH) {
            s_full_up[t] = 1.0;
            continue;
        }

        let scale = vec2<f32>(half_dims) / vec2<f32>(full_dims);
        let coord = (vec2<f32>(fxy) + 0.5) * scale - 0.5;
        let base = vec2<i32>(floor(coord));
        let lb = clamp(base - half_tile_origin, vec2<i32>(0), vec2<i32>(6));

        let row0 = lb.y * 8;
        let row1 = row0 + 8;

        let i00 = u32(lb.x + row0);
        let i10 = u32(lb.x + 1 + row0);
        let i01 = u32(lb.x + row1);
        let i11 = u32(lb.x + 1 + row1);

        let v = vec4<f32>(
            s_half_vis[i00], s_half_vis[i10],
            s_half_vis[i01], s_half_vis[i11]
        );
        let dg = vec4<f32>(
            s_half_dg[i00], s_half_dg[i10],
            s_half_dg[i01], s_half_dg[i11]
        );

        let diff = abs(dg - vec4<f32>(d0));

        var hard_vis = v.x;
        var best = diff.x;
        if (diff.y < best) { hard_vis = v.y; best = diff.y; }
        if (diff.z < best) { hard_vis = v.z; best = diff.z; }
        if (diff.w < best) { hard_vis = v.w; }

        s_full_up[t] = hard_vis;
    }

    workgroupBarrier();

    // ════════════════════════════════════════
    // Early out for out-of-bounds threads
    // ════════════════════════════════════════
    if (gid.x >= full_dims.x || gid.y >= full_dims.y) {
        return;
    }

    let out_lxy = vec2<i32>(i32(li) & 7, i32(li) >> 3);
    let cxy12 = out_lxy + vec2<i32>(2);
    let cidx = u32(cxy12.x + cxy12.y * 12);

    let d_center = s_full_d[cidx];

    // Sky pixels: no TAA needed
    if (d_center > SKY_DEPTH) {
        textureStore(rt_out, vec2<i32>(gid.xy), vec4<f32>(1.0));
        return;
    }

    // ════════════════════════════════════════
    // Stage B: 4×4 bilateral filter on upsampled data
    // ════════════════════════════════════════
    var depth_extent: f32 = 0.0;
    for (var oy: i32 = 0; oy < 4; oy++) {
        for (var ox: i32 = 0; ox < 4; ox++) {
            let sx = cxy12.x + (ox - 1);
            let sy = cxy12.y + (oy - 1);
            let sidx = u32(sx + sy * 12);
            depth_extent = max(depth_extent, abs(s_full_d[sidx] - d_center));
        }
    }

    let sigma = max(BASE_SIGMA, depth_extent);

    var wsum: f32 = 0.0;
    var vsum: f32 = 0.0;

    for (var oy: i32 = 0; oy < 4; oy++) {
        let wy = w4(oy);
        for (var ox: i32 = 0; ox < 4; ox++) {
            let wx = w4(ox);
            let sx = cxy12.x + (ox - 1);
            let sy = cxy12.y + (oy - 1);
            let sidx = u32(sx + sy * 12);

            let w_spatial = wx * wy;
            let w_depth = lorentz(abs(s_full_d[sidx] - d_center), sigma);
            let w = w_spatial * w_depth;

            vsum += s_full_up[sidx] * w;
            wsum += w;
        }
    }

    let current_vis = select(1.0, vsum / wsum, wsum > 1e-8);

    // ════════════════════════════════════════
    // Stage C: TAA — reproject and blend with history
    // ════════════════════════════════════════

    // Read motion vector at this pixel
    let motion = textureLoad(motion_tex, vec2<i32>(gid.xy), 0).xy;

    // Compute reprojected UV
    let curr_uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(full_dims);
    let prev_uv = curr_uv - motion;

    // Check if reprojected coordinate is on-screen
    let on_screen = all(prev_uv >= vec2<f32>(0.0)) && all(prev_uv < vec2<f32>(1.0));

    var result = current_vis;

    if (on_screen) {
        // Sample history with bilinear interpolation via integer taps
        let prev_pixel = prev_uv * vec2<f32>(full_dims) - 0.5;
        let base_px = vec2<i32>(floor(prev_pixel));
        let frac = prev_pixel - vec2<f32>(base_px);

        let p00 = clamp(base_px, vec2<i32>(0), full_max);
        let p10 = clamp(base_px + vec2<i32>(1, 0), vec2<i32>(0), full_max);
        let p01 = clamp(base_px + vec2<i32>(0, 1), vec2<i32>(0), full_max);
        let p11 = clamp(base_px + vec2<i32>(1, 1), vec2<i32>(0), full_max);

        let h00 = textureLoad(rt_history, p00, 0).x;
        let h10 = textureLoad(rt_history, p10, 0).x;
        let h01 = textureLoad(rt_history, p01, 0).x;
        let h11 = textureLoad(rt_history, p11, 0).x;

        let history_val = mix(
            mix(h00, h10, frac.x),
            mix(h01, h11, frac.x),
            frac.y
        );

        // Neighborhood clamp: prevent ghosting by clamping history
        // to the range of values in the current frame's 3×3 neighborhood
        let minmax = neighborhood_minmax(cxy12);
        let clamped_history = clamp(history_val, minmax.x, minmax.y);

        // Choose blend factor: trust current more if we had to clamp
        let was_clamped = abs(history_val - clamped_history) > 0.01;
        let blend = select(TAA_BLEND, TAA_BLEND_CLAMP, was_clamped);

        result = mix(clamped_history, current_vis, blend);
    }

    textureStore(rt_out, vec2<i32>(gid.xy), vec4<f32>(result));
}
