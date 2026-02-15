#include "../includes/uniforms.wgsl"
@group(0) @binding(0) var rt_visibility_full_history : texture_2d<u32>;
@group(0) @binding(1) var rt_visibility_half : texture_2d<f32>;
@group(0) @binding(2) var depth_full_tex     : texture_2d<f32>;
@group(1) @binding(0) var rt_visibility_full : texture_storage_2d<r8unorm, write>;

const WG: u32 = 8u;

const SKY_DEPTH  : f32 = 10000.0;
const BASE_SIGMA : f32 = 1e-3;

// Half-res tile we cache (8×8)
var<workgroup> s_half_vis  : array<f32, 64>;
var<workgroup> s_half_dg   : array<f32, 64>; // guide depth per half texel (sampled from full depth)

// Full-res upsampled tile we cache for the bilateral stage (12×12 = 8×8 + 2-pixel halo)
var<workgroup> s_full_up   : array<f32, 144>;
var<workgroup> s_full_d    : array<f32, 144>;

fn lorentz(diff: f32, sigma: f32) -> f32 {
  let t = diff / sigma;
  return 1.0 / (1.0 + t * t);
}

// Spatial weights: a 4-tap separable “tent-ish” kernel that keeps all 16 taps non-zero.
// This corresponds to a tent sampled at half offsets: [-1.5, -0.5, +0.5, +1.5]
fn w4(i: i32) -> f32 {
  // i in {0,1,2,3} -> weights {1,3,3,1} (sum = 8)
  return select(
    select(3.0, 1.0, (i == 0) || (i == 3)),
    0.0,
    (i < 0) || (i > 3)
  );
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

  // ----------------------------
  // Load 8×8 half-res tile
  // Covers half coords needed for full tile (12×12) hard upsample.
  // For an 8×8 output group, the nominal half origin is wid*4.
  // We need a 1-texel halo for the 2×2 gather across the 12×12 full halo.
  // This origin works out to: [4w-2 .. 4w+5] => 8 texels.
  // ----------------------------
  let half_tile_origin = vec2<i32>(wid.xy) * 4 - vec2<i32>(2);

  let lxy8 = vec2<i32>(i32(li) & 7, i32(li) >> 3);
  let hxy  = clamp(half_tile_origin + lxy8, vec2<i32>(0), half_max);

  s_half_vis[li] = textureLoad(rt_visibility_half, hxy, 0).x;

  // Guide depth at corresponding full-res quad center (odd pixel)
  let guide_xy = clamp(hxy * 2 + vec2<i32>(1), vec2<i32>(0), full_max);
  s_half_dg[li] = textureLoad(depth_full_tex, guide_xy, 0).x;

  workgroupBarrier();

  // ----------------------------
  // Stage A: build 12×12 full-res tile of hard depth-matched upsample
  // Cooperative fill: 144 texels, 64 threads => ~3 iterations.
  // Full tile origin adds 2-pixel halo around the 8×8 output region.
  // ----------------------------
  let full_tile_origin = vec2<i32>(wid.xy) * 8 - vec2<i32>(2);

  for (var t: u32 = li; t < 144u; t += 64u) {
    let lxy12 = vec2<i32>(i32(t % 12u), i32(t / 12u));
    let fxy   = clamp(full_tile_origin + lxy12, vec2<i32>(0), full_max);

    let d0 = textureLoad(depth_full_tex, fxy, 0).x;
    s_full_d[t] = d0;

    // Sky/invalid depth => fully visible
    if (d0 > SKY_DEPTH) {
      s_full_up[t] = 1.0;
      continue;
    }

    // Map full-res pixel to half-res space (same mapping you used before)
    let scale = vec2<f32>(half_dims) / vec2<f32>(full_dims); // typically 0.5
    let coord = (vec2<f32>(fxy) + 0.5) * scale - 0.5;
    let base  = vec2<i32>(floor(coord));

    // base is in half coords; convert to local shared half tile coords
    // Clamp so (lb + {0,1} + {0,1}) stays within 0..7
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

  // ----------------------------
  // Output pixel for this thread (center 8×8 region inside 12×12 cache)
  // ----------------------------
  if (gid.x >= full_dims.x || gid.y >= full_dims.y) {
    return;
  }

  let out_lxy = vec2<i32>(i32(li) & 7, i32(li) >> 3); // 0..7
  let cxy12   = out_lxy + vec2<i32>(2);               // shift into 12×12 (halo=2)
  let cidx    = u32(cxy12.x + cxy12.y * 12);

  let d_center = s_full_d[cidx];
  if (d_center > SKY_DEPTH) {
    textureStore(rt_visibility_full, vec2<i32>(gid.xy), vec4<f32>(1.0));
    return;
  }

  // ----------------------------
  // Stage B: 4×4 full-res bilateral over the upsampled tile
  // sigma = max(BASE_SIGMA, depth_extent over 4×4)
  // ----------------------------
  var depth_extent: f32 = 0.0;
  for (var oy: i32 = 0; oy < 4; oy++) {
    for (var ox: i32 = 0; ox < 4; ox++) {
      let sx = cxy12.x + (ox - 1); // -1..+2
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
      let w_depth   = lorentz(abs(s_full_d[sidx] - d_center), sigma);
      let w         = w_spatial * w_depth;

      vsum += s_full_up[sidx] * w;
      wsum += w;
    }
  }

  let vis = select(1.0, vsum / wsum, wsum > 1e-8);
  textureStore(rt_visibility_full, vec2<i32>(gid.xy), vec4<f32>(vis));
}
