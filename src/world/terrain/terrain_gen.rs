#![allow(dead_code, unused_variables)]
use crate::helpers::hsv::{HSV, hsv_to_rgb, lerp_hsv};
use crate::helpers::positions::{ChunkCoord, WorldPos, chunk_size};
use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};
use std::f32::consts::PI;
use wgpu::Extent3d;

const TAU: f32 = PI * 2.0;

fn hash01(mut x: u32) -> f32 {
    x ^= x >> 13;
    x = x.wrapping_mul(0x85ebca6b);
    x ^= x >> 16;
    (x as f32) / (u32::MAX as f32)
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn smootherstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
fn bias(x: f32, b: f32) -> f32 {
    let b = b.clamp(0.0001, 0.9999);
    x / ((1.0 / b - 2.0) * (1.0 - x) + 1.0)
}

#[inline]
fn gain(x: f32, g: f32) -> f32 {
    if x < 0.5 {
        0.5 * bias(2.0 * x, 1.0 - g)
    } else {
        1.0 - 0.5 * bias(2.0 - 2.0 * x, 1.0 - g)
    }
}

#[inline]
fn micro_flatten(rel: f32, strength: f32) -> f32 {
    let a = rel.abs();
    let t = smoothstep(0.0, 0.15, a);
    if strength <= 0.0 {
        rel
    } else {
        lerp(rel, rel * 0.4, strength * (1.0 - t))
    }
}

#[inline]
fn ridged(n: f32) -> f32 {
    let r = 1.0 - n.abs();
    if r <= 0.0 { 0.0 } else { r * r * r }
}

// ─────────────────────────────────────────────────────────────────────────────
// f64 helpers for jitter-free world coordinates
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn world_xz_f64(p: &WorldPos) -> (f64, f64) {
    let cs = chunk_size() as f64;
    let wx = p.chunk.x as f64 * cs + p.local.x as f64;
    let wz = p.chunk.z as f64 * cs + p.local.z as f64;
    (wx, wz)
}

#[inline]
fn noise2_f64(n: &FastNoiseLite, x: f64, z: f64, scale: f64) -> f32 {
    n.get_noise_2d((x * scale) as f32, (z * scale) as f32)
}

#[inline]
fn noise2_f64_raw(n: &FastNoiseLite, x: f64, z: f64) -> f32 {
    n.get_noise_2d(x as f32, z as f32)
}

#[inline]
fn grad2_f64(noise: &FastNoiseLite, x: f64, z: f64, eps: f64) -> (f32, f32) {
    let a = noise.get_noise_2d((x + eps) as f32, z as f32);
    let b = noise.get_noise_2d((x - eps) as f32, z as f32);
    let c = noise.get_noise_2d(x as f32, (z + eps) as f32);
    let d = noise.get_noise_2d(x as f32, (z - eps) as f32);
    let inv_2eps = 1.0 / (2.0 * eps as f32);
    ((a - b) * inv_2eps, (c - d) * inv_2eps)
}

// ─────────────────────────────────────────────────────────────────────────────
// TerrainParams
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct TerrainParams {
    pub seed: u32,
    pub world_scale: f32,

    pub height_scale: f32,
    pub sea_level: f32,

    pub lat_extent: f32,
    pub continent_radius: f32,
    pub ring_radius: f32,
    pub coast_noise_scale: f32,
    pub coast_noise_amp: f32,
    pub island_threshold0: f32,
    pub island_threshold1: f32,
    pub island_amp: f32,

    pub force_land_at_origin: bool,
    pub origin_island_radius: f32,
    pub origin_min_height: f32,
    pub pull_one_continent_to_origin: bool,
    pub origin_pull_strength: f32,

    pub warp_large_scale: f32,
    pub warp_small_scale: f32,
    pub warp_large_amp: f32,
    pub warp_small_amp: f32,
    pub warp_mix_large: f32,
    pub warp_mix_small: f32,

    pub macro_freq: f32,
    pub hills_freq: f32,
    pub mountains_freq: f32,
    pub _belts_freq: f32,
    pub moisture_freq: f32,

    pub macro_octaves: usize,
    pub macro_persistence: f32,
    pub hills_octaves: usize,
    pub hills_persistence: f32,
    pub mountains_octaves: usize,
    pub mountains_persistence: f32,
    pub continent_octaves: usize,
    pub continent_persistence: f32,
    pub moisture_octaves: usize,
    pub moisture_persistence: f32,
    pub warp_large_octaves: usize,
    pub warp_large_persistence: f32,
    pub warp_small_octaves: usize,
    pub warp_small_persistence: f32,

    pub ocean_floor: f32,
    pub inland_plateau: f32,
    pub macro_amp: f32,
    pub hills_amp: f32,
    pub mountains_amp: f32,
    pub belt_amp: f32,

    pub coast_soften_width: f32,
    pub coast_soften_strength: f32,
    pub interior_lo: f32,
    pub interior_hi: f32,
    pub belt_lo: f32,
    pub belt_hi: f32,

    pub flatten: f32,
    pub flatten_curve: f32,
    pub mountain_smooth: f32,
    pub hills_detail: f32,
    pub micro_flatten: f32,

    pub plate_freq: f32,
    pub plate_mountain_amp: f32,

    pub erosion_strength: f32,
    pub erosion_iters: usize,

    pub river_freq: f32,
    pub river_depth: f32,
    pub river_width: f32,

    pub snow_slope_limit: f32,
}

impl Default for TerrainParams {
    fn default() -> Self {
        Self {
            seed: 0,
            world_scale: 0.04,

            height_scale: 1000.0,
            sea_level: 0.0,

            lat_extent: 140_000.0,
            continent_radius: 70_000.0,

            ring_radius: 48_000.0,

            coast_noise_scale: 0.0035,
            coast_noise_amp: 0.45,
            island_threshold0: 0.74,
            island_threshold1: 0.92,
            island_amp: 0.55,

            force_land_at_origin: true,
            origin_island_radius: 10_000.0,
            origin_min_height: 50.0,
            pull_one_continent_to_origin: false,

            origin_pull_strength: 0.0,

            warp_large_scale: 0.00042,
            warp_small_scale: 0.0040,
            warp_large_amp: 160.0,
            warp_small_amp: 110.0,
            warp_mix_large: 0.65,
            warp_mix_small: 0.35,

            macro_freq: 0.00028,

            hills_freq: 0.0016,
            mountains_freq: 0.0035,
            _belts_freq: 0.00162,
            moisture_freq: 0.0008,

            macro_octaves: 5,
            macro_persistence: 0.50,
            hills_octaves: 6,
            hills_persistence: 0.52,

            mountains_octaves: 5,
            mountains_persistence: 0.58,

            continent_octaves: 4,
            continent_persistence: 0.85,
            moisture_octaves: 5,
            moisture_persistence: 0.60,
            warp_large_octaves: 4,
            warp_large_persistence: 0.68,
            warp_small_octaves: 3,
            warp_small_persistence: 0.52,

            ocean_floor: -1.10,
            inland_plateau: 0.88,
            macro_amp: 0.2,
            hills_amp: 1.05,

            mountains_amp: 1.75,

            belt_amp: 0.85,

            coast_soften_width: 0.22,
            coast_soften_strength: 0.1,
            interior_lo: 0.38,
            interior_hi: 0.92,
            belt_lo: 0.50,
            belt_hi: 0.78,

            flatten: 0.06,
            flatten_curve: 1.8,
            mountain_smooth: 0.55,

            hills_detail: 0.45,

            micro_flatten: 0.05,

            plate_freq: 0.0022,
            plate_mountain_amp: 1.8,

            erosion_strength: 0.20,
            erosion_iters: 24,

            river_freq: 0.0011,
            river_depth: 0.09,
            river_width: 0.28,

            snow_slope_limit: 0.55,
        }
    }
}

fn make_fbm(seed: u32, freq: f32, oct: usize, gain: f32) -> FastNoiseLite {
    let mut n = FastNoiseLite::new();
    n.set_seed(Some(seed as i32));
    n.set_noise_type(Some(NoiseType::Perlin));
    n.set_fractal_type(Some(FractalType::FBm));
    n.set_fractal_octaves(Some(oct as i32));
    n.set_frequency(Some(freq));
    n.set_fractal_gain(Some(gain));
    n
}

#[derive(Clone, Copy)]
struct BaseSample {
    cont: f32,
    wx2: f64,
    wz2: f64,
    rel: f32,
    _mountain_mask: f32,
    _uplift: f32,
    slope_proxy: f32,
}

pub struct TerrainGenerator {
    p: TerrainParams,

    macro_elev: FastNoiseLite,
    hills: FastNoiseLite,
    mountains: FastNoiseLite,
    plates: FastNoiseLite,
    rivers: FastNoiseLite,

    continent_noise: FastNoiseLite,

    continent_large: FastNoiseLite,
    continent_ridge: FastNoiseLite,
    continent_warp: FastNoiseLite,

    moisture_noise: FastNoiseLite,
    warp_large: FastNoiseLite,
    warp_small: FastNoiseLite,

    detail: FastNoiseLite,
    rock: FastNoiseLite,

    continent_shape_noise: FastNoiseLite,
}

impl Clone for TerrainGenerator {
    fn clone(&self) -> Self {
        Self::new(self.p)
    }
}

impl TerrainGenerator {
    pub fn new(terrain_params: TerrainParams) -> Self {
        Self::with_params(terrain_params)
    }

    pub fn with_params(mut p: TerrainParams) -> Self {
        let seed = p.seed;
        p.world_scale = p.world_scale.max(0.000001);

        let macro_elev = make_fbm(seed, p.macro_freq, p.macro_octaves, p.macro_persistence);
        let hills = make_fbm(
            seed.wrapping_add(10),
            p.hills_freq,
            p.hills_octaves,
            p.hills_persistence,
        );
        let mountains = make_fbm(
            seed.wrapping_add(1),
            p.mountains_freq,
            p.mountains_octaves,
            p.mountains_persistence,
        );
        let plates = make_fbm(seed.wrapping_add(42), p.plate_freq, 3, 0.5);
        let rivers = make_fbm(seed.wrapping_add(77), p.river_freq.max(0.0), 4, 0.55);

        let continent_noise = make_fbm(
            seed.wrapping_add(2),
            0.0003,
            p.continent_octaves,
            p.continent_persistence,
        );
        let continent_large = make_fbm(seed.wrapping_add(100), 0.000045, 3, 0.52);

        let continent_ridge = make_fbm(seed.wrapping_add(101), 0.00012, 4, 0.58);

        let continent_warp = make_fbm(seed.wrapping_add(102), 0.00009, 3, 0.55);
        let continent_shape_noise = make_fbm(seed.wrapping_add(3), 0.0012, 5, 0.58);

        let moisture_noise = make_fbm(
            seed,
            p.moisture_freq,
            p.moisture_octaves,
            p.moisture_persistence,
        );

        let warp_large = make_fbm(
            seed.wrapping_add(4),
            p.warp_large_scale,
            p.warp_large_octaves,
            p.warp_large_persistence,
        );
        let warp_small = make_fbm(
            seed.wrapping_add(5),
            p.warp_small_scale,
            p.warp_small_octaves,
            p.warp_small_persistence,
        );

        let detail = make_fbm(seed.wrapping_add(9001), 0.020, 4, 0.55);
        let rock = make_fbm(seed.wrapping_add(9002), 0.012, 3, 0.55);

        Self {
            p,
            macro_elev,
            hills,
            mountains,
            plates,
            rivers,
            continent_noise,
            continent_large,
            continent_ridge,
            continent_warp,
            continent_shape_noise,
            moisture_noise,
            warp_large,
            warp_small,
            detail,
            rock,
        }
    }

    #[inline]
    fn scaled_coords_f64(&self, p: &WorldPos) -> (f64, f64) {
        let (wx, wz) = world_xz_f64(p);
        let s = self.p.world_scale as f64;
        (wx * s, wz * s)
    }

    fn warped_coords_f64(&self, p: &WorldPos) -> (f64, f64) {
        let (sx, sz) = self.scaled_coords_f64(p);

        let w1x = noise2_f64_raw(&self.warp_large, sx, sz);
        let w1z = noise2_f64_raw(&self.warp_large, sx + 1234.0, sz - 5678.0);

        let w2x = noise2_f64(&self.warp_small, sx, sz, 2.0);
        let w2z = noise2_f64(&self.warp_small, sx - 500.0, sz + 500.0, 2.0);

        let dx = (w1x * self.p.warp_large_amp * self.p.warp_mix_large
            + w2x * self.p.warp_small_amp * self.p.warp_mix_small) as f64;
        let dz = (w1z * self.p.warp_large_amp * self.p.warp_mix_large
            + w2z * self.p.warp_small_amp * self.p.warp_mix_small) as f64;

        (sx + dx, sz + dz)
    }

    fn continental_mask(&self, p: &WorldPos) -> f32 {
        let (sx, sz) = self.scaled_coords_f64(p);
        let warp_amp = 14000.0;

        let warp_x = noise2_f64_raw(&self.continent_warp, sx * 0.7, sz * 0.7) as f64;

        let warp_z =
            noise2_f64_raw(&self.continent_warp, sx * 0.7 + 4000.0, sz * 0.7 - 2000.0) as f64;

        let wx = sx + warp_x * warp_amp;
        let wz = sz + warp_z * warp_amp;

        let base = noise2_f64_raw(&self.continent_large, wx, wz);

        let breakup = noise2_f64_raw(&self.continent_noise, wx * 1.8, wz * 1.8);

        let ridge_raw = noise2_f64_raw(&self.continent_ridge, wx * 0.22, wz * 0.08);

        let ridge = ridged(ridge_raw) * 0.65;

        let shape = noise2_f64_raw(&self.continent_shape_noise, wx * 0.7, wz * 0.7);

        let mut v = base * 1.00 + breakup * 0.32 + ridge * 0.55 + shape * 0.18;

        // Normalize from [-1,1]-ish toward [0,1]
        v = v * 0.5 + 0.5;

        // Strong continent thresholding
        let mut cont = smootherstep(0.42, 0.68, v);

        // Coastline detail
        let coast = noise2_f64_raw(&self.detail, wx * 0.0022, wz * 0.0022);

        cont += coast * 0.045;

        // Spawn continent bias
        if self.p.force_land_at_origin {
            let d = ((sx * sx + sz * sz).sqrt() / 24000.0) as f32;

            let spawn = 1.0 - smootherstep(0.25, 1.0, d);

            cont += spawn * 0.22;
        }

        cont.clamp(0.0, 1.0)
    }

    fn latitude_factor(&self, p: &WorldPos) -> f32 {
        let (_, sz) = self.scaled_coords_f64(p);
        let lat_ext = self.p.lat_extent as f64 * self.p.world_scale as f64;
        ((sz / lat_ext).abs() as f32).min(1.0)
    }

    #[inline]
    fn flatten_profile(&self, rel: f32, cont: f32) -> f32 {
        let f = self.p.flatten.clamp(0.0, 1.0);
        if f <= 0.0001 {
            return rel;
        }

        let sign = rel.signum();
        let a = rel.abs();

        let c = self.p.flatten_curve.max(1.0);
        let k = 1.25 * f;
        let a2 = a / (1.0 + k * a.powf(c));

        let land = smoothstep(0.25, 0.65, cont);
        let mix = f * (0.30 + 0.70 * land);
        lerp(rel, sign * a2, mix)
    }

    #[inline]
    fn base_sample(&self, p: &WorldPos) -> BaseSample {
        let cont = self.continental_mask(p);
        let (wx2, wz2) = self.warped_coords_f64(p);

        let basin_raw = noise2_f64(&self.macro_elev, wx2 + 9000.0, wz2 - 4000.0, 0.12);
        let basin = gain(((basin_raw + 1.0) * 0.5).clamp(0.0, 1.0), 0.62);
        let ocean_floor = lerp(self.p.ocean_floor, self.p.ocean_floor * 1.45, basin);

        let mut rel = ocean_floor + (self.p.inland_plateau - ocean_floor) * cont;

        let shelf = smoothstep(0.18, 0.42, cont) * (1.0 - smoothstep(0.42, 0.58, cont));

        rel = lerp(rel, -0.16, shelf * 0.75);

        let macro_raw = noise2_f64(&self.macro_elev, wx2, wz2, 0.60);
        let macro_e = macro_raw * self.p.macro_amp;

        let hills_raw = noise2_f64(&self.hills, wx2, wz2, 2.05);
        let hills = hills_raw * self.p.hills_amp * self.p.hills_detail;

        let m_raw = noise2_f64_raw(&self.mountains, wx2, wz2);
        let mut rg = ridged(m_raw);
        if self.p.mountain_smooth > 0.0 {
            let s = self.p.mountain_smooth.clamp(0.0, 1.0);
            if s > 0.0001 {
                rg = lerp(rg, rg.sqrt(), s);
            }
        }

        let belts_raw = noise2_f64(&self.mountains, wx2 + 1234.0, wz2 - 5678.0, 0.18);
        let belt_n = (belts_raw + 1.0) * 0.5;
        let belt_mask = smootherstep(self.p.belt_lo, self.p.belt_hi, belt_n);

        let interior = smootherstep(self.p.interior_lo, self.p.interior_hi, cont);
        let chain_noise = noise2_f64_raw(&self.plates, wx2 * 0.11, wz2 * 0.035);

        let chains = ridged(chain_noise);

        let chain_mask = smootherstep(0.18, 0.72, chains);

        let mountain_mask = belt_mask * interior * chain_mask;

        let eps = 1.0;
        let (pgx, pgz) = grad2_f64(&self.plates, wx2 * 0.70, wz2 * 0.70, eps);
        let g = (pgx * pgx + pgz * pgz).sqrt();
        let plate_edges = smoothstep(0.02, 0.08, g);
        let uplift = (plate_edges * 0.65 + mountain_mask * 0.55).clamp(0.0, 1.0);

        rel += macro_e * (0.28 + 0.72 * cont);
        rel += hills * cont;

        let peak_detail = noise2_f64(&self.detail, wx2 + 400.0, wz2 - 700.0, 1.7);
        let crag = (ridged(peak_detail) * 0.55 + 0.45).clamp(0.0, 1.0);

        let m_amp = self.p.mountains_amp * self.p.belt_amp * (0.55 + 0.85 * uplift);
        rel += rg * mountain_mask * m_amp * crag;
        rel += plate_edges * mountain_mask * self.p.plate_mountain_amp * 0.52;

        let coast = (cont - 0.5).abs();
        let w = self.p.coast_soften_width.max(0.0001);
        let coast_t = ((w - coast) / w).clamp(0.0, 1.0);
        let coast_noise = noise2_f64_raw(&self.detail, wx2 * 0.0035, wz2 * 0.0035);

        let coast_variation = 1.0 + coast_noise * 0.18;

        rel *= 1.0 - self.p.coast_soften_strength * coast_t * coast_variation;

        let shelf = smootherstep(0.36, 0.62, cont) * smootherstep(-0.10, 0.06, rel);
        rel = lerp(rel, rel * 0.72, shelf * 0.55);

        let eps = 0.65;
        let n1 = noise2_f64_raw(&self.hills, wx2 + eps, wz2);
        let n2 = noise2_f64_raw(&self.hills, wx2 - eps, wz2);
        let n3 = noise2_f64_raw(&self.hills, wx2, wz2 + eps);
        let n4 = noise2_f64_raw(&self.hills, wx2, wz2 - eps);
        let slope_proxy = ((n1 - n2).abs() + (n3 - n4).abs()).clamp(0.0, 2.0) * 0.5;

        let e = self.p.erosion_strength.clamp(0.0, 2.0) * cont;
        if e > 0.0 {
            let it = (self.p.erosion_iters.max(1) as f32).min(64.0);
            let s = (slope_proxy * 0.95 + uplift * 0.35).clamp(0.0, 1.0);
            let k = 1.0 - (1.0 - s).powf(0.22 * it);
            rel -= k * e * (0.55 + 0.45 * cont);
        }

        rel = self.flatten_profile(rel, cont);

        BaseSample {
            cont,
            wx2,
            wz2,
            rel,
            _mountain_mask: mountain_mask,
            _uplift: uplift,
            slope_proxy,
        }
    }

    #[inline]
    fn apply_rivers_fast(&self, s: &BaseSample) -> f32 {
        if self.p.river_freq <= 0.0 || self.p.river_depth <= 0.0 || self.p.river_width <= 0.0 {
            return s.rel;
        }

        let land = smoothstep(-0.01, 0.03, s.rel) * smoothstep(0.10, 1.0, s.cont);
        if land <= 0.0001 {
            return s.rel;
        }

        let mut channel = 0.0f32;
        let mut f = 1.0f64;
        let mut a = 1.0f32;

        for _ in 0..2 {
            let n = noise2_f64(&self.rivers, s.wx2, s.wz2, f);
            let v = 1.0 - n.abs();
            let w = (self.p.river_width / f as f32).clamp(0.01, 0.48);
            let line = smoothstep(1.0 - w, 1.0, v);
            channel += line * a;
            f *= 2.15;
            a *= 0.62;
        }
        channel = (channel / 1.1).clamp(0.0, 1.0);

        let slope_n = (s.slope_proxy * 1.2).clamp(0.0, 1.0);
        let valley_like = (1.0 - slope_n).clamp(0.0, 1.0);

        let wet_n = noise2_f64(&self.moisture_noise, s.wx2, s.wz2, 0.55);
        let wet = ((wet_n + 1.0) * 0.5).clamp(0.0, 1.0);

        let lowland = 1.0 - smoothstep(0.18, 0.85, s.rel.max(0.0));
        let discharge = (0.22 + 0.78 * wet) * (0.30 + 0.70 * lowland);

        let coast_guard = smoothstep(-0.02, 0.08, s.rel) * smoothstep(0.16, 1.0, s.cont);
        let mask = (channel * valley_like).clamp(0.0, 1.0) * land * coast_guard;

        let carve = mask.powf(1.6) * self.p.river_depth * discharge;

        let mouth_soft = smoothstep(-0.02, 0.05, s.rel);
        let carve = carve * (0.60 + 0.40 * mouth_soft);

        s.rel - carve
    }

    pub fn height(&self, p: &WorldPos) -> f32 {
        let s = self.base_sample(p);
        let mut rel = self.apply_rivers_fast(&s);

        rel = micro_flatten(rel, self.p.micro_flatten);

        let h = rel * self.p.height_scale + self.p.sea_level;

        //h = self.apply_origin_land_override(p, h);

        h
    }

    pub fn moisture(&self, p: &WorldPos, h: f32) -> f32 {
        let h_rel = (h - self.p.sea_level) / self.p.height_scale;

        let (wx2, wz2) = self.warped_coords_f64(p);
        let n = noise2_f64_raw(&self.moisture_noise, wx2, wz2);
        let mut m = (n + 1.0) * 0.5;

        let cont = self.continental_mask(p);
        let lat = self.latitude_factor(p);

        let tropical = smoothstep(0.25, 0.0, lat); // wet near equator
        let subtropical_dry = smoothstep(0.18, 0.38, lat) * smoothstep(0.58, 0.38, lat); // dry belt
        let midlat_wet = smoothstep(0.38, 0.55, lat) * smoothstep(0.78, 0.62, lat); // westerlies
        let polar_dry = smoothstep(0.70, 0.90, lat); // polar dry

        let zonal = (tropical * 0.90 + midlat_wet * 0.70)
            * (1.0 - subtropical_dry * 0.55)
            * (1.0 - polar_dry * 0.75);
        let zonal = gain(zonal.clamp(0.0, 1.0), 0.55);

        let eps = 0.90;
        let (gx, gz) = grad2_f64(&self.hills, wx2 * 0.8, wz2 * 0.8, eps);

        const WIND_X: f32 = 1.0;
        const WIND_Z: f32 = 0.22;
        let dot = gx * WIND_X + gz * WIND_Z;

        let o = dot.clamp(-0.02, 0.02);
        let orographic = (o * 25.0 * 0.5 + 0.5).clamp(0.0, 1.0);

        m = m * 0.35 + zonal * 0.55 + orographic * 0.10;

        let height_dry = h_rel.clamp(0.0, 1.6);
        m *= 1.0 - height_dry * 0.30;

        let interior = smoothstep(0.54, 0.92, cont);
        m *= 1.0 - interior * 0.22;
        m = 1.0;
        m.clamp(0.0, 1.0)
    }

    pub fn color(&self, p: &WorldPos, h: f32, moisture: f32) -> [f32; 3] {
        let hs = self.p.height_scale;
        let h_rel = h - self.p.sea_level;
        let h_norm = (h_rel / hs).clamp(-1.2, 1.8);

        let lat = self.latitude_factor(p).clamp(0.0, 1.0);
        let s = self.base_sample(p);

        let mut temp = (1.0 - lat).powf(1.6);
        let t_noise = noise2_f64(&self.macro_elev, s.wx2, s.wz2, 0.020);
        temp = (temp + t_noise * 0.05).clamp(0.0, 1.0);

        let dry = (1.0 - moisture).clamp(0.0, 1.0);
        let wet = 1.0 - dry;

        let eps = 0.8f64;
        let (gx, gz) = grad2_f64(&self.hills, s.wx2 * 0.9, s.wz2 * 0.9, eps);
        let slope = (gx * gx + gz * gz).sqrt();
        let slope_n = smoothstep(0.015, 0.08, slope);

        // Water path
        if h_rel < 0.1 {
            let depth = (-h_rel / (0.75 * hs)).clamp(0.0, 1.0);

            let warm = HSV {
                h: 0.55,
                s: 0.62,
                v: 0.68,
            };
            let cold = HSV {
                h: 0.58,
                s: 0.26,
                v: 0.86,
            };
            let surface = lerp_hsv(warm, cold, lat);

            let shallow = HSV {
                h: 0.52,
                s: 0.38,
                v: 0.78,
            };
            let shelf = smoothstep(-0.18 * hs, -0.02 * hs, h_rel) * smoothstep(0.0, 0.55, slope_n);
            let surface2 = lerp_hsv(surface, shallow, shelf);

            let deep = HSV {
                h: surface2.h,
                s: surface2.s * 0.82,
                v: surface2.v * 0.33,
            };
            let mut col = lerp_hsv(surface2, deep, depth);

            let ice = smoothstep(0.78, 0.96, lat) * smoothstep(-6.0, 2.0, h_rel);
            let ice_col = HSV {
                h: 0.58,
                s: 0.05,
                v: 0.985,
            };
            col = lerp_hsv(col, ice_col, ice);

            return hsv_to_rgb(col);
        }

        // Land path
        let beach = smoothstep(-0.01, 0.06, h_norm) * (1.0 - smoothstep(0.08, 0.22, h_norm));
        let beach = beach * (1.0 - smoothstep(0.20, 0.55, slope_n));
        let cliff = smoothstep(0.55, 0.92, slope_n);
        let rockiness = (cliff * 0.75 + smoothstep(0.85, 1.35, h_norm) * 0.55).clamp(0.0, 1.0);

        let low_cold_dry = HSV {
            h: 0.24,
            s: 0.42,
            v: 0.55,
        };
        let low_hot_dry = HSV {
            h: 0.14,
            s: 0.78,
            v: 0.83,
        };

        let mid_cold_dry = HSV {
            h: 0.22,
            s: 0.40,
            v: 0.60,
        };
        let mid_hot_dry = HSV {
            h: 0.12,
            s: 0.66,
            v: 0.72,
        };

        let high_cold = HSV {
            h: 0.58,
            s: 0.15,
            v: 0.92,
        };
        let high_hot = HSV {
            h: 0.06,
            s: 0.18,
            v: 0.68,
        };

        let low_cold_wet = HSV {
            h: 0.34,
            s: 0.65,
            v: 0.48,
        };
        let low_hot_wet = HSV {
            h: 0.38,
            s: 0.82,
            v: 0.68,
        };
        let mid_cold_wet = HSV {
            h: 0.32,
            s: 0.58,
            v: 0.48,
        };
        let mid_hot_wet = HSV {
            h: 0.36,
            s: 0.78,
            v: 0.65,
        };

        fn climate(temp: f32, wet: f32, cd: HSV, cw: HSV, hd: HSV, hw: HSV) -> HSV {
            let cold = lerp_hsv(cd, cw, wet);
            let hot = lerp_hsv(hd, hw, wet);
            lerp_hsv(cold, hot, temp)
        }

        let c_low = climate(
            temp,
            wet,
            low_cold_dry,
            low_cold_wet,
            low_hot_dry,
            low_hot_wet,
        );
        let c_mid = climate(
            temp,
            wet,
            mid_cold_dry,
            mid_cold_wet,
            mid_hot_dry,
            mid_hot_wet,
        );
        let c_high = lerp_hsv(high_cold, high_hot, temp);

        let alt = h_norm.clamp(0.0, 1.35);
        let t_mid = ((alt - 0.25) / (0.60 - 0.25)).clamp(0.0, 1.0);
        let t_high = ((alt - 0.60) / (1.10 - 0.60)).clamp(0.0, 1.0);

        let mut col = lerp_hsv(c_low, c_mid, t_mid);
        col = lerp_hsv(col, c_high, t_high);

        let sand = HSV {
            h: 0.12,
            s: 0.32,
            v: 0.90,
        };
        col = lerp_hsv(col, sand, beach);

        let rock_n = noise2_f64(&self.rock, s.wx2, s.wz2, 1.25);
        let rock_tint = (rock_n * 0.5 + 0.5).clamp(0.0, 1.0);
        let rock_dry = HSV {
            h: lerp(0.08, 0.12, rock_tint),
            s: 0.16,
            v: 0.62,
        };
        let rock_wet = HSV {
            h: lerp(0.08, 0.12, rock_tint),
            s: 0.10,
            v: 0.52,
        };
        let rock_col = lerp_hsv(rock_dry, rock_wet, wet);
        col = lerp_hsv(col, rock_col, rockiness);

        let detail = noise2_f64(&self.detail, s.wx2, s.wz2, 3.2);
        let var = (detail * 0.06).clamp(-0.08, 0.08);
        col.v = (col.v * (1.0 + var)).clamp(0.0, 1.0);

        let shadow = (slope_n * 0.20).clamp(0.0, 0.20);
        col.v = (col.v * (1.0 - shadow)).clamp(0.0, 1.0);

        let snow_lat = smoothstep(0.80, 0.98, lat);
        let snow_alt = smoothstep(0.40, 1.18, alt);
        let snow_temp = smoothstep(0.55, 0.15, temp);
        let slope_mask = smoothstep(self.p.snow_slope_limit, 1.0, 1.0 - slope_n);

        let mut snow = (snow_lat * 0.55 + snow_alt * 0.95)
            * snow_temp
            * slope_mask
            * smoothstep(0.25, 0.65, moisture);
        snow = gain(snow, 0.65);
        snow = snow * (1.0 - smoothstep(0.45, 0.9, slope_n));

        let snow_col = HSV {
            h: 0.58,
            s: 0.04,
            v: 0.94,
        };
        col = lerp_hsv(col, snow_col, snow.clamp(0.0, 1.0));

        hsv_to_rgb(col)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tree placement
    // ─────────────────────────────────────────────────────────────────────────

    /// Returns local (x, z) positions within the chunk where trees should be placed.
    ///
    /// Uses a jittered grid so trees aren't perfectly regular, but sampling is
    /// O(chunk_cells) with no spatial data structures needed.
    ///
    /// `grid_spacing` controls how densely the chunk is sampled (e.g. 6 means
    /// one candidate per 6×6 block). Lower = denser forest, higher = sparser.
    /// A value of 4–8 works well for most use-cases.
    pub fn tree_positions(&self, chunk_x: i32, chunk_z: i32, grid_spacing: u32) -> Vec<[f32; 2]> {
        use crate::helpers::positions::{ChunkCoord, LocalPos};

        let gs = grid_spacing.max(1);
        let cs = chunk_size() as u32;
        let cols = cs / gs;
        let mut out = Vec::new();

        for gz in 0..cols {
            for gx in 0..cols {
                // Stable, unique seed per cell — incorporates chunk position so
                // identical cells in different chunks don't produce the same jitter.
                let cell_seed = (chunk_x as u32)
                    .wrapping_mul(0x9e3779b9)
                    .wrapping_add((chunk_z as u32).wrapping_mul(0x517cc1b7))
                    .wrapping_add(gz.wrapping_mul(0x45d9f3b))
                    .wrapping_add(gx)
                    .wrapping_add(self.p.seed.wrapping_mul(0xdeadbeef));

                // Jitter within cell
                let jx = hash01(cell_seed) * gs as f32;
                let jz = hash01(cell_seed.wrapping_add(1)) * gs as f32;

                let lx = gx as f32 * gs as f32 + jx;
                let lz = gz as f32 * gs as f32 + jz;

                // Clamp strictly inside chunk
                if lx < 0.0 || lx >= cs as f32 || lz < 0.0 || lz >= cs as f32 {
                    continue;
                }

                let wp = WorldPos {
                    chunk: ChunkCoord {
                        x: chunk_x,
                        z: chunk_z,
                    },
                    local: LocalPos {
                        x: lx,
                        z: lz,
                        y: 0.0,
                    },
                };

                let h = self.height(&wp);
                let h_norm = (h - self.p.sea_level) / self.p.height_scale;

                // Below water or barely above — no trees
                if h_norm < 0.015 {
                    continue;
                }

                // Above treeline — bare rock/snow
                if h_norm > 0.80 {
                    continue;
                }

                let m = self.moisture(&wp, h);

                // Too dry for any tree growth
                if m < 0.22 {
                    continue;
                }

                let lat = self.latitude_factor(&wp);

                // Approximate temperature (no re-running full color pipeline)
                let t_noise = noise2_f64(
                    &self.macro_elev,
                    (wp.chunk.x as f64 * cs as f64 + lx as f64) * self.p.world_scale as f64,
                    (wp.chunk.z as f64 * cs as f64 + lz as f64) * self.p.world_scale as f64,
                    0.020,
                );
                let temp = ((1.0 - lat).powf(1.6) + t_noise * 0.05).clamp(0.0, 1.0);

                // Too cold (polar / high alpine)
                if temp < 0.12 {
                    continue;
                }

                // Slope check — trees don't grow on steep cliffs
                let (gx_s, gz_s) = grad2_f64(
                    &self.hills,
                    (wp.chunk.x as f64 * cs as f64 + lx as f64) * self.p.world_scale as f64,
                    (wp.chunk.z as f64 * cs as f64 + lz as f64) * self.p.world_scale as f64,
                    0.8,
                );
                let slope = (gx_s * gx_s + gz_s * gz_s).sqrt();
                if slope > 0.55 {
                    continue;
                }

                // Density function: moist + warm + lowland = dense forest;
                // marginal conditions = sparse. Roll against a hash to thin out.
                let alt_penalty = smoothstep(0.55, 0.80, h_norm); // fewer trees near peaks
                let density = (m - 0.22) / 0.78       // moisture drive (0..1)
                    * (0.30 + 0.70 * temp)             // colder biomes are sparser
                    * (1.0 - alt_penalty * 0.85)       // altitude thins canopy
                    * (1.0 - slope / 0.55 * 0.40); // steeper = sparser

                let roll = hash01(cell_seed.wrapping_add(2));
                if roll < density {
                    out.push([lx, lz]);
                }
            }
        }

        out
    }

    #[inline]
    fn height_preview_rgb(&self, h: f32) -> [f32; 3] {
        let h_rel = h - self.p.sea_level;
        let h_norm = (h_rel / self.p.height_scale).clamp(-1.2, 1.8);

        let t = ((h_norm + 1.2) / 3.0).clamp(0.0, 1.0);

        let low = HSV {
            h: 0.62,
            s: 0.95,
            v: 0.28,
        };
        let high = HSV {
            h: 0.08,
            s: 0.05,
            v: 0.98,
        };

        hsv_to_rgb(lerp_hsv(low, high, t))
    }

    pub fn make_texture(
        &self,
        texture_size: Extent3d,
        chunk_coord: ChunkCoord,
        size: u32,
    ) -> TerrainTextures {
        let size = size.max(1);
        let tex_w = texture_size.width as usize;
        let tex_h = texture_size.height as usize;

        let cs = chunk_size() as f64;
        let start_wx = chunk_coord.x as f64 * cs;
        let start_wz = chunk_coord.z as f64 * cs;
        let world_span = cs * size as f64;

        // ─────────────────────────────────────────────
        // PASS 1: gather heights + find min/max
        // ─────────────────────────────────────────────
        let mut heights = vec![0.0f32; tex_w * tex_h];
        let mut min_h = f32::MAX;
        let mut max_h = f32::MIN;

        for py in 0..tex_h {
            let vz = (py as f64 + 0.5) / tex_h as f64;
            let wz = start_wz + vz * world_span;

            for px in 0..tex_w {
                let vx = (px as f64 + 0.5) / tex_w as f64;
                let wx = start_wx + vx * world_span;

                let wp = world_pos_from_world_xz(wx, wz);
                let h = self.height(&wp);

                heights[py * tex_w + px] = h;

                min_h = min_h.min(h);
                max_h = max_h.max(h);
            }
        }

        let inv_range = if max_h > min_h {
            1.0 / (max_h - min_h)
        } else {
            1.0
        };

        // ─────────────────────────────────────────────
        // PASS 2: encode grayscale
        // ─────────────────────────────────────────────
        let mut height_data = Vec::with_capacity(tex_w * tex_h * 4);
        let mut color_data = Vec::with_capacity(tex_w * tex_h * 4);

        for i in 0..heights.len() {
            let h = heights[i];

            let t = ((h - min_h) * inv_range).clamp(0.0, 1.0);

            // pure grayscale debug ramp
            let v = (t * 255.0 + 0.5) as u8;

            height_data.push(v);
            height_data.push(v);
            height_data.push(v);
            height_data.push(255);

            // still your biome color
            let wx = 0.0; // not used here anymore
            let wz = 0.0;
            let color_rgb = [t, t, t]; // optional fallback if you want speed debug
            color_data.push((color_rgb[0] * 255.0) as u8);
            color_data.push((color_rgb[1] * 255.0) as u8);
            color_data.push((color_rgb[2] * 255.0) as u8);
            color_data.push(255);
        }

        TerrainTextures {
            height: TextureBuffer {
                width: tex_w as u32,
                height: tex_h as u32,
                data: height_data,
            },
            color: TextureBuffer {
                width: tex_w as u32,
                height: tex_h as u32,
                data: color_data,
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct TextureBuffer {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>, // RGBA8
}

#[derive(Clone, Debug)]
pub struct TerrainTextures {
    pub height: TextureBuffer,
    pub color: TextureBuffer,
}
#[inline]
fn f32_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

#[inline]
fn push_rgba(out: &mut Vec<u8>, rgb: [f32; 3]) {
    out.push(f32_to_u8(rgb[0]));
    out.push(f32_to_u8(rgb[1]));
    out.push(f32_to_u8(rgb[2]));
    out.push(255);
}

#[inline]
fn world_pos_from_world_xz(wx: f64, wz: f64) -> WorldPos {
    use crate::helpers::positions::LocalPos;

    let cs = chunk_size() as f64;

    let cx = (wx / cs).floor() as i32;
    let cz = (wz / cs).floor() as i32;

    let lx = (wx - cx as f64 * cs) as f32;
    let lz = (wz - cz as f64 * cs) as f32;

    WorldPos {
        chunk: ChunkCoord { x: cx, z: cz },
        local: LocalPos {
            x: lx,
            y: 0.0,
            z: lz,
        },
    }
}
