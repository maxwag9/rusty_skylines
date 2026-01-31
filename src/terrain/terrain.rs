use crate::hsv::{HSV, hsv_to_rgb, lerp_hsv};
use crate::positions::{ChunkSize, WorldPos};
use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};
use std::f32::consts::PI;

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
fn world_xz_f64(p: &WorldPos, chunk_size: ChunkSize) -> (f64, f64) {
    let wx = p.chunk.x as f64 * chunk_size as f64 + p.local.x as f64;
    let wz = p.chunk.z as f64 * chunk_size as f64 + p.local.z as f64;
    (wx, wz)
}

/// Sample noise at derived f64 coordinates with a scale factor
#[inline]
fn noise2_f64(n: &FastNoiseLite, x: f64, z: f64, scale: f64) -> f32 {
    n.get_noise_2d((x * scale) as f32, (z * scale) as f32)
}

/// Sample noise at derived f64 coordinates (scale = 1.0)
#[inline]
fn noise2_f64_raw(n: &FastNoiseLite, x: f64, z: f64) -> f32 {
    n.get_noise_2d(x as f32, z as f32)
}

/// Gradient via central difference in f64 space
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
// TerrainParams (unchanged)
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
            world_scale: 0.05,

            height_scale: 1000.0,
            sea_level: 0.0,

            lat_extent: 140_000.0,
            continent_radius: 70_000.0,
            ring_radius: 8_000.0,
            coast_noise_scale: 0.0035,
            coast_noise_amp: 0.45,
            island_threshold0: 0.74,
            island_threshold1: 0.92,
            island_amp: 0.55,

            force_land_at_origin: false,
            origin_island_radius: 12_000.0,
            origin_min_height: 50.0,
            pull_one_continent_to_origin: true,
            origin_pull_strength: 0.85,

            warp_large_scale: 0.00042,
            warp_small_scale: 0.0040,
            warp_large_amp: 160.0,
            warp_small_amp: 110.0,
            warp_mix_large: 0.65,
            warp_mix_small: 0.35,

            macro_freq: 0.0006,
            hills_freq: 0.0045,
            mountains_freq: 0.0070,
            _belts_freq: 0.00162,
            moisture_freq: 0.0012,

            macro_octaves: 5,
            macro_persistence: 0.50,
            hills_octaves: 6,
            hills_persistence: 0.52,
            mountains_octaves: 0,
            mountains_persistence: 0.0,
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
            hills_amp: 0.95,
            mountains_amp: 0.1,
            belt_amp: 0.5,

            coast_soften_width: 0.22,
            coast_soften_strength: 0.1,
            interior_lo: 0.38,
            interior_hi: 0.92,
            belt_lo: 0.50,
            belt_hi: 0.78,

            flatten: 0.20,
            flatten_curve: 1.8,
            mountain_smooth: 0.55,
            hills_detail: 0.18,
            micro_flatten: 0.20,

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

// ─────────────────────────────────────────────────────────────────────────────
// BaseSample (unchanged)
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// TerrainGenerator
// ─────────────────────────────────────────────────────────────────────────────

pub struct TerrainGenerator {
    p: TerrainParams,

    macro_elev: FastNoiseLite,
    hills: FastNoiseLite,
    mountains: FastNoiseLite,
    plates: FastNoiseLite,
    rivers: FastNoiseLite,

    continent_noise: FastNoiseLite,
    moisture_noise: FastNoiseLite,
    warp_large: FastNoiseLite,
    warp_small: FastNoiseLite,

    detail: FastNoiseLite,
    rock: FastNoiseLite,

    continent_centers: [(f64, f64); 6],
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

        let ws = p.world_scale as f64;
        let mut centers = [(0.0f64, 0.0f64); 6];
        for i in 0..6 {
            let base_angle = (i as f64) / 6.0 * (TAU as f64);
            let jitter = (hash01(seed.wrapping_add(i as u32)) as f64 - 0.5) * 0.6;
            let angle = base_angle + jitter;

            let radial_jitter = (hash01(seed.wrapping_add(100 + i as u32)) as f64 - 0.5) * 0.25;
            let r = p.ring_radius as f64 * ws * (1.0 + radial_jitter);

            let lat_band = if i < 2 {
                let sign = if i == 0 { 1.0 } else { -1.0 };
                let band_jitter = (hash01(seed.wrapping_add(200 + i as u32)) as f64 - 0.5) * 0.1;
                sign * (0.85 + band_jitter)
            } else {
                let raw = hash01(seed.wrapping_add(200 + i as u32)) as f64;
                (raw * 0.9) - 0.45
            };
            let lat_ext = p.lat_extent as f64 * ws;
            centers[i] = (angle.cos() * r, lat_band * lat_ext);
        }

        if p.pull_one_continent_to_origin {
            let idx = 2;
            let (cx, cz) = centers[idx];
            let pull = p.origin_pull_strength as f64;
            centers[idx] = (cx * (1.0 - pull), cz * (1.0 - pull));
        }

        Self {
            p,
            macro_elev,
            hills,
            mountains,
            plates,
            rivers,
            continent_noise,
            moisture_noise,
            warp_large,
            warp_small,
            detail,
            rock,
            continent_centers: centers,
        }
    }

    /// Returns scaled world coordinates in f64 precision
    #[inline]
    fn scaled_coords_f64(&self, p: &WorldPos, chunk_size: ChunkSize) -> (f64, f64) {
        let (wx, wz) = world_xz_f64(p, chunk_size);
        let s = self.p.world_scale as f64;
        (wx * s, wz * s)
    }

    /// Returns warped coordinates in f64 precision
    fn warped_coords_f64(&self, p: &WorldPos, chunk_size: ChunkSize) -> (f64, f64) {
        let (sx, sz) = self.scaled_coords_f64(p, chunk_size);

        // Warp noise has baked-in frequency, so we pass scaled coords directly
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

    fn continental_mask(&self, p: &WorldPos, chunk_size: ChunkSize) -> f32 {
        let (sx, sz) = self.scaled_coords_f64(p, chunk_size);

        let mut best = 0.0f32;
        for &(cx, cz) in &self.continent_centers {
            let cr = self.p.continent_radius as f64 * self.p.world_scale as f64;
            let dx = (sx - cx) / cr;
            let dz = (sz - cz) / (cr * 0.6);
            let dist = (dx * dx + dz * dz).sqrt() as f32;
            let v = (1.0 - dist).clamp(0.0, 1.0);
            let shaped = v * v * (3.0 - 2.0 * v);
            if shaped > best {
                best = shaped;
            }
        }

        let coast_scale = self.p.coast_noise_scale as f64;
        let nx = sx * coast_scale;
        let nz = sz * coast_scale;
        let noise = noise2_f64_raw(&self.continent_noise, nx, nz) * self.p.coast_noise_amp;

        let mut c = (best + noise).clamp(0.0, 1.0);

        let island_raw = noise2_f64(&self.continent_noise, nx, nz, 3.0);
        let island_v = (island_raw + 1.0) * 0.5;
        let island = smoothstep(self.p.island_threshold0, self.p.island_threshold1, island_v);
        c += island * self.p.island_amp;
        c
    }

    #[inline]
    fn apply_origin_land_override(&self, p: &WorldPos, h: f32, chunk_size: ChunkSize) -> f32 {
        if !self.p.force_land_at_origin {
            return h;
        }

        let (wx, wz) = world_xz_f64(p, chunk_size);
        let r = self.p.origin_island_radius.max(1.0) as f64;
        let d2 = wx * wx + wz * wz;

        if d2 >= r * r {
            return h;
        }

        let d = d2.sqrt();
        let t = ((1.0 - d / r) as f32).clamp(0.0, 1.0);
        let s = t * t * (3.0 - 2.0 * t);

        let min_land = self.p.sea_level + self.p.origin_min_height;
        lerp(h, min_land, s)
    }

    fn latitude_factor(&self, p: &WorldPos, chunk_size: ChunkSize) -> f32 {
        let (_, sz) = self.scaled_coords_f64(p, chunk_size);
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
    fn base_sample(&self, p: &WorldPos, chunk_size: ChunkSize) -> BaseSample {
        let cont = self.continental_mask(p, chunk_size);
        let (wx2, wz2) = self.warped_coords_f64(p, chunk_size);

        // Basin/macro sampling in f64
        let basin_raw = noise2_f64(&self.macro_elev, wx2 + 9000.0, wz2 - 4000.0, 0.12);
        let basin = gain(((basin_raw + 1.0) * 0.5).clamp(0.0, 1.0), 0.62);
        let ocean_floor = lerp(self.p.ocean_floor, self.p.ocean_floor * 1.45, basin);

        let mut rel = ocean_floor + (self.p.inland_plateau - ocean_floor) * cont;

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
        let mountain_mask = belt_mask * interior;

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
        rel *= 1.0 - self.p.coast_soften_strength * coast_t;

        let shelf = smootherstep(0.36, 0.62, cont) * smootherstep(-0.10, 0.06, rel);
        rel = lerp(rel, rel * 0.72, shelf * 0.55);

        // Slope proxy via central difference in f64
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

    pub fn height(&self, p: &WorldPos, chunk_size: ChunkSize) -> f32 {
        let s = self.base_sample(p, chunk_size);
        let mut rel = self.apply_rivers_fast(&s);

        rel = micro_flatten(rel, self.p.micro_flatten);

        let mut h = rel * self.p.height_scale + self.p.sea_level;

        h = self.apply_origin_land_override(p, h, chunk_size);

        h
    }

    pub fn moisture(&self, p: &WorldPos, h: f32, chunk_size: ChunkSize) -> f32 {
        let h_rel = (h - self.p.sea_level) / self.p.height_scale;

        let (wx2, wz2) = self.warped_coords_f64(p, chunk_size);
        let n = noise2_f64_raw(&self.moisture_noise, wx2, wz2);
        let mut m = (n + 1.0) * 0.5;

        let cont = self.continental_mask(p, chunk_size);
        let lat = self.latitude_factor(p, chunk_size);

        let mut zonal = 1.0 - ((lat - 0.18) * 1.55).abs();
        zonal = zonal.max(0.0).min(1.0);
        zonal = gain(zonal, 0.55);

        let eps = 0.90;
        let (gx, gz) = grad2_f64(&self.hills, wx2 * 0.8, wz2 * 0.8, eps);

        const WIND_X: f32 = 1.0;
        const WIND_Z: f32 = 0.22;
        let dot = gx * WIND_X + gz * WIND_Z;

        let o = dot.clamp(-0.02, 0.02);
        let orographic = (o * 25.0 * 0.5 + 0.5).clamp(0.0, 1.0);

        m = m * 0.40 + zonal * 0.52 + orographic * 0.08;

        let height_dry = h_rel.clamp(0.0, 1.6);
        m *= 1.0 - height_dry * 0.55;

        let interior = smoothstep(0.54, 0.92, cont);
        m *= 1.0 - interior * 0.48;

        m.clamp(0.0, 1.0)
    }

    pub fn color(&self, p: &WorldPos, h: f32, moisture: f32, chunk_size: ChunkSize) -> [f32; 3] {
        let hs = self.p.height_scale;
        let h_rel = h - self.p.sea_level;
        let h_norm = (h_rel / hs).clamp(-1.2, 1.8);

        let lat = self.latitude_factor(p, chunk_size).clamp(0.0, 1.0);
        let s = self.base_sample(p, chunk_size);

        let mut temp = (1.0 - lat).powf(1.6);
        let t_noise = noise2_f64(&self.macro_elev, s.wx2, s.wz2, 0.020);
        temp = (temp + t_noise * 0.05).clamp(0.0, 1.0);

        let dry = (1.0 - moisture).clamp(0.0, 1.0);
        let wet = 1.0 - dry;

        let eps = 0.8f64;
        let (gx, gz) = grad2_f64(&self.hills, s.wx2 * 0.9, s.wz2 * 0.9, eps);
        let slope = (gx * gx + gz * gz).sqrt();
        let slope_n = smoothstep(0.08, 0.015, slope);

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

        let cliff = smoothstep(0.35, 0.95, slope_n) * smoothstep(0.02, 0.60, h_norm);
        let rockiness = (cliff * 0.75 + smoothstep(0.85, 1.35, h_norm) * 0.55).clamp(0.0, 1.0);

        let low_cold_dry = HSV {
            h: 0.24,
            s: 0.42,
            v: 0.55,
        };
        let low_cold_wet = HSV {
            h: 0.33,
            s: 0.78,
            v: 0.44,
        };
        let low_hot_dry = HSV {
            h: 0.14,
            s: 0.78,
            v: 0.83,
        };
        let low_hot_wet = HSV {
            h: 0.33,
            s: 0.98,
            v: 0.56,
        };

        let mid_cold_dry = HSV {
            h: 0.22,
            s: 0.40,
            v: 0.60,
        };
        let mid_cold_wet = HSV {
            h: 0.30,
            s: 0.62,
            v: 0.50,
        };
        let mid_hot_dry = HSV {
            h: 0.12,
            s: 0.66,
            v: 0.72,
        };
        let mid_hot_wet = HSV {
            h: 0.30,
            s: 0.88,
            v: 0.56,
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
}
