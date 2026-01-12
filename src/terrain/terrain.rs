use crate::hsv::{HSV, hsv_to_rgb, lerp_hsv};
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
    // same effect, slightly cheaper math
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
    // keep the cubic but avoid extra temporaries
    let r = 1.0 - n.abs();
    if r <= 0.0 { 0.0 } else { r * r * r }
}

#[inline]
fn grad2(noise: &FastNoiseLite, x: f32, z: f32, eps: f32) -> (f32, f32) {
    // unchanged: central difference. Inline for speed.
    let a = noise.get_noise_2d(x + eps, z);
    let b = noise.get_noise_2d(x - eps, z);
    let c = noise.get_noise_2d(x, z + eps);
    let d = noise.get_noise_2d(x, z - eps);
    ((a - b) / (2.0 * eps), (c - d) / (2.0 * eps))
}

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
    pub origin_island_strength: f32,
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
    pub plate_sharpness: f32,
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
            seed: 201035458,
            world_scale: 0.1,

            height_scale: 2000.0,
            sea_level: 0.0,

            lat_extent: 140_000.0,
            continent_radius: 30_000.0,
            ring_radius: 85_000.0,
            coast_noise_scale: 0.00035,
            coast_noise_amp: 0.45,
            island_threshold0: 0.74,
            island_threshold1: 0.92,
            island_amp: 0.65,

            force_land_at_origin: true,
            origin_island_radius: 12_000.0,
            origin_island_strength: 0.75,
            origin_min_height: 50.0,
            pull_one_continent_to_origin: true,
            origin_pull_strength: 0.50,

            warp_large_scale: 0.00042,
            warp_small_scale: 0.0040,
            warp_large_amp: 160.0,
            warp_small_amp: 110.0,
            warp_mix_large: 0.65,
            warp_mix_small: 0.35,

            macro_freq: 0.0006,
            hills_freq: 0.0045,
            mountains_freq: 0.0090,
            _belts_freq: 0.00162,
            moisture_freq: 0.0012,

            macro_octaves: 5,
            macro_persistence: 0.50,
            hills_octaves: 6,
            hills_persistence: 0.52,
            mountains_octaves: 7,
            mountains_persistence: 0.50,
            continent_octaves: 4,
            continent_persistence: 0.75,
            moisture_octaves: 5,
            moisture_persistence: 0.60,
            warp_large_octaves: 4,
            warp_large_persistence: 0.68,
            warp_small_octaves: 3,
            warp_small_persistence: 0.52,

            ocean_floor: -1.10,
            inland_plateau: 0.28,
            macro_amp: 0.55,
            hills_amp: 0.95,
            mountains_amp: 0.6,
            belt_amp: 0.5,

            coast_soften_width: 0.22,
            coast_soften_strength: 0.22,
            interior_lo: 0.38,
            interior_hi: 0.92,
            belt_lo: 0.50,
            belt_hi: 0.88,

            flatten: 0.45,
            flatten_curve: 1.8,
            mountain_smooth: 0.45,
            hills_detail: 0.28,
            micro_flatten: 0.5,

            plate_freq: 0.00022,
            plate_sharpness: 4.2,
            plate_mountain_amp: 1.8,

            erosion_strength: 0.55,
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
    wx2: f32,
    wz2: f32,
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
    moisture_noise: FastNoiseLite,
    warp_large: FastNoiseLite,
    warp_small: FastNoiseLite,

    detail: FastNoiseLite,
    rock: FastNoiseLite,

    continent_centers: [(f32, f32); 6],
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
        let ws = p.world_scale;

        let macro_elev = make_fbm(
            seed,
            p.macro_freq * ws,
            p.macro_octaves,
            p.macro_persistence,
        );
        let hills = make_fbm(
            seed.wrapping_add(10),
            p.hills_freq * ws,
            p.hills_octaves,
            p.hills_persistence,
        );
        let mountains = make_fbm(
            seed.wrapping_add(1),
            p.mountains_freq * ws,
            p.mountains_octaves,
            p.mountains_persistence,
        );
        let plates = make_fbm(seed.wrapping_add(42), p.plate_freq * ws, 3, 0.5);
        let rivers = make_fbm(seed.wrapping_add(77), p.river_freq.max(0.0) * ws, 4, 0.55);

        let continent_noise = make_fbm(
            seed.wrapping_add(2),
            0.0003 * ws,
            p.continent_octaves,
            p.continent_persistence,
        );
        let moisture_noise = make_fbm(
            seed,
            p.moisture_freq * ws,
            p.moisture_octaves,
            p.moisture_persistence,
        );

        let warp_large = make_fbm(
            seed.wrapping_add(4),
            p.warp_large_scale * ws,
            p.warp_large_octaves,
            p.warp_large_persistence,
        );
        let warp_small = make_fbm(
            seed.wrapping_add(5),
            p.warp_small_scale * ws,
            p.warp_small_octaves,
            p.warp_small_persistence,
        );

        let detail = make_fbm(seed.wrapping_add(9001), 0.020 * ws, 4, 0.55);
        let rock = make_fbm(seed.wrapping_add(9002), 0.012 * ws, 3, 0.55);

        let mut centers = [(0.0f32, 0.0f32); 6];
        for i in 0..6 {
            let base_angle = (i as f32) / 6.0 * TAU;
            let jitter = (hash01(seed.wrapping_add(i as u32)) - 0.5) * 0.6;
            let angle = base_angle + jitter;

            let radial_jitter = (hash01(seed.wrapping_add(100 + i as u32)) - 0.5) * 0.25;
            let r = p.ring_radius * (1.0 + radial_jitter);

            let lat_band = if i < 2 {
                let sign = if i == 0 { 1.0 } else { -1.0 };
                let band_jitter = (hash01(seed.wrapping_add(200 + i as u32)) - 0.5) * 0.1;
                sign * (0.85 + band_jitter)
            } else {
                let raw = hash01(seed.wrapping_add(200 + i as u32));
                (raw * 0.9) - 0.45
            };

            centers[i] = (angle.cos() * r, lat_band * p.lat_extent);
        }

        if p.pull_one_continent_to_origin {
            let idx = 2;
            let (cx, cz) = centers[idx];
            centers[idx] = (
                cx * (1.0 - p.origin_pull_strength),
                cz * (1.0 - p.origin_pull_strength),
            );
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

    #[inline]
    fn scaled_coords(&self, wx: f32, wz: f32) -> (f32, f32) {
        let s = self.p.world_scale;
        (wx * s, wz * s)
    }

    fn warped_coords(&self, wx: f32, wz: f32) -> (f32, f32) {
        let (wx, wz) = self.scaled_coords(wx, wz);
        let x = wx;
        let z = wz;

        let w1x = self.warp_large.get_noise_2d(x, z);
        let w1z = self.warp_large.get_noise_2d(x + 1234.0, z - 5678.0);

        let w2x = self.warp_small.get_noise_2d(x * 2.0, z * 2.0);
        let w2z = self
            .warp_small
            .get_noise_2d(x * 2.0 - 500.0, z * 2.0 + 500.0);

        let dx = w1x * self.p.warp_large_amp * self.p.warp_mix_large
            + w2x * self.p.warp_small_amp * self.p.warp_mix_small;
        let dz = w1z * self.p.warp_large_amp * self.p.warp_mix_large
            + w2z * self.p.warp_small_amp * self.p.warp_mix_small;

        (x + dx, z + dz)
    }

    fn continental_mask(&self, orig_wx: f32, orig_wz: f32) -> f32 {
        let (wx, wz) = self.scaled_coords(orig_wx, orig_wz);

        let mut best = 0.0f32;
        for &(cx, cz) in &self.continent_centers {
            let dx = (wx - cx) / self.p.continent_radius;
            let dz = (wz - cz) / (self.p.continent_radius * 0.6);
            let dist = (dx * dx + dz * dz).sqrt();
            let v = (1.0 - dist).clamp(0.0, 1.0);
            let shaped = v * v * (3.0 - 2.0 * v);
            if shaped > best {
                best = shaped;
            }
        }

        let nx = wx * self.p.coast_noise_scale;
        let nz = wz * self.p.coast_noise_scale;
        let noise = self.continent_noise.get_noise_2d(nx, nz) * self.p.coast_noise_amp;

        let mut c = (best + noise).clamp(0.0, 1.0);

        let island_raw = self.continent_noise.get_noise_2d(nx * 3.0, nz * 3.0);
        let island_v = (island_raw + 1.0) * 0.5;
        let island = smoothstep(self.p.island_threshold0, self.p.island_threshold1, island_v);
        c += island * self.p.island_amp;
        c
    }
    #[inline]
    fn apply_origin_land_override(&self, wx: f32, wz: f32, h: f32) -> f32 {
        if !self.p.force_land_at_origin {
            return h;
        }

        let r = self.p.origin_island_radius.max(1.0);
        let d2 = wx * wx + wz * wz;

        if d2 >= r * r {
            return h;
        }

        let d = d2.sqrt();
        let t = (1.0 - d / r).clamp(0.0, 1.0);
        let s = t * t * (3.0 - 2.0 * t);

        // guaranteed minimum height ABOVE sea level
        let min_land = self.p.sea_level + self.p.origin_min_height;

        // blend, do not add
        lerp(h, min_land, s)
    }

    fn latitude_factor(&self, wz: f32) -> f32 {
        let (_, wz) = self.scaled_coords(0.0, wz);
        let t = (wz / self.p.lat_extent).abs();
        t.min(1.0)
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
    fn base_sample(&self, wx: f32, wz: f32) -> BaseSample {
        // Compute continental mask and warped coords once
        let cont = self.continental_mask(wx, wz);
        let (wx2, wz2) = self.warped_coords(wx, wz);

        // Basin/macro: two scaled samples from the same macro_elev noise, reuse minimal ops
        // Keep the basin offset but sample macro_elev only twice (basin + main)
        let basin_raw = self
            .macro_elev
            .get_noise_2d(wx2 * 0.12 + 9000.0, wz2 * 0.12 - 4000.0);
        let basin = gain(((basin_raw + 1.0) * 0.5).clamp(0.0, 1.0), 0.62);
        let ocean_floor = lerp(self.p.ocean_floor, self.p.ocean_floor * 1.45, basin);

        // Base relative height starts between ocean_floor and inland_plateau
        let mut rel = ocean_floor + (self.p.inland_plateau - ocean_floor) * cont;

        // macro elevation and hills (one sample each)
        let macro_raw = self.macro_elev.get_noise_2d(wx2 * 0.60, wz2 * 0.60);
        let macro_e = macro_raw * self.p.macro_amp;

        let hills_raw = self.hills.get_noise_2d(wx2 * 2.05, wz2 * 2.05);
        let hills = hills_raw * self.p.hills_amp * self.p.hills_detail;

        // mountains: one sample then ridged once
        let m_raw = self.mountains.get_noise_2d(wx2, wz2);
        let mut rg = ridged(m_raw);
        if self.p.mountain_smooth > 0.0 {
            let s = self.p.mountain_smooth.clamp(0.0, 1.0);
            // lerp(rg, sqrt(rg), s) -> keep but avoid repeated sqrt if s==0
            if s > 0.0001 {
                rg = lerp(rg, rg.sqrt(), s);
            }
        }

        // belts/mountain mask (one sample)
        let belts_raw = self
            .mountains
            .get_noise_2d(wx2 * 0.18 + 1234.0, wz2 * 0.18 - 5678.0);
        let belt_n = (belts_raw + 1.0) * 0.5;
        let belt_mask = smootherstep(self.p.belt_lo, self.p.belt_hi, belt_n);

        let interior = smootherstep(self.p.interior_lo, self.p.interior_hi, cont);
        let mountain_mask = belt_mask * interior;

        // plates and uplift (single plate sample)
        let plate_raw = self.plates.get_noise_2d(wx2 * 0.70, wz2 * 0.70);
        // plate_edges used to be powf a lot; keep but clamp exponent to avoid huge cost
        let plate_edges = (1.0 - plate_raw.abs()).powf(self.p.plate_sharpness.max(0.0001));
        let uplift = (plate_edges * 0.65 + mountain_mask * 0.55).clamp(0.0, 1.0);

        // compose rel with macro & hills
        rel += macro_e * (0.28 + 0.72 * cont);
        rel += hills * cont;

        // small peak detail from one detail sample
        let peak_detail = self
            .detail
            .get_noise_2d(wx2 * 1.7 + 400.0, wz2 * 1.7 - 700.0);
        let crag = (ridged(peak_detail) * 0.55 + 0.45).clamp(0.0, 1.0);

        let m_amp = self.p.mountains_amp * self.p.belt_amp * (0.55 + 0.85 * uplift);
        rel += rg * mountain_mask * m_amp * crag;
        rel += plate_edges * mountain_mask * self.p.plate_mountain_amp * 0.52;

        // coastline soften
        let coast = (cont - 0.5).abs();
        let w = self.p.coast_soften_width.max(0.0001);
        let coast_t = ((w - coast) / w).clamp(0.0, 1.0);
        rel *= 1.0 - self.p.coast_soften_strength * coast_t;

        // shelf blending
        let shelf = smootherstep(0.36, 0.62, cont) * smootherstep(-0.10, 0.06, rel);
        rel = lerp(rel, rel * 0.72, shelf * 0.55);

        // fast slope proxy using two hill offset samples (we already have hills_raw; reuse small offsets)
        let eps = 0.65;
        let n1 = self.hills.get_noise_2d(wx2 + eps, wz2);
        let n2 = self.hills.get_noise_2d(wx2 - eps, wz2);
        let n3 = self.hills.get_noise_2d(wx2, wz2 + eps);
        let n4 = self.hills.get_noise_2d(wx2, wz2 - eps);
        let slope_proxy = ((n1 - n2).abs() + (n3 - n4).abs()).clamp(0.0, 2.0) * 0.5;

        // simplified erosion: compute k cheaply and apply
        let e = self.p.erosion_strength.clamp(0.0, 2.0) * cont;
        if e > 0.0 {
            let it = (self.p.erosion_iters.max(1) as f32).min(64.0);
            let s = (slope_proxy * 0.95 + uplift * 0.35).clamp(0.0, 1.0);
            // approximate powf chain with a single cheaper pow if iter count large
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
    fn apply_rivers_fast(&self, s: &BaseSample, _wx: f32, _wz: f32) -> f32 {
        // fast approximate river carving:
        // keep the river noise channels but reduce iterations and remove heavy hydrology calls.
        if self.p.river_freq <= 0.0 || self.p.river_depth <= 0.0 || self.p.river_width <= 0.0 {
            return s.rel;
        }

        // quick land test using same smoothstep heuristic
        let land = smoothstep(-0.01, 0.03, s.rel) * smoothstep(0.10, 1.0, s.cont);
        if land <= 0.0001 {
            return s.rel;
        }

        // channel: fewer octaves, same flavor
        let mut channel = 0.0f32;
        let mut f = 1.0f32;
        let mut a = 1.0f32;
        // reduce octaves from 4 to 2 for huge speed win
        for _i in 0..2 {
            let n = self.rivers.get_noise_2d(s.wx2 * f, s.wz2 * f);
            let v = 1.0 - n.abs();
            let w = (self.p.river_width / f).clamp(0.01, 0.48); // scale width per octave
            let line = smoothstep(1.0 - w, 1.0, v);
            channel += line * a;
            f *= 2.15;
            a *= 0.62;
        }
        channel = (channel / 1.1).clamp(0.0, 1.0);

        // valley likelihood: approximate using slope_proxy and a cheap local average on hills noise
        // we reuse s.slope_proxy which was precomputed in base_sample
        let slope_n = (s.slope_proxy * 1.2).clamp(0.0, 1.0);
        let valley_like = (1.0 - slope_n).clamp(0.0, 1.0);

        // wetness proxy from moisture noise at a coarse scale
        let wet_n = self.moisture_noise.get_noise_2d(s.wx2 * 0.55, s.wz2 * 0.55);
        let wet = ((wet_n + 1.0) * 0.5).clamp(0.0, 1.0);

        // lowland and discharge approximations
        let lowland = 1.0 - smoothstep(0.18, 0.85, s.rel.max(0.0));
        let discharge = (0.22 + 0.78 * wet) * (0.30 + 0.70 * lowland);

        // coast guard and land mask
        let coast_guard = smoothstep(-0.02, 0.08, s.rel) * smoothstep(0.16, 1.0, s.cont);
        let mask = (channel * valley_like).clamp(0.0, 1.0) * land * coast_guard;

        // carve strength: reduce exponent and use discharge proxy
        let carve = mask.powf(1.6) * self.p.river_depth * discharge;

        // soften mouth effect with a single smoothstep
        let mouth_soft = smoothstep(-0.02, 0.05, s.rel);
        let carve = carve * (0.60 + 0.40 * mouth_soft);

        s.rel - carve
    }

    pub fn height(&self, wx: f32, wz: f32) -> f32 {
        let s = self.base_sample(wx, wz);
        let mut rel = self.apply_rivers_fast(&s, wx, wz);

        rel = micro_flatten(rel, self.p.micro_flatten);

        let mut h = rel * self.p.height_scale + self.p.sea_level;

        h = self.apply_origin_land_override(wx, wz, h);

        h
    }

    pub fn moisture(&self, wx: f32, wz: f32, h: f32) -> f32 {
        // Normalize height once
        let h_rel = (h - self.p.sea_level) / self.p.height_scale;

        // Warp coords and sample moisture noise exactly once
        let (wx2, wz2) = self.warped_coords(wx, wz);
        let n = self.moisture_noise.get_noise_2d(wx2, wz2);
        // convert from [-1,1] to [0,1]
        let mut m = (n + 1.0) * 0.5;

        // continental mask and latitude
        let cont = self.continental_mask(wx, wz);
        let lat = self.latitude_factor(wz);

        // Zonal factor (compact and cheap)
        let mut zonal = 1.0 - ((lat - 0.18) * 1.55).abs();
        zonal = zonal.max(0.0).min(1.0);
        // keep gain for the same curve, it's cheap relative to noise
        zonal = gain(zonal, 0.55);

        // Orographic influence: one grad2 call, tiny algebraic work
        // scale coordinates for the hills gradient once
        let eps = 0.90;
        let (gx, gz) = grad2(&self.hills, wx2 * 0.8, wz2 * 0.8, eps);

        // dot with preferred wind direction
        const WIND_X: f32 = 1.0;
        const WIND_Z: f32 = 0.22;
        let dot = gx * WIND_X + gz * WIND_Z;

        // replicate original mapping: clamp to [-0.02,0.02], scale, then remap to [0,1]
        // combine into minimal ops
        let o = if dot <= -0.02 {
            -0.02
        } else if dot >= 0.02 {
            0.02
        } else {
            dot
        };
        // o in [-0.02,0.02], times 25 -> [-0.5,0.5], remap -> [0.25,0.75]
        let orographic = (o * 25.0 * 0.5 + 0.5).clamp(0.0, 1.0);

        // Blend the three main contributions using original weights
        m = m * 0.40 + zonal * 0.52 + orographic * 0.08;

        // Height dryness factor (clamped)
        let height_dry = if h_rel <= 0.0 {
            0.0
        } else if h_rel >= 1.6 {
            1.6
        } else {
            h_rel
        };
        // apply height dryness multiplier
        m *= 1.0 - height_dry * 0.55;

        // Continental interior effect (smoothstep)
        let interior = smoothstep(0.54, 0.92, cont);
        m *= 1.0 - interior * 0.48;

        // Final clamp to [0,1]
        m.clamp(0.0, 1.0)
    }

    pub fn color(&self, wx: f32, wz: f32, h: f32, moisture: f32) -> [f32; 3] {
        // local shortcuts
        let hs = self.p.height_scale;
        let h_rel = h - self.p.sea_level;
        let h_norm = (h_rel / hs).clamp(-1.2, 1.8);

        // latitude / temperature base
        let lat = self.latitude_factor(wz).clamp(0.0, 1.0);
        // base sample once
        let s = self.base_sample(wx, wz);

        // temperature: mostly lat-driven, slightly modulated by macro_elev noise
        let mut temp = (1.0 - lat).powf(1.6); // cheaper exponent than before but similar curve
        let t_noise = self.macro_elev.get_noise_2d(s.wx2 * 0.020, s.wz2 * 0.020);
        temp = (temp + t_noise * 0.05).clamp(0.0, 1.0);

        // wet/dry
        let dry = (1.0 - moisture).clamp(0.0, 1.0);
        let wet = 1.0 - dry;

        // approximate slope using one cheap gradient sample (instead of 4 hydrology queries)
        let eps = 0.8f32;
        let (gx, gz) = grad2(&self.hills, s.wx2 * 0.9, s.wz2 * 0.9, eps);
        let slope = (gx * gx + gz * gz).sqrt();
        let slope_n = smoothstep(0.08, 0.015, slope);

        // water path (shallow/deep/shelf) — simplified but expressive
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

            // small ice tint at high lat / shallow
            let ice = smoothstep(0.78, 0.96, lat) * smoothstep(-6.0, 2.0, h_rel);
            let ice_col = HSV {
                h: 0.58,
                s: 0.05,
                v: 0.985,
            };
            col = lerp_hsv(col, ice_col, ice);

            return hsv_to_rgb(col);
        }

        // beach / cliff / rockiness
        let beach = smoothstep(-0.01, 0.06, h_norm) * (1.0 - smoothstep(0.08, 0.22, h_norm));
        let beach = beach * (1.0 - smoothstep(0.20, 0.55, slope_n));

        let cliff = smoothstep(0.35, 0.95, slope_n) * smoothstep(0.02, 0.60, h_norm);
        let rockiness = (cliff * 0.75 + smoothstep(0.85, 1.35, h_norm) * 0.55).clamp(0.0, 1.0);

        // climate palettes (kept, but we blend cheaper)
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

        // helper reduced-cost climate
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

        // altitude blend simplified: pick two bands and blend between them
        let alt = h_norm.clamp(0.0, 1.35);
        let t_mid = ((alt - 0.25) / (0.60 - 0.25)).clamp(0.0, 1.0);
        let t_high = ((alt - 0.60) / (1.10 - 0.60)).clamp(0.0, 1.0);

        // start with low->mid, then blend toward high
        let mut col = lerp_hsv(c_low, c_mid, t_mid);
        col = lerp_hsv(col, c_high, t_high);

        // beach + rock influence
        let sand = HSV {
            h: 0.12,
            s: 0.32,
            v: 0.90,
        };
        col = lerp_hsv(col, sand, beach);

        // rock tint (single noise sample)
        let rock_n = self.rock.get_noise_2d(s.wx2 * 1.25, s.wz2 * 1.25);
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

        // small per-vertex detail + shadow (one detail noise)
        let detail = self.detail.get_noise_2d(s.wx2 * 3.2, s.wz2 * 3.2);
        let var = (detail * 0.06).clamp(-0.08, 0.08);
        col.v = (col.v * (1.0 + var)).clamp(0.0, 1.0);

        let shadow = (slope_n * 0.20).clamp(0.0, 0.20);
        col.v = (col.v * (1.0 - shadow)).clamp(0.0, 1.0);

        // snow/ice tint (simplified, cheap)
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
