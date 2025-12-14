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

#[derive(Clone, Copy, Debug)]
pub struct TerrainParams {
    pub seed: u32,
    // Global scaling: >1 = larger features (zoomed out), <1 = smaller features (zoomed in)
    pub world_scale: f32,

    // Output scaling
    pub height_scale: f32,
    pub sea_level: f32,

    // Continents / planet layout
    pub lat_extent: f32,
    pub continent_radius: f32,
    pub ring_radius: f32,
    pub coast_noise_scale: f32,
    pub coast_noise_amp: f32,
    pub island_threshold0: f32,
    pub island_threshold1: f32,
    pub island_amp: f32,

    // Force land at origin
    pub force_land_at_origin: bool,
    pub origin_island_radius: f32,   // world units
    pub origin_island_strength: f32, // 0..1-ish
    pub pull_one_continent_to_origin: bool,
    pub origin_pull_strength: f32, // 0..1

    // Domain warp
    pub warp_large_scale: f32,
    pub warp_small_scale: f32,
    pub warp_large_amp: f32,
    pub warp_small_amp: f32,
    pub warp_mix_large: f32, // 0..1
    pub warp_mix_small: f32, // 0..1

    // Noise frequencies (these are in “world units”; they get multiplied by world_scale internally)
    pub macro_freq: f32,
    pub hills_freq: f32,
    pub mountains_freq: f32,
    pub belts_freq: f32,
    pub moisture_freq: f32,

    // Octaves/persistence
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

    // Amplitudes
    pub ocean_floor: f32,
    pub inland_plateau: f32,
    pub macro_amp: f32,
    pub hills_amp: f32,
    pub mountains_amp: f32,
    pub belt_amp: f32,

    // Coast / mountains shaping
    pub coast_soften_width: f32, // around cont=0.5
    pub coast_soften_strength: f32,
    pub interior_lo: f32,
    pub interior_hi: f32,
    pub belt_lo: f32,
    pub belt_hi: f32,

    // Flatten/smooth controls
    pub flatten: f32,         // 0..1, higher = flatter
    pub flatten_curve: f32,   // >= 1, higher compresses extremes more
    pub mountain_smooth: f32, // 0..1, reduces ridged harshness
    pub hills_detail: f32,    // 0..1
    pub micro_flatten: f32,   // 0..1

    // Tectonics
    pub plate_freq: f32,
    pub plate_sharpness: f32,
    pub plate_mountain_amp: f32,

    // Erosion
    pub erosion_strength: f32,
    pub erosion_iters: usize,

    // Rivers
    pub river_freq: f32,
    pub river_depth: f32,
    pub river_width: f32,

    // Slope effects
    pub snow_slope_limit: f32,
}

impl Default for TerrainParams {
    fn default() -> Self {
        Self {
            seed: 201035458,
            world_scale: 0.5,

            height_scale: 1200.0,
            sea_level: 0.0,

            lat_extent: 120_000.0,
            continent_radius: 40_000.0,
            ring_radius: 70_000.0,
            coast_noise_scale: 0.00025,
            coast_noise_amp: 0.25,
            island_threshold0: 0.80,
            island_threshold1: 0.95,
            island_amp: 0.45,

            force_land_at_origin: true,
            origin_island_radius: 18_000.0,
            origin_island_strength: 0.85,
            pull_one_continent_to_origin: true,
            origin_pull_strength: 0.65,

            warp_large_scale: 0.0005,
            warp_small_scale: 0.0025,
            warp_large_amp: 90.0,
            warp_small_amp: 90.0,
            warp_mix_large: 0.8,
            warp_mix_small: 0.2,

            macro_freq: 0.0009,
            hills_freq: 0.0035,
            mountains_freq: 0.006,
            belts_freq: 0.00108, // = 0.006 * 0.18
            moisture_freq: 0.0015,

            macro_octaves: 4,
            macro_persistence: 0.45,
            hills_octaves: 5,
            hills_persistence: 0.5,
            mountains_octaves: 6,
            mountains_persistence: 0.48,
            continent_octaves: 3,
            continent_persistence: 0.8,
            moisture_octaves: 4,
            moisture_persistence: 0.6,
            warp_large_octaves: 3,
            warp_large_persistence: 0.7,
            warp_small_octaves: 2,
            warp_small_persistence: 0.5,

            ocean_floor: -0.9,
            inland_plateau: 0.35,
            macro_amp: 0.45,
            hills_amp: 0.25,
            mountains_amp: 1.3,
            belt_amp: 1.0,

            coast_soften_width: 0.28,
            coast_soften_strength: 0.15,
            interior_lo: 0.45,
            interior_hi: 0.95,
            belt_lo: 0.55,
            belt_hi: 0.9,

            // Smoothing: start a bit flatter by default
            flatten: 0.7,
            flatten_curve: 2.4,
            mountain_smooth: 0.6,

            hills_detail: 0.18,
            micro_flatten: 0.0,

            plate_freq: 0.00035,
            plate_sharpness: 3.5,
            plate_mountain_amp: 1.2,

            erosion_strength: 0.35,
            erosion_iters: 2,

            river_freq: 0.0000,
            river_depth: 0.85,
            river_width: 0.085,

            snow_slope_limit: 0.65,
        }
    }
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

    continent_centers: [(f32, f32); 6],
}

impl Clone for TerrainGenerator {
    fn clone(&self) -> Self {
        Self::new(self.p.clone())
    }
}

impl TerrainGenerator {
    pub fn new(terrain_params: TerrainParams) -> Self {
        Self::with_params(terrain_params)
    }

    pub fn with_params(mut p: TerrainParams) -> Self {
        let seed = p.seed;
        // Avoid weird stuff
        p.world_scale = p.world_scale.max(0.000001);

        // Frequencies are multiplied by world_scale so you can scale the entire world in one place.
        // If you prefer inverse behaviour, flip to (1.0 / p.world_scale).
        let ws = p.world_scale;

        let mut macro_elev = FastNoiseLite::new();
        macro_elev.set_seed(Some(seed as i32));
        macro_elev.set_noise_type(Some(NoiseType::Perlin));
        macro_elev.set_fractal_type(Some(FractalType::FBm));
        macro_elev.set_fractal_octaves(Some(p.macro_octaves as i32));
        macro_elev.set_frequency(Some(p.macro_freq * ws));
        macro_elev.set_fractal_gain(Some(p.macro_persistence));

        let mut hills = FastNoiseLite::new();
        hills.set_seed(Some(seed.wrapping_add(10) as i32));
        hills.set_noise_type(Some(NoiseType::Perlin));
        hills.set_fractal_type(Some(FractalType::FBm));
        hills.set_fractal_octaves(Some(p.hills_octaves as i32));
        hills.set_frequency(Some(p.hills_freq * ws));
        hills.set_fractal_gain(Some(p.hills_persistence));

        let mut mountains = FastNoiseLite::new();
        mountains.set_seed(Some(seed.wrapping_add(1) as i32));
        mountains.set_noise_type(Some(NoiseType::Perlin));
        mountains.set_fractal_type(Some(FractalType::FBm));
        mountains.set_fractal_octaves(Some(p.mountains_octaves as i32));
        mountains.set_frequency(Some(p.mountains_freq * ws));
        mountains.set_fractal_gain(Some(p.mountains_persistence));

        let mut plates = FastNoiseLite::new();
        plates.set_seed(Some(seed.wrapping_add(42) as i32));
        plates.set_noise_type(Some(NoiseType::Perlin));
        plates.set_fractal_type(Some(FractalType::FBm));
        plates.set_fractal_octaves(Some(3));
        plates.set_frequency(Some(p.plate_freq * ws));
        plates.set_fractal_gain(Some(0.5));

        let mut rivers = FastNoiseLite::new();
        rivers.set_seed(Some(seed.wrapping_add(77) as i32));
        rivers.set_noise_type(Some(NoiseType::Perlin));
        rivers.set_fractal_type(Some(FractalType::FBm));
        rivers.set_fractal_octaves(Some(4));
        rivers.set_frequency(Some(p.river_freq * ws));
        rivers.set_fractal_gain(Some(0.55));

        let mut continent_noise = FastNoiseLite::new();
        continent_noise.set_seed(Some(seed.wrapping_add(2) as i32));
        continent_noise.set_noise_type(Some(NoiseType::Perlin));
        continent_noise.set_fractal_type(Some(FractalType::FBm));
        continent_noise.set_fractal_octaves(Some(p.continent_octaves as i32));
        continent_noise.set_frequency(Some(0.0003 * ws));
        continent_noise.set_fractal_gain(Some(p.continent_persistence));

        let mut moisture_noise = FastNoiseLite::new();
        moisture_noise.set_seed(Some(seed as i32));
        moisture_noise.set_noise_type(Some(NoiseType::Perlin));
        moisture_noise.set_fractal_type(Some(FractalType::FBm));
        moisture_noise.set_fractal_octaves(Some(p.moisture_octaves as i32));
        moisture_noise.set_frequency(Some(p.moisture_freq * ws));
        moisture_noise.set_fractal_gain(Some(p.moisture_persistence));

        let mut warp_large = FastNoiseLite::new();
        warp_large.set_seed(Some(seed.wrapping_add(4) as i32));
        warp_large.set_noise_type(Some(NoiseType::Perlin));
        warp_large.set_fractal_type(Some(FractalType::FBm));
        warp_large.set_fractal_octaves(Some(p.warp_large_octaves as i32));
        warp_large.set_frequency(Some(p.warp_large_scale * ws));
        warp_large.set_fractal_gain(Some(p.warp_large_persistence));

        let mut warp_small = FastNoiseLite::new();
        warp_small.set_seed(Some(seed.wrapping_add(5) as i32));
        warp_small.set_noise_type(Some(NoiseType::Perlin));
        warp_small.set_fractal_type(Some(FractalType::FBm));
        warp_small.set_fractal_octaves(Some(p.warp_small_octaves as i32));
        warp_small.set_frequency(Some(p.warp_small_scale * ws));
        warp_small.set_fractal_gain(Some(p.warp_small_persistence));

        // Continent centers
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

            let cx = angle.cos() * r;
            let cz = lat_band * p.lat_extent;

            centers[i] = (cx, cz);
        }

        // Optional: pull the first “mid-lat” continent toward origin so (0,0) is more likely land,
        // even if you disable the origin island.
        if p.pull_one_continent_to_origin {
            let idx = 2; // one of the non-polar ones
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
            continent_centers: centers,
        }
    }

    // world_scale applied to coords here, so every system sees the same “zoom”.
    // If you want opposite behaviour, use inv = 1.0 / world_scale.
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

        let dx =
            (w1x * self.p.warp_mix_large + w2x * self.p.warp_mix_small) * self.p.warp_large_amp;
        let dz =
            (w1z * self.p.warp_mix_large + w2z * self.p.warp_mix_small) * self.p.warp_small_amp;

        (x + dx, z + dz)
    }

    // 6 supercontinents + noisy coasts + islands + optional forced origin land
    fn continental_mask(&self, wx: f32, wz: f32) -> f32 {
        let (wx, wz) = self.scaled_coords(wx, wz);

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

        let nx = wx * (self.p.coast_noise_scale);
        let nz = wz * (self.p.coast_noise_scale);
        let noise = self.continent_noise.get_noise_2d(nx, nz) * self.p.coast_noise_amp;

        let mut c = (best + noise).clamp(0.0, 1.0);

        // random island chains out in the ocean
        let island_raw = self.continent_noise.get_noise_2d(nx * 3.0, nz * 3.0);
        let island_v = (island_raw + 1.0) * 0.5;
        let island =
            smoothstep(self.p.island_threshold0, self.p.island_threshold1, island_v) * (1.0 - c);
        c += island * self.p.island_amp;

        // Force land at origin: blend in a deterministic “origin island” mask.
        if self.p.force_land_at_origin {
            let r = self.p.origin_island_radius.max(1.0);
            let d = (wx * wx + wz * wz).sqrt();
            let t = (1.0 - d / r).clamp(0.0, 1.0);
            let island = t * t * (3.0 - 2.0 * t); // smoothstep-ish
            c = (c + island * self.p.origin_island_strength).clamp(0.0, 1.0);
        }

        c.clamp(0.0, 1.0)
    }

    // 0 at equator (wz ~= 0), 1 at poles
    fn latitude_factor(&self, wz: f32) -> f32 {
        let (_, wz) = self.scaled_coords(0.0, wz);
        let t = (wz / self.p.lat_extent).abs();
        t.min(1.0)
    }

    #[inline]
    fn flatten_profile(&self, rel: f32, cont: f32) -> f32 {
        // Compress extremes while keeping sign and sea-level behaviour.
        // Apply more on land than in deep ocean.
        let f = self.p.flatten.clamp(0.0, 1.0);
        if f <= 0.0001 {
            return rel;
        }

        // Map [-1..+1]ish into a softened curve.
        let sign = rel.signum();
        let a = rel.abs();

        // Curve that compresses big heights more than small ones:
        // a' = a / (1 + k*a^c) (a smooth saturating curve)
        let c = self.p.flatten_curve.max(1.0);
        let k = 1.25 * f; // strength
        let a2 = a / (1.0 + k * a.powf(c));

        // Blend: more flattening inland, less in ocean.
        let land = smoothstep(0.25, 0.65, cont);
        let mix = f * (0.35 + 0.65 * land);
        let out = lerp(rel, sign * a2, mix);

        out
    }

    pub fn height(&self, wx: f32, wz: f32) -> f32 {
        let hs = self.p.height_scale;

        let cont = self.continental_mask(wx, wz);
        let (wx2, wz2) = self.warped_coords(wx, wz);

        // Macro/hills/mountains
        let macro_raw = self.macro_elev.get_noise_2d(wx2 * 0.6, wz2 * 0.6);
        let macro_elev = macro_raw * self.p.macro_amp;

        let hills_raw = self.hills.get_noise_2d(wx2 * 2.0, wz2 * 2.0);
        let hills = hills_raw * self.p.hills_amp * self.p.hills_detail;

        let m_raw = self.mountains.get_noise_2d(wx2, wz2);
        let mut ridged = 1.0 - m_raw.abs();
        ridged = ridged.max(0.0);
        ridged = ridged * ridged * ridged;

        // Smooth harsh ridges (knob)
        if self.p.mountain_smooth > 0.0 {
            let s = self.p.mountain_smooth.clamp(0.0, 1.0);
            // Blend ridged toward a softer bump
            let soft = ridged.sqrt();
            ridged = lerp(ridged, soft, s);
        }

        let belts_raw = self
            .mountains
            .get_noise_2d(wx2 * 0.18 + 1234.0, wz2 * 0.18 - 5678.0);
        let belt_n = (belts_raw + 1.0) * 0.5;
        let belt_mask = smoothstep(self.p.belt_lo, self.p.belt_hi, belt_n);

        let interior = smoothstep(self.p.interior_lo, self.p.interior_hi, cont);
        let mountain_mask = belt_mask * interior;

        // Base continental profile
        let mut rel = self.p.ocean_floor + (self.p.inland_plateau - self.p.ocean_floor) * cont;

        // Detail mostly on land
        rel += macro_elev * (0.3 + 0.7 * cont);
        rel += hills * cont;

        rel += ridged * mountain_mask * self.p.mountains_amp * self.p.belt_amp;

        let river_n = self.rivers.get_noise_2d(wx2, wz2);
        let river = 1.0 - river_n.abs();
        let river_mask = smoothstep(1.0 - self.p.river_width, 1.0, river);

        let downhill = smoothstep(0.05, 0.6, -rel);
        let river_cut = river_mask * downhill;

        rel -= river_cut * self.p.river_depth;

        let n1 = self.hills.get_noise_2d(wx2 + 1.0, wz2);
        let n2 = self.hills.get_noise_2d(wx2, wz2 + 1.0);

        let slope = (n1 - n2).abs();

        rel = erode(rel, slope, self.p.erosion_strength * cont);

        // Coastline softening (knobs)
        let coast = (cont - 0.5).abs();
        let w = self.p.coast_soften_width.max(0.0001);
        let coast_t = ((w - coast) / w).clamp(0.0, 1.0);
        rel = rel * (1.0 - self.p.coast_soften_strength * coast_t);

        let plate_raw = self.plates.get_noise_2d(wx2 * 0.7, wz2 * 0.7);
        let plate_edges = (1.0 - plate_raw.abs()).powf(self.p.plate_sharpness);

        rel += plate_edges * mountain_mask * self.p.plate_mountain_amp * 0.6;

        // Global flattening/smoothing (main request)
        rel = self.flatten_profile(rel, cont);

        rel = micro_flatten(rel, self.p.micro_flatten);

        rel *= hs + self.p.sea_level;
        if rel > -0.3 && rel < 0.3 {
            return 0.3;
        }
        rel
    }

    pub fn moisture(&self, wx: f32, wz: f32) -> f32 {
        let hs = self.p.height_scale;
        let h = self.height(wx, wz);
        let h_rel = (h - self.p.sea_level) / hs;

        let (wx2, wz2) = self.warped_coords(wx, wz);
        let n = self.moisture_noise.get_noise_2d(wx2, wz2);
        let mut m = (n + 1.0) * 0.5;

        let cont = self.continental_mask(wx, wz);
        let lat = self.latitude_factor(wz);

        let mut zonal = 1.0 - ((lat - 0.2) * 1.4).abs();
        zonal = zonal.clamp(0.0, 1.0);

        m = m * 0.4 + zonal * 0.6;

        let height_dry = h_rel.clamp(0.0, 1.4);
        m *= 1.0 - height_dry * 0.55;

        let interior = smoothstep(0.55, 0.9, cont);
        m *= 1.0 - interior * 0.45;

        m.clamp(0.0, 1.0)
    }

    pub fn color(&self, wx: f32, wz: f32, h: f32, moisture: f32) -> [f32; 3] {
        let hs = self.p.height_scale;
        let h_rel = h - self.p.sea_level;
        let h_norm = (h_rel / hs).clamp(-1.0, 1.5);

        let lat = self.latitude_factor(wz).clamp(0.0, 1.0);
        let (wx2, wz2) = self.warped_coords(wx, wz);

        // ---------- TEMPERATURE ----------
        let mut temp = (1.0 - lat).powf(1.8);

        let alt_cool = h_norm.max(0.0).powf(1.2) * 0.8;
        temp *= 1.0 - alt_cool;

        let t_noise = self.macro_elev.get_noise_2d(wx2 * 0.02, wz2 * 0.02);
        temp = (temp + t_noise * 0.04).clamp(0.0, 1.0);

        let dry = (1.0 - moisture).clamp(0.0, 1.0);
        let wet = 1.0 - dry;

        // ---------- OCEAN ----------
        if h_rel < 0.0 {
            let depth = (-h_rel / (0.7 * hs)).clamp(0.0, 1.0);

            let warm = HSV {
                h: 0.55,
                s: 0.65,
                v: 0.65,
            };
            let cold = HSV {
                h: 0.58,
                s: 0.25,
                v: 0.85,
            };

            let surface = lerp_hsv(warm, cold, lat);
            let deep = HSV {
                h: surface.h,
                s: surface.s * 0.8,
                v: surface.v * 0.35,
            };

            let mut col = lerp_hsv(surface, deep, depth);

            let ice = smoothstep(0.75, 0.95, lat) * smoothstep(-6.0, 2.0, h_rel);
            let ice_col = HSV {
                h: 0.58,
                s: 0.05,
                v: 0.98,
            };
            col = lerp_hsv(col, ice_col, ice);

            return hsv_to_rgb(col);
        }

        // ---------- BIOME PALETTES (HSV) ----------
        let low_cold_dry = HSV {
            h: 0.25,
            s: 0.25,
            v: 0.55,
        };
        let low_cold_wet = HSV {
            h: 0.33,
            s: 0.55,
            v: 0.45,
        };
        let low_hot_dry = HSV {
            h: 0.15,
            s: 0.55,
            v: 0.80,
        };
        let low_hot_wet = HSV {
            h: 0.33,
            s: 0.75,
            v: 0.55,
        };

        let mid_cold_dry = HSV {
            h: 0.22,
            s: 0.20,
            v: 0.60,
        };
        let mid_cold_wet = HSV {
            h: 0.30,
            s: 0.40,
            v: 0.50,
        };
        let mid_hot_dry = HSV {
            h: 0.12,
            s: 0.45,
            v: 0.70,
        };
        let mid_hot_wet = HSV {
            h: 0.30,
            s: 0.65,
            v: 0.55,
        };

        let high_cold = HSV {
            h: 0.58,
            s: 0.05,
            v: 0.92,
        };
        let high_hot = HSV {
            h: 0.00,
            s: 0.00,
            v: 0.65,
        };

        fn climate(
            temp: f32,
            wet: f32,
            cold_dry: HSV,
            cold_wet: HSV,
            hot_dry: HSV,
            hot_wet: HSV,
        ) -> HSV {
            let cold = lerp_hsv(cold_dry, cold_wet, wet);
            let hot = lerp_hsv(hot_dry, hot_wet, wet);
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

        // ---------- ALTITUDE BLEND ----------
        let alt = h_norm.clamp(0.0, 1.3);

        let w_low = tri_weight(alt, 0.20, 0.25);
        let w_mid = tri_weight(alt, 0.60, 0.25);
        let w_high = tri_weight(alt, 1.05, 0.30);
        let sum = (w_low + w_mid + w_high).max(0.0001);

        let mut col = HSV {
            h: (c_low.h * w_low + c_mid.h * w_mid + c_high.h * w_high) / sum,
            s: (c_low.s * w_low + c_mid.s * w_mid + c_high.s * w_high) / sum,
            v: (c_low.v * w_low + c_mid.v * w_mid + c_high.v * w_high) / sum,
        };

        // ---------- DETAIL VARIATION ----------
        let detail = self.hills.get_noise_2d(wx2 * 3.0, wz2 * 3.0);
        col.v = (col.v * (1.0 + detail * 0.05)).clamp(0.0, 1.0);

        // ---------- SNOW OVERLAY ----------
        let snow_lat = smoothstep(0.80, 0.98, lat);
        let snow_alt = smoothstep(0.40, 1.15, alt);

        let nx = self.hills.get_noise_2d(wx2 + 0.5, wz2);
        let nz = self.hills.get_noise_2d(wx2, wz2 + 0.5);
        let slope = (nx - nz).abs();

        let slope_mask = smoothstep(self.p.snow_slope_limit, 1.0, 1.0 - slope);

        let snow = (snow_lat * 0.6 + snow_alt * 0.9) * slope_mask;
        let snow_col = HSV {
            h: 0.58,
            s: 0.03,
            v: 0.98,
        };

        col = lerp_hsv(col, snow_col, snow.clamp(0.0, 1.0));

        hsv_to_rgb(col)
    }
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
fn lerp_color(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

#[inline]
fn tri_weight(x: f32, center: f32, width: f32) -> f32 {
    let d = (x - center).abs();
    if d >= width { 0.0 } else { 1.0 - d / width }
}

fn micro_flatten(rel: f32, strength: f32) -> f32 {
    let a = rel.abs();
    let t = smoothstep(0.0, 0.15, a);
    lerp(rel, rel * 0.4, strength * (1.0 - t))
}

fn erode(h: f32, n: f32, strength: f32) -> f32 {
    let slope = (n.abs()).clamp(0.0, 1.0);
    h - slope * strength
}
