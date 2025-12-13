use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
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
    pub macro_persistence: f64,
    pub hills_octaves: usize,
    pub hills_persistence: f64,
    pub mountains_octaves: usize,
    pub mountains_persistence: f64,
    pub continent_octaves: usize,
    pub continent_persistence: f64,
    pub moisture_octaves: usize,
    pub moisture_persistence: f64,
    pub warp_large_octaves: usize,
    pub warp_large_persistence: f64,
    pub warp_small_octaves: usize,
    pub warp_small_persistence: f64,

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
}

impl Default for TerrainParams {
    fn default() -> Self {
        Self {
            world_scale: 1.0,

            height_scale: 120.0,
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
            micro_flatten: 0.6,
        }
    }
}

#[derive(Clone)]
pub struct TerrainGenerator {
    p: TerrainParams,

    macro_elev: Fbm<Perlin>,
    hills: Fbm<Perlin>,
    mountains: Fbm<Perlin>,
    continent_noise: Fbm<Perlin>,
    moisture_noise: Fbm<Perlin>,
    warp_large: Fbm<Perlin>,
    warp_small: Fbm<Perlin>,

    continent_centers: [(f32, f32); 6],
}

impl TerrainGenerator {
    pub fn new(seed: u32) -> Self {
        Self::with_params(seed, TerrainParams::default())
    }

    pub fn with_params(seed: u32, mut p: TerrainParams) -> Self {
        // Avoid weird stuff
        p.world_scale = p.world_scale.max(0.000001);

        // Frequencies are multiplied by world_scale so you can scale the entire world in one place.
        // If you prefer inverse behaviour, flip to (1.0 / p.world_scale).
        let ws = p.world_scale;

        let macro_elev = Fbm::<Perlin>::new(seed)
            .set_octaves(p.macro_octaves)
            .set_frequency((p.macro_freq * ws) as f64)
            .set_persistence(p.macro_persistence);

        let hills = Fbm::<Perlin>::new(seed.wrapping_add(10))
            .set_octaves(p.hills_octaves)
            .set_frequency((p.hills_freq * ws) as f64)
            .set_persistence(p.hills_persistence);

        let mountains = Fbm::<Perlin>::new(seed.wrapping_add(1))
            .set_octaves(p.mountains_octaves)
            .set_frequency((p.mountains_freq * ws) as f64)
            .set_persistence(p.mountains_persistence);

        let continent_noise = Fbm::<Perlin>::new(seed.wrapping_add(2))
            .set_octaves(p.continent_octaves)
            .set_frequency((0.0003 * ws) as f64)
            .set_persistence(p.continent_persistence);

        let moisture_noise = Fbm::<Perlin>::new(seed.wrapping_add(3))
            .set_octaves(p.moisture_octaves)
            .set_frequency((p.moisture_freq * ws) as f64)
            .set_persistence(p.moisture_persistence);

        let warp_large = Fbm::<Perlin>::new(seed.wrapping_add(4))
            .set_octaves(p.warp_large_octaves)
            .set_frequency((p.warp_large_scale * ws) as f64)
            .set_persistence(p.warp_large_persistence);

        let warp_small = Fbm::<Perlin>::new(seed.wrapping_add(5))
            .set_octaves(p.warp_small_octaves)
            .set_frequency((p.warp_small_scale * ws) as f64)
            .set_persistence(p.warp_small_persistence);

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

    fn warped_coords(&self, wx: f32, wz: f32) -> (f64, f64) {
        let (wx, wz) = self.scaled_coords(wx, wz);

        let x = wx as f64;
        let z = wz as f64;

        let w1x = self.warp_large.get([x, z]) as f32;
        let w1z = self.warp_large.get([x + 1234.0, z - 5678.0]) as f32;

        let w2x = self.warp_small.get([x * 2.0, z * 2.0]) as f32;
        let w2z = self.warp_small.get([x * 2.0 - 500.0, z * 2.0 + 500.0]) as f32;

        let dx =
            (w1x * self.p.warp_mix_large + w2x * self.p.warp_mix_small) * self.p.warp_large_amp;
        let dz =
            (w1z * self.p.warp_mix_large + w2z * self.p.warp_mix_small) * self.p.warp_small_amp;

        (x + dx as f64, z + dz as f64)
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

        let nx = wx as f64 * (self.p.coast_noise_scale as f64);
        let nz = wz as f64 * (self.p.coast_noise_scale as f64);
        let noise = self.continent_noise.get([nx, nz]) as f32 * self.p.coast_noise_amp;

        let mut c = (best + noise).clamp(0.0, 1.0);

        // random island chains out in the ocean
        let island_raw = self.continent_noise.get([nx * 3.0, nz * 3.0]) as f32;
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
        let macro_raw = self.macro_elev.get([wx2 * 0.6, wz2 * 0.6]) as f32;
        let macro_elev = macro_raw * self.p.macro_amp;

        let hills_raw = self.hills.get([wx2 * 2.0, wz2 * 2.0]) as f32;
        let hills = hills_raw * self.p.hills_amp * self.p.hills_detail;

        let m_raw = self.mountains.get([wx2, wz2]) as f32;
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
            .get([wx2 * 0.18 + 1234.0, wz2 * 0.18 - 5678.0]) as f32;
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

        // Coastline softening (knobs)
        let coast = (cont - 0.5).abs();
        let w = self.p.coast_soften_width.max(0.0001);
        let coast_t = ((w - coast) / w).clamp(0.0, 1.0);
        rel = rel * (1.0 - self.p.coast_soften_strength * coast_t);

        // Global flattening/smoothing (main request)
        rel = self.flatten_profile(rel, cont);

        rel = micro_flatten(rel, self.p.micro_flatten);

        rel * hs + self.p.sea_level
    }

    pub fn moisture(&self, wx: f32, wz: f32) -> f32 {
        let hs = self.p.height_scale;
        let h = self.height(wx, wz);
        let h_rel = (h - self.p.sea_level) / hs;

        let (wx2, wz2) = self.warped_coords(wx, wz);
        let n = self.moisture_noise.get([wx2, wz2]) as f32;
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

    // color() unchanged except swap self.height_scale/self.sea_level and latitude_factor uses params now.
    pub fn color(&self, wx: f32, wz: f32, h: f32, moisture: f32) -> [f32; 3] {
        let hs = self.p.height_scale;
        let h_rel = h - self.p.sea_level;
        let h_norm = (h_rel / hs).clamp(-1.0, 1.5);
        let lat = self.latitude_factor(wz);

        let (wx2, wz2) = self.warped_coords(wx, wz);

        let mut temp_lat = 1.0 - lat * lat;
        let mut temp = temp_lat;

        let alt_cool = (h_norm.max(0.0) * 0.7).clamp(0.0, 0.9);
        temp *= 1.0 - alt_cool;

        let t_noise = self.macro_elev.get([wx2 * 0.02, wz2 * 0.02]) as f32;
        temp += t_noise * 0.03;
        temp = temp.clamp(0.0, 1.0);

        let dry = (1.0 - moisture).clamp(0.0, 1.0);

        if h_rel < 0.0 {
            let depth = (-h_rel / (0.7 * hs)).clamp(0.0, 1.0);

            let warm_shallow = [0.06, 0.42, 0.60];
            let warm_deep = [0.02, 0.10, 0.24];
            let cold_shallow = [0.70, 0.80, 0.87];
            let cold_deep = [0.03, 0.08, 0.20];

            let cold = lat.clamp(0.0, 1.0);

            let shallow_col = lerp_color(warm_shallow, cold_shallow, cold);
            let deep_col = lerp_color(warm_deep, cold_deep, cold);
            let mut color = lerp_color(shallow_col, deep_col, depth);

            let ice_lat = smoothstep(0.78, 0.98, lat);
            let ice_depth = smoothstep(-5.0, 3.0, h_rel);
            let ice = (ice_lat * ice_depth).clamp(0.0, 1.0);
            let ice_color = [0.94, 0.97, 1.0];
            color = lerp_color(color, ice_color, ice);

            return color;
        }

        let alt = h_norm.clamp(0.0, 1.4);

        let w_low = tri_weight(alt, 0.15, 0.35);
        let w_mid = tri_weight(alt, 0.55, 0.40);
        let w_high = tri_weight(alt, 1.05, 0.55);
        let sum = (w_low + w_mid + w_high).max(0.0001);
        let w_low = w_low / sum;
        let w_mid = w_mid / sum;
        let w_high = w_high / sum;

        let low_cold_dry = [0.72, 0.76, 0.70];
        let low_cold_wet = [0.16, 0.44, 0.26];
        let low_hot_dry = [0.91, 0.83, 0.55];
        let low_hot_wet = [0.07, 0.45, 0.18];

        let mid_cold_dry = [0.62, 0.64, 0.62];
        let mid_cold_wet = [0.15, 0.40, 0.27];
        let mid_hot_dry = [0.80, 0.76, 0.52];
        let mid_hot_wet = [0.18, 0.50, 0.26];

        let high_cold_dry = [0.93, 0.95, 0.97];
        let high_cold_wet = [0.96, 0.98, 1.0];
        let high_hot_dry = [0.64, 0.64, 0.66];
        let high_hot_wet = [0.58, 0.60, 0.64];

        fn climate_color(
            temp: f32,
            dry: f32,
            cold_dry: [f32; 3],
            cold_wet: [f32; 3],
            hot_dry: [f32; 3],
            hot_wet: [f32; 3],
        ) -> [f32; 3] {
            let wet = 1.0 - dry;
            let cold_mix = lerp_color(cold_dry, cold_wet, wet);
            let hot_mix = lerp_color(hot_dry, hot_wet, wet);
            lerp_color(cold_mix, hot_mix, temp)
        }

        let c_low = climate_color(
            temp,
            dry,
            low_cold_dry,
            low_cold_wet,
            low_hot_dry,
            low_hot_wet,
        );
        let c_mid = climate_color(
            temp,
            dry,
            mid_cold_dry,
            mid_cold_wet,
            mid_hot_dry,
            mid_hot_wet,
        );
        let c_high = climate_color(
            temp,
            dry,
            high_cold_dry,
            high_cold_wet,
            high_hot_dry,
            high_hot_wet,
        );

        let mut color = [
            c_low[0] * w_low + c_mid[0] * w_mid + c_high[0] * w_high,
            c_low[1] * w_low + c_mid[1] * w_mid + c_high[1] * w_high,
            c_low[2] * w_low + c_mid[2] * w_mid + c_high[2] * w_high,
        ];

        let snow_lat = smoothstep(0.80, 0.98, lat);
        let snow_alt = smoothstep(0.35, 1.2, alt);
        let snow = (snow_lat * 0.8 + snow_alt * 0.6).clamp(0.0, 1.0);
        let snow_color = [0.96, 0.97, 0.99];
        color = lerp_color(color, snow_color, snow);

        let light = 0.97 + alt * 0.08;
        color[0] = (color[0] * light).min(1.0);
        color[1] = (color[1] * light).min(1.0);
        color[2] = (color[2] * light).min(1.0);

        let detail_noise = self.hills.get([wx2 * 3.0, wz2 * 3.0]) as f32;
        let tint = 1.0 + detail_noise * 0.03;
        color[0] = (color[0] * tint).clamp(0.0, 1.0);
        color[1] = (color[1] * tint).clamp(0.0, 1.0);
        color[2] = (color[2] * tint).clamp(0.0, 1.0);

        color
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
