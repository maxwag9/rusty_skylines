use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use std::f32::consts::PI;

const TAU: f32 = PI * 2.0;

fn hash01(mut x: u32) -> f32 {
    x ^= x >> 13;
    x = x.wrapping_mul(0x85ebca6b);
    x ^= x >> 16;
    (x as f32) / (u32::MAX as f32)
}

#[derive(Clone)]
pub struct TerrainGenerator {
    macro_elev: Fbm<Perlin>,
    hills: Fbm<Perlin>,
    mountains: Fbm<Perlin>,
    continent_noise: Fbm<Perlin>,
    moisture_noise: Fbm<Perlin>,
    warp_large: Fbm<Perlin>,
    warp_small: Fbm<Perlin>,

    // 6 big landmasses laid out on a "planet"
    continent_centers: [(f32, f32); 6],
    continent_radius: f32,
    // distance from equator to pole in world units
    lat_extent: f32,

    pub height_scale: f32,
    pub sea_level: f32,
}

impl TerrainGenerator {
    pub fn new(seed: u32) -> Self {
        let macro_elev = Fbm::<Perlin>::new(seed)
            .set_octaves(4)
            .set_frequency(0.0009)
            .set_persistence(0.45);

        let hills = Fbm::<Perlin>::new(seed.wrapping_add(10))
            .set_octaves(5)
            .set_frequency(0.0035)
            .set_persistence(0.5);

        let mountains = Fbm::<Perlin>::new(seed.wrapping_add(1))
            .set_octaves(6)
            .set_frequency(0.006)
            .set_persistence(0.48);

        let continent_noise = Fbm::<Perlin>::new(seed.wrapping_add(2))
            .set_octaves(3)
            .set_frequency(0.0003)
            .set_persistence(0.8);

        let moisture_noise = Fbm::<Perlin>::new(seed.wrapping_add(3))
            .set_octaves(4)
            .set_frequency(0.0015)
            .set_persistence(0.6);

        let warp_large = Fbm::<Perlin>::new(seed.wrapping_add(4))
            .set_octaves(3)
            .set_frequency(0.0005)
            .set_persistence(0.7);

        let warp_small = Fbm::<Perlin>::new(seed.wrapping_add(5))
            .set_octaves(2)
            .set_frequency(0.0025)
            .set_persistence(0.5);

        // scale: 0 at equator, ±lat_extent at poles
        let lat_extent = 120_000.0;
        let continent_radius = 40_000.0;
        let ring_radius = 70_000.0;

        let mut centers = [(0.0f32, 0.0f32); 6];

        // Two polar continents, four mid-latitude/equatorial
        for i in 0..6 {
            let base_angle = (i as f32) / 6.0 * TAU;
            let jitter = (hash01(seed.wrapping_add(i as u32)) - 0.5) * 0.6;
            let angle = base_angle + jitter;

            let radial_jitter = (hash01(seed.wrapping_add(100 + i as u32)) - 0.5) * 0.25;
            let r = ring_radius * (1.0 + radial_jitter);

            let lat_band = if i < 2 {
                // 0: Arctic, 1: Antarctic
                let sign = if i == 0 { 1.0 } else { -1.0 };
                let band_jitter = (hash01(seed.wrapping_add(200 + i as u32)) - 0.5) * 0.1;
                sign * (0.85 + band_jitter)
            } else {
                // mid-lats, biased into [-0.45, +0.45]
                let raw = hash01(seed.wrapping_add(200 + i as u32));
                (raw * 0.9) - 0.45
            };

            let cx = angle.cos() * r;
            let cz = lat_band * lat_extent;

            centers[i] = (cx, cz);
        }

        Self {
            macro_elev,
            hills,
            mountains,
            continent_noise,
            moisture_noise,
            warp_large,
            warp_small,
            continent_centers: centers,
            continent_radius,
            lat_extent,
            height_scale: 120.0,
            sea_level: 0.0,
        }
    }

    fn warped_coords(&self, wx: f32, wz: f32) -> (f64, f64) {
        let x = wx as f64;
        let z = wz as f64;

        let w1x = self.warp_large.get([x, z]) as f32;
        let w1z = self.warp_large.get([x + 1234.0, z - 5678.0]) as f32;

        let w2x = self.warp_small.get([x * 2.0, z * 2.0]) as f32;
        let w2z = self.warp_small.get([x * 2.0 - 500.0, z * 2.0 + 500.0]) as f32;

        let dx = (w1x * 0.8 + w2x * 0.2) * 90.0;
        let dz = (w1z * 0.8 + w2z * 0.2) * 90.0;

        (x + dx as f64, z + dz as f64)
    }

    // 6 supercontinents + noisy coasts + islands
    fn continental_mask(&self, wx: f32, wz: f32) -> f32 {
        let mut best = 0.0f32;
        for &(cx, cz) in &self.continent_centers {
            let dx = (wx - cx) / self.continent_radius;
            let dz = (wz - cz) / (self.continent_radius * 0.6);
            let dist = (dx * dx + dz * dz).sqrt();
            let v = (1.0 - dist).clamp(0.0, 1.0);
            let shaped = v * v * (3.0 - 2.0 * v);
            if shaped > best {
                best = shaped;
            }
        }

        let nx = wx as f64 * 0.00025;
        let nz = wz as f64 * 0.00025;
        let noise = self.continent_noise.get([nx, nz]) as f32 * 0.25;

        let mut c = (best + noise).clamp(0.0, 1.0);

        // random island chains out in the ocean
        let island_raw = self.continent_noise.get([nx * 3.0, nz * 3.0]) as f32;
        let island_v = (island_raw + 1.0) * 0.5;
        let island = smoothstep(0.80, 0.95, island_v) * (1.0 - c);
        c += island * 0.45;

        c.clamp(0.0, 1.0)
    }

    // 0 at equator (wz ~= 0), 1 at poles
    fn latitude_factor(&self, wz: f32) -> f32 {
        let t = (wz / self.lat_extent).abs();
        t.min(1.0)
    }

    pub fn height(&self, wx: f32, wz: f32) -> f32 {
        let hs = self.height_scale;

        let cont = self.continental_mask(wx, wz);

        let (wx2, wz2) = self.warped_coords(wx, wz);
        let x2 = wx2;
        let z2 = wz2;

        let macro_raw = self.macro_elev.get([x2 * 0.6, z2 * 0.6]) as f32;
        let macro_elev = macro_raw * 0.45;

        let hills_raw = self.hills.get([x2 * 2.0, z2 * 2.0]) as f32;
        let hills = hills_raw * 0.25;

        let m_raw = self.mountains.get([x2, z2]) as f32;
        let mut ridged = 1.0 - m_raw.abs();
        ridged = ridged.max(0.0);
        ridged = ridged * ridged * ridged;

        let belts_raw = self.mountains.get([x2 * 0.18 + 1234.0, z2 * 0.18 - 5678.0]) as f32;
        let belt_n = (belts_raw + 1.0) * 0.5;
        let belt_mask = smoothstep(0.55, 0.9, belt_n);

        let interior = smoothstep(0.45, 0.95, cont);
        let mountain_mask = belt_mask * interior;

        // base continental profile: deep basins -> inland plateaus
        let ocean_floor = -0.9;
        let inland_plateau = 0.35;
        let mut rel = ocean_floor + (inland_plateau - ocean_floor) * cont;

        // macro/hills detail mostly on land, not much on ocean floor
        rel += macro_elev * (0.3 + 0.7 * cont);
        rel += hills * cont;
        rel += ridged * mountain_mask * 1.3;

        // soften coastlines so they do not become razor cliffs
        let coast = (cont - 0.5).abs();
        let coast_t = ((0.28 - coast) / 0.28).clamp(0.0, 1.0);
        rel = rel * (1.0 - 0.15 * coast_t);

        rel * hs + self.sea_level
    }

    pub fn moisture(&self, wx: f32, wz: f32) -> f32 {
        let hs = self.height_scale;
        let h = self.height(wx, wz);
        let h_rel = (h - self.sea_level) / hs;

        let (wx2, wz2) = self.warped_coords(wx, wz);
        let n = self.moisture_noise.get([wx2, wz2]) as f32;
        let mut m = (n + 1.0) * 0.5;

        let cont = self.continental_mask(wx, wz);
        let lat = self.latitude_factor(wz);

        // zonal pattern: equator wet, subtropics drier, mid-lats moderate, poles dry
        let mut zonal = 1.0 - ((lat - 0.2) * 1.4).abs();
        zonal = zonal.clamp(0.0, 1.0);

        m = m * 0.4 + zonal * 0.6;

        // altitude dries
        let height_dry = h_rel.clamp(0.0, 1.4);
        m *= 1.0 - height_dry * 0.55;

        // continental interiors drier than coasts
        let interior = smoothstep(0.55, 0.9, cont);
        m *= 1.0 - interior * 0.45;

        m.clamp(0.0, 1.0)
    }

    // new: needs wx, wz for latitude-based winter biomes
    pub fn color(&self, wx: f32, wz: f32, h: f32, moisture: f32) -> [f32; 3] {
        let hs = self.height_scale;
        let h_rel = h - self.sea_level;
        let h_norm = (h_rel / hs).clamp(-1.0, 1.5);
        let lat = self.latitude_factor(wz);

        let (wx2, wz2) = self.warped_coords(wx, wz);

        // temperature from latitude + altitude
        let mut temp_lat = 1.0 - lat * lat; // 1 hot equator, 0 cold pole
        let mut temp = temp_lat;

        let alt_cool = (h_norm.max(0.0) * 0.7).clamp(0.0, 0.9);
        temp *= 1.0 - alt_cool;

        // tiny noise so bands are not perfectly straight
        let t_noise = self.macro_elev.get([wx2 * 0.02, wz2 * 0.02]) as f32;
        temp += t_noise * 0.03;
        temp = temp.clamp(0.0, 1.0);

        let dry = (1.0 - moisture).clamp(0.0, 1.0);

        // water: depth + latitude gradient, plus polar sea ice
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

            // sea ice near poles, close to surface
            let ice_lat = smoothstep(0.78, 0.98, lat);
            let ice_depth = smoothstep(-5.0, 3.0, h_rel);
            let ice = (ice_lat * ice_depth).clamp(0.0, 1.0);
            let ice_color = [0.94, 0.97, 1.0];
            color = lerp_color(color, ice_color, ice);

            return color;
        }

        let alt = h_norm.clamp(0.0, 1.4);

        // smooth altitude weights: lowlands / midlands / highlands
        let w_low = tri_weight(alt, 0.15, 0.35);
        let w_mid = tri_weight(alt, 0.55, 0.40);
        let w_high = tri_weight(alt, 1.05, 0.55);
        let sum = (w_low + w_mid + w_high).max(0.0001);
        let w_low = w_low / sum;
        let w_mid = w_mid / sum;
        let w_high = w_high / sum;

        // palettes per altitude band (colors are just anchors, everything is interpolated)

        // lowlands: deserts, savannas, grasslands, rainforests, tundra
        let low_cold_dry = [0.72, 0.76, 0.70]; // tundra-ish
        let low_cold_wet = [0.16, 0.44, 0.26]; // cool forest
        let low_hot_dry = [0.91, 0.83, 0.55]; // desert
        let low_hot_wet = [0.07, 0.45, 0.18]; // rainforest

        // midlands: taiga, shrubland, montane forests, dry plateaus
        let mid_cold_dry = [0.62, 0.64, 0.62]; // rocky shrub
        let mid_cold_wet = [0.15, 0.40, 0.27]; // taiga
        let mid_hot_dry = [0.80, 0.76, 0.52]; // dry plateau / savanna
        let mid_hot_wet = [0.18, 0.50, 0.26]; // montane forest

        // highlands: rock vs snow
        let high_cold_dry = [0.93, 0.95, 0.97]; // bright snow
        let high_cold_wet = [0.96, 0.98, 1.0]; // fresh snow
        let high_hot_dry = [0.64, 0.64, 0.66]; // bare rock
        let high_hot_wet = [0.58, 0.60, 0.64]; // wet rock / glacier ice

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

        // blend altitude bands
        let mut color = [
            c_low[0] * w_low + c_mid[0] * w_mid + c_high[0] * w_high,
            c_low[1] * w_low + c_mid[1] * w_mid + c_high[1] * w_high,
            c_low[2] * w_low + c_mid[2] * w_mid + c_high[2] * w_high,
        ];

        // polar snow on land, even at low altitude → Antarctica / Arctic
        let snow_lat = smoothstep(0.80, 0.98, lat);
        let snow_alt = smoothstep(0.35, 1.2, alt);
        let snow = (snow_lat * 0.8 + snow_alt * 0.6).clamp(0.0, 1.0);
        let snow_color = [0.96, 0.97, 0.99];
        color = lerp_color(color, snow_color, snow);

        // gentle lightening with altitude
        let light = 0.97 + alt * 0.08;
        color[0] = (color[0] * light).min(1.0);
        color[1] = (color[1] * light).min(1.0);
        color[2] = (color[2] * light).min(1.0);

        // tiny variation to break up large uniform fields
        let detail_noise = self.hills.get([wx2 * 3.0, wz2 * 3.0]) as f32;
        let tint = 1.0 + detail_noise * 0.03;
        color[0] = (color[0] * tint).clamp(0.0, 1.0);
        color[1] = (color[1] * tint).clamp(0.0, 1.0);
        color[2] = (color[2] * tint).clamp(0.0, 1.0);

        color
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp_color(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn tri_weight(x: f32, center: f32, width: f32) -> f32 {
    let d = (x - center).abs();
    if d >= width { 0.0 } else { 1.0 - d / width }
}
