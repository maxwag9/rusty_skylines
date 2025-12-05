use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

pub struct TerrainGenerator {
    elevation: Fbm<Perlin>,
    continent: Fbm<Perlin>,
    moisture: Fbm<Perlin>,

    pub height_scale: f32,
    pub sea_level: f32,
}

impl TerrainGenerator {
    pub fn new(seed: u32) -> Self {
        Self {
            elevation: Fbm::<Perlin>::new(seed)
                .set_octaves(5)
                .set_frequency(0.03)
                .set_persistence(0.5),
            continent: Fbm::<Perlin>::new(seed + 1)
                .set_octaves(3)
                .set_frequency(0.005)
                .set_persistence(0.8),
            moisture: Fbm::<Perlin>::new(seed + 2)
                .set_octaves(4)
                .set_frequency(0.02)
                .set_persistence(0.6),
            height_scale: 40.0,
            sea_level: 0.0,
        }
    }

    pub fn height(&self, wx: f32, wz: f32) -> f32 {
        let e = self.elevation.get([wx as f64, wz as f64]) as f32;
        let c = self.continent.get([wx as f64, wz as f64]) as f32;

        let continent = ((c + 1.0) * 0.5).powf(1.4);
        let elevation = e * continent;

        elevation * self.height_scale
    }

    pub fn moisture(&self, wx: f32, wz: f32) -> f32 {
        let m = self.moisture.get([wx as f64, wz as f64]) as f32;
        (m + 1.0) * 0.5
    }

    pub fn color(&self, h: f32, moisture: f32) -> [f32; 3] {
        let sea = [0.02, 0.14, 0.34];
        let shallow = [0.05, 0.28, 0.44];
        let beach = [0.90, 0.82, 0.58];
        let grass = [0.20, 0.68, 0.24];
        let dry_grass = [0.64, 0.70, 0.32];
        let forest = [0.07, 0.40, 0.14];
        let rock = [0.55, 0.55, 0.60];
        let snow = [0.95, 0.96, 0.98];

        let h_rel = h - self.sea_level;

        if h_rel < -15.0 {
            return sea;
        } else if h_rel < -3.0 {
            return shallow;
        } else if h_rel < 1.0 {
            return beach;
        }

        if h_rel > 32.0 {
            return snow;
        } else if h_rel > 24.0 {
            return rock;
        }

        if h_rel > 8.0 {
            if moisture > 0.5 {
                return forest;
            } else {
                return dry_grass;
            }
        }

        if moisture > 0.6 { forest } else { grass }
    }
}
