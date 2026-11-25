#![allow(dead_code, unused_variables, unused_assignments)]
use crate::helpers::hsv::{HSV, hsv_to_rgb, lerp_hsv};
use crate::world::props::props::Props;
use crate::world::terrain::chunk_builder::TreeSpawner;
use glam::DVec2;
use noisy_skylines::compiler::GenWorkspace;
use noisy_skylines::presets::CompiledPreset;
use rand_distr::num_traits::real::Real;
use rusty_skylines_utils::positions::{ChunkCoord, WorldPos, chunk_size};
use std::f32::consts::PI;
use tracing::error;
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


// ─────────────────────────────────────────────────────────────────────────────
// TerrainParams
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct TerrainParams {
    pub seed: u32,
    pub world_scale: f32,

    pub height_scale: f32,
    pub sea_level: f32
}

impl Default for TerrainParams {
    fn default() -> Self {
        Self {
            seed: 0,
            world_scale: 1.0,

            height_scale: 30.0,
            sea_level: 0.0,

        }
    }
}

#[derive(Clone, Copy)]
struct BaseSample {
    rel: f32
}

#[derive(Clone)]
pub struct TerrainGenerator {
    p: TerrainParams,
    pub tree_spawner: TreeSpawner,
    pub preset: CompiledPreset
}

impl TerrainGenerator {
    pub fn new(terrain_params: TerrainParams, props: &Props, preset: CompiledPreset) -> Self {
        Self::with_params(terrain_params, Some(props), None, preset)
    }

    pub fn with_params(mut p: TerrainParams, props: Option<&Props>, tree_spawner: Option<TreeSpawner>, preset: CompiledPreset) -> Self {
        let seed = p.seed;
        p.world_scale = p.world_scale.max(0.000001);

        let tree_spawner = tree_spawner.unwrap_or_else(|| {
            TreeSpawner::new(props.expect("Props required when creating a new TreeSpawner"))
        });

        Self {
            p,
            tree_spawner,
            preset
        }
    }
    // pub fn from_terrain_gen(terrain_gen: &TerrainGeneratorpreset: CompiledPreset) -> TerrainGenerator {
    //     TerrainGenerator::with_params(terrain_gen.p, None, Some(terrain_gen.tree_spawner.clone()))
    // }
    #[inline]
    fn scaled_coords_f64(&self, p: &WorldPos) -> (f64, f64) {
        let (wx, wz) = world_xz_f64(p);
        let s = self.p.world_scale as f64;
        (wx * s, wz * s)
    }

    //
    // #[inline]
    // fn flatten_profile(&self, rel: f32, cont: f32) -> f32 {
    //     let f = self.p.flatten.clamp(0.0, 1.0);
    //     if f <= 0.0001 {
    //         return rel;
    //     }
    //
    //     let sign = rel.signum();
    //     let a = rel.abs();
    //
    //     let c = self.p.flatten_curve.max(1.0);
    //     let k = 1.25 * f;
    //     let a2 = a / (1.0 + k * a.powf(c));
    //
    //     let land = smoothstep(0.25, 0.65, cont);
    //     let mix = f * (0.30 + 0.70 * land);
    //     lerp(rel, sign * a2, mix)
    // }
    pub fn heights(&self, points: &[DVec2]) -> Vec<f32> {
        let result = GenWorkspace::execute(&self.preset, points, None);
        match result {
            Ok((heights, _)) => heights.into_iter().map(|h| self.change_height(h)).collect(),
            Err(err) => {
                error!("[Terrain] IMPORTANT ERROR! Error while calculating heights: {:?}", err);
                Vec::new()
            }
        }
    }
    pub fn height(&self, point: DVec2) -> f32 {
        let points = vec![point];
        let result = GenWorkspace::execute(&self.preset, points.as_slice(), None);
        match result {
            Ok((heights, _)) => self.change_height(heights[0]),
            Err(err) => {
                error!("[Terrain] IMPORTANT ERROR! Error while calculating one height: {:?}", err);
                0.0
            }
        }
    }
    fn change_height(&self, h: f32) -> f32 {
        (h + self.p.sea_level) * self.p.height_scale
    }
    pub fn moisture(&self, p: &WorldPos, h: f32) -> f32 {
        1.0
    }

    pub fn color(&self, h: f32, moisture: f32) -> [f32; 3] {
        match h {
            -1.0..=1.0 => [1.0, 1.0, 0.0],
            -100.0..=-1.0 => [0.0, 0.0, 1.0],
            400.0..=1000.0 => [1.0, 1.0, 1.0],
            1.0..=200.0 => [0.0, 0.8, 0.0],
            200.0..=400.0 => [0.3, 0.8, 0.3],
            _ => [0.0, 1.0, 0.0]
        }
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

    // pub fn make_texture(
    //     &self,
    //     texture_size: Extent3d,
    //     chunk_coord: ChunkCoord,
    //     size: u32,
    // ) -> TerrainTextures {
    //     let size = size.max(1);
    //     let tex_w = texture_size.width as usize;
    //     let tex_h = texture_size.height as usize;
    //
    //     let cs = chunk_size() as f64;
    //     let start_wx = chunk_coord.x as f64 * cs;
    //     let start_wz = chunk_coord.z as f64 * cs;
    //     let world_span = cs * size as f64;
    //
    //     // ─────────────────────────────────────────────
    //     // PASS 1: gather heights + find min/max
    //     // ─────────────────────────────────────────────
    //     let mut points = Vec::with_capacity(size as usize);
    //     let mut heights = vec![0.0f32; tex_w * tex_h];
    //     let mut min_h = f32::MAX;
    //     let mut max_h = f32::MIN;
    //
    //     for py in 0..tex_h {
    //         let vz = (py as f64 + 0.5) / tex_h as f64;
    //         let wz = start_wz + vz * world_span;
    //
    //         for px in 0..tex_w {
    //             let vx = (px as f64 + 0.5) / tex_w as f64;
    //             let wx = start_wx + vx * world_span;
    //
    //             let h = self.heights();
    //
    //             heights[py * tex_w + px] = h;
    //
    //             min_h = min_h.min(h);
    //             max_h = max_h.max(h);
    //         }
    //     }
    //
    //     let inv_range = if max_h > min_h {
    //         1.0 / (max_h - min_h)
    //     } else {
    //         1.0
    //     };
    //
    //     // ─────────────────────────────────────────────
    //     // PASS 2: encode grayscale
    //     // ─────────────────────────────────────────────
    //     let mut height_data = Vec::with_capacity(tex_w * tex_h * 4);
    //     let mut color_data = Vec::with_capacity(tex_w * tex_h * 4);
    //
    //     for i in 0..heights.len() {
    //         let h = heights[i];
    //
    //         let t = ((h - min_h) * inv_range).clamp(0.0, 1.0);
    //
    //         // pure grayscale debug ramp
    //         let v = (t * 255.0 + 0.5) as u8;
    //
    //         height_data.push(v);
    //         height_data.push(v);
    //         height_data.push(v);
    //         height_data.push(255);
    //
    //         // still your biome color
    //         let wx = 0.0; // not used here anymore
    //         let wz = 0.0;
    //         let color_rgb = [t, t, t]; // optional fallback if you want speed debug
    //         color_data.push((color_rgb[0] * 255.0) as u8);
    //         color_data.push((color_rgb[1] * 255.0) as u8);
    //         color_data.push((color_rgb[2] * 255.0) as u8);
    //         color_data.push(255);
    //     }
    //
    //     TerrainTextures {
    //         height: TextureBuffer {
    //             width: tex_w as u32,
    //             height: tex_h as u32,
    //             data: height_data,
    //         },
    //         color: TextureBuffer {
    //             width: tex_w as u32,
    //             height: tex_h as u32,
    //             data: color_data,
    //         },
    //     }
    // }
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
    use rusty_skylines_utils::positions::LocalPos;

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
