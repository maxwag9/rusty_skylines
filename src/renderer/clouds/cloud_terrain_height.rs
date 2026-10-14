use crate::helpers::positions::{LocalPos, WorldPos, chunk_size};
use crate::world::terrain::terrain_subsystem::Terrain;
use wgpu::{
    Device, Extent3d, Origin3d, Queue, TexelCopyBufferLayout, TexelCopyTextureInfo, Texture,
    TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor,
};

pub const CLOUD_TERRAIN_MAP_SIZE: u32 = 128;
pub const CLOUD_TERRAIN_MAP_WORLD_SIZE: f32 = 32768.0;
const ROWS_PER_UPDATE: u32 = 8;
const REBUILD_DISTANCE: f32 = 8192.0;

struct HeightTarget {
    texture: Texture,
    view: TextureView,
}

impl HeightTarget {
    fn new(device: &Device, label: &'static str) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: CLOUD_TERRAIN_MAP_SIZE,
                height: CLOUD_TERRAIN_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        Self { texture, view }
    }
}

pub struct CloudTerrainHeightMap {
    targets: [HeightTarget; 2],
    active_index: usize,
    active_center: Option<WorldPos>,
    pending_center: Option<WorldPos>,
    next_row: u32,
    pending_data: Vec<u16>,
    frames_since_refresh: u32,
}

impl CloudTerrainHeightMap {
    pub fn new(device: &Device) -> Self {
        Self {
            targets: [
                HeightTarget::new(device, "Cloud Terrain Height A"),
                HeightTarget::new(device, "Cloud Terrain Height B"),
            ],
            active_index: 0,
            active_center: None,
            pending_center: None,
            next_row: 0,
            pending_data: vec![0; (CLOUD_TERRAIN_MAP_SIZE * CLOUD_TERRAIN_MAP_SIZE) as usize],
            frames_since_refresh: 0,
        }
    }

    fn f32_to_f16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let raw_exp = ((bits >> 23) & 0xff) as i32;
        let mantissa = bits & 0x7fffff;

        if raw_exp == 0xff {
            return sign | 0x7c00 | if mantissa != 0 { 0x0200 } else { 0 };
        }

        let mut exponent = raw_exp - 127 + 15;
        if exponent >= 31 {
            return sign | 0x7c00;
        }

        if exponent <= 0 {
            if exponent < -10 {
                return sign;
            }
            let significand = mantissa | 0x800000;
            let shift = (14 - exponent) as u32;
            let rounded = significand + (1u32 << (shift - 1));
            return sign | (rounded >> shift) as u16;
        }

        let rounded = mantissa + 0x1000;
        if rounded & 0x800000 != 0 {
            exponent += 1;
            if exponent >= 31 {
                return sign | 0x7c00;
            }
            return sign | ((exponent as u16) << 10);
        }

        sign | ((exponent as u16) << 10) | ((rounded >> 13) as u16 & 0x03ff)
    }

    fn world_delta_xz(a: WorldPos, b: WorldPos) -> [f32; 2] {
        let size = chunk_size() as f64;
        [
            (((a.chunk.x as f64 - b.chunk.x as f64) * size) + a.local.x as f64 - b.local.x as f64)
                as f32,
            (((a.chunk.z as f64 - b.chunk.z as f64) * size) + a.local.z as f64 - b.local.z as f64)
                as f32,
        ]
    }

    pub fn update(&mut self, queue: &Queue, terrain: &Terrain, camera_world: WorldPos) {
        self.frames_since_refresh = self.frames_since_refresh.saturating_add(1);

        if self.pending_center.is_none() {
            let should_build = match self.active_center {
                None => true,
                Some(center) => {
                    let delta = Self::world_delta_xz(camera_world, center);
                    delta[0] * delta[0] + delta[1] * delta[1] >= REBUILD_DISTANCE * REBUILD_DISTANCE
                        || self.frames_since_refresh >= 600
                }
            };

            if should_build {
                self.pending_center = Some(camera_world);
                self.next_row = 0;
                self.frames_since_refresh = 0;
            }
        }

        let Some(center) = self.pending_center else {
            return;
        };

        let start_row = self.next_row;
        let end_row = (start_row + ROWS_PER_UPDATE).min(CLOUD_TERRAIN_MAP_SIZE);
        let size = CLOUD_TERRAIN_MAP_SIZE as usize;
        let cell_size = CLOUD_TERRAIN_MAP_WORLD_SIZE / CLOUD_TERRAIN_MAP_SIZE as f32;
        let half_world = CLOUD_TERRAIN_MAP_WORLD_SIZE * 0.5;

        for z in start_row..end_row {
            let relative_z = (z as f32 + 0.5) * cell_size - half_world;
            for x in 0..CLOUD_TERRAIN_MAP_SIZE {
                let relative_x = (x as f32 + 0.5) * cell_size - half_world;
                let position = WorldPos::new(
                    center.chunk,
                    LocalPos::new(
                        center.local.x + relative_x,
                        0.0,
                        center.local.z + relative_z,
                    ),
                )
                .normalize();

                self.pending_data[z as usize * size + x as usize] =
                    Self::f32_to_f16_bits(terrain.get_height_at(position, true));
            }
        }

        let rows = end_row - start_row;
        if rows > 0 {
            let write_index = 1 - self.active_index;
            let first = start_row as usize * size;
            let last = end_row as usize * size;
            queue.write_texture(
                TexelCopyTextureInfo {
                    texture: &self.targets[write_index].texture,
                    mip_level: 0,
                    origin: Origin3d {
                        x: 0,
                        y: start_row,
                        z: 0,
                    },
                    aspect: TextureAspect::All,
                },
                bytemuck::cast_slice(&self.pending_data[first..last]),
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(CLOUD_TERRAIN_MAP_SIZE * 2),
                    rows_per_image: Some(rows),
                },
                Extent3d {
                    width: CLOUD_TERRAIN_MAP_SIZE,
                    height: rows,
                    depth_or_array_layers: 1,
                },
            );
        }

        self.next_row = end_row;
        if self.next_row >= CLOUD_TERRAIN_MAP_SIZE {
            self.active_index = 1 - self.active_index;
            self.active_center = Some(center);
            self.pending_center = None;
            self.next_row = 0;
        }
    }

    pub fn view(&self) -> &TextureView {
        &self.targets[self.active_index].view
    }

    pub fn origin_relative_to(&self, camera_world: WorldPos) -> ([f32; 2], bool) {
        match self.active_center {
            Some(center) => (Self::world_delta_xz(camera_world, center), true),
            None => ([0.0, 0.0], false),
        }
    }
}
