use crate::chunk_builder::{ChunkBuilder, ChunkMesh};
use crate::renderer::pipelines::Pipelines;
use crate::terrain::TerrainGenerator;
use glam::Vec3;
use std::collections::HashMap;
use wgpu::{Device, IndexFormat, RenderPass};

pub struct WorldRenderer {
    pub chunks: HashMap<(i32, i32), ChunkMesh>,
    pub terrain_gen: TerrainGenerator,
    pub chunk_size: i32,
    pub view_radius_generate: i32,
    pub view_radius_render: i32,
    pub max_new_chunks_per_frame: usize,
}

impl WorldRenderer {
    pub fn new(device: &Device) -> Self {
        let terrain_gen = TerrainGenerator::new(0);
        let chunk_size = 32;
        let view_radius_render = 64;
        let view_radius_generate = 10;
        let mut chunks = HashMap::new();
        let origin = (0, 0);

        let origin_chunk =
            ChunkBuilder::build_chunk(device, origin.0, origin.1, chunk_size as u32, &terrain_gen);
        chunks.insert(origin, origin_chunk);

        Self {
            chunks,
            terrain_gen,
            chunk_size,
            view_radius_generate,
            view_radius_render,
            max_new_chunks_per_frame: 2,
        }
    }

    pub fn update(&mut self, device: &Device, camera_pos: Vec3) {
        let cs = self.chunk_size as f32;
        let cam_cx = (camera_pos.x / cs).floor() as i32;
        let cam_cz = (camera_pos.z / cs).floor() as i32;

        let mut created = 0usize;

        for dx in -self.view_radius_generate..=self.view_radius_generate {
            for dz in -self.view_radius_generate..=self.view_radius_generate {
                if created >= self.max_new_chunks_per_frame {
                    return;
                }

                let coord = (cam_cx + dx, cam_cz + dz);
                if self.chunks.contains_key(&coord) {
                    continue;
                }

                let (cx, cz) = coord;
                let chunk = ChunkBuilder::build_chunk(
                    device,
                    cx,
                    cz,
                    self.chunk_size as u32,
                    &self.terrain_gen,
                );
                self.chunks.insert(coord, chunk);
                created += 1;
            }
        }
    }

    pub fn render<'a>(
        &'a self,
        pass: &mut RenderPass<'a>,
        pipelines: &'a Pipelines,
        camera_pos: Vec3,
    ) {
        pass.set_pipeline(&pipelines.pipeline);
        pass.set_bind_group(0, &pipelines.uniform_bind_group, &[]);

        let cs = self.chunk_size as f32;
        let cam_cx = (camera_pos.x / cs).floor() as i32;
        let cam_cz = (camera_pos.z / cs).floor() as i32;
        let r2 = (self.view_radius_render * self.view_radius_render) as i32;

        for ((cx, cz), chunk) in self.chunks.iter() {
            let dx = cx - cam_cx;
            let dz = cz - cam_cz;
            if dx * dx + dz * dz > r2 {
                continue;
            }

            pass.set_vertex_buffer(0, chunk.mesh.vertex_buf.slice(..));
            pass.set_index_buffer(chunk.mesh.index_buf.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..chunk.mesh.index_count, 0, 0..1);
        }
    }
}
