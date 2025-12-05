use crate::terrain::TerrainGenerator;
use crate::ui::vertex::Vertex;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferUsages, Device};

pub struct Mesh {
    pub vertex_buf: Buffer,
    pub index_buf: Buffer,
    pub index_count: u32,
}

pub struct ChunkMesh {
    pub x: i32,
    pub z: i32,
    pub mesh: Mesh,
}

pub struct ChunkBuilder;

impl ChunkBuilder {
    pub fn build_chunk(
        device: &Device,
        chunk_x: i32,
        chunk_z: i32,
        size: u32,
        terrain_gen: &TerrainGenerator,
    ) -> ChunkMesh {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let base_x = chunk_x as f32 * size as f32;
        let base_z = chunk_z as f32 * size as f32;

        for x in 0..size {
            for z in 0..size {
                let wx = base_x + x as f32;
                let wz = base_z + z as f32;

                let h00 = terrain_gen.height(wx, wz);
                let h10 = terrain_gen.height(wx + 1.0, wz);
                let h01 = terrain_gen.height(wx, wz + 1.0);
                let h11 = terrain_gen.height(wx + 1.0, wz + 1.0);

                let m = terrain_gen.moisture(wx, wz);
                let col = terrain_gen.color(h00, m);

                let i = vertices.len() as u32;

                vertices.push(Vertex {
                    position: [wx, h00, wz],
                    color: col,
                });
                vertices.push(Vertex {
                    position: [wx + 1.0, h10, wz],
                    color: col,
                });
                vertices.push(Vertex {
                    position: [wx, h01, wz + 1.0],
                    color: col,
                });
                vertices.push(Vertex {
                    position: [wx + 1.0, h11, wz + 1.0],
                    color: col,
                });

                indices.extend_from_slice(&[i, i + 1, i + 2, i + 2, i + 1, i + 3]);
            }
        }

        let vertex_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Chunk vertex buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: BufferUsages::VERTEX,
        });

        let index_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Chunk index buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: BufferUsages::INDEX,
        });

        ChunkMesh {
            x: chunk_x,
            z: chunk_z,
            mesh: Mesh {
                vertex_buf,
                index_buf,
                index_count: indices.len() as u32,
            },
        }
    }
}
