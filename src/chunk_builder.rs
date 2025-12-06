use crate::terrain::TerrainGenerator;
use crate::ui::vertex::Vertex;
use wgpu::util::DeviceExt;
use wgpu::{Buffer, Device};

pub struct GpuChunkMesh {
    pub vertex_buf: Buffer,
    pub index_buf: Buffer,
    pub index_count: u32,
}
impl GpuChunkMesh {
    pub fn from_cpu(device: &Device, cpu: &CpuChunkMesh) -> Self {
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Chunk vertex buffer"),
            contents: bytemuck::cast_slice(&cpu.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Chunk index buffer"),
            contents: bytemuck::cast_slice(&cpu.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buf,
            index_buf,
            index_count: cpu.indices.len() as u32,
        }
    }
}

#[derive(Clone)]
pub struct CpuChunkMesh {
    pub cx: i32,
    pub cz: i32,
    pub step: usize,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

pub struct ChunkMesh {
    pub x: i32,
    pub z: i32,
    pub mesh: GpuChunkMesh,
}

pub struct ChunkMeshLod {
    pub step: usize,
    pub mesh: GpuChunkMesh,
}

pub struct ChunkBuilder;

impl ChunkBuilder {
    pub fn build_chunk_cpu(
        chunk_x: i32,
        chunk_z: i32,
        size: u32,
        step: usize,
        terrain_gen: &TerrainGenerator,
    ) -> CpuChunkMesh {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let base_x = chunk_x as f32 * size as f32;
        let base_z = chunk_z as f32 * size as f32;

        let size_usize = size as usize;

        for x in (0..size_usize).step_by(step) {
            for z in (0..size_usize).step_by(step) {
                // stay inside the chunk bounds
                if x + step > size_usize || z + step > size_usize {
                    continue;
                }

                let wx = base_x + x as f32;
                let wz = base_z + z as f32;

                let h00 = terrain_gen.height(wx, wz);
                let h10 = terrain_gen.height(wx + step as f32, wz);
                let h01 = terrain_gen.height(wx, wz + step as f32);
                let h11 = terrain_gen.height(wx + step as f32, wz + step as f32);

                let m = terrain_gen.moisture(wx, wz);
                let col = terrain_gen.color(h00, m);

                let i = vertices.len() as u32;

                vertices.push(Vertex {
                    position: [wx, h00, wz],
                    color: col,
                });
                vertices.push(Vertex {
                    position: [wx + step as f32, h10, wz],
                    color: col,
                });
                vertices.push(Vertex {
                    position: [wx, h01, wz + step as f32],
                    color: col,
                });
                vertices.push(Vertex {
                    position: [wx + step as f32, h11, wz + step as f32],
                    color: col,
                });

                indices.extend_from_slice(&[i, i + 1, i + 2, i + 2, i + 1, i + 3]);
            }
        }

        CpuChunkMesh {
            cx: chunk_x,
            cz: chunk_z,
            step,
            vertices,
            indices,
        }
    }
}

pub fn lod_step_for_distance(dist2: i32) -> usize {
    if dist2 < 8 {
        1
    } else if dist2 < 32 {
        2
    } else if dist2 < 128 {
        4
    } else {
        8
    }
}
