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

#[derive(Clone)]
pub struct CpuChunkMesh {
    pub cx: i32,
    pub cz: i32,
    pub step: usize,
    pub version: u64,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl ChunkBuilder {
    pub fn build_chunk_cpu(
        chunk_x: i32,
        chunk_z: i32,
        size: u32,
        step: usize,
        _ns_x_neg: usize,
        _ns_x_pos: usize,
        _ns_z_neg: usize,
        _ns_z_pos: usize,
        version: u64,
        terrain_gen: &TerrainGenerator,
    ) -> CpuChunkMesh {
        let size = size as usize;
        let base_x = chunk_x as f32 * size as f32;
        let base_z = chunk_z as f32 * size as f32;
        let stepf = step as f32;

        let verts_x = size / step + 1;
        let verts_z = size / step + 1;

        let mut vertices = Vec::with_capacity(verts_x * verts_z);
        let mut heights = vec![0.0; verts_x * verts_z];

        for gx in 0..verts_x {
            for gz in 0..verts_z {
                let wx = base_x + (gx * step) as f32;
                let wz = base_z + (gz * step) as f32;

                let h = terrain_gen.height(wx, wz);
                let m = if step >= 8 {
                    0.5
                } else {
                    terrain_gen.moisture(wx, wz)
                };
                let col = terrain_gen.color(wx, wz, h, m);

                heights[gx * verts_z + gz] = h;

                vertices.push(Vertex {
                    position: [wx, h, wz],
                    normal: [0.0, 1.0, 0.0],
                    color: col,
                });
            }
        }

        let inv = 1.0 / stepf;
        for gx in 0..verts_x {
            for gz in 0..verts_z {
                let xm = gx.saturating_sub(1);
                let xp = (gx + 1).min(verts_x - 1);
                let zm = gz.saturating_sub(1);
                let zp = (gz + 1).min(verts_z - 1);

                let h_l = heights[xm * verts_z + gz];
                let h_r = heights[xp * verts_z + gz];
                let h_d = heights[gx * verts_z + zm];
                let h_u = heights[gx * verts_z + zp];

                let dhdx = (h_r - h_l) * 0.5 * inv;
                let dhdz = (h_u - h_d) * 0.5 * inv;

                let n = glam::Vec3::new(-dhdx, 1.0, -dhdz).normalize();
                vertices[gx * verts_z + gz].normal = [n.x, n.y, n.z];
            }
        }

        let mut indices = Vec::new();
        for gx in 0..verts_x - 1 {
            for gz in 0..verts_z - 1 {
                let i0 = (gx * verts_z + gz) as u32;
                let i1 = ((gx + 1) * verts_z + gz) as u32;
                let i2 = (gx * verts_z + gz + 1) as u32;
                let i3 = ((gx + 1) * verts_z + gz + 1) as u32;
                indices.extend_from_slice(&[i0, i1, i2, i2, i1, i3]);
            }
        }

        CpuChunkMesh {
            cx: chunk_x,
            cz: chunk_z,
            step,
            version,
            vertices,
            indices,
        }
    }
}

pub fn lod_step_for_distance(dist2: i32) -> usize {
    if dist2 < 4 {
        1
    } else if dist2 < 6 {
        2
    } else if dist2 < 8 {
        4
    } else if dist2 < 12 {
        8
    } else if dist2 < 16 {
        16
    } else {
        32
    }
}

pub fn generate_spiral_offsets(radius: i32) -> Vec<(i32, i32)> {
    let mut v = Vec::new();
    for dx in -radius..=radius {
        for dz in -radius..=radius {
            v.push((dx, dz));
        }
    }

    // sort by distance from center
    v.sort_by_key(|(dx, dz)| dx * dx + dz * dz);
    v
}

fn normalize_edge_order(edge: &mut Vec<usize>, vertices: &Vec<Vertex>, axis: char) {
    match axis {
        'x' => {
            // sort by world X
            edge.sort_by(|&a, &b| {
                let ax = vertices[a].position[0];
                let bx = vertices[b].position[0];
                ax.partial_cmp(&bx).unwrap()
            });
        }
        'z' => {
            // sort by world Z
            edge.sort_by(|&a, &b| {
                let az = vertices[a].position[2];
                let bz = vertices[b].position[2];
                az.partial_cmp(&bz).unwrap()
            });
        }
        _ => {}
    }
}
