use crate::terrain::TerrainGenerator;
use crate::threads::ChunkWorkerPool;
use crate::ui::vertex::Vertex;
use glam::Vec3;
use std::sync::atomic::AtomicU64;
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
        version_atomic: &AtomicU64,
        terrain_gen: &TerrainGenerator,
    ) -> Option<CpuChunkMesh> {
        let size = size as usize;
        let base_x = chunk_x as f32 * size as f32;
        let base_z = chunk_z as f32 * size as f32;
        let stepf = step as f32;

        let verts_x = size / step + 1;
        let verts_z = size / step + 1;

        let mut vertices = Vec::with_capacity(verts_x * verts_z);
        let mut heights = vec![0.0; verts_x * verts_z];

        let check_mask = match step {
            1 => 0b11,  // every 4 rows
            2 => 0b111, // every 8
            4 => 0b1111,
            8 => 0b1_1111,
            _ => 0b11_1111,
        };

        for gx in 0..verts_x {
            if gx & check_mask == 0 && !ChunkWorkerPool::still_current(version_atomic, version) {
                return None;
            }

            for gz in 0..verts_z {
                let wx = base_x + (gx * step) as f32;
                let wz = base_z + (gz * step) as f32;
                let cx = base_x + gx as f32;
                let cz = base_z + gz as f32;

                let h = terrain_gen.height(wx, wz);
                let m = if step >= 8 {
                    0.5
                } else {
                    terrain_gen.moisture(wx, wz)
                };
                let col = terrain_gen.color(wx, wz, h, m);

                heights[gx * verts_z + gz] = h;

                let h = terrain_gen.height(wx, wz);

                let hx = terrain_gen.height(wx + step as f32, wz);
                let hz = terrain_gen.height(wx, wz + step as f32);
                let dx = Vec3::new(step as f32, hx - h, 0.0);
                let dz = Vec3::new(0.0, hz - h, step as f32);
                let n = dx.cross(dz).normalize();

                vertices.push(Vertex {
                    position: [wx, h, wz],
                    normal: [n.x, n.y, n.z],
                    color: col,
                });
            }
        }

        let inv = 1.0 / stepf;
        for gx in 0..verts_x {
            if gx & check_mask == 0 && !ChunkWorkerPool::still_current(version_atomic, version) {
                return None;
            }

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
            if gx & check_mask == 0 && !ChunkWorkerPool::still_current(version_atomic, version) {
                return None;
            }

            for gz in 0..verts_z - 1 {
                let i0 = (gx * verts_z + gz) as u32;
                let i1 = ((gx + 1) * verts_z + gz) as u32;
                let i2 = (gx * verts_z + gz + 1) as u32;
                let i3 = ((gx + 1) * verts_z + gz + 1) as u32;
                indices.extend_from_slice(&[i0, i1, i2, i2, i1, i3]);
            }
        }

        Some(CpuChunkMesh {
            cx: chunk_x,
            cz: chunk_z,
            step,
            version,
            vertices,
            indices,
        })
    }
}

pub fn lod_step_for_distance(dist2: i32) -> usize {
    if dist2 < 8 {
        1
    } else if dist2 < 14 {
        2
    } else if dist2 < 20 {
        4
    } else if dist2 < 28 {
        8
    } else if dist2 < 42 {
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
