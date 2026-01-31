#![allow(dead_code)]
use std::mem::size_of;

use wgpu::{Buffer, BufferUsages, Device, Queue};

use crate::terrain::chunk_builder::GpuChunkHandle;
use crate::ui::vertex::VertexWithPosition;

/// Represents a free byte range inside a GPU buffer
#[derive(Clone, Copy, Debug)]
pub struct FreeRange {
    pub start: u64,
    pub size: u64,
}

/// Scratch buffer used during mesh upload
pub struct GeometryScratch<V> {
    pub vertices: Vec<V>,
    pub indices: Vec<u32>,
}

impl<V> Default for GeometryScratch<V> {
    fn default() -> Self {
        Self {
            vertices: Vec::with_capacity(4096),
            indices: Vec::with_capacity(12000),
        }
    }
}

impl<V> GeometryScratch<V> {
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }
}

/// Internal allocation description
struct AllocationRequest {
    vertex_bytes: u64,
    vertex_align: u64,
    index_bytes: u64,
}

impl AllocationRequest {
    fn from_scratch<V>(scratch: &GeometryScratch<V>) -> Self {
        Self {
            vertex_bytes: scratch.vertices.len() as u64 * size_of::<V>() as u64,
            vertex_align: size_of::<V>() as u64,
            index_bytes: scratch.indices.len() as u64 * 4,
        }
    }
}

/// A single GPU page containing vertex and index buffers
pub struct MeshPage {
    pub vertex_buf: Buffer,
    pub index_buf: Buffer,

    pub vcap: u64,
    pub icap: u64,

    pub free_v: Vec<FreeRange>,
    pub free_i: Vec<FreeRange>,
}

impl MeshPage {
    fn try_allocate_and_upload<V: bytemuck::Pod>(
        &mut self,
        queue: &Queue,
        scratch: &GeometryScratch<V>,
        request: &AllocationRequest,
        page_index: usize,
    ) -> Option<GpuChunkHandle> {
        let v_off = find_fit(&self.free_v, request.vertex_bytes, request.vertex_align)?;
        let i_off = find_fit(&self.free_i, request.index_bytes, 4)?;

        commit_alloc(&mut self.free_v, v_off, request.vertex_bytes);
        commit_alloc(&mut self.free_i, i_off, request.index_bytes);

        queue.write_buffer(
            &self.vertex_buf,
            v_off,
            bytemuck::cast_slice(&scratch.vertices),
        );

        queue.write_buffer(
            &self.index_buf,
            i_off,
            bytemuck::cast_slice(&scratch.indices),
        );

        Some(GpuChunkHandle {
            page: page_index as u8,
            base_vertex: (v_off / request.vertex_align) as i32,
            first_index_above: (i_off / 4) as u32,
            index_count_above: scratch.indices.len() as u32,
            first_index_under: 0,
            index_count_under: 0,
            vertex_count: scratch.vertices.len() as u32,
        })
    }
}

/// General purpose mesh arena
///
/// This is NOT terrain specific.
/// It assumes no clipping and no semantic meaning of geometry.
pub struct GeneralMeshArena {
    pub pages: Vec<MeshPage>,
    pub page_v_bytes: u64,
    pub page_i_bytes: u64,
}

impl GeneralMeshArena {
    pub fn new(device: &Device, page_v_bytes: u64, page_i_bytes: u64) -> Self {
        let mut arena = Self {
            pages: Vec::new(),
            page_v_bytes,
            page_i_bytes,
        };
        arena.add_page(device);
        arena
    }

    fn add_page(&mut self, device: &Device) {
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("general_mesh_v_page"),
            size: self.page_v_bytes,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("general_mesh_i_page"),
            size: self.page_i_bytes,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.pages.push(MeshPage {
            vertex_buf,
            index_buf,
            vcap: self.page_v_bytes,
            icap: self.page_i_bytes,
            free_v: vec![FreeRange {
                start: 0,
                size: self.page_v_bytes,
            }],
            free_i: vec![FreeRange {
                start: 0,
                size: self.page_i_bytes,
            }],
        });
    }

    pub fn alloc_and_upload<V: bytemuck::Pod + VertexWithPosition + Copy>(
        &mut self,
        device: &Device,
        queue: &Queue,
        vertices: &[V],
        indices: &[u32],
        scratch: &mut GeometryScratch<V>,
    ) -> GpuChunkHandle {
        scratch.clear();
        scratch.vertices.extend_from_slice(vertices);
        scratch.indices.extend_from_slice(indices);

        let request = AllocationRequest::from_scratch(scratch);

        loop {
            for (page_index, page) in self.pages.iter_mut().enumerate() {
                if let Some(handle) =
                    page.try_allocate_and_upload(queue, scratch, &request, page_index)
                {
                    return handle;
                }
            }
            self.add_page(device);
        }
    }

    pub fn free<V>(&mut self, handle: GpuChunkHandle) {
        let page = &mut self.pages[handle.page as usize];
        let stride = size_of::<V>() as u64;

        let v_off = handle.base_vertex as u64 * stride;
        let v_bytes = handle.vertex_count as u64 * stride;
        free_insert_and_coalesce(&mut page.free_v, v_off, v_bytes);

        if handle.index_count_above > 0 {
            let i_off = handle.first_index_above as u64 * 4;
            let i_bytes = handle.index_count_above as u64 * 4;
            free_insert_and_coalesce(&mut page.free_i, i_off, i_bytes);
        }
    }
}

/* =========================
Allocation helpers
========================= */

fn align_up(x: u64, a: u64) -> u64 {
    if a == 0 {
        return x;
    }
    ((x + a - 1) / a) * a
}

pub fn find_fit(free: &[FreeRange], size: u64, align: u64) -> Option<u64> {
    for r in free {
        let aligned = align_up(r.start, align);
        let pad = aligned - r.start;
        if pad + size <= r.size {
            return Some(aligned);
        }
    }
    None
}

pub fn commit_alloc(free: &mut Vec<FreeRange>, start: u64, size: u64) {
    for i in 0..free.len() {
        let r = free[i];
        if start >= r.start && start + size <= r.start + r.size {
            let mut new_ranges = Vec::with_capacity(2);

            if start > r.start {
                new_ranges.push(FreeRange {
                    start: r.start,
                    size: start - r.start,
                });
            }

            let end = start + size;
            let r_end = r.start + r.size;
            if end < r_end {
                new_ranges.push(FreeRange {
                    start: end,
                    size: r_end - end,
                });
            }

            free.swap_remove(i);
            free.extend(new_ranges);
            return;
        }
    }

    unreachable!("commit_alloc with invalid range");
}

pub fn free_insert_and_coalesce(free: &mut Vec<FreeRange>, start: u64, size: u64) {
    free.push(FreeRange { start, size });
    free.sort_by_key(|r| r.start);

    let mut out: Vec<FreeRange> = Vec::with_capacity(free.len());
    for r in free.drain(..) {
        if let Some(last) = out.last_mut() {
            if last.start + last.size == r.start {
                last.size += r.size;
                continue;
            }
        }
        out.push(r);
    }
    *free = out;
}
