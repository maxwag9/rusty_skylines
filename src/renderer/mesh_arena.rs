use crate::chunk_builder::GpuChunkHandle;
use wgpu::{Buffer, BufferUsages, Device, Queue};

pub struct MeshPage {
    pub vertex_buf: Buffer,
    pub index_buf: Buffer,
    pub vcap: u64,
    pub icap: u64,
    pub free_v: Vec<FreeRange>,
    pub free_i: Vec<FreeRange>,
}

pub struct MeshArena {
    pub pages: Vec<MeshPage>,
    pub page_v_bytes: u64,
    pub page_i_bytes: u64,
}

impl MeshArena {
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
            label: Some("terrain_v_page"),
            size: self.page_v_bytes,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain_i_page"),
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

    pub fn alloc_and_upload<V: bytemuck::Pod>(
        &mut self,
        device: &Device,
        queue: &Queue,
        vertices: &[V],
        indices: &[u32],
    ) -> GpuChunkHandle {
        let v_bytes = vertices.len() as u64 * std::mem::size_of::<V>() as u64;
        let i_bytes = indices.len() as u64 * 4;

        let v_align = std::mem::size_of::<V>() as u64;
        let i_align = 4u64;

        loop {
            for (pi, page) in self.pages.iter_mut().enumerate() {
                // 1) Probe WITHOUT mutating
                let v_off = find_fit(&page.free_v, v_bytes, v_align);
                let i_off = find_fit(&page.free_i, i_bytes, i_align);

                if let (Some(v_off), Some(i_off)) = (v_off, i_off) {
                    // 2) Commit allocations
                    commit_alloc(&mut page.free_v, v_off, v_bytes);
                    commit_alloc(&mut page.free_i, i_off, i_bytes);

                    // 3) Upload
                    queue.write_buffer(&page.vertex_buf, v_off, bytemuck::cast_slice(vertices));
                    queue.write_buffer(&page.index_buf, i_off, bytemuck::cast_slice(indices));

                    let base_vertex = (v_off / std::mem::size_of::<V>() as u64) as i32;
                    let first_index = (i_off / 4) as u32;

                    return GpuChunkHandle {
                        page: pi as u32,
                        base_vertex,
                        first_index,
                        index_count: indices.len() as u32,
                        vertex_count: vertices.len() as u32,
                    };
                }
            }

            // No page fits: add one and retry
            self.add_page(device);
        }
    }

    pub fn free<V>(&mut self, handle: GpuChunkHandle) {
        let page = &mut self.pages[handle.page as usize];

        let stride = std::mem::size_of::<V>() as u64;

        let v_off = handle.base_vertex as u64 * stride;
        let v_bytes = handle.vertex_count as u64 * stride;

        let i_off = handle.first_index as u64 * 4;
        let i_bytes = handle.index_count as u64 * 4;

        free_insert_and_coalesce(&mut page.free_v, v_off, v_bytes);
        free_insert_and_coalesce(&mut page.free_i, i_off, i_bytes);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FreeRange {
    start: u64, // bytes
    size: u64,  // bytes
}

fn align_up(x: u64, a: u64) -> u64 {
    if a == 0 {
        return x;
    }
    ((x + a - 1) / a) * a
}

fn free_insert_and_coalesce(free: &mut Vec<FreeRange>, start: u64, size: u64) {
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
fn find_fit(free: &[FreeRange], size: u64, align: u64) -> Option<u64> {
    for r in free {
        let aligned = align_up(r.start, align);
        let pad = aligned - r.start;
        if pad + size <= r.size {
            return Some(aligned);
        }
    }
    None
}
fn commit_alloc(free: &mut Vec<FreeRange>, start: u64, size: u64) {
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

    unreachable!("commit_alloc called with invalid range");
}
