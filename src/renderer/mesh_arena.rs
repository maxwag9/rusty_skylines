use crate::terrain::chunk_builder::GpuChunkHandle;
use crate::ui::vertex::VertexWithPosition;
use wgpu::{Buffer, BufferUsages, Device, Queue};

pub struct GeometryScratch<V> {
    pub new_vertices: Vec<V>,
    pub indices_above: Vec<u32>,
    pub indices_under: Vec<u32>,
}

impl<V> Default for GeometryScratch<V> {
    fn default() -> Self {
        Self {
            new_vertices: Vec::with_capacity(4096),
            indices_above: Vec::with_capacity(12000),
            indices_under: Vec::with_capacity(12000),
        }
    }
}

impl<V> GeometryScratch<V> {
    pub fn clear(&mut self) {
        self.new_vertices.clear();
        self.indices_above.clear();
        self.indices_under.clear();
    }
}

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

    pub fn alloc_and_upload<V: bytemuck::Pod + VertexWithPosition + Clone + Copy>(
        &mut self,
        device: &Device,
        queue: &Queue,
        vertices: &[V],
        indices: &[u32], // Changed to slice
        scratch: &mut GeometryScratch<V>,
    ) -> GpuChunkHandle {
        scratch.clear();

        // 1) Fast Copy: Copy base vertices to scratch
        //    We avoid 'to_vec()' which does a raw allocator call.
        //    'extend_from_slice' into a reserved capacity is much faster.
        scratch.new_vertices.extend_from_slice(vertices);

        // 2) Optimized Clipping
        // Define closure to capture scratch context
        let make_intersection = |i: usize, j: usize, new_verts: &mut Vec<V>| -> u32 {
            // Standard lerp logic...
            let vi = new_verts[i]; // Copy, V is Pod+Copy
            let vj = new_verts[j];
            let yi = vi.position()[1];
            let yj = vj.position()[1];

            let t = if (yj - yi).abs() < f32::EPSILON {
                0.5
            } else {
                (0.0 - yi) / (yj - yi)
            };
            let new_v = V::lerp(&vi, &vj, t);
            new_verts.push(new_v);
            (new_verts.len() - 1) as u32
        };

        // Process triangles
        for tri in indices.chunks(3) {
            let a = tri[0] as usize;
            let b = tri[1] as usize;
            let c = tri[2] as usize;

            // read y positions first (no long-lived immutable borrow)
            let ya = scratch.new_vertices[a].position()[1];
            let yb = scratch.new_vertices[b].position()[1];
            let yc = scratch.new_vertices[c].position()[1];

            let above_a = ya >= 0.0;
            let above_b = yb >= 0.0;
            let above_c = yc >= 0.0;

            let above_count = (above_a as u8 + above_b as u8 + above_c as u8) as usize;

            match above_count {
                0 => {
                    // all under
                    scratch.indices_under.extend_from_slice(tri);
                }
                3 => {
                    // all above
                    scratch.indices_above.extend_from_slice(tri);
                }
                1 => {
                    // one vertex above, two below
                    // ia is the single above vertex, ib & ic are below
                    let (ia, ib, ic) = if above_a {
                        (a, b, c)
                    } else if above_b {
                        (b, c, a)
                    } else {
                        (c, a, b)
                    };

                    // intersections on edges ia-ib and ia-ic
                    let i_ab = make_intersection(ia, ib, &mut scratch.new_vertices);
                    let i_ac = make_intersection(ia, ic, &mut scratch.new_vertices);

                    // Above side: single triangle (ia, i_ab, i_ac)
                    scratch.indices_above.push(ia as u32);
                    scratch.indices_above.push(i_ab);
                    scratch.indices_above.push(i_ac);

                    // Under side: quad (ib, ic, i_ac, i_ab) -> two triangles
                    scratch.indices_under.push(ib as u32);
                    scratch.indices_under.push(ic as u32);
                    scratch.indices_under.push(i_ac);

                    scratch.indices_under.push(ib as u32);
                    scratch.indices_under.push(i_ac);
                    scratch.indices_under.push(i_ab);
                }
                2 => {
                    // two vertices above, one below
                    // ib is the single below vertex, ia & ic are above
                    let (ib, ia, ic) = if !above_a {
                        (a, b, c)
                    } else if !above_b {
                        (b, c, a)
                    } else {
                        (c, a, b)
                    };

                    // intersections on edges ib-ia and ib-ic
                    let i_ba = make_intersection(ib, ia, &mut scratch.new_vertices);
                    let i_bc = make_intersection(ib, ic, &mut scratch.new_vertices);

                    // Under side: single triangle (ib, i_ba, i_bc)
                    scratch.indices_under.push(ib as u32);
                    scratch.indices_under.push(i_ba);
                    scratch.indices_under.push(i_bc);

                    // Above side: quad (ia, ic, i_bc, i_ba) -> two triangles
                    scratch.indices_above.push(ia as u32);
                    scratch.indices_above.push(ic as u32);
                    scratch.indices_above.push(i_bc);

                    scratch.indices_above.push(ia as u32);
                    scratch.indices_above.push(i_bc);
                    scratch.indices_above.push(i_ba);
                }
                _ => {
                    // unreachable
                }
            }
        }

        // 3) Compute sizes
        let v_bytes = scratch.new_vertices.len() as u64 * std::mem::size_of::<V>() as u64;
        let i_bytes_above = scratch.indices_above.len() as u64 * 4;
        let i_bytes_under = scratch.indices_under.len() as u64 * 4;
        let v_align = std::mem::size_of::<V>() as u64;
        let i_align = 4u64;

        // 4) Allocation Loop (Same logic, just using scratch data)
        loop {
            for (pi, page) in self.pages.iter_mut().enumerate() {
                // 1. Find Vertex Fit
                let v_off = match find_fit(&page.free_v, v_bytes, v_align) {
                    Some(off) => off,
                    None => continue,
                };

                // 2. Find Index Fit (Co-locating above and under indices to reduce fragmentation)
                let total_i_bytes = i_bytes_above + i_bytes_under;
                let i_off_base = if total_i_bytes > 0 {
                    match find_fit(&page.free_i, total_i_bytes, i_align) {
                        Some(off) => Some(off),
                        None => continue,
                    }
                } else {
                    None
                };

                // 3. Commit Allocations (No new Vec allocations inside)
                commit_alloc(&mut page.free_v, v_off, v_bytes);
                if let Some(off) = i_off_base {
                    commit_alloc(&mut page.free_i, off, total_i_bytes);
                }

                // 4. Batch Upload to GPU
                queue.write_buffer(
                    &page.vertex_buf,
                    v_off,
                    bytemuck::cast_slice(&scratch.new_vertices),
                );

                let i_off_above = i_off_base.unwrap_or(0);
                let i_off_under = i_off_above + i_bytes_above;

                if i_bytes_above > 0 {
                    queue.write_buffer(
                        &page.index_buf,
                        i_off_above,
                        bytemuck::cast_slice(&scratch.indices_above),
                    );
                }
                if i_bytes_under > 0 {
                    queue.write_buffer(
                        &page.index_buf,
                        i_off_under,
                        bytemuck::cast_slice(&scratch.indices_under),
                    );
                }

                return GpuChunkHandle {
                    page: pi as u8,
                    base_vertex: (v_off / v_align) as i32,
                    first_index_above: (i_off_above / 4) as u32,
                    index_count_above: scratch.indices_above.len() as u32,
                    first_index_under: (i_off_under / 4) as u32,
                    index_count_under: scratch.indices_under.len() as u32,
                    vertex_count: scratch.new_vertices.len() as u32,
                };
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
            let i_off_above = handle.first_index_above as u64 * 4;
            let i_bytes_above = handle.index_count_above as u64 * 4;
            free_insert_and_coalesce(&mut page.free_i, i_off_above, i_bytes_above);
        }
        if handle.index_count_under > 0 {
            let i_off_under = handle.first_index_under as u64 * 4;
            let i_bytes_under = handle.index_count_under as u64 * 4;
            free_insert_and_coalesce(&mut page.free_i, i_off_under, i_bytes_under);
        }
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
