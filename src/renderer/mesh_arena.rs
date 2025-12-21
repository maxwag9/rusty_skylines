use crate::terrain::chunk_builder::GpuChunkHandle;
use crate::ui::vertex::VertexWithPosition;
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

    pub fn alloc_and_upload<V: bytemuck::Pod + VertexWithPosition + Clone>(
        &mut self,
        device: &Device,
        queue: &Queue,
        vertices: &[V],
        indices: &[u32],
    ) -> GpuChunkHandle {
        // 1) split indices by sea level, creating new vertices where edges cross y = 0
        let mut new_vertices: Vec<V> = vertices.to_vec(); // start with original vertices
        let mut indices_above: Vec<u32> = Vec::new();
        let mut indices_under: Vec<u32> = Vec::new();

        // helper to create an interpolated vertex between i and j, caching not implemented
        let mut make_intersection = |i: usize, j: usize| -> u32 {
            let vi = &new_vertices[i].clone();
            let vj = &new_vertices[j].clone();
            let yi = vi.position()[1];
            let yj = vj.position()[1];
            // guard against division by zero (shouldn't happen if signs differ)
            let t = if (yj - yi).abs() < std::f32::EPSILON {
                0.5
            } else {
                (0.0 - yi) / (yj - yi)
            };
            let new_v = V::lerp(vi, vj, t);
            new_vertices.push(new_v);
            (new_vertices.len() - 1) as u32
        };

        for tri in indices.chunks(3) {
            let a = tri[0] as usize;
            let b = tri[1] as usize;
            let c = tri[2] as usize;

            let ya = vertices[a].position()[1];
            let yb = vertices[b].position()[1];
            let yc = vertices[c].position()[1];

            let above_a = ya >= 0.0;
            let above_b = yb >= 0.0;
            let above_c = yc >= 0.0;

            let above_count = (above_a as u8 + above_b as u8 + above_c as u8) as usize;

            match above_count {
                0 => {
                    // all under
                    indices_under.extend_from_slice(tri);
                }
                3 => {
                    // all above
                    indices_above.extend_from_slice(tri);
                }
                1 => {
                    // one vertex above, two below
                    // find which is above
                    let (ia, ib, ic, ya_flag) = if above_a {
                        (a, b, c, above_a)
                    } else if above_b {
                        (b, c, a, above_b)
                    } else {
                        (c, a, b, above_c)
                    };
                    // ia is the single above vertex, ib & ic are below
                    // intersections on edges ia-ib and ia-ic
                    let i_ab = make_intersection(ia, ib);
                    let i_ac = make_intersection(ia, ic);

                    // Above side: single triangle (ia, i_ab, i_ac)
                    indices_above.push(ia as u32);
                    indices_above.push(i_ab);
                    indices_above.push(i_ac);

                    // Under side: quad (ib, ic, i_ac, i_ab) -> two triangles (ib, ic, i_ac) and (ib, i_ac, i_ab)
                    indices_under.push(ib as u32);
                    indices_under.push(ic as u32);
                    indices_under.push(i_ac);

                    indices_under.push(ib as u32);
                    indices_under.push(i_ac);
                    indices_under.push(i_ab);
                }
                2 => {
                    // two vertices above, one below
                    // find which is below
                    let (ib, ia, ic, _) = if !above_a {
                        (a, b, c, above_a)
                    } else if !above_b {
                        (b, c, a, above_b)
                    } else {
                        (c, a, b, above_c)
                    };
                    // ib is the single below vertex, ia & ic are above
                    // intersections on edges ib-ia and ib-ic
                    let i_ba = make_intersection(ib, ia);
                    let i_bc = make_intersection(ib, ic);

                    // Under side: single triangle (ib, i_ba, i_bc)
                    indices_under.push(ib as u32);
                    indices_under.push(i_ba);
                    indices_under.push(i_bc);

                    // Above side: quad (ia, ic, i_bc, i_ba) -> two triangles (ia, ic, i_bc) and (ia, i_bc, i_ba)
                    indices_above.push(ia as u32);
                    indices_above.push(ic as u32);
                    indices_above.push(i_bc);

                    indices_above.push(ia as u32);
                    indices_above.push(i_bc);
                    indices_above.push(i_ba);
                }
                _ => {
                    // unreachable but keep for completeness
                }
            }
        }

        // 2) compute byte sizes based on new_vertices
        let v_bytes = new_vertices.len() as u64 * std::mem::size_of::<V>() as u64;
        let i_bytes_above = indices_above.len() as u64 * 4;
        let i_bytes_under = indices_under.len() as u64 * 4;

        let v_align = std::mem::size_of::<V>() as u64;
        let i_align = 4u64;

        loop {
            for (pi, page) in self.pages.iter_mut().enumerate() {
                // find vertex slot first
                let v_off_opt = find_fit(&page.free_v, v_bytes, v_align);
                if v_off_opt.is_none() {
                    continue;
                }
                let v_off = v_off_opt.unwrap();

                // handle index allocation cases:
                let (i_off_above, i_off_under) = match (i_bytes_above > 0, i_bytes_under > 0) {
                    (false, false) => (0u64, 0u64), // no indices at all
                    (true, false) => {
                        if let Some(i_off) = find_fit(&page.free_i, i_bytes_above, i_align) {
                            (i_off, 0)
                        } else {
                            continue;
                        }
                    }
                    (false, true) => {
                        if let Some(i_off) = find_fit(&page.free_i, i_bytes_under, i_align) {
                            (0, i_off)
                        } else {
                            continue;
                        }
                    }
                    (true, true) => {
                        let total = i_bytes_above + i_bytes_under;
                        if let Some(i_off_total) = find_fit(&page.free_i, total, i_align) {
                            let off_above = i_off_total;
                            let off_under = i_off_total + i_bytes_above;
                            (off_above, off_under)
                        } else {
                            continue;
                        }
                    }
                };

                // Commit allocations now. Commit vertex first, then index region(s).
                commit_alloc(&mut page.free_v, v_off, v_bytes);

                if i_bytes_above > 0 {
                    commit_alloc(&mut page.free_i, i_off_above, i_bytes_above);
                }
                if i_bytes_under > 0 {
                    commit_alloc(&mut page.free_i, i_off_under, i_bytes_under);
                }

                // Upload buffers
                queue.write_buffer(&page.vertex_buf, v_off, bytemuck::cast_slice(&new_vertices));
                if i_bytes_above > 0 {
                    queue.write_buffer(
                        &page.index_buf,
                        i_off_above,
                        bytemuck::cast_slice(&indices_above),
                    );
                }
                if i_bytes_under > 0 {
                    queue.write_buffer(
                        &page.index_buf,
                        i_off_under,
                        bytemuck::cast_slice(&indices_under),
                    );
                }

                let base_vertex = (v_off / std::mem::size_of::<V>() as u64) as i32;
                let first_index_above = if i_bytes_above > 0 {
                    (i_off_above / 4) as u32
                } else {
                    0
                };
                let index_count_above = indices_above.len() as u32;
                let first_index_under = if i_bytes_under > 0 {
                    (i_off_under / 4) as u32
                } else {
                    0
                };
                let index_count_under = indices_under.len() as u32;

                return GpuChunkHandle {
                    page: pi as u8,
                    base_vertex,
                    first_index_above,
                    index_count_above,
                    first_index_under,
                    index_count_under,
                    vertex_count: new_vertices.len() as u32,
                };
            }

            // no page fit; add one and retry
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
