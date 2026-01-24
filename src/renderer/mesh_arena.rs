use crate::renderer::general_mesh_arena::{
    FreeRange, commit_alloc, find_fit, free_insert_and_coalesce,
};
use crate::terrain::chunk_builder::GpuChunkHandle;
use crate::ui::vertex::VertexWithPosition;
use wgpu::{Buffer, BufferUsages, Device, Queue};

struct AllocationOffsets {
    vertex: u64,
    index_base: Option<u64>,
}

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

struct AllocationRequest {
    vertex_bytes: u64,
    vertex_align: u64,
    index_bytes_above: u64,
    index_bytes_under: u64,
}

impl AllocationRequest {
    fn from_scratch<V>(scratch: &GeometryScratch<V>) -> Self {
        Self {
            vertex_bytes: scratch.new_vertices.len() as u64 * size_of::<V>() as u64,
            vertex_align: size_of::<V>() as u64,
            index_bytes_above: scratch.indices_above.len() as u64 * 4,
            index_bytes_under: scratch.indices_under.len() as u64 * 4,
        }
    }

    fn total_index_bytes(&self) -> u64 {
        self.index_bytes_above + self.index_bytes_under
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

impl MeshPage {
    fn try_allocate_and_upload<V: bytemuck::Pod>(
        &mut self,
        queue: &Queue,
        scratch: &GeometryScratch<V>,
        request: &AllocationRequest,
        page_index: usize,
    ) -> Option<GpuChunkHandle> {
        let offsets = self.find_allocation_offsets(request)?;

        self.commit_allocations(&offsets, request);
        self.upload_buffers(queue, scratch, &offsets, request);

        Some(self.create_handle(scratch, &offsets, request, page_index))
    }

    fn find_allocation_offsets(&self, request: &AllocationRequest) -> Option<AllocationOffsets> {
        let vertex_offset = find_fit(&self.free_v, request.vertex_bytes, request.vertex_align)?;

        let index_offset_base = if request.total_index_bytes() > 0 {
            Some(find_fit(&self.free_i, request.total_index_bytes(), 4)?)
        } else {
            None
        };

        Some(AllocationOffsets {
            vertex: vertex_offset,
            index_base: index_offset_base,
        })
    }

    fn commit_allocations(&mut self, offsets: &AllocationOffsets, request: &AllocationRequest) {
        commit_alloc(&mut self.free_v, offsets.vertex, request.vertex_bytes);

        if let Some(index_base) = offsets.index_base {
            commit_alloc(&mut self.free_i, index_base, request.total_index_bytes());
        }
    }

    fn upload_buffers<V: bytemuck::Pod>(
        &self,
        queue: &Queue,
        scratch: &GeometryScratch<V>,
        offsets: &AllocationOffsets,
        request: &AllocationRequest,
    ) {
        queue.write_buffer(
            &self.vertex_buf,
            offsets.vertex,
            bytemuck::cast_slice(&scratch.new_vertices),
        );

        let index_above_offset = offsets.index_base.unwrap_or(0);
        let index_under_offset = index_above_offset + request.index_bytes_above;

        if request.index_bytes_above > 0 {
            queue.write_buffer(
                &self.index_buf,
                index_above_offset,
                bytemuck::cast_slice(&scratch.indices_above),
            );
        }

        if request.index_bytes_under > 0 {
            queue.write_buffer(
                &self.index_buf,
                index_under_offset,
                bytemuck::cast_slice(&scratch.indices_under),
            );
        }
    }

    fn create_handle<V>(
        &self,
        scratch: &GeometryScratch<V>,
        offsets: &AllocationOffsets,
        request: &AllocationRequest,
        page_index: usize,
    ) -> GpuChunkHandle {
        let index_above_offset = offsets.index_base.unwrap_or(0);
        let index_under_offset = index_above_offset + request.index_bytes_above;

        GpuChunkHandle {
            page: page_index as u8,
            base_vertex: (offsets.vertex / request.vertex_align) as i32,
            first_index_above: (index_above_offset / 4) as u32,
            index_count_above: scratch.indices_above.len() as u32,
            first_index_under: (index_under_offset / 4) as u32,
            index_count_under: scratch.indices_under.len() as u32,
            vertex_count: scratch.new_vertices.len() as u32,
        }
    }
}

pub struct TerrainMeshArena {
    pub pages: Vec<MeshPage>,
    pub page_v_bytes: u64,
    pub page_i_bytes: u64,
}

impl TerrainMeshArena {
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
        indices: &[u32],
        scratch: &mut GeometryScratch<V>,
    ) -> GpuChunkHandle {
        scratch.clear();
        scratch.new_vertices.extend_from_slice(vertices);

        clip_triangles_by_plane(indices, scratch);

        let allocation = AllocationRequest::from_scratch::<V>(scratch);

        loop {
            if let Some(handle) = self.try_allocate_and_upload(queue, scratch, &allocation) {
                return handle;
            }
            self.add_page(device);
        }
    }

    fn try_allocate_and_upload<V: bytemuck::Pod>(
        &mut self,
        queue: &Queue,
        scratch: &GeometryScratch<V>,
        request: &AllocationRequest,
    ) -> Option<GpuChunkHandle> {
        for (page_index, page) in self.pages.iter_mut().enumerate() {
            if let Some(handle) = page.try_allocate_and_upload(queue, scratch, request, page_index)
            {
                return Some(handle);
            }
        }
        None
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

fn align_up(x: u64, a: u64) -> u64 {
    if a == 0 {
        return x;
    }
    ((x + a - 1) / a) * a
}

// Triangle clipping logic - completely separate from allocation

fn clip_triangles_by_plane<V: bytemuck::Pod + VertexWithPosition + Clone + Copy>(
    indices: &[u32],
    scratch: &mut GeometryScratch<V>,
) {
    for tri in indices.chunks(3) {
        clip_single_triangle(tri, scratch);
    }
}

fn clip_single_triangle<V: bytemuck::Pod + VertexWithPosition + Clone + Copy>(
    tri: &[u32],
    scratch: &mut GeometryScratch<V>,
) {
    let [a, b, c] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];

    let ya = scratch.new_vertices[a].local_position()[1];
    let yb = scratch.new_vertices[b].local_position()[1];
    let yc = scratch.new_vertices[c].local_position()[1];

    let above = [ya >= 0.0, yb >= 0.0, yc >= 0.0];
    let above_count = above.iter().filter(|&&x| x).count();

    match above_count {
        0 => scratch.indices_under.extend_from_slice(tri),
        3 => scratch.indices_above.extend_from_slice(tri),
        1 => clip_one_above(tri, above, scratch),
        2 => clip_two_above(tri, above, scratch),
        _ => unreachable!(),
    }
}

fn clip_one_above<V: bytemuck::Pod + VertexWithPosition + Clone + Copy>(
    tri: &[u32],
    above: [bool; 3],
    scratch: &mut GeometryScratch<V>,
) {
    let (ia, ib, ic) = reorder_one_above(tri, above);

    let i_ab = create_intersection_vertex(ia, ib, scratch);
    let i_ac = create_intersection_vertex(ia, ic, scratch);

    // Above: single triangle
    scratch
        .indices_above
        .extend_from_slice(&[ia as u32, i_ab, i_ac]);

    // Under: quad as two triangles
    scratch
        .indices_under
        .extend_from_slice(&[ib as u32, ic as u32, i_ac]);
    scratch
        .indices_under
        .extend_from_slice(&[ib as u32, i_ac, i_ab]);
}

fn clip_two_above<V: bytemuck::Pod + VertexWithPosition + Clone + Copy>(
    tri: &[u32],
    above: [bool; 3],
    scratch: &mut GeometryScratch<V>,
) {
    let (ib, ia, ic) = reorder_one_below(tri, above);

    let i_ba = create_intersection_vertex(ib, ia, scratch);
    let i_bc = create_intersection_vertex(ib, ic, scratch);

    // Under: single triangle
    scratch
        .indices_under
        .extend_from_slice(&[ib as u32, i_ba, i_bc]);

    // Above: quad as two triangles
    scratch
        .indices_above
        .extend_from_slice(&[ia as u32, ic as u32, i_bc]);
    scratch
        .indices_above
        .extend_from_slice(&[ia as u32, i_bc, i_ba]);
}

fn reorder_one_above(tri: &[u32], above: [bool; 3]) -> (usize, usize, usize) {
    let [a, b, c] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
    match above {
        [true, false, false] => (a, b, c),
        [false, true, false] => (b, c, a),
        [false, false, true] => (c, a, b),
        _ => unreachable!(),
    }
}

fn reorder_one_below(tri: &[u32], above: [bool; 3]) -> (usize, usize, usize) {
    let [a, b, c] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
    match above {
        [false, true, true] => (a, b, c),
        [true, false, true] => (b, c, a),
        [true, true, false] => (c, a, b),
        _ => unreachable!(),
    }
}

fn create_intersection_vertex<V: bytemuck::Pod + VertexWithPosition + Clone + Copy>(
    i: usize,
    j: usize,
    scratch: &mut GeometryScratch<V>,
) -> u32 {
    let vi = scratch.new_vertices[i];
    let vj = scratch.new_vertices[j];

    let yi = vi.local_position()[1];
    let yj = vj.local_position()[1];

    let t = if (yj - yi).abs() < f32::EPSILON {
        0.5
    } else {
        (0.0 - yi) / (yj - yi)
    };

    let new_vertex = V::lerp(&vi, &vj, t);
    scratch.new_vertices.push(new_vertex);
    (scratch.new_vertices.len() - 1) as u32
}
