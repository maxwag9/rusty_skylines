use crate::renderer::general_mesh_arena::{
    FreeRange, commit_alloc, find_fit, free_insert_and_coalesce,
};
use crate::ui::vertex::VertexWithPosition;
use crate::world::terrain::chunk_builder::GpuChunkHandle;
use wgpu::{Buffer, BufferUsages, Device, Queue};
use crate::resources::{GPU_TELEMETRY};

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
            vertex_bytes: (scratch.new_vertices.len() as u64)
                .saturating_mul(size_of::<V>() as u64),
            vertex_align: size_of::<V>() as u64,
            index_bytes_above: (scratch.indices_above.len() as u64)
                .saturating_mul(4),
            index_bytes_under: (scratch.indices_under.len() as u64)
                .saturating_mul(4),
        }
    }

    fn total_index_bytes(&self) -> u64 {
        self.index_bytes_above.saturating_add(self.index_bytes_under)
    }
}

pub struct MeshPage {
    pub vertex_buf: Buffer,
    pub index_buf: Buffer,
    pub _vcap: u64,
    pub _icap: u64,
    pub free_v: Vec<FreeRange>,
    pub free_i: Vec<FreeRange>,
}

impl MeshPage {
    fn range_waste(
        free: &[FreeRange],
        offset: u64,
        size: u64,
    ) -> Option<u64> {
        let end = offset.checked_add(size)?;

        free.iter().find_map(|range| {
            let range_end = range.start.checked_add(range.size)?;

            if offset >= range.start && end <= range_end {
                Some(
                    (offset - range.start)
                        .saturating_add(range_end - end),
                )
            } else {
                None
            }
        })
    }

    fn allocation_waste(&self, request: &AllocationRequest) -> Option<u64> {
        let offsets = self.find_allocation_offsets(request)?;

        let vertex_waste = Self::range_waste(
            &self.free_v,
            offsets.vertex,
            request.vertex_bytes,
        )?;

        let index_waste = match offsets.index_base {
            Some(offset) => Self::range_waste(
                &self.free_i,
                offset,
                request.total_index_bytes(),
            )?,
            None => 0,
        };

        Some(vertex_waste.saturating_add(index_waste))
    }

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

        Some(self.create_handle(
            scratch,
            &offsets,
            request,
            page_index,
        ))
    }

    fn find_allocation_offsets(
        &self,
        request: &AllocationRequest,
    ) -> Option<AllocationOffsets> {
        let vertex_offset = find_fit(
            &self.free_v,
            request.vertex_bytes,
            request.vertex_align,
        )?;

        let index_offset_base = if request.total_index_bytes() > 0 {
            Some(find_fit(
                &self.free_i,
                request.total_index_bytes(),
                4,
            )?)
        } else {
            None
        };

        Some(AllocationOffsets {
            vertex: vertex_offset,
            index_base: index_offset_base,
        })
    }

    fn commit_allocations(
        &mut self,
        offsets: &AllocationOffsets,
        request: &AllocationRequest,
    ) {
        commit_alloc(
            &mut self.free_v,
            offsets.vertex,
            request.vertex_bytes,
        );

        if let Some(index_base) = offsets.index_base {
            commit_alloc(
                &mut self.free_i,
                index_base,
                request.total_index_bytes(),
            );
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
        let index_under_offset =
            index_above_offset + request.index_bytes_above;

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
        let index_under_offset =
            index_above_offset + request.index_bytes_above;

        GpuChunkHandle {
            page: page_index,
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
    pub pages: Vec<Option<MeshPage>>,
    pub page_v_bytes: u64,
    pub page_i_bytes: u64,
    pub(crate) allocated_bytes: u64,
}

impl TerrainMeshArena {
    pub fn new(
        _device: &Device,
        page_v_bytes: u64,
        page_i_bytes: u64,
    ) -> Self {
        Self {
            pages: Vec::new(),
            page_v_bytes,
            page_i_bytes,
            allocated_bytes: 0,
        }
    }

    pub fn clear(&mut self) {
        self.pages.clear();
        self.allocated_bytes = 0;
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    pub fn active_page_count(&self) -> usize {
        self.pages.iter().filter(|page| page.is_some()).count()
    }

    pub fn get_page(&self, index: u32) -> Option<&MeshPage> {
        self.pages.get(index as usize)?.as_ref()
    }

    pub fn trim_empty_pages(&mut self) -> u64 {
        let mut released_bytes = 0u64;
        let mut released_pages = 0usize;

        for (page_index, slot) in self.pages.iter_mut().enumerate() {
            let is_empty = slot.as_ref().is_some_and(|page| {
                page.free_v.len() == 1
                    && page.free_v[0].start == 0
                    && page.free_v[0].size == page._vcap
                    && page.free_i.len() == 1
                    && page.free_i[0].start == 0
                    && page.free_i[0].size == page._icap
            });

            if is_empty {
                if let Some(page) = slot.take() {
                    released_bytes = released_bytes
                        .saturating_add(page._vcap)
                        .saturating_add(page._icap);

                    released_pages += 1;
                    drop(page);

                    tracing::debug!(
                        page_index,
                        "Released empty terrain GPU page"
                    );
                }
            }
        }

        self.allocated_bytes = self.allocated_bytes
            .saturating_sub(released_bytes);

        if released_pages > 0 {
            tracing::info!(
                released_pages,
                released_mib = released_bytes / (1024 * 1024),
                remaining_pages = self.active_page_count(),
                remaining_mib = self.allocated_bytes / (1024 * 1024),
                "Terrain GPU arena trimmed"
            );
        }

        released_bytes
    }


    fn add_page(&mut self, device: &Device) -> bool {
        let Some(page_bytes) = self.page_v_bytes.checked_add(self.page_i_bytes) else {
            return false;
        };

        if page_bytes == 0
            || self.page_v_bytes == 0
            || self.page_i_bytes == 0
        {
            return false;
        }

        let budget = terrain_vram_budget(self.allocated_bytes);

        let Some(new_allocated) = self.allocated_bytes.checked_add(page_bytes) else {
            return false;
        };

        if new_allocated > budget {
            tracing::warn!(
                allocated_mib = self.allocated_bytes / (1024 * 1024),
                requested_mib = page_bytes / (1024 * 1024),
                budget_mib = budget / (1024 * 1024),
                "Terrain GPU allocation refused: VRAM budget exhausted"
            );

            return false;
        }

        let error_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);

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

        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        if let Some(error) = pollster::block_on(error_scope.pop()) {
            tracing::error!(
                error = %error,
                "Terrain GPU page allocation failed"
            );
            return false;
        }

        let page = MeshPage {
            vertex_buf,
            index_buf,
            _vcap: self.page_v_bytes,
            _icap: self.page_i_bytes,
            free_v: vec![FreeRange {
                start: 0,
                size: self.page_v_bytes,
            }],
            free_i: vec![FreeRange {
                start: 0,
                size: self.page_i_bytes,
            }],
        };

        let page_index = if let Some(index) =
            self.pages.iter().position(|page| page.is_none())
        {
            self.pages[index] = Some(page);
            index
        } else {
            self.pages.push(Some(page));
            self.pages.len() - 1
        };

        self.allocated_bytes = new_allocated;

        tracing::debug!(
            page_index,
            active_pages = self.active_page_count(),
            allocated_mib = self.allocated_bytes / (1024 * 1024),
            budget_mib = budget / (1024 * 1024),
            "Terrain GPU page allocated"
        );

        true
    }

    pub(crate) fn alloc_and_upload<
        V: bytemuck::Pod + VertexWithPosition + Clone + Copy,
    >(
        &mut self,
        device: &Device,
        queue: &Queue,
        vertices: &[V],
        indices: &[u32],
        scratch: &mut GeometryScratch<V>,
    ) -> Option<GpuChunkHandle> {
        scratch.clear();
        scratch.new_vertices.extend_from_slice(vertices);

        clip_triangles_by_plane(indices, scratch);

        let allocation = AllocationRequest::from_scratch(scratch);

        if allocation.vertex_bytes == 0
            || allocation.vertex_align == 0
            || allocation.vertex_bytes > self.page_v_bytes
            || allocation.total_index_bytes() > self.page_i_bytes
        {
            tracing::warn!(
                vertex_bytes = allocation.vertex_bytes,
                index_bytes = allocation.total_index_bytes(),
                "Terrain mesh exceeds GPU page capacity"
            );

            return None;
        }

        loop {
            if let Some(handle) =
                self.try_allocate_and_upload(queue, scratch, &allocation)
            {
                return Some(handle);
            }

            if !self.add_page(device) {
                return None;
            }
        }
    }

    fn try_allocate_and_upload<V: bytemuck::Pod>(
        &mut self,
        queue: &Queue,
        scratch: &GeometryScratch<V>,
        request: &AllocationRequest,
    ) -> Option<GpuChunkHandle> {
        let mut best_page: Option<(usize, u64)> = None;

        for (page_index, slot) in self.pages.iter().enumerate() {
            let Some(page) = slot.as_ref() else {
                continue;
            };

            let Some(waste) = page.allocation_waste(request) else {
                continue;
            };

            if best_page.is_none_or(|(_, best_waste)| waste < best_waste) {
                best_page = Some((page_index, waste));
            }
        }

        let (page_index, _) = best_page?;

        self.pages[page_index].as_mut()?.try_allocate_and_upload(
            queue,
            scratch,
            request,
            page_index,
        )
    }
    pub fn theoretical_minimum_pages(&self) -> usize {
        if self.page_v_bytes == 0 || self.page_i_bytes == 0 {
            return 0;
        }

        let mut used_v = 0u64;
        let mut used_i = 0u64;

        for page in self.pages.iter().flatten() {
            let free_v = page.free_v.iter()
                .fold(0u64, |sum, r| sum.saturating_add(r.size));

            let free_i = page.free_i.iter()
                .fold(0u64, |sum, r| sum.saturating_add(r.size));

            used_v = used_v.saturating_add(
                page._vcap.saturating_sub(free_v)
            );

            used_i = used_i.saturating_add(
                page._icap.saturating_sub(free_i)
            );
        }

        let pages_v = used_v.div_ceil(self.page_v_bytes);
        let pages_i = used_i.div_ceil(self.page_i_bytes);

        pages_v.max(pages_i) as usize
    }

    pub fn potentially_reclaimable_pages(&self) -> usize {
        self.active_page_count()
            .saturating_sub(self.theoretical_minimum_pages())
    }

    pub fn should_rebuild_for_fragmentation(&self) -> bool {
        self.potentially_reclaimable_pages() >= 2
    }
    pub(crate) fn free<V>(&mut self, handle: GpuChunkHandle) {
        let page_index = handle.page as usize;

        let Some(page) = self.pages
            .get_mut(page_index)
            .and_then(Option::as_mut)
        else {
            tracing::error!(
                page_index,
                "Attempted to free allocation from a missing terrain GPU page"
            );
            return;
        };

        let stride = size_of::<V>() as u64;

        if stride == 0 || handle.base_vertex < 0 {
            tracing::error!(
                page_index,
                "Invalid terrain vertex allocation handle"
            );
            return;
        }

        let vertex_offset = (handle.base_vertex as u64).saturating_mul(stride);
        let vertex_bytes = (handle.vertex_count as u64).saturating_mul(stride);

        let index_above_offset =
            (handle.first_index_above as u64).saturating_mul(4);
        let index_above_bytes =
            (handle.index_count_above as u64).saturating_mul(4);

        let index_under_offset =
            (handle.first_index_under as u64).saturating_mul(4);
        let index_under_bytes =
            (handle.index_count_under as u64).saturating_mul(4);

        if !can_free_range(
            &page.free_v,
            vertex_offset,
            vertex_bytes,
            page._vcap,
        ) || !can_free_range(
            &page.free_i,
            index_above_offset,
            index_above_bytes,
            page._icap,
        ) || !can_free_range(
            &page.free_i,
            index_under_offset,
            index_under_bytes,
            page._icap,
        ) {
            tracing::error!(
                page_index,
                "Invalid or duplicate terrain GPU allocation free"
            );
            return;
        }

        if vertex_bytes > 0 {
            free_insert_and_coalesce(
                &mut page.free_v,
                vertex_offset,
                vertex_bytes,
            );
        }

        if index_above_bytes > 0 {
            free_insert_and_coalesce(
                &mut page.free_i,
                index_above_offset,
                index_above_bytes,
            );
        }

        if index_under_bytes > 0 {
            free_insert_and_coalesce(
                &mut page.free_i,
                index_under_offset,
                index_under_bytes,
            );
        }

        let page_empty = page.free_v.len() == 1
            && page.free_v[0].start == 0
            && page.free_v[0].size == page._vcap
            && page.free_i.len() == 1
            && page.free_i[0].start == 0
            && page.free_i[0].size == page._icap;

        if page_empty {
            let page = self.pages[page_index].take().unwrap();
            let page_bytes = page._vcap.saturating_add(page._icap);

            drop(page);

            self.allocated_bytes = self.allocated_bytes
                .saturating_sub(page_bytes);
        }
    }
}

fn can_free_range(
    free: &[FreeRange],
    start: u64,
    size: u64,
    capacity: u64,
) -> bool {
    if size == 0 {
        return true;
    }

    let Some(end) = start.checked_add(size) else {
        return false;
    };

    if end > capacity {
        return false;
    }

    !free.iter().any(|range| {
        let range_end = range.start.saturating_add(range.size);

        start < range_end && range.start < end
    })
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

const TERRAIN_VRAM_RESERVE: u64 = 1024 * 1024 * 1024;
const TERRAIN_MAX_VRAM: u64 = 2 * 1024 * 1024 * 1024;
const TERRAIN_FALLBACK_BUDGET: u64 = 512 * 1024 * 1024;

fn terrain_vram_budget(arena_allocated_bytes: u64) -> u64 {
    let Ok(gpu) = GPU_TELEMETRY.read() else {
        return TERRAIN_FALLBACK_BUDGET;
    };

    let (Some(total), Some(used)) =
        (gpu.memory_total_bytes, gpu.memory_used_bytes)
    else {
        return TERRAIN_FALLBACK_BUDGET;
    };

    if total == 0 {
        return TERRAIN_FALLBACK_BUDGET;
    }

    let other_gpu_usage = used.saturating_sub(arena_allocated_bytes);

    total
        .saturating_sub(other_gpu_usage)
        .saturating_sub(TERRAIN_VRAM_RESERVE)
        .min(TERRAIN_MAX_VRAM)
}