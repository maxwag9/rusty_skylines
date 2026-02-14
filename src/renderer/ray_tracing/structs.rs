use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use std::collections::HashMap;
use std::mem::size_of;

const SAH_BINS: usize = 12;
const SAH_TRAVERSAL_COST: f32 = 1.0;
const SAH_INTERSECTION_COST: f32 = 1.0;
const MAX_LEAF_PRIMITIVES: usize = 4;

//
// Triangle geometry buffers (BLAS data)
//
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct RTVertex {
    pub position: [f32; 3],
    pub _pad: f32, // align to 16
}

impl RTVertex {
    pub fn size() -> usize {
        size_of::<RTVertex>()
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct RTIndex(pub u32);

#[derive(Clone)]
struct BuildPrimitive {
    aabb: Aabb,
    centroid: [f32; 3],
    index: u32,
}

#[derive(Clone, Copy, Default)]
struct SahBin {
    bounds: Aabb,
    count: u32,
}

impl Default for Aabb {
    fn default() -> Self {
        Self::EMPTY
    }
}

// ============================================================================
// BLAS - Bottom Level Acceleration Structure
// ============================================================================

pub struct Blas {
    pub vertices: Vec<RTVertex>,
    pub indices: Vec<u32>,
    pub bvh_nodes: Vec<BvhNode>,
    root_aabb: Aabb,
}

impl Blas {
    /// Build BLAS with SAH BVH from mesh data
    pub fn build(positions: Vec<[f32; 3]>, indices: Vec<u32>) -> Self {
        let vertices: Vec<RTVertex> = positions
            .iter()
            .map(|&p| RTVertex {
                position: p,
                _pad: 0.0,
            })
            .collect();

        let tri_count = indices.len() / 3;
        if tri_count == 0 {
            return Self {
                vertices,
                indices,
                bvh_nodes: vec![],
                root_aabb: Aabb::EMPTY,
            };
        }

        // Build primitive list with AABBs
        let mut primitives: Vec<BuildPrimitive> = Vec::with_capacity(tri_count);
        for i in 0..tri_count {
            let i0 = indices[i * 3] as usize;
            let i1 = indices[i * 3 + 1] as usize;
            let i2 = indices[i * 3 + 2] as usize;

            let mut aabb = Aabb::EMPTY;
            aabb.expand_point(positions[i0]);
            aabb.expand_point(positions[i1]);
            aabb.expand_point(positions[i2]);

            primitives.push(BuildPrimitive {
                aabb,
                centroid: aabb.centroid(),
                index: i as u32,
            });
        }

        let mut nodes = Vec::with_capacity(tri_count * 2);
        let mut ordered_indices = Vec::with_capacity(indices.len());

        Self::build_recursive(
            &mut nodes,
            &mut primitives,
            0,
            tri_count,
            &mut ordered_indices,
            &indices,
        );

        let root_aabb = if !nodes.is_empty() {
            Aabb::new(nodes[0].aabb_min, nodes[0].aabb_max)
        } else {
            Aabb::EMPTY
        };

        Self {
            vertices,
            indices: ordered_indices,
            bvh_nodes: nodes,
            root_aabb,
        }
    }

    fn build_recursive(
        nodes: &mut Vec<BvhNode>,
        primitives: &mut [BuildPrimitive],
        start: usize,
        end: usize,
        ordered_indices: &mut Vec<u32>,
        original_indices: &[u32],
    ) -> u32 {
        let node_idx = nodes.len() as u32;
        nodes.push(BvhNode {
            aabb_min: [0.0; 3],
            _pad0: 0.0,
            aabb_max: [0.0; 3],
            _pad1: 0.0,
            child_or_tri_offset: 0,
            meta: 0,
            tri_count_or_right: 0,
            _pad2: 0,
        });

        // Compute bounds
        let mut bounds = Aabb::EMPTY;
        for prim in &primitives[start..end] {
            bounds.expand(&prim.aabb);
        }

        let count = end - start;

        // Create leaf if few primitives
        if count <= MAX_LEAF_PRIMITIVES {
            let first_tri = (ordered_indices.len() / 3) as u32;
            for prim in &primitives[start..end] {
                let ti = prim.index as usize;
                ordered_indices.push(original_indices[ti * 3]);
                ordered_indices.push(original_indices[ti * 3 + 1]);
                ordered_indices.push(original_indices[ti * 3 + 2]);
            }

            nodes[node_idx as usize] = BvhNode {
                aabb_min: bounds.min,
                _pad0: 0.0,
                aabb_max: bounds.max,
                _pad1: 0.0,
                child_or_tri_offset: first_tri,
                meta: BvhNode::make_leaf(),
                tri_count_or_right: count as u32,
                _pad2: 0,
            };
            return node_idx;
        }

        // Compute centroid bounds
        let mut centroid_bounds = Aabb::EMPTY;
        for prim in &primitives[start..end] {
            centroid_bounds.expand_point(prim.centroid);
        }

        // Find best SAH split
        let split = Self::find_best_split(&primitives[start..end], &bounds, &centroid_bounds);

        let mid = if let Some((axis, split_pos)) = split {
            Self::partition_primitives(primitives, start, end, axis, split_pos)
        } else {
            start + count / 2
        };

        // Ensure valid split
        let mid = mid.clamp(start + 1, end - 1);

        // Recurse
        let left_child = Self::build_recursive(
            nodes,
            primitives,
            start,
            mid,
            ordered_indices,
            original_indices,
        );
        let right_child = Self::build_recursive(
            nodes,
            primitives,
            mid,
            end,
            ordered_indices,
            original_indices,
        );

        nodes[node_idx as usize] = BvhNode {
            aabb_min: bounds.min,
            _pad0: 0.0,
            aabb_max: bounds.max,
            _pad1: 0.0,
            child_or_tri_offset: left_child,
            meta: BvhNode::make_inner(),
            tri_count_or_right: right_child,
            _pad2: 0,
        };

        node_idx
    }

    fn find_best_split(
        primitives: &[BuildPrimitive],
        bounds: &Aabb,
        centroid_bounds: &Aabb,
    ) -> Option<(usize, f32)> {
        let parent_sa = bounds.surface_area();
        if parent_sa <= 0.0 {
            return None;
        }

        let leaf_cost = SAH_INTERSECTION_COST * primitives.len() as f32;
        let mut best_cost = leaf_cost;
        let mut best_split: Option<(usize, f32)> = None;

        for axis in 0..3 {
            let extent = centroid_bounds.max[axis] - centroid_bounds.min[axis];
            if extent <= f32::EPSILON {
                continue;
            }

            // Initialize bins
            let mut bins = [SahBin::default(); SAH_BINS];

            // Populate bins
            for prim in primitives {
                let offset = (prim.centroid[axis] - centroid_bounds.min[axis]) / extent;
                let bin_idx = ((offset * SAH_BINS as f32) as usize).min(SAH_BINS - 1);
                bins[bin_idx].count += 1;
                bins[bin_idx].bounds.expand(&prim.aabb);
            }

            // Sweep left->right to accumulate
            let mut left_area = [0.0f32; SAH_BINS - 1];
            let mut left_count = [0u32; SAH_BINS - 1];
            let mut cumulative_bounds = Aabb::EMPTY;
            let mut cumulative_count = 0u32;

            for i in 0..(SAH_BINS - 1) {
                if bins[i].bounds.is_valid() {
                    cumulative_bounds.expand(&bins[i].bounds);
                }
                cumulative_count += bins[i].count;
                left_area[i] = cumulative_bounds.surface_area();
                left_count[i] = cumulative_count;
            }

            // Sweep right->left and compute SAH cost
            let mut right_bounds = Aabb::EMPTY;
            let mut right_count = 0u32;

            for i in (1..SAH_BINS).rev() {
                if bins[i].bounds.is_valid() {
                    right_bounds.expand(&bins[i].bounds);
                }
                right_count += bins[i].count;

                let lc = left_count[i - 1];
                if lc == 0 || right_count == 0 {
                    continue;
                }

                let la = left_area[i - 1];
                let ra = right_bounds.surface_area();

                let cost = SAH_TRAVERSAL_COST
                    + SAH_INTERSECTION_COST * (lc as f32 * la + right_count as f32 * ra)
                        / parent_sa;

                if cost < best_cost {
                    best_cost = cost;
                    let split_pos =
                        centroid_bounds.min[axis] + extent * (i as f32 / SAH_BINS as f32);
                    best_split = Some((axis, split_pos));
                }
            }
        }

        best_split
    }

    fn partition_primitives(
        primitives: &mut [BuildPrimitive],
        start: usize,
        end: usize,
        axis: usize,
        split_pos: f32,
    ) -> usize {
        let mut left = start;
        let mut right = end;

        while left < right {
            if primitives[left].centroid[axis] < split_pos {
                left += 1;
            } else {
                right -= 1;
                primitives.swap(left, right);
            }
        }

        left
    }

    /// Get root AABB for TLAS instance creation
    #[inline]
    pub fn root_aabb(&self) -> &Aabb {
        &self.root_aabb
    }

    /// Get total memory size for GPU upload
    pub fn gpu_memory_size(&self) -> usize {
        self.vertices.len() * RTVertex::size()
            + self.indices.len() * 4
            + self.bvh_nodes.len() * size_of::<BvhNode>()
    }
}

// ============================================================================
// TLAS - Top Level Acceleration Structure
// ============================================================================

pub struct Tlas {
    pub instances: Vec<TlasInstance>,
    pub bvh_nodes: Vec<BvhNode>,
    pub dirty: bool,
    refit_quality: f32, // Track quality degradation for auto-rebuild
    id_to_index: HashMap<u32, usize>,
}

impl Tlas {
    pub fn new() -> Self {
        Self {
            instances: vec![],
            bvh_nodes: vec![],
            dirty: true,
            refit_quality: 1.0,
            id_to_index: HashMap::new(),
        }
    }

    /// Full SAH rebuild of TLAS (expensive but optimal)
    pub fn rebuild(&mut self, instances: Vec<TlasInstance>) {
        self.instances = instances;
        self.dirty = false;
        self.refit_quality = 1.0;

        if self.instances.is_empty() {
            self.bvh_nodes.clear();
            return;
        }

        let mut primitives: Vec<BuildPrimitive> = self
            .instances
            .iter()
            .enumerate()
            .map(|(i, inst)| {
                let aabb = Aabb::new(inst.aabb_min, inst.aabb_max);
                BuildPrimitive {
                    aabb,
                    centroid: aabb.centroid(),
                    index: i as u32,
                }
            })
            .collect();

        self.bvh_nodes.clear();
        let mut ordered_indices = Vec::new();
        let end = primitives.len();
        Self::build_recursive(
            &mut self.bvh_nodes,
            &mut primitives,
            0,
            end,
            &mut ordered_indices,
        );

        // Reorder instances to match BVH leaf order (improves cache coherence)
        let old_instances = std::mem::take(&mut self.instances);
        self.instances = ordered_indices
            .iter()
            .map(|&i| old_instances[i as usize])
            .collect();
        self.rebuild_id_map();
    }

    fn build_recursive(
        nodes: &mut Vec<BvhNode>,
        primitives: &mut [BuildPrimitive],
        start: usize,
        end: usize,
        ordered_indices: &mut Vec<u32>,
    ) -> u32 {
        let node_idx = nodes.len() as u32;
        nodes.push(BvhNode {
            aabb_min: [0.0; 3],
            _pad0: 0.0,
            aabb_max: [0.0; 3],
            _pad1: 0.0,
            child_or_tri_offset: 0,
            meta: 0,
            tri_count_or_right: 0,
            _pad2: 0,
        });

        let mut bounds = Aabb::EMPTY;
        for prim in &primitives[start..end] {
            bounds.expand(&prim.aabb);
        }

        let count = end - start;

        // TLAS uses single-instance leaves for simplicity
        if count <= 1 {
            let first_instance = ordered_indices.len() as u32;
            for prim in &primitives[start..end] {
                ordered_indices.push(prim.index);
            }

            nodes[node_idx as usize] = BvhNode {
                aabb_min: bounds.min,
                _pad0: 0.0,
                aabb_max: bounds.max,
                _pad1: 0.0,
                child_or_tri_offset: first_instance,
                meta: BvhNode::make_leaf(),
                tri_count_or_right: count as u32,
                _pad2: 0,
            };
            return node_idx;
        }

        // Find best split
        let mut centroid_bounds = Aabb::EMPTY;
        for prim in &primitives[start..end] {
            centroid_bounds.expand_point(prim.centroid);
        }

        let split = Self::find_best_split_tlas(&primitives[start..end], &bounds, &centroid_bounds);

        let mid = if let Some((axis, split_pos)) = split {
            Self::partition(primitives, start, end, axis, split_pos)
        } else {
            start + count / 2
        };

        let mid = mid.clamp(start + 1, end - 1);

        let left = Self::build_recursive(nodes, primitives, start, mid, ordered_indices);
        let right = Self::build_recursive(nodes, primitives, mid, end, ordered_indices);

        nodes[node_idx as usize] = BvhNode {
            aabb_min: bounds.min,
            _pad0: 0.0,
            aabb_max: bounds.max,
            _pad1: 0.0,
            child_or_tri_offset: left,
            meta: BvhNode::make_inner(),
            tri_count_or_right: right,
            _pad2: 0,
        };

        node_idx
    }

    fn find_best_split_tlas(
        primitives: &[BuildPrimitive],
        bounds: &Aabb,
        centroid_bounds: &Aabb,
    ) -> Option<(usize, f32)> {
        const TLAS_BINS: usize = 8;

        let parent_sa = bounds.surface_area();
        if parent_sa <= 0.0 {
            return None;
        }

        let mut best_cost = f32::MAX;
        let mut best_split = None;

        for axis in 0..3 {
            let extent = centroid_bounds.max[axis] - centroid_bounds.min[axis];
            if extent <= f32::EPSILON {
                continue;
            }

            let mut bins = [SahBin::default(); TLAS_BINS];

            for prim in primitives {
                let offset = (prim.centroid[axis] - centroid_bounds.min[axis]) / extent;
                let bin_idx = ((offset * TLAS_BINS as f32) as usize).min(TLAS_BINS - 1);
                bins[bin_idx].count += 1;
                bins[bin_idx].bounds.expand(&prim.aabb);
            }

            let mut left_area = [0.0f32; TLAS_BINS - 1];
            let mut left_count = [0u32; TLAS_BINS - 1];
            let mut cb = Aabb::EMPTY;
            let mut cc = 0u32;

            for i in 0..(TLAS_BINS - 1) {
                if bins[i].bounds.is_valid() {
                    cb.expand(&bins[i].bounds);
                }
                cc += bins[i].count;
                left_area[i] = cb.surface_area();
                left_count[i] = cc;
            }

            let mut rb = Aabb::EMPTY;
            let mut rc = 0u32;

            for i in (1..TLAS_BINS).rev() {
                if bins[i].bounds.is_valid() {
                    rb.expand(&bins[i].bounds);
                }
                rc += bins[i].count;

                let lc = left_count[i - 1];
                if lc == 0 || rc == 0 {
                    continue;
                }

                let cost = 1.0
                    + (lc as f32 * left_area[i - 1] + rc as f32 * rb.surface_area()) / parent_sa;

                if cost < best_cost {
                    best_cost = cost;
                    let sp = centroid_bounds.min[axis] + extent * (i as f32 / TLAS_BINS as f32);
                    best_split = Some((axis, sp));
                }
            }
        }

        best_split
    }

    fn partition(
        primitives: &mut [BuildPrimitive],
        start: usize,
        end: usize,
        axis: usize,
        split_pos: f32,
    ) -> usize {
        let mut left = start;
        let mut right = end;
        while left < right {
            if primitives[left].centroid[axis] < split_pos {
                left += 1;
            } else {
                right -= 1;
                primitives.swap(left, right);
            }
        }
        left
    }

    /// Fast refit: propagate updated AABBs from leaves up (O(n))
    /// Use when instances move but count stays same
    pub fn refit(&mut self, updated_instances: &[TlasInstance]) {
        if updated_instances.len() != self.instances.len() {
            self.rebuild(updated_instances.to_vec());
            return;
        }
        if self.bvh_nodes.is_empty() {
            return;
        }

        // Update instance transforms/AABBs by instance_id (order-independent input)
        for inst in updated_instances {
            if let Some(&dst) = self.id_to_index.get(&inst.instance_id) {
                self.instances[dst] = *inst;
            }
        }

        // Bottom-up refit (unchanged)
        let old_root_sa =
            Aabb::new(self.bvh_nodes[0].aabb_min, self.bvh_nodes[0].aabb_max).surface_area();

        Self::refit_recursive(&mut self.bvh_nodes, &self.instances, 0);

        let new_root_sa =
            Aabb::new(self.bvh_nodes[0].aabb_min, self.bvh_nodes[0].aabb_max).surface_area();

        if old_root_sa > 0.0 {
            self.refit_quality *= (old_root_sa / new_root_sa).min(1.0);
        }
    }

    fn refit_recursive(nodes: &mut [BvhNode], instances: &[TlasInstance], idx: usize) -> Aabb {
        let node = nodes[idx];

        if node.is_leaf() {
            let first = node.child_or_tri_offset as usize;
            let count = node.tri_count_or_right as usize;

            let mut bounds = Aabb::EMPTY;
            for i in first..(first + count).min(instances.len()) {
                bounds.expand_point(instances[i].aabb_min);
                bounds.expand_point(instances[i].aabb_max);
            }

            nodes[idx].aabb_min = bounds.min;
            nodes[idx].aabb_max = bounds.max;
            bounds
        } else {
            let left_idx = node.child_or_tri_offset as usize;
            let right_idx = node.tri_count_or_right as usize;

            let left_bounds = Self::refit_recursive(nodes, instances, left_idx);
            let right_bounds = Self::refit_recursive(nodes, instances, right_idx);

            let mut bounds = Aabb::EMPTY;
            bounds.expand(&left_bounds);
            bounds.expand(&right_bounds);

            nodes[idx].aabb_min = bounds.min;
            nodes[idx].aabb_max = bounds.max;
            bounds
        }
    }

    /// Check if refit quality has degraded enough to warrant rebuild
    pub fn needs_rebuild(&self) -> bool {
        self.refit_quality < 0.7 || self.dirty
    }

    /// Get refit quality (1.0 = perfect, lower = degraded)
    pub fn quality(&self) -> f32 {
        self.refit_quality
    }

    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    pub fn node_count(&self) -> usize {
        self.bvh_nodes.len()
    }

    fn rebuild_id_map(&mut self) {
        self.id_to_index.clear();
        self.id_to_index.reserve(self.instances.len());
        for (i, inst) in self.instances.iter().enumerate() {
            self.id_to_index.insert(inst.instance_id, i);
        }
    }
}

impl Default for Tlas {
    fn default() -> Self {
        Self::new()
    }
}
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BvhNode {
    pub aabb_min: [f32; 3],
    pub _pad0: f32,
    pub aabb_max: [f32; 3],
    pub _pad1: f32,
    pub child_or_tri_offset: u32,
    pub meta: u32,
    pub tri_count_or_right: u32,
    pub _pad2: u32,
}

impl BvhNode {
    #[inline]
    pub fn is_leaf(&self) -> bool {
        (self.meta & 1) == 1
    }
    #[inline]
    pub fn make_inner() -> u32 {
        0
    }
    #[inline]
    pub fn make_leaf() -> u32 {
        1
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TlasInstance {
    pub transform: [[f32; 4]; 4],
    pub inv_transform: [[f32; 4]; 4],

    pub blas_root: u32,
    pub instance_id: u32,
    pub _pad0: [u32; 2],

    pub aabb_min: [f32; 3],
    pub _pad1: f32,
    pub aabb_max: [f32; 3],
    pub _pad2: f32,
}

impl TlasInstance {
    pub fn size() -> usize {
        size_of::<TlasInstance>()
    }

    /// Create instance from transform matrix and BLAS
    pub fn new(
        transform: [[f32; 4]; 4],
        blas_root_aabb: &Aabb,
        blas_root: u32,
        instance_id: u32,
    ) -> Self {
        let m = Mat4::from_cols_array_2d(&transform);
        let inv = m.inverse();
        let inv_transform = inv.to_cols_array_2d();

        let world_aabb = blas_root_aabb.transform(&transform);

        Self {
            transform,
            inv_transform,
            blas_root,
            instance_id,
            _pad0: [0; 2],
            aabb_min: world_aabb.min,
            _pad1: 0.0,
            aabb_max: world_aabb.max,
            _pad2: 0.0,
        }
    }

    /// Update transform and recompute world AABB
    pub fn update_transform(&mut self, transform: [[f32; 4]; 4], blas_root_aabb: &Aabb) {
        let m = Mat4::from_cols_array_2d(&transform);
        let inv = m.inverse();

        self.transform = transform;
        self.inv_transform = inv.to_cols_array_2d();

        let world_aabb = blas_root_aabb.transform(&transform);
        self.aabb_min = world_aabb.min;
        self.aabb_max = world_aabb.max;
    }
}

// ============================================================================
// AABB Helper
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    pub const EMPTY: Self = Self {
        min: [f32::MAX, f32::MAX, f32::MAX],
        max: [f32::MIN, f32::MIN, f32::MIN],
    };

    #[inline]
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn is_valid(&self) -> bool {
        self.min[0] <= self.max[0] && self.min[1] <= self.max[1] && self.min[2] <= self.max[2]
    }

    #[inline]
    pub fn expand_point(&mut self, p: [f32; 3]) {
        self.min[0] = self.min[0].min(p[0]);
        self.min[1] = self.min[1].min(p[1]);
        self.min[2] = self.min[2].min(p[2]);
        self.max[0] = self.max[0].max(p[0]);
        self.max[1] = self.max[1].max(p[1]);
        self.max[2] = self.max[2].max(p[2]);
    }

    #[inline]
    pub fn expand(&mut self, other: &Aabb) {
        self.min[0] = self.min[0].min(other.min[0]);
        self.min[1] = self.min[1].min(other.min[1]);
        self.min[2] = self.min[2].min(other.min[2]);
        self.max[0] = self.max[0].max(other.max[0]);
        self.max[1] = self.max[1].max(other.max[1]);
        self.max[2] = self.max[2].max(other.max[2]);
    }

    #[inline]
    pub fn surface_area(&self) -> f32 {
        let d = [
            (self.max[0] - self.min[0]).max(0.0),
            (self.max[1] - self.min[1]).max(0.0),
            (self.max[2] - self.min[2]).max(0.0),
        ];
        2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])
    }

    #[inline]
    pub fn centroid(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Transform AABB by 4x4 column-major matrix (fast 8-corner transform)
    pub fn transform(&self, m: &[[f32; 4]; 4]) -> Aabb {
        // Optimized: compute transformed center + extent
        let center = self.centroid();
        let extent = [
            (self.max[0] - self.min[0]) * 0.5,
            (self.max[1] - self.min[1]) * 0.5,
            (self.max[2] - self.min[2]) * 0.5,
        ];

        // Transform center
        let new_center = [
            m[0][0] * center[0] + m[1][0] * center[1] + m[2][0] * center[2] + m[3][0],
            m[0][1] * center[0] + m[1][1] * center[1] + m[2][1] * center[2] + m[3][1],
            m[0][2] * center[0] + m[1][2] * center[1] + m[2][2] * center[2] + m[3][2],
        ];

        // Compute new extent using absolute values of rotation matrix
        let new_extent = [
            m[0][0].abs() * extent[0] + m[1][0].abs() * extent[1] + m[2][0].abs() * extent[2],
            m[0][1].abs() * extent[0] + m[1][1].abs() * extent[1] + m[2][1].abs() * extent[2],
            m[0][2].abs() * extent[0] + m[1][2].abs() * extent[1] + m[2][2].abs() * extent[2],
        ];

        Aabb {
            min: [
                new_center[0] - new_extent[0],
                new_center[1] - new_extent[1],
                new_center[2] - new_extent[2],
            ],
            max: [
                new_center[0] + new_extent[0],
                new_center[1] + new_extent[1],
                new_center[2] + new_extent[2],
            ],
        }
    }
}
