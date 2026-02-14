use crate::helpers::positions::WorldPos;
use crate::renderer::pipelines::Pipelines;
use crate::renderer::ray_tracing::structs::{Aabb, Blas, BvhNode, RTVertex, Tlas, TlasInstance};
use crate::world::camera::Camera;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
// rt_subsystem.rs
use std::mem::size_of;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue};
use wgpu_render_manager::compute_system::BufferSet;
// ============================================================================
// GPU Buffer Management
// ============================================================================

pub struct BlasGpu {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub bvh_buffer: Buffer,
    pub vertex_count: u32,
    pub index_count: u32,
    pub node_count: u32,
}

pub struct TlasGpu {
    pub instance_buffer: Buffer,
    pub bvh_buffer: Buffer,
    pub instance_count: u32,
    pub node_count: u32,
}

// ============================================================================
// RT Subsystem - Manages All Ray Tracing Resources
// ============================================================================

pub struct RTSubsystem {
    // GPU resources
    pub car_blas_gpu: Option<BlasGpu>,
    pub tlas_gpu: Option<TlasGpu>,

    // CPU acceleration structures
    pub car_blas: Option<Blas>,
    pub tlas: Tlas,

    // Buffer capacity tracking (avoid frequent reallocs)
    instance_capacity: usize,
    tlas_node_capacity: usize,

    // Stats
    frames_since_rebuild: u32,
    pub buffer_sets: Option<Vec<BufferSet>>,
    rt_params_buffer: Buffer,
}

impl RTSubsystem {
    pub fn new(device: &Device) -> Self {
        let rt_params = RTParams {
            max_distance: 200.0, // e.g. 200.0
            normal_bias: 0.05,   // e.g. 0.05
            origin_bias: 0.02,   // e.g. 0.02
            ndotl_fade: 0.10,    // e.g. 0.10  (smooth ramp around N·L=0)

            soft_start: 5.0,          // e.g. 5.0   (close blockers = hard/dark)
            soft_end: 80.0,           // e.g. 80.0  (far blockers = softer/lighter)
            far_hit_visibility: 0.85, // e.g. 0.85  (visibility when blocked very far)
            _pad0: 0.0,
        };
        let rt_params_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("RT Params Uniforms"),
            contents: bytemuck::bytes_of(&rt_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        Self {
            car_blas_gpu: None,
            tlas_gpu: None,
            car_blas: None,
            tlas: Tlas::new(),
            instance_capacity: 0,
            tlas_node_capacity: 0,
            frames_since_rebuild: 0,
            buffer_sets: None,
            rt_params_buffer,
        }
    }

    /// Initialize car mesh BLAS (call once at startup)
    pub fn init_car_blas(
        &mut self,
        device: &Device,
        queue: &Queue,
        vertices: &[[f32; 3]],
        indices: &[u32],
    ) {
        let blas = Blas::build(vertices.to_vec(), indices.to_vec());

        if blas.bvh_nodes.is_empty() {
            log::warn!("BLAS build produced no nodes");
            return;
        }

        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("RT Car BLAS Vertices"),
            size: (blas.vertices.len() * RTVertex::size()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("RT Car BLAS Indices"),
            size: (blas.indices.len() * 4).max(4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bvh_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("RT Car BLAS BVH Nodes"),
            size: (blas.bvh_nodes.len() * size_of::<BvhNode>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Upload data
        queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&blas.vertices));
        queue.write_buffer(&index_buffer, 0, bytemuck::cast_slice(&blas.indices));
        queue.write_buffer(&bvh_buffer, 0, bytemuck::cast_slice(&blas.bvh_nodes));

        log::info!(
            "BLAS initialized: {} vertices, {} indices, {} BVH nodes",
            blas.vertices.len(),
            blas.indices.len(),
            blas.bvh_nodes.len()
        );

        self.car_blas_gpu = Some(BlasGpu {
            vertex_buffer,
            index_buffer,
            bvh_buffer,
            vertex_count: blas.vertices.len() as u32,
            index_count: blas.indices.len() as u32,
            node_count: blas.bvh_nodes.len() as u32,
        });

        self.car_blas = Some(blas);
    }

    /// Update TLAS with car instances for current frame
    ///
    /// # Arguments
    /// * `instances` - Pre-built TlasInstance list with transforms and world AABBs
    /// * `force_rebuild` - Force full SAH rebuild instead of refit
    pub fn update_tlas(
        &mut self,
        device: &Device,
        queue: &Queue,
        pipelines: &Pipelines,
        instances: Vec<TlasInstance>,
        force_rebuild: bool,
    ) {
        if instances.is_empty() {
            self.tlas.instances.clear();
            self.tlas.bvh_nodes.clear();
            return;
        }

        self.frames_since_rebuild += 1;

        // Decide: rebuild vs refit
        let topology_changed = instances.len() != self.tlas.instances.len();
        let quality_degraded = self.tlas.needs_rebuild();
        let periodic_rebuild = self.frames_since_rebuild > 120; // Every ~2 seconds at 60fps

        if force_rebuild || topology_changed || quality_degraded || periodic_rebuild {
            self.tlas.rebuild(instances);
            self.frames_since_rebuild = 0;
        } else {
            self.tlas.refit(&instances);
        }

        self.upload_tlas(device, queue, pipelines);
    }

    fn upload_tlas(&mut self, device: &Device, queue: &Queue, pipelines: &Pipelines) {
        let instances = &self.tlas.instances;
        let nodes = &self.tlas.bvh_nodes;

        if instances.is_empty() {
            return;
        }

        let inst_bytes = instances.len() * TlasInstance::size();
        let node_bytes = nodes.len() * size_of::<BvhNode>();

        // Check if buffers need resize
        let need_inst_resize = inst_bytes > self.instance_capacity;
        let need_node_resize = node_bytes > self.tlas_node_capacity;

        if need_inst_resize || need_node_resize || self.tlas_gpu.is_none() {
            // Grow with 2x headroom to avoid frequent reallocs
            if need_inst_resize {
                self.instance_capacity = (inst_bytes * 2).max(4096);
            }
            if need_node_resize {
                self.tlas_node_capacity = (node_bytes * 2).max(2048);
            }

            let instance_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("RT TLAS Instances"),
                size: self.instance_capacity as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bvh_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("RT TLAS BVH Nodes"),
                size: self.tlas_node_capacity as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            self.tlas_gpu = Some(TlasGpu {
                instance_buffer,
                bvh_buffer,
                instance_count: instances.len() as u32,
                node_count: nodes.len() as u32,
            });
        }

        // Upload
        if let Some(tlas_gpu) = &mut self.tlas_gpu {
            tlas_gpu.instance_count = instances.len() as u32;
            tlas_gpu.node_count = nodes.len() as u32;

            queue.write_buffer(
                &tlas_gpu.instance_buffer,
                0,
                bytemuck::cast_slice(instances),
            );
            queue.write_buffer(&tlas_gpu.bvh_buffer, 0, bytemuck::cast_slice(nodes));
        }
        if need_inst_resize || need_node_resize || self.tlas_gpu.is_none() {
            self.ensure_rt_buffer_sets(pipelines, false);
        }
    }

    /// Get car BLAS root AABB (for instance creation)
    pub fn car_blas_aabb(&self) -> Option<&Aabb> {
        self.car_blas.as_ref().map(|b| b.root_aabb())
    }

    /// Check if RT is ready for rendering
    pub fn is_ready(&self) -> bool {
        self.car_blas_gpu.is_some() && self.tlas_gpu.is_some() && !self.tlas.instances.is_empty()
    }

    /// Get statistics for debug display
    pub fn stats(&self) -> RTStats {
        RTStats {
            blas_nodes: self
                .car_blas
                .as_ref()
                .map(|b| b.bvh_nodes.len())
                .unwrap_or(0),
            blas_triangles: self
                .car_blas
                .as_ref()
                .map(|b| b.indices.len() / 3)
                .unwrap_or(0),
            tlas_instances: self.tlas.instances.len(),
            tlas_nodes: self.tlas.bvh_nodes.len(),
            tlas_quality: self.tlas.quality(),
        }
    }

    /// Create buffer sets for the compute system (all read-only storage buffers)
    pub fn ensure_rt_buffer_sets(&mut self, pipelines: &Pipelines, just_ensure: bool) -> bool {
        if self.buffer_sets.is_some() && just_ensure {
            return true;
        }

        let Some(blas) = self.car_blas_gpu.as_ref() else {
            return false;
        };
        let Some(tlas) = self.tlas_gpu.as_ref() else {
            return false;
        };

        self.buffer_sets = Some(vec![
            BufferSet {
                buffer: blas.vertex_buffer.clone(),
                read_only: true,
            },
            BufferSet {
                buffer: blas.index_buffer.clone(),
                read_only: true,
            },
            BufferSet {
                buffer: blas.bvh_buffer.clone(),
                read_only: true,
            },
            BufferSet {
                buffer: tlas.instance_buffer.clone(),
                read_only: true,
            },
            BufferSet {
                buffer: tlas.bvh_buffer.clone(),
                read_only: true,
            },
            BufferSet::from_uniform(&pipelines.buffers.camera.clone()),
            BufferSet::from_uniform(&self.rt_params_buffer.clone()),
        ]);

        true
    }

    /// Get cached buffer sets, or None if not ready
    pub fn get_buffer_sets(&mut self, pipelines: &Pipelines) -> Option<&Vec<BufferSet>> {
        if !self.ensure_rt_buffer_sets(pipelines, true) {
            return None;
        }
        Some(self.buffer_sets.as_ref().unwrap())
    }
}

#[derive(Debug, Clone)]
pub struct RTStats {
    pub blas_nodes: usize,
    pub blas_triangles: usize,
    pub tlas_instances: usize,
    pub tlas_nodes: usize,
    pub tlas_quality: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RTParams {
    max_distance: f32, // e.g. 200.0
    normal_bias: f32,  // e.g. 0.05
    origin_bias: f32,  // e.g. 0.02
    ndotl_fade: f32,   // e.g. 0.10  (smooth ramp around N·L=0)

    soft_start: f32,         // e.g. 5.0   (close blockers = hard/dark)
    soft_end: f32,           // e.g. 80.0  (far blockers = softer/lighter)
    far_hit_visibility: f32, // e.g. 0.85  (visibility when blocked very far)
    _pad0: f32,
}

pub fn make_render_space_transform(
    world_pos: WorldPos,
    rot: Quat,
    scale: Vec3,
    camera: &Camera,
) -> Mat4 {
    let eye = camera.eye_world();
    let t_render: Vec3 = world_pos.to_render_pos(eye, camera.chunk_size);
    Mat4::from_scale_rotation_translation(scale, rot, t_render)
}

pub fn build_render_space_instances(
    objects: impl Iterator<
        Item = (
            u32, /*instance_id*/
            WorldPos,
            Quat,
            Vec3, /*scale*/
            u32,  /*blas_root*/
        ),
    >,
    camera: &Camera,
    blas_root_aabb: &Aabb,
) -> Vec<TlasInstance> {
    objects
        .map(|(instance_id, world_pos, rot, scale, blas_root)| {
            let m = make_render_space_transform(world_pos, rot, scale, camera);
            TlasInstance::new(m.to_cols_array_2d(), blas_root_aabb, blas_root, instance_id)
        })
        .collect()
}
