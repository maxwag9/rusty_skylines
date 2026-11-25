use crate::data::Settings;
use std::collections::{HashMap, HashSet};
use std::{fs, mem};
use std::path::{Path, PathBuf};
use anyhow::Context;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use rusty_skylines_utils::positions::{chunk_size, ChunkCoord, WorldPos, LocalPos};
use serde::Deserialize;
use tracing::{error, warn};
use wgpu::{Buffer, BufferAddress, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Device, Face, IndexFormat, Queue, RenderPass, VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu_render_manager::compute_system::{BufferSet, ComputePipelineOptions};
use wgpu_render_manager::generator::{MipmapMode, TextureCompressionMode, TextureKey, TextureParams};
use wgpu_render_manager::pipelines::{FragmentOption, PipelineOptions};
use wgpu_render_manager::renderer::RenderManager;
use crate::helpers::modpack::ModManager;
use crate::renderer::pipelines::Pipelines;
use crate::renderer::shadows::{shadow_bias_for_cascade, shadow_pipeline_options};
use crate::world::camera::camera::Camera;
use crate::world::props::props::{ArchetypeId, PropBindingResource, PropInstanceId, Props};
use crate::world::props::uniform_layout::{UniformFieldYaml, UniformLayout, UniformYaml};
use crate::world::terrain::terrain_subsystem::{extract_frustum_planes, Terrain};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PropVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
    pub uv: [f32; 2],
    pub texture_id: u32,
}

impl PropVertex {
    pub fn layout<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            array_stride: size_of::<PropVertex>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &[
                VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: VertexFormat::Float32x3,
                }, // position
                VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: VertexFormat::Float32x3,
                }, // normal
                VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: VertexFormat::Float32x4,
                }, // color
                VertexAttribute {
                    offset: 40,
                    shader_location: 3,
                    format: VertexFormat::Float32x2,
                }, // uv
                // texture_id
                VertexAttribute {
                    offset: 48,
                    shader_location: 4,
                    format: VertexFormat::Uint32,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GpuPropInstance {
    pub chunk_xz: [i32; 2],  // 8 bytes
    pub local_pos: [f32; 3], // 12 bytes

    pub scale: f32,        // 4 bytes
    pub rotation_sin: f32, // 4 bytes
    pub rotation_cos: f32, // 4 bytes

    pub seed: f32,          // 4 bytes
    pub wind_strength: f32, // 4 bytes

    pub color: [f32; 4], // 16 bytes
}

impl GpuPropInstance {
    pub fn new(
        chunk_coord: ChunkCoord,
        local_pos: LocalPos,
        scale: f32,
        rotation: f32,
        color: [f32; 4],
        seed: f32,
        wind_strength: f32,
    ) -> Self {
        Self {
            chunk_xz: chunk_coord.as_slice(),
            local_pos: local_pos.as_slice(),
            scale,
            rotation_sin: rotation.sin(),
            rotation_cos: rotation.cos(),
            seed,
            wind_strength,
            color,
        }
    }

    pub fn layout<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            array_stride: size_of::<GpuPropInstance>() as BufferAddress,
            step_mode: VertexStepMode::Instance,
            attributes: &[
                VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: VertexFormat::Sint32x2,
                },
                VertexAttribute {
                    offset: 8,
                    shader_location: 6,
                    format: VertexFormat::Float32x3,
                },
                VertexAttribute {
                    offset: 20,
                    shader_location: 7,
                    format: VertexFormat::Float32,
                },
                VertexAttribute {
                    offset: 24,
                    shader_location: 8,
                    format: VertexFormat::Float32,
                },
                VertexAttribute {
                    offset: 28,
                    shader_location: 9,
                    format: VertexFormat::Float32,
                },
                VertexAttribute {
                    offset: 32,
                    shader_location: 10,
                    format: VertexFormat::Float32,
                },
                VertexAttribute {
                    offset: 36,
                    shader_location: 11,
                    format: VertexFormat::Float32,
                },
                VertexAttribute {
                    offset: 40,
                    shader_location: 12,
                    format: VertexFormat::Float32x4,
                },
            ],
        }
    }
}
#[repr(C, align(16))]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuPropCullParams {
    camera_chunk: [i32; 2],
    _pad0: [u32; 2],
    camera_local: [f32; 4],
    chunk_size: f32,
    bounds_radius: f32,
    instance_count: u32,
    capacity: u32,
    lod_distances_squared: [f32; 4],
    frustum_planes: [[f32; 4]; 6],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuDrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct GpuInstanceBuffer {
    buffer: Buffer,
    count: u32,
    capacity: u32,
}

struct GpuVisibleBuffer {
    buffers: [Buffer; 4],
    capacity: u32,
}

pub struct PropsRenderer {
    archetypes: Vec<Option<PropRenderArchetype>>,
    prev_models: HashMap<u64, [[f32; 4]; 4]>,
    chunk_gpu_instance_buffers: HashMap<ChunkCoord, HashMap<ArchetypeId, (Buffer, u32)>>,
    gpu_global_instances: HashMap<ArchetypeId, GpuInstanceBuffer>,
    gpu_visible_instances: HashMap<ArchetypeId, GpuVisibleBuffer>,
    gpu_indirect_args: HashMap<ArchetypeId, Buffer>,
    gpu_cull_params: HashMap<ArchetypeId, Buffer>,
    gpu_active_chunks: HashSet<ChunkCoord>,
    world_instance_revision: u64,
    world_archetype_revision: u64,
}

impl PropsRenderer {
    pub fn new() -> Self {
        Self {
            archetypes: vec![],
            prev_models: HashMap::new(),
            chunk_gpu_instance_buffers: HashMap::new(),
            gpu_global_instances: HashMap::new(),
            gpu_visible_instances: HashMap::new(),
            gpu_indirect_args: HashMap::new(),
            gpu_cull_params: HashMap::new(),
            gpu_active_chunks: HashSet::new(),
            world_instance_revision: 0,
            world_archetype_revision: 0,
        }
    }

    fn sync_world(&mut self, props: &Props) {
        if self.world_archetype_revision != props.archetype_revision {
            self.clear_instance_data();
            self.archetypes.clear();

            self.world_archetype_revision = props.archetype_revision;
            self.world_instance_revision = props.instance_revision;
        } else if self.world_instance_revision != props.instance_revision {
            self.clear_instance_data();
            self.world_instance_revision = props.instance_revision;
        }
    }
    fn instance_key(
        chunk: ChunkCoord, // TODO: Use prop chunk coord, not random-ass given chunkcoord
        archetype_id: ArchetypeId,
        id: Option<PropInstanceId>,
    ) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        chunk.hash(&mut hasher);
        archetype_id.hash(&mut hasher);
        id.hash(&mut hasher);
        hasher.finish()
    }

    pub fn upload_instances(
        &mut self,
        props: &mut Props,
        device: &Device,
        queue: &Queue,
        camera: &Camera,
        terrain: &Terrain,
        render_manager: &mut RenderManager,
        mod_manager: &ModManager,
    ) {
        self.initialize_gpu_geometry(
            props,
            device,
            queue,
            render_manager,
        );
        let active_chunks: HashSet<ChunkCoord> =
            terrain.chunks.keys().copied().collect();

        let changed_chunks: Vec<ChunkCoord> = active_chunks
            .symmetric_difference(&self.gpu_active_chunks)
            .copied()
            .collect();

        let mut affected_archetypes = HashSet::new();

        for coord in changed_chunks {
            if let Some(chunk) = props.chunks.get_mut(&coord) {
                let archetypes: Vec<ArchetypeId> =
                    chunk.archetype_instances.keys().copied().collect();

                for archetype_id in archetypes {
                    affected_archetypes.insert(archetype_id);

                    if active_chunks.contains(&coord) {
                        chunk.dirty_archetypes.insert(archetype_id);
                    }
                }

                if active_chunks.contains(&coord) {
                    props.dirty_chunks.insert(coord);
                } else {
                    let gpu_buffers = self.chunk_gpu_instance_buffers.entry(coord).or_default();
                    gpu_buffers.clear();
                    chunk.dirty_archetypes.clear();
                }
            }
        }

        self.gpu_active_chunks = active_chunks.clone();

        let dirty_chunks = mem::take(&mut props.dirty_chunks);

        for coord in dirty_chunks {
            let Some(chunk) = props.chunks.get_mut(&coord) else {
                continue;
            };

            let dirty: Vec<ArchetypeId> =
                chunk.dirty_archetypes.drain().collect();

            if !active_chunks.contains(&coord) {
                affected_archetypes.extend(dirty);
                let gpu_buffers = self.chunk_gpu_instance_buffers.entry(coord).or_default();
                gpu_buffers.clear();
                continue;
            }

            for archetype_id in dirty {
                affected_archetypes.insert(archetype_id);

                let Some(instances) =
                    chunk.archetype_instances.get(&archetype_id)
                else {
                    let gpu_buffers = self.chunk_gpu_instance_buffers.entry(coord).or_default();
                    gpu_buffers.remove(&archetype_id);
                    continue;
                };

                if instances.is_empty() {
                    let gpu_buffers = self.chunk_gpu_instance_buffers.entry(coord).or_default();
                    gpu_buffers.remove(&archetype_id);
                    continue;
                }

                let mut gpu_instances =
                    Vec::with_capacity(instances.len());

                for &id in instances {
                    let Some(inst) = props.prop_instances
                        .get(id as usize)
                        .and_then(|v| v.as_ref())
                    else {
                        continue;
                    };

                    gpu_instances.push(GpuPropInstance::new(
                        inst.pos.chunk,
                        inst.pos.local,
                        inst.scale,
                        inst.rotation_y_rad,
                        inst.color,
                        inst.seed as f32 / u32::MAX as f32,
                        inst.wind_strength,
                    ));
                }

                let count = gpu_instances.len() as u32;
                let gpu_buffers = self.chunk_gpu_instance_buffers.entry(coord).or_default();
                if count == 0 {
                    gpu_buffers.remove(&archetype_id);
                    continue;
                }

                let bytes = bytemuck::cast_slice(&gpu_instances);

                let recreate = match gpu_buffers.get(&archetype_id) {
                    None => true,
                    Some((_, capacity)) => *capacity < count,
                };

                if recreate {
                    let capacity = count.max(1) * 2;

                    let buffer = device.create_buffer(&BufferDescriptor {
                        label: Some("Chunk Prop Shadow Instances"),
                        size: capacity as BufferAddress
                            * size_of::<GpuPropInstance>() as BufferAddress,
                        usage: BufferUsages::VERTEX
                            | BufferUsages::STORAGE
                            | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    gpu_buffers.insert(archetype_id, (buffer, count));
                }

                if let Some((buffer, stored_count)) =
                    gpu_buffers.get_mut(&archetype_id)
                {
                    queue.write_buffer(buffer, 0, bytes);
                    *stored_count = count;
                }
            }
        }

        for archetype_id in affected_archetypes {
            let mut gpu_instances = Vec::new();

            for (coord, chunk) in &props.chunks {
                if !active_chunks.contains(coord) {
                    continue;
                }

                let Some(ids) =
                    chunk.archetype_instances.get(&archetype_id)
                else {
                    continue;
                };

                for &id in ids {
                    let Some(inst) = props.prop_instances
                        .get(id as usize)
                        .and_then(|v| v.as_ref())
                    else {
                        continue;
                    };

                    gpu_instances.push(GpuPropInstance::new(
                        inst.pos.chunk,
                        inst.pos.local,
                        inst.scale,
                        inst.rotation_y_rad,
                        inst.color,
                        inst.seed as f32 / u32::MAX as f32,
                        inst.wind_strength,
                    ));
                }
            }

            if gpu_instances.is_empty() {
                self.gpu_global_instances.remove(&archetype_id);
                self.gpu_visible_instances.remove(&archetype_id);
                self.gpu_indirect_args.remove(&archetype_id);
                self.gpu_cull_params.remove(&archetype_id);
                continue;
            }

            let count = gpu_instances.len() as u32;
            let input_capacity = count.max(1) * 2;

            let recreate_input = self
                .gpu_global_instances
                .get(&archetype_id)
                .map(|v| v.capacity < count)
                .unwrap_or(true);

            if recreate_input {
                let buffer = device.create_buffer(&BufferDescriptor {
                    label: Some("Global Prop Instances"),
                    size: input_capacity as BufferAddress
                        * size_of::<GpuPropInstance>() as BufferAddress,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                self.gpu_global_instances.insert(
                    archetype_id,
                    GpuInstanceBuffer {
                        buffer,
                        count,
                        capacity: input_capacity,
                    },
                );
            }

            let input = self.gpu_global_instances.get_mut(&archetype_id).unwrap();
            input.count = count;

            queue.write_buffer(
                &input.buffer,
                0,
                bytemuck::cast_slice(&gpu_instances),
            );

            let recreate_visible = self
                .gpu_visible_instances
                .get(&archetype_id)
                .map(|v| v.capacity < count)
                .unwrap_or(true);

            if recreate_visible {
                let capacity = count.max(1);

                let buffers = std::array::from_fn(|lod| {
                    device.create_buffer(&BufferDescriptor {
                        label: Some(match lod {
                            0 => "Visible Prop LOD 0",
                            1 => "Visible Prop LOD 1",
                            2 => "Visible Prop LOD 2",
                            _ => "Visible Prop LOD 3",
                        }),
                        size: capacity as BufferAddress
                            * size_of::<GpuPropInstance>() as BufferAddress,
                        usage: BufferUsages::STORAGE | BufferUsages::VERTEX,
                        mapped_at_creation: false,
                    })
                });

                self.gpu_visible_instances.insert(
                    archetype_id,
                    GpuVisibleBuffer {
                        buffers,
                        capacity,
                    },
                );
            }

            self.gpu_indirect_args
                .entry(archetype_id)
                .or_insert_with(|| {
                    device.create_buffer(&BufferDescriptor {
                        label: Some("Prop Indirect Draw Arguments"),
                        size: 4 * size_of::<GpuDrawIndexedIndirectArgs>() as BufferAddress,
                        usage: BufferUsages::STORAGE
                            | BufferUsages::INDIRECT
                            | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                });

            self.gpu_cull_params
                .entry(archetype_id)
                .or_insert_with(|| {
                    device.create_buffer(&BufferDescriptor {
                        label: Some("Prop Culling Parameters"),
                        size: size_of::<GpuPropCullParams>() as BufferAddress,
                        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                });
        }

        let Some(cull_shader) =
            mod_manager.resource_path("shaders/compute/props_cull.wgsl")
        else {
            error!("[Props] Missing props_cull.wgsl");
            return;
        };

        let eye = camera.eye_world();

        let frustum_planes = extract_frustum_planes(camera.view_proj())
            .map(|p| [p.normal.x, p.normal.y, p.normal.z, p.d]);

        let archetype_ids: Vec<ArchetypeId> =
            self.gpu_global_instances.keys().copied().collect();

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("GPU Prop Culling"),
        });

        for archetype_id in archetype_ids {
            let Some(global) = self.gpu_global_instances.get(&archetype_id) else {
                continue;
            };

            let source_buffer = global.buffer.clone();
            let count = global.count;

            let Some(visible) =
                self.gpu_visible_instances.get(&archetype_id)
            else {
                continue;
            };

            let visible_buffers = visible.buffers.clone();
            let visible_capacity = visible.capacity;

            let Some(args) = self.gpu_indirect_args.get(&archetype_id) else {
                continue;
            };

            let args_buffer = args.clone();

            let Some(params_buffer) =
                self.gpu_cull_params.get(&archetype_id)
            else {
                continue;
            };

            let params_buffer = params_buffer.clone();

            let Some(archetype) = self
                .archetypes
                .get(archetype_id as usize)
                .and_then(|v| v.as_ref())
            else {
                continue;
            };

            let mut bounds_radius = 0.1f32;
            let mut draw_args = [
                GpuDrawIndexedIndirectArgs::zeroed();
                4
            ];

            for level in 0..4 {
                if let Some(mesh) = archetype.get_lod(level) {
                    draw_args[level as usize].index_count = mesh.index_count;
                    bounds_radius = bounds_radius.max(
                        mesh.bounds.0.length() + mesh.bounds.1
                    );
                }
            }
            let lod_distances_squared = std::array::from_fn(|index| {
                if index == 3 {
                    f32::MAX
                } else {
                    props.definitions
                        .get(&archetype_id)
                        .and_then(|definition| definition.lods[index + 1].as_ref())
                        .map(|lod| lod.distance.max(0.0).powi(2))
                        .unwrap_or(f32::MAX)
                }
            });
            let params = GpuPropCullParams {
                camera_chunk: eye.chunk.as_slice(),
                _pad0: [0; 2],
                camera_local: [
                    eye.local.x,
                    eye.local.y,
                    eye.local.z,
                    0.0,
                ],
                chunk_size: chunk_size() as f32,
                bounds_radius,
                instance_count: count,
                capacity: visible_capacity,
                lod_distances_squared,
                frustum_planes,
            };

            queue.write_buffer(
                &args_buffer,
                0,
                bytemuck::cast_slice(&draw_args),
            );

            queue.write_buffer(
                &params_buffer,
                0,
                bytemuck::bytes_of(&params),
            );

            render_manager.compute(
                Some(&mut encoder),
                "Prop Frustum Culling and LOD",
                vec![],
                vec![],
                cull_shader,
                ComputePipelineOptions {
                    dispatch_size: [count.div_ceil(64), 1, 1],
                },
                &[
                    BufferSet::from_storage(&source_buffer, true),
                    BufferSet::from_storage(&visible_buffers[0], false),
                    BufferSet::from_storage(&visible_buffers[1], false),
                    BufferSet::from_storage(&visible_buffers[2], false),
                    BufferSet::from_storage(&visible_buffers[3], false),
                    BufferSet::from_storage(&args_buffer, false),
                    BufferSet::from_uniform(&params_buffer),
                ],
            );
        }

        queue.submit(Some(encoder.finish()));

        props.visible_generated_prop_count = 0;
        props.visible_manual_prop_count = 0;

        for visible_chunk in &terrain.visible {
            if let Some(chunk) = props.chunks.get(&visible_chunk.chunk_coord) {
                props.visible_generated_prop_count += chunk.generated_count;
                props.visible_manual_prop_count += chunk.manual_count;
            }
        }
    }
    fn clear_instance_data(&mut self) {
        self.prev_models.clear();
        self.chunk_gpu_instance_buffers.clear();
        self.gpu_global_instances.clear();
        self.gpu_visible_instances.clear();
        self.gpu_indirect_args.clear();
        self.gpu_cull_params.clear();
        self.gpu_active_chunks.clear();
    }

    /// Normal rendering pass

    pub fn render<'a>(
        &mut self,
        render_manager: &mut RenderManager,
        pass: &mut RenderPass<'a>,
        shader_path: &Path,
        opts: PipelineOptions,
        _camera: &'a Camera,
        _terrain: &'a Terrain,
        pipelines: &Pipelines,
        _settings: &Settings,
    ) {
        for (index, slot) in self.archetypes.iter().enumerate() {
            let archetype_id = index as ArchetypeId;

            let Some(archetype) = slot.as_ref() else {
                continue;
            };

            let Some(visible) = self.gpu_visible_instances.get(&archetype_id) else {
                continue;
            };

            let Some(args) = self.gpu_indirect_args.get(&archetype_id) else {
                continue;
            };

            for level in 0..4u32 {
                let Some(mesh) = archetype.get_lod(level) else {
                    continue;
                };

                render_manager.render(
                    &archetype.texture_keys,
                    shader_path,
                    &opts,
                    &[
                        &pipelines.buffers.camera,
                        &mesh.alpha_cutoff_buffer,
                    ],
                    pass,
                );

                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_vertex_buffer(
                    1,
                    visible.buffers[level as usize].slice(..),
                );
                pass.set_index_buffer(
                    mesh.index_buffer.slice(..),
                    IndexFormat::Uint32,
                );

                pass.draw_indexed_indirect(
                    args,
                    level as BufferAddress
                        * size_of::<GpuDrawIndexedIndirectArgs>() as BufferAddress,
                );
            }
        }
    }


    /// Shadow pass rendering.  DUH!!
    pub fn render_shadows<'a>(
        &'a self,
        props: &'a Props,
        render_manager: &mut RenderManager,
        pass: &mut RenderPass<'a>,
        camera: &'a Camera,
        terrain: &'a Terrain,
        pipelines: &Pipelines,
        settings: &Settings,
        shadow_mat_buffer: &'a Buffer,
        cascade_idx: usize,
        mod_manager: &ModManager,
    ) {
        let eye = camera.eye_world();

        let bias = shadow_bias_for_cascade(
            cascade_idx,
            pipelines.resources.csm_shadows.texels[cascade_idx],
            settings.reversed_depth_z,
        );

        let Some(shader) = mod_manager.resource_path("shaders/props_shadows.wgsl") else {
            error!("[Renderer] Missing shader 'shaders/props_shadows.wgsl'");
            return;
        };

        let opts = shadow_pipeline_options(
            settings,
            bias,
            vec![Some(PropVertex::layout()), Some(GpuPropInstance::layout())],
            Face::Back,
            FragmentOption::Default { targets: vec![] },
        );

        for visible_chunk in terrain.visible.iter() {
            let coord = visible_chunk.chunk_coord;

            let Some(chunk) = props.chunks.get(&coord) else {
                continue;
            };

            let instance_terrain_height = terrain.get_height_at(eye, true);
            let dist_sq = eye.distance_squared(WorldPos::new(
                coord,
                LocalPos::new(0.0, instance_terrain_height, 0.0),
            ));
            let dist = dist_sq.sqrt();

            for (&archetype_id, instances) in &chunk.archetype_instances {
                if instances.is_empty() {
                    continue;
                }

                let Some(archetype) = self
                    .archetypes
                    .get(archetype_id as usize)
                    .and_then(|a| a.as_ref())
                else {
                    continue;
                };

                let lod_level = if let Some(definition) = props.definitions.get(&archetype_id) {
                    let mut chosen = 0u32;

                    for level in 0..4u32 {
                        if let Some(lod) = definition.lods[level as usize].as_ref() {
                            if dist >= lod.distance as f64 {
                                chosen = level;
                            }
                        }
                    }

                    chosen
                } else {
                    0
                };

                let Some(mesh) = archetype.get_lod(lod_level) else {
                    continue;
                };

                let Some((inst_buf, count)) = self
                    .chunk_gpu_instance_buffers
                    .get(&coord)
                    .and_then(|buffers| buffers.get(&archetype_id))
                else {
                    continue;
                };

                render_manager.render(
                    &archetype.texture_keys,
                    shader,
                    &opts,
                    &[
                        &pipelines.buffers.camera,
                        shadow_mat_buffer,
                        &mesh.alpha_cutoff_buffer,
                    ],
                    pass,
                );

                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, inst_buf.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..*count);
            }
        }
    }

    pub fn initialize_gpu_geometry(
        &mut self,
        props: &Props,
        device: &Device,
        queue: &Queue,
        render_manager: &mut RenderManager,
    ) {
        if self.world_archetype_revision == props.archetype_revision {
            return;
        }

        self.sync_world(props);

        let archetype_count = props.definitions.keys()
            .map(|id| *id as usize + 1)
            .max()
            .unwrap_or(0);

        self.archetypes.clear();
        self.archetypes.resize_with(archetype_count, || None);

        for (&archetype_id, definition) in &props.definitions {
            let mut texture_keys: [TextureKey; 4] =
                std::array::from_fn(|_| TextureKey::notex());

            for texture in &definition.textures {
                if !(1..=4).contains(&texture.slot) {
                    continue;
                }

                texture_keys[(texture.slot - 1) as usize] = TextureKey::new(
                    texture.shader.as_path(),
                    texture.params.build(),
                    texture.resolution,
                    texture.mipmap_mode.clone(),
                    texture.compression.clone(),
                );
            }

            self.archetypes[archetype_id as usize] = Some(PropRenderArchetype {
                lod0: None,
                lod1: None,
                lod2: None,
                lod3: None,
                texture_keys,
            });
        }

        let mut owned_buffers: Vec<Buffer> = Vec::new();

        let mut pending_meshes: Vec<(
            ArchetypeId,
            usize,
            usize,
            usize,
            Option<Buffer>,
            Option<u32>,
            String,
        )> = Vec::new();

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Prop Geometry Generation"),
        });

        for (archetype_id, definition) in &props.definitions {
            let prop_label = format!(
                "Archetype {:?}, shader {:?}",
                archetype_id,
                definition.shader
            );

            for level in 0..4usize {
                let Some(lod) = definition.lods[level].as_ref() else {
                    continue;
                };

                let mut buffer_sets: Vec<BufferSet> = Vec::new();
                let mut vertex_idx: Option<usize> = None;
                let mut index_idx: Option<usize> = None;
                let mut counter_idx: Option<usize> = None;

                let mut sorted_bindings = lod.bindings.clone();
                sorted_bindings.sort_by_key(|b| b.binding);

                for binding in &sorted_bindings {
                    match &binding.resource {
                        PropBindingResource::Uniform(layout) => {
                            let label = format!(
                                "Prop {:?} LOD{} Uniform b{}",
                                archetype_id,
                                level,
                                binding.binding
                            );

                            match layout.create_buffer(device, &label) {
                                Ok(buf) => {
                                    buffer_sets.push(BufferSet::from_uniform(&buf));
                                    owned_buffers.push(buf);
                                }
                                Err(err) => {
                                    error!(
                                    "[Props] {} LOD {} binding {}: failed to create uniform buffer: {}",
                                    prop_label,
                                    level,
                                    binding.binding,
                                    err
                                );
                                }
                            }
                        }
                        PropBindingResource::Storage {
                            access,
                            format: _,
                            source: _,
                            data: _,
                            capacity,
                            usage,
                        } => {
                            let cap = capacity
                                .unwrap_or(1024)
                                .max(if binding.binding == 3 { 4 } else { 1 });

                            let size = (cap * 4).max(4);

                            let mut usages =
                                BufferUsages::STORAGE | BufferUsages::COPY_DST;

                            for usage_entry in usage {
                                match usage_entry.as_str() {
                                    "vertex" => usages |= BufferUsages::VERTEX,
                                    "index" => usages |= BufferUsages::INDEX,
                                    "copy_src" => usages |= BufferUsages::COPY_SRC,
                                    "copy_dst" => usages |= BufferUsages::COPY_DST,
                                    "storage" => {}
                                    other => {
                                        error!(
                                        "[Props] {} LOD {} binding {}: unknown buffer usage '{}'",
                                        prop_label,
                                        level,
                                        binding.binding,
                                        other
                                    );
                                    }
                                }
                            }

                            if binding.binding == 3 {
                                usages |= BufferUsages::COPY_SRC;
                            }

                            let buf = device.create_buffer(&BufferDescriptor {
                                label: Some(&format!(
                                    "Prop {:?} LOD{} Storage b{}",
                                    archetype_id,
                                    level,
                                    binding.binding
                                )),
                                size,
                                usage: usages,
                                mapped_at_creation: false,
                            });

                            let idx = owned_buffers.len();

                            if usages.contains(BufferUsages::VERTEX)
                                && vertex_idx.is_none()
                            {
                                vertex_idx = Some(idx);
                            }

                            if usages.contains(BufferUsages::INDEX)
                                && index_idx.is_none()
                            {
                                index_idx = Some(idx);
                            }

                            if binding.binding == 3 {
                                counter_idx = Some(idx);
                            }

                            let read_only = access == "read";
                            buffer_sets.push(BufferSet::from_storage(&buf, read_only));
                            owned_buffers.push(buf);
                        }
                    }
                }

                let mut missing = Vec::new();

                if vertex_idx.is_none() {
                    missing.push("VERTEX");
                }

                if index_idx.is_none() {
                    missing.push("INDEX");
                }

                if counter_idx.is_none() && definition.index_count.is_none() {
                    missing.push("COUNTERS (binding 3) or configured index_count");
                }

                if !missing.is_empty() {
                    error!(
                    "[Props] {} LOD {}: missing {}. Skipping geometry generation.",
                    prop_label,
                    level,
                    missing.join(" and ")
                );
                    continue;
                }

                let (Some(v_idx), Some(i_idx)) = (vertex_idx, index_idx) else {
                    continue;
                };

                render_manager.compute(
                    Some(&mut encoder),
                    "Prop Geometry Generation",
                    vec![],
                    vec![],
                    &definition.shader,
                    ComputePipelineOptions {
                        dispatch_size: definition.workgroups,
                    },
                    &buffer_sets,
                );

                let counter_readback = if let Some(counter_idx) = counter_idx {
                    let staging = device.create_buffer(&BufferDescriptor {
                        label: Some("Prop Geometry Counter Readback"),
                        size: 16,
                        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    encoder.copy_buffer_to_buffer(
                        &owned_buffers[counter_idx],
                        0,
                        &staging,
                        0,
                        16,
                    );

                    Some(staging)
                } else {
                    None
                };

                pending_meshes.push((
                    *archetype_id,
                    level,
                    v_idx,
                    i_idx,
                    counter_readback,
                    definition.index_count,
                    prop_label.clone(),
                ));
            }
        }

        let submission_index = queue.submit(Some(encoder.finish()));

        let mut readback_receivers = Vec::new();

        for (pending_index, pending) in pending_meshes.iter().enumerate() {
            let Some(staging) = pending.4.as_ref() else {
                continue;
            };

            let (sender, receiver) = std::sync::mpsc::channel();

            staging.slice(..).map_async(
                wgpu::MapMode::Read,
                move |result| {
                    let _ = sender.send(result);
                },
            );

            readback_receivers.push((pending_index, receiver));
        }

        let mut generated_counters: Vec<Option<[u32; 4]>> =
            vec![None; pending_meshes.len()];

        if !readback_receivers.is_empty() {
            match device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: None,
            }) {
                Ok(_) => {
                    for (pending_index, receiver) in readback_receivers {
                        match receiver.recv() {
                            Ok(Ok(())) => {
                                let Some(staging) =
                                    pending_meshes[pending_index].4.as_ref()
                                else {
                                    continue;
                                };

                                let Ok(mapped) = staging.slice(..).get_mapped_range() else { continue };
                                let values: &[u32] =
                                    bytemuck::cast_slice(&mapped[..]);

                                if values.len() >= 4 {
                                    generated_counters[pending_index] = Some([
                                        values[0],
                                        values[1],
                                        values[2],
                                        values[3],
                                    ]);
                                } else {
                                    error!(
                                    "[Props] {}: invalid geometry counter readback",
                                    pending_meshes[pending_index].6
                                );
                                }

                                drop(mapped);
                                staging.unmap();
                            }
                            Ok(Err(err)) => {
                                error!(
                                "[Props] {}: failed to map geometry counters: {}",
                                pending_meshes[pending_index].6,
                                err
                            );
                            }
                            Err(err) => {
                                error!(
                                "[Props] {}: geometry counter readback channel failed: {}",
                                pending_meshes[pending_index].6,
                                err
                            );
                            }
                        }
                    }
                }
                Err(err) => {
                    error!(
                    "[Props] Failed waiting for geometry generation: {}",
                    err
                );
                }
            }
        }

        for (
            pending_index,
            (
                archetype_id,
                level,
                v_idx,
                i_idx,
                _counter_readback,
                configured_index_count,
                prop_label,
            ),
        ) in pending_meshes.into_iter().enumerate()
        {
            let vertex_capacity = (
                owned_buffers[v_idx].size()
                    / size_of::<PropVertex>() as BufferAddress
            ) as u32;

            let index_capacity = (
                owned_buffers[i_idx].size()
                    / size_of::<u32>() as BufferAddress
            ) as u32;

            let counters = generated_counters[pending_index];

            if let Some(counters) = counters {
                if counters[2] != 0 {
                    error!(
                    "[Props] {} LOD {}: geometry generation reported an error. Vertices: {}, indices: {}, commands: {}, capacities: {} vertices / {} indices",
                    prop_label,
                    level,
                    counters[0],
                    counters[1],
                    counters[3],
                    vertex_capacity,
                    index_capacity
                );
                    continue;
                }

                tracing::info!(
                "[Props] {} LOD {}: generated {} vertices and {} indices. Capacities: {} vertices / {} indices",
                prop_label,
                level,
                counters[0],
                counters[1],
                vertex_capacity,
                index_capacity
            );
            }

            let mut index_count = counters
                .map(|c| c[1])
                .or(configured_index_count)
                .unwrap_or(0);

            if index_count > index_capacity {
                error!(
                "[Props] {} LOD {}: generated index count {} exceeds capacity {}",
                prop_label,
                level,
                index_count,
                index_capacity
            );
                index_count = index_capacity;
            }

            if index_count == 0 {
                error!(
                "[Props] {} LOD {}: index count is zero. Skipping mesh.",
                prop_label,
                level
            );
                continue;
            }

            let alpha_cutoff = props
                .definitions
                .get(&archetype_id)
                .and_then(|definition| definition.lods[level].as_ref())
                .map(|lod| lod.alpha_cutoff)
                .unwrap_or(0.25);

            let alpha_cutoff_data = [alpha_cutoff, if level == 3 { 1.0 } else { 0.0 }, 0.0, 0.0];

            let alpha_cutoff_buffer = device.create_buffer_init(&BufferInitDescriptor {
                label: Some(&format!(
                    "Prop {:?} LOD{} Alpha Cutoff",
                    archetype_id,
                    level
                )),
                contents: bytemuck::cast_slice(&alpha_cutoff_data),
                usage: BufferUsages::UNIFORM,
            });

            let mesh = Mesh {
                vertex_buffer: owned_buffers[v_idx].clone(),
                index_buffer: owned_buffers[i_idx].clone(),
                index_count,
                bounds: (
                    Vec3::ZERO,
                    props.definitions
                        .get(&archetype_id)
                        .map(|definition| definition.bounds_radius)
                        .unwrap_or(5.0),
                ),
                alpha_cutoff_buffer,
            };

            if let Some(archetype) = self
                .archetypes
                .get_mut(archetype_id as usize)
                .and_then(|v| v.as_mut())
            {
                match level {
                    0 => archetype.lod0 = Some(mesh),
                    1 => archetype.lod1 = Some(mesh),
                    2 => archetype.lod2 = Some(mesh),
                    3 => archetype.lod3 = Some(mesh),
                    _ => {}
                }
            } else {
                error!(
                "[Props] {} LOD {}: generated mesh could not be assigned because the archetype does not exist",
                prop_label,
                level
            );
            }
        }

        self.world_archetype_revision = props.archetype_revision;
    }
}

pub struct PropRenderArchetype {
    lod0: Option<Mesh>,
    lod1: Option<Mesh>,
    lod2: Option<Mesh>,
    lod3: Option<Mesh>,
    texture_keys: [TextureKey; 4], // 4 slots for textures in the shader.
}

impl PropRenderArchetype {
    /// Get mesh for requested LOD level, falling back to nearest available
    pub fn get_lod(&self, level: u32) -> Option<&Mesh> {
        match level {
            0 => self
                .lod0
                .as_ref()
                .or(self.lod1.as_ref())
                .or(self.lod2.as_ref())
                .or(self.lod3.as_ref()),
            1 => self
                .lod1
                .as_ref()
                .or(self.lod0.as_ref())
                .or(self.lod2.as_ref())
                .or(self.lod3.as_ref()),
            2 => self
                .lod2
                .as_ref()
                .or(self.lod1.as_ref())
                .or(self.lod3.as_ref())
                .or(self.lod0.as_ref()),
            _ => self
                .lod3
                .as_ref()
                .or(self.lod2.as_ref())
                .or(self.lod1.as_ref())
                .or(self.lod0.as_ref()),
        }
    }
}
pub struct Mesh {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,
    pub alpha_cutoff_buffer: Buffer,
    pub bounds: (Vec3, f32), // center, radius
}
