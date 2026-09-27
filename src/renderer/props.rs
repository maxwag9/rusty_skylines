use crate::data::Settings;
use crate::helpers::modpack::ModManager;
use crate::helpers::positions::{ChunkCoord, LocalPos, WorldPos};
use crate::renderer::pipelines::Pipelines;
use crate::renderer::shadows::{shadow_bias_for_cascade, shadow_pipeline_options};
use crate::ui::input::Input;
use crate::world::camera::Camera;
use crate::world::terrain::terrain_subsystem::{CursorMode, Terrain};
use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec2, Vec3};
use revision::revisioned;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::f32::consts::{PI, TAU};
use std::mem;
use std::path::Path;
use tracing::error;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    Buffer, BufferAddress, BufferDescriptor, BufferUsages, Device, Face, IndexFormat, Queue,
    RenderPass, VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode,
};
use wgpu_render_manager::generator::{MipmapMode, TextureKey, TextureParams};
use wgpu_render_manager::pipelines::{FragmentOption, PipelineOptions};
use wgpu_render_manager::renderer::RenderManager;

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
// struct InstanceInput {
//     @location(5) chunk_xz: vec2<i32>,
//     @location(6) local_pos: vec3<f32>,
//     @location(7) scale: f32,
//
//     @location(8) rotation_sin: f32,
//     @location(9) rotation_cos: f32,
//     @location(10) seed: f32,
//     @location(11) wind_strength: f32,
//
//     @location(12) color: vec4<f32>,
// };
#[derive(Clone)]
pub struct PropInstance {
    pub id: Option<PropInstanceId>,
    pub archetype_id: Option<ArchetypeId>,
    pub pos: WorldPos,
    pub rotation_y_rad: f32,
    pub scale: f32,
    pub color: [f32; 4],
    pub seed: u32,
    pub variant: u16,
    pub wind_strength: f32,
    pub generated: bool, // true = procedurally placed by terrain gen (trees), regenerated deterministically, never saved
}

pub struct Mesh {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,
    pub bounds: (Vec3, f32), // center, radius
}

pub struct PropChunk {
    pub chunk_coord: ChunkCoord,
    pub archetype_instances: HashMap<ArchetypeId, Vec<PropInstanceId>>,
    pub gpu_instance_buffers: HashMap<ArchetypeId, (Buffer, u32)>,
    pub dirty_archetypes: HashSet<ArchetypeId>,
}

impl PropChunk {
    pub fn new(chunk_coord: ChunkCoord) -> Self {
        Self {
            chunk_coord,
            archetype_instances: HashMap::new(),
            gpu_instance_buffers: HashMap::new(),
            dirty_archetypes: HashSet::new(),
        }
    }

    pub fn remove_instance(&mut self, archetype_id: ArchetypeId, instance_id: PropInstanceId) {
        let Some(chunk_instances) = self.archetype_instances.get_mut(&archetype_id) else {
            // // It SHOULD panic, because instances in the instance list 100% have an archetype id Not relevant here, I JUST COPIED this text from the other function to here.
            return;
        };
        chunk_instances.retain(|&pid| pid != instance_id);
        self.dirty_archetypes.insert(archetype_id);
    }

    pub fn add_instance(&mut self, archetype_id: ArchetypeId, instance_id: PropInstanceId) {
        self.archetype_instances
            .entry(archetype_id)
            .or_default()
            .push(instance_id); // Do you want to hear something insane? Here: self.archetype_instances.entry(archetype_id).or_insert(vec![instance_id]);
        self.dirty_archetypes.insert(archetype_id);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SavePropChunk {
    pub chunk_coord: ChunkCoord,
    pub archetype_instances: HashMap<ArchetypeId, Vec<PropInstanceId>>,
}
#[derive(Clone)]
#[revisioned(revision = 1)]
pub struct SavedPropInstance {
    pub archetype: String,
    pub pos: WorldPos,
    pub rotation_y_rad: f32,
    pub scale: f32,
    pub color: [f32; 4],
    pub seed: u32,
    pub variant: u16,
    pub wind_strength: f32,
}
#[derive(Default, Clone)]
#[revisioned(revision = 1)]
pub struct SavedProps {
    pub instances: Vec<SavedPropInstance>,
}
pub type PropInstanceId = u64;
pub type ArchetypeId = u32;
pub struct Props {
    archetypes: Vec<Option<Archetype>>,
    archetypes_free: Vec<ArchetypeId>,
    archetype_to_id: HashMap<String, ArchetypeId>,
    prop_instances: Vec<Option<PropInstance>>,
    prop_instances_free: Vec<PropInstanceId>,
    pub chunks: HashMap<ChunkCoord, PropChunk>,
    pub prev_models: HashMap<u64, [[f32; 4]; 4]>,
    device: Device,
    dirty_chunks: HashSet<ChunkCoord>,
    generated_prop_count: usize,
    manual_prop_count: usize,
    visible_generated_prop_count: usize,
    visible_manual_prop_count: usize,
    preview_prop_id: Option<PropInstanceId>,
}

impl Props {
    pub fn new(device: &Device) -> Self {
        Self {
            archetypes: vec![],
            archetypes_free: vec![],
            archetype_to_id: HashMap::new(),
            prop_instances: vec![],
            prop_instances_free: vec![],
            chunks: HashMap::new(),
            prev_models: HashMap::new(),
            device: device.clone(),
            dirty_chunks: HashSet::new(),
            generated_prop_count: 0,
            manual_prop_count: 0,
            visible_generated_prop_count: 0,
            visible_manual_prop_count: 0,
            preview_prop_id: None,
        }
    }

    pub fn clear(&mut self) {
        self.archetypes.clear();
        self.archetypes_free.clear();
        self.archetype_to_id.clear();
        self.prop_instances.clear();
        self.prop_instances_free.clear();
        self.chunks.clear();
        self.prev_models.clear();
        self.dirty_chunks.clear();
    }

    #[inline]
    pub fn generated_prop_count(&self) -> usize {
        self.generated_prop_count
    }

    #[inline]
    pub fn manual_prop_count(&self) -> usize {
        self.manual_prop_count
    }

    #[inline]
    pub fn prop_count(&self) -> usize {
        self.generated_prop_count + self.manual_prop_count
    }

    #[inline]
    pub fn visible_generated_prop_count(&self) -> usize {
        self.visible_generated_prop_count
    }

    #[inline]
    pub fn visible_manual_prop_count(&self) -> usize {
        self.visible_manual_prop_count
    }

    #[inline]
    pub fn visible_prop_count(&self) -> usize {
        self.visible_generated_prop_count + self.visible_manual_prop_count
    }

    pub fn get_props(&self) -> SavedProps {
        let mut instances = Vec::new();

        for prop in self.prop_instances.iter().flatten() {
            if prop.generated {
                continue; // procedural, regenerates deterministically, don't bloat the save
            }

            let archetype_name = self.archetypes[prop.archetype_id.unwrap() as usize]
                .as_ref()
                .unwrap()
                .name
                .to_lowercase();

            instances.push(SavedPropInstance {
                archetype: archetype_name,
                pos: prop.pos,
                rotation_y_rad: prop.rotation_y_rad,
                scale: prop.scale,
                color: prop.color,
                seed: prop.seed,
                variant: prop.variant,
                wind_strength: prop.wind_strength,
            });
        }

        SavedProps { instances }
    }
    pub fn load_props(&mut self, saved: SavedProps, mod_manager: &ModManager) {
        self.clear();

        for prop in saved.instances {
            let key = &prop.archetype.to_lowercase();
            if !self.is_registered(key) {
                if let Some(archetype) = make_archetype(key, &self.device, mod_manager) {
                    self.register_archetype(archetype);
                }
            }
            let Some(archetype_id) = self.archetype_to_id.get(key).copied() else {
                continue;
            };
            let prop_instance = PropInstance {
                id: None,
                archetype_id: Some(archetype_id),
                pos: prop.pos,
                rotation_y_rad: prop.rotation_y_rad,
                scale: prop.scale,
                color: prop.color,
                seed: prop.seed,
                variant: prop.variant,
                wind_strength: prop.wind_strength,
                generated: false,
            };
            self.add_instance(prop_instance);
        }
    }
    pub fn register_archetype(&mut self, mut archetype: Archetype) -> ArchetypeId {
        let name = archetype.name.to_lowercase();
        archetype.name = name;
        let name = archetype.name.clone();
        let id = if let Some(id) = self.archetypes_free.pop() {
            self.archetypes[id as usize] = Some(archetype);
            id
        } else {
            let id = self.archetypes.len() as ArchetypeId;
            self.archetypes.push(Some(archetype));
            id
        };

        self.archetype_to_id.insert(name, id);

        id
    }
    pub fn get_archetype_id_for_name(&self, name: &str) -> Option<ArchetypeId> {
        self.archetype_to_id
            .get(name.to_lowercase().as_str())
            .copied()
    }
    pub fn is_registered(&self, key: impl Into<String>) -> bool {
        self.archetype_to_id.contains_key(&key.into())
    }

    pub fn add_instance(&mut self, mut instance: PropInstance) -> PropInstanceId {
        let chunk_coord = instance.pos.chunk;
        let chunk = self
            .chunks
            .entry(chunk_coord)
            .or_insert_with(|| PropChunk::new(chunk_coord));

        let chunk_instances = chunk
            .archetype_instances
            .entry(instance.archetype_id.unwrap())
            .or_default(); // It SHOULD panic to ensure I coded it correctly.

        if instance.generated {
            self.generated_prop_count += 1;
        } else {
            self.manual_prop_count += 1;
        }

        chunk
            .dirty_archetypes
            .insert(instance.archetype_id.unwrap());
        self.dirty_chunks.insert(chunk_coord);
        let id = if let Some(id) = self.prop_instances_free.pop() {
            instance.id = Some(id);
            self.prop_instances[id as usize] = Some(instance);
            id
        } else {
            let id = self.prop_instances.len() as PropInstanceId;
            instance.id = Some(id);
            self.prop_instances.push(Some(instance));
            id
        };

        chunk_instances.push(id);

        id
    }

    pub fn remove_instance(&mut self, id: PropInstanceId) -> bool {
        let Some(instance) = self
            .prop_instances
            .get(id as usize)
            .and_then(|p| p.as_ref())
        else {
            return false;
        };
        let chunk_coord = instance.pos.chunk;
        let chunk = self
            .chunks
            .entry(chunk_coord)
            .or_insert_with(|| PropChunk::new(chunk_coord));
        if instance.generated {
            debug_assert!(self.generated_prop_count > 0);
            self.generated_prop_count -= 1;
        } else {
            debug_assert!(self.manual_prop_count > 0);
            self.manual_prop_count -= 1;
        }

        let archetype_id = instance.archetype_id.unwrap();
        chunk.remove_instance(archetype_id, id);

        let slot = self.prop_instances.get_mut(id as usize).unwrap();

        self.prop_instances_free.push(id);
        self.dirty_chunks.insert(chunk_coord);

        if slot.is_none() {
            // Slot was empty anyway
            return false;
        }

        *slot = None;

        true
    }
    pub fn remove_instances(&mut self, ids: &[PropInstanceId]) {
        for &id in ids.iter() {
            self.remove_instance(id);
        }
    }
    #[inline]
    pub fn get_instance(&self, id: PropInstanceId) -> Option<&PropInstance> {
        self.prop_instances.get(id as usize)?.as_ref()
    }
    // NO get instance mut!! Edits must happen through helper functions here!

    /// Swaps all procedurally-generated instances (trees) for a chunk with a
    /// fresh batch. Idempotent: call it every time a chunk's vegetation is
    /// (re)computed -- LOD rebuild, reload, whatever -- and it will never
    /// accumulate duplicates, because the previous generation's instances for
    /// this chunk are dropped first. Manually-placed props (generated == false)
    /// in the same chunk, even of the same archetype, are left untouched.
    /// Single pass per archetype list, no per-id search, no position math.
    pub fn replace_generated_instances(
        &mut self,
        chunk_coord: ChunkCoord,
        new_instances: Vec<PropInstance>,
        archetypes: Vec<String>,
        mod_manager: &ModManager,
    ) {
        for archetype in archetypes.into_iter() {
            self.ensure_archetype(archetype, mod_manager);
        }

        let prop_instances = &mut self.prop_instances;
        let prop_instances_free = &mut self.prop_instances_free;
        let mut removed_count = 0usize;
        let mut touched_archetypes: Vec<ArchetypeId> = Vec::new();

        if let Some(chunk) = self.chunks.get_mut(&chunk_coord) {
            for (&archetype_id, ids) in chunk.archetype_instances.iter_mut() {
                let before_len = ids.len();

                ids.retain(|&id| {
                    let is_generated = prop_instances
                        .get(id as usize)
                        .and_then(|p| p.as_ref())
                        .map(|p| p.generated)
                        .unwrap_or(false);

                    if is_generated {
                        prop_instances[id as usize] = None;
                        prop_instances_free.push(id);
                    }

                    !is_generated
                });

                let removed_here = before_len - ids.len();
                if removed_here > 0 {
                    removed_count += removed_here;
                    touched_archetypes.push(archetype_id);
                }
            }

            // Mark every archetype we actually mutated as dirty, so
            // upload_instances rebuilds (or drops) its GPU buffer instead of
            // leaving stale instance data resident and rendering phantoms.
            for archetype_id in &touched_archetypes {
                chunk.dirty_archetypes.insert(*archetype_id);
            }
        }

        if removed_count > 0 {
            debug_assert!(self.generated_prop_count >= removed_count);
            self.generated_prop_count -= removed_count;
            // The chunk was mutated even if none of the new_instances below
            // happen to reuse a touched archetype — make sure upload_instances
            // actually visits it this frame regardless.
            self.dirty_chunks.insert(chunk_coord);
        }

        for mut inst in new_instances {
            inst.generated = true;
            self.add_instance(inst);
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
        device: &Device,
        queue: &Queue,
        camera: &Camera,
        terrain: &Terrain,
    ) {
        // let visible_chunks: Vec<ChunkCoord> = terrain
        //     .visible
        //     .iter()
        //     .map(|c| c.coords.chunk_coord)
        //     .collect();

        let dirty_chunks = mem::take(&mut self.dirty_chunks);

        for coord in dirty_chunks {
            let chunk = self
                .chunks
                .entry(coord)
                .or_insert_with(|| PropChunk::new(coord));

            if chunk.dirty_archetypes.is_empty() {
                continue;
            }

            let dirty: Vec<ArchetypeId> = chunk.dirty_archetypes.drain().collect();

            for archetype_id in dirty {
                let Some(instances) = chunk.archetype_instances.get(&archetype_id) else {
                    continue;
                };

                if instances.is_empty() {
                    chunk.gpu_instance_buffers.remove(&archetype_id);
                    continue;
                }

                let mut gpu_instances = Vec::with_capacity(instances.len());

                for &id in instances {
                    let Some(inst) = self
                        .prop_instances
                        .get(id as usize)
                        .and_then(|x| x.as_ref())
                    else {
                        continue;
                    };

                    let seed = inst.seed as f32 / u32::MAX as f32;

                    gpu_instances.push(GpuPropInstance::new(
                        inst.pos.chunk,
                        inst.pos.local,
                        inst.scale,
                        inst.rotation_y_rad,
                        inst.color,
                        seed,
                        inst.wind_strength,
                    ));
                }

                let count = gpu_instances.len() as u32;

                if count == 0 {
                    continue;
                }

                let bytes = bytemuck::cast_slice(&gpu_instances);

                let recreate = match chunk.gpu_instance_buffers.get(&archetype_id) {
                    None => true,
                    Some((_, capacity)) => *capacity < count,
                };

                if recreate {
                    let capacity = (count.max(1) * 2) as BufferAddress
                        * size_of::<GpuPropInstance>() as BufferAddress;

                    let buffer = device.create_buffer(&BufferDescriptor {
                        label: Some("prop_instance_buffer"),
                        size: capacity,
                        usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    chunk
                        .gpu_instance_buffers
                        .insert(archetype_id, (buffer, count));
                }

                if let Some((buffer, stored_count)) =
                    chunk.gpu_instance_buffers.get_mut(&archetype_id)
                {
                    queue.write_buffer(buffer, 0, bytes);
                    *stored_count = count;
                }
            }
        }
    }

    pub fn place_props(
        &mut self,
        terrain: &Terrain,
        input: &mut Input,
        device: &Device,
        mod_manager: &ModManager,
    ) {
        match &terrain.cursor.mode {
            CursorMode::Props => {
                let name = &terrain.cursor.prop_name;
                let name = &name.to_lowercase();
                if !self.is_registered(name) {
                    if let Some(archetype) = make_archetype(name, device, mod_manager) {
                        self.register_archetype(archetype);
                    }
                }
                if let Some(picked_point) = &terrain.last_picked {
                    if let Some(&archetype_id) = self.archetype_to_id.get(name) {
                        if input.action_repeat("Place Prop") {
                            self.add_instance(PropInstance {
                                id: None,
                                archetype_id: Some(archetype_id),
                                pos: picked_point.pos,
                                scale: rand::random_range(0.8..1.5),
                                rotation_y_rad: rand::random_range(0.0..5.0),
                                seed: rand::random(),
                                color: [1.0, 1.0, 1.0, 1.0],
                                wind_strength: 0.2,
                                variant: 0,
                                generated: false,
                            });
                        }
                        if self.preview_prop_id.is_none() {
                            let prop_instance = PropInstance {
                                id: None,
                                archetype_id: Some(archetype_id),
                                pos: picked_point.pos,
                                scale: rand::random_range(0.8..1.5),
                                rotation_y_rad: rand::random_range(0.0..5.0),
                                seed: rand::random(),
                                color: [1.0, 1.0, 1.0, 1.0],
                                wind_strength: 0.2,
                                variant: 0,
                                generated: false,
                            };
                            self.preview_prop_id =
                                Some(self.place_prop(name, prop_instance, mod_manager));
                        }
                        if let Some(id) = self.preview_prop_id {
                            if let Some(prop) = self
                                .prop_instances
                                .get_mut(id as usize)
                                .and_then(|x| x.as_mut())
                            {
                                self.dirty_chunks.insert(prop.pos.chunk);
                                let archetype_id = prop.archetype_id.unwrap();
                                let chunk = self
                                    .chunks
                                    .entry(prop.pos.chunk)
                                    .or_insert_with(|| PropChunk::new(prop.pos.chunk));
                                chunk.remove_instance(archetype_id, id);

                                prop.pos = picked_point.pos;
                                self.dirty_chunks.insert(prop.pos.chunk);
                                let chunk = self
                                    .chunks
                                    .entry(prop.pos.chunk)
                                    .or_insert_with(|| PropChunk::new(prop.pos.chunk));
                                chunk.add_instance(archetype_id, id);
                            }
                        }
                    }
                }
            }
            _ => return,
        }
    }

    pub fn place_prop(
        &mut self,
        archetype_name: impl Into<String>,
        mut prop_instance: PropInstance,
        mod_manager: &ModManager,
    ) -> PropInstanceId {
        let archetype_id = self.ensure_archetype(archetype_name, mod_manager);
        prop_instance.archetype_id = Some(archetype_id);
        self.add_instance(prop_instance)
    }
    pub fn move_multiple_offset(
        &mut self,
        prop_ids: &[PropInstanceId],
        old_center_pos: WorldPos,
        old_dir: Vec3,
        new_center_pos: WorldPos,
        new_dir: Vec3,
    ) {
        let old_xz = Vec2::new(old_dir.x, old_dir.z);
        let new_xz = Vec2::new(new_dir.x, new_dir.z);

        let delta_angle = if old_xz.length_squared() < 1e-10 || new_xz.length_squared() < 1e-10 {
            0.0f32
        } else {
            let old_angle = old_xz.y.atan2(old_xz.x);
            let new_angle = new_xz.y.atan2(new_xz.x);
            new_angle - old_angle
        };

        let (sin_a, cos_a) = delta_angle.sin_cos();

        for &id in prop_ids {
            if let Some(prop) = self
                .prop_instances
                .get_mut(id as usize)
                .and_then(|x| x.as_mut())
            {
                let offset = old_center_pos.delta_to(prop.pos);

                let rotated_offset = Vec3::new(
                    offset.x * cos_a - offset.z * sin_a,
                    offset.y,
                    offset.x * sin_a + offset.z * cos_a,
                );
                self.dirty_chunks.insert(prop.pos.chunk);
                let archetype_id = prop.archetype_id.unwrap();
                let chunk = self
                    .chunks
                    .entry(prop.pos.chunk)
                    .or_insert_with(|| PropChunk::new(prop.pos.chunk));
                chunk.remove_instance(archetype_id, id);

                prop.pos = new_center_pos.add_vec3(rotated_offset);

                prop.rotation_y_rad += delta_angle;

                self.dirty_chunks.insert(prop.pos.chunk);
                let chunk = self
                    .chunks
                    .entry(prop.pos.chunk)
                    .or_insert_with(|| PropChunk::new(prop.pos.chunk));
                chunk.add_instance(archetype_id, id);
            }
        }
    }
    pub fn ensure_archetype(
        &mut self,
        archetype_name: impl Into<String>,
        mod_manager: &ModManager,
    ) -> ArchetypeId {
        let key = &archetype_name.into().to_lowercase();
        if !self.is_registered(key) {
            if let Some(archetype) = make_archetype(key, &self.device, mod_manager) {
                self.register_archetype(archetype);
            }
        }
        self.archetype_to_id.get(key).copied().unwrap()
    }
    /// Normal rendering pass
    pub fn render<'a>(
        &mut self,
        render_manager: &mut RenderManager,
        pass: &mut RenderPass<'a>,
        shader_path: &Path,
        opts: PipelineOptions,
        camera: &'a Camera,
        terrain: &'a Terrain,
        pipelines: &Pipelines,
        settings: &Settings,
    ) {
        let eye = camera.eye_world();
        // let mut eye = WorldPos::default();
        // eye.local.y = eye_real.local.y;
        let terrain_height = terrain.get_height_at(eye, true);
        self.visible_generated_prop_count = 0;
        self.visible_manual_prop_count = 0;

        for visible_chunk in terrain.visible.iter() {
            let coord = visible_chunk.chunk_coord;

            let Some(chunk) = self.chunks.get(&coord) else {
                continue;
            };

            //let instance_terrain_height = terrain.get_height_at(eye, true);
            let dist = eye.distance_squared(WorldPos::new(
                coord,
                LocalPos::new(0.0, terrain_height, 0.0),
            ));
            let lod_level = select_lod(dist);

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
                let Some(mesh) = archetype.get_lod(lod_level) else {
                    continue;
                };
                let Some((inst_buf, count)) = chunk.gpu_instance_buffers.get(&archetype_id) else {
                    continue;
                };
                for &id in instances {
                    let inst = self.prop_instances[id as usize].as_ref().unwrap();

                    if inst.generated {
                        self.visible_generated_prop_count += 1;
                    } else {
                        self.visible_manual_prop_count += 1;
                    }
                }
                //println!("{:?}", archetype.texture_keys);
                render_manager.render(
                    &archetype.texture_keys,
                    shader_path,
                    &opts,
                    &[&pipelines.buffers.camera],
                    pass,
                );
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, inst_buf.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..*count);
            }
        }
    }

    /// Shadow pass rendering
    pub fn render_shadows<'a>(
        &'a self,
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
        //let mut visible_vertex_count: usize = 0;
        for visible_chunk in terrain.visible.iter() {
            let coord = visible_chunk.chunk_coord;

            let Some(chunk) = self.chunks.get(&coord) else {
                continue;
            };

            let instance_terrain_height = terrain.get_height_at(eye, true);
            let dist = eye.distance_squared(WorldPos::new(
                coord,
                LocalPos::new(0.0, instance_terrain_height, 0.0),
            ));
            let lod_level = select_lod(dist);

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
                let Some(mesh) = archetype.get_lod(lod_level) else {
                    continue;
                };
                let Some((inst_buf, count)) = chunk.gpu_instance_buffers.get(&archetype_id) else {
                    continue;
                };

                render_manager.render(
                    &archetype.texture_keys,
                    shader,
                    &opts,
                    &[&pipelines.buffers.camera, shadow_mat_buffer],
                    pass,
                );
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, inst_buf.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..*count);
                //visible_vertex_count += mesh.index_count as usize * *count as usize;
            }
        }
        //println!("Visible vertex count of props: {}", visible_vertex_count);
    }
}
struct Archetype {
    name: String,
    lod0: Option<Mesh>,
    lod1: Option<Mesh>,
    lod2: Option<Mesh>,
    lod3: Option<Mesh>,
    texture_keys: [TextureKey; 4], // 4 slots for textures in the shader.
}

impl Archetype {
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
/// LOD distance thresholds (in world units)
const LOD0_MAX_DIST: f64 = 30.0; // Full detail
const LOD1_MAX_DIST: f64 = 180.0; // Medium detail
const LOD2_MAX_DIST: f64 = 500.0; // Low detail

fn select_lod(dist2: f64) -> u32 {
    //return 3;
    if dist2 < LOD0_MAX_DIST * LOD0_MAX_DIST {
        0
    } else if dist2 < LOD1_MAX_DIST * LOD1_MAX_DIST {
        1
    } else if dist2 < LOD2_MAX_DIST * LOD2_MAX_DIST {
        2
    } else {
        3
    }
}

fn make_archetype(key: &str, device: &Device, mod_manager: &ModManager) -> Option<Archetype> {
    match key.to_lowercase().as_str() {
        "oak" | "oak_tree" => make_oak_tree(device, mod_manager),
        "pine" | "pine_tree" => make_pine_tree(device),
        _ => None,
    }
}

fn make_oak_tree(device: &Device, mod_manager: &ModManager) -> Option<Archetype> {
    let Some(leaves) = mod_manager.resource_path("shaders/textures/leaves.wgsl") else {
        error!("[Props] Missing shader 'shaders/textures/leaves.wgsl'");
        return None;
    };
    let Some(bark) = mod_manager.resource_path("shaders/textures/bark.wgsl") else {
        error!("[Props] Missing shader 'shaders/textures/bark.wgsl'");
        return None;
    };
    Some(Archetype {
        name: "oak".to_string(),
        lod0: Some(make_oak_lod(device, 0)),
        lod1: Some(make_oak_lod(device, 1)),
        lod2: Some(make_oak_lod(device, 2)),
        lod3: Some(make_oak_lod(device, 3)),
        texture_keys: [
            TextureKey::new(
                leaves,
                TextureParams {
                    color_primary: [0.22, 0.40, 0.12, 1.0],
                    color_secondary: [0.30, 0.50, 0.18, 1.0],
                    seed: 69,
                    scale: 1.5,
                    roughness: 0.8,
                    octaves: 0.6,
                    persistence: 0.5,
                    lacunarity: 0.3,
                    _pad0: 0.0,
                    _pad1: 0.0,
                },
                256,
                MipmapMode::AlphaPreserving,
            ),
            TextureKey::new(
                bark,
                TextureParams {
                    color_primary: [0.36, 0.26, 0.18, 1.0],
                    color_secondary: [0.25, 0.18, 0.12, 1.0],
                    seed: 69,
                    scale: 1.5,
                    roughness: 0.8,
                    octaves: 0.0,
                    persistence: 0.0,
                    lacunarity: 0.0,
                    _pad0: 0.0,
                    _pad1: 0.0,
                },
                256,
                MipmapMode::Generate,
            ),
            TextureKey::notex(),
            TextureKey::notex(),
        ],
    })
}

fn make_pine_tree(device: &Device) -> Option<Archetype> {
    Some(Archetype {
        name: "pine".to_string(),
        lod0: Some(make_pine_lod(device, 0)),
        lod1: Some(make_pine_lod(device, 1)),
        lod2: Some(make_pine_lod(device, 2)),
        lod3: Some(make_pine_lod(device, 3)),
        texture_keys: [
            TextureKey::notex(),
            TextureKey::notex(),
            TextureKey::notex(),
            TextureKey::notex(),
        ],
    })
}

fn make_pine_lod(device: &Device, lod: u32) -> Mesh {
    if lod >= 3 {
        let mut vertices: Vec<PropVertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        generate_tree_billboard_cross(
            Vec3::new(0.0, 3.5, 0.0),
            3.0,
            7.0,
            &mut vertices,
            &mut indices,
        );
        return create_mesh(device, &vertices, &indices);
    }

    let structure = pine_tree_structure();
    let render_params = LodRenderParams::pine_for_lod(lod);
    generate_tree_mesh(device, &structure, &render_params)
}

struct LSystemRule {
    from: char,
    to: &'static str,
}

#[derive(Clone)]
struct TurtleState {
    position: Vec3,
    direction: Vec3,
    right: Vec3,
    up: Vec3,
    thickness: f32,
    length: f32,
    depth: u32,
}

impl TurtleState {
    fn new(base_thickness: f32, base_length: f32) -> Self {
        Self {
            position: Vec3::ZERO,
            direction: Vec3::Y,
            right: Vec3::X,
            up: Vec3::NEG_Z,
            thickness: base_thickness,
            length: base_length,
            depth: 0,
        }
    }

    fn rotate_yaw(&mut self, angle: f32) {
        let rotation = Quat::from_axis_angle(self.up, angle);
        self.direction = rotation * self.direction;
        self.right = rotation * self.right;
    }

    fn rotate_pitch(&mut self, angle: f32) {
        let rotation = Quat::from_axis_angle(self.right, angle);
        self.direction = rotation * self.direction;
        self.up = rotation * self.up;
    }

    fn rotate_roll(&mut self, angle: f32) {
        let rotation = Quat::from_axis_angle(self.direction, angle);
        self.right = rotation * self.right;
        self.up = rotation * self.up;
    }
}

struct BranchSegment {
    start: Vec3,
    end: Vec3,
    start_radius: f32,
    end_radius: f32,
}

// Replaces individual LeafCard - one cluster = what was 3-5 leaves
struct LeafCluster {
    position: Vec3,
    direction: Vec3,
    right: Vec3,
    up: Vec3,
    size: f32,
}

struct SimpleRng(u32);

impl SimpleRng {
    fn new(seed: u32) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.0 >> 16) & 0x7FFF) as f32 / 32767.0
    }

    fn range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next() * (max - min)
    }
}

fn expand_lsystem(axiom: &str, rules: &[LSystemRule], iterations: u32) -> String {
    let mut current = axiom.to_string();

    for _ in 0..iterations {
        let mut next = String::with_capacity(current.len() * 2);
        for c in current.chars() {
            let replacement = rules
                .iter()
                .find(|r| r.from == c)
                .map(|r| r.to)
                .unwrap_or("");

            if replacement.is_empty() {
                next.push(c);
            } else {
                next.push_str(replacement);
            }
        }
        current = next;
    }

    current
}

fn interpret_lsystem(
    lsystem: &str,
    base_angle: f32,
    length_decay: f32,
    thickness_decay: f32,
    base_length: f32,
    base_thickness: f32,
    seed: u32,
    leaf_scale: f32,
) -> (Vec<BranchSegment>, Vec<LeafCluster>) {
    let mut branches = Vec::new();
    let mut leaves = Vec::new();
    let mut stack: Vec<TurtleState> = Vec::new();
    let mut state = TurtleState::new(base_thickness, base_length);
    let mut rng = SimpleRng::new(seed);

    for c in lsystem.chars() {
        let angle_variation = rng.range(0.65, 1.35);
        let length_variation = rng.range(0.8, 1.2);

        match c {
            'F' | 'G' => {
                state.rotate_pitch(rng.range(-0.12, 0.12));
                state.rotate_yaw(rng.range(-0.12, 0.12));

                let start = state.position;
                let start_radius = state.thickness;
                let actual_length = state.length * length_variation;

                state.position += state.direction * actual_length;

                // More aggressive taper, especially for thin branches
                let base_taper = rng.range(0.65, 0.78);
                // Extra taper for already-thin branches (makes ends properly thin)
                let thin_branch_factor = if start_radius < 0.05 {
                    0.75
                } else if start_radius < 0.08 {
                    0.85
                } else {
                    1.0
                };
                let end_radius = (start_radius * base_taper * thin_branch_factor).max(0.004);
                state.thickness = end_radius;

                branches.push(BranchSegment {
                    start,
                    end: state.position,
                    start_radius,
                    end_radius,
                });
            }
            'f' => {
                state.position += state.direction * state.length * length_variation;
            }
            '+' => state.rotate_yaw(base_angle * angle_variation),
            '-' => state.rotate_yaw(-base_angle * angle_variation),
            '&' => state.rotate_pitch(base_angle * angle_variation),
            '^' => state.rotate_pitch(-base_angle * angle_variation),
            '\\' => state.rotate_roll(base_angle * angle_variation + rng.range(-0.1, 0.1)),
            '/' => state.rotate_roll(-base_angle * angle_variation + rng.range(-0.1, 0.1)),
            '|' => state.rotate_yaw(PI),
            '[' => {
                stack.push(state.clone());
                state.depth += 1;
                state.length *= length_decay * rng.range(0.85, 1.15);
                // More aggressive thickness decay when branching
                state.thickness *= thickness_decay * rng.range(0.65, 0.90);
            }
            ']' => {
                if let Some(s) = stack.pop() {
                    state = s;
                }
            }
            'L' => {
                // ONLY create leaves at branch ends (terminal branches)
                // Terminal = thin enough OR deep enough in the tree
                let is_terminal = state.thickness < 0.04 || state.depth >= 3;

                if is_terminal {
                    let size = rng.range(0.4, 0.75)
                        * (state.length / base_length).sqrt().max(0.35)
                        * leaf_scale;

                    leaves.push(LeafCluster {
                        position: state.position + state.direction * rng.range(-0.05, 0.12),
                        direction: state.direction,
                        right: state.right,
                        up: state.up,
                        size,
                    });
                }
            }
            _ => {}
        }
    }

    (branches, leaves)
}

fn generate_cylinder(
    segment: &BranchSegment,
    radial_segments: u32,
    vertices: &mut Vec<PropVertex>,
    indices: &mut Vec<u32>,
) {
    let base_idx = vertices.len() as u32;
    let axis = (segment.end - segment.start).normalize();

    let up = if axis.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let right = axis.cross(up).normalize();
    let forward = right.cross(axis).normalize();

    let bark_color = [0.30, 0.20, 0.12, 1.0];

    for ring in 0..2 {
        let (center, radius) = if ring == 0 {
            (segment.start, segment.start_radius)
        } else {
            (segment.end, segment.end_radius)
        };

        for i in 0..=radial_segments {
            let angle = (i as f32 / radial_segments as f32) * PI * 2.0;
            let (sin_a, cos_a) = angle.sin_cos();
            let normal = right * cos_a + forward * sin_a;
            let position = center + normal * radius;

            vertices.push(PropVertex {
                position: position.into(),
                normal: normal.into(),
                color: bark_color,
                uv: [i as f32 / radial_segments as f32, ring as f32 * 2.0],
                texture_id: 2,
            });
        }
    }

    let ring_verts = radial_segments + 1;
    for i in 0..radial_segments {
        let bl = base_idx + i;
        let br = base_idx + i + 1;
        let tl = base_idx + ring_verts + i;
        let tr = base_idx + ring_verts + i + 1;

        indices.extend_from_slice(&[bl, tl, br, br, tl, tr]);
    }
}

/// Generates a complex bent leaf cluster with 3-4 twisted quads
/// NOT just two 45° planes - uses golden angle distribution and curved vertices
fn generate_bent_leaf_cluster(
    cluster: &LeafCluster,
    seed: u32,
    vertices: &mut Vec<PropVertex>,
    indices: &mut Vec<u32>,
) {
    let mut rng = SimpleRng::new(seed);

    // Varied green tones for more natural look
    let base_green = 0.50 + rng.range(-0.08, 0.08);
    let leaf_color = [
        0.30 + rng.range(-0.05, 0.05),
        base_green,
        0.22 + rng.range(-0.04, 0.04),
        1.0,
    ];

    // 3-4 bent quads at various angles (NOT just two 45° planes)
    let num_planes = 3 + (rng.next() * 2.0) as u32;

    for plane_idx in 0..num_planes {
        let base_idx = vertices.len() as u32;

        // Golden angle distribution for non-uniform, natural-looking spread
        // Plus random offset so it's never perfectly regular
        let golden_angle = 2.39996; // ~137.5 degrees
        let base_angle = (plane_idx as f32) * golden_angle + rng.range(-0.35, 0.35);

        // Each plane has different tilt relative to branch
        let tilt_amount = rng.range(0.2, 0.55);
        let twist = rng.range(-0.25, 0.25);

        // Rotate plane around the branch direction
        let rotation = Quat::from_axis_angle(cluster.direction, base_angle);
        let plane_right = rotation * cluster.right;
        let plane_forward = rotation * cluster.up;

        // Create tilted up vector - not aligned with branch, more natural
        let tilted_up = (cluster.direction * (1.0 - tilt_amount)
            + plane_forward * tilt_amount
            + plane_right * twist)
            .normalize();

        // Asymmetric dimensions for organic look
        let width_left = cluster.size * rng.range(0.32, 0.52);
        let width_right = cluster.size * rng.range(0.32, 0.52);
        let height = cluster.size * rng.range(0.75, 1.15);

        // Bend parameters - vertices curve AROUND the branch
        let bend_out_bottom = cluster.size * rng.range(0.06, 0.16);
        let bend_out_top = cluster.size * rng.range(-0.04, 0.08);
        let curve_inward = cluster.size * rng.range(0.02, 0.10);

        // Slight random wobble for each vertex
        let wobble = |rng: &mut SimpleRng| -> Vec3 {
            Vec3::new(
                rng.range(-0.025, 0.025),
                rng.range(-0.025, 0.025),
                rng.range(-0.025, 0.025),
            ) * cluster.size
        };

        // Four corners with bending that wraps around the branch
        let corners = [
            // Bottom left - bends outward from branch
            cluster.position - plane_right * width_left
                + plane_forward * bend_out_bottom
                + wobble(&mut rng),
            // Bottom right - bends outward
            cluster.position
                + plane_right * width_right
                + plane_forward * bend_out_bottom
                + wobble(&mut rng),
            // Top right - curves back toward branch center, narrower
            cluster.position
                + plane_right * (width_right * rng.range(0.7, 0.92))
                + tilted_up * height
                + plane_forward * bend_out_top
                - cluster.direction * curve_inward
                + wobble(&mut rng),
            // Top left - curves back, narrower
            cluster.position - plane_right * (width_left * rng.range(0.7, 0.92))
                + tilted_up * height
                + plane_forward * bend_out_top
                - cluster.direction * curve_inward
                + wobble(&mut rng),
        ];

        // Calculate face normal from bent geometry
        let edge_bottom = corners[1] - corners[0];
        let edge_left = corners[3] - corners[0];
        let face_normal = edge_bottom.cross(edge_left).normalize();

        // Varying normals per vertex for curved surface shading
        let normal_bend = plane_forward * 0.25;
        let normals = [
            (face_normal + normal_bend).normalize(),
            (face_normal + normal_bend).normalize(),
            (face_normal - normal_bend * 0.4 + tilted_up * 0.15).normalize(),
            (face_normal - normal_bend * 0.4 + tilted_up * 0.15).normalize(),
        ];

        let uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

        for i in 0..4 {
            vertices.push(PropVertex {
                position: corners[i].into(),
                normal: normals[i].into(),
                color: leaf_color,
                uv: uvs[i],
                texture_id: 1,
            });
        }

        // Double-sided
        indices.extend_from_slice(&[base_idx, base_idx + 1, base_idx + 2]);
        indices.extend_from_slice(&[base_idx, base_idx + 2, base_idx + 3]);
        indices.extend_from_slice(&[base_idx + 2, base_idx + 1, base_idx]);
        indices.extend_from_slice(&[base_idx + 3, base_idx + 2, base_idx]);
    }
}

fn generate_tree_billboard_cross(
    center: Vec3,
    width: f32,
    height: f32,
    vertices: &mut Vec<PropVertex>,
    indices: &mut Vec<u32>,
) {
    let leaf_color = [0.32, 0.50, 0.22, 1.0];
    let bark_color = [0.30, 0.20, 0.12, 1.0];

    //
    // Leaf planes
    //
    let planes = [
        (0.0_f32, 1.00),
        (60.0_f32.to_radians(), 0.90),
        (120.0_f32.to_radians(), 0.82),
    ];

    for (angle, scale) in planes {
        let right = Vec3::new(angle.cos(), 0.0, angle.sin());

        let half_width = width * scale * 0.5;
        let bottom = center.y - height * 0.25;
        let top = center.y + height * (0.70 + (scale - 0.8) * 0.15);

        let base = vertices.len() as u32;

        let corners = [
            Vec3::new(
                center.x - right.x * half_width,
                bottom,
                center.z - right.z * half_width,
            ),
            Vec3::new(
                center.x + right.x * half_width,
                bottom,
                center.z + right.z * half_width,
            ),
            Vec3::new(
                center.x + right.x * half_width,
                top,
                center.z + right.z * half_width,
            ),
            Vec3::new(
                center.x - right.x * half_width,
                top,
                center.z - right.z * half_width,
            ),
        ];

        let normal = Vec3::new(-right.z, 0.0, right.x);

        let uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

        for i in 0..4 {
            vertices.push(PropVertex {
                position: corners[i].into(),
                normal: normal.into(),
                color: leaf_color,
                uv: uvs[i],
                texture_id: 1,
            });
        }

        // Double sided
        indices.extend_from_slice(&[
            base,
            base + 1,
            base + 2,
            base,
            base + 2,
            base + 3,
            base + 2,
            base + 1,
            base,
            base + 3,
            base + 2,
            base,
        ]);
    }

    //
    // Trunk (triangular prism with pointed top)
    //
    let trunk_radius = width * 0.07;
    let trunk_bottom = center.y - height * 0.35;
    let trunk_top = center.y + height * 0.20;
    let trunk_tip = center + Vec3::Y * (height * 0.45);

    let trunk_base = vertices.len() as u32;

    let mut bottom_ring = [Vec3::ZERO; 3];
    let mut top_ring = [Vec3::ZERO; 3];

    for i in 0..3 {
        let a = i as f32 * TAU / 3.0;
        let dir = Vec3::new(a.cos(), 0.0, a.sin());

        bottom_ring[i] = Vec3::new(center.x, trunk_bottom, center.z) + dir * trunk_radius;
        top_ring[i] = Vec3::new(center.x, trunk_top, center.z) + dir * trunk_radius;
    }

    for i in 0..3 {
        let a = i as f32 * TAU / 3.0;
        let dir = Vec3::new(a.cos(), 0.0, a.sin());

        vertices.push(PropVertex {
            position: bottom_ring[i].into(),
            normal: dir.into(),
            color: bark_color,
            uv: [i as f32 / 3.0, 0.0],
            texture_id: 2,
        });

        vertices.push(PropVertex {
            position: top_ring[i].into(),
            normal: dir.into(),
            color: bark_color,
            uv: [i as f32 / 3.0, 0.8],
            texture_id: 2,
        });
    }

    let tip_index = vertices.len() as u32;

    vertices.push(PropVertex {
        position: trunk_tip.into(),
        normal: Vec3::Y.into(),
        color: bark_color,
        uv: [0.5, 1.0],
        texture_id: 2,
    });

    // Prism sides
    for i in 0..3 {
        let next = (i + 1) % 3;

        let b0 = trunk_base + i * 2;
        let t0 = trunk_base + i * 2 + 1;

        let b1 = trunk_base + next * 2;
        let t1 = trunk_base + next * 2 + 1;

        indices.extend_from_slice(&[b0, t0, b1, b1, t0, t1, b1, t0, b0, t1, t0, b1]);
    }

    // Pointed top
    for i in 0..3 {
        let next = (i + 1) % 3;

        let t0 = trunk_base + i * 2 + 1;
        let t1 = trunk_base + next * 2 + 1;

        indices.extend_from_slice(&[t0, t1, tip_index, tip_index, t1, t0]);
    }
}

fn calculate_bounds(vertices: &[PropVertex]) -> (Vec3, f32) {
    if vertices.is_empty() {
        return (Vec3::ZERO, 1.0);
    }

    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);

    for v in vertices {
        let p = Vec3::from(v.position);
        min = min.min(p);
        max = max.max(p);
    }

    let center = (min + max) * 0.5;
    let radius = (max - min).length() * 0.5;

    (center, radius.max(0.1))
}

fn create_mesh(device: &Device, vertices: &[PropVertex], indices: &[u32]) -> Mesh {
    let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Tree Vertex Buffer"),
        contents: bytemuck::cast_slice(vertices),
        usage: BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Tree Index Buffer"),
        contents: bytemuck::cast_slice(indices),
        usage: BufferUsages::INDEX,
    });

    let bounds = calculate_bounds(vertices);

    Mesh {
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        bounds,
    }
}

// ============================================================================
// Tree generation parameters
// ============================================================================

struct TreeStructure {
    axiom: &'static str,
    rules: Vec<LSystemRule>,
    iterations: u32,
    base_angle: f32,
    length_decay: f32,
    thickness_decay: f32,
    base_length: f32,
    base_thickness: f32,
    seed: u32,
}

struct LodRenderParams {
    branch_segments: u32,
    leaf_density: f32,
    leaf_scale: f32,
    min_branch_thickness: f32,
}

impl LodRenderParams {
    fn oak_for_lod(lod: u32) -> Self {
        match lod {
            0 => Self {
                branch_segments: 8,
                leaf_density: 1.0,
                leaf_scale: 1.0,
                min_branch_thickness: 0.005,
            },
            1 => Self {
                branch_segments: 4,
                leaf_density: 0.6,
                leaf_scale: 1.18,
                min_branch_thickness: 0.008,
            },
            2 => Self {
                branch_segments: 3,
                leaf_density: 0.3,
                leaf_scale: 1.45,
                min_branch_thickness: 0.012,
            },
            _ => Self {
                branch_segments: 3,
                leaf_density: 0.1,
                leaf_scale: 1.2,
                min_branch_thickness: 1.00,
            },
        }
    }

    fn pine_for_lod(lod: u32) -> Self {
        match lod {
            0 => Self {
                branch_segments: 8,
                leaf_density: 1.0,
                leaf_scale: 1.0,
                min_branch_thickness: 0.005,
            },
            1 => Self {
                branch_segments: 4,
                leaf_density: 0.6,
                leaf_scale: 1.18,
                min_branch_thickness: 0.008,
            },
            2 => Self {
                branch_segments: 3,
                leaf_density: 0.3,
                leaf_scale: 1.45,
                min_branch_thickness: 0.012,
            },
            _ => Self {
                branch_segments: 3,
                leaf_density: 0.0,
                leaf_scale: 1.0,
                min_branch_thickness: 1.0,
            },
        }
    }
}

fn oak_tree_structure() -> TreeStructure {
    TreeStructure {
        axiom: "FFFA",
        rules: vec![LSystemRule {
            from: 'A',
            to: "[&FLAL]////[&FLAL]////[&FLAL]////^FAL",
        }],
        iterations: 4,
        base_angle: 14.0_f32.to_radians(),
        length_decay: 0.74,
        thickness_decay: 0.48, // More aggressive - branches thin faster
        base_length: 0.8,
        base_thickness: 0.28, // Start thicker so ends can be properly thin
        seed: 2,
    }
}

fn pine_tree_structure() -> TreeStructure {
    TreeStructure {
        axiom: "FFA",
        rules: vec![LSystemRule {
            from: 'A',
            to: "[&FL]////[&FL]////[&FL]////[&FL]^FA",
        }],
        iterations: 5,
        base_angle: 35.0_f32.to_radians(),
        length_decay: 0.80,
        thickness_decay: 0.55,
        base_length: 0.7,
        base_thickness: 0.18,
        seed: 123,
    }
}

fn generate_tree_mesh(
    device: &Device,
    structure: &TreeStructure,
    render_params: &LodRenderParams,
) -> Mesh {
    let mut vertices: Vec<PropVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let lsystem_string = expand_lsystem(structure.axiom, &structure.rules, structure.iterations);

    let (branches, leaf_clusters) = interpret_lsystem(
        &lsystem_string,
        structure.base_angle,
        structure.length_decay,
        structure.thickness_decay,
        structure.base_length,
        structure.base_thickness,
        structure.seed,
        render_params.leaf_scale,
    );

    // Generate branch geometry
    for branch in &branches {
        if branch.start_radius >= render_params.min_branch_thickness {
            generate_cylinder(
                branch,
                render_params.branch_segments,
                &mut vertices,
                &mut indices,
            );
        }
    }

    // Generate leaf clusters with bent cross-quads
    if render_params.leaf_density > 0.0 {
        let cluster_step = (1.0 / render_params.leaf_density).ceil() as usize;
        for (i, cluster) in leaf_clusters.iter().enumerate() {
            if i % cluster_step == 0 {
                // Use index as part of seed for variation
                let cluster_seed = structure.seed.wrapping_add(i as u32 * 7919);
                generate_bent_leaf_cluster(cluster, cluster_seed, &mut vertices, &mut indices);
            }
        }
    }

    create_mesh(device, &vertices, &indices)
}

fn make_oak_lod(device: &Device, lod: u32) -> Mesh {
    if lod >= 3 {
        let mut vertices: Vec<PropVertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        generate_tree_billboard_cross(
            Vec3::new(0.0, 3.0, 0.0),
            4.0,
            5.0,
            &mut vertices,
            &mut indices,
        );
        return create_mesh(device, &vertices, &indices);
    }

    let structure = oak_tree_structure();
    let render_params = LodRenderParams::oak_for_lod(lod);
    generate_tree_mesh(device, &structure, &render_params)
}
