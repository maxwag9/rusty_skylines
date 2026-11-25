use crate::data::Settings;
use crate::helpers::modpack::ModManager;
use rusty_skylines_utils::positions::{chunk_size, ChunkCoord, LocalPos, WorldPos};
use crate::renderer::pipelines::Pipelines;
use crate::renderer::shadows::{shadow_bias_for_cascade, shadow_pipeline_options};
use crate::ui::input::Input;
use crate::world::camera::camera::Camera;
use crate::world::terrain::terrain_subsystem::{extract_frustum_planes, CursorMode, Terrain};
use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec2, Vec3};
use revision::revisioned;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::f32::consts::{PI, TAU};
use std::{fs, mem};
use std::path::{Path, PathBuf};
use tracing::{error, warn};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferAddress, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Device, Face, IndexFormat, Queue, RenderPass, VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode};
use wgpu_render_manager::compute_system::{BufferSet, ComputePipelineOptions};
use wgpu_render_manager::generator::{MipmapMode, TextureCompressionMode, TextureKey, TextureParams};
use wgpu_render_manager::pipelines::{FragmentOption, PipelineOptions};
use wgpu_render_manager::renderer::RenderManager;
use crate::world::props::uniform_layout::{UniformFieldYaml, UniformLayout, UniformYaml};
use anyhow::{Context, Result};
use crate::world::props::props_renderer::PropRenderArchetype;

#[derive(Deserialize)]
struct PropDefinitionYaml {
    name: String,
    geometry: PropGeometryYaml,
    #[serde(default)]
    textures: Vec<PropTextureYaml>,
}
#[derive(Deserialize)]
struct PropGeometryYaml {
    shader: String,
    #[serde(default = "default_entry_point")]
    entry_point: String,
    #[serde(default = "default_workgroups")]
    workgroups: [u32; 3],
    #[serde(default = "default_bounds_radius")]
    bounds_radius: f32,
    index_count: Option<u32>,
    /// Base bindings shared by every LOD. LODs only supply overrides.
    #[serde(default)]
    bindings: Vec<PropBindingYaml>,
    #[serde(default)]
    lods: Vec<PropGeometryLodYaml>,
}

fn default_entry_point() -> String {
    "main".to_string()
}

fn default_workgroups() -> [u32; 3] {
    [1, 1, 1]
}

fn default_bounds_radius() -> f32 {
    5.0
}

#[derive(Deserialize, Clone)]
struct PropBindingYaml {
    binding: u32,
    #[serde(flatten)]
    resource: PropBindingResourceYaml,
}

#[derive(Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
enum PropBindingResourceYaml {
    Uniform {
        fields: Vec<UniformFieldYaml>,
    },
    Storage {
        access: String,
        format: String,
        source: Option<String>,
        data: Option<serde_yaml::Value>,
        capacity: Option<u64>,
        #[serde(default)]
        usage: Vec<String>,
    },
}

#[derive(Deserialize)]
struct PropGeometryLodYaml {
    distance: f32,
    #[serde(default = "default_alpha_cutoff")]
    alpha_cutoff: f32,
    #[serde(default)]
    overrides: HashMap<u32, HashMap<String, serde_yaml::Value>>,
    #[serde(default)]
    storage_capacities: HashMap<u32, u64>,
}

fn default_alpha_cutoff() -> f32 {
    0.25
}
#[derive(Deserialize)]
struct PropTextureYaml {
    slot: u32,
    shader: String,
    #[serde(default = "default_prop_texture_resolution")]
    resolution: u32,
    #[serde(default)]
    mipmap_mode: MipmapMode,
    #[serde(default)]
    params: PropTextureParamsYaml,
    #[serde(default)]
    compression: TextureCompressionMode,
}

#[derive(Default, Deserialize)]
pub struct PropTextureParamsYaml {
    color_primary: Option<[f32; 4]>,
    color_secondary: Option<[f32; 4]>,
    seed: Option<u32>,
    scale: Option<f32>,
    roughness: Option<f32>,
    octaves: Option<f32>,
    persistence: Option<f32>,
    lacunarity: Option<f32>,
}

impl PropTextureParamsYaml {
    pub fn build(&self) -> TextureParams {
        let mut p = TextureParams::default();

        if let Some(v) = self.color_primary {
            p.color_primary = v;
        }
        if let Some(v) = self.color_secondary {
            p.color_secondary = v;
        }
        if let Some(v) = self.seed {
            p.seed = v;
        }
        if let Some(v) = self.scale {
            p.scale = v;
        }
        if let Some(v) = self.roughness {
            p.roughness = v;
        }
        if let Some(v) = self.octaves {
            p.octaves = v;
        }
        if let Some(v) = self.persistence {
            p.persistence = v;
        }
        if let Some(v) = self.lacunarity {
            p.lacunarity = v;
        }

        p
    }
}

pub struct PropTextureDefinition {
    pub slot: u32,
    pub shader: PathBuf,
    pub resolution: u32,
    pub mipmap_mode: MipmapMode,
    pub params: PropTextureParamsYaml,
    pub compression: TextureCompressionMode,
}

pub struct PropGeometryDefinition {
    pub shader: PathBuf,
    pub entry_point: String,
    pub workgroups: [u32; 3],
    pub bounds_radius: f32,
    pub index_count: Option<u32>,
    pub lods: [Option<PropGeometryLod>; 4],
    pub textures: Vec<PropTextureDefinition>,
}

#[derive(Clone)]
pub struct PropGeometryLod {
    pub distance: f32,
    pub alpha_cutoff: f32,
    pub bindings: Vec<PropBindingDefinition>,
}

#[derive(Clone)]
pub struct PropBindingDefinition {
    pub binding: u32,
    pub resource: PropBindingResource,
}

#[derive(Clone)]
pub enum PropBindingResource {
    Uniform(UniformLayout),
    Storage {
        access: String,
        format: String,
        source: Option<PathBuf>,
        data: Option<serde_yaml::Value>,
        capacity: Option<u64>,
        usage: Vec<String>,
    },
}

fn default_prop_texture_resolution() -> u32 {
    256
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


pub struct PropChunk {
    pub chunk_coord: ChunkCoord,
    pub archetype_instances: HashMap<ArchetypeId, Vec<PropInstanceId>>,
    pub dirty_archetypes: HashSet<ArchetypeId>,
    pub generated_count: usize,
    pub manual_count: usize,
}

impl PropChunk {
    pub fn new(chunk_coord: ChunkCoord) -> Self {
        Self {
            chunk_coord,
            archetype_instances: HashMap::new(),
            dirty_archetypes: HashSet::new(),
            generated_count: 0,
            manual_count: 0,
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
    archetypes: Vec<Option<String>>,
    pub definitions: HashMap<ArchetypeId, PropGeometryDefinition>,
    archetypes_free: Vec<ArchetypeId>,
    archetype_to_id: HashMap<String, ArchetypeId>,
    pub prop_instances: Vec<Option<PropInstance>>,
    prop_instances_free: Vec<PropInstanceId>,
    pub chunks: HashMap<ChunkCoord, PropChunk>,
    pub dirty_chunks: HashSet<ChunkCoord>,
    generated_prop_count: usize,
    manual_prop_count: usize,
    pub visible_generated_prop_count: usize,
    pub visible_manual_prop_count: usize,
    preview_prop_id: Option<PropInstanceId>,
    pub instance_revision: u64,
    pub archetype_revision: u64,
}

impl Props {
    pub fn new() -> Self {
        let props = Self {
            archetypes: vec![],
            definitions: Default::default(),
            archetypes_free: vec![],
            archetype_to_id: HashMap::new(),
            prop_instances: vec![],
            prop_instances_free: vec![],
            chunks: HashMap::new(),
            dirty_chunks: HashSet::new(),
            generated_prop_count: 0,
            manual_prop_count: 0,
            visible_generated_prop_count: 0,
            visible_manual_prop_count: 0,
            preview_prop_id: None,
            instance_revision: 0,
            archetype_revision: 0,
        };

        props
    }
    pub fn load_prop_definitions(&mut self, mod_manager: &ModManager) {
        let mut definitions_changed = false;

        for path in mod_manager.paths_in("simulation/props", &["yaml"]) {
            let source = match fs::read_to_string(&path) {
                Ok(v) => v,
                Err(e) => {
                    error!("[Props] Failed reading {}: {}", path.display(), e);
                    continue;
                }
            };

            let definition = match serde_yaml::from_str::<PropDefinitionYaml>(&source) {
                Ok(v) => v,
                Err(e) => {
                    error!("[Props] Invalid {}: {}", path.display(), e);
                    continue;
                }
            };

            let Some(geometry_shader) =
                mod_manager.resource_path(&definition.geometry.shader)
            else {
                error!(
                "[Props] Missing geometry shader {}",
                definition.geometry.shader
            );
                continue;
            };

            if definition.geometry.bindings.is_empty() {
                error!(
                "[Props] {} has no base bindings, every prop needs geometry.bindings",
                definition.name
            );
                continue;
            }

            if definition.geometry.lods.is_empty() {
                error!(
                "[Props] {} has no LODs, every prop needs at least one LOD",
                definition.name
            );
                continue;
            }

            if definition.geometry.lods.len() > 4 {
                error!(
                "[Props] {} has {} LODs (max 4)",
                definition.name,
                definition.geometry.lods.len()
            );
                continue;
            }

            let mut lods: [Option<PropGeometryLod>; 4] =
                std::array::from_fn(|_| None);
            let mut lods_ok = true;

            for (lod_index, lod_yaml) in definition.geometry.lods.into_iter().enumerate() {
                if !lod_yaml.alpha_cutoff.is_finite()
                    || !(0.0..=1.0).contains(&lod_yaml.alpha_cutoff)
                {
                    error!(
                    "[Props] Invalid alpha cutoff {} in {} LOD {}",
                    lod_yaml.alpha_cutoff,
                    definition.name,
                    lod_index
                );
                    lods_ok = false;
                    break;
                }

                let mut bindings = Vec::new();

                for base_binding in &definition.geometry.bindings {
                    let resource = match &base_binding.resource {
                        PropBindingResourceYaml::Uniform { fields } => {
                            let mut fields = fields.clone();

                            if let Some(field_overrides) =
                                lod_yaml.overrides.get(&base_binding.binding)
                            {
                                for field in &mut fields {
                                    if let Some(new_value) = field_overrides.get(&field.name) {
                                        field.value = new_value.clone();
                                    }
                                }
                            }

                            let uniform_decl = UniformYaml { fields };

                            match UniformLayout::from_declaration(uniform_decl, 65_536) {
                                Ok(layout) => PropBindingResource::Uniform(layout),
                                Err(e) => {
                                    error!(
                                    "[Props] Invalid uniform binding {} in {} LOD {}: {:#}",
                                    base_binding.binding,
                                    definition.name,
                                    lod_index,
                                    e
                                );
                                    lods_ok = false;
                                    break;
                                }
                            }
                        }
                        PropBindingResourceYaml::Storage {
                            access,
                            format,
                            source,
                            data,
                            capacity,
                            usage,
                        } => {
                            let source_path = source.as_ref().and_then(|s| {
                                mod_manager
                                    .resource_path(s)
                                    .map(|p| p.to_path_buf())
                            });

                            if source.is_some() && source_path.is_none() {
                                error!(
                                "[Props] Missing storage source for binding {} in {}",
                                base_binding.binding,
                                definition.name
                            );
                            }

                            PropBindingResource::Storage {
                                access: access.clone(),
                                format: format.clone(),
                                source: source_path,
                                data: data.clone(),
                                capacity: lod_yaml.storage_capacities.get(&base_binding.binding).copied().or(*capacity),
                                usage: usage.clone(),
                            }
                        }
                    };

                    if !lods_ok {
                        break;
                    }

                    bindings.push(PropBindingDefinition {
                        binding: base_binding.binding,
                        resource,
                    });
                }

                if !lods_ok {
                    break;
                }

                lods[lod_index] = Some(PropGeometryLod {
                    distance: lod_yaml.distance,
                    alpha_cutoff: lod_yaml.alpha_cutoff,
                    bindings,
                });
            }

            if !lods_ok {
                continue;
            }

            let mut textures = Vec::new();

            for texture in definition.textures {
                if !(1..=4).contains(&texture.slot) {
                    error!(
                    "[Props] Invalid texture slot {} in {}",
                    texture.slot,
                    definition.name
                );
                    continue;
                }

                let Some(shader) = mod_manager.resource_path(&texture.shader) else {
                    error!(
                    "[Props] Missing texture shader {}",
                    texture.shader
                );
                    continue;
                };

                textures.push(PropTextureDefinition {
                    slot: texture.slot,
                    shader: shader.to_path_buf(),
                    resolution: texture.resolution,
                    mipmap_mode: texture.mipmap_mode,
                    params: texture.params,
                    compression: texture.compression,
                });
            }

            let id = self.register_archetype(definition.name.clone());

            self.definitions.insert(
                id,
                PropGeometryDefinition {
                    shader: geometry_shader.to_path_buf(),
                    entry_point: definition.geometry.entry_point,
                    workgroups: definition.geometry.workgroups,
                    bounds_radius: definition.geometry.bounds_radius,
                    index_count: definition.geometry.index_count,
                    lods,
                    textures,
                },
            );

            definitions_changed = true;
        }

        if definitions_changed {
            self.archetype_revision = self.archetype_revision.wrapping_add(1);
        }
    }
    fn clear_instances(&mut self) {
        self.prop_instances.clear();
        self.prop_instances_free.clear();
        self.chunks.clear();
        self.dirty_chunks.clear();


        self.generated_prop_count = 0;
        self.manual_prop_count = 0;
        self.visible_generated_prop_count = 0;
        self.visible_manual_prop_count = 0;
        self.preview_prop_id = None;
        self.instance_revision = self.instance_revision.wrapping_add(1);
    }

    pub fn clear(&mut self) {
        self.clear_instances();

        self.archetypes.clear();
        self.definitions.clear();
        self.archetypes_free.clear();
        self.archetype_to_id.clear();

        self.archetype_revision = self.archetype_revision.wrapping_add(1);
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

            let archetype_name = self.archetypes[prop.archetype_id.unwrap() as usize].as_ref().unwrap().to_lowercase();

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
        self.clear_instances();

        for prop in saved.instances {
            let key = &prop.archetype.to_lowercase();

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
    pub fn register_archetype(&mut self, name: impl Into<String>) -> ArchetypeId {
        let name = name.into().to_lowercase();

        if let Some(id) = self.archetype_to_id.get(&name).copied() {
            self.archetypes[id as usize] = Some(name);
            return id;
        }

        let id = if let Some(id) = self.archetypes_free.pop() {
            self.archetypes[id as usize] = Some(name.clone());
            id
        } else {
            let id = self.archetypes.len() as ArchetypeId;
            self.archetypes.push(Some(name.clone()));
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
            chunk.generated_count += 1;
        } else {
            self.manual_prop_count += 1;
            chunk.manual_count += 1;
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
        let is_generated = instance.generated;
        let chunk_coord = instance.pos.chunk;
        let chunk = self
            .chunks
            .entry(chunk_coord)
            .or_insert_with(|| PropChunk::new(chunk_coord));
        if is_generated {
            debug_assert!(self.generated_prop_count > 0);
            self.generated_prop_count -= 1;
            chunk.generated_count = chunk.generated_count.saturating_sub(1);
        } else {
            debug_assert!(self.manual_prop_count > 0);
            self.manual_prop_count -= 1;
            chunk.manual_count = chunk.manual_count.saturating_sub(1);
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
                chunk.generated_count = chunk.generated_count.saturating_sub(removed_here);
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
        _mod_manager: &ModManager,
    ) -> ArchetypeId {
        let key = archetype_name.into().to_lowercase();

        self.archetype_to_id
            .get(&key)
            .copied()
            .unwrap_or_else(|| {
                error!("[Props] ensure_archetype: '{}' not registered (load definitions first)", key);
                // Fallback so callers don't panic immediately; real fix is registering all props at startup
                0
            })
    }
}


