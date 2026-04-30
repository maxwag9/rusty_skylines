use crate::data::Settings;
use crate::helpers::paths::shader_dir;
use crate::helpers::positions::{ChunkCoord, LocalPos, WorldPos};
use crate::renderer::pipelines::Pipelines;
use crate::renderer::render_passes::{
    color_and_normals_and_instance_targets, depth_stencil, make_shadow_option,
};
use crate::renderer::shadows::{shadow_bias_for_cascade, shadow_pipeline_options};
use crate::ui::input::Input;
use crate::world::camera::Camera;
use crate::world::terrain::terrain_subsystem::{CursorMode, Terrain};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::PI;
use wgpu::PrimitiveTopology::TriangleList;
use wgpu::util::DeviceExt;
use wgpu::*;
use wgpu_render_manager::generator::{TextureKey, TextureParams};
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
    pub model: [[f32; 4]; 4],
    pub prev_model: [[f32; 4]; 4],
    pub color: [f32; 4],
    pub misc: [f32; 4], // x: seed, y: wind_strength, z: type_variant, w: padding
}

impl GpuPropInstance {
    pub fn new(
        model: Mat4,
        prev_model: Mat4,
        color: [f32; 4],
        seed: f32,
        wind: f32,
        variant: f32,
    ) -> Self {
        Self {
            model: model.to_cols_array_2d(),
            prev_model: prev_model.to_cols_array_2d(),
            color,
            misc: [seed, wind, variant, 0.0],
        }
    }

    pub fn layout<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            array_stride: size_of::<GpuPropInstance>() as BufferAddress,
            step_mode: VertexStepMode::Instance,
            attributes: &[
                // model mat4 (locations 5-8)
                VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: VertexFormat::Float32x4,
                },
                VertexAttribute {
                    offset: 16,
                    shader_location: 6,
                    format: VertexFormat::Float32x4,
                },
                VertexAttribute {
                    offset: 32,
                    shader_location: 7,
                    format: VertexFormat::Float32x4,
                },
                VertexAttribute {
                    offset: 48,
                    shader_location: 8,
                    format: VertexFormat::Float32x4,
                },
                // prev_model mat4 (locations 9-12)
                VertexAttribute {
                    offset: 64,
                    shader_location: 9,
                    format: VertexFormat::Float32x4,
                },
                VertexAttribute {
                    offset: 80,
                    shader_location: 10,
                    format: VertexFormat::Float32x4,
                },
                VertexAttribute {
                    offset: 96,
                    shader_location: 11,
                    format: VertexFormat::Float32x4,
                },
                VertexAttribute {
                    offset: 112,
                    shader_location: 12,
                    format: VertexFormat::Float32x4,
                },
                // color (location 13)
                VertexAttribute {
                    offset: 128,
                    shader_location: 13,
                    format: VertexFormat::Float32x4,
                },
                // misc (location 14)
                VertexAttribute {
                    offset: 144,
                    shader_location: 14,
                    format: VertexFormat::Float32x4,
                },
            ],
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PropInstance {
    pub pos: WorldPos,
    pub rotation_y_rad: f32,
    pub scale: f32,
    pub color: [f32; 4],
    pub seed: u32,
    pub variant: u16,
    pub wind_strength: f32,
}

pub struct Mesh {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,
    pub bounds: (Vec3, f32), // center, radius
}

pub struct PropChunk {
    pub chunk_coord: ChunkCoord,
    pub archetype_instances: HashMap<String, Vec<PropInstance>>,
    pub gpu_instance_buffers: HashMap<String, (Buffer, u32)>,
}

impl PropChunk {
    pub fn new(chunk_coord: ChunkCoord) -> Self {
        Self {
            chunk_coord,
            archetype_instances: HashMap::new(),
            gpu_instance_buffers: HashMap::new(),
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
pub struct SavePropChunk {
    pub chunk_coord: ChunkCoord,
    pub archetype_instances: HashMap<String, Vec<PropInstance>>,
}
pub struct Props {
    pub props: HashMap<String, Prop>,
    pub chunks: HashMap<ChunkCoord, PropChunk>,
    pub prev_models: HashMap<u64, [[f32; 4]; 4]>, // key: hash of (chunk, archetype, index)
    device: Device,
}

impl Props {
    pub fn new(device: &Device) -> Self {
        Self {
            props: HashMap::new(),
            chunks: HashMap::new(),
            prev_models: HashMap::new(),
            device: device.clone(),
        }
    }

    pub fn get_props(&self) -> Vec<SavePropChunk> {
        let mut prop_chunks: Vec<SavePropChunk> = Vec::new();

        for chunk in self.chunks.values() {
            prop_chunks.push(SavePropChunk {
                chunk_coord: chunk.chunk_coord,
                archetype_instances: chunk.archetype_instances.clone(),
            })
        }

        prop_chunks
    }
    pub fn load_props(&mut self, chunks: Vec<SavePropChunk>) {
        for chunk in chunks {
            for archetype in chunk.archetype_instances.keys() {
                let key = &archetype.to_lowercase();
                if !self.is_registered(key) {
                    if let Some(prop) = make_prop(key, &self.device) {
                        self.register_prop(key.clone(), prop);
                    }
                }
            }
            self.chunks.insert(
                chunk.chunk_coord,
                PropChunk {
                    chunk_coord: chunk.chunk_coord,
                    archetype_instances: chunk.archetype_instances.clone(),
                    gpu_instance_buffers: HashMap::new(),
                },
            );
        }
    }
    pub fn register_prop(&mut self, key: impl Into<String>, prop: Prop) {
        self.props.insert(key.into(), prop);
    }

    pub fn is_registered(&self, key: impl Into<String>) -> bool {
        self.props.contains_key(&key.into())
    }

    pub fn add_instance(
        &mut self,
        chunk_coord: ChunkCoord,
        archetype: &str,
        instance: PropInstance,
    ) {
        let chunk = self
            .chunks
            .entry(chunk_coord)
            .or_insert_with(|| PropChunk::new(chunk_coord));
        chunk
            .archetype_instances
            .entry(archetype.to_string())
            .or_default()
            .push(instance);
    }

    pub fn clear_chunk(&mut self, chunk_coord: ChunkCoord) {
        self.chunks.remove(&chunk_coord);
    }

    pub fn clear_archetype_in_chunk(&mut self, chunk_coord: ChunkCoord, archetype: &str) {
        if let Some(chunk) = self.chunks.get_mut(&chunk_coord) {
            chunk.archetype_instances.remove(archetype);
            chunk.gpu_instance_buffers.remove(archetype);
        }
    }

    fn instance_key(chunk: ChunkCoord, archetype: &str, index: usize) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        chunk.hash(&mut hasher);
        archetype.hash(&mut hasher);
        index.hash(&mut hasher);
        hasher.finish()
    }

    pub fn upload_instances(
        &mut self,
        device: &Device,
        queue: &Queue,
        camera: &Camera,
        terrain: &Terrain,
    ) {
        for coord in terrain
            .visible
            .iter()
            .map(|visible_chunk| visible_chunk.coords.chunk_coord)
        {
            if let Some(chunk) = self.chunks.get_mut(&coord) {
                for (archetype, instances) in chunk.archetype_instances.iter() {
                    let count = instances.len() as u32;
                    if count == 0 {
                        chunk.gpu_instance_buffers.remove(archetype);
                        continue;
                    }

                    let mut gpu_instances: Vec<GpuPropInstance> =
                        Vec::with_capacity(instances.len());

                    for (i, inst) in instances.iter().enumerate() {
                        // Convert to render-space position relative to camera
                        let render_pos = inst.pos.to_render_pos(camera.eye_world());
                        let model = Mat4::from_scale_rotation_translation(
                            Vec3::splat(inst.scale),
                            Quat::from_rotation_y(inst.rotation_y_rad),
                            render_pos,
                        );

                        let key = Self::instance_key(coord, archetype, i);
                        let prev_model_array = self
                            .prev_models
                            .get(&key)
                            .copied()
                            .unwrap_or_else(|| model.to_cols_array_2d());

                        let seed = (inst.seed as f32) / u32::MAX as f32;

                        gpu_instances.push(GpuPropInstance::new(
                            model,
                            Mat4::from_cols_array_2d(&prev_model_array),
                            inst.color,
                            seed,
                            inst.wind_strength,
                            inst.variant as f32,
                        ));

                        // Store for next frame
                        self.prev_models.insert(key, model.to_cols_array_2d());
                    }

                    let bytes = bytemuck::cast_slice(&gpu_instances);

                    // Recreate buffer if needed (double size for growth)
                    let recreate = match chunk.gpu_instance_buffers.get(archetype) {
                        None => true,
                        Some((_, existing_count)) => *existing_count < count,
                    };

                    if recreate {
                        let capacity = (count.max(1) * 2) as BufferAddress
                            * size_of::<GpuPropInstance>() as BufferAddress;
                        let buffer = device.create_buffer(&BufferDescriptor {
                            label: Some(&format!("prop_inst_{}_{:?}", archetype, coord)),
                            size: capacity,
                            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        chunk
                            .gpu_instance_buffers
                            .insert(archetype.clone(), (buffer, count));
                    }

                    if let Some((buf, stored_count)) = chunk.gpu_instance_buffers.get_mut(archetype)
                    {
                        queue.write_buffer(buf, 0, bytes);
                        *stored_count = count;
                    }
                }
            }
        }
    }

    pub fn place_props(&mut self, terrain: &Terrain, input: &mut Input, device: &Device) {
        match &terrain.cursor.mode {
            CursorMode::Props => {
                let Some(key) = &terrain.cursor.prop_name else {
                    return;
                };
                let key = &key.to_lowercase();
                if !self.is_registered(key) {
                    if let Some(prop) = make_prop(key, device) {
                        self.register_prop(key.clone(), prop);
                    }
                }
                if input.action_repeat("Place Prop") {
                    if let Some(picked_point) = &terrain.last_picked {
                        self.add_instance(
                            picked_point.chunk.coords.chunk_coord,
                            key,
                            PropInstance {
                                pos: picked_point.pos,
                                scale: rand::random_range(0.8..1.5),
                                rotation_y_rad: rand::random_range(0.0..5.0),
                                seed: rand::random(),
                                color: [1.0, 1.0, 1.0, 1.0],
                                wind_strength: 0.2,
                                variant: 0,
                            },
                        );
                    }
                }
            }
            _ => return,
        }
    }

    /// Normal rendering pass
    pub fn render<'a>(
        &'a self,
        render_manager: &mut RenderManager,
        pass: &mut RenderPass<'a>,
        camera: &'a Camera,
        terrain: &'a Terrain,
        pipelines: &Pipelines,
        settings: &Settings,
    ) {
        let eye = camera.eye_world();
        let terrain_height = terrain.get_height_at(eye, true);

        let shader_path = shader_dir().join("props.wgsl");
        let shadow = make_shadow_option(settings, pipelines);
        let targets = color_and_normals_and_instance_targets(pipelines);

        let opts = PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(depth_stencil(Default::default(), settings)),
            msaa_samples: settings.msaa_samples,
            vertex_layouts: Vec::from([PropVertex::layout(), GpuPropInstance::layout()]),
            cull_mode: Some(Face::Back),
            fragment: FragmentOption::Default {
                targets: targets.clone(),
            },
            shadow: shadow.clone(),
            ..Default::default()
        };

        for visible_chunk in terrain.visible.iter() {
            let coord = visible_chunk.coords.chunk_coord;

            let Some(chunk) = self.chunks.get(&coord) else {
                continue;
            };

            let instance_terrain_height = terrain.get_height_at(eye, true);
            let dist = eye.distance_to(WorldPos::new(
                coord,
                LocalPos::new(0.0, instance_terrain_height, 0.0),
            ));
            let lod_level = select_lod(dist);

            for (archetype, instances) in &chunk.archetype_instances {
                if instances.is_empty() {
                    continue;
                }

                let Some(prop) = self.props.get(archetype) else {
                    continue;
                };
                let Some(mesh) = prop.get_lod(lod_level) else {
                    continue;
                };
                let Some((inst_buf, count)) = chunk.gpu_instance_buffers.get(archetype) else {
                    continue;
                };

                render_manager.render(
                    &prop.texture_keys,
                    shader_path.as_path(),
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
    ) {
        let eye = camera.eye_world();

        let bias = shadow_bias_for_cascade(
            cascade_idx,
            pipelines.resources.csm_shadows.texels[cascade_idx],
            settings.reversed_depth_z,
        );

        let shader = shader_dir().join("props_shadows.wgsl");
        let opts = shadow_pipeline_options(
            settings,
            bias,
            vec![PropVertex::layout(), GpuPropInstance::layout()],
            Face::Back,
            FragmentOption::Default { targets: vec![] },
        );

        for visible_chunk in terrain.visible.iter() {
            let coord = visible_chunk.coords.chunk_coord;

            let Some(chunk) = self.chunks.get(&coord) else {
                continue;
            };

            let instance_terrain_height = terrain.get_height_at(eye, true);
            let dist = eye.distance_to(WorldPos::new(
                coord,
                LocalPos::new(0.0, instance_terrain_height, 0.0),
            ));
            let lod_level = select_lod(dist);

            for (archetype, instances) in &chunk.archetype_instances {
                if instances.is_empty() {
                    continue;
                }

                let Some(prop) = self.props.get(archetype) else {
                    continue;
                };
                let Some(mesh) = prop.get_lod(lod_level) else {
                    continue;
                };
                let Some((inst_buf, count)) = chunk.gpu_instance_buffers.get(archetype) else {
                    continue;
                };

                render_manager.render(
                    &prop.texture_keys,
                    shader.as_path(),
                    &opts,
                    &[&pipelines.buffers.camera, shadow_mat_buffer],
                    pass,
                );
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, inst_buf.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..*count);
            }
        }
    }
}
struct Prop {
    lod0: Option<Mesh>,
    lod1: Option<Mesh>,
    lod2: Option<Mesh>,
    lod3: Option<Mesh>,
    texture_keys: [TextureKey; 4], // 4 slots for textures in the shader.
}

impl Prop {
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
const LOD0_MAX_DIST: f64 = 250.0; // Full detail
const LOD1_MAX_DIST: f64 = 700.0; // Medium detail
const LOD2_MAX_DIST: f64 = 1700.0; // Low detail

fn select_lod(dist: f64) -> u32 {
    if dist < LOD0_MAX_DIST {
        0
    } else if dist < LOD1_MAX_DIST {
        1
    } else if dist < LOD2_MAX_DIST {
        2
    } else {
        3
    }
}

fn make_prop(key: &str, device: &Device) -> Option<Prop> {
    match key.to_lowercase().as_str() {
        "oak" | "oak_tree" => make_oak_tree(device),
        "pine" | "pine_tree" => make_pine_tree(device),
        _ => None,
    }
}

fn make_oak_tree(device: &Device) -> Option<Prop> {
    Some(Prop {
        lod0: Some(make_oak_lod(device, 0)),
        lod1: Some(make_oak_lod(device, 1)),
        lod2: Some(make_oak_lod(device, 2)),
        lod3: Some(make_oak_lod(device, 3)),
        texture_keys: [
            TextureKey::new(
                "leaves",
                TextureParams {
                    color_primary: [0.25, 0.45, 0.15, 1.0],
                    color_secondary: [0.35, 0.55, 0.20, 1.0],
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
            ),
            TextureKey::new(
                "bark",
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
            ),
            TextureKey::notex(),
            TextureKey::notex(),
        ],
    })
}

fn make_pine_tree(device: &Device) -> Option<Prop> {
    Some(Prop {
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
        generate_billboard_cross(
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
                    // One cluster replaces what was 3-5 individual leaves (~70% reduction)
                    let size = rng.range(0.4, 0.75) * (state.length / base_length).sqrt().max(0.35);

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

fn generate_billboard_cross(
    center: Vec3,
    width: f32,
    height: f32,
    vertices: &mut Vec<PropVertex>,
    indices: &mut Vec<u32>,
) {
    let half_width = width * 0.5;
    let bottom_y = center.y - height * 0.3;
    let top_y = center.y + height * 0.7;
    let leaf_color = [0.32, 0.50, 0.22, 1.0];

    let orientations = [
        (Vec3::X, Vec3::Z),
        (Vec3::new(0.707, 0.0, 0.707), Vec3::new(-0.707, 0.0, 0.707)),
    ];

    for (right_dir, normal) in orientations {
        let base_idx = vertices.len() as u32;

        let corners = [
            Vec3::new(
                center.x - right_dir.x * half_width,
                bottom_y,
                center.z - right_dir.z * half_width,
            ),
            Vec3::new(
                center.x + right_dir.x * half_width,
                bottom_y,
                center.z + right_dir.z * half_width,
            ),
            Vec3::new(
                center.x + right_dir.x * half_width,
                top_y,
                center.z + right_dir.z * half_width,
            ),
            Vec3::new(
                center.x - right_dir.x * half_width,
                top_y,
                center.z - right_dir.z * half_width,
            ),
        ];

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

        indices.extend_from_slice(&[base_idx, base_idx + 1, base_idx + 2]);
        indices.extend_from_slice(&[base_idx, base_idx + 2, base_idx + 3]);
        indices.extend_from_slice(&[base_idx, base_idx + 2, base_idx + 1]);
        indices.extend_from_slice(&[base_idx, base_idx + 3, base_idx + 2]);
    }

    let trunk_base_idx = vertices.len() as u32;
    let trunk_radius = width * 0.08;
    let trunk_color = [0.30, 0.20, 0.12, 1.0];

    for i in 0..4 {
        let angle = (i as f32 / 4.0) * PI * 2.0;
        let (sin_a, cos_a) = angle.sin_cos();
        let offset = Vec3::new(cos_a * trunk_radius, 0.0, sin_a * trunk_radius);
        let normal = Vec3::new(cos_a, 0.0, sin_a);

        vertices.push(PropVertex {
            position: (center - Vec3::Y * height * 0.3 + offset).into(),
            normal: normal.into(),
            color: trunk_color,
            uv: [i as f32 / 4.0, 0.0],
            texture_id: 0,
        });
        vertices.push(PropVertex {
            position: (center + offset).into(),
            normal: normal.into(),
            color: trunk_color,
            uv: [i as f32 / 4.0, 1.0],
            texture_id: 0,
        });
    }

    for i in 0..4 {
        let next = (i + 1) % 4;
        let b1 = trunk_base_idx + i * 2;
        let t1 = trunk_base_idx + i * 2 + 1;
        let b2 = trunk_base_idx + next * 2;
        let t2 = trunk_base_idx + next * 2 + 1;
        indices.extend_from_slice(&[b1, t1, b2, b2, t1, t2]);
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
    let vertex_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("Tree Vertex Buffer"),
        contents: bytemuck::cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Tree Index Buffer"),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
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
    min_branch_thickness: f32,
}

impl LodRenderParams {
    fn oak_for_lod(lod: u32) -> Self {
        match lod {
            0 => Self {
                branch_segments: 8,
                leaf_density: 1.0,
                min_branch_thickness: 0.005,
            },
            1 => Self {
                branch_segments: 6,
                leaf_density: 0.7,
                min_branch_thickness: 0.008,
            },
            2 => Self {
                branch_segments: 4,
                leaf_density: 0.4,
                min_branch_thickness: 0.012,
            },
            _ => Self {
                branch_segments: 3,
                leaf_density: 0.0,
                min_branch_thickness: 1.0,
            },
        }
    }

    fn pine_for_lod(lod: u32) -> Self {
        match lod {
            0 => Self {
                branch_segments: 8,
                leaf_density: 1.0,
                min_branch_thickness: 0.005,
            },
            1 => Self {
                branch_segments: 6,
                leaf_density: 0.5,
                min_branch_thickness: 0.015,
            },
            2 => Self {
                branch_segments: 4,
                leaf_density: 0.2,
                min_branch_thickness: 0.025,
            },
            _ => Self {
                branch_segments: 4,
                leaf_density: 0.0,
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
        generate_billboard_cross(
            Vec3::new(0.0, 3.0, 0.0),
            4.5,
            6.0,
            &mut vertices,
            &mut indices,
        );
        return create_mesh(device, &vertices, &indices);
    }

    let structure = oak_tree_structure();
    let render_params = LodRenderParams::oak_for_lod(lod);
    generate_tree_mesh(device, &structure, &render_params)
}
