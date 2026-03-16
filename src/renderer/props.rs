use crate::data::Settings;
use crate::helpers::paths::shader_dir;
use crate::helpers::positions::{ChunkCoord, ChunkSize, LocalPos, WorldPos};
use crate::renderer::pipelines::Pipelines;
use crate::renderer::render_passes::{
    color_and_normals_and_instance_targets, depth_stencil, make_shadow_option,
};
use crate::ui::input::Input;
use crate::world::camera::Camera;
use crate::world::terrain::terrain_subsystem::{CursorMode, Terrain};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};
use std::collections::HashMap;
use std::f32::consts::PI;
use wgpu::PrimitiveTopology::TriangleList;
use wgpu::util::DeviceExt;
use wgpu::*;
use wgpu_render_manager::generator::{TextureKey, TextureParams};
use wgpu_render_manager::pipelines::PipelineOptions;
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

pub struct Props {
    pub props: HashMap<String, Prop>,
    pub chunks: HashMap<ChunkCoord, PropChunk>,
    pub prev_models: HashMap<u64, [[f32; 4]; 4]>, // key: hash of (chunk, archetype, index)
}

impl Props {
    pub fn new() -> Self {
        Self {
            props: HashMap::new(),
            chunks: HashMap::new(),
            prev_models: HashMap::new(),
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
                        let render_pos = inst
                            .pos
                            .to_render_pos(camera.eye_world(), camera.chunk_size);
                        let model = Mat4::from_scale_rotation_translation(
                            Vec3::splat(inst.scale),
                            glam::Quat::from_rotation_y(inst.rotation_y_rad),
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
        let height_above_terrain = (eye.local.y - terrain_height).max(0.0);
        let chunk_size = camera.chunk_size;

        for visible_chunk in terrain.visible.iter() {
            let coord = visible_chunk.coords.chunk_coord;

            let Some(chunk) = self.chunks.get(&coord) else {
                continue;
            };

            // Calculate LOD for this chunk
            let instance_terrain_height = terrain.get_height_at(eye, true);
            let dist = eye.distance_to(
                WorldPos::new(coord, LocalPos::new(0.0, instance_terrain_height, 0.0)),
                chunk_size,
            );
            let lod_level = select_lod(dist, chunk_size);

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

                let shader_path = shader_dir().join("props.wgsl");
                let shadow = make_shadow_option(settings, pipelines);
                let targets = color_and_normals_and_instance_targets(pipelines);

                // Set up pipeline
                render_manager.render(
                    &prop.texture_keys,
                    shader_path.as_path(),
                    &PipelineOptions {
                        topology: TriangleList,
                        depth_stencil: Some(depth_stencil(Default::default(), settings)),
                        msaa_samples: settings.msaa_samples,
                        vertex_layouts: Vec::from([
                            PropVertex::layout(),
                            GpuPropInstance::layout(),
                        ]),
                        cull_mode: Some(Face::Front),
                        targets: targets.clone(),
                        shadow: shadow.clone(),
                        ..Default::default()
                    },
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

fn select_lod(dist: f64, chunk_size: ChunkSize) -> u32 {
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

pub fn make_prop(key: &str, device: &Device) -> Option<Prop> {
    match key.to_lowercase().as_str() {
        "oak" | "oak_tree" => make_oak_tree(device),
        "pine" | "pine_tree" => make_pine_tree(device),
        _ => None,
    }
}

fn make_oak_tree(device: &Device) -> Option<Prop> {
    Some(Prop {
        lod0: Some(make_oak_lod0(device)),
        lod1: Some(make_oak_lod1(device)),
        lod2: Some(make_oak_lod2(device)),
        lod3: Some(make_oak_billboard(device)),
        texture_keys: [
            TextureKey::new(
                "leaves",
                TextureParams {
                    // Good defaults for leaf cards:
                    color_primary: [0.25, 0.45, 0.15, 1.0], // Deep green
                    color_secondary: [0.35, 0.55, 0.20, 1.0], // Lighter green variation
                    seed: 42,
                    scale: 1.5,     // 1.0 = moderate density, 2.0+ = very dense
                    roughness: 0.8, // Surface texture amount
                    octaves: 0.6,
                    persistence: 0.5,
                    lacunarity: 0.3,
                    _pad0: 0.0,
                    _pad1: 0.0,
                },
                512,
            ),
            TextureKey::notex(),
            TextureKey::notex(),
            TextureKey::notex(),
        ],
    })
}

/// LOD0: Full detail (~400 verts)
fn make_oak_lod0(device: &Device) -> Mesh {
    let mut verts: Vec<PropVertex> = Vec::new();
    let mut idxs: Vec<u32> = Vec::new();

    // Trunk parameters
    let trunk_h = 4.0;
    let trunk_r_bot = 0.35;
    let trunk_r_top = 0.18;
    let trunk_color = [0.35, 0.22, 0.12, 1.0];

    // Generate trunk
    generate_tapered_cylinder(
        &mut verts,
        &mut idxs,
        Vec3::ZERO,
        trunk_h,
        trunk_r_bot,
        trunk_r_top,
        12,
        5, // segments, rings
        trunk_color,
    );

    // Main branches (3 big ones)
    let branch_starts = [
        (Vec3::new(0.0, trunk_h * 0.7, 0.0), Vec3::new(1.2, 1.5, 0.3)),
        (
            Vec3::new(0.0, trunk_h * 0.75, 0.0),
            Vec3::new(-0.8, 1.6, 1.0),
        ),
        (
            Vec3::new(0.0, trunk_h * 0.8, 0.0),
            Vec3::new(0.2, 1.4, -1.1),
        ),
    ];
    for (start, dir) in branch_starts {
        generate_branch(
            &mut verts,
            &mut idxs,
            start,
            dir.normalize(),
            dir.length(),
            0.1,
            0.04, // radius start/end
            6,
            3,
            trunk_color,
        );
    }

    // Leaf crown parameters
    let crown_center = Vec3::new(0.0, trunk_h + 1.8, 0.0);
    let crown_radius_h = 2.8;
    let crown_radius_v = 2.2;
    let leaf_color = [0.25, 0.55, 0.18, 0.95];

    // Generate leaf cards (12 cards in spherical arrangement)
    generate_leaf_sphere(
        &mut verts,
        &mut idxs,
        crown_center,
        crown_radius_h,
        crown_radius_v,
        16,  // num cards
        1.6, // card size
        leaf_color,
    );

    // Inner leaf cards for density
    generate_leaf_sphere(
        &mut verts,
        &mut idxs,
        crown_center + Vec3::new(0.0, -0.3, 0.0),
        crown_radius_h * 0.6,
        crown_radius_v * 0.6,
        8,
        1.0,
        [0.18, 0.45, 0.12, 0.9],
    );

    let bounds_center = Vec3::new(0.0, trunk_h / 2.0 + 1.0, 0.0);
    let bounds_radius = trunk_h + crown_radius_v;

    build_mesh(device, &verts, &idxs, (bounds_center, bounds_radius))
}

/// LOD1: Medium detail (~200 verts)
fn make_oak_lod1(device: &Device) -> Mesh {
    let mut verts: Vec<PropVertex> = Vec::new();
    let mut idxs: Vec<u32> = Vec::new();

    let trunk_h = 4.0;
    let trunk_color = [0.35, 0.22, 0.12, 1.0];

    // Simpler trunk
    generate_tapered_cylinder(
        &mut verts,
        &mut idxs,
        Vec3::ZERO,
        trunk_h,
        0.35,
        0.18,
        8,
        3,
        trunk_color,
    );

    // Fewer leaf cards
    let crown_center = Vec3::new(0.0, trunk_h + 1.8, 0.0);
    generate_leaf_sphere(
        &mut verts,
        &mut idxs,
        crown_center,
        2.8,
        2.2,
        8,
        2.0,
        [0.25, 0.55, 0.18, 0.95],
    );

    let bounds_center = Vec3::new(0.0, trunk_h / 2.0 + 1.0, 0.0);
    build_mesh(device, &verts, &idxs, (bounds_center, 6.0))
}

/// LOD2: Low detail (~80 verts)
fn make_oak_lod2(device: &Device) -> Mesh {
    let mut verts: Vec<PropVertex> = Vec::new();
    let mut idxs: Vec<u32> = Vec::new();

    let trunk_h = 4.0;

    // Very simple trunk
    generate_tapered_cylinder(
        &mut verts,
        &mut idxs,
        Vec3::ZERO,
        trunk_h,
        0.35,
        0.18,
        6,
        2,
        [0.35, 0.22, 0.12, 1.0],
    );

    // Just 4 big leaf cards
    let crown_center = Vec3::new(0.0, trunk_h + 1.5, 0.0);
    generate_leaf_sphere(
        &mut verts,
        &mut idxs,
        crown_center,
        2.5,
        2.0,
        4,
        2.5,
        [0.25, 0.55, 0.18, 0.95],
    );

    build_mesh(device, &verts, &idxs, (Vec3::new(0.0, 3.0, 0.0), 6.0))
}

/// LOD3: Billboard (2 crossed quads)
fn make_oak_billboard(device: &Device) -> Mesh {
    let mut verts: Vec<PropVertex> = Vec::new();
    let mut idxs: Vec<u32> = Vec::new();

    let h = 7.0;
    let w = 5.0;
    let center_y = h / 2.0;

    // Two crossed quads
    for angle in [0.0, PI / 2.0] {
        let rot = Mat3::from_rotation_y(angle);
        let right = rot * Vec3::new(w / 2.0, 0.0, 0.0);

        let base = idxs.len() as u32 / 6 * 4;

        // Mixed color: trunk at bottom, leaves at top
        let positions = [
            Vec3::new(0.0, 0.0, 0.0) - right,
            Vec3::new(0.0, 0.0, 0.0) + right,
            Vec3::new(0.0, h, 0.0) + right,
            Vec3::new(0.0, h, 0.0) - right,
        ];
        let colors = [
            [0.35, 0.25, 0.12, 1.0], // bottom left - trunk
            [0.35, 0.25, 0.12, 1.0], // bottom right - trunk
            [0.25, 0.55, 0.18, 0.9], // top right - leaves
            [0.25, 0.55, 0.18, 0.9], // top left - leaves
        ];
        let uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

        let normal = rot * Vec3::Z;

        for i in 0..4 {
            verts.push(PropVertex {
                position: positions[i].to_array(),
                normal: normal.to_array(),
                color: colors[i],
                uv: uvs[i],
                texture_id: 0,
            });
        }

        idxs.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    build_mesh(
        device,
        &verts,
        &idxs,
        (Vec3::new(0.0, h / 2.0, 0.0), h / 2.0 + 1.0),
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// GEOMETRY GENERATORS
// ═══════════════════════════════════════════════════════════════════════════════

fn generate_tapered_cylinder(
    verts: &mut Vec<PropVertex>,
    idxs: &mut Vec<u32>,
    base: Vec3,
    height: f32,
    radius_bottom: f32,
    radius_top: f32,
    segments: u32,
    rings: u32,
    color: [f32; 4],
) {
    let base_idx = verts.len() as u32;

    // Generate vertices
    for ring in 0..=rings {
        let t = ring as f32 / rings as f32;
        let y = base.y + height * t;
        let radius = radius_bottom + (radius_top - radius_bottom) * t;

        // Slight wobble for organic look
        let wobble = 1.0 + (ring as f32 * 2.3).sin() * 0.05;

        for seg in 0..segments {
            let angle = (seg as f32 / segments as f32) * PI * 2.0;
            let seg_wobble = 1.0 + (seg as f32 * 1.7 + ring as f32).sin() * 0.08;
            let r = radius * wobble * seg_wobble;

            let x = base.x + angle.cos() * r;
            let z = base.z + angle.sin() * r;

            // Normal points outward
            let normal = Vec3::new(angle.cos(), 0.15, angle.sin()).normalize();

            // Slight color variation
            let bark_variation = 1.0 + ((seg as f32 + ring as f32 * 0.5).sin() * 0.1);
            let varied_color = [
                color[0] * bark_variation,
                color[1] * bark_variation,
                color[2] * bark_variation,
                color[3],
            ];

            verts.push(PropVertex {
                position: [x, y, z],
                normal: normal.to_array(),
                color: varied_color,
                uv: [seg as f32 / segments as f32, t],
                texture_id: 0,
            });
        }
    }

    // Generate indices
    for ring in 0..rings {
        for seg in 0..segments {
            let curr = base_idx + ring * segments + seg;
            let next = base_idx + ring * segments + (seg + 1) % segments;
            let curr_up = curr + segments;
            let next_up = next + segments;

            idxs.extend_from_slice(&[curr, next, curr_up]);
            idxs.extend_from_slice(&[next, next_up, curr_up]);
        }
    }

    // Cap the bottom
    let center_idx = verts.len() as u32;
    verts.push(PropVertex {
        position: base.to_array(),
        normal: [0.0, -1.0, 0.0],
        color,
        uv: [0.5, 0.5],
        texture_id: 0,
    });
    for seg in 0..segments {
        let next = (seg + 1) % segments;
        idxs.extend_from_slice(&[center_idx, base_idx + next, base_idx + seg]);
    }
}

fn generate_branch(
    verts: &mut Vec<PropVertex>,
    idxs: &mut Vec<u32>,
    start: Vec3,
    direction: Vec3,
    length: f32,
    radius_start: f32,
    radius_end: f32,
    segments: u32,
    rings: u32,
    color: [f32; 4],
) {
    let base_idx = verts.len() as u32;

    // Build local coordinate frame
    let up = if direction.y.abs() > 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let right = direction.cross(up).normalize();
    let forward = right.cross(direction).normalize();

    for ring in 0..=rings {
        let t = ring as f32 / rings as f32;
        let pos = start + direction * length * t;
        let radius = radius_start + (radius_end - radius_start) * t;

        for seg in 0..segments {
            let angle = (seg as f32 / segments as f32) * PI * 2.0;
            let offset = right * angle.cos() * radius + forward * angle.sin() * radius;
            let p = pos + offset;
            let normal = offset.normalize();

            verts.push(PropVertex {
                position: p.to_array(),
                normal: normal.to_array(),
                color,
                uv: [seg as f32 / segments as f32, t],
                texture_id: 0,
            });
        }
    }

    for ring in 0..rings {
        for seg in 0..segments {
            let curr = base_idx + ring * segments + seg;
            let next = base_idx + ring * segments + (seg + 1) % segments;
            let curr_up = curr + segments;
            let next_up = next + segments;

            idxs.extend_from_slice(&[curr, next, curr_up]);
            idxs.extend_from_slice(&[next, next_up, curr_up]);
        }
    }
}

fn generate_leaf_sphere(
    verts: &mut Vec<PropVertex>,
    idxs: &mut Vec<u32>,
    center: Vec3,
    radius_h: f32,
    radius_v: f32,
    num_cards: u32,
    card_size: f32,
    color: [f32; 4],
) {
    let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;

    for i in 0..num_cards {
        let t = i as f32 / num_cards as f32;

        // Fibonacci sphere distribution
        let theta = 2.0 * PI * t * golden_ratio;
        let phi = (1.0 - 2.0 * (i as f32 + 0.5) / num_cards as f32).acos();

        let dir = Vec3::new(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin());

        let card_center = center + dir * Vec3::new(radius_h, radius_v, radius_h);

        // Card faces outward with some random rotation
        let seed = i as f32 * 1.618;
        generate_leaf_card(
            verts,
            idxs,
            card_center,
            dir,
            card_size * (0.8 + (seed * 2.3).sin().abs() * 0.4),
            color,
            seed,
        );
    }
}

fn generate_leaf_card(
    verts: &mut Vec<PropVertex>,
    idxs: &mut Vec<u32>,
    center: Vec3,
    facing: Vec3,
    size: f32,
    color: [f32; 4],
    seed: f32,
) {
    let base_idx = verts.len() as u32;

    // Build card orientation
    let up = Vec3::Y;
    let right = if facing.y.abs() > 0.9 {
        Vec3::X
    } else {
        facing.cross(up).normalize()
    };
    let card_up = right.cross(facing).normalize();

    // Rotate card by seed for variety
    let rot_angle = seed * 0.5;
    let cos_a = rot_angle.cos();
    let sin_a = rot_angle.sin();
    let rotated_right = right * cos_a + card_up * sin_a;
    let rotated_up = card_up * cos_a - right * sin_a;

    let half = size / 2.0;

    // Quad corners
    let corners = [
        center - rotated_right * half - rotated_up * half,
        center + rotated_right * half - rotated_up * half,
        center + rotated_right * half + rotated_up * half,
        center - rotated_right * half + rotated_up * half,
    ];

    let uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

    // Alpha falloff at edges for soft look
    let alphas = [color[3] * 0.7, color[3] * 0.7, color[3], color[3]];

    // Slight color variation per card
    let variation = 1.0 + (seed * 3.7).sin() * 0.15;
    let varied_color = |alpha: f32| {
        [
            color[0] * variation,
            color[1] * (variation * 0.9 + 0.1),
            color[2] * variation,
            alpha,
        ]
    };

    for i in 0..4 {
        verts.push(PropVertex {
            position: corners[i].to_array(),
            normal: facing.to_array(),
            color: varied_color(alphas[i]),
            uv: uvs[i],
            texture_id: 1,
        });
    }

    // Two triangles, both sides (for alpha cards we want double-sided)
    idxs.extend_from_slice(&[
        base_idx,
        base_idx + 1,
        base_idx + 2,
        base_idx,
        base_idx + 2,
        base_idx + 3,
        // Backface
        base_idx,
        base_idx + 2,
        base_idx + 1,
        base_idx,
        base_idx + 3,
        base_idx + 2,
    ]);
}

fn build_mesh(
    device: &Device,
    vertices: &[PropVertex],
    indices: &[u32],
    bounds: (Vec3, f32),
) -> Mesh {
    let vertex_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("Prop Vertex Buffer"),
        contents: bytemuck::cast_slice(vertices),
        usage: BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
        label: Some("Prop Index Buffer"),
        contents: bytemuck::cast_slice(indices),
        usage: BufferUsages::INDEX,
    });

    Mesh {
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        bounds,
    }
}

fn make_pine_tree(device: &Device) -> Option<Prop> {
    Some(Prop {
        lod0: Some(make_pine_lod0(device)),
        lod1: Some(make_pine_lod1(device)),
        lod2: Some(make_pine_lod2(device)),
        lod3: Some(make_pine_billboard(device)),
        texture_keys: [
            TextureKey::notex(),
            TextureKey::notex(),
            TextureKey::notex(),
            TextureKey::notex(),
        ],
    })
}

fn make_pine_lod0(device: &Device) -> Mesh {
    let mut verts: Vec<PropVertex> = Vec::new();
    let mut idxs: Vec<u32> = Vec::new();

    let trunk_h = 6.0;
    let trunk_color = [0.3, 0.18, 0.08, 1.0];
    let needle_color = [0.1, 0.35, 0.15, 0.95];

    // Trunk
    generate_tapered_cylinder(
        &mut verts,
        &mut idxs,
        Vec3::ZERO,
        trunk_h,
        0.25,
        0.08,
        8,
        4,
        trunk_color,
    );

    // Cone-shaped foliage layers
    let layers = [
        (1.5, 2.2, 1.6), // (y, radius, height)
        (3.0, 1.8, 1.4),
        (4.2, 1.3, 1.2),
        (5.2, 0.8, 1.0),
    ];

    for (y, radius, height) in layers {
        generate_cone_layer(&mut verts, &mut idxs, y, radius, height, 8, needle_color);
    }

    build_mesh(
        device,
        &verts,
        &idxs,
        (Vec3::new(0.0, trunk_h / 2.0, 0.0), trunk_h),
    )
}

fn generate_cone_layer(
    verts: &mut Vec<PropVertex>,
    idxs: &mut Vec<u32>,
    base_y: f32,
    radius: f32,
    height: f32,
    segments: u32,
    color: [f32; 4],
) {
    let base_idx = verts.len() as u32;

    // Tip
    verts.push(PropVertex {
        position: [0.0, base_y + height, 0.0],
        normal: [0.0, 1.0, 0.0],
        color,
        uv: [0.5, 0.0],
        texture_id: 0,
    });

    // Base ring
    for seg in 0..segments {
        let angle = (seg as f32 / segments as f32) * PI * 2.0;
        let x = angle.cos() * radius;
        let z = angle.sin() * radius;

        let normal = Vec3::new(angle.cos(), 0.3, angle.sin()).normalize();

        let edge_alpha = color[3] * (0.6 + (seg as f32 * 0.7).sin().abs() * 0.4);

        verts.push(PropVertex {
            position: [x, base_y, z],
            normal: normal.to_array(),
            color: [color[0], color[1], color[2], edge_alpha],
            uv: [seg as f32 / segments as f32, 1.0],
            texture_id: 0,
        });
    }

    // Indices (cone)
    for seg in 0..segments {
        let next = (seg + 1) % segments;
        idxs.extend_from_slice(&[
            base_idx,            // tip
            base_idx + 1 + seg,  // current base
            base_idx + 1 + next, // next base
        ]);
    }
}

fn make_pine_lod1(device: &Device) -> Mesh {
    make_pine_lod0(device) // Simplified version
}

fn make_pine_lod2(device: &Device) -> Mesh {
    make_pine_lod0(device) // Even simpler
}

fn make_pine_billboard(device: &Device) -> Mesh {
    let mut verts: Vec<PropVertex> = Vec::new();
    let mut idxs: Vec<u32> = Vec::new();

    let h = 8.0;
    let w = 3.5;

    for angle in [0.0, PI / 2.0] {
        let rot = Mat3::from_rotation_y(angle);
        let right = rot * Vec3::new(w / 2.0, 0.0, 0.0);
        let base = verts.len() as u32;

        let positions = [
            -right,
            right,
            right + Vec3::new(0.0, h, 0.0),
            -right + Vec3::new(0.0, h, 0.0),
        ];

        let colors = [
            [0.3, 0.18, 0.08, 1.0],
            [0.3, 0.18, 0.08, 1.0],
            [0.1, 0.35, 0.15, 0.9],
            [0.1, 0.35, 0.15, 0.9],
        ];

        for i in 0..4 {
            verts.push(PropVertex {
                position: positions[i].to_array(),
                normal: (rot * Vec3::Z).to_array(),
                color: colors[i],
                uv: [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]][i],
                texture_id: 0,
            });
        }

        idxs.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    build_mesh(
        device,
        &verts,
        &idxs,
        (Vec3::new(0.0, h / 2.0, 0.0), h / 2.0 + 1.0),
    )
}
