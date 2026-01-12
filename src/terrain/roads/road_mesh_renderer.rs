use crate::renderer::world_renderer::{PickedPoint, VisibleChunk};
use crate::resources::InputState;
use crate::terrain::roads::road_editor::RoadEditor;
use crate::terrain::roads::road_mesh_manager::{
    ChunkId, ChunkMesh, CrossSection, MeshConfig, RoadMeshManager, RoadVertex,
};
use crate::terrain::roads::roads::{RoadManager, apply_commands};
/// road_mesh_renderer.rs
use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use wgpu::Device;
use wgpu::util::DeviceExt;

pub struct RoadRenderSubsystem {
    pub mesh_manager: RoadMeshManager,
    pub mesh_renderer: RoadMeshRenderer,

    pub road_gpu_storage: RoadGpuStorage,
    pub gpu_chunks: HashMap<ChunkId, u64>, // topo_version per chunk

    pub cross_section: CrossSection,
    pub material_array: MaterialArray,

    pub visible_chunks: Vec<ChunkId>,
    pub road_manager: RoadManager,
    pub road_editor: RoadEditor,
    pub road_vertex_buffer: Option<wgpu::Buffer>,
    pub road_index_buffer: Option<wgpu::Buffer>,
}
impl RoadRenderSubsystem {
    pub fn new(cross_section: CrossSection) -> Self {
        Self {
            mesh_manager: RoadMeshManager::new(MeshConfig::default()),
            mesh_renderer: RoadMeshRenderer::new(),
            road_gpu_storage: RoadGpuStorage::new(),
            gpu_chunks: HashMap::new(),
            cross_section,
            material_array: MaterialArray::new(),
            visible_chunks: Vec::new(),
            road_manager: RoadManager::new(),
            road_editor: RoadEditor::new(),
            road_vertex_buffer: None,
            road_index_buffer: None,
        }
    }

    /// Call this each frame with visible chunk IDs
    pub fn update(
        &mut self,
        visible_chunks: &Vec<VisibleChunk>,
        device: &Device,
        input: &mut InputState,
        picked_point: &Option<PickedPoint>,
    ) {
        let road_commands = self
            .road_editor
            .update(&self.road_manager, input, picked_point);
        if !road_commands.is_empty() {
            println!("{:?}", road_commands);
            let results = apply_commands(
                &mut self.mesh_manager,
                &mut self.road_manager,
                &road_commands,
            );
        }
        // Clear old GPU buffers
        self.road_vertex_buffer = None;
        self.road_index_buffer = None;
        self.road_gpu_storage.draw_calls.clear();

        let mut all_vertices: Vec<RoadVertex> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();

        let mut current_index_offset: u32 = 0;

        for v in visible_chunks {
            let chunk_id = v.id;
            let needs_rebuild = self.mesh_manager.chunk_needs_update(
                chunk_id,
                &self.cross_section,
                &self.road_manager,
            );

            let mesh = if needs_rebuild {
                self.mesh_manager.update_chunk_mesh(
                    chunk_id,
                    &self.cross_section,
                    &self.road_manager,
                )
            } else {
                // Use cached mesh or skip if None
                match self.mesh_manager.get_chunk_mesh(chunk_id) {
                    Some(mesh) => mesh,
                    None => continue,
                }
            };

            // Update version tracking
            self.gpu_chunks.insert(chunk_id, mesh.topo_version);

            // Append vertices
            let vertex_start = all_vertices.len() as u32;
            all_vertices.extend(mesh.vertices.iter().cloned());

            // Append offset indices (all materials mixed — safe because layer is per-vertex)
            let chunk_index_count = mesh.indices.len() as u32;
            let offset_indices: Vec<u32> = mesh.indices.iter().map(|&i| i + vertex_start).collect();

            let index_start = current_index_offset;
            all_indices.extend(offset_indices);
            current_index_offset += chunk_index_count;

            // One draw call per chunk
            let draw_call = DrawCall {
                chunk_id,
                material_layer_index: 0, // unused now — layer comes from vertices
                index_range: index_start..index_start + chunk_index_count,
                vertex_offset: vertex_start,
            };
            self.road_gpu_storage.draw_calls.push(draw_call);
        }

        if all_vertices.is_empty() {
            return;
        }

        // Create big GPU buffers
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Road Vertex Buffer (all chunks)"),
            contents: bytemuck::cast_slice(&all_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Road Index Buffer (all chunks)"),
            contents: bytemuck::cast_slice(&all_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        self.road_vertex_buffer = Some(vertex_buffer);
        self.road_index_buffer = Some(index_buffer);
    }

    /// Get draw calls for current frame
    pub fn draw_calls(&self) -> &[DrawCall] {
        &self.road_gpu_storage.draw_calls
    }
}

struct GpuRoadChunk {
    topo_version: u64,
}

#[derive(Clone, Debug, Default)]
pub struct MaterialArray {
    mapping: BTreeMap<u32, u32>,
}

impl MaterialArray {
    pub fn new() -> Self {
        Self {
            mapping: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, material_id: u32, layer_index: u32) {
        self.mapping.insert(material_id, layer_index);
    }

    pub fn get(&self, material_id: u32) -> Option<u32> {
        self.mapping.get(&material_id).copied()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DrawCall {
    pub chunk_id: ChunkId,
    pub material_layer_index: u32,
    pub index_range: Range<u32>,
    pub vertex_offset: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct UploadMesh {
    pub chunk_id: ChunkId,
    pub vertices: Vec<RoadVertex>,
}

#[derive(Clone, Debug, Default)]
pub struct RoadGpuStorage {
    pub uploaded_vertex_buffers: Vec<UploadMesh>,
    pub uploaded_index_buffers: Vec<(u32, Vec<u32>)>,
    pub draw_calls: Vec<DrawCall>,
}

impl RoadGpuStorage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn upload_vertices(&mut self, chunk_id: ChunkId, vertices: Vec<RoadVertex>) {
        self.uploaded_vertex_buffers
            .push(UploadMesh { chunk_id, vertices });
    }

    pub fn upload_indices(&mut self, material_layer_index: u32, indices: Vec<u32>) {
        self.uploaded_index_buffers
            .push((material_layer_index, indices));
    }

    pub fn add_draw_call(&mut self, draw_call: DrawCall) {
        self.draw_calls.push(draw_call);
    }
}

pub struct RoadMeshRenderer {
    last_draw_calls: Vec<DrawCall>,
}

impl Default for RoadMeshRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl RoadMeshRenderer {
    pub fn new() -> Self {
        Self {
            last_draw_calls: Vec::new(),
        }
    }

    pub fn prepare_and_upload(
        &mut self,
        chunk_id: ChunkId,
        chunk_mesh: &ChunkMesh,
        material_array: &MaterialArray,
        gpu: &mut RoadGpuStorage,
    ) {
        self.last_draw_calls.clear();

        gpu.upload_vertices(chunk_id, chunk_mesh.vertices.clone());

        let mut per_material: BTreeMap<u32, (usize, Vec<u32>)> = BTreeMap::new();

        let triangle_count = chunk_mesh.indices.len() / 3;
        for tri_idx in 0..triangle_count {
            let base = tri_idx * 3;
            let i0 = chunk_mesh.indices[base];
            let i1 = chunk_mesh.indices[base + 1];
            let i2 = chunk_mesh.indices[base + 2];

            let material_id = chunk_mesh.vertices[i0 as usize].material_id;

            let entry = per_material
                .entry(material_id)
                .or_insert((tri_idx, Vec::new()));
            entry.1.push(i0);
            entry.1.push(i1);
            entry.1.push(i2);
        }

        let mut global_index_offset: u32 = 0;

        for (material_id, (_first_tri_idx, indices)) in per_material.iter() {
            let layer_index = material_array.get(*material_id).unwrap_or(0);
            let index_count = indices.len() as u32;

            gpu.upload_indices(layer_index, indices.clone());

            let draw_call = DrawCall {
                chunk_id,
                material_layer_index: layer_index,
                index_range: global_index_offset..(global_index_offset + index_count),
                vertex_offset: 0,
            };

            self.last_draw_calls.push(draw_call.clone());
            gpu.add_draw_call(draw_call);

            global_index_offset += index_count;
        }
    }

    pub fn last_draw_calls(&self) -> &[DrawCall] {
        &self.last_draw_calls
    }
}

pub const WGSL_FRAGMENT_SHADER: &str = r#"
@group(0) @binding(0) var texture_array: texture_2d_array<f32>;
@group(0) @binding(1) var texture_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) material_layer_index: u32,
}

fn sample_road_texture(uv: vec2<f32>, material_layer_index: u32) -> vec4<f32> {
    return textureSample(texture_array, texture_sampler, uv, i32(material_layer_index));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let road_color = sample_road_texture(in.uv, in.material_layer_index);
    return road_color;
}
"#;

pub const WGSL_SDF_TEXT_OVERLAY: &str = r#"
@group(1) @binding(0) var glyph_atlas: texture_2d_array<f32>;
@group(1) @binding(1) var glyph_sampler: sampler;

const SDF_EDGE_VALUE: f32 = 0.5;
const SDF_SMOOTHING: f32 = 0.1;
const TEXT_BLEND_FACTOR: f32 = 1.0;
const TEXT_COLOR: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);

fn sample_sdf_glyph(text_uv: vec2<f32>, glyph_layer: u32) -> f32 {
    let distance = textureSample(glyph_atlas, glyph_sampler, text_uv, i32(glyph_layer)).r;
    let alpha = smoothstep(SDF_EDGE_VALUE - SDF_SMOOTHING, SDF_EDGE_VALUE + SDF_SMOOTHING, distance);
    return alpha;
}

fn blend_sdf_text_additive(
    base_color: vec4<f32>,
    text_uv: vec2<f32>,
    glyph_layer: u32,
    text_color: vec3<f32>
) -> vec4<f32> {
    let text_alpha = sample_sdf_glyph(text_uv, glyph_layer);
    let text_contribution = text_color * text_alpha * TEXT_BLEND_FACTOR;
    return vec4<f32>(base_color.rgb + text_contribution, base_color.a);
}

fn apply_text_overlay(base_color: vec4<f32>, text_uv: vec2<f32>, glyph_layer: u32) -> vec4<f32> {
    return blend_sdf_text_additive(base_color, text_uv, glyph_layer, TEXT_COLOR);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn vertex(
        x: f32,
        y: f32,
        z: f32,
        nx: f32,
        ny: f32,
        nz: f32,
        u: f32,
        v: f32,
        mat: u32,
    ) -> RoadVertex {
        RoadVertex {
            position: [x, y, z],
            normal: [nx, ny, nz],
            uv: [u, v],
            material_id: mat,
        }
    }

    fn synthetic_chunk_two_materials_interleaved() -> ChunkMesh {
        let vertices = vec![
            vertex(0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0),
            vertex(1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0),
            vertex(0.5, 1.0, 0.0, 0.5, 0.5, 0.5, 0.25, 1.0, 0),
            vertex(2.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 0.0, 1),
            vertex(3.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.5, 0.0, 1),
            vertex(2.5, 1.0, 0.0, 0.5, 0.5, 0.5, 1.25, 1.0, 1),
            vertex(4.0, 0.0, 0.0, 0.5, 0.5, 0.5, 2.0, 0.0, 0),
            vertex(5.0, 0.0, 0.0, 0.5, 0.5, 0.5, 2.5, 0.0, 0),
            vertex(4.5, 1.0, 0.0, 0.5, 0.5, 0.5, 2.25, 1.0, 0),
            vertex(6.0, 0.0, 0.0, 0.5, 0.5, 0.5, 3.0, 0.0, 1),
            vertex(7.0, 0.0, 0.0, 0.5, 0.5, 0.5, 3.5, 0.0, 1),
            vertex(6.5, 1.0, 0.0, 0.5, 0.5, 0.5, 3.25, 1.0, 1),
        ];
        let indices = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        ChunkMesh {
            vertices,
            indices,
            topo_version: 1,
        }
    }

    fn standard_material_array() -> MaterialArray {
        let mut arr = MaterialArray::new();
        arr.insert(0, 10);
        arr.insert(1, 20);
        arr
    }

    #[test]
    fn test_single_vertex_upload_and_material_grouping() {
        let chunk = synthetic_chunk_two_materials_interleaved();
        let mats = standard_material_array();
        let mut gpu = RoadGpuStorage::new();
        let mut renderer = RoadMeshRenderer::new();

        renderer.prepare_and_upload(0, &chunk, &mats, &mut gpu);

        assert_eq!(gpu.uploaded_vertex_buffers.len(), 1);
        assert_eq!(gpu.uploaded_vertex_buffers[0].vertices.len(), 12);

        assert_eq!(gpu.uploaded_index_buffers.len(), 2);

        assert_eq!(gpu.draw_calls.len(), 2);
        assert!(gpu.draw_calls[0].material_layer_index < gpu.draw_calls[1].material_layer_index);
    }

    #[test]
    fn test_determinism_identical_output() {
        let chunk = synthetic_chunk_two_materials_interleaved();
        let mats = standard_material_array();

        let mut gpu1 = RoadGpuStorage::new();
        let mut r1 = RoadMeshRenderer::new();
        r1.prepare_and_upload(0, &chunk, &mats, &mut gpu1);

        let mut gpu2 = RoadGpuStorage::new();
        let mut r2 = RoadMeshRenderer::new();
        r2.prepare_and_upload(0, &chunk, &mats, &mut gpu2);

        assert_eq!(gpu1.draw_calls, gpu2.draw_calls);
        assert_eq!(gpu1.uploaded_index_buffers, gpu2.uploaded_index_buffers);
    }

    #[test]
    fn test_material_layer_index_matches_mapping() {
        let chunk = synthetic_chunk_two_materials_interleaved();
        let mats = standard_material_array();
        let mut gpu = RoadGpuStorage::new();
        let mut renderer = RoadMeshRenderer::new();

        renderer.prepare_and_upload(0, &chunk, &mats, &mut gpu);

        let calls = renderer.last_draw_calls();
        let layer_10_present = calls.iter().any(|c| c.material_layer_index == 10);
        let layer_20_present = calls.iter().any(|c| c.material_layer_index == 20);

        assert!(layer_10_present);
        assert!(layer_20_present);
    }

    #[test]
    fn test_batching_reduces_draw_calls() {
        let vertices = vec![
            vertex(0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0),
            vertex(1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0),
            vertex(0.5, 1.0, 0.0, 0.5, 0.5, 0.5, 0.25, 1.0, 0),
            vertex(2.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 0.0, 0),
            vertex(3.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.5, 0.0, 0),
            vertex(2.5, 1.0, 0.0, 0.5, 0.5, 0.5, 1.25, 1.0, 0),
        ];
        let indices = vec![0, 1, 2, 3, 4, 5];
        let chunk = ChunkMesh {
            vertices,
            indices,
            topo_version: 1,
        };

        let mut mats = MaterialArray::new();
        mats.insert(0, 0);

        let mut gpu = RoadGpuStorage::new();
        let mut renderer = RoadMeshRenderer::new();

        renderer.prepare_and_upload(0, &chunk, &mats, &mut gpu);

        assert_eq!(gpu.draw_calls.len(), 1);
        assert_eq!(
            gpu.draw_calls[0].index_range.end - gpu.draw_calls[0].index_range.start,
            6
        );
    }

    #[test]
    fn test_wgsl_fragment_shader_contains_texture_array() {
        assert!(WGSL_FRAGMENT_SHADER.contains("texture_array"));
        assert!(WGSL_FRAGMENT_SHADER.contains("texture_2d_array"));
        assert!(WGSL_FRAGMENT_SHADER.contains("material_layer_index"));
        assert!(WGSL_FRAGMENT_SHADER.contains("uv"));
        assert!(WGSL_FRAGMENT_SHADER.contains("sample_road_texture"));
    }

    #[test]
    fn test_wgsl_sdf_overlay_contains_required() {
        assert!(WGSL_SDF_TEXT_OVERLAY.contains("glyph_atlas"));
        assert!(WGSL_SDF_TEXT_OVERLAY.contains("texture_2d_array"));
        assert!(WGSL_SDF_TEXT_OVERLAY.contains("smoothstep"));
        assert!(WGSL_SDF_TEXT_OVERLAY.contains("blend_sdf_text_additive"));
        assert!(WGSL_SDF_TEXT_OVERLAY.contains("SDF_SMOOTHING"));
        assert!(WGSL_SDF_TEXT_OVERLAY.contains("TEXT_BLEND_FACTOR"));
    }

    #[test]
    fn test_index_buffer_contents_correct() {
        let chunk = synthetic_chunk_two_materials_interleaved();
        let mats = standard_material_array();
        let mut gpu = RoadGpuStorage::new();
        let mut renderer = RoadMeshRenderer::new();

        renderer.prepare_and_upload(0, &chunk, &mats, &mut gpu);

        let mat0_buf = gpu
            .uploaded_index_buffers
            .iter()
            .find(|(layer, _)| *layer == 10)
            .map(|(_, idx)| idx.clone())
            .unwrap();
        assert_eq!(mat0_buf, vec![0, 1, 2, 6, 7, 8]);

        let mat1_buf = gpu
            .uploaded_index_buffers
            .iter()
            .find(|(layer, _)| *layer == 20)
            .map(|(_, idx)| idx.clone())
            .unwrap();
        assert_eq!(mat1_buf, vec![3, 4, 5, 9, 10, 11]);
    }

    #[test]
    fn test_draw_call_ordering_is_deterministic() {
        let chunk = synthetic_chunk_two_materials_interleaved();
        let mats = standard_material_array();

        for _ in 0..5 {
            let mut gpu = RoadGpuStorage::new();
            let mut renderer = RoadMeshRenderer::new();
            renderer.prepare_and_upload(0, &chunk, &mats, &mut gpu);

            assert_eq!(gpu.draw_calls[0].material_layer_index, 10);
            assert_eq!(gpu.draw_calls[1].material_layer_index, 20);
        }
    }

    #[test]
    fn test_vertex_buffer_not_duplicated() {
        let chunk = synthetic_chunk_two_materials_interleaved();
        let mats = standard_material_array();
        let mut gpu = RoadGpuStorage::new();
        let mut renderer = RoadMeshRenderer::new();

        renderer.prepare_and_upload(0, &chunk, &mats, &mut gpu);

        assert_eq!(gpu.uploaded_vertex_buffers.len(), 1);
        assert_eq!(gpu.uploaded_vertex_buffers[0].vertices, chunk.vertices);
    }

    #[test]
    fn test_last_draw_calls_matches_gpu() {
        let chunk = synthetic_chunk_two_materials_interleaved();
        let mats = standard_material_array();
        let mut gpu = RoadGpuStorage::new();
        let mut renderer = RoadMeshRenderer::new();

        assert!(renderer.last_draw_calls().is_empty());

        renderer.prepare_and_upload(0, &chunk, &mats, &mut gpu);

        assert_eq!(renderer.last_draw_calls().len(), 2);
        assert_eq!(renderer.last_draw_calls(), &gpu.draw_calls[..]);
    }

    #[test]
    fn test_three_materials_sorted_correctly() {
        let vertices = vec![
            vertex(0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 2),
            vertex(1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 2),
            vertex(0.5, 1.0, 0.0, 0.5, 0.5, 0.5, 0.25, 1.0, 2),
            vertex(2.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 0.0, 0),
            vertex(3.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.5, 0.0, 0),
            vertex(2.5, 1.0, 0.0, 0.5, 0.5, 0.5, 1.25, 1.0, 0),
            vertex(4.0, 0.0, 0.0, 0.5, 0.5, 0.5, 2.0, 0.0, 1),
            vertex(5.0, 0.0, 0.0, 0.5, 0.5, 0.5, 2.5, 0.0, 1),
            vertex(4.5, 1.0, 0.0, 0.5, 0.5, 0.5, 2.25, 1.0, 1),
        ];
        let indices = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
        let chunk = ChunkMesh {
            vertices,
            indices,
            topo_version: 1,
        };

        let mut mats = MaterialArray::new();
        mats.insert(0, 100);
        mats.insert(1, 200);
        mats.insert(2, 300);

        let mut gpu = RoadGpuStorage::new();
        let mut renderer = RoadMeshRenderer::new();

        renderer.prepare_and_upload(0, &chunk, &mats, &mut gpu);

        assert_eq!(gpu.draw_calls.len(), 3);
        assert_eq!(gpu.draw_calls[0].material_layer_index, 100);
        assert_eq!(gpu.draw_calls[1].material_layer_index, 200);
        assert_eq!(gpu.draw_calls[2].material_layer_index, 300);
    }

    #[test]
    fn test_arc_length_uv_preserved() {
        let vertices = vec![
            vertex(0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0),
            vertex(1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.123, 0.5, 0),
            vertex(0.5, 1.0, 0.0, 0.5, 0.5, 0.5, 0.456, 1.0, 0),
        ];
        let indices = vec![0, 1, 2];
        let chunk = ChunkMesh {
            vertices: vertices.clone(),
            indices,
            topo_version: 1,
        };

        let mut mats = MaterialArray::new();
        mats.insert(0, 0);

        let mut gpu = RoadGpuStorage::new();
        let mut renderer = RoadMeshRenderer::new();

        renderer.prepare_and_upload(0, &chunk, &mats, &mut gpu);

        assert_eq!(gpu.uploaded_vertex_buffers[0].vertices[0].uv, [0.0, 0.0]);
        assert_eq!(gpu.uploaded_vertex_buffers[0].vertices[1].uv, [0.123, 0.5]);
        assert_eq!(gpu.uploaded_vertex_buffers[0].vertices[2].uv, [0.456, 1.0]);
    }

    #[test]
    fn test_chunk_id_propagated() {
        let vertices = vec![
            vertex(0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0),
            vertex(1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0),
            vertex(0.5, 1.0, 0.0, 0.5, 0.5, 0.5, 0.25, 1.0, 0),
        ];
        let indices = vec![0, 1, 2];
        let chunk = ChunkMesh {
            vertices,
            indices,
            topo_version: 42,
        };

        let mut mats = MaterialArray::new();
        mats.insert(0, 0);

        let mut gpu = RoadGpuStorage::new();
        let mut renderer = RoadMeshRenderer::new();

        renderer.prepare_and_upload(99, &chunk, &mats, &mut gpu);

        assert_eq!(gpu.uploaded_vertex_buffers[0].chunk_id, 99);
        assert_eq!(gpu.draw_calls[0].chunk_id, 99);
    }

    #[test]
    fn test_empty_mesh_no_draw_calls() {
        let chunk = ChunkMesh {
            vertices: vec![],
            indices: vec![],
            topo_version: 1,
        };
        let mats = MaterialArray::new();
        let mut gpu = RoadGpuStorage::new();
        let mut renderer = RoadMeshRenderer::new();

        renderer.prepare_and_upload(0, &chunk, &mats, &mut gpu);

        assert_eq!(gpu.uploaded_vertex_buffers.len(), 1);
        assert!(gpu.uploaded_vertex_buffers[0].vertices.is_empty());
        assert!(gpu.uploaded_index_buffers.is_empty());
        assert!(gpu.draw_calls.is_empty());
    }

    #[test]
    fn test_index_ranges_sequential() {
        let chunk = synthetic_chunk_two_materials_interleaved();
        let mats = standard_material_array();
        let mut gpu = RoadGpuStorage::new();
        let mut renderer = RoadMeshRenderer::new();

        renderer.prepare_and_upload(0, &chunk, &mats, &mut gpu);

        assert_eq!(gpu.draw_calls[0].index_range, 0..6);
        assert_eq!(gpu.draw_calls[1].index_range, 6..12);
    }

    #[test]
    fn test_unknown_material_defaults_to_zero() {
        let vertices = vec![
            vertex(0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 999),
            vertex(1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 999),
            vertex(0.5, 1.0, 0.0, 0.5, 0.5, 0.5, 0.25, 1.0, 999),
        ];
        let indices = vec![0, 1, 2];
        let chunk = ChunkMesh {
            vertices,
            indices,
            topo_version: 1,
        };

        let mats = MaterialArray::new();
        let mut gpu = RoadGpuStorage::new();
        let mut renderer = RoadMeshRenderer::new();

        renderer.prepare_and_upload(0, &chunk, &mats, &mut gpu);

        assert_eq!(gpu.draw_calls[0].material_layer_index, 0);
    }
}
