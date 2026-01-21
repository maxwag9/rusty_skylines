use crate::components::camera::Camera;
use crate::paths::shader_dir;
use crate::renderer::pipelines::Pipelines;
use crate::renderer::procedural_render_manager::{PipelineOptions, RenderManager};
use crate::renderer::world_renderer::TerrainRenderer;
use crate::terrain::roads::road_mesh_manager::RoadVertex;
use crate::terrain::roads::road_mesh_renderer::RoadRenderSubsystem;
use crate::ui::vertex::Vertex;
use glam::{Mat4, Vec3};
use wgpu::PrimitiveTopology::TriangleList;
use wgpu::TextureFormat::Depth32Float;
use wgpu::{
    BlendState, CompareFunction, DepthBiasState, DepthStencilState, Face, IndexFormat, RenderPass,
    StencilState,
};

// Helper to create the shadow texture
pub fn create_shadow_texture(device: &wgpu::Device, size: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Shadow Map"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // DEPTH32 is standard for shadows.
        // TEXTURE_BINDING allows reading it in the main pass.
        // RENDER_ATTACHMENT allows writing to it in the shadow pass.
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    // Create a "comparison" sampler for PCF (Soft Shadows)
    // You likely need to store this sampler alongside the texture view
    // or in your global resource bind group.

    texture.create_view(&wgpu::TextureViewDescriptor::default())
}
pub fn compute_light_matrix(target_pos: Vec3, sun_direction: Vec3) -> Mat4 {
    let center = target_pos;
    let box_size = 100.0;

    // sun_direction points FROM surface TO sun
    // So light position is center + sun_direction * distance
    let light_pos = center + sun_direction * box_size;

    let view = Mat4::look_at_rh(light_pos, center, Vec3::Y);

    // Make sure near/far encompasses your scene
    let projection = Mat4::orthographic_rh(
        -box_size,
        box_size,
        -box_size,
        box_size,
        0.1,
        box_size * 3.0, // Adjusted for better depth precision
    );

    projection * view
}

pub fn render_roads_shadows(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    road_renderer: &RoadRenderSubsystem,
    pipelines: &Pipelines,
) {
    let bias = DepthBiasState {
        constant: -3, // Positive pushes geometry away from light
        slope_scale: -2.0,
        clamp: 0.0,
    };
    // This sets the road pipeline + texture array bind group (bind group 0)
    let shadow_shader_path = shader_dir().join("shadows.wgsl");
    render_manager.render(
        Vec::new(),
        "Roads Shadows",
        shadow_shader_path.as_path(), // file containing shadow vertex-only shader
        PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(DepthStencilState {
                format: Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::LessEqual,
                stencil: Default::default(),
                bias,
            }),
            msaa_samples: 1,
            vertex_layouts: Vec::from([RoadVertex::layout()]),
            cull_mode: Some(Face::Back),
            blend: Some(BlendState::ALPHA_BLENDING),
            shadow_pass: true,
        },
        &[&pipelines.uniforms.buffer],
        pass,
        pipelines,
    );

    for chunk_id in &road_renderer.visible_draw_list {
        if let Some(gpu) = road_renderer.chunk_gpu.get(chunk_id) {
            pass.set_vertex_buffer(0, gpu.vertex.slice(..));
            pass.set_index_buffer(gpu.index.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..gpu.index_count, 0, 0..1);
        }
    }
    if road_renderer.preview_gpu.is_empty() {
        return;
    }
    let (Some(vb), Some(ib)) = (&road_renderer.preview_gpu.vb, &road_renderer.preview_gpu.ib)
    else {
        return;
    };

    let nudge_bias = DepthBiasState {
        constant: -4,
        slope_scale: -2.0,
        clamp: 0.0,
    };
    render_manager.render(
        Vec::new(),
        "Roads Preview Shadows",
        shadow_shader_path.as_path(), // file containing full vertex+fragment shader
        PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(DepthStencilState {
                format: Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: nudge_bias,
            }),
            msaa_samples: 1,
            vertex_layouts: Vec::from([RoadVertex::layout()]),
            cull_mode: Some(Face::Back),
            blend: Some(BlendState::ALPHA_BLENDING),
            shadow_pass: true,
        },
        &[&pipelines.uniforms.buffer],
        pass,
        pipelines,
    );
    pass.set_vertex_buffer(0, vb.slice(..));
    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
    pass.draw_indexed(0..road_renderer.preview_gpu.index_count, 0, 0..1);
}
pub fn render_terrain_shadows(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    terrain_renderer: &TerrainRenderer,
    pipelines: &Pipelines,
    camera: &Camera,
    aspect: f32,
) {
    let shadows_shader_path = shader_dir().join("shadows.wgsl");
    render_manager.render(
        Vec::new(),
        "Terrain Pipeline (Above Water) Shadows",
        shadows_shader_path.as_path(), // file containing vertex shader
        PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(DepthStencilState {
                format: Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::LessEqual,
                stencil: StencilState::default(),
                bias: Default::default(),
            }),
            msaa_samples: 1,
            vertex_layouts: Vec::from([Vertex::desc()]),
            blend: Some(BlendState::REPLACE),
            cull_mode: Some(Face::Back),
            shadow_pass: true,
        },
        &[&pipelines.uniforms.buffer],
        pass,
        pipelines,
    );
    terrain_renderer.render(pass, camera, aspect, false);

    render_manager.render(
        Vec::new(),
        "Terrain Pipeline (Under Water) Shadows",
        shadows_shader_path.as_path(), // file containing vertex shader
        PipelineOptions {
            topology: TriangleList,
            depth_stencil: Some(DepthStencilState {
                format: Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::LessEqual,
                stencil: StencilState::default(),
                bias: Default::default(),
            }),
            msaa_samples: 1,
            vertex_layouts: Vec::from([Vertex::desc()]),
            blend: Some(BlendState::REPLACE),
            cull_mode: Some(Face::Back),
            shadow_pass: true,
        },
        &[&pipelines.uniforms.buffer],
        pass,
        pipelines,
    );
    terrain_renderer.render(pass, camera, aspect, true);
}
