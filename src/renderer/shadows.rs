use crate::components::camera::Camera;
use crate::data::Settings;
use crate::paths::shader_dir;
use crate::renderer::pipelines::Pipelines;
use crate::renderer::procedural_render_manager::{PipelineOptions, RenderManager};
use crate::renderer::render_passes::draw_visible_roads;
use crate::renderer::world_renderer::TerrainRenderer;
use crate::terrain::roads::road_mesh_manager::RoadVertex;
use crate::terrain::roads::road_subsystem::RoadRenderSubsystem;
use crate::ui::vertex::Vertex;
use glam::{Mat4, Vec3, Vec4};
use wgpu::PrimitiveTopology::TriangleList;
use wgpu::TextureFormat::Depth32Float;
use wgpu::{
    Buffer, CompareFunction, DepthBiasState, DepthStencilState, Device, Face, IndexFormat,
    RenderPass, StencilState, TextureView,
};

pub const CSM_CASCADES: usize = 4;
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowMatUniform {
    pub light_view_proj: [[f32; 4]; 4],
}
pub struct CascadedShadowMap {
    pub array_view: TextureView,
    pub layer_views: [TextureView; CSM_CASCADES],
    pub size: u32,

    pub light_mats: [Mat4; 4],
    pub splits: [f32; 4],

    pub shadow_mat_buffers: [Buffer; CSM_CASCADES], // <- NEW
}
pub fn create_shadow_mat_uniform_buffer(device: &Device) -> Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("CSM Shadow Mat Buffer"),
        size: size_of::<ShadowMatUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub fn create_csm_shadow_texture(device: &Device, size: u32, label: &str) -> CascadedShadowMap {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&format!(
            "CSM Shadow Map Array ({CSM_CASCADES} layers). {label}"
        )),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: CSM_CASCADES as u32,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    // Array view for sampling
    let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("CSM Shadow Array View"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        base_array_layer: 0,
        array_layer_count: Some(CSM_CASCADES as u32),
        ..Default::default()
    });

    // Per-layer views for rendering
    let layer_views = std::array::from_fn(|i| {
        texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("CSM Shadow Layer View {i}")),
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: i as u32,
            array_layer_count: Some(1),
            ..Default::default()
        })
    });

    CascadedShadowMap {
        array_view,
        layer_views,
        size,
        light_mats: [Mat4::IDENTITY; 4],
        splits: [1.0; 4],
        shadow_mat_buffers: std::array::from_fn(|_| create_shadow_mat_uniform_buffer(device)),
    }
}

// Defaults (tweak later)
pub const DEFAULT_SHADOW_DISTANCE: f32 = 200.0; // how far from camera to cast shadows

fn camera_basis_from_view_rotation_only(view: Mat4) -> (Vec3, Vec3, Vec3) {
    let inv = view.inverse();
    let right = inv.x_axis.truncate().normalize();
    let up = inv.y_axis.truncate().normalize();
    let forward = (-inv.z_axis.truncate()).normalize(); // view forward is -Z
    (right, up, forward)
}

fn frustum_slice_corners_ws(
    eye: Vec3,
    right: Vec3,
    up: Vec3,
    forward: Vec3,
    fov_y_radians: f32,
    aspect: f32,
    slice_near: f32,
    slice_far: f32,
) -> [Vec3; 8] {
    let tan_half = (fov_y_radians * 0.5).tan();

    let nh = tan_half * slice_near;
    let nw = nh * aspect;

    let fh = tan_half * slice_far;
    let fw = fh * aspect;

    let nc = eye + forward * slice_near;
    let fc = eye + forward * slice_far;

    // Near plane corners
    let ntl = nc + up * nh - right * nw;
    let ntr = nc + up * nh + right * nw;
    let nbl = nc - up * nh - right * nw;
    let nbr = nc - up * nh + right * nw;

    // Far plane corners
    let ftl = fc + up * fh - right * fw;
    let ftr = fc + up * fh + right * fw;
    let fbl = fc - up * fh - right * fw;
    let fbr = fc - up * fh + right * fw;

    [ntl, ntr, nbl, nbr, ftl, ftr, fbl, fbr]
}

pub fn compute_light_matrix_fit_frustum_slice_stable(
    frustum_corners_ws: &[Vec3; 8],
    sun_dir_surface_to_sun: Vec3,
    shadow_map_size: u32,
    stabilize: bool,
    reversed_z: bool,
) -> Mat4 {
    let sun_dir_raw = sun_dir_surface_to_sun;
    let sun_dir = if sun_dir_raw.length_squared() < 1e-8 {
        Vec3::new(0.3, 1.0, 0.2).normalize()
    } else {
        sun_dir_raw.normalize()
    };

    // Pick a stable "up"
    let up = if sun_dir.dot(Vec3::Y).abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };

    // Compute slice center + radius (bounding sphere)
    let mut center = Vec3::ZERO;
    for &c in frustum_corners_ws.iter() {
        center += c;
    }
    center *= 1.0 / 8.0;

    let mut radius = 0.0f32;
    for &c in frustum_corners_ws.iter() {
        radius = radius.max((c - center).length());
    }

    // Slight pad to avoid clipping due to floating error
    radius *= 1.05;

    // Light view: place the light "behind" the slice along sun_dir, looking at center.
    let light_eye = center + sun_dir * (radius * 3.0 + 10.0);
    let light_view = Mat4::look_at_rh(light_eye, center, up);

    // Center in light space
    let center_ls = (light_view * Vec4::new(center.x, center.y, center.z, 1.0)).truncate();

    // Snap center to texel grid (in LIGHT SPACE)
    let mut snapped_center = center_ls;
    if stabilize && shadow_map_size > 0 {
        let diameter = 2.0 * radius;
        let texel = diameter / shadow_map_size as f32;
        snapped_center.x = (snapped_center.x / texel).round() * texel;
        snapped_center.y = (snapped_center.y / texel).round() * texel;
    }

    // XY bounds are fixed by radius (stable)
    let min_x = snapped_center.x - radius;
    let max_x = snapped_center.x + radius;
    let min_y = snapped_center.y - radius;
    let max_y = snapped_center.y + radius;

    // Z bounds must still cover all corners (in light space)
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    for &c in frustum_corners_ws.iter() {
        let p = (light_view * Vec4::new(c.x, c.y, c.z, 1.0)).truncate();
        min_z = min_z.min(p.z);
        max_z = max_z.max(p.z);
    }

    // Convert light-space z extents to positive near/far distances for orthographic_rh().
    // In RH view space, visible points typically have negative z, so distances are -z.
    const Z_PAD: f32 = 150.0;
    let near = (-max_z - Z_PAD).max(0.1);
    let far = (-min_z + Z_PAD).max(near + 0.1);
    let light_proj = if reversed_z {
        // reversed Z: near maps to 1, far maps to 0
        Mat4::orthographic_rh(min_x, max_x, min_y, max_y, far, near)
    } else {
        Mat4::orthographic_rh(min_x, max_x, min_y, max_y, near, far)
    };
    light_proj * light_view
}
pub fn cascade_splits_from_ratios(near: f32, far: f32, ratios: [f32; 4]) -> [f32; 4] {
    let range = (far - near).max(1.0);
    [
        near + range * ratios[0],
        near + range * ratios[1],
        near + range * ratios[2],
        near + range * ratios[3],
    ]
}
pub fn compute_csm_matrices(
    camera_view: Mat4,
    camera_fov_y_radians: f32,
    aspect: f32,
    camera_near: f32,
    camera_far: f32,
    sun_dir_surface_to_sun: Vec3,
    shadow_map_size: u32,
    stabilize: bool,
    reversed_z: bool, // <--- add
) -> ([Mat4; CSM_CASCADES], [f32; 4]) {
    let shadow_far = camera_far
        .min(DEFAULT_SHADOW_DISTANCE)
        .max(camera_near + 1.0);

    let splits = cascade_splits_from_ratios(camera_near, shadow_far, [0.05, 0.15, 0.55, 1.0]);

    let (right, up, forward) = camera_basis_from_view_rotation_only(camera_view);
    let eye = Vec3::ZERO; // IMPORTANT: camera-relative space

    let matrices: [Mat4; CSM_CASCADES] = std::array::from_fn(|i| {
        let slice_near = if i == 0 { camera_near } else { splits[i - 1] };
        let slice_far = splits[i];

        let corners = frustum_slice_corners_ws(
            eye,
            right,
            up,
            forward,
            camera_fov_y_radians,
            aspect,
            slice_near,
            slice_far,
        );

        compute_light_matrix_fit_frustum_slice_stable(
            &corners,
            sun_dir_surface_to_sun,
            shadow_map_size,
            stabilize,
            reversed_z, // <--- pass through
        )
    });

    (matrices, splits)
}
pub fn render_roads_shadows(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    road_renderer: &RoadRenderSubsystem,
    pipelines: &Pipelines,
    settings: &Settings,
    shadow_mat_buffer: &Buffer,
) {
    let bias = DepthBiasState {
        constant: 0, // for Depth32Float, constants often need to be “large”
        slope_scale: 0.5,
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
                depth_compare: if settings.reversed_depth_z {
                    CompareFunction::GreaterEqual
                } else {
                    CompareFunction::LessEqual
                },
                stencil: Default::default(),
                bias,
            }),
            msaa_samples: 1,
            vertex_layouts: Vec::from([RoadVertex::layout()]),
            cull_mode: Some(Face::Back),
            shadow_pass: true,
            targets: vec![],
            ..Default::default()
        },
        &[&pipelines.camera_uniforms.buffer, &shadow_mat_buffer],
        pass,
        pipelines,
        settings,
    );

    draw_visible_roads(pass, road_renderer);

    if road_renderer.preview_gpu.is_empty() {
        return;
    }
    let (Some(vb), Some(ib)) = (&road_renderer.preview_gpu.vb, &road_renderer.preview_gpu.ib)
    else {
        return;
    };

    let nudge_bias = DepthBiasState {
        constant: 1,
        slope_scale: 0.5,
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
                depth_compare: if settings.reversed_depth_z {
                    CompareFunction::GreaterEqual
                } else {
                    CompareFunction::LessEqual
                },
                stencil: Default::default(),
                bias: nudge_bias,
            }),
            msaa_samples: 1,
            vertex_layouts: Vec::from([RoadVertex::layout()]),
            cull_mode: Some(Face::Back),
            shadow_pass: true,
            targets: Vec::new(),
            ..Default::default()
        },
        &[&pipelines.camera_uniforms.buffer, &shadow_mat_buffer],
        pass,
        pipelines,
        settings,
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
    settings: &Settings,
    camera: &Camera,
    aspect: f32,
    shadow_mat_buffer: &Buffer,
) {
    let bias = DepthBiasState {
        constant: 0, // for Depth32Float, constants often need to be “large”
        slope_scale: 0.5,
        clamp: 0.0,
    };
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
                depth_compare: if settings.reversed_depth_z {
                    CompareFunction::GreaterEqual
                } else {
                    CompareFunction::LessEqual
                },
                stencil: StencilState::default(),
                bias,
            }),
            msaa_samples: 1,
            vertex_layouts: Vec::from([Vertex::desc()]),
            cull_mode: Some(Face::Back),
            shadow_pass: true,
            targets: Vec::new(),
            ..Default::default()
        },
        &[&pipelines.camera_uniforms.buffer, &shadow_mat_buffer],
        pass,
        pipelines,
        settings,
    );
    terrain_renderer.render(pass, camera, aspect, settings, false);
}
