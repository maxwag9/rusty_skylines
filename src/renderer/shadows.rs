use crate::data::Settings;
use crate::helpers::paths::shader_dir;
use crate::renderer::pipelines::Pipelines;
use crate::renderer::ray_tracing::rt_subsystem::RTSubsystem;
use crate::renderer::render_passes::draw_visible_roads;
use crate::ui::vertex::Vertex;
use crate::world::camera::Camera;
use crate::world::cars::car_mesh::CarVertex;
use crate::world::cars::car_render::CarInstance;
use crate::world::cars::car_structs::CarStorage;
use crate::world::cars::car_subsystem::CarRenderSubsystem;
use crate::world::roads::road_mesh_manager::RoadVertex;
use crate::world::roads::road_subsystem::RoadRenderSubsystem;
use crate::world::terrain::terrain_subsystem::{TerrainRenderSubsystem, TerrainSubsystem};
use glam::{Mat4, Vec3, Vec4};
use wgpu::PrimitiveTopology::TriangleList;
use wgpu::TextureFormat::Depth32Float;
use wgpu::{
    Buffer, CompareFunction, DepthBiasState, DepthStencilState, Device, Face, IndexFormat,
    RenderPass, StencilState, TextureView,
};
use wgpu_render_manager::pipelines::PipelineOptions;
use wgpu_render_manager::renderer::RenderManager;

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

    pub light_mats: [Mat4; CSM_CASCADES],
    pub splits: [f32; CSM_CASCADES],
    pub texels: [f32; CSM_CASCADES],

    pub shadow_mat_buffers: [Buffer; CSM_CASCADES],
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
        texels: [1.0; 4],
        shadow_mat_buffers: std::array::from_fn(|_| create_shadow_mat_uniform_buffer(device)),
    }
}

pub const CSM_SPLIT_LAMBDA: f32 = 0.90; // 0=uniform, 1=log. 0.8-0.95 is typical.
pub const CSM_OVERLAP: f32 = 0.04; // 0.03-0.10 helps hide seams between cascades.

fn smoothstep01(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

pub fn cascade_splits_practical(near: f32, far: f32, lambda: f32) -> [f32; CSM_CASCADES] {
    let near = near.max(0.01);
    let far = far.max(near + 0.5);
    let n = CSM_CASCADES as f32;

    let mut out = [0.0; CSM_CASCADES];
    for i in 1..=CSM_CASCADES {
        let p = i as f32 / n;
        let log = near * (far / near).powf(p);
        let lin = near + (far - near) * p;
        out[i - 1] = lerp(lin, log, lambda.clamp(0.0, 1.0));
    }
    out
}

pub fn camera_basis_from_view_rotation_only(view: Mat4) -> (Vec3, Vec3, Vec3) {
    let inv = view.inverse();
    let right = inv.x_axis.truncate().normalize();
    let up = inv.y_axis.truncate().normalize();
    let forward = (-inv.z_axis.truncate()).normalize();
    (right, up, forward)
}

pub fn frustum_slice_corners_ws(
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

    let ntl = nc + up * nh - right * nw;
    let ntr = nc + up * nh + right * nw;
    let nbl = nc - up * nh - right * nw;
    let nbr = nc - up * nh + right * nw;

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
) -> (Mat4, f32) {
    const MAX_CASTER_HEIGHT: f32 = 520.0;
    const GUARD_BAND: f32 = 1.02;

    let sun_dir = if sun_dir_surface_to_sun.length_squared() < 1e-8 {
        Vec3::new(0.3, 1.0, 0.2).normalize()
    } else {
        sun_dir_surface_to_sun.normalize()
    };

    let up = if sun_dir.dot(Vec3::Y).abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };

    let mut center = Vec3::ZERO;
    for &c in frustum_corners_ws.iter() {
        center += c;
    }
    center *= 1.0 / 8.0;

    let light_eye = center + sun_dir * 2000.0;
    let light_view = Mat4::look_at_rh(light_eye, center, up);

    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;

    for &c in frustum_corners_ws.iter() {
        let p = (light_view * Vec4::new(c.x, c.y, c.z, 1.0)).truncate();
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
        min_z = min_z.min(p.z);
        max_z = max_z.max(p.z);
    }

    let sun_y = sun_dir.y.abs().max(0.08);
    let max_extent = (max_x - min_x).max(max_y - min_y).max(1.0);
    let xy_pad = ((MAX_CASTER_HEIGHT / sun_y) * 0.15)
        .min(0.20 * max_extent)
        .max(2.0);

    min_x -= xy_pad;
    max_x += xy_pad;
    min_y -= xy_pad;
    max_y += xy_pad;

    let cx = 0.5 * (min_x + max_x);
    let cy = 0.5 * (min_y + max_y);

    let ex = (max_x - min_x) * GUARD_BAND;
    let ey = (max_y - min_y) * GUARD_BAND;

    let mut min_x = cx - 0.5 * ex;
    let mut max_x = cx + 0.5 * ex;
    let mut min_y = cy - 0.5 * ey;
    let mut max_y = cy + 0.5 * ey;

    let size = shadow_map_size.max(1) as f32;
    let mut texel_x = ((max_x - min_x) / size).max(1e-6);
    let mut texel_y = ((max_y - min_y) / size).max(1e-6);

    if stabilize {
        min_x = (min_x / texel_x).floor() * texel_x;
        max_x = (max_x / texel_x).ceil() * texel_x;
        min_y = (min_y / texel_y).floor() * texel_y;
        max_y = (max_y / texel_y).ceil() * texel_y;

        texel_x = ((max_x - min_x) / size).max(1e-6);
        texel_y = ((max_y - min_y) / size).max(1e-6);
    }

    let texel_world = texel_x.max(texel_y);

    let mut near_d = (-max_z).max(0.05);
    let mut far_d = (-min_z).max(near_d + 0.25);

    let z_range = (far_d - near_d).max(1.0);
    let caster_pad = (MAX_CASTER_HEIGHT / sun_y).min(2500.0);
    let z_pad = (0.06 * z_range + 20.0 + 0.35 * caster_pad).min(1200.0);

    near_d = (near_d - z_pad).max(0.01);
    far_d = far_d + z_pad;

    let light_proj = if reversed_z {
        Mat4::orthographic_rh(min_x, max_x, min_y, max_y, far_d, near_d)
    } else {
        Mat4::orthographic_rh(min_x, max_x, min_y, max_y, near_d, far_d)
    };

    (light_proj * light_view, texel_world)
}

fn compute_shadow_distance(eye_height_agl: f32, orbit_radius: f32) -> f32 {
    let height_t = smoothstep01((eye_height_agl - 8.0) / 260.0);
    let orbit_t = smoothstep01((orbit_radius - 25.0) / 850.0);

    let by_orbit = lerp(160.0, 1350.0, orbit_t);
    let by_height = lerp(140.0, 900.0, height_t);

    by_orbit.max(by_height).clamp(120.0, 1500.0)
}

pub fn compute_csm_matrices(
    terrain_renderer: &TerrainSubsystem,
    camera: &Camera,
    aspect: f32,
    sun_dir_surface_to_sun: Vec3,
    shadow_map_size: u32,
    stabilize: bool,
    reversed_z: bool,
) -> (
    [Mat4; CSM_CASCADES],
    [f32; CSM_CASCADES],
    [f32; CSM_CASCADES],
) {
    let eye_ws = camera.eye_world();
    let terrain_y = terrain_renderer.get_height_at(eye_ws, false);
    let eye_height_agl = (eye_ws.local.y - terrain_y).max(0.0);

    let camera_fov_y_radians = camera.fov.to_radians();
    let camera_near = camera.near.max(0.01);
    let camera_far = camera.far.max(camera_near + 1.0);

    let (right, up, forward) = camera_basis_from_view_rotation_only(camera.view());

    let looking_down_t = smoothstep01(((-forward.y) - 0.25) / 0.65);
    let orbit_t = smoothstep01((camera.orbit_radius - 30.0) / 850.0);

    let lambda = (lerp(0.62, 0.86, looking_down_t) * lerp(1.0, 0.85, orbit_t)).clamp(0.55, 0.92);

    let mut shadow_distance = compute_shadow_distance(eye_height_agl, camera.orbit_radius);
    shadow_distance *= lerp(1.25, 0.85, looking_down_t);

    let shadow_far = camera_far.min(shadow_distance).max(camera_near + 10.0);

    let splits = cascade_splits_practical(camera_near, shadow_far, lambda);

    let eye = Vec3::ZERO;

    let mut mats = [Mat4::IDENTITY; CSM_CASCADES];
    let mut texels = [0.0f32; CSM_CASCADES];

    for i in 0..CSM_CASCADES {
        let slice_near = if i == 0 { camera_near } else { splits[i - 1] };

        let mut slice_far = splits[i];
        if i + 1 < CSM_CASCADES {
            slice_far = (slice_far * (1.0 + CSM_OVERLAP)).min(shadow_far);
        }

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

        let (m, texel_world) = compute_light_matrix_fit_frustum_slice_stable(
            &corners,
            sun_dir_surface_to_sun,
            shadow_map_size,
            stabilize,
            reversed_z,
        );

        mats[i] = m;
        texels[i] = texel_world;
    }

    (mats, splits, texels)
}

pub fn shadow_bias_for_cascade(
    cascade_idx: usize,
    texel_world: f32,
    reversed_z: bool,
) -> DepthBiasState {
    let c = cascade_idx as f32;

    let slope = (1.2 + 0.9 * c).min(6.0);
    let constant_f = (texel_world * (650.0 + 350.0 * c)) + (70.0 * c);
    let constant_abs = constant_f.clamp(10.0, 14000.0) as i32;

    let sign_i: i32 = if reversed_z { -1 } else { 1 };
    let sign_f: f32 = if reversed_z { -1.0 } else { 1.0 };

    DepthBiasState {
        constant: sign_i * constant_abs,
        slope_scale: sign_f * slope,
        clamp: 0.02,
    }
}
fn shadow_pipeline_options(
    settings: &Settings,
    bias: DepthBiasState,
    vertex_layouts: Vec<wgpu::VertexBufferLayout<'static>>,
    cull_mode: Face,
) -> PipelineOptions {
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
        vertex_layouts,
        cull_mode: Some(cull_mode),
        vertex_only: true,
        targets: vec![],
        ..Default::default()
    }
}
pub fn render_roads_shadows(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    road_renderer: &RoadRenderSubsystem,
    pipelines: &Pipelines,
    settings: &Settings,
    shadow_mat_buffer: &Buffer,
    cascade_idx: usize,
) {
    let bias = shadow_bias_for_cascade(
        cascade_idx,
        pipelines.resources.csm_shadows.texels[cascade_idx],
        settings.reversed_depth_z,
    );

    let shader = shader_dir().join("shadows.wgsl");
    let opts = shadow_pipeline_options(settings, bias, vec![RoadVertex::layout()], Face::Back);

    render_manager.render(
        &[],
        shader.as_path(),
        &opts,
        &[&pipelines.buffers.camera, shadow_mat_buffer],
        pass,
    );

    draw_visible_roads(pass, road_renderer);

    // preview (optional extra bias)
    if let (Some(vb), Some(ib)) = (&road_renderer.preview_gpu.vb, &road_renderer.preview_gpu.ib) {
        let mut preview_bias = bias;
        preview_bias.constant = (preview_bias.constant
            + if settings.reversed_depth_z { -200 } else { 200 })
        .clamp(-15000, 15000);

        let opts2 = shadow_pipeline_options(
            settings,
            preview_bias,
            vec![RoadVertex::layout()],
            Face::Back,
        );

        render_manager.render(
            &[],
            shader.as_path(),
            &opts2,
            &[&pipelines.buffers.camera, shadow_mat_buffer],
            pass,
        );

        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
        pass.draw_indexed(0..road_renderer.preview_gpu.index_count, 0, 0..1);
    }
}
pub fn render_terrain_shadows(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    terrain_renderer: &TerrainRenderSubsystem,
    terrain_subsystem: &TerrainSubsystem,
    pipelines: &Pipelines,
    settings: &Settings,
    camera: &Camera,
    aspect: f32,
    shadow_mat_buffer: &Buffer,
    cascade_idx: usize,
) {
    let bias = shadow_bias_for_cascade(
        cascade_idx,
        pipelines.resources.csm_shadows.texels[cascade_idx],
        settings.reversed_depth_z,
    );

    let shader = shader_dir().join("shadows.wgsl");
    let opts = shadow_pipeline_options(settings, bias, vec![Vertex::desc()], Face::Back);

    render_manager.render(
        &[],
        shader.as_path(),
        &opts,
        &[&pipelines.buffers.camera, shadow_mat_buffer],
        pass,
    );

    terrain_renderer.render(pass, terrain_subsystem, camera, aspect, settings, false);
}
pub fn render_cars_shadows(
    pass: &mut RenderPass,
    render_manager: &mut RenderManager,
    rt_subsystem: &mut RTSubsystem,
    car_renderer: &mut CarRenderSubsystem,
    car_storage: &CarStorage,
    pipelines: &Pipelines,
    settings: &Settings,
    camera: &Camera,
    shadow_mat_buffer: &Buffer,
    cascade_idx: usize,
) {
    let bias = shadow_bias_for_cascade(
        cascade_idx,
        pipelines.resources.csm_shadows.texels[cascade_idx],
        settings.reversed_depth_z,
    );

    let shader = shader_dir().join("car_shadows.wgsl");
    let opts = shadow_pipeline_options(
        settings,
        bias,
        vec![CarVertex::layout(), CarInstance::layout()],
        Face::Front,
    );

    render_manager.render(&[], shader.as_path(), &opts, &[shadow_mat_buffer], pass);

    car_renderer.render(pipelines, rt_subsystem, car_storage, camera, pass)
}
