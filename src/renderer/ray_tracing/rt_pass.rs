use crate::gpu_timestamp;
use crate::helpers::paths::{compute_shader_dir, shader_dir};
use crate::renderer::gpu_profiler::GpuProfiler;
use crate::renderer::pipelines::Pipelines;
use crate::renderer::ray_tracing::rt_subsystem::{RTSubsystem, build_render_space_instances};
use crate::renderer::render_core::create_color_attachment_load;
use crate::world::camera::Camera;
use crate::world::cars::car_structs::CarStorage;
use crate::world::cars::car_subsystem::{CAR_BASE_LENGTH, CAR_BASE_WIDTH};
use glam::Vec3;
use wgpu::PrimitiveTopology::TriangleList;
use wgpu::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, ColorTargetState, ColorWrites,
    CommandEncoder, Device, Queue, RenderPassDescriptor, SurfaceConfiguration,
};
use wgpu_render_manager::compute_system::ComputePipelineOptions;
use wgpu_render_manager::pipelines::PipelineOptions;
use wgpu_render_manager::renderer::RenderManager;

pub fn update_rt_instances(
    rt: &mut RTSubsystem,
    device: &Device,
    queue: &Queue,
    pipelines: &Pipelines,
    car_storage: &CarStorage,
    camera: &Camera,
) {
    let Some(blas_aabb) = rt.car_blas_aabb() else {
        return;
    };

    let close_cars = car_storage.car_chunks().close_cars();

    let instances = build_render_space_instances(
        close_cars.iter().filter_map(|&car_id| {
            let car = car_storage.get(car_id)?;

            let rot = car.quat.normalize();
            let scale = Vec3::new(
                car.width / CAR_BASE_WIDTH,
                1.0,
                car.length / CAR_BASE_LENGTH,
            );

            let blas_root = 0u32; // car BLAS root node index
            Some((car_id as u32, car.pos, rot, scale, blas_root))
        }),
        camera,
        blas_aabb,
    );

    rt.update_tlas(device, queue, pipelines, instances, false);
}

pub fn render_ray_tracing(
    encoder: &mut CommandEncoder,
    config: &SurfaceConfiguration,
    rt: &mut RTSubsystem,
    render_manager: &mut RenderManager,
    pipelines: &Pipelines,
    profiler: &mut GpuProfiler,
    msaa_samples: u32,
) {
    let half_width = (config.width / 2).max(1);
    let half_height = (config.height / 2).max(1);

    const WG: u32 = 8;

    // === Pass 1: Ray trace at half-res ===
    let rt_dispatch = [(half_width + WG - 1) / WG, (half_height + WG - 1) / WG, 1];

    let Some(buffer_sets) = rt.get_buffer_sets(pipelines) else {
        return;
    };

    gpu_timestamp!(encoder, profiler, "RTX_Tracing", {
        render_manager.compute(
            Some(encoder),
            "Ray Tracing Pass",
            vec![
                &pipelines.post_fx.linear_depth_half,
                &pipelines.post_fx.normal_half,
                &pipelines.post_fx.rt_instance,
            ],
            vec![&pipelines.post_fx.rt_raw_half],
            &compute_shader_dir().join("ray_tracing.wgsl"),
            make_ray_tracing_options(rt_dispatch),
            buffer_sets.as_slice(),
        );
    });

    // === Pass 2: Upsample half→full (compute, no MSAA overhead) ===
    let upsample_dispatch = [
        (config.width + WG - 1) / WG,
        (config.height + WG - 1) / WG,
        1,
    ];

    gpu_timestamp!(encoder, profiler, "RTX_Upsample", {
        render_manager.compute(
            Some(encoder),
            "RTX Upsample",
            vec![
                &pipelines.post_fx.rt_full_history,
                &pipelines.post_fx.rt_raw_half,
                &pipelines.post_fx.linear_depth_full,
            ],
            vec![&pipelines.post_fx.rt_full],
            &compute_shader_dir().join("ray_tracing_upsample.wgsl"),
            make_ray_tracing_options(upsample_dispatch),
            &[],
        );
    });

    // === Pass 3: Apply shadow to HDR (trivial per-sample cost) ===
    gpu_timestamp!(encoder, profiler, "RTX_Apply", {
        let color_attachment = create_color_attachment_load(
            &pipelines.msaa.hdr,
            &pipelines.resolved.hdr,
            msaa_samples,
        );

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("RTX Shadow Apply"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        let options = PipelineOptions::default()
            .with_topology(TriangleList)
            .with_msaa(msaa_samples)
            .with_target(ColorTargetState {
                format: pipelines.msaa.hdr.texture().format(),
                blend: Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::Zero,
                        dst_factor: BlendFactor::Src, // result = hdr * shadow
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent::OVER,
                }),
                write_mask: ColorWrites::ALL,
            });

        render_manager.render_with_textures(
            &[&pipelines.post_fx.rt_full],
            shader_dir().join("ray_tracing_apply.wgsl").as_path(),
            &options,
            &[&pipelines.buffers.camera],
            &mut pass,
        );

        pass.draw(0..3, 0..1);
    });
}

fn make_ray_tracing_options(dispatch_size: [u32; 3]) -> ComputePipelineOptions {
    ComputePipelineOptions { dispatch_size }
}
