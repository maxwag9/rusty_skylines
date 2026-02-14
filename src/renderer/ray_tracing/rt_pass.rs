use crate::cars::car_structs::CarStorage;
use crate::cars::car_subsystem::{CAR_BASE_LENGTH, CAR_BASE_WIDTH};
use crate::helpers::paths::compute_shader_dir;
use crate::renderer::pipelines::Pipelines;
use crate::renderer::ray_tracing::rt_subsystem::{RTSubsystem, build_render_space_instances};
use crate::world::camera::Camera;
use glam::Vec3;
use wgpu::{CommandEncoder, Device, Queue, SurfaceConfiguration};
use wgpu_render_manager::compute_system::ComputePipelineOptions;
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
) {
    let width = (config.width / 2).max(1);
    let height = (config.height / 2).max(1);

    const WG_X: u32 = 8;
    const WG_Y: u32 = 8;

    let dispatch_size = [(width + WG_X - 1) / WG_X, (height + WG_Y - 1) / WG_Y, 1];

    let options = make_ray_tracing_options(dispatch_size);

    let Some(buffer_sets) = rt.get_buffer_sets(pipelines) else {
        return;
    };
    let buffer_sets = buffer_sets.as_slice();
    let shader_path = &compute_shader_dir().join("ray_tracing.wgsl");

    render_manager.compute(
        Some(encoder),
        "Ray Tracing Pass",
        vec![
            &pipelines.post_fx.linear_depth_half,
            &pipelines.post_fx.normal_half,
            &pipelines.post_fx.rt_instance,
        ],
        vec![&pipelines.post_fx.rt_raw_half],
        shader_path,
        options,
        buffer_sets,
    );
}

fn make_ray_tracing_options(dispatch_size: [u32; 3]) -> ComputePipelineOptions {
    ComputePipelineOptions { dispatch_size }
}
