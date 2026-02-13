use crate::helpers::paths::compute_shader_dir;
use wgpu::{
    BufferDescriptor, BufferUsages, Device, Extent3d, Queue, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};
use wgpu_render_manager::compute_system::ComputePipelineOptions;
use wgpu_render_manager::renderer::RenderManager;

pub fn create_blue_noise_texture_gpu(
    render_manager: &mut RenderManager,
    device: &Device,
    queue: &Queue,
    size: u32,
    seed: u32,
) -> TextureView {
    let n = size * size;
    let target = (n as f32 * 0.1) as u32;

    let texture = device.create_texture(&TextureDescriptor {
        label: Some("blue_noise"),
        size: Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());

    let storage = |label, bytes| {
        device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };

    let binary = storage("binary", (n * 4) as u64);
    let ranks = storage("ranks", (n * 4) as u64);
    let energy = storage("energy", (n * 4) as u64);
    let extremum = storage("extremum", 8);

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        size: u32,
        seed: u32,
        sigma: f32,
        kernel_size: i32,
    }

    let params = device.create_buffer(&BufferDescriptor {
        label: Some("params"),
        size: size_of::<Params>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(
        &params,
        0,
        bytemuck::bytes_of(&Params {
            size,
            seed,
            sigma: 1.5,
            kernel_size: 6,
        }),
    );

    // Add a state buffer for remaining/rank
    let state = device.create_buffer(&BufferDescriptor {
        label: Some("state"),
        size: 8, // [remaining, rank]
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Initialize
    queue.write_buffer(&state, 0, bytemuck::bytes_of(&[target + 1, 0u32]));

    // Dispatch init pass once
    render_manager.compute(
        None,
        "blue_noise_init",
        vec![],
        vec![&view],
        &compute_shader_dir().join("blue_noise_init.wgsl"),
        ComputePipelineOptions {
            dispatch_size: [(n + 255) / 256, 1, 1],
        },
        &[&binary, &ranks, &energy, &params],
    );

    // One dispatch per pruning iteration (CPU loop)
    for iteration in 0..target {
        render_manager.compute(
            None,
            "blue_noise_prune_step",
            vec![],
            vec![],
            &compute_shader_dir().join("blue_noise_step.wgsl"),
            ComputePipelineOptions {
                dispatch_size: [(n + 255) / 256, 1, 1],
            },
            &[&binary, &ranks, &energy, &extremum, &params, &state],
        );
    }

    // Final output pass
    render_manager.compute(
        None,
        "blue_noise_output",
        vec![],
        vec![&view],
        &compute_shader_dir().join("blue_noise_output.wgsl"),
        ComputePipelineOptions {
            dispatch_size: [(n + 255) / 256, 1, 1],
        },
        &[&ranks, &params],
    );

    view
}
