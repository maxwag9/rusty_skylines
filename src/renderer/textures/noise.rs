use rand::prelude::*;
use std::f32::consts::PI;
use wgpu::{
    Device, Extent3d, Origin3d, Queue, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor,
};
/// Generate a blue noise texture using the void-and-cluster algorithm
/// Returns RGBA8 data where R = noise value, G = cos(angle), B = sin(angle), A = 1
fn generate_blue_noise_texture(size: u32, seed: u64) -> Vec<u8> {
    let n = (size * size) as usize;
    let mut rng = StdRng::seed_from_u64(seed);

    // Phase 1: Generate blue noise scalar values using void-and-cluster
    let scalar_noise = generate_blue_noise_scalar(size, &mut rng);

    // Phase 2: Generate blue noise rotation angles (separate pattern)
    let angle_noise = generate_blue_noise_scalar(size, &mut rng);

    // Pack into RGBA8 texture
    let mut data = Vec::with_capacity(n * 4);
    for i in 0..n {
        let noise_val = scalar_noise[i];
        let angle = angle_noise[i] * 2.0 * PI; // Map [0,1] to [0, 2π]

        data.push((noise_val * 255.0) as u8); // R: noise value
        data.push(((angle.cos() * 0.5 + 0.5) * 255.0) as u8); // G: cos(angle) encoded
        data.push(((angle.sin() * 0.5 + 0.5) * 255.0) as u8); // B: sin(angle) encoded
        data.push(255); // A: unused
    }

    data
}

/// Void-and-cluster algorithm for blue noise generation
fn generate_blue_noise_scalar(size: u32, rng: &mut StdRng) -> Vec<f32> {
    let n = (size * size) as usize;
    let sigma: f32 = 1.5; // Controls the "blueness" of the noise

    // Initialize with random binary pattern (10% filled)
    let initial_density = 0.1;
    let mut binary: Vec<bool> = (0..n)
        .map(|_| rng.random::<f32>() < initial_density)
        .collect();

    // Precompute Gaussian kernel for energy calculation
    let kernel_size = (sigma * 4.0).ceil() as i32;
    let kernel = precompute_gaussian_kernel(kernel_size, sigma);

    // Phase 1: Remove tightest clusters until we have ~10% points
    let target_points = (n as f32 * initial_density) as usize;
    loop {
        let point_count: usize = binary.iter().filter(|&&b| b).count();
        if point_count <= target_points {
            break;
        }

        // Find the tightest cluster (highest energy point that is ON)
        let energies = compute_energy_field(&binary, size, &kernel, kernel_size);
        let mut max_energy = f32::MIN;
        let mut max_idx = 0;
        for i in 0..n {
            if binary[i] && energies[i] > max_energy {
                max_energy = energies[i];
                max_idx = i;
            }
        }
        binary[max_idx] = false;
    }

    // Phase 2: Progressive point insertion
    let mut ranks: Vec<f32> = vec![0.0; n];
    let mut rank = 0;

    // First, assign ranks to initial points by removing them in cluster order
    let mut temp_binary = binary.clone();
    let initial_points: Vec<usize> = temp_binary
        .iter()
        .enumerate()
        .filter(|&(_, &b)| b)
        .map(|(i, _)| i)
        .collect();

    for _ in 0..initial_points.len() {
        let energies = compute_energy_field(&temp_binary, size, &kernel, kernel_size);
        let mut max_energy = f32::MIN;
        let mut max_idx = 0;
        for i in 0..n {
            if temp_binary[i] && energies[i] > max_energy {
                max_energy = energies[i];
                max_idx = i;
            }
        }
        temp_binary[max_idx] = false;
        ranks[max_idx] = rank as f32;
        rank += 1;
    }

    // Phase 3: Fill remaining points by finding largest voids
    while rank < n {
        let energies = compute_energy_field(&binary, size, &kernel, kernel_size);
        let mut min_energy = f32::MAX;
        let mut min_idx = 0;
        for i in 0..n {
            if !binary[i] && energies[i] < min_energy {
                min_energy = energies[i];
                min_idx = i;
            }
        }
        binary[min_idx] = true;
        ranks[min_idx] = rank as f32;
        rank += 1;
    }

    // Normalize ranks to [0, 1]
    let max_rank = (n - 1) as f32;
    ranks.iter().map(|&r| r / max_rank).collect()
}

fn precompute_gaussian_kernel(size: i32, sigma: f32) -> Vec<f32> {
    let width = (size * 2 + 1) as usize;
    let mut kernel = vec![0.0; width * width];
    let sigma2 = sigma * sigma;

    for dy in -size..=size {
        for dx in -size..=size {
            let d2 = (dx * dx + dy * dy) as f32;
            let idx = ((dy + size) as usize) * width + (dx + size) as usize;
            kernel[idx] = (-d2 / (2.0 * sigma2)).exp();
        }
    }
    kernel
}

fn compute_energy_field(binary: &[bool], size: u32, kernel: &[f32], kernel_size: i32) -> Vec<f32> {
    let n = (size * size) as usize;
    let size_i = size as i32;
    let kernel_width = (kernel_size * 2 + 1) as usize;
    let mut energy = vec![0.0; n];

    for y in 0..size_i {
        for x in 0..size_i {
            let idx = (y * size_i + x) as usize;
            let mut e = 0.0;

            for dy in -kernel_size..=kernel_size {
                for dx in -kernel_size..=kernel_size {
                    // Toroidal wrapping
                    let nx = ((x + dx).rem_euclid(size_i)) as usize;
                    let ny = ((y + dy).rem_euclid(size_i)) as usize;
                    let neighbor_idx = ny * (size as usize) + nx;

                    if binary[neighbor_idx] {
                        let kidx = ((dy + kernel_size) as usize) * kernel_width
                            + (dx + kernel_size) as usize;
                        e += kernel[kidx];
                    }
                }
            }
            energy[idx] = e;
        }
    }
    energy
}

/// Create the wgpu texture from the generated blue noise
pub fn create_blue_noise_texture(device: &Device, queue: &Queue, size: u32) -> TextureView {
    let data = generate_blue_noise_texture(size, 69); // Fixed seed for reproducibility

    let texture = device.create_texture(&TextureDescriptor {
        label: Some("Blue Noise Texture"),
        size: Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        &data,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size * 4),
            rows_per_image: Some(size),
        },
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );

    texture.create_view(&TextureViewDescriptor::default())
}
