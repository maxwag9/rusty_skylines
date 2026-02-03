#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GtaoParams {
    pub radius_world: f32,         // World-space AO radius (e.g., 0.5 to 2.0)
    pub intensity: f32,            // AO intensity multiplier (e.g., 1.0 to 2.0)
    pub bias: f32,                 // Depth bias to prevent self-occlusion (e.g., 0.01)
    pub frame_index: u32,          // For temporal jitter
    pub screen_size: [f32; 2],     // Half-res screen dimensions
    pub inv_screen_size: [f32; 2], // 1.0 / screen_size
}

impl Default for GtaoParams {
    fn default() -> Self {
        Self {
            radius_world: 1.0,
            intensity: 1.5,
            bias: 0.02,
            frame_index: 0,
            screen_size: [1920.0 / 2.0, 1080.0 / 2.0],
            inv_screen_size: [2.0 / 1920.0, 2.0 / 1080.0],
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GtaoBlurParams {
    pub direction: [i32; 2],  // (1,0) for horizontal, (0,1) for vertical
    pub texel_size: [f32; 2], // 1.0 / texture_dimensions
    pub depth_sigma: f32,     // Depth difference sensitivity (e.g., 0.5)
    pub normal_sigma: f32,    // Normal difference sensitivity (e.g., 0.1)
    pub kernel_radius: i32,   // Blur radius in pixels (3-5)
    pub _padding: i32,
}

impl GtaoBlurParams {
    pub fn horizontal(width: u32, height: u32) -> Self {
        Self {
            direction: [1, 0],
            texel_size: [1.0 / width as f32, 1.0 / height as f32],
            depth_sigma: 0.5,
            normal_sigma: 0.1,
            kernel_radius: 4,
            _padding: 0,
        }
    }

    pub fn vertical(width: u32, height: u32) -> Self {
        Self {
            direction: [0, 1],
            texel_size: [1.0 / width as f32, 1.0 / height as f32],
            depth_sigma: 0.5,
            normal_sigma: 0.1,
            kernel_radius: 4,
            _padding: 0,
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GtaoUpsampleParams {
    pub full_size: [f32; 2],
    pub half_size: [f32; 2],
    pub inv_full_size: [f32; 2],
    pub inv_half_size: [f32; 2],
    pub depth_threshold: f32,  // Relative depth difference threshold
    pub normal_threshold: f32, // Normal dot product threshold (0.9 = ~25 degrees)
    pub use_normal_check: u32, // 0 = off, 1 = on
    pub _padding: u32,
}

impl GtaoUpsampleParams {
    pub fn new(full_width: u32, full_height: u32) -> Self {
        let half_width = full_width / 2;
        let half_height = full_height / 2;

        Self {
            full_size: [full_width as f32, full_height as f32],
            half_size: [half_width as f32, half_height as f32],
            inv_full_size: [1.0 / full_width as f32, 1.0 / full_height as f32],
            inv_half_size: [1.0 / half_width as f32, 1.0 / half_height as f32],
            depth_threshold: 0.05, // 5% relative depth difference
            normal_threshold: 0.9, // ~25 degree angle tolerance
            use_normal_check: 1,   // Enable normal check
            _padding: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GtaoApplyParams {
    pub power: f32,      // Power curve (1.0-3.0, default 1.5)
    pub intensity: f32,  // AO strength (0.0-1.0, default 1.0)
    pub min_ao: f32,     // Minimum AO to prevent over-darkening (0.0-0.5, default 0.1)
    pub debug_mode: u32, // 0 = normal, 1 = AO only, 2 = color viz, 3 = raw AO
}

impl Default for GtaoApplyParams {
    fn default() -> Self {
        Self {
            power: 1.5,
            intensity: 1.0,
            min_ao: 0.1,
            debug_mode: 0,
        }
    }
}
