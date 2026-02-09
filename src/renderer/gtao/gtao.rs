#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GtaoParams {
    pub radius_world: f32,         // World-space AO radius (e.g., 0.5 to 2.0)
    pub intensity: f32,            // AO intensity multiplier (e.g., 1.0 to 2.0)
    pub bias: f32,                 // Depth bias to prevent self-occlusion (e.g., 0.01)
    pub frame_index: u32,          // For temporal jitter
    pub screen_size: [f32; 2],     // Half-res screen dimensions
    pub inv_screen_size: [f32; 2], // 1.0 / screen_size
    pub temporal_blend: f32,       // 0.05 typical, 1.0 = reset
    pub _pad: [u32; 3],
    pub prev_view_proj: [[f32; 4]; 4],
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
            temporal_blend: 1.0,
            _pad: [0; 3],
            prev_view_proj: Default::default(),
        }
    }
}
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GtaoBlurParams {
    pub(crate) depth_sigma: f32,
    pub(crate) normal_sigma: f32,
    pub(crate) kernel_radius: i32,
    pub(crate) _padding: i32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GtaoUpsampleApplyParams {
    pub(crate) full_size: [f32; 2],
    pub(crate) half_size: [f32; 2],
    pub(crate) inv_full_size: [f32; 2],
    pub(crate) inv_half_size: [f32; 2],
    pub(crate) depth_threshold: f32,
    pub(crate) normal_threshold: f32,
    pub(crate) use_normal_check: u32,
    pub(crate) power: f32,
    pub(crate) apply_intensity: f32,
    pub(crate) min_ao: f32,
    pub(crate) debug_mode: u32,
    pub(crate) _padding: u32,
}
