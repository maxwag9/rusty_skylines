/// Return Halton sequence value in [0,1) for given index (1-based) and base.
/// If you pass index = 0 we return 0.0.
fn halton(index: u64, base: u64) -> f32 {
    if index == 0 || base < 2 {
        return 0.0;
    }
    let mut i = index;
    let b = base as f32;
    let mut f = 1.0f32;
    let mut r = 0.0f32;
    while i > 0 {
        f /= b;
        r += f * ((i % base) as f32);
        i /= base;
    }
    r
}

/// Compute (prev_jitter, curr_jitter) in NDC units for TAA.
/// - `frame_index` is the 0-based frame counter you increment every frame.
/// - `width`/`height` are render target dimensions in pixels.
/// Returned tuples are [x, y].
pub fn taa_jitter_pair(frame_index: u64, width: u32, height: u32) -> ([f32; 2], [f32; 2]) {
    // pixel -> NDC scale: 1 pixel = 2.0 / resolution
    let px_to_ndc_x = 2.0f32 / (width as f32);
    let px_to_ndc_y = 2.0f32 / (height as f32);

    // helper: convert Halton sample [0,1) -> centered pixel offset [-0.5, 0.5)
    let halton_to_pixel_offset = |idx: u64| -> (f32, f32) {
        // use index+1 so frame_index==0 still produces a non-zero Halton point on first sample if desired
        let sample_x = halton(idx + 1, 2);
        let sample_y = halton(idx + 1, 3);
        // center to [-0.5, 0.5)
        (sample_x - 0.5, sample_y - 0.5)
    };

    // current frame jitter (in pixel units centered)
    let (curr_px_x, curr_px_y) = halton_to_pixel_offset(frame_index);

    // previous frame jitter: if frame_index == 0 use zero (no previous)
    let (prev_px_x, prev_px_y) = if frame_index == 0 {
        (0.0f32, 0.0f32)
    } else {
        halton_to_pixel_offset(frame_index - 1)
    };

    // convert to NDC
    let curr_jitter = [curr_px_x * px_to_ndc_x, curr_px_y * px_to_ndc_y];
    let prev_jitter = [prev_px_x * px_to_ndc_x, prev_px_y * px_to_ndc_y];

    (prev_jitter, curr_jitter)
}
