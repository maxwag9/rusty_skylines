
struct Uniforms {
    // ── Current frame matrices ──────────────────────────────────
    view:           mat4x4<f32>,
    inv_view:       mat4x4<f32>,
    proj:           mat4x4<f32>,
    inv_proj:       mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,

    // ── Previous frame reprojection ─────────────────────────────
    prev_view_proj: mat4x4<f32>,

    // ── Shadow cascades ─────────────────────────────────────────
    lighting_view_proj: array<mat4x4<f32>, 4>,  // CSM_CASCADES
    cascade_splits:     vec4<f32>,

    // ── Lighting ────────────────────────────────────────────────
    sun_direction:  vec3<f32>,
    time:           f32,
    moon_direction: vec3<f32>,
    orbit_radius:   f32,

    // ── Current camera (chunk-relative) ─────────────────────────
    camera_local:   vec3<f32>,
    chunk_size:     f32,
    camera_chunk:   vec2<i32>,
    _pad_cam:       vec2<u32>,

    // ── Previous camera (chunk-relative) ────────────────────────
    prev_camera_local: vec3<f32>,
    _pad_prev0:        f32,
    prev_camera_chunk: vec2<i32>,
    _pad_prev1:        vec2<i32>,

    // ── TAA jitter ──────────────────────────────────────────────
    curr_jitter:    vec2<f32>,
    prev_jitter:    vec2<f32>,

    // ── Misc settings ───────────────────────────────────────────
    reversed_depth_z:  u32,
    shadows_enabled:   u32,
    near_far_depth:    vec2<f32>,
};
