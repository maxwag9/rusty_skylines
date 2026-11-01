struct CelestialBodyUniform {
    direction_size: vec4<f32>,
    properties: vec4<f32>,
    parent_relative_position: vec4<f32>,
}

struct SkyUniform {
    star_rotation: mat4x4<f32>,
    exposure: f32,
    moon_phase: f32,
    sun_size: f32,
    sun_intensity: f32,
    moon_size: f32,
    moon_intensity: f32,
    _pad1: f32,
    _pad2: f32,
    celestial_bodies: array<CelestialBodyUniform, 20>,
}