use crate::renderer::textures::procedural_texture_manager::{
    MaterialKind, Params, TextureCacheKey,
};

pub fn road_material_keys() -> Vec<TextureCacheKey> {
    vec![
        TextureCacheKey {
            kind: MaterialKind::Concrete,
            params: Params {
                seed: 1,
                scale: 2.0,
                roughness: 1.0,
                color_primary: [0.32, 0.30, 0.28, 1.0],
                color_secondary: [0.15, 0.13, 0.10, 1.0],
                moisture: 0.0,
                shadow_strength: 0.0,
                sheen_strength: 0.0,
                ..Default::default()
            },
            resolution: 512,
        },
        TextureCacheKey {
            kind: MaterialKind::Goo,
            params: Params {
                seed: 0,
                scale: 3.0,
                roughness: 0.3,
                color_primary: [0.02, 0.02, 0.03, 1.0],
                color_secondary: [0.10, 0.10, 0.12, 1.0],
                moisture: 0.0,
                shadow_strength: 0.0,
                sheen_strength: 0.0,
                ..Default::default()
            },
            resolution: 512,
        },
        TextureCacheKey {
            kind: MaterialKind::Asphalt,
            params: Params {
                seed: 0,
                scale: 16.0,
                roughness: 0.5,
                color_primary: [0.004, 0.004, 0.004, 1.0],
                color_secondary: [0.015, 0.015, 0.015, 1.0],
                moisture: 0.0,
                shadow_strength: 0.0,
                sheen_strength: 0.0,
                ..Default::default()
            },
            resolution: 512,
        },
        TextureCacheKey {
            kind: MaterialKind::Asphalt,
            params: Params {
                seed: 1,
                scale: 16.0,
                roughness: 0.3,
                color_primary: [0.006, 0.006, 0.006, 1.0],
                color_secondary: [0.020, 0.020, 0.020, 1.0],
                moisture: 0.0,
                shadow_strength: 0.0,
                sheen_strength: 0.0,
                ..Default::default()
            },
            resolution: 512,
        },
        TextureCacheKey {
            kind: MaterialKind::Asphalt,
            params: Params {
                seed: 2,
                scale: 16.0,
                roughness: 0.5,
                color_primary: [0.04, 0.04, 0.006, 1.0],
                color_secondary: [0.120, 0.120, 0.120, 1.0],
                moisture: 0.0,
                shadow_strength: 0.0,
                sheen_strength: 0.0,
                ..Default::default()
            },
            resolution: 512,
        },
        TextureCacheKey {
            kind: MaterialKind::Asphalt,
            params: Params {
                seed: 3,
                scale: 16.0,
                roughness: 0.8,
                color_primary: [0.02, 0.02, 0.02, 1.0],
                color_secondary: [0.080, 0.080, 0.080, 1.0],
                moisture: 0.0,
                shadow_strength: 0.0,
                sheen_strength: 0.0,
                ..Default::default()
            },
            resolution: 512,
        },
    ]
}

pub fn terrain_material_keys() -> Vec<TextureCacheKey> {
    vec![
        TextureCacheKey {
            kind: MaterialKind::Grass,
            params: Params {
                seed: 1337,
                scale: 40.0,
                roughness: 0.78, // Heavily reduced dry influence
                moisture: 0.65,  // Max lush bias
                color_primary: [0.02, 0.34, 0.01, 1.0], // Deep muted olive green—no neon
                color_secondary: [0.10, 0.60, 0.10, 1.0], // Dark neutral brown, zero yellow pop
                shadow_strength: 1.80, // Hard punchy shadows
                sheen_strength: 0.05, // Barely any highlight
                ..Default::default()
            },
            resolution: 1024,
        },
        TextureCacheKey {
            kind: MaterialKind::Grass,
            params: Params {
                seed: 42,
                scale: 80.0,
                roughness: 0.65,                       // Way down—minimal dry yellow
                moisture: 0.70,                        // Balanced but not dead
                color_primary: [0.0, 0.33, 0.00, 1.0], // Ultra-dark muted base
                color_secondary: [0.02, 0.50, 0.00, 1.0], // Super dark brown shadow tone
                shadow_strength: 1.75,                 // Even deeper volume
                sheen_strength: 0.02,                  // None basically
                ..Default::default()
            },
            resolution: 1024,
        },
    ]
}
