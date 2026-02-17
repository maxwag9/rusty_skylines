use wgpu_render_manager::generator::{TextureKey, TextureParams};

pub fn road_material_keys() -> Vec<TextureKey> {
    vec![
        TextureKey {
            shader_id: "concrete".to_string(),
            params: TextureParams {
                seed: 1,
                scale: 2.0,
                roughness: 1.0,
                color_primary: [0.32, 0.30, 0.28, 1.0],
                color_secondary: [0.15, 0.13, 0.10, 1.0],
                ..Default::default()
            },
            resolution: 512,
        },
        TextureKey {
            shader_id: "goo".to_string(),
            params: TextureParams {
                seed: 0,
                scale: 3.0,
                roughness: 0.3,
                color_primary: [0.02, 0.02, 0.03, 1.0],
                color_secondary: [0.10, 0.10, 0.12, 1.0],
                ..Default::default()
            },
            resolution: 512,
        },
        TextureKey {
            shader_id: "asphalt".to_string(),
            params: TextureParams {
                seed: 0,
                scale: 20.0,
                roughness: 0.7,
                color_primary: [0.05, 0.05, 0.05, 1.0],
                color_secondary: [0.06, 0.06, 0.08, 1.0],
                ..Default::default()
            },
            resolution: 512,
        },
        TextureKey {
            shader_id: "asphalt".to_string(),
            params: TextureParams {
                seed: 1,
                scale: 16.0,
                roughness: 0.3,
                color_primary: [0.006, 0.006, 0.006, 1.0],
                color_secondary: [0.020, 0.020, 0.020, 1.0],
                ..Default::default()
            },
            resolution: 512,
        },
        TextureKey {
            shader_id: "asphalt".to_string(),
            params: TextureParams {
                seed: 2,
                scale: 16.0,
                roughness: 0.7,
                color_primary: [0.04, 0.04, 0.006, 1.0],
                color_secondary: [0.070, 0.070, 0.070, 1.0],
                ..Default::default()
            },
            resolution: 512,
        },
        TextureKey {
            shader_id: "asphalt".to_string(),
            params: TextureParams {
                seed: 3,
                scale: 16.0,
                roughness: 0.8,
                color_primary: [0.02, 0.02, 0.02, 1.0],
                color_secondary: [0.080, 0.080, 0.080, 1.0],
                ..Default::default()
            },
            resolution: 512,
        },
    ]
}

pub fn cars_material_keys() -> Vec<TextureKey> {
    vec![TextureKey {
        shader_id: "shiny_metal".to_string(),
        params: TextureParams {
            seed: 3,
            scale: 16.0,
            roughness: 0.4,
            color_primary: [1.00, 1.00, 1.00, 1.0],
            color_secondary: [0.80, 0.80, 0.80, 1.0],
            ..Default::default()
        },
        resolution: 512,
    }]
}

pub fn terrain_material_keys() -> Vec<TextureKey> {
    vec![
        TextureKey {
            shader_id: "grass".to_string(),
            params: TextureParams {
                seed: 1337,
                scale: 40.0,
                roughness: 0.78, // Heavily reduced dry influence
                color_primary: [0.02, 0.34, 0.01, 1.0], // Deep muted olive green—no neon
                color_secondary: [0.10, 0.60, 0.10, 1.0], // Dark neutral brown, zero yellow pop
                ..Default::default()
            },
            resolution: 1024,
        },
        TextureKey {
            shader_id: "grass".to_string(),
            params: TextureParams {
                seed: 42,
                scale: 80.0,
                roughness: 0.65,                       // Way down—minimal dry yellow
                color_primary: [0.0, 0.33, 0.00, 1.0], // Ultra-dark muted base
                color_secondary: [0.02, 0.50, 0.00, 1.0], // Super dark brown shadow tone
                ..Default::default()
            },
            resolution: 1024,
        },
    ]
}
