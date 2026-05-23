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
                roughness: 0.35,
                color_primary: [0.07, 0.30, 0.04, 1.0],
                color_secondary: [0.38, 0.30, 0.06, 1.0],
                octaves: 7.0,
                persistence: 0.50,
                lacunarity: 2.0,
                ..Default::default()
            },
            resolution: 1024,
        },
        TextureKey {
            shader_id: "grass".to_string(),
            params: TextureParams {
                seed: 42,
                scale: 64.0,
                roughness: 0.42,
                color_primary: [0.04, 0.24, 0.02, 1.0],
                color_secondary: [0.34, 0.26, 0.05, 1.0],
                octaves: 6.0,
                persistence: 0.48,
                lacunarity: 2.0,
                ..Default::default()
            },
            resolution: 1024,
        },
        TextureKey {
            shader_id: "rock".to_string(),
            params: TextureParams {
                seed: 9001,
                scale: 52.0,
                roughness: 0.62,
                color_primary: [0.18, 0.18, 0.19, 1.0],
                color_secondary: [0.52, 0.50, 0.48, 1.0],
                octaves: 6.0,
                persistence: 0.52,
                lacunarity: 2.0,
                ..Default::default()
            },
            resolution: 1024,
        },
        TextureKey {
            shader_id: "rock".to_string(),
            params: TextureParams {
                seed: 9002,
                scale: 76.0,
                roughness: 0.58,
                color_primary: [0.12, 0.13, 0.14, 1.0],
                color_secondary: [0.44, 0.43, 0.41, 1.0],
                octaves: 7.0,
                persistence: 0.50,
                lacunarity: 2.0,
                ..Default::default()
            },
            resolution: 1024,
        },
        TextureKey {
            shader_id: "dirt".to_string(),
            params: TextureParams {
                seed: 31415,
                scale: 48.0,
                roughness: 0.46,
                color_primary: [0.16, 0.10, 0.05, 1.0],
                color_secondary: [0.35, 0.24, 0.11, 1.0],
                octaves: 6.0,
                persistence: 0.50,
                lacunarity: 2.0,
                ..Default::default()
            },
            resolution: 1024,
        },
    ]
}
