use crate::hsv::HSV;
use bytemuck::{Pod, Zeroable};
use rand::{Rng, RngExt};
use wgpu::{VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode};

#[derive(Copy, Clone)]
pub struct WeightedHsv {
    pub weight: f32, // percentage weight, normalized later
    pub h_min: f32,
    pub h_max: f32,
    pub s_min: f32,
    pub s_max: f32,
    pub v_min: f32,
    pub v_max: f32,
}

pub static CAR_COLOR_DISTRIBUTION: &[WeightedHsv] = &[
    // Neutrals (~70%)
    WeightedHsv {
        // White
        weight: 30.0,
        h_min: 0.0,
        h_max: 1.0,
        s_min: 0.00,
        s_max: 0.05,
        v_min: 0.90,
        v_max: 1.00,
    },
    WeightedHsv {
        // Black
        weight: 22.0,
        h_min: 0.0,
        h_max: 1.0,
        s_min: 0.00,
        s_max: 0.05,
        v_min: 0.02,
        v_max: 0.07,
    },
    WeightedHsv {
        // Gray / Silver
        weight: 18.0,
        h_min: 0.0,
        h_max: 1.0,
        s_min: 0.00,
        s_max: 0.08,
        v_min: 0.45,
        v_max: 0.65,
    },
    // Chromatic (~30%)
    WeightedHsv {
        // Blue
        weight: 8.0,
        h_min: 0.55,
        h_max: 0.65,
        s_min: 0.60,
        s_max: 0.85,
        v_min: 0.40,
        v_max: 0.85,
    },
    WeightedHsv {
        // Red
        weight: 4.0,
        h_min: 0.97,
        h_max: 1.00,
        s_min: 0.70,
        s_max: 0.90,
        v_min: 0.35,
        v_max: 0.80,
    },
    WeightedHsv {
        // Green
        weight: 1.0,
        h_min: 0.30,
        h_max: 0.40,
        s_min: 0.60,
        s_max: 0.75,
        v_min: 0.45,
        v_max: 0.75,
    },
    WeightedHsv {
        // Yellow / Orange
        weight: 1.0,
        h_min: 0.08,
        h_max: 0.15,
        s_min: 0.60,
        s_max: 0.95,
        v_min: 0.60,
        v_max: 0.95,
    },
];
pub fn sample_car_color<R: Rng>(rng: &mut R) -> HSV {
    let total_weight: f32 = CAR_COLOR_DISTRIBUTION.iter().map(|c| c.weight).sum();

    let mut pick = rng.random_range(0.0..total_weight);

    for c in CAR_COLOR_DISTRIBUTION {
        if pick < c.weight {
            let mut h = rng.random_range(c.h_min..c.h_max);
            if c.h_min > c.h_max {
                h = (h + 1.0).fract();
            }

            return HSV {
                h,
                s: rng.random_range(c.s_min..c.s_max),
                v: rng.random_range(c.v_min..c.v_max),
            };
        }
        pick -= c.weight;
    }

    // fallback, should never happen
    HSV {
        h: 0.0,
        s: 0.0,
        v: 1.0,
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CarVertex {
    pub position: [f32; 3],
    pub normals: [f32; 3],
    pub color: [f32; 3],
    pub uv: [f32; 2],
    pub _pad: f32, // padding to 16-byte alignment (total size = 48)
}
impl CarVertex {
    pub fn layout<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            array_stride: size_of::<CarVertex>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: &[
                VertexAttribute {
                    shader_location: 0,
                    format: VertexFormat::Float32x3,
                    offset: 0,
                },
                VertexAttribute {
                    shader_location: 1,
                    format: VertexFormat::Float32x3,
                    offset: 12,
                },
                VertexAttribute {
                    shader_location: 2,
                    format: VertexFormat::Float32x3,
                    offset: 24,
                },
                VertexAttribute {
                    shader_location: 3,
                    format: VertexFormat::Float32x2,
                    offset: 36,
                },
            ],
        }
    }
}

pub fn create_procedural_car() -> (Vec<CarVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let wheel_radius: f32 = 0.35;
    let body_width: f32 = 1.7;
    let body_length: f32 = 4.5;
    let body_height: f32 = 0.6;
    let cabin_width: f32 = body_width * 0.8;
    let cabin_length: f32 = body_length * 0.6;
    let cabin_height: f32 = 0.8;
    let wheel_thickness: f32 = 0.3;

    let axle_front_z: f32 = body_length * 0.35;
    let axle_rear_z: f32 = -axle_front_z;

    let body_center_y = wheel_radius + body_height / 2.0;
    let cabin_center_y = wheel_radius + body_height + cabin_height / 2.0;

    const UV_WORLD_SCALE: f32 = 1.0;

    fn add_box(
        vertices: &mut Vec<CarVertex>,
        indices: &mut Vec<u32>,
        center: [f32; 3],
        size: [f32; 3],
        color: [f32; 3],
    ) {
        let hx = size[0] / 2.0;
        let hy = size[1] / 2.0;
        let hz = size[2] / 2.0;

        let x0 = center[0] - hx;
        let x1 = center[0] + hx;
        let y0 = center[1] - hy;
        let y1 = center[1] + hy;
        let z0 = center[2] - hz;
        let z1 = center[2] + hz;

        fn push_face(
            vertices: &mut Vec<CarVertex>,
            indices: &mut Vec<u32>,
            a: [f32; 3],
            b: [f32; 3],
            c: [f32; 3],
            d: [f32; 3],
            normals: [f32; 3],
            uv_size: [f32; 2],
            color: [f32; 3],
        ) {
            let base = vertices.len() as u32;

            // UVs: (0,0),(1,0),(1,1),(0,1) scaled by uv_size (tiling)
            let uvs = [
                [0.0, 0.0],
                [uv_size[0], 0.0],
                [uv_size[0], uv_size[1]],
                [0.0, uv_size[1]],
            ];

            let corners = [a, b, c, d];
            for i in 0..4 {
                vertices.push(CarVertex {
                    position: corners[i],
                    normals,
                    color,
                    uv: uvs[i],
                    _pad: 0.0,
                });
            }

            // two triangles (CW).
            indices.extend_from_slice(&[base + 2, base + 1, base, base + 3, base + 2, base]);
        }

        push_face(
            vertices,
            indices,
            [x0, y0, z0], // a
            [x1, y0, z0], // b
            [x1, y1, z0], // c
            [x0, y1, z0], // d
            [0.0, 0.0, -1.0],
            [size[0] / UV_WORLD_SCALE, size[1] / UV_WORLD_SCALE],
            color,
        );

        // +Z face (front)
        push_face(
            vertices,
            indices,
            [x1, y0, z1],
            [x0, y0, z1],
            [x0, y1, z1],
            [x1, y1, z1],
            [0.0, 0.0, 1.0],
            [size[0] / UV_WORLD_SCALE, size[1] / UV_WORLD_SCALE],
            color,
        );

        // -X face (left) : plane Z (u) vs Y (v) -> uv_size = (size.z, size.y)
        push_face(
            vertices,
            indices,
            [x0, y0, z1],
            [x0, y0, z0],
            [x0, y1, z0],
            [x0, y1, z1],
            [-1.0, 0.0, 0.0],
            [size[2] / UV_WORLD_SCALE, size[1] / UV_WORLD_SCALE],
            color,
        );

        // +X face (right)
        push_face(
            vertices,
            indices,
            [x1, y0, z0],
            [x1, y0, z1],
            [x1, y1, z1],
            [x1, y1, z0],
            [1.0, 0.0, 0.0],
            [size[2] / UV_WORLD_SCALE, size[1] / UV_WORLD_SCALE],
            color,
        );

        // +Y face (top)
        push_face(
            vertices,
            indices,
            [x0, y1, z0],
            [x1, y1, z0],
            [x1, y1, z1],
            [x0, y1, z1],
            [0.0, 1.0, 0.0],
            [size[0] / UV_WORLD_SCALE, size[2] / UV_WORLD_SCALE],
            color,
        );

        // -Y face (bottom)
        push_face(
            vertices,
            indices,
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y0, z0],
            [x0, y0, z0],
            [0.0, -1.0, 0.0],
            [size[0] / UV_WORLD_SCALE, size[2] / UV_WORLD_SCALE],
            color,
        );
    }

    // Body - red (albedo)
    add_box(
        &mut vertices,
        &mut indices,
        [0.0, body_center_y, 0.0],
        [body_width, body_height, body_length],
        [0.9, 0.15, 0.15],
    );

    // Cabin - cyan "glass"
    add_box(
        &mut vertices,
        &mut indices,
        [0.0, cabin_center_y, -0.2], // slight rear shift for style
        [cabin_width, cabin_height, cabin_length],
        [0.4, 0.8, 0.95],
    );

    // Wheels - black, blocky
    let wheel_size = [wheel_thickness, wheel_radius * 2.0, wheel_radius * 2.0];
    let wheel_positions = [
        [-body_width / 2.0, wheel_radius, axle_front_z],
        [body_width / 2.0, wheel_radius, axle_front_z],
        [-body_width / 2.0, wheel_radius, axle_rear_z],
        [body_width / 2.0, wheel_radius, axle_rear_z],
    ];

    for pos in wheel_positions {
        add_box(
            &mut vertices,
            &mut indices,
            pos,
            wheel_size,
            [0.05, 0.05, 0.05],
        );
    }

    (vertices, indices)
}
