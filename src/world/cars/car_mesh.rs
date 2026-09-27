use crate::helpers::hsv::HSV;
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
        v_max: 0.980,
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
    use std::f32::consts::TAU;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let body_width = 1.78;
    let body_length = 4.6;
    let body_bottom = 0.32;
    let body_height = 0.58;

    let cabin_width_bottom = 1.52;
    let cabin_width_top = 1.28;
    let cabin_front_bottom = 1.05;
    let cabin_rear_bottom = -1.15;
    let cabin_front_top = 0.68;
    let cabin_rear_top = -0.85;
    let cabin_bottom_y = body_bottom + body_height * 0.82;
    let cabin_top_y = 1.72;

    let wheel_radius = 0.36;
    let wheel_width = 0.28;
    let wheel_x = body_width * 0.5 + wheel_width * 0.38;
    let front_axle_z = 1.48;
    let rear_axle_z = -1.46;

    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn normalize(v: [f32; 3]) -> [f32; 3] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if len > 0.0 {
            [v[0] / len, v[1] / len, v[2] / len]
        } else {
            [0.0, 1.0, 0.0]
        }
    }

    fn add_quad(
        vertices: &mut Vec<CarVertex>,
        indices: &mut Vec<u32>,
        corners: [[f32; 3]; 4],
        color: [f32; 3],
        uv_size: [f32; 2],
        normal: [f32; 3],
    ) {
        let base = vertices.len() as u32;

        let uvs = [
            [0.0, 0.0],
            [uv_size[0], 0.0],
            [uv_size[0], uv_size[1]],
            [0.0, uv_size[1]],
        ];

        for i in 0..4 {
            vertices.push(CarVertex {
                position: corners[i],
                normals: normal,
                color,
                uv: uvs[i],
                _pad: 0.0,
            });
        }

        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    fn add_box(
        vertices: &mut Vec<CarVertex>,
        indices: &mut Vec<u32>,
        center: [f32; 3],
        size: [f32; 3],
        color: [f32; 3],
    ) {
        let hx = size[0] * 0.5;
        let hy = size[1] * 0.5;
        let hz = size[2] * 0.5;

        let x0 = center[0] - hx;
        let x1 = center[0] + hx;
        let y0 = center[1] - hy;
        let y1 = center[1] + hy;
        let z0 = center[2] - hz;
        let z1 = center[2] + hz;

        add_quad(
            vertices,
            indices,
            [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]],
            color,
            [size[2], size[1]],
            [-1.0, 0.0, 0.0],
        );

        add_quad(
            vertices,
            indices,
            [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]],
            color,
            [size[2], size[1]],
            [1.0, 0.0, 0.0],
        );

        add_quad(
            vertices,
            indices,
            [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
            color,
            [size[0], size[1]],
            [0.0, 0.0, 1.0],
        );

        add_quad(
            vertices,
            indices,
            [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]],
            color,
            [size[0], size[1]],
            [0.0, 0.0, -1.0],
        );

        add_quad(
            vertices,
            indices,
            [[x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0]],
            color,
            [size[0], size[2]],
            [0.0, 1.0, 0.0],
        );

        add_quad(
            vertices,
            indices,
            [[x0, y0, z1], [x0, y0, z0], [x1, y0, z0], [x1, y0, z1]],
            color,
            [size[0], size[2]],
            [0.0, -1.0, 0.0],
        );
    }

    fn add_cabin(
        vertices: &mut Vec<CarVertex>,
        indices: &mut Vec<u32>,
        width_bottom: f32,
        width_top: f32,
        front_bottom: f32,
        rear_bottom: f32,
        front_top: f32,
        rear_top: f32,
        bottom_y: f32,
        top_y: f32,
        color: [f32; 3],
    ) {
        let xb = width_bottom * 0.5;
        let xt = width_top * 0.5;

        let v = [
            [-xb, bottom_y, rear_bottom],
            [xb, bottom_y, rear_bottom],
            [xb, bottom_y, front_bottom],
            [-xb, bottom_y, front_bottom],
            [-xt, top_y, rear_top],
            [xt, top_y, rear_top],
            [xt, top_y, front_top],
            [-xt, top_y, front_top],
        ];

        let faces = [
            ([0, 3, 2, 1], [0.0, -1.0, 0.0]),
            ([4, 5, 6, 7], [0.0, 1.0, 0.0]),
            ([0, 4, 7, 3], [-1.0, 0.0, 0.0]),
            ([1, 2, 6, 5], [1.0, 0.0, 0.0]),
            ([3, 7, 6, 2], [0.0, 0.0, 1.0]),
            ([1, 5, 4, 0], [0.0, 0.0, -1.0]),
        ];

        for (face, fallback_normal) in faces {
            let a = v[face[0]];
            let b = v[face[1]];
            let c = v[face[2]];

            let normal = normalize(cross(
                [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
                [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
            ));

            let normal = if normal[0].is_finite()
                && normal[1].is_finite()
                && normal[2].is_finite()
                && (normal[0].abs() + normal[1].abs() + normal[2].abs()) > 0.0
            {
                normal
            } else {
                fallback_normal
            };

            add_quad(
                vertices,
                indices,
                [v[face[0]], v[face[1]], v[face[2]], v[face[3]]],
                color,
                [1.0, 1.0],
                normal,
            );
        }
    }

    fn add_cylinder(
        vertices: &mut Vec<CarVertex>,
        indices: &mut Vec<u32>,
        center: [f32; 3],
        radius: f32,
        width: f32,
        color: [f32; 3],
        segments: u32,
    ) {
        let base = vertices.len() as u32;
        let x0 = center[0] - width * 0.5;
        let x1 = center[0] + width * 0.5;
        let y = center[1];
        let z = center[2];

        for i in 0..segments {
            let t = i as f32 / segments as f32;
            let a = t * TAU;
            let cy = a.cos();
            let sz = a.sin();

            vertices.push(CarVertex {
                position: [x0, y + cy * radius, z + sz * radius],
                normals: [0.0, cy, sz],
                color,
                uv: [t, 0.0],
                _pad: 0.0,
            });

            vertices.push(CarVertex {
                position: [x1, y + cy * radius, z + sz * radius],
                normals: [0.0, cy, sz],
                color,
                uv: [t, 1.0],
                _pad: 0.0,
            });
        }

        for i in 0..segments {
            let n = (i + 1) % segments;

            let a = base + i * 2;
            let b = base + i * 2 + 1;
            let c = base + n * 2 + 1;
            let d = base + n * 2;

            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }

        let left_center = vertices.len() as u32;
        vertices.push(CarVertex {
            position: [x0, y, z],
            normals: [-1.0, 0.0, 0.0],
            color,
            uv: [0.5, 0.5],
            _pad: 0.0,
        });

        let right_center = vertices.len() as u32;
        vertices.push(CarVertex {
            position: [x1, y, z],
            normals: [1.0, 0.0, 0.0],
            color,
            uv: [0.5, 0.5],
            _pad: 0.0,
        });

        for i in 0..segments {
            let n = (i + 1) % segments;

            let p0 = base + i * 2;
            let p1 = base + n * 2;

            indices.extend_from_slice(&[left_center, p1, p0]);

            let p0 = base + i * 2 + 1;
            let p1 = base + n * 2 + 1;

            indices.extend_from_slice(&[right_center, p0, p1]);
        }
    }

    let body_color = [0.72, 0.07, 0.06];
    let body_dark = [0.12, 0.12, 0.13];
    let glass_color = [0.12, 0.32, 0.42];
    let tire_color = [0.025, 0.025, 0.028];
    let hub_color = [0.38, 0.40, 0.42];
    let headlight_color = [0.95, 0.92, 0.72];
    let taillight_color = [0.8, 0.025, 0.02];

    add_box(
        &mut vertices,
        &mut indices,
        [0.0, body_bottom + body_height * 0.5, 0.0],
        [body_width, body_height, body_length],
        body_color,
    );

    add_box(
        &mut vertices,
        &mut indices,
        [0.0, body_bottom + body_height * 0.82, 1.28],
        [body_width * 0.94, 0.22, 1.65],
        body_color,
    );

    add_box(
        &mut vertices,
        &mut indices,
        [0.0, body_bottom + body_height * 0.82, -1.48],
        [body_width * 0.94, 0.22, 1.1],
        body_color,
    );

    add_cabin(
        &mut vertices,
        &mut indices,
        cabin_width_bottom,
        cabin_width_top,
        cabin_front_bottom,
        cabin_rear_bottom,
        cabin_front_top,
        cabin_rear_top,
        cabin_bottom_y,
        cabin_top_y,
        glass_color,
    );

    add_box(
        &mut vertices,
        &mut indices,
        [0.0, 0.47, body_length * 0.5 + 0.025],
        [body_width * 0.86, 0.18, 0.10],
        body_dark,
    );

    add_box(
        &mut vertices,
        &mut indices,
        [0.0, 0.47, -body_length * 0.5 - 0.025],
        [body_width * 0.86, 0.18, 0.10],
        body_dark,
    );

    for x in [-0.55, 0.55] {
        add_box(
            &mut vertices,
            &mut indices,
            [x, 0.66, body_length * 0.5 + 0.035],
            [0.34, 0.16, 0.05],
            headlight_color,
        );

        add_box(
            &mut vertices,
            &mut indices,
            [x, 0.66, -body_length * 0.5 - 0.035],
            [0.34, 0.16, 0.05],
            taillight_color,
        );
    }

    for x in [-wheel_x, wheel_x] {
        for z in [front_axle_z, rear_axle_z] {
            add_cylinder(
                &mut vertices,
                &mut indices,
                [x, wheel_radius, z],
                wheel_radius,
                wheel_width,
                tire_color,
                16,
            );

            let hub_x = x + if x > 0.0 {
                wheel_width * 0.5 + 0.006
            } else {
                -wheel_width * 0.5 - 0.006
            };

            add_cylinder(
                &mut vertices,
                &mut indices,
                [hub_x, wheel_radius, z],
                wheel_radius * 0.48,
                0.055,
                hub_color,
                12,
            );
        }
    }

    (vertices, indices)
}
