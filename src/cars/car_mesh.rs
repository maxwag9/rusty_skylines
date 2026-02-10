use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CarVertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}
pub fn create_procedural_car() -> (Vec<CarVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Tunable params – tweak these to taste
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

    // Helper: add colored axis-aligned box
    fn add_box(
        vertices: &mut Vec<CarVertex>,
        indices: &mut Vec<u32>,
        center: [f32; 3],
        size: [f32; 3],
        color: [f32; 3],
    ) {
        let half = [size[0] / 2.0, size[1] / 2.0, size[2] / 2.0];
        let base_idx = vertices.len() as u32;

        let rel_corners = [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ];

        for &c in &rel_corners {
            vertices.push(CarVertex {
                position: [c[0] + center[0], c[1] + center[1], c[2] + center[2]],
                color,
            });
        }

        // 12 triangles (counter-clockwise)
        let face_indices = [
            [0, 1, 2, 2, 3, 0], // -Z
            [4, 6, 5, 6, 4, 7], // +Z (fixed order for correct winding)
            [0, 3, 7, 7, 4, 0], // -X
            [1, 5, 6, 6, 2, 1], // +X
            [3, 2, 6, 6, 7, 3], // +Y
            [0, 4, 5, 5, 1, 0], // -Y
        ];

        for face in face_indices {
            for &i in &face {
                indices.push(base_idx + i);
            }
        }
    }

    // Body - red
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

    // Wheels - black, blocky (square wheels = retro vibe)
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
            [0.1, 0.1, 0.1],
        );
    }

    (vertices, indices)
}
