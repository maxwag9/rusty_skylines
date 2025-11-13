use crate::vertex::UiVertex;
use wgpu::*;

pub fn point_in_polygon(px: f32, py: f32, verts: &[UiVertex]) -> bool {
    let mut inside = false;
    let mut j = verts.len() - 1;

    for i in 0..verts.len() {
        let xi = verts[i].pos[0];
        let yi = verts[i].pos[1];
        let xj = verts[j].pos[0];
        let yj = verts[j].pos[1];

        // raycasting method
        let intersect =
            ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi + 0.000001) + xi);

        if intersect {
            inside = !inside;
        }

        j = i;
    }
    inside
}

pub fn distance_to_segment(px: f32, py: f32, a: &UiVertex, b: &UiVertex) -> f32 {
    let ax = a.pos[0];
    let ay = a.pos[1];
    let bx = b.pos[0];
    let by = b.pos[1];

    let apx = px - ax;
    let apy = py - ay;
    let abx = bx - ax;
    let aby = by - ay;

    let ab_len2 = abx * abx + aby * aby;
    let t = if ab_len2 > 0.0 {
        (apx * abx + apy * aby) / ab_len2
    } else {
        0.0
    }
    .clamp(0.0, 1.0);

    let cx = ax + abx * t;
    let cy = ay + aby * t;

    ((px - cx).powi(2) + (py - cy).powi(2)).sqrt()
}

pub fn polygon_edge_distance(px: f32, py: f32, verts: &[UiVertex]) -> f32 {
    let mut min_d = f32::MAX;

    for i in 0..verts.len() {
        let a = &verts[i];
        let b = &verts[(i + 1) % verts.len()];
        let d = distance_to_segment(px, py, a, b);
        if d < min_d {
            min_d = d;
        }
    }

    min_d
}

pub(crate) fn make_pipeline(
    device: &Device,
    label: &str,
    layout: &PipelineLayout,
    shader: &ShaderModule,
    vs_entry: &str,
    fs_entry: &str,
    buffers: &[VertexBufferLayout],
    format: TextureFormat,
    blend: Option<BlendState>,
    topology: PrimitiveTopology,
) -> RenderPipeline {
    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: VertexState {
            module: shader,
            entry_point: Some(vs_entry),
            buffers,
            compilation_options: Default::default(),
        },
        fragment: Some(FragmentState {
            module: shader,
            entry_point: Some(fs_entry),
            targets: &[Some(ColorTargetState {
                format,
                blend,
                write_mask: ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: PrimitiveState {
            topology,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

pub fn make_uniform_layout(device: &Device, label: &str) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX_FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}
