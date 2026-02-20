use crate::renderer::pipelines::DEPTH_FORMAT;
use crate::ui::input::Input;
use crate::ui::vertex::UiVertex;
use std::cmp::Ordering;
use wgpu::*;

pub fn calc_move_speed(input: &Input) -> f32 {
    let mut base = 8.0;

    match (input.shift, input.ctrl) {
        (true, false) => base *= 12.0, // fast
        (false, true) => base *= 0.6,  // slow
        (true, true) => base *= 0.2,   // ultra-fine
        _ => {}
    }

    base
}

pub fn ensure_ccw(verts: &mut [UiVertex]) {
    if verts.len() < 3 {
        return;
    }

    // 1) centroid
    let mut cx = 0.0;
    let mut cy = 0.0;
    for v in verts.iter() {
        cx += v.pos[0];
        cy += v.pos[1];
    }
    let n = verts.len() as f32;
    cx /= n;
    cy /= n;

    // 2) sort by angle around centroid
    verts.sort_by(|a, b| {
        let ax = a.pos[0] - cx;
        let ay = a.pos[1] - cy;
        let bx = b.pos[0] - cx;
        let by = b.pos[1] - cy;

        let ang_a = ay.atan2(ax);
        let ang_b = by.atan2(bx);

        ang_a.partial_cmp(&ang_b).unwrap_or(Ordering::Equal)
    });

    // 3) enforce CCW orientation (for backface culling etc.)
    if compute_signed_area(verts) < 0.0 {
        verts.reverse();
    }
}

pub(crate) fn triangulate_polygon(verts: &mut Vec<UiVertex>) -> Vec<UiVertex> {
    let n = verts.len();
    if n < 3 {
        return Vec::new();
    }

    // working index list
    let mut idx: Vec<usize> = (0..n).collect();
    let mut tri_indices: Vec<u32> = Vec::with_capacity((n - 2) * 3);

    let is_convex = |a: [f32; 2], b: [f32; 2], c: [f32; 2]| -> bool {
        let cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        cross > 0.0
    };

    let point_in_tri = |p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]| -> bool {
        fn sign(p1: [f32; 2], p2: [f32; 2], p3: [f32; 2]) -> f32 {
            (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        }
        let d1 = sign(p, a, b);
        let d2 = sign(p, b, c);
        let d3 = sign(p, c, a);
        let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
        let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
        !(has_neg && has_pos)
    };

    let mut guard = 0;
    while idx.len() > 3 && guard < 20_000 {
        guard += 1;
        let m = idx.len();
        let mut ear_found = false;

        for i in 0..m {
            let a = idx[(i + m - 1) % m];
            let b = idx[i];
            let c = idx[(i + 1) % m];

            let p0 = verts[a].pos;
            let p1 = verts[b].pos;
            let p2 = verts[c].pos;

            // avoid degeneracy
            let area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]);
            if area.abs() < 1e-6 {
                continue;
            }

            if !is_convex(p0, p1, p2) {
                continue;
            }

            let mut contains = false;
            for &j in &idx {
                if j == a || j == b || j == c {
                    continue;
                }
                if point_in_tri(verts[j].pos, p0, p1, p2) {
                    contains = true;
                    break;
                }
            }
            if contains {
                continue;
            }

            // to guarantee CCW triangles:
            push_triangle_indices(&mut tri_indices, area, a, b, c);

            idx.remove(i);
            ear_found = true;
            break;
        }

        if !ear_found {
            // try brute fallback: triangulate fan (keeps UI working)
            for i in 1..idx.len() - 1 {
                tri_indices.push(idx[0] as u32);
                tri_indices.push(idx[i] as u32);
                tri_indices.push(idx[i + 1] as u32);
            }
            idx.clear();
            break;
        }
    }

    if idx.len() == 3 {
        let a = idx[0];
        let b = idx[1];
        let c = idx[2];

        let p0 = verts[a].pos;
        let p1 = verts[b].pos;
        let p2 = verts[c].pos;

        let area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]);

        push_triangle_indices(&mut tri_indices, area, a, b, c);
    }

    // convert index list into raw UiVertex list
    tri_indices.into_iter().map(|i| verts[i as usize]).collect()
}

fn push_triangle_indices(tri_indices: &mut Vec<u32>, area: f32, a: usize, b: usize, c: usize) {
    if area < 0.0 {
        tri_indices.push(a as u32);
        tri_indices.push(c as u32);
        tri_indices.push(b as u32);
    } else {
        tri_indices.push(a as u32);
        tri_indices.push(b as u32);
        tri_indices.push(c as u32);
    }
}
fn compute_signed_area(verts: &[UiVertex]) -> f32 {
    let mut area = 0.0;
    let n = verts.len();
    for i in 0..n {
        let p0 = verts[i].pos;
        let p1 = verts[(i + 1) % n].pos;
        area += p0[0] * p1[1] - p1[0] * p0[1];
    }
    area * 0.5
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
    multisample: MultisampleState,
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
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare: CompareFunction::Always,
            stencil: Default::default(),
            bias: Default::default(),
        }),

        multisample,
        cache: None,
        multiview_mask: None,
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

pub fn make_storage_layout(device: &Device, label: &str) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX_FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}
