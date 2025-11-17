use crate::vertex::UiVertex;
use wgpu::*;

pub fn polygon_sdf(px: f32, py: f32, verts: &[UiVertex]) -> f32 {
    let mut min_dist = f32::MAX;
    let mut winding = 0;
    let n = verts.len();

    for i in 0..n {
        let ax = verts[i].pos[0];
        let ay = verts[i].pos[1];
        let bx = verts[(i + 1) % n].pos[0];
        let by = verts[(i + 1) % n].pos[1];

        // edge vector
        let abx = bx - ax;
        let aby = by - ay;

        // point-to-a
        let apx = px - ax;
        let apy = py - ay;

        // projection factor
        let t = ((apx * abx + apy * aby) / (abx * abx + aby * aby)).clamp(0.0, 1.0);

        // closest point
        let cx = ax + abx * t;
        let cy = ay + aby * t;

        // distance to edge
        let dx = px - cx;
        let dy = py - cy;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist < min_dist {
            min_dist = dist;
        }

        // winding (stable inside/outside)
        let cond1 = (ay <= py) && (py < by);
        let cond2 = (by <= py) && (py < ay);
        let side = (px - ax) * (by - ay) - (bx - ax) * (py - ay);

        if cond1 && side > 0.0 {
            winding += 1;
        }
        if cond2 && side < 0.0 {
            winding -= 1;
        }
    }

    // inside = negative distance, outside = positive distance
    if winding == 0 {
        min_dist // outside
    } else {
        -min_dist // inside
    }
}

#[inline]
pub fn dist(a: f32, b: f32, c: f32, d: f32) -> f32 {
    ((a - c) * (a - c) + (b - d) * (b - d)).sqrt()
}

pub fn ensure_ccw(verts: &mut [UiVertex]) {
    if compute_signed_area(verts) < 0.0 {
        verts.reverse();
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

#[inline]
fn is_convex(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    let cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    cross > 0.0
}

fn point_in_triangle(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    let v0 = [c[0] - a[0], c[1] - a[1]];
    let v1 = [b[0] - a[0], b[1] - a[1]];
    let v2 = [p[0] - a[0], p[1] - a[1]];

    let dot00 = v0[0] * v0[0] + v0[1] * v0[1];
    let dot01 = v0[0] * v1[0] + v0[1] * v1[1];
    let dot02 = v0[0] * v2[0] + v0[1] * v2[1];
    let dot11 = v1[0] * v1[0] + v1[1] * v1[1];
    let dot12 = v1[0] * v2[0] + v1[1] * v2[1];

    let inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    u >= 0.0 && v >= 0.0 && (u + v) < 1.0
}

pub fn triangulate_polygon(verts: &[UiVertex]) -> Vec<[usize; 3]> {
    let n = verts.len();
    if n < 3 {
        return Vec::new();
    }

    // working index list
    let mut idx: Vec<usize> = (0..n).collect();
    let mut tris = Vec::new();

    // ear clipping loop
    while idx.len() > 3 {
        let mut ear_found = false;
        let m = idx.len();

        for i in 0..m {
            let i_prev = idx[(i + m - 1) % m];
            let i_curr = idx[i];
            let i_next = idx[(i + 1) % m];

            let a = verts[i_prev].pos;
            let b = verts[i_curr].pos;
            let c = verts[i_next].pos;

            // Check convexity
            if !is_convex(a, b, c) {
                continue;
            }

            // Check if any other vertex lies inside this triangle
            let mut contains = false;
            for &j in &idx {
                if j == i_prev || j == i_curr || j == i_next {
                    continue;
                }
                if point_in_triangle(verts[j].pos, a, b, c) {
                    contains = true;
                    break;
                }
            }
            if contains {
                continue;
            }

            // This is an ear
            tris.push([i_prev, i_curr, i_next]);
            idx.remove(i);
            ear_found = true;
            break;
        }

        if !ear_found {
            // Degenerate polygon or precision issue.
            break;
        }
    }

    // Final triangle
    if idx.len() == 3 {
        tris.push([idx[0], idx[1], idx[2]]);
    }

    tris
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
