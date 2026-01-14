use crate::ui::vertex::LineVtx;
use glam::Vec3;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue};

const CIRCLE_SEGMENT_COUNT: usize = 16;

pub struct PendingGizmoRender {
    pub vertices: Vec<LineVtx>,
}
pub struct Gizmo {
    pub pending_renders: Vec<PendingGizmoRender>,
    pub gizmo_buffer: Buffer,
}

impl Gizmo {
    pub(crate) fn new(device: &Device) -> Self {
        let gizmo_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Ez Gizmo VB"),
            size: (size_of::<LineVtx>() * 2048) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pending_renders: Vec::new(),
            gizmo_buffer,
        }
    }
    pub fn clear(&mut self) {
        self.pending_renders.clear();
    }
    pub fn update_buffer(&mut self, device: &Device, queue: &Queue) -> u32 {
        let vertices = self.collect_vertices();

        let byte_size = (vertices.len() * size_of::<LineVtx>()) as u64;
        let max_size = self.gizmo_buffer.size();

        if byte_size > max_size {
            self.gizmo_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Ez Gizmo VB"),
                size: max_size * 2,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        queue.write_buffer(&self.gizmo_buffer, 0, bytemuck::cast_slice(&vertices));

        vertices.len() as u32
    }

    pub fn update_gizmo_vertices(
        &mut self,
        target: Vec3,
        orbit_radius: f32,
        scale_with_orbit: bool,
    ) {
        let t = target;

        // 1 m per axis or radius-scaled gizmo
        let s = if scale_with_orbit {
            orbit_radius * 0.2
        } else {
            1.0
        };

        let axes = [
            LineVtx {
                pos: [t.x, t.y, t.z],
                color: [1.0, 0.2, 0.2],
            },
            LineVtx {
                pos: [t.x + s, t.y, t.z],
                color: [1.0, 0.2, 0.2],
            },
            LineVtx {
                pos: [t.x, t.y, t.z],
                color: [0.2, 1.0, 0.2],
            },
            LineVtx {
                pos: [t.x, t.y + s, t.z],
                color: [0.2, 1.0, 0.2],
            },
            LineVtx {
                pos: [t.x, t.y, t.z],
                color: [0.2, 0.6, 1.0],
            },
            LineVtx {
                pos: [t.x, t.y, t.z + s],
                color: [0.2, 0.6, 1.0],
            },
        ];
        self.pending_renders.push(PendingGizmoRender {
            vertices: axes.to_vec(),
        });
    }
    pub fn collect_vertices(&self) -> Vec<LineVtx> {
        let mut out = Vec::new();
        for draw in &self.pending_renders {
            out.extend_from_slice(&draw.vertices);
        }
        out
    }
    pub fn render_circle(&mut self, position: [f32; 3], radius: f32, color: [f32; 3]) {
        let mut circle_vertices = Vec::<LineVtx>::new();
        let (cx, cy, cz) = (position[0], position[1], position[2]);
        for i in 0..CIRCLE_SEGMENT_COUNT {
            let t = i as f32 / CIRCLE_SEGMENT_COUNT as f32;
            let angle = t * std::f32::consts::TAU;
            let x = cx + radius * angle.cos();
            let z = cz + radius * angle.sin();
            circle_vertices.push(LineVtx {
                pos: [x, cy, z],
                color,
            })
        }
        self.pending_renders.push(PendingGizmoRender {
            vertices: circle_vertices,
        })
    }
    pub fn render_arrow(&mut self, start: [f32; 3], end: [f32; 3], color: [f32; 3]) {
        let mut vertices = Vec::<LineVtx>::new();

        // Main line
        vertices.push(LineVtx { pos: start, color });
        vertices.push(LineVtx { pos: end, color });

        // Direction
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        let dz = end[2] - start[2];

        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len == 0.0 {
            return;
        }

        let nx = dx / len;
        let ny = dy / len;
        let nz = dz / len;

        // Pick an up vector that is not parallel
        let (ux, uy, uz) = if ny.abs() > 0.99 {
            (1.0, 0.0, 0.0)
        } else {
            (0.0, 1.0, 0.0)
        };

        // side = dir × up
        let mut sx = ny * uz - nz * uy;
        let mut sy = nz * ux - nx * uz;
        let mut sz = nx * uy - ny * ux;

        let slen = (sx * sx + sy * sy + sz * sz).sqrt();
        if slen == 0.0 {
            return;
        }

        sx /= slen;
        sy /= slen;
        sz /= slen;

        let head_len = 0.3;
        let head_width = 0.15;

        let hx = end[0] - nx * head_len;
        let hy = end[1] - ny * head_len;
        let hz = end[2] - nz * head_len;

        // Left flap
        vertices.push(LineVtx { pos: end, color });
        vertices.push(LineVtx {
            pos: [
                hx + sx * head_width,
                hy + sy * head_width,
                hz + sz * head_width,
            ],
            color,
        });

        // Right flap
        vertices.push(LineVtx { pos: end, color });
        vertices.push(LineVtx {
            pos: [
                hx - sx * head_width,
                hy - sy * head_width,
                hz - sz * head_width,
            ],
            color,
        });

        self.pending_renders.push(PendingGizmoRender { vertices });
    }
}
