use crate::terrain::roads::roads::RoadManager;
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
    total_game_time: f64,
}

impl Gizmo {
    pub fn update(&mut self, total_game_time: f64, road_manager: &RoadManager) {
        self.total_game_time = total_game_time;

        for node in &road_manager.nodes {
            // 1. Render the Node
            let node_color = if node.enabled {
                [0.0, 0.0, 0.9]
            } else {
                [1.0, 0.0, 0.0]
            };
            self.render_circle([node.x, node.y, node.z], 2.0, node_color);

            // 2. Render Incoming Lane
            for lane_id in &node.incoming_lanes {
                let lane = road_manager.lane(*lane_id);
                let segment = road_manager.segment(lane.segment());

                let lane_is_forward = lane.from_node() == segment.start();

                // Determine lane color once
                let lane_color = if lane.is_enabled() {
                    if lane_is_forward {
                        [0.0, 0.9, 0.0]
                    } else {
                        [0.2, 0.9, 0.0]
                    }
                } else {
                    [1.0, 0.05, 0.0]
                };

                // 3. Safe Polyline Iteration
                // .windows(2) gives [current, next] safely and stops before the overflow, such a cool function, you learn something new every day...!
                for points in lane.polyline().windows(2) {
                    let start = points[0];
                    let end = points[1];

                    self.render_arrow(
                        [start.x, start.y, start.z],
                        [end.x, end.y, end.z],
                        lane_color,
                        false,
                    );
                }
                if let Some(end) = lane.polyline().last() {
                    self.render_arrow(
                        [end.x, end.y, end.z],
                        [node.x, node.y, node.z],
                        lane_color,
                        false,
                    );
                }
            }
        }
    }

    pub fn new(device: &Device) -> Self {
        let gizmo_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Ez Gizmo VB"),
            size: (size_of::<LineVtx>() * 2048) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pending_renders: Vec::new(),
            gizmo_buffer,
            total_game_time: 0.0,
        }
    }
    pub fn clear(&mut self) {
        self.pending_renders.clear();
    }
    pub fn update_buffer(&mut self, device: &Device, queue: &Queue) -> u32 {
        let vertices = self.collect_vertices();

        let byte_size = (vertices.len() * size_of::<LineVtx>()) as u64;
        let mut max_size = self.gizmo_buffer.size();

        while byte_size > max_size {
            self.gizmo_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Ez Gizmo VB"),
                size: max_size * 2,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            max_size = self.gizmo_buffer.size();
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
    pub fn render_arrow(&mut self, start: [f32; 3], end: [f32; 3], color: [f32; 3], dashed: bool) {
        let mut vertices = Vec::<LineVtx>::new();

        // Direction + length
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        let dz = end[2] - start[2];
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len <= 0.0001 {
            return;
        }

        let inv_len = 1.0 / len;
        let nx = dx * inv_len;
        let ny = dy * inv_len;
        let nz = dz * inv_len;

        // Normalized dash parameters --------------------------------------------

        // Visual rule: about one arrow head every ~5 meters
        let target_spacing = 5.0;
        let steps = (len / target_spacing).ceil().max(1.0);
        let step_len = len / steps;

        // Dash ratio
        let dash_ratio = if dashed { 0.6 } else { 1.0 };
        let dash_len = step_len * dash_ratio;

        // Collect arrow-head positions (from END backwards)
        let mut head_positions = Vec::<f32>::new();

        let mut i = 0.0;
        while i < steps {
            let t_head = len - i * step_len;
            if t_head <= 0.0 {
                break;
            }
            head_positions.push(t_head);
            i += 1.0;
        }

        // Shaft -----------------------------------------------------------------

        for &t_head in &head_positions {
            let t0 = (t_head - dash_len).max(0.0);
            let t1 = t_head;

            let p0 = [start[0] + nx * t0, start[1] + ny * t0, start[2] + nz * t0];
            let p1 = [start[0] + nx * t1, start[1] + ny * t1, start[2] + nz * t1];

            vertices.push(LineVtx { pos: p0, color });
            vertices.push(LineVtx { pos: p1, color });
        }

        // Colors ----------------------------------------------------------------

        let flap_color = [
            (color[0] + 1.0) * 0.5,
            (color[1] + 1.0) * 0.5,
            (color[2] + 1.0) * 0.5,
        ];

        // Orthonormal frame -----------------------------------------------------

        let (ux0, uy0, uz0) = if ny.abs() > 0.99 {
            (1.0, 0.0, 0.0)
        } else {
            (0.0, 1.0, 0.0)
        };

        // side = dir × up
        let mut sx = ny * uz0 - nz * uy0;
        let mut sy = nz * ux0 - nx * uz0;
        let mut sz = nx * uy0 - ny * ux0;

        let sl = (sx * sx + sy * sy + sz * sz).sqrt();
        if sl <= 0.0001 {
            return;
        }

        let inv_sl = 1.0 / sl;
        sx *= inv_sl;
        sy *= inv_sl;
        sz *= inv_sl;

        // up_perp = dir × side
        let ux = ny * sz - nz * sy;
        let uy = nz * sx - nx * sz;
        let uz = nx * sy - ny * sx;

        // Arrow head params -----------------------------------------------------

        let head_len = step_len * 0.15;
        let head_width = 0.30;

        let spin_speed = 1.0;
        let time = self.total_game_time as f32;

        // Arrow heads -----------------------------------------------------------

        for (i, &t) in head_positions.iter().enumerate() {
            let px = start[0] + nx * t;
            let py = start[1] + ny * t;
            let pz = start[2] + nz * t;

            let bx = px - nx * head_len;
            let by = py - ny * head_len;
            let bz = pz - nz * head_len;

            let angle = time * spin_speed + i as f32 * 0.6;
            let c = angle.cos();
            let s = angle.sin();

            let rx = sx * c + ux * s;
            let ry = sy * c + uy * s;
            let rz = sz * c + uz * s;

            // Left flap
            vertices.push(LineVtx {
                pos: [px, py, pz],
                color: flap_color,
            });
            vertices.push(LineVtx {
                pos: [
                    bx + rx * head_width,
                    by + ry * head_width,
                    bz + rz * head_width,
                ],
                color: flap_color,
            });

            // Right flap
            vertices.push(LineVtx {
                pos: [px, py, pz],
                color: flap_color,
            });
            vertices.push(LineVtx {
                pos: [
                    bx - rx * head_width,
                    by - ry * head_width,
                    bz - rz * head_width,
                ],
                color: flap_color,
            });
        }

        self.pending_renders.push(PendingGizmoRender { vertices });
    }
}
