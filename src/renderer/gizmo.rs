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
        let render_disabled_lanes = false;
        let render_arrow_lane_to_node = false;

        // iterate both storages but resolve everything against the storage the node belongs to
        for storage in [&road_manager.roads, &road_manager.preview_roads] {
            for (_node_id, node) in storage.iter_nodes() {
                // 1. Render the Node
                let node_color = if node.is_enabled() {
                    [0.0, 0.0, 0.9]
                } else {
                    [1.0, 0.0, 0.0]
                };
                self.render_circle([node.x(), node.y(), node.z()], 2.0, node_color);

                // 2. Render Incoming Lane (resolve lanes/segments from the same storage)
                for &lane_id in node.incoming_lanes().iter() {
                    let lane = storage.lane(lane_id);
                    let segment = storage.segment(lane.segment());

                    let lane_is_forward = lane.from_node() == segment.start();

                    // Determine lane color once
                    let lane_color = if lane.is_enabled() {
                        if lane_is_forward {
                            [0.0, 0.9, 0.0]
                        } else {
                            [0.2, 0.9, 0.0]
                        }
                    } else {
                        if !render_disabled_lanes {
                            continue;
                        }
                        [1.0, 0.05, 0.0]
                    };

                    // 3. Safe Polyline Iteration
                    let poly: Vec<[f32; 3]> =
                        lane.polyline().iter().map(|p| [p.x, p.y, p.z]).collect();

                    self.render_polyline(&poly, lane_color, false);

                    if render_arrow_lane_to_node {
                        if let Some(end) = lane.polyline().last() {
                            self.render_arrow(
                                [end.x, end.y, end.z],
                                [node.x(), node.y(), node.z()],
                                lane_color,
                                false,
                            );
                        }
                    }
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
    pub fn render_line(&mut self, start: [f32; 3], end: [f32; 3], color: [f32; 3]) {
        let line_vertices = [LineVtx { pos: start, color }, LineVtx { pos: end, color }];

        self.pending_renders.push(PendingGizmoRender {
            vertices: line_vertices.to_vec(),
        });
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
        let target_spacing = 150.0;
        let steps = (len / target_spacing).ceil().max(1.0);
        let step_len = target_spacing;

        // Dash ratio
        let dash_ratio = if dashed { 0.8 } else { 1.0 };
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

        let head_len = 0.15;
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

            let angle = time * spin_speed + (i as f32 * 12.9898).sin() * 3.14;
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

    pub fn render_polyline(&mut self, polyline: &[[f32; 3]], color: [f32; 3], dashed: bool) {
        if polyline.len() < 2 {
            return;
        }

        let mut vertices = Vec::<LineVtx>::new();

        // ---------- draw full shaft first ----------

        for i in 0..polyline.len() - 1 {
            vertices.push(LineVtx {
                pos: polyline[i],
                color,
            });
            vertices.push(LineVtx {
                pos: polyline[i + 1],
                color,
            });
        }

        // ---------- cumulative lengths ----------

        let mut lengths = Vec::with_capacity(polyline.len());
        lengths.push(0.0);

        for i in 1..polyline.len() {
            let dx = polyline[i][0] - polyline[i - 1][0];
            let dy = polyline[i][1] - polyline[i - 1][1];
            let dz = polyline[i][2] - polyline[i - 1][2];
            lengths.push(lengths[i - 1] + (dx * dx + dy * dy + dz * dz).sqrt());
        }

        let total_len = *lengths.last().unwrap();
        if total_len <= 0.0001 {
            return;
        }

        // ---------- arrow parameters ----------

        let spacing = 7.0;
        let head_len = 0.30;
        let head_width = 0.25;

        let flap_color = [
            (color[0] + 1.0) * 0.5,
            (color[1] + 1.0) * 0.5,
            (color[2] + 1.0) * 0.5,
        ];

        let time = self.total_game_time as f32;
        let spin_speed = 1.0;

        // ---------- helper: sample position + dir ----------

        let sample_at = |t: f32| -> ([f32; 3], [f32; 3]) {
            let mut i = 1;
            while i < lengths.len() && lengths[i] < t {
                i += 1;
            }

            let i0 = i - 1;
            let i1 = i.min(polyline.len() - 1);

            let l0 = lengths[i0];
            let l1 = lengths[i1];
            let s = if l1 > l0 { (t - l0) / (l1 - l0) } else { 0.0 };

            let p0 = polyline[i0];
            let p1 = polyline[i1];

            let pos = [
                p0[0] + (p1[0] - p0[0]) * s,
                p0[1] + (p1[1] - p0[1]) * s,
                p0[2] + (p1[2] - p0[2]) * s,
            ];

            let dx = p1[0] - p0[0];
            let dy = p1[1] - p0[1];
            let dz = p1[2] - p0[2];
            let dl = (dx * dx + dy * dy + dz * dz).sqrt().max(0.0001);

            (pos, [dx / dl, dy / dl, dz / dl])
        };

        // ---------- arrow heads only ----------

        let mut i = 1;
        let mut t = spacing;

        while t < total_len {
            let (p, dir) = sample_at(t);

            // build frame
            let up = if dir[1].abs() > 0.99 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };

            let mut side = [
                dir[1] * up[2] - dir[2] * up[1],
                dir[2] * up[0] - dir[0] * up[2],
                dir[0] * up[1] - dir[1] * up[0],
            ];

            let sl = (side[0] * side[0] + side[1] * side[1] + side[2] * side[2]).sqrt();
            side[0] /= sl;
            side[1] /= sl;
            side[2] /= sl;

            let up_perp = [
                dir[1] * side[2] - dir[2] * side[1],
                dir[2] * side[0] - dir[0] * side[2],
                dir[0] * side[1] - dir[1] * side[0],
            ];

            let angle = time * spin_speed + i as f32 * 1.7;
            let c = angle.cos();
            let s = angle.sin();

            let rx = side[0] * c + up_perp[0] * s;
            let ry = side[1] * c + up_perp[1] * s;
            let rz = side[2] * c + up_perp[2] * s;

            let bx = p[0] - dir[0] * head_len;
            let by = p[1] - dir[1] * head_len;
            let bz = p[2] - dir[2] * head_len;

            // left flap
            vertices.push(LineVtx {
                pos: p,
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

            // right flap
            vertices.push(LineVtx {
                pos: p,
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

            i += 1;
            t += spacing;
        }

        self.pending_renders.push(PendingGizmoRender { vertices });
    }
}
