use crate::data::Settings;
use crate::positions::{ChunkSize, LocalPos, WorldPos};
use crate::renderer::world_renderer::TerrainRenderer;
use crate::terrain::roads::roads::RoadManager;
use crate::ui::vertex::{LineVtxRender, LineVtxWorld};
use glam::Vec3;
use std::f32::consts::TAU;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue};

const CIRCLE_SEGMENT_COUNT: usize = 16;
pub const DEBUG_DRAW_DURATION: f32 = 10.0; // Seconds
pub struct PendingGizmoRender {
    pub vertices: Vec<LineVtxWorld>,
    pub duration: f32,
    pub start_time: f64,
}
pub struct Gizmo {
    pub pending_renders: Vec<PendingGizmoRender>,
    pub gizmo_buffer: Buffer,
    total_game_time: f64,
    pub(crate) chunk_size: ChunkSize,
}

impl Gizmo {
    pub fn new(device: &Device, chunk_size: ChunkSize) -> Self {
        let gizmo_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Gizmo VB"),
            size: (size_of::<LineVtxRender>() * 2048) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pending_renders: Vec::new(),
            gizmo_buffer,
            total_game_time: 0.0,
            chunk_size,
        }
    }

    pub fn clear(&mut self) {
        let now = self.total_game_time;
        self.pending_renders
            .retain(|g| now - g.start_time < g.duration as f64);
    }

    pub fn collect_vertices(&self, camera: WorldPos) -> Vec<LineVtxRender> {
        self.pending_renders
            .iter()
            .flat_map(|g| {
                g.vertices
                    .iter()
                    .map(|v| v.to_render(camera, self.chunk_size))
            })
            .collect()
    }

    pub fn update_buffer(&mut self, device: &Device, queue: &Queue, camera_pos: WorldPos) -> u32 {
        let vertices = self.collect_vertices(camera_pos);
        let byte_size = (vertices.len() * size_of::<LineVtxRender>()) as u64;

        // Grow buffer if needed
        if byte_size > self.gizmo_buffer.size() {
            let new_size = (self.gizmo_buffer.size() * 2).max(byte_size);
            self.gizmo_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("Gizmo VB"),
                size: new_size,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        queue.write_buffer(&self.gizmo_buffer, 0, bytemuck::cast_slice(&vertices));
        vertices.len() as u32
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal: push a gizmo render
    // ─────────────────────────────────────────────────────────────────────────

    #[inline]
    fn push(&mut self, vertices: Vec<LineVtxWorld>, duration: f32) {
        self.pending_renders.push(PendingGizmoRender {
            vertices,
            duration,
            start_time: self.total_game_time,
        });
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Basic primitives
    // ─────────────────────────────────────────────────────────────────────────

    pub fn line(&mut self, start: WorldPos, end: WorldPos, color: [f32; 3], duration: f32) {
        self.push(
            vec![
                LineVtxWorld::new(start, color),
                LineVtxWorld::new(end, color),
            ],
            duration,
        );
    }

    pub fn circle(&mut self, center: WorldPos, radius: f32, color: [f32; 3], duration: f32) {
        let cs = self.chunk_size;
        let verts: Vec<_> = (0..CIRCLE_SEGMENT_COUNT)
            .map(|i| {
                let angle = (i as f32 / CIRCLE_SEGMENT_COUNT as f32) * TAU;
                let offset = Vec3::new(radius * angle.cos(), 0.0, radius * angle.sin());
                LineVtxWorld::new(center.add_vec3(offset, cs), color)
            })
            .collect();
        self.push(verts, duration);
    }

    pub fn box_xz(&mut self, center: WorldPos, half_size: f32, color: [f32; 3], duration: f32) {
        let cs = self.chunk_size;
        let corners = [
            center.add_vec3(Vec3::new(-half_size, 0.0, -half_size), cs),
            center.add_vec3(Vec3::new(half_size, 0.0, -half_size), cs),
            center.add_vec3(Vec3::new(half_size, 0.0, half_size), cs),
            center.add_vec3(Vec3::new(-half_size, 0.0, half_size), cs),
        ];
        for i in 0..4 {
            self.line(corners[i], corners[(i + 1) % 4], color, duration);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Axes gizmo
    // ─────────────────────────────────────────────────────────────────────────

    pub fn axes(&mut self, origin: WorldPos, scale: f32, duration: f32) {
        let cs = self.chunk_size;
        let axes = [
            (Vec3::X, [1.0, 0.2, 0.2]),
            (Vec3::Y, [0.2, 1.0, 0.2]),
            (Vec3::Z, [0.2, 0.6, 1.0]),
        ];
        for (dir, color) in axes {
            self.line(origin, origin.add_vec3(dir * scale, cs), color, duration);
        }
    }

    pub fn axes_with_sun(&mut self, origin: WorldPos, scale: f32, sun_dir: Vec3, duration: f32) {
        self.axes(origin, scale, duration);
        let cs = self.chunk_size;
        let sun_end = origin.add_vec3(sun_dir.normalize_or_zero() * scale, cs);
        self.arrow(origin, sun_end, [1.0, 1.0, 0.0], false, duration);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Arrow
    // ─────────────────────────────────────────────────────────────────────────

    pub fn arrow(
        &mut self,
        start: WorldPos,
        end: WorldPos,
        color: [f32; 3],
        dashed: bool,
        duration: f32,
    ) {
        let cs = self.chunk_size;
        let delta = end.to_render_pos(start, cs);
        let len = delta.length();
        if len < 0.0001 {
            return;
        }

        let dir = delta / len;
        let (side, up_perp) = build_frame(dir);
        let flap = flap_color(color);

        let mut verts = Vec::new();

        // Parameters
        let target_spacing = 150.0;
        let steps = (len / target_spacing).ceil().max(1.0) as usize;
        let step_len = len / steps as f32;
        let dash_ratio = if dashed { 0.8 } else { 1.0 };
        let dash_len = step_len * dash_ratio;

        let head_len = 0.15;
        let head_width = 0.30;
        let spin_speed = 1.0;
        let time = self.total_game_time as f32;

        // Generate arrow heads from end backwards
        for i in 0..steps {
            let t_head = len - i as f32 * step_len;
            if t_head <= 0.0 {
                break;
            }

            let t0 = (t_head - dash_len).max(0.0);
            let p0 = start.add_vec3(dir * t0, cs);
            let p1 = start.add_vec3(dir * t_head, cs);

            // Shaft segment
            verts.push(LineVtxWorld::new(p0, color));
            verts.push(LineVtxWorld::new(p1, color));

            // Arrow head
            let angle = time * spin_speed + (i as f32 * 12.9898).sin() * 3.14;
            let rot = rotate_frame(side, up_perp, angle);
            let back = p1.add_vec3(-dir * head_len, cs);

            verts.push(LineVtxWorld::new(p1, flap));
            verts.push(LineVtxWorld::new(back.add_vec3(rot * head_width, cs), flap));
            verts.push(LineVtxWorld::new(p1, flap));
            verts.push(LineVtxWorld::new(
                back.add_vec3(-rot * head_width, cs),
                flap,
            ));
        }

        self.push(verts, duration);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Polyline (anchor + relative points for now, or full WorldPos slice)
    // ─────────────────────────────────────────────────────────────────────────

    /// Render polyline with arrows. Points are WorldPos.
    pub fn polyline(
        &mut self,
        points: &[WorldPos],
        color: [f32; 3],
        arrow_spacing: f32,
        duration: f32,
    ) {
        if points.len() < 2 {
            return;
        }

        let cs = self.chunk_size;
        let flap = flap_color(color);
        let mut verts = Vec::new();

        // Draw line segments
        for w in points.windows(2) {
            verts.push(LineVtxWorld::new(w[0], color));
            verts.push(LineVtxWorld::new(w[1], color));
        }

        // Compute cumulative lengths
        let mut lengths = vec![0.0f32];
        for w in points.windows(2) {
            let d = w[1].to_render_pos(w[0], cs).length();
            lengths.push(lengths.last().unwrap() + d);
        }
        let total_len = *lengths.last().unwrap();
        if total_len < 0.001 {
            self.push(verts, duration);
            return;
        }

        // Sample position and direction at distance t along polyline
        let sample_at = |t: f32| -> (WorldPos, Vec3) {
            let mut i = 1;
            while i < lengths.len() && lengths[i] < t {
                i += 1;
            }
            let i0 = i - 1;
            let i1 = i.min(points.len() - 1);
            let seg_t = if lengths[i1] > lengths[i0] {
                (t - lengths[i0]) / (lengths[i1] - lengths[i0])
            } else {
                0.0
            };

            let pos = points[i0].lerp(points[i1], seg_t, cs);
            let dir = points[i1].to_render_pos(points[i0], cs).normalize_or_zero();
            (pos, dir)
        };

        // Arrow parameters
        let head_len = 0.30;
        let head_width = 0.25;
        let spin_speed = 1.0;
        let time = self.total_game_time as f32;

        // Place arrows along polyline
        let mut t = arrow_spacing;
        let mut idx = 0;
        while t < total_len {
            let (pos, dir) = sample_at(t);
            if dir.length_squared() < 0.0001 {
                t += arrow_spacing;
                continue;
            }

            let (side, up_perp) = build_frame(dir);
            let angle = time * spin_speed + idx as f32 * 1.7;
            let rot = rotate_frame(side, up_perp, angle);
            let back = pos.add_vec3(-dir * head_len, cs);

            verts.push(LineVtxWorld::new(pos, flap));
            verts.push(LineVtxWorld::new(back.add_vec3(rot * head_width, cs), flap));
            verts.push(LineVtxWorld::new(pos, flap));
            verts.push(LineVtxWorld::new(
                back.add_vec3(-rot * head_width, cs),
                flap,
            ));

            idx += 1;
            t += arrow_spacing;
        }

        self.push(verts, duration);
    }

    /// Polyline from an anchor WorldPos and relative Vec3 offsets.
    /// Useful when you have data in local/relative coordinates.
    pub fn polyline_relative(
        &mut self,
        anchor: WorldPos,
        offsets: &[Vec3],
        color: [f32; 3],
        arrow_spacing: f32,
        duration: f32,
    ) {
        let cs = self.chunk_size;
        let points: Vec<_> = offsets.iter().map(|&o| anchor.add_vec3(o, cs)).collect();
        self.polyline(&points, color, arrow_spacing, duration);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Digit rendering
    // ─────────────────────────────────────────────────────────────────────────

    pub fn digit(&mut self, digit: u8, pos: WorldPos, scale: f32, color: [f32; 3], duration: f32) {
        const SEGMENTS: [&[([f32; 2], [f32; 2])]; 10] = [
            &[
                ([0., 0.], [1., 0.]),
                ([1., 0.], [1., 2.]),
                ([1., 2.], [0., 2.]),
                ([0., 2.], [0., 0.]),
            ], // 0
            &[([1., 0.], [1., 2.])], // 1
            &[
                ([0., 0.], [1., 0.]),
                ([1., 0.], [1., 1.]),
                ([1., 1.], [0., 1.]),
                ([0., 1.], [0., 2.]),
                ([0., 2.], [1., 2.]),
            ], // 2
            &[
                ([0., 0.], [1., 0.]),
                ([1., 0.], [1., 2.]),
                ([1., 2.], [0., 2.]),
                ([0., 1.], [1., 1.]),
            ], // 3
            &[
                ([0., 0.], [0., 1.]),
                ([0., 1.], [1., 1.]),
                ([1., 0.], [1., 2.]),
            ], // 4
            &[
                ([0., 0.], [1., 0.]),
                ([0., 0.], [0., 1.]),
                ([0., 1.], [1., 1.]),
                ([1., 1.], [1., 2.]),
                ([1., 2.], [0., 2.]),
            ], // 5
            &[
                ([1., 0.], [0., 0.]),
                ([0., 0.], [0., 2.]),
                ([0., 2.], [1., 2.]),
                ([1., 2.], [1., 1.]),
                ([1., 1.], [0., 1.]),
            ], // 6
            &[([0., 0.], [1., 0.]), ([1., 0.], [1., 2.])], // 7
            &[
                ([0., 0.], [1., 0.]),
                ([1., 0.], [1., 2.]),
                ([1., 2.], [0., 2.]),
                ([0., 2.], [0., 0.]),
                ([0., 1.], [1., 1.]),
            ], // 8
            &[
                ([0., 0.], [1., 0.]),
                ([1., 0.], [1., 2.]),
                ([1., 2.], [0., 2.]),
                ([0., 1.], [0., 0.]),
                ([0., 1.], [1., 1.]),
            ], // 9
        ];

        let cs = self.chunk_size;
        let segs = SEGMENTS.get(digit as usize).copied().unwrap_or(&[]);

        let verts: Vec<_> = segs
            .iter()
            .flat_map(|&([x0, z0], [x1, z1])| {
                let p0 = pos.add_vec3(Vec3::new(x0 * scale, 0.0, z0 * scale), cs);
                let p1 = pos.add_vec3(Vec3::new(x1 * scale, 0.0, z1 * scale), cs);
                [LineVtxWorld::new(p0, color), LineVtxWorld::new(p1, color)]
            })
            .collect();

        self.push(verts, duration);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Update (road visualization example)
    // ─────────────────────────────────────────────────────────────────────────

    pub fn update(
        &mut self,
        terrain_renderer: &TerrainRenderer,
        target: WorldPos,
        total_game_time: f64,
        road_manager: &RoadManager,
        settings: &Settings,
    ) {
        self.total_game_time = total_game_time;

        if !settings.render_lanes_gizmo {
            return;
        }

        let cs = self.chunk_size;

        // Optional: render chunk bounds
        if settings.render_chunk_bounds {
            if terrain_renderer.chunks.contains_key(&target.chunk) {
                let chunk_size_f = terrain_renderer.chunk_size as f32;
                let corner = WorldPos::new(target.chunk, LocalPos::new(0.0, target.local.y, 0.0));
                self.box_xz(
                    corner.add_vec3(Vec3::new(chunk_size_f * 0.5, 0.0, chunk_size_f * 0.5), cs),
                    chunk_size_f * 0.5,
                    [0.2, 0.8, 0.6],
                    0.0,
                );
            }
        }

        // Road visualization
        let render_disabled = false;
        let render_lane_arrows = false;

        for storage in [&road_manager.roads, &road_manager.preview_roads] {
            for (_node_id, node) in storage.iter_nodes() {
                // Node circle
                let node_pos = node.position();
                let node_color = if node.is_enabled() {
                    [0.0, 0.0, 0.9]
                } else {
                    [1.0, 0.0, 0.0]
                };
                self.circle(node_pos, 2.0, node_color, 0.0);

                // Incoming lanes
                for lane_id in node.incoming_lanes() {
                    let lane = storage.lane(lane_id);
                    if !lane.is_enabled() && !render_disabled {
                        continue;
                    }

                    let segment = storage.segment(lane.segment());
                    let is_forward = lane.from_node() == segment.start();

                    let color = if lane.is_enabled() {
                        if is_forward {
                            [0.0, 0.9, 0.0]
                        } else {
                            [0.2, 0.9, 0.0]
                        }
                    } else {
                        [1.0, 0.05, 0.0]
                    };

                    // Convert polyline to WorldPos
                    let points: &Vec<WorldPos> = lane.polyline();

                    self.polyline(&points, color, 15.0, 0.0);

                    if render_lane_arrows {
                        if let Some(last) = points.last() {
                            self.arrow(*last, node_pos, color, false, 0.0);
                        }
                    }
                }

                // Node lanes
                for node_lane in node.node_lanes() {
                    if !node_lane.is_enabled() && !render_disabled {
                        continue;
                    }

                    let color = if node_lane.is_enabled() {
                        [0.7, 0.5, 0.0]
                    } else {
                        [1.0, 0.05, 0.0]
                    };

                    let points: &Vec<WorldPos> = node_lane.polyline();

                    self.polyline(&points, color, 4.0, 0.0);

                    if render_lane_arrows {
                        if let Some(last) = points.last() {
                            self.arrow(*last, node_pos, color, false, 0.0);
                        }
                    }
                }
            }
        }
    }

    /// Render a whole number centered at position.
    pub fn number(
        &mut self,
        mut value: usize,
        center: WorldPos,
        scale: f32,
        color: [f32; 3],
        duration: f32,
    ) {
        let digits: Vec<u8> = if value == 0 {
            vec![0]
        } else {
            let mut ds = Vec::new();
            while value > 0 {
                ds.push((value % 10) as u8);
                value /= 10;
            }
            ds.reverse();
            ds
        };

        let cs = self.chunk_size;
        let spacing = scale * 1.2;
        let total_width = (digits.len() as f32 - 1.0) * spacing;
        let start_offset = -total_width * 0.5;

        for (i, &d) in digits.iter().enumerate() {
            let offset = Vec3::new(start_offset + i as f32 * spacing, 0.0, 0.0);
            self.digit(d, center.add_vec3(offset, cs), scale, color, duration);
        }
    }

    /// Draw a cross marker at position.
    pub fn cross(&mut self, pos: WorldPos, size: f32, color: [f32; 3], duration: f32) {
        let cs = self.chunk_size;
        let half = size * 0.5;
        self.line(
            pos.add_vec3(Vec3::new(-half, 0.0, 0.0), cs),
            pos.add_vec3(Vec3::new(half, 0.0, 0.0), cs),
            color,
            duration,
        );
        self.line(
            pos.add_vec3(Vec3::new(0.0, 0.0, -half), cs),
            pos.add_vec3(Vec3::new(0.0, 0.0, half), cs),
            color,
            duration,
        );
    }
    pub fn update_gizmo_vertices(
        &mut self,
        target: WorldPos,
        orbit_radius: f32,
        scale_with_orbit: bool,
        sun_direction: Vec3,
        chunk_size: ChunkSize,
    ) {
        let s = if scale_with_orbit {
            orbit_radius * 0.2
        } else {
            1.0
        };

        let axes = [
            LineVtxWorld {
                pos: target,
                color: [1.0, 0.2, 0.2],
            },
            LineVtxWorld {
                pos: target.add_vec3(Vec3::X * s, chunk_size),
                color: [1.0, 0.2, 0.2],
            },
            LineVtxWorld {
                pos: target,
                color: [0.2, 1.0, 0.2],
            },
            LineVtxWorld {
                pos: target.add_vec3(Vec3::Y * s, chunk_size),
                color: [0.2, 1.0, 0.2],
            },
            LineVtxWorld {
                pos: target,
                color: [0.2, 0.6, 1.0],
            },
            LineVtxWorld {
                pos: target.add_vec3(Vec3::Z * s, chunk_size),
                color: [0.2, 0.6, 1.0],
            },
        ];

        self.pending_renders.push(PendingGizmoRender {
            vertices: axes.to_vec(),
            duration: 0.0,
            start_time: self.total_game_time,
        });

        let arrow_length = s;
        let sun_end = target.add_vec3(sun_direction.normalize_or_zero() * arrow_length, chunk_size);

        // Make render_arrow take WorldPos endpoints too (recommended)
        self.arrow(target, sun_end, [1.0, 1.0, 0.0], false, 0.0);
    }
}
#[inline]
fn flap_color(c: [f32; 3]) -> [f32; 3] {
    [(c[0] + 1.0) * 0.5, (c[1] + 1.0) * 0.5, (c[2] + 1.0) * 0.5]
}

/// Build orthonormal frame from direction vector
#[inline]
fn build_frame(dir: Vec3) -> (Vec3, Vec3) {
    let up = if dir.y.abs() > 0.99 { Vec3::X } else { Vec3::Y };
    let side = dir.cross(up).normalize();
    let up_perp = dir.cross(side);
    (side, up_perp)
}

/// Rotate vector in the side/up_perp plane
#[inline]
fn rotate_frame(side: Vec3, up_perp: Vec3, angle: f32) -> Vec3 {
    side * angle.cos() + up_perp * angle.sin()
}

impl LineVtxWorld {
    #[inline]
    pub fn new(pos: WorldPos, color: [f32; 3]) -> Self {
        Self { pos, color }
    }

    #[inline]
    pub fn to_render(&self, camera_pos: WorldPos, chunk_size: ChunkSize) -> LineVtxRender {
        let rp = self.pos.to_render_pos(camera_pos, chunk_size);
        LineVtxRender {
            pos: rp.to_array(),
            color: self.color,
        }
    }
}
