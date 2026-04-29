#![allow(dead_code)]

use crate::data::Settings;
use crate::helpers::hsv::{HSV, depth_to_color, hsv_to_rgb};
use crate::helpers::positions::{ChunkCoord, ChunkSize, LocalPos, WorldPos};
use crate::renderer::gizmo::partition_gizmo::{PartitionGizmo, PartitionVisualizationConfig};
use crate::renderer::pipelines::Pipelines;
use crate::renderer::ray_tracing::rt_subsystem::RTSubsystem;
use crate::renderer::ray_tracing::structs::{Aabb, Blas, BvhNode, Tlas};
use crate::ui::ui_editor::Ui;
use crate::ui::vertex::{LineVtxRender, LineVtxWorld, TextVtxRender};
use crate::world::buildings::zoning::point_in_polygon_xz;
use crate::world::camera::Camera;
use crate::world::cars::partitions::PartitionManager;
use crate::world::roads::road_structs::NodeId;
use crate::world::roads::roads::{RoadManager, RoadStorage};
use crate::world::terrain::terrain_subsystem::Terrain;
use glam::Vec3;
use std::cell::{Cell, RefCell};
use std::f32::consts::{PI, TAU};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue, SurfaceConfiguration};
use wgpu_text::glyph_brush::ab_glyph::{FontArc, PxScale, Rect};
use wgpu_text::glyph_brush::{Extra, GlyphBrush, GlyphBrushBuilder, OwnedSection, Section, Text};

const CIRCLE_SEGMENT_COUNT: usize = 16;
pub const DEBUG_DRAW_DURATION: f32 = 20.0; // Seconds
pub const ROAD_GIZMO_THICKNESS: f32 = 0.0; // M!
pub struct PendingGizmoRender {
    pub vertices: Vec<LineVtxWorld>,
    pub text: Option<PendingGizmoTextRender>,
    pub thickness: f32,
    pub duration: f32,
    pub start_time: f64,
    pub filled: bool,
}
pub struct PendingGizmoTextRender {
    pub section: OwnedSection,
    pub center: WorldPos,
    pub scale: f32,
    pub color: [f32; 4],
    pub vertices: Vec<TextVertex3D>,
    pub facing: Option<Vec3>,
}
pub struct Gizmo {
    partition_gizmo: PartitionGizmo,
    pub pending_renders: Vec<PendingGizmoRender>,
    pub gizmo_buffer: Buffer,
    pub thick_buffer: Buffer,
    pub text_buffer: Buffer,
    total_game_time: f64,
    pub chunk_size: ChunkSize,
    pub brush: GlyphBrush<(), Extra>,
    pub text_raster_factor: f32, // good start: 48.0
    pub text_raster_min: f32,    // good start: 8.0
    pub text_raster_max: f32,    // good start: 256.0
}
#[derive(Default)]
pub struct GizmoBatches {
    pub thin_vertices: Vec<LineVtxRender>,
    pub thick_vertices: Vec<LineVtxRender>,
    pub text_vertices: Vec<TextVtxRender>,
}
impl Gizmo {
    pub fn new(
        device: &Device,
        chunk_size: ChunkSize,
        config: &SurfaceConfiguration,
        font_arc: &FontArc,
        msaa_samples: u32,
    ) -> Self {
        let gizmo_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Gizmo VB"),
            size: (size_of::<LineVtxRender>() * 2048) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let thick_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Gizmo Custom Thickness VB"),
            size: (size_of::<LineVtxRender>() * 2048) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let text_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Gizmo Text VB"),
            size: (size_of::<TextVtxRender>() * 2048) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let brush: GlyphBrush<(), Extra> = GlyphBrushBuilder::using_font(font_arc.clone())
            .initial_cache_size((2048, 2048))
            .build();
        Self {
            partition_gizmo: PartitionGizmo::new(),
            pending_renders: Vec::new(),
            gizmo_buffer,
            thick_buffer,
            text_buffer,
            total_game_time: 0.0,
            chunk_size,
            brush,
            text_raster_factor: 1512.0,
            text_raster_min: 8.0,
            text_raster_max: 128.0,
        }
    }
    pub fn clear(&mut self) {
        let now = self.total_game_time;
        self.pending_renders
            .retain(|g| now - g.start_time < g.duration as f64);
    }

    pub fn update_buffers(
        &mut self,
        device: &Device,
        queue: &Queue,
        batches: &GizmoBatches,
    ) -> (u32, u32, u32) {
        let thin_count = batches.thin_vertices.len() as u32;
        let thick_count = batches.thick_vertices.len() as u32;
        let text_count = batches.text_vertices.len() as u32;
        // Update thin line buffer
        if thin_count > 0 {
            let byte_size = (batches.thin_vertices.len() * size_of::<LineVtxRender>()) as u64;
            if byte_size > self.gizmo_buffer.size() {
                let new_size = (self.gizmo_buffer.size() * 2).max(byte_size);
                self.gizmo_buffer = device.create_buffer(&BufferDescriptor {
                    label: Some("Gizmo Thin Lines VB"),
                    size: new_size,
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            queue.write_buffer(
                &self.gizmo_buffer,
                0,
                bytemuck::cast_slice(&batches.thin_vertices),
            );
        }

        // Update thick geometry buffer
        if thick_count > 0 {
            let byte_size = (batches.thick_vertices.len() * size_of::<LineVtxRender>()) as u64;
            if byte_size > self.thick_buffer.size() {
                let new_size = (self.thick_buffer.size() * 2).max(byte_size);
                self.thick_buffer = device.create_buffer(&BufferDescriptor {
                    label: Some("Gizmo Thick Geometry VB"),
                    size: new_size,
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            queue.write_buffer(
                &self.thick_buffer,
                0,
                bytemuck::cast_slice(&batches.thick_vertices),
            );
        }

        // Update text geometry buffer
        if text_count > 0 {
            let byte_size = (batches.text_vertices.len() * size_of::<TextVtxRender>()) as u64;
            if byte_size > self.text_buffer.size() {
                let new_size = (self.text_buffer.size() * 2).max(byte_size);
                self.text_buffer = device.create_buffer(&BufferDescriptor {
                    label: Some("Gizmo Text Geometry VB"),
                    size: new_size,
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            queue.write_buffer(
                &self.text_buffer,
                0,
                bytemuck::cast_slice(&batches.text_vertices),
            );
        }

        (thin_count, thick_count, text_count)
    }

    pub fn visualize_partitions(
        &mut self,
        partition_gizmo: &mut PartitionGizmo,
        manager: &PartitionManager,
        road_storage: &RoadStorage,
    ) {
        let config = PartitionVisualizationConfig::detailed();

        partition_gizmo.visualize(self, manager, road_storage, config)
    }

    /// Visualizes all road regions with distinct colors and region ID numbers.
    ///
    /// Each region is displayed with:
    /// - A unique color based on its ID using golden ratio hue distribution
    /// - Circles around each node belonging to the region
    /// - Cross markers at node positions for clarity
    /// - A number label showing the region ID at the region centroid
    /// - A small white circle marking the centroid position
    pub fn visualize_regions(&mut self, road_storage: &RoadStorage, thickness: f32, duration: f32) {
        let cs = self.chunk_size;

        for (region_id, region) in road_storage.iter_active_regions() {
            let hue = (region_id as f32 * 0.618033988749895) % 1.0;
            let color = hsv_to_rgb(HSV {
                h: hue,
                s: 0.8,
                v: 0.9,
            });
            let color = [color[0], color[1], color[2], 1.0];
            let secondary_color = hsv_to_rgb(HSV {
                h: hue,
                s: 0.5,
                v: 0.7,
            });
            let secondary_color = [
                secondary_color[0],
                secondary_color[1],
                secondary_color[2],
                1.0,
            ];

            let node_indices = region.node_indices();
            if node_indices.is_empty() {
                continue;
            }

            let mut positions: Vec<WorldPos> = Vec::new();

            for &node_idx in node_indices {
                if let Some(node) = road_storage.node(NodeId::new(node_idx)) {
                    let pos = node.position();
                    positions.push(pos);
                    self.circle(pos, 4.0, color, thickness, duration);
                    self.cross(pos, 2.0, secondary_color, thickness, duration);
                }
            }

            if positions.is_empty() {
                continue;
            }

            let first_pos = positions[0];
            let centroid = if positions.len() == 1 {
                first_pos
            } else {
                let mut offset_sum = Vec3::ZERO;
                for pos in &positions {
                    offset_sum += pos.to_render_pos(first_pos, cs);
                }
                offset_sum /= positions.len() as f32;
                first_pos.add_vec3(offset_sum, cs)
            };

            let label_pos = centroid.add_vec3(Vec3::new(0.0, 5.0, 0.0), cs);
            self.text(
                region_id.to_string().as_str(),
                label_pos,
                4.0,
                color,
                None,
                thickness,
                duration,
            );
            self.circle(centroid, 2.0, [1.0, 1.0, 1.0, 1.0], thickness, duration);
        }
    }

    // Internal: push a gizmo render
    #[inline]
    fn push(&mut self, vertices: Vec<LineVtxWorld>, thickness: f32, duration: f32, filled: bool) {
        self.pending_renders.push(PendingGizmoRender {
            vertices,
            text: None,
            thickness,
            duration,
            filled,
            start_time: self.total_game_time,
        });
    }
    #[inline]
    fn push_text(&mut self, text: PendingGizmoTextRender, thickness: f32, duration: f32) {
        self.pending_renders.push(PendingGizmoRender {
            vertices: Vec::new(),
            text: Some(text),
            thickness,
            duration,
            filled: false,
            start_time: self.total_game_time,
        });
    }
    // Basic primitives
    pub fn line(
        &mut self,
        start: WorldPos,
        end: WorldPos,
        color: [f32; 4],
        thickness: f32,
        duration: f32,
    ) {
        self.push(
            vec![
                LineVtxWorld::new(start, color),
                LineVtxWorld::new(end, color),
            ],
            thickness,
            duration,
            false,
        );
    }

    pub fn circle(
        &mut self,
        center: WorldPos,
        radius: f32,
        color: [f32; 4],
        thickness: f32,
        duration: f32,
    ) {
        let cs = self.chunk_size;

        let mut verts = Vec::with_capacity(CIRCLE_SEGMENT_COUNT * 2);

        for i in 0..CIRCLE_SEGMENT_COUNT {
            let a0 = (i as f32 / CIRCLE_SEGMENT_COUNT as f32) * TAU;
            let a1 = ((i + 1) as f32 / CIRCLE_SEGMENT_COUNT as f32) * TAU;

            let p0 = center.add_vec3(Vec3::new(radius * a0.cos(), 0.0, radius * a0.sin()), cs);
            let p1 = center.add_vec3(Vec3::new(radius * a1.cos(), 0.0, radius * a1.sin()), cs);

            verts.push(LineVtxWorld::new(p0, color));
            verts.push(LineVtxWorld::new(p1, color));
        }

        self.push(verts, thickness, duration, false);
    }

    pub fn sphere(
        &mut self,
        center: WorldPos,
        radius: f32,
        color: [f32; 4],
        thickness: f32,
        duration: f32,
    ) {
        let cs = self.chunk_size;
        let mut verts = Vec::new();
        let rings = CIRCLE_SEGMENT_COUNT;
        for j in 1..rings {
            let phi = (j as f32 / rings as f32) * PI;
            let y = radius * phi.cos();
            let r = radius * phi.sin();
            for i in 0..CIRCLE_SEGMENT_COUNT {
                let a0 = (i as f32 / CIRCLE_SEGMENT_COUNT as f32) * TAU;
                let a1 = ((i + 1) as f32 / CIRCLE_SEGMENT_COUNT as f32) * TAU;
                verts.push(LineVtxWorld::new(
                    center.add_vec3(Vec3::new(r * a0.cos(), y, r * a0.sin()), cs),
                    color,
                ));
                verts.push(LineVtxWorld::new(
                    center.add_vec3(Vec3::new(r * a1.cos(), y, r * a1.sin()), cs),
                    color,
                ));
            }
        }
        for j in 0..rings {
            let theta = (j as f32 / rings as f32) * PI;
            let (ct, st) = (theta.cos(), theta.sin());
            for i in 0..CIRCLE_SEGMENT_COUNT {
                let phi0 = (i as f32 / CIRCLE_SEGMENT_COUNT as f32) * TAU;
                let phi1 = ((i + 1) as f32 / CIRCLE_SEGMENT_COUNT as f32) * TAU;
                verts.push(LineVtxWorld::new(
                    center.add_vec3(
                        Vec3::new(
                            radius * phi0.sin() * ct,
                            radius * phi0.cos(),
                            radius * phi0.sin() * st,
                        ),
                        cs,
                    ),
                    color,
                ));
                verts.push(LineVtxWorld::new(
                    center.add_vec3(
                        Vec3::new(
                            radius * phi1.sin() * ct,
                            radius * phi1.cos(),
                            radius * phi1.sin() * st,
                        ),
                        cs,
                    ),
                    color,
                ));
            }
        }
        self.push(verts, thickness, duration, false);
    }

    pub fn box_xz(
        &mut self,
        center: WorldPos,
        half_size: f32,
        color: [f32; 4],
        thickness: f32,
        duration: f32,
    ) {
        let cs = self.chunk_size;
        let corners = [
            center.add_vec3(Vec3::new(-half_size, 0.0, -half_size), cs),
            center.add_vec3(Vec3::new(half_size, 0.0, -half_size), cs),
            center.add_vec3(Vec3::new(half_size, 0.0, half_size), cs),
            center.add_vec3(Vec3::new(-half_size, 0.0, half_size), cs),
        ];
        for i in 0..4 {
            self.line(corners[i], corners[(i + 1) % 4], color, thickness, duration);
        }
    }

    pub fn direction(
        &mut self,
        center: WorldPos,
        direction: Vec3,
        color: [f32; 4],
        thickness: f32,
        duration: f32,
    ) {
        self.arrow(
            center,
            center.add_vec3(direction, self.chunk_size),
            color,
            false,
            thickness,
            duration,
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Axes gizmo
    // ─────────────────────────────────────────────────────────────────────────

    pub fn axes(&mut self, origin: WorldPos, scale: f32, thickness: f32, duration: f32) {
        let cs = self.chunk_size;
        let axes = [
            (Vec3::X, [1.0, 0.2, 0.2, 1.0]),
            (Vec3::Y, [0.2, 1.0, 0.2, 1.0]),
            (Vec3::Z, [0.2, 0.6, 1.0, 1.0]),
        ];
        for (dir, color) in axes {
            self.line(
                origin,
                origin.add_vec3(dir * scale, cs),
                color,
                thickness,
                duration,
            );
        }
    }

    pub fn axes_with_sun(
        &mut self,
        origin: WorldPos,
        scale: f32,
        sun_dir: Vec3,
        moon_dir: Vec3,
        thickness: f32,
        duration: f32,
    ) {
        self.axes(origin, scale, thickness, duration);
        let cs = self.chunk_size;
        let sun_end = origin.add_vec3(sun_dir.normalize_or_zero() * scale, cs);
        self.arrow(
            origin,
            sun_end,
            [1.0, 1.0, 0.0, 1.0],
            false,
            thickness,
            duration,
        );
        let moon_end = origin.add_vec3(moon_dir.normalize_or_zero() * scale, cs);
        self.arrow(
            origin,
            moon_end,
            [1.0, 1.0, 1.0, 1.0],
            false,
            thickness,
            duration,
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Arrow
    // ─────────────────────────────────────────────────────────────────────────

    pub fn arrow(
        &mut self,
        start: WorldPos,
        end: WorldPos,
        color: [f32; 4],
        dashed: bool,
        thickness: f32,
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
            let angle = time * spin_speed + (i as f32 * 12.9898).sin() * PI;
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

        self.push(verts, thickness, duration, false);
    }

    // Polyline (anchor + relative points for now, or full WorldPos slice)
    /// Render polyline with arrows. Points are WorldPos.
    pub fn polyline(
        &mut self,
        points: &[WorldPos],
        color: [f32; 4],
        arrow_spacing: f32,
        thickness: f32,
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
            self.push(verts, thickness, duration, false);
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

            let pos = points[i0].lerp(points[i1], seg_t as f64, cs);
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
        if t == 0.0 {
            self.push(verts, thickness, duration, false);
            return;
        }
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

        self.push(verts, thickness, duration, false);
    }

    /// Polyline from an anchor WorldPos and relative Vec3 offsets.
    /// Useful when you have data in local/relative coordinates.
    pub fn polyline_relative(
        &mut self,
        anchor: WorldPos,
        offsets: &[Vec3],
        color: [f32; 4],
        arrow_spacing: f32,
        thickness: f32,
        duration: f32,
    ) {
        let cs = self.chunk_size;
        let points: Vec<_> = offsets.iter().map(|&o| anchor.add_vec3(o, cs)).collect();
        self.polyline(&points, color, arrow_spacing, thickness, duration);
    }

    /// Render area. Points are WorldPos.
    pub fn area(&mut self, points: &[WorldPos], color: [f32; 4], duration: f32) {
        if points.len() < 3 {
            return;
        }

        let verts: Vec<LineVtxWorld> = points
            .iter()
            .map(|&p| LineVtxWorld::new(p, color))
            .collect();

        self.push(verts, duration, 0.0, true);
    }

    pub fn area_textured(&mut self, points: &[WorldPos], color: [f32; 4], duration: f32) {
        self.area(points, color, duration);

        if points.len() < 3 {
            return;
        }

        let color = [
            color[0] * 1.1,
            color[1] * 1.1,
            color[2] * 1.1,
            color[3] * 1.1,
        ];
        let cs = self.chunk_size;
        let origin = points[0];

        // Precompute render positions ONCE
        let renders: Vec<Vec3> = points
            .iter()
            .map(|&p| p.to_render_pos(origin, cs))
            .collect();

        // AABB in render space
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_z = f32::INFINITY;
        let mut max_z = f32::NEG_INFINITY;

        for rp in &renders {
            min_x = min_x.min(rp.x);
            max_x = max_x.max(rp.x);
            min_z = min_z.min(rp.z);
            max_z = max_z.max(rp.z);
        }

        let spacing = cs as f32 * 0.025;

        // height sampling via triangle fan
        let sample_y = |x: f32, z: f32| -> f32 {
            let p = Vec3::new(x, 0.0, z);

            for i in 1..renders.len() - 1 {
                if let Some(y) = barycentric_y(p, renders[0], renders[i], renders[i + 1]) {
                    return y;
                }
            }

            // fallback (should rarely happen if inside polygon)
            renders[0].y
        };

        let mut z = min_z;
        let mut row = 0;

        while z <= max_z {
            let mut x = min_x;

            // offset every second row (less grid look)
            if row % 2 == 1 {
                x += spacing * 0.678;
            }

            while x <= max_x {
                let y = sample_y(x, z);
                let p = origin.add_vec3(Vec3::new(x, y, z), cs);

                if point_in_polygon_xz(p, points, cs) {
                    self.cross(p, spacing * 0.1, color, 0.0, duration);
                }

                x += spacing;
            }

            z += spacing;
            row += 1;
        }
    }

    pub fn text<S>(
        &mut self,
        text: S,
        center: WorldPos,
        scale: f32,
        color: [f32; 4],
        facing: Option<Vec3>,
        thickness: f32,
        duration: f32,
    ) where
        S: Into<String>,
    {
        let text = text.into();
        // Text scale is scaled dynamically in the to render conversion anyway.
        let section = Section::default()
            .with_text(vec![Text::new(text.as_str()).with_color(color)])
            .to_owned();
        let text = PendingGizmoTextRender {
            section,
            center,
            scale,
            color,
            facing,
            vertices: vec![],
        };
        self.push_text(text, thickness, duration);
    }

    pub fn update(
        &mut self,
        terrain_subsystem: &Terrain,
        rt_subsystem: &RTSubsystem,
        total_game_time: f64,
        road_manager: &RoadManager,
        partition_gizmo: &mut PartitionGizmo,
        partition_manager: &PartitionManager,
        settings: &Settings,
        camera: &Camera,
    ) {
        self.total_game_time = total_game_time;
        self.chunk_size = camera.chunk_size;
        let target = camera.target;

        if settings.render_partitions_gizmo {
            self.visualize_partitions(partition_gizmo, partition_manager, &road_manager.roads);
            self.visualize_regions(&road_manager.roads, 0.0, 0.0);
        }

        // self.sphere(camera.eye_world(), 400.0, [1.0, 1.0, 1.0], 0.0);
        if settings.render_rt_gizmo {
            self.visualize_rt(
                rt_subsystem,
                camera.eye_world(), // reference position
                false,              // show TLAS instances
                true,               // show TLAS BVH
                8,                  // TLAS BVH max depth to show
                false,              // show BLAS (usually false - it's object-space)
                8,                  // BLAS BVH max depth
                0.0,
                0.0, // duration (0 = single frame)
            );
        }
        if settings.render_chunk_bounds {
            if terrain_subsystem.chunks.contains_key(&target.chunk) {
                let chunk_size_f = self.chunk_size as f32;
                let corner = WorldPos::new(
                    target.chunk,
                    LocalPos::new(0.0, terrain_subsystem.get_height_at(target, false), 0.0),
                );
                self.box_xz(
                    corner.add_vec3(
                        Vec3::new(chunk_size_f * 0.5, 0.0, chunk_size_f * 0.5),
                        self.chunk_size,
                    ),
                    chunk_size_f * 0.5,
                    [0.2, 0.8, 0.6, 0.9],
                    0.0,
                    0.0,
                );
            }
        }
        if !settings.render_lanes_gizmo {
            return;
        }
        // Road visualization
        let render_disabled = false;
        let render_lane_arrows = false;

        for storage in [&road_manager.roads, &road_manager.preview_roads] {
            for (_node_id, node) in storage.iter_nodes() {
                // Node circle
                let node_pos = node.position();
                let node_color = if node.is_enabled() {
                    [0.0, 0.0, 0.9, 1.0]
                } else {
                    [1.0, 0.0, 0.0, 1.0]
                };
                self.circle(node_pos, 2.0, node_color, 0.0, 0.0);

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
                            [0.0, 0.9, 0.0, 1.0]
                        } else {
                            [0.2, 0.9, 0.0, 1.0]
                        }
                    } else {
                        [1.0, 0.05, 0.0, 1.0]
                    };

                    // Convert polyline to WorldPos
                    let points: &Vec<WorldPos> = lane.polyline();

                    self.polyline(&points, color, 15.0, 0.0, 0.0);

                    if render_lane_arrows {
                        if let Some(last) = points.last() {
                            self.arrow(*last, node_pos, color, false, 0.0, 0.0);
                        }
                    }
                }

                // Node lanes
                for node_lane in node.node_lanes() {
                    if !node_lane.is_enabled() && !render_disabled {
                        continue;
                    }

                    let color = if node_lane.is_enabled() {
                        [0.7, 0.5, 0.0, 1.0]
                    } else {
                        [1.0, 0.05, 0.0, 1.0]
                    };

                    let points: &Vec<WorldPos> = node_lane.polyline();

                    self.polyline(&points, color, 4.0, 0.0, 0.0);

                    if render_lane_arrows {
                        if let Some(last) = points.last() {
                            self.arrow(*last, node_pos, color, false, 0.0, 0.0);
                        }
                    }
                }
            }
        }
    }

    /// Draw a cross marker at position.
    pub fn cross(
        &mut self,
        pos: WorldPos,
        size: f32,
        color: [f32; 4],
        thickness: f32,
        duration: f32,
    ) {
        let cs = self.chunk_size;
        let half = size * 0.5;
        self.line(
            pos.add_vec3(Vec3::new(-half, 0.0, 0.0), cs),
            pos.add_vec3(Vec3::new(half, 0.0, 0.0), cs),
            color,
            thickness,
            duration,
        );
        self.line(
            pos.add_vec3(Vec3::new(0.0, 0.0, -half), cs),
            pos.add_vec3(Vec3::new(0.0, 0.0, half), cs),
            color,
            thickness,
            duration,
        );
    }
    pub fn update_orbit_gizmo(
        &mut self,
        ui: &mut Ui,
        target: WorldPos,
        orbit_radius: f32,
        sun_direction: Vec3,
        moon_direction: Vec3,
        scale_with_orbit: bool,
    ) {
        let debug_menu_active = ui.menus.get("Debug_Menu").unwrap().active;
        ui.variables.set_bool("debug_mode", debug_menu_active);
        if debug_menu_active {
            let scale = scale_with_orbit
                .then_some(orbit_radius * 0.1)
                .unwrap_or(1.0);
            self.axes_with_sun(target, scale, sun_direction, moon_direction, 0.0, 0.0);
        }
    }

    /// Draw a 3D wireframe axis-aligned bounding box
    pub fn aabb(
        &mut self,
        aabb: &Aabb,
        reference: WorldPos,
        color: [f32; 4],
        thickness: f32,
        duration: f32,
    ) {
        if !aabb.is_valid() {
            return;
        }

        let cs = self.chunk_size;
        let min = Vec3::new(aabb.min[0], aabb.min[1], aabb.min[2]);
        let max = Vec3::new(aabb.max[0], aabb.max[1], aabb.max[2]);

        // 8 corners of the box
        let c = [
            reference.add_vec3(Vec3::new(min.x, min.y, min.z), cs),
            reference.add_vec3(Vec3::new(max.x, min.y, min.z), cs),
            reference.add_vec3(Vec3::new(max.x, max.y, min.z), cs),
            reference.add_vec3(Vec3::new(min.x, max.y, min.z), cs),
            reference.add_vec3(Vec3::new(min.x, min.y, max.z), cs),
            reference.add_vec3(Vec3::new(max.x, min.y, max.z), cs),
            reference.add_vec3(Vec3::new(max.x, max.y, max.z), cs),
            reference.add_vec3(Vec3::new(min.x, max.y, max.z), cs),
        ];

        // 12 edges of the box
        let edges: [(usize, usize); 12] = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0), // bottom face
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4), // top face
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7), // vertical edges
        ];

        for (i, j) in edges {
            self.line(c[i], c[j], color, thickness, duration);
        }
    }

    /// Visualize BVH nodes with depth-based coloring
    pub fn visualize_bvh_nodes(
        &mut self,
        nodes: &[BvhNode],
        reference: WorldPos,
        max_depth: u32,
        leaves_only: bool,
        thickness: f32,
        duration: f32,
    ) {
        if nodes.is_empty() {
            return;
        }
        self.visualize_bvh_recursive(
            nodes,
            0,
            reference,
            0,
            max_depth,
            leaves_only,
            thickness,
            duration,
        );
    }

    fn visualize_bvh_recursive(
        &mut self,
        nodes: &[BvhNode],
        node_idx: usize,
        reference: WorldPos,
        depth: u32,
        max_depth: u32,
        leaves_only: bool,
        thickness: f32,
        duration: f32,
    ) {
        if node_idx >= nodes.len() || depth > max_depth {
            return;
        }

        let node = &nodes[node_idx];
        let is_leaf = node.is_leaf();

        // Draw this node's AABB
        if !leaves_only || is_leaf {
            let aabb = Aabb::new(node.aabb_min, node.aabb_max);
            let color = depth_to_color(depth, max_depth);
            self.aabb(&aabb, reference, color, thickness, duration);
        }

        // Recurse into children if not a leaf
        if !is_leaf {
            let left_idx = node.child_or_tri_offset as usize;
            let right_idx = node.tri_count_or_right as usize;

            self.visualize_bvh_recursive(
                nodes,
                left_idx,
                reference,
                depth + 1,
                max_depth,
                leaves_only,
                thickness,
                duration,
            );
            self.visualize_bvh_recursive(
                nodes,
                right_idx,
                reference,
                depth + 1,
                max_depth,
                leaves_only,
                thickness,
                duration,
            );
        }
    }

    /// Visualize TLAS (Top Level Acceleration Structure)
    pub fn visualize_tlas(
        &mut self,
        tlas: &Tlas,
        reference: WorldPos,
        show_instances: bool,
        show_bvh: bool,
        bvh_max_depth: u32,
        thickness: f32,
        duration: f32,
    ) {
        // Draw instance AABBs in cyan
        if show_instances {
            for (i, inst) in tlas.instances.iter().enumerate() {
                let aabb = Aabb::new(inst.aabb_min, inst.aabb_max);
                // Alternate colors for different instances
                let hue = (i as f32 * 0.618033988749895) % 1.0; // golden ratio for spread
                let color = hsv_to_rgb(HSV {
                    h: hue,
                    s: 0.7,
                    v: 0.9,
                });
                let color = [color[0], color[1], color[2], 1.0];
                self.aabb(&aabb, reference, color, thickness, duration);
            }
        }

        // Draw BVH structure with depth coloring
        if show_bvh && !tlas.bvh_nodes.is_empty() {
            self.visualize_bvh_nodes(
                &tlas.bvh_nodes,
                reference,
                bvh_max_depth,
                false,
                thickness,
                duration,
            );
        }
    }

    /// Visualize BLAS (Bottom Level Acceleration Structure)
    pub fn visualize_blas(
        &mut self,
        blas: &Blas,
        reference: WorldPos,
        show_root: bool,
        show_bvh: bool,
        bvh_max_depth: u32,
        thickness: f32,
        duration: f32,
    ) {
        // Draw root AABB in magenta
        if show_root {
            self.aabb(
                blas.root_aabb(),
                reference,
                [1.0, 0.0, 1.0, 1.0],
                thickness,
                duration,
            );
        }

        // Draw BVH structure
        if show_bvh {
            self.visualize_bvh_nodes(
                &blas.bvh_nodes,
                reference,
                bvh_max_depth,
                false,
                thickness,
                duration,
            );
        }
    }

    /// Visualize entire RT subsystem
    pub fn visualize_rt(
        &mut self,
        rt_subsystem: &RTSubsystem,
        reference: WorldPos,
        show_tlas_instances: bool,
        show_tlas_bvh: bool,
        tlas_bvh_depth: u32,
        show_blas: bool,
        blas_bvh_depth: u32,
        thickness: f32,
        duration: f32,
    ) {
        // Visualize TLAS
        self.visualize_tlas(
            &rt_subsystem.tlas,
            reference,
            show_tlas_instances,
            show_tlas_bvh,
            tlas_bvh_depth,
            thickness,
            duration,
        );

        // Visualize BLAS (at origin reference - instances handle world transforms)
        if show_blas {
            if let Some(blas) = &rt_subsystem.car_blas {
                self.visualize_blas(
                    blas,
                    reference,
                    true,
                    true,
                    blas_bvh_depth,
                    thickness,
                    duration,
                );
            }
        }
    }

    /// Visualize chunks that were just updated with colorful pulsing boxes.
    /// Call this right after `job_system.update_chunks()` with the results.
    pub fn visualize_chunk_updates(
        &mut self,
        updated_chunks: &[ChunkCoord],
        current_time: f64,
        thickness: f32,
    ) {
        let cs = self.chunk_size;
        let chunk_size_f = cs as f32;
        let duration = 2.0; // Visible for 2 seconds (matches tick interval)

        for &chunk_coord in updated_chunks.iter() {
            // Rainbow color based on chunk position + time for variety
            let hash = (chunk_coord.x.wrapping_mul(73856093) ^ chunk_coord.z.wrapping_mul(19349663))
                as f32;
            let hue = ((hash.abs() % 1000.0) / 1000.0 + current_time as f32 * 0.1) % 1.0;

            let color = hsv_to_rgb(HSV {
                h: hue,
                s: 0.9,
                v: 1.0,
            });
            let color = [color[0], color[1], color[2], 1.0];
            // Chunk center at y=0
            let center = WorldPos::new(
                chunk_coord,
                LocalPos::new(chunk_size_f * 0.5, 0.0, chunk_size_f * 0.5),
            );

            // Draw box outline
            self.box_xz(center, chunk_size_f * 0.48, color, thickness, duration);

            // Draw an X across the chunk for extra visibility
            let corners = [
                center.add_vec3(
                    Vec3::new(-chunk_size_f * 0.45, 0.0, -chunk_size_f * 0.45),
                    cs,
                ),
                center.add_vec3(Vec3::new(chunk_size_f * 0.45, 0.0, chunk_size_f * 0.45), cs),
                center.add_vec3(
                    Vec3::new(-chunk_size_f * 0.45, 0.0, chunk_size_f * 0.45),
                    cs,
                ),
                center.add_vec3(
                    Vec3::new(chunk_size_f * 0.45, 0.0, -chunk_size_f * 0.45),
                    cs,
                ),
            ];
            self.line(corners[0], corners[1], color, thickness, duration);
            self.line(corners[2], corners[3], color, thickness, duration);

            // Small circle in center
            self.circle(center, chunk_size_f * 0.15, color, thickness, duration);
        }
    }

    /// Simpler version: just bright-colored boxes with chunk index number
    pub fn visualize_chunk_updates_numbered(
        &mut self,
        updated_chunks: &[ChunkCoord],
        current_time: f64,
        thickness: f32,
    ) {
        let cs = self.chunk_size;
        let chunk_size_f = cs as f32;
        let duration = 1.5;

        for (i, &chunk_coord) in updated_chunks.iter().enumerate() {
            // Cycle through bright colors
            let hue = (i as f32 * 0.137 + current_time as f32 * 0.05) % 1.0;
            let color = hsv_to_rgb(HSV {
                h: hue,
                s: 0.85,
                v: 1.0,
            });
            let color = [color[0], color[1], color[2], 1.0];

            let center = WorldPos::new(
                chunk_coord,
                LocalPos::new(chunk_size_f * 0.5, 1.0, chunk_size_f * 0.5),
            );

            // Box outline
            self.box_xz(center, chunk_size_f * 0.49, color, thickness, duration);

            // Show update order number
            let label_pos = center.add_vec3(Vec3::new(0.0, 0.0, 0.0), cs);
            self.text(
                i.to_string(),
                label_pos,
                chunk_size_f * 0.15,
                color,
                None,
                thickness,
                duration,
            );
        }
    }

    pub fn collect_batches(
        &mut self,
        camera: &Camera,
        pipelines: &Pipelines,
        queue: &Queue,
    ) -> GizmoBatches {
        let mut batches = GizmoBatches::default();
        let eye = camera.eye_world();
        let mut text_idx = 0;
        for render in self.pending_renders.iter_mut() {
            if render.filled {
                // Triangulate filled area
                batches.thick_vertices.extend(Self::triangulate_filled(
                    &render.vertices,
                    eye,
                    self.chunk_size,
                ));
            } else if render.thickness > 0.0 {
                // Generate thick line quads

                batches.thick_vertices.extend(Self::generate_thick_lines(
                    &render.vertices,
                    render.thickness,
                    eye,
                    camera,
                    self.chunk_size,
                ));
            } else {
                let verts = RefCell::new(Vec::<TextVertex3D>::new());
                if let Some(text) = &mut render.text {
                    let world_scale = text.scale;

                    let distance = (camera.eye_world().distance_to(text.center, self.chunk_size)
                        as f32)
                        .max(0.001);
                    // Bigger factor = higher glyph resolution overall.
                    // More distance = lower raster scale.
                    // Bigger world text = higher raster scale.
                    let raw = self.text_raster_factor * world_scale
                        / (distance * camera.fov.to_radians().tan());
                    let step = 2.0;
                    let raster_scale = ((raw / step).round() * step)
                        .clamp(self.text_raster_min, self.text_raster_max);
                    if text_idx == 0 {
                        println!(
                            "Raw: {}, FOV: {}, FOV.tan(): {}, Final Res: {}",
                            raw,
                            camera.fov,
                            camera.fov.to_radians().tan(),
                            raster_scale
                        );
                    }
                    text_idx += 1;
                    for t in text.section.text.iter_mut() {
                        t.scale = PxScale::from(raster_scale);
                    }
                    let facing: Option<Vec3> = text.facing;

                    let section = text.section.to_borrowed();

                    self.brush.queue(&section);

                    let color = section
                        .text
                        .first()
                        .map(|t| t.extra.color)
                        .unwrap_or([1.0, 1.0, 1.0, 1.0]);

                    let glyphs = RefCell::new(Vec::<GlyphQuad>::new());

                    let min_x = Cell::new(f32::INFINITY);
                    let min_y = Cell::new(f32::INFINITY);
                    let max_x = Cell::new(f32::NEG_INFINITY);
                    let max_y = Cell::new(f32::NEG_INFINITY);

                    let _ = self.brush.process_queued(
                        |rect, tex_data| {
                            let width = rect.width();
                            let height = rect.height();

                            queue.write_texture(
                                wgpu::TexelCopyTextureInfo {
                                    texture: &pipelines.resolved.atlas.texture(),
                                    mip_level: 0,
                                    origin: wgpu::Origin3d {
                                        x: rect.min[0],
                                        y: rect.min[1],
                                        z: 0,
                                    },
                                    aspect: wgpu::TextureAspect::All,
                                },
                                tex_data,
                                wgpu::TexelCopyBufferLayout {
                                    offset: 0,
                                    bytes_per_row: Some(width),
                                    rows_per_image: Some(height),
                                },
                                wgpu::Extent3d {
                                    width,
                                    height,
                                    depth_or_array_layers: 1,
                                },
                            );
                        },
                        |v| {
                            let px = v.pixel_coords;
                            let uv = v.tex_coords;

                            min_x.set(min_x.get().min(px.min.x));
                            min_y.set(min_y.get().min(px.min.y));
                            max_x.set(max_x.get().max(px.max.x));
                            max_y.set(max_y.get().max(px.max.y));

                            glyphs.borrow_mut().push(GlyphQuad { px, uv });
                        },
                    );

                    let center_x = (min_x.get() + max_x.get()) * 0.5;
                    let center_y = (min_y.get() + max_y.get()) * 0.5;
                    let glyphs = glyphs.into_inner();

                    let forward = if let Some(dir) = facing {
                        dir.normalize()
                    } else {
                        text.center
                            .direction_to(camera.eye_world(), self.chunk_size)
                            .normalize()
                    };

                    let world_up = if forward.dot(Vec3::Y).abs() > 0.99 {
                        Vec3::X
                    } else {
                        Vec3::Y
                    };

                    let right = world_up.cross(forward).normalize();
                    let up = right.cross(forward).normalize();
                    let world_to_geom = world_scale / raster_scale;

                    for glyph in glyphs {
                        let px = glyph.px;
                        let uv = glyph.uv;

                        let x0 = (px.min.x - center_x) * world_to_geom;
                        let y0 = (px.min.y - center_y) * world_to_geom;
                        let x1 = (px.max.x - center_x) * world_to_geom;
                        let y1 = (px.max.y - center_y) * world_to_geom;

                        let p0 = text.center.add_vec3(right * x0 + up * y0, self.chunk_size);
                        let p1 = text.center.add_vec3(right * x1 + up * y0, self.chunk_size);
                        let p2 = text.center.add_vec3(right * x1 + up * y1, self.chunk_size);
                        let p3 = text.center.add_vec3(right * x0 + up * y1, self.chunk_size);

                        let uv0 = [uv.min.x, uv.min.y];
                        let uv1 = [uv.max.x, uv.min.y];
                        let uv2 = [uv.max.x, uv.max.y];
                        let uv3 = [uv.min.x, uv.max.y];

                        verts.borrow_mut().extend([
                            TextVertex3D::new(p0, uv0, color),
                            TextVertex3D::new(p1, uv1, color),
                            TextVertex3D::new(p2, uv2, color),
                            TextVertex3D::new(p2, uv2, color),
                            TextVertex3D::new(p3, uv3, color),
                            TextVertex3D::new(p0, uv0, color),
                        ]);
                    }

                    text.vertices = verts.into_inner();

                    batches.text_vertices.extend(
                        text.vertices
                            .iter()
                            .map(|v| v.to_render(eye, self.chunk_size)),
                    );
                }

                batches.thin_vertices.extend(
                    render
                        .vertices
                        .iter()
                        .map(|v| v.to_render(eye, self.chunk_size)),
                );
            }
        }
        batches
    }

    fn generate_thick_lines(
        vertices: &[LineVtxWorld],
        thickness: f32,
        eye: WorldPos,
        _camera: &Camera,
        chunk_size: ChunkSize,
    ) -> Vec<LineVtxRender> {
        let mut result = Vec::new();
        let half_thick = thickness * 0.5;

        // Process pairs of vertices (line segments)
        for pair in vertices.chunks_exact(2) {
            let v0 = pair[0].to_render(eye, chunk_size);
            let v1 = pair[1].to_render(eye, chunk_size);

            // Calculate perpendicular offset in screen space (we'll do this in shader for better performance)
            // For now, generate a quad with metadata
            // We'll use a modified shader approach - pass thickness as a vertex attribute

            // Generate quad (2 triangles = 6 vertices)
            let quad = Self::line_to_quad(v0, v1, half_thick, eye, chunk_size);
            result.extend_from_slice(&quad);
        }

        result
    }

    fn line_to_quad(
        v0: LineVtxRender,
        v1: LineVtxRender,
        half_thickness: f32,
        eye: WorldPos,
        chunk_size: ChunkSize,
    ) -> [LineVtxRender; 6] {
        let dx = v1.pos[0] - v0.pos[0];
        let dy = v1.pos[1] - v0.pos[1];
        let dz = v1.pos[2] - v0.pos[2];

        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len < 0.0001 {
            return [v0; 6];
        }

        let dir = [dx / len, dy / len, dz / len];

        // midpoint of segment
        let mx = (v0.pos[0] + v1.pos[0]) * 0.5;
        let my = (v0.pos[1] + v1.pos[1]) * 0.5;
        let mz = (v0.pos[2] + v1.pos[2]) * 0.5;

        let camera_pos = eye.to_render_pos(eye, chunk_size);

        // view direction
        let mut vx = camera_pos[0] - mx;
        let mut vy = camera_pos[1] - my;
        let mut vz = camera_pos[2] - mz;

        let vlen = (vx * vx + vy * vy + vz * vz).sqrt();
        if vlen > 0.0001 {
            vx /= vlen;
            vy /= vlen;
            vz /= vlen;
        }

        // perpendicular = dir × view
        let mut ox = dir[1] * vz - dir[2] * vy;
        let mut oy = dir[2] * vx - dir[0] * vz;
        let mut oz = dir[0] * vy - dir[1] * vx;

        let olen = (ox * ox + oy * oy + oz * oz).sqrt();
        if olen < 0.0001 {
            return [v0; 6];
        }

        let scale = half_thickness / olen;
        ox *= scale;
        oy *= scale;
        oz *= scale;

        let p0 = LineVtxRender {
            pos: [v0.pos[0] + ox, v0.pos[1] + oy, v0.pos[2] + oz],
            color: v0.color,
        };
        let p1 = LineVtxRender {
            pos: [v0.pos[0] - ox, v0.pos[1] - oy, v0.pos[2] - oz],
            color: v0.color,
        };
        let p2 = LineVtxRender {
            pos: [v1.pos[0] + ox, v1.pos[1] + oy, v1.pos[2] + oz],
            color: v1.color,
        };
        let p3 = LineVtxRender {
            pos: [v1.pos[0] - ox, v1.pos[1] - oy, v1.pos[2] - oz],
            color: v1.color,
        };

        [p0, p1, p2, p1, p3, p2]
    }

    fn triangulate_filled(
        vertices: &[LineVtxWorld],
        camera: WorldPos,
        chunk_size: ChunkSize,
    ) -> Vec<LineVtxRender> {
        if vertices.len() < 3 {
            return Vec::new();
        }
        let result = triangulate_ear_clipping(vertices, camera, chunk_size);

        result
    }
}
#[inline]
fn flap_color(c: [f32; 4]) -> [f32; 4] {
    [
        (c[0] + 1.0) * 0.5,
        (c[1] + 1.0) * 0.5,
        (c[2] + 1.0) * 0.5,
        c[3],
    ]
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
    pub fn new(pos: WorldPos, color: [f32; 4]) -> Self {
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
pub fn barycentric_y(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<f32> {
    let v0x = b.x - a.x;
    let v0z = b.z - a.z;
    let v1x = c.x - a.x;
    let v1z = c.z - a.z;
    let v2x = p.x - a.x;
    let v2z = p.z - a.z;

    let denom = v0x * v1z - v1x * v0z;
    if denom.abs() < 1e-6 {
        return None;
    }

    let v = (v2x * v1z - v1x * v2z) / denom;
    let w = (v0x * v2z - v2x * v0z) / denom;
    let u = 1.0 - v - w;

    if u >= 0.0 && v >= 0.0 && w >= 0.0 {
        Some(a.y * u + b.y * v + c.y * w)
    } else {
        None
    }
}

#[derive(Clone, Copy)]
struct P {
    x: f32,
    z: f32,
}

fn cross(a: P, b: P, c: P) -> f32 {
    (b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x)
}

fn is_convex(prev: P, curr: P, next: P) -> bool {
    cross(prev, curr, next) < 0.0
}

fn point_in_triangle(a: P, b: P, c: P, p: P) -> bool {
    let c1 = cross(a, b, p);
    let c2 = cross(b, c, p);
    let c3 = cross(c, a, p);
    (c1 < 0.0) && (c2 < 0.0) && (c3 < 0.0)
}

pub fn triangulate_ear_clipping(
    vertices: &[LineVtxWorld],
    camera: WorldPos,
    chunk_size: ChunkSize,
) -> Vec<LineVtxRender> {
    let n = vertices.len();
    if n < 3 {
        return vec![];
    }

    let poly: Vec<P> = vertices
        .iter()
        .map(|v| {
            let r = v.to_render(camera, chunk_size);
            P {
                x: r.pos[0],
                z: r.pos[2],
            }
        })
        .collect();

    // Detect and Correct Winding Order

    // Calculate signed area using the Shoelace formula.
    // Area > 0 indicates Counter-Clockwise (CCW) in a standard coordinate system (X-right, Z-up).
    // Area < 0 indicates Clockwise (CW).
    let mut signed_area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        signed_area += (poly[i].x * poly[j].z) - (poly[j].x * poly[i].z);
    }

    // Your logic assumes CW winding (convex check is cross < 0).
    // If the polygon is CCW (positive area), we reverse the indices to make it CW.
    let mut indices: Vec<usize> = (0..n).collect();
    if signed_area > 0.0 {
        indices.reverse();
    }

    let mut result = Vec::new();
    let mut guard = 0;

    while indices.len() > 3 && guard < 10_000 {
        guard += 1;
        let mut ear_found = false;

        for i in 0..indices.len() {
            let prev_i = indices[(i + indices.len() - 1) % indices.len()];
            let curr_i = indices[i];
            let next_i = indices[(i + 1) % indices.len()];

            let a = poly[prev_i];
            let b = poly[curr_i];
            let c = poly[next_i];

            // This check now works correctly because we ensured CW winding above
            if !is_convex(a, b, c) {
                continue;
            }

            let mut contains_point = false;
            for &j in &indices {
                if j == prev_i || j == curr_i || j == next_i {
                    continue;
                }
                // This check now works correctly because we ensured CW winding above
                if point_in_triangle(a, b, c, poly[j]) {
                    contains_point = true;
                    break;
                }
            }

            if contains_point {
                continue;
            }

            // ear found
            let v_a = vertices[prev_i].to_render(camera, chunk_size);
            let v_b = vertices[curr_i].to_render(camera, chunk_size);
            let v_c = vertices[next_i].to_render(camera, chunk_size);

            result.push(v_a);
            result.push(v_b);
            result.push(v_c);

            indices.remove(i);
            ear_found = true;
            break;
        }

        if !ear_found {
            // This should theoretically not happen for valid simple polygons
            // if winding is correct, but we keep the guard.
            break;
        }
    }

    if indices.len() == 3 {
        let a = vertices[indices[0]].to_render(camera, chunk_size);
        let b = vertices[indices[1]].to_render(camera, chunk_size);
        let c = vertices[indices[2]].to_render(camera, chunk_size);

        result.push(a);
        result.push(b);
        result.push(c);
    }

    result
}

struct TextVertex3D {
    pos: WorldPos,
    uv: [f32; 2],
    color: [f32; 4],
}
impl TextVertex3D {
    pub fn new(pos: WorldPos, uv: [f32; 2], color: [f32; 4]) -> Self {
        Self { pos, uv, color }
    }
    #[inline]
    pub fn to_render(&self, camera_pos: WorldPos, chunk_size: ChunkSize) -> TextVtxRender {
        let rp = self.pos.to_render_pos(camera_pos, chunk_size);
        TextVtxRender {
            pos: rp.to_array(),
            uv: self.uv,
            color: self.color,
        }
    }
}

#[derive(Clone, Copy)]
struct GlyphQuad {
    px: Rect,
    uv: Rect,
}
