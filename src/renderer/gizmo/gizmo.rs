#![allow(dead_code)]

use crate::data::Settings;
use crate::helpers::hsv::{HSV, depth_to_color, hsv_to_rgb};
use crate::helpers::positions::{ChunkCoord, LocalPos, WorldPos, chunk_size};

use crate::helpers::stupid_color_from_rgba;
use crate::renderer::pipelines::{COLOR_FORMAT, Pipelines};
use crate::renderer::ray_tracing::rt_subsystem::RTSubsystem;
use crate::renderer::ray_tracing::structs::{Aabb, Blas, BvhNode, Tlas};
use crate::renderer::ui_pipelines::UI_DEPTH_FORMAT;
use crate::ui::ui_editor::Ui;
use crate::ui::vertex::{LineVtxWorld, TextVtxRender, ThickLineVtxRender, ThinLineVtxRender};
use crate::world::buildings::buildings::Buildings;
use crate::world::buildings::zoning::{Zoning, ZoningStorage, point_in_polygon_xz};
use crate::world::camera::{Camera, CameraMode};
use crate::world::cars::car_structs::CarStorage;
use crate::world::cars::parking::{PARK_L, PARK_W, ParkingStorage};
use crate::world::cars::partitions::PartitionId;
use crate::world::cars::signfinding::{SFTurnType, get_last_turn};
use crate::world::roads::road_structs::SnapPreview;
use crate::world::roads::roads::{RoadManager, RoadStorage};
use crate::world::terrain::terrain_subsystem::Terrain;
use glam::Vec3;
use sluggrs_skylines::cosmic_text::{Attrs, Family, Metrics, Shaping, Wrap};
use sluggrs_skylines::{
    Cache, FontSystem, Resolution, TextArea, TextAtlas, TextBounds, TextDecoration, TextRenderer,
    Viewport, cosmic_text,
};
use std::collections::HashMap;
use std::f32::consts::{PI, TAU};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::mem;
use std::sync::{Mutex, OnceLock};
use tracing::error;
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, DepthStencilState, Device,
    MultisampleState, Queue, SurfaceConfiguration,
};

static PENDING_GIZMO_RENDERS: OnceLock<Mutex<Vec<PendingGizmoRender>>> = OnceLock::new();

fn pending_gizmo_renders() -> &'static Mutex<Vec<PendingGizmoRender>> {
    PENDING_GIZMO_RENDERS.get_or_init(|| Mutex::new(Vec::new()))
}
pub fn push_gizmo_render(render: PendingGizmoRender) {
    if let Ok(mut queue) = pending_gizmo_renders().lock() {
        queue.push(render);
    }
}
pub fn push_gizmo_renders(renders: Vec<PendingGizmoRender>) {
    if let Ok(mut queue) = pending_gizmo_renders().lock() {
        queue.extend(renders);
    }
}
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
    pub buffer: cosmic_text::Buffer,
    pub center: WorldPos,
    pub scale: f32,
    pub color: [f32; 4],
    pub scale_with_cam: bool,
    pub decorations: Vec<TextDecoration>,
}

pub struct GizmoBuffers {
    pub thin_buffer: Buffer,
    pub thick_buffer: Buffer,
    pub filled_buffer: Buffer,
    pub text_renderer: TextRenderer,
}
pub struct Gizmo {
    pub pending_renders: Vec<PendingGizmoRender>,
    pub gizmo_buffers: Option<GizmoBuffers>,
    total_game_time: f64,
}

#[derive(Default)]
pub struct GizmoBatches {
    pub thin_vertices: Vec<ThinLineVtxRender>,
    pub thick_vertices: Vec<ThickLineVtxRender>,
    pub filled_vertices: Vec<ThinLineVtxRender>,
}
impl Gizmo {
    pub fn new(
        device: &Device,
        queue: &Queue,
        config: &SurfaceConfiguration,
        text_atlas: &mut TextAtlas,
        msaa_samples: u32,
    ) -> Self {
        let thin_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Gizmo Thin VB"),
            size: (size_of::<ThinLineVtxRender>() * 2048) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let filled_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Gizmo Filled VB"),
            size: (size_of::<ThinLineVtxRender>() * 2048) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let thick_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Gizmo Thick VB"),
            size: (size_of::<ThickLineVtxRender>() * 2048) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let text_renderer = TextRenderer::new(
            text_atlas,
            device,
            MultisampleState {
                count: msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            None,
        );

        let gizmo_buffers = Some(GizmoBuffers {
            thin_buffer,
            thick_buffer,
            filled_buffer,
            text_renderer,
        });

        Self {
            pending_renders: Vec::new(),
            gizmo_buffers,
            total_game_time: 0.0,
        }
    }
    pub fn new_empty() -> Self {
        Self {
            pending_renders: Vec::new(),
            gizmo_buffers: None,
            total_game_time: 0.0,
        }
    }
    pub fn clear(&mut self) {
        let now = self.total_game_time;
        self.pending_renders
            .retain(|g| now - g.start_time < g.duration as f64);
    }
    pub fn update_msaa(&mut self, msaa_samples: u32, device: &Device, text_atlas: &mut TextAtlas) {
        if let Some(gizmo_buffers) = self.gizmo_buffers.as_mut() {
            let text_renderer = TextRenderer::new(
                text_atlas,
                device,
                MultisampleState {
                    count: msaa_samples,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                None,
            );
            gizmo_buffers.text_renderer = text_renderer;
        }
    }
    pub fn update_buffers(
        &mut self,
        device: &Device,
        queue: &Queue,
        batches: &GizmoBatches,
    ) -> (u32, u32, u32) {
        let thin_count = batches.thin_vertices.len() as u32;
        let thick_count = batches.thick_vertices.len() as u32;
        let filled_count = batches.filled_vertices.len() as u32;

        let Some(gb) = self.gizmo_buffers.as_mut() else {
            return (0, 0, 0);
        };

        if thin_count > 0 {
            let byte_size = (batches.thin_vertices.len() * size_of::<ThinLineVtxRender>()) as u64;

            if byte_size > gb.thin_buffer.size() {
                let new_size = (gb.thin_buffer.size() * 2).max(byte_size);

                if new_size > device.limits().max_buffer_size {
                    error!("Gizmo Thin Buffer tried to become larger than the max_buffer_size");
                    return (0, 0, 0);
                }

                gb.thin_buffer = device.create_buffer(&BufferDescriptor {
                    label: Some("Gizmo Thin VB"),
                    size: new_size,
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }

            queue.write_buffer(
                &gb.thin_buffer,
                0,
                bytemuck::cast_slice(&batches.thin_vertices),
            );
        }

        if thick_count > 0 {
            let byte_size = (batches.thick_vertices.len() * size_of::<ThickLineVtxRender>()) as u64;

            if byte_size > gb.thick_buffer.size() {
                let new_size = (gb.thick_buffer.size() * 2).max(byte_size);

                if new_size > device.limits().max_buffer_size {
                    error!("Gizmo Thick Buffer tried to become larger than the max_buffer_size");
                    return (0, 0, 0);
                }

                gb.thick_buffer = device.create_buffer(&BufferDescriptor {
                    label: Some("Gizmo Thick VB"),
                    size: new_size,
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }

            queue.write_buffer(
                &gb.thick_buffer,
                0,
                bytemuck::cast_slice(&batches.thick_vertices),
            );
        }

        if filled_count > 0 {
            let byte_size = (batches.filled_vertices.len() * size_of::<ThinLineVtxRender>()) as u64;

            if byte_size > gb.filled_buffer.size() {
                let new_size = (gb.filled_buffer.size() * 2).max(byte_size);

                if new_size > device.limits().max_buffer_size {
                    error!("Gizmo Filled Buffer tried to become larger than the max_buffer_size");
                    return (0, 0, 0);
                }

                gb.filled_buffer = device.create_buffer(&BufferDescriptor {
                    label: Some("Gizmo Filled VB"),
                    size: new_size,
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }

            queue.write_buffer(
                &gb.filled_buffer,
                0,
                bytemuck::cast_slice(&batches.filled_vertices),
            );
        }

        (thin_count, thick_count, filled_count)
    }

    pub fn visualize_utilities(
        &mut self,
        road_manager: &RoadManager,
        buildings: &mut Buildings,
        zoning: &ZoningStorage,
        camera: &Camera,
    ) {
        for segment_id in road_manager
            .roads
            .segment_ids_touching_chunks(&camera.target.chunk.get_chunks_in_chunk_distance(5))
        {
            let Some(segment) = road_manager.roads.segment_safe(segment_id) else {
                continue;
            };

            let Some(road_type) = road_manager.road_types.get_road_type(segment.road_type_id)
            else {
                continue;
            };

            if segment.lanes.is_empty() || segment.utility_rails.is_empty() {
                continue;
            }

            let lane_count = segment.lanes.len();
            let rail_count = segment.utility_rails.len();

            let rails_per_lane = rail_count / lane_count;
            let extra_rails = rail_count % lane_count;

            let mut rail_index = 0;

            for (lane_index, &lane_id) in segment.lanes.iter().enumerate() {
                let rails_on_lane = rails_per_lane + usize::from(lane_index < extra_rails);

                if rails_on_lane == 0 {
                    continue;
                }

                let Some(lane) = road_manager.roads.lane_safe(lane_id) else {
                    continue;
                };

                let geometry = lane.geometry();

                for local_rail_index in 0..rails_on_lane {
                    let utility_rail = &segment.utility_rails[rail_index];
                    let utility = &buildings.utilities.utilities[utility_rail.utility_id as usize];

                    let lane_offset = ((local_rail_index as f32 + 0.7) / rails_on_lane as f32
                        - 0.7)
                        * road_type.lane_width;

                    let points = WorldPos::offset_polyline(geometry.points.as_slice(), lane_offset);
                    let (a_color, b_color) = if local_rail_index % 2 == 0 {
                        (utility.rail_color, utility.second_rail_color)
                    } else {
                        (utility.second_rail_color, utility.rail_color)
                    };
                    self.polyline_flowing(
                        points.as_slice(),
                        a_color,
                        b_color,
                        0.0,
                        false,
                        0.5,
                        0.0,
                    );

                    rail_index += 1;
                }
            }
        }

        for (idx, endpoint) in buildings
            .utilities
            .endpoints
            .iter()
            .enumerate()
            .filter_map(|(idx, endpoint)| endpoint.as_ref().map(|endpoint| (idx, endpoint)))
        {
            let Some(building_id) = endpoint.endpoint_type.building_id() else {
                continue;
            };
            let Some(building) = buildings.storage.get(building_id) else {
                continue;
            };
            let Some(lot) = zoning.get_lot(building.lot_id) else {
                continue;
            };
            let mut usages = String::new();
            for (idx, usage) in endpoint.utility_usages.iter().enumerate() {
                let utility = &buildings.utilities.utilities[idx]; // Made sure.
                usages.push_str(
                    format!(
                        "{}: Cons: {} {}, Prod: {} {}; ",
                        utility.name,
                        clean_float(usage.consumption),
                        utility.primary_unit,
                        clean_float(usage.production),
                        utility.primary_unit
                    )
                    .as_str(),
                );
            }
            self.text(
                format!("Util Endpoint {idx}, {:?}", usages),
                lot.entrance.pos,
                0.5,
                [0.1, 0.5, 0.05, 1.0],
                None,
                false,
                1.0,
                0.0,
            );
        }
    }

    pub fn visualize_partitions(&mut self, buildings: &Buildings) {
        use crate::helpers::hsv::hsv_to_rgb;

        let storage = &buildings.partitions.storage;

        for (idx, partition) in storage.partitions.iter().enumerate() {
            let id = idx as PartitionId;

            if !storage.is_alive(id) {
                continue;
            }

            // Pick a stable, distinct color per partition ID using golden-ratio hue spread
            let hue = (id as f32 * 137.508) % 360.0; // golden angle
            let [r, g, b] = hsv_to_rgb(HSV::new(hue, 0.75, 0.95));
            let color: [f32; 4] = [r, g, b, 0.85];
            let faint: [f32; 4] = [r, g, b, 0.15];

            let positions = partition.positions(buildings);

            if positions.is_empty() {
                continue;
            }

            // Draw a cross at each building belonging to this partition
            for &pos in &positions {
                self.cross(pos, 5.0, color, 0.0, 0.0);
            }

            if positions.len() >= 3 {
                let hull = WorldPos::convex_hull(&positions);
                self.area(&hull, faint, 0.0);
            }

            // Draw a label at the centroid with the partition ID
            let barycenter = WorldPos::barycenter(&positions);
            self.text(
                format!("P{}", id),
                barycenter,
                12.0,
                color,
                None,
                true,
                0.0,
                0.0,
            );

            // draw lines from centroid to each building so it's clear what belongs to what
            for &pos in &positions {
                self.arrow(barycenter, pos, [r, g, b, 0.95], true, false, 0.1, 0.0);
            }
        }
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
        let cs = chunk_size() as f32;

        for (region_id, region) in road_storage.road_regions.iter_regions() {
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

            let node_ids = region.node_ids();
            if node_ids.is_empty() {
                continue;
            }

            let mut positions: Vec<WorldPos> = Vec::new();

            for &node_id in node_ids {
                let node = road_storage.node(node_id);
                let pos = node.pos();
                positions.push(pos);
                self.circle(pos, 4.0, color, thickness, duration);
                self.cross(pos, 2.0, secondary_color, thickness, duration);
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
                    offset_sum += pos.to_relative_pos(first_pos);
                }
                offset_sum /= positions.len() as f32;
                first_pos.add_vec3(offset_sum)
            };

            let label_pos = centroid.add_vec3(Vec3::new(0.0, 5.0, 0.0));
            self.text(
                region_id.to_string(),
                label_pos,
                4.0,
                color,
                None,
                false,
                thickness,
                duration,
            );
            self.circle(centroid, 2.0, [1.0, 1.0, 1.0, 1.0], thickness, duration);
        }
    }

    pub fn visualize_signfinding_trajectories(
        &mut self,
        car_storage: &CarStorage,
        terrain: &Terrain,
        road_storage: &RoadStorage,
        buildings: &Buildings,
        zoning: &Zoning,
    ) {
        for car in car_storage
            .car_chunk_storage
            .close_car_ids()
            .flat_map(|car_id| car_storage.get(car_id))
        {
            if let Some(physical_traj) = &car.physical_trajectory {
                let offsets: Vec<Vec3> = physical_traj
                    .points
                    .iter()
                    .map(|p| p.pos)
                    .collect::<Vec<Vec3>>();

                self.polyline_relative(
                    physical_traj.origin,
                    offsets,
                    [car.color[0], car.color[1], car.color[2], 0.45],
                    4.0,
                    false,
                    0.0,
                    0.0,
                );
            }
        }

        let Some(picked) = &terrain.last_picked else {
            return;
        };

        let max_d2 = 625.0; // 25 m
        let picked_car_id = car_storage
            .car_chunk_storage
            .close_car_ids()
            .filter_map(|id| {
                let car = car_storage.get(id)?;
                let dist2 = car.pos.distance_squared(picked.pos);
                if dist2 < max_d2 {
                    Some((dist2, id))
                } else {
                    None
                }
            })
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .map(|(_, id)| id);

        let Some(car_id) = picked_car_id else {
            return;
        };

        let Some(car) = car_storage.get(car_id) else {
            return;
        };

        self.circle(car.pos, car.length, [0.1, 0.3, 0.3, 0.9], 0.1, 0.0);

        let Some(sf_traj) = &car.signfinding_trajectory else {
            return;
        };

        if sf_traj.turns.is_empty() {
            return;
        }

        if let Some(destination) = car.mode.destination() {
            if let Some(building) = buildings.storage.get(destination.as_building_id()) {
                if let Some(lot) = zoning.zoning_storage.get_lot(building.lot_id) {
                    if let Some(segment_id) = lot.segment_id {
                        let pos = car.pos.add_vec3(Vec3::new(0.0, 5.0, 0.0));
                        self.text(
                            format!(
                                "Current Turn: {:?}",
                                get_last_turn(
                                    &sf_traj.turns,
                                    car,
                                    road_storage,
                                    building.pos,
                                    segment_id
                                )
                            ),
                            pos,
                            1.0,
                            [0.1, 0.96, 0.64, 0.9],
                            None,
                            false,
                            0.0,
                            0.0,
                        );
                        let pos = car.pos.add_vec3(Vec3::new(0.0, 4.0, 0.0));
                        self.text(
                            format!("Current Lane: {:?}", car.current_lane),
                            pos,
                            1.0,
                            [0.1, 0.66, 0.64, 0.9],
                            None,
                            false,
                            0.0,
                            0.0,
                        );
                    }
                };
            };
        };

        // Assumes turns are ordered from closest to farthest.
        // We render them in reverse so farther turns are drawn on top.
        let turn_count = sf_traj.turns.len() as f32;

        for (i, turn) in sf_traj.turns.iter().enumerate().rev() {
            let t = if turn_count <= 1.0 {
                0.0
            } else {
                i as f32 / (turn_count - 1.0)
            };

            let thickness = 0.1; //2.0 - t * 5.5;
            let alpha = 0.25 + (1.0 - t) * 0.50;
            let color = if turn.is_final_turn_to_building {
                [1.0, 0.15, 0.25, alpha.max(0.75)]
            } else {
                [1.0 - t * 0.35, 1.0 - t * 0.30, 1.0, alpha]
            };
            match &turn.turn_type {
                SFTurnType::SegmentLanes {
                    segment_id,
                    possible_lanes,
                } => {
                    for &lane_id in possible_lanes {
                        let lane = road_storage.lane(lane_id);
                        self.polyline(lane.polyline(), color, 5.0, false, thickness, 0.0);
                    }
                    let segment = road_storage.segment(*segment_id);
                    let from_pos = road_storage.node(segment.start()).pos();
                    let to_pos = road_storage.node(segment.start()).pos();
                    // Midpoint label, lifted a bit so it reads cleanly
                    let mid = from_pos.lerp(to_pos, 0.5);
                    let label_pos = mid.add_vec3(Vec3::new(0.0, 4.45, 0.0));

                    // Small badge behind the text
                    self.circle(
                        label_pos,
                        0.55 + thickness * 0.03,
                        [0.0, 0.0, 0.0, 0.35 + (1.0 - t) * 0.15],
                        0.0,
                        0.0,
                    );

                    let label = if turn.is_final_turn_to_building {
                        format!("T: {}  FINAL  Seg: {}", i + 1, segment_id.index())
                    } else {
                        format!("T: {}  Seg: {}", i + 1, segment_id.index())
                    };

                    self.text(label, label_pos, 1.0, color, None, false, 1.0, 0.0);
                }
                SFTurnType::IntersectionLanes {
                    node_id,
                    possible_paths,
                    to_segment_id,
                } => {
                    let node = road_storage.node(*node_id);
                    for path in possible_paths {
                        for &nodelane_id in path.0.iter() {
                            let Some(lane) = node.node_lane(nodelane_id) else {
                                continue;
                            };
                            self.polyline(lane.polyline(), color, 5.0, false, thickness, 0.0);
                        }
                    }
                }
            }
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
        let cs = chunk_size() as f32;

        let mut verts = Vec::with_capacity(CIRCLE_SEGMENT_COUNT * 2);

        for i in 0..CIRCLE_SEGMENT_COUNT {
            let a0 = (i as f32 / CIRCLE_SEGMENT_COUNT as f32) * TAU;
            let a1 = ((i + 1) as f32 / CIRCLE_SEGMENT_COUNT as f32) * TAU;

            let p0 = center.add_vec3(Vec3::new(radius * a0.cos(), 0.0, radius * a0.sin()));
            let p1 = center.add_vec3(Vec3::new(radius * a1.cos(), 0.0, radius * a1.sin()));

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
        let cs = chunk_size() as f32;
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
                    center.add_vec3(Vec3::new(r * a0.cos(), y, r * a0.sin())),
                    color,
                ));
                verts.push(LineVtxWorld::new(
                    center.add_vec3(Vec3::new(r * a1.cos(), y, r * a1.sin())),
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
                    center.add_vec3(Vec3::new(
                        radius * phi0.sin() * ct,
                        radius * phi0.cos(),
                        radius * phi0.sin() * st,
                    )),
                    color,
                ));
                verts.push(LineVtxWorld::new(
                    center.add_vec3(Vec3::new(
                        radius * phi1.sin() * ct,
                        radius * phi1.cos(),
                        radius * phi1.sin() * st,
                    )),
                    color,
                ));
            }
        }
        self.push(verts, thickness, duration, false);
    }

    pub fn square(
        &mut self,
        center: WorldPos,
        half_size: f32,
        color: [f32; 4],
        thickness: f32,
        duration: f32,
    ) {
        let corners = [
            center.add_vec3(Vec3::new(-half_size, 0.0, -half_size)),
            center.add_vec3(Vec3::new(half_size, 0.0, -half_size)),
            center.add_vec3(Vec3::new(half_size, 0.0, half_size)),
            center.add_vec3(Vec3::new(-half_size, 0.0, half_size)),
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
            center.add_vec3(direction),
            color,
            false,
            true,
            thickness,
            duration,
        );
    }

    // Axes gizmo
    pub fn axes(&mut self, origin: WorldPos, scale: f32, thickness: f32, duration: f32) {
        let cs = chunk_size() as f32;
        let axes = [
            (Vec3::X, [1.0, 0.2, 0.2, 1.0]),
            (Vec3::Y, [0.2, 1.0, 0.2, 1.0]),
            (Vec3::Z, [0.2, 0.6, 1.0, 1.0]),
        ];
        for (dir, color) in axes {
            self.line(
                origin,
                origin.add_vec3(dir * scale),
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
        let cs = chunk_size() as f32;
        let sun_end = origin.add_vec3(sun_dir.normalize_or_zero() * scale);
        self.arrow(
            origin,
            sun_end,
            [1.0, 1.0, 0.0, 1.0],
            false,
            false,
            thickness,
            duration,
        );
        let moon_end = origin.add_vec3(moon_dir.normalize_or_zero() * scale);
        self.arrow(
            origin,
            moon_end,
            [1.0, 1.0, 1.0, 1.0],
            false,
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
        with_start_stopper: bool,
        thickness: f32,
        duration: f32,
    ) {
        let delta = end.delta_to(start);
        let len = delta.length();

        if len < 0.0001 {
            return;
        }

        let dir = delta / len;
        let (side, up_perp) = build_frame(dir);
        let flap = flap_color(color);

        let mut verts = Vec::new();

        let target_spacing = 150.0;
        let steps = (len / target_spacing).ceil().max(1.0) as usize;
        let step_len = len / steps as f32;
        let dash_ratio = if dashed { 0.5 } else { 1.0 };
        let dash_len = step_len * dash_ratio;

        let head_ratio = 0.15;
        let head_width_ratio = 2.0;

        let spin_speed = 1.0;
        let time = self.total_game_time as f32;

        for i in 0..steps {
            let t0 = i as f32 * step_len;
            let t1 = (t0 + dash_len).min(len);

            if t0 >= len {
                break;
            }

            let p0 = start.add_vec3(dir * t0);
            let p1 = start.add_vec3(dir * t1);

            verts.push(LineVtxWorld::new(p0, color));
            verts.push(LineVtxWorld::new(p1, color));

            let segment_len = t1 - t0;
            if segment_len <= 0.0001 {
                continue;
            }

            let head_len = segment_len * head_ratio;
            let head_width = head_len * head_width_ratio;

            let angle = time * spin_speed + (i as f32 * 12.9898).sin() * PI;
            let rot = rotate_frame(side, up_perp, angle);
            let back = p1.add_vec3(-dir * head_len);

            verts.push(LineVtxWorld::new(p1, flap));
            verts.push(LineVtxWorld::new(back.add_vec3(rot * head_width), flap));
            verts.push(LineVtxWorld::new(p1, flap));
            verts.push(LineVtxWorld::new(back.add_vec3(-rot * head_width), flap));
        }

        self.push(verts, thickness, duration, false);

        if with_start_stopper {
            let right = Vec3::new(dir.z, 0.0, -dir.x).normalize() * thickness;
            let left_start = start.sub_vec3(right);
            let right_start = start.add_vec3(right);

            self.line(left_start, right_start, color, thickness, duration);
        }
    }

    pub fn tile(
        &mut self,
        pos: WorldPos,
        dir: Vec3,
        width: f32,
        length: f32,
        color: [f32; 4],
        thickness: f32,
        duration: f32,
    ) {
        let forward = dir.normalize();
        let right = Vec3::new(forward.z, 0.0, -forward.x);

        let half_w = width * 0.5;
        let half_l = length * 0.5;

        let corners = [
            pos.add_vec3(-forward * half_l - right * half_w), // back left
            pos.add_vec3(-forward * half_l + right * half_w), // back right
            pos.add_vec3(forward * half_l + right * half_w),  // front right
            pos.add_vec3(forward * half_l - right * half_w),  // front left
        ];
        self.polyline(corners.as_slice(), color, 0.0, true, thickness, duration);
    }

    /// Render polyline with arrows. Points are WorldPos.
    pub fn polyline(
        &mut self,
        points: &[WorldPos],
        color: [f32; 4],
        arrow_spacing: f32,
        closed: bool,
        thickness: f32,
        duration: f32,
    ) {
        if points.len() < 2 {
            return;
        }

        let cs = chunk_size() as f32;
        let flap = flap_color(color);
        let mut verts = Vec::new();

        let seg_count = if closed {
            points.len()
        } else {
            points.len() - 1
        };

        // Draw line segments
        for i in 0..seg_count {
            let a = points[i];
            let b = if i + 1 < points.len() {
                points[i + 1]
            } else {
                points[0]
            };
            verts.push(LineVtxWorld::new(a, color));
            verts.push(LineVtxWorld::new(b, color));
        }

        // Compute cumulative lengths
        let mut lengths = vec![0.0f32];
        for i in 0..seg_count {
            let a = points[i];
            let b = if i + 1 < points.len() {
                points[i + 1]
            } else {
                points[0]
            };
            let d = b.to_relative_pos(a).length();
            lengths.push(lengths.last().unwrap() + d);
        }

        let total_len = *lengths.last().unwrap();
        if total_len < 0.001 {
            self.push(verts, thickness, duration, false);
            return;
        }

        let sample_at = |t: f32| -> (WorldPos, Vec3) {
            let mut i = 1;
            while i < lengths.len() && lengths[i] < t {
                i += 1;
            }
            let i0 = i - 1;
            let i1 = i.min(seg_count);

            let a = points[i0];
            let b = if i1 < points.len() {
                points[i1]
            } else {
                points[0]
            };

            let seg_t = if lengths[i1] > lengths[i0] {
                (t - lengths[i0]) / (lengths[i1] - lengths[i0])
            } else {
                0.0
            };

            let pos = a.lerp(b, seg_t as f64);
            let dir = b.to_relative_pos(a).normalize_or_zero();
            (pos, dir)
        };

        let head_len = 0.30;
        let head_width = 0.25;
        let spin_speed = 1.0;
        let time = self.total_game_time as f32;

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
            let back = pos.add_vec3(-dir * head_len);

            verts.push(LineVtxWorld::new(pos, flap));
            verts.push(LineVtxWorld::new(back.add_vec3(rot * head_width), flap));
            verts.push(LineVtxWorld::new(pos, flap));
            verts.push(LineVtxWorld::new(back.add_vec3(-rot * head_width), flap));

            idx += 1;
            t += arrow_spacing;
        }

        self.push(verts, thickness, duration, false);
    }
    pub fn polyline_flowing(
        &mut self,
        points: &[WorldPos],
        color_a: [f32; 4],
        color_b: [f32; 4],
        arrow_spacing: f32,
        closed: bool,
        thickness: f32,
        duration: f32,
    ) {
        if points.len() < 2 {
            return;
        }

        let seg_count = if closed {
            points.len()
        } else {
            points.len() - 1
        };

        let mut lengths = vec![0.0f32];

        for i in 0..seg_count {
            let a = points[i];
            let b = if i + 1 < points.len() {
                points[i + 1]
            } else {
                points[0]
            };

            let d = b.to_relative_pos(a).length();
            lengths.push(lengths.last().unwrap() + d);
        }

        let total_len = *lengths.last().unwrap();

        if total_len < 0.001 {
            return;
        }

        let average_section_len = total_len / seg_count as f32;
        let band_length = average_section_len.max(0.05);
        let pattern_length = band_length * 2.0;

        let flow_speed = -7.0;
        let phase = self.total_game_time as f32 * flow_speed;

        let sample_at = |t: f32| -> (WorldPos, Vec3) {
            let mut i = 1;

            while i < lengths.len() && lengths[i] < t {
                i += 1;
            }

            let i0 = i - 1;
            let i1 = i.min(seg_count);

            let a = points[i0];
            let b = if i1 < points.len() {
                points[i1]
            } else {
                points[0]
            };

            let seg_t = if lengths[i1] > lengths[i0] {
                (t - lengths[i0]) / (lengths[i1] - lengths[i0])
            } else {
                0.0
            };

            let pos = a.lerp(b, seg_t as f64);
            let dir = b.to_relative_pos(a).normalize_or_zero();

            (pos, dir)
        };

        let sample_step = (band_length * 0.15).max(0.1);
        let sample_count = (total_len / sample_step).ceil() as usize;

        let mut verts = Vec::with_capacity(sample_count * 2 + 64);

        let color_at = |distance: f32| -> [f32; 4] {
            let p = (distance + phase).rem_euclid(pattern_length);
            let color_index = (p / band_length).floor() as i32;

            if color_index & 1 == 0 {
                color_a
            } else {
                color_b
            }
        };

        let mut previous_distance = 0.0;
        let mut previous_pos = sample_at(0.0).0;

        for i in 1..=sample_count {
            let distance = (i as f32 * sample_step).min(total_len);
            let pos = sample_at(distance).0;

            let color = color_at((previous_distance + distance) * 0.5);

            verts.push(LineVtxWorld::new(previous_pos, color));
            verts.push(LineVtxWorld::new(pos, color));

            previous_distance = distance;
            previous_pos = pos;

            if distance >= total_len {
                break;
            }
        }

        let time = self.total_game_time as f32;
        let head_len = 0.30;
        let head_width = 0.25;
        let spin_speed = 1.0;

        if arrow_spacing > 0.0 {
            let mut t = arrow_spacing;
            let mut idx = 0;

            while t < total_len {
                let (pos, dir) = sample_at(t);

                if dir.length_squared() >= 0.0001 {
                    let color = flap_color(color_at(t));

                    let (side, up_perp) = build_frame(dir);
                    let angle = time * spin_speed + idx as f32 * 1.7;
                    let rot = rotate_frame(side, up_perp, angle);
                    let back = pos.add_vec3(-dir * head_len);

                    verts.push(LineVtxWorld::new(pos, color));
                    verts.push(LineVtxWorld::new(back.add_vec3(rot * head_width), color));

                    verts.push(LineVtxWorld::new(pos, color));
                    verts.push(LineVtxWorld::new(back.add_vec3(-rot * head_width), color));

                    idx += 1;
                }

                t += arrow_spacing;
            }
        }

        self.push(verts, thickness, duration, false);
    }
    /// Polyline from an anchor WorldPos and relative Vec3 offsets.
    /// Useful when you have data in local/relative coordinates.
    pub fn polyline_relative<I>(
        &mut self,
        anchor: WorldPos,
        offsets: I,
        color: [f32; 4],
        arrow_spacing: f32,
        closed: bool,
        thickness: f32,
        duration: f32,
    ) where
        I: IntoIterator<Item = Vec3>,
    {
        let points: Vec<_> = offsets.into_iter().map(|o| anchor.add_vec3(o)).collect();

        self.polyline(&points, color, arrow_spacing, closed, thickness, duration);
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

        self.push(verts, 0.0, duration, true);
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
        let cs = chunk_size() as f32;
        let origin = points[0];

        // Precompute render positions ONCE
        let renders: Vec<Vec3> = points.iter().map(|&p| p.to_relative_pos(origin)).collect();

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

        let spacing = cs * 0.025;

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
                let p = origin.add_vec3(Vec3::new(x, y, z));

                if point_in_polygon_xz(p, points) {
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
        _facing: Option<Vec3>,
        scale_with_cam: bool,
        thickness: f32,
        duration: f32,
    ) where
        S: Into<String>,
    {
        let text = text.into();
        let metrics = Metrics::new(16.0, 19.0);
        let mut buffer = cosmic_text::Buffer::new_empty(metrics);
        //buffer.set_size(None, None);
        let attrs = Attrs::new().family(Family::Name("Noto Sans"));

        buffer.set_text(&text, &attrs, Shaping::Advanced, None);

        self.push_text(
            PendingGizmoTextRender {
                buffer,
                center,
                scale,
                color,
                scale_with_cam,
                decorations: if thickness > 0.0 {
                    vec![TextDecoration::outline(
                        stupid_color_from_rgba([0.0, 0.0, 0.0, 1.0]),
                        thickness,
                    )]
                } else {
                    vec![]
                },
            },
            thickness,
            duration,
        );
    }

    pub fn update(
        &mut self,
        terrain_subsystem: &Terrain,
        rt_subsystem: &RTSubsystem,
        total_game_time: f64,
        road_manager: &RoadManager,
        parking: &ParkingStorage,
        buildings: &mut Buildings,
        zoning: &ZoningStorage,
        settings: &Settings,
        camera: &Camera,
    ) {
        self.total_game_time = total_game_time;
        let target = camera.debug_anchor(10.0);

        let mut renders = pending_gizmo_renders()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut taken_renders = mem::take(&mut *renders);
        taken_renders
            .iter_mut()
            .for_each(|render| render.start_time = self.total_game_time);
        self.pending_renders.extend(taken_renders);

        if let Some(district) = zoning.get_closest_district(camera.target) {
            self.polyline(
                district.points.as_slice(),
                [0.4, 0.0, 0.0, 0.6],
                0.0,
                true,
                0.4,
                0.0,
            );
        }
        self.visualize_utilities(road_manager, buildings, zoning, camera);
        if settings.render_partitions_gizmo {
            self.visualize_partitions(buildings);
            self.visualize_regions(&road_manager.roads, 0.0, 0.0);
        }
        if settings.render_node_ids_gizmo {
            self.visualize_road_node_numbers(&road_manager.roads);
        }
        // self.sphere(camera.eye_world(), 400.0, [1.0, 1.0, 1.0], 0.0);
        if settings.render_rt_gizmo {
            self.visualize_rt(
                rt_subsystem,
                camera.eye_world(), // reference position
                false,              // show TLAS instances
                true,               // show TLAS BVH
                8,                  // TLAS BVH max depth to show
                false,              // show BLAS
                8,                  // BLAS BVH max depth
                0.0,
                0.0, // duration (0 = single frame)
            );
        }
        if settings.render_chunk_bounds {
            if terrain_subsystem.chunks.contains_key(&target.chunk) {
                let cs = chunk_size() as f32;
                let corner = WorldPos::new(
                    target.chunk,
                    LocalPos::new(0.0, terrain_subsystem.get_height_at(target, false), 0.0),
                );
                self.square(
                    corner.add_vec3(Vec3::new(cs * 0.5, 0.0, cs * 0.5)),
                    cs * 0.5,
                    [0.2, 0.8, 0.6, 0.9],
                    0.0,
                    0.0,
                );
            }
        }
        if settings.render_parking_gizmo {
            for parking_spot in parking.iter() {
                let Some(ps) = parking_spot else { continue };

                self.tile(
                    ps.pos,
                    ps.dir,
                    PARK_W as f32,
                    PARK_L as f32,
                    [1.0, 0.0, 0.0, 0.5],
                    0.0,
                    0.0,
                );
            }
        }
        if settings.render_lot_info {
            for lot in zoning
                .lots_in_chunks(camera.eye_world().chunk.get_chunks_3x3())
                .iter()
                .flat_map(|&lot_id| zoning.get_lot(lot_id))
            {
                let entrance = lot.entrance;
                self.arrow(
                    entrance.pos,
                    entrance.pos.add_vec3(entrance.dir.as_vec3().normalize()),
                    [1.0, 0.2, 0.0, 1.0],
                    false,
                    true,
                    0.05,
                    0.0,
                );
                let dir = entrance.dir.as_vec3().normalize();
                let right = Vec3::new(dir.z, 0.0, -dir.x).normalize() * 0.3;
                let left_start = entrance.pos.sub_vec3(right);
                let right_end = entrance.pos.add_vec3(right);
                self.line(left_start, right_end, [1.0, 0.2, 0.0, 1.0], 0.05, 0.0);
                if let Some(layout) = lot.layout.as_ref() {
                    for entrance in layout.driveway_entrances.iter() {
                        self.arrow(
                            entrance.pos,
                            entrance.pos.add_vec3(entrance.dir.as_vec3().normalize()),
                            [0.8, 0.6, 0.0, 1.0],
                            false,
                            true,
                            0.05,
                            0.0,
                        );
                    }
                }
            }
        }
        if !settings.render_lanes_gizmo {
            return;
        }
        // Road visualization
        let render_lane_arrows = false;

        for storage in [&road_manager.roads, &road_manager.preview_roads] {
            for (_node_id, node) in storage.iter_nodes() {
                // Node circle
                let node_pos = node.pos();
                let node_color = [0.0, 0.0, 0.9, 1.0];

                self.circle(node_pos, 2.0, node_color, 0.0, 0.0);
                //println!("{}", node_pos);
                // Incoming lanes
                for &lane_id in node.incoming_lanes() {
                    let lane = storage.lane(lane_id);

                    let segment = storage.segment(lane.segment());
                    let is_forward = lane.from_node() == segment.start();

                    let color = if is_forward {
                        [0.0, 0.9, 0.0, 1.0]
                    } else {
                        [0.2, 0.9, 0.0, 1.0]
                    };

                    // Convert polyline to WorldPos
                    let points: &Vec<WorldPos> = lane.polyline();

                    self.polyline(&points, color, 15.0, false, 0.1, 0.0);

                    if render_lane_arrows {
                        if let Some(last) = points.last() {
                            self.arrow(*last, node_pos, color, false, false, 0.0, 0.0);
                        }
                    }
                    if let Some(&middle) = points.get(points.len() / 2) {
                        //let (pos, _, tangent, _) = lane.geometry().closest_point_to(middle);
                        self.text(
                            format!("{}", lane_id),
                            middle,
                            1.0,
                            color,
                            None,
                            false,
                            0.0,
                            0.0,
                        );
                    }
                }

                // Node lanes
                for node_lane in node.node_lanes() {
                    let color = [0.7, 0.5, 0.0, 1.0];

                    let points: &Vec<WorldPos> = node_lane.polyline();

                    self.polyline(&points, color, 4.0, false, 0.0, 0.0);

                    if render_lane_arrows {
                        if let Some(last) = points.last() {
                            self.arrow(*last, node_pos, color, false, true, 0.0, 0.0);
                        }
                    }
                    if let Some(first) = points.first() {
                        self.text(
                            format!(
                                "Nl_id: {}, merging: {:?}",
                                node_lane.id(),
                                node_lane.merging()
                            ),
                            *first,
                            0.4,
                            color,
                            None,
                            false,
                            0.0,
                            0.0,
                        );
                    }

                    if let Some(last) = points.last() {
                        self.text(
                            format!(
                                "Nl_id: {}, splitting: {:?}",
                                node_lane.id(),
                                node_lane.splitting()
                            ),
                            *last,
                            0.4,
                            color,
                            None,
                            false,
                            0.0,
                            0.0,
                        );
                    }
                }
            }
            for (segment_id, segment) in storage.iter_segments() {
                let mut left_lane = None;
                let mut right_lane = None;

                for &lane_id in segment.lanes() {
                    let lane = storage.lane(lane_id);

                    let idx = lane.lane_index();

                    if idx < 0 {
                        // Highest negative index: -1 beats -2
                        if left_lane
                            .map(|(_, best_idx)| idx > best_idx)
                            .unwrap_or(true)
                        {
                            left_lane = Some((lane_id, idx));
                        }
                    } else if idx > 0 {
                        // Lowest positive index: 1 beats 2
                        if right_lane
                            .map(|(_, best_idx)| idx < best_idx)
                            .unwrap_or(true)
                        {
                            right_lane = Some((lane_id, idx));
                        }
                    }
                }

                let middle = match (left_lane, right_lane) {
                    (Some((left_id, _)), Some((right_id, _))) => {
                        let left = storage.lane(left_id);
                        let right = storage.lane(right_id);

                        let left_points = left.polyline();
                        let right_points = right.polyline();

                        // Use the midpoint sample of the innermost lanes.
                        let li = (left_points.len() / 2).saturating_sub(2);
                        let ri = (right_points.len() / 2).saturating_add(2);
                        let Some(left_point) = left_points.get(li) else {
                            continue;
                        };
                        let Some(right_point) = right_points.get(ri) else {
                            continue;
                        };
                        WorldPos::barycenter(vec![*left_point, *right_point].as_slice())
                    }

                    // One-way road
                    (Some((lane_id, _)), None) | (None, Some((lane_id, _))) => {
                        let points = storage.lane(lane_id).polyline();
                        points[points.len() / 2]
                    }

                    (None, None) => continue,
                };

                self.text(
                    format!("Seg_id: {}", segment_id.raw()),
                    middle,
                    1.0,
                    [1.0, 0.0, 0.0, 1.0],
                    None,
                    false,
                    0.0,
                    0.0,
                );
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
        let cs = chunk_size() as f32;
        let half = size * 0.5;
        self.line(
            pos.add_vec3(Vec3::new(-half, 0.0, 0.0)),
            pos.add_vec3(Vec3::new(half, 0.0, 0.0)),
            color,
            thickness,
            duration,
        );
        self.line(
            pos.add_vec3(Vec3::new(0.0, 0.0, -half)),
            pos.add_vec3(Vec3::new(0.0, 0.0, half)),
            color,
            thickness,
            duration,
        );
    }
    pub fn update_orbit_gizmo(
        &mut self,
        ui: &mut Ui,
        camera: &Camera,
        sun_direction: Vec3,
        moon_direction: Vec3,
        scale_with_orbit: bool,
    ) {
        let debug_menu_active = ui
            .menus
            .get("Debug_Menu")
            .map(|m| m.active)
            .unwrap_or(false);
        ui.variables.set_bool("debug_mode", debug_menu_active);
        if debug_menu_active {
            let anchor = camera.debug_anchor(1.5);
            let scale = match camera.mode {
                CameraMode::Orbit if scale_with_orbit => camera.orbit_radius * 0.1,
                CameraMode::Orbit => 1.0,
                CameraMode::FirstPerson => return, // 0.01, // fixed, human-scale gizmo
            };
            self.axes_with_sun(anchor, scale, sun_direction, moon_direction, 0.0, 0.0);
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

        let cs = chunk_size() as f32;
        let min = Vec3::new(aabb.min[0], aabb.min[1], aabb.min[2]);
        let max = Vec3::new(aabb.max[0], aabb.max[1], aabb.max[2]);

        // 8 corners of the box
        let c = [
            reference.add_vec3(Vec3::new(min.x, min.y, min.z)),
            reference.add_vec3(Vec3::new(max.x, min.y, min.z)),
            reference.add_vec3(Vec3::new(max.x, max.y, min.z)),
            reference.add_vec3(Vec3::new(min.x, max.y, min.z)),
            reference.add_vec3(Vec3::new(min.x, min.y, max.z)),
            reference.add_vec3(Vec3::new(max.x, min.y, max.z)),
            reference.add_vec3(Vec3::new(max.x, max.y, max.z)),
            reference.add_vec3(Vec3::new(min.x, max.y, max.z)),
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
    pub fn visualize_road_node_numbers(&mut self, storage: &RoadStorage) {
        for (id, node) in storage.iter_nodes() {
            self.text(
                format!("Node ID: {}", id.raw()),
                node.pos(),
                1.0,
                [1.0, 1.0, 1.0, 1.0],
                None,
                false,
                0.0,
                0.0,
            );
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
        let cs = chunk_size() as f32;
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
            let center = WorldPos::new(chunk_coord, LocalPos::new(cs * 0.5, 0.0, cs * 0.5));

            // Draw box outline
            self.square(center, cs * 0.48, color, thickness, duration);

            // Draw an X across the chunk for extra visibility
            let corners = [
                center.add_vec3(Vec3::new(cs * 0.45, 0.0, cs * 0.45)),
                center.add_vec3(Vec3::new(cs * 0.45, 0.0, cs * 0.45)),
                center.add_vec3(Vec3::new(cs * 0.45, 0.0, cs * 0.45)),
                center.add_vec3(Vec3::new(cs * 0.45, 0.0, cs * 0.45)),
            ];
            self.line(corners[0], corners[1], color, thickness, duration);
            self.line(corners[2], corners[3], color, thickness, duration);

            // Small circle in center
            self.circle(center, cs * 0.15, color, thickness, duration);
        }
    }

    /// Simpler version: just bright-colored boxes with chunk index number
    pub fn visualize_chunk_updates_numbered(
        &mut self,
        updated_chunks: &[ChunkCoord],
        current_time: f64,
        thickness: f32,
    ) {
        let cs = chunk_size() as f32;
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

            let center = WorldPos::new(chunk_coord, LocalPos::new(cs * 0.5, 1.0, cs * 0.5));

            // Box outline
            self.square(center, cs * 0.49, color, thickness, duration);

            // Show update order number
            let label_pos = center.add_vec3(Vec3::new(0.0, 0.0, 0.0));
            self.text(
                i.to_string(),
                label_pos,
                cs * 0.15,
                color,
                None,
                false,
                thickness,
                duration,
            );
        }
    }

    /// It works, but I am still mad at glyph_brush. Nothing personal, just pain.
    pub fn collect_batches(&mut self, camera: &Camera) -> GizmoBatches {
        let mut batches = GizmoBatches::default();
        let eye = camera.eye_world();

        for render in self.pending_renders.iter_mut() {
            if render.text.is_some() {
                continue;
            }

            if render.filled {
                batches
                    .filled_vertices
                    .extend(Self::triangulate_filled(&render.vertices, eye));
            } else if render.thickness > 0.0 {
                batches.thick_vertices.extend(Self::generate_thick_lines(
                    &render.vertices,
                    render.thickness,
                    eye,
                ));
            } else {
                batches
                    .thin_vertices
                    .extend(render.vertices.iter().map(|v| v.to_render(eye)));
            }
        }

        batches
    }

    fn generate_thick_lines(
        vertices: &[LineVtxWorld],
        thickness: f32,
        eye: WorldPos,
    ) -> Vec<ThickLineVtxRender> {
        let mut result = Vec::new();

        for pair in vertices.chunks_exact(2) {
            let v0 = pair[0].to_render(eye);
            let v1 = pair[1].to_render(eye);

            result.extend(Self::line_to_quad(v0, v1, thickness));
        }

        result
    }

    fn line_to_quad(
        v0: ThinLineVtxRender,
        v1: ThinLineVtxRender,
        thickness: f32,
    ) -> [ThickLineVtxRender; 6] {
        let p0 = v0.pos;
        let p1 = v1.pos;
        let c0 = v0.color;
        let c1 = v1.color;

        [
            ThickLineVtxRender {
                start: p0,
                end: p1,
                side_sign: -1.0,
                end_sign: 0.0,
                width: thickness,
                color: c0,
            },
            ThickLineVtxRender {
                start: p0,
                end: p1,
                side_sign: 1.0,
                end_sign: 0.0,
                width: thickness,
                color: c0,
            },
            ThickLineVtxRender {
                start: p0,
                end: p1,
                side_sign: 1.0,
                end_sign: 1.0,
                width: thickness,
                color: c1,
            },
            ThickLineVtxRender {
                start: p0,
                end: p1,
                side_sign: 1.0,
                end_sign: 1.0,
                width: thickness,
                color: c1,
            },
            ThickLineVtxRender {
                start: p0,
                end: p1,
                side_sign: -1.0,
                end_sign: 1.0,
                width: thickness,
                color: c1,
            },
            ThickLineVtxRender {
                start: p0,
                end: p1,
                side_sign: -1.0,
                end_sign: 0.0,
                width: thickness,
                color: c0,
            },
        ]
    }

    fn triangulate_filled(vertices: &[LineVtxWorld], camera: WorldPos) -> Vec<ThinLineVtxRender> {
        if vertices.len() < 3 {
            return Vec::new();
        }
        let result = triangulate_ear_clipping(vertices, camera);

        result
    }

    pub fn snap_preview(&mut self, snap_preview: &SnapPreview) {}
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
    pub fn to_render(&self, camera_pos: WorldPos) -> ThinLineVtxRender {
        let rp = self.pos.to_relative_pos(camera_pos);
        ThinLineVtxRender {
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
) -> Vec<ThinLineVtxRender> {
    let n = vertices.len();
    if n < 3 {
        return vec![];
    }

    let poly: Vec<P> = vertices
        .iter()
        .map(|v| {
            let r = v.to_render(camera);
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
            let v_a = vertices[prev_i].to_render(camera);
            let v_b = vertices[curr_i].to_render(camera);
            let v_c = vertices[next_i].to_render(camera);

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
        let a = vertices[indices[0]].to_render(camera);
        let b = vertices[indices[1]].to_render(camera);
        let c = vertices[indices[2]].to_render(camera);

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
impl Hash for TextVertex3D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos.hash(state);
        self.uv.map(|c| c.to_bits()).hash(state);
        self.color.map(|c| c.to_bits()).hash(state);
    }
}
impl TextVertex3D {
    pub fn new(pos: WorldPos, uv: [f32; 2], color: [f32; 4]) -> Self {
        Self { pos, uv, color }
    }
    #[inline]
    pub fn to_render(&self, camera_pos: WorldPos) -> TextVtxRender {
        let rp = self.pos.to_relative_pos(camera_pos);
        TextVtxRender {
            pos: rp.to_array(),
            uv: self.uv,
            color: self.color,
        }
    }
}

fn clean_float(value: f32) -> String {
    if value == 0.0 {
        return "0".to_string();
    }

    let significant_digits = 6;
    let digits_before_decimal = value.abs().log10().floor() as i32 + 1;
    let decimals = (significant_digits - digits_before_decimal).max(0) as usize;

    let mut result = format!("{value:.decimals$}");

    if result.contains('.') {
        result = result
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
    }

    result
}

const GIZMO_TEXT_BASE_SIZE: f32 = 16.0;

impl Gizmo {
    fn project_text_position(
        camera: &Camera,
        center: WorldPos,
        width: f32,
        height: f32,
    ) -> Option<(f32, f32)> {
        let relative = center.to_relative_pos(camera.eye_world());
        let clip = camera.view_proj() * relative.extend(1.0);

        if clip.w <= 0.0 || !clip.w.is_finite() {
            return None;
        }

        let ndc = clip.truncate() / clip.w;

        if !ndc.is_finite() {
            return None;
        }

        if ndc.z < 0.0 || ndc.z > 1.0 {
            return None;
        }

        let screen_x = (ndc.x * 0.5 + 0.5) * width;
        let screen_y = (0.5 - ndc.y * 0.5) * height;

        Some((screen_x, screen_y))
    }

    fn text_layout_size(buffer: &cosmic_text::Buffer) -> (f32, f32) {
        let mut width: f32 = 0.0;
        let mut height: f32 = 0.0;

        for run in buffer.layout_runs() {
            width = width.max(run.line_w);
            height = height.max(run.line_top + run.line_height);
        }

        (width, height.max(GIZMO_TEXT_BASE_SIZE))
    }

    fn gizmo_text_scale(
        camera: &Camera,
        text: &PendingGizmoTextRender,
        viewport_height: f32,
    ) -> f32 {
        let distance = camera.eye_world().distance_to(text.center).max(0.001) as f32;

        let tan_half_fov = (camera.fov.to_radians() * 0.5).tan().max(0.0001);

        let desired_pixels = if text.scale_with_cam {
            text.scale * viewport_height * 0.005
        } else {
            let pixels_per_world = viewport_height / (2.0 * distance * tan_half_fov);

            text.scale * pixels_per_world
        };

        (desired_pixels / GIZMO_TEXT_BASE_SIZE).max(0.0625)
    }

    pub fn prepare_text(
        &mut self,
        camera: &Camera,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        text_viewport: &Viewport,
        text_atlas: &mut TextAtlas,
        font_system: &mut FontSystem,
        width: u32,
        height: u32,
    ) -> bool {
        let Some(gb) = self.gizmo_buffers.as_mut() else {
            return false;
        };

        let pending = &mut self.pending_renders;
        let mut areas = Vec::<TextArea<'_>>::new();

        let screen_width = width as f32;
        let screen_height = height as f32;

        let bounds = TextBounds {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };

        for render in pending {
            let Some(text) = render.text.as_mut() else {
                continue;
            };
            text.buffer.shape_until_scroll(font_system, false);
            let Some((screen_x, screen_y)) =
                Self::project_text_position(camera, text.center, screen_width, screen_height)
            else {
                continue;
            };

            let scale = Self::gizmo_text_scale(camera, text, screen_height);

            let (text_width, text_height) = Self::text_layout_size(&text.buffer);
            //println!("Layout size: {}x{}", text_width, text_height);
            //println!("{screen_x}:{screen_y} {scale} {text_width}x{text_height}");
            let left = screen_x - text_width * scale * 0.5;

            let top = screen_y - text_height * scale * 0.5;
            areas.push(TextArea {
                buffer: &text.buffer,
                left,
                top,
                scale,
                bounds,
                default_color: stupid_color_from_rgba(text.color),
                decorations: text.decorations.as_slice(),
            });
        }

        let has_text = !areas.is_empty();

        if let Err(error) = gb.text_renderer.prepare(
            device,
            queue,
            encoder,
            font_system,
            text_atlas,
            text_viewport,
            areas,
        ) {
            error!("Failed to prepare gizmo text: {}", error);
            return false;
        }

        has_text
    }
}
