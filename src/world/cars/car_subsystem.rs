use crate::helpers::hsv::hsv_to_rgb;
use crate::helpers::positions::WorldPos;
use crate::renderer::pipelines::Pipelines;
use crate::renderer::ray_tracing::rt_pass::update_rt_instances;
use crate::renderer::ray_tracing::rt_subsystem::RTSubsystem;
use crate::resources::TimeSystem;
use crate::ui::input::InputState;
use crate::ui::variables::UiVariableRegistry;
use crate::ui::vertex::Vertex;
use crate::world::camera::Camera;
use crate::world::cars::car_mesh::{create_procedural_car, sample_car_color};
use crate::world::cars::car_render::CarInstance;
use crate::world::cars::car_structs::{Car, CarChunkDistance, CarId, CarStorage};
use crate::world::roads::road_structs::NodeId;
use crate::world::roads::roads::RoadManager;
use crate::world::terrain::terrain_subsystem::{CursorMode, TerrainSubsystem};
use glam::{Mat4, Vec3};
use rand::rngs::ThreadRng;
use rand::{RngExt, rng};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, IndexFormat, Queue, RenderPass};

pub const CAR_BASE_LENGTH: f32 = 4.5;
pub const CAR_BASE_WIDTH: f32 = 2.5;

pub struct SpawningNode {
    pub node_id: NodeId,
    pub spawn_accumulator: f32, // in seconds
}

impl SpawningNode {
    fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            spawn_accumulator: 0.0,
        }
    }
}

struct CarSubsystemTiming {
    carchunk_update_last: Instant,
}

impl CarSubsystemTiming {
    fn new() -> CarSubsystemTiming {
        Self {
            carchunk_update_last: Instant::now(),
        }
    }
}

pub struct CarRenderSubsystem {
    pub device: Device,
    pub queue: Queue,
    pub vb: Buffer,
    pub ib: Buffer,
    pub instance_buf: Buffer,
    pub index_count: u32,
    pub current_instance_count: u32,

    // NEW: map from car id -> previous model matrix (column-major [[f32;4];4])
    prev_models: HashMap<u64, [[f32; 4]; 4]>,
}

impl CarRenderSubsystem {
    pub fn new(device: &Device, queue: &Queue, rt_subsystem: &mut RTSubsystem) -> Self {
        // Calculate exact sizes from procedural mesh (generate temporarily, discard data)
        let (vertices, indices) = create_procedural_car();
        let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();

        // Build BLAS for car mesh
        rt_subsystem.init_car_blas(&device, &queue, &positions, &indices);
        let vb_size = (vertices.len() * size_of::<Vertex>()) as u64;
        let ib_size = (indices.len() * size_of::<u32>()) as u64;

        let vb = device.create_buffer(&BufferDescriptor {
            label: Some("Cars Vertex Buffer (uninitialized)"),
            size: vb_size,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ib = device.create_buffer(&BufferDescriptor {
            label: Some("Cars IB"),
            size: ib_size,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let initial_instances = 16384;
        let instance_size = (initial_instances * size_of::<CarInstance>()) as u64;

        let instance_buf = device.create_buffer(&BufferDescriptor {
            label: Some("Cars Instance Buffer"),
            size: instance_size,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // UPLOAD THE FUCKING DATA
        queue.write_buffer(&vb, 0, bytemuck::cast_slice(&vertices));
        queue.write_buffer(&ib, 0, bytemuck::cast_slice(&indices));

        Self {
            device: device.clone(),
            queue: queue.clone(),
            vb,
            ib,
            instance_buf,
            index_count: indices.len() as u32,
            current_instance_count: 0,
            prev_models: HashMap::with_capacity(initial_instances),
        }
    }

    pub fn render(
        &mut self,
        pipelines: &Pipelines,
        rt_subsystem: &mut RTSubsystem,
        car_storage: &CarStorage,
        camera: &Camera,
        pass: &mut RenderPass,
    ) {
        let mut close_ids = car_storage.car_chunks().close_cars();
        close_ids.sort_unstable();

        let mut instances: Vec<CarInstance> = Vec::with_capacity(close_ids.len());

        for &car_id in &close_ids {
            if let Some(car) = car_storage.get(car_id) {
                let render_pos = car.pos.to_render_pos(camera.eye_world(), camera.chunk_size);

                let quat = car.quat.normalize();

                let length_scale = car.length / CAR_BASE_LENGTH;
                let width_scale = car.width / CAR_BASE_WIDTH;
                let scale = Vec3::new(width_scale, 1.0, length_scale);
                let model_mat: Mat4 =
                    Mat4::from_scale_rotation_translation(scale, quat, render_pos);

                let key = car_id as u64;
                let prev_mat_array = self
                    .prev_models
                    .get(&key)
                    .copied()
                    .unwrap_or_else(|| model_mat.to_cols_array_2d());

                instances.push(CarInstance {
                    model: model_mat.to_cols_array_2d(),
                    prev_model: prev_mat_array,
                    color: car.color,
                    _pad: 0.0,
                });

                // Update stored previous matrix for next frame
                self.prev_models.insert(key, model_mat.to_cols_array_2d());
            }
        }

        update_rt_instances(
            rt_subsystem,
            &self.device,
            &self.queue,
            pipelines,
            car_storage,
            camera,
        );

        let instance_count = instances.len() as u32;

        if instance_count == 0 {
            return;
        }

        // Smart buffer resize — only recreate when too small, double size for growth
        let needed_bytes = (instances.len() * size_of::<CarInstance>()) as u64;
        if needed_bytes > self.instance_buf.size() {
            let new_size = (needed_bytes * 2).next_multiple_of(256);
            self.instance_buf = self.device.create_buffer(&BufferDescriptor {
                label: Some("Car Instance Buffer (resized)"),
                size: new_size,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        // Upload instance data
        self.queue
            .write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(&instances));

        self.current_instance_count = instance_count;

        // Bind and draw — one call for ALL close cars
        pass.set_vertex_buffer(0, self.vb.slice(..));
        pass.set_vertex_buffer(1, self.instance_buf.slice(..));
        pass.set_index_buffer(self.ib.slice(..), IndexFormat::Uint32);

        pass.draw_indexed(0..self.index_count, 0, 0..instance_count);
    }
}

pub struct CarSubsystem {
    car_storage: CarStorage,
    spawning_nodes: Vec<SpawningNode>,
    timing: CarSubsystemTiming,
    player_car_id: CarId,
}

impl CarSubsystem {
    pub(crate) fn player_car_id(&self) -> CarId {
        self.player_car_id
    }
    pub(crate) fn get_player_car(&mut self) -> Option<&mut Car> {
        let id = self.player_car_id;
        self.car_storage_mut().get_mut(id)
    }
}

impl CarSubsystem {
    pub fn new() -> Self {
        Self {
            car_storage: CarStorage::new(),
            spawning_nodes: vec![],
            timing: CarSubsystemTiming::new(),
            player_car_id: 1,
        }
    }
    pub fn car_storage(&self) -> &CarStorage {
        &self.car_storage
    }
    pub fn car_storage_mut(&mut self) -> &mut CarStorage {
        &mut self.car_storage
    }
    pub fn update(
        &mut self,
        road_manager: &RoadManager,
        terrain_renderer: &TerrainSubsystem,
        input_state: &mut InputState,
        time_system: &TimeSystem,
        variables: &mut UiVariableRegistry,
        target_pos: WorldPos,
    ) {
        variables.set_i32("car_count", self.car_storage.car_count() as i32);
        self.car_storage
            .update_target_and_chunk_size(target_pos.chunk, terrain_renderer.chunk_size);
        self.spawn_cars(road_manager, terrain_renderer, target_pos, time_system);

        match terrain_renderer.cursor.mode {
            CursorMode::Cars => {
                if let Some(picked) = &terrain_renderer.last_picked {
                    if input_state.gameplay_repeat("Place Car") {
                        let mut rng = rng();
                        let car = make_random_car(picked.pos, &mut rng);
                        self.car_storage
                            .spawn(picked.pos.chunk, CarChunkDistance::Close, car);
                    }
                }
            }
            _ => {}
        }
        if self.timing.carchunk_update_last.elapsed() >= Duration::from_secs(1) {
            self.timing.carchunk_update_last = Instant::now();
            self.car_storage
                .update_carchunk_distances(target_pos, terrain_renderer.chunk_size);
        }
    }

    fn spawn_cars(
        &mut self,
        road_manager: &RoadManager,
        terrain_renderer: &TerrainSubsystem,
        target_pos: WorldPos,
        time_system: &TimeSystem,
    ) {
        let mut rng = rng();
        let dt = time_system.target_sim_dt;
        let mut to_remove: Vec<usize> = Vec::new();
        for (idx, spawning_node) in self.spawning_nodes.iter_mut().enumerate() {
            let Some(node) = road_manager.roads.node(spawning_node.node_id) else {
                continue;
            };

            if !node.is_enabled() {
                to_remove.push(idx);
                continue;
            }

            let spawning_rate = node.car_spawning_rate(); // Cars per minute: f32
            if !(spawning_rate > 0.0) {
                to_remove.push(idx);
                continue;
            }

            let seconds_per_car = 60.0 / spawning_rate;
            spawning_node.spawn_accumulator += dt;

            // Clamp to prevent burst-spawning after lag spikes or large dt
            let max_accumulator = seconds_per_car * 2.0; // spawn at most 2 cars per update
            spawning_node.spawn_accumulator = spawning_node.spawn_accumulator.min(max_accumulator);

            while spawning_node.spawn_accumulator >= seconds_per_car {
                spawning_node.spawn_accumulator -= seconds_per_car;

                // Pick ONE lane, don't spawn on ALL outgoing lanes
                let outgoing = node.outgoing_lanes();
                if outgoing.is_empty() {
                    break;
                }
                let lane_id = outgoing[rng.random_range(0..outgoing.len())];
                let lane = road_manager.roads.lane(&lane_id);
                let polyline = lane.polyline();
                let first_point = polyline.first().unwrap();
                let car_chunk_distance = CarChunkDistance::from_chunk_positions(
                    target_pos.chunk,
                    first_point.chunk,
                    terrain_renderer.chunk_size,
                );
                self.car_storage.spawn(
                    first_point.chunk,
                    car_chunk_distance,
                    make_random_car(*first_point, &mut rng),
                );
            }
        }
        for idx in to_remove.into_iter().rev() {
            self.spawning_nodes.remove(idx);
        }
    }
    pub fn add_spawning_node(&mut self, id: NodeId) {
        self.spawning_nodes.push(SpawningNode::new(id))
    }
}
pub fn make_random_car(position: WorldPos, mut rng: &mut ThreadRng) -> Car {
    let mut car = Car::default();
    car.pos = position;
    let hsv = sample_car_color(&mut rng);
    let rgb = hsv_to_rgb(hsv);

    car.color = Vec3::new(rgb[0], rgb[1], rgb[2]).to_array();
    let random_length_scale = rng.random_range(0.8..1.3);
    let random_width_scale = rng.random_range(0.8..1.2);
    car.length = car.length * random_length_scale;
    car.width = car.width * random_width_scale;
    car
}
