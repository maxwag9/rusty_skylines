use crate::cars::car_mesh::{create_procedural_car, sample_car_color};
use crate::cars::car_render::CarInstance;
use crate::cars::car_structs::{Car, CarChunkDistance, CarStorage};
use crate::components::camera::Camera;
use crate::hsv::hsv_to_rgb;
use crate::positions::WorldPos;
use crate::renderer::world_renderer::{CursorMode, TerrainRenderer};
use crate::resources::TimeSystem;
use crate::ui::input::InputState;
use crate::ui::vertex::Vertex;
use glam::{Mat4, Vec3};
use rand::prelude::IndexedRandom;
use rand::{RngExt, rng};
use std::time::{Duration, Instant};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, IndexFormat, Queue, RenderPass};

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
pub struct CarSubsystemRender {
    pub device: Device,
    pub queue: Queue,
    pub vb: Buffer,
    pub ib: Buffer,
    pub instance_buf: Buffer,
    pub index_count: u32,
    pub current_instance_count: u32,
}
impl CarSubsystemRender {
    pub fn new(device: Device, queue: Queue) -> Self {
        // Calculate exact sizes from procedural mesh (generate temporarily, discard data)
        let (vertices, indices) = create_procedural_car();
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
            label: Some("Cars IB"),
            size: instance_size,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // UPLOAD THE FUCKING DATA
        queue.write_buffer(&vb, 0, bytemuck::cast_slice(&vertices));
        queue.write_buffer(&ib, 0, bytemuck::cast_slice(&indices));
        Self {
            device,
            queue,
            vb,
            ib,
            instance_buf,
            index_count: indices.len() as u32,
            current_instance_count: 0,
        }
    }
}

pub struct CarSubsystem {
    car_storage: CarStorage,
    timing: CarSubsystemTiming,
    render: CarSubsystemRender,
}
impl CarSubsystem {
    pub fn new(device: &Device, queue: &Queue) -> Self {
        Self {
            car_storage: CarStorage::new(),
            timing: CarSubsystemTiming::new(),
            render: CarSubsystemRender::new(device.clone(), queue.clone()),
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
        terrain_renderer: &TerrainRenderer,
        input_state: &mut InputState,
        time_system: &TimeSystem,
        target_pos: WorldPos,
    ) {
        match terrain_renderer.cursor.mode {
            CursorMode::Cars => {
                if let Some(picked) = &terrain_renderer.last_picked {
                    if input_state.gameplay_repeat("Place Car") {
                        let mut car = Car::default();
                        car.pos = picked.pos;
                        let mut rng = rng();
                        let hsv = sample_car_color(&mut rng);
                        let rgb = hsv_to_rgb(hsv);

                        car.color = Vec3::new(rgb[0], rgb[1], rgb[2]).to_array();
                        let random_length_scale = rng.random_range(0.8..1.3);
                        let random_width_scale = rng.random_range(0.8..1.2);
                        car.length = car.length * random_length_scale;
                        car.width = car.width * random_width_scale;
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

    pub fn render(&mut self, camera: &Camera, pass: &mut RenderPass) {
        let render = &mut self.render;
        const BASE_LENGTH: f32 = 4.5;
        const BASE_WIDTH: f32 = 2.5;
        let close_ids = self.car_storage.car_chunks().close_cars();

        let mut instances: Vec<CarInstance> = Vec::with_capacity(close_ids.len());

        for &car_id in &close_ids {
            if let Some(car) = self.car_storage.get(car_id) {
                let render_pos = car.pos.to_render_pos(camera.eye_world(), camera.chunk_size);

                let quat = car.quat.normalize();

                let length_scale = car.length / BASE_LENGTH;
                let width_scale = car.width / BASE_WIDTH;
                let scale = Vec3::new(width_scale, 1.0, length_scale);
                let model = Mat4::from_scale_rotation_translation(scale, quat, render_pos);

                instances.push(CarInstance {
                    model: model.to_cols_array_2d(),
                    color: car.color,
                    _pad: 0.0,
                });
            }
        }

        let instance_count = instances.len() as u32;

        if instance_count == 0 {
            return;
        }

        // Smart buffer resize — only recreate when too small, double size for growth
        let needed_bytes = (instances.len() * size_of::<CarInstance>()) as u64;
        if needed_bytes > render.instance_buf.size() {
            let new_size = (needed_bytes * 2).next_multiple_of(256);
            render.instance_buf = render.device.create_buffer(&BufferDescriptor {
                label: Some("Car Instance Buffer (resized)"),
                size: new_size,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        // Upload instance data
        render
            .queue
            .write_buffer(&render.instance_buf, 0, bytemuck::cast_slice(&instances));

        render.current_instance_count = instance_count;

        // Bind and draw — one call for ALL close cars
        pass.set_vertex_buffer(0, render.vb.slice(..));
        pass.set_vertex_buffer(1, render.instance_buf.slice(..));
        pass.set_index_buffer(render.ib.slice(..), IndexFormat::Uint32);

        pass.draw_indexed(0..render.index_count, 0, 0..instance_count);
    }
}
