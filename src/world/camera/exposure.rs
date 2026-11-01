use std::path::PathBuf;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::*;
use wgpu_render_manager::compute_system::{BufferSet, ComputePipelineOptions};
use wgpu_render_manager::renderer::RenderManager;

use crate::helpers::modpack::ModManager;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ExposureState {
    pub exposure: f32,
    pub average_log_luminance: f32,
    pub target_exposure: f32,
    pub _padding: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AutoExposureParams {
    delta_time: f32,
    key_value: f32,
    min_exposure: f32,
    max_exposure: f32,
    bright_scene_speed: f32,
    dark_scene_speed: f32,
    padding: [f32; 2],
}

pub struct AutoExposure {
    states: [Buffer; 2],
    current_index: usize,
    params_buffer: Buffer,
    shader_path: PathBuf,
    last_frame: Instant,
}

impl AutoExposure {
    pub fn new(device: &Device, _queue: &Queue, mod_manager: &ModManager) -> Self {
        let shader_path = mod_manager
            .resource_path("shaders/compute/auto_exposure.wgsl")
            .expect("Missing shader 'shaders/compute/auto_exposure.wgsl'");

        let initial_state = ExposureState {
            exposure: 1.0,
            average_log_luminance: 0.0,
            target_exposure: 1.0,
            _padding: 0.0,
        };

        let states = [
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Auto Exposure State A"),
                contents: bytemuck::bytes_of(&initial_state),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            }),
            device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Auto Exposure State B"),
                contents: bytemuck::bytes_of(&initial_state),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            }),
        ];

        let initial_params = AutoExposureParams {
            delta_time: 1.0 / 60.0,
            key_value: 0.18,
            min_exposure: 0.025,
            max_exposure: 32.0,
            bright_scene_speed: 5.0,
            dark_scene_speed: 1.25,
            padding: [0.0; 2],
        };

        let params_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Auto Exposure Parameters"),
            contents: bytemuck::bytes_of(&initial_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        Self {
            states,
            current_index: 0,
            params_buffer,
            shader_path: shader_path.to_path_buf(),
            last_frame: Instant::now(),
        }
    }

    pub fn current_buffer(&self) -> &Buffer {
        &self.states[self.current_index]
    }

    pub fn encode(
        &mut self,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        hdr_view: &TextureView,
        render_manager: &mut RenderManager,
    ) {
        let now = Instant::now();
        let delta_time = now
            .duration_since(self.last_frame)
            .as_secs_f32()
            .clamp(1.0 / 240.0, 0.1);

        self.last_frame = now;

        let params = AutoExposureParams {
            delta_time,
            key_value: 0.18,
            min_exposure: 0.055,
            max_exposure: 12.0,
            bright_scene_speed: 5.0,
            dark_scene_speed: 1.25,
            padding: [0.0; 2],
        };

        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        let next_index = 1 - self.current_index;

        render_manager.compute(
            Some(encoder),
            "Auto Exposure",
            vec![hdr_view],
            Vec::new(),
            &self.shader_path,
            ComputePipelineOptions {
                dispatch_size: [1, 1, 1],
            },
            &[
                BufferSet::from_storage(&self.states[self.current_index], true),
                BufferSet::from_storage(&self.states[next_index], false),
                BufferSet::from_uniform(&self.params_buffer),
            ],
        );
    }

    pub fn advance(&mut self) {
        self.current_index = 1 - self.current_index;
    }
}
