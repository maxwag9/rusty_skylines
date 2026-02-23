use crate::helpers::positions::{ChunkSize, WorldPos};
use crate::resources::Resources;
use crate::world::sound::car_sounds::{CarAudioState, collect_car_audio};
use cpal::Stream;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::f32::consts::PI;
use std::sync::{Arc, Mutex};

mod car_sounds;

pub fn run_sounds(resources: &mut Resources) {
    let sounds = &mut resources.sounds;
    let mut state = sounds.state.lock().unwrap();

    let camera = &resources.world_core.world_state.camera;
    let terrain = &resources.world_core.terrain;
    let car_storage = resources.world_core.cars.car_storage();

    let current_eye = camera.eye_world();
    let prev_eye = camera.prev_eye_world();

    let vel = current_eye.delta_to(prev_eye, terrain.chunk_size);

    state.listener_pos = current_eye;
    state.listener_velocity = vel;
    state.listener_yaw = camera.yaw;
    state.listener_pitch = camera.pitch;
    state.chunk_size = camera.chunk_size;
    state.cars.clear();

    collect_car_audio(&mut state, camera, terrain, car_storage);
}

pub struct AudioState {
    pub cars: Vec<CarAudioState>,
    pub listener_pos: WorldPos,
    pub listener_velocity: glam::Vec3,
    pub listener_yaw: f32,
    pub listener_pitch: f32,
    pub chunk_size: ChunkSize,
}
impl AudioState {
    pub fn clear(&mut self) {
        self.cars.clear();
    }
}
pub struct Sounds {
    stream: Stream,
    pub state: Arc<Mutex<AudioState>>,
}
impl Sounds {
    pub fn new() -> Self {
        let host = cpal::default_host();

        let device = host
            .output_devices()
            .unwrap()
            .find(|d| d.description().unwrap().name().contains("PipeWire"))
            .expect("No PipeWire device found");

        println!(
            "Using output device: {}",
            device.description().unwrap().name()
        );

        let config = device
            .default_output_config()
            .expect("Failed to get default config");

        let sample_rate = config.sample_rate() as f32;

        let state = Arc::new(Mutex::new(AudioState {
            cars: Vec::with_capacity(32),
            listener_pos: WorldPos::zero(),

            listener_velocity: Default::default(),
            listener_yaw: 0.0,
            listener_pitch: 0.0,
            chunk_size: 128,
        }));

        let state_clone = state.clone();

        let stream = device
            .build_output_stream(
                &config.into(),
                move |output: &mut [f32], _| {
                    let mut state = state_clone.lock().unwrap();

                    const SPEED_OF_SOUND: f32 = 343.0;

                    for frame in output.chunks_mut(2) {
                        let mut left = 0.0;
                        let mut right = 0.0;
                        // derive forward/right from yaw + pitch
                        let yaw = state.listener_yaw;
                        let pitch = state.listener_pitch;
                        let forward = glam::Vec3::new(
                            yaw.cos() * pitch.cos(),
                            pitch.sin(),
                            yaw.sin() * pitch.cos(),
                        )
                        .normalize();

                        let right_vec = forward.cross(glam::Vec3::Y).normalize();
                        let listener_pos = state.listener_pos;
                        let listener_velocity = state.listener_velocity;
                        let chunk_size = state.chunk_size;
                        for car in &mut state.cars {
                            // --- WORLD SPACE ---
                            let to_car = car.position.delta_to(listener_pos, chunk_size);
                            let distance = to_car.length().max(0.01);
                            let dir = to_car.normalize();

                            // --- DISTANCE ATTENUATION ---
                            let attenuation = 1.0 / (1.0 + 0.015 * distance * distance);

                            // --- STEREO PANNING ---
                            let pan = dir.dot(right_vec).clamp(-1.0, 1.0);
                            let pan_l = ((1.0 - pan) * 0.5).sqrt();
                            let pan_r = ((1.0 + pan) * 0.5).sqrt();

                            // --- DOPPLER ---
                            let rel_vel = car.velocity.dot(dir) - listener_velocity.dot(dir);

                            let doppler =
                                (SPEED_OF_SOUND / (SPEED_OF_SOUND - rel_vel)).clamp(0.5, 2.0);

                            // --- ENGINE SYNTH ---
                            let cylinders = 4.0;
                            let engine_freq = car.rpm / 60.0 * (cylinders * 0.5) * doppler;

                            car.phase += engine_freq / sample_rate;
                            if car.phase >= 1.0 {
                                car.phase -= 1.0;
                            }

                            let t = car.phase * 2.0 * PI;

                            let fundamental = t.sin();
                            let second = (2.0 * t).sin() * 0.5;
                            let third = (3.0 * t).sin() * 0.25;
                            let rough = (t * 13.0).sin() * 0.05;

                            let rpm_norm = (car.rpm / 7000.0).clamp(0.1, 1.0);

                            let mut engine =
                                (fundamental * 0.5 + second * 0.5 + third * 0.3 + rough) * rpm_norm;

                            engine = engine.tanh();

                            // --- SIMPLE HRTF HEAD SHADOW ---
                            let shadow = 1.0 - pan.abs() * 0.4;
                            engine *= shadow;

                            let sample = engine * attenuation * 0.7;

                            left += sample * pan_l;
                            right += sample * pan_r;
                        }

                        frame[0] = left;
                        frame[1] = right;
                    }
                },
                move |err| {
                    eprintln!("Audio error: {err}");
                },
                None,
            )
            .unwrap();

        stream.play().unwrap();

        Self { stream, state }
    }
}
