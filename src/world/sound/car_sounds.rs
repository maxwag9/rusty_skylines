use crate::helpers::positions::WorldPos;
use crate::world::camera::Camera;
use crate::world::cars::car_structs::CarStorage;
use crate::world::sound::MAX_CARS_AUDIO;
use crate::world::sound::sound::AudioState;
use crate::world::terrain::terrain_subsystem::Terrain;
use glam::Vec3;
use std::sync::MutexGuard;

#[derive(Default, Clone)]
pub struct CarAudioState {
    pub position: WorldPos,
    pub velocity: Vec3,
    pub rpm: f32,
    pub throttle: f32,
    pub doppler: f32,
    pub attenuation: f32,
    pub pan_l: f32,
    pub pan_r: f32,
}

pub fn collect_car_audio(
    state: &mut MutexGuard<AudioState>,
    camera: &Camera,
    terrain: &Terrain,
    car_storage: &mut CarStorage,
) {
    const MAX_DISTANCE: f64 = 300.0;

    for car in car_storage.iter_cars() {
        let Some(car) = car else { continue };

        if state.cars.len() >= MAX_CARS_AUDIO {
            break;
        }

        let distance = car.pos.distance_to(camera.eye_world());

        if distance > MAX_DISTANCE {
            continue;
        }

        //if car.physical_trajectory.is_none() {continue};

        state.cars.push(CarAudioState {
            position: car.pos,
            velocity: car.current_velocity,
            rpm: car.engine_rpm,
            throttle: car.throttle,
            doppler: 1.0,
            attenuation: 1.0,
            pan_l: 0.0,
            pan_r: 0.0,
        });
    }
}
