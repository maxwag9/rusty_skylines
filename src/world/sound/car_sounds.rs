use crate::helpers::positions::WorldPos;
use crate::world::camera::Camera;
use crate::world::cars::car_structs::CarStorage;
use crate::world::sound::AudioState;
use crate::world::terrain::terrain_subsystem::Terrain;
use glam::Vec3;
use std::sync::MutexGuard;

#[derive(Clone, Copy)]
pub struct CarAudioState {
    pub position: WorldPos,
    pub velocity: Vec3,
    pub rpm: f32,
    pub phase: f32,
}

pub fn collect_car_audio(
    state: &mut MutexGuard<AudioState>,
    camera: &Camera,
    terrain: &Terrain,
    car_storage: &CarStorage,
) {
    const MAX_DISTANCE: f64 = 300.0;
    const MAX_CARS: usize = 32;

    for car in car_storage.iter_cars() {
        let Some(car) = car else { continue };

        if state.cars.len() >= MAX_CARS {
            break;
        }

        let distance = car.pos.distance_to(camera.eye_world());

        if distance > MAX_DISTANCE {
            continue;
        }

        state.cars.push(CarAudioState {
            position: car.pos,
            velocity: car.current_velocity,
            rpm: car.engine_rpm,
            phase: rand::random::<f32>(),
        });
    }
}
