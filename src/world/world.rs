use crate::data::Settings;
use crate::resources::TimeSystem;
use crate::ui::ui_editor::UiButtonLoader;
use crate::ui::variables::update_ui_variables;
use crate::world::astronomy::{AstronomyState, ObserverParams, TimeScales, compute_astronomy};
use crate::world::camera::{Camera, CameraController};
use glam::Mat4;
use std::collections::HashMap;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Entity(u32);

pub struct CameraBundle {
    pub camera: Camera,
    pub controller: CameraController,
}

pub struct WorldState {
    next_entity: u32,
    main_camera: Entity,
    cameras: HashMap<Entity, CameraBundle>,
    pub(crate) astronomy: AstronomyState,
}

impl WorldState {
    pub fn new() -> Self {
        let mut world = Self {
            next_entity: 0,
            main_camera: Entity(0),
            cameras: HashMap::new(),
            astronomy: AstronomyState::default(),
        };
        let camera = world.spawn_camera(Camera::new());
        world.main_camera = camera;
        world
    }

    pub fn spawn_camera(&mut self, camera: Camera) -> Entity {
        let entity = self.spawn_entity();
        let controller = CameraController::new(&camera);
        self.cameras
            .insert(entity, CameraBundle { camera, controller });
        entity
    }

    fn spawn_entity(&mut self) -> Entity {
        let id = self.next_entity;
        self.next_entity += 1;
        Entity(id)
    }

    pub fn main_camera(&self) -> Entity {
        self.main_camera
    }

    pub fn camera(&self, entity: Entity) -> Option<&Camera> {
        self.cameras.get(&entity).map(|bundle| &bundle.camera)
    }

    pub fn camera_mut(&mut self, entity: Entity) -> Option<&mut Camera> {
        self.cameras
            .get_mut(&entity)
            .map(|bundle| &mut bundle.camera)
    }

    pub fn camera_controller_mut(&mut self, entity: Entity) -> Option<&mut CameraController> {
        self.cameras
            .get_mut(&entity)
            .map(|bundle| &mut bundle.controller)
    }

    pub fn camera_and_controller_mut(&mut self, entity: Entity) -> Option<&mut CameraBundle> {
        self.cameras.get_mut(&entity)
    }

    pub fn update(
        &mut self,
        ui_loader: &mut UiButtonLoader,
        time: &TimeSystem,
        settings: &Settings,
        proj: Mat4,
    ) {
        let time_scales = TimeScales::from_game_time(time.total_game_time, settings.always_day);
        let observer = ObserverParams::new(time_scales.day_angle);
        let astronomy = compute_astronomy(&time_scales, proj);

        update_ui_variables(
            ui_loader,
            &time_scales,
            &astronomy,
            observer.obliquity,
            settings,
        );
        self.astronomy = astronomy;
    }
}
