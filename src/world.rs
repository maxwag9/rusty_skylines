use crate::components::camera::{Camera, CameraController};
use std::collections::HashMap;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Entity(u32);

pub struct CameraBundle {
    pub camera: Camera,
    pub controller: CameraController,
}

pub struct World {
    next_entity: u32,
    main_camera: Entity,
    cameras: HashMap<Entity, CameraBundle>,
}

impl World {
    pub fn new() -> Self {
        let mut world = Self {
            next_entity: 0,
            main_camera: Entity(0),
            cameras: HashMap::new(),
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

    pub fn _camera_mut(&mut self, entity: Entity) -> Option<&mut Camera> {
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
}
