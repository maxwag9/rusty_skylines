#![allow(dead_code)]

use glam::Vec3;

#[derive(Debug, Clone, Default)]
pub struct Rigidbody {
    pub velocity: Vec3,
    pub mass: f32,
}
