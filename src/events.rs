#![allow(dead_code)]

use crate::world::cars::car_structs::CarChunk;
use crate::world::terrain::terrain_subsystem::CursorMode;

#[derive(Debug, Clone)]
pub enum Event {
    SetCursorMode(CursorMode),
    ToggleSimulation,
    CarNavigate(Vec<CarChunk>),
}

#[derive(Default)]
pub struct Events {
    write: Vec<Event>,
    read: Vec<Event>,
}

impl Events {
    pub fn new() -> Self {
        Self {
            write: Vec::with_capacity(64),
            read: Vec::with_capacity(64),
        }
    }

    /// Push an event for the *next* tick
    pub fn send(&mut self, event: Event) {
        self.write.push(event);
    }

    /// Call once per frame / tick
    pub fn flip(&mut self) {
        self.read.clear();
        std::mem::swap(&mut self.read, &mut self.write);
    }

    /// Drain events for this tick
    pub fn drain(&mut self) -> impl Iterator<Item = Event> + '_ {
        self.read.drain(..)
    }
}
