#![allow(dead_code)]

#[derive(Debug, Clone)]
pub enum Event {
    ToggleSimulation,
}

#[derive(Default)]
pub struct Events {
    queue: Vec<Event>,
}

impl Events {
    pub fn new() -> Self {
        Self { queue: Vec::new() }
    }

    pub fn send(&mut self, event: Event) {
        self.queue.push(event);
    }

    pub fn drain(&mut self) -> impl Iterator<Item = Event> + '_ {
        self.queue.drain(..)
    }
}
