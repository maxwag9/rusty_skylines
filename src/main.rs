use glam::{Mat4, Vec3};
use std::time::Instant;
use winit::event_loop::{ControlFlow, EventLoop};

mod app;
mod camera;
mod input;
mod state;
mod vertex;

mod data;
mod renderer;
mod simulation;

use app::App;
use camera::Camera;
use input::{InputState, MouseState};

pub struct FrameTimer {
    last: Instant,
    start: Instant,
    total: f32,
}

impl FrameTimer {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            last: now,
            start: now,
            total: 0.0,
        }
    }

    /// Returns the delta time in seconds since last frame.
    pub fn dt(&mut self) -> f32 {
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32();
        self.last = now;
        self.total += dt;
        dt
    }

    /// Returns the total elapsed time since creation, in seconds.
    pub fn total_time(&self) -> f32 {
        self.total
    }

    /// Optionally reset both counters.
    pub fn reset(&mut self) {
        self.start = Instant::now();
        self.last = self.start;
        self.total = 0.0;
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
}

impl Uniforms {
    fn new() -> Self {
        // Eye position: x, y, z
        let eye = Vec3::new(5.0, 15.0, 0.0); // elevated above origin
        let target = Vec3::new(0.0, 0.0, 0.0); // looking at center
        let up = Vec3::Y; // y-axis = up

        let view = Mat4::look_at_rh(eye, target, up);
        let proj = Mat4::perspective_rh_gl(45f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
        Self {
            view_proj: (proj * view).to_cols_array_2d(),
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll); // <- continuous
    let mut app = App::default();
    event_loop
        .run_app(&mut app)
        .expect("Failed to run event_loop application");
}
