use glam::{Mat4, Vec3};
use std::time::Instant;
use winit::event_loop::{ControlFlow, EventLoop};

mod app;
mod camera;
mod input;
mod state;
mod vertex;

mod renderer;
mod simulation_controls;
mod ui;

use app::App;
use camera::Camera;
use input::{InputState, MouseState};

struct FrameTimer {
    last: Instant,
}
impl FrameTimer {
    fn new() -> Self {
        Self {
            last: Instant::now(),
        }
    }
    fn dt(&mut self) -> f32 {
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32();
        self.last = now;
        dt
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
