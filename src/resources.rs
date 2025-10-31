use crate::data::Settings;
use crate::events::Events;
use crate::renderer::Renderer;
use crate::renderer::ui_editor::UiButtonLoader;
use crate::simulation::Simulation;
use glam::{Mat4, Vec2, Vec3};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use winit::window::Window;

pub struct Resources {
    pub settings: Settings,
    pub timing: TimingData,
    pub time: Time,
    pub input: InputState,
    pub mouse: MouseState,
    pub renderer: Renderer,
    pub simulation: Simulation,
    pub ui_loader: UiButtonLoader,
    pub events: Events,
    pub window: Arc<Window>,
}

impl Resources {
    pub fn new(window: Arc<Window>) -> Self {
        let settings = Settings::load("src/settings.toml");
        let renderer = Renderer::new(window.clone());
        Self {
            settings,
            timing: TimingData::default(),
            time: Time::new(),
            input: InputState::new(),
            mouse: MouseState::new(),
            renderer,
            simulation: Simulation::new(),
            ui_loader: UiButtonLoader::new(),
            events: Events::new(),
            window,
        }
    }

    pub fn update_time(&mut self) -> f32 {
        let dt = self.time.update();
        self.timing.sim_dt = dt;
        self.timing.render_dt = dt;
        self.timing.render_fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
        dt
    }
}

#[derive(Debug, Default, Clone)]
pub struct TimingData {
    pub sim_dt: f32,
    pub render_dt: f32,
    pub render_fps: f32,
}

pub struct Time {
    last: Instant,
    pub delta: f32,
    pub total: f32,
}

impl Time {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            last: now,
            delta: 0.0,
            total: 0.0,
        }
    }

    pub fn update(&mut self) -> f32 {
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32();
        self.last = now;
        self.delta = dt;
        self.total += dt;
        dt
    }
}

pub struct InputState {
    pressed: HashSet<String>,
    pub shift_pressed: bool,
    pub ctrl_pressed: bool,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            pressed: HashSet::new(),
            shift_pressed: false,
            ctrl_pressed: false,
        }
    }

    pub fn set_key(&mut self, key: &str, down: bool) {
        if down {
            self.pressed.insert(key.to_string());
        } else {
            self.pressed.remove(key);
        }
    }

    pub fn pressed(&self, key: &str) -> bool {
        self.pressed.contains(key)
    }
}

#[derive(Debug, Clone)]
pub struct MouseState {
    pub last_pos: Option<Vec2>,
    pub pos: Vec2,
    pub middle_pressed: bool,
    pub left_pressed: bool,
    pub right_pressed: bool,
    pub back_pressed: bool,
    pub forward_pressed: bool,
}

impl MouseState {
    pub fn new() -> Self {
        Self {
            last_pos: None,
            pos: Vec2::ZERO,
            middle_pressed: false,
            left_pressed: false,
            right_pressed: false,
            back_pressed: false,
            forward_pressed: false,
        }
    }
}

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

    pub fn dt(&mut self) -> f32 {
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32();
        self.last = now;
        self.total += dt;
        dt
    }

    pub fn total_time(&self) -> f32 {
        self.total
    }

    pub fn reset(&mut self) {
        self.start = Instant::now();
        self.last = self.start;
        self.total = 0.0;
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub view_proj: [[f32; 4]; 4],
}

impl Uniforms {
    pub fn new() -> Self {
        let eye = Vec3::new(5.0, 15.0, 0.0);
        let target = Vec3::ZERO;
        let up = Vec3::Y;

        let view = Mat4::look_at_rh(eye, target, up);
        let proj = Mat4::perspective_rh_gl(45f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
        Self {
            view_proj: (proj * view).to_cols_array_2d(),
        }
    }
}
