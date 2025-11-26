use crate::data::Settings;
use crate::events::Events;
use crate::renderer::Renderer;
pub(crate) use crate::renderer::input::InputState;
use crate::renderer::ui_editor::UiButtonLoader;
use crate::simulation::Simulation;
use glam::{Mat4, Vec2, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

pub struct Resources {
    pub settings: Settings,
    pub time: TimeSystem,
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
        let editor_mode = settings.editor_mode.clone();
        let renderer = Renderer::new(window.clone(), &settings);
        Self {
            settings,
            time: TimeSystem::new(),
            input: InputState::new(),
            mouse: MouseState::new(),
            renderer,
            simulation: Simulation::new(),
            ui_loader: UiButtonLoader::new(editor_mode),
            events: Events::new(),
            window,
        }
    }

    pub fn update_sim_time(&mut self) {
        self.time.update_sim();
    }

    pub fn update_render_time(&mut self) {
        self.time.update_render();
    }
}

#[derive(Debug, Clone)]
pub struct TimeSystem {
    pub last_sim: Instant,
    pub last_render: Instant,
    pub start: Instant,

    pub sim_dt: f32,
    pub sim_accumulator: f32,
    pub sim_target_step: f32,

    pub render_dt: f32,
    pub render_fps: f32,
    pub target_fps: f32,
    pub target_frametime: f32,
    pub total_time: f32,
}

impl TimeSystem {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            last_sim: now,
            last_render: now,
            start: now,

            sim_dt: 0.0,
            sim_accumulator: 0.0,
            sim_target_step: 0.0,

            render_dt: 0.0,
            render_fps: 0.0,
            target_fps: 100.0,
            target_frametime: 0.0,
            total_time: 0.0,
        }
    }

    pub fn set_tps(&mut self, tps: f32) {
        self.sim_target_step = 1.0 / tps;
        self.sim_accumulator = 0.0;
    }

    pub fn set_fps(&mut self, target_fps: f32) {
        self.target_fps = target_fps;
        self.target_frametime = 1.0 / target_fps;
    }

    pub fn update_sim(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_sim).as_secs_f32();
        self.last_sim = now;
        self.sim_dt = dt;
        self.total_time += dt;
    }

    pub fn update_render(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_render).as_secs_f32();
        self.last_render = now;
        self.render_dt = dt;
        self.render_fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
        self.sim_accumulator += dt;
    }
}

#[derive(Serialize, Deserialize)]
pub struct SerializableKeybind {
    pub key: String,
    pub action: String,
}

pub struct KeyRepeatState {
    pub first_press_time: f32,
    pub last_repeat_time: f32,
    pub phase: u8, // 0 = fresh, 1 = warmup, 2 = fast
}

#[derive(Debug, Clone)]
pub struct KeyState {
    pub pressed: bool,
    pub pressed_once: bool,
    pub repeated: bool,
    pub first_press: f32,
    pub last_repeat: f32,
}

impl KeyState {
    pub fn new() -> Self {
        Self {
            pressed: false,
            pressed_once: false,
            repeated: false,
            first_press: -1.0,
            last_repeat: -1.0,
        }
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

    pub left_just_pressed: bool,
    pub left_just_released: bool,
    pub right_just_pressed: bool,
    pub right_just_released: bool,

    pub scroll_delta: Vec2,
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

            left_just_pressed: false,
            left_just_released: false,
            right_just_pressed: false,
            right_just_released: false,

            scroll_delta: Vec2::ZERO,
        }
    }

    pub fn update_just_states(&mut self) {
        // Reset per-frame flags
        self.left_just_pressed = false;
        self.left_just_released = false;
        self.right_just_pressed = false;
        self.right_just_released = false;
        self.scroll_delta = Vec2::ZERO;
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

fn default_keybinds() -> HashMap<PhysicalKey, String> {
    use winit::keyboard::{KeyCode, PhysicalKey};

    let mut m = HashMap::new();

    m.insert(PhysicalKey::Code(KeyCode::KeyW), "editor.move_up".into());
    m.insert(PhysicalKey::Code(KeyCode::KeyA), "editor.move_left".into());
    m.insert(PhysicalKey::Code(KeyCode::KeyS), "editor.move_down".into());
    m.insert(PhysicalKey::Code(KeyCode::KeyD), "editor.move_right".into());
    m.insert(PhysicalKey::Code(KeyCode::Space), "simulation.pause".into());
    m.insert(PhysicalKey::Code(KeyCode::Escape), "editor.cancel".into());

    m
}
