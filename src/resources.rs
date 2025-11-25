use crate::data::Settings;
use crate::events::Events;
use crate::renderer::Renderer;
use crate::renderer::ui_editor::UiButtonLoader;
use crate::simulation::Simulation;
use glam::{Mat4, Vec2, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use winit::keyboard::{NamedKey, PhysicalKey};
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

pub struct InputState {
    pub physical: HashMap<PhysicalKey, bool>,
    pub logical: HashMap<NamedKey, bool>,
    pub character: HashSet<String>, // for text keys only

    pub keybinds: HashMap<PhysicalKey, String>, // <key, action>
    pub shift_pressed: bool,
    pub ctrl_pressed: bool,

    // timing for repeat
    pub last_backspace_time: f32,
    pub last_char_time: f32,
    pub backspace_first_press: f32,
    pub char_first_press: f32,

    pub arrow_first_press: f32,
    pub last_arrow_time: f32,
}

impl InputState {
    pub fn new() -> Self {
        let mut s = Self {
            physical: HashMap::new(),
            logical: HashMap::new(),
            character: HashSet::new(),
            keybinds: HashMap::new(),
            shift_pressed: false,
            ctrl_pressed: false,
            last_backspace_time: -999.0,
            last_char_time: -999.0,
            backspace_first_press: -999.0,
            char_first_press: -999.0,

            arrow_first_press: -999.0,
            last_arrow_time: -999.0,
        };

        s.load_keybinds("keybinds.toml");
        s
    }

    pub fn arrow_tick(&mut self, now: f32) -> bool {
        let arrows = [
            NamedKey::ArrowLeft,
            NamedKey::ArrowRight,
            NamedKey::ArrowUp,
            NamedKey::ArrowDown,
        ];

        let any_pressed = arrows.iter().any(|k| self.pressed_logical(k));

        if !any_pressed {
            self.arrow_first_press = -999.0;
            self.last_arrow_time = -999.0;
            return false;
        }

        let initial_delay = 0.25;
        let warmup_time = 0.15;
        let warmup_interval = 0.07;
        let fast_interval = 0.03;

        let now_pressed = any_pressed;

        if self.arrow_first_press < 0.0 {
            self.arrow_first_press = now;
            self.last_arrow_time = now;
            return true;
        }

        let held_for = now - self.arrow_first_press;
        let since_last = now - self.last_arrow_time;

        if held_for < initial_delay {
            return false;
        }

        if held_for < initial_delay + warmup_time {
            if since_last >= warmup_interval {
                self.last_arrow_time = now;
                return true;
            }
            return false;
        }

        if since_last >= fast_interval {
            self.last_arrow_time = now;
            return true;
        }

        false
    }

    pub fn set_physical(&mut self, key: PhysicalKey, down: bool) {
        self.physical.insert(key, down);
    }

    /// Returns true when we should perform one backspace action this frame.
    pub fn backspace_tick(&mut self, now: f32) -> bool {
        let pressed = self.pressed_logical(&NamedKey::Backspace);

        // If not pressed, reset state.
        if !pressed {
            self.backspace_first_press = -999.0;
            self.last_backspace_time = -999.0;
            return false;
        }

        // Parameters for "satisfying" feel.
        let initial_delay = 0.25; // no repeat for first 250 ms
        let warmup_time = 0.15; // 150 ms warmup phase
        let warmup_interval = 0.07; // 70 ms between repeats during warmup
        let fast_interval = 0.03; // 30 ms between repeats in fast phase

        // First frame we see it pressed.
        if self.backspace_first_press < 0.0 {
            self.backspace_first_press = now;
            self.last_backspace_time = now;
            return true; // first backspace happens instantly
        }

        let held_for = now - self.backspace_first_press;

        // While key not held long enough: no repeat at all.
        if held_for < initial_delay {
            return false;
        }

        let since_last = now - self.last_backspace_time;

        // Warmup phase
        if held_for < initial_delay + warmup_time {
            if since_last >= warmup_interval {
                self.last_backspace_time = now;
                return true;
            }
            return false;
        }

        // Fast phase
        if since_last >= fast_interval {
            self.last_backspace_time = now;
            return true;
        }

        false
    }

    /// Returns true when we should insert characters for currently held text keys.
    pub fn char_tick(&mut self, now: f32) -> bool {
        let any_char_down = !self.character.is_empty();

        // No char pressed → reset state.
        if !any_char_down {
            self.char_first_press = -999.0;
            self.last_char_time = -999.0;
            return false;
        }

        // Params: same feel as backspace. You can tweak separately if you want.
        let initial_delay = 0.25;
        let warmup_time = 0.15;
        let warmup_interval = 0.07;
        let fast_interval = 0.03;

        // First time we see any char pressed.
        if self.char_first_press < 0.0 {
            self.char_first_press = now;
            self.last_char_time = now;
            return true; // first character immediately
        }

        let held_for = now - self.char_first_press;

        // Short tap → only first char, no repeat.
        if held_for < initial_delay {
            return false;
        }

        let since_last = now - self.last_char_time;

        // Warmup phase
        if held_for < initial_delay + warmup_time {
            if since_last >= warmup_interval {
                self.last_char_time = now;
                return true;
            }
            return false;
        }

        // Fast phase
        if since_last >= fast_interval {
            self.last_char_time = now;
            return true;
        }

        false
    }

    pub fn set_logical(&mut self, key: NamedKey, down: bool) {
        let prev = *self.logical.get(&key).unwrap_or(&false);
        self.logical.insert(key.clone(), prev);
        self.logical.insert(key.clone(), down);

        if down && !prev {
            self.logical.insert(key, true);
        } else {
            self.logical.insert(key, false);
        }
    }

    pub fn set_character(&mut self, ch: &str, down: bool) {
        if down {
            self.character.insert(ch.to_string());
        } else {
            self.character.remove(ch);
        }
    }

    pub fn just_pressed(&self, key: &NamedKey) -> bool {
        *self.logical.get(key).unwrap_or(&false)
    }

    pub fn pressed_physical(&self, key: &PhysicalKey) -> bool {
        *self.physical.get(key).unwrap_or(&false)
    }

    pub fn pressed_logical(&self, key: &NamedKey) -> bool {
        *self.logical.get(key).unwrap_or(&false)
    }

    pub fn pressed_char(&self, ch: &str) -> bool {
        self.character.contains(ch)
    }

    pub fn save_keybinds(&self, path: &str) {
        let toml = toml::to_string(&self.keybinds).expect("Failed to serialize keybinds");

        std::fs::write(path, toml).expect("Failed to write keybind file");
    }

    pub fn load_keybinds(&mut self, path: &str) {
        let data = match std::fs::read_to_string(path) {
            Ok(d) => d,
            Err(_) => {
                // file missing → load defaults
                self.keybinds = default_keybinds();

                // save defaults to disk
                self.save_keybinds(path);
                return;
            }
        };

        // try parsing file
        let parsed = toml::from_str::<HashMap<PhysicalKey, String>>(&data);

        match parsed {
            Ok(map) => {
                self.keybinds = map;
            }
            Err(_) => {
                // corrupted file → reset to defaults
                self.keybinds = default_keybinds();
                self.save_keybinds(path);
            }
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
