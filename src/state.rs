use crate::data::{Data, Settings};
use crate::renderer::Renderer;
use crate::renderer::ui_editor::UiButtonLoader;
use crate::simulation::Simulation;
use crate::{Camera, InputState, MouseState};
use glam::Vec3;
use std::sync::{Arc, Mutex};
use winit::window::Window;

pub(crate) struct State {
    _window: Arc<Window>,
    pub(crate) camera: Camera,
    pub(crate) input: InputState,
    pub(crate) mouse: MouseState,
    velocity: Vec3,
    pub(crate) zoom_vel: f32,
    pub(crate) target_yaw: f32,
    pub(crate) target_pitch: f32,
    orbit_smoothness: f32,
    pub(crate) yaw_velocity: f32,
    pub(crate) pitch_velocity: f32,
    orbit_damping_release: f32,
    zoom_damping: f32,
    pub renderer: Arc<Mutex<Renderer>>,
    simulation_running: bool,
    pub simulation: Arc<Mutex<Simulation>>,
    pub ui_loader: Arc<Mutex<UiButtonLoader>>,
    pub settings: Settings,
}

impl State {
    pub fn new(window: Arc<Window>) -> Self {
        let data = Data::empty();
        let settings = Settings::load("src/settings.toml");
        println!("Loaded settings: {:?}", settings);

        let renderer = Arc::new(Mutex::new(Renderer::new(window.clone(), data.clone())));
        let camera = Camera::new();
        let simulation = Arc::new(Mutex::new(Simulation::new()));
        let ui_loader = Arc::new(Mutex::new(UiButtonLoader::new()));

        {
            let mut d = data.lock().unwrap();
            d.set_cores(renderer.clone(), simulation.clone(), ui_loader.clone());
        }
        {
            let mut r = renderer.lock().unwrap();
            r.core.make_circles()
        }
        Self {
            _window: window,
            input: InputState::new(),
            mouse: MouseState {
                last_pos: None,
                dragging: false,
            },
            velocity: Vec3::ZERO,
            zoom_vel: 0.0,
            target_yaw: camera.yaw,
            target_pitch: camera.pitch,
            orbit_smoothness: 0.25,
            yaw_velocity: 0.0,
            pitch_velocity: 0.0,
            orbit_damping_release: 4.0,
            zoom_damping: 12.0,
            camera,
            renderer,
            simulation,
            simulation_running: true,
            ui_loader,
            settings,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.renderer.lock().unwrap().resize(new_size);
    }
    pub fn render(&mut self, simulation_dt: f32) {
        let dt;
        {
            let mut renderer = self.renderer.lock().unwrap();
            renderer.render(&self.camera);
            dt = renderer.core.timer.dt();
        }
        self.update_camera(dt);
    }

    pub(crate) fn update_camera(&mut self, dt: f32) {
        let eye = self.camera.position();
        let mut fwd3d = self.camera.target - eye;
        if fwd3d.length_squared() > 0.0 {
            fwd3d = fwd3d.normalize();
        }

        let mut forward = Vec3::new(fwd3d.x, 0.0, fwd3d.z);
        if forward.length_squared() > 0.0 {
            forward = forward.normalize();
        }

        let right = forward.cross(Vec3::Y).normalize();
        let up = Vec3::Y;

        let mut wish = Vec3::ZERO;
        if self.input.pressed.contains("w") {
            wish += forward;
        }
        if self.input.pressed.contains("s") {
            wish -= forward;
        }
        if self.input.pressed.contains("a") {
            wish -= right;
        }
        if self.input.pressed.contains("d") {
            wish += right;
        }
        if self.input.pressed.contains("q") {
            wish += up;
        }
        if self.input.pressed.contains("e") {
            wish -= up;
        }

        // --- Adaptive movement speed ---
        let base_speed = 8.0; // base units/sec
        let mut speed = base_speed;

        match (self.input.shift_pressed, self.input.ctrl_pressed) {
            (true, false) => speed *= 3.0, // faster
            (false, true) => speed *= 0.4, // slower
            (true, true) => speed *= 0.1,  // ultra slow
            _ => {}                        // normal speed
        }

        let decay_rate = 6.0; // damping when no input
        let dist = self.camera.radius;
        let speed_factor = (dist / 10.0).clamp(0.1, 10.0); // scales with zoom

        if wish.length_squared() > 0.0 {
            wish = wish.normalize();
            self.velocity = wish * speed * speed_factor;
        } else {
            // exponential decay towards zero
            let k = (1.0 - decay_rate * dt).max(0.0);
            self.velocity *= k;
            if self.velocity.length_squared() < 1e-5 {
                self.velocity = Vec3::ZERO;
            }
        }

        // smooth zoom update
        if self.zoom_vel.abs() > 0.0001 {
            self.camera.radius += self.zoom_vel * dt;
            self.zoom_vel *= (1.0 - self.zoom_damping * dt).max(0.0);
            self.camera.radius = self.camera.radius.clamp(1.0, 10_000.0);
        } else {
            self.zoom_vel = 0.0;
        }

        // --- Smooth orbit follow ---
        let t = 1.0 - (-self.orbit_smoothness * 60.0 * dt).exp();
        self.camera.yaw += (self.target_yaw - self.camera.yaw) * t;
        self.camera.pitch += (self.target_pitch - self.camera.pitch) * t;

        // --- Gentle decel after release ---
        if !self.mouse.dragging {
            self.target_yaw += self.yaw_velocity;
            self.target_pitch += self.pitch_velocity;
            self.yaw_velocity *= (1.0 - self.orbit_damping_release * dt).max(0.0);
            self.pitch_velocity *= (1.0 - self.orbit_damping_release * dt).max(0.0);
        }

        // --- Clamp and pan ---
        self.camera.pitch = self
            .camera
            .pitch
            .clamp(10.0f32.to_radians(), 89.0f32.to_radians());
        println!("{}", dt);
        self.camera.target += self.velocity * dt;
    }
    pub fn update_simulation(&self, dt: f32) {
        if let Ok(mut sim) = self.simulation.lock() {
            sim.update(dt);
        }
    }
}
