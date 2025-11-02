use crate::resources::Resources;
use crate::systems::audio::audio_system;
use crate::systems::input::camera_input_system;
use crate::systems::physics::simulation_system;
use crate::systems::render::render_system;
use crate::systems::ui::ui_system;
use crate::world::World;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::Key;
use winit::window::{Window, WindowId};

pub struct App {
    window: Option<Arc<Window>>,
    world: Option<World>,
    resources: Option<Resources>,
    schedule: Schedule,
    target_frame_time: Duration,
}

impl App {
    pub fn new() -> Self {
        Self {
            window: None,
            world: None,
            resources: None,
            schedule: Schedule::new(),
            target_frame_time: Duration::from_secs_f32(1.0 / 60.0),
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

struct Schedule {
    systems: Vec<SystemFn>,
}

type SystemFn = fn(&mut World, &mut Resources);

impl Schedule {
    fn new() -> Self {
        Self {
            systems: vec![
                camera_input_system,
                simulation_system,
                ui_system,
                audio_system,
                render_system,
            ],
        }
    }

    fn run(&self, world: &mut World, resources: &mut Resources) {
        for system in &self.systems {
            (system)(world, resources);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Rusty City")
                        .with_inner_size(winit::dpi::PhysicalSize::new(2560, 1400)),
                )
                .expect("Failed to create window"),
        );

        let world = World::new();
        let resources = Resources::new(window.clone());
        self.target_frame_time =
            Duration::from_secs_f32(1.0 / resources.settings.target_fps.max(1.0));

        self.window = Some(window.clone());
        self.world = Some(world);
        self.resources = Some(resources);

        event_loop.set_control_flow(ControlFlow::Poll);
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(resources) = self.resources.as_mut() {
                    resources.renderer.resize(size);
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(world), Some(resources)) =
                    (self.world.as_mut(), self.resources.as_mut())
                {
                    let frame_start = Instant::now();
                    resources.update_render_time();
                    self.schedule.run(world, resources);

                    let elapsed = frame_start.elapsed();
                    if elapsed < self.target_frame_time {
                        thread::sleep(self.target_frame_time - elapsed);
                    }
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let Some(resources) = self.resources.as_mut() else {
                    return;
                };
                let pressed = event.state == ElementState::Pressed;

                match &event.logical_key {
                    Key::Character(ch) => {
                        let key = ch.to_lowercase();
                        if ["w", "a", "s", "d", "q", "e"].contains(&key.as_str()) {
                            resources.input.set_key(&key, pressed);
                        }
                    }
                    Key::Named(winit::keyboard::NamedKey::Shift) => {
                        resources.input.shift_pressed = pressed;
                    }
                    Key::Named(winit::keyboard::NamedKey::Control) => {
                        resources.input.ctrl_pressed = pressed;
                    }
                    _ => {}
                }

                if pressed {
                    if let Key::Named(winit::keyboard::NamedKey::F5) = &event.logical_key {
                        resources.renderer.core.cycle_msaa();
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let Some(resources) = self.resources.as_mut() else {
                    return;
                };
                match button {
                    winit::event::MouseButton::Left => {
                        resources.mouse.left_pressed = state == ElementState::Pressed;
                    }
                    winit::event::MouseButton::Middle => {
                        let pressed = state == ElementState::Pressed;
                        resources.mouse.middle_pressed = pressed;
                        resources.mouse.last_pos = None;
                    }
                    winit::event::MouseButton::Right => {
                        resources.mouse.right_pressed = state == ElementState::Pressed;
                    }
                    winit::event::MouseButton::Back => {
                        resources.mouse.back_pressed = state == ElementState::Pressed;
                    }
                    winit::event::MouseButton::Forward => {
                        resources.mouse.forward_pressed = state == ElementState::Pressed;
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let delta = if let Some(resources) = self.resources.as_mut() {
                    let pos = glam::Vec2::new(position.x as f32, position.y as f32);
                    resources.mouse.pos = pos;

                    if resources.mouse.middle_pressed {
                        let delta = resources.mouse.last_pos.map(|last| pos - last);
                        resources.mouse.last_pos = Some(pos);
                        delta
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let (Some(delta), Some(world)) = (delta, self.world.as_mut()) {
                    if let Some(controller) = world.camera_controller_mut(world.main_camera()) {
                        let pitch_sensitivity = 0.002;
                        let yaw_sensitivity = 0.0016;
                        controller.target_yaw += delta.x * yaw_sensitivity;
                        controller.target_pitch += delta.y * pitch_sensitivity;
                        controller.target_pitch = controller
                            .target_pitch
                            .clamp(10.0f32.to_radians(), 89.0f32.to_radians());
                        controller.yaw_velocity = delta.x * yaw_sensitivity;
                        controller.pitch_velocity = delta.y * pitch_sensitivity;
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let Some(world) = self.world.as_mut() else {
                    return;
                };
                let entity = world.main_camera();
                let radius = world.camera(entity).map(|c| c.radius).unwrap_or(1.0);
                if let Some(controller) = world.camera_controller_mut(entity) {
                    match delta {
                        MouseScrollDelta::LineDelta(_, y) => {
                            controller.zoom_velocity -= y * 1.0 * radius;
                        }
                        MouseScrollDelta::PixelDelta(pos) => {
                            controller.zoom_velocity -= pos.y as f32 * 0.04 * radius;
                        }
                    }
                }
            }
            _ => {}
        }
    }
}
