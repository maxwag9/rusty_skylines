use crate::state::State;
use std::sync::{Arc, Mutex};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::Key;
use winit::window::{Window, WindowId};

pub(crate) struct App {
    window: Option<Arc<Window>>,
    state: Option<Arc<Mutex<State>>>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            state: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create the window when resumed
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Rusty City")
                        .with_inner_size(winit::dpi::PhysicalSize::new(2560, 1400)),
                )
                .unwrap(),
        );

        // Initialize shared state
        let state = Arc::new(Mutex::new(State::new(window.clone())));
        self.state = Some(state);
        self.window = Some(window);
        if let Some(state_arc) = &self.state {
            let state_clone = state_arc.clone();
            std::thread::spawn(move || {
                use std::time::{Duration, Instant};
                const TICK: Duration = Duration::from_millis(16); // ~60 Hz
                let mut last = Instant::now();

                loop {
                    let now = Instant::now();
                    let dt = (now - last).as_secs_f32();
                    last = now;

                    {
                        let mut state = state_clone.lock().unwrap();
                        state.simulation.update(dt);
                    }

                    std::thread::sleep(TICK);
                }
            });
        }

    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state_arc) = &self.state else {
            return;
        };
        let mut state = state_arc.lock().unwrap();

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size),
            WindowEvent::RedrawRequested => state.render(),
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;

                // Movement keys
                if let Key::Character(ch) = &event.logical_key {
                    let key = ch.to_lowercase();
                    if ["w", "a", "s", "d", "q", "e"].contains(&key.as_str()) {
                        state.input.set_key(&key, pressed);
                    }
                }

                // F5: toggle MSAA
                if pressed {
                    if let Key::Named(winit::keyboard::NamedKey::F5) = &event.logical_key {
                        state.renderer.core.cycle_msaa();
                    }
                }
            }

            WindowEvent::MouseInput {
                state: mouse_state,
                button,
                ..
            } => {
                if button == winit::event::MouseButton::Middle {
                    state.mouse.dragging = mouse_state == ElementState::Pressed;
                    state.mouse.last_pos = None;
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if state.mouse.dragging {
                    if let Some((lx, ly)) = state.mouse.last_pos {
                        let dx = position.x - lx;
                        let dy = position.y - ly;
                        let pitch_sensitivity = 0.002;
                        let yaw_sensitivity = 0.0016;

                        state.target_yaw += dx as f32 * yaw_sensitivity;
                        state.target_pitch += dy as f32 * pitch_sensitivity;
                        state.target_pitch = state
                            .target_pitch
                            .clamp(10.0f32.to_radians(), 89.0f32.to_radians());

                        state.yaw_velocity = dx as f32 * yaw_sensitivity;
                        state.pitch_velocity = dy as f32 * pitch_sensitivity;
                    }
                    state.mouse.last_pos = Some((position.x, position.y));
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => {
                        state.zoom_vel -= y * 1.0 * state.camera.radius;
                    }
                    winit::event::MouseScrollDelta::PixelDelta(pos) => {
                        state.zoom_vel -= pos.y as f32 * 0.04 * state.camera.radius;
                    }
                }
            }

            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}
