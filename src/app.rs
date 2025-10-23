use crate::state::State;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::Key;
use winit::window::{Window, WindowId};

#[derive(Default)]
pub(crate) struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
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

        let state = pollster::block_on(State::new(window.clone()));

        self.window = Some(window);
        self.state = Some(state);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let state = match self.state.as_mut() {
            Some(s) => s,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size),
            WindowEvent::RedrawRequested => state.render(),
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;

                // Movement keys (continuous)
                if let Key::Character(ch) = &event.logical_key {
                    let key = ch.to_lowercase();
                    if ["w", "a", "s", "d", "q", "e"].contains(&key.as_str()) {
                        state.input.set_key(&key, pressed);
                    }
                }

                // F5 (toggle once)
                if pressed {
                    if let Key::Named(winit::keyboard::NamedKey::F5) = &event.logical_key {
                        if let Some(state) = &mut self.state {
                            state.renderer.core.cycle_msaa();
                        }
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

                        // Apply smoothing directly to target angles
                        state.target_yaw += dx as f32 * yaw_sensitivity;
                        state.target_pitch += dy as f32 * pitch_sensitivity;
                        state.target_pitch = state
                            .target_pitch
                            .clamp(10.0f32.to_radians(), 89.0f32.to_radians());

                        // record last angular velocity for soft stop
                        state.yaw_velocity = dx as f32 * yaw_sensitivity;
                        state.pitch_velocity = dy as f32 * pitch_sensitivity;
                    }
                    state.mouse.last_pos = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // zoom in/out
                match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => {
                        // scroll direction affects velocity, proportional to distance
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
        // Request redraw every frame
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}
