use crate::state::{SimulationHandle, State};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::Key;
use winit::window::{Window, WindowId};

#[derive(Default, Clone)]
pub struct TimingData {
    pub sim_dt: f32,
    pub render_dt: f32,
    pub render_fps: f32,
}

pub(crate) struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
    last_frame: Instant,
    target_fps: f32,
    target_frame_time: Duration,
    sim_dt: f32,
    timing: Arc<Mutex<TimingData>>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            state: None,
            last_frame: Instant::now(),
            target_fps: 100.0,
            target_frame_time: Duration::from_millis(10),
            sim_dt: 1.0 / 60.0,
            timing: Arc::new(Mutex::new(Default::default())),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create the window
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Rusty City")
                        .with_inner_size(winit::dpi::PhysicalSize::new(2560, 1400)),
                )
                .unwrap(),
        );
        self.last_frame = Instant::now();
        // Shared state
        let state = State::new(window.clone());
        self.target_fps = state.settings.target_fps;
        let simulation_handle: SimulationHandle = state.simulation_handle();
        self.state = Some(state);

        self.target_frame_time = Duration::from_secs_f32(1.0 / self.target_fps);
        self.window = Some(window);

        let timing_clone = self.timing.clone();

        // Simulation thread (fixed timestep)
        thread::spawn(move || {
            let tick = Duration::from_secs_f64(1.0 / 60.0); // 60 Hz

            let mut last = Instant::now();

            loop {
                let now = Instant::now();
                let dt = (now - last).as_secs_f32();
                last = now;

                if let Ok(mut t) = timing_clone.lock() {
                    t.sim_dt = dt;
                }
                if let Ok(mut sim) = simulation_handle.lock() {
                    sim.update(dt);
                }

                let elapsed = now.elapsed();
                if elapsed < tick {
                    thread::sleep(tick - elapsed);
                }
            }
        });

        //self.window.as_ref().unwrap().request_redraw();
        event_loop.set_control_flow(ControlFlow::Poll);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size),
            WindowEvent::RedrawRequested => {
                let frame_start = Instant::now();
                {
                    let mut timing = self.timing.lock().unwrap();
                    let now = Instant::now();
                    let dt = (now - self.last_frame).as_secs_f32();
                    self.last_frame = now;
                    timing.render_dt = dt;
                    timing.render_fps = 1.0 / dt;
                }
                state.render(self.timing.clone());

                // Measure and schedule next frame
                let elapsed = frame_start.elapsed();
                if elapsed < self.target_frame_time {
                    thread::sleep(self.target_frame_time - elapsed);
                }

                self.window.as_ref().unwrap().request_redraw();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;

                match &event.logical_key {
                    Key::Character(ch) => {
                        let key = ch.to_lowercase();
                        if ["w", "a", "s", "d", "q", "e"].contains(&key.as_str()) {
                            state.input.set_key(&key, pressed);
                        }
                    }
                    Key::Named(winit::keyboard::NamedKey::Shift) => {
                        state.input.shift_pressed = pressed
                    }
                    Key::Named(winit::keyboard::NamedKey::Control) => {
                        state.input.ctrl_pressed = pressed
                    }

                    _ => {}
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

            WindowEvent::MouseWheel { delta, .. } => match delta {
                winit::event::MouseScrollDelta::LineDelta(_, y) => {
                    state.zoom_vel -= y * 1.0 * state.camera.radius;
                }
                winit::event::MouseScrollDelta::PixelDelta(pos) => {
                    state.zoom_vel -= pos.y as f32 * 0.04 * state.camera.radius;
                }
            },

            _ => (),
        }
    }
}
