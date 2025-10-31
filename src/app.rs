use crate::state::State;
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
    state: Option<Arc<Mutex<State>>>,
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

        let state = Arc::new(Mutex::new(State::new(window.clone())));

        self.target_fps = {
            let s = state.lock().unwrap();
            s.settings.target_fps
        };

        let simulation_handle = {
            let s = state.lock().unwrap();
            s.simulation_handle()
        };

        self.state = Some(state.clone());
        self.window = Some(window.clone());
        self.target_frame_time = Duration::from_secs_f32(1.0 / self.target_fps);

        let timing_clone = self.timing.clone();

        // spawn sim thread
        let state_clone = state.clone();
        let mut state_clone = state.clone();
        thread::spawn(move || {
            let tick = Duration::from_secs_f64(1.0 / 60.0);
            let mut last = Instant::now();

            loop {
                let now = Instant::now();
                let dt = (now - last).as_secs_f32();
                last = now;

                if let Ok(mut t) = timing_clone.lock() {
                    t.sim_dt = dt;
                }

                if let Ok(state) = state_clone.lock() {
                    let ui_loader = state.ui_loader.clone(); // safely clone Arc!!!!
                    drop(state); // drop before calling update (so that no double-lock happens ok??)

                    if let Ok(mut sim) = simulation_handle.lock() {
                        sim.update(dt, &ui_loader, &state_clone);
                    }
                }

                let elapsed = now.elapsed();
                if elapsed < tick {
                    thread::sleep(tick - elapsed);
                }
            }
        });

        event_loop.set_control_flow(ControlFlow::Poll);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state_arc) = &self.state else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let Ok(mut state) = state_arc.lock() {
                    state.resize(size);
                }
            }

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

                if let Ok(mut state) = state_arc.lock() {
                    state.render(self.timing.clone());
                }

                // maintain FPS cap
                let elapsed = frame_start.elapsed();
                if elapsed < self.target_frame_time {
                    thread::sleep(self.target_frame_time - elapsed);
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let Ok(mut state) = state_arc.lock() {
                    let pressed = event.state == ElementState::Pressed;

                    match &event.logical_key {
                        Key::Character(ch) => {
                            let key = ch.to_lowercase();
                            if ["w", "a", "s", "d", "q", "e"].contains(&key.as_str()) {
                                state.input.set_key(&key, pressed);
                            }
                        }
                        Key::Named(winit::keyboard::NamedKey::Shift) => {
                            state.input.shift_pressed = pressed;
                        }
                        Key::Named(winit::keyboard::NamedKey::Control) => {
                            state.input.ctrl_pressed = pressed;
                        }
                        _ => {}
                    }

                    // F5 toggles MSAA
                    if pressed {
                        if let Key::Named(winit::keyboard::NamedKey::F5) = &event.logical_key {
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
                if let Ok(mut state) = state_arc.lock() {
                    match button {
                        winit::event::MouseButton::Left => {
                            state.mouse.left_pressed = mouse_state == ElementState::Pressed;
                        }
                        winit::event::MouseButton::Middle => {
                            state.mouse.middle_pressed = mouse_state == ElementState::Pressed;
                            state.mouse.last_pos = None;
                        }
                        winit::event::MouseButton::Right => {
                            state.mouse.right_pressed = mouse_state == ElementState::Pressed;
                        }
                        winit::event::MouseButton::Back => {
                            state.mouse.back_pressed = mouse_state == ElementState::Pressed;
                        }
                        winit::event::MouseButton::Forward => {
                            state.mouse.forward_pressed = mouse_state == ElementState::Pressed;
                        }
                        _ => {}
                    }
                    // if mouse_state == ElementState::Pressed {
                    //
                    // }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if let Ok(mut state) = state_arc.lock() {
                    let pos = (position.x as f32, position.y as f32);
                    state.mouse.pos_x = pos.0;
                    state.mouse.pos_y = pos.1;

                    if state.mouse.middle_pressed {
                        if let Some((lx, ly)) = state.mouse.last_pos {
                            let dx = pos.0 - lx;
                            let dy = pos.1 - ly;
                            let pitch_sensitivity = 0.002;
                            let yaw_sensitivity = 0.0016;

                            state.target_yaw += dx * yaw_sensitivity;
                            state.target_pitch += dy * pitch_sensitivity;
                            state.target_pitch = state
                                .target_pitch
                                .clamp(10.0f32.to_radians(), 89.0f32.to_radians());

                            state.yaw_velocity = dx * yaw_sensitivity;
                            state.pitch_velocity = dy * pitch_sensitivity;
                        }

                        state.mouse.last_pos = Some(pos);
                    }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if let Ok(mut state) = state_arc.lock() {
                    match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => {
                            state.zoom_vel -= y * 1.0 * state.camera.radius;
                        }
                        winit::event::MouseScrollDelta::PixelDelta(pos) => {
                            state.zoom_vel -= pos.y as f32 * 0.04 * state.camera.radius;
                        }
                    }
                }
            }

            _ => (),
        }
    }
}
