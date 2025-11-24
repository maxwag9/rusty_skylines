use crate::resources::Resources;
use crate::systems::audio::audio_system;
use crate::systems::input::camera_input_system;
use crate::systems::physics::simulation_system;
use crate::systems::render::render_system;
use crate::systems::ui::ui_system;
use crate::vertex::UiButtonPolygon;
use crate::vertex::UiElement::Polygon;
use crate::world::World;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::window::{Window, WindowId};

pub struct App {
    window: Option<Arc<Window>>,
    world: Option<World>,
    resources: Option<Resources>,
    schedule: Schedule,
}

impl App {
    pub fn new() -> Self {
        Self {
            window: None,
            world: None,
            resources: None,
            schedule: Schedule::new(),
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

struct Schedule {
    sim_systems: Vec<SystemFn>,
    render_systems: Vec<SystemFn>,
    input_systems: Vec<SystemFn>,
}

type SystemFn = fn(&mut World, &mut Resources);

impl Schedule {
    fn new() -> Self {
        Self {
            sim_systems: vec![simulation_system, audio_system],
            render_systems: vec![ui_system, render_system],
            input_systems: vec![camera_input_system],
        }
    }

    pub fn run_sim(&self, world: &mut World, resources: &mut Resources) {
        for system in &self.sim_systems {
            (system)(world, resources);
        }
    }

    pub fn run_render(&self, world: &mut World, resources: &mut Resources) {
        for system in &self.render_systems {
            (system)(world, resources);
        }
    }

    pub fn run_inputs(&self, world: &mut World, resources: &mut Resources) {
        for system in &self.input_systems {
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
        let mut resources = Resources::new(window.clone());
        resources
            .time
            .set_tps(resources.settings.target_tps.max(1.0));
        resources
            .time
            .set_fps(resources.settings.target_fps.max(1.0));

        self.window = Some(window.clone());
        self.world = Some(world);
        self.resources = Some(resources);

        event_loop.set_control_flow(ControlFlow::Poll);
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                let Some(resources) = self.resources.as_mut() else {
                    return;
                };

                let down = event.state == ElementState::Pressed;
                let input = &mut resources.input;

                // Always track physical key
                let phys = event.physical_key;
                input.set_physical(phys, down);

                // Track logical key (NamedKey or Character)
                match &event.logical_key {
                    Key::Named(named) => {
                        input.set_logical(*named, down);

                        if matches!(named, NamedKey::Shift) {
                            input.shift_pressed = down;
                        }
                        if matches!(named, NamedKey::Control) {
                            input.ctrl_pressed = down;
                        }
                    }

                    Key::Character(s) => {
                        // This is text meaning, not hardware key
                        // Use only first char for keybindings
                        if let Some(c) = s.chars().next() {
                            let lower = c.to_ascii_lowercase().to_string();
                            input.set_character(&lower, down);
                        }
                    }

                    _ => {}
                }

                // -------- special logic --------
                if down {
                    if let Key::Named(NamedKey::F5) = &event.logical_key {
                        resources.renderer.core.cycle_msaa();
                    }

                    if let Key::Named(NamedKey::F6) = &event.logical_key {
                        resources.settings.editor_mode = !resources.settings.editor_mode;
                        resources
                            .ui_loader
                            .ui_runtime
                            .update_editor_mode(resources.settings.editor_mode);
                    }

                    if input.ctrl_pressed {
                        if let Key::Character(ch) = &event.logical_key {
                            if ch.eq_ignore_ascii_case("s") {
                                if let Err(e) = resources
                                    .ui_loader
                                    .save_gui_to_file("ui_data/gui_layout.json")
                                {
                                    eprintln!("Failed to save GUI layout: {e}");
                                } else {
                                    println!("GUI layout saved");
                                }
                            }
                        }
                    }
                    if input.pressed_physical(&PhysicalKey::Code(KeyCode::KeyP)) {
                        let result = resources.ui_loader.add_element(
                            "base_gui",
                            Polygon(UiButtonPolygon::default()),
                            &resources.mouse,
                        );
                        match result {
                            Ok(r) => {
                                println!("Added GUI element: {:?}", r);
                            }
                            Err(r) => {
                                println!("Failed adding GUI element: {:?}", r);
                            }
                        }
                    }
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(resources) = self.resources.as_mut() {
                    let mouse = &mut resources.mouse;
                    match button {
                        winit::event::MouseButton::Left => {
                            let pressed = state == ElementState::Pressed;
                            mouse.left_just_pressed = pressed && !mouse.left_pressed;
                            mouse.left_just_released = !pressed && mouse.left_pressed;
                            mouse.left_pressed = pressed;
                        }
                        winit::event::MouseButton::Right => {
                            let pressed = state == ElementState::Pressed;
                            mouse.right_just_pressed = pressed && !mouse.right_pressed;
                            mouse.right_just_released = !pressed && mouse.right_pressed;
                            mouse.right_pressed = pressed;
                        }
                        winit::event::MouseButton::Middle => {
                            let pressed = state == ElementState::Pressed;
                            mouse.middle_pressed = pressed;
                            mouse.last_pos = None;
                        }
                        winit::event::MouseButton::Back => {
                            mouse.back_pressed = state == ElementState::Pressed;
                        }
                        winit::event::MouseButton::Forward => {
                            mouse.forward_pressed = state == ElementState::Pressed;
                        }
                        _ => {}
                    }
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
                let mut editor_mode = false;
                if let Some(resources) = self.resources.as_mut() {
                    editor_mode = resources.settings.editor_mode;
                    let mouse = &mut resources.mouse;
                    match delta {
                        MouseScrollDelta::LineDelta(x, y) => {
                            mouse.scroll_delta.x += x;
                            mouse.scroll_delta.y += y;
                        }
                        MouseScrollDelta::PixelDelta(pos) => {
                            mouse.scroll_delta.x += pos.x as f32 * 0.1;
                            mouse.scroll_delta.y += pos.y as f32 * 0.1;
                        }
                    }
                }
                if !editor_mode {
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
            }
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

                    // update render time
                    resources.time.update_render();

                    let ui = &mut resources.ui_loader;

                    ui.variables
                        .set("fps", format!("{}", resources.time.render_fps));
                    ui.variables
                        .set("render_dt", format!("{}", resources.time.render_dt));
                    ui.variables
                        .set("sim_dt", format!("{}", resources.time.sim_dt));

                    // simulate time independent of render fps
                    resources.time.update_sim();
                    resources.time.sim_accumulator += resources.time.sim_dt;

                    while resources.time.sim_accumulator >= resources.time.sim_target_step {
                        resources.time.sim_dt = resources.time.sim_target_step;
                        self.schedule.run_inputs(world, resources);
                        self.schedule.run_sim(world, resources);
                        resources.time.sim_accumulator -= resources.time.sim_target_step;
                    }

                    self.schedule.run_render(world, resources);

                    let elapsed = frame_start.elapsed();
                    if elapsed < Duration::from_secs_f32(resources.time.target_frametime) {
                        thread::sleep(
                            Duration::from_secs_f32(resources.time.target_frametime) - elapsed,
                        );
                    }
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }

                if let Some(resources) = self.resources.as_mut() {
                    resources.mouse.update_just_states();
                }
            }

            _ => {}
        }
    }
}
