use crate::paths::data_dir;
use crate::resources::Resources;
use crate::systems::audio::audio_system;
use crate::systems::input::camera_input_system;
use crate::systems::physics::simulation_system;
use crate::systems::render::render_system;
use crate::systems::ui::ui_system;
use crate::ui::vertex::UiButtonPolygon;
use crate::ui::vertex::UiElement::Polygon;
use crate::world::World;
use glam::Vec2;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{Key, NamedKey};
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
    fn new_events(&mut self, _event_loop: &ActiveEventLoop, _cause: StartCause) {
        if let Some(resources) = self.resources.as_mut() {
            resources.input.mouse.delta = Vec2::ZERO;
            resources.input.begin_frame(resources.time.total_time);
            let pos = resources.input.mouse.pos;
            let delta = resources.input.mouse.delta;

            resources.ui_loader.variables.set_f32("mouse_pos.x", pos.x);
            resources
                .ui_loader
                .variables
                .set_f32("mouse_pos_delta.x", delta.x);
            resources.ui_loader.variables.set_f32("mouse_pos.y", pos.y);
            resources
                .ui_loader
                .variables
                .set_f32("mouse_pos_delta.y", delta.y);
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Rusty Skylines")
                        .with_inner_size(winit::dpi::PhysicalSize::new(2560, 1400)),
                )
                .expect("Failed to create window"),
        );

        let world = World::new();
        let mut resources = Resources::new(window.clone(), &world);
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

                let input = &mut resources.input;
                let down = event.state == ElementState::Pressed;

                // physical
                input.set_physical(event.physical_key, down);

                // logical + printable text
                match &event.logical_key {
                    Key::Named(named) => {
                        input.set_logical(*named, down);

                        if *named == NamedKey::Shift {
                            input.shift = down;
                        }
                        if *named == NamedKey::Control {
                            input.ctrl = down;
                        }
                        if *named == NamedKey::Alt {
                            input.alt = down;
                        }
                        if *named == NamedKey::Space {
                            input.set_character(" ", down);
                        }
                    }

                    Key::Character(ch) => {
                        if !ch.is_empty() {
                            input.set_character(ch, down);
                        }
                    }

                    _ => {}
                }

                // ---------------------------------------------------------------------
                // ENGINE ACTIONS (ALL action-based, no direct key checks)
                // ---------------------------------------------------------------------

                // MSAA cycle
                if input.action_repeat("Cycle MSAA") {
                    resources.renderer.core.cycle_msaa();
                }

                // Toggle editor mode
                if input.action_repeat("Toggle editor mode") {
                    resources.settings.editor_mode = !resources.settings.editor_mode;
                    resources
                        .ui_loader
                        .ui_runtime
                        .update_editor_mode(resources.settings.editor_mode);
                    resources
                        .ui_loader
                        .variables
                        .set_bool("editor_mode", resources.settings.editor_mode)
                }
                // Toggle override_mode
                if input.action_repeat("Toggle override mode") {
                    resources.settings.override_mode = !resources.settings.override_mode;
                    resources
                        .ui_loader
                        .ui_runtime
                        .update_override_mode(resources.settings.override_mode);
                    resources
                        .ui_loader
                        .variables
                        .set_bool("override_mode", resources.settings.override_mode)
                }
                // Toggle show_gui
                if input.action_repeat("Toggle show gui") {
                    resources.settings.show_gui = !resources.settings.show_gui;
                    resources
                        .ui_loader
                        .ui_runtime
                        .update_show_gui(resources.settings.show_gui);
                    resources
                        .ui_loader
                        .variables
                        .set_bool("show_gui", resources.settings.show_gui);
                    resources.settings.override_mode = false;
                    resources
                        .ui_loader
                        .ui_runtime
                        .update_override_mode(resources.settings.override_mode);
                    resources
                        .ui_loader
                        .variables
                        .set_bool("override_mode", resources.settings.override_mode);
                    resources.settings.editor_mode = false;
                    resources
                        .ui_loader
                        .ui_runtime
                        .update_editor_mode(resources.settings.editor_mode);
                    resources
                        .ui_loader
                        .variables
                        .set_bool("editor_mode", resources.settings.editor_mode)
                }

                // Save GUI
                if input.action_pressed_once("Save GUI layout") {
                    match resources
                        .ui_loader
                        .save_gui_to_file(data_dir("ui_data/gui_layout.json"))
                    {
                        Ok(_) => println!("GUI layout saved"),
                        Err(e) => eprintln!("Failed to save GUI layout: {e}"),
                    }
                }

                if input.action_pressed_once("Leave Game") {
                    resources.settings.total_game_time = resources.time.total_game_time;
                    match resources.settings.save(data_dir("settings.toml")) {
                        Ok(_) => println!("Settings saved"),
                        Err(e) => eprintln!("Failed to save Settings: {e}"),
                    }
                    match resources
                        .renderer
                        .core
                        .world
                        .terrain_editor
                        .save_edits(data_dir("edited_chunks"))
                    {
                        Ok(_) => println!("World saved"),
                        Err(e) => eprintln!("Failed to save World: {e}"),
                    }
                    event_loop.exit()
                }
                // Add GUI element
                if input.action_repeat("Add GUI element")
                    && resources.ui_loader.ui_runtime.editor_mode
                {
                    let result = resources.ui_loader.add_element(
                        resources
                            .ui_loader
                            .ui_runtime
                            .selected_ui_element_primary
                            .menu_name
                            .clone()
                            .as_str(),
                        resources
                            .ui_loader
                            .ui_runtime
                            .selected_ui_element_primary
                            .layer_name
                            .clone()
                            .as_str(),
                        Polygon(UiButtonPolygon::default()),
                        &resources.input.mouse,
                        true,
                    );

                    match result {
                        Ok(r) => println!("Added GUI element: {:?}", r),
                        Err(r) => println!("Failed adding GUI element: {:?}", r),
                    }
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(resources) = self.resources.as_mut() {
                    resources.input.handle_mouse_button(button, state);
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if let Some(resources) = self.resources.as_mut() {
                    resources.input.handle_mouse_move(position.x, position.y);

                    let pos = resources.input.mouse.pos;
                    let delta = resources.input.mouse.delta;

                    resources.ui_loader.variables.set_f32("mouse_pos.x", pos.x);
                    resources
                        .ui_loader
                        .variables
                        .set_f32("mouse_pos_delta.x", delta.x);
                    resources.ui_loader.variables.set_f32("mouse_pos.y", pos.y);
                    resources
                        .ui_loader
                        .variables
                        .set_f32("mouse_pos_delta.y", delta.y);
                    // camera rotation ONLY if needed & dragging
                    if resources.input.mouse.middle_pressed {
                        if let Some(world) = self.world.as_mut() {
                            if let Some(controller) =
                                world.camera_controller_mut(world.main_camera())
                            {
                                let pitch_s = 0.002;
                                let yaw_s = 0.0016;

                                controller.target_yaw += delta.x * yaw_s;
                                controller.target_pitch += delta.y * pitch_s;

                                controller.yaw_velocity = delta.x * yaw_s;
                                controller.pitch_velocity = delta.y * pitch_s;
                            }
                        }
                    }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(resources) = self.resources.as_mut() {
                    let scroll = resources.input.handle_mouse_wheel(delta);

                    if !resources.settings.editor_mode {
                        if let Some(world) = self.world.as_mut() {
                            let cam = world.main_camera();
                            let radius = world.camera(cam).unwrap().orbit_radius;

                            if let Some(controller) = world.camera_controller_mut(cam) {
                                let zoom_factor = 100.0;
                                controller.zoom_velocity -= scroll.y * zoom_factor * radius.sqrt();
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
            WindowEvent::ScaleFactorChanged { .. } => {
                if let Some(resources) = self.resources.as_mut() {
                    let size = resources.window.inner_size(); // << get the real physical size
                    if size.width > 0 && size.height > 0 {
                        resources.renderer.resize(size);
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                if let (Some(world), Some(resources)) =
                    (self.world.as_mut(), self.resources.as_mut())
                {
                    let frame_start = Instant::now();

                    // update render time
                    let mut time_speed: f32 = 1.0f32; // f32 btw
                    if resources.input.action_down("Speed up Time 20x") {
                        time_speed = 20.0
                    } else if resources.input.action_down("Speed up Time 8x") {
                        time_speed = 8.0
                    } else if resources.input.action_down("Speed up Time 2x") {
                        time_speed = 2.0
                    }
                    resources
                        .ui_loader
                        .variables
                        .set_f32("time_speed", time_speed);
                    resources.time.update_render(time_speed);
                    resources
                        .ui_loader
                        .variables
                        .set_f32("total_game_time", resources.time.total_game_time as f32);

                    // update UI global vars
                    let ui = &mut resources.ui_loader;

                    ui.variables.set_f32("fps", resources.time.render_fps);
                    ui.variables.set_f32("render_dt", resources.time.render_dt);
                    ui.variables.set_f32("sim_dt", resources.time.sim_dt);

                    // simulation timing
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
            }

            _ => {}
        }
    }
}
