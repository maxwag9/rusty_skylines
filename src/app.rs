use crate::events::Event;
use crate::helpers::paths::data_dir;
use crate::resources::Resources;
use crate::systems::audio::audio_system;
use crate::systems::car_events::run_car_events;
use crate::systems::input::camera_input_system;
use crate::systems::render::render_system;
use crate::systems::simulation::simulation_system;
use crate::systems::small_systems::*;
use crate::systems::ui::ui_system;
use crate::ui::ui_edit_manager::CreateElementCommand;
use crate::ui::vertex::UiButtonCircle;
use crate::ui::vertex::UiElement::Circle;
use crate::world::roads::road_structs::RoadType;
use crate::world::terrain::terrain_subsystem::CursorMode;
use crate::world::world_core::WorldCore;
use glam::Vec2;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const TIME_SPEED_BINDINGS: [(&str, f32); 7] = [
    ("Speed up Time 100x", 100.0),
    ("Speed up Time 16x", 16.0),
    ("Speed up Time 2x", 2.0),
    ("Reverse Time 100x", -100.0),
    ("Reverse Time 16x", -16.0),
    ("Reverse Time 2x", -2.0),
    ("Slow down Time 2x", 0.5),
];
const MAX_SIM_STEPS_PER_FRAME: usize = 1_000;

pub struct App {
    window: Option<Arc<Window>>,
    resources: Option<Resources>,
    schedule: Schedule,
}

impl App {
    pub fn new() -> Self {
        Self {
            window: None,
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

type SystemFn = fn(&mut Resources);

impl Schedule {
    fn new() -> Self {
        Self {
            sim_systems: vec![simulation_system, audio_system],
            render_systems: vec![ui_system, render_system],
            input_systems: vec![camera_input_system],
        }
    }

    pub fn run_sim(&self, resources: &mut Resources) {
        for system in &self.sim_systems {
            system(resources);
        }
    }

    pub fn run_render(&self, resources: &mut Resources) {
        let aspect =
            resources.render_core.config.width as f32 / resources.render_core.config.height as f32;
        let settings = &resources.settings;
        resources
            .world_core
            .world_state
            .camera_mut(resources.world_core.world_state.main_camera())
            .unwrap()
            .compute_matrices(aspect, settings);
        for system in &self.render_systems {
            system(resources);
        }
        resources
            .world_core
            .world_state
            .camera_mut(resources.world_core.world_state.main_camera())
            .unwrap()
            .end_frame();
    }

    pub fn run_inputs(&self, resources: &mut Resources) {
        for system in &self.input_systems {
            system(resources);
        }
    }
    pub fn run_events(&self, resources: &mut Resources) {
        let world = &mut resources.world_core;
        world.events.flip();

        for event in world.events.drain() {
            // order is explicit and intentional
            world.simulation.process_event(&event);
            cursor_system(&mut world.terrain_subsystem.cursor, &event);
            run_car_events(event, &mut world.car_subsystem, &world.road_subsystem);
            // later:
            // audio_event_system(...)
            // ui_event_system(...)
        }
    }
}

impl ApplicationHandler for App {
    fn new_events(&mut self, _event_loop: &ActiveEventLoop, _cause: StartCause) {
        if let Some(resources) = self.resources.as_mut() {
            let world = &mut resources.world_core;
            let input = &mut world.input;
            let time = &world.time;
            input.mouse.delta = Vec2::ZERO;
            input.begin_frame(time.total_time);
            let pos = input.mouse.pos;
            let delta = input.mouse.delta;

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

        let mut resources = Resources::new(window.clone());
        let world = &mut resources.world_core;
        let time = &mut world.time;
        time.set_tps(resources.settings.target_tps.max(1.0));
        time.set_fps(resources.settings.target_fps.max(1.0));
        self.window = Some(window.clone());
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
                let world = &mut resources.world_core;
                let Some(camera) = world
                    .world_state
                    .camera_mut(world.world_state.main_camera())
                else {
                    return;
                };
                let variables = &mut resources.ui_loader.variables;
                let settings = &mut resources.settings;
                let ui_options = &mut resources.ui_loader.touch_manager.options;
                let input = &mut world.input;
                let time = &world.time;
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
                // ENGINE ACTIONS (ALL action-based, no direct key checks)! factorial
                // ---------------------------------------------------------------------

                // MSAA cycle
                if input.action_repeat("Cycle MSAA") {
                    resources.render_core.cycle_msaa(settings);
                }

                if input.action_repeat("Switch Debug Render Mode") {
                    settings.debug_view_state = settings.debug_view_state.next();
                }
                if input.action_repeat("Cycle Shadow Type") {
                    settings.shadow_type = settings.shadow_type.next();
                }
                if input.action_repeat("Toggle Reversed Depth Z") {
                    settings.reversed_depth_z = !settings.reversed_depth_z;
                }
                // Toggle editor mode
                if input.action_repeat("Toggle editor mode") {
                    settings.editor_mode = !settings.editor_mode;
                    resources.ui_loader.touch_manager.editor.enabled = settings.editor_mode;
                    variables.set_bool("editor_mode", settings.editor_mode);
                    settings.show_world = !settings.editor_mode;
                    variables.set_bool("show_world", settings.show_world);
                    settings.show_gui = true;
                    variables.set_bool("show_gui", settings.show_gui);
                }
                // Toggle override_mode
                if input.action_repeat("Toggle override mode") {
                    settings.override_mode = !settings.override_mode;
                    ui_options.override_mode = settings.override_mode;
                    variables.set_bool("override_mode", settings.override_mode)
                }
                if input.action_repeat("Toggle show world") {
                    settings.show_world = !settings.show_world;
                    variables.set_bool("show_world", settings.show_world);
                }
                if input.action_repeat("Toggle drive car") {
                    settings.drive_car = !settings.drive_car;
                    variables.set_bool("drive_car", settings.drive_car);
                }
                // Toggle show_gui
                if input.action_repeat("Toggle show gui") {
                    settings.show_gui = !settings.show_gui;
                    ui_options.show_gui = settings.show_gui;
                    variables.set_bool("show_gui", settings.show_gui);
                    settings.override_mode = false;
                    ui_options.override_mode = settings.override_mode;
                    variables.set_bool("override_mode", settings.override_mode);
                    settings.editor_mode = false;
                    resources.ui_loader.touch_manager.editor.enabled = settings.editor_mode;
                    variables.set_bool("editor_mode", settings.editor_mode)
                }

                // Save GUI
                if input.action_pressed_once("Save GUI layout") {
                    match resources
                        .ui_loader
                        .save_gui_to_file(data_dir("ui_data/menus"), resources.window.inner_size())
                    {
                        Ok(_) => println!("GUI layout saved"),
                        Err(e) => eprintln!("Failed to save GUI layout: {e}"),
                    }
                }
                if input.action_pressed_once("Toggle Cursor Mode") {
                    match world.terrain_subsystem.cursor.mode {
                        CursorMode::Roads(_) => {
                            world.events.send(Event::SetCursorMode(CursorMode::Cars))
                        }
                        CursorMode::Cars => world
                            .events
                            .send(Event::SetCursorMode(CursorMode::TerrainEditing)),
                        CursorMode::TerrainEditing => {
                            world.events.send(Event::SetCursorMode(CursorMode::None))
                        }
                        CursorMode::None => world
                            .events
                            .send(Event::SetCursorMode(CursorMode::Roads(RoadType::default()))),
                    }
                }
                if input.action_pressed_once("Leave Game") {
                    settings.total_game_time = time.total_game_time;
                    settings.player_pos = camera.target;
                    match settings.save(data_dir("settings.toml")) {
                        Ok(_) => println!("Settings saved"),
                        Err(e) => eprintln!("Failed to save Settings: {e}"),
                    }
                    if settings.show_world {
                        match resources
                            .world_core
                            .terrain_subsystem
                            .terrain_editor
                            .save_edits(data_dir("edited_chunks"))
                        {
                            Ok(_) => println!("World saved"),
                            Err(e) => eprintln!("Failed to save World: {e}"),
                        }
                    }

                    event_loop.exit();
                    std::process::exit(69); // Die.
                }
                // Add GUI element
                if input.action_repeat("Add GUI element")
                    && resources.ui_loader.touch_manager.editor.enabled
                {
                    if let Some(sel) = &resources.ui_loader.touch_manager.selection.primary {
                        resources.ui_loader.ui_edit_manager.execute_command(
                            CreateElementCommand {
                                affected_element: sel.clone(),
                                element: Circle(UiButtonCircle::default()),
                            },
                            &mut resources.ui_loader.touch_manager,
                            &mut resources.ui_loader.menus,
                            &mut resources.ui_loader.variables,
                            &input.mouse,
                        )
                    }
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(resources) = self.resources.as_mut() {
                    resources
                        .world_core
                        .input
                        .handle_mouse_button(button, state);
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if let Some(resources) = self.resources.as_mut() {
                    let input = &mut resources.world_core.input;
                    input.handle_mouse_move(position.x, position.y);

                    let pos = input.mouse.pos;
                    let delta = input.mouse.delta;

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
                    if input.mouse.middle_pressed {
                        if let Some(controller) = resources
                            .world_core
                            .world_state
                            .camera_controller_mut(resources.world_core.world_state.main_camera())
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

            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(resources) = self.resources.as_mut() {
                    let scroll = resources.world_core.input.handle_mouse_wheel(delta);

                    if !resources.settings.editor_mode {
                        let cam = resources.world_core.world_state.main_camera();

                        if let Some(controller) =
                            resources.world_core.world_state.camera_controller_mut(cam)
                        {
                            let zoom_factor = 10.0;
                            controller.zoom_velocity -= scroll.y * zoom_factor;
                        }
                    }
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(resources) = self.resources.as_mut() {
                    resources.render_core.resize(&resources.surface, size);
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                if let Some(resources) = self.resources.as_mut() {
                    let size = resources.window.inner_size(); // << get the real physical size
                    if size.width > 0 && size.height > 0 {
                        resources.render_core.resize(&resources.surface, size);
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                let Some(resources) = self.resources.as_mut() else {
                    event_loop.exit();
                    return;
                };
                let frame_start = Instant::now();
                update_time_and_ui(resources);

                // -----------------------------
                // Fixed-step simulation with budget + spiral-of-death protection
                // -----------------------------
                // Do input/events once per render frame (prevents "pressed_once" from firing N times).
                self.schedule.run_inputs(resources);
                self.schedule.run_events(resources);

                // Clamp runaway backlog (e.g., window drag / breakpoint)
                resources
                    .world_core
                    .time
                    .clamp_sim_accumulator(MAX_SIM_STEPS_PER_FRAME);

                let sim_budget = Duration::from_secs_f32(
                    (resources.world_core.time.target_frametime * 0.7).max(0.0),
                );
                let sim_deadline = Instant::now() + sim_budget;

                let mut steps = 0usize;
                while resources.world_core.time.can_step_sim()
                    && steps < MAX_SIM_STEPS_PER_FRAME
                    && Instant::now() < sim_deadline
                {
                    self.schedule.run_sim(resources);
                    resources.world_core.time.consume_sim_step();
                    steps += 1;
                }

                let achieved_speed = if resources.world_core.time.render_dt > 0.0 {
                    (steps as f32 * resources.world_core.time.target_sim_dt)
                        / resources.world_core.time.render_dt
                } else {
                    0.0
                };
                resources
                    .ui_loader
                    .variables
                    .set_f32("achieved_time_speed", achieved_speed);

                // -----------------------------
                // Render
                // -----------------------------
                resources.world_core.world_state.update(
                    &mut resources.ui_loader,
                    &resources.world_core.time,
                    &resources.settings,
                );
                self.schedule.run_render(resources);

                // -----------------------------
                // FPS cap
                // -----------------------------
                let elapsed = frame_start.elapsed();
                let target =
                    Duration::from_secs_f32(resources.world_core.time.target_frametime.max(0.0));
                if target > Duration::ZERO && elapsed < target {
                    thread::sleep(target - elapsed);
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            WindowEvent::Focused(false) | WindowEvent::Focused(true) => {
                if let Some(resources) = self.resources.as_mut() {
                    let input = &mut resources.world_core.input;
                    let time = &mut resources.world_core.time;
                    input.reset_all(time.total_time);
                }
            }

            _ => {}
        }
    }
}

fn update_time_and_ui(resources: &mut Resources) {
    let Resources {
        world_core,
        settings,
        ui_loader,
        ..
    } = resources;
    let WorldCore {
        world_state,
        time,
        input,
        simulation,
        ..
    } = world_core;

    // -----------------------------
    // Time controls (incl. real toggle stop)
    // -----------------------------
    let can_time_control = !settings.editor_mode && !settings.drive_car && settings.show_world;

    if can_time_control && input.action_pressed_once("Toggle Stop Time") {
        simulation.toggle();

        // Important: kill any backlog so the sim stops immediately.
        if !simulation.running {
            time.clear_sim_accumulator();
        }
    }

    // Base timescale from "stopped" state
    let mut time_speed = if simulation.running { 1.0 } else { 0.0 };

    // Optional override from held bindings (also resumes time)
    for (action, speed) in TIME_SPEED_BINDINGS {
        if input.action_down(action) {
            time_speed = speed;
            simulation.running = true; // resume if user is explicitly controlling time
            break;
        }
    }

    // Ensure stop always wins if it's enabled (even if other keys are held)
    if !simulation.running {
        time_speed = 0.0;
    }

    resources
        .ui_loader
        .variables
        .set_f32("time_speed", time_speed);

    // -----------------------------
    // Frame timing (accumulator lives here)
    // -----------------------------
    time.begin_frame(time_speed);

    // UI globals
    {
        let ui = &mut resources.ui_loader;
        ui.variables.set_f32("fps", time.render_fps);
        ui.variables.set_f32("render_dt", time.render_dt);
        ui.variables.set_f32("sim_dt", time.target_sim_dt);
        ui.variables
            .set_f32("total_game_time", time.total_game_time as f32);
    }
}
