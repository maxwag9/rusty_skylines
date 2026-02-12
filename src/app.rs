use crate::events::Event;
use crate::paths::data_dir;
use crate::renderer::world_renderer::CursorMode;
use crate::resources::Resources;
use crate::systems::audio::audio_system;
use crate::systems::car_events::run_car_events;
use crate::systems::input::camera_input_system;
use crate::systems::render::render_system;
use crate::systems::simulation::simulation_system;
use crate::systems::small_systems::*;
use crate::systems::ui::ui_system;
use crate::terrain::roads::road_structs::RoadType;
use crate::ui::ui_edit_manager::CreateElementCommand;
use crate::ui::vertex::UiButtonCircle;
use crate::ui::vertex::UiElement::Circle;
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
            system(world, resources);
        }
    }

    pub fn run_render(&self, world: &mut World, resources: &mut Resources) {
        for system in &self.render_systems {
            system(world, resources);
        }
    }

    pub fn run_inputs(&self, world: &mut World, resources: &mut Resources) {
        for system in &self.input_systems {
            system(world, resources);
        }
    }
    pub fn run_events(&self, _world: &mut World, resources: &mut Resources) {
        resources.events.flip();

        for event in resources.events.drain() {
            // order is explicit and intentional
            resources.simulation.process_event(&event);
            cursor_system(&mut resources.renderer.core.terrain_renderer.cursor, &event);
            run_car_events(
                event,
                &mut resources.renderer.core.car_subsystem,
                &resources.renderer.core.road_renderer,
            );
            // later:
            // audio_event_system(...)
            // ui_event_system(...)
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

        let mut world = World::new();
        let mut resources = Resources::new(window.clone(), &mut world);
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
                let (Some(world), Some(resources)) = (self.world.as_mut(), self.resources.as_mut())
                else {
                    return;
                };
                let Some(camera) = world.camera_mut(world.main_camera()) else {
                    return;
                };
                let variables = &mut resources.ui_loader.variables;
                let settings = &mut resources.settings;
                let ui_options = &mut resources.ui_loader.touch_manager.options;
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
                // ENGINE ACTIONS (ALL action-based, no direct key checks)! factorial
                // ---------------------------------------------------------------------

                // MSAA cycle
                if input.action_repeat("Cycle MSAA") {
                    resources.renderer.core.cycle_msaa(settings);
                }

                if input.action_repeat("Switch Debug Render Mode") {
                    settings.debug_view_state = settings.debug_view_state.next();
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
                    match resources.renderer.core.terrain_renderer.cursor.mode {
                        CursorMode::Roads(_) => resources
                            .events
                            .send(Event::SetCursorMode(CursorMode::Cars)),
                        CursorMode::Cars => resources
                            .events
                            .send(Event::SetCursorMode(CursorMode::TerrainEditing)),
                        CursorMode::TerrainEditing => resources
                            .events
                            .send(Event::SetCursorMode(CursorMode::None)),
                        CursorMode::None => resources
                            .events
                            .send(Event::SetCursorMode(CursorMode::Roads(RoadType::default()))),
                    }
                }
                if input.action_pressed_once("Leave Game") {
                    settings.total_game_time = resources.time.total_game_time;
                    settings.player_pos = camera.target;
                    match settings.save(data_dir("settings.toml")) {
                        Ok(_) => println!("Settings saved"),
                        Err(e) => eprintln!("Failed to save Settings: {e}"),
                    }
                    if settings.show_world {
                        match resources
                            .renderer
                            .core
                            .terrain_renderer
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
                            &resources.input.mouse,
                        )
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

                            if let Some(controller) = world.camera_controller_mut(cam) {
                                let zoom_factor = 10.0;
                                controller.zoom_velocity -= scroll.y * zoom_factor;
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

                    // -----------------------------
                    // Time controls (incl. real toggle stop)
                    // -----------------------------
                    let can_time_control = !resources.settings.editor_mode
                        && !resources.settings.drive_car
                        && resources.settings.show_world;

                    if can_time_control && resources.input.action_pressed_once("Toggle Stop Time") {
                        resources.simulation.toggle();

                        // Important: kill any backlog so the sim stops immediately.
                        if !resources.simulation.running {
                            resources.time.clear_sim_accumulator();
                        }
                    }

                    // Base timescale from "stopped" state
                    let mut time_speed = if resources.simulation.running {
                        1.0
                    } else {
                        0.0
                    };

                    // Optional override from held bindings (also resumes time)
                    for (action, speed) in TIME_SPEED_BINDINGS {
                        if resources.input.action_down(action) {
                            time_speed = speed;
                            resources.simulation.running = true; // resume if user is explicitly controlling time
                            break;
                        }
                    }

                    // Ensure stop always wins if it's enabled (even if other keys are held)
                    if !resources.simulation.running {
                        time_speed = 0.0;
                    }

                    resources
                        .ui_loader
                        .variables
                        .set_f32("time_speed", time_speed);

                    // -----------------------------
                    // Frame timing (accumulator lives here)
                    // -----------------------------
                    resources.time.begin_frame(time_speed);

                    // UI globals
                    {
                        let ui = &mut resources.ui_loader;
                        ui.variables.set_f32("fps", resources.time.render_fps);
                        ui.variables.set_f32("render_dt", resources.time.render_dt);
                        ui.variables.set_f32("sim_dt", resources.time.target_sim_dt);
                        ui.variables
                            .set_f32("total_game_time", resources.time.total_game_time as f32);
                    }

                    // -----------------------------
                    // Fixed-step simulation with budget + spiral-of-death protection
                    // -----------------------------
                    // Do input/events once per render frame (prevents "pressed_once" from firing N times).
                    self.schedule.run_inputs(world, resources);
                    self.schedule.run_events(world, resources);

                    // Clamp runaway backlog (e.g., window drag / breakpoint)
                    resources
                        .time
                        .clamp_sim_accumulator(MAX_SIM_STEPS_PER_FRAME);

                    let sim_budget =
                        Duration::from_secs_f32((resources.time.target_frametime * 0.7).max(0.0));
                    let sim_deadline = Instant::now() + sim_budget;

                    let mut steps = 0usize;
                    while resources.time.can_step_sim()
                        && steps < MAX_SIM_STEPS_PER_FRAME
                        && Instant::now() < sim_deadline
                    {
                        self.schedule.run_sim(world, resources);
                        resources.time.consume_sim_step();
                        steps += 1;
                    }

                    let achieved_speed = if resources.time.render_dt > 0.0 {
                        (steps as f32 * resources.time.target_sim_dt) / resources.time.render_dt
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
                    self.schedule.run_render(world, resources);

                    // -----------------------------
                    // FPS cap
                    // -----------------------------
                    let elapsed = frame_start.elapsed();
                    let target = Duration::from_secs_f32(resources.time.target_frametime.max(0.0));
                    if target > Duration::ZERO && elapsed < target {
                        thread::sleep(target - elapsed);
                    }
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            WindowEvent::Focused(false) | WindowEvent::Focused(true) => {
                if let Some(resources) = self.resources.as_mut() {
                    resources.input.reset_all(resources.time.total_time);
                }
            }

            _ => {}
        }
    }
}
