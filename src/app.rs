use crate::data::Cycle;
use crate::helpers::paths::{data_dir, next_screenshot_path};
use crate::resources::Resources;
use crate::simulation::update_picked_pos;
use crate::systems::input::run_inputs;
use crate::systems::small_systems::run_commands;
use crate::systems::systems::{run_interpolation, run_render, run_sim, run_ticked, run_ui};
use crate::ui::actions::UiCommand;
use crate::ui::ui_edit_manager::CreateElementCommand;
use crate::ui::ui_touch_manager::ElementRef;
use crate::ui::vertex::UiButtonCircle;
use crate::ui::vertex::UiElement::Circle;
use crate::world::sound::run_sounds;
use crate::world::world::World;
use glam::Vec2;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use wgpu::{
    BufferAddress, BufferDescriptor, BufferUsages, Extent3d, MapMode, Origin3d, PollType,
    TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
};
use winit::application::ApplicationHandler;
use winit::cursor::CustomCursorSource;
use winit::event::{ElementState, StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

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
    window: Option<Arc<Box<dyn Window>>>,
    resources: Option<Resources>,
}

impl App {
    pub fn new() -> Self {
        Self {
            window: None,
            resources: None,
        }
    }
}

impl ApplicationHandler for App {
    fn new_events(&mut self, _event_loop: &dyn ActiveEventLoop, _cause: StartCause) {
        if let Some(resources) = self.resources.as_mut() {
            let world = &mut resources.world;
            let input = &mut world.input;
            let time = &world.time;
            input.mouse.delta = Vec2::ZERO;
            input.begin_frame(time.total_time);
            let pos = input.mouse.pos;
            let delta = input.mouse.delta;

            resources.ui.variables.set_f64("mouse_pos.x", pos.x);
            resources.ui.variables.set_f64("mouse_pos_delta.x", delta.x);
            resources.ui.variables.set_f64("mouse_pos.y", pos.y);
            resources.ui.variables.set_f64("mouse_pos_delta.y", delta.y);
        }
    }

    fn resumed(&mut self, event_loop: &dyn ActiveEventLoop) {} // For Mobile only

    fn can_create_surfaces(&mut self, event_loop: &dyn ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("Rusty Skylines")
                        .with_surface_size(winit::dpi::PhysicalSize::new(2560, 1400)),
                )
                .expect("Failed to create window"),
        );

        let mut resources = Resources::new(window.clone(), event_loop);
        let world = &mut resources.world;
        let time = &mut world.time;
        time.set_tps(resources.settings.target_tps.max(1.0));
        time.set_fps(resources.settings.target_fps.max(1.0));
        resources
            .ui
            .variables
            .set_string("cursor_mode", format!("{:#?}", world.terrain.cursor.mode));
        resources.ui.variables.set_array(
            "screen",
            vec![window.surface_size().width, window.surface_size().height],
        );

        let width = 32u16;
        let height = 32u16;

        let rgba: Vec<u8> = vec![64; width as usize * height as usize * 4];

        let source = CustomCursorSource::from_rgba(
            rgba, width, height, 0, // hotspot x
            0, // hotspot y
        )
        .unwrap();
        let custom_cursor = event_loop.create_custom_cursor(source);
        //window.set_cursor(Cursor::Custom(custom_cursor));

        self.window = Some(window.clone());
        self.resources = Some(resources);

        event_loop.set_control_flow(ControlFlow::Poll);
        window.request_redraw();
    } // resumed() but new

    fn window_event(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                let Some(resources) = self.resources.as_mut() else {
                    return;
                };
                let world = &mut resources.world;
                let camera = &mut world.world_state.camera;
                let cam_ctrl = &mut world.world_state.cam_controller;
                let ui = &mut resources.ui;
                let settings = &mut resources.settings;
                let ui_options = &mut ui.touch_manager.options;
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
                    ui.touch_manager.editor.enabled = settings.editor_mode;
                    ui.variables.set_bool("editor_mode", settings.editor_mode);
                    settings.show_world = !settings.editor_mode;
                    ui.variables.set_bool("show_world", settings.show_world);
                    settings.show_gui = true;
                    ui.variables.set_bool("show_gui", settings.show_gui);
                }
                // Toggle override_mode
                if input.action_repeat("Toggle override mode") {
                    settings.override_mode = !settings.override_mode;
                    ui_options.override_mode = settings.override_mode;
                    ui.variables
                        .set_bool("override_mode", settings.override_mode)
                }
                if input.action_repeat("Toggle show world") {
                    settings.show_world = !settings.show_world;
                    ui.variables.set_bool("show_world", settings.show_world);
                }
                if input.action_repeat("Toggle drive car") {
                    settings.drive_car = !settings.drive_car;
                    ui.variables.set_bool("drive_car", settings.drive_car);
                }
                if input.action_repeat("Toggle noclip") {
                    settings.noclip = !settings.noclip;
                    ui.variables.set_bool("noclip", settings.noclip);
                }
                // Toggle show_gui
                if input.action_repeat("Toggle show gui") {
                    settings.show_gui = !settings.show_gui;
                    ui_options.show_gui = settings.show_gui;
                    ui.variables.set_bool("show_gui", settings.show_gui);
                    settings.override_mode = false;
                    ui_options.override_mode = settings.override_mode;
                    ui.variables
                        .set_bool("override_mode", settings.override_mode);
                    settings.editor_mode = false;
                    ui.touch_manager.editor.enabled = settings.editor_mode;
                    ui.variables.set_bool("editor_mode", settings.editor_mode)
                }
                if input.action_repeat("Screenshot") {
                    let view = &resources.render_core.pipelines.resolved.tonemapped;
                    let width = view.texture().width();
                    let height = view.texture().height();

                    let bytes_per_pixel = 4; // Rgba8UnormSrgb
                    let unpadded_bytes_per_row = bytes_per_pixel * width;
                    let padded_bytes_per_row = unpadded_bytes_per_row.next_multiple_of(256);
                    let buffer_size = (padded_bytes_per_row * height) as BufferAddress;

                    let output_buffer =
                        resources
                            .render_core
                            .device
                            .create_buffer(&BufferDescriptor {
                                label: Some("Screenshot Buffer"),
                                size: buffer_size,
                                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                                mapped_at_creation: false,
                            });

                    let mut encoder = resources
                        .render_core
                        .device
                        .create_command_encoder(&Default::default());
                    encoder.copy_texture_to_buffer(
                        TexelCopyTextureInfo {
                            texture: &view.texture(),
                            mip_level: 0,
                            origin: Origin3d::ZERO,
                            aspect: TextureAspect::All,
                        },
                        TexelCopyBufferInfo {
                            buffer: &output_buffer,
                            layout: TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(padded_bytes_per_row),
                                rows_per_image: Some(height),
                            },
                        },
                        Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                    );

                    resources.render_core.queue.submit(Some(encoder.finish()));

                    let buffer_slice = output_buffer.slice(..);
                    buffer_slice.map_async(MapMode::Read, |_| {});
                    resources
                        .render_core
                        .device
                        .poll(PollType::Wait {
                            submission_index: None,
                            timeout: Some(Duration::from_secs(1)),
                        })
                        .ok();

                    let data = buffer_slice.get_mapped_range();

                    // Handle row padding when saving
                    if padded_bytes_per_row != unpadded_bytes_per_row {
                        // Strip padding
                        let mut pixels =
                            Vec::with_capacity((unpadded_bytes_per_row * height) as usize);
                        for row in 0..height {
                            let start = (row * padded_bytes_per_row) as usize;
                            let end = start + unpadded_bytes_per_row as usize;
                            pixels.extend_from_slice(&data[start..end]);
                        }
                        image::save_buffer(
                            next_screenshot_path(),
                            &pixels,
                            width,
                            height,
                            image::ColorType::Rgba8,
                        )
                        .unwrap();
                    } else {
                        image::save_buffer(
                            next_screenshot_path(),
                            &data,
                            width,
                            height,
                            image::ColorType::Rgba8,
                        )
                        .unwrap();
                    }
                }

                // Save GUI
                if input.action_pressed_once("Save GUI layout") {
                    match ui.save_gui_to_file(
                        data_dir("ui_data/menus"),
                        data_dir("ui_data/menus/advanced_primitives"),
                        resources.window.surface_size(),
                    ) {
                        Ok(_) => println!("GUI layout saved"),
                        Err(e) => eprintln!("Failed to save GUI layout: {e}"),
                    }
                }
                if input.action_pressed_once("Toggle Cursor Mode") {
                    world.events.send(world.terrain.cursor.mode.next_command());
                }
                if input.action_repeat("Toggle Debug Menu") {
                    resources
                        .command_queues
                        .ui_command_queue
                        .push(UiCommand::ToggleMenu {
                            menu_name: "Debug_Menu".to_string(),
                        });
                    let debug_menu_active = ui.menus.get("Debug_Menu").unwrap().active;
                    ui.variables.set_bool("debug_mode", debug_menu_active);
                }
                let main_menu_active = resources.ui.menus.get("MainMenu").unwrap().active;

                if !main_menu_active && input.action_released("Exit to Main Menu") {
                    let cmds: Vec<UiCommand> = vec![
                        UiCommand::OpenMenu {
                            menu_name: "MainMenu".to_string(),
                        },
                        UiCommand::CloseMenu {
                            menu_name: "Editor_Menu".to_string(),
                        },
                        UiCommand::CloseMenu {
                            menu_name: "Debug_Menu".to_string(),
                        },
                        UiCommand::SetVar {
                            element_ref: ElementRef::default(),
                            name: "editor_mode".to_string(),
                            value: "false".to_string(),
                        },
                        // UiCommand::SetVar {
                        //     element_ref: ElementRef::default(),
                        //     name: "show_world".to_string(),
                        //     value: "false".to_string(),
                        // },
                        UiCommand::SetVar {
                            element_ref: ElementRef::default(),
                            name: "override_mode".to_string(),
                            value: "false".to_string(),
                        },
                    ];
                    resources.command_queues.ui_command_queue.push_many(cmds);
                } else if main_menu_active && input.action_released("Leave Game") {
                    resources
                        .command_queues
                        .ui_command_queue
                        .push(UiCommand::ExitGame);
                }
                // Add GUI element
                if input.action_repeat("Add GUI element")
                    && resources.ui.touch_manager.editor.enabled
                {
                    if let Some(sel) = resources.ui.touch_manager.selection.selected.first() {
                        resources.ui.ui_edit_manager.execute_command(
                            CreateElementCommand {
                                affected_element: sel.clone(),
                                element: Circle(UiButtonCircle::default()),
                            },
                            &mut resources.ui.touch_manager,
                            &mut resources.ui.menus,
                            &mut resources.ui.variables,
                            &input.mouse,
                        )
                    }
                }
            }

            WindowEvent::PointerButton { state, button, .. } => {
                if let Some(resources) = self.resources.as_mut() {
                    if let Some(mouse_button) = button.mouse_button() {
                        resources
                            .world
                            .input
                            .handle_mouse_button(mouse_button, state);
                    }
                }
            }

            WindowEvent::PointerMoved { position, .. } => {
                if let Some(resources) = self.resources.as_mut() {
                    let input = &mut resources.world.input;
                    input.handle_mouse_move(position.x, position.y);

                    let pos = input.mouse.pos;
                    let delta = input.mouse.delta;

                    resources.ui.variables.set_f64("mouse_pos.x", pos.x);
                    resources.ui.variables.set_f64("mouse_pos_delta.x", delta.x);
                    resources.ui.variables.set_f64("mouse_pos.y", pos.y);
                    resources.ui.variables.set_f64("mouse_pos_delta.y", delta.y);
                    // camera rotation ONLY if needed & dragging
                    if input.mouse.buttons.middle.pressed {
                        let cam_controller = &mut resources.world.world_state.cam_controller;
                        let pitch_s = 0.002;
                        let yaw_s = 0.0016;

                        cam_controller.target_yaw += delta.x * yaw_s;
                        cam_controller.target_pitch += delta.y * pitch_s;

                        cam_controller.yaw_velocity = delta.x * yaw_s;
                        cam_controller.pitch_velocity = delta.y * pitch_s;
                    }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(resources) = self.resources.as_mut() {
                    let scroll = resources.world.input.handle_mouse_wheel(delta);

                    if !resources.settings.editor_mode {
                        let cam_controller = &mut resources.world.world_state.cam_controller;
                        let zoom_factor = 10.0;
                        cam_controller.zoom_velocity -= scroll.y * zoom_factor;
                    }
                }
            }
            WindowEvent::SurfaceResized(size) => {
                if let Some(resources) = self.resources.as_mut() {
                    resources
                        .render_core
                        .resize(&resources.surface, size, &mut resources.ui);
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                if let Some(resources) = self.resources.as_mut() {
                    let size = resources.window.surface_size(); // << get the real physical size
                    if size.width > 0 && size.height > 0 {
                        resources
                            .render_core
                            .resize(&resources.surface, size, &mut resources.ui);
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                let Some(resources) = self.resources.as_mut() else {
                    event_loop.exit();
                    return;
                };
                let frame_start = Instant::now();
                update_time(resources);

                run_inputs(resources);

                run_ui(resources, event_loop);

                run_commands(resources);
                run_ticked(resources);
                let mut steps = 0u32;

                // If speed just changed, skip sim this frame for clean transition
                if !resources.world.time.speed_just_changed {
                    resources
                        .world
                        .time
                        .clamp_sim_accumulator(MAX_SIM_STEPS_PER_FRAME);

                    let sim_budget = Duration::from_secs_f32(
                        (resources.world.time.target_frametime * 0.6).max(0.0),
                    );
                    let sim_deadline = Instant::now() + sim_budget;

                    while resources.world.time.can_step_sim()
                        && (steps as usize) < MAX_SIM_STEPS_PER_FRAME
                        && Instant::now() < sim_deadline
                    {
                        run_sim(resources);
                        resources.world.time.consume_sim_step();
                        steps += 1;
                    }
                }

                // Update achieved speed (windowed measurement)
                resources.world.time.update_achieved_speed(steps);
                resources
                    .ui
                    .variables
                    .set_f64("achieved_time_speed", resources.world.time.achieved_speed);

                // Render
                {
                    let camera = &resources.world.world_state.camera;
                    let proj = camera.proj();
                    resources.world.world_state.update(
                        &mut resources.ui,
                        &resources.world.time,
                        &resources.settings,
                        proj,
                    );
                }

                run_interpolation(resources);
                run_sounds(resources);
                run_render(resources); // use commands output

                // FPS cap
                let elapsed = frame_start.elapsed();
                let target =
                    Duration::from_secs_f32(resources.world.time.target_frametime.max(0.0));
                if target > Duration::ZERO && elapsed < target {
                    thread::sleep(target - elapsed);
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            WindowEvent::Focused(false) | WindowEvent::Focused(true) => {
                if let Some(resources) = self.resources.as_mut() {
                    let input = &mut resources.world.input;
                    let time = &mut resources.world.time;
                    input.reset_all(time.total_time);
                }
            }

            _ => {}
        }
    }
}

fn update_time(resources: &mut Resources) {
    let Resources {
        world,
        settings,
        ui,
        render_core,
        ..
    } = resources;
    let World {
        world_state,
        time,
        input,
        simulation,
        terrain,
        ..
    } = world;
    update_picked_pos(
        terrain,
        &world_state.camera,
        settings,
        &render_core.config,
        input,
    );
    terrain.make_pick_uniforms(
        &render_core.queue,
        &render_core.pipelines.buffers.pick,
        &world_state.camera,
    );
    let can_time_control = !settings.editor_mode && !settings.drive_car && settings.show_world;

    if can_time_control && input.action_pressed_once("Toggle Stop Time") {
        simulation.toggle();

        if !simulation.running {
            time.clear_sim_accumulator();
        }
    }

    let mut time_speed = if simulation.running { 1.0 } else { 0.0 };

    for (action, speed) in TIME_SPEED_BINDINGS {
        if can_time_control && input.action_down(action) {
            time_speed = speed;
            simulation.running = true;
            break;
        }
    }

    if !simulation.running {
        time_speed = 0.0;
    }

    resources.ui.variables.set_f64("time_speed", time_speed);

    time.begin_frame(time_speed);

    {
        let ui = &mut resources.ui;
        ui.variables.set_f64("fps", time.render_fps);
        ui.variables.set_f64("render_dt", time.render_dt);
        ui.variables.set_f64("sim_dt", time.target_sim_dt);
        ui.variables
            .set_f64("total_game_time", time.total_game_time as f32);
    }
}
