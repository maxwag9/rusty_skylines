use crate::data::Settings;
use crate::events::Events;
use crate::paths::data_dir;
use crate::renderer::Renderer;
use crate::renderer::shadows::CSM_CASCADES;
use crate::simulation::Simulation;
use crate::ui::actions::CommandQueue;
use crate::ui::input::InputState;
use crate::ui::ui_editor::UiButtonLoader;
use crate::world::World;
use std::sync::Arc;
use std::time::Instant;
use winit::window::Window;

pub struct CommandQueues {
    pub(crate) ui_command_queue: CommandQueue,
}
impl CommandQueues {
    pub fn new() -> Self {
        Self {
            ui_command_queue: CommandQueue::new(),
        }
    }
}
pub struct Resources {
    pub settings: Settings,
    pub time: TimeSystem,
    pub input: InputState,
    pub renderer: Renderer,
    pub simulation: Simulation,
    pub ui_loader: UiButtonLoader,
    pub events: Events,
    pub window: Arc<Window>,
    pub command_queues: CommandQueues,
}

impl Resources {
    pub fn new(window: Arc<Window>, world: &World) -> Self {
        let settings = Settings::load(data_dir("settings.toml"));
        let editor_mode = settings.editor_mode.clone();
        let camera_entity = world.main_camera();
        let camera = world.camera(camera_entity).unwrap();
        let renderer = Renderer::new(window.clone(), &settings, camera);
        let mut ui_loader = UiButtonLoader::new(
            editor_mode,
            settings.override_mode,
            settings.show_gui,
            &settings.bend_mode.clone(),
            window.inner_size(),
        );
        ui_loader
            .variables
            .set_bool("editor_mode", settings.editor_mode);
        let mut command_queues = CommandQueues::new();
        ui_loader.set_starting_menu(&settings, &mut command_queues.ui_command_queue);
        let mut time = TimeSystem::new();
        time.total_game_time = settings.total_game_time;

        Self {
            settings,
            time,
            input: InputState::new(),
            renderer,
            simulation: Simulation::new(),
            ui_loader,
            events: Events::new(),
            window,
            command_queues,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TimeSystem {
    pub last_sim: Instant,
    pub last_render: Instant,
    pub start: Instant,

    pub sim_dt: f32,
    pub sim_accumulator: f32,
    pub sim_target_step: f32,

    pub render_dt: f32,
    pub render_fps: f32,
    pub target_fps: f32,
    pub target_frametime: f32,
    pub total_time: f64,
    pub total_game_time: f64,
}

impl TimeSystem {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            last_sim: now,
            last_render: now,
            start: now,

            sim_dt: 0.0,
            sim_accumulator: 0.0,
            sim_target_step: 0.0,

            render_dt: 0.0,
            render_fps: 0.0,
            target_fps: 100.0,
            target_frametime: 0.0,
            total_time: 0.0,
            total_game_time: 0.0,
        }
    }

    pub fn set_tps(&mut self, tps: f32) {
        self.sim_target_step = 1.0 / tps;
        self.sim_accumulator = 0.0;
    }

    pub fn set_fps(&mut self, target_fps: f32) {
        self.target_fps = target_fps;
        self.target_frametime = 1.0 / target_fps;
    }

    pub fn update_sim(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_sim).as_secs_f32();
        self.last_sim = now;
        self.sim_dt = dt;
    }

    pub fn update_render(&mut self, time_speed: f32) {
        let now = Instant::now();
        let dt = (now - self.last_render).as_secs_f32();
        self.last_render = now;

        self.render_dt = dt;
        self.render_fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };

        self.total_time += dt as f64;
        self.total_game_time += dt as f64 * time_speed as f64;

        self.sim_accumulator += dt;
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub view: [[f32; 4]; 4],
    pub inv_view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub lighting_view_proj: [[[f32; 4]; 4]; CSM_CASCADES],
    pub cascade_splits: [f32; 4], // end distance of each cascade in view-space units
    pub sun_direction: [f32; 3],
    pub time: f32,
    pub camera_local: [f32; 3], // eye_world.local (x,y,z) where x/z are within chunk
    pub chunk_size: f32,
    pub camera_chunk: [i32; 2], // eye_world.chunk (x,z)
    pub _pad_cam: [u32; 2],     // padding to 16 bytes
    pub moon_direction: [f32; 3],
    pub orbit_radius: f32,
    pub reversed_depth_z: u32,
    pub shadows_enabled: u32,
    pub _pad_2: [u32; 2], // padding to 16 bytes
}
