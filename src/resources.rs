use crate::data::Settings;
use crate::helpers::paths::rusty_skylines_dir;
use crate::renderer::render_core::{
    Renderer, create_device, create_surface_and_adapter, create_surface_config,
};
use crate::renderer::shadows::CSM_CASCADES;
use crate::simulation::Simulation;
use crate::ui::actions::CommandQueue;
use crate::ui::ui_editor::Ui;
use crate::ui::variables::load_colors;
use crate::world::game_state::GameState;
use crate::world::sound::Sounds;
use crate::world::world::World;
use std::sync::Arc;
use std::time::Instant;
use wgpu::Surface;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

pub struct CommandQueues {
    pub ui_command_queue: CommandQueue,
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
    pub window: Arc<Box<dyn Window>>,
    pub command_queues: CommandQueues,

    // The simulation & world core:
    pub world: World,
    pub simulation: Simulation,
    pub game_state: GameState,
    // The GPU + render-only subsystems:
    pub render_core: Renderer,

    pub ui: Ui,
    pub sounds: Sounds,
    pub surface: Surface<'static>,
}

impl Resources {
    pub fn new(window: Arc<Box<dyn Window>>, event_loop: &dyn ActiveEventLoop) -> Self {
        let mut settings = Settings::load(rusty_skylines_dir("settings.toml"));
        let editor_mode = settings.editor_mode.clone();

        let (surface, adapter, size) = create_surface_and_adapter(window.clone(), event_loop);
        let (config, msaa_samples) = create_surface_config(&surface, &adapter, &mut settings, size);
        let (device, queue) = &create_device(&adapter);

        surface.configure(device, &config);
        let mut game_state = GameState::new();
        let mut world_core = World::new(device, &settings, &mut game_state);
        let camera = &mut world_core.world_state.camera;

        let render_core = Renderer::new(device, queue, &config, size, adapter, &settings, camera);

        let mut ui_loader = Ui::new(&settings, window.surface_size());
        ui_loader
            .variables
            .set_bool("editor_mode", settings.editor_mode);
        load_colors(
            rusty_skylines_dir("colors.toml"),
            &settings,
            &mut ui_loader.variables,
        );
        let mut command_queues = CommandQueues::new();
        ui_loader.set_starting_menu(&settings, &mut command_queues.ui_command_queue);
        world_core.time.total_game_time = settings.total_game_time;

        Self {
            surface,
            settings,
            ui: ui_loader,
            window,
            command_queues,
            world: world_core,
            simulation: Simulation::new(),
            game_state,
            render_core,
            sounds: Sounds::new(),
        }
    }
}

pub struct Time {
    pub last_frame: Instant,
    pub start: Instant,

    pub render_dt: f32,
    pub render_fps: f32,
    pub target_fps: f32,
    pub target_frametime: f32,

    pub sim_accumulator: f32,
    pub target_sim_dt: f32,
    pub prev_time_scale: f32,

    pub achieved_speed: f32,
    achieved_speed_window_time: f32,
    achieved_speed_window_steps: u32,

    pub total_time: f64,
    pub total_game_time: f64,
    pub frame_count: u64,

    pub max_frame_dt: f32,

    pub speed_just_changed: bool,
    pub current_time_speed: f32,
}

impl Time {
    pub fn new() -> Self {
        let now = Instant::now();
        let target_fps = 100.0;
        let target_frametime = 1.0 / target_fps;

        let tps = 60.0;
        let target_sim_dt = 1.0 / tps;

        Self {
            last_frame: now,
            start: now,

            render_dt: 0.0,
            render_fps: 0.0,
            target_fps,
            target_frametime,

            sim_accumulator: 0.0,
            target_sim_dt,

            prev_time_scale: 1.0,

            achieved_speed: 1.0,
            achieved_speed_window_time: 0.0,
            achieved_speed_window_steps: 0,

            total_time: 0.0,
            total_game_time: 0.0,
            frame_count: 0,

            max_frame_dt: 0.25,

            speed_just_changed: false,
            current_time_speed: 1.0,
        }
    }

    pub fn set_tps(&mut self, tps: f32) {
        let tps = tps.max(1.0);
        self.target_sim_dt = 1.0 / tps;
        self.sim_accumulator = 0.0;
    }

    pub fn set_fps(&mut self, target_fps: f32) {
        self.target_fps = target_fps.max(1.0);
        self.target_frametime = 1.0 / self.target_fps;
    }

    #[inline]
    pub fn is_rewinding(&self) -> bool {
        self.current_time_speed < 0.0
    }

    #[inline]
    pub fn time_direction(&self) -> f32 {
        if self.current_time_speed >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    pub fn begin_frame(&mut self, time_speed: f32) {
        let now = Instant::now();
        let raw_dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Pure wall-clock dt, no time_speed here
        let raw_dt = raw_dt.clamp(0.0, self.max_frame_dt);

        if self.render_dt == 0.0 {
            self.render_dt = raw_dt;
        } else {
            self.render_dt += (raw_dt - self.render_dt) * 0.05;
        }

        self.render_fps = if self.render_dt > 0.0 {
            1.0 / self.render_dt
        } else {
            0.0
        };

        self.total_time += self.render_dt as f64;
        self.frame_count += 1;

        let speed_changed = (time_speed - self.current_time_speed).abs() > 1e-6;
        self.speed_just_changed = speed_changed;

        if speed_changed {
            self.sim_accumulator = 0.0;
            self.prev_time_scale = self.current_time_speed;
            self.current_time_speed = time_speed;
            self.achieved_speed = time_speed;
            self.achieved_speed_window_time = 0.0;
            self.achieved_speed_window_steps = 0;
        }

        self.achieved_speed_window_time += self.render_dt;

        // time_speed scaling only applies to the sim accumulator
        self.sim_accumulator += self.render_dt * time_speed.abs();
    }
    pub fn update_achieved_speed(&mut self, steps: u32) {
        self.achieved_speed_window_steps += steps;

        const WINDOW_DURATION: f32 = 0.5;

        if self.achieved_speed_window_time >= WINDOW_DURATION {
            let sim_time = self.achieved_speed_window_steps as f32 * self.target_sim_dt;
            let raw_speed = if self.achieved_speed_window_time > 0.0 {
                sim_time / self.achieved_speed_window_time
            } else {
                0.0
            };

            self.achieved_speed = raw_speed * self.time_direction();

            self.achieved_speed_window_time = 0.0;
            self.achieved_speed_window_steps = 0;
        }
    }

    pub fn clear_sim_accumulator(&mut self) {
        self.sim_accumulator = 0.0;
    }

    pub fn clamp_sim_accumulator(&mut self, max_steps: usize) {
        let max = self.target_sim_dt * max_steps as f32;
        if self.sim_accumulator > max {
            self.sim_accumulator = max;
        }
    }

    #[inline]
    pub fn can_step_sim(&self) -> bool {
        if self.target_sim_dt <= 0.0 || self.sim_accumulator < self.target_sim_dt {
            return false;
        }
        if self.is_rewinding() && self.total_game_time < 1e-9 {
            return false;
        }
        true
    }

    #[inline]
    pub fn consume_sim_step(&mut self) {
        self.sim_accumulator -= self.target_sim_dt;
        let dt = self.target_sim_dt as f64;
        if self.current_time_speed >= 0.0 {
            self.total_game_time += dt;
        } else {
            self.total_game_time = (self.total_game_time - dt).max(0.0);
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    // ── Current frame matrices ──────────────────────────────────
    pub view: [[f32; 4]; 4],
    pub inv_view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],

    // ── Previous frame reprojection ─────────────────────────────
    pub prev_view_proj: [[f32; 4]; 4],

    // ── Shadow cascades ─────────────────────────────────────────
    pub lighting_view_proj: [[[f32; 4]; 4]; CSM_CASCADES],
    pub cascade_splits: [f32; 4],

    // ── Lighting ────────────────────────────────────────────────
    pub sun_direction: [f32; 3],
    pub time: f32,
    pub moon_direction: [f32; 3],
    pub orbit_radius: f32,

    // ── Current camera (chunk-relative) ─────────────────────────
    pub camera_local: [f32; 3], // vec3<f32> + 1 float pad
    pub chunk_size: f32,
    pub camera_chunk: [i32; 2], // vec2<i32>
    pub _pad_cam: [u32; 2],     // align to 16

    // ── Previous camera (chunk-relative) ────────────────────────
    pub prev_camera_local: [f32; 3], // vec3<f32> + 1 float pad
    pub frame_index: u32,
    pub prev_camera_chunk: [i32; 2], // vec2<i32>
    pub _pad_prev1: [i32; 2],        // align to 16

    // ── TAA jitter ──────────────────────────────────────────────
    pub curr_jitter: [f32; 2],
    pub prev_jitter: [f32; 2],

    // ── Misc settings ───────────────────────────────────────────
    pub reversed_depth_z: u32,
    pub csm_enabled: u32,
    pub near_far_depth: [f32; 2],
}
